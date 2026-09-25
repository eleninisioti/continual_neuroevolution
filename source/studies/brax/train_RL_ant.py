"""
Train PPO on Brax Ant (non-continual).

Uses the standard Brax Ant environment with default parameters.

Usage:
    python source/studies/brax/train_RL_ant.py --gpus 0
    python source/studies/brax/train_RL_ant.py --num_timesteps 50000000 --gpus 0
"""

import argparse
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# One dirname deeper since 2026-09-08: the trainers moved from
# `source/<suite>/` into `source/studies/<suite>/`. The runner
# invokes them as SCRIPTS, so `source` is importable only via
# this insert -- a short walk here is a ModuleNotFoundError at
# launch, not a subtle one.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(SCRIPT_DIR)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from source.algorithms.rl.redo import DEFAULT_TAU as REDO_DEFAULT_TAU  # one threshold for the study
from source.algorithms.networks import ACTIVATIONS, POLICY_ARCH  # one activation for the suite

# Must run before jax is imported: CUDA_VISIBLE_DEVICES has no effect once the
# backend is up. This file carried its own byte-identical copy of the argv scan
# and the preallocation setting; `select_gpus` is the shared one every gymnax
# trainer already uses.
from source.utils.runtime import Tee, select_gpus, write_run_config

select_gpus()

import functools
import json
import pickle
import time

import jax
import jax.numpy as jnp
import numpy as np
from brax.envs import create

# One source of truth for which simulator the ant runs on.
from source.envs.brax_common import DEFAULT_BACKEND
from brax.training.agents.ppo import networks as ppo_networks
from source.studies.brax.my_brax import networks as my_brax_networks
import wandb

from source.studies.brax.my_brax.cchain import add_chain_args
from source.studies.brax.my_brax.ppo_train import train as ppo_train
from source.metrics import rl_diagnostics


def parse_args():
    parser = argparse.ArgumentParser(description='PPO on Brax Ant (Non-Continual)')
    parser.add_argument('--env', type=str, default='ant',
                        help='Brax environment name')
    parser.add_argument('--num_timesteps', type=int, default=50_000_000,
                        help='Total timesteps')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--trial', type=int, default=1)
    parser.add_argument('--gpus', type=str, default=None)

    # PPO hyperparameters (from brax defaults for ant)
    parser.add_argument('--num_envs', type=int, default=4096)
    parser.add_argument('--episode_length', type=int, default=1000)
    parser.add_argument('--backend', type=str, default=DEFAULT_BACKEND,
                        choices=['mjx', 'generalized', 'spring', 'positional'],
                        help="Physics backend. Must match the continual block's, "
                             "which this run is the control for.")
    parser.add_argument('--learning_rate', type=float, default=3e-4)
    parser.add_argument('--entropy_cost', type=float, default=1e-2)
    parser.add_argument('--discounting', type=float, default=0.97)
    parser.add_argument('--unroll_length', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=2048)
    parser.add_argument('--num_minibatches', type=int, default=32)
    parser.add_argument('--num_updates_per_batch', type=int, default=4)
    parser.add_argument('--normalize_observations', type=bool, default=True)
    parser.add_argument('--reward_scaling', type=float, default=10.0)
    parser.add_argument('--num_evals', type=int, default=10)
    parser.add_argument('--num_eval_envs', type=int, default=128)
    parser.add_argument('--action_repeat', type=int, default=1)
    # Defaults are what this file hardcoded before they were flags, so an
    # existing stationary run is reproduced by not passing them. The continual
    # block runs 128,128 for both -- pass that here when this block is being
    # used as its control.
    parser.add_argument('--policy_hidden_sizes', type=str, default='128,128')
    parser.add_argument('--value_hidden_sizes', type=str, default='256,256')
    # A FLAG, AND READ FROM POLICY_ARCH, because this file hardcoded tanh with
    # no way to change it -- which is why its ant runs collapsed to 69-139
    # while the reference reached 4379. The continual trainer reads the same
    # entry, so the stationary and continual arms cannot end up on different
    # networks again.
    parser.add_argument('--activation', type=str,
                        default=POLICY_ARCH['brax']['activation'],
                        choices=['tanh', 'swish', 'relu'])
    # TWO PLACES ppo_train.py AND ppo_continual_train.py DISAGREE. They were
    # added as flags to test whether either explains why this trainer cannot
    # learn the ant at the small-batch shape (512/16/32/10/5) that
    # ppo_continual_train.py takes to 4110 within one 24.5M sub-task.
    #
    #   max_grad_norm       None here, 1.0 there
    #   deterministic_eval  False here (sampled actions), hardcoded True there
    #
    # NEITHER IS THE CAUSE. Measured at 40M steps, ant + speed target 2.0, one
    # seed, everything else equal:
    #
    #   unclipped, sampled eval, value 128,128     259
    #   unclipped, sampled eval, value 256,256     238
    #   CLIPPED,   sampled eval, value 128,128     139   <- clipping does not help
    #   CLIPPED,   det eval,     value 128,128     494   <- best, still 8x short
    #
    # So the value net is not it, clipping is not it, and the eval policy is
    # worth about 3.5x but nowhere near the gap. All four climb to 600-900 by
    # 4.4M and then decay; the continual trainer at the same shape keeps
    # climbing. Whatever the real difference is, it is elsewhere in the two
    # implementations and is unidentified -- which is the argument for having
    # ONE of them rather than for finding it.
    #
    # Until then the ant's stationary control is produced by the continual
    # trainer with the ground held still (`PHASES=reference` in
    # scripts/outdated/train/queue_brax_fixed.sh), which is what runs/brax/reference is
    # and which avoids this path entirely.
    #
    # Defaults here are ppo_train.py's own, so an existing stationary tree is
    # reproduced by not passing them.
    parser.add_argument('--max_grad_norm', type=float, default=None)
    parser.add_argument('--deterministic_eval',
                        type=lambda x: str(x).lower() == 'true', default=False)

    # Method variants
    parser.add_argument('--use_trac', action='store_true', default=False,
                        help='Use TRAC optimizer for adaptive learning rates')
    parser.add_argument('--use_redo', action='store_true', default=False,
                        help='Use ReDo (Reinitializing Dormant Neurons)')
    parser.add_argument('--redo_frequency', type=int, default=10,
                        help='Apply ReDo every N epochs')
    parser.add_argument('--redo_tau', type=float, default=REDO_DEFAULT_TAU,
                        help='Threshold for dormant neuron detection')
    # Dormancy is MEASURED for every method, not just ReDo. ReDo recycles
    # dormant units; this only counts them, and gymnax reports the count for
    # every method -- so leaving it off here would give brax one populated
    # column where gymnax has four. On by default for that reason; it costs one
    # extra forward pass per evaluation.
    parser.add_argument('--track_dormant', type=int, default=1,
                        help='Log dormant-neuron counts and ages every eval')
    parser.add_argument('--dormant_tau', type=float, default=REDO_DEFAULT_TAU,
                        help='Dormancy threshold when only measuring')
    # C-CHAIN, at the reference implementation's continuous-control defaults --
    # ant's policy is a Normal, so the churn term is an MSE between action means
    # and sits on a completely different scale from the gymnax logit version.
    add_chain_args(parser)

    # Output
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--wandb_project', type=str, default='brax_ant_ppo')
    parser.add_argument('--run_name', type=str, default=None)

    parser.add_argument('--speed_target', type=float, default=None,
                        help='Track this speed instead of brax\'s unbounded '
                             'forward-velocity reward. Must match the continual '
                             'block\'s target for the two to be comparable.')
    parser.add_argument('--speed_margin', type=float, default=None)
    parser.add_argument('--speed_weight', type=float, default=None)

    return parser.parse_args()


def main():
    args = parse_args()

    env_name = args.env
    num_timesteps = args.num_timesteps
    seed = args.seed
    trial = args.trial

    output_dir = args.output_dir or f"projects/brax/ppo_{env_name}/trial_{trial}"
    os.makedirs(output_dir, exist_ok=True)

    # Checkpoints for the post-hoc metrics. The NE trainers have written a
    # per-sub-task checkpoint directory for a long time and this trainer wrote
    # only a single final pickle, so nothing about a stationary ant RL run could
    # be re-scored at an intermediate point the way a GA run can.
    checkpoint_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    # `train.log`, as every gymnax trainer writes and no brax one did.
    sys.stdout = Tee(os.path.join(output_dir, "train.log"))

    print("=" * 60)
    print(f"PPO on Brax {env_name} (Non-Continual)")
    print("=" * 60)
    print(f"  Total timesteps: {num_timesteps:,}")
    print(f"  Seed: {seed}")
    print(f"  Output: {output_dir}")

    # Create environment
    # Target-speed reward when asked, so this block runs the continual block's
    # objective and the two are comparable at all; see train_GA_ant.create_env.
    # Delegating to the continual factory with the sequence switched off means
    # the reward wrapper is the same code path, not a second copy of it.
    if args.speed_target is not None:
        from source.envs.brax_ant import create_env_with_damaged_leg
        env = create_env_with_damaged_leg(
            env_name, leg_idx=None, episode_length=args.episode_length,
            target_speed=float(args.speed_target),
            speed_margin=args.speed_margin, speed_weight=args.speed_weight,
            backend=args.backend)
    else:
        env = create(env_name, episode_length=args.episode_length,
                     backend=args.backend)

    print(f"  Obs size: {env.observation_size}, Action size: {env.action_size}")

    # Determine algorithm name
    if args.use_redo:
        algo_name = "redo_ppo"
    elif args.use_trac:
        algo_name = "trac_ppo"
    elif args.use_cchain:
        algo_name = "cchain_ppo"
    else:
        algo_name = "ppo"

    # Initialize wandb
    config = {
        'env': env_name, 'num_timesteps': num_timesteps,
        'seed': seed, 'trial': trial, 'algorithm': algo_name,
        'activation': args.activation,
        'num_envs': args.num_envs, 'episode_length': args.episode_length,
        'learning_rate': args.learning_rate, 'entropy_cost': args.entropy_cost,
        'discounting': args.discounting, 'unroll_length': args.unroll_length,
        'batch_size': args.batch_size, 'num_minibatches': args.num_minibatches,
        'num_updates_per_batch': args.num_updates_per_batch,
        'reward_scaling': args.reward_scaling, 'action_repeat': args.action_repeat,
        'use_trac': args.use_trac, 'use_redo': args.use_redo,
        'track_dormant': bool(args.track_dormant),
        'use_cchain': args.use_cchain,
        'chain_target_rel_scale': args.chain_target_rel_scale,
        'chain_warmup_iterations': args.chain_warmup_iterations,
        'chain_coef_window': args.chain_coef_window,
    }
    run_name = args.run_name or f"{algo_name}_{env_name}_trial{trial}"
    wandb.init(project=args.wandb_project, config=config,
               name=run_name, reinit=True)
    # The run says what network it searched, so scripts/check_architectures.py
    # can confirm every method compared on this task used the same one. This
    # trainer wrote no config.json at all, which is why every brax RL group is
    # listed UNVERIFIABLE by that check.
    write_run_config(output_dir, config, policy_arch='brax')

    # Track metrics
    best_reward = -float('inf')
    training_metrics = []
    start_time_metrics = time.time()

    # The diagnostics this run is expected to carry, under the gymnax names, so
    # one reader serves both trees. See source/metrics/rl_diagnostics.py.
    #
    # THIS IS THE REASON THE BLOCK EXISTS. `ppo_train` already returns losses,
    # entropy, the evaluation spread and -- under C-CHAIN -- policy churn, and
    # progress_fn dropped every one of them on the floor: the old body kept
    # `step`, `reward` and `best_reward` and nothing else, so a stationary ant
    # RL run's training_metrics.json was three columns against the gymnax
    # trainer's sixteen. Every plasticity figure read an empty series off this
    # tree. Same failure, and same fix, as the continual ant trainer's.
    report_missing = rl_diagnostics.make_missing_reporter()

    # Weight statistics and checkpoints both need the parameters, which
    # progress_fn is not given -- its signature is (step, metrics). ppo_train
    # calls policy_params_fn(step, make_policy, params) just BEFORE progress_fn
    # on every iteration (my_brax/ppo_train.py:1132 then the eval), so the stats
    # are computed there and merged into the record progress_fn is about to
    # append. `pending` is emptied on use, so a record can never inherit the
    # previous iteration's numbers if that call order ever changes.
    pending_param_stats = {}

    def policy_params_fn(step, make_policy, params):
        del make_policy
        normalizer_params, policy_params, value_params = params
        # Weight statistics are NOT computed here any more: `ppo_train` writes
        # them into `metrics` itself, by the same shared function, so the
        # stationary ant and the stationary cheetah cannot drift apart. This
        # callback keeps only the job progress_fn cannot do -- the checkpoint.
        pending_param_stats.clear()

        # A checkpoint per evaluation, for the post-hoc metrics. The normalizer
        # is stored WITH the parameters and is not optional: this policy is
        # trained behind a running observation normaliser, so a checkpoint
        # re-scored without it is being fed a different observation
        # distribution from the one it was trained on. That is the failure
        # source/studies/brax/evaluate_continual.py's docstring already warns about.
        ckpt_path = os.path.join(checkpoint_dir, f"step_{int(step):012d}.pkl")
        with open(ckpt_path, 'wb') as f:
            pickle.dump({
                'step': int(step),
                'normalizer_params': normalizer_params,
                'policy_params': policy_params,
                'value_params': value_params,
                'config': config,
            }, f)

    def progress_fn(step, metrics):
        nonlocal best_reward

        reward = float(metrics.get('eval/episode_reward', 0.0))
        if reward > best_reward:
            best_reward = reward

        record = {
            'step': int(step),
            'timestep': int(step),
            'reward': reward,
            # 'mean_reward' as well as 'reward': gymnax names this column
            # mean_reward and the readers key off that name.
            'mean_reward': reward,
            'best_reward': best_reward,
            'elapsed_time': time.time() - start_time_metrics,
            **rl_diagnostics.extract(metrics),
            **pending_param_stats,
        }
        pending_param_stats.clear()
        training_metrics.append(record)
        report_missing(metrics)

        wandb.log({k: v for k, v in record.items() if v is not None})

        print(f"Step {step:10,} | Reward: {reward:8.2f} | Best: {best_reward:8.2f}")

    # Network factory.
    #
    # The policy is (128, 128) to match the NE trainers' MLPPolicy exactly
    # (source/studies/brax/train_GA_ant.py), so the methods search the same-sized policy:
    # 21,128 parameters on both sides. It was (256, 256) -- 77,072 params, 3.6x
    # the NE policy -- which confounded the RL-vs-NE gap on ant with a capacity
    # difference. The env-step budget was already matched in
    # block_brax_noncontinual; this matches the thing the budget is spent on.
    #
    # The value network is deliberately NOT matched: it has no NE counterpart, so
    # there is nothing to match it to, and shrinking the critic would handicap PPO
    # for no comparability gain. RL therefore still has more total trainable
    # parameters (150k vs 21k), and that difference is inherent to comparing a
    # critic-based method against a population rather than a free choice.
    #
    # The hidden activation is tanh for the same reason the width is (128, 128):
    # it is what the NE policy uses (ContinuousMLPPolicy, and
    # source/algorithms/networks.py POLICY_ARCH['brax']). It has to be passed --
    # make_ppo_networks defaults to swish, so leaving it off silently gave PPO a
    # different network from the one GA/ES/DNS were searching.
    #
    # Two asymmetries remain and belong in the caption, not in this file: PPO
    # normalizes observations (--normalize_observations, default True) and its
    # policy is a sampled NormalTanhDistribution with a learned scale, while the
    # NE policy is deterministic and tanh-squashed.
    # my_brax's factory, not brax's. brax's networks cannot report dormant
    # neurons at all, so `--use_redo` against them was a silent no-op: the old
    # `_apply_redo` tested `hasattr(net, 'apply_with_dormant_indices')`, found
    # False, and returned the parameters unchanged. Every stationary ant and
    # cheetah "redo" run was plain PPO. The continual trainers already switched
    # factory when ReDo was on; these did not.
    #
    # The sizes are flags rather than constants because this block is the
    # CONTROL for block_brax_continual, whose trainer is given 128,128 for BOTH
    # nets, and a control whose value net is twice as wide is not measuring the
    # same learner. The defaults below are the values that were hardcoded here,
    # so every existing stationary tree is reproduced unless a caller asks
    # otherwise.
    def _sizes(spec):
        return tuple(int(x) for x in str(spec).split(',') if x != '')

    network_factory = functools.partial(
        my_brax_networks.make_ppo_networks,
        policy_hidden_layer_sizes=_sizes(args.policy_hidden_sizes),
        value_hidden_layer_sizes=_sizes(args.value_hidden_sizes),
        activation=ACTIVATIONS[args.activation],
    )

    start_time = time.time()

    # Train PPO
    make_inference_fn, params, metrics = ppo_train(
        environment=env,
        num_timesteps=num_timesteps,
        num_envs=args.num_envs,
        episode_length=args.episode_length,
        learning_rate=args.learning_rate,
        entropy_cost=args.entropy_cost,
        discounting=args.discounting,
        unroll_length=args.unroll_length,
        batch_size=args.batch_size,
        num_minibatches=args.num_minibatches,
        num_updates_per_batch=args.num_updates_per_batch,
        normalize_observations=args.normalize_observations,
        reward_scaling=args.reward_scaling,
        action_repeat=args.action_repeat,
        num_evals=args.num_evals,
        num_eval_envs=args.num_eval_envs,
        max_grad_norm=args.max_grad_norm,
        deterministic_eval=args.deterministic_eval,
        network_factory=network_factory,
        seed=seed,
        progress_fn=progress_fn,
        policy_params_fn=policy_params_fn,
        use_trac=args.use_trac,
        use_redo=args.use_redo,
        redo_frequency=args.redo_frequency,
        redo_tau=args.redo_tau,
        track_dormant=bool(args.track_dormant),
        dormant_tau=args.dormant_tau,
        use_cchain=args.use_cchain,
        chain_target_rel_scale=args.chain_target_rel_scale,
        chain_warmup_iterations=args.chain_warmup_iterations,
        chain_coef_window=args.chain_coef_window,
    )

    total_time = time.time() - start_time
    print(f"\nTraining complete! Time: {total_time:.1f}s, Best: {best_reward:.2f}")

    # Save checkpoint
    ckpt_path = os.path.join(output_dir, f"{algo_name}_{env_name}_best.pkl")
    with open(ckpt_path, 'wb') as f:
        pickle.dump({
            'params': params,
            'best_reward': best_reward,
            'config': config,
        }, f)
    print(f"Saved: {ckpt_path}")

    # Save training metrics
    metrics_path = os.path.join(output_dir, "training_metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(training_metrics, f, indent=2)

    wandb.finish()
    print("Done!")


if __name__ == "__main__":
    main()
