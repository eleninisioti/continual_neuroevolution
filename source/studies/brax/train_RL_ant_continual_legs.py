"""
Train PPO on Brax Ant with Continual Learning (leg damage).

Same continual setup as source/studies/mujoco/train_RL_quadruped_continual.py, but on
the Brax Ant: every task damages one leg (the leg's actuators are disabled and
its joints are locked in place) and a different leg is damaged when the task
switches. The full training state (including optimizer) is preserved across
tasks.

The sub-task sequence, the damage wrapper and the flags that select them live in
source/envs/brax_ant.py, shared verbatim with the GA/ES/DNS ant
continual trainers so all four face the same sequence. By default it is the
deterministic cycle leg 1 -> 2 -> 3 -> 4 -> 1 ...; `--leg_order random` restores
the sampled sequence this file used before 2026-07-29.

Usage:
    python source/studies/brax/train_RL_ant_continual_legs.py --gpus 0
    python source/studies/brax/train_RL_ant_continual_legs.py --num_tasks 12 --gpus 0
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
from source.algorithms.networks import POLICY_ARCH  # one activation for the suite

def _get_gpu_arg():
    for i, arg in enumerate(sys.argv):
        if arg == '--gpus' and i + 1 < len(sys.argv):
            return sys.argv[i + 1]
    return None

_gpu_arg = _get_gpu_arg()
if _gpu_arg:
    os.environ['CUDA_VISIBLE_DEVICES'] = _gpu_arg
    print(f"Setting CUDA_VISIBLE_DEVICES={_gpu_arg}")

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
# Headless renderer for the sub-task GIFs, set before any mujoco import --
# the three NE ant trainers have always set it and this one had no need to
# until it started rendering. Without it brax's renderer falls through to
# GLFW and every boundary logs "an OpenGL platform library has not been
# loaded" instead of writing footage.
os.environ.setdefault("MUJOCO_GL", "egl")

import functools
import json
import pickle
import time

import jax
from brax.training.agents.ppo import networks as ppo_networks
import wandb

from source.envs.brax_ant import LEG_NAMES, create_env_with_damaged_leg
from source.studies.brax.cli import add_continual_args, add_gif_args, frictions_from_args, flips_from_args, legs_from_args, save_rl_task_gifs, speeds_from_args, gravities_from_args, task_label
from source.metrics import rl_diagnostics
from source.studies.brax.my_brax.cchain import add_chain_args
from source.studies.brax.my_brax.ppo_continual_train import train_continual


def parse_args():
    parser = argparse.ArgumentParser(description='PPO Continual Learning on Brax Ant (leg damage)')
    # Accepted for one reason: run_experiments.sh passes --env to every
    # trainer it drives. The leg-damage protocol is specific to the ant xml
    # (it resolves hip_N/ankle_N by name), so anything else is refused rather
    # than silently trained on the wrong body.
    parser.add_argument('--env', type=str, default='ant', choices=['ant'])
    parser.add_argument('--timesteps_per_task', type=int, default=50_000_000,
                        help='Timesteps per task (default 50M, matches the non-continual run)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--trial', type=int, default=1)
    parser.add_argument('--gpus', type=str, default=None)
    # --num_tasks, --leg_order and --allow_consecutive_legs, shared with the
    # NE continual ant trainers so all four see the same sub-task sequence.
    add_continual_args(parser)
    add_gif_args(parser)

    # PPO hyperparameters (same as the non-continual ant script)
    parser.add_argument('--num_envs', type=int, default=4096)
    parser.add_argument('--num_eval_envs', type=int, default=128)
    parser.add_argument('--episode_length', type=int, default=1000)
    parser.add_argument('--learning_rate', type=float, default=3e-4)
    parser.add_argument('--entropy_cost', type=float, default=1e-2)
    parser.add_argument('--discounting', type=float, default=0.97)
    parser.add_argument('--unroll_length', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=2048)
    parser.add_argument('--num_minibatches', type=int, default=32)
    parser.add_argument('--num_updates_per_batch', type=int, default=4)
    parser.add_argument('--normalize_observations', type=lambda x: x.lower() == 'true',
                        default=True, metavar='BOOL')
    parser.add_argument('--reward_scaling', type=float, default=10.0)
    parser.add_argument('--clipping_epsilon', type=float, default=0.3)
    parser.add_argument('--gae_lambda', type=float, default=0.95)
    parser.add_argument('--max_grad_norm', type=float, default=1.0)
    parser.add_argument('--action_repeat', type=int, default=1)

    # Network architecture.
    #
    # (128, 128) tanh is the ant policy every method searches: it is what
    # source/algorithms/networks.py POLICY_ARCH['brax'] says and what the NE trainers'
    # ContinuousMLPPolicy is. The default read 256,256 while every run on disk
    # was launched with an explicit 128,128 -- and the activation was not a flag
    # at all, so it fell through to make_ppo_networks' swish while GA/ES/DNS
    # searched a tanh network.
    parser.add_argument('--policy_hidden_sizes', type=str, default='128,128')
    parser.add_argument('--value_hidden_sizes', type=str, default='128,128',
                        help='Critic width. Not matched to anything: the value '
                             'network has no NE counterpart.')
    parser.add_argument('--activation', type=str,
                        default=POLICY_ARCH['brax']['activation'],
                        choices=['tanh', 'swish', 'relu'],
                        help='Hidden activation, shared by the policy and value '
                             'networks. Defaults to POLICY_ARCH["brax"], which '
                             'is what the NE trainers read, so the RL and NE '
                             'policies cannot drift apart from each other -- '
                             'they did, and PPO could not learn the ant as a '
                             'result. Must agree with the NE policy to keep the '
                             'comparison about the method.')

    # Eval and logging
    parser.add_argument('--num_evals_per_task', type=int, default=100)

    # Method variants
    parser.add_argument('--use_trac', action='store_true', default=False)
    parser.add_argument('--use_redo', action='store_true', default=False)
    parser.add_argument('--redo_frequency', type=int, default=10)
    parser.add_argument('--redo_tau', type=float, default=REDO_DEFAULT_TAU)
    # C-CHAIN at the reference implementation's continuous-control defaults;
    # the coefficient is reset at every leg change, which is the point of the
    # baseline here. Same flags as the cheetah continual block passes.
    add_chain_args(parser)
    parser.add_argument('--track_dormant', action='store_true', default=False)
    parser.add_argument('--dormant_tau', type=float, default=REDO_DEFAULT_TAU)

    # Output
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--wandb_project', type=str, default='brax_ant_continual')
    parser.add_argument('--run_name', type=str, default=None)

    return parser.parse_args()


def main():
    args = parse_args()

    policy_hidden_sizes = tuple(int(x) for x in args.policy_hidden_sizes.split(','))
    value_hidden_sizes = tuple(int(x) for x in args.value_hidden_sizes.split(','))
    activation_fn = {'tanh': jax.nn.tanh, 'swish': jax.nn.swish,
                     'relu': jax.nn.relu}[args.activation]

    num_tasks = args.num_tasks
    timesteps_per_task = args.timesteps_per_task

    # Which leg breaks in each task. `cycle` by default, so every method and
    # every trial faces the same damage at the same point in its budget; see
    # source/envs/brax_ant.py.
    leg_sequence = legs_from_args(args, args.seed)
    friction_sequence = frictions_from_args(args, args.seed)
    flip_sequence = flips_from_args(args)
    speed_sequence = speeds_from_args(args)
    gravity_sequence = gravities_from_args(args)

    if args.use_redo:
        algo_name = "redo_ppo"
    elif args.use_trac:
        algo_name = "trac_ppo"
    elif args.use_cchain:
        algo_name = "cchain_ppo"
    else:
        algo_name = "ppo"

    output_dir = args.output_dir or f"projects/brax/{algo_name}_ant_continual_legs/trial_{args.trial}"
    os.makedirs(output_dir, exist_ok=True)

    checkpoint_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    gen_checkpoint_dir = os.path.join(checkpoint_dir, "generations")
    os.makedirs(gen_checkpoint_dir, exist_ok=True)

    print("=" * 60)
    print("PPO Continual Learning on Brax Ant (leg damage)")
    print("=" * 60)
    print(f"  Algorithm: {algo_name}")
    print(f"  Number of tasks: {num_tasks}")
    print(f"  Timesteps per task: {timesteps_per_task:,}")
    print(f"  Total timesteps: {num_tasks * timesteps_per_task:,}")
    print(f"  Backend: {args.backend}")
    print(f"  Damaged leg sequence: {[LEG_NAMES[l] for l in leg_sequence]}")
    print(f"  Friction sequence ({args.friction_order}): {friction_sequence}")
    print(f"  Speed sequence ({args.speed_order}): {speed_sequence}")
    print(f"  Gravity sequence ({args.gravity_order}): {gravity_sequence}")
    print(f"  Output: {output_dir}")
    print(f"  *** Optimizer state preserved across tasks ***")

    config = {
        "algorithm": f"{algo_name}-Continual",
        "env_name": "ant",
        # Recorded because it changes what the numbers mean, not just how fast
        # they arrive: an mjx run and a generalized run are different physics.
        "backend": args.backend,
        "task_mod": "leg_damage_friction",
        "num_tasks": num_tasks,
        "timesteps_per_task": timesteps_per_task,
        "total_timesteps": num_tasks * timesteps_per_task,
        "leg_sequence": leg_sequence,
        "leg_names": [LEG_NAMES[l] for l in leg_sequence],
        "leg_order": args.leg_order,
        "friction_sequence": friction_sequence,
        "friction_order": args.friction_order,
        "speed_sequence": speed_sequence,
        "speed_order": args.speed_order,
        "gravity_sequence": gravity_sequence,
        "gravity_order": args.gravity_order,
        # The observation-offset axis. Recorded for the same reason `backend`
        # is: it changes what the numbers mean, and before this the ONLY record
        # that an obs-noise tree had noise at all was its directory name. A
        # run whose OBS_NOISE_RANGE was dropped by the shell produced a
        # config.json byte-identical to one that kept it.
        "obs_noise_sigma": args.obs_noise_range,
        "obs_task_period": args.task_period,
        "avoid_consecutive_legs": not args.allow_consecutive_legs,
        "num_envs": args.num_envs,
        "num_eval_envs": args.num_eval_envs,
        "episode_length": args.episode_length,
        "unroll_length": args.unroll_length,
        "num_minibatches": args.num_minibatches,
        "num_updates_per_batch": args.num_updates_per_batch,
        "learning_rate": args.learning_rate,
        "entropy_cost": args.entropy_cost,
        "discounting": args.discounting,
        "reward_scaling": args.reward_scaling,
        "clipping_epsilon": args.clipping_epsilon,
        "gae_lambda": args.gae_lambda,
        "policy_hidden_sizes": policy_hidden_sizes,
        "value_hidden_sizes": value_hidden_sizes,
        "activation": args.activation,
        "batch_size": args.batch_size,
        "max_grad_norm": args.max_grad_norm,
        # brax's running observation normalizer is part of the preserved
        # training state, so its sample count accumulates across sub-tasks and
        # never resets. On the obs-offset task that pools 12 differently
        # centred distributions into one estimate: the mean stays ~correct
        # (the offsets are zero-mean and cancel) while the variance absorbs
        # every offset, so the normalizer removes NONE of the offset in force
        # and attenuates the signal underneath it. Whether that is what the
        # obs-noise plasticity result is measuring is the question
        # run_ant_nonorm.sh exists to answer -- which it cannot do unless the
        # setting is on disk per run.
        "normalize_observations": args.normalize_observations,
        "seed": args.seed,
        "trial": args.trial,
        "continual": True,
        "optimizer_preserved": True,
        "use_trac": args.use_trac,
        "use_redo": args.use_redo,
        "use_cchain": args.use_cchain,
        "chain_target_rel_scale": args.chain_target_rel_scale,
        "chain_warmup_iterations": args.chain_warmup_iterations,
        "chain_coef_window": args.chain_coef_window,
        "track_dormant": args.track_dormant,
        "output_dir": output_dir,
    }
    run_name = args.run_name or f"{algo_name}_ant_continual_legs_trial{args.trial}"
    wandb.init(project=args.wandb_project, config=config, name=run_name, reinit=True)

    config_path = os.path.join(output_dir, "config.json")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

    network_factory = functools.partial(
        ppo_networks.make_ppo_networks,
        policy_hidden_layer_sizes=policy_hidden_sizes,
        value_hidden_layer_sizes=value_hidden_sizes,
        activation=activation_fn,
    )

    # train_continual identifies tasks by a float "multiplier"; here that is just
    # the task index, which selects the damaged leg for that task.
    task_multipliers = list(range(num_tasks))

    def env_factory(task_idx):
        damaged_leg = leg_sequence[int(task_idx)]
        flipped_leg = flip_sequence[int(task_idx)]
        friction_mult = friction_sequence[int(task_idx)]
        target_speed = speed_sequence[int(task_idx)]
        gravity_mult = gravity_sequence[int(task_idx)]
        # wrap=False: train_continual runs both of these through brax's
        # training.wrap, which supplies the Episode and AutoReset wrappers.
        # Letting envs.create supply them too made the evaluator average over
        # twice the true episode count and report half the real return.
        train_env = create_env_with_damaged_leg(
            'ant', damaged_leg, args.episode_length, friction_mult,
            target_speed, args.speed_margin, args.speed_weight, wrap=False,
            backend=args.backend, gravity_mult=gravity_mult,
            flipped_leg=flipped_leg,
            obs_noise_sigma=args.obs_noise_range, obs_noise_seed=args.seed,
            obs_task_period=args.task_period,
            task_idx=int(task_idx))
        eval_env = create_env_with_damaged_leg(
            'ant', damaged_leg, args.episode_length, friction_mult,
            target_speed, args.speed_margin, args.speed_weight, wrap=False,
            backend=args.backend, gravity_mult=gravity_mult,
            flipped_leg=flipped_leg,
            obs_noise_sigma=args.obs_noise_range, obs_noise_seed=args.seed,
            obs_task_period=args.task_period,
            task_idx=int(task_idx))
        return train_env, eval_env

    start_time = time.time()
    best_reward_overall = -float('inf')
    best_reward_per_task = {}
    training_metrics_list = []

    # Diagnostics land in training_metrics.json under the gymnax names, so one
    # reader serves both trees. The mapping itself lives in
    # source/metrics/rl_diagnostics.py, which the stationary ant trainer also uses:
    # this file used to carry a private copy of it, and the copy drifted --
    # ppo_continual_train emits `chain_reg_loss` and `chain_coef`, and the copy
    # only looked for `chain_p_reg_loss`, so every C-CHAIN continual run logged
    # None for its own regulariser. One mapping, one place to fix it.
    report_missing = rl_diagnostics.make_missing_reporter()

    def progress_fn(global_step, task_idx, multiplier, metrics):
        nonlocal best_reward_overall

        leg_idx = leg_sequence[int(multiplier)]
        friction_mult = friction_sequence[int(multiplier)]
        target_speed = speed_sequence[int(multiplier)]
        log_data = {
            "global_step": global_step,
            "task": task_idx,
            "damaged_leg": leg_idx,
            "friction_mult": friction_mult,
            "target_speed": target_speed,
        }
        log_data.update(metrics)

        if 'eval/episode_reward' in metrics:
            reward = metrics['eval/episode_reward']

            if task_idx not in best_reward_per_task:
                best_reward_per_task[task_idx] = -float('inf')
            if reward > best_reward_per_task[task_idx]:
                best_reward_per_task[task_idx] = reward
            if reward > best_reward_overall:
                best_reward_overall = reward

            log_data['fitness/best'] = float(reward)
            log_data['fitness/best_task'] = float(best_reward_per_task[task_idx])
            log_data['fitness/best_overall'] = float(best_reward_overall)

            elapsed = time.time() - start_time
            print(f"Task {task_idx+1} Step {global_step:>10,} | "
                  f"Reward: {reward:8.2f} | "
                  f"Best Task: {best_reward_per_task[task_idx]:8.2f} | "
                  f"Best Overall: {best_reward_overall:8.2f} | "
                  f"leg: {LEG_NAMES[leg_idx]} fric x{friction_mult:g}"
                  + (f" spd {target_speed:g}" if target_speed is not None else "")
                  + " | "
                  f"Time: {elapsed:6.1f}s", flush=True)

            training_metrics_list.append({
                'step': global_step,
                # 'generation' as well as 'step': every figure that puts an NE
                # and an RL curve on one axis keys off the generation-equivalent,
                # and deriving it here means the two trees agree on it rather
                # than each reader re-deriving it from a step count.
                'generation': metrics.get('generation'),
                'task': task_idx,
                'damaged_leg': leg_idx,
                'friction_mult': friction_mult,
                'target_speed': target_speed,
                'reward': float(reward),
                'mean_reward': float(reward),
                'best_task_reward': float(best_reward_per_task[task_idx]),
                'best_overall_reward': float(best_reward_overall),
                'elapsed_time': elapsed,
                **rl_diagnostics.extract(metrics),
            })
            report_missing(metrics)

        generation = metrics.get('generation', 0)
        log_data['generation'] = generation
        wandb.log(log_data, step=generation)

    def checkpoint_fn(task_idx, params_dict):
        seq_idx = int(params_dict.get('multiplier', task_idx))
        leg_idx = leg_sequence[seq_idx]
        friction_mult = friction_sequence[seq_idx]
        target_speed = speed_sequence[seq_idx]
        gravity_mult = gravity_sequence[seq_idx]
        checkpoint_path = os.path.join(
            checkpoint_dir,
            f"task_{task_idx:02d}_"
            f"{task_label(leg_idx, friction_mult, target_speed, gravity_mult)}.pkl")
        with open(checkpoint_path, 'wb') as f:
            pickle.dump({
                'normalizer_params': params_dict.get('normalizer_params'),
                'policy_params': params_dict.get('policy_params'),
                'value_params': params_dict.get('value_params'),
                'task_idx': task_idx,
                'damaged_leg': leg_idx,
                'friction_mult': friction_mult,
                'target_speed': target_speed,
                'gravity_mult': gravity_mult,
                'global_step': params_dict.get('global_step'),
                'config': {
                    'env_name': 'ant',
                    # The backend belongs here, not only in the run's
                    # config.json: a checkpoint that does not name its
                    # simulator gets evaluated on whatever the reader guesses,
                    # which is how an mjx policy was first scored on
                    # generalized physics it had never trained against.
                    'backend': args.backend,
                    'policy_hidden_sizes': policy_hidden_sizes,
                    'value_hidden_sizes': value_hidden_sizes,
                }
            }, f)
        print(f"  Checkpoint saved: {checkpoint_path}")

        # Footage of the policy this sub-task ended with, in the same layout
        # the NE trainers write (gifs/task_NN_<label>/trajectory_KK_*.gif) so
        # an RL run and a GA run can be flipped through side by side. Drawn
        # from its own key stream so --gifs_per_task cannot move the training
        # random numbers.
        save_rl_task_gifs(
            'ant', leg_idx, friction_mult, target_speed,
            params_dict.get('normalizer_params'),
            params_dict.get('policy_params'),
            policy_hidden_sizes, value_hidden_sizes,
            output_dir, task_idx, args.episode_length,
            activation=activation_fn,
            key=jax.random.fold_in(jax.random.key(args.seed + 10_000), task_idx),
            num_gifs=args.gifs_per_task,
            speed_margin=args.speed_margin, speed_weight=args.speed_weight,
            backend=args.backend, gravity_mult=gravity_mult)

    def generation_checkpoint_fn(generation, params_dict):
        checkpoint_path = os.path.join(gen_checkpoint_dir, f"gen_{generation:05d}.pkl")
        with open(checkpoint_path, 'wb') as f:
            pickle.dump({
                'normalizer_params': params_dict.get('normalizer_params'),
                'policy_params': params_dict.get('policy_params'),
                'value_params': params_dict.get('value_params'),
                'task_idx': params_dict.get('task_idx'),
                'damaged_leg': leg_sequence[int(params_dict.get('multiplier', 0))],
                'generation': generation,
                'global_step': params_dict.get('global_step'),
                'config': {
                    'env_name': 'ant',
                    'policy_hidden_sizes': policy_hidden_sizes,
                    'value_hidden_sizes': value_hidden_sizes,
                }
            }, f)
        print(f"  Generation checkpoint saved: {checkpoint_path}")

    make_inference_fn, params, final_metrics = train_continual(
        env_factory=env_factory,
        task_multipliers=task_multipliers,
        timesteps_per_task=timesteps_per_task,
        num_envs=args.num_envs,
        episode_length=args.episode_length,
        action_repeat=args.action_repeat,
        wrap_env_fn=None,
        learning_rate=args.learning_rate,
        entropy_cost=args.entropy_cost,
        discounting=args.discounting,
        unroll_length=args.unroll_length,
        batch_size=args.batch_size,
        num_minibatches=args.num_minibatches,
        num_updates_per_batch=args.num_updates_per_batch,
        normalize_observations=args.normalize_observations,
        reward_scaling=args.reward_scaling,
        clipping_epsilon=args.clipping_epsilon,
        gae_lambda=args.gae_lambda,
        max_grad_norm=args.max_grad_norm,
        network_factory=network_factory,
        seed=args.seed,
        num_eval_envs=args.num_eval_envs,
        num_evals_per_task=args.num_evals_per_task,
        use_trac=args.use_trac,
        use_redo=args.use_redo,
        redo_frequency=args.redo_frequency,
        redo_tau=args.redo_tau,
        use_cchain=args.use_cchain,
        # Was declared by add_chain_args and then never passed, so
        # --cchain_reset_on_switch was silently inert here while it worked in
        # source/studies/mujoco/train_RL_cheetah_continual.py. The default is off either
        # way, so no run's behaviour changes -- but the flag now does what it says.
        cchain_reset_on_switch=bool(args.cchain_reset_on_switch),
        chain_target_rel_scale=args.chain_target_rel_scale,
        chain_warmup_iterations=args.chain_warmup_iterations,
        chain_coef_window=args.chain_coef_window,
        track_dormant=args.track_dormant,
        dormant_tau=args.dormant_tau,
        progress_fn=progress_fn,
        checkpoint_fn=checkpoint_fn,
        generation_checkpoint_fn=generation_checkpoint_fn,
    )

    total_time = time.time() - start_time

    print("\n" + "=" * 60)
    print(f"Continual training complete! Time: {total_time:.1f}s")
    print(f"  Best reward overall: {best_reward_overall:.2f}")

    final_ckpt = os.path.join(output_dir, f"{algo_name}_ant_continual_legs_final.pkl")
    with open(final_ckpt, 'wb') as f:
        pickle.dump({
            'params': params,
            'best_reward_overall': best_reward_overall,
            'best_reward_per_task': best_reward_per_task,
            'leg_sequence': leg_sequence,
            'config': config,
        }, f)
    print(f"Saved: {final_ckpt}")

    metrics_path = os.path.join(output_dir, "training_metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(training_metrics_list, f, indent=2)

    wandb.finish()
    print("Done!")


if __name__ == "__main__":
    main()
