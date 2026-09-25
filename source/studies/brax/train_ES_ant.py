"""
Train OpenES on Brax Ant (non-continual).

Same evaluation setup as source/studies/brax/train_GA_ant.py (identical policy, rollout
and env-step accounting), so GA / OpenES / PPO curves are directly comparable.

Usage:
    python source/studies/brax/train_ES_ant.py --gpus 0
    python source/studies/brax/train_ES_ant.py --pop_size 512 --num_generations 500 --gpus 0
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
os.environ["MUJOCO_GL"] = "egl"

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec
import optax
from evosax.algorithms import Open_ES
import json
import time
import pickle
import wandb

from source.utils.runtime import Tee, write_run_config
from source.metrics.weight_stats import population_weight_stats, weight_stats
from source.metrics import plasticity
from source.algorithms.rl import redo as redo_mod
from source.algorithms.networks import ACTIVATIONS, POLICY_ARCH
import numpy as np
import imageio

from source.envs.brax_common import DEFAULT_BACKEND
from source.studies.brax.train_GA_ant import (
    create_env,
    create_policy_network,
    get_flat_params,
    make_scoring_fn,
    unflatten_params,
)
from source.metrics.behaviour_descriptors import BehaviourConfig
from source.metrics.behaviour_tracking import (
    PopulationDiversityTracker,
    add_diversity_args,
    summarise,
)
from source.studies.brax.behaviour_brax import AntFeet
from brax.io import image as brax_image


def parse_args():
    parser = argparse.ArgumentParser(description='OpenES on Brax Ant (Non-Continual)')
    parser.add_argument('--env', type=str, default='ant')
    parser.add_argument('--backend', type=str, default=DEFAULT_BACKEND,
                        choices=['mjx', 'generalized', 'spring', 'positional'],
                        help="Physics backend. Must match the continual block's, "
                             "which this run is the control for.")
    parser.add_argument('--num_generations', type=int, default=500)
    parser.add_argument('--pop_size', type=int, default=512,
                        help='Must be even (antithetic sampling)')
    # sigma/lr from a 40-generation sweep on ant (pop 256): 0.02/0.01 climbs
    # fastest, lr 0.03 diverges, sigma 0.05 stagnates, sigma 0.01 is slow.
    parser.add_argument('--sigma', type=float, default=0.02,
                        help='Std of the search distribution')
    parser.add_argument('--sigma_final', type=float, default=None,
                        help='Std at the last generation; sigma then decays '
                             'geometrically from --sigma. Default (unset) keeps '
                             'sigma constant, which is what the sweep tested.')
    parser.add_argument('--learning_rate', type=float, default=0.01,
                        help='Adam learning rate on the distribution mean')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--trial', type=int, default=1)
    parser.add_argument('--episode_length', type=int, default=1000)
    parser.add_argument('--num_evals', type=int, default=1,
                        help='Number of evaluations per individual (averaged)')
    parser.add_argument('--eval_mean_interval', type=int, default=10,
                        help='Evaluate the distribution mean every N generations '
                             '(0 disables). The mean is what OpenES actually optimizes.')
    parser.add_argument('--gpus', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--wandb_project', type=str, default='brax_ant_es')
    parser.add_argument('--run_name', type=str, default=None)
    parser.add_argument('--log_interval', type=int, default=10)
    parser.add_argument('--healthy_reward', type=float, default=None,
                        help="brax ant's per-step survival bonus (brax default "
                             "1.0). Standing still for a full episode is worth "
                             "1000 of it, which on mjx is a local optimum "
                             "evolution reaches at initialisation and does not "
                             "leave. 0 makes the reward pure locomotion, like "
                             "dm_control CheetahRun. Unset = brax default.")
    parser.add_argument('--save_gifs', action='store_true', default=False)
    parser.add_argument('--num_gifs', type=int, default=3,
                        help='Evaluation rollouts rendered at the end of training. '
                             'Each is a FRESH episode of the best genome and prints '
                             'its own return, so raising this also sharpens the '
                             'eval-vs-training-fitness comparison -- the training '
                             'number is a maximum over pop_size x num_evals noisy '
                             'rollouts and sits above a fresh-episode score by '
                             'construction. 3 is the historical default and is kept '
                             'so existing runs are reproduced.')
    parser.add_argument('--traj_steps', type=int, default=10,
                        help='Steps sub-sampled per episode for the AURORA encoder')
    add_diversity_args(parser)
    # Target-speed reward, so this block can run the continual block's
    # objective. Off by default, which reproduces every existing run here.
    parser.add_argument('--speed_target', type=float, default=None,
                        help='Track this speed instead of brax\'s unbounded '
                             'forward-velocity reward. Must match the continual '
                             'block\'s target for the two to be comparable.')
    parser.add_argument('--speed_margin', type=float, default=None,
                        help='Gaussian sigma of the speed-tracking credit.')
    parser.add_argument('--speed_weight', type=float, default=None,
                        help='Weight on the tracking term.')

    return parser.parse_args()


def main():
    args = parse_args()

    env_name = args.env
    pop_size = args.pop_size
    num_generations = args.num_generations
    episode_length = args.episode_length
    seed = args.seed
    trial = args.trial
    hidden_dims = (128, 128)

    if pop_size % 2 != 0:
        raise ValueError(f"pop_size must be even for antithetic sampling, got {pop_size}")

    output_dir = args.output_dir or f"projects/brax/es_{env_name}/trial_{trial}"
    os.makedirs(output_dir, exist_ok=True)

    # `train.log`, as every gymnax trainer writes and no brax one did.
    sys.stdout = Tee(os.path.join(output_dir, "train.log"))

    print("=" * 60)
    print(f"OpenES on Brax {env_name} (Non-Continual)")
    print("=" * 60)
    print(f"  Generations: {num_generations}")
    print(f"  Population: {pop_size}")
    print(f"  Sigma: {args.sigma}" +
          (f" -> {args.sigma_final}" if args.sigma_final is not None else " (constant)"))
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  Output: {output_dir}")

    key = jax.random.key(seed)

    # Create environment
    env = create_env(env_name, episode_length, backend=args.backend,
                     target_speed=args.speed_target,
                     speed_margin=args.speed_margin,
                     speed_weight=args.speed_weight,
                     healthy_reward=args.healthy_reward)
    key, reset_key = jax.random.split(key)
    state = env.reset(reset_key)
    obs_dim = state.obs.shape[-1]
    action_dim = env.action_size

    print(f"  Obs dim: {obs_dim}, Action dim: {action_dim}")

    # Create policy
    key, init_key = jax.random.split(key)
    policy, param_template = create_policy_network(init_key, obs_dim, action_dim, hidden_dims)
    flat_params = get_flat_params(param_template)
    num_params = flat_params.shape[0]
    print(f"  Network: {hidden_dims}, {num_params} params")

    steps_per_gen = pop_size * args.num_evals * episode_length
    print(f"  Env steps/generation: {steps_per_gen:,} "
          f"(total {steps_per_gen * num_generations:,})")

    # Behaviour tracking. Two scoring functions rather than one: collecting the
    # descriptors costs an extra pass over every rollout, so the cheap one runs
    # on the generations the tracker does not need (see tracker.needs).
    track_diversity = bool(args.track_diversity)
    feet = AntFeet.from_env(env) if track_diversity else None
    behaviour_cfg = BehaviourConfig(
        env_name=env_name, num_actions=int(action_dim),
        occupancy_bins=args.occupancy_bins) if track_diversity else None

    scoring_fn, rollout_with_trajectory = make_scoring_fn(
        env, policy, param_template, episode_length, num_evals=args.num_evals)
    scoring_fn_bd = make_scoring_fn(
        env, policy, param_template, episode_length, num_evals=args.num_evals,
        behaviour_cfg=behaviour_cfg, traj_steps=args.traj_steps, feet=feet,
    )[0] if track_diversity else None

    config = {
        'env': env_name, 'num_generations': num_generations,
        'pop_size': pop_size, 'seed': seed, 'trial': trial,
        'num_evals': args.num_evals, 'sigma': args.sigma,
        'sigma_final': args.sigma_final, 'learning_rate': args.learning_rate,
        'episode_length': episode_length, 'algorithm': 'openes',
        'track_diversity': track_diversity,
    }
    run_name = args.run_name or f"es_{env_name}_pop{pop_size}_trial{trial}"
    wandb.init(project=args.wandb_project, config=config, name=run_name, reinit=True)
    # The run says what network it searched, so
    # scripts/check_architectures.py can confirm every method compared on
    # this task used the same one. Was recorded only inside the pickles.
    write_run_config(output_dir, config, policy_arch='brax')

    # Initialize OpenES
    devices = jax.devices()
    mesh = Mesh(np.array(devices), axis_names=('p',))
    replicate_sharding = NamedSharding(mesh, PartitionSpec())

    if args.sigma_final is not None and args.sigma_final != args.sigma:
        decay = (args.sigma_final / args.sigma) ** (1.0 / max(1, num_generations - 1))
        std_schedule = lambda gen: args.sigma * decay ** gen
    else:
        std_schedule = lambda gen: args.sigma

    optimizer = optax.adam(learning_rate=args.learning_rate)

    es = Open_ES(
        population_size=pop_size,
        solution=jnp.zeros(num_params),
        std_schedule=std_schedule,
        optimizer=optimizer,
        use_antithetic_sampling=True,
    )
    es_params = jax.device_put(es.default_params, replicate_sharding)

    key, init_key, mean_key = jax.random.split(key, 3)
    # Start from a small random policy, matching the GA initialisation scale.
    init_mean = jax.random.normal(mean_key, (num_params,)) * 0.1
    es_state = es.init(init_key, init_mean, es_params)
    es_state = jax.device_put(es_state, replicate_sharding)

    tracker = None
    if track_diversity:
        def apply_flat(flat_params, obs_batch):
            return policy.apply(unflatten_params(flat_params, param_template), obs_batch)

        tracker = PopulationDiversityTracker(
            env_name=env_name, obs_dim=obs_dim, num_actions=int(action_dim),
            traj_steps=args.traj_steps, num_generations=num_generations,
            interval=args.diversity_interval, occupancy_bins=args.occupancy_bins,
            max_pairwise=args.diversity_max_pairwise,
            num_probe=args.num_probe_states, apply_flat=apply_flat,
            snapshots=args.behaviour_snapshots, seed=seed,
        )
    # Diversity draws from its own stream so that switching tracking on or off
    # leaves the search's random numbers -- and therefore the run -- unchanged.
    # Cadence for the weight statistics; see train_GA_ant.py for why it reuses
    # --diversity_interval rather than taking a flag of its own.
    weight_interval = max(1, int(args.diversity_interval))

    # --- Plasticity diagnostics, reported for every method (core/plasticity.py).
    # Churn and dormancy on a frozen probe batch, the same two columns the
    # gymnax trainers report, so NE and RL sit in one table. See
    # train_GA_ant.py for why continuous=True, action_range=2.0 and
    # max_pairwise=64 on the ant.
    #
    # Rides on --track_diversity because the frozen probe batch comes from the
    # behaviour-descriptor rollout, which is the only thing that returns
    # observations. Every paper block passes --track_diversity 1.
    _brax_activation = POLICY_ARCH['brax']['activation']
    plast = plasticity.NEPlasticityTracker(
        apply_flat=lambda f, o: policy.apply(unflatten_params(f, param_template), o),
        unflatten=lambda f: unflatten_params(f, param_template),
        num_hidden=len(hidden_dims),
        activation_fn=ACTIVATIONS[_brax_activation],
        criterion=redo_mod.criterion_for_activation(_brax_activation),
        num_probe=args.num_probe_states,
        max_pairwise=64,
        continuous=True,
        action_range=2.0,
    ) if track_diversity else None
    if plast is None:
        print("  (plasticity diagnostics off: they need --track_diversity 1)")
    plasticity_key = jax.random.key(seed + 3_000_000)

    diversity_key = jax.random.key(seed + 1_000_000)
    probe_key = jax.random.key(seed + 2_000_000)

    # Training loop
    best_overall_fitness = -float('inf')
    best_mean_fitness = -float('inf')
    best_mean_params = jax.device_get(es_state.mean)
    training_metrics = []
    start_time = time.time()

    print(f"\nStarting training...")

    for gen in range(num_generations):
        key, ask_key, eval_key, tell_key = jax.random.split(key, 4)

        population, es_state = es.ask(ask_key, es_state, es_params)
        # `or gen == 0`: the plasticity probe batch is frozen from the initial
        # population's visited states, as the diversity probes are.
        measure_now = ((tracker is not None and tracker.needs(gen))
                       or (gen == 0 and scoring_fn_bd is not None))
        if measure_now:
            fitness, extras = scoring_fn_bd(population, eval_key)
        else:
            fitness, extras = scoring_fn(population, eval_key), None
        # evosax minimizes, so negate fitness for maximization
        es_state, _ = es.tell(tell_key, population, -fitness, es_state, es_params)

        fitness_host = jax.device_get(fitness)
        gen_best = float(np.max(fitness_host))
        gen_mean = float(np.mean(fitness_host))

        if gen_best > best_overall_fitness:
            best_overall_fitness = gen_best

        env_steps = (gen + 1) * steps_per_gen
        log_data = {
            "generation": gen, "env_steps": env_steps, "best_fitness": gen_best,
            "mean_fitness": gen_mean, "best_overall": best_overall_fitness,
            "std": float(std_schedule(gen)),
        }

        # Weight statistics, on the diversity cadence and independent of
        # --track_diversity (see train_GA_ant.py). Two sets, because for ES the
        # population and the solution are not the same object: `population` is a
        # cloud of antithetic noise samples around the mean, so its pooled norms
        # carry the search noise, while `es_state.mean` IS what OpenES returns
        # and is the thing to compare against a GA genome. The population keys
        # match GA's and DNS's so one reader serves all three.
        if gen % weight_interval == 0 or gen == num_generations - 1:
            log_data.update(population_weight_stats(jax.device_get(population)))
            log_data.update(weight_stats(jax.device_get(es_state.mean),
                                         prefix="mean_weight"))

        # Plasticity. ES's "elite" is the DISTRIBUTION MEAN, not the best
        # sample: the mean is what OpenES returns and the only thing with a
        # stable identity from one generation to the next, which is what an
        # across-generation churn measure needs. The sampled population is a
        # cloud of antithetic noise and its best member changes arbitrarily.
        if plast is not None:
            if gen == 0 and measure_now and not plast.started():
                plast.start(plasticity_key, extras["observations"])
            # incumbent=: Open_ES has no parent/offspring relation -- the population
        # is mean + sigma*eps, pure exploration noise, and no sample descends
        # from another. Its ONE update is mean_t -> mean_t+1, which is the
        # closest NE analogue to a gradient step, so that is what ne_churn
        # measures here. See NEPlasticityTracker.update.
            log_data.update(plast.update(
                jax.random.fold_in(plasticity_key, gen),
                jax.device_get(population), jax.device_get(es_state.mean),
                incumbent=jax.device_get(es_state.mean)))

        diversity = None
        if measure_now and tracker is not None:
            if gen == 0:
                tracker.start(probe_key, extras["observations"])
            diversity = tracker.update(
                jax.random.fold_in(diversity_key, gen), gen, population,
                extras["observations"], extras["behaviour"], fitness)
            if diversity:
                log_data.update(diversity)

        # The population is only a set of noise samples used to estimate the
        # gradient; the distribution mean is the solution OpenES returns, so
        # score it directly every so often.
        mean_fitness = None
        if args.eval_mean_interval and (gen % args.eval_mean_interval == 0
                                        or gen == num_generations - 1):
            key, mean_eval_key = jax.random.split(key)
            current_mean = es_state.mean
            mean_fitness = float(scoring_fn(current_mean[None, :], mean_eval_key)[0])
            log_data["mean_solution_fitness"] = mean_fitness
            if mean_fitness > best_mean_fitness:
                best_mean_fitness = mean_fitness
                best_mean_params = jax.device_get(current_mean)

        training_metrics.append({
            **log_data,
            'mean_solution_fitness': mean_fitness,
            'elapsed_time': time.time() - start_time,
        })
        wandb.log(log_data)

        if gen % args.log_interval == 0 or gen == num_generations - 1:
            mean_str = f" | Mean sol: {mean_fitness:8.2f}" if mean_fitness is not None else ""
            line = (f"Gen {gen:4d} | Steps {env_steps:12,} | Best: {gen_best:8.2f} | "
                    f"Pop mean: {gen_mean:8.2f}{mean_str}")
            if diversity:
                line += f" | {summarise(diversity)}"
            print(line, flush=True)

    total_time = time.time() - start_time
    print(f"\nTraining complete! Time: {total_time:.1f}s")
    print(f"  Best population sample: {best_overall_fitness:.2f}")
    print(f"  Best distribution mean: {best_mean_fitness:.2f}")

    # Checkpoint the best distribution mean: that is the solution OpenES returns.
    ckpt_path = os.path.join(output_dir, f"es_{env_name}_best.pkl")
    with open(ckpt_path, 'wb') as f:
        pickle.dump({
            'flat_params': np.array(best_mean_params),
            'param_template': param_template,
            'best_fitness': best_mean_fitness,
            'best_population_fitness': best_overall_fitness,
            'final_mean': np.array(jax.device_get(es_state.mean)),
            'config': config,
        }, f)
    print(f"Saved: {ckpt_path}")

    metrics_path = os.path.join(output_dir, "training_metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(training_metrics, f, indent=2)

    # Trajectories at the snapshot generations, for the offline shared-encoder
    # analysis (behaviour_diversity_analysis.py). Per-run AURORA latents are not
    # comparable between runs; these are what make a shared space possible.
    if tracker is not None:
        snapshot_path = tracker.save(output_dir)
        if snapshot_path:
            print(f"Saved: {snapshot_path}")

    # Re-score the checkpointed solution on fresh episodes
    print(f"\nVerifying best mean on fresh episodes...")
    key, verify_key = jax.random.split(key)
    verify_fitness = scoring_fn(jnp.asarray(best_mean_params)[None, :], verify_key)
    print(f"  Reported best fitness: {best_mean_fitness:.2f}")
    print(f"  Verification fitness:  {float(verify_fitness[0]):.2f}")

    if args.save_gifs:
        try:
            gifs_dir = os.path.join(output_dir, "gifs")
            os.makedirs(gifs_dir, exist_ok=True)

            print(f"\nSaving {args.num_gifs} evaluation GIFs...")
            for gif_idx in range(args.num_gifs):
                key, gif_key = jax.random.split(key)
                total_reward, trajectory_states = rollout_with_trajectory(
                    jnp.asarray(best_mean_params), gif_key)
                total_reward = float(total_reward)

                trajectory_list = [jax.tree.map(lambda x: x[i], trajectory_states)
                                   for i in range(0, episode_length, 4)]

                images = brax_image.render_array(env.sys, trajectory_list,
                                                 height=240, width=320)
                gif_path = os.path.join(gifs_dir, f"trial{gif_idx}_reward{total_reward:.0f}.gif")
                imageio.mimsave(gif_path, images, fps=30, loop=0)
                print(f"  GIF {gif_idx}: reward={total_reward:.2f}")

            print(f"Saved GIFs to: {gifs_dir}")
        except Exception as e:
            print(f"Warning: Failed to save GIFs: {e}")

    wandb.finish()
    print("Done!")


if __name__ == "__main__":
    main()
