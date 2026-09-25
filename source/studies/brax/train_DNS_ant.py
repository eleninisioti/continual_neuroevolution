"""
Train DNS (Dominated Novelty Search) on Brax Ant (non-continual).

The ant DNS trainer: same algorithm as every other body --
Iso+Line-DD variation, then survivor selection by dominated novelty -- with the
ant's own novelty descriptor.

DNS's *selection* descriptor here is where the ant ends up: the torso's x/y
position at the last step of the episode. That is the standard novelty
descriptor for a maze/locomotion task, it is not the return (the return is
forward velocity integrated over the episode, so a fast ant that circles ends
far from a fast ant that runs straight), and it is deliberately a different
quantity from the foot duty factors the diversity tracker observes -- scoring
DNS's diversity on the descriptor it optimises would guarantee it wins that row.

Usage:
    python source/studies/brax/train_DNS_ant.py --gpus 0
    python source/studies/brax/train_DNS_ant.py --pop_size 512 --num_generations 500 --gpus 0
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
from jax import random
import json
import time
import pickle
import wandb

from source.utils.runtime import Tee, write_run_config
from source.metrics.weight_stats import population_weight_stats
from source.metrics import plasticity
from source.algorithms.rl import redo as redo_mod
from source.algorithms.networks import ACTIVATIONS, POLICY_ARCH
import numpy as np
import imageio
from brax.io import image as brax_image

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
# Dominated novelty search -- one implementation for every suite.
from source.algorithms.ne.dns import (
    _compute_dominated_novelty,
    compute_fitness_diversity,
    compute_genomic_diversity,
    dns_selection,
    isoline_variation,
)


def parse_args():
    parser = argparse.ArgumentParser(description='DNS on Brax Ant (Non-Continual)')
    parser.add_argument('--env', type=str, default='ant')
    parser.add_argument('--backend', type=str, default=DEFAULT_BACKEND,
                        choices=['mjx', 'generalized', 'spring', 'positional'],
                        help="Physics backend. Must match the continual block's, "
                             "which this run is the control for.")
    parser.add_argument('--num_generations', type=int, default=500)
    parser.add_argument('--pop_size', type=int, default=512)
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Offspring per generation (default: pop_size // 2)')
    parser.add_argument('--k', type=int, default=3,
                        help='Number of neighbours the novelty is averaged over')
    # iso/line sigma are the ant equivalents of the cheetah's 0.05/0.5. Ant is
    # far more sensitive to parameter noise -- the GA sweep on this env settled
    # on sigma 0.005 against the cheetah's 0.1 -- so the defaults here are set
    # by the ant tuning sweep rather than inherited.
    parser.add_argument('--iso_sigma', type=float, default=0.005)
    parser.add_argument('--line_sigma', type=float, default=0.05)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--trial', type=int, default=1)
    parser.add_argument('--episode_length', type=int, default=1000)
    parser.add_argument('--num_evals', type=int, default=3,
                        help='Number of evaluations per individual (averaged)')
    parser.add_argument('--gpus', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--wandb_project', type=str, default='brax_ant_dns')
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
    batch_size = args.batch_size if args.batch_size is not None else max(1, pop_size // 2)
    batch_size = min(batch_size, pop_size)
    num_generations = args.num_generations
    episode_length = args.episode_length
    seed = args.seed
    trial = args.trial
    k = args.k
    hidden_dims = (128, 128)

    output_dir = args.output_dir or f"projects/brax/dns_{env_name}/trial_{trial}"
    os.makedirs(output_dir, exist_ok=True)

    # `train.log`, as every gymnax trainer writes and no brax one did.
    sys.stdout = Tee(os.path.join(output_dir, "train.log"))

    print("=" * 60)
    print(f"DNS on Brax {env_name} (Non-Continual)")
    print("=" * 60)
    print(f"  Generations: {num_generations}")
    print(f"  Population: {pop_size}, Batch size: {batch_size}")
    print(f"  k (novelty neighbours): {k}")
    print(f"  iso_sigma {args.iso_sigma}, line_sigma {args.line_sigma}")
    print(f"  Output: {output_dir}")

    key = jax.random.key(seed)

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

    key, init_key = jax.random.split(key)
    policy, param_template = create_policy_network(init_key, obs_dim, action_dim, hidden_dims)
    flat_params = get_flat_params(param_template)
    num_params = flat_params.shape[0]
    print(f"  Network: {hidden_dims}, {num_params} params")

    # Env steps consumed per generation. Unlike GA and ES, DNS only rolls out
    # the `batch_size` offspring: the surviving parents keep the fitness they
    # were scored with. The generation count is what the sweep matches on, so
    # this is reported rather than corrected for.
    steps_per_gen = batch_size * args.num_evals * episode_length
    print(f"  Env steps/generation: {steps_per_gen:,} "
          f"(total {steps_per_gen * num_generations:,})")

    # Behaviour tracking. Kept in a second scoring function so that the cheap
    # one runs on the generations the tracker does not need (see tracker.needs).
    track_diversity = bool(args.track_diversity)
    feet = AntFeet.from_env(env)
    behaviour_cfg = BehaviourConfig(
        env_name=env_name, num_actions=int(action_dim),
        occupancy_bins=args.occupancy_bins) if track_diversity else None

    scoring_fn, rollout_with_trajectory = make_scoring_fn(
        env, policy, param_template, episode_length, num_evals=args.num_evals,
        feet=feet, return_descriptor=True)
    scoring_fn_bd = make_scoring_fn(
        env, policy, param_template, episode_length, num_evals=args.num_evals,
        behaviour_cfg=behaviour_cfg, traj_steps=args.traj_steps, feet=feet,
        return_descriptor=True,
    )[0] if track_diversity else None

    config = {
        'env': env_name, 'num_generations': num_generations,
        'pop_size': pop_size, 'batch_size': batch_size, 'k': k,
        'iso_sigma': args.iso_sigma, 'line_sigma': args.line_sigma,
        'seed': seed, 'trial': trial, 'num_evals': args.num_evals,
        'episode_length': episode_length, 'algorithm': 'dns',
        'track_diversity': track_diversity,
    }
    run_name = args.run_name or f"dns_{env_name}_pop{pop_size}_trial{trial}"
    wandb.init(project=args.wandb_project, config=config, name=run_name, reinit=True)
    # The run says what network it searched, so
    # scripts/check_architectures.py can confirm every method compared on
    # this task used the same one. Was recorded only inside the pickles.
    write_run_config(output_dir, config, policy_arch='brax')

    # Initial population
    key, pop_key = jax.random.split(key)
    population = random.normal(pop_key, (pop_size, num_params)) * 0.1

    print(f"\nEvaluating initial population...")
    key, eval_key = jax.random.split(key)
    fitnesses, descriptors = scoring_fn(population, eval_key)
    novelties = _compute_dominated_novelty(fitnesses, descriptors, k)
    print(f"  Initial best fitness: {float(jnp.max(fitnesses)):.2f}")

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
    diversity_key = random.key(seed + 1_000_000)
    probe_key = random.key(seed + 2_000_000)

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
    plasticity_key = random.key(seed + 3_000_000)

    best_overall_fitness = -float('inf')
    training_metrics = []
    start_time = time.time()

    print(f"\nStarting training...")

    for gen in range(num_generations):
        key, var_key = jax.random.split(key)
        # return_parents: x1, the iso base. Taken before dns_selection merges
        # parents and offspring, after which nothing is in correspondence.
        offspring, _parents = isoline_variation(population, var_key, args.iso_sigma,
                                                args.line_sigma, batch_size,
                                                return_parents=True)

        key, eval_key = jax.random.split(key)
        offspring_fitnesses, offspring_descriptors = scoring_fn(offspring, eval_key)

        # `None` observations: the ant descriptors are hand-designed (AntFeet
        # contacts), so no trajectory needs to travel with a survivor. With
        # them None the shared selection is numerically what the brax-local
        # copy did -- it just also returns the (empty) observation slot.
        population, fitnesses, descriptors, _, novelties = dns_selection(
            population, fitnesses, descriptors, None,
            offspring, offspring_fitnesses, offspring_descriptors, None,
            pop_size, k,
        )

        fitness_host = jax.device_get(fitnesses)
        population_host = jax.device_get(population)

        gen_best = float(np.max(fitness_host))
        gen_mean = float(np.mean(fitness_host))
        best_idx = int(np.argmax(fitness_host))

        final_best_fitness = gen_best
        final_best_params = population_host[best_idx].copy()

        if gen_best > best_overall_fitness:
            best_overall_fitness = gen_best

        env_steps = (gen + 1) * steps_per_gen
        record = {
            "generation": gen, "env_steps": env_steps, "best_fitness": gen_best,
            "mean_fitness": gen_mean, "best_overall": best_overall_fitness,
            "fitness_diversity": compute_fitness_diversity(fitnesses),
            "genomic_diversity": compute_genomic_diversity(population),
            "mean_novelty": float(jnp.nanmean(novelties)),
        }

        # Weight statistics, on the diversity cadence and independent of
        # --track_diversity (see train_GA_ant.py). `population` here is the
        # surviving archive after selection, not the offspring batch, which is
        # what makes it the counterpart of GA's population.
        if gen % weight_interval == 0 or gen == num_generations - 1:
            record.update(population_weight_stats(jax.device_get(population)))

        # Plasticity. DNS's probe batch needs observations, and its BD path
        # re-evaluates the survivors below; generation 0 is forced through it so
        # the batch can be frozen there, exactly as the diversity probes are.
        if plast is not None and gen == 0 and not plast.started():
            key, plast_key = random.split(key)
            _, _, plast_extras = scoring_fn_bd(population, plast_key)
            plast.start(plasticity_key, plast_extras["observations"])
        if plast is not None:
            record.update(plast.update(
                random.fold_in(plasticity_key, gen), population_host,
                population_host[best_idx],
                parents=jax.device_get(_parents),
                offspring=jax.device_get(offspring)))

        diversity = None
        if tracker is not None and tracker.needs(gen):
            # Re-evaluate the survivors rather than describing the offspring:
            # DNS carries a population across generations and only the offspring
            # were rolled out above, so the offspring's descriptors are not the
            # population's. This is the extra cost of tracking DNS, and it is
            # why it is paid only on the generations the tracker asks for.
            key, reeval_key = random.split(key)
            _, _, extras = scoring_fn_bd(population, reeval_key)
            if gen == 0:
                tracker.start(probe_key, extras["observations"])
            diversity = tracker.update(
                random.fold_in(diversity_key, gen), gen, population,
                extras["observations"], extras["behaviour"], fitnesses)
            if diversity:
                record.update(diversity)

        wandb.log(record)
        training_metrics.append({**record, 'elapsed_time': time.time() - start_time})

        if gen % args.log_interval == 0 or gen == num_generations - 1:
            line = (f"Gen {gen:4d} | Steps {env_steps:12,} | Best: {gen_best:8.2f} "
                    f"| Mean: {gen_mean:8.2f} | Overall: {best_overall_fitness:8.2f} "
                    f"| Nov: {record['mean_novelty']:.3f}")
            if diversity:
                line += f" | {summarise(diversity)}"
            print(line, flush=True)

    total_time = time.time() - start_time
    print(f"\nTraining complete! Time: {total_time:.1f}s")
    print(f"  Cached best overall: {best_overall_fitness:.2f}")

    # Re-evaluate the final population: the carried fitnesses are the scores the
    # survivors were selected on, which are optimistically biased.
    print(f"\nRe-evaluating final population for accurate fitness...")
    key, reeval_key = jax.random.split(key)
    final_fitnesses, _ = scoring_fn(population, reeval_key)
    final_best_idx = int(jnp.argmax(final_fitnesses))
    population_host = jax.device_get(population)
    best_params = population_host[final_best_idx].copy()
    best_fitness = float(final_fitnesses[final_best_idx])
    print(f"  Re-evaluated best fitness: {best_fitness:.2f}")

    ckpt_path = os.path.join(output_dir, f"dns_{env_name}_best.pkl")
    with open(ckpt_path, 'wb') as f:
        pickle.dump({
            'flat_params': np.array(best_params),
            'param_template': param_template,
            'best_fitness': best_fitness,
            'best_overall_fitness': best_overall_fitness,
            'final_gen_flat_params': np.array(final_best_params),
            'final_gen_fitness': final_best_fitness,
            'config': config,
        }, f)
    print(f"Saved: {ckpt_path}")

    # Written before the GIF step below, which is best-effort: the sweep driver
    # treats a run with no training_metrics.json as failed.
    metrics_path = os.path.join(output_dir, "training_metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(training_metrics, f, indent=2)
    print(f"Saved: {metrics_path}")

    # Trajectories at the snapshot generations, for the offline shared-encoder
    # analysis (behaviour_diversity_analysis.py). Per-run AURORA latents are not
    # comparable between runs; these are what make a shared space possible.
    if tracker is not None:
        snapshot_path = tracker.save(output_dir)
        if snapshot_path:
            print(f"Saved: {snapshot_path}")

    if args.save_gifs:
        try:
            gifs_dir = os.path.join(output_dir, "gifs")
            os.makedirs(gifs_dir, exist_ok=True)

            print(f"\nSaving {args.num_gifs} evaluation GIFs...")
            for gif_idx in range(args.num_gifs):
                key, gif_key = jax.random.split(key)
                total_reward, trajectory_states = rollout_with_trajectory(
                    jnp.asarray(best_params), gif_key)
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
