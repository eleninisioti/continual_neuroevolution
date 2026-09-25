"""Dominated Novelty Search on brax ant as a sequence of leg-damage sub-tasks.

The continual counterpart of train_DNS_ant.py; see
train_GA_ant_continual_legs.py for why the algorithm itself is imported from the
noncontinual trainer rather than reimplemented, and for how the leg sequence is
built.

The novelty descriptor is the ant's, not the cheetah's: where the ant ends up,
i.e. the torso's x/y position at the last step of the episode. There is no
AURORA path here because train_DNS_ant.py has none -- the ant's selection
descriptor was always the handcrafted endpoint -- so unlike
source/studies/mujoco/train_DNS_cheetah_continual.py there is no encoder to retrain at a
switch, and `--descriptor` is not a flag.

One thing happens at a sub-task boundary that has no analogue in GA or ES: the
carried population is re-evaluated on the new damage before any offspring are
produced. DNS selects on fitness *and* novelty jointly, so a population still
carrying the previous sub-task's fitnesses and endpoints would spend a
generation making dominance comparisons against numbers measured on a different
robot. The endpoints in particular are not transferable -- an ant that reached
(8, 3) on three legs will not reach it once a different leg dies.

Usage:
    python source/studies/brax/train_DNS_ant_continual_legs.py --env ant --num_tasks 12 \
        --gens_per_task 100 --pop_size 512 --num_evals 3 --gpus 0
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
import time
import pickle
import wandb

from source.utils.runtime import Tee, write_run_config
from source.metrics.weight_stats import population_weight_stats
from source.metrics import plasticity
from source.algorithms.rl import redo as redo_mod
from source.algorithms.networks import ACTIVATIONS, POLICY_ARCH
import json
import numpy as np

from source.metrics.behaviour_descriptors import BehaviourConfig
from source.metrics.behaviour_tracking import (
    PopulationDiversityTracker,
    add_diversity_args,
    summarise,
)
from source.metrics.zero_shot import attach as attach_zero_shot, zero_shot_fitness
# Dominated novelty search -- one implementation for every suite.
from source.algorithms.ne.dns import (
    _compute_dominated_novelty,
    compute_fitness_diversity,
    compute_genomic_diversity,
    dns_selection,
    isoline_variation,
)
from source.studies.brax.behaviour_brax import AntFeet
from source.envs.brax_ant import LEG_NAMES, create_env_with_damaged_leg
from source.studies.brax.cli import add_continual_args, add_gif_args, frictions_from_args, flips_from_args, legs_from_args, speeds_from_args, gravities_from_args, save_task_gifs, task_label
from source.studies.brax.train_DNS_ant import (
    create_policy_network,
    get_flat_params,
    make_scoring_fn,
    unflatten_params,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description='DNS continual learning on brax ant (leg damage)')
    parser.add_argument('--env', type=str, default='ant', choices=['ant'])
    parser.add_argument('--gens_per_task', type=int, default=100,
                        help='Generations before the damaged leg changes')
    parser.add_argument('--pop_size', type=int, default=512)
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Offspring per generation (default pop_size // 2, the '
                             'noncontinual ant trainer\'s value). The sweep passes '
                             'the full population so GA, ES and DNS spend the same '
                             'env steps per generation.')
    parser.add_argument('--k', type=int, default=3,
                        help='Number of neighbours the novelty is averaged over')
    parser.add_argument('--iso_sigma', type=float, default=0.005)
    parser.add_argument('--line_sigma', type=float, default=0.05)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--trial', type=int, default=1)
    parser.add_argument('--episode_length', type=int, default=1000)
    parser.add_argument('--num_evals', type=int, default=3,
                        help='Rollouts per individual, averaged into fitness')
    parser.add_argument('--gpus', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--wandb_project', type=str, default='brax_ant_continual_dns')
    parser.add_argument('--run_name', type=str, default=None)
    parser.add_argument('--log_interval', type=int, default=10)
    parser.add_argument('--traj_steps', type=int, default=10,
                        help='Steps sub-sampled per episode for the AURORA encoder '
                             'used by the diversity observer (not by selection)')
    add_continual_args(parser)
    add_gif_args(parser)
    add_diversity_args(parser)
    return parser.parse_args()


def main():
    args = parse_args()

    env_name = args.env
    pop_size = args.pop_size
    batch_size = args.batch_size if args.batch_size is not None else max(1, pop_size // 2)
    batch_size = min(batch_size, pop_size)
    gens_per_task = args.gens_per_task
    num_tasks = args.num_tasks
    total_generations = num_tasks * gens_per_task
    episode_length = args.episode_length
    seed = args.seed
    trial = args.trial
    k = args.k
    hidden_dims = (128, 128)

    legs = legs_from_args(args, seed)
    frictions = frictions_from_args(args, seed)
    flips = flips_from_args(args)
    speeds = speeds_from_args(args)
    gravities = gravities_from_args(args)

    output_dir = args.output_dir or (
        f"projects/brax/dns_{env_name}_continual_legs/trial_{trial}")
    os.makedirs(output_dir, exist_ok=True)
    checkpoint_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    # `train.log`, as every gymnax trainer writes and no brax one did.
    sys.stdout = Tee(os.path.join(output_dir, "train.log"))

    # DNS only rolls out the `batch_size` offspring within a sub-task: the
    # surviving parents keep the fitness they were scored with. The extra
    # population-wide evaluation at each boundary is charged separately below.
    steps_per_gen = batch_size * args.num_evals * episode_length
    steps_per_switch = pop_size * args.num_evals * episode_length

    print("=" * 60)
    print(f"DNS on brax {env_name} (Continual, leg damage)")
    print("=" * 60)
    print(f"  Sub-tasks: {num_tasks} x {gens_per_task} gens = {total_generations}")
    print(f"  Backend: {args.backend}")
    print(f"  Leg order ({args.leg_order}): {[LEG_NAMES[l] for l in legs]}")
    print(f"  Friction ({args.friction_order}): {frictions}")
    print(f"  Speed ({args.speed_order}): {speeds}")
    print(f"  Gravity ({args.gravity_order}): {gravities}")
    print(f"  Population: {pop_size}, batch: {batch_size}, evals: {args.num_evals}")
    print(f"  k (novelty neighbours): {k}")
    print(f"  iso_sigma {args.iso_sigma}, line_sigma {args.line_sigma}")
    print(f"  Env steps/generation: {steps_per_gen:,} "
          f"(+ {steps_per_switch:,} per sub-task boundary)")

    key = jax.random.key(seed)

    env = create_env_with_damaged_leg(env_name, legs[0], episode_length,
                                      frictions[0], speeds[0],
                                      args.speed_margin, args.speed_weight,
                                      backend=args.backend, flipped_leg=flips[0],
                                      gravity_mult=gravities[0])
    key, reset_key = jax.random.split(key)
    state = env.reset(reset_key)
    obs_dim = state.obs.shape[-1]
    action_dim = env.action_size
    print(f"  Obs dim: {obs_dim}, Action dim: {action_dim}")

    key, init_key = jax.random.split(key)
    policy, param_template = create_policy_network(init_key, obs_dim, action_dim, hidden_dims)
    num_params = get_flat_params(param_template).shape[0]
    print(f"  Network: {hidden_dims}, {num_params} params")

    track_diversity = bool(args.track_diversity)
    behaviour_cfg = BehaviourConfig(
        env_name=env_name, num_actions=int(action_dim),
        occupancy_bins=args.occupancy_bins) if track_diversity else None

    config = {
        'env': env_name, 'backend': args.backend,
        'task_mod': 'leg_damage_friction', 'num_tasks': num_tasks,
        'gens_per_task': gens_per_task, 'total_generations': total_generations,
        'leg_sequence': legs, 'leg_names': [LEG_NAMES[l] for l in legs],
        'leg_order': args.leg_order, 'friction_sequence': frictions,
        'friction_order': args.friction_order,
        'speed_sequence': speeds, 'speed_order': args.speed_order,
        'gravity_sequence': gravities, 'gravity_order': args.gravity_order, 'pop_size': pop_size,
        'batch_size': batch_size, 'k': k, 'iso_sigma': args.iso_sigma,
        'line_sigma': args.line_sigma, 'seed': seed, 'trial': trial,
        'num_evals': args.num_evals, 'episode_length': episode_length,
        'hidden_dims': hidden_dims, 'track_diversity': track_diversity,
        'descriptor': 'torso_xy', 'algorithm': 'dns', 'continual': True,
    }
    run_name = args.run_name or f"dns_{env_name}_continual_legs_trial{trial}"
    wandb.init(project=args.wandb_project, config=config, name=run_name, reinit=True)
    # The run says what network it searched, so
    # scripts/check_architectures.py can confirm every method compared on
    # this task used the same one. Was recorded only inside the pickles.
    write_run_config(output_dir, config, policy_arch='brax')

    key, pop_key = jax.random.split(key)
    population = random.normal(pop_key, (pop_size, num_params)) * 0.1

    tracker = None
    if track_diversity:
        def apply_flat(flat_params, obs_batch):
            return policy.apply(unflatten_params(flat_params, param_template), obs_batch)

        # num_generations is the whole run, not one sub-task, so the snapshot
        # generations spread across the sequence.
        tracker = PopulationDiversityTracker(
            env_name=env_name, obs_dim=obs_dim, num_actions=int(action_dim),
            traj_steps=args.traj_steps, num_generations=total_generations,
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
    # gymnax trainers report. See train_GA_ant.py for why continuous=True,
    # action_range=2.0 and max_pairwise=64 on the ant.
    #
    # The probe batch is frozen ONCE, at the first generation of sub-task 0, and
    # never refreshed -- the same trade the diversity probes make. It means late
    # churn is measured on states the ant no longer visits, which is deliberate:
    # a probe batch that moved with the sub-tasks would confound "the policy
    # changed" with "the state distribution changed", and the latter is exactly
    # what this block manipulates. See the module docstring.
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
    # Third stream, same reason as the two above: rendering must not move the
    # search's random numbers, so a run is identical with GIFs on or off.
    gif_key = random.key(seed + 3_000_000)

    best_overall_fitness = -float('inf')
    start_time = time.time()
    training_metrics = []  # Per-generation history, saved to training_metrics.json
    gen = 0  # Global generation index, continuous across sub-tasks.
    env_steps = 0  # Charged per generation, plus the boundary re-evaluations.

    # Carried across sub-tasks together with the population.
    fitnesses = descriptors = novelties = None
    final_best_params = None

    print(f"\nStarting training...")

    for task_idx, (leg_idx, friction_mult, target_speed, gravity_mult) in enumerate(
            zip(legs, frictions, speeds, gravities)):
        print("\n" + "=" * 60)
        print(f"SUB-TASK {task_idx}/{num_tasks - 1} | damaged {LEG_NAMES[leg_idx]} "
              f"| friction x{friction_mult:g}"
              + (f" | speed {target_speed:g}" if target_speed is not None else "")
              + (f" | gravity x{gravity_mult:g}" if gravity_mult != 1.0 else "")
              + f" | gen {gen}")
        print("=" * 60)

        env = create_env_with_damaged_leg(env_name, leg_idx, episode_length,
                                          friction_mult, target_speed,
                                          args.speed_margin, args.speed_weight,
                                          backend=args.backend, flipped_leg=flips[task_idx],
                                          obs_noise_sigma=args.obs_noise_range,
                                          obs_task_period=args.task_period,
                                          obs_noise_seed=args.seed,
                                          task_idx=int(task_idx),
                                          gravity_mult=gravity_mult)
        feet = AntFeet.from_env(env)
        scoring_fn, rollout_with_trajectory = make_scoring_fn(
            env, policy, param_template, episode_length, num_evals=args.num_evals,
            feet=feet, return_descriptor=True)
        scoring_fn_bd = make_scoring_fn(
            env, policy, param_template, episode_length, num_evals=args.num_evals,
            behaviour_cfg=behaviour_cfg, traj_steps=args.traj_steps, feet=feet,
            return_descriptor=True,
        )[0] if track_diversity else None

        # Re-measure the carried population under the new damage. On sub-task 0
        # this is simply the initial evaluation. See the module docstring on why
        # DNS needs this and GA/ES do not.
        key, eval_key = jax.random.split(key)
        fitnesses, descriptors = scoring_fn(population, eval_key)
        novelties = _compute_dominated_novelty(fitnesses, descriptors, k)
        env_steps += steps_per_switch

        # Zero-shot transfer, measured before a single offspring is produced.
        # See source/metrics/zero_shot.py for why the first logged generation is
        # not this quantity and cannot be substituted for it.
        key, zs_key = jax.random.split(key)
        zero_shot = zero_shot_fitness(
            lambda genomes: scoring_fn(genomes, zs_key), final_best_params)
        # DNS alone can report the carried POPULATION under the new damage for
        # free -- the re-measure above already is that. Kept separate from
        # `zero_shot_carried_best` and deliberately not computed for GA/ES,
        # where it would cost a full generation of env steps and distort the
        # budget the NE and RL curves are matched on. It is a best-of-512 order
        # statistic: comparable across DNS sub-tasks, NOT against another
        # method's carried-best or against an RL policy.
        pop_zero_shot_best = float(jnp.max(fitnesses))
        pop_zero_shot_mean = float(jnp.mean(fitnesses))
        if zero_shot is not None:
            print(f"  Zero-shot (carried best of sub-task {task_idx - 1}): "
                  f"{zero_shot:.2f} | carried population best "
                  f"{pop_zero_shot_best:.2f}, mean {pop_zero_shot_mean:.2f}",
                  flush=True)

        task_best_fitness = -float('inf')
        task_best_params = None

        for task_gen in range(gens_per_task):
            key, var_key = jax.random.split(key)
            # return_parents: see train_DNS_ant.py.
            offspring, _parents = isoline_variation(population, var_key, args.iso_sigma,
                                                    args.line_sigma, batch_size,
                                                    return_parents=True)

            key, eval_key = jax.random.split(key)
            offspring_fitnesses, offspring_descriptors = scoring_fn(offspring, eval_key)

            # `None` observations -- see the note in train_DNS_ant.py.
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

            final_best_params = population_host[best_idx].copy()
            task_best_params = final_best_params
            if gen_best > task_best_fitness:
                task_best_fitness = gen_best
            if gen_best > best_overall_fitness:
                best_overall_fitness = gen_best

            env_steps += steps_per_gen
            record = {
                "generation": gen, "task": task_idx,
                "task_generation": task_gen, "damaged_leg": leg_idx,
                "friction_mult": friction_mult,
                "target_speed": target_speed,
                "env_steps": env_steps,
                "best_fitness": gen_best, "mean_fitness": gen_mean,
                "best_overall": best_overall_fitness,
                "best_task_fitness": task_best_fitness,
                "fitness_diversity": compute_fitness_diversity(fitnesses),
                "genomic_diversity": compute_genomic_diversity(population),
                "mean_novelty": float(jnp.nanmean(novelties)),
            }
            if task_gen == 0:
                attach_zero_shot(record, zero_shot, task_idx - 1)
                record["zero_shot_carried_pop_best"] = pop_zero_shot_best
                record["zero_shot_carried_pop_mean"] = pop_zero_shot_mean

            # Weight statistics; see train_GA_ant_continual_legs.py. The
            # array here is the surviving archive, DNS's counterpart of GA's
            # population, not the offspring batch.
            if (gen % weight_interval == 0 or task_gen == 0
                    or gen == total_generations - 1):
                record.update(population_weight_stats(population_host))

            # Plasticity: elite churn across generations, within-population
            # churn, and dormancy. See source/metrics/plasticity.py.
            # DNS's `extras` only exists inside the diversity block below,
            # which re-evaluates the survivors on the tracker's own cadence. The
            # probe batch is therefore taken with one explicit re-evaluation at
            # the very first generation, as train_DNS_ant.py does.
            if plast is not None:
                if not plast.started():
                    key, plast_key = random.split(key)
                    _, _, plast_extras = scoring_fn_bd(population, plast_key)
                    plast.start(plasticity_key, plast_extras["observations"])
                record.update(plast.update(
                    random.fold_in(plasticity_key, gen), population_host,
                    population_host[best_idx],
                    parents=jax.device_get(_parents),
                    offspring=jax.device_get(offspring)))

            diversity = None
            if tracker is not None and tracker.needs(gen):
                # Re-evaluate the survivors rather than describing the offspring:
                # DNS carries a population across generations and only the
                # offspring were rolled out above.
                key, bd_key = random.split(key)
                _, _, extras = scoring_fn_bd(population, bd_key)
                if gen == 0:
                    tracker.start(probe_key, extras["observations"])
                diversity = tracker.update(
                    random.fold_in(diversity_key, gen), gen, population,
                    extras["observations"], extras["behaviour"], fitnesses)
                if diversity:
                    record.update(diversity)

            wandb.log(record)
            training_metrics.append({**record, 'elapsed_time': time.time() - start_time})

            if task_gen % args.log_interval == 0 or task_gen == gens_per_task - 1:
                line = (f"Task {task_idx} Gen {gen:5d} | Best: {gen_best:8.2f} "
                        f"| Mean: {gen_mean:8.2f} | Nov: {record['mean_novelty']:.3f} "
                        f"| leg: {LEG_NAMES[leg_idx]} fric x{friction_mult:g}"
                        + (f" spd {target_speed:g}" if target_speed is not None else "")
                        + "")
                if diversity:
                    line += f" | {summarise(diversity)}"
                print(line, flush=True)

            gen += 1

        ckpt_path = os.path.join(
            checkpoint_dir,
            f"task_{task_idx:02d}_"
            f"{task_label(leg_idx, friction_mult, target_speed, gravity_mult)}.pkl")
        with open(ckpt_path, 'wb') as f:
            pickle.dump({
                'flat_params': np.array(task_best_params),
                'param_template': param_template,
                'best_fitness': task_best_fitness,
                'task_idx': task_idx, 'task_mod': 'leg_damage_friction',
                'damaged_leg': leg_idx, 'friction_mult': friction_mult,
                'target_speed': target_speed,
                'generation': gen,
                'config': config,
            }, f)
        print(f"  Sub-task {task_idx} done (best {task_best_fitness:.2f}) -> {ckpt_path}")

        # Footage of what this sub-task ended up with, rendered before the next
        # iteration rebuilds the env -- rollout_with_trajectory is compiled
        # against the current damage. Drawn from its own stream so that
        # --gifs_per_task does not move the search's random numbers.
        save_task_gifs(
            env, rollout_with_trajectory, task_best_params, output_dir,
            task_idx, leg_idx, episode_length,
            key=random.fold_in(gif_key, task_idx),
            num_gifs=args.gifs_per_task, friction_mult=friction_mult,
            target_speed=target_speed, gravity_mult=gravity_mult)

    total_time = time.time() - start_time
    print(f"\nTraining complete! Time: {total_time:.1f}s over {gen} generations")
    print(f"  Best overall: {best_overall_fitness:.2f}")

    ckpt_path = os.path.join(output_dir, f"dns_{env_name}_continual_legs_best.pkl")
    with open(ckpt_path, 'wb') as f:
        pickle.dump({
            'flat_params': np.array(final_best_params),
            'param_template': param_template,
            'best_fitness': training_metrics[-1]['best_fitness'],
            'config': config,
        }, f)
    print(f"Saved: {ckpt_path}")

    # Written before the best-effort steps below: the sweep driver treats a run
    # with no training_metrics.json as failed.
    metrics_path = os.path.join(output_dir, "training_metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(training_metrics, f, indent=2)
    print(f"Saved: {metrics_path}")

    if tracker is not None:
        snapshot_path = tracker.save(output_dir)
        if snapshot_path:
            print(f"Saved: {snapshot_path}")

    wandb.finish()
    print("Done!")


if __name__ == "__main__":
    main()
