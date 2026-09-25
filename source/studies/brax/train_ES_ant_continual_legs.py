"""OpenES on brax ant presented as a sequence of leg-damage sub-tasks.

The continual counterpart of train_ES_ant.py; see
train_GA_ant_continual_legs.py for why the algorithm itself is imported from the
noncontinual trainer rather than reimplemented, and for how the leg sequence is
built.

ES differs from GA and DNS in what the run's answer is: the population is a
cloud of perturbations used to estimate a gradient, and the solution is the
distribution mean. Sub-task checkpoints therefore store `es_state.mean`, not the
best sampled individual, and the mean is re-evaluated at each sub-task boundary
so the stored fitness is the mean's own rather than the best perturbation's.

Usage:
    python source/studies/brax/train_ES_ant_continual_legs.py --env ant --num_tasks 12 \
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
from jax.sharding import Mesh, NamedSharding, PartitionSpec
import optax
from evosax.algorithms import Open_ES
import time
import pickle
import wandb

from source.utils.runtime import Tee, write_run_config
from source.metrics.weight_stats import population_weight_stats, weight_stats
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
from source.studies.brax.behaviour_brax import AntFeet
from source.envs.brax_ant import LEG_NAMES, create_env_with_damaged_leg
from source.studies.brax.cli import add_continual_args, add_gif_args, frictions_from_args, flips_from_args, legs_from_args, speeds_from_args, gravities_from_args, save_task_gifs, task_label
from source.studies.brax.train_ES_ant import (
    create_policy_network,
    get_flat_params,
    make_scoring_fn,
    unflatten_params,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description='OpenES continual learning on brax ant (leg damage)')
    parser.add_argument('--env', type=str, default='ant', choices=['ant'])
    parser.add_argument('--gens_per_task', type=int, default=100,
                        help='Generations before the damaged leg changes')
    parser.add_argument('--pop_size', type=int, default=512)
    parser.add_argument('--sigma', type=float, default=0.02)
    parser.add_argument('--sigma_final', type=float, default=None,
                        help='Perturbation std at the end of the decay window. None '
                             'keeps sigma constant, which is what the noncontinual '
                             'ant block runs.')
    parser.add_argument('--sigma_schedule', type=str, default='sequence',
                        choices=['sequence', 'per_task'],
                        help='Which window sigma decays over when --sigma_final is '
                             'set. sequence: once across the whole run. per_task: '
                             'restarts at every sub-task boundary, so each sub-task '
                             'gets the same exploration budget.')
    parser.add_argument('--learning_rate', type=float, default=0.01)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--trial', type=int, default=1)
    parser.add_argument('--episode_length', type=int, default=1000)
    parser.add_argument('--num_evals', type=int, default=3,
                        help='Rollouts per individual, averaged into fitness')
    parser.add_argument('--eval_mean_interval', type=int, default=10,
                        help='Generations between scoring the distribution mean, '
                             'which is the solution OpenES actually returns. 0 '
                             'evaluates it only at the sub-task boundaries.')
    parser.add_argument('--gpus', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--wandb_project', type=str, default='brax_ant_continual_es')
    parser.add_argument('--run_name', type=str, default=None)
    parser.add_argument('--log_interval', type=int, default=10)
    parser.add_argument('--traj_steps', type=int, default=10,
                        help='Steps sub-sampled per episode for the AURORA encoder')
    add_continual_args(parser)
    add_gif_args(parser)
    add_diversity_args(parser)
    return parser.parse_args()


def main():
    args = parse_args()

    env_name = args.env
    pop_size = args.pop_size
    gens_per_task = args.gens_per_task
    num_tasks = args.num_tasks
    total_generations = num_tasks * gens_per_task
    episode_length = args.episode_length
    seed = args.seed
    trial = args.trial
    hidden_dims = (128, 128)

    legs = legs_from_args(args, seed)
    frictions = frictions_from_args(args, seed)
    flips = flips_from_args(args)
    speeds = speeds_from_args(args)
    gravities = gravities_from_args(args)

    output_dir = args.output_dir or (
        f"projects/brax/es_{env_name}_continual_legs/trial_{trial}")
    os.makedirs(output_dir, exist_ok=True)
    checkpoint_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    # `train.log`, as every gymnax trainer writes and no brax one did.
    sys.stdout = Tee(os.path.join(output_dir, "train.log"))

    steps_per_gen = pop_size * args.num_evals * episode_length

    print("=" * 60)
    print(f"OpenES on brax {env_name} (Continual, leg damage)")
    print("=" * 60)
    print(f"  Sub-tasks: {num_tasks} x {gens_per_task} gens = {total_generations}")
    print(f"  Backend: {args.backend}")
    print(f"  Leg order ({args.leg_order}): {[LEG_NAMES[l] for l in legs]}")
    print(f"  Friction ({args.friction_order}): {frictions}")
    print(f"  Speed ({args.speed_order}): {speeds}")
    print(f"  Gravity ({args.gravity_order}): {gravities}")
    print(f"  Population: {pop_size}, evals per individual: {args.num_evals}")
    print(f"  Env steps/generation: {steps_per_gen:,} "
          f"(total {steps_per_gen * total_generations:,})")

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
        'task_mod': 'leg_damage', 'num_tasks': num_tasks,
        'gens_per_task': gens_per_task, 'total_generations': total_generations,
        'leg_sequence': legs, 'leg_names': [LEG_NAMES[l] for l in legs],
        'friction_sequence': frictions, 'friction_order': args.friction_order,
        'speed_sequence': speeds, 'speed_order': args.speed_order,
        'gravity_sequence': gravities, 'gravity_order': args.gravity_order,
        'leg_order': args.leg_order, 'pop_size': pop_size, 'seed': seed,
        'trial': trial, 'num_evals': args.num_evals, 'episode_length': episode_length,
        'sigma': args.sigma, 'sigma_final': args.sigma_final,
        'sigma_schedule': args.sigma_schedule,
        'learning_rate': args.learning_rate,
        'hidden_dims': hidden_dims, 'track_diversity': track_diversity,
        'algorithm': 'es', 'continual': True,
    }
    run_name = args.run_name or f"es_{env_name}_continual_legs_trial{trial}"
    wandb.init(project=args.wandb_project, config=config, name=run_name, reinit=True)
    # The run says what network it searched, so
    # scripts/check_architectures.py can confirm every method compared on
    # this task used the same one. Was recorded only inside the pickles.
    write_run_config(output_dir, config, policy_arch='brax')

    devices = jax.devices()
    mesh = Mesh(np.array(devices), axis_names=('p',))
    replicate_sharding = NamedSharding(mesh, PartitionSpec())

    # Decayed over the whole sequence rather than per sub-task, for the reason
    # train_GA_ant_continual_legs.py gives. Constant by default, matching the
    # noncontinual ant block.
    if args.sigma_final is not None and args.sigma_final != args.sigma:
        if args.sigma_schedule == 'per_task':
            decay = (args.sigma_final / args.sigma) ** (1.0 / max(1, gens_per_task - 1))
            std_schedule = lambda gen: args.sigma * decay ** (gen % gens_per_task)
        else:
            decay = (args.sigma_final / args.sigma) ** (1.0 / max(1, total_generations - 1))
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

        # num_generations is the whole run, not one sub-task, so the snapshot
        # generations spread across the sequence rather than piling up in
        # sub-task 0.
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
    diversity_key = jax.random.key(seed + 1_000_000)
    probe_key = jax.random.key(seed + 2_000_000)

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
    plasticity_key = jax.random.key(seed + 3_000_000)
    # Third stream, same reason as the two above: rendering must not move the
    # search's random numbers, so a run is identical with GIFs on or off.
    gif_key = jax.random.key(seed + 3_000_000)

    best_overall_fitness = -float('inf')
    start_time = time.time()
    training_metrics = []  # Per-generation history, saved to training_metrics.json
    gen = 0  # Global generation index, continuous across sub-tasks.

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

        # The ES state -- mean, Adam moments, std schedule position -- carries
        # across the switch; only the environment is rebuilt.
        env = create_env_with_damaged_leg(env_name, leg_idx, episode_length,
                                          friction_mult, target_speed,
                                          args.speed_margin, args.speed_weight,
                                          backend=args.backend, flipped_leg=flips[task_idx],
                                          obs_noise_sigma=args.obs_noise_range,
                                          obs_task_period=args.task_period,
                                          obs_noise_seed=args.seed,
                                          task_idx=int(task_idx),
                                          gravity_mult=gravity_mult)
        feet = AntFeet.from_env(env) if track_diversity else None
        scoring_fn, rollout_with_trajectory = make_scoring_fn(
            env, policy, param_template, episode_length, num_evals=args.num_evals)
        scoring_fn_bd = make_scoring_fn(
            env, policy, param_template, episode_length, num_evals=args.num_evals,
            behaviour_cfg=behaviour_cfg, traj_steps=args.traj_steps, feet=feet,
        )[0] if track_diversity else None

        # Zero-shot transfer, measured before a single perturbation is drawn.
        # The carried policy is `es_state.mean`, not a population member: the
        # population is noise around the mean and is redrawn every generation,
        # and the mean is what the checkpoints store. See
        # source/metrics/zero_shot.py for why the first logged generation is not
        # this quantity.
        key, zs_key = jax.random.split(key)
        zero_shot = zero_shot_fitness(
            lambda genomes: scoring_fn(genomes, zs_key),
            es_state.mean if task_idx > 0 else None)
        if zero_shot is not None:
            print(f"  Zero-shot (carried mean of sub-task {task_idx - 1}): "
                  f"{zero_shot:.2f}", flush=True)

        task_best_fitness = -float('inf')

        for task_gen in range(gens_per_task):
            key, ask_key, eval_key, tell_key = jax.random.split(key, 4)

            population, es_state = es.ask(ask_key, es_state, es_params)
            # `or gen == 0`: the plasticity probe batch is frozen from the first
            # generation's visited states, as the diversity probes are.
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

            if gen_best > task_best_fitness:
                task_best_fitness = gen_best
            if gen_best > best_overall_fitness:
                best_overall_fitness = gen_best

            record = {
                "generation": gen, "task": task_idx,
                "task_generation": task_gen, "damaged_leg": leg_idx,
                "friction_mult": friction_mult,
                "target_speed": target_speed,
                "env_steps": (gen + 1) * steps_per_gen,
                "best_fitness": gen_best, "mean_fitness": gen_mean,
                "best_overall": best_overall_fitness,
                "best_task_fitness": task_best_fitness,
                "std": float(std_schedule(gen)),
            }
            if task_gen == 0:
                attach_zero_shot(record, zero_shot, task_idx - 1)

            # Weight statistics; see train_GA_ant_continual_legs.py for the
            # cadence and train_ES_ant.py for why ES records two sets -- the
            # sampled population and the distribution mean, which is the
            # solution OpenES actually returns.
            if (gen % weight_interval == 0 or task_gen == 0
                    or gen == total_generations - 1):
                record.update(population_weight_stats(jax.device_get(population)))
                record.update(weight_stats(jax.device_get(es_state.mean),
                                           prefix="mean_weight"))

            # Plasticity: elite churn across generations, within-population
            # churn, and dormancy. See source/metrics/plasticity.py.
            if plast is not None:
                if not plast.started() and extras is not None:
                    plast.start(plasticity_key, extras["observations"])
                # incumbent=: Open_ES has no parent/offspring relation -- the population
        # is mean + sigma*eps, pure exploration noise, and no sample descends
        # from another. Its ONE update is mean_t -> mean_t+1, which is the
        # closest NE analogue to a gradient step, so that is what ne_churn
        # measures here. See NEPlasticityTracker.update.
                record.update(plast.update(
                    jax.random.fold_in(plasticity_key, gen),
                    jax.device_get(population), jax.device_get(es_state.mean),
                    incumbent=jax.device_get(es_state.mean)))

            diversity = None
            if measure_now:
                if gen == 0:
                    tracker.start(probe_key, extras["observations"])
                diversity = tracker.update(
                    jax.random.fold_in(diversity_key, gen), gen, population,
                    extras["observations"], extras["behaviour"], fitness)
                if diversity:
                    record.update(diversity)

            # The population is only a set of noise samples used to estimate the
            # gradient; the distribution mean is the solution OpenES returns, so
            # score it directly every so often.
            mean_fitness = None
            if args.eval_mean_interval and task_gen % args.eval_mean_interval == 0:
                key, mean_eval_key = jax.random.split(key)
                mean_fitness = float(scoring_fn(es_state.mean[None, :], mean_eval_key)[0])
                record["mean_solution_fitness"] = mean_fitness

            wandb.log(record)
            # The key is written on every generation, None where the mean was not
            # scored, so training_metrics.json has one schema rather than two.
            training_metrics.append({
                'mean_solution_fitness': mean_fitness,
                **record,
                'elapsed_time': time.time() - start_time,
            })

            if task_gen % args.log_interval == 0 or task_gen == gens_per_task - 1:
                mean_str = f" | Mean sol: {mean_fitness:8.2f}" if mean_fitness is not None else ""
                line = (f"Task {task_idx} Gen {gen:5d} | Best: {gen_best:8.2f} "
                        f"| Mean: {gen_mean:8.2f} | Task best: {task_best_fitness:8.2f}"
                        f"{mean_str} | leg: {LEG_NAMES[leg_idx]} fric x{friction_mult:g}"
                        + (f" spd {target_speed:g}" if target_speed is not None else "")
                        + "")
                if diversity:
                    line += f" | {summarise(diversity)}"
                print(line, flush=True)

            gen += 1

        # The distribution mean is the solution ES has actually produced, so it
        # is what the checkpoint stores -- evaluated here, on this sub-task's
        # damage, rather than reusing a perturbation's fitness.
        task_mean = jnp.asarray(jax.device_get(es_state.mean))
        key, mean_eval_key = jax.random.split(key)
        mean_fitness_value = float(scoring_fn(task_mean[None, :], mean_eval_key)[0])

        ckpt_path = os.path.join(
            checkpoint_dir,
            f"task_{task_idx:02d}_"
            f"{task_label(leg_idx, friction_mult, target_speed, gravity_mult)}.pkl")
        with open(ckpt_path, 'wb') as f:
            pickle.dump({
                'flat_params': np.array(task_mean),
                'param_template': param_template,
                'best_fitness': mean_fitness_value,
                'best_sample_fitness': task_best_fitness,
                'task_idx': task_idx, 'task_mod': 'leg_damage_friction',
                'damaged_leg': leg_idx, 'friction_mult': friction_mult,
                'target_speed': target_speed,
                'generation': gen,
                'config': config,
            }, f)
        print(f"  Sub-task {task_idx} done (mean {mean_fitness_value:.2f}, "
              f"best sample {task_best_fitness:.2f}) -> {ckpt_path}")

        # Footage of what this sub-task ended up with, rendered before the next
        # iteration rebuilds the env -- rollout_with_trajectory is compiled
        # against the current damage. Drawn from its own stream so that
        # --gifs_per_task does not move the search's random numbers.
        save_task_gifs(
            env, rollout_with_trajectory, task_mean, output_dir,
            task_idx, leg_idx, episode_length,
            key=jax.random.fold_in(gif_key, task_idx),
            num_gifs=args.gifs_per_task, friction_mult=friction_mult,
            target_speed=target_speed, gravity_mult=gravity_mult)

    total_time = time.time() - start_time
    print(f"\nTraining complete! Time: {total_time:.1f}s over {gen} generations")
    print(f"  Best overall: {best_overall_fitness:.2f}")

    best_params = jnp.asarray(jax.device_get(es_state.mean))
    key, eval_key = jax.random.split(key)
    best_fitness = float(scoring_fn(best_params[None, :], eval_key)[0])
    print(f"  Final mean fitness: {best_fitness:.2f}")

    ckpt_path = os.path.join(output_dir, f"es_{env_name}_continual_legs_best.pkl")
    with open(ckpt_path, 'wb') as f:
        pickle.dump({
            'flat_params': np.array(best_params),
            'param_template': param_template,
            'best_fitness': best_fitness,
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
