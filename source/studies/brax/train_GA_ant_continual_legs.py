"""SimpleGA on brax ant presented as a sequence of leg-damage sub-tasks.

The continual counterpart of train_GA_ant.py, and the ant counterpart of
source/studies/mujoco/train_GA_cheetah_continual.py. A single GA state runs across
`num_tasks` sub-tasks of `gens_per_task` generations; a different leg is
disabled at every boundary and the learner is never told.

Everything except the task loop is imported from train_GA_ant.py -- the policy,
the scoring function, the behaviour descriptors. That is deliberate: the
brax_noncontinual block is this block's control, so any drift between the two
implementations would show up in the figures as a setting effect. The GA's own
hyperparameters (elite ratio, mutation std and its decay) are therefore the
noncontinual trainer's.

Two ant-specific details that the cheetah version has no analogue for:

  * `--sigma_schedule` chooses the window the mutation std decays over.
    'sequence' (the default, and what the pre-2026-07-29 runs used) decays it
    once across all num_tasks*gens_per_task generations, inheriting
    train_GA_ant.py's schedule wholesale. That turned out to be the wrong shape
    for a continual run -- each sub-task explores less than the one before, and
    GA's per-sub-task best fell 1908 -> 1108 -> 897 -> 650 -> 278 across the
    first five sub-tasks of the 2026-07-29 sweep while DNS held ~2000.
    'per_task' restarts the decay at every boundary so each sub-task gets the
    same exploration budget; 'const' disables decay entirely.
  * The elite archive carries across a switch like the rest of the state, and
    its stored fitnesses were measured under the PREVIOUS damage.
    `--reeval_archive` re-scores the archive on the new sub-task at every
    boundary and re-sorts it. The genomes carry over; only the numbers attached
    to them are refreshed.

    DEFAULT OFF SINCE 2026-09-08, where it was on from 2026-08-23. Doing it
    requires knowing where the boundary is, and no other method in the
    comparison is told that -- so with it on, the GA row was the only arm
    running a schedule-aware algorithm. The boundary-FREE way to get the same
    honesty is to re-score the archive EVERY generation out of the same
    evaluation budget, which is `--refresh_archive` in
    source/studies/gymnax/train_GA_gymnax_continual.py and `GASearcher(refresh=True)`
    in source/studies/generalists/ne.py. That is not implemented here yet, so this
    trainer now carries a stale archive by default and the freeze described
    below is live again. `--reeval_archive 1` restores the boundary re-score.

    This paragraph used to argue the opposite -- that a stale-but-high score
    holds a slot for only "a few generations" until an offspring beats it, and
    is therefore a property of (mu+lambda) elitism rather than a bug. That
    argument depends on consecutive sub-tasks scoring on the SAME SCALE, which
    is true for leg damage (every variant returns hundreds) and false in
    general. `tell` concatenates the new fitnesses with `state.fitness` and
    keeps the smallest, so if a later sub-task's achievable score is
    systematically worse than an earlier one's, NO offspring can ever displace
    the stale entries and the archive freezes for the rest of the run --
    permanently, not for a few generations. That was measured in the scheduling
    suite (dropped 2026-09-08), where an objective switch moves fitness from ~1.44 to ~0.43: the
    elite's evaluation was byte-identical for 74 consecutive generations across
    a whole sub-task, and the run reported a forgetting of exactly 0.000 that
    meant "the search stopped", not "the policy held up".

    Leg damage is the benign case, so this fix changed ant numbers little --
    but "little" is a measurement, not an assumption, and the failure mode is
    silent when it does bite. Runs between 2026-08-23 and 2026-09-08 were made
    with `--reeval_archive` on and record it in their config.

    ES needs none of this: Open_ES keeps a mean, not an archive.

Usage:
    python source/studies/brax/train_GA_ant_continual_legs.py --env ant --num_tasks 12 \
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
from source.algorithms.ne.ga import SimpleGA
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
from source.studies.brax.behaviour_brax import AntFeet
from source.envs.brax_ant import LEG_NAMES, create_env_with_damaged_leg
from source.studies.brax.cli import add_continual_args, add_gif_args, frictions_from_args, flips_from_args, legs_from_args, speeds_from_args, gravities_from_args, save_task_gifs, task_label
# The algorithm itself, unchanged from the noncontinual block.
from source.studies.brax.train_GA_ant import (
    create_policy_network,
    get_flat_params,
    make_scoring_fn,
    unflatten_params,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description='SimpleGA continual learning on brax ant (leg damage)')
    # The leg-damage protocol resolves hip_N/ankle_N by name, so anything other
    # than the ant is refused rather than silently trained on the wrong body.
    parser.add_argument('--env', type=str, default='ant', choices=['ant'])
    parser.add_argument('--gens_per_task', type=int, default=100,
                        help='Generations before the damaged leg changes')
    parser.add_argument('--pop_size', type=int, default=512)
    parser.add_argument('--elite_ratio', type=float, default=0.1)
    parser.add_argument('--reeval_archive', type=int, default=0,
                        help='Re-score the elite archive on the new sub-task at '
                             'every boundary, so `tell` compares like with like. '
                             'OFF by default since 2026-09-08: it needs to know '
                             'where the boundary is, and no other method here '
                             'is told. 1 reproduces runs made between '
                             '2026-08-23 and 2026-09-08.')
    parser.add_argument('--sigma', type=float, default=0.01)
    parser.add_argument('--sigma_final', type=float, default=0.002,
                        help='Mutation std at the end of the decay window; sigma '
                             'decays geometrically from --sigma to this value. Set '
                             'equal to --sigma for a constant schedule.')
    parser.add_argument('--sigma_schedule', type=str, default='sequence',
                        choices=['sequence', 'per_task', 'const'],
                        help="Which window sigma decays over. sequence: once across "
                             "all num_tasks*gens_per_task generations (the "
                             "pre-2026-07-29 behaviour). per_task: restarts at every "
                             "sub-task boundary, so each sub-task gets the same "
                             "exploration budget. const: no decay, ignores "
                             "--sigma_final.")
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--trial', type=int, default=1)
    parser.add_argument('--episode_length', type=int, default=1000)
    parser.add_argument('--num_evals', type=int, default=3,
                        help='Rollouts per individual, averaged into fitness')
    parser.add_argument('--gpus', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--wandb_project', type=str, default='brax_ant_continual_ga')
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
        f"projects/brax/ga_{env_name}_continual_legs/trial_{trial}")
    os.makedirs(output_dir, exist_ok=True)
    checkpoint_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    # `train.log`, as every gymnax trainer writes and no brax one did.
    sys.stdout = Tee(os.path.join(output_dir, "train.log"))

    steps_per_gen = pop_size * args.num_evals * episode_length

    print("=" * 60)
    print(f"SimpleGA on brax {env_name} (Continual, leg damage)")
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

    # Sub-task 0's environment, used to fix the observation/action dimensions.
    # Damage changes neither -- the dead actuators keep their slots and the
    # frozen joints keep reporting angles -- so the policy is built once.
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
        'gravity_sequence': gravities, 'gravity_order': args.gravity_order, 'pop_size': pop_size, 'seed': seed,
        'trial': trial, 'num_evals': args.num_evals, 'episode_length': episode_length,
        'elite_ratio': args.elite_ratio,
        'reeval_archive': bool(args.reeval_archive),
        'sigma': args.sigma, 'sigma_final': args.sigma_final,
        'sigma_schedule': args.sigma_schedule,
        'hidden_dims': hidden_dims, 'track_diversity': track_diversity,
        'algorithm': 'ga', 'continual': True,
    }
    run_name = args.run_name or f"ga_{env_name}_continual_legs_trial{trial}"
    wandb.init(project=args.wandb_project, config=config, name=run_name, reinit=True)
    # The run says what network it searched, so
    # scripts/check_architectures.py can confirm every method compared on
    # this task used the same one. Was recorded only inside the pickles.
    write_run_config(output_dir, config, policy_arch='brax')

    devices = jax.devices()
    mesh = Mesh(np.array(devices), axis_names=('p',))
    replicate_sharding = NamedSharding(mesh, PartitionSpec())

    # The GA's own sigma is continuous across sub-tasks, because the GA state
    # carries across the boundaries; 'per_task' is the schedule that restarts
    # it, via ga.reset_sigma() at each switch below.
    #
    # 'sequence' is the original schedule and is kept as the default only so the
    # earlier runs reproduce. It decays sigma monotonically over all 1200
    # generations, which means each sub-task explores less than the one before
    # it: in the 2026-07-29 trial-1 sweep GA's per-sub-task best fell 1908 ->
    # 1108 -> 897 -> 650 -> 278 while DNS held ~2000 on the same sequence. The
    # module docstring of this file used to argue that restarting the decay
    # would "re-inject generation-0 noise into a converged population twelve
    # times"; re-injecting noise into a converged population at a task boundary
    # is the point in a continual setting, and the argument was inherited from
    # the noncontinual trainer, where there are no boundaries.
    if args.sigma_schedule == 'const':
        decay_window = None
    elif args.sigma_schedule == 'per_task':
        decay_window = gens_per_task
    else:
        decay_window = total_generations
    restart_sigma_each_task = args.sigma_schedule == 'per_task'

    sigma_decay = (
        1.0 if decay_window is None
        else SimpleGA.decay_for(args.sigma, args.sigma_final, decay_window)
    )

    ga = SimpleGA(
        popsize=pop_size,
        num_dims=num_params,
        elite_ratio=args.elite_ratio,
        sigma_init=args.sigma,
        sigma_decay=sigma_decay,
        sigma_limit=args.sigma_final if args.sigma_final is not None else args.sigma,
    )
    ga_params = jax.device_put(ga.default_params, replicate_sharding)

    key, state_init_key = jax.random.split(key)
    ga_state = ga.init(state_init_key, ga_params)
    ga_state = jax.device_put(ga_state, replicate_sharding)

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

        # Rebuild the environment and everything closed over it. The GA state is
        # NOT rebuilt -- carrying it across the switch is what makes this
        # continual rather than a sequence of independent runs.
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

        # Zero-shot transfer, measured before a single offspring is produced.
        # See source/metrics/zero_shot.py for why the first logged generation is
        # not this quantity and cannot be substituted for it.
        key, zs_key = jax.random.split(key)
        zero_shot = zero_shot_fitness(
            lambda genomes: scoring_fn(genomes, zs_key), final_best_params)
        if zero_shot is not None:
            print(f"  Zero-shot (carried best of sub-task {task_idx - 1}): "
                  f"{zero_shot:.2f}", flush=True)

        # The archive carries over, but its stored fitnesses were measured on
        # the PREVIOUS sub-task and `tell` will compare them against fitnesses
        # from this one. Re-score them so the comparison is like with like; the
        # genomes are untouched. See the module docstring for what goes wrong
        # without this.
        #
        # Costs elite_ratio * pop_size * num_evals episodes per boundary -- a
        # tenth of a generation at the default elite_ratio, so roughly 0.1% of a
        # 12x100 run. Not enough to disturb the compute match with PPO.
        if task_idx > 0 and args.reeval_archive:
            key, reval_key = jax.random.split(key)
            stale_best = float(-ga_state.fitness[0])
            reval_fitness = scoring_fn(ga_state.archive, reval_key)
            neg_fitness = -reval_fitness
            order = jnp.argsort(neg_fitness)
            reordered = ga_state.archive[order]
            ga_state = ga_state.replace(
                archive=reordered,
                fitness=neg_fitness[order],
                mean=reordered[0],
                best_member=reordered[0],
                best_fitness=neg_fitness[order][0],
            )
            print(f"  Archive re-scored on sub-task {task_idx}: best "
                  f"{stale_best:.2f} (stale) -> {float(-ga_state.fitness[0]):.2f}",
                  flush=True)

        # 'per_task' restarts the mutation schedule at every boundary: the
        # archive carries over, the exploration noise does not.
        if restart_sigma_each_task:
            ga_state = ga.reset_sigma(ga_state)

        task_best_fitness = -float('inf')
        task_best_params = None

        for task_gen in range(gens_per_task):
            key, ask_key, eval_key, tell_key = jax.random.split(key, 4)

            gen_sigma = float(ga_state.sigma)  # the std this generation mutates with
            # ask_with_parents: see train_GA_ant.py. Same offspring as ask().
            population, ga_state, parents = ga.ask_with_parents(ask_key, ga_state, ga_params)
            # `or gen == 0`: the plasticity probe batch is frozen from the first
            # generation's visited states, as the diversity probes are.
            measure_now = ((tracker is not None and tracker.needs(gen))
                           or (gen == 0 and scoring_fn_bd is not None))
            if measure_now:
                fitness, extras = scoring_fn_bd(population, eval_key)
            else:
                fitness, extras = scoring_fn(population, eval_key), None

            # Copy to host BEFORE tell() - tell() may reuse buffers
            fitness_host = jax.device_get(fitness)
            population_host = jax.device_get(population)

            ga_state = ga.tell(population, -fitness, ga_state, ga_params)

            gen_best = float(np.max(fitness_host))
            gen_mean = float(np.mean(fitness_host))
            best_idx = int(np.argmax(fitness_host))

            final_best_params = population_host[best_idx].copy()
            task_best_params = final_best_params
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
                "std": gen_sigma,
            }
            if task_gen == 0:
                attach_zero_shot(record, zero_shot, task_idx - 1)

            # Weight statistics, the third plasticity signal alongside the
            # RL side's dormancy and churn. Also taken at every sub-task
            # boundary (task_gen == 0), not just on the interval: the switch is
            # the moment the weights are asked to move, so a reading that
            # straddles it is the one worth having.
            if (gen % weight_interval == 0 or task_gen == 0
                    or gen == total_generations - 1):
                record.update(population_weight_stats(population_host))

            # Plasticity: elite churn across generations, within-population
            # churn, and dormancy. See source/metrics/plasticity.py.
            if plast is not None:
                if not plast.started() and extras is not None:
                    plast.start(plasticity_key, extras["observations"])
                record.update(plast.update(
                    jax.random.fold_in(plasticity_key, gen), population_host,
                    population_host[best_idx], parents=jax.device_get(parents)))

            diversity = None
            if measure_now:
                if gen == 0:
                    tracker.start(probe_key, extras["observations"])
                diversity = tracker.update(
                    jax.random.fold_in(diversity_key, gen), gen, population,
                    extras["observations"], extras["behaviour"], fitness)
                if diversity:
                    record.update(diversity)

            wandb.log(record)
            training_metrics.append({**record, 'elapsed_time': time.time() - start_time})

            if task_gen % args.log_interval == 0 or task_gen == gens_per_task - 1:
                line = (f"Task {task_idx} Gen {gen:5d} | Best: {gen_best:8.2f} "
                        f"| Mean: {gen_mean:8.2f} | Task best: {task_best_fitness:8.2f} "
                        f"| leg: {LEG_NAMES[leg_idx]} fric x{friction_mult:g}"
                        + (f" spd {target_speed:g}" if target_speed is not None else "")
                        + "")
                if diversity:
                    line += f" | {summarise(diversity)}"
                print(line, flush=True)

            gen += 1

        # End-of-sub-task checkpoint, in the same format as the noncontinual
        # trainer's _best.pkl so the same loaders can render any of them.
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
            key=jax.random.fold_in(gif_key, task_idx),
            num_gifs=args.gifs_per_task, friction_mult=friction_mult,
            target_speed=target_speed, gravity_mult=gravity_mult)

    total_time = time.time() - start_time
    print(f"\nTraining complete! Time: {total_time:.1f}s over {gen} generations")
    print(f"  Best overall: {best_overall_fitness:.2f}")

    # Final-generation best, matching the cheetah continual trainer's convention:
    # a best-of-run genome would be one selected under whichever damage happened
    # to suit it, which is not the thing a continual run is asked to produce.
    ckpt_path = os.path.join(output_dir, f"ga_{env_name}_continual_legs_best.pkl")
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
