"""
Train SimpleGA on Brax Ant (non-continual).

Mirrors source/studies/mujoco/train_GA_cheetah.py, but on the standard Brax Ant
environment instead of a mujoco_playground task.

Usage:
    python source/studies/brax/train_GA_ant.py --gpus 0
    python source/studies/brax/train_GA_ant.py --pop_size 1024 --num_generations 500 --gpus 0
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
import flax.linen as nn
from source.algorithms.ne.ga import SimpleGA
from brax import envs

# One source of truth for which simulator the ant runs on.
from source.envs.brax_common import DEFAULT_BACKEND
from brax.io import image as brax_image
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

from source.metrics.aurora import episode_relative_indices, subsample_indices
from source.metrics.behaviour_descriptors import (
    BehaviourConfig,
    average_over_evals,
    rollout_behaviour,
)
from source.metrics.behaviour_tracking import (
    PopulationDiversityTracker,
    add_diversity_args,
    summarise,
)
from source.studies.brax.behaviour_brax import AntFeet


from source.algorithms.networks import (  # one definition of the policy and
    ContinuousMLPPolicy,             # of the flat<->pytree bridge, shared
    create_continuous_policy_network,  # with the RL side
    get_flat_params,
    unflatten_params,
)


# The cheetah/ant NE policy. One definition, in source/algorithms/networks.py, so
# that the RL trainers have a single thing to match and the four NE trainers
# cannot drift apart -- this was four byte-identical copies. `MLPPolicy` stays
# as a name because the evaluators import it from here.
MLPPolicy = ContinuousMLPPolicy
create_policy_network = create_continuous_policy_network


# `create_env` moved to source/envs/brax_ant.py: building an ant must not
# require importing this trainer, whose module body assigns CUDA_VISIBLE_DEVICES.
from source.envs.brax_ant import create_env


def make_scoring_fn(env, policy, param_template, episode_length, num_evals=1,
                    behaviour_cfg=None, traj_steps=10, feet=None,
                    return_descriptor=False):
    """Create the JIT-compiled scoring function.

    With `behaviour_cfg` set it additionally returns, per individual, the
    sub-sampled observation trajectory of the first rollout (what AURORA
    encodes) and the behaviour descriptors averaged over all `num_evals`
    rollouts. Selection sees neither, so switching tracking on cannot change
    what the run does -- the same contract as the cheetah and gymnax trainers.

    `return_descriptor` adds DNS's *selection* descriptor (the torso's final x/y
    position) to the return, which is a separate quantity from the observer
    descriptors above and is unaffected by tracking.
    """
    jit_reset = jax.jit(env.reset)
    jit_step = jax.jit(env.step)
    track = behaviour_cfg is not None
    num_traj_steps = int(subsample_indices(episode_length, traj_steps).shape[0])

    def evaluate_single(flat_params, eval_key):
        params = unflatten_params(flat_params, param_template)
        reset_key, _ = jax.random.split(eval_key)
        state = jit_reset(reset_key)

        def step_fn(carry, _):
            state, total_reward, active, descriptor = carry
            obs = state.obs
            action = policy.apply(params, obs)
            # Foot contact is read off the *current* state, so it lines up with
            # the observation and action of the same step. Reading it after the
            # step would sample the auto-reset pose on the terminating step.
            contact = feet.contact(state.pipeline_state) if track else None
            xy = feet.torso_xy(state.pipeline_state) if return_descriptor else None
            next_state = jit_step(state, action)
            # nan_to_num guards the terminating step itself: a diverging sim
            # reports NaN reward on the same step it flags done, and that one
            # is still inside the active window.
            reward = jnp.nan_to_num(next_state.reward, nan=0.0,
                                    posinf=0.0, neginf=0.0)
            total_reward = total_reward + reward * active
            # The step's own signals are part of the episode iff it started
            # inside it; `active` is updated only afterwards.
            valid = active
            if return_descriptor:
                descriptor = jnp.where(valid > 0, xy, descriptor)
            active = active * (1.0 - next_state.done)
            per_step = (obs, action, valid, contact) if track else None
            return (next_state, total_reward, active, descriptor), per_step

        init_descriptor = (feet.torso_xy(state.pipeline_state)
                           if return_descriptor else 0.0)
        (_, total_reward, _, descriptor), per_step = jax.lax.scan(
            step_fn, (state, 0.0, 1.0, init_descriptor), None,
            length=episode_length
        )

        out = (total_reward,)
        if return_descriptor:
            out += (descriptor,)
        if track:
            all_obs, all_actions, valid, contact = per_step
            behaviour = rollout_behaviour(all_obs, all_actions, valid,
                                          behaviour_cfg,
                                          aux={"foot_contact": contact})
            out += (all_obs[episode_relative_indices(valid, num_traj_steps)],
                    behaviour)
        return out[0] if len(out) == 1 else out

    vmapped_eval = jax.vmap(evaluate_single)

    # JIT-compiled trajectory collection for fast GIF generation
    @jax.jit
    def rollout_with_trajectory(flat_params, eval_key):
        """Rollout episode and return pipeline states for rendering."""
        params = unflatten_params(flat_params, param_template)
        state = jit_reset(eval_key)

        def step_fn(carry, _):
            state, total_reward, active = carry
            action = policy.apply(params, state.obs)
            next_state = jit_step(state, action)
            reward = jnp.nan_to_num(next_state.reward, nan=0.0,
                                    posinf=0.0, neginf=0.0)
            total_reward = total_reward + reward * active
            active = active * (1.0 - next_state.done)
            return (next_state, total_reward, active), state.pipeline_state

        (_, total_reward, _), trajectory_states = jax.lax.scan(
            step_fn, (state, 0.0, 1.0), None, length=episode_length
        )
        return total_reward, trajectory_states

    @jax.jit
    def scoring_fn(flat_genotypes, key):
        """Fitnesses, plus the selection descriptor and `extras` when asked for."""
        pop_size = flat_genotypes.shape[0]
        all_keys = jax.random.split(key, pop_size * num_evals)
        flat_params_repeated = jnp.repeat(flat_genotypes, num_evals, axis=0)
        results = vmapped_eval(flat_params_repeated, all_keys)
        if not (track or return_descriptor):
            return jnp.mean(results.reshape(pop_size, num_evals), axis=1)

        all_rewards, results = results[0], list(results[1:])
        out = (jnp.mean(all_rewards.reshape(pop_size, num_evals), axis=1),)
        if return_descriptor:
            descriptors = results.pop(0)
            out += (jnp.mean(descriptors.reshape(pop_size, num_evals, -1), axis=1),)
        if track:
            all_traj, all_behaviour = results
            out += ({
                "observations": all_traj.reshape(
                    pop_size, num_evals, num_traj_steps, -1)[:, 0],
                "behaviour": average_over_evals(all_behaviour, pop_size, num_evals),
            },)
        return out

    return scoring_fn, rollout_with_trajectory


def parse_args():
    parser = argparse.ArgumentParser(description='SimpleGA on Brax Ant (Non-Continual)')
    parser.add_argument('--env', type=str, default='ant')
    parser.add_argument('--backend', type=str, default=DEFAULT_BACKEND,
                        choices=['mjx', 'generalized', 'spring', 'positional'],
                        help="Physics backend. Must match the continual block's, "
                             "which this run is the control for.")
    parser.add_argument('--num_generations', type=int, default=500)
    parser.add_argument('--pop_size', type=int, default=512)
    parser.add_argument('--elite_ratio', type=float, default=0.1,
                        help='Fraction of the population used as parents')
    # sigma=0.005 picked from a 30-generation sweep on ant (pop 256):
    # 0.1 collapses, 0.02 stagnates, 0.005 climbs fastest, 0.001 is slower.
    parser.add_argument('--sigma', type=float, default=0.005)
    parser.add_argument('--sigma_final', type=float, default=0.001,
                        help='Mutation std at the last generation; sigma decays '
                             'geometrically from --sigma to this value. Set '
                             'equal to --sigma for a constant schedule.')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--trial', type=int, default=1)
    parser.add_argument('--episode_length', type=int, default=1000)
    parser.add_argument('--num_evals', type=int, default=1,
                        help='Number of evaluations per individual (averaged). '
                             '1 keeps the env-step budget low but makes selection '
                             'noisy: a single-episode best score re-scores far '
                             'lower on fresh episodes. Raise to 3 to trade env '
                             'steps for less optimistic selection.')
    parser.add_argument('--gpus', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--wandb_project', type=str, default='brax_ant_ga')
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

    output_dir = args.output_dir or f"projects/brax/ga_{env_name}/trial_{trial}"
    os.makedirs(output_dir, exist_ok=True)

    # Every gymnax trainer keeps a `train.log` next to the run and no brax one
    # did, so an ant run's only record of what it printed was the launcher's log
    # -- which is named after the job, not the run, and is gone once the sweep's
    # log directory is cleaned. Same shared Tee, same filename.
    sys.stdout = Tee(os.path.join(output_dir, "train.log"))

    print("=" * 60)
    print(f"SimpleGA on Brax {env_name} (Non-Continual)")
    print("=" * 60)
    print(f"  Generations: {num_generations}")
    print(f"  Population: {pop_size}")
    print(f"  Evals per individual: {args.num_evals}")
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

    # Total env steps consumed, for comparison against PPO's timestep budget
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

    # Create scoring function and trajectory rollout
    scoring_fn, rollout_with_trajectory = make_scoring_fn(
        env, policy, param_template, episode_length, num_evals=args.num_evals)
    scoring_fn_bd = make_scoring_fn(
        env, policy, param_template, episode_length, num_evals=args.num_evals,
        behaviour_cfg=behaviour_cfg, traj_steps=args.traj_steps, feet=feet,
    )[0] if track_diversity else None

    # Initialize wandb
    config = {
        'env': env_name, 'num_generations': num_generations,
        'pop_size': pop_size, 'seed': seed, 'trial': trial,
        'num_evals': args.num_evals, 'elite_ratio': args.elite_ratio,
        'sigma': args.sigma, 'sigma_final': args.sigma_final,
        'episode_length': episode_length,
        'algorithm': 'ga', 'track_diversity': track_diversity,
    }
    run_name = args.run_name or f"ga_{env_name}_pop{pop_size}_trial{trial}"
    wandb.init(project=args.wandb_project, config=config,
               name=run_name, reinit=True)
    # The run says what network it searched, so
    # scripts/check_architectures.py can confirm every method compared on
    # this task used the same one. Was recorded only inside the pickles.
    write_run_config(output_dir, config, policy_arch='brax')

    # Initialize GA
    devices = jax.devices()
    mesh = Mesh(np.array(devices), axis_names=('p',))
    replicate_sharding = NamedSharding(mesh, PartitionSpec())

    # Mutation std schedule. A constant sigma made ant peak around generation
    # 200 and then decay (population mean fell too, so it was drift, not eval
    # noise): late in the run every offspring is still perturbed as hard as at
    # generation 0. Decaying sigma geometrically to sigma_final lets the
    # population refine once it is near a good solution.
    sigma_decay = SimpleGA.decay_for(args.sigma, args.sigma_final, num_generations)

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

        tracker = PopulationDiversityTracker(
            env_name=env_name, obs_dim=obs_dim, num_actions=int(action_dim),
            traj_steps=args.traj_steps, num_generations=num_generations,
            interval=args.diversity_interval, occupancy_bins=args.occupancy_bins,
            max_pairwise=args.diversity_max_pairwise,
            num_probe=args.num_probe_states, apply_flat=apply_flat,
            snapshots=args.behaviour_snapshots, seed=seed,
        )
    # How often the weight statistics are recorded. Reuses --diversity_interval
    # rather than adding a flag: both are observers sampled off the same
    # generation axis, and a second cadence is a second thing to keep in step
    # between the suites. Reductions over a (512, 21128) population are not free
    # at every generation -- ~1e7 values -- and nothing about the weight norms
    # moves fast enough for per-generation sampling to show anything extra.
    weight_interval = max(1, int(args.diversity_interval))

    # --- Plasticity diagnostics, reported for every method (core/plasticity.py).
    # Churn and dormancy on a frozen probe batch, the same two columns the
    # gymnax trainers already report, so NE and RL sit in one table.
    #
    # continuous=True is the ant-specific part: the gymnax policies act by
    # argmax over logits and churn is the fraction of probe states whose chosen
    # action changed, but ContinuousMLPPolicy emits a tanh-squashed torque
    # vector with no argmax. Churn is therefore the mean absolute change per
    # actuator over the tanh range, which lands in the same [0, 1] as the
    # discrete measure. action_range=2.0 because the output is tanh.
    #
    # max_pairwise is 64 rather than the 128 default: the within-population term
    # materialises an (n, n, probe, action_dim) array, and at n=128 with 512
    # probe states and 8 actuators that is 67M floats (~268 MB) every time it is
    # measured. gymnax gets away with 128 because its action dim is 1.
    # The frozen probe batch is taken from generation 0's behaviour-descriptor
    # rollout, and `scoring_fn_bd` -- the only thing that returns observations --
    # is built only when --track_diversity is on. So plasticity rides on that
    # flag rather than silently reporting nothing. The paper blocks all pass
    # --track_diversity 1, so this is off only for a deliberate bare run.
    _brax_activation = POLICY_ARCH['brax']['activation']
    plast = None if scoring_fn_bd is None else plasticity.NEPlasticityTracker(
        apply_flat=lambda f, o: policy.apply(unflatten_params(f, param_template), o),
        unflatten=lambda f: unflatten_params(f, param_template),
        num_hidden=len(hidden_dims),
        activation_fn=ACTIVATIONS[_brax_activation],
        criterion=redo_mod.criterion_for_activation(_brax_activation),
        num_probe=args.num_probe_states,
        max_pairwise=64,
        continuous=True,
        action_range=2.0,
    )
    if plast is None:
        print("  (plasticity diagnostics off: they need --track_diversity 1)")
    plasticity_key = jax.random.key(seed + 3_000_000)

    # Diversity draws from its own stream so that switching tracking on or off
    # leaves the search's random numbers -- and therefore the run -- unchanged.
    diversity_key = jax.random.key(seed + 1_000_000)
    probe_key = jax.random.key(seed + 2_000_000)

    # Training loop
    best_overall_fitness = -float('inf')
    best_overall_params = None
    training_metrics = []
    start_time = time.time()

    print(f"\nStarting training...")

    for gen in range(num_generations):
        key, ask_key, eval_key, tell_key = jax.random.split(key, 4)

        # ask_with_parents: the churn column pairs each offspring with the
        # genome it came from, the only pairing with the same network before
        # and after one update. Same offspring as ask() for the same key.
        population, ga_state, parents = ga.ask_with_parents(ask_key, ga_state, ga_params)
        # `or gen == 0`: the plasticity probe batch is frozen from the initial
        # population's visited states, exactly as the diversity probes are, so
        # generation 0 has to produce observations even when --track_diversity
        # is 0. Same condition the gymnax trainers use.
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

        # Track best from current generation (will use final generation for eval)
        final_best_fitness = gen_best
        final_best_params = population_host[best_idx].copy()

        # Best genome seen in any generation, kept for the final checkpoint.
        if gen_best > best_overall_fitness:
            best_overall_fitness = gen_best
            best_overall_params = population_host[best_idx].copy()

        env_steps = (gen + 1) * steps_per_gen
        record = {
            'generation': gen,
            'env_steps': env_steps,
            'best_fitness': gen_best,
            'mean_fitness': gen_mean,
            'best_overall': best_overall_fitness,
        }

        # Weight statistics of the searched parameters themselves -- the third
        # plasticity signal CLAUDE.md asks for, next to dormancy and churn on the
        # RL side. On the same cadence as the diversity descriptors but NOT
        # gated on `measure_now`: `--track_diversity 0` must still leave a run
        # with its weight series, and this needs no rollouts to compute.
        if gen % weight_interval == 0 or gen == num_generations - 1:
            record.update(population_weight_stats(population_host))

        # Plasticity: elite churn across generations, within-population churn,
        # and dormancy. Same probe batch and same definitions as the RL
        # trainers -- see source/metrics/plasticity.py.
        if plast is not None:
            if gen == 0 and measure_now and not plast.started():
                plast.start(plasticity_key, extras["observations"])
            record.update(plast.update(
                jax.random.fold_in(plasticity_key, gen), population_host,
                population_host[best_idx], parents=jax.device_get(parents)))

        diversity = None
        if measure_now and tracker is not None:
            if gen == 0:
                tracker.start(probe_key, extras["observations"])
            diversity = tracker.update(
                jax.random.fold_in(diversity_key, gen), gen, population,
                extras["observations"], extras["behaviour"], fitness)
            if diversity:
                record.update(diversity)

        training_metrics.append({**record, 'elapsed_time': time.time() - start_time})
        wandb.log(record)

        if gen % args.log_interval == 0 or gen == num_generations - 1:
            line = (f"Gen {gen:4d} | Steps {env_steps:12,} | Best: {gen_best:8.2f} | "
                    f"Mean: {gen_mean:8.2f} | Overall: {best_overall_fitness:8.2f}")
            if diversity:
                line += f" | {summarise(diversity)}"
            print(line, flush=True)

    total_time = time.time() - start_time
    print(f"\nTraining complete! Time: {total_time:.1f}s")
    print(f"  Best overall: {best_overall_fitness:.2f}, Final gen best: {final_best_fitness:.2f}")

    # Checkpoint the best genome of the whole run, not of the last generation:
    # the last generation is not reliably the best one.
    best_params = best_overall_params
    best_fitness = best_overall_fitness

    ckpt_path = os.path.join(output_dir, f"ga_{env_name}_best.pkl")
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

    # Save training metrics
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

    # Re-score the checkpointed genome on fresh episodes: the recorded fitness
    # is a single noisy rollout, so it is optimistically biased by selection.
    print(f"\nVerifying best params on fresh episodes...")
    key, verify_key = jax.random.split(key)
    verify_fitness = scoring_fn(best_params[None, :], verify_key)
    print(f"  Reported best fitness: {best_fitness:.2f}")
    print(f"  Verification fitness:  {float(verify_fitness[0]):.2f}")

    # Save GIFs of the best policy
    if args.save_gifs:
        try:
            gifs_dir = os.path.join(output_dir, "gifs")
            os.makedirs(gifs_dir, exist_ok=True)

            print(f"\nSaving {args.num_gifs} evaluation GIFs...")
            for gif_idx in range(args.num_gifs):
                key, gif_key = jax.random.split(key)
                total_reward, trajectory_states = rollout_with_trajectory(best_params, gif_key)
                total_reward = float(total_reward)

                # Render every 4th frame for speed
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
