"""Per-generation behavioural-diversity tracking for the gymnax NE trainers.

One object that GA, ES and DNS all drive the same way, so that the numbers in
`training_metrics.json` mean the same thing whichever method produced them:

    tracker = PopulationDiversityTracker(
        env_name=env_name, obs_dim=obs_dim, num_actions=action_dim,
        traj_steps=args.traj_steps, num_generations=num_generations,
        interval=args.diversity_interval, apply_flat=apply_flat,
    )
    tracker.start(key, observations)                      # freeze probes, fit AURORA
    ...
    metrics = tracker.update(key, gen, population, observations, behaviour,
                             fitnesses)                   # every generation
    if metrics:
        wandb.log(metrics); record.update(metrics)
    ...
    tracker.save(output_dir)                              # behaviour_snapshots.npz

The descriptor families and their units are documented in
`source/metrics/behaviour_descriptors.py`; the tracker only decides *when* to
measure and what to keep on disk.

Two things it is careful about:

* It is an observer. Nothing it computes is fed back into selection, so
  switching tracking on cannot change the trajectory of a run. The AURORA
  encoder it trains for GA/ES is separate from the one DNS selects with.
* AURORA latents live in a per-run space, so `bd_aurora_diversity` is only
  comparable *within* a run. The snapshots written to `behaviour_snapshots.npz`
  exist so that one encoder can be fitted on the pooled trajectories offline
  and every population re-encoded in a single shared space --- see
  `scripts/neurips_2026_rebuttal/behaviour_diversity_analysis.py`.
"""

from __future__ import annotations

import os

import jax
import jax.numpy as jnp
import numpy as np

from source.metrics.behaviour_descriptors import (
    AuroraTracker,
    BehaviourConfig,
    collect_probe_states,
    get_spec,
    make_probe_diversity_fn,
    mean_pairwise_euclidean,
    population_diversity,
)


class PopulationDiversityTracker:
    """Measures how behaviourally diverse a population is, every `interval` gens.

    Args:
        env_name: gymnax env id; selects the hand-designed descriptors.
        obs_dim, num_actions: policy interface sizes.
        traj_steps: length of the sub-sampled trajectory fed to AURORA.
        num_generations: total run length (sets the AURORA training schedule and
            the generations at which snapshots are kept).
        interval: measure every this many generations (the last generation is
            always measured).
        apply_flat: (flat_params, obs_batch) -> logits, for the probe-state
            measure. If None, the probe family is skipped.
        own_aurora: train an observer AURORA encoder. DNS passes False and
            hands its own latents to `update` instead.
        snapshots: how many generations to store trajectories for.
        snapshot_pop: individuals kept per snapshot.
    """

    def __init__(self, env_name, obs_dim, num_actions, traj_steps, num_generations,
                 interval=10, occupancy_bins=12, max_pairwise=128, num_probe=512,
                 apply_flat=None, own_aurora=True, snapshots=8, snapshot_pop=256,
                 latent_dim=6, aurora_lr=1e-3, aurora_batch_size=128,
                 aurora_train_ratio=8, seed=0):
        self.cfg = BehaviourConfig(env_name=env_name, num_actions=num_actions,
                                   occupancy_bins=occupancy_bins)
        spec = get_spec(env_name)  # fail now, not 600 generations in

        self.env_name = env_name
        self.num_generations = num_generations
        self.interval = max(1, int(interval))
        self.max_pairwise = max_pairwise
        self.num_probe = num_probe
        self.snapshot_pop = snapshot_pop
        self.seed = seed

        self.probe_fn = (make_probe_diversity_fn(apply_flat, max_sample=max_pairwise,
                                                 action_kind=spec.action_kind)
                         if apply_flat is not None else None)
        self.probe_obs = None

        self.aurora = None
        if own_aurora:
            self.aurora = AuroraTracker(
                obs_size=obs_dim, traj_steps=traj_steps,
                num_generations=num_generations, latent_dim=latent_dim,
                learning_rate=aurora_lr, batch_size=aurora_batch_size,
                train_ratio=aurora_train_ratio,
            )

        if snapshots > 0 and num_generations > 0:
            self.snapshot_gens = set(np.linspace(
                0, num_generations - 1, min(snapshots, num_generations)
            ).astype(int).tolist())
        else:
            self.snapshot_gens = set()
        self._snapshots = {}

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self, key, observations):
        """Freeze the probe batch and fit the observer encoder once.

        `observations` are the initial population's sub-sampled trajectories,
        (pop_size, traj_steps, obs_dim).
        """
        if self.probe_fn is not None:
            key, probe_key = jax.random.split(key)
            self.probe_obs = collect_probe_states(
                observations, num_probe=self.num_probe, key=probe_key)
        if self.aurora is not None:
            key, aurora_key = jax.random.split(key)
            self.aurora.init(aurora_key, observations)

    def tracks(self, generation):
        """Is this a generation whose diversity gets logged?"""
        return (generation % self.interval == 0
                or generation == self.num_generations - 1)

    def needs(self, generation):
        """Does this generation need the descriptors computed at all?

        Collecting them costs an extra pass over every rollout, so trainers
        keep a second scoring function without them and ask this before
        choosing which to call. Besides the logged generations, the encoder's
        own retraining generations and the snapshot generations need the data.
        """
        return (self.tracks(generation)
                or generation in self.snapshot_gens
                or (self.aurora is not None
                    and (generation + 1) in self.aurora.schedule))

    def update(self, key, generation, population, observations, behaviour,
               fitnesses, aurora_descriptors=None, aurora_loss=None):
        """Diversity of the current population, or None on untracked generations.

        Must be called every generation: the observer encoder retrains on
        AURORA's own schedule, which does not line up with `interval`.

        Args:
            population: (pop_size, num_params) flat genomes.
            observations: (pop_size, traj_steps, obs_dim) sub-sampled trajectories.
            behaviour: dict from `rollout_behaviour`, averaged over the repeat
                evaluations, with a leading population axis.
            fitnesses: (pop_size,) fitness used for selection.
            aurora_descriptors: latents to use instead of the observer encoder's
                (DNS passes the ones it selected with).
            aurora_loss: reconstruction loss that goes with them.
        """
        if self.aurora is not None:
            key, train_key = jax.random.split(key)
            self.aurora.maybe_train(train_key, generation, observations)

        if generation in self.snapshot_gens:
            self._store_snapshot(generation, observations, behaviour, fitnesses)

        if not self.tracks(generation):
            return None

        metrics = population_diversity(behaviour, max_n=self.max_pairwise,
                                       seed=self.seed)
        metrics["bd_genomic_diversity"] = mean_pairwise_euclidean(
            jax.device_get(population), max_n=self.max_pairwise, seed=self.seed)
        metrics["bd_fitness_std"] = float(np.std(jax.device_get(fitnesses)))

        if aurora_descriptors is not None:
            metrics["bd_aurora_diversity"] = mean_pairwise_euclidean(
                jax.device_get(aurora_descriptors), max_n=self.max_pairwise,
                seed=self.seed)
            if aurora_loss is not None:
                metrics["bd_aurora_loss"] = float(aurora_loss)
        elif self.aurora is not None:
            metrics.update(self.aurora.diversity(
                observations, max_n=self.max_pairwise, seed=self.seed))

        if self.probe_fn is not None and self.probe_obs is not None:
            key, probe_key = jax.random.split(key)
            metrics.update(self.probe_fn(population, self.probe_obs, probe_key))

        return metrics

    # ------------------------------------------------------------------
    # Snapshots
    # ------------------------------------------------------------------

    def _store_snapshot(self, generation, observations, behaviour, fitnesses):
        obs = np.asarray(jax.device_get(observations), dtype=np.float32)
        n = min(self.snapshot_pop, obs.shape[0])
        idx = np.random.default_rng(self.seed + generation).choice(
            obs.shape[0], n, replace=False)

        snap = {f"gen_{generation}_observations": obs[idx],
                f"gen_{generation}_fitness": np.asarray(
                    jax.device_get(fitnesses), dtype=np.float32)[idx]}
        for name, value in behaviour.items():
            arr = np.asarray(jax.device_get(value), dtype=np.float32)
            snap[f"gen_{generation}_{name}"] = arr[idx]
        self._snapshots.update(snap)

    def save(self, output_dir, filename="behaviour_snapshots.npz"):
        """Write the stored snapshots; returns the path, or None if empty."""
        if not self._snapshots:
            return None
        path = os.path.join(output_dir, filename)
        np.savez_compressed(
            path,
            snapshot_gens=np.array(sorted(self.snapshot_gens)),
            env_name=np.array(self.env_name),
            occupancy_bins=np.array(self.cfg.occupancy_bins),
            handcrafted_names=np.array(get_spec(self.env_name).handcrafted_names),
            **self._snapshots,
        )
        return path


def continual_snapshot_gens(num_generations, task_interval, per_task):
    """Snapshot generations, pinned relative to each sub-task.

    The tracker's own schedule spreads snapshots evenly over the whole run,
    which in the continual setting scatters them across sub-task boundaries and
    leaves some sub-tasks unsampled. Diversity *at the switch* is the quantity
    of interest, so each sub-task gets the same schedule: `per_task`
    generations spread over its own range, which always includes its first
    generation and the one immediately before the next switch.

    Set `tracker.snapshot_gens` to the result; build the tracker with
    `snapshots=0` so it does not also install its own.
    """
    if per_task <= 0 or num_generations <= 0:
        return set()
    gens = set()
    for start in range(0, num_generations, task_interval):
        stop = min(start + task_interval, num_generations) - 1
        if stop < start:
            continue
        gens.update(np.linspace(start, stop,
                                min(per_task, stop - start + 1)).astype(int).tolist())
    return gens


def add_diversity_args(parser, default_interval=10):
    """The flags every gymnax NE trainer exposes for this."""
    group = parser.add_argument_group("behavioural diversity tracking")
    group.add_argument('--track_diversity', type=int, default=1,
                       help='Log behavioural diversity of the population '
                            '(0 disables it and the extra rollout bookkeeping)')
    group.add_argument('--diversity_interval', type=int, default=default_interval,
                       help='Measure diversity every N generations')
    group.add_argument('--occupancy_bins', type=int, default=12,
                       help='Bins per dimension of the state-occupancy histogram')
    group.add_argument('--diversity_max_pairwise', type=int, default=128,
                       help='Individuals compared pairwise per diversity number')
    group.add_argument('--num_probe_states', type=int, default=512,
                       help='Size of the frozen probe batch for the '
                            'action-distribution measure')
    group.add_argument('--behaviour_snapshots', type=int, default=8,
                       help='Generations whose trajectories are saved to '
                            'behaviour_snapshots.npz for the offline shared-'
                            'encoder analysis (0 disables)')
    return parser


def summarise(metrics):
    """Short one-line form of a diversity dict, for the training log."""
    parts = []
    for key, label in (("bd_handcrafted_diversity", "hand"),
                       ("bd_occupancy_diversity", "occ"),
                       ("bd_action_usage_diversity", "act"),
                       ("bd_aurora_diversity", "aur"),
                       ("bd_probe_js", "probe"),
                       ("bd_probe_action_dist", "probe")):
        if key in metrics:
            parts.append(f"{label} {metrics[key]:.3f}")
    return " | ".join(parts)
