"""Result recording for the Kinetix continual neuroevolution runs.

Why this file exists
--------------------
The GA/DNS/ES continual trainers wrote almost nothing usable. Between them they
produced a single ``<method>_continual_best.pkl`` holding the best genome over
the *entire* 20-task run, plus a ``training_metrics.json`` of a dozen summary
scalars. Per-generation history existed only as wandb calls, and no per-sub-task
agent was ever written.

That makes the continual quantities unrecoverable, not merely inconvenient:

* **forgetting / backward transfer** needs the agent as it *left* each sub-task,
  so it can be re-scored on earlier ones. One global-best genome cannot give it.
* **zero-shot transfer** needs the same per-sub-task agents.
* **learning curves** need the per-generation series.
* **behavioural diversity** needs the *population*, not one individual --
  and a population is a property of a moment in the run, so it cannot be
  reconstructed afterwards from any checkpoint.

Unlike the RL side, where each sub-task is a separate process that already saves
its own agent, none of this can be back-filled: the data never existed. It has
to be written while training runs.

What it writes, per run directory
---------------------------------
Deliberately the same layout ``source/studies/gymnax/train_*_continual.py`` produces, so
one loader and one set of figure code reads both benchmarks:

    training_metrics.json    list of per-generation dicts, each carrying `task`
    results.json             config + per-sub-task record + `agent_sources`
    checkpoints.npz          agents stacked (num_tasks, num_dims), per source,
                             plus the sub-task sequence
    behaviour_snapshots.npz  populations saved at pinned generations, for the
                             offline shared-encoder diversity analysis
    metrics.yaml             training-time summary (not the reported numbers)

Two agents are recorded per sub-task, because they answer different questions --
matching the `agent_sources` convention in the gymnax tree:

    finalgen   best individual of the sub-task's final generation
    incumbent  the elite/archive mean, i.e. what the search would carry forward

Snapshots are pinned *per sub-task* rather than spread over the whole run, for
the reason `continual_snapshot_gens` gives in
``source/metrics/behaviour_tracking.py``: diversity at the switch is the quantity
of interest, and an evenly-spread schedule scatters samples across boundaries
and leaves some sub-tasks unsampled.
"""

import json
import os
import time

import numpy as np
import yaml


def snapshot_generations(generations_per_task, per_task):
    """Generations within a sub-task at which to save the population.

    Always includes the sub-task's first generation and its last, so every
    sub-task is sampled at the switch in and the switch out.
    """
    if per_task <= 0 or generations_per_task <= 0:
        return set()
    n = min(per_task, generations_per_task)
    return set(np.linspace(0, generations_per_task - 1, n).astype(int).tolist())


def rerank_by_rollout(flat_pop, batched_mean_fit, rollout_single, reshaper, rng,
                      top_k=32, reps=10):
    """Pick the agent to SAVE using the same evaluator that will judge it.

    The population is scored by a batched evaluator (vmap over individuals inside
    a lax.scan over chunks); the GIFs, the [verify] block and every number we
    report come from calling the rollout directly. On stable levels the two agree
    -- measured max |batched - single| = 0.06 on h0_unicycle. On chaotic ones they
    do not: on h3_car_thrust the same genome under the same RNG key scores 1.21
    apart, and 13 of 16 random genomes disagree. XLA fuses the batched path
    differently, and 100 steps of rigid-body physics amplify the rounding until
    the car either reaches the goal or does not.

    Selecting on the batched score therefore optimises a measure we never report.
    DNS saved an h3 agent at +1.282 batched that scored -0.082 over 16 direct
    rollouts, in both seeds. This takes the top `top_k` by batched score -- cheap
    and enough to hold every genuine contender -- then re-ranks just those by the
    direct rollout, so the agent we keep is the one that wins the measure we
    publish.

    Returns (best_flat_params, best_fitness, n_changed) where n_changed reports
    whether the re-rank moved the winner, which is worth logging.
    """
    import jax
    import jax.numpy as jnp
    import jax.random as jr

    k = int(min(top_k, flat_pop.shape[0]))
    cand = jnp.argsort(batched_mean_fit)[-k:]
    keys = jr.split(rng, reps)
    scores = []
    def _reward(out):
        # GA and ES rollouts return (reward, ep_len); DNS returns
        # (reward, ep_len, descriptor, trajectory) whose leaves have different
        # shapes, so asarray() on the whole tuple raises. The reward is always
        # the first element.
        while isinstance(out, (tuple, list)):
            out = out[0]
        return jnp.asarray(out).reshape(-1)[0]

    for idx in cand:
        tree = reshaper.reshape_single(flat_pop[idx])
        r = jnp.array([_reward(rollout_single(tree, kk)) for kk in keys])
        scores.append(jnp.mean(r))
    scores = jnp.array(scores)
    win = int(jnp.argmax(scores))
    best_idx = int(cand[win])
    moved = int(best_idx != int(cand[-1]))
    return flat_pop[best_idx], float(scores[win]), moved


class ContinualRecorder:
    """Accumulates a continual NE run and writes it out in the gymnax layout."""

    def __init__(self, output_dir, method, envs, generations_per_task, seed,
                 trial, pop_size, config=None, behaviour_snapshots=4,
                 descriptor=None, snapshot_pop=32):
        self.output_dir = output_dir
        self.method = method
        self.envs = list(envs)
        self.generations_per_task = int(generations_per_task)
        self.seed = int(seed)
        self.trial = int(trial)
        self.pop_size = int(pop_size)
        self.config = config or {}
        self.descriptor = descriptor
        self.snapshot_gens = snapshot_generations(generations_per_task,
                                                  behaviour_snapshots)
        # How many population members each snapshot keeps.
        #
        # Kinetix policies are ~1.13M parameters, so a full 512-member snapshot
        # is 2.3 GB and four per sub-task is 9.2 GB -- which at the observed
        # ~13 MB/s of compressed write is ~12 minutes of pure I/O per sub-task,
        # i.e. longer than the search itself. (Gymnax nets are ~1e4 params, so
        # this never came up there.) Every diversity measure is a mean over
        # pairs, and 32 members give 496 pairs, which estimates a population
        # mean pairwise distance perfectly well. Subsampling is uniform without
        # replacement, so it is an unbiased sample of the population.
        self.snapshot_pop = int(snapshot_pop) if snapshot_pop else 0

        self.history = []          # per-generation dicts
        self.per_task = []         # one record per sub-task
        self.agents = {"finalgen": [], "incumbent": []}
        self.snapshots = []        # (task_idx, gen, population array)
        self._t0 = time.time()

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

    # -- during training ---------------------------------------------------

    def log_generation(self, task_idx, task_name, gen_in_task, metrics):
        """One generation. `metrics` values are plain floats, or strings for
        the few label columns such as `ne_churn_kind`."""
        row = {
            "generation": int(task_idx * self.generations_per_task + gen_in_task),
            "gen_in_task": int(gen_in_task),
            "task": int(task_idx),
            "task_name": task_name,
        }
        # Numeric values are coerced to float; STRINGS pass through unchanged.
        # `ne_churn_kind` names which of the four churn estimators in CLAUDE.md
        # produced the `ne_churn` column, and every other suite carries it as a
        # string beside the number. Coercing everything broke on it.
        def _coerce(v):
            if v is None or isinstance(v, str):
                return v
            return float(v)
        row.update({k: _coerce(v) for k, v in metrics.items()})
        self.history.append(row)

    def maybe_snapshot(self, task_idx, gen_in_task, population):
        """Save the population if this generation is on the pinned schedule.

        Cheap enough to call every generation; it no-ops off-schedule.
        """
        if gen_in_task in self.snapshot_gens and population is not None:
            pop = np.asarray(population, dtype=np.float32)
            if 0 < self.snapshot_pop < pop.shape[0]:
                idx = np.random.default_rng(
                    self.seed * 1000 + task_idx * 10 + gen_in_task
                ).choice(pop.shape[0], self.snapshot_pop, replace=False)
                pop = pop[np.sort(idx)]
            self.snapshots.append((int(task_idx), int(gen_in_task), pop))

    def end_task(self, task_idx, task_name, finalgen, incumbent, record):
        """Close out a sub-task: its two agents and its summary record."""
        self.agents["finalgen"].append(np.asarray(finalgen, dtype=np.float32).reshape(-1))
        self.agents["incumbent"].append(np.asarray(incumbent, dtype=np.float32).reshape(-1))
        rec = {"task": int(task_idx), "task_name": task_name}
        rec.update({k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
                    for k, v in record.items()})
        self.per_task.append(rec)
        # Flush everything now, not only at the end of the chain. Every output
        # of this class used to be written once, by `finalise`, after the last
        # sub-task -- so a run stopped part-way left nothing but `train.log`.
        # That is exactly how the 2026-08-03 kinetix drop lost its
        # checkpoints.npz and behaviour_snapshots.npz across six runs, and why
        # F, BD and the whole diversity block had no NE rows. Writing per
        # sub-task costs a fraction of a second against ~30 min of compute and
        # makes an interrupted run yield every sub-task it finished.
        self._dump(time.time() - self._t0, final=False)

    # -- writing -----------------------------------------------------------

    def finalise(self, elapsed_seconds, extra=None):
        self._dump(elapsed_seconds, extra=extra, final=True)

    def _atomic(self, name, writer):
        """Write via a temp file + os.replace.

        Without this a mid-run flush is readable in a torn state -- reading a
        half-written behaviour_snapshots.npz raises BadZipFile, which is not an
        obvious symptom to debug.
        """
        # The temp file keeps the real extension: `np.savez_compressed`
        # appends ".npz" when the path lacks it, so a plain "<name>.tmp" is
        # written as "<name>.tmp.npz" and the replace below then looks for a
        # file that was never created.
        base, ext = os.path.splitext(name)
        tmp = os.path.join(self.output_dir, f"{base}.tmp{ext}")
        writer(tmp)
        os.replace(tmp, os.path.join(self.output_dir, name))

    def _dump(self, elapsed_seconds, extra=None, final=True):
        if not self.output_dir:
            return

        self._atomic("training_metrics.json",
                     lambda t: json.dump(self.history, open(t, "w"), indent=2))

        agents = {k: np.stack(v) for k, v in self.agents.items() if v}
        npz = dict(agents)
        if agents:
            npz["task_index"] = np.arange(len(next(iter(agents.values()))), dtype=np.int32)
        # The sub-task sequence, as names. Kinetix sub-tasks are distinct levels
        # rather than noise vectors, so this is the analogue of gymnax's
        # `noise_vectors` -- what an evaluator needs to rebuild each sub-task.
        npz["task_levels"] = np.array(self.envs[:len(self.per_task)], dtype=object)
        self._atomic("checkpoints.npz", lambda t: np.savez_compressed(
            t,
            **{k: v for k, v in npz.items() if k != "task_levels"},
            task_levels=np.array([str(s) for s in self.envs[:len(self.per_task)]]),
        ))

        if self.snapshots:
            self._atomic("behaviour_snapshots.npz", lambda t: np.savez_compressed(
                t,
                populations=np.stack([s[2] for s in self.snapshots]),
                snapshot_task=np.array([s[0] for s in self.snapshots], dtype=np.int32),
                snapshot_gen=np.array([s[1] for s in self.snapshots], dtype=np.int32),
            ))

        num_dims = int(next(iter(agents.values())).shape[1]) if agents else None
        results = {
            "method": self.method,
            "env": "kinetix_m_h0_h19",
            "trial": self.trial,
            "seed": self.seed,
            "pop_size": self.pop_size,
            "num_tasks": len(self.per_task),
            "task_levels": self.envs[:len(self.per_task)],
            "generations_per_task": self.generations_per_task,
            "num_params": num_dims,
            "descriptor": self.descriptor,
            # Which agents checkpoints.npz carries, in preference order. Same
            # meaning as the gymnax tree's field of the same name.
            "agent_sources": [k for k in ("finalgen", "incumbent") if self.agents[k]],
            "elapsed_seconds": float(elapsed_seconds),
            "config": self.config,
            "per_task": self.per_task,
        }
        if extra:
            results.update(extra)
        self._atomic("results.json",
                     lambda t: json.dump(results, open(t, "w"), indent=2))

        summary = {
            "method": self.method,
            "num_tasks": len(self.per_task),
            "generations_per_task": self.generations_per_task,
            "per_task": self.per_task,
        }
        with open(os.path.join(self.output_dir, "metrics.yaml"), "w") as f:
            yaml.dump(summary, f, default_flow_style=False)

        if not final:
            print(f"  [flush] {len(self.per_task)} sub-task(s) written "
                  f"({len(self.snapshots)} snapshots) -> {self.output_dir}")
            return
        print(f"  Saved {len(self.history)} generation records, "
              f"{len(self.per_task)} sub-task agents"
              + (f", {len(self.snapshots)} population snapshots" if self.snapshots else "")
              + f" -> {self.output_dir}")
