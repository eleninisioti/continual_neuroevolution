"""What both trainers need: the sub-task schedule, and the run artifacts.

``train_nes.py`` and ``train_ppo.py`` have to agree on two things or the two
methods cannot be put on one figure. Both live here rather than in either
trainer, so neither is the other's dependency:

    the schedule    which sub-task each phase is scored on. A PPO run and an
                    NES run at the same trial must face the identical sequence.
    the artifacts   ``results.json`` / ``training_metrics.json`` /
                    ``trajectory.npz``, with the same keys, so every analysis
                    script runs on either tag unchanged.

This module IMPORTS NO JAX, and that is load-bearing rather than tidiness:
``scripts/outdated/generalists/train/train_all.py`` needs ``SCHEDULES`` to build its parser
*before* it sets ``CUDA_VISIBLE_DEVICES``, and importing jax first freezes the
device mask. It used to keep a hand-copied duplicate of the tuple plus a
drift check for exactly that reason; it now imports from here. Keep numpy the
heaviest thing this file pulls in.
"""

from __future__ import annotations

import json
import os

import numpy as np

SCHEDULES = ('task0', 'task1', 'switch', 'joint', 'pairwise', 'sampled',
             'chained', 'growing')


def make_phase_grid(num_steps, task_interval, warmup=0):
    """Which PHASE each step belongs to, and how many phases there are.

    The uniform grid -- phase ``step // task_interval`` -- unless ``warmup``,
    which makes the FIRST phase ``warmup`` steps long and every later one
    ``task_interval``. ``warmup=0`` reproduces the uniform grid exactly, so
    every run made before this existed is unchanged.

    A warmup exists because a phase has to be read against how long the SEARCH
    takes on that body, and the two can differ by an order of magnitude. The
    ant's stationary ES run reaches 50% of its final score at generation 104
    and 90% at 184, while the continual grid gives it 16 generations a phase:
    every arm is then measured mid-climb, and the continual curve plateaus at
    ~2000 against a 4100 stationary ceiling -- a number about the phase length,
    not about continual learning. Converging on the first sub-task first and
    then switching fast measures what the design is for: adaptation, and
    retention, from a policy that HAS the skill.

    Returns ``(phase_of_step, num_phases)``; the phase array is what the
    trainers index, so a schedule change is this function and nothing else.
    """
    steps = np.arange(int(num_steps))
    warmup = int(warmup or 0)
    if warmup > 0:
        phase = np.where(steps < warmup, 0,
                         1 + (steps - warmup) // int(task_interval))
    else:
        phase = steps // int(task_interval)
    phase = phase.astype(int)
    return phase, (int(phase[-1]) + 1 if len(phase) else 0)


def make_task_sequence(schedule, num_phases, num_tasks, trial,
                       pool_size=None, pair_repeats=5,
                       expand_every=25):
    """Which sub-task each phase is scored on, as an array of length num_phases.

    Replaces a per-generation formula so that schedules which depend on more
    than the phase index -- a random draw, a curriculum -- are expressible at
    all, and so the whole sequence can be written into results.json and read
    back by the analysis rather than reconstructed.

        task0 / task1   one sub-task throughout; the specialist controls.
        switch          round-robin, 0,1,2,...,n-1,0,1,... The original.
        pairwise        alternate WITHIN a pair for `pair_repeats` rounds, then
                        move to the next DISJOINT pair: 0,1,0,1 | 2,3,2,3 | ...
        chained         the same, but pairs OVERLAP: 0,1,0,1 | 1,2,1,2 |
                        2,3,2,3 | ... Each new sub-task arrives alongside one
                        the search has just been holding, so a generalist can
                        grow by one sub-task at a time instead of having to
                        cover a fresh pair from scratch. Disjoint pairs give no
                        such ladder -- every pair starts over.
        sampled         drawn uniformly from the first `pool_size` sub-tasks,
                        so revisits happen by chance rather than on a cycle.
        joint           -1 everywhere; the caller scores all sub-tasks.

    `pairwise` and `sampled` exist to break a confound in the round-robin arm.
    At a fixed 50 phases, round-robin gives each of ten sub-tasks FIVE visits
    against twenty-five when there are two, so "more sub-tasks" and "fewer
    revisits" move together and a failure at ten cannot be attributed to
    either. Both of these hold the sub-task count at ten (or a pool of five)
    while raising how often the search comes back to one it has already seen.

    The draw for `sampled` is seeded from the trial index alone, the same
    convention as the sub-task offsets, so every method at a given trial faces
    the identical sequence.
    """
    if schedule == 'task0':
        return np.zeros(num_phases, dtype=int)
    if schedule == 'task1':
        return np.ones(num_phases, dtype=int)
    if schedule == 'joint':
        return np.full(num_phases, -1, dtype=int)
    if schedule == 'switch':
        return np.arange(num_phases) % num_tasks
    if schedule == 'pairwise':
        num_pairs = max(num_tasks // 2, 1)
        per_pair = 2 * pair_repeats
        phase = np.arange(num_phases)
        pair = (phase // per_pair) % num_pairs
        return (pair * 2 + phase % 2) % num_tasks
    if schedule == 'growing':
        # A curriculum WITH REPLAY: the pool starts at two sub-tasks and gains
        # one every `expand_every` phases; each phase draws uniformly from the
        # whole pool so far, so an old sub-task keeps being re-scored for the
        # rest of the run instead of being left behind.
        #
        # This exists because `chained` failed for a diagnosable reason. There,
        # once pair (k, k+1) ended, sub-task k was never scored again -- and the
        # search dropped it while learning the next pair, so breadth oscillated
        # around 2-4 and never ratcheted, even with 6000 generations a pair.
        # The two-sub-task result worked because BOTH tasks kept being scored,
        # which is what made the generalist an attractor. A sliding window
        # destroys exactly that; a growing pool keeps it.
        #
        # The cost: re-exposure thins as the pool grows. With a pool of k a
        # sub-task comes round every ~k phases, so late sub-tasks get less
        # practice per phase than early ones. That is the trade, and it is why
        # the run has to be long.
        rng = np.random.default_rng(int(trial) * 7919 + 23)
        out = np.empty(num_phases, dtype=int)
        for p in range(num_phases):
            pool = min(2 + p // max(expand_every, 1), num_tasks)
            out[p] = rng.integers(0, pool)
        return out
    if schedule == 'chained':
        num_pairs = max(num_tasks - 1, 1)
        per_pair = 2 * pair_repeats
        phase = np.arange(num_phases)
        pair = (phase // per_pair) % num_pairs
        return (pair + phase % 2) % num_tasks
    if schedule == 'sampled':
        pool = pool_size or num_tasks
        rng = np.random.default_rng(int(trial) * 7919 + 11)
        return rng.integers(0, min(pool, num_tasks), size=num_phases)
    raise ValueError(f'unknown schedule {schedule!r}')


def record_scores(record, per_task, prefix='centroid'):
    """Write the per-sub-task scores of one genome into a metrics record.

    ``<prefix>_generalist`` is the WORST sub-task, whatever fitness the search
    was actually optimising -- it is the run's result, and it has to be the
    same reduction for both methods.

    The two prefixes the runners write (since 2026-09-09; the older runs are
    renamed in place by scripts/analysis/migrate_shared_runner_columns.py):

        centroid   the coordinate-wise MEAN OF THE POPULATION'S WEIGHTS, always.
                   On NES/OpenES that is the distribution mean; on GA and DNS
                   it is a point no search optimises, and a low score there is
                   evidence of a spread archive rather than of a bad search.
                   Single-policy RL runs write the policy itself here.
        incumbent  what the search HANDS BACK: the best archive member on GA
                   and DNS (the elite), the distribution mean on NES/OpenES,
                   the best PBT member by training return. The figures read it
                   as `elite_eval`.

    Until 2026-09-09 this docstring described the opposite convention
    (`centroid` = what the search hands back, `popmean` = the weight mean),
    which is what the kinetix_repo runs on CLUSTER still record.
    """
    per_task = np.asarray(per_task)
    for t in range(per_task.shape[0]):
        record[f'{prefix}_task{t}'] = float(per_task[t])
    record[f'{prefix}_generalist'] = float(per_task.min())
    record[f'{prefix}_mean_over_tasks'] = float(per_task.mean())
    return record


def record_centroid_scores(record, per_task):
    """``record_scores`` under the ``centroid`` prefix. Kept as the name every
    existing caller uses."""
    return record_scores(record, per_task, prefix='centroid')


def summarise_records(records, num_tasks, threshold):
    """The ``final`` / ``best_generalist`` / ``final_solves_all_tasks`` block.

    Shared so the two methods are summarised by one piece of code: `best` is
    the generation whose centroid had the highest WORST sub-task, not the
    highest mean, and a run "solves all tasks" only on its final generation's
    scores -- never on the best it ever passed through.

    ``threshold`` is None on a suite that has no solved threshold -- CheetahRun
    and the ant, where ``evaluation_metrics.py`` says outright there is nothing
    to compare against. Then ``final_solves_all_tasks`` is None rather than
    False: "not measured here" and "measured, and it did not" are different
    facts about a run, and anything reading the key must be able to tell them
    apart.
    """
    final = records[-1]
    best = int(np.argmax([r['centroid_generalist'] for r in records]))
    return {
        'final': {k: final[k] for k in final if k.startswith('centroid')},
        'best_generalist': {
            'generation': best,
            'score': records[best]['centroid_generalist'],
            **{f'task{t}': records[best][f'centroid_task{t}']
               for t in range(num_tasks)},
        },
        'final_solves_all_tasks': None if threshold is None else all(
            final[f'centroid_task{t}'] >= threshold for t in range(num_tasks)),
    }


def write_run(output_dir, result, records, **arrays):
    """Write the three artifacts every analysis script expects to find.

    ``arrays`` goes into ``trajectory.npz``; NES adds population snapshots and
    champion genomes there, PPO writes only the checkpoints, and the keys they
    share (``centroids``, ``generations``, ``noise_vectors``) mean the same
    thing in both -- a PPO checkpoint and an NES centroid are points in the
    same parameter space.
    """
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, 'results.json'), 'w') as f:
        json.dump(result, f, indent=2)
    with open(os.path.join(output_dir, 'training_metrics.json'), 'w') as f:
        json.dump(records, f)
    np.savez_compressed(os.path.join(output_dir, 'trajectory.npz'), **arrays)
    print(f"  wrote {output_dir}")
