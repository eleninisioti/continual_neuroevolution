"""The files a finished population-based run leaves on disk.

Was the bottom of ``source/studies/gymnax/continual_common.py``, which mixed the
gymnax environment with the run's output format. Writing ``results.json`` and
``checkpoints.npz`` has nothing to do with gymnax -- the format is what
``evaluate_continual.py`` and the figure scripts read, and it is the same
format whichever suite produced it -- so it lives with the other
process-level utilities instead.
"""

from __future__ import annotations

import json
import os

import numpy as np


# ============================================================================
# Artifacts the analysis reads
# ============================================================================
#
# A finished run is read by two things that never re-run training:
#   training_metrics.json  scripts/neurips_2026_rebuttal/make_figures.py, for
#                          the learning curve. One record per generation.
#   results.json +         source/studies/evaluate_continual.py, which scores
#   checkpoints.npz        the saved sub-task agents at a controlled episode
#                          count. Everything it needs to rebuild the policy and
#                          the sub-task sequence is in results.json.
#
# Both are written by the population-based trainers through the helpers below
# so the three of them cannot drift apart in what they emit.


def save_training_metrics(output_dir, records):
    """Per-generation history, the file make_figures.py reads."""
    path = os.path.join(output_dir, "training_metrics.json")
    with open(path, "w") as f:
        json.dump(records, f, indent=2)
    return path


def save_eval_artifacts(output_dir, *, method, env, trial, seed, pop_size,
                        finalgen, incumbent, noise_vectors, hidden_dims,
                        centroid=None,
                        episode_length, num_generations, task_interval,
                        num_evals, noise_range, elapsed_seconds,
                        gen_best_trace=None, gen_mean_trace=None,
                        per_task=None, config=None,
                        task_type='noise', param_name=None, param_mults=None,
                        action_flips=None):
    """Write ``checkpoints.npz`` + ``results.json`` for evaluate_continual.py.

    ``finalgen`` and ``incumbent`` are one flat parameter vector per sub-task,
    in sub-task order. They answer different questions and the evaluator scores
    both:

    finalgen
        Best member of the sub-task's final generation, selected on training
        fitness -- the definition the gymnax training scripts already use.
    incumbent
        What the optimizer would hand back: `archive[0]` for the GA -- its best
        archive member, NOT a mean -- the distribution mean for OpenES/NES, and
        the repertoire best for DNS. Unbiased in population size.
    centroid
        Optional, and the one the CENTROID LINEPLOT scores: the coordinate-wise
        mean of the population's WEIGHTS, which is `archive.mean(0)` for the GA
        and `population.mean(0)` for DNS -- neither of them `incumbent`, and
        neither a network the search ever evaluated. It coincides with
        `incumbent` for OpenES/NES, where the distribution mean is both. Saved
        beside the other two rather than replacing either, so a figure can be
        drawn against whichever object its curve plots. Runs that predate it
        simply have no `centroid` key and the readers fall back.

    ``noise_vectors`` must be the *whole* sub-task sequence: zero-shot transfer
    is reconstructed by scoring sub-task t's agent on sub-task t+1.

    ``task_type`` says what a sub-task IS. Under ``param`` the observation is
    untouched -- ``noise_vectors`` is all zeros -- and ``param_mults`` carries
    the sequence instead: sub-task t is the stock body with the ``param_name``
    physics group rescaled by ``param_mults[t]``. evaluate_continual.py rebuilds
    those bodies with ``apply_physics``. Without it a param run's saved agents
    could not be scored at all, which is why the trainers used to emit no
    artifacts under that task type -- the sub-task sequence was unrecoverable.

    ``action_flips`` is the ``actions`` family's counterpart of ``param_mults``:
    sub-task t runs the stock body with its action order REVERSED when
    ``action_flips[t]`` is 1. The observation and the body are both untouched,
    so ``noise_vectors`` is all zeros there too and this array is the only
    record of what the sub-task sequence was.

    ``gen_best_trace`` is the per-generation best fitness over the whole run.
    It is the one piece of training-side data the metrics read: speed-up is a
    time-to-threshold and so exists nowhere else. It is sliced into per-sub-task
    traces by task_interval, so it must cover every generation in order.
    """
    agents_finalgen = np.stack([np.asarray(p) for p in finalgen])
    agents_incumbent = np.stack([np.asarray(p) for p in incumbent])
    num_tasks = int(agents_finalgen.shape[0])

    arrays = dict(
        finalgen=agents_finalgen,
        incumbent=agents_incumbent,
        noise_vectors=np.stack([np.asarray(v) for v in noise_vectors[:num_tasks]]),
    )
    if centroid is not None:
        arrays["centroid"] = np.stack([np.asarray(p) for p in centroid])
    if param_mults is not None:
        arrays["param_mults"] = np.asarray(param_mults[:num_tasks], dtype=np.float64)
    if action_flips is not None:
        arrays["action_flips"] = np.asarray(action_flips[:num_tasks], dtype=np.int32)
    np.savez_compressed(os.path.join(output_dir, "checkpoints.npz"), **arrays)

    results = {
        "method": method,
        "env": env,
        "trial": int(trial),
        "seed": int(seed),
        "pop_size": int(pop_size),
        "num_tasks": num_tasks,
        "num_generations": int(num_generations),
        "task_interval": int(task_interval),
        "num_evals": int(num_evals),
        "episode_length": int(episode_length),
        "hidden_dims": list(hidden_dims),
        "num_params": int(agents_finalgen.shape[1]),
        "noise_range": float(noise_range),
        "task_type": task_type,
        "param_name": param_name,
        "action_flips": ([int(f) for f in action_flips[:num_tasks]]
                         if action_flips is not None else None),
        "agent_sources": ["finalgen", "incumbent"],
        "elapsed_seconds": float(elapsed_seconds),
        # Training-side diagnostics only. No success rate is computed here by
        # design -- that is evaluate_continual.py's job.
        "per_task": per_task or [],
        "gen_best_trace": list(gen_best_trace or []),
        "gen_mean_trace": list(gen_mean_trace or []),
        "config": config or {},
    }
    path = os.path.join(output_dir, "results.json")
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    return path


def save_checkpoints(output_dir, *, noise_vectors, finalgen=None,
                     incumbent=None, centroid=None, final=None):
    """Write ``checkpoints.npz`` ALONE, leaving ``results.json`` untouched.

    ``save_eval_artifacts`` above writes both, which is right for the gymnax
    trainers -- they have no other ``results.json``. The suite-generic runners
    do: ``source/studies/generalists/common.py:write_run`` already writes one,
    with the nested-``config`` schema every analysis script reads through
    ``load_config``. Calling ``save_eval_artifacts`` from those runners would
    silently overwrite it with the flat schema and drop ``task_sequence``,
    ``arch`` and the suite's ``task`` description -- without which a MiniGrid
    run cannot be rebuilt at all. So the checkpoint half is separable, and this
    is it.

    The keys are exactly the ones ``scripts/analysis/plasticity_checkpoints.py``
    looks for, in its own fallback order (``AGENT_SOURCES``):

        centroid   the network the centroid lineplot scores -- the mean of the
                   population's WEIGHTS. `population_mean` on every searcher:
                   the distribution mean for NES/OpenES, the elite archive's
                   mean for the GA, the repertoire's for DNS. NOT `incumbent`
                   on the last two, which is the whole point of storing it.
        incumbent  what the optimizer would hand back (archive[0] / the mean /
                   the repertoire best).
        finalgen   best member of the sub-task's final generation.
        final      the one moving policy of a gradient method, which has no
                   population and so no three-way distinction.

    One row per SUB-TASK PHASE, in the order the phases ran, and
    ``noise_vectors`` must be the matching per-phase sequence -- not the
    per-sub-task table -- or zero-shot transfer is reconstructed against the
    wrong sub-task. A schedule that revisits sub-tasks has more phases than
    sub-tasks, and it is the phases that are checkpointed.
    """
    arrays = {}
    for name, value in (('finalgen', finalgen), ('incumbent', incumbent),
                        ('centroid', centroid), ('final', final)):
        if value is not None and len(value):
            arrays[name] = np.stack([np.asarray(p) for p in value])
    if not arrays:
        raise ValueError('save_checkpoints: nothing to save')
    num_phases = int(next(iter(arrays.values())).shape[0])
    for name, arr in arrays.items():
        if arr.shape[0] != num_phases:
            raise ValueError(
                f'save_checkpoints: {name} has {arr.shape[0]} phases, '
                f'expected {num_phases}')
    vectors = np.stack([np.asarray(v) for v in noise_vectors[:num_phases]])
    if vectors.shape[0] != num_phases:
        raise ValueError(
            f'save_checkpoints: {vectors.shape[0]} sub-task vectors for '
            f'{num_phases} checkpoints; pass the PER-PHASE sequence')
    arrays['noise_vectors'] = vectors
    path = os.path.join(output_dir, 'checkpoints.npz')
    np.savez_compressed(path, **arrays)
    return path
