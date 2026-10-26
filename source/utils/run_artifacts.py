"""The checkpoint file a finished run leaves on disk."""

from __future__ import annotations

import os

import numpy as np


# ============================================================================
# Artifacts the analysis reads
# ============================================================================
#
# A finished run is read by two things that never re-run training:
#   training_metrics.json  the learning curve. One record per generation.
#   results.json +         the post-hoc evaluation, which scores the saved
#   checkpoints.npz        sub-task agents at a controlled episode count.
#                          Everything it needs to rebuild the policy and the
#                          sub-task sequence is in results.json.
#
# `source/runners/common.py:write_run` writes the first two; the checkpoints
# are written here, by every runner, so they cannot drift apart.


def save_checkpoints(output_dir, *, noise_vectors, finalgen=None,
                     incumbent=None, centroid=None, final=None):
    """Write ``checkpoints.npz`` ALONE, leaving ``results.json`` untouched.

    ``source/runners/common.py:write_run`` writes ``results.json``, with the
    nested-``config`` schema read through ``load_config``; this writes the
    checkpoints beside it.

    The keys, one saved agent per sub-task phase each:

        centroid   the network the centroid lineplot scores -- the mean of the
                   population's WEIGHTS. `population_mean` on every searcher:
                   the distribution mean for ES, the elite archive's
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
