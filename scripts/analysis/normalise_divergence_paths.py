"""Rewrite `run_dir` in behavioural_divergence outputs to start at `projects/`.

`behavioural_divergence.py` records each run's directory relative to the
checkout it ran from. On CLUSTER the job runs from `repo_pbt`, a sibling of the
runs' tree, so the paths read `../cont_ne_playground/projects/...`; the tables
(`make_lineplot.load_divergence`) key runs by `projects/...` and find nothing,
which leaves F `--`. Paths that exist are resolved first, so a run reached
through a symlink tree is keyed by its real directory. Idempotent; a `.orig` copy is kept the first time.

    python scripts/analysis/normalise_divergence_paths.py <results_dir> [...]
"""
import json
import os
import shutil
import sys

import numpy as np


def normalise(path):
    # A run reached through a symlink tree (a staged `paper/.../data` copy) is
    # keyed at home by its resolved directory, so resolve first; run from the
    # checkout the pass ran from.
    if path and os.path.exists(path):
        path = os.path.realpath(path)
    i = path.find('projects/')
    return path[i:] if i >= 0 else path


def fix_dir(results_dir):
    changed = 0
    jp = os.path.join(results_dir, 'behavioural_divergence.json')
    if os.path.exists(jp):
        blob = json.load(open(jp))
        for rec in blob.get('runs') or []:
            new = normalise(rec.get('run_dir', ''))
            changed += new != rec.get('run_dir')
            rec['run_dir'] = new
        if changed:
            if not os.path.exists(jp + '.orig'):
                shutil.copy2(jp, jp + '.orig')
            json.dump(blob, open(jp, 'w'))
    npz = os.path.join(results_dir, 'behavioural_divergence.npz')
    n_idx = 0
    if os.path.exists(npz):
        arrays = dict(np.load(npz, allow_pickle=True))
        index = []
        for raw in arrays['index']:
            meta = json.loads(str(raw))
            if 'run_dir' in meta:
                new = normalise(meta['run_dir'])
                n_idx += new != meta['run_dir']
                meta['run_dir'] = new
            index.append(json.dumps(meta))
        if n_idx:
            if not os.path.exists(npz + '.orig'):
                shutil.copy2(npz, npz + '.orig')
            arrays['index'] = np.array(index)
            np.savez(npz, **arrays)
    print(f'{results_dir}: {changed} JSON run_dir and {n_idx} npz index paths rewritten')


if __name__ == '__main__':
    for d in sys.argv[1:]:
        fix_dir(d)
