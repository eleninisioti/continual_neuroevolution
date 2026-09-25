"""Does Figure 2 depend on how forgetting is defined? The stability-plasticity
ring of every setting with more than two tasks, under four definitions of F.

    .venv/bin/python scripts/analysis/forgetting_definitions.py

    -> paper/visuals/final/forgetting_definitions.tex   the appendix table (booktabs)
       paper/visuals/final/forgetting_definitions.md    the same, with definitions
       paper/visuals/final/data/forgetting_definitions.json
    (paper = projects/iclr_2027/paper)

Reads the figure's saved extract (plot_stability_plasticity.py --extract, the
same arms and trials) and swaps only F. R[i][j] is the centroid saved at the end
of phase j on the task of phase i, from the forgetting pass for gymnax and the
training records for HalfCheetah and Kinetix, as the figure reads it:

    final        mean_{i<T} R[i][i] - R[i][T]      the paper's F (Lopez-Paz & Ranzato 2017,
                                                    Wolczyk et al. 2021)
    last visit   the same, over the LAST visit of each task other than the final
                 one: gymnax noise visits each task twice, and the first visit's
                 agent is compared with a final agent that trained on it again
    previous     mean_i R[i][i] - R[i][i+1]        the task just left, one switch later
    all later    mean_{i<j} R[i][i] - R[i][j]      every later agent (Diaz-Rodriguez et al. 2018)

A second table splits each run in half (the first and the second ten tasks)
and gives the centroid's LA and its previous-task F in each: does a switch cost
more late in the run, and is less forgetting there bought by less learning?

The two-task settings are left out: their F is already the switch forgetting,
the same under every definition. LA, the rescaling and the ring are
plot_stability_plasticity's (`normalise`, `summary`), so the `final` row
reproduces Figure 2.
"""

from __future__ import annotations

import copy
import contextlib
import io
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import plot_stability_plasticity as sp                     # noqa: E402

lp, es_arm, REPO = sp.lp, sp.es_arm, sp.REPO
STEM = sp.FINAL / 'forgetting_definitions'
DATA = sp.FINAL / 'data' / 'forgetting_definitions.json'


def _final(R, ids):
    return np.mean(np.diag(R)[:-1] - R[:-1, -1])


def _last_visit(R, ids):
    T = len(R) - 1
    keep = [i for i in range(T) if ids[i] != ids[T] and ids[i] not in ids[i + 1:]]
    return np.mean([R[i, i] - R[i, T] for i in keep])


def _previous(R, ids):
    return np.mean(np.diag(R)[:-1] - np.diag(R, 1))


def _all_later(R, ids):
    n = len(R)
    return np.mean([R[i, i] - R[i, j] for i in range(n) for j in range(i + 1, n)])


DEFINITIONS = {'final': ('Final agent (reported)', _final),
               'last_visit': ('Final agent, last visit', _last_visit),
               'previous': ('Previous task', _previous),
               'all_later': ('All later agents', _all_later)}


def pass_matrices():
    """`{run_dir: R}` from the forgetting passes the figure reads."""
    out = {}
    for sub in sp.PASS.values():
        npz = sp.PROJECT / 'paper' / sub / 'results/centroid/behavioural_divergence.npz'
        with np.load(npz) as z:
            for i, raw in enumerate(z['index']):
                key = f'run{i}_reward'
                if key in z.files:
                    out[str(pathlib.Path(json.loads(str(raw))['run_dir']))] = z[key]
    return out


def task_ids(run, R):
    """One label a phase, equal for phases on the same task: from the run's
    checkpoint sequence (make_lineplot.TASK_SEQUENCE_ARRAYS)."""
    with np.load(REPO / run / 'checkpoints.npz') as z:
        seq = np.concatenate([np.asarray(z[k], float).reshape(len(z[k]), -1)
                              for k in lp.TASK_SEQUENCE_ARRAYS if k in z.files], axis=1)
    seq = seq[:len(R)]
    return [int(np.flatnonzero((np.unique(seq, axis=0) == row).all(1))[0]) for row in seq]


def records_matrix(run):
    """`load_records`' R and the task of each phase."""
    records = json.loads((REPO / run / 'training_metrics.json').read_text())
    tasks = np.array([r['task'] for r in records])
    ends = list(np.flatnonzero(tasks[1:] != tasks[:-1])) + [len(tasks) - 1]
    R = np.array([[records[e][f'centroid_task{tasks[i]}'] for e in ends] for i in ends], float)
    return R, [int(tasks[e]) for e in ends]


def matrix(run, matrices):
    """R and the task of each phase: the forgetting pass when it scored the
    run, else the training records, as the figure reads it."""
    if run in matrices:
        return matrices[run], task_ids(run, matrices[run])
    return records_matrix(run)


def multi_task(meta, key):
    """A panel whose runs visit more than two distinct tasks."""
    return key.startswith(('paper/gymnax/data/noise_10task', 'paper/mjx/cheetah/data/noise_10task',
                           'paper/kinetix/data'))


def variants():
    """`{definition: extract}`: the figure's saved extract with F swapped in
    every multi-task panel."""
    meta = json.loads(sp.DATA.read_text())
    matrices = pass_matrices()
    out = {}
    for name, (_, fn) in DEFINITIONS.items():
        mm = copy.deepcopy(meta)
        for key, arms in mm['panels'].items():
            if not multi_task(mm, key):
                continue
            for trials in arms.values():
                for t in trials:
                    R, ids = matrix(t['run'], matrices)
                    f = float(fn(R, ids))
                    if name == 'final' and not np.isclose(f, t['F']):
                        sys.exit(f'final F of {t["run"]} is {f}, the figure has {t["F"]}')
                    t['F'] = f
        out[name] = mm
    return out


def halves():
    """Per multi-task panel and method: LA and previous-task F of the centroid
    over the first and the second half of the run, on the figure's rescaled
    axes (untrained network 0, best mean LA of any method 1)."""
    meta = json.loads(sp.DATA.read_text())
    matrices = pass_matrices()
    rows = []
    for fig, tree, cell, title in sp.pcl.PANELS:
        key = sp.pcl._key(tree, cell)
        if fig != 'main' or not multi_task(meta, key):
            continue
        env = cell.split('_sigma')[0]
        by_m = {m: [matrix(t['run'], matrices)[0] for t in trials]
                for m, trials in meta['panels'][key].items()}
        floor = sp.FLOOR[env]
        span = max(np.mean([np.diag(R).mean() for R in v]) for v in by_m.values()) - floor
        for m in [m for m in lp.METHOD_ORDER if m in by_m] + [m for m in by_m if m not in lp.METHOD_ORDER]:
            la = np.array([(np.diag(R) - floor) / span for R in by_m[m]])
            f = np.array([(np.diag(R)[:-1] - np.diag(R, 1)) / span for R in by_m[m]])
            h, g = la.shape[1] // 2, f.shape[1] // 2
            rows.append({'setting': title, 'method': m,
                         'la': [float(la[:, :h].mean()), float(la[:, h:].mean())],
                         'F': [float(f[:, :g].mean()), float(f[:, g:].mean())]})
    return rows


def main() -> int:
    # One PBT-PPO entry, whichever N a panel kept, as in Figure 2.
    lp.METHOD_STYLE['pbt'] = {**lp.METHOD_STYLE['pbt'], 'label': 'PBT-PPO'}
    rows = []
    for name, mm in variants().items():
        with contextlib.redirect_stdout(io.StringIO()):
            panels, _ = sp.main_panels(mm)
        keys = [sp.pcl._key(tree, cell) for fig, tree, cell, _ in sp.pcl.PANELS if fig == 'main']
        for key, (title, points) in zip(keys, panels):
            if not multi_task(mm, key) or not points:
                continue
            s = sp.summary(points)
            es = next(m for m in points if m in es_arm.ARMS)
            rl = [m for m in points if lp.FAMILY.get(m) == 'rl']
            best_rl = max(rl, key=lambda m: points[m]['s'][0])
            rows.append({'setting': title, 'definition': name,
                         'ring': s['ringed'], 'es': points[es]['s'][0],
                         'ga': points['ga']['s'][0] if 'ga' in points else None,
                         'best_rl': best_rl, 'best_rl_s': points[best_rl]['s'][0]})

    DATA.parent.mkdir(parents=True, exist_ok=True)
    split = halves()
    DATA.write_text(json.dumps({'definitions': rows, 'halves': split}, indent=1) + '\n')
    name_of = lambda m: '--' if m is None else sp._label(m)

    lines = ['% Built by scripts/analysis/forgetting_definitions.py -- rerun it, do not edit.',
             r'\begin{tabular}{llcrrl}', r'\toprule',
             r'Setting & Forgetting & Ring & ES & GA & Best RL \\', r'\midrule']
    settings = list(dict.fromkeys(r['setting'] for r in rows))
    for k, setting in enumerate(settings):
        for r in (r for r in rows if r['setting'] == setting):
            first = r['definition'] == 'final'
            ga = '--' if r['ga'] is None else f"{r['ga']:.2f}"
            lines.append(f"{setting if first else ''} & {DEFINITIONS[r['definition']][0]} & "
                         f"{name_of(r['ring'])} & {r['es']:.2f} & {ga} & "
                         f"{name_of(r['best_rl'])} {r['best_rl_s']:.2f} \\\\")
        if k < len(settings) - 1:
            lines.append(r'\midrule')
    lines += [r'\bottomrule', r'\end{tabular}']
    pathlib.Path(f'{STEM}.tex').write_text('\n'.join(lines) + '\n')

    md = ['# Forgetting definitions', '',
          'LA - F of the centroid on Figure 2\'s rescaled axes under four definitions of F; '
          'ring by the figure\'s rule. Built by `scripts/analysis/forgetting_definitions.py` '
          '(its docstring defines each F).', '',
          '| Setting | F | Ring | ES | GA | Best RL |', '|---|---|---|---|---|---|']
    for r in rows:
        ga = '--' if r['ga'] is None else f"{r['ga']:.2f}"
        md.append(f"| {r['setting']} | {DEFINITIONS[r['definition']][0]} | {name_of(r['ring'])} | "
                  f"{r['es']:.2f} | {ga} | {name_of(r['best_rl'])} {r['best_rl_s']:.2f} |")
    md += ['', '## First and second half of the run', '',
           'LA and previous-task F of the centroid over the first and the second half of '
           'the tasks, rescaled as Figure 2. On classic control and HalfCheetah the second '
           'half revisits the same ten tasks.', '',
           '| Setting | Method | LA 1st | LA 2nd | F 1st | F 2nd |', '|---|---|---|---|---|---|']
    for r in split:
        md.append(f"| {r['setting']} | {name_of(r['method'])} | {r['la'][0]:.2f} | {r['la'][1]:.2f} | "
                  f"{r['F'][0]:.2f} | {r['F'][1]:.2f} |")
    pathlib.Path(f'{STEM}.md').write_text('\n'.join(md) + '\n')
    print('\n'.join(md))
    return 0


if __name__ == '__main__':
    sys.exit(main())
