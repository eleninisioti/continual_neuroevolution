"""The shared basin over training: the companion of `basin_width_evolution`
for the right panel of Figure 5, on the same 13 ReLU panels and grid.

    .venv/bin/python scripts/analysis/plot_shared_basin_evolution.py --extract
    .venv/bin/python scripts/analysis/plot_shared_basin_evolution.py

    -> paper/visuals/final/appendix/shared_basin_evolution.{pdf,png,md}
       paper/visuals/final/data/shared_basin_evolution.json

Per panel, method and checkpoint t (the centroid at the end of task t), the
shared basin of plot_shared_basin: the share of 64 random moves of radius 0.1
of every tensor's norm (KEY) after which the policy solves both task t and
task t + 1 (rescaled score >= GOOD on each), from the
overlap probe (`child_survival.py overlap`). The paper's probe covered the
second half of every run (results/overlap_probe, checkpoints 9..18); the first
half comes from results/overlap_first_half (posthoc_overlap_first_half.sh).
Checkpoints whose own task is not solved (parent < GOOD) are left out, as in
overlap_points. Mean over trials with a 95% bootstrap band. Panels, runs and
trials are those of basin_width_main (plot_basin_width_methods.panel_trials).

The markdown gives, per panel and method, the mean over the first three and
the last five checkpoints and the Spearman correlation of the share with the
checkpoint index over (trial, checkpoint) pairs (negative = the shared basin
shrinks over training), and the same pooled over panels.
"""
from __future__ import annotations

import argparse
import datetime
import json
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt                            # noqa: E402
from scipy.stats import spearmanr                          # noqa: E402

import plot_basin_width_evolution as evo                   # noqa: E402
import plot_basin_width_methods as b                        # noqa: E402
import plot_basin_width_return as ret                       # noqa: E402
import child_survival as cs                                 # noqa: E402
from make_lineplot import METHOD_STYLE                      # noqa: E402

pgs = b.pgs
REPO = b.REPO
PROJECT = pgs.PROJECT
RAWS = [REPO / 'results' / 'overlap_probe' / 'raw', REPO / 'results' / 'overlap_first_half' / 'raw']
OUT = pgs.FINAL / 'appendix' / 'shared_basin_evolution'
DATA = pgs.FINAL / 'data' / 'shared_basin_evolution.json'
KEY, GOOD, TOL = 'fixed:0.1', 0.5, 0.05   # plot_shared_basin's definition (own step at 0.8 until 2026-09-25)


def extract():
    names = ret.overlap_panels()
    out = {'panels': {}, 'groups': {}, 'key': KEY, 'good': GOOD,
           'raw': [str(r.relative_to(REPO)) for r in RAWS],
           'extracted': datetime.date.today().isoformat()}
    for label, tree, cell, key, arms, rel, scores in b.panel_trials(relu_only=True):
        name = names[(tree, cell)]
        rows = {}
        for row in arms:
            for raw in RAWS:
                path = raw / f'{name}__{row}.json'
                if path.exists():
                    rows.setdefault(row, []).extend(json.loads(path.read_text()))
        # the rescaling reference of overlap_points: the best mean parent score
        # of any arm, over every probed checkpoint
        best = max(np.mean([np.asarray(r['scores']['copies:0'])[:, 0].mean() for r in rs])
                   for rs in rows.values())
        panel = {}
        for row, (arm, trials, solved) in arms.items():
            if row not in rows:
                print(f'SKIP {label} {row}: no raw rows')
                continue
            by_trial = {}
            for r in rows[row]:
                assert (PROJECT / tree / 'continual' / arm / cell / r['trial']).resolve() == \
                    (REPO / r['run']).resolve(), f'{name}__{row}: {r["run"]} is not the figure\'s run'
                by_trial.setdefault(r['trial'], {})[r['phase']] = r
            both, steps, probed = [], [], 0
            for t in trials:
                T = len(solved[t]) - 1                     # checkpoints 0..T-2 have a next task
                series, step = [None] * T, [None] * T
                for p, r in by_trial.get(t, {}).items():
                    st = cs.overlap_stats(r, best, TOL, GOOD)
                    probed += 1
                    step[p] = float(r['step_rel'])
                    if st['parent_A'] >= GOOD:
                        series[p] = st[KEY]['both']
                both.append(series)
                steps.append(step)
            panel[row] = {'arm': arm, 'trials': list(trials), 'both': both, 'step_rel': steps}
            print(f'{label:34s} {row:7s} {probed:4d} probed checkpoints over {len(trials)} trials')
        out['panels'][label] = panel
    out['groups'] = {f'{body}, {change}': group for group, body, change, tree, _ in b.panel_list()
                     if not b.is_absolute(tree)}
    DATA.parent.mkdir(parents=True, exist_ok=True)
    DATA.write_text(json.dumps(out, indent=1) + '\n')
    print(f'wrote {DATA}')


def series(blob, key='both'):
    """{panel: {method: (trials, T) array, NaN where not probed or not solved}}."""
    out = {}
    for label, panel in blob['panels'].items():
        for m, v in panel.items():
            arr = np.array([[np.nan if x is None else x for x in s] for s in v[key]], float)
            if arr.size and np.isfinite(arr).any():
                out.setdefault(label, {})[m] = arr
    return out


def trend(arr):
    idx = np.tile(np.arange(arr.shape[1]), arr.shape[0])
    ok = np.isfinite(arr.ravel())
    return spearmanr(idx[ok], arr.ravel()[ok])[0] if ok.sum() > 2 else np.nan


def write_markdown(blob, data, path):
    lab = lambda m: 'ES' if m == 'es' else METHOD_STYLE[m]['label']                       # noqa: E731
    steps = series(blob, 'step_rel')
    lines = ['# shared_basin_evolution', '',
             'Shared basin per checkpoint (share of random moves of radius 0.1 of each tensor\'s norm '
             f'that solve both the task just trained and the next; rescaled score >= {GOOD}), '
             'mean over trials. Per method: first three checkpoints, last five, Spearman rho of '
             'the share vs checkpoint index over (trial, checkpoint) pairs (- = shrinks over '
             f'training), probed (trial, checkpoint) pairs, and the relative step |Delta| / |theta| '
             'over the first three and last five checkpoints (the unit of the moves). '
             f'Data `{DATA.relative_to(REPO)}`, '
             f'extracted {blob["extracted"]}.', '']
    for label, ser in data.items():
        lines += [f'## {label}', '', '| Method | trials | first 3 | last 5 | rho | probed | step first 3 / last 5 |',
                  '|---|---|---|---|---|---|---|']
        for m in evo.METHODS:
            arr = ser.get(m)
            if arr is None:
                continue
            lines.append(f'| {lab(m)} | {arr.shape[0]} | {np.nanmean(arr[:, :3]):.2f} | '
                         f'{np.nanmean(arr[:, -5:]):.2f} | {trend(arr):+.2f} | '
                         f'{int(np.isfinite(arr).sum())}/{arr.size} | '
                         f'{np.nanmean(steps[label][m][:, :3]):.3f} / {np.nanmean(steps[label][m][:, -5:]):.3f} |')
        lines.append('')
    lines += ['## Across panels', '', '| Method | median rho | panels rho < 0 | median first 3 | median last 5 | panels | median step first 3 / last 5 |',
              '|---|---|---|---|---|---|---|']
    for m in evo.METHODS:
        rhos = np.array([trend(s[m]) for s in data.values() if m in s])
        first = np.array([np.nanmean(s[m][:, :3]) for s in data.values() if m in s])
        last = np.array([np.nanmean(s[m][:, -5:]) for s in data.values() if m in s])
        sf = np.array([np.nanmean(steps[p][m][:, :3]) for p in data if m in data[p]])
        sl = np.array([np.nanmean(steps[p][m][:, -5:]) for p in data if m in data[p]])
        lines.append(f'| {lab(m)} | {np.nanmedian(rhos):+.2f} | {(rhos < 0).sum()} | '
                     f'{np.nanmedian(first):.2f} | {np.nanmedian(last):.2f} | {rhos.size} | '
                     f'{np.nanmedian(sf):.3f} / {np.nanmedian(sl):.3f} |')
    path.with_suffix('.md').write_text('\n'.join(lines) + '\n')
    print(f'wrote {path}.md')


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--extract', action='store_true')
    ap.add_argument('--font-size', type=float, default=7.0)
    args = ap.parse_args()
    if args.extract:
        extract()
    if not DATA.exists():
        sys.exit(f'no {DATA}: run with --extract first')
    blob = json.loads(DATA.read_text())
    data = series(blob)
    fs = args.font_size
    plt.rcParams.update({'font.size': fs, 'axes.titlesize': fs, 'axes.linewidth': 0.5,
                         'xtick.major.width': 0.5, 'ytick.major.width': 0.5})
    METHOD_STYLE['pbt'] = {**METHOD_STYLE['pbt'], 'label': 'PBT-PPO'}
    stats = {m: f' {np.nanmedian([trend(s[m]) for s in data.values() if m in s]):+.2f}'
             for m in evo.METHODS}
    evo.draw({'groups': blob['groups'], 'absolute': []}, None, fs, OUT, series=data,
             ylabel='Shared basin (share of moves keeping both tasks)', invert=False,
             ylim=(0, 1), relative_row=False, legend_stats=stats)
    write_markdown(blob, data, OUT)
    return 0


if __name__ == '__main__':
    sys.exit(main())
