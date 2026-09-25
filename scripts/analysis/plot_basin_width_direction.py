"""The basin along the method's own path against random directions of the same
size (appendix; Figure 5's width uses random directions only).

Why: NE explores in weight space, so a random perturbation is one of its own
moves, but RL follows the gradient of the return, and Rahn et al. (2023) look
at the landscape along the algorithm's update directions. A basin that is
narrow in random directions could still be wide along the path RL takes.

Data: `child_survival.py direction` (results/direction_width/raw). Around every
centroid theta_A at the end of a task, with Delta = theta_B - theta_A the step
the run took over the next task, it scores on task A
  forward    theta_A + f Delta        (16 re-scores; f = 1 is theta_B)
  backward   theta_A - f Delta
  random     64 random directions with, in every tensor, f times the norm of
             Delta (the `matched` set of the shared-basin probe)
at f in DIRECTION_FRACTIONS (0.1 .. 8). The loss curve is the share of points
(random directions, or re-scores) whose rescaled return on A drops below
LEARNED (0.5, the rule of Figure 5); the width in units of the method's own
step is the f at which it crosses one half, log-linear between rungs, censored
at the ladder's ends. Same 13 ReLU panels, runs, solved masks and window (the
last five probed checkpoints) as plot_basin_width_return.py. The width in
relative-norm units is the step-unit width times the checkpoint's relative
step |Delta| / |theta_A| (mean over the window).

Reported:
  path / random   per trial the paired log ratio of the forward (and backward)
                  width to the random one; geometric mean over trials, Wilcoxon
                  over trials (Holm over a method's panels); the figure and the
                  primary table. 1 = the basin is as wide along the path as in
                  random directions; > 1 = the path stays inside the basin
                  further than random moves do.
  vs PPO          the forward width in relative units against PPO's, as Figure
                  5's ratio (Mann-Whitney on log width, Holm).

  .venv/bin/python scripts/analysis/plot_basin_width_direction.py --extract
  -> visuals/final/data/basin_width_direction.json,
     visuals/final/appendix/basin_width_direction.{pdf,png,md}
"""
import argparse
import datetime
import json
import pathlib
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt                            # noqa: E402
from scipy.stats import mannwhitneyu, wilcoxon             # noqa: E402

REPO = pathlib.Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO / 'scripts'))
sys.path.insert(0, str(REPO / 'scripts/analysis'))
import plot_basin_width_methods as b                        # noqa: E402
import plot_basin_width_return as ret                       # noqa: E402
import plot_generalist_scores as pgs                        # noqa: E402
import child_survival as cs                                 # noqa: E402
from make_lineplot import METHOD_STYLE                      # noqa: E402
from plot_stability_plasticity import FLOOR                 # noqa: E402
from plot_landscape_slices import LEARNED                   # noqa: E402

PROJECT = pgs.PROJECT
RAW = REPO / 'results' / 'direction_width' / 'raw'
OUT = pgs.FINAL / 'appendix' / 'basin_width_direction'
DATA = pgs.FINAL / 'data' / 'basin_width_direction.json'
LADDER = tuple(float(f) for f in cs.DIRECTION_FRACTIONS)
KINDS = ('random', 'forward', 'backward')
WINDOW = ret.WINDOW


def checkpoint_losses(row, floor, best):
    """{kind: [loss share per multiple]} of one probed checkpoint, plus its
    parent's rescaled return and relative step."""
    span = best - floor
    sc = {k: (np.asarray(v)[:, 0] - floor) / span for k, v in row['scores'].items()}
    out = {'parent': float(sc['copies:0'].mean()), 'step_rel': float(row['step_rel'])}
    for kind, key in (('random', 'matched:{:g}'), ('forward', 'actual:{:g}'), ('backward', 'actual:-{:g}')):
        out[kind] = [float(np.mean(sc[key.format(f)] < LEARNED)) for f in LADDER]
    return out


def extract():
    names = ret.overlap_panels()
    out = {'panels': {}, 'extracted': datetime.date.today().isoformat(),
           'raw': str(RAW.relative_to(REPO)), 'window': WINDOW, 'ladder': list(LADDER),
           'rule': f'rescaled return on A < {LEARNED}', 'half': ret.HALF}
    for label, tree, cell, key, arms, rel, scores in b.panel_trials(relu_only=True):
        name = names.get((tree, cell))
        floor = FLOOR[cell.split('_sigma')[0]]
        best = ret.panel_best(tree, cell, arms)
        panel = {}
        for row, (arm, trials, solved) in arms.items():
            path = RAW / f'{name}__{row}.json'
            if not path.exists():
                print(f'SKIP {label} {row}: no {path.name}')
                continue
            by_trial = {}
            for r in json.loads(path.read_text()):
                assert (PROJECT / tree / 'continual' / arm / cell / r['trial']).resolve() == \
                    (REPO / r['run']).resolve(), f'{path.name}: {r["run"]} is not the figure\'s run'
                by_trial.setdefault(r['trial'], {})[r['phase']] = r
            per_trial = {k: {'curve': [], 'width': [], 'censored': []} for k in KINDS}
            per_trial['step_rel'] = []
            counts = {'checkpoints': 0, 'unsolved': 0, 'missing': 0, 'frozen': 0}
            for t in trials:
                mask = np.asarray(solved[t], bool)
                T = len(mask)
                probed = by_trial.get(t, {})
                acc = {k: [] for k in KINDS}
                steps = []
                for p in range(T - 1 - WINDOW, T - 1):
                    if p not in probed:
                        counts['missing'] += 1
                        continue
                    if not mask[p]:
                        counts['unsolved'] += 1
                        continue
                    loss = checkpoint_losses(probed[p], floor, best)
                    if loss['step_rel'] <= 0:          # frozen: no step, so no direction
                        counts['frozen'] += 1
                        continue
                    counts['checkpoints'] += 1
                    steps.append(loss['step_rel'])
                    for k in KINDS:
                        acc[k].append(loss[k])
                per_trial['step_rel'].append(float(np.mean(steps)) if steps else None)
                for k in KINDS:
                    if acc[k]:
                        curve = np.mean(acc[k], axis=0)
                        w, cens = ret.crossing(curve, LADDER)
                        per_trial[k]['curve'].append(curve.tolist())
                        per_trial[k]['width'].append(w)
                        per_trial[k]['censored'].append(cens)
                    else:
                        for kk in per_trial[k]:
                            per_trial[k][kk].append(None)
            panel[row] = {'arm': arm, 'trials': list(trials), 'counts': counts, **per_trial}
        out['panels'][label] = panel
        print(f'{label:34s} {name:22s} ' + ' '.join(
            f'{r}={v["counts"]["checkpoints"]}/{v["counts"]["unsolved"]}u/{v["counts"]["frozen"]}f'
            for r, v in panel.items()))
    DATA.parent.mkdir(parents=True, exist_ok=True)
    DATA.write_text(json.dumps(out, indent=1) + '\n')
    print(f'wrote {DATA}')


def load():
    """{panel: {method: {kind: per-trial widths (step units)}, 'step_rel': ...}},
    trials with a solved checkpoint only."""
    blob = json.loads(DATA.read_text())
    out = {}
    for label, panel in blob['panels'].items():
        for m, v in panel.items():
            keep = [i for i, w in enumerate(v['random']['width']) if w is not None]
            if not keep:
                continue
            d = {k: np.array([v[k]['width'][i] for i in keep], float) for k in KINDS}
            d['step_rel'] = np.array([v['step_rel'][i] for i in keep], float)
            d['censored'] = {k: [v[k]['censored'][i] for i in keep] for k in KINDS}
            out.setdefault(label, {})[m] = d
    return out


def path_ratios(data, kind):
    """{method: {panel: (mean over trials of log2 kind/random width, Holm p)}}:
    paired Wilcoxon over trials, Holm over a method's panels."""
    out = {}
    for m in b.METHODS:
        cells = {p: d[m] for p, d in data.items() if m in d}
        lr = {p: np.log2(d[kind]) - np.log2(d['random']) for p, d in cells.items()}
        ps = {}
        for p, v in lr.items():
            ps[p] = 1.0 if (v.size < 5 or np.allclose(v, 0)) else float(wilcoxon(v).pvalue)
        adj = dict(zip(ps, b.holm(np.array(list(ps.values()))))) if ps else {}
        out[m] = {p: (float(np.mean(v)), adj[p]) for p, v in lr.items()}
    return out


def vs_ppo(data, kind):
    """{method: {panel: (log2 ratio to PPO of the geometric-mean width in
    relative units, Holm p)}}, as plot_basin_width_return.ratios."""
    out = {}
    for m in b.METHODS:
        if m == b.REFERENCE:
            continue
        cells = {p: d for p, d in data.items() if m in d and b.REFERENCE in d}
        rel = lambda d, mm: np.log(d[mm][kind] * d[mm]['step_rel'])                   # noqa: E731
        ps = {p: mannwhitneyu(rel(d, m), rel(d, b.REFERENCE)).pvalue for p, d in cells.items()}
        adj = dict(zip(ps, b.holm(np.array(list(ps.values()))))) if ps else {}
        out[m] = {p: (float((np.mean(rel(d, m)) - np.mean(rel(d, b.REFERENCE))) / np.log(2)), adj[p])
                  for p, d in cells.items()}
    return out


def write_markdown(path, data):
    lab = lambda m: 'ES' if m == 'es' else METHOD_STYLE[m]['label']                       # noqa: E731
    labels = list(data)
    md = [f'# {path.name}', '',
          'See the docstring of `scripts/analysis/plot_basin_width_direction.py`. Widths in units '
          f'of the method\'s own step over the next task, ladder {list(LADDER)}; loss = rescaled '
          f'return on the task just trained < {LEARNED}; width = the multiple at which the loss '
          'share crosses one half. "c" = some trial censored at an end of the ladder.', '']
    for kind in ('forward', 'backward'):
        R = path_ratios(data, kind)
        md += [f'## Width along the {kind} path / width in random directions of the same size', '',
               'Geometric mean over trials of the paired ratio (Holm p, Wilcoxon over trials); '
               'median over panels and Wilcoxon over panels in the last columns.', '',
               '| Method | ' + ' | '.join(labels) + ' | median | Wilcoxon p | n |',
               '|---|' + '---|' * (len(labels) + 3)]
        for m in b.METHODS:
            cells = R.get(m, {})
            med, pw, n = b.across([v[0] for v in cells.values()])
            md.append(f'| {lab(m)} | ' + ' | '.join(
                (f'{2 ** cells[p][0]:.2f} ({cells[p][1]:.2g})'
                 + (' c' if any(data[p][m]['censored'][k].count(None) < len(data[p][m]['censored'][k])
                               for k in (kind, 'random')) else ''))
                if p in cells else '-' for p in labels)
                + f' | x{2 ** med:.2f} | {pw:.2g} | {n} |')
        md.append('')
    md += ['## Geometric mean width per method (step units): random / forward / backward', '',
           '| Method | ' + ' | '.join(labels) + ' |', '|---|' + '---|' * len(labels)]
    for m in b.METHODS:
        md.append(f'| {lab(m)} | ' + ' | '.join(
            ' / '.join(f'{np.exp(np.mean(np.log(data[p][m][k]))):.2f}' for k in KINDS)
            if m in data[p] else '-' for p in labels) + ' |')
    md += ['', '## Mean relative step |Delta| / |theta_A| per method (window checkpoints)', '',
           '| Method | ' + ' | '.join(labels) + ' |', '|---|' + '---|' * len(labels)]
    for m in b.METHODS:
        md.append(f'| {lab(m)} | ' + ' | '.join(
            f'{np.mean(data[p][m]["step_rel"]):.3f}' if m in data[p] else '-' for p in labels) + ' |')
    for kind in ('forward', 'random'):
        R = vs_ppo(data, kind)
        md += ['', f'## Width along the {kind} path in relative-norm units, ratio to PPO', '',
               '(step-unit width x relative step; Mann-Whitney on log width over trials, Holm over '
               'a method\'s panels; compare Figure 5\'s random-direction ratios)', '',
               '| Method | ' + ' | '.join(labels) + ' | median | Wilcoxon p | n |',
               '|---|' + '---|' * (len(labels) + 3)]
        for m in b.TABLE_ORDER:
            cells = R.get(m, {})
            med, pw, n = b.across([v[0] for v in cells.values()])
            md.append(f'| {lab(m)} | ' + ' | '.join(
                f'{2 ** cells[p][0]:.2f} ({cells[p][1]:.2g})' if p in cells else '-' for p in labels)
                + f' | x{2 ** med:.2f} | {pw:.2g} | {n} |')
    md += ['', '## Mean loss curve per method along the ladder (forward path | random), over trials', '']
    for p in labels:
        md += [f'### {p}', '', '| Method | ' + ' | '.join(f'{f:g}' for f in LADDER) + ' |',
               '|---|' + '---|' * len(LADDER)]
        blob = json.loads(DATA.read_text())['panels'][p]
        for m in b.METHODS:
            if m not in blob:
                continue
            row = []
            for k in ('forward', 'random'):
                curves = [c for c in blob[m][k]['curve'] if c is not None]
                row.append(np.mean(curves, axis=0) if curves else None)
            if row[0] is not None:
                md.append(f'| {lab(m)} | ' + ' | '.join(f'{a:.2f} | {r:.2f}' for a, r in zip(*row)) + ' |')
        md.append('')
    md += ['## Trials with a solved, moving checkpoint in the window (frozen checkpoints, '
           'step 0, are left out: no direction)', '',
           '| Method | ' + ' | '.join(labels) + ' |', '|---|' + '---|' * len(labels)]
    for m in b.METHODS:
        md.append(f'| {lab(m)} | ' + ' | '.join(
            str(len(data[p][m]['random'])) if m in data[p] else '-' for p in labels) + ' |')
    path.with_suffix('.md').write_text('\n'.join(md) + '\n')
    print(f'wrote {path}.md')


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--extract', action='store_true')
    ap.add_argument('--lim', type=float, default=4.0, help='axis range in log2 units')
    ap.add_argument('--font-size', type=float, default=7.0)
    args = ap.parse_args()
    if args.extract:
        extract()
    if not DATA.exists():
        sys.exit(f'no {DATA}: run with --extract first')
    fs = args.font_size
    plt.rcParams.update({
        'font.size': fs, 'axes.titlesize': fs, 'xtick.labelsize': fs - 1,
        'ytick.labelsize': fs, 'axes.linewidth': 0.5, 'xtick.major.width': 0.5,
        'ytick.major.width': 0.5, 'xtick.major.size': 2, 'ytick.major.size': 0,
    })
    METHOD_STYLE['pbt'] = {**METHOD_STYLE['pbt'], 'label': 'PBT-PPO'}
    b.LIM = args.lim
    panels = [p for p in b.panel_list() if not b.is_absolute(p[3])]
    data = load()
    R = path_ratios(data, 'forward')
    b.draw_summary(OUT, panels, R, list(b.METHODS), None, fs,
                   xlabel='Basin width along the own path / random directions (×)', ref_label='')
    write_markdown(OUT, data)
    return 0


if __name__ == '__main__':
    sys.exit(main())
