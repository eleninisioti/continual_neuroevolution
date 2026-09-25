"""Basin width of every method relative to PPO, measured on the RETURN: the
size of random weight perturbation at which half of them lose the task the
policy was trained on. The return-space version of `plot_basin_width_methods`
(the action-change proxy): same panels, same runs, same solved-checkpoint
rule, same drawing.

    # re-read the ladder passes (only when they change)
    .venv/bin/python scripts/analysis/plot_basin_width_return.py --extract
    # redraw from visuals/final/data/
    .venv/bin/python scripts/analysis/plot_basin_width_return.py [--rule good|keep]

    -> paper/visuals/final/basin_width_return.{pdf,png,md}       (main text: the summary)
       paper/visuals/final/data/basin_width_return.json
       paper/visuals/landscape/basin_width_return_table.{pdf,png} (appendix: every panel)

Data: `child_survival.py overlap --radii 0.001 0.003 0.01 0.03 0.1 0.3 1 3 --out
overlap_ladder` (results/overlap_ladder/raw; MiniGrid was run on the six rungs
from 0.01, every width there lies inside them). Around every centroid saved at
the end of a task in the second half of a run it scores 64 random Gaussian
directions at each radius on the task the checkpoint was trained on, with the reported
curve's episode count (10; MiniGrid 16), plus 4 exact copies of the parent
scored the same way. The noise in every parameter tensor is scaled to the
radius times that tensor's norm (the convention of the action-change figure;
ReLU networks only, whose function is invariant to a layer's scale), so a
radius of 1 moves every tensor by its own norm. Every point is a fresh
rollout, so the parent's reference score is the mean of its copies, never the
stored fitness.

Width. A perturbed copy is IN the basin while it still solves the task: its
return, rescaled per panel (0 = untrained network, 1 = the best mean
post-switch return of any method), stays at or above LEARNED (0.5), the same
rule that admits the checkpoint (below). `--rule good` asks for 0.8 instead
(the shared-basin figure's "solves"), `--rule keep` for staying within 0.05 of
the parent's own rescaled return. Per trial, the loss share (fraction of the 64
directions that leave the basin) at each radius is averaged over the trial's
solved checkpoints in the window, and the WIDTH is the radius at which this
curve crosses one half, by log-linear interpolation between the ladder's
radii. A curve still under one half at the top rung gives that radius as its
width, one already over one half at the bottom rung that radius; both are
censored and counted in the markdown. A method's width relative to PPO on a panel is the ratio of
the geometric means over trials, tested with a Mann-Whitney U test over trials
on log width, Holm over the panels of a method; the summary takes the median
log ratio over panels with a Wilcoxon signed-rank test, as the action figure.

Panels, runs, arms, trials and the solved masks are `plot_basin_width_methods.
panel_trials`'s: the 13 ReLU panels (the generalist grid's gymnax and MiniGrid
panels and Figure 2's ten-task noise runs). The tanh panels (HalfCheetah,
Kinetix) are not on the ladder; Kinetix's return width is tab:basin_return. A
checkpoint counts only when it solves its task by evaluation.json, as there.
Window: the last five PROBED checkpoints by position (the final checkpoint of
a run has no next task and the overlap probe skips it), unsolved ones left
out; a trial with none drops out of the panel.
"""

from __future__ import annotations

import argparse
import datetime
import json
import pathlib
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt                            # noqa: E402
from scipy.stats import mannwhitneyu                       # noqa: E402

REPO = pathlib.Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO / 'scripts'))
sys.path.insert(0, str(REPO / 'scripts/analysis'))
import plot_basin_width_methods as b                        # noqa: E402
import plot_generalist_scores as pgs                        # noqa: E402
from make_lineplot import METHOD_STYLE                      # noqa: E402
from plot_stability_plasticity import FLOOR                 # noqa: E402
from plot_landscape_slices import LEARNED                   # noqa: E402

PROJECT = pgs.PROJECT
RAW = REPO / 'results' / 'overlap_ladder' / 'raw'
MAIN = pgs.FINAL / 'basin_width_return'
TABLE = b.TABLE.with_name('basin_width_return_table')
DATA = pgs.FINAL / 'data' / 'basin_width_return.json'
LADDER = (0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0)   # gymnax; MiniGrid has no 0.001 / 0.003 rung
GOOD = 0.8                 # plot_shared_basin's "solves"
KEEP_TOL = 0.05            # keep: within tol x span of the parent's re-score
RULES = ('solved', 'good', 'keep')
HALF = 0.5
WINDOW = 5
N_DIR = 64


def overlap_panels():
    """{(tree, cell): overlap panel name} from child_survival."""
    import child_survival as cs
    return {(tree, cell): name for name, tree, cell, _ in cs.OVERLAP_PANELS}


def panel_best(tree, cell, arms):
    """The solved rule's `best`: highest over methods of the mean over trials of
    the mean post-switch return (evaluation.json), as plot_basin_width_methods."""
    means = []
    for row, (arm, trials, _) in arms.items():
        per_trial = []
        for t in trials:
            got = pgs.from_evaluation(PROJECT / tree / 'continual' / arm / cell / t)
            per_trial.append(np.asarray(got[0], dtype=float)[1:].mean())
        means.append(np.mean(per_trial))
    return max(means)


def checkpoint_losses(row, floor, best):
    """{rule: [loss share per ladder radius]} of one probed checkpoint."""
    span = best - floor
    radii = [float(r) for r in row['radii']]
    assert all(r in LADDER for r in radii), f'{row["run"]} radii {radii} off the ladder'
    sc = {k: (np.asarray(v)[:, 0] - floor) / span for k, v in row['scores'].items()}
    ref = float(sc['copies:0'].mean())
    out = {'parent': ref, 'radii': radii}
    for rule in RULES:
        thr = {'solved': LEARNED, 'good': GOOD, 'keep': ref - KEEP_TOL}[rule]
        out[rule] = [float(np.mean(sc[f'fixed:{e:g}'] < thr)) for e in radii]
    return out


def crossing(curve, radii, level=HALF):
    """(width, censored): the radius where the loss curve reaches `level`,
    log-linear between ladder points; censored at the ladder's ends."""
    c = np.asarray(curve, float)
    lr = np.log10(radii)
    if c[0] >= level:
        return float(radii[0]), 'low'
    if c[-1] < level:
        return float(radii[-1]), 'high'
    i = int(np.argmax(c >= level))               # first radius at or over the level
    f = (level - c[i - 1]) / max(c[i] - c[i - 1], 1e-12)
    return float(10 ** (lr[i - 1] + f * (lr[i] - lr[i - 1]))), None


def extract():
    names = overlap_panels()
    out = {'panels': {}, 'extracted': datetime.date.today().isoformat(),
           'raw': str(RAW.relative_to(REPO)), 'window': WINDOW, 'n_dir': N_DIR,
           'ladder': list(LADDER), 'half': HALF, 'minigrid_ladder': list(LADDER[2:]),
           'rules': {'solved': f'rescaled return < {LEARNED}', 'good': f'rescaled return < {GOOD}',
                     'keep': f'rescaled return < parent - {KEEP_TOL}'}}
    for label, tree, cell, key, arms, rel, scores in b.panel_trials():
        if b.is_absolute(tree):
            continue                                   # tanh panels: not on the ladder
        name = names.get((tree, cell))
        if name is None:
            print(f'SKIP {label}: no overlap panel for {tree} {cell}')
            continue
        floor = FLOOR[cell.split('_sigma')[0]]
        best = panel_best(tree, cell, arms)
        panel = {}
        for row, (arm, trials, solved) in arms.items():
            path = RAW / f'{name}__{row}.json'
            if not path.exists():
                print(f'SKIP {label} {row}: no {path.name}')
                continue
            raw = json.loads(path.read_text())
            by_trial = {}
            for r in raw:
                assert (PROJECT / tree / 'continual' / arm / cell / r['trial']).resolve() == \
                    (REPO / r['run']).resolve(), f'{path.name}: {r["run"]} is not the figure\'s run'
                by_trial.setdefault(r['trial'], {})[r['phase']] = r
            per_trial = {rule: {'curve': [], 'width': [], 'censored': []} for rule in RULES}
            radii = None
            counts = {'checkpoints': 0, 'unsolved': 0, 'missing': 0}
            for t in trials:
                mask = np.asarray(solved[t], bool)
                T = len(mask)
                probed = by_trial.get(t, {})
                positions = list(range(T - 1 - WINDOW, T - 1))     # the last WINDOW probed ones
                acc = {rule: [] for rule in RULES}
                for p in positions:
                    if p not in probed:
                        counts['missing'] += 1
                        continue
                    if not mask[p]:
                        counts['unsolved'] += 1
                        continue
                    counts['checkpoints'] += 1
                    loss = checkpoint_losses(probed[p], floor, best)
                    assert radii in (None, loss['radii']), f'{path.name}: rows with different radii'
                    radii = loss['radii']
                    for rule in RULES:
                        acc[rule].append(loss[rule])
                for rule in RULES:
                    if acc[rule]:
                        curve = np.mean(acc[rule], axis=0)
                        w, cens = crossing(curve, radii)
                        per_trial[rule]['curve'].append(curve.tolist())
                        per_trial[rule]['width'].append(w)
                        per_trial[rule]['censored'].append(cens)
                    else:
                        for k in per_trial[rule]:
                            per_trial[rule][k].append(None)
            panel[row] = {'arm': arm, 'trials': list(trials), 'counts': counts, 'radii': radii,
                          **per_trial}
        out['panels'][label] = panel
        print(f'{label:34s} {name:22s} ' + ' '.join(
            f'{r}={v["counts"]["checkpoints"]}/{v["counts"]["unsolved"]}u/{v["counts"]["missing"]}m'
            for r, v in panel.items()))
    DATA.parent.mkdir(parents=True, exist_ok=True)
    DATA.write_text(json.dumps(out, indent=1) + '\n')
    print(f'wrote {DATA}')


def widths(rule):
    """({panel label: {method: np.array(per-trial width)}},
    {(panel, method): (n censored high, n censored low)})."""
    blob = json.loads(DATA.read_text())
    out, cens = {}, {}
    for label, panel in blob['panels'].items():
        for m, v in panel.items():
            vals = np.array([x for x in v[rule]['width'] if x is not None], float)
            if not vals.size:
                continue
            out.setdefault(label, {})[m] = vals
            c = v[rule]['censored']
            cens[(label, m)] = (c.count('high'), c.count('low'))
    return out, cens


def ratios(data):
    """{method: {panel: (log2 width ratio vs PPO, Holm p)}}: geometric means over
    trials, Mann-Whitney on log widths, Holm over a method's panels."""
    out = {}
    for m in b.METHODS:
        if m == b.REFERENCE:
            continue
        cells = {p: d for p, d in data.items() if m in d and b.REFERENCE in d}
        ps = {p: mannwhitneyu(np.log(d[m]), np.log(d[b.REFERENCE])).pvalue for p, d in cells.items()}
        adj = dict(zip(ps, b.holm(np.array(list(ps.values()))))) if ps else {}
        out[m] = {p: (float(np.mean(np.log2(d[m])) - np.mean(np.log2(d[b.REFERENCE]))), adj[p])
                  for p, d in cells.items()}
    return out


def write_markdown(path, rule):
    lab = lambda m: 'ES' if m == 'es' else METHOD_STYLE[m]['label']                       # noqa: E731
    blob = json.loads(DATA.read_text())
    labels = list(blob['panels'])
    md = [f'# {path.name} (and {TABLE.name})', '',
          'See the docstring of `scripts/analysis/plot_basin_width_return.py`. Width = the '
          'relative radius of Gaussian weight noise at which half of 64 random perturbations '
          f'no longer solve the task (drawn rule `{rule}`: {blob["rules"][rule]}); ladder '
          f'{blob["ladder"]} (MiniGrid: without the two lowest rungs). Ratio = geometric mean width of the method / PPO\'s over trials '
          '(Holm p over the method\'s panels, Mann-Whitney on log width). Above 1 = wider than '
          'PPO. "c" marks a cell where some trial\'s curve never crossed one half inside the '
          'ladder (width set to the end of the ladder; the count is in the last table).', '']
    for r in RULES:
        data, cens = widths(r)
        R = ratios(data)
        head = 'PRIMARY, drawn' if r == rule else 'robustness'
        md += [f'## Rule `{r}` ({head})', '',
               '| Method | ' + ' | '.join(labels) + ' | median | Wilcoxon p | n |',
               '|---|' + '---|' * (len(labels) + 3)]
        for m in b.TABLE_ORDER:
            cells = R.get(m, {})
            med, pw, n = b.across([v[0] for v in cells.values()])
            md.append(f'| {lab(m)} | ' + ' | '.join(
                (f'{2 ** cells[p][0]:.2f} ({cells[p][1]:.2g})'
                 + (' c' if sum(cens.get((p, m), (0, 0))) + sum(cens.get((p, 'ppo'), (0, 0))) else ''))
                if p in cells else '-' for p in labels)
                + f' | x{2 ** med:.2f} | {pw:.2g} | {n} |')
        md += ['', 'Geometric mean width per method (relative radius), the ratio\'s terms:', '',
               '| Method | ' + ' | '.join(labels) + ' |', '|---|' + '---|' * len(labels)]
        for m in b.METHODS:
            md.append(f'| {lab(m)} | ' + ' | '.join(
                f'{np.exp(np.mean(np.log(data[p][m]))):.3g}' if p in data and m in data[p] else '-'
                for p in labels) + ' |')
        md += ['', 'Trials censored high / low (curve never reached one half by radius 3 / '
               'already over it at 0.01) out of trials with a solved checkpoint:', '',
               '| Method | ' + ' | '.join(labels) + ' |', '|---|' + '---|' * len(labels)]
        for m in b.METHODS:
            md.append(f'| {lab(m)} | ' + ' | '.join(
                f'{cens[(p, m)][0]}/{cens[(p, m)][1]} of {len(data[p][m])}' if (p, m) in cens else '-'
                for p in labels) + ' |')
        md.append('')
    md += ['## Mean loss curve per method at the ladder radii (rule `solved`, over trials)', '']
    for p in labels:
        radii = next((v['radii'] for v in blob['panels'][p].values() if v.get('radii')), LADDER)
        md += [f'### {p}', '', '| Method | ' + ' | '.join(f'{e:g}' for e in radii) + ' |',
               '|---|' + '---|' * len(radii)]
        for m in b.METHODS:
            if m not in blob['panels'][p]:
                continue
            curves = [c for c in blob['panels'][p][m]['solved']['curve'] if c is not None]
            if curves:
                md.append(f'| {lab(m)} | ' + ' | '.join(f'{x:.2f}' for x in np.mean(curves, axis=0)) + ' |')
        md.append('')
    md += ['## Checkpoints per panel (solved in window / unsolved / not probed)', '',
           '| Method | ' + ' | '.join(labels) + ' |', '|---|' + '---|' * len(labels)]
    for m in b.METHODS:
        md.append(f'| {lab(m)} | ' + ' | '.join(
            (lambda c: f'{c["checkpoints"]}/{c["unsolved"]}/{c["missing"]}')(blob['panels'][p][m]['counts'])
            if m in blob['panels'][p] else '-' for p in labels) + ' |')
    path.with_suffix('.md').write_text('\n'.join(md) + '\n')
    print(f'wrote {path}.md')


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--extract', action='store_true')
    ap.add_argument('--rule', default='solved', choices=RULES)
    ap.add_argument('--lim', type=float, default=b.LIM,
                    help='colour / axis range in log2 units (the action figure uses 4)')
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
    data, _ = widths(args.rule)
    R = ratios(data)
    rows = [m for m in b.METHODS if m != b.REFERENCE]
    sfx = '' if args.rule == 'solved' else f'_{args.rule}'
    b.draw_table(TABLE.with_name(TABLE.name + sfx), panels, R,
                 [m for m in b.TABLE_ORDER if m in rows], None, fs)
    b.draw_summary(MAIN.with_name(MAIN.name + sfx), panels, R, rows, None, fs)
    write_markdown(MAIN.with_name(MAIN.name + sfx), args.rule)
    return 0


if __name__ == '__main__':
    sys.exit(main())
