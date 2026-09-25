"""Is the basin width's level arbitrary? The width of `plot_basin_width_return`
is the radius at which HALF of the random perturbations no longer SOLVE the
task. This script reads the same ladder passes and re-summarises them under
other levels and other references, then reports the ranking of the methods
under each, so the appendix can say that the choice moves the magnitudes and
not the ordering. It also draws the survival curves the width is read from.

    # re-read the ladder passes (only when they change)
    .venv/bin/python scripts/analysis/plot_basin_width_levels.py --extract
    # redraw / retabulate from visuals/final/data/
    .venv/bin/python scripts/analysis/plot_basin_width_levels.py

    -> paper/visuals/final/appendix/basin_width_levels.{tex,md}   (appendix: tab:basin_levels)
       paper/visuals/final/appendix/basin_width_curves.{pdf,png}  (appendix: fig:basin_curves)
       paper/visuals/final/data/basin_width_levels.json

Definitions. Every one is a radius read off a curve over the ladder (log-linear
between rungs, censored at its ends as in plot_basin_width_return):

- `solved` at level L: the share of the 64 perturbations whose rescaled return
  falls below the solved line (0.5; 0 = untrained network, 1 = the panel's best
  method) reaches L. L = 0.5 is the paper's width; 0.25, 0.75 and 0.9 are the
  sweep.
- `good` at L = 0.5: the same with the shared basin's stricter line (0.8).
- `ratio` at level r: Lehman et al.'s robustness score, the median return of the
  perturbed copies divided by the unperturbed policy's own re-scored return (both
  rescaled), falls below r. r = 0.5 is the score they quote as an example; 0.9,
  0.75 and 0.25 are the sweep. No solved line enters.

Per trial the curve is the mean over its solved checkpoints in the window
(plot_basin_width_return's rule and window), and the width its crossing. A
method's width relative to PPO on a panel is the ratio of geometric means over
trials; the table gives the median over the 13 ReLU panels with a Wilcoxon
signed-rank test, and the Spearman correlation of the 78 per-panel log ratios
(6 methods x 13 panels) with those of the paper's definition.

The curves figure draws, per panel, the share of perturbations that still solve
the task against the radius (rule `solved`, mean over trials), one line per
method; the dotted line is the level the width is read at.
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
from matplotlib.lines import Line2D                        # noqa: E402
from scipy.stats import spearmanr                          # noqa: E402

REPO = pathlib.Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO / 'scripts'))
sys.path.insert(0, str(REPO / 'scripts/analysis'))
import plot_basin_width_methods as b                        # noqa: E402
import plot_basin_width_return as r                         # noqa: E402
import plot_generalist_scores as pgs                        # noqa: E402
from make_lineplot import METHOD_STYLE                      # noqa: E402
from plot_stability_plasticity import FLOOR                 # noqa: E402

PROJECT = pgs.PROJECT
APPX = pgs.FINAL / 'appendix'
TABLE = APPX / 'basin_width_levels'
CURVES = APPX / 'basin_width_curves'
DATA = pgs.FINAL / 'data' / 'basin_width_levels.json'
MAIN_DEF = ('solved', 0.5)
# (curve, level) in table order; the curve is a loss share except `ratio`
DEFS = [('solved', 0.25), ('solved', 0.5), ('solved', 0.75), ('solved', 0.9),
        ('good', 0.5),
        ('ratio', 0.9), ('ratio', 0.75), ('ratio', 0.5), ('ratio', 0.25)]
CURVE_LABEL = {                        # the caption carries the explanation
    'solved': r'Share no longer solving, return $< 0.5$',
    'good': r'Share no longer solving, return $< 0.8$',
    'ratio': r'Median return relative to own',
}
LABEL = {m: ('ES' if m == 'es' else 'PBT-PPO' if m == 'pbt' else METHOD_STYLE[m]['label'])
         for m in b.METHODS}


def ratio_curve(row, floor, best):
    """Lehman's robustness score at every rung: median rescaled return of the 64
    perturbed copies over the parent's own re-scored (rescaled) return."""
    span = best - floor
    sc = {k: (np.asarray(v)[:, 0] - floor) / span for k, v in row['scores'].items()}
    ref = float(sc['copies:0'].mean())
    return [float(np.median(sc[f'fixed:{e:g}']) / ref) for e in row['radii']]


def width_of(curve, radii, kind, level):
    """(width, censored) of a definition; a ratio curve falls, a loss curve rises."""
    if kind == 'ratio':
        return r.crossing(1 - np.asarray(curve, float), radii, level=1 - level)
    return r.crossing(curve, radii, level=level)


def extract():
    names = r.overlap_panels()
    out = {'panels': {}, 'extracted': datetime.date.today().isoformat(),
           'raw': str(r.RAW.relative_to(REPO)), 'window': r.WINDOW, 'n_dir': r.N_DIR,
           'ladder': list(r.LADDER), 'defs': [list(d) for d in DEFS], 'main': list(MAIN_DEF)}
    # only the ReLU panels are on the ladder; drop the tanh ones before
    # panel_trials checks their width passes (HalfCheetah's may lag a relink)
    relu = [p for p in b.panel_list() if not b.is_absolute(p[3])]
    b.panel_list = lambda: relu
    for label, tree, cell, key, arms, rel, scores in b.panel_trials():
        name = names.get((tree, cell))
        if name is None:
            print(f'SKIP {label}: no overlap panel for {tree} {cell}')
            continue
        floor = FLOOR[cell.split('_sigma')[0]]
        best = r.panel_best(tree, cell, arms)
        panel = {}
        for row, (arm, trials, solved) in arms.items():
            path = r.RAW / f'{name}__{row}.json'
            if not path.exists():
                print(f'SKIP {label} {row}: no {path.name}')
                continue
            raw = json.loads(path.read_text())
            by_trial = {}
            for x in raw:
                assert (PROJECT / tree / 'continual' / arm / cell / x['trial']).resolve() == \
                    (REPO / x['run']).resolve(), f'{path.name}: {x["run"]} is not the figure\'s run'
                by_trial.setdefault(x['trial'], {})[x['phase']] = x
            curves = {'solved': [], 'good': [], 'ratio': []}
            radii = None
            n_ckpt = 0
            for t in trials:
                mask = np.asarray(solved[t], bool)
                T = len(mask)
                probed = by_trial.get(t, {})
                acc = {k: [] for k in curves}
                for p in range(T - 1 - r.WINDOW, T - 1):
                    if p not in probed or not mask[p]:
                        continue
                    n_ckpt += 1
                    loss = r.checkpoint_losses(probed[p], floor, best)
                    assert radii in (None, loss['radii']), f'{path.name}: rows with different radii'
                    radii = loss['radii']
                    acc['solved'].append(loss['solved'])
                    acc['good'].append(loss['good'])
                    acc['ratio'].append(ratio_curve(probed[p], floor, best))
                for k in curves:
                    curves[k].append(np.mean(acc[k], axis=0).tolist() if acc[k] else None)
            panel[row] = {'arm': arm, 'trials': list(trials), 'radii': radii,
                          'checkpoints': n_ckpt, 'curves': curves}
        out['panels'][label] = panel
        print(f'{label:34s} ' + ' '.join(f'{m}={v["checkpoints"]}' for m, v in panel.items()))
    DATA.parent.mkdir(parents=True, exist_ok=True)
    DATA.write_text(json.dumps(out, indent=1) + '\n')
    print(f'wrote {DATA}')


def widths(blob, kind, level):
    """({panel: {method: per-trial widths}}, censored count) under one definition."""
    out, cens = {}, 0
    for label, panel in blob['panels'].items():
        for m, v in panel.items():
            vals = []
            for c in v['curves'][kind]:
                if c is None:
                    continue
                w, flag = width_of(c, v['radii'], kind, level)
                cens += flag is not None
                vals.append(w)
            if vals:
                out.setdefault(label, {})[m] = np.array(vals, float)
    return out, cens


def summarise(blob):
    """[(kind, level, {method: (median log2 ratio, Wilcoxon p, n)}, rho vs main, censored)]."""
    main_R = r.ratios(widths(blob, *MAIN_DEF)[0])
    rows = []
    for kind, level in DEFS:
        data, cens = widths(blob, kind, level)
        R = r.ratios(data)
        per_method = {m: b.across([v[0] for v in R[m].values()]) for m in b.TABLE_ORDER}
        pairs = [(R[m][p][0], main_R[m][p][0]) for m in b.TABLE_ORDER for p in R[m] if p in main_R[m]]
        rho = spearmanr(*zip(*pairs)).correlation if len(pairs) > 2 else np.nan
        rows.append((kind, level, per_method, float(rho), cens, len(pairs)))
    return rows


def write_table(rows):
    cols = b.TABLE_ORDER
    lines = [r'% Built by scripts/analysis/plot_basin_width_levels.py -- rerun it, do not edit.',
             r'\begin{tabular}{lr' + 'r' * len(cols) + 'r}', r'\toprule',
             'Curve & Level & ' + ' & '.join(LABEL[m] for m in cols) + r' & $\rho$ \\', r'\midrule']
    last = None
    for kind, level, per_method, rho, cens, n in rows:
        first = CURVE_LABEL[kind] if kind != last else ''
        if kind != last and last is not None:
            lines.append(r'\midrule')
        last = kind
        cells = []
        for m in cols:
            med, pw, _ = per_method[m]
            mark = b.stars(pw) if np.isfinite(pw) else ''
            cells.append(f'{2 ** med:.1f}' + (f'$^{{{mark}}}$' if mark else ''))
        lev = f'{level:g}' + (r' (reported)' if (kind, level) == MAIN_DEF else '')
        rho_s = '--' if (kind, level) == MAIN_DEF else f'{rho:.2f}'
        lines.append(f'{first} & {lev} & ' + ' & '.join(cells) + f' & {rho_s} \\\\')
    lines += [r'\bottomrule', r'\end{tabular}']
    TABLE.with_suffix('.tex').write_text('\n'.join(lines) + '\n')
    print(f'wrote {TABLE}.tex')


def write_markdown(blob, rows):
    md = [f'# {TABLE.name} and {CURVES.name}', '',
          'See the docstring of `scripts/analysis/plot_basin_width_levels.py`. Each row: the '
          'basin width relative to PPO under one definition, median over the 13 ReLU panels '
          '(Wilcoxon over panels: * p<0.05, ** p<0.01, *** p<0.001), the Spearman rho of the 78 per-panel '
          'log ratios with the reported definition, and how many trial widths were censored '
          f'at an end of the ladder. Extracted {blob["extracted"]} from `{blob["raw"]}`.', '',
          '| Curve | Level | ' + ' | '.join(LABEL[m] for m in b.TABLE_ORDER) + ' | rho | censored | n pairs |',
          '|---|---|' + '---|' * (len(b.TABLE_ORDER) + 3)]
    for kind, level, per_method, rho, cens, n in rows:
        cells = [f'{2 ** v[0]:.2f} ({v[1]:.2g})' for v in (per_method[m] for m in b.TABLE_ORDER)]
        md.append(f'| {kind} | {level:g} | ' + ' | '.join(cells) + f' | {rho:.2f} | {cens} | {n} |')
    md += ['', '## Ranking of the methods under each definition (widest first)', '']
    for kind, level, per_method, rho, cens, n in rows:
        order = sorted(b.TABLE_ORDER, key=lambda m: -per_method[m][0])
        md.append(f'- {kind} @ {level:g}: ' + ' > '.join(LABEL[m] for m in order))
    md += ['', '## Survival curve per panel (rule `solved`, share still solving, mean over trials)', '']
    for label, panel in blob['panels'].items():
        radii = next(v['radii'] for v in panel.values())
        md += [f'### {label}', '', '| Method | ' + ' | '.join(f'{e:g}' for e in radii) + ' |',
               '|---|' + '---|' * len(radii)]
        for m in b.METHODS:
            if m in panel:
                c = [x for x in panel[m]['curves']['solved'] if x is not None]
                md.append(f'| {LABEL[m]} | ' + ' | '.join(f'{1 - x:.2f}' for x in np.mean(c, axis=0)) + ' |')
        md.append('')
    TABLE.with_suffix('.md').write_text('\n'.join(md) + '\n')
    print(f'wrote {TABLE}.md')


def draw_curves(blob, fs):
    labels = list(blob['panels'])
    groups = {f'{body}, {change}': g for g, body, change, _, _ in b.panel_list()}
    order = [g for g in ('Noise', 'Physics', 'Action reversal', 'Sequences')]
    rows = [[p for p in labels if groups.get(p) == g] for g in order]
    ncol = max(len(x) for x in rows)
    W = 5.5
    H = 1.1 * len(rows) + 0.2
    fig, axes = plt.subplots(len(rows), ncol, figsize=(W, H), sharey=True)
    axes = np.atleast_2d(axes)
    lowest = {}
    for i, (g, ps) in enumerate(zip(order, rows)):
        for j in range(ncol):
            ax = axes[i, j]
            if j >= len(ps):
                ax.axis('off')
                continue
            lowest[j] = ax
            panel = blob['panels'][ps[j]]
            for m in b.METHODS:
                if m not in panel:
                    continue
                c = [x for x in panel[m]['curves']['solved'] if x is not None]
                if not c:
                    continue
                st = METHOD_STYLE[m]
                ax.plot(panel[m]['radii'], 1 - np.mean(c, axis=0), color=st['color'],
                        lw=1.0, ls=st.get('ls', '-'), zorder=3)
            ax.axhline(1 - MAIN_DEF[1], color='0.5', lw=0.5, ls=':', zorder=1)
            ax.set_xscale('log')
            ax.set_xlim(blob['ladder'][0] * 0.8, blob['ladder'][-1] * 1.25)
            ax.set_ylim(-0.03, 1.03)
            ax.set_xticks([0.001, 0.01, 0.1, 1])
            ax.set_xticklabels([])
            ax.set_yticks([0, 0.5, 1])
            ax.minorticks_off()
            ax.tick_params(length=2, width=0.5, labelsize=fs - 1)
            ax.spines[['right', 'top']].set_visible(False)
            ax.set_title(ps[j].replace(', ', '\n').replace(' → ', '→'), fontsize=fs - 0.5, pad=2)
            if j == 0:
                ax.set_ylabel('share still\nsolving', fontsize=fs)
    for ax in lowest.values():
        ax.set_xticklabels(['0.001', '0.01', '0.1', '1'])
        ax.set_xlabel('radius (× tensor norm)', fontsize=fs)
    handles = [Line2D([], [], color=METHOD_STYLE[m]['color'], lw=1.0,
                      ls=METHOD_STYLE[m].get('ls', '-')) for m in b.METHOD_ORDER if m in b.METHODS]
    axes[0, -1].legend(handles, [LABEL[m] for m in b.METHOD_ORDER if m in b.METHODS],
                       loc='center', frameon=False, fontsize=fs, handlelength=1.6,
                       labelspacing=0.3)
    fig.subplots_adjust(left=0.09, right=0.99, top=0.93, bottom=0.08, hspace=0.75, wspace=0.12)
    b.save(fig, CURVES)


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
    plt.rcParams.update({'font.size': args.font_size, 'axes.linewidth': 0.5})
    blob = json.loads(DATA.read_text())
    rows = summarise(blob)
    write_table(rows)
    write_markdown(blob, rows)
    draw_curves(blob, args.font_size)
    return 0


if __name__ == '__main__':
    sys.exit(main())
