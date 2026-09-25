"""Basin width of every method against training, on every panel of the
basin-width table (`plot_basin_width_methods`). An appendix paper figure: the
companion of `basin_width_main` that shows WHEN the widths of that figure
arise, since it pools the last five sub-task checkpoints and Figure 1 draws a
single one.

    # re-read the width passes (only when they change)
    .venv/bin/python scripts/analysis/plot_basin_width_evolution.py --extract
    # redraw from visuals/final/data/
    .venv/bin/python scripts/analysis/plot_basin_width_evolution.py [--eps 0.03]

    -> paper/visuals/final/appendix/basin_width_evolution.{pdf,png,md}
       paper/visuals/final/data/basin_width_evolution.json

Last row (since 2026-09-22, when the separate basin_width_lines figure was folded
in here, since it drew the same data): one panel per environment, one line per
method, the median over that environment's settings of the method's width
relative to PPO at every task (log scale, 3-checkpoint running mean; MiniGrid has
one setting, so it has no relative panel). Relative width per setting and
checkpoint = PPO's mean action change over trials / the method's, solved
checkpoints only, and undefined where either mean is below RATIO_FLOOR (0.5% of
probe states: a near-constant policy makes the ratio explode). Grouped by
environment, not by the axis of change, because environment explains most of the
spread between settings and the axis of change almost none (adjusted eta^2 over
the 12 classic-control settings, 2026-09-21: ~0.7 vs <= 0 for the late ratio of
ES, GA and TRAC-PPO). The legend carries basin_width_main's pooled statistic
over all 13 settings: the median late ratio and its Wilcoxon mark.

Panels, runs, arms, trials and the width column are exactly those of
`basin_width_main` (`plot_basin_width_methods.panel_trials`): the 13 panels of
`generalist_scores_centroid` and the five Figure 2 sequences; relative noise
(0.1 times each tensor's norm) on the ReLU networks of gymnax and MiniGrid,
absolute noise (s.d. 0.1 on every weight) on the tanh networks of HalfCheetah
and Kinetix. The FIGURE draws the 13 ReLU panels only, as basin_width_main and
the appendix table do (2026-09-21): on a tanh network neither kind of noise is
free of the weight scale (paper app:basin_noise_scale); the tanh panels stay in
the data file and the markdown, marked with a dagger. Each panel draws, per method, the action
change under weight noise of the centroid saved at the end of each sub-task
(mean over trials, 95% bootstrap band): the raw width measure, on an INVERTED
axis (0 at the top) so that up is a wider basin, as in basin_width_main; no
ratio to PPO is taken. The markdown gives, per panel and
method, the mean over the first three and the last five checkpoints and the
Spearman correlation of the width with the checkpoint index over all
(trial, checkpoint) pairs: positive = the basin narrows over training.

Only checkpoints that solve the sub-task they were trained on are drawn, the
rule of `basin_width_main` (its docstring): a failed policy sits on a plateau
of failure, not in a basin of a solution. The markdown counts the exact zeros
among the checkpoints that remain.
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
from matplotlib.lines import Line2D                        # noqa: E402
from matplotlib.ticker import FuncFormatter, MaxNLocator   # noqa: E402
from scipy.stats import spearmanr                          # noqa: E402

import plot_basin_width_methods as b                       # noqa: E402
from make_lineplot import METHOD_STYLE                     # noqa: E402
from source.metrics.continual_metrics import bootstrap_ci  # noqa: E402

pgs = b.pgs
OUT = pgs.FINAL / 'appendix' / 'basin_width_evolution'
DATA = pgs.FINAL / 'data' / 'basin_width_evolution.json'
METHODS = b.METHODS
RADII = b.RADII
INK = b.INK
PRINT_W = 5.5                       # \linewidth
COLS = 4                            # panels per row: bodies, in the table's order
BODIES = ['CartPole', 'Acrobot', 'MountainCar', 'MiniGrid']


def extract():
    """Per panel and method, the per-trial width at every checkpoint and radius."""
    out = {'panels': {}, 'passes': {}, 'absolute': [], 'groups': {},
           'extracted': datetime.date.today().isoformat()}
    for label, tree, cell, key, arms, rel, _ in b.panel_trials():
        out['passes'][tree] = rel
        out['panels'][label] = {row: {'arm': arm, **{
            f'width_{e}': [rec[key.format(e)] for rec in trials.values()] for e in RADII},
            'solved': None if solved is None else [solved[t] for t in trials]}
            for row, (arm, trials, solved) in arms.items()}
        if b.is_absolute(tree):
            out['absolute'].append(label)
        print(f'{label:34s} {rel:62s} '
              + ' '.join(f'{r}={len(v["width_0.1"])}x{len(v["width_0.1"][0])}'
                         for r, v in out['panels'][label].items()))
    out['groups'] = {f'{body}, {change}': group for group, body, change, _, _ in b.panel_list()}
    DATA.parent.mkdir(parents=True, exist_ok=True)
    DATA.write_text(json.dumps(out, indent=1) + '\n')
    print(f'wrote {DATA}')


def curves(blob, eps):
    """{panel: {method: (n_trials, T) array}}, T the shortest trial."""
    out = {}
    for label, panel in blob['panels'].items():
        for m, v in panel.items():
            ok = v.get('solved') or [None] * len(v[f'width_{eps}'])
            rows = [np.where(np.asarray(s, bool), np.asarray(r, dtype=float), np.nan)
                    if s is not None else np.asarray(r, dtype=float)
                    for r, s in zip(v[f'width_{eps}'], ok) if r]
            if not rows:
                continue
            T = min(len(r) for r in rows)
            out.setdefault(label, {})[m] = np.stack([r[:T] for r in rows])
    return out


def draw_panel(ax, series, fs, invert=True, ylim=None):
    drawn = set()
    for m in METHODS:
        arr = series.get(m)
        if arr is None:
            continue
        T = arr.shape[1]
        mean, lo, hi = np.full(T, np.nan), np.full(T, np.nan), np.full(T, np.nan)
        for t in range(T):
            col = arr[:, t][np.isfinite(arr[:, t])]
            if col.size:
                mean[t], lo[t], hi[t] = bootstrap_ci(col)
        colour = METHOD_STYLE[m]['color']
        x = np.arange(1, T + 1)
        ax.plot(x, mean, color=colour, lw=0.9, zorder=3)
        ax.fill_between(x, lo, hi, color=colour, alpha=0.18, lw=0, zorder=2)
        drawn.add(m)
    ax.grid(True, color='0.92', lw=0.5, zorder=0)
    ax.spines[['top', 'right']].set_visible(False)
    ax.tick_params(length=2, pad=1.5, labelsize=fs - 1.5)
    ax.set_xticks([1, 10, 20])
    ax.set_xlim(0.5, 20.5)
    ax.set_ylim(*(ylim or (0, None)))
    if invert:
        ax.invert_yaxis()            # up = a wider basin, as basin_width_main reads
    ax.yaxis.set_major_locator(MaxNLocator(nbins=3))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:g}'))
    return drawn


def draw(blob, eps, fs, path, series=None, ylabel='Action change under weight noise (up = wider basin)',
         invert=True, ylim=None, relative_row=True, legend_stats=None):
    """`series` ({panel: {method: (trials, T)}}) replaces the width curves, for
    another per-checkpoint measure on the same grid (plot_shared_basin_evolution);
    `legend_stats` ({method: text}) then replaces the pooled width ratios."""
    data = curves(blob, eps) if series is None else series
    groups = blob['groups']
    absolute = set(blob['absolute'])
    groups = {p: g for p, g in groups.items() if p not in absolute}   # ReLU panels only
    order = list(dict.fromkeys(groups.values()))            # Noise, Physics, ..., Sequences
    rows = [[p for p in groups if groups[p] == g] for g in order]
    left, right, top, bottom, gap_x, gap_y = 0.6, 0.04, 0.3, 0.3, 0.22, 0.3
    W = PRINT_W
    cell_w = (W - left - right - gap_x * (COLS - 1)) / COLS
    cell_h, rel_h, rel_gap = 0.72, 0.9, 0.42     # the last row: width relative to PPO
    if not relative_row:
        rel_gap = rel_h = 0
    H = top + bottom + len(rows) * cell_h + (len(rows) - 1) * gap_y + rel_gap + rel_h
    fig = plt.figure(figsize=(W, H))
    drawn = set()
    x_of = lambda c: (left + c * (cell_w + gap_x)) / W     # noqa: E731
    for c, body in enumerate(BODIES[:COLS - 1]):            # column headers: the body
        fig.text(x_of(c) + cell_w / 2 / W, 1 - 0.1 / H, body, ha='center', va='center',
                 fontsize=fs)
    for r, (g, panels) in enumerate(zip(order, rows)):
        y0 = H - top - (r + 1) * cell_h - r * gap_y
        for label in panels:
            body, change = label.split(', ', 1)
            c = BODIES.index(body)
            ax = fig.add_axes([x_of(c), y0 / H, cell_w / W, cell_h / H])
            drawn |= draw_panel(ax, data.get(label, {}), fs, invert, ylim)
            title = label if c == COLS - 1 else change
            ax.set_title(title.replace(', room ', ' ').replace(' → ', '→'), fontsize=fs - 1, pad=2)
        fig.text(0.1 / W, (y0 + cell_h / 2) / H, g, rotation=90, ha='center', va='center',
                 fontsize=fs)
    y_abs = H - top - len(rows) * cell_h - (len(rows) - 1) * gap_y   # bottom of the raw rows
    fig.text(0.33 / W, (y_abs + (H - top - y_abs) / 2) / H, ylabel,
             rotation=90, ha='center', va='center', fontsize=fs)
    # last row: the same data relative to PPO, the median over an environment's panels
    if relative_row:
        rel = relative(blob, eps)
        for c, env in enumerate(REL_ENVS):
            ax = fig.add_axes([x_of(c), bottom / H, cell_w / W, rel_h / H])
            draw_relative(ax, rel.get(env, {}), fs, first=c == 0)
        fig.text(0.1 / W, (bottom + rel_h / 2) / H, 'Width relative to PPO', rotation=90,
                 ha='center', va='center', fontsize=fs)
    # legend in the spare slot of the first row (the last column); the number after
    # each method is its median width relative to PPO over the 13 panels
    pooled = {} if legend_stats is not None else \
        b.ratios({p: v for p, v in b.widths(eps).items() if p not in absolute})
    handles = []
    for m in METHODS:
        if m not in drawn:
            continue
        name = 'ES' if m == 'es' else METHOD_STYLE[m]['label']
        if m in pooled:
            med, pw, _ = b.across([v[0] for v in pooled[m].values()])
            name += f' \u00d7{2 ** med:.2g}' + (b.stars(pw) if np.isfinite(pw) else '')
        elif legend_stats:
            name += legend_stats.get(m, '')
        handles.append(Line2D([], [], color=METHOD_STYLE[m]['color'], lw=1.2, label=name))
    y0 = H - top - cell_h
    fig.legend(handles=handles, loc='center',
               bbox_to_anchor=(x_of(COLS - 1) + cell_w / 2 / W, (y0 + cell_h / 2) / H),
               frameon=False, fontsize=fs - 0.5, handlelength=1.4, labelspacing=0.35)
    b.save(fig, path)


RATIO_FLOOR = 0.005
# The relative row leaves MiniGrid out: one setting, and every method's action
# change is near zero early on there, so its ratio swings by 4x between checkpoints.
REL_ENVS = ['CartPole', 'Acrobot', 'MountainCar']
from make_lineplot import METHOD_ORDER                     # noqa: E402
REL_ROWS = [m for m in METHOD_ORDER if m in ('es', 'ga', 'pbt', 'trac', 'cchain', 'redo')]


def relative(blob, eps):
    """{env: {method: (n_settings, T) log2 width relative to PPO}}, ReLU panels."""
    data = curves(blob, eps)
    out = {}
    for label, series in data.items():
        if label in blob['absolute'] or 'ppo' not in series:
            continue
        q = np.nanmean(series['ppo'], 0)
        for m in REL_ROWS:
            if m not in series:
                continue
            a = np.nanmean(series[m], 0)
            T = min(len(a), len(q))
            with np.errstate(invalid='ignore', divide='ignore'):
                r = np.log2(q[:T] / a[:T])
            r[~((a[:T] >= RATIO_FLOOR) & (q[:T] >= RATIO_FLOOR))] = np.nan
            out.setdefault(label.split(',')[0], {}).setdefault(m, []).append(r)
    return {e: {m: np.array(v) for m, v in by.items()} for e, by in out.items()}


def draw_relative(ax, rel, fs, first):
    """One environment: per method, the median over its panels of the width relative
    to PPO at every task, as a running mean over three tasks (log scale)."""
    x = np.arange(1, 21)
    for m in sorted(REL_ROWS, key=lambda m: m == 'es'):     # ES drawn last, on top
        arr = rel.get(m)
        if arr is None:
            continue
        with np.errstate(all='ignore'):
            med = np.nanmedian(arr, 0)
        sm = np.array([np.nanmean(med[max(0, i - 1):i + 2]) if np.isfinite(med[max(0, i - 1):i + 2]).any()
                       else np.nan for i in range(len(med))])
        ax.plot(x[:len(sm)], 2 ** sm, color=METHOD_STYLE[m]['color'], lw=0.9, zorder=3)
    ax.axhline(1, color='0.35', lw=0.7, zorder=2)
    ax.set_yscale('log', base=2)
    ax.set_ylim(0.5, 24)
    ax.set_yticks([0.5, 1, 2, 4, 8, 16])
    ax.set_yticklabels(['0.5', '1', '2', '4', '8', '16'])
    ax.minorticks_off()
    ax.set_xlim(0.5, 20.5)
    ax.set_xticks([1, 10, 20])
    ax.set_xlabel('Task', fontsize=fs - 1, labelpad=1)
    ax.tick_params(length=2, pad=1.5, labelsize=fs - 1.5)
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(True, color='0.92', lw=0.5, zorder=0)
    if first:
        ax.text(1.4, 0.94, 'PPO', ha='left', va='top', fontsize=fs - 1.5, color='0.35')


def write_markdown(blob, eps, path):
    data = curves(blob, eps)
    absolute = set(blob['absolute'])
    lines = [f'# basin_width_evolution (eps {eps:g})', '',
             'Action change under weight noise per sub-task checkpoint, mean over trials '
             '(lower = wider basin). Per method: first three checkpoints, last five, '
             'Spearman rho of width vs checkpoint index over (trial, checkpoint) pairs '
             '(+ = narrows over training), and exact zeros / values (a constant policy). '
             '† absolute noise (tanh network). Data: '
             f'`{DATA.relative_to(pgs.FINAL.parent.parent)}`, extracted {blob["extracted"]}.', '']
    for label, series in data.items():
        lines += [f'## {label}' + (' †' if label in absolute else ''), '',
                  '| Method | n | first 3 | last 5 | rho | zeros |', '|---|---|---|---|---|---|']
        for m in METHODS:
            arr = series.get(m)
            if arr is None:
                continue
            T = arr.shape[1]
            idx = np.tile(np.arange(T), arr.shape[0])
            ok = np.isfinite(arr.ravel())
            rho = spearmanr(idx[ok], arr.ravel()[ok])[0] if ok.sum() > 2 else np.nan
            lines.append(f'| {"ES" if m == "es" else METHOD_STYLE[m]["label"]} | {arr.shape[0]} | '
                         f'{np.nanmean(arr[:, :3]):.3f} | {np.nanmean(arr[:, -5:]):.3f} | '
                         f'{rho:+.2f} | {int((arr == 0).sum())}/{int(ok.sum())} |')
        lines.append('')
    # the trend of every method pooled over panels: median rho, panels narrowing
    lines += ['## Across panels', '', '| Method | median rho | panels rho > 0 | panels |',
              '|---|---|---|---|']
    for m in METHODS:
        rhos = []
        for label, series in data.items():
            arr = series.get(m)
            if arr is None:
                continue
            idx = np.tile(np.arange(arr.shape[1]), arr.shape[0])
            ok = np.isfinite(arr.ravel())
            rhos.append(spearmanr(idx[ok], arr.ravel()[ok])[0])
        rhos = np.array(rhos)
        lines.append(f'| {"ES" if m == "es" else METHOD_STYLE[m]["label"]} | '
                     f'{np.median(rhos):+.2f} | {(rhos > 0).sum()} | {rhos.size} |')
    path.with_suffix('.md').write_text('\n'.join(lines) + '\n')
    print(f'wrote {path}.md')


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--extract', action='store_true',
                    help='read the width passes and write the data file first')
    ap.add_argument('--eps', type=float, default=0.1, choices=RADII)
    ap.add_argument('--font-size', type=float, default=7.0, help='the size in print')
    args = ap.parse_args()
    if args.extract:
        extract()
    if not DATA.exists():
        sys.exit(f'no {DATA}: run with --extract first')
    fs = args.font_size
    plt.rcParams.update({
        'font.size': fs, 'axes.titlesize': fs, 'xtick.labelsize': fs - 1,
        'ytick.labelsize': fs - 1, 'axes.linewidth': 0.5, 'xtick.major.width': 0.5,
        'ytick.major.width': 0.5, 'xtick.major.size': 2, 'ytick.major.size': 2,
    })
    METHOD_STYLE['pbt'] = {**METHOD_STYLE['pbt'], 'label': 'PBT-PPO'}
    blob = json.loads(DATA.read_text())
    sfx = '' if args.eps == 0.1 else f'_eps{args.eps:g}'
    path = OUT.with_name(OUT.name + sfx)
    draw(blob, args.eps, fs, path)
    write_markdown(blob, args.eps, path)
    return 0


if __name__ == '__main__':
    sys.exit(main())
