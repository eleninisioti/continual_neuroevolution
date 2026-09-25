#!/usr/bin/env python
"""The novelty comparison as curves: elite and centroid return over the run.

`fig:novelty` in the paper (2026-09-19) and `fig:novelty_all` in the appendix
(`--figure all --diversity`, the four rows of the main figure with PBT-PPO,
since 2026-09-21). One column per cell and two rows per block,
the reported ELITE (best individual, fresh episodes) over the CENTROID (mean of
the population weights). ES, GA and GA + Novelty, mean and 95% bootstrap CI
over seeds, smoothed with the paper lineplots' rolling median (1% of the
records), task switches dashed, the revisit half shaded.

`--figure main`: the three collapse cells -- the tasks on which a switch drops
the GA's whole population into a local optimum: MountainCar under observation
noise (sigma 0.5), MountainCar under action reversal, DeepSea 12 (the
action-map family, gymnax_classic.DeepSeaEnv) -- in one block, legend on top.

`--figure all`: every kept cell, one block a kind of change: observation noise
(CartPole, Acrobot at sigma 1.0, MountainCar at 0.5) over action reversal
(the three bodies) plus DeepSea 12; the noise block's spare slot holds the
legend.

Reads the same `paper/diversity/data/kept_*` links as
plot_diversity_metrics.py --figure kept, so the figures describe the same
runs; that figure (appendix) carries the statistics.

    .venv/bin/python scripts/analysis/plot_novelty_lineplots.py
        -> paper/visuals/final/novelty_lineplots.{pdf,png,md}
    .venv/bin/python scripts/analysis/plot_novelty_lineplots.py --figure all
        -> paper/visuals/final/novelty_lineplots_all.{pdf,png,md}
    .venv/bin/python scripts/analysis/plot_novelty_lineplots.py --diversity
        -> paper/visuals/final/novelty_lineplots_bd.{pdf,png,md}   a third row,
           the population's behavioural diversity (probe disagreement, logged
           every 10 generations and interpolated between the logged points)
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt                            # noqa: E402
import numpy as np                                         # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import plot_noncontinual_solve as ncs                      # noqa: E402
import plot_diversity_plasticity as pdp                    # noqa: E402
lp, PROJECT, REPO = pdp.lp, pdp.PROJECT, pdp.REPO

STEM = 'novelty_lineplots'
FINAL = PROJECT / 'paper/visuals/final'
RUNS = PROJECT / 'paper/diversity/data'
# Cells are (data root, cell, column title); a figure is a list of blocks
# (rows of columns), each block an ELITE row over a CENTROID row.
NOISE = [('kept_noise', 'CartPole_v1_sigma1.0', 'CartPole, noise'),
         ('kept_noise', 'Acrobot_v1_sigma1.0', 'Acrobot, noise'),
         ('kept_noise', 'MountainCar_v0_sigma0.5', 'MountainCar, noise')]
ACTIONS = [('kept_actions', 'CartPole_v1_sigma1.0', 'CartPole, action reversal'),
           ('kept_actions', 'Acrobot_v1_sigma1.0', 'Acrobot, action reversal'),
           ('kept_actions', 'MountainCar_v0_sigma1.0', 'MountainCar, action reversal'),
           ('kept_deepsea', 'DeepSea12_bsuite_sigma1.0', 'DeepSea 12, action map')]
FIGURES = {'main': [[NOISE[2], ACTIONS[2], ACTIONS[3]]],   # the collapse cells
           'all': [NOISE, ACTIONS]}
ROWS = [('elite_eval_fitness', 'Elite return'), ('centroid_fitness', 'Centroid return')]
# The population's diversity (--diversity, the main-text figure's default):
# behavioural = disagreement of the members' greedy actions on the probe
# states (`bd_probe_disagreement` on the gymnax trainers,
# `bd_behavioural_diversity` on the shared runner's PBT), genomic = mean
# pairwise distance between members over sqrt(2P), P the parameter count, as
# in plot_population_diversity.py, on a log axis.
DIVERSITY_ROWS = [('behaviour', 'Behav. diversity'), ('genomic', 'Genomic diversity')]
COLUMN_SOURCES = {'behaviour': ('bd_behavioural_diversity', 'bd_probe_disagreement'),
                  'genomic': ('bd_genomic_diversity',)}
LOG_ROWS = {'genomic'}
ARM_DIR = {'es': 'nes', 'ga': 'ga', 'dns_gaussian': 'dns_gaussian', 'pbt': 'pbt'}   # ES is NES
# The paper's one method order (make_lineplot.METHOD_ORDER, as Figure 2) since
# 2026-09-21: GA, GA + Novelty, ES, PBT-PPO. It used to put ES first.
METHODS = [m for m in lp.METHOD_ORDER if m in ARM_DIR]
# Drawn with GA + Novelty on top so it stays visible; the legend keeps METHODS.
DRAW_ORDER = [m for m in METHODS if m != 'dns_gaussian'] + ['dns_gaussian']
TASK_INTERVAL, PERIOD = 200, 10
# Smoothing window as a fraction of the run. DeepSea's return is 0 or 1, so
# the ten-seed mean jumps between the two within a task and 1% draws
# overlapping zigzags; one task (200 generations) shows each method's level.
SMOOTH = {'DeepSea12_bsuite_sigma1.0': 0.05}
DEFAULT_SMOOTH = 0.01
PLOT_POINTS = 1000
GENS = 2 * PERIOD * TASK_INTERVAL      # 4000 generations; PBT's records (one an
                                       # update, 30000 on MountainCar, 720 on
                                       # DeepSea) are placed on the same axis


def _num_params(trial_dir):
    with np.load(trial_dir / 'checkpoints.npz', allow_pickle=True) as ck:
        key = next(k for k in ('centroid', 'final', 'finalgen') if k in ck.files)
        return int(ck[key].shape[-1])


def curves(root, m, cell, column):
    """(generations, seeds x records) of `column` for one arm of one cell;
    `column` is a record column or a DIVERSITY_ROWS key."""
    out = []
    sources = COLUMN_SOURCES.get(column, (column,))
    for t in sorted((RUNS / root / 'continual' / ARM_DIR[m] / cell).glob('trial_*')):
        f = t / 'training_metrics.json'
        if not f.exists():
            continue
        r = json.loads(f.read_text())
        col = next((c for c in sources if any(x.get(c) is not None for x in r)), None)
        if col is None:
            continue
        row = np.array([np.nan if x.get(col) is None else float(x[col]) for x in r])
        if column == 'genomic':
            row = row / np.sqrt(2 * _num_params(t))
        out.append(row)
    if not out:
        return np.arange(0), np.zeros((0, 0))
    n = min(map(len, out))
    arr = np.array([o[:n] for o in out], float)
    if arr.size and np.isnan(arr).mean() > 0.5:      # logged every PERIOD gens
        for row in arr:
            ok = np.isfinite(row)
            if ok.sum() > 1:
                row[~ok] = np.interp(np.flatnonzero(~ok), np.flatnonzero(ok), row[ok])
    # One record a generation on the gymnax trainers; PBT's records are
    # updates over the same compute, spread over the same generations.
    gens = np.arange(n) if n == GENS else (np.arange(n) + 0.5) / n * GENS
    return gens, arr


def load(cells):
    """`{(row, root, cell): col}` with `col` as plot_continual_lineplots builds
    it; keyed by the root too, since noise and reversal share cell names."""
    panels = {}
    for root, cell, _title in cells:
        for column, _label in ROWS:
            col = {'curves': {}, 'edges': None}
            for m in METHODS:
                gens, arr = curves(root, m, cell, column)
                if arr.size:
                    arr = lp.smooth(arr, max(int(arr.shape[1] * SMOOTH.get(cell, DEFAULT_SMOOTH)) | 1, 1))
                    # At most PLOT_POINTS after smoothing (50 a sub-task, more
                    # than a 1.2 in panel resolves): PBT's 30000 records a
                    # seed made the seven-cell PDF 15 MB.
                    step = max(arr.shape[1] // PLOT_POINTS, 1)
                    col['curves'][m] = (gens[::step], arr[:, ::step])
                    col['edges'] = list(range(TASK_INTERVAL, GENS, TASK_INTERVAL))
            panels[column, root, cell] = col
    return panels


def draw_block(axes, block, panels, last, transpose=False):
    """One block: `axes` (rows x ncols) holds `block`'s cells, ELITE over
    CENTROID; `last` puts the x label on. `transpose`: one row a cell and one
    column a metric instead (`axes` is cells x metrics). Returns the empty axes."""
    spare = []
    for i, (column, label) in enumerate(ROWS):
        for j in range(axes.shape[1] if not transpose else len(block)):
            ax = axes[j, i] if transpose else axes[i, j]
            if j >= len(block):
                ax.set_axis_off()
                spare.append(ax)
                continue
            root, cell, title = block[j]
            col = panels[column, root, cell]
            ncs.draw_curves(ax, col, DRAW_ORDER, xmax=GENS)
            ax.axvspan(GENS * 0.5, GENS, color='0.95', lw=0, zorder=0)   # the revisit half
            if column in LOG_ROWS:
                ax.set_yscale('log')
                # A seed whose population collapses to one genome drags the
                # lower CI towards 0; bound the axis by the mean curves.
                lo = min(np.nanmin(np.nanmean(a, 0)) for _g, a in col['curves'].values())
                ax.set_ylim(bottom=lo / 3)
            ax.spines[['top', 'right']].set_visible(False)
            ax.tick_params(length=2, pad=1.5)
            first_row, last_row, first_col = ((j == 0, j == len(block) - 1, i == 0) if transpose
                                              else (i == 0, i == len(ROWS) - 1, j == 0))
            if first_row:
                ax.set_title(label if transpose else title.replace(', ', ',\n'),
                             linespacing=1.1)
            if last_row:
                if last:
                    ax.set_xlabel('Generations')
            else:
                ax.tick_params(labelbottom=False)
            if first_col:
                ax.set_ylabel(title.replace(', ', ',\n') if transpose else label,
                              linespacing=1.1)
    return spare


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--figure', choices=sorted(FIGURES), default='main')
    ap.add_argument('--font_size', type=float, default=7.0)
    ap.add_argument('--panel', type=float, nargs=2, default=None,
                    help='inches per panel, width height (main 1.55 x 1.15; all 1.2 x 0.95)')
    ap.add_argument('--diversity', action=argparse.BooleanOptionalAction, default=None,
                    help='two more rows, the behavioural and genomic diversity of the '
                         'population (default: on for the main figure, off for all)')
    ap.add_argument('--transpose', action=argparse.BooleanOptionalAction, default=None,
                    help='one row a cell and one column a metric, so the figure spans the '
                         'text width with the caption below (default: on for the main figure)')
    args = ap.parse_args()
    if args.transpose is None:
        args.transpose = args.figure == 'main'
    if args.transpose and len(FIGURES[args.figure]) > 1:
        ap.error('--transpose needs a one-block figure')
    if args.diversity is None:
        args.diversity = args.figure == 'main'
    if args.diversity:
        ROWS.extend(DIVERSITY_ROWS)
    lp.METHOD_STYLE['pbt'] = {**lp.METHOD_STYLE['pbt'], 'label': 'PBT-PPO'}
    os.chdir(REPO)
    plt.rcParams.update({'font.size': args.font_size})
    blocks = FIGURES[args.figure]
    cells = [c for block in blocks for c in block]
    panels = load(cells)
    pw, ph = args.panel or ((1.55, 1.15 if len(ROWS) == 2 else 0.95)
                            if len(blocks) == 1 else (1.2, 0.95))
    ncols = max(map(len, blocks))
    width = pw * ncols + 0.55
    handles = [plt.Line2D([], [], color=lp.METHOD_STYLE[m]['color'], lw=1.8) for m in METHODS]
    labels = [lp.METHOD_STYLE[m]['label'] for m in METHODS]
    if len(blocks) == 1:
        # One block: the legend on top, tight_layout (the main-text figure).
        nrows = ncols if args.transpose else len(ROWS)
        if args.transpose:
            pw, ph = args.panel or (1.25, 0.9)
            ncols = len(ROWS)
            width = pw * ncols + 0.55
        height = ph * nrows + 0.75
        fig, axes = plt.subplots(nrows, ncols, figsize=(width, height), squeeze=False)
        draw_block(axes, blocks[0], panels, True, args.transpose)
        fig.legend(handles, labels, loc='upper center', ncol=len(METHODS), frameon=False,
                   bbox_to_anchor=(0.5, 1.0), columnspacing=1.4, handlelength=1.6)
        fig.tight_layout(rect=(0, 0, 1, 0.94))
    else:
        # Blocks as subfigures under constrained layout (plot_noncontinual_solve.draw);
        # the legend sits in a block's spare slot, as an axes legend so the
        # layout does not reserve room for it.
        height = ph * len(ROWS) * len(blocks) + 0.35 * len(blocks) + 0.3
        fig = plt.figure(figsize=(width, height), layout='constrained')
        fig.get_layout_engine().set(h_pad=0.02, hspace=0.02)
        subs = fig.subfigures(len(blocks), 1, squeeze=False, hspace=0.04)[:, 0]
        spare = []
        for b, (sub, block) in enumerate(zip(subs, blocks)):
            axes = sub.subplots(len(ROWS), ncols, squeeze=False)
            spare += draw_block(axes, block, panels, b == len(blocks) - 1)
        # Centred on the spare slot's rows: the top spare axes span y in [0, 1]
        # of their own frame, the one below reaches about -1.1.
        leg = spare[0].legend(handles, labels, loc='center', bbox_to_anchor=(0.5, -0.05),
                              bbox_transform=spare[0].transAxes, ncol=1, frameon=False,
                              handlelength=1.8, labelspacing=1.0)
        leg.set_in_layout(False)
    FINAL.mkdir(parents=True, exist_ok=True)
    stem = FINAL / (STEM + ('' if args.figure == 'main' else f'_{args.figure}')
                    + ('_bd' if args.diversity and args.figure != 'main' else ''))
    for ext in ('pdf', 'png'):
        fig.savefig(f'{stem}.{ext}', dpi=300)
    plt.close(fig)
    print(f'wrote {stem}.pdf, {stem}.png  ({width:.1f} x {height:.1f} in)')
    lines = [f'# {stem.name}', '', 'Built by `scripts/analysis/plot_novelty_lineplots.py '
             f'--figure {args.figure}` from the '
             '`paper/diversity/data/kept_*` links (the runs of `diversity_metrics_kept_elite`). '
             'Mean over seeds of the run-average return, per column and row; the statistics '
             'are in `diversity_metrics_kept_elite_paper.md`.', '',
             '| Row | Task | Method | n | run mean |', '|---|---|---|---|---|']
    for column, label in ROWS:
        for root, cell, title in cells:
            for m, (_g, arr) in panels[column, root, cell]['curves'].items():
                lines.append(f'| {label} | {title} | {lp.METHOD_STYLE[m]["label"]} | {arr.shape[0]} | '
                             f'{np.nanmean(arr):.3g} |')
    pathlib.Path(f'{stem}.md').write_text('\n'.join(lines) + '\n')
    return 0


if __name__ == '__main__':
    sys.exit(main())
