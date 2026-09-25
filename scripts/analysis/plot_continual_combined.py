#!/usr/bin/env python3
"""continual_main and stability_plasticity as one figure: each task's centroid
curve with its learning accuracy vs forgetting panel directly below it.

Four rows of five: noise curves, noise LA vs F, action-reversal curves,
action-reversal LA vs F (MiniGrid and Kinetix in the last column, as in both
source figures). One legend, methods only: the dominated region, the equal
LA - F lines and the ring are explained in the caption.

Draws from the two figures' saved data only; refresh those first with
    plot_continual_lineplots.py --extract
    plot_stability_plasticity.py --extract

    .venv/bin/python scripts/analysis/plot_continual_combined.py
-> paper/visuals/final/continual_combined.{pdf,png}

--part draws one half alone, in the same look:
    --part tradeoff -> paper/visuals/final/continual_tradeoff.{pdf,png,md}   (Figure 2)
    --part curves   -> paper/visuals/final/appendix/continual_curves.{pdf,png}

Figure 2 (`tradeoff`, since 2026-09-24) is wider than the curves' grid: it
holds every continual setting of the paper on the LA vs F plane, a row a kind
of change, in TRADEOFF_ROWS -- the multiple-task row (noise over ten tasks,
Kinetix's twenty levels) over the two-task rows (noise, physics with MiniGrid,
action reversal), so the generalist figure's settings and Figure 2's are one
figure (the generalist performance profiles stayed in the appendix figure
fig:generalist_panels; Figure 2 dropped its profiles row on 2026-09-25).
Each panel's title names its environment and the change; the two groups are labelled and bracketed at the left and separated by a rule. `--rows` picks a subset of the rows
(`--rows ten noise actions` leaves physics to the appendix). The two-task
noise and physics panels come from plot_stability_plasticity.py's `tradeoff`
panels (pcl.PANELS), so refresh with
    plot_stability_plasticity.py --extract
"""

import argparse
import json
import pathlib
import sys

import matplotlib
matplotlib.use('Agg')
import numpy as np                                         # noqa: E402
import matplotlib.pyplot as plt                            # noqa: E402
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec  # noqa: E402
from matplotlib.lines import Line2D                        # noqa: E402

import plot_stability_plasticity as sps                    # noqa: E402

pcl, ncs, lp, es_arm = sps.pcl, sps.ncs, sps.lp, sps.es_arm
STEMS = {'both': sps.FINAL / 'continual_combined',
         'tradeoff': sps.FINAL / 'continual_tradeoff',
         'curves': sps.FINAL / 'appendix' / 'continual_curves'}
NCOLS = sps.NCOLS
PANEL_W, CURVE_H, SCATTER_H = 2.4, 1.85, 1.85       # inches
PAIR_GAP, BLOCK_GAP = 0.95, 1.75                     # inside a curve/scatter pair, between the two
# Figure 2: (row key, group, [(tree, cell) or None] a column). Consecutive
# rows of one group share a rotated label and a bracket at the far left, and a
# rule separates the groups: the sequence of many tasks, each seen once, over
# the settings that alternate two. Every panel's title names its environment
# and the kind of change, so the fifth column can hold Kinetix (many levels)
# beside the multiple-task row and MiniGrid (a room change) beside physics.
TRADEOFF_ROWS = [
    ('ten', 'Multiple tasks', [
        ('paper/gymnax/data/noise_10task', 'CartPole_v1_sigma1.0'),
        ('paper/gymnax/data/noise_10task', 'Acrobot_v1_sigma1.0'),
        ('paper/gymnax/data/noise_10task', 'MountainCar_v0_sigma0.1'),
        ('paper/mjx/cheetah/data/noise_10task', 'cheetah_noise'),
        ('paper/kinetix/data', 'Kinetix20')]),
    ('noise', 'Two tasks', [
        ('paper/gymnax/data/noise_2task', 'CartPole_v1_sigma0.5'),
        ('paper/gymnax/data/noise_2task', 'Acrobot_v1_sigma0.5'),
        ('paper/gymnax/data/noise_2task', 'MountainCar_v0_sigma0.05'),
        ('paper/mjx/cheetah/data/noise_2task', 'cheetah_noise'),
        None]),
    ('physics', 'Two tasks', [
        ('paper/gymnax/data/physics_2task', 'CartPole_v1_sigma1.0'),
        ('paper/gymnax/data/physics_2task', 'Acrobot_v1_sigma1.0'),
        ('paper/gymnax/data/physics_2task', 'MountainCar_v0_sigma1.0'),
        ('paper/mjx/cheetah/data/physics_2task', 'cheetah_friction'),
        ('paper/minigrid/data', 'MiniGrid_8x8_16x16')]),
    ('actions', 'Two tasks', [
        ('paper/gymnax/data/actions_2task', 'CartPole_v1_sigma1.0'),
        ('paper/gymnax/data/actions_2task', 'Acrobot_v1_sigma1.0'),
        ('paper/gymnax/data/actions_2task', 'MountainCar_v0_sigma1.0'),
        ('paper/mjx/cheetah/data/actions_2task', 'cheetah_action'),
        None]),
]
TRADEOFF_H, ROW_GAP, GROUP_GAP = 1.75, 1.05, 0.5     # inches; GROUP_GAP is added between groups
GROUP_RULE = '0.3'


def panel_title(title):
    """The panel's two-line title from its pcl.PANELS title: environment over
    the kind of change (`CartPole, noise, two tasks` -> `CartPole\nnoise`)."""
    title = title.replace(', two tasks', '').replace('MiniGrid ', 'MiniGrid, ')
    return title.replace(', ', '\n', 1).replace(', ', ' ')


def _key(m):
    """ES and NES are one method (`es`), PBT N=8 and N=2 one (`pbt`), whichever
    a family kept."""
    return 'es' if m in es_arm.ARMS else 'pbt' if m in es_arm.PBT_ARMS else m


def _legend(fig, methods, height):
    legend = list(dict.fromkeys(_key(m) for m in methods))
    handles = [Line2D([], [], color=lp.METHOD_STYLE[m]['color'], lw=1.6,
                      marker='o' if lp.FAMILY.get(m) == 'ne' else 's', ms=5,
                      mec='white', mew=0.5) for m in legend]
    fig.legend(handles, [lp.METHOD_STYLE[m]['label'] for m in legend], loc='upper center',
               ncol=len(handles), frameon=False, bbox_to_anchor=(0.5, 1 - 0.02 / height),
               handlelength=2.2, handletextpad=0.4, columnspacing=1.3)


def _save(fig, stem, width, height):
    stem.parent.mkdir(parents=True, exist_ok=True)
    for ext in ('pdf', 'png'):
        fig.savefig(f'{stem}.{ext}', dpi=300)
    plt.close(fig)
    print(f'wrote {stem}.pdf, {stem}.png  ({width:.1f} x {height:.1f} in)')


def draw_tradeoff(meta, row_keys, stem):
    """Figure 2: TRADEOFF_ROWS on the LA vs F plane, every panel scaled and
    ringed as plot_stability_plasticity.py does it, plus its markdown."""
    rows = [r for r in TRADEOFF_ROWS if r[0] in row_keys]
    all_panels, scales = sps.main_panels(meta, figs=sps.FIGS)
    by_key = {pcl._key(t, c): (title, points) for (title, points), (_, t, c, _) in zip(
        all_panels, [p for p in pcl.PANELS if p[0] in sps.FIGS])}
    # Groups of consecutive rows; a wider gap where the group changes.
    groups = []
    for r, (_, group, _) in enumerate(rows):
        if groups and groups[-1][0] == group:
            groups[-1][1].append(r)
        else:
            groups.append((group, [r]))
    left, right, top, bottom, wgap = 1.75, 0.65, 1.4, 0.85, 0.7
    width = left + right + NCOLS * PANEL_W + (NCOLS - 1) * wgap
    height = (top + bottom + len(rows) * TRADEOFF_H + (len(rows) - 1) * ROW_GAP
              + (len(groups) - 1) * GROUP_GAP)
    fig = plt.figure(figsize=(width, height))
    # One GridSpec a group, stacked by hand so the group gap is exact.
    y = 1 - top / height
    axes, used, drawn = {}, set(), []
    for g, (group, members) in enumerate(groups):
        n = len(members)
        block_h = n * TRADEOFF_H + (n - 1) * ROW_GAP
        gs = GridSpec(n, NCOLS, figure=fig, left=left / width, right=1 - right / width,
                      top=y, bottom=y - block_h / height,
                      wspace=wgap / PANEL_W, hspace=ROW_GAP / TRADEOFF_H)
        for i, r in enumerate(members):
            for c, spec in enumerate(rows[r][2]):
                ax = axes[r, c] = fig.add_subplot(gs[i, c])
                if spec is None:
                    ax.set_axis_off()
                    continue
                title, points = by_key[pcl._key(*spec)]
                drawn.append((title, points))
                used |= set(points)
                sps.draw_panel(ax, points, title, 'F')
                ax.set_title(panel_title(title), linespacing=1.1)
                if c == 0:
                    ax.set_ylabel('Forgetting')
        # The group's label and bracket at the far left, over its rows.
        y_top, y_bot = y, y - block_h / height
        x_rule = 0.62 / width
        fig.add_artist(Line2D([x_rule, x_rule], [y_bot, y_top], lw=1.0, color=GROUP_RULE))
        fig.text(0.28 / width, (y_top + y_bot) / 2, group, rotation=90, ha='center',
                 va='center', fontsize=plt.rcParams['font.size'] + 1)
        if g < len(groups) - 1:
            # The rule between the groups: below this group's tick labels
            # (~0.4 in), above the next group's two-line titles.
            y_sep = y_bot - 0.58 / height
            fig.add_artist(Line2D([x_rule, 1 - right / width], [y_sep, y_sep], lw=1.0,
                                  color=GROUP_RULE))
        y = y_bot - (ROW_GAP + GROUP_GAP) / height
    fig.align_ylabels([axes[r, 0] for r in range(len(rows))])
    fig.supxlabel('Learning accuracy', y=0.22 / height)
    _legend(fig, [m for m in lp.METHOD_ORDER if m in used], height)
    # The caption's (a) many tasks, (b) two tasks: a bold letter right of the
    # bracket, its top level with the first title of the group.
    starts = [members[0] for _, members in groups]
    fig.canvas.draw()
    inv = fig.transFigure.inverted()
    for letter, r in zip('abcdefg', sorted(starts)):
        y_top = inv.transform(axes[r, 0].title.get_window_extent())[1, 1]
        fig.text(0.72 / width, y_top, letter, ha='left', va='top', fontweight='bold',
                 fontsize=plt.rcParams['font.size'] + 4)
    _save(fig, stem, width, height)
    sps.write_markdown(drawn, scales, str(stem), 'F')


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--part', choices=list(STEMS), default='both')
    ap.add_argument('--rows', nargs='+', default=[k for k, *_ in TRADEOFF_ROWS],
                    choices=[k for k, *_ in TRADEOFF_ROWS],
                    help='the rows of --part tradeoff (default all)')
    ap.add_argument('--out', help='output stem (default the part\'s under visuals/final)')
    args = ap.parse_args()
    part = args.part
    rows = {'both': ['curve', 'scatter'], 'tradeoff': ['scatter'], 'curves': ['curve']}[part]
    heights = [{'curve': CURVE_H, 'scatter': SCATTER_H}[r] for r in rows]
    # the figure is drawn ~16 in wide and printed at \textwidth (5.5 in, x0.35):
    # these sizes print at ~6.3 pt text and ~5.4 pt ticks, as in the intro schematic
    plt.rcParams.update({'font.size': 19, 'xtick.labelsize': 16, 'ytick.labelsize': 16})
    lp.METHOD_STYLE['pbt'] = {**lp.METHOD_STYLE['pbt'], 'label': 'PBT-PPO'}
    lp.METHOD_STYLE['pbt2'] = {**lp.METHOD_STYLE['pbt'], 'label': 'PBT-PPO'}   # one PBT-PPO, as in continual_main
    if not sps.DATA.exists():
        sys.exit(f'no {sps.DATA}: run plot_stability_plasticity.py --extract first')
    meta = json.loads(sps.DATA.read_text())
    stem = pathlib.Path(args.out) if args.out else STEMS[part]
    if part == 'tradeoff':
        draw_tradeoff(meta, args.rows, stem)
        return 0
    curves = pcl.load_main()
    scatters, _ = sps.main_panels(meta)
    assert [t for _, _, t, _ in curves] == [t for t, _ in scatters], 'panel order differs'

    union = {m for *_, col in curves for m in col['curves']}
    union |= {m for _, points in scatters for m in points}
    methods = [m for m in lp.METHOD_ORDER if m in union]

    nblocks = -(-len(curves) // NCOLS)
    left, right, top, bottom, wgap = 1.05, 0.4, 1.45, 0.75, 0.85
    block_h = sum(heights) + PAIR_GAP * (len(rows) - 1)
    width = left + right + NCOLS * PANEL_W + (NCOLS - 1) * wgap
    height = top + bottom + nblocks * block_h + (nblocks - 1) * BLOCK_GAP
    fig = plt.figure(figsize=(width, height))
    outer = GridSpec(nblocks, 1, figure=fig, left=left / width, right=1 - right / width,
                     top=1 - top / height, bottom=bottom / height,
                     hspace=BLOCK_GAP / block_h)
    for i, ((_tree, _cell, title, col), (_, points)) in enumerate(zip(curves, scatters)):
        block, c = divmod(i, NCOLS)
        inner = GridSpecFromSubplotSpec(len(rows), NCOLS, subplot_spec=outer[block],
                                        height_ratios=heights,
                                        hspace=PAIR_GAP / (sum(heights) / len(rows)),
                                        wspace=wgap / PANEL_W)
        panel_title = title.replace(', ', '\n').replace('MiniGrid ', 'MiniGrid\n')
        if 'curve' in rows:
            ax = fig.add_subplot(inner[rows.index('curve'), c])
            ncs.draw_curves(ax, col, [m for m in methods if m in col['curves']])
            ax.set_title(panel_title, linespacing=1.1)
            ax.set_xlabel('Generations', labelpad=1)
            ax.spines[['top', 'right']].set_visible(False)
            if c == 0:
                ax.set_ylabel('Episodic reward')
        if 'scatter' in rows:
            ax = fig.add_subplot(inner[rows.index('scatter'), c])
            sps.draw_panel(ax, points, title, 'F')
            ax.set_title('' if 'curve' in rows else panel_title, linespacing=1.1)
            ax.set_xlabel('Learning accuracy', labelpad=1)
            if c == 0:
                ax.set_ylabel('Forgetting')
    fig.align_ylabels()
    _legend(fig, methods, height)
    _save(fig, stem, width, height)
    return 0


if __name__ == '__main__':
    sys.exit(main())
