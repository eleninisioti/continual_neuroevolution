"""Generalist, switching specialist, stuck or neither: one figure over every
two-sub-task family, read from the `generalist_checkpoints_<agent>.json` in
each paper directory.

    .venv/bin/python scripts/analysis/plot_generalist_outcomes.py
    .venv/bin/python scripts/analysis/plot_generalist_outcomes.py --agent centroid

    -> paper/visuals/generalist_outcomes_{centroid,elite}.{pdf,png,md}

Rows are what the switch changes, columns the body, each panel titled
"Body, change" like continual_main. Each bar is one method: the post-switch checkpoints of all
its trials pooled and split by outcome, as defined in
`generalist_checkpoints.py` (the agent saved at the end of a phase scored on
that phase's sub-task and on the previous one). The number to the right of
the bar is retention, the markdown table's column: per trial that ever held a
generalist, the fraction of checkpoints from the first one on that still are,
averaged over those trials only; (k) when only k of the trials ever held one,
a dash when none did. The method with the longest generalist segment in a
panel is outlined (every one, on a tie; none when no method has a generalist
checkpoint): unlike retention, it counts the trials that never found one.

Arms are the other paper figures': one of ES/NES (drawn as ES) and one of PBT
N=8/N=2 a paper directory (`plot_metrics_overview.keep_one_arm`), no novelty
arm.

The cheetah tables are not written by `finish_iclr.sh`, which has no solved
threshold for the MJX bodies. They were classified at 2000, about three times
the 677 a cheetah that does not move scores, with
`generalist_checkpoints.py --threshold 2000` on runs_mjx_noise025,
runs_mjx_2task (friction) and runs_mjx_action. A panel whose JSON is missing
is drawn empty with the reason.

MiniGrid keeps only the checkpoints that end an 8x8 phase (ONE_DIRECTION).
Its rooms are nested: an agent that crosses the 16x16 room solves the 8x8 one,
so every checkpoint ending a 16x16 phase is a generalist for every method
(GA/ES 1.00, RL 0.73-0.98) and pooling the two directions caps the switching
fraction at one half. The kept direction asks whether the 8x8 phase erased
the 16x16 room, which is where the methods differ. Retention is over the same
checkpoints.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt                            # noqa: E402
from matplotlib.gridspec import GridSpec                    # noqa: E402
from matplotlib.patches import Patch                        # noqa: E402
from matplotlib.transforms import blended_transform_factory  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO / 'scripts'))
sys.path.insert(0, str(REPO / 'scripts/analysis'))
from make_lineplot import METHOD_STYLE                     # noqa: E402
import plot_metrics_overview as pmo                        # noqa: E402

PAPER = pmo.PAPER
OUT = pmo.OUT
# The figures the paper includes go to visuals/main.
# generalist_scores_centroid (plot_generalist_scores.py) replaced this one
# there on 2026-09-17: the outcome split moves with the solved threshold.
IN_PAPER: set[str] = set()

# Columns: the body a panel title starts with.
COLUMNS = ['CartPole', 'Acrobot', 'MountainCar', 'HalfCheetah', 'MiniGrid']
# (row label, [(paper directory, cell, what the switch changes) or None] a
# column). Rows are what the switch touches: what the agent sees, the world it
# acts in, or what its actions do; MiniGrid's room swap changes the world.
# Values are the run configs' (param_range, noise width, task.options).
GRID = [
    ('Noise', [
        ('gymnax/noise/2task', 'CartPole_v1_sigma0.5', 'offset 0.5'),
        ('gymnax/noise/2task', 'Acrobot_v1_sigma0.5', 'offset 0.5'),
        ('gymnax/noise/2task', 'MountainCar_v0_sigma0.05', 'offset 0.05'),
        ('mjx/cheetah/noise/2task', 'cheetah_noise', 'offset 0.25'),
        None,
    ]),
    ('Physics', [
        ('gymnax/physics/2task', 'CartPole_v1_sigma1.0', 'pole length ×3'),
        ('gymnax/physics/2task', 'Acrobot_v1_sigma1.0', 'link mass ×1.15'),
        ('gymnax/physics/2task', 'MountainCar_v0_sigma1.0', 'gravity ×1.5'),
        ('mjx/cheetah/physics/2task', 'cheetah_friction', 'ground friction'),
        ('minigrid/minigrid', 'MiniGrid_8x8_16x16', 'room 16x16 → 8x8'),
    ]),
    ('Action reversal', [
        ('gymnax/actions/2task', 'CartPole_v1_sigma1.0', 'order reversed'),
        ('gymnax/actions/2task', 'Acrobot_v1_sigma1.0', 'order reversed'),
        ('gymnax/actions/2task', 'MountainCar_v0_sigma1.0', 'order reversed'),
        # Continuous controls: the sub-task negates the action (envs/mjx.py).
        ('mjx/cheetah/actions/2task', 'cheetah_action', 'sign flipped'),
        None,
    ]),
]

# One row a method; `es` and `pbt` hold whichever arm of their pair the paper
# directory keeps.
NE_ARMS = ['ga', 'es']
RL_ARMS = ['ppo', 'trac', 'redo', 'cchain', 'pbt']
ARMS = NE_ARMS + RL_ARMS

# Blue = learned the sub-task it was shown, dark when it also kept the
# previous one; orange = kept the previous one but did not learn; grey = neither.
OUTCOMES = [('generalist', 'Generalist', '#1c5cab'),
            ('switching', 'Switching specialist', '#9ec5f4'),
            ('stuck', 'Stuck on previous', '#eb6834'),
            ('neither', 'Neither', '#d3d2cb')]
INK, SURFACE = '#0b0b0b', '#ffffff'
MUTED = pmo.MUTED
BAR_H = 0.7
# cell -> the parity of the checkpoints kept (index = phase; sub-task = index
# mod 2). MiniGrid's sub-task 0 is the 8x8 room (envs/minigrid.py), so the
# even checkpoints are scored on 8x8 and on the 16x16 room they came from.
ONE_DIRECTION = {'MiniGrid_8x8_16x16': 0}


def summarise(detail: dict, cell: str, method: str):
    """Outcome fractions and retention for one cell and method, recomputed
    from the per-trial outcomes exactly as generalist_checkpoints.py does:
    (fractions, retention or None, trials that ever held a generalist, trials)."""
    keep = ONE_DIRECTION.get(cell)
    trials = [[o for i, o in enumerate(v['outcome']) if keep is None or i % 2 == keep]
              for k, v in detail.items() if k.startswith(f'{cell}/{method}/')]
    if not trials:
        return None
    pooled, ret = [], []
    for outcome in trials:
        g = np.array([o == 'generalist' for o in outcome])
        pooled += [o for o in outcome if o is not None]
        if g.any():
            ret.append(g[int(np.argmax(g)):].mean())
    pooled = np.asarray(pooled, dtype=object)
    frac = [float(np.mean(pooled == key)) for key, _, _ in OUTCOMES]
    return frac, (float(np.mean(ret)) if ret else None), len(ret), len(trials)


def panel_stats(sub, cell, agent):
    """({row arm: summary or None}, {row arm: arm drawn}) for one panel, or
    (None, reason) when its JSON is missing."""
    path = PAPER / sub / f'generalist_checkpoints_{agent}.json'
    if not path.exists():
        return None, path
    detail = json.loads(path.read_text())
    present = sorted({k.split('/')[1] for k in detail if k.startswith(f'{cell}/')})
    remap, _ = pmo.keep_one_arm(sub, present)
    stats, source = {}, {}
    for m in present:
        row = remap.get(m, m)
        if row in ARMS and m not in pmo.NOT_REPORTED:
            stats[row], source[row] = summarise(detail, cell, m), m
    return {m: stats.get(m) for m in ARMS}, source


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--agent', nargs='+', default=['centroid', 'elite'],
                    choices=['centroid', 'elite'])
    # Wider than the text, like the other main figures: the one-line titles need it.
    ap.add_argument('--width', type=float, default=10.0)
    ap.add_argument('--row-height', type=float, default=1.0)
    ap.add_argument('--font-size', type=float, default=6.0)
    ap.add_argument('--out', default=None,
                    help='output stem; _<agent>.pdf/.png/.md is appended (default: '
                         'generalist_outcomes under visuals/, or visuals/main/ for IN_PAPER)')
    args = ap.parse_args()

    fs = args.font_size
    # One entry a pair: `pbt` is PBT-PPO, whichever N the directory kept.
    METHOD_STYLE['pbt'] = {**METHOD_STYLE['pbt'], 'label': 'PBT-PPO'}
    plt.rcParams.update({
        'font.size': fs, 'axes.titlesize': fs + 0.5, 'xtick.labelsize': fs - 1,
        'ytick.labelsize': fs - 0.5, 'axes.linewidth': 0.5,
        'xtick.major.width': 0.5, 'xtick.major.size': 2, 'ytick.major.size': 0,
    })
    # A gap between the NE and the RL block; bars run top to bottom.
    gap = 0.4
    ypos = {m: i + (gap if m in RL_ARMS else 0.0) for i, m in enumerate(ARMS)}
    split = len(NE_ARMS) - 0.5 + gap / 2

    for agent in args.agent:
        width = args.width
        top_in, bottom_in = 0.4, 0.3
        height = args.row_height * len(GRID) + top_in + bottom_in
        fig = plt.figure(figsize=(width, height))
        gs = GridSpec(len(GRID), len(COLUMNS), figure=fig,
                      left=0.86 / width, right=1 - 0.32 / width,
                      top=1 - top_in / height, bottom=bottom_in / height,
                      wspace=0.95, hspace=0.42)
        axes = [[fig.add_subplot(gs[r, c]) for c in range(len(COLUMNS))]
                for r in range(len(GRID))]
        # A retention label runs past its panel into the gap; transparent axes
        # keep the next panel from painting over it.
        for ax in (a for row in axes for a in row):
            ax.patch.set_visible(False)
        md = [f'# generalist_outcomes_{agent}', '',
              'Pooled post-switch checkpoints by outcome; retention over the trials '
              'that ever held a generalist (found / n). See the docstring of '
              '`scripts/analysis/plot_generalist_outcomes.py`.', '',
              '| Row | Task | Change | Method | Arm | n | found | retention | '
              + ' | '.join(label for _, label, _ in OUTCOMES) + ' | most |',
              '|' + '---|' * (9 + len(OUTCOMES))]
        print(f'--- {agent}')
        lowest = {c: max(r for r in range(len(GRID)) if GRID[r][1][c] is not None)
                  for c in range(len(COLUMNS))}
        for r, (row_label, cells) in enumerate(GRID):
            for c, spec in enumerate(cells):
                ax = axes[r][c]
                if spec is None:
                    ax.axis('off')
                    continue
                sub, cell, change = spec
                body = COLUMNS[c]
                ax.set_xlim(0, 1)
                ax.set_ylim(max(ypos.values()) + 0.6, -0.6)
                ax.set_title(f'{body}, {change}', fontweight='bold', pad=3)
                ax.spines[['top', 'right']].set_visible(False)
                ax.set_xticks([0, 0.5, 1])
                ax.set_xticklabels(['0', '.5', '1'] if r == lowest[c] else [])
                ax.tick_params(axis='x', pad=1.5)
                ax.grid(axis='x', color='0.9', lw=0.5, zorder=0)
                ax.set_axisbelow(True)
                ax.set_yticks([ypos[m] for m in ARMS])
                ax.set_yticklabels([METHOD_STYLE[m]['label'] for m in ARMS]
                                   if c == 0 else [], color=INK)
                ax.axhline(split, color='0.45', lw=0.6, ls=':', zorder=1)
                if c == 0:
                    ax.annotate(row_label, xy=(0.1 / width, 0.5),
                                xycoords=('figure fraction', 'axes fraction'),
                                rotation=90, ha='center', va='center',
                                fontweight='bold', fontsize=fs + 0.5)
                stats, source = panel_stats(sub, cell, agent)
                if stats is None:
                    ax.text(0.5, 0.5, 'no solved threshold', transform=ax.transAxes,
                            ha='center', va='center', color=MUTED, fontsize=fs - 1)
                    print(f'{sub}: missing {source.name}')
                    continue
                right = blended_transform_factory(ax.transAxes, ax.transData)
                best = max((s[0][0] for s in stats.values() if s), default=0.0)
                for m in ARMS:
                    y = ypos[m]
                    if stats[m] is None:
                        ax.text(0.03, y, 'not run', va='center', color=MUTED,
                                fontsize=fs - 1)
                        continue
                    frac, retention, found, n = stats[m]
                    left = 0.0
                    for (_, _, colour), f in zip(OUTCOMES, frac):
                        ax.barh(y, f, left=left, height=BAR_H, color=colour,
                                edgecolor=SURFACE, linewidth=0.4, zorder=2)
                        left += f
                    top = best > 0 and np.isclose(frac[0], best)
                    if top:
                        ax.barh(y, left, height=BAR_H, fill=False, edgecolor=INK,
                                linewidth=0.9, zorder=3)
                    # Retention averages only the trials that ever found a
                    # generalist, so how many did goes beside it when not all.
                    label = ('–' if retention is None else f'{retention:.2f}'
                             + ('' if found == n else f' ({found})'))
                    ax.text(1.04, y, label, transform=right, ha='left', va='center',
                            color=MUTED if retention is None else INK,
                            fontsize=fs - 1.5, zorder=5, clip_on=False)
                    ret = '-' if retention is None else f'{retention:.2f}'
                    md.append(f'| {row_label} | {body} | {change} | '
                              f'{METHOD_STYLE[m]["label"]} | {source[m]} | {n} | {found} | '
                              f'{ret} | ' + ' | '.join(f'{f:.2f}' for f in frac)
                              + f' | {"yes" if top else ""} |')
                    print(f'{sub:26s} {cell:24s} {source[m]:7s} n={n:2d} found {found:2d} '
                          f'ret {ret:>5s} ' + ' '.join(f'{f:.2f}' for f in frac)
                          + ('  <- most generalists' if top else ''))
        fig.legend([Patch(color=col) for _, _, col in OUTCOMES]
                   + [Patch(facecolor=SURFACE, edgecolor=INK, linewidth=0.9)],
                   [label for _, label, _ in OUTCOMES] + ['Most generalists'],
                   loc='upper center', bbox_to_anchor=(0.5, 1.0), ncol=len(OUTCOMES) + 1,
                   frameon=False, fontsize=fs, handlelength=1.2, handleheight=0.8,
                   columnspacing=1.2, handletextpad=0.5)
        mid = axes[-1][0].get_position().x0, axes[-1][len(COLUMNS) - 1].get_position().x1
        fig.text(sum(mid) / 2, 0.02 / height * 2, 'Fraction of post-switch checkpoints',
                 ha='center', va='bottom', fontsize=fs)
        name = f'generalist_outcomes_{agent}'
        out = (pathlib.Path(f'{args.out}_{agent}') if args.out
               else (OUT / 'main' if name in IN_PAPER else OUT) / name)
        out.parent.mkdir(parents=True, exist_ok=True)
        for ext in ('pdf', 'png'):
            fig.savefig(out.with_suffix(f'.{ext}'), dpi=300)
        plt.close(fig)
        out.with_suffix('.md').write_text('\n'.join(md) + '\n')
        print(f'wrote {out}.pdf/.png/.md ({width:.2f} x {height:.2f} in)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
