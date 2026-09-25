"""
Figure 1: a schematic of `appendix/landscape_slices_centroid`.

Three rows by four columns over the same pair of sub-tasks, A (pink) and B
(teal); both at once reads as indigo, as in the appendix figure.

    row 1  the end of sub-task A (only A is drawn)
    row 2  after the switch to B
    row 3  after the switch back to A

    ES        tight population in the middle of a wide basin; the centroid
              moves towards the overlap and stays there: a generalist.
    GA        wider population; after each switch it re-forms around an
              individual in the new sub-task's basin: a switching specialist.
    PPO       narrow ridge, policy on its edge; it never leaves: plasticity
              loss.
    CRL       a continual PPO variant, no method named: a slightly wider
              ridge, and the policy hops between the two ridge tips.

Basin areas follow the paper's basin-width ratios (ES 8x, GA 3.5x, continual
RL 1.5x, PPO 1x). Everything you are likely to edit is in the CONFIG block.

    .venv/bin/python scripts/analysis/plot_intro_schematic.py
"""
from pathlib import Path

import matplotlib as mpl
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyArrowPatch

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / 'projects/iclr_2027/paper/visuals/final'
NAME = 'intro_schematic'

# ----------------------------------------------------------------- CONFIG
COLS = ['ES', 'GA', 'PPO', 'CRL']
TITLE = {'ES': 'ES', 'GA': 'GA', 'PPO': 'PPO', 'CRL': 'Continual PPO variant'}
GROUPS = [('Neuroevolution', ['ES', 'GA']), ('Reinforcement learning', ['PPO', 'CRL'])]
ROWS = ['end of\nsub-task A', 'after the\nswitch to B', 'after the switch\nback to A']
AREA = {'ES': 8.0, 'GA': 3.5, 'PPO': 1.0, 'CRL': 1.5}   # basin area relative to PPO
FIG_W = 5.5                    # ICLR text width, inches
R0 = 0.27                      # PPO basin radius, plot units
D = 0.45                       # half-distance between the NE basin centres
RIDGE_ASPECT = 3.0             # RL ridge length / width at constant area
RIDGE_TIP = -0.3               # x of the tip of ridge A in the RL panels
GAP = {'PPO': 0.7, 'CRL': 0.45}   # ridge A tip -> ridge B tip
XLIM, YLIM = 1.4, 0.78         # panel half-extent; wide panels keep 3 rows short
SEED = 7
ES_SIGMA, ES_N = 0.3, 40       # ES search distribution
GA_SPREAD, GA_N = 0.4, 40      # GA population
ES_X = {1: 0.0, 2: 0.02}      # ES centroid x after each switch (row index)

# The appendix figure's inks (plot_landscape_main.py): two inks multiplied on
# paper, so the overlap is the darkest colour.
GROUND = '#f4f3ef'
INK_A, INK_B = '#c2549d', '#3fb8c0'
INK, MUTED = '#0b0b0b', '#6b6a65'
EDGE_A, EDGE_B = '#8f2f6f', '#1f7f86'
# ------------------------------------------------------------------------

mpl.rcParams.update({'font.size': 6.5, 'axes.linewidth': 0.5,
                     'pdf.fonttype': 42, 'ps.fonttype': 42})
RL = ('PPO', 'CRL')


def _ramp(score, ink):
    return 1 - score[..., None] * (1 - np.array(to_rgb(ink)))


def overlay(a, b):
    return np.array(to_rgb(GROUND)) * _ramp(a, INK_A) * _ramp(b, INK_B)


def field(X, Y, centre, r, phase, aspect=1.0, warp=0.08):
    """Plateau in [0, 1], 0.9 on the basin edge; aspect > 1 makes a ridge
    along x at constant area; warp bends the outline so it is not a circle."""
    Xw = X + warp * r * np.sin(2.6 * Y / r + phase) + 0.5 * warp * r * np.sin(4.1 * X / r + 2 * phase)
    Yw = Y + warp * r * np.sin(2.2 * X / r + 1.7 * phase)
    sa = np.sqrt(aspect)
    q = ((Xw - centre[0]) / (r * sa)) ** 2 + ((Yw - centre[1]) / (r / sa)) ** 2
    return 1.0 / (1.0 + (1 / 0.9 - 1) * q ** 5)


def geometry(name):
    """Basin radius and the two basin centres of a column."""
    r = R0 * np.sqrt(AREA[name])
    if name in RL:
        half = r * np.sqrt(RIDGE_ASPECT)
        ca = (RIDGE_TIP - half, 0.0)
        cb = (RIDGE_TIP + GAP[name] + half, 0.0)
    else:
        ca, cb = (-D, 0.0), (D, 0.0)
    return r, ca, cb


def landscape(ax, name, with_b):
    r, ca, cb = geometry(name)
    rl = name in RL
    aspect = RIDGE_ASPECT if rl else 1.0
    warp = 0.03 if rl else 0.08
    gx = np.linspace(-XLIM, XLIM, 500)
    gy = np.linspace(-YLIM, YLIM, int(500 * YLIM / XLIM))
    X, Y = np.meshgrid(gx, gy)
    fa = field(X, Y, ca, r, 0.4, aspect, warp)
    fb = field(X, Y, cb, r, 2.1, aspect, warp) if with_b else np.zeros_like(fa)
    ax.imshow(np.clip(overlay(fa, fb), 0, 1), extent=[-XLIM, XLIM, -YLIM, YLIM],
              origin='lower', interpolation='bilinear', zorder=0, aspect='auto')
    ax.contour(X, Y, fa, levels=[0.9], colors=[EDGE_A], linewidths=0.45, zorder=1)
    if with_b:
        ax.contour(X, Y, fb, levels=[0.9], colors=[EDGE_B], linewidths=0.45, zorder=1)
    return r, ca, cb


def solution(ax, xy):
    ax.plot(*xy, 'o', ms=4.2, mfc='white', mec=INK, mew=0.8, zorder=9)


def arrow(ax, p, q, rad=0.0, lw=1.0, shrink=3, scale=6):
    """Movement since the previous row; the tail is where the solution was."""
    a = FancyArrowPatch(p, q, connectionstyle=f'arc3,rad={rad}', arrowstyle='-|>',
                        mutation_scale=scale, lw=lw, color=INK,
                        shrinkA=shrink, shrinkB=shrink, zorder=7)
    a.set_path_effects([pe.withStroke(linewidth=lw + 1.4, foreground='white')])
    ax.add_patch(a)


def population(ax, pts):
    ax.plot(pts[:, 0], pts[:, 1], 'o', ms=2.0, mfc=INK, mec='white', mew=0.35,
            ls='', alpha=0.85, zorder=4)


def draw(ax, name, row, rng):
    with_b = row > 0
    r, ca, cb = landscape(ax, name, with_b)
    if name == 'ES':
        a0 = np.array([ca[0] - 0.04, 0.05])
        c = a0 if row == 0 else np.array([ES_X[row], 0.02 if row == 1 else 0.0])
        ax.add_patch(Circle(c, ES_SIGMA, fill=False, ls=(0, (2.2, 1.6)), lw=0.7,
                            ec='white', zorder=4))
        population(ax, c + rng.normal(0, ES_SIGMA / 2.0, (ES_N, 2)))
        if row == 1:
            arrow(ax, a0, c)
        solution(ax, c)
    elif name == 'GA':
        at_a = np.array([ca[0], 0.0])
        at_b = np.array([cb[0] - 0.04, 0.04])
        c = at_b if row == 1 else at_a + (np.array([0.04, -0.06]) if row == 2 else 0)
        pts = c + rng.normal(0, GA_SPREAD, (GA_N, 2))
        pts = pts[np.hypot(pts[:, 0] - c[0], pts[:, 1] - c[1]) < 1.5 * r]
        population(ax, pts)
        if row == 1:
            arrow(ax, at_a, c, rad=-0.3)
        elif row == 2:
            arrow(ax, at_b, c, rad=-0.3)
        solution(ax, c)
    else:
        tip_a, tip_b = (RIDGE_TIP, 0.0), (RIDGE_TIP + GAP[name], 0.0)
        if name == 'PPO' or row == 0:
            solution(ax, tip_a)
        elif row == 1:
            arrow(ax, tip_a, tip_b, rad=-0.7)
            solution(ax, tip_b)
        else:
            arrow(ax, tip_b, tip_a, rad=-0.7)
            solution(ax, tip_a)

    if row == 0:
        ax.set_title(TITLE[name], fontsize=7, pad=3)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlim(-XLIM, XLIM); ax.set_ylim(-YLIM, YLIM)
    for s in ax.spines.values():
        s.set_color('#b5b3ad')


def main():
    rng = np.random.default_rng(SEED)
    n, nrow = len(COLS), len(ROWS)
    left_in, right_in, gap_in, group_gap_in = 0.34, 0.04, 0.05, 0.14
    top_in, bottom_in = 0.38, 0.04
    cell_w = (FIG_W - left_in - right_in - gap_in * (n - 1) - group_gap_in) / n
    cell_h = cell_w * YLIM / XLIM
    fig_h = top_in + nrow * cell_h + (nrow - 1) * gap_in + bottom_in
    fig = plt.figure(figsize=(FIG_W, fig_h))
    xs, x = {}, left_in
    for group, members in GROUPS:
        for m in members:
            xs[m] = x
            x += cell_w + gap_in
        x += group_gap_in
    for row in range(nrow):
        y = bottom_in + (nrow - 1 - row) * (cell_h + gap_in)
        for name in COLS:
            ax = fig.add_axes([xs[name] / FIG_W, y / fig_h, cell_w / FIG_W, cell_h / fig_h])
            draw(ax, name, row, rng)
            if name == COLS[0]:
                ax.set_ylabel(ROWS[row], fontsize=6, labelpad=4, linespacing=1.1)
    yg = (fig_h - top_in + 0.19) / fig_h
    for group, members in GROUPS:
        x0, x1 = xs[members[0]], xs[members[-1]] + cell_w
        fig.text((x0 + x1) / 2 / FIG_W, yg, group, ha='center', va='bottom', fontsize=7,
                 color=MUTED)
        fig.add_artist(Line2D([x0 / FIG_W, x1 / FIG_W], [yg - 0.02 / fig_h] * 2,
                              color=MUTED, lw=0.5))

    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f'{NAME}.pdf')
    fig.savefig(OUT / f'{NAME}.png', dpi=300)
    print(OUT / f'{NAME}.png')


if __name__ == '__main__':
    main()
