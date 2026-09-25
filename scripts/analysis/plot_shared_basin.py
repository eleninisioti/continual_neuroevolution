"""Shared basin: does the next task's solution lie in this task's basin?

    .venv/bin/python scripts/analysis/plot_shared_basin.py --extract   # data file first
    .venv/bin/python scripts/analysis/plot_shared_basin.py

Reads `child_survival.py overlap`'s raw passes (results/overlap_probe/raw)
through `child_survival.overlap_points` and writes
`paper/visuals/final/data/shared_basin.json`, then
`paper/visuals/final/appendix/shared_basin.{pdf,png,md}`:

  appendix/shared_basin   per Figure 2 setting (gymnax and MiniGrid), the share of random moves
         after which the policy solves both the task just trained and the
         next one (rescaled score >= 0.5 on each, the width's "solves").
         Left, the definition (KEY): moves of radius 0.1 of each parameter
         tensor's norm, the width's perturbation at one rung of its ladder,
         the same for every method. Right (OWN_STEP): moves as large, per
         tensor, as the step the method took over the next task -- the
         distance it actually travels. Crossed: frozen specialists (Figure
         3's class).
  shared_basin_main       the definition against Figure 2's LA - F, one point per setting and
         method, frozen specialists left out; Spearman over all points.

Until 2026-09-25 the definition was OWN_STEP at 0.8. A method that moves less
scores higher on it and also forgets less, whatever its landscape, so the
radius is now method-independent; own-step stays as the right-hand panel.

HalfCheetah and Kinetix are not probed (see child_survival's overlap section).
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
from matplotlib.transforms import blended_transform_factory  # noqa: E402
from scipy.stats import spearmanr                          # noqa: E402

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'scripts'))
sys.path.insert(0, str(REPO / 'scripts' / 'analysis'))
from make_lineplot import METHOD_ORDER, METHOD_STYLE       # noqa: E402

FINAL = REPO / 'projects/iclr_2027/paper/visuals/final'
DATA = FINAL / 'data' / 'shared_basin.json'
OUT = FINAL / 'appendix' / 'shared_basin'      # left: overlap per setting (appendix)
MAIN = FINAL / 'shared_basin_main'               # right: overlap against LA - F (main text)
MAIN_W = 2.55                                    # drawn width of the main-text panel (in)
# Figure 5 (fig:basin) puts basin_width_main and shared_basin_main side by side
# in [b]-aligned minipages of 0.52 and 0.44 \linewidth. Both panels share this
# vertical layout, in PRINT inches from the bottom of the page: the same page
# height, axes bottom, x-label top and legend bottom (two rows at y = 0), so
# the two axes, their x-labels and their legend rows line up. Each script draws
# at its own width and scales these by drawn / printed width.
FIG5 = dict(linewidth=5.5, left=0.52, right=0.44,   # \linewidth and the minipage shares
            height=2.05, ax_bottom=0.70, ax_top=0.18, xlabel_top=0.51)
SETTINGS = [('cartpole_noise10', 'CartPole, noise'), ('acrobot_noise10', 'Acrobot, noise'),
            ('mountaincar_noise10', 'MountainCar, noise'), ('minigrid', 'MiniGrid'),
            ('cartpole_actions', 'CartPole, reversal'), ('acrobot_actions', 'Acrobot, reversal'),
            ('mountaincar_actions', 'MountainCar, reversal')]
NE = {'es', 'ga'}
PRINT_W = 5.5
KEY, OWN_STEP, GOOD = 'fixed:0.1', 'matched:1', 0.5   # plot_basin_width_return's LEARNED


def extract():
    import child_survival as cs
    pts = cs.overlap_points(KEY, GOOD)
    own = {(p[0], p[1]): p[2] for p in cs.overlap_points(OWN_STEP, GOOD)}
    DATA.write_text(json.dumps(dict(
        extracted=str(datetime.date.today()), key=KEY, own_step_key=OWN_STEP, good=GOOD,
        points=[dict(setting=p[0], method=p[1], overlap=p[2], own_step=own.get((p[0], p[1])),
                     keep=p[3], frozen=bool(p[4]), checkpoints=p[5]) for p in pts]),
        indent=1) + '\n')
    print(f'wrote {DATA} ({len(pts)} points)')


def style(m):
    s = METHOD_STYLE['es' if m == 'es' else m]
    label = {'pbt': 'PBT-PPO', 'es': 'ES'}.get(m, s['label'])
    return s['color'], label


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--extract', action='store_true')
    ap.add_argument('--font-size', type=float, default=7.0)
    args = ap.parse_args()
    if args.extract:
        extract()
    pts = json.loads(DATA.read_text())['points']
    fs = args.font_size
    plt.rcParams.update({'font.size': fs, 'axes.titlesize': fs, 'xtick.labelsize': fs - 1,
                         'ytick.labelsize': fs - 1, 'axes.linewidth': 0.5,
                         'xtick.major.width': 0.5, 'ytick.major.width': 0.5,
                         'xtick.major.size': 2, 'ytick.major.size': 2})
    methods = [m for m in METHOD_ORDER if any(p['method'] == m for p in pts)]
    # Appendix: one row per setting, one marker per method, jittered vertically;
    # left the definition, right the method's own step.
    fig, axes = plt.subplots(1, 2, figsize=(PRINT_W, 2.3), sharey=True)
    offs = np.linspace(-0.3, 0.3, len(methods))
    for ax, field, xlabel in ((axes[0], 'overlap', 'moves of radius 0.1 (definition)'),
                              (axes[1], 'own_step', "moves of the method's own step")):
        for i, (key, title) in enumerate(SETTINGS):
            for m, dy in zip(methods, offs):
                p = [q for q in pts if q['setting'] == key and q['method'] == m]
                if not p or p[0][field] is None:
                    continue
                c, lab = style(m)
                x, y = p[0][field], i + dy
                if p[0]['frozen']:
                    ax.scatter(x, y, s=14, facecolor='white', edgecolor=c, lw=0.8, zorder=3)
                    ax.scatter(x, y, s=10, marker='x', color=c, lw=0.7, zorder=4)
                else:
                    ax.scatter(x, y, s=14, color=c, lw=0, zorder=3,
                               label=lab if i == 0 and ax is axes[0] else None)
            if i:
                ax.axhline(i - 0.5, color='0.85', lw=0.4, zorder=0)
        ax.set_xlim(-0.03, 1.0)
        ax.set_xlabel(xlabel)
        ax.spines[['top', 'right']].set_visible(False)
    axes[0].set_yticks(range(len(SETTINGS)), [t for _, t in SETTINGS])
    axes[0].set_ylim(len(SETTINGS) - 0.5, -0.5)
    fig.legend(loc='center right', frameon=False, handletextpad=0.1)
    fig.subplots_adjust(left=0.19, right=0.86, top=0.97, bottom=0.18, wspace=0.08)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    for ext in ('pdf', 'png'):
        fig.savefig(OUT.with_suffix(f'.{ext}'), dpi=300)
    plt.close(fig)
    # Main text: overlap against Figure 2's LA - F, frozen methods left out.
    kept = [p for p in pts if not p['frozen']]
    k = MAIN_W / (FIG5['right'] * FIG5['linewidth'])   # drawn inches per printed inch
    H = FIG5['height'] * k
    fig, bx = plt.subplots(figsize=(MAIN_W, H))
    for m in methods:
        q = [p for p in kept if p['method'] == m]
        c, lab = style(m)
        bx.scatter([p['overlap'] for p in q], [p['keep'] for p in q], s=12, color=c, lw=0,
                   label=lab, zorder=3)
    r = spearmanr([p['overlap'] for p in kept], [p['keep'] for p in kept])
    rn = spearmanr([p['overlap'] for p in kept if p['method'] in NE],
                   [p['keep'] for p in kept if p['method'] in NE])
    rr = spearmanr([p['overlap'] for p in kept if p['method'] not in NE],
                   [p['keep'] for p in kept if p['method'] not in NE])
    bx.text(0.98, 0.04, f'Spearman $\\rho$ = {r.correlation:.2f}\n'
            f'NE {rn.correlation:.2f}, RL {rr.correlation:.2f}',
            transform=bx.transAxes, ha='right', va='bottom')
    bx.set_xlabel('shared basin')
    bx.set_ylabel('LA $-$ F')
    bx.set_xlim(-0.03, 1.0)
    bx.spines[['top', 'right']].set_visible(False)
    bx.xaxis.set_label_coords(0.5, FIG5['xlabel_top'] * k, transform=blended_transform_factory(
        bx.transAxes, fig.dpi_scale_trans))
    # Two rows at (fs - 0.5) pt in print, as basin_width_main's legend: same pitch.
    fig.legend(loc='lower center', bbox_to_anchor=(0.5, 0.0), ncol=4, frameon=False,
               fontsize=(fs - 0.5) * k, handletextpad=0.1, columnspacing=0.5)
    fig.subplots_adjust(left=0.17, right=0.98, top=1 - FIG5['ax_top'] * k / H,
                        bottom=FIG5['ax_bottom'] * k / H)
    for ext in ('pdf', 'png'):
        fig.savefig(MAIN.with_suffix(f'.{ext}'), dpi=300)
    plt.close(fig)
    lines = ['# shared_basin', '', 'Written by `scripts/analysis/plot_shared_basin.py` from '
             f'`{DATA.relative_to(REPO)}` (see the script docstring).', '',
             f'Spearman over settings x methods, frozen left out: all {r.correlation:.2f} '
             f'(p {r.pvalue:.2g}, n {len(kept)}); NE {rn.correlation:.2f} (p {rn.pvalue:.2g}); '
             f'RL {rr.correlation:.2f} (p {rr.pvalue:.2g}).', '',
             '| Setting | Method | overlap (radius 0.1) | overlap (own step) | LA - F | frozen | checkpoints |',
              '|---|---|---|---|---|---|---|']
    lines += [f"| {p['setting']} | {p['method']} | {p['overlap']:.2f} | {p['own_step']:.2f} | {p['keep']:.2f} | "
              f"{'yes' if p['frozen'] else ''} | {p['checkpoints']} |" for p in pts]
    OUT.with_suffix('.md').write_text('\n'.join(lines) + '\n')
    print(f'wrote {OUT}.pdf/.png/.md and {MAIN}.pdf/.png')


if __name__ == '__main__':
    main()
