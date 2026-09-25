"""Does a method's exploration noise find anything at a switch? Weight noise
against action noise on the nine classic-control two-task panels. An appendix
paper figure (app:exploration_signal), the companion of Figure 1.

    # re-read the exploration_signal.py passes (only when they change)
    .venv/bin/python scripts/analysis/plot_exploration_signal.py --extract
    # redraw from visuals/final/data/
    .venv/bin/python scripts/analysis/plot_exploration_signal.py

    -> paper/visuals/final/appendix/exploration_signal.{pdf,png,md}
       paper/visuals/final/data/exploration_signal.json

Figure 1 draws a plane in weight space. NE explores in that space: a
perturbed copy of the centroid is a point of the plane, so what NE can reach
is what its basin covers. PPO explores through its actions and moves along the
gradient, which needs no basin to cross, only sampled episodes whose returns
differ. This figure asks, for the checkpoints that still fail the task they are
about to train on (greedy return below the panel's threshold), how often each
kind of noise finds anything: the share of those checkpoints at which at least
one of 32 samples beats the unperturbed greedy policy, from the same start
state, by more than 5% of the panel's range (threshold - floor, as
landscape_slices.GYM). Mean over trials of the per-trial share, 95% bootstrap
interval over trials; a method with fewer than MIN_N such checkpoints or
MIN_TRIALS trials on a panel is not drawn.

    own sampling   softmax over the policy's logits, PPO's exploration. Not drawn
                   for ES and the GA: nothing fixes an NE network's logit scale
    random actions each step's action uniform with prob 0.3 (eps_0.3)
    held actions   20-step segments, each with prob 0.3 playing ONE uniform
                   action throughout (sticky_0.3): as much noise, correlated in
                   time
    weight noise   fig:basin's perturbation: gaussian noise of norm 0.1 times each
                   tensor's norm (weight_0.1), then greedy actions

Episode return is exactly what ES and the GA select on, but only a lower bound
on what PPO can use: its critic credits single states. The markdown adds the
other strengths (eps 0.1, held 0.1, weight 0.03 / 0.3) and the counts.
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

REPO = pathlib.Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'scripts'))
sys.path.insert(0, str(REPO / 'scripts/analysis'))

import plot_basin_width_methods as b                       # noqa: E402
from make_lineplot import METHOD_STYLE                     # noqa: E402
from source.metrics.continual_metrics import bootstrap_ci  # noqa: E402

pgs = b.pgs
PAPER = REPO / 'projects/iclr_2027/paper'
OUT = pgs.FINAL / 'appendix' / 'exploration_signal'
DATA = pgs.FINAL / 'data' / 'exploration_signal.json'
METHODS = b.METHODS                                         # the paper's one order
NE = {'ga', 'es'}
MARGIN = 0.05
MIN_N, MIN_TRIALS = 5, 3
PRINT_W = 5.5                                               # \linewidth
BODIES = ['CartPole', 'Acrobot', 'MountainCar']
# (condition, legend label, marker, face, edge)
SHOWN = [('softmax', 'own sampling', 'o', '#222222', '#222222'),
         ('eps_0.3', 'random actions', 'o', 'white', '#7a7a7a'),
         ('sticky_0.3', 'held actions', '^', '#9a9a9a', '#9a9a9a'),
         ('weight_0.1', 'weight noise', 's', '#2A6FBB', '#2A6FBB')]
MD_EXTRA = ['eps_0.1', 'sticky_0.1', 'weight_0.03', 'weight_0.3']


def panels():
    """(group, body, change, results dir, cell) for the gymnax two-task panels,
    in basin_width_main's order."""
    sub = {'noise_2task': 'gymnax/noise/2task', 'physics_2task': 'gymnax/physics/2task',
           'actions_2task': 'gymnax/actions/2task'}
    out = []
    for group, body, change, tree, cell in b.panel_list():
        name = tree.rsplit('/', 1)[-1]
        if body in BODIES and name in sub:
            out.append((group, body, change, sub[name], cell))
    return out


def extract():
    """Per panel, method and condition: the per-trial share of failing
    checkpoints at which the noise finds a better episode, and the counts."""
    out = {'panels': {}, 'margin': MARGIN, 'extracted': datetime.date.today().isoformat()}
    for group, body, change, sub, cell in panels():
        f = PAPER / sub / 'results/centroid/exploration_signal' / f'{cell}.npz'
        z = np.load(f, allow_pickle=True)
        meta = json.loads(str(z['meta']))
        C = meta['conditions']
        rng = meta['threshold'] - meta['floor']
        rows = {}
        for m in METHODS:
            if m not in z.files:
                continue
            R = z[m]                                   # trial, t, condition, sample
            greedy = R[:, :, C.index('greedy')]
            failing = greedy.mean(-1) < meta['threshold']
            rows[m] = {'arm': meta['arms'][m]['run_arm'], 'root': meta['arms'][m]['root'],
                       'n_failing': failing.sum(1).tolist(), 'n_checkpoints': int(failing.shape[1]),
                       'found': {}}
            for c in [s[0] for s in SHOWN] + MD_EXTRA:
                hit = ((R[:, :, C.index(c)] - greedy) / rng > MARGIN).any(-1)
                rows[m]['found'][c] = [float(h[f_].mean()) if f_.any() else None
                                       for h, f_ in zip(hit, failing)]
        out['panels'][f'{body}, {change}'] = dict(group=group, body=body, change=change,
                                                  file=str(f.relative_to(REPO)), rows=rows)
        print(f'{group:16s} {body:12s} {sorted(rows)}')
    DATA.parent.mkdir(parents=True, exist_ok=True)
    DATA.write_text(json.dumps(out, indent=1) + '\n')
    print(f'wrote {DATA}')


def summary(row, cond):
    """(mean, lo, hi, n failing checkpoints, trials) or None where too thin."""
    vals = np.array([v for v in row['found'][cond] if v is not None], dtype=float)
    n = int(sum(row['n_failing']))
    if n < MIN_N or vals.size < MIN_TRIALS:
        return None
    mean, lo, hi = bootstrap_ci(vals)
    return float(mean), float(lo), float(hi), n, int(vals.size)


def label(m):
    return 'ES' if m == 'es' else METHOD_STYLE[m]['label']


def draw(blob, fs, path):
    groups = list(dict.fromkeys(p['group'] for p in blob['panels'].values()))
    left, right, top, bottom, gap_x, gap_y = 0.42, 0.04, 0.42, 0.42, 0.14, 0.3
    W = PRINT_W
    cw = (W - left - right - gap_x * (len(BODIES) - 1)) / len(BODIES)
    ch = 0.95
    H = top + bottom + len(groups) * ch + (len(groups) - 1) * gap_y
    fig = plt.figure(figsize=(W, H))
    offs = np.linspace(-0.27, 0.27, len(SHOWN))
    x = np.arange(len(METHODS))
    for r, g in enumerate(groups):
        for c, body in enumerate(BODIES):
            key = next(k for k, p in blob['panels'].items() if p['group'] == g and p['body'] == body)
            p = blob['panels'][key]
            ax = fig.add_axes([(left + c * (cw + gap_x)) / W,
                               1 - (top + r * (ch + gap_y) + ch) / H, cw / W, ch / H])
            for i, m in enumerate(METHODS):
                row = p['rows'].get(m)
                if row is None:
                    continue
                for k, (cond, _, mk, face, edge) in enumerate(SHOWN):
                    if cond == 'softmax' and m in NE:
                        continue
                    s = summary(row, cond)
                    if s is None:
                        continue
                    xi = i + offs[k]
                    ax.plot([xi, xi], [s[1], s[2]], color=edge, lw=0.6, zorder=2)
                    ax.plot(xi, s[0], mk, ms=2.8, mfc=face, mec=edge, mew=0.6, zorder=3)
            for i in range(1, len(METHODS)):
                ax.axvline(i - 0.5, color='0.93', lw=0.5, zorder=0)
            ax.set_xlim(-0.5, len(METHODS) - 0.5)
            ax.set_ylim(-0.03, 1.03)
            ax.set_yticks([0, 0.5, 1])
            ax.set_yticklabels(['0', '.5', '1'] if c == 0 else [])
            ax.set_xticks(x)
            if r == len(groups) - 1:
                ax.set_xticklabels([label(m) for m in METHODS], rotation=45, ha='right',
                                   rotation_mode='anchor', fontsize=fs - 1.5)
            else:
                ax.set_xticklabels([])
            ax.tick_params(length=2, pad=1.5, labelsize=fs - 1.5)
            ax.spines[['top', 'right']].set_visible(False)
            ax.set_title(f'{body}, {p["change"]}', fontsize=fs, pad=2)
            if c == 0:
                ax.set_ylabel(f'{g}\nshare with a signal', fontsize=fs - 0.5, labelpad=2)
    handles = [Line2D([], [], ls='', marker=mk, ms=3.2, mfc=face, mec=edge, mew=0.6, label=lab)
               for _, lab, mk, face, edge in SHOWN]
    fig.legend(handles=handles, loc='upper center', ncol=len(SHOWN), frameon=False,
               fontsize=fs, bbox_to_anchor=(0.5, 1.0), handletextpad=0.2, columnspacing=1.2)
    path.parent.mkdir(parents=True, exist_ok=True)
    for ext in ('pdf', 'png'):
        fig.savefig(path.with_suffix(f'.{ext}'), dpi=300)
    plt.close(fig)
    print(f'wrote {path}.pdf/.png ({W:.2f} x {H:.2f} in)')


def write_markdown(blob, path):
    conds = [s[0] for s in SHOWN] + MD_EXTRA
    lines = [f'# Exploration signal at a switch ({blob["extracted"]})', '',
             'Share of the checkpoints that fail the next task at which at least one of 32 '
             f'samples beats the greedy policy from the same start by > {MARGIN:.0%} of the '
             'range; mean over trials [95% bootstrap CI]; n = failing checkpoints (trials). '
             'Blank: fewer than '
             f'{MIN_N} checkpoints or {MIN_TRIALS} trials.', '']
    for key, p in blob['panels'].items():
        lines += [f'## {p["group"]}: {key}', '',
                  '| method | n | ' + ' | '.join(conds) + ' |',
                  '|---|---|' + '---|' * len(conds)]
        for m in METHODS:
            row = p['rows'].get(m)
            if row is None:
                continue
            cells, n = [], ''
            for cnd in conds:
                s = summary(row, cnd)
                if s is None or (cnd == 'softmax' and m in NE):
                    cells.append('')
                    continue
                cells.append(f'{s[0]:.2f} [{s[1]:.2f}, {s[2]:.2f}]')
                n = f'{s[3]} ({s[4]})'
            lines.append(f'| {label(m)} | {n} | ' + ' | '.join(cells) + ' |')
        lines.append('')
    # The comparison the text makes: RL arms, own sampling vs weight noise.
    lines += ['## RL: weight noise minus own sampling, per panel', '',
              '| panel | ' + ' | '.join(label(m) for m in METHODS if m not in NE) + ' |',
              '|---|' + '---|' * len([m for m in METHODS if m not in NE])]
    for key, p in blob['panels'].items():
        cells = []
        for m in METHODS:
            if m in NE:
                continue
            row = p['rows'].get(m)
            a = row and summary(row, 'weight_0.1')
            s = row and summary(row, 'softmax')
            cells.append(f'{a[0] - s[0]:+.2f}' if a and s else '')
        lines.append(f'| {key} | ' + ' | '.join(cells) + ' |')
    path.with_suffix('.md').write_text('\n'.join(lines) + '\n')
    print(f'wrote {path}.md')


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--extract', action='store_true',
                    help='read the exploration_signal passes and write the data file first')
    ap.add_argument('--font-size', type=float, default=7.0, help='the size in print')
    args = ap.parse_args()
    if args.extract:
        extract()
    if not DATA.exists():
        sys.exit(f'no {DATA}: run with --extract first')
    fs = args.font_size
    plt.rcParams.update({
        'font.size': fs, 'axes.titlesize': fs, 'axes.linewidth': 0.5,
        'xtick.major.width': 0.5, 'ytick.major.width': 0.5,
        'xtick.major.size': 2, 'ytick.major.size': 2,
    })
    METHOD_STYLE['pbt'] = {**METHOD_STYLE['pbt'], 'label': 'PBT-PPO'}
    blob = json.loads(DATA.read_text())
    draw(blob, fs, OUT)
    write_markdown(blob, OUT)
    return 0


if __name__ == '__main__':
    sys.exit(main())
