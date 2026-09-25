"""Why some methods keep generalists: the return landscape around a saved agent
on every panel of `generalist_outcomes_centroid` (appendix figure; the
main-text basin-width summary is `plot_basin_width_methods.py`).

    .venv/bin/python scripts/analysis/plot_landscape_main.py
    .venv/bin/python scripts/analysis/plot_landscape_main.py --rl-arm cchain

    -> paper/visuals/landscape/landscape_main.{pdf,png,md}

One slice per method (rows) and panel (columns, `plot_generalist_outcomes.GRID`
order, grouped by what the switch changes), from `landscape_slices.py`'s grids
in `<paper dir>/results/centroid/landscape/`. The plane passes through
checkpoint t (circle) along the run's movement over the next phase (square at
x = 1) and a random orthogonal direction of the same length, window
-1..2 x -1.5..1.5 drifts. Two continuous layers are superimposed: the return
on the sub-task at risk in pink and on the sub-task trained next in teal, each
scaled to (return - floor) / (threshold - floor); below the threshold an ink
grows continuously to BELOW_MAX of its strength and jumps to full strength
once the sub-task is solved. The inks multiply,
so where both sub-tasks are solved the colour is dark indigo; a contour in the
darker shade of each ink marks its threshold. The
slice is a generalist whenever the method has one, else its most common
checkpoint type (`landscape_slices.choose`), named in the corner: kept / lost (a generalist that is / is not one a phase
later), switching, stuck, neither.

The markdown adds, per method of the figure, on the checkpoints that learned the sub-task
they were shown (generalist or switching specialist; stuck and neither are a
learning failure, not a choice between solutions): a point per outcome panel
with at least MIN_LEARNED of them, x = mean action change under 10% relative
weight noise over those checkpoints (`curvature_width.py`'s `width_0.1`, low =
wide basin), y = the generalist share among them. MiniGrid keeps the outcome
figure's direction. Spearman rho across panels, per method. The markdown adds
the family-gap version, and two checkpoint-level tests within each method
(generalist vs switching, kept vs lost).
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
from matplotlib.colors import to_rgb                        # noqa: E402
from matplotlib.gridspec import GridSpec                   # noqa: E402
from matplotlib.lines import Line2D                        # noqa: E402
from scipy.stats import mannwhitneyu, spearmanr            # noqa: E402

REPO = pathlib.Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO / 'scripts'))
sys.path.insert(0, str(REPO / 'scripts/analysis'))
from make_lineplot import FAMILY, METHOD_STYLE             # noqa: E402
import plot_generalist_outcomes as pgo                     # noqa: E402
import plot_metrics_overview as pmo                        # noqa: E402

PAPER = pmo.PAPER
OUT = pmo.OUT / 'landscape'
STEM = 'landscape_main'
INK, MUTED = pgo.INK, pmo.MUTED
OUTCOME_COLOUR = {k: c for k, _, c in pgo.OUTCOMES}
# One ramp a sub-task, multiplied like two inks on paper: the overlap is the
# darkest colour (pink x teal = indigo). The pair is Stevens' bivariate
# scheme; simulated deutan separation of the two inks is dE 12.
GROUND = '#f4f3ef'
INK_AT_RISK, INK_NEXT = '#c2549d', '#3fb8c0'


def _ramp(score, ink):
    """score in [0, 1] (...,) -> the multiplicative filter of that ink (..., 3)."""
    return 1 - score[..., None] * (1 - np.array(to_rgb(ink)))


# Below its threshold an ink reaches at most this strength, and it jumps to
# full strength once the sub-task is solved: the colour stays continuous
# underneath, but solved and nearly solved no longer look alike.
BELOW_MAX = 0.6


def ink_level(score):
    """Scaled return ((return - floor) / (threshold - floor)) -> ink strength."""
    return np.where(score >= 1, 1.0, BELOW_MAX * np.clip(score, 0, 1))


def overlay(a, b):
    """Scores a (at risk), b (trained next) in [0, 1] -> RGB on the ground."""
    return np.array(to_rgb(GROUND)) * _ramp(a, INK_AT_RISK) * _ramp(b, INK_NEXT)


PERT_MARKER = {'Noise': 'o', 'Physics': 's', 'Action reversal': '^'}
MIN_LEARNED = 10
CELL_IN = 0.45
SPACER = 0.35                 # width of the gap between groups, in columns
TOP_IN, BOTTOM_IN = 0.78, 0.52
# The outcome figure's words. A checkpoint type (`landscape_slices.choose`) ->
# its corner label; generalists add what the next phase did to them.
TYPE_LABEL = {'kept': 'Generalist\n(kept)', 'lost': 'Generalist\n(lost)',
              'generalist': 'Generalist', 'switching': 'Switching\nspecialist',
              'stuck': 'Stuck on\nprevious', 'neither': 'Neither'}


def wrap(change, width=11):
    """The outcome figure's change label on two lines (so the titles align),
    broken at the space nearest its middle when longer than a slice is wide."""
    if len(change) <= width or ' ' not in change:
        return change + '\n'
    cut = min((i for i, ch in enumerate(change) if ch == ' '),
              key=lambda i: abs(i - len(change) / 2))
    return f'{change[:cut]}\n{change[cut + 1:]}'


# Bodies left out of both landscape figures (2026-09-16: HalfCheetah, whose
# slices and PBT/GA runs are still coming); `--include` puts one back.
EXCLUDED = {'HalfCheetah'}


def panel_list(excluded=EXCLUDED):
    """[(group, body, change, paper dir, cell)] in outcome-figure order, by group."""
    return [(group, pgo.COLUMNS[c], p[2], p[0], p[1])
            for group, row in pgo.GRID for c, p in enumerate(row)
            if p and pgo.COLUMNS[c] not in excluded]


def load_slice(sub, cell, arm):
    """(RGB grid, alphas, betas, row) or None."""
    path = PAPER / sub / 'results/centroid/landscape' / f'{cell}.json'
    if not path.exists():
        return None
    blob = json.loads(path.read_text())
    row = blob['rows'].get(arm)
    npz = path.parent / f'{cell}_{arm}.npz'
    if row is None or not npz.exists():
        return None
    z = np.load(npz)
    lo, thr = blob['floor'], blob['threshold']
    a, b = ((z[k] - lo) / (thr - lo) for k in ('at_risk', 'trained_next'))
    return (a, b), z['alphas'], z['betas'], row


def draw_slices(fig, gs, col_of, panels, arms, fs):
    rows_md, span = [], {}
    for i, (group, body, change, sub, cell) in enumerate(panels):
        for r, arm in enumerate(arms):
            ax = fig.add_subplot(gs[r, col_of[i]])
            ax.set_xticks([])
            ax.set_yticks([])
            for s in ax.spines.values():
                s.set_linewidth(0.4)
                s.set_color('0.65')
            if r == 0:
                ax.set_title(f'{body}\n{wrap(change)}', fontsize=fs - 1, pad=2,
                             linespacing=1.1)
                span.setdefault(group, [ax, ax])[1] = ax
            if i == 0:
                ax.set_ylabel(METHOD_STYLE[arm]['label'], rotation=0, ha='right', va='center',
                              fontsize=fs, labelpad=3)
            got = load_slice(sub, cell, arm)
            if got is None:
                ax.set_facecolor('white')
                ax.text(0.5, 0.5, 'not run', transform=ax.transAxes, ha='center',
                        va='center', color=MUTED, fontsize=fs - 1.5)
                continue
            (a, b), al, be, row = got
            ax.imshow(overlay(ink_level(a), ink_level(b)), origin='lower',
                      extent=[al[0], al[-1], be[0], be[-1]], aspect='auto',
                      interpolation='nearest')
            # Where each sub-task is solved, in a darker shade of its ink.
            for score, ink in ((a, INK_AT_RISK), (b, INK_NEXT)):
                if score.min() < 1 <= score.max():
                    ax.contour(al, be, score, levels=[1.0],
                               colors=[tuple(0.6 * np.array(to_rgb(ink)))], linewidths=0.4)
            ax.plot(0, 0, 'o', ms=2.4, mfc='white', mec=INK, mew=0.5, zorder=5)
            ax.plot(1, 0, 's', ms=2.0, mfc=INK, mec='white', mew=0.4, zorder=5)
            ax.text(0.04, 0.95, TYPE_LABEL[row['desc']], transform=ax.transAxes, ha='left', va='top',
                    linespacing=1.0,
                    fontsize=fs - 2, color=INK,
                    bbox=dict(boxstyle='round,pad=0.1', fc='white', ec='none', alpha=0.8))
            rows_md.append((group, f'{body}, {change}', METHOD_STYLE[arm]['label'], row))
    fig.canvas.draw()
    for group, (a, b) in span.items():
        pa, pb = a.get_position(), b.get_position()
        fig.text((pa.x0 + pb.x1) / 2, pa.y1 + 0.33 / fig.get_figheight(), group,
                 ha='center', va='bottom', fontweight='bold', fontsize=fs + 0.5)
    return rows_md


def checkpoints(panels, methods):
    """{(method, group, panel label): [(t, width, outcome at t, outcome at t+1)]}
    for every trial, from the outcome figure's JSON and `curvature_width.json`."""
    out = {}
    for group, body, change, sub, cell in panels:
        det = json.loads((PAPER / sub / 'generalist_checkpoints_centroid.json').read_text())
        path = PAPER / sub / 'results/centroid/curvature_width.json'
        cw = json.loads(path.read_text())['cells'].get(cell, {}) if path.exists() else {}
        present = sorted({k.split('/')[1] for k in det if k.startswith(f'{cell}/')})
        remap, _ = pmo.keep_one_arm(sub, present)
        for m in present:
            row = remap.get(m, m)
            if row not in methods or m in pmo.NOT_REPORTED:
                continue
            key_out = (row, group, f'{body}, {change}')
            for key, v in det.items():
                if not key.startswith(f'{cell}/{m}/'):
                    continue
                w = (cw.get(m, {}).get('trials', {}).get(key.rsplit('/', 1)[1]) or {}).get('width_0.1')
                if not w:
                    continue
                o = v['outcome']
                for t in range(len(o)):
                    if o[t] is not None and w[t] is not None:
                        out.setdefault(key_out, []).append(
                            (t, w[t], o[t], o[t + 1] if t + 1 < len(o) else None))
    return out


def keep_direction(panel, t, shift=0):
    """MiniGrid: only the checkpoints the outcome figure keeps (`shift` 1 for the
    checkpoint BEFORE one of them)."""
    keep = pgo.ONE_DIRECTION.get('MiniGrid_8x8_16x16') if panel.startswith('MiniGrid') else None
    return keep is None or (t + shift) % 2 == keep


def per_method(ck):
    """{method: [(width, generalist share, group, panel, learned)]} on generalist +
    switching checkpoints, panels with at least MIN_LEARNED of them."""
    data = {}
    for (m, group, panel), rows in ck.items():
        learned = [(w, o == 'generalist') for t, w, o, _ in rows
                   if o in ('generalist', 'switching') and keep_direction(panel, t)]
        if len(learned) >= MIN_LEARNED:
            data.setdefault(m, []).append((float(np.mean([w for w, _ in learned])),
                                           float(np.mean([g for _, g in learned])),
                                           group, panel, len(learned)))
    return data


def spearman(pts):
    if len(pts) < 4:
        return None
    return spearmanr([q[0] for q in pts], [q[1] for q in pts])


def auc(ck, methods, pick):
    """{method: (AUC, n_a, n_b)}: P(a class-b checkpoint changes more actions than a
    class-a one), pooled within panels. `pick(t, o, o_next, panel)` -> 'a', 'b' or None."""
    out = {}
    for m in methods:
        num = den = na = nb = 0
        for (mm, _group, panel), rows in ck.items():
            if mm != m:
                continue
            a = [w for t, w, o, n in rows if pick(t, o, n, panel) == 'a']
            b = [w for t, w, o, n in rows if pick(t, o, n, panel) == 'b']
            na, nb = na + len(a), nb + len(b)
            if len(a) >= 3 and len(b) >= 3:
                num += mannwhitneyu(b, a).statistic
                den += len(a) * len(b)
        if den:
            out[m] = (num / den, na, nb)
    return out


def draw_key(fig, x, y, fs):
    """One line at figure coordinates (x, y), left to right: the two ink ramps and
    the overlap patch. Returns the right end."""
    W, H = fig.get_figwidth(), fig.get_figheight()
    zero, one = np.zeros((1, 64)), ink_level(np.linspace(0, 1.02, 64))[None, :]
    items = [('previous sub-task', overlay(one, zero), True),
             ('shown sub-task', overlay(zero, one), True),
             ('Generalist (both)', overlay(np.ones((1, 4)), np.ones((1, 4))), False)]
    for label, img, ramp in items:
        w = (0.5 if ramp else 0.14) / W
        ax = fig.add_axes([x, y - 0.045 / H, w, 0.09 / H])
        ax.imshow(img, aspect='auto', interpolation='bilinear')
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_linewidth(0.4)
        if ramp:
            fig.text(x, y - 0.07 / H, '0', ha='left', va='top', fontsize=fs - 1.5, color=MUTED)
            fig.text(x + w, y - 0.07 / H, 'solved', ha='right', va='top', fontsize=fs - 1.5,
                     color=MUTED)
        t = fig.text(x + w + 0.05 / W, y, label, ha='left', va='center', fontsize=fs)
        fig.canvas.draw()
        x = t.get_window_extent().x1 / fig.dpi / W + 0.2 / W
    return x


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--rl-arm', default='redo', help='the continual-RL row besides PPO and PBT')
    ap.add_argument('--include', nargs='*', default=[], metavar='BODY',
                    help=f'bodies to draw despite EXCLUDED {sorted(EXCLUDED)}')
    ap.add_argument('--font-size', type=float, default=6.0)
    args = ap.parse_args()
    fs = args.font_size
    plt.rcParams.update({
        'font.size': fs, 'axes.titlesize': fs, 'xtick.labelsize': fs - 1,
        'ytick.labelsize': fs - 1, 'axes.linewidth': 0.5, 'xtick.major.width': 0.5,
        'ytick.major.width': 0.5, 'xtick.major.size': 2, 'ytick.major.size': 2,
    })
    # One entry a pair, as in the outcome figure: `pbt` is PBT-PPO, whichever N was kept.
    METHOD_STYLE['pbt'] = {**METHOD_STYLE['pbt'], 'label': 'PBT-PPO'}
    arms = ['ga', 'es', 'ppo', args.rl_arm, 'pbt']
    panels = panel_list(EXCLUDED - set(args.include))
    ratios, col_of = [], []
    for i, (group, *_rest) in enumerate(panels):
        if i and group != panels[i - 1][0]:
            ratios.append(SPACER)
        col_of.append(len(ratios))
        ratios.append(1)
    W = 0.62 + sum(ratios) * CELL_IN * 1.08 + 0.12
    H = len(arms) * CELL_IN * 1.08 + TOP_IN + BOTTOM_IN
    fig = plt.figure(figsize=(W, H))
    left = 0.62 / W
    grid_right = left + sum(ratios) * CELL_IN * 1.08 / W
    gs_a = GridSpec(len(arms), len(ratios), figure=fig, left=left, right=grid_right,
                    top=1 - TOP_IN / H, bottom=BOTTOM_IN / H, wspace=0.08, hspace=0.08,
                    width_ratios=ratios)
    rows_md = draw_slices(fig, gs_a, col_of, panels, arms, fs)

    ck = checkpoints(panels, pgo.ARMS)
    data = per_method(ck)

    # Legend along the top, reading note below.
    y_top = 1 - 0.11 / H
    x_key = draw_key(fig, left, y_top, fs)
    fig.legend([Line2D([], [], color='0.35', lw=0.6)], ['solved threshold'],
               loc='center left', bbox_to_anchor=(x_key, y_top), ncol=1,
               frameon=False, fontsize=fs, handlelength=1.2,
               columnspacing=1.2, handletextpad=0.4, borderaxespad=0)
    fig.text(left, 0.04 / H,
             'Circle: checkpoint $t$; square: checkpoint $t{+}1$, at 1 on x, the movement over '
             'the next phase;\ny: a random orthogonal direction of the same length. '
             'Sub-tasks are named as at checkpoint $t{+}1$.\n'
             'Corner: a Generalist (kept or lost one phase later) if the method ever has one, '
             'else its most common outcome.',
             ha='left', va='bottom', fontsize=fs - 0.5, linespacing=1.3)

    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ('pdf', 'png'):
        fig.savefig(OUT / f'{STEM}.{ext}', dpi=300)
    plt.close(fig)
    write_markdown(rows_md, ck, data)
    print(f'wrote {OUT / STEM}.pdf/.png/.md ({W:.2f} x {H:.2f} in)')
    return 0


def write_markdown(rows_md, ck, data):
    md = [f'# {STEM}', '', 'See the docstring of `scripts/analysis/plot_landscape_main.py`.', '',
          '## (a) Landscape slices', '',
          'A generalist if the method has one, else its most common type '
          '(`landscape_slices.choose`). within 1 drift '
          '= share of the unit disc where both sub-tasks clear the threshold, drawn checkpoint '
          'only.', '',
          '| Group | Panel | Method | arm | trial | ckpt | type | kept / lost / switching / stuck '
          '/ neither | drift / \\|θ\\| | within 1 drift |',
          '|---|---|---|---|---|---|---|---|---|---|']
    for group, panel, label, row in rows_md:
        c = row['counts']
        md.append(f'| {group} | {panel} | {label} | {row["arm"]} | {row["trial"]} | {row["t"]} '
                  f'| {row["desc"]} | {c["kept"]} / {c["lost"]} / {c["switching"]} / '
                  f'{c["stuck"]} / {c["neither"]} | {row["rel_drift"]:.3f} '
                  f'| {row["within_unit_drift"]:.2f} |')
    md += ['', '## (b) Width against generalist share, learned checkpoints only', '',
           f'Generalist + switching checkpoints, panels with at least {MIN_LEARNED}.', '',
           '| Method | panels | rho | p |', '|---|---|---|---|']
    for m in pgo.ARMS:
        res = spearman(data.get(m, []))
        if res is not None:
            md.append(f'| {METHOD_STYLE[m]["label"]} | {len(data[m])} | {res[0]:+.3f} '
                      f'| {res[1]:.3g} |')
    # Family gap: NE mean over RL mean of the same per-method points.
    fam = {}
    for m, pts in data.items():
        for x, y, _g, panel, _n in pts:
            fam.setdefault(panel, {}).setdefault(FAMILY[m], []).append((x, y))
    gaps = [(p, np.mean([x for x, _ in f['rl']]) / np.mean([x for x, _ in f['ne']]),
             np.mean([y for _, y in f['ne']]) - np.mean([y for _, y in f['rl']]))
            for p, f in fam.items() if 'ne' in f and 'rl' in f]
    if len(gaps) >= 4:
        rho, p = spearmanr([g[1] for g in gaps], [g[2] for g in gaps])
        md += ['', f'Family gap (width RL / NE against generalist share NE − RL): '
               f'rho = {rho:+.3f}, p = {p:.3g}, {len(gaps)} panels.', '',
               '| Panel | width RL / NE | share NE − RL |', '|---|---|---|']
        md += [f'| {g[0]} | {g[1]:.2f} | {g[2]:+.2f} |' for g in gaps]
    gs_auc = auc(ck, pgo.ARMS, lambda t, o, n, panel: (
        None if not keep_direction(panel, t) else
        'a' if o == 'generalist' else 'b' if o == 'switching' else None))
    kl_auc = auc(ck, pgo.ARMS, lambda t, o, n, panel: (
        None if (o != 'generalist' or n is None or not keep_direction(panel, t, 1)) else
        'a' if n == 'generalist' else 'b'))
    md += ['', '## Within a method, checkpoint by checkpoint', '',
           'AUC = P(a checkpoint of the second class changes more actions under 10% weight '
           'noise than one of the first), pooled within panels; 0.5 = width does not '
           'separate them, above 0.5 = the first class is wider.', '',
           '| Method | generalist vs switching | n | kept vs lost generalist | n |',
           '|---|---|---|---|---|']
    for m in pgo.ARMS:
        a, b = gs_auc.get(m), kl_auc.get(m)
        md.append(f'| {METHOD_STYLE[m]["label"]} | {"-" if a is None else f"{a[0]:.2f}"} '
                  f'| {"-" if a is None else f"{a[1]}/{a[2]}"} '
                  f'| {"-" if b is None else f"{b[0]:.2f}"} '
                  f'| {"-" if b is None else f"{b[1]}/{b[2]}"} |')
    md += ['', '| Method | Panel | learned checkpoints | width | generalist share |',
           '|---|---|---|---|---|']
    md += [f'| {METHOD_STYLE[m]["label"]} | {q[3]} | {q[4]} | {q[0]:.3f} | {q[1]:.2f} |'
           for m in pgo.ARMS for q in data.get(m, [])]
    (OUT / f'{STEM}.md').write_text('\n'.join(md) + '\n')


if __name__ == '__main__':
    sys.exit(main())
