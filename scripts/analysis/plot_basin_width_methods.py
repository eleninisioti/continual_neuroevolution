"""Basin width of every method relative to PPO, on every panel of the final
`generalist_scores_centroid` figure and of `continual_combined` (Figure 2). A
final paper figure.

    # re-read the width passes (only when they change)
    .venv/bin/python scripts/analysis/plot_basin_width_methods.py --extract
    # redraw from visuals/final/data/
    .venv/bin/python scripts/analysis/plot_basin_width_methods.py [--eps 0.03]

    -> paper/visuals/final/basin_width_main.{pdf,png,md}     (main text: the summary)
       paper/visuals/final/data/basin_width_main.json
       paper/visuals/landscape/basin_width_table.{pdf,png}   (appendix draft: every panel)

Panels, runs and arms are generalist_scores_centroid's (plot_generalist_scores.GRID
and its data file): ES = NES, the PBT-PPO it kept, the same trials. Figure 2's
panels that the grid lacks (SEQUENCES: the 10-sub-task noise runs and Kinetix)
follow Figure 2 instead: its kept ES and PBT arm, its trial count. Each panel's
width comes from the `curvature_width.json` pass over its data root (WIDTH_PASS).
The extract refuses a pass that scored other runs than the data root links to,
or other trials than the figure it follows shows.

Width is `curvature_width.py`'s perturbation robustness at radius `--eps`: the
fraction of probe states whose greedy action changes under gaussian weight
noise (continuous heads: the normalised action distance). On ReLU networks
(gymnax, MiniGrid) the noise is filter-normalised (Li et al., 2018): eps times
each tensor's own norm, since rescaling a ReLU layer leaves the policy
unchanged and weight scale says nothing about it. On tanh networks (HalfCheetah,
Kinetix; ABSOLUTE) weight scale is not a symmetry -- large weights saturate the
units -- and methods' scales differ up to 8x (Kinetix ES against PPO), so the
noise is absolute, s.d. eps on every weight (`width_abs`, the
`centroid_abswidth` pass of scripts/analysis/posthoc_paper_width.sh ABS=1). One value per trial, the mean over the last
five sub-task checkpoints, on the centroid (the policy for PPO, TRAC, ReDo and C-CHAIN; the weight
average of the population for PBT). A method's width relative to PPO is PPO's
action change over the method's: above 1 = a wider basin than PPO.

Solved checkpoints only (ReLU panels, since 2026-09-21). A basin is the region
of high return around a SOLUTION, and the action change stands in for it only
while the policy solves its sub-task. A failed policy can read as very wide --
TRAC-PPO's MountainCar checkpoints that push right in every state change no
action under any noise, on a plateau where every return is -500 -- so a
checkpoint counts only when its return on the sub-task it was trained on
(evaluation.json, the centroid) reaches LEARNED (half) of the panel's best gain
above FLOOR, the generalist figure's `learned` rule. `best` is the highest mean
post-switch return of any method on the panel. The window is the last five
checkpoints BY POSITION, unsolved ones left out; a trial with none solved there
drops out of that panel. The tanh panels (tab:basin_noise) are unmasked.

Table (appendix): rows methods, columns panels: the ratio (trial means), marked
where the method differs from PPO over trials (two-sided Mann-Whitney, Holm
over the panels of the row). Summary (main text): the same ratios as one point
per panel, with the median over panels and a two-sided Wilcoxon signed-rank
test of log ratio = 0 over panels. The markdown also counts the panels where
PPO has the narrowest basin of all methods, and repeats the summary at the
other two radii.
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
from matplotlib.colors import LinearSegmentedColormap      # noqa: E402
from matplotlib.lines import Line2D                        # noqa: E402
from matplotlib.transforms import blended_transform_factory  # noqa: E402
from scipy.stats import mannwhitneyu, wilcoxon             # noqa: E402

REPO = pathlib.Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO / 'scripts'))
sys.path.insert(0, str(REPO / 'scripts/analysis'))
from make_lineplot import METHOD_STYLE                     # noqa: E402
import plot_generalist_outcomes as pgo                     # noqa: E402
import plot_generalist_scores as pgs                       # noqa: E402
import plot_landscape_main as plm                          # noqa: E402
from plot_stability_plasticity import FLOOR                 # noqa: E402
from plot_landscape_slices import LEARNED                   # noqa: E402
from plot_shared_basin import FIG5                          # noqa: E402

PROJECT = pgs.PROJECT
TABLE = plm.OUT / 'basin_width_table'
MAIN = pgs.FINAL / 'basin_width_main'
DATA = pgs.FINAL / 'data' / 'basin_width_main.json'
METHODS = pgo.ARMS                  # ga es ppo trac redo cchain pbt
# The paper's one method order (make_lineplot.METHOD_ORDER, as Figure 2) since
# 2026-09-21; the table used to put ES first. PPO is the reference, not a column.
from make_lineplot import METHOD_ORDER                     # noqa: E402
TABLE_ORDER = [m for m in METHOD_ORDER if m in ('es', 'ga', 'trac', 'redo', 'cchain', 'pbt')]
REFERENCE = 'ppo'
RADII = (0.03, 0.1, 0.3)
INK, MUTED = plm.INK, plm.MUTED
# Figure 2's panels that plot_generalist_scores.GRID lacks: (data root, cell,
# body, change). The extract checks these are exactly that difference.
SEQUENCES = [
    ('paper/gymnax/data/noise_10task', 'CartPole_v1_sigma1.0', 'CartPole', '10 noise tasks'),
    ('paper/gymnax/data/noise_10task', 'Acrobot_v1_sigma1.0', 'Acrobot', '10 noise tasks'),
    ('paper/gymnax/data/noise_10task', 'MountainCar_v0_sigma0.1', 'MountainCar', '10 noise tasks'),
    ('paper/mjx/cheetah/data/noise_10task', 'cheetah_noise', 'HalfCheetah', '10 noise tasks'),
    ('paper/kinetix/data', 'Kinetix20', 'Kinetix', '20 levels'),
]
MARKER = {**plm.PERT_MARKER, 'Sequences': 'D'}
# Data roots whose policy is tanh-hidden (envs/mjx.py ContinuousMLPPolicy,
# POLICY_ARCH['kinetix']): absolute noise, read from the root's own pass.
ABSOLUTE = ('paper/mjx/', 'paper/kinetix/')
# data root -> the paper directory whose results/centroid/curvature_width.json
# scored its runs: finish_iclr.sh's, or the root's own
# (scripts/analysis/posthoc_paper_{cheetah,width}.sh).
WIDTH_PASS = {
    'paper/gymnax/data/noise_2task': 'paper/gymnax/noise/2task',
    'paper/gymnax/data/physics_2task': 'paper/gymnax/data/physics_2task',
    'paper/gymnax/data/actions_2task': 'paper/gymnax/actions/2task',
    'paper/minigrid/data': 'paper/minigrid/minigrid',
    'paper/mjx/cheetah/data/noise_2task': 'paper/mjx/cheetah/data/noise_2task',
    'paper/mjx/cheetah/data/physics_2task': 'paper/mjx/cheetah/physics/2task',
    'paper/mjx/cheetah/data/actions_2task': 'paper/mjx/cheetah/data/actions_2task',
    'paper/gymnax/data/noise_10task': 'paper/gymnax/noise/10task',
}
# Diverging, neutral midpoint: purple = wider than PPO, orange = narrower. Kept
# off the method palette's red/green so a cell is not read as a method.
CMAP = LinearSegmentedColormap.from_list(
    'width_ratio', ['#b35806', '#f1a340', '#fee0b6', '#f2f1ee', '#d8daeb', '#998ec3', '#542788'])
LIM = 4.0                           # colour range, log2 units (16x either way)


def panel_list():
    """[(group, body, change, data root, cell)]: the generalist figure's order,
    then Figure 2's sequences."""
    return ([(group, pgo.COLUMNS[c], p[2], p[0], p[1])
             for group, row in pgs.GRID for c, p in enumerate(row) if p]
            + [('Sequences', body, change, tree, cell)
               for tree, cell, body, change in SEQUENCES])


def is_absolute(tree):
    return tree.startswith(ABSOLUTE)


def pass_path(tree):
    if is_absolute(tree):
        return PROJECT / tree / 'results/centroid_abswidth/curvature_width.json'
    return PROJECT / WIDTH_PASS[tree] / 'results/centroid/curvature_width.json'


def figure2_arms():
    """{(data root, cell): {row: (arm, trial count)}} as continual_combined
    plots them (plot_continual_lineplots' data file)."""
    import plot_continual_lineplots as pcl
    blob = json.loads(pcl.DATA.with_suffix('.json').read_text())
    curves = np.load(pcl.DATA.with_suffix('.npz'))
    grid = {(p[0], p[1]) for _, row in pgs.GRID for p in row if p}
    main = [(tree, cell) for g, tree, cell, _ in pcl.PANELS if g == 'main']
    assert [(t, c) for t, c in main if (t, c) not in grid] == [(t, c) for t, c, *_ in SEQUENCES], \
        'SEQUENCES is not the set of Figure 2 panels the generalist grid lacks'
    out = {}
    for tree, cell, *_ in SEQUENCES:
        kept = blob['kept'][tree]
        rows = {}
        for row in METHODS:
            # the kept one of es/nes for the ES row, of pbt/pbt2 for the PBT row
            arm = next((k for k in kept if k.rstrip('2') == row or {row, k} <= {'es', 'nes'}), row)
            key = f'{tree}|{cell}|{row}|curves'      # the npz names the kept arm by its row
            if key in curves.files:
                rows[row] = (arm, curves[key].shape[0])
        out[(tree, cell)] = rows
    return out


def panel_trials(relu_only=False):
    """Yield (label, tree, cell, key, {row: (arm, {trial: record})}) for every
    panel: the width-pass records of the runs the generalist figure (or, for
    SEQUENCES, Figure 2) shows, with the same checks. `key` is the pass's width
    column, 'width_{}' or 'width_abs_{}' (tanh panels). `relu_only` skips the
    tanh panels before their checks (the ladder analyses never read them)."""
    scores = json.loads(pgs.DATA.read_text())
    fig2 = figure2_arms()
    for _, body, change, tree, cell in panel_list():
        if relu_only and is_absolute(tree):
            continue
        path = pass_path(tree)
        blob = json.loads(path.read_text())
        pass_root = (REPO / blob['meta']['root']).resolve()
        key = 'width_abs_{}' if is_absolute(tree) else 'width_{}'
        if (tree, cell) in fig2:
            arms = {row: (arm, None, n) for row, (arm, n) in fig2[(tree, cell)].items()}
        else:
            arms = {}
            for row, v in scores['panels'][f'{tree}|{cell}']['arms'].items():
                shown = {(PROJECT / t['run']).parent.resolve() for t in v['trials']}
                now = (PROJECT / tree / 'continual' / v['arm'] / cell).resolve()
                if shown == {now}:
                    arms[row] = (v['arm'], {pathlib.Path(t['run']).name for t in v['trials']}, None)
                else:   # the data root was relinked after the generalist figure's extract
                    print(f'WARNING: generalist_scores_centroid shows {shown} for '
                          f'{v["arm"]}/{cell}, the data root now links {now}; '
                          f'following the data root -- re-extract that figure')
                    arms[row] = (v['arm'], None, None)
        panel = {}
        for row, (arm, want, n) in arms.items():
            scored = (pass_root / 'continual' / arm / cell).resolve()
            runs = (PROJECT / tree / 'continual' / arm / cell).resolve()
            assert runs == scored, f'{path}: {arm}/{cell} scored {scored}, figure shows {runs}'
            trials = blob['cells'].get(cell, {}).get(arm, {}).get('trials', {})
            if want is not None:
                assert set(trials) == want, (f'{path}: {arm}/{cell} has trials '
                                             f'{sorted(set(trials) ^ want)} on one side only')
            elif n is not None:
                assert len(trials) == n, f'{path}: {arm}/{cell} has {len(trials)} trials, Figure 2 {n}'
            panel[row] = (arm, {t: trials[t] for t in sorted(trials)})
        solved = None if is_absolute(tree) else solved_masks(tree, cell, panel)
        panel = {row: (arm, trials, solved[row] if solved else None)
                 for row, (arm, trials) in panel.items()}
        yield f'{body}, {change}', tree, cell, key, panel, str(path.relative_to(PROJECT)), scores


def solved_masks(tree, cell, panel):
    """{row: {trial: [bool per checkpoint]}}: the checkpoint solves the sub-task
    it was trained on (see the docstring)."""
    shown = {}
    for row, (arm, trials) in panel.items():
        shown[row] = {}
        for t in trials:
            got = pgs.from_evaluation(PROJECT / tree / 'continual' / arm / cell / t)
            if got is None:
                sys.exit(f'{tree}/continual/{arm}/{cell}/{t} has no evaluation.json')
            shown[row][t] = np.asarray(got[0], dtype=float)
    floor = FLOOR[cell.split('_sigma')[0]]
    best = max(np.mean([v[1:].mean() for v in by.values()]) for by in shown.values())
    return {row: {t: ((v - floor) / (best - floor) >= LEARNED).tolist() for t, v in by.items()}
            for row, by in shown.items()}


def window(values, solved, k=5):
    """Mean over the last k checkpoints by position, unsolved ones left out;
    None when none is left."""
    a = np.array([np.nan if x is None else float(x) for x in values], dtype=float)[-k:]
    if solved is not None:
        a = np.where(np.asarray(solved, bool)[-k:], a, np.nan)
    a = a[np.isfinite(a)]
    return float(a.mean()) if a.size else None


def extract():
    """Per panel and row arm, the per-trial late action change at every radius,
    checked against the generalist figure's runs."""
    out = {'panels': {}, 'passes': {}, 'absolute': [],
           'generalist_scores_extracted': None,
           'extracted': datetime.date.today().isoformat()}
    for label, tree, cell, key, arms, rel, scores in panel_trials():
        out['generalist_scores_extracted'] = scores['extracted']
        out['passes'][tree] = rel
        panel = {row: {'arm': arm, **{
            f'width_{e}': [window(rec[key.format(e)], solved[t] if solved else None)
                           for t, rec in trials.items()]
            for e in RADII}, 'unsolved_late': None if solved is None else int(sum(
                5 - sum(solved[t][-5:]) for t in trials))}
            for row, (arm, trials, solved) in arms.items()}
        out['panels'][label] = panel
        if is_absolute(tree):
            out['absolute'].append(label)
        print(f'{label:34s} {rel:62s} '
              + ' '.join(f'{r}={len(v["width_0.1"])}' for r, v in panel.items()))
    DATA.parent.mkdir(parents=True, exist_ok=True)
    DATA.write_text(json.dumps(out, indent=1) + '\n')
    print(f'wrote {DATA}')


def widths(eps):
    """{panel label: {method: np.array(per-trial late action change)}}."""
    blob = json.loads(DATA.read_text())
    out = {}
    for label, panel in blob['panels'].items():
        for m, v in panel.items():
            vals = np.array([x for x in v[f'width_{eps}'] if x is not None and np.isfinite(x)])
            if vals.size:
                out.setdefault(label, {})[m] = vals
    return out


def holm(ps):
    """Holm-adjusted p-values, same order."""
    order = np.argsort(ps)
    adj = np.empty(len(ps))
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(ps) - rank) * ps[i]))
        adj[i] = running
    return adj


def ratios(data):
    """{method: {panel: (log2 width ratio vs PPO, p vs PPO)}}; Holm over each row."""
    out = {}
    for m in METHODS:
        if m == REFERENCE:
            continue
        cells = {p: d for p, d in data.items() if m in d and REFERENCE in d}
        ps = {p: mannwhitneyu(d[m], d[REFERENCE]).pvalue for p, d in cells.items()}
        adj = dict(zip(ps, holm(np.array(list(ps.values()))))) if ps else {}
        out[m] = {p: (float(np.log2(d[REFERENCE].mean() / d[m].mean())), adj[p])
                  for p, d in cells.items()}
    return out


def stars(p):
    return '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''


def across(values):
    """(median log2 ratio, Wilcoxon p, n) over panels."""
    v = np.array(values)
    if v.size < 5 or np.allclose(v, 0):
        return float(np.median(v)) if v.size else np.nan, np.nan, v.size
    return float(np.median(v)), float(wilcoxon(v).pvalue), v.size


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--extract', action='store_true',
                    help='read the width passes and write the data file first')
    ap.add_argument('--eps', type=float, default=0.1, choices=RADII)
    ap.add_argument('--font-size', type=float, default=7.0,
                    help='the size in print; the figures are drawn wider than they print '
                         'and scale their text by PRINT_SCALE')
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
    panels = panel_list()
    labels = [f'{b}, {c}' for _, b, c, _, _ in panels]
    all_data = {eps: widths(eps) for eps in RADII}
    rat = {eps: ratios(d) for eps, d in all_data.items()}
    R = rat[args.eps]
    rows = [m for m in METHODS if m != REFERENCE]
    absolute = set(json.loads(DATA.read_text()).get('absolute', []))

    sfx = '' if args.eps == 0.1 else f'_eps{args.eps:g}'
    table = TABLE.with_name(TABLE.name + sfx)
    summary = MAIN.with_name(MAIN.name + sfx)
    # Both figures show the ReLU panels only: relative width is invariant to
    # weight scale there; the tanh panels are the paper's tab:basin_noise.
    relu = {eps: {p: v for p, v in d.items() if p not in absolute} for eps, d in all_data.items()}
    relu_rat = {eps: ratios(d) for eps, d in relu.items()}
    relu_panels = [p for p in panels if f'{p[1]}, {p[2]}' not in absolute]
    draw_table(table, relu_panels, relu_rat[args.eps], [m for m in TABLE_ORDER if m in rows],
               args.eps, fs)
    draw_summary(summary, relu_panels, relu_rat[args.eps], rows, args.eps, fs)
    write_markdown(summary, labels, R, relu, relu_rat, args.eps, absolute)
    return 0


# Printed widths (in): the summary goes in the left minipage of fig:basin
# (FIG5['left'] \linewidth) and is drawn wider, scaling its text to match; the
# table is drawn at \linewidth.
PRINT_W = {'summary': FIG5['left'] * FIG5['linewidth'], 'table': 5.5}


def scale_fonts(fs, drawn_w, printed_w):
    """The drawing's font size for `fs` in print, rcParams set to match."""
    fs = fs * drawn_w / printed_w
    plt.rcParams.update({'font.size': fs, 'axes.titlesize': fs,
                         'xtick.labelsize': fs - 1, 'ytick.labelsize': fs})
    return fs


def save(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    for ext in ('pdf', 'png'):
        fig.savefig(path.with_suffix(f'.{ext}'), dpi=300)
    plt.close(fig)
    print(f'wrote {path}.pdf/.png ({fig.get_figwidth():.2f} x {fig.get_figheight():.2f} in)')


def draw_table(path, panels, R, rows, eps, fs):
    """Appendix: rows panels (grouped by what the switch changes), columns
    methods, the ratio in each cell. Panels no method has a width for are
    left out. Drawn at print size (\\linewidth)."""
    panels = [p for p in panels if any(f'{p[1]}, {p[2]}' in R[m] for m in rows)]
    groups = [g for g, *_ in panels]
    ys, y = [], 0.0
    for i, g in enumerate(groups):
        y += 1.2 if i == 0 or g != groups[i - 1] else 0   # room for the group label
        ys.append(y)
        y += 1
    cell_w, cell_h = 0.62, 0.17                         # inches
    left, top, bottom = 1.75, 0.35, 0.55
    W = PRINT_W['table']
    H = top + bottom + cell_h * (ys[-1] + 1)
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([left / W, bottom / H, cell_w * len(rows) / W, 1 - (top + bottom) / H])
    for c, m in enumerate(rows):
        for r, (_, b, ch, _, _) in enumerate(panels):
            val = R[m].get(f'{b}, {ch}')
            if val is None:
                ax.add_patch(plt.Rectangle((c - 0.5, ys[r] - 0.5), 1, 1, fc='white',
                                           ec='0.85', lw=0.4, hatch='////', zorder=1))
                continue
            lr, pv = val
            ax.add_patch(plt.Rectangle((c - 0.5, ys[r] - 0.5), 1, 1,
                                       fc=CMAP(0.5 + np.clip(lr, -LIM, LIM) / (2 * LIM)),
                                       ec='white', lw=0.8, zorder=1))
            ink = 'white' if abs(lr) > 2.3 else INK
            mark = stars(pv)
            ax.text(c, ys[r], f'{2 ** lr:.2g}' + (f' {mark}' if mark else ''),
                    ha='center', va='center', fontsize=fs - 1, color=ink, zorder=2)
    ax.set_xlim(-0.5, len(rows) - 0.5)
    ax.set_ylim(ys[-1] + 0.5, -0.5)
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels(['ES' if m == 'es' else METHOD_STYLE[m]['label'] for m in rows],
                       fontsize=fs - 0.5)
    ax.xaxis.tick_top()
    ax.axvline(1.5, color=INK, lw=0.6)                  # NE left, RL right
    ax.set_yticks(ys)
    ax.set_yticklabels([f'{b}, {ch}' for _, b, ch, _, _ in panels], fontsize=fs - 0.5)
    ax.tick_params(length=0, pad=2)
    for sp in ax.spines.values():
        sp.set_visible(False)
    for g in dict.fromkeys(groups):
        first = ys[groups.index(g)]
        ax.text(-0.5, first - 0.75, g, ha='left', va='center',
                fontsize=fs, transform=ax.transData, clip_on=False)
    cax = fig.add_axes([left / W, 0.3 / H, 1.1 / W, 0.07 / H])
    cax.imshow(np.linspace(-LIM, LIM, 128)[None, :], aspect='auto', cmap=CMAP,
               vmin=-LIM, vmax=LIM, extent=[-LIM, LIM, 0, 1])
    cax.set_yticks([])
    cax.set_xticks([-LIM, 0, LIM])
    cax.set_xticklabels([f'×1/{2 ** LIM:g}', '×1', f'×{2 ** LIM:g}'], fontsize=fs - 1)
    cax.tick_params(length=1.5, pad=1)
    for sp in cax.spines.values():
        sp.set_linewidth(0.4)
    fig.text((left + 1.3) / W, 0.335 / H, 'Basin width relative to PPO (purple = wider)',
             fontsize=fs - 0.5, va='center')
    save(fig, path)


def draw_summary(path, panels, R, rows, eps, fs, xlabel='Basin width relative to PPO (×)',
                 ref_label='PPO'):
    """Main text: one point per panel per method, the median, Wilcoxon over panels."""
    group_of = {f'{b}, {c}': g for g, b, c, _, _ in panels}
    W = 3.4
    k = W / PRINT_W['summary']          # drawn inches per printed inch
    H = FIG5['height'] * k              # the vertical layout shared with shared_basin_main
    fs = scale_fonts(fs, W, PRINT_W['summary'])
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0.9 / W, FIG5['ax_bottom'] * k / H, (W - 1.6) / W,
                       1 - (FIG5['ax_bottom'] + FIG5['ax_top']) * k / H])
    rng = np.random.default_rng(0)
    for r, m in enumerate(rows):
        col = METHOD_STYLE[m]['color']
        for p, (lr, pv) in R[m].items():
            ax.scatter([2 ** lr], [r + rng.uniform(-0.2, 0.2)], s=12,
                       marker=MARKER[group_of[p]],
                       color=col if stars(pv) else 'white', edgecolors=col,
                       linewidths=0.6, zorder=3)
        med, pw, n = across([v[0] for v in R[m].values()])
        ax.plot([2 ** med] * 2, [r - 0.34, r + 0.34], color=INK, lw=1.2, zorder=4)
        mark = stars(pw) if np.isfinite(pw) else ''
        ax.text(1.03, r, f'×{2 ** med:.2g}{" " + mark if mark else ""}',
                transform=ax.get_yaxis_transform(), ha='left', va='center', fontsize=fs)
    ax.set_xscale('log')
    ax.axvline(1, color='0.45', lw=0.7, zorder=1)
    ax.text(1, -0.75, ref_label, ha='center', va='bottom', fontsize=fs - 0.5, color='0.35')
    ax.set_ylim(len(rows) - 0.5, -0.5)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(['ES' if m == 'es' else METHOD_STYLE[m]['label'] for m in rows])
    ax.tick_params(axis='y', length=0)
    ax.axhline(1.5, color='0.7', lw=0.5, ls=':')
    low = min([0.7] + [0.9 * 2 ** lr for m in rows for lr, _ in R[m].values()])
    ax.set_xlim(low, 2 ** (LIM + 0.3))
    ticks = [t for t in [2.0 ** k for k in range(-2, int(LIM) + 1)] if t >= low]
    ax.set_xticks(ticks)
    ax.set_xticklabels([{0.25: '¼', 0.5: '½'}.get(t, f'{t:g}') for t in ticks])
    ax.minorticks_off()
    ax.spines[['right', 'top', 'left']].set_visible(False)
    ax.grid(axis='x', color='0.92', lw=0.4, zorder=0)
    ax.set_xlabel(xlabel, fontsize=fs)
    ax.xaxis.set_label_coords(0.5, FIG5['xlabel_top'] * k, transform=blended_transform_factory(
        ax.transAxes, fig.dpi_scale_trans))
    fig.legend([Line2D([], [], ls='', marker=mk, ms=3.2, mfc='0.8', mec=INK, mew=0.3)
                for mk in MARKER.values()], list(MARKER),
               loc='lower center', bbox_to_anchor=(0.5, 0.0), ncol=2, frameon=False,
               fontsize=fs - 0.5 * k, handletextpad=0.2, columnspacing=1.0)  # fs - 0.5 in print
    save(fig, path)


def write_markdown(path, labels, R_all, all_data, rat, eps, absolute):
    """Per-panel ratios for every panel (R_all); the rest for the summary's
    panels (all_data, rat: the ReLU ones)."""
    lab = lambda m: 'ES' if m == 'es' else METHOD_STYLE[m]['label']                       # noqa: E731
    data, R = all_data[eps], rat[eps]
    md = [f'# {path.name} (and {TABLE.name})', '', 'See the docstring of `scripts/analysis/plot_basin_width_methods.py`. '
          'Ratio = PPO action change / method action change (above 1 = wider than PPO). '
          'Radius = relative (times each tensor\'s norm) on the ReLU panels, absolute noise '
          f's.d. on the tanh panels ({", ".join(sorted(absolute)) or "none"}).', '',
          f'## Radius {eps:g}: ratio per panel (Holm p vs PPO), every panel', '',
          '| Method | ' + ' | '.join(labels) + ' |', '|---|' + '---|' * len(labels)]
    for m, cells in R_all.items():
        md.append(f'| {lab(m)} | ' + ' | '.join(
            f'{2 ** cells[p][0]:.2f} ({cells[p][1]:.2g})' if p in cells else '-'
            for p in labels) + ' |')
    md += ['', f'Everything below is over the {len(data)} ReLU panels of the main-text '
           'summary; the tanh panels are in the table only.']
    md += ['', '## PPO narrowest?', '',
           'Panels where PPO has the largest mean action change (narrowest basin) of all '
           'methods present, per radius.', '']
    for e, d in all_data.items():
        narrowest = [p for p, v in d.items() if REFERENCE in v and
                     max(v, key=lambda m: v[m].mean()) == REFERENCE]
        rl_only = [p for p, v in d.items() if REFERENCE in v and
                   max((m for m in v if m not in ('ga', 'es')),
                       key=lambda m: v[m].mean()) == REFERENCE]
        md.append(f'- radius {e:g}: narrowest of all in {len(narrowest)}/{len(d)} panels '
                  f'({", ".join(narrowest) or "none"}); narrowest among RL methods in '
                  f'{len(rl_only)}/{len(d)}.')
    md += ['', '## Median ratio over panels (Wilcoxon signed-rank on log ratio)', '',
           '| Method | ' + ' | '.join(f'radius {e:g}' for e in RADII) + ' |',
           '|---|' + '---|' * len(RADII)]
    for m in R:
        cells = []
        for e in RADII:
            med, p, n = across([v[0] for v in rat[e][m].values()])
            cells.append(f'×{2 ** med:.2f}, p = {p:.3g}, n = {n}' if np.isfinite(p)
                         else f'×{2 ** med:.2f}, n = {n}')
        md.append(f'| {lab(m)} | ' + ' | '.join(cells) + ' |')
    md += ['', '## Per-panel significant differences from PPO (Holm p < 0.05), radius '
           f'{eps:g}', '']
    for m, cells in R.items():
        wider = [p for p, (lr, pv) in cells.items() if pv < 0.05 and lr > 0]
        narrower = [p for p, (lr, pv) in cells.items() if pv < 0.05 and lr < 0]
        md.append(f'- {lab(m)}: wider on {len(wider)}/{len(cells)} '
                  f'({", ".join(wider) or "none"}); narrower on {len(narrower)} '
                  f'({", ".join(narrower) or "none"}).')
    md += ['', 'Caveat: the GA, ES and PBT values are for the population centroid (the '
           'agent the figure scores); PBT\'s is the weight average of its PPO '
           'members, not one PPO network.']
    # Pairs of RL variants on the panels both have (e.g. whether PBT widens more than TRAC).
    md += ['', '## RL variants against each other (panels both have)', '',
           '| Pair | radius | panels | median ratio A | median ratio B | median log2(B/A) | '
           'Wilcoxon p |', '|---|---|---|---|---|---|---|']
    rl = [m for m in rat[eps] if m not in ('ga', 'es')]
    for i, a in enumerate(rl):
        for b in rl[i + 1:]:
            for e in RADII:
                common = [p for p in rat[e][a] if p in rat[e][b]]
                if len(common) < 5:
                    continue
                la = np.array([rat[e][a][p][0] for p in common])
                lb = np.array([rat[e][b][p][0] for p in common])
                d = lb - la
                pw = wilcoxon(d).pvalue if not np.allclose(d, 0) else np.nan
                md.append(f'| {lab(a)} vs {lab(b)} | {e:g} | {len(common)} '
                          f'| ×{2 ** np.median(la):.2f} | ×{2 ** np.median(lb):.2f} '
                          f'| {np.median(d):+.2f} | {pw:.3g} |')
    path.with_suffix('.md').write_text('\n'.join(md) + '\n')
    print(f'wrote {path}.md')


if __name__ == '__main__':
    sys.exit(main())
