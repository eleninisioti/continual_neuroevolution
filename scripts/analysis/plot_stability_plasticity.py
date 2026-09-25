"""The stability-plasticity trade-off of every continual family the paper
reports, in the style of the other paper visuals: can a method learn each new
sub-task, and does it keep what it learnt?

    .venv/bin/python scripts/analysis/plot_stability_plasticity.py             # from the saved data
    .venv/bin/python scripts/analysis/plot_stability_plasticity.py --extract   # re-read the runs first
        [--stability BD] [--plasticity cum]                                    # the variants

    -> paper/visuals/final/stability_plasticity.{pdf,png}   one panel a task, continual_main's ten
       paper/visuals/final/stability_plasticity.tex         one row a task
       paper/visuals/final/stability_plasticity.md          every number, with definitions
       paper/visuals/final/data/stability_plasticity.json   per-trial LA, F, BD and Cum. it is drawn from
       paper/visuals/stability_plasticity{_bd,_cum}.*       the variants (not in the paper)
    (paper = projects/iclr_2027/paper)

Built in two steps, as plot_continual_lineplots.py's main figure: `--extract`
reads the runs through the symlink trees under paper/<suite>/data (the same
panels, trees and arms as continual_main) and saves one value a trial;
without it only the saved data is read.

The axes are the continual-learning literature's decomposition, both read for
the CENTROID from the family's forgetting pass (`results/centroid/
behavioural_divergence.{json,npz}`), whose reward matrix R[i][j] is the agent
saved at the end of phase j on the sub-task of phase i:

    Plasticity  learning accuracy, LA = mean_i R[i][i]: how well each sub-task
                was learnt by the end of its own phase (Chaudhry et al. 2018's
                intransigence against a fixed reference; Mirzadeh et al. 2022
                and Jung et al. 2023 report it as plasticity).
    Stability   forgetting, F, as every other figure reads it
                (`make_lineplot.load_divergence`): the final agent's drop on
                every earlier sub-task, or with two alternating sub-tasks the
                drop at every switch (Chaudhry et al. 2018; Lopez-Paz & Ranzato
                2017; Wolczyk et al. 2021).

A method that never learns cannot forget, so neither axis means anything
alone. For the same reason a FROZEN specialist -- a method that, with two
alternating sub-tasks, only ever solved one of them (the generalist figure's
class, `plot_generalist_scores.classify`) -- is not eligible for the ring:
its F is ~0 because it never learnt the other sub-task, not because it kept
it, and its LA - F would otherwise beat every method that learns both. Its
point is still drawn and it still counts as a rival. Cum. is not an axis: it already integrates both. Both axes are rescaled
per environment the way rliable does (Agarwal et al. 2021),

    LA -> (LA - untrained) / (best - untrained)      F -> F / (best - untrained)

with `untrained` the return of the untrained network (FLOOR) and `best` the
highest mean LA any reported method reaches in that family and environment.
The same line is applied to every method, so no comparison inside a panel
changes. With both axes on one scale, LA - F is the performance a sub-task
keeps: at the end of the sequence with more than two sub-tasks (the
final-average-performance identity, up to (T-1)/T), right after the next
switch with two. The grey lines are equal LA - F; better is up and right.

A panel is one task -- the ten of the main continual figure, gymnax's three
environments drawn apart rather than pooled -- and draws every method's mean
over seeds with a 95% percentile-bootstrap interval. The grey region lies beyond
some RL method's interval on both axes (lower LA than its lower bound, more F
than its upper bound); an NE point outside it is not clearly beaten on both
axes by any RL method. The `Dominated by` column tests the same claim: RL
methods better on LA and on F, each by a one-sided Mann-Whitney U test over
seeds (p < 0.05). Before 2026-09-24 the region and the column used the means. The mean, not rliable's IQM, as in every other
figure of the paper (make_lineplot.bootstrap_ci says why). The tables add
rliable's probability of improvement on LA - F.

Arms are continual_main's (plot_continual_lineplots.py): superseded arms
dropped, an arm left out while any of its trials still trains, ES = NES except
on Kinetix (OpenES, `es`, plain OpenES since 2026-09-24), and PBT N=8 or N=2 by the higher Cum.
elite (es_arm.py).

gymnax and MiniGrid read the reward matrix from the home forgetting pass
(PASS). Kinetix and HalfCheetah read it from the training records instead
(`load_records`), which log the centroid on every sub-task: Kinetix's pass runs
on the GH200 only (CLAUDE.md), and the records gave its LA and F exactly on
all 45 runs it scored (2026-09-15); the cheetah's RL arms (and its noise GA)
are the CLUSTER ant-PPO-shape runs, which the home pass never scored, so every
cheetah arm is read the same way. On the cheetah NES runs both sources score,
mean LA agrees to 0.1% and a run's F to within 0.07 of the rescaled axis (MJX
evaluation noise, 2026-09-17). Their BD, where present, is a CLUSTER centroid
pass over three trials an arm (BD_PASS).

`--stability BD` puts behavioural divergence at the switch on the y axis
instead (already in [0, 1], not rescaled, no equal-performance lines), written
to `stability_plasticity_bd.*`.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import pathlib
import sys

import numpy as np
from scipy.stats import mannwhitneyu
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt                            # noqa: E402
from matplotlib.gridspec import GridSpec                   # noqa: E402
from matplotlib.lines import Line2D                        # noqa: E402
from matplotlib.legend_handler import HandlerTuple         # noqa: E402
from matplotlib.patches import Patch                       # noqa: E402
from matplotlib.ticker import MaxNLocator                  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import plot_continual_lineplots as pcl                     # noqa: E402
from source.metrics.continual_metrics import mann_whitney_tests  # noqa: E402

ncs, lp, PROJECT = pcl.ncs, pcl.lp, pcl.PROJECT
es_arm, REPO = ncs.es_arm, ncs.REPO
FINAL = pcl.FINAL
STEM = 'stability_plasticity'
DATA = FINAL / 'data' / f'{STEM}.json'
VARIANTS = PROJECT / 'paper/visuals' / STEM          # --stability BD / --plasticity cum

GYMNAX_ENVS = ('CartPole_v1', 'Acrobot_v1', 'MountainCar_v0')
# The panels are continual_main's (plot_continual_lineplots.PANELS, `main`),
# in its order. A tree listed here reads the forgetting pass of that paper
# directory (a finish_iclr.sh family); the others read their training records.
PASS = {'paper/gymnax/data/noise_10task': 'gymnax/noise/10task',
        'paper/gymnax/data/actions_2task': 'gymnax/actions/2task',
        'paper/gymnax/data/noise_2task': 'gymnax/noise/2task',
        # The family pass ran on runs_param_2task, where es_arm kept OpenES and
        # PBT does not exist (the paper tree links it per cell from CLUSTER):
        # posthoc_paper_forgetting.sh scores the paper tree's arms instead.
        'paper/gymnax/data/physics_2task': 'gymnax/data/physics_2task',
        'paper/minigrid/data': 'minigrid/minigrid'}
# The panels beside continual_main's that Figure 2 draws (plot_continual_combined.py
# --part tradeoff): the two-sub-task noise and physics families.
FIGS = ('main', 'tradeoff')
# Records-read trees whose BD comes from a centroid pass run on CLUSTER over
# three trials an arm (scripts/train/cluster/submit_bd_posthoc.sh), shipped
# into paper/<sub>/results/centroid_bd; LA and F stay the records'.
BD_PASS = {'paper/mjx/cheetah/data/noise_10task': 'mjx/cheetah',
           'paper/mjx/cheetah/data/actions_2task': 'mjx/cheetah',
           'paper/kinetix/data': 'kinetix/kinetix'}
# The untrained network's return, the zero of the plasticity axis: the NE
# arms' generation-0 centroid in these trees (CartPole 9.2-10.0; Acrobot and
# MountainCar never reach the goal in the 500-step episode; MiniGrid never
# reaches it either; the cheetah's 604-726 brackets its do-nothing floor, 677;
# every Kinetix arm's first record is -1).
FLOOR = {'CartPole_v1': 10.0, 'Acrobot_v1': -500.0, 'MountainCar_v0': -500.0,
         'MiniGrid_8x8_16x16': 0.0,
         'cheetah_noise': 677.0, 'cheetah_friction': 677.0, 'cheetah_action': 677.0,
         'Kinetix20': -1.0}
# continual_main's grid: the noise row over the action-reversal row, MiniGrid
# and Kinetix in the last column. Each panel's title names its task, so the
# figure needs no suite headers.
NCOLS = 5
# CLAUDE.md (2026-09-16): ES and NES are ONE method, reported as ES -- NES
# everywhere but Kinetix, which keeps plain OpenES. One label, one colour and one
# legend entry here, whichever of the two a family kept.
ES_LABEL, ES_COLOUR = 'ES', lp.METHOD_STYLE['es']['color']
N_BOOT = 2000
ISO = '0.86'
DOMINATED = '0.93'
BEST = '0.35'          # the ring and iso line of the best LA - F in a panel
# A panel never spans less than this on either axis (rescaled units): without
# it a setting every method solves (Acrobot physics: LA within 0.03, F within
# 0.05) is zoomed until noise-sized gaps fill the panel. Extra room is added
# on both sides of the data.
MIN_SPAN = 0.5
# The ring needs a lead of at least this much LA - F (rescaled units) over
# every method of the other family, besides the Holm-corrected test: with
# near-deterministic seeds a 0.01 lead passes the test (C-CHAIN, MountainCar
# physics, p 0.045) yet is no result.
MIN_EFFECT = 0.05


def reported_arms(root, cells, es_kept=None):
    """The arms plot_continual_lineplots.py draws for `cells` of one tree;
    `es_kept` fixes the ES variant as its main figure does."""
    cell_filter = lp.cell_selector(cells, None)[0]
    arms = [m for m in pcl.complete_arms(root, cells)
            if m not in lp.SUPERSEDED or lp.defect_state(
                root, 'continual', m, cell_filter, lp.SUPERSEDED[m][0]) == 'none']
    pairs = (es_arm.ARMS, es_arm.PBT_ARMS)
    cum = es_arm.load(root, 'continual', cells, arms=sum(pairs, ()))
    for pair in pairs:
        if es_kept and pair == es_arm.ARMS:
            assert es_kept in arms, f'{es_kept} has no finished runs under {root}'
            arms = [m for m in arms if m not in pair or m == es_kept]
        elif set(pair) <= set(arms):
            kept = es_arm.pick(cum, pair)[0]
            arms = [m for m in arms if m not in pair or m == kept]
    return arms


def curve_mean(trial_dir):
    """The centroid curve's mean over the whole run: the paper's Cum. centroid
    up to the constant (generations / 1000), because `make_lineplot.collect`
    spaces every method's records evenly across its own budget whatever its
    logging stride was. On the same scale as LA, so it can replace it on the x
    axis (`--plasticity cum`) and stay comparable with F."""
    records = json.loads((trial_dir / 'training_metrics.json').read_text())
    key = lp.resolve_column('centroid', 'generalist', records[0], lp.METRIC_COLUMNS)
    if key is None:
        return np.nan
    series = np.array([float(r[key]) for r in records], dtype=float)
    return float(np.trapz(series, dx=1) / max(len(series) - 1, 1))


def _trial(la, forgetting, bd, cum, run):
    return {'la': la, 'F': forgetting, 'BD': bd, 'cum': cum, 'run': run}


def load_pass(sub, tree, cells, arms):
    """`{cell: {method: [trial]}}` from the centroid forgetting pass of paper
    directory `sub`, a run matched to an arm through the arm's symlink under
    `tree`; `{}` if the pass has not run."""
    results = PROJECT / 'paper' / sub / 'results/centroid'
    npz = results / 'behavioural_divergence.npz'
    if not npz.exists():
        return {}
    divergence = lp.load_divergence(results)
    # A trial matched by its resolved directory: the pass may have walked the
    # run tree (run_dir the real path) or the paper tree (run_dir through the
    # arm's link, or a per-cell link under a real arm directory).
    arm_of = {str(t.resolve().relative_to(REPO)): m for m in arms for c in cells
              for t in (PROJECT / tree / 'continual' / m / c).glob('trial_*')}
    per_run = {}
    with np.load(npz) as archive:
        for i, raw in enumerate(archive['index']):
            rec = json.loads(str(raw))
            run_dir = str(pathlib.Path(rec['run_dir']))
            _arm_dir, cell, _trial_name = run_dir.rsplit('/', 2)
            method = arm_of.get(str((REPO / run_dir).resolve().relative_to(REPO)))
            if (method is None or cell not in cells
                    or rec.get('source') not in lp.ZT_SOURCES['centroid']):
                continue
            la = float(np.mean(np.diag(archive[f'run{i}_reward'])))
            summary = divergence.get(run_dir, {})
            if np.isfinite(la) and summary.get('F') is not None:
                bd = summary.get('BD')
                per_run[run_dir] = (cell, method, _trial(
                    la, float(summary['F']), None if bd is None else float(bd),
                    curve_mean(REPO / run_dir), run_dir))
    out = {}
    for cell, method, trial in per_run.values():
        out.setdefault(cell, {}).setdefault(method, []).append(trial)
    return out


def load_records(tree, cells, arms):
    """`load_pass`'s result from the training records, for a run that logs the
    centroid on every sub-task (`centroid_task<k>`): R[i][j] is the record at the
    last generation of phase j, on phase i's sub-task. F is the final-agent
    forgetting, or the switch forgetting with two alternating sub-tasks."""
    out = {}
    for m in arms:
        for cell in cells:
            for trial in sorted((PROJECT / tree / 'continual' / m / cell).glob('trial_*')):
                path = trial / 'training_metrics.json'
                if not path.exists():
                    continue
                records = json.loads(path.read_text())
                tasks = np.array([r['task'] for r in records])
                ends = list(np.flatnonzero(tasks[1:] != tasks[:-1])) + [len(tasks) - 1]
                R = np.array([[records[e][f'centroid_task{tasks[i]}'] for e in ends]
                              for i in ends], dtype=float)
                if len(set(tasks.tolist())) == 2:
                    # Two alternating sub-tasks: the switch forgetting, as the
                    # pass reports it (make_lineplot.load_divergence) -- each
                    # phase's agent against the agent after the next switch.
                    forgetting = float((R.diagonal()[:-1] - R.diagonal(1)).mean())
                else:
                    forgetting = float((R.diagonal()[:-1] - R[:-1, -1]).mean())
                run = str(trial.resolve().relative_to(REPO))
                out.setdefault(cell, {}).setdefault(m, []).append(_trial(
                    float(R.diagonal().mean()), forgetting, None, curve_mean(trial), run))
    return out


def fill_bd(loaded, sub):
    """Set each records trial's BD from the BD_PASS results of `sub`, matched
    by run directory; trials the pass did not score keep None."""
    divergence = lp.load_divergence(PROJECT / 'paper' / sub / 'results/centroid_bd')
    for by_method in loaded.values():
        for trials in by_method.values():
            for t in trials:
                t['BD'] = divergence.get(t['run'], {}).get('BD')
    return loaded


def as_arrays(by_method):
    """`{method: [trial]}` -> `{method: {'la', 'F', 'BD', 'cum': array}}`."""
    return {m: {k: np.array([np.nan if t[k] is None else t[k] for t in trials], dtype=float)
                for k in ('la', 'F', 'BD', 'cum')}
            for m, trials in by_method.items()}


def normalise(seeds, stability, plasticity='la'):
    """Per trial `{env: {method: {'x', 'y', 's'}}}` on the rescaled axes, and
    `{env: (untrained, best)}`. `s` = x - y, the performance kept (F only).
    The x axis is LA, or the curve mean under `plasticity='cum'`; `best` is the
    best LA either way, so the two versions share a scale."""
    scaled, scale = {}, {}
    for env, by_m in seeds.items():
        floor = FLOOR[env]
        best = max(float(np.mean(r['la'])) for r in by_m.values())
        scale[env] = (floor, best)
        span = best - floor
        scaled[env] = {}
        for m, r in by_m.items():
            x = (r[plasticity] - floor) / span
            y = r['F'] / span if stability == 'F' else r['BD']
            keep = np.isfinite(x) & np.isfinite(y)
            scaled[env][m] = {'x': x[keep], 'y': y[keep], 's': (x - y)[keep]}
    return scaled, scale


def pooled(per_env, seed=0):
    """`(mean, lo, hi)` of the mean over environments of the seed mean, the
    interval resampling seeds within each environment (stratified bootstrap)."""
    rng = np.random.default_rng(seed)
    point = float(np.mean([v.mean() for v in per_env]))
    boots = np.mean([rng.choice(v, (N_BOOT, v.size)).mean(axis=1) for v in per_env], axis=0)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return point, float(lo), float(hi)


def prob_improvement(a, b):
    """rliable's probability of improvement: the chance a seed of `a` beats a
    seed of `b` (ties count half), meaned over environments."""
    return float(np.mean([((x[:, None] > y[None]) + 0.5 * (x[:, None] == y[None])).mean()
                          for x, y in zip(a, b)]))


def points_for(scaled, envs, stability):
    """`{method: {'x', 'y'[, 's']: (mean, lo, hi), 'per_env': {...}}}` pooled over
    `envs`; a method missing from any of them is left out and named."""
    out = {}
    methods = {m for env in envs for m in scaled.get(env, {})}
    for m in [m for m in lp.METHOD_ORDER if m in methods]:
        if not all(scaled.get(env, {}).get(m, {}).get('x', np.empty(0)).size for env in envs):
            print(f'note: {m} is not in every environment of {envs}; left out of the pooled point')
            continue
        keys = ('x', 'y', 's') if stability == 'F' else ('x', 'y')
        out[m] = {k: pooled([scaled[env][m][k] for env in envs]) for k in keys}
        out[m]['per_env'] = {k: [scaled[env][m][k] for env in envs] for k in keys}
    return out


def dominators(points, alpha=0.05):
    """`{method: [RL methods SIGNIFICANTLY better on both axes]}`: higher LA and
    lower F, each by a one-sided Mann-Whitney U test over seeds at `alpha`.
    Better means alone did not count as dominance since 2026-09-24: on Kinetix
    ES trailed ReDo-PPO by 0.02 LA (p 0.84) and sat in the region."""
    def seeds(p, k):
        return np.concatenate(p['per_env'][k])
    out = {}
    for a, pa in points.items():
        out[a] = [b for b, pb in points.items() if b != a and lp.FAMILY.get(b) == 'rl'
                  and mannwhitneyu(seeds(pb, 'x'), seeds(pa, 'x'), alternative='greater').pvalue < alpha
                  and mannwhitneyu(seeds(pb, 'y'), seeds(pa, 'y'), alternative='less').pvalue < alpha]
    return out


def draw_panel(ax, points, title, stability):
    """RL-dominated region, equal-performance lines, then every method's point."""
    if not points:
        ax.set_axis_off()
        ax.text(0.5, 0.5, f'{title}\nnot built', ha='center', va='center',
                transform=ax.transAxes, color='0.4')
        return
    x_lo = min(p['x'][1] for p in points.values())
    x_hi = max(p['x'][2] for p in points.values())
    y_lo = min(p['y'][1] for p in points.values())
    y_hi = max(p['y'][2] for p in points.values())
    pad_x, pad_y = 0.1 * (x_hi - x_lo or 1), 0.1 * (y_hi - y_lo or 1)
    x_lo, x_hi, y_lo, y_hi = x_lo - pad_x, x_hi + pad_x, y_lo - pad_y, y_hi + pad_y
    if x_hi - x_lo < MIN_SPAN:
        x_lo, x_hi = (x_lo + x_hi) / 2 - MIN_SPAN / 2, (x_lo + x_hi) / 2 + MIN_SPAN / 2
    if y_hi - y_lo < MIN_SPAN:
        y_lo, y_hi = (y_lo + y_hi) / 2 - MIN_SPAN / 2, (y_lo + y_hi) / 2 + MIN_SPAN / 2
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(y_hi, y_lo)                      # inverted: up is less forgetting
    ax.set_autoscale_on(False)

    if stability == 'F':
        span = (x_hi - y_lo) - (x_lo - y_hi)
        step = next(s for s in (0.05, 0.1, 0.2, 0.25, 0.5, 1.0, 2.0) if span / s <= 8)
        for c in np.arange(np.floor((x_lo - y_hi) / step), np.ceil((x_hi - y_lo) / step) + 1) * step:
            ax.plot([x_lo, x_hi], [x_lo - c, x_hi - c], color=ISO, lw=0.6, zorder=0)

    # Each RL method shades what lies beyond its 95% interval on both axes (the
    # corner of its lower LA and upper F bound), not beyond its mean, so a point
    # a noise-sized gap behind it stays white; `dominators` tests the same claim.
    rl = sorted((p['x'][1], p['y'][2]) for m, p in points.items() if lp.FAMILY.get(m) == 'rl')
    if rl:
        xs = [x for x, _ in rl]
        best = np.minimum.accumulate([y for _, y in rl][::-1])[::-1]
        edge_x, edge_y = [x_lo] + xs, [best[0]] + list(best)
        ax.fill_between(edge_x, edge_y, y_hi, step='pre', color=DOMINATED, lw=0, zorder=0.5)
        ax.step(edge_x + [xs[-1]], edge_y + [y_hi], where='pre', color='0.65', lw=0.7, zorder=0.6)

    for m, p in points.items():
        ne = lp.FAMILY.get(m) == 'ne'
        (x, xl, xh), (y, yl, yh) = p['x'], p['y']
        ax.errorbar([x], [y], xerr=[[x - xl], [xh - x]], yerr=[[y - yl], [yh - y]],
                    fmt='o' if ne else 's', ms=6.5 if ne else 5.5,
                    color=_colour(m), mec='white', mew=0.6,
                    elinewidth=0.8, capsize=0, zorder=4 if ne else 3)
    # The best trade-off: the highest LA - F, the point on the highest grey
    # line. Its own line is drawn darker, so the ring says which point it is
    # and the line says by how much it leads.
    rank = summary(points) if any('s' in p for p in points.values()) else None
    best = rank['ringed'] if rank else None
    if best is not None:
        x, y = points[best]['x'][0], points[best]['y'][0]
        ax.plot([x_lo, x_hi], [x_lo - x + y, x_hi - x + y], color=BEST, lw=0.9, zorder=1)
        ax.plot([x], [y], ls='', marker='o', ms=13, mfc='none', mec=BEST, mew=1.1, zorder=5)
    ax.xaxis.set_major_locator(MaxNLocator(4))
    ax.yaxis.set_major_locator(MaxNLocator(4))
    ax.set_title(title)
    ax.spines[['top', 'right']].set_visible(False)


def draw(panels, headers, shape, stem, stability, methods, axes_in=(1.85, 1.7),
         legend_rows=1, x_label='Learning accuracy', x_name='LA'):
    """`panels` = [(row, col, title, points)] on a `shape` grid, suite
    `headers` = [(row, first col, last col, text)], to `<stem>.pdf/.png`;
    `legend_rows` wraps the legend for a figure too narrow for one row."""
    nrows, ncols = shape
    left, right, gap_w = 0.7, 0.45, 0.65
    top, bottom, gap_h = 0.73 + 0.22 * legend_rows, 0.5, 0.95
    width = left + right + ncols * axes_in[0] + (ncols - 1) * gap_w
    height = top + bottom + nrows * axes_in[1] + (nrows - 1) * gap_h
    fig = plt.figure(figsize=(width, height))
    gs = GridSpec(nrows, ncols, figure=fig, left=left / width, right=1 - right / width,
                  top=1 - top / height, bottom=bottom / height,
                  wspace=gap_w / axes_in[0], hspace=gap_h / axes_in[1])
    axes = {}
    for row, col, title, points in panels:
        ax = axes[row, col] = fig.add_subplot(gs[row, col])
        draw_panel(ax, points, title, stability)
        ax.set_xlabel(x_label)
        if col == 0:
            ax.set_ylabel('Forgetting' if stability == 'F' else 'Behavioural divergence')
    pt = 1 / 72 / height
    for row, c0, c1, text in headers:
        a, b = axes[row, c0].get_position(), axes[row, c1].get_position()
        y = a.y1 + 19 * pt
        fig.add_artist(Line2D([a.x0, b.x1], [y, y], lw=0.6, color='0.3'))
        fig.text((a.x0 + b.x1) / 2, y + 2 * pt, text, ha='center', va='bottom',
                 fontsize=plt.rcParams['font.size'] + 1)
    # ES and NES share a label, so whichever of them a family kept is one entry.
    shown, seen = [], set()
    for m in methods:
        key = 'es' if m in es_arm.ARMS else m
        if key not in seen:
            seen.add(key)
            shown.append(m)
    handles = [Line2D([], [], ls='', marker='o' if lp.FAMILY.get(m) == 'ne' else 's',
                      ms=6, color=_colour(m)) for m in shown]
    labels = [_label(m) for m in shown]
    handles.append(Patch(color=DOMINATED))
    labels.append('Worse than an RL method on both axes')
    if stability == 'F':
        handles.append(Line2D([], [], color=ISO, lw=1.2))
        labels.append(f'Equal {x_name} $-$ F')
        handles.append(Line2D([], [], ls='', marker='o', ms=9, mfc='none',
                              mec=BEST, mew=1.1))
        labels.append(f'Best {x_name} $-$ F (p < 0.05)')
    fig.legend(handles, labels, loc='upper center', ncol=-(-len(handles) // legend_rows), frameon=False,
               handler_map={tuple: HandlerTuple(ndivide=1)},
               bbox_to_anchor=(0.5, 1 - 0.05 / height), handletextpad=0.3, columnspacing=1.1)
    for ext in ('pdf', 'png'):
        fig.savefig(f'{stem}.{ext}', dpi=300)
    plt.close(fig)
    print(f'wrote {stem}.pdf, {stem}.png  ({width:.1f} x {height:.1f} in)')


def _ci(p, digits=2):
    return f'{p[0]:.{digits}f} [{p[1]:.{digits}f}, {p[2]:.{digits}f}]'


def _label(m):
    return ES_LABEL if m in es_arm.ARMS else lp.METHOD_STYLE[m]['label']


def _colour(m):
    return ES_COLOUR if m in es_arm.ARMS else lp.METHOD_STYLE[m]['color']


def summary(points):
    """The family's best NE and best RL arm on LA - F, whether that NE arm is
    dominated, and its lowest probability of improvement over the RL arms."""
    by_fam = {f: [m for m in points if lp.FAMILY.get(m) == f] for f in ('ne', 'rl')}
    if not all(by_fam.values()) or 's' not in next(iter(points.values())):
        return None
    ne = max(by_fam['ne'], key=lambda m: points[m]['s'][0])
    rl = max(by_fam['rl'], key=lambda m: points[m]['s'][0])
    poi = {r: prob_improvement(points[ne]['per_env']['s'], points[r]['per_env']['s'])
           for r in by_fam['rl']}
    order = sorted(points, key=lambda m: points[m]['s'][0], reverse=True)
    best, second = order[0], order[1]
    # A lead is only called a win when it survives the paper's cross-family
    # test (the metrics figure's marks): the winner against EVERY method of the
    # OTHER family on the per-seed LA - F, one-sided Mann-Whitney U, Holm over
    # those comparisons, graded on the weakest. A tie inside the winner's own
    # family (ES vs GA) does not block it. Ranking by the mean alone made a 0.02
    # lead (MountainCar noise) look like the same result as a 0.4 one.
    # The RING goes to the highest LA - F among the methods that pass, so a
    # near-tie at the top (Acrobot noise: GA 0.60 misses at p 0.052, ES 0.59
    # passes at 0.038) rings the one the test supports rather than nobody.
    tests = mann_whitney_tests({m: np.concatenate(points[m]['per_env']['s']) for m in points},
                               lp.FAMILY, True)
    p_of = {m: max((c['p_holm'] for c in tests[m]['comparisons']), default=1.0) for m in order}
    # A frozen specialist (main_panels marks it from the generalist figure's
    # classes) is not eligible: nothing to forget is not stability.
    frozen = sorted(m for m, p in points.items() if p.get('frozen'))
    # ... and by a lead that means something: MIN_EFFECT over the best of the
    # other family, on the means.
    lead = {m: points[m]['s'][0] - max(points[o]['s'][0] for o in points
                                       if lp.FAMILY.get(o) != lp.FAMILY.get(m))
            for m in order}
    ringed = next((m for m in order if m not in frozen and p_of[m] < 0.05
                   and lead[m] >= MIN_EFFECT), None)
    return dict(ne=ne, rl=rl, dominated_by=dominators(points)[ne], poi=poi,
                weakest=min(poi, key=poi.get), best=best, second=second,
                p=p_of[best], ringed=ringed, p_ringed=p_of.get(ringed), frozen=frozen,
                lead=lead)


def write_tex(panels, stem):
    lines = ['% Built by scripts/analysis/plot_stability_plasticity.py -- rerun it, do not edit.',
             r'% Needs \usepackage{booktabs}.',
             r'\begin{table}[t]', r'\centering',
             r'\caption{Stability--plasticity trade-off of the centroid. Learning accuracy (LA) and '
             r'forgetting (F) are rescaled per environment so that 0 is the untrained network and 1 '
             r'the best LA of any method; LA$-$F is the performance a sub-task keeps. Mean over seeds '
             r'intervals in the figure. '
             r'$P$: probability that a seed of the NE method keeps more than a seed of the '
             r'RL method, the lowest over the RL methods. $^\dagger$: no RL method is at least as good '
             r'on both LA and F. PBT$_N$: PBT-PPO with $N$ members.}',
             r'\label{tab:stability_plasticity}', r'\footnotesize', r'\setlength{\tabcolsep}{3pt}',
             r'\begin{tabular}{llrlrr}', r'\toprule',
             r'Task & Best NE & LA$-$F & Best RL & LA$-$F & $P$ \\', r'\midrule']
    for title, points in panels:
        s = summary(points) if points else None
        if s is None:
            continue
        mark = '' if s['dominated_by'] else r'$^\dagger$'
        tex = {m: ncs.TEX_METHOD.get(m, _label(m)) for m in (s['ne'], s['rl'])}
        label = title.replace('8x8 / 16x16', r'8$\times$8 / 16$\times$16')
        lines.append(f"{label} & {tex[s['ne']]}{mark} & {points[s['ne']]['s'][0]:.2f} & "
                     f"{tex[s['rl']]} & {points[s['rl']]['s'][0]:.2f} & "
                     f"{s['poi'][s['weakest']]:.2f} \\\\")
    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    pathlib.Path(f'{stem}.tex').write_text('\n'.join(lines) + '\n')
    print(f'wrote {stem}.tex')


def write_markdown(panels, scales, stem, stability, plasticity='la'):
    y_name = 'F' if stability == 'F' else 'BD'
    x_name = 'LA' if plasticity == 'la' else 'Cum'
    columns = ['| **LA** | Learning accuracy, mean_i R[i][i] from the centroid forgetting pass: '
               'each sub-task at the end of its own phase. Rescaled: (LA - untrained) / '
               '(best - untrained), `best` the highest mean LA of any method there. |']
    if plasticity == 'cum':
        columns = ['| **Cum** | The centroid curve\'s mean over the whole run (the paper\'s '
                   'Cum. centroid up to generations / 1000): reward collected THROUGHOUT, not '
                   'at the end of each sub-task. Rescaled by the same (best LA - untrained), so '
                   'the two versions share a scale. Unlike LA it already credits keeping a '
                   'sub-task on its later repeats, so it is not independent of F. |']
    if stability == 'F':
        columns += ['| **F** | Forgetting as the metrics tables read it (final-agent, or switch '
                    'forgetting with two sub-tasks), divided by the same (best - untrained). '
                    'Lower is better. |',
                    f'| **{x_name} - F** | What a sub-task keeps, on the same scale. Exactly the '
                    'final average performance only under LA. |']
    else:
        columns.append('| **BD** | Behavioural divergence at the switch, in [0, 1], not '
                       'rescaled. Lower is better. |')
    columns.append(f'| **Dominated by** | RL methods significantly better on both LA and {y_name} '
                   '(one-sided Mann-Whitney U over seeds, p < 0.05 on each). The grey '
                   'region lies beyond some RL method\'s 95% interval on both axes (its '
                   'lower LA bound, its upper F bound). |')
    if stability == 'F':
        columns.append('| **P(NE > RL)** | Probability of improvement on LA - F (rliable): a '
                       'seed of the best NE method against a seed of each RL method. |')
    if stability == 'F':
        columns.append('| **Best LA - F** | The ringed point of a panel, on the darker line: '
                       'the method that keeps the most. The RING goes to the highest LA - F '
                       'among the methods that beat EVERY method of the other family (NE vs RL) '
                       'on the per-seed values -- one-sided Mann-Whitney U, Holm over those '
                       'comparisons, weakest p < 0.05 -- AND lead the best of the other family by at '
                       f'least {MIN_EFFECT} LA - F on the means (the "lead" column; a test alone '
                       'rings a 0.01 lead when seeds hardly vary) -- and to nobody if none does. A FROZEN '
                       'specialist (the generalist figure\'s class: with two alternating '
                       'sub-tasks it only ever solved one) is not eligible, since its F is ~0 '
                       'for never having learnt the other sub-task. |')
    lines = [f'# {pathlib.Path(stem).name}', '',
             'The stability-plasticity trade-off of the centroid, built by '
             '`scripts/analysis/plot_stability_plasticity.py` (its docstring has the '
             'definitions and the literature).', '',
             '| Column | What it is |', '|---|---|', *columns,
             '', 'One row a task, the ten panels of `continual_main`. Mean over seeds '
             '[95% percentile-bootstrap CI]. Kinetix and HalfCheetah are read from their training '
             'records (`centroid_task<k>`; the module docstring says why) and have no BD.', '']
    if stability == 'F':
        lines += ['## Summary', '',
                  f'| Task | Best {x_name} - F | Holm p vs other family | lead over other family | ringed | its {x_name} | ringed p | '
                  f'frozen (not eligible) | '
                  f'Best NE | {x_name} - F | Best RL | {x_name} - F | NE dominated by | '
                  'P(NE > RL), lowest | P(NE > RL), every RL method |',
                  '|' + '---|' * 15]
        for title, points in panels:
            s = summary(points) if points else None
            if s is None:
                lines.append(f'| {title} | not built |' + ' |' * 13)
                continue
            lines.append(
                f"| {title} | {_label(s['best'])} | {s['p']:.3g} | {s['lead'][s['best']]:+.2f} | "
                + (f"{_label(s['ringed'])} | {points[s['ringed']]['x'][0]:.2f} | "
                   f"{s['p_ringed']:.3g}" if s['ringed'] else 'none | | ')
                + f" | {', '.join(map(_label, s['frozen'])) or 'none'}"
                + f" | {_label(s['ne'])} | {_ci(points[s['ne']]['s'])} | "
                f"{_label(s['rl'])} | {_ci(points[s['rl']]['s'])} | "
                f"{', '.join(map(_label, s['dominated_by'])) or 'none'} | "
                f"{s['poi'][s['weakest']]:.2f} ({_label(s['weakest'])}) | "
                + ', '.join(f'{_label(r)} {p:.2f}' for r, p in s['poi'].items()) + ' |')
        lines.append('')
    lines += ['## Every point', '',
              f'| Task | Method | n | LA | {y_name} |'
              + (' LA - F |' if stability == 'F' else '') + ' Dominated by |',
              '|---|---|---|---|---|' + ('---|' if stability == 'F' else '') + '---|']
    for title, pts in panels:
        dom = dominators(pts)
        for m, p in pts.items():
            seeds = p['per_env']['x'][0].size
            lines.append(f'| {title} | {_label(m)} | {seeds} | {_ci(p["x"])} | '
                         f'{_ci(p["y"])} |' + (f' {_ci(p["s"])} |' if stability == 'F' else '')
                         + f' {", ".join(map(_label, dom[m])) or ""} |')
    lines += ['', '## Rescaling', '', '| Family | Environment | untrained | best LA (raw) |',
              '|---|---|---|---|']
    for key, by_env in scales.items():
        for env, (floor, best) in by_env.items():
            lines.append(f'| {", ".join(filter(None, key))} | {lp.ENV_TITLES.get(env, env)} | '
                         f'{floor:g} | {best:.4g} |')
    pathlib.Path(f'{stem}.md').write_text('\n'.join(lines) + '\n')
    print(f'wrote {stem}.md')


def extract():
    """Read continual_main's runs and write DATA: per panel, per arm, one
    `{la, F, BD, cum, run}` a trial, plus the kept arms and the runs each arm
    link resolves to."""
    # load_divergence counts a run's distinct sub-tasks from its checkpoint,
    # found by the repo-relative run_dir the forgetting pass recorded.
    os.chdir(REPO)
    cells = pcl._cells(lambda f: f in FIGS)
    meta = {'arms': {}, 'source': {}, 'links': {}, 'panels': {}}
    for tree, tree_cells in cells.items():
        root = PROJECT / tree
        arms = reported_arms(root, tree_cells,
                             es_kept='es' if tree.startswith('paper/kinetix') else 'nes')
        meta['arms'][tree] = arms
        meta['source'][tree] = (f'forgetting pass, paper/{PASS[tree]}/results/centroid'
                                if tree in PASS else 'training records')
        loaded = (load_pass(PASS[tree], tree, tree_cells, arms) if tree in PASS
                  else load_records(tree, tree_cells, arms))
        if tree in BD_PASS:
            loaded = fill_bd(loaded, BD_PASS[tree])
        for cell in tree_cells:
            by_m = loaded.get(cell, {})
            for m in arms:
                if not by_m.get(m):
                    print(f'WARNING: {tree} {cell} {m}: no scored trial')
            meta['panels'][pcl._key(tree, cell)] = {m: by_m[m] for m in arms if by_m.get(m)}
        for m in arms:
            meta['links'][f'{tree}/continual/{m}'] = str(
                (root / 'continual' / m).resolve().relative_to(PROJECT))
    meta['extracted'] = datetime.date.today().isoformat()
    DATA.parent.mkdir(parents=True, exist_ok=True)
    DATA.write_text(json.dumps(meta, indent=1) + '\n')
    print(f'wrote {DATA}')


def main_panels(meta, stability='F', plasticity='la', figs=('main',)):
    """`([(title, points)], scales)` for the panels of `figs` (continual_main's
    by default; `FIGS` adds the two-sub-task noise and physics ones), in
    pcl.PANELS order."""
    panels, scales = [], {}
    for fig, tree, cell, title in pcl.PANELS:
        if fig not in figs:
            continue
        env = cell.split('_sigma')[0]
        seeds = as_arrays(meta['panels'][pcl._key(tree, cell)])
        scaled, scales[title, ] = normalise({env: seeds}, stability, plasticity)
        points = points_for(scaled, [env], stability) if seeds else {}
        for m in frozen_methods(pcl._key(tree, cell)):
            if m in points:
                points[m]['frozen'] = True
        panels.append((title, points))
        s = summary(points) if points and stability == 'F' else None
        print(f'{title:32s} arms {" ".join(meta["arms"][tree])}'
              + (f"  best NE {s['ne']} (dominated by {s['dominated_by'] or 'none'}), "
                 f"best RL {s['rl']}, ringed {s['ringed']}, frozen {s['frozen'] or 'none'}"
                 if s else ''))
    return panels, scales


def frozen_methods(key):
    """The frozen specialists of a panel by the generalist figure's rule, from
    its saved data (`plot_generalist_scores.py --extract` writes it); empty for
    a panel that figure does not have (the ten-sub-task and Kinetix families,
    where two sub-tasks do not alternate). Its `es` is whichever ES arm the
    family kept, so it names every ES arm here."""
    import plot_generalist_scores as pgs   # it imports FLOOR from this module
    if not pgs.DATA.exists():
        return set()
    panel = json.loads(pgs.DATA.read_text())['panels'].get(key)
    if not panel or not panel['arms']:
        return set()
    classes = pgs.classify(panel, key.split('|')[1])
    out = {m for m, k in classes.items() if k == 'frozen'}
    if 'es' in out:
        out |= set(es_arm.ARMS)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--extract', action='store_true',
                    help='re-read the runs into the saved data first')
    ap.add_argument('--stability', choices=['F', 'BD'], default='F')
    ap.add_argument('--plasticity', choices=['la', 'cum'], default='la',
                    help="'cum' puts the centroid curve's mean over the whole run on the x "
                         'axis instead of the end-of-phase score: how much reward was '
                         'collected THROUGHOUT, not at the end of each sub-task')
    args = ap.parse_args()
    if args.extract:
        extract()
    if not DATA.exists():
        sys.exit(f'no {DATA}: run with --extract first')
    meta = json.loads(DATA.read_text())
    plt.rcParams.update({'font.size': 9})
    # One PBT-PPO entry, whichever N a panel kept (continual_main's label).
    lp.METHOD_STYLE['pbt'] = {**lp.METHOD_STYLE['pbt'], 'label': 'PBT-PPO'}
    # The paper's figure (F, learning accuracy) goes to visuals/final, the
    # variants stay in visuals.
    stem = (f'{FINAL / STEM}' if (args.stability, args.plasticity) == ('F', 'la')
            else f'{VARIANTS}' + ('' if args.stability == 'F' else '_bd')
            + ('' if args.plasticity == 'la' else '_cum'))
    x_label = ('Learning accuracy' if args.plasticity == 'la'
               else 'Mean return during training')

    panels, scales = main_panels(meta, args.stability, args.plasticity)

    union = {m for _, points in panels for m in points}
    methods = [m for m in lp.METHOD_ORDER if m in union]
    pathlib.Path(stem).parent.mkdir(parents=True, exist_ok=True)
    draw([(i // NCOLS, i % NCOLS, label, points) for i, (label, points) in enumerate(panels)],
         [], (-(-len(panels) // NCOLS), NCOLS), stem, args.stability, methods,
         axes_in=(2.0, 1.8), x_label=x_label)
    if args.stability == 'F' and args.plasticity == 'la':
        write_tex(panels, stem)
    write_markdown(panels, scales, stem, args.stability, args.plasticity)
    return 0


if __name__ == '__main__':
    sys.exit(main())
