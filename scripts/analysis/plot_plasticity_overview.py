"""Plasticity diagnostics and basin width across suites, one column per task.

    .venv/bin/python scripts/analysis/plot_plasticity_overview.py            # both figures, 2-sub-task set
    .venv/bin/python scripts/analysis/plot_plasticity_overview.py --set more
    .venv/bin/python scripts/analysis/plot_plasticity_overview.py --set main [--extract]
        -> final/plasticity_main_continual.{pdf,png}: continual_main's tasks, runs and arms
           final/appendix/saturation_main_continual.{pdf,png}: saturated units, tanh bodies only
           final/data/plasticity_main_continual.{npz,json}: what it is drawn from

    -> paper/visuals/plasticity_diagnosis[_more].{pdf,png}   does NE show the usual signs of plasticity loss?
       paper/visuals/basin_width[_more].{pdf,png}            how wide the agent's basin is (read beside
                                                             generalist_outcomes_centroid, which has the outcomes)
       paper/visuals/basin_width_scatter[_more].{pdf,png}    per task: width gap against generalist gap
       paper/visuals/plasticity_overview[_more].md           every panel as numbers
       paper/visuals/dormancy_lines[_more].{pdf,png}         `--figure lines`: the dormancy rows
                                                             against sub-task, mean and 95% CI;
                                                             `--figure main` draws them above its dot rows

The final figures (`--set main`) are built in two steps, as the
other figures in paper/visuals/final: `--extract` reads the checkpoint passes
of the runs continual_main shows and saves one curve a (task, row, arm, trial);
without it only the saved data is read. Its arms are continual_main's kept arms
(ES = NES, OpenES on Kinetix; its PBT size), not `keep_one_arm`'s, and its
cheetah columns read the passes of paper/mjx/cheetah/data/<family>
(`scripts/analysis/posthoc_paper_cheetah.sh`), whose RL arms and noise GA are
the CLUSTER ant-PPO-shape runs.

Columns are `plot_metrics_overview.SETS` (the same tasks, in the same order,
as the metrics overview), and so are the arms: one of ES/NES (drawn as ES) and
one of PBT N=8/N=2 a column (`plot_metrics_overview.keep_one_arm`). Every panel is `make_metrics_figure.draw_panel`: the
mean over trials with a 95% bootstrap CI, and the mark when a method beats
every method of the other family. All values describe the CENTROID agent,
the network the centroid lineplot scores (the policy for RL).

Diagnosis rows (the network at the end of the run, from the checkpoint passes
in `<paper dir>/results/centroid/`):

    FT         forward transfer from `metrics_centroid_values.json`: a sub-task
               learnt as well as the method's own stationary run is 0. The
               performance-side definition of plasticity loss.
    Dormant    fraction of hidden units dormant, last 5 sub-tasks,
    Persistence chance-corrected lag-1 persistence of the dormant set, mean
               over the run, and
    Age        mean number of consecutive sub-tasks the units dormant at the
               end have been dormant (0 when nothing is dormant) --
               all three from `results/centroid_pooled/`, the pass rerun with
               `--probe pooled --criterion magnitude`: one fixed batch drawn
               from every sub-task, so a unit is compared with itself on the
               same states, and a unit is dormant when it is SILENT (ReDo's
               mean |output| <= tau x layer mean) on every body. On the tanh
               bodies (Kinetix, HalfCheetah) ReDo's own `variability` test also
               counts units saturated at +/-1; those are not counted here. Under the
               matched probe an observation offset redraws the dormant set at
               every switch whatever the network does (scoring on the NEXT
               sub-task's states gives the same numbers), and persistence
               then measures the input, not the network (checked 2026-09-16).
    NTK rank   effective rank of the centred-logit NTK at the START of the
               next sub-task, last 5 sub-tasks over the first 3 (1 = no rank
               lost).
    Weight     parameter RMS at the end over the start.

Specialist rows:

    Generalist fraction of post-switch sub-task checkpoints that clear the
               solved threshold on the sub-task just trained AND the previous
               one (`generalist_checkpoints.classify`, computed here from each
               run's `evaluation.json` so every reported arm has it).
    Width      fraction of greedy actions (normalised action distance on a
               continuous head) that change under a random perturbation of
               every weight tensor by 10% of its own norm, last 5 sub-tasks.
               Lower = a wider basin around the agent in policy space.
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
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec  # noqa: E402
from matplotlib.lines import Line2D                        # noqa: E402
from matplotlib.ticker import (FixedLocator, FuncFormatter, LogLocator,  # noqa: E402
                               MaxNLocator, NullLocator)

REPO = pathlib.Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'scripts'))
sys.path.insert(0, str(REPO / 'scripts' / 'analysis'))
import make_lineplot as lp                                 # noqa: E402
from make_metrics_figure import draw_panel                 # noqa: E402
import generalist_checkpoints as gc                        # noqa: E402
import plot_metrics_overview as pmo                        # noqa: E402
from source.envs.registry import threshold_for             # noqa: E402

PAPER = REPO / 'projects/iclr_2027/paper'
OUT = PAPER / 'visuals'
# The figures built from the saved data in visuals/final go there, the
# appendix ones to final/appendix.
FINAL = OUT / 'final'
IN_FINAL = {'plasticity_main_continual': FINAL,
            'plasticity_all_continual': FINAL / 'appendix',
            'saturation_main_continual': FINAL / 'appendix'}
# The figures `--set main` builds from the saved data, and the rows they need.
FINAL_FIGURES = ('main', 'all', 'saturation_lines')
# Final figures drawn to sit beside their caption, at print size: (width in,
# row height in, font size). Figure 6 takes 0.655\linewidth (2026-09-21).
SIDE_CAPTION = {'main': (3.6, 0.56, 6.5)}
FINAL_DATA = FINAL / 'data' / 'plasticity_main_continual'      # .npz curves, .json the rest
# The final figure's columns: the continual_main tree whose kept arms each
# draws, and the results directory where the default (PAPER/<sub>/results)
# scored other runs than that tree's.
FINAL_TREE = {
    'gymnax/noise/10task': 'paper/gymnax/data/noise_10task',
    'gymnax/actions/2task': 'paper/gymnax/data/actions_2task',
    'mjx/cheetah/noise/10task': 'paper/mjx/cheetah/data/noise_10task',
    'mjx/cheetah/actions/2task': 'paper/mjx/cheetah/data/actions_2task',
    'minigrid/minigrid': 'paper/minigrid/data',
    'kinetix/kinetix': 'paper/kinetix/data',
}
FINAL_RESULTS = {
    'mjx/cheetah/noise/10task': 'mjx/cheetah/data/noise_10task/results',
    'mjx/cheetah/actions/2task': 'mjx/cheetah/data/actions_2task/results',
}


def out_path(stem):
    return IN_FINAL.get(stem, OUT) / stem
# Bodies without a registry threshold: the values the paper's other visuals use
# (finish_iclr.sh for MiniGrid, the cheetah generalist tables, the ant's
# learner-vs-floor gap in plot_noncontinual_solve.py).
THRESHOLD = {'minigrid': 0.8, 'cheetah': 2000.0, 'ant': 3000.0, 'kinetix': 1.0}
# The scatter draws a task hollow when either family learns the sub-task it is
# shown in fewer than this fraction of its post-switch checkpoints.
LEARNED_MIN = 0.5
# Tasks the scatter adds to a set's columns without a row in the overview
# figures: the ant task of the main continual figure (paper/visuals
# continual_main), at the same threshold as the other ant columns.
SCATTER_EXTRA = {}
# Columns of `plot_metrics_overview.SETS` these figures leave out. The ant
# (2026-09-15): the GA does not solve it and ES/NES learn the shown sub-task in
# at most 60% of checkpoints, so the paper does not report it.
DROPPED = {'ant-noise', 'ant-physics'}
# Columns the two plasticity figures add to a set: Kinetix is a twenty-level
# chain, not a two-sub-task schedule, but it is the one body where the
# dormancy picture reverses (NE dormant, PPO not), so it is drawn beside them.
# No NTK or width there (`curvature_width.py` has no
# multi-discrete head): those panels read n/a.
PLASTIC_EXTRA = {
    'two': [('kinetix', 'Kinetix', 'kinetix/kinetix', 'Kinetix20', '20 levels', 'level chain')],
}

# The line figure's rows: (label, higher is better), all on the pooled batch.
LINES = {
    'dormant_pooled': ('Dormant units', False),
    'persistence':    ('Dormancy\npersistence', False),
    'age':            ('Dormancy age\n(tasks)', False),
    'saturated':      ('Saturated units', False),
    # Absolute, not late / early: the ratio cannot tell a steady decline from
    # a one-off drop after the first sub-task or a rank low from the start.
    'ntk':            ('NTK rank', True),
    # Parameter RMS of the saved agent, ABSOLUTE and on a log axis shared
    # across tasks (2026-09-16). The ratio to the first checkpoint made
    # MountainCar look special only because ES learns little in its first
    # sub-task (RMS 0.18 there against 0.29-0.34); in absolute terms it ends
    # beside CartPole and Acrobot, and Kinetix is the outlier.
    'weight':         ('Weight RMS', False),
    # Basin width against sub-task: the relative test (10% of each tensor's
    # norm) and the absolute one (noise s.d. ABS_WIDTH_SIGMA on every weight).
    # Both, because they disagree where weight scales differ (Kinetix).
    'width':          ('Action change,\n10% rel. noise', False),
    'width_abs':      ('Action change,\nnoise s.d. 0.03', False),
}
# Line rows drawn on a log axis.
LOG_LINES = {'weight'}
# Line rows drawn on a symlog axis, {row: linear threshold}. The dormant
# fraction was one for an hour on 2026-09-20: it made Kinetix (0.001-0.03)
# readable beside the ReLU bodies (0.03-0.7) but flattened the RL climb the
# text quotes, so the row is linear again, shared across every task.
SYMLOG_LINES = {}
# Line rows whose scale is a property of the task (NTK rank is ~2 on gymnax,
# ~150 on HalfCheetah): each panel keeps its own y range.
UNSHARED_LINES = {'ntk'}
# The rows the line figure draws (2026-09-16): the fraction and the age, BOTH
# on the pooled batch, so a panel pair describes the same dormant units -- the
# age is 0 exactly where the fraction above it is 0. Silent units only, as in
# the dot figures. The saturated fraction (a tanh unit pinned at one sign of
# +/-1, `plasticity_checkpoints.SAT_LEVEL`), which the silent test does not
# count, has its own figure: a ReLU cannot saturate, so it exists only for the
# tanh bodies (`TANH_COLUMNS`).
LINE_ROWS = ['dormant_pooled', 'age']


# Output suffix a set. `main` is the ten tasks of continual_main and
# stability_plasticity, with the same arms.
SUFFIX = {'two': '', 'main': '_continual'}

ROWS = {
    # key: (label, higher is better)
    'ft':         ('FT', True),
    'dormant':    ('Dormant units', False),
    'age':        ('Dormancy age\n(tasks)', False),
    'ntk':        ('NTK rank\n(late / early)', True),
    'weight':     ('Weight RMS\n(end)', False),
    # Basin width at a FIXED absolute radius (`curvature_width.py
    # --width-abs-only`, `results/centroid_abswidth/`): gaussian noise of s.d.
    # ABS_WIDTH_SIGMA on every weight, the same for every method.
    'width_abs':  ('Action change,\nnoise s.d. 0.03', False),
    'generalist': ('Generalist\ncheckpoints', True),
    # Not drawn as a row: the scatter's marker rule. Fraction of post-switch
    # checkpoints that clear the threshold on the sub-task just trained.
    'learned':    ('Learned shown\ntask', True),
    'width':      ('Action change,\n10% relative noise', False),
    # Chance-corrected persistence of the dormant set at a lag of one
    # sub-task, mean over the run: 0 = a different set each time (functional
    # sparsity), 1 = the same units throughout. n/a where nothing is dormant.
    'persistence': ('Dormancy\npersistence', False),
    # Fraction of hidden units saturated at one sign, last 5 sub-tasks, pooled
    # batch; tanh bodies only.
    'saturated':   ('Saturated units', False),
}
# Performance (FT, Cum.) is reported in its own figures, so neither plasticity
# figure repeats it; `ft` stays selectable with --rows.
FIGURES = {
    # A row is (kind, key): 'dot' is a `ROWS` key drawn as `draw_panel`,
    # 'line' a `LINES` key drawn against sub-task.
    'appendix':    ('plasticity_appendix', [('dot', k) for k in (
                        'dormant', 'persistence', 'age', 'ntk', 'weight', 'width')]),
    # Every row against sub-task: silent units (fraction, age), then NTK rank
    # and weight RMS. Saturated units have their own figure (`saturation`).
    # The paper's figure is one block of representative columns (MAIN_COLUMNS);
    # `all` is the ten tasks of continual_main, for the appendix (2026-09-19).
    # Dormancy rows only since 2026-09-21: NTK rank and weight RMS are in `all`.
    # The dormant fraction only since 2026-09-24: the age row backed one
    # sentence and showed no family gap on MountainCar, so it is in `all`.
    'main':        ('plasticity_main', [('line', 'dormant_pooled')]),
    'all':         ('plasticity_all', [('line', k) for k in
                                        LINE_ROWS + ['ntk', 'weight']]),
    # Width only: the generalist fraction per method is the dark segment of
    # `plot_generalist_outcomes.py`'s bars and is read there. It is still
    # loaded, for the scatter's y axis.
    'specialists': ('basin_width', [('dot', 'width'), ('dot', 'width_abs')]),
    'lines':       ('dormancy_lines', [('line', k) for k in LINE_ROWS]),
    # Saturated units, tanh bodies only: against sub-task, then the last five.
    'saturation':  ('saturation', [('line', 'saturated'), ('dot', 'saturated')]),
    'saturation_lines': ('saturation_main', [('line', 'saturated')]),
}
# Columns a figure is restricted to; every other figure draws the whole set.
# Only a bounded activation saturates: Kinetix and HalfCheetah are tanh,
# gymnax and MiniGrid ReLU.
TANH_COLUMNS = {'kinetix', 'cheetah-noise', 'cheetah-physics', 'cheetah-actions',
                'ant-noise', 'ant-physics'}
# The main figure (2026-09-19): representative columns, the rest is in `all`.
# Acrobot left out (flat on every row, under both changes); HalfCheetah
# dropped 2026-09-21 and Kinetix 2026-09-24 (tanh units are rarely dormant);
# MountainCar under noise replaced 2026-09-25, because there no RL method
# gains dormant units (PPO 0.23 -> 0.31, PBT-PPO and TRAC-PPO fall). CartPole
# under noise is the only noise setting with the RL rise, so the second column
# varies the environment instead: MountainCar under action reversal (PPO
# 0.22 -> 0.38, C-CHAIN 0.25 -> 0.33, TRAC-PPO falls, GA/ES flat at 0.4-0.5).
# CartPole under reversal has the largest rise (PPO 0.50) but would repeat the
# environment.
MAIN_COLUMNS = ('cartpole-noise', 'mountaincar-actions', 'minigrid')
FIGURE_COLUMNS = {'saturation': TANH_COLUMNS, 'saturation_lines': TANH_COLUMNS,
                  'main': set(MAIN_COLUMNS)}
# A lines-only figure with more columns than one block holds is wrapped into
# these blocks, one above the other, as continual_combined: noise over action
# reversal, MiniGrid and Kinetix in the last column. At \textwidth a block of
# five keeps the text at print size.
WRAP = (('cartpole-noise', 'acrobot-noise', 'mountaincar-noise', 'cheetah-noise', 'minigrid'),
        ('cartpole-actions', 'acrobot-actions', 'mountaincar-actions', 'cheetah-actions',
         'kinetix'))
BLOCK_GAP_IN = 0.75
WRAP_ROW_HEIGHT_IN = 0.72
# Row labels on two lines, so they fit the wrapped figure's shorter rows.
WRAP_LABEL = {'dormant_pooled': 'Dormant\nunits', 'ntk': 'NTK\nrank', 'weight': 'Weight\nRMS'}
ROW_HEIGHT_IN = 0.95
# (line row, columns) that keep their own y range instead of the row's shared
# one: on the tanh bodies a unit is rarely silent (Kinetix ~0.02 against up to
# 0.5 on gymnax), so on the shared axis their dormant curves lie on zero.
# Since 2026-09-20 the dormant row shares its linear axis with the tanh
# bodies too (it used to give them their own): Kinetix and HalfCheetah then
# read as flat near zero, which is the point.
OWN_SCALE = {}
# Dot rows that are fractions or counts: their axis starts at 0, so an
# all-zero panel does not centre on nothing.
LOG_DOTS = {'weight'}
# The absolute radius the `width_abs` row reads (one of curvature_width.ABS_SIGMA).
ABS_WIDTH_SIGMA = '0.03'
NONNEGATIVE_ROWS = {'width_abs', 'dormant', 'age', 'saturated', 'generalist', 'learned', 'width'}


def _tick(x, pos):
    """`plot_metrics_overview._compact`, with tiny values as 4e-4 so they fit."""
    if x and abs(x) < 1e-2:
        mant, exp = f'{x:.0e}'.split('e')
        return f'{mant}e{int(exp)}'
    return pmo._compact(x, pos)


# The line figures title their panels like continual_main.
panel_title = pmo.panel_title


def _symlog_tick(x, pos):
    return '0' if x == 0 else f'{x:g}'


def draw_line_panel(ax, series, methods, log=False, symlog=None, transitions=False):
    """One (line row, task) panel: each method's mean over trials against
    sub-task, with a 95% bootstrap band. A point is the agent saved at the end
    of that sub-task. `log` draws a log axis, `symlog` a symlog one with that
    linear threshold. `transitions` marks each task switch between two points
    with the dashed line of make_lineplot. Returns the methods drawn."""
    from source.metrics.continual_metrics import bootstrap_ci
    drawn = set()
    for m in methods:
        curves = series.get(m)
        if not curves:
            continue
        T = min(len(v) for v in curves)
        arr = np.stack([v[:T] for v in curves])
        ok = np.isfinite(arr).any(axis=0)
        if not ok.any():
            continue
        mean, lo, hi = np.full(T, np.nan), np.full(T, np.nan), np.full(T, np.nan)
        for t in np.flatnonzero(ok):
            col = arr[:, t][np.isfinite(arr[:, t])]
            mean[t], lo[t], hi[t] = bootstrap_ci(col)
        colour = lp.METHOD_STYLE[m]['color']
        x = np.arange(1, T + 1)
        ax.plot(x, mean, color=colour, lw=0.9)
        ax.fill_between(x, lo, hi, color=colour, alpha=0.18, lw=0)
        drawn.add(m)
    if not transitions:         # the switch lines are the only rules then
        ax.grid(True, color='0.92', lw=0.5)
    if transitions and drawn:
        T = max(min(len(v) for v in series[m]) for m in drawn)
        for b in np.arange(1.5, T):
            ax.axvline(b, color='0.6', lw=0.45, ls='--', zorder=0)
    ax.spines[['top', 'right']].set_visible(False)
    ax.tick_params(length=2, pad=1.5)
    ax.set_xticks([1, 10, 20])
    if log:
        if drawn:
            ax.set_yscale('log')
            # Decades only: minor ticks read as a black bar at this size.
            ax.yaxis.set_major_locator(LogLocator(numticks=4))
            ax.yaxis.set_minor_locator(NullLocator())
        return drawn
    if symlog is not None:
        ax.set_yscale('symlog', linthresh=symlog, linscale=0.4)
        ax.set_ylim(0, 1)
        ax.yaxis.set_major_locator(FixedLocator([0, 1e-2, 1e-1, 1]))
        ax.yaxis.set_minor_locator(NullLocator())
        ax.yaxis.set_major_formatter(FuncFormatter(_symlog_tick))
        return drawn
    ax.yaxis.set_major_locator(MaxNLocator(nbins=3))
    ax.yaxis.set_major_formatter(FuncFormatter(_tick))
    # Every line row is a fraction or an age: never below 0, and an all-zero
    # panel reads 0 to 1 rather than a symmetric band around nothing.
    top = ax.get_ylim()[1]
    ax.set_ylim(0, top if drawn and top > 1e-9 else 1)
    return drawn


def _finite(v):
    a = np.array([np.nan if x is None else float(x) for x in v], dtype=float)
    return a[np.isfinite(a)]


def late(v, k=5):
    a = _finite(v)
    return float(a[-k:].mean()) if a.size else np.nan


def early(v, k=3):
    a = _finite(v)
    return float(a[:k].mean()) if a.size else np.nan


def ratio(v):
    e = early(v)
    return late(v) / e if e and np.isfinite(e) else np.nan


def _json(path):
    return json.loads(path.read_text()) if path.exists() else None


def _cell(cells, env):
    return next((c for c in cells if c == env or c.startswith(env + '_sigma')), None)


def threshold(col_id, env_name):
    for prefix, thr in THRESHOLD.items():
        if col_id.startswith(prefix):
            return thr
    return threshold_for(env_name)


def outcomes(root, method, cell, col_id):
    """{trial: (generalist fraction, failed fraction)} from evaluation.json."""
    out = {}
    for trial_dir in sorted((root / 'continual' / method / cell).glob('trial_*')):
        blob = _json(trial_dir / 'evaluation.json')
        if not blob:
            continue
        src = next((s for s in gc.SOURCES['centroid'] if s in blob['agent_sources']), None)
        thr = threshold(col_id, blob['env'])
        if src is None or thr is None:
            continue
        own, prev, _ = gc.per_checkpoint([e for e in blob['per_task'] if e['source'] == src])
        post = [c for c in gc.classify(own, prev, thr)[1:] if c is not None]
        if post:
            out[trial_dir.name] = (np.mean([c == 'generalist' for c in post]),
                                   np.mean([c in ('stuck', 'neither') for c in post]))
    return out


def load_column(col, results=None, kept=None):
    """{row: {method: [per-trial values]}} for one column, or None. `results`
    replaces PAPER/<sub>/results; `kept` = [ES arm, PBT arm] replaces
    `keep_one_arm`'s choice."""
    col_id, _suite, sub, env = col[:4]
    res = PAPER / (results or f'{sub}/results') / 'centroid'
    ck, cw = _json(res / 'plasticity_checkpoints.json'), _json(res / 'curvature_width.json')
    pooled = _json(res.parent / 'centroid_pooled' / 'plasticity_checkpoints.json')
    absw = _json(res.parent / 'centroid_abswidth' / 'curvature_width.json')
    if ck and not pooled:
        print(f'  {sub}: no centroid_pooled pass, persistence and age left n/a')
    vals = _json(PAPER / sub / 'metrics_centroid_values.json')
    if not (ck or cw or vals):
        return None
    rows = {k: {} for k in ROWS}
    # Per-checkpoint curves for the line figure: {LINES key: {method: [array]}}.
    rows['series'] = {k: {} for k in LINES}
    # One arm a pair (ES/NES, PBT N=8/N=2), filed under the pair's first name,
    # as in the metrics overview and stability_plasticity.
    present = set((vals or {}).get('methods', []))
    for blob in (ck, cw):
        for by_m in (blob or {}).get('cells', {}).values():
            present |= set(by_m)
    if kept is None:
        remap, _kept = pmo.keep_one_arm(sub, sorted(present))
    else:
        remap = {a: (pair[0] if a == k else None)
                 for pair, k in zip(pmo.PAIRS, kept) for a in pair}

    def keep(m):
        return m not in pmo.NOT_REPORTED and remap.get(m, m) is not None

    if vals:
        ft = next((r for r in vals['rows'] if r['key'] == 'ft'), None)
        for m, v in ((ft or {}).get('values', {}).get(env) or {}).items():
            if keep(m):
                rows['ft'][remap.get(m, m)] = list(v)
    root = None
    for blob, key in ((ck, 'runs_root'), (cw, 'root')):
        if blob and root is None:
            root = REPO / blob['meta'][key]
    ck_cells = (ck or {}).get('cells', {})
    cw_cells = (cw or {}).get('cells', {})
    cell = _cell(ck_cells, env) or _cell(cw_cells, env)
    pl_cells = (pooled or {}).get('cells', {})
    aw_cells = (absw or {}).get('cells', {})
    if cell is None:
        return rows
    methods = set(ck_cells.get(cell, {})) | set(cw_cells.get(cell, {}))
    for m in sorted(filter(keep, methods)):
        ck_t = ck_cells.get(cell, {}).get(m, {}).get('trials', {})
        cw_t = cw_cells.get(cell, {}).get(m, {}).get('trials', {})
        oc = outcomes(root, m, cell, col_id) if root is not None else {}
        pl_t = pl_cells.get(cell, {}).get(m, {}).get('trials', {})
        aw_t = aw_cells.get(cell, {}).get(m, {}).get('trials', {})
        for trial in sorted(set(ck_t) | set(cw_t) | set(oc)):
            a, b, c = ck_t.get(trial), cw_t.get(trial), pl_t.get(trial)
            got = {}
            curves = {}
            if b:
                curves['ntk'] = b['ntk_logit']
                curves['width'] = b['width_0.1']
            aw_curve = aw_t.get(trial, {}).get(f'width_abs_{ABS_WIDTH_SIGMA}')
            if aw_curve:
                curves['width_abs'] = aw_curve
            if a:
                w = np.array([np.nan if x is None else float(x) for x in a['weight_rms']])
                if w.size:
                    curves['weight'] = list(w)
            if c:
                curves['dormant_pooled'] = c['dormant_fraction']
                # Index at checkpoint 0 is undefined: pad so every curve has T points.
                curves['persistence'] = [None] + list(c['persistence_index'])
                # The pass leaves age undefined where nothing is dormant; drawn
                # as 0 (no dormant unit has any age), so a curve does not break.
                curves['age'] = [0.0 if x is None else x for x in c['dormant_age']]
                if c.get('saturated_fraction') is not None:
                    curves['saturated'] = c['saturated_fraction']
            for k, v in curves.items():
                rows['series'][k].setdefault(remap.get(m, m), []).append(
                    np.array([np.nan if x is None else float(x) for x in v]))
            if c:
                got['dormant'] = late(c['dormant_fraction'])
                if c.get('saturated_fraction') is not None:
                    got['saturated'] = late(c['saturated_fraction'])
                # A run with no dormant unit at any checkpoint has no dormant
                # set to persist; its index reads 0, which would look like
                # perfect turnover.
                if _finite(c['dormant_fraction']).max(initial=0) > 0:
                    pi = _finite(c['persistence_index'])
                    got['persistence'] = float(pi.mean()) if pi.size else np.nan
                got['age'] = late([0.0 if x is None else x for x in c['dormant_age']], 1)
            if a:
                w = _finite(a['weight_rms'])
                got['weight'] = w[-1] if w.size else np.nan
            if b:
                got['ntk'] = ratio(b['ntk_logit'])
                got['width'] = late(b['width_0.1'])
            aw = aw_t.get(trial, {}).get(f'width_abs_{ABS_WIDTH_SIGMA}')
            if aw:
                got['width_abs'] = late(aw)
            if trial in oc:
                got['generalist'] = oc[trial][0]
                got['learned'] = 1.0 - oc[trial][1]
            for k, v in got.items():
                rows[k].setdefault(remap.get(m, m), []).append(v)
    return rows


def draw(stem, cols, rows, data, methods, args):
    """One figure, a column per task and a panel per (row, task). `rows` is
    a list of (kind, key), see `FIGURES`. Line rows come first and carry the
    colour legend; dot rows carry the method names and the significance note.
    Returns one summary tuple per dot drawn. No note on the figure: arrows
    and marks (one-sided Mann-Whitney U against every method of the other
    family, Holm; * p<.05, ** p<.01, *** p<.001) are for the caption."""
    fs = args.font_size
    kinds = [k for k, _ in rows]
    has_lines, has_dots = 'line' in kinds, 'dot' in kinds
    # A lines-only figure is drawn like continual_main and stability_plasticity:
    # one "Task, perturbation" title a panel, evenly spaced columns, no
    # suite headers.
    plain = has_lines and not has_dots
    blocks = [cols]
    # A block of WRAP's width is drawn at \textwidth with two-line titles,
    # whether it is one of two blocks or the whole figure.
    # A figure given its width (SIDE_CAPTION) is drawn at print size too.
    print_block = plain and (len(cols) >= len(WRAP[0]) or bool(args.width))
    if plain and len(cols) > len(WRAP[0]):
        blocks = [[c for k in keys for c in cols if c[0] == k] for keys in WRAP]
        assert sum(map(len, blocks)) == len(cols), 'a column is in no WRAP block'
        cols = [c for b in blocks for c in b]
    if args.width:
        width = args.width
    elif print_block:
        width = pmo.TEXT_WIDTH_IN
    elif plain:
        width = 0.6 + 1.45 * len(cols) + 0.35
    elif has_lines:
        width = pmo.LEFT_IN + 0.9 * len(cols)
    else:
        width = (pmo.TEXT_WIDTH_IN if len(cols) <= 5
                 else pmo.LEFT_IN + pmo.COLUMN_IN * len(cols))
    # Narrower than \textwidth: drawn to sit beside its caption (Figure 6).
    narrow = print_block and width < pmo.TEXT_WIDTH_IN
    ratios, slot = [], {}
    for b in blocks:
        ratios = []
        for i, c in enumerate(b):
            if i and c[1] != b[i - 1][1] and not plain:
                ratios.append(0.2)
            slot[c[0]] = len(ratios)
            ratios.append(1.0)
    # The colour legend wraps to the figure's width, about an inch an entry
    # (0.75 in for the short labels of a lines-only figure).
    per_entry = 0.7 if plain else 1.05
    legend_cols = max(1, min(len(methods), int((width - 0.2) / per_entry)))
    if narrow:
        legend_cols = len(methods)      # one row at the smaller legend size
    legend_rows = -(-len(methods) // legend_cols)
    top_in = ((0.45 if plain else 0.72) + 0.13 * (legend_rows - 1)) if has_lines else 0.48
    if print_block:
        top_in += 0.15 if not narrow else -0.02   # the two-line titles
    # Room for the last row's tick labels, and the sub-task label when the
    # last row is a line row. The legend for the marks is the caption's.
    bottom_in = 0.3 if kinds[-1] == 'line' else 0.18
    # Between blocks: the upper block's sub-task label and the lower one's titles.
    row_h = WRAP_ROW_HEIGHT_IN if print_block and args.row_height == ROW_HEIGHT_IN else args.row_height
    block_h = row_h * len(rows)
    height = block_h * len(blocks) + BLOCK_GAP_IN * (len(blocks) - 1) + top_in + bottom_in
    fig = plt.figure(figsize=(width, height))
    outer = GridSpec(len(blocks), 1, figure=fig,
                     left=(0.5 if narrow else 0.6 if plain else pmo.LEFT_IN) / width,
                     right=1 - (0.08 if narrow else 0.35 if plain else 0.12) / width,
                     top=1 - top_in / height, bottom=bottom_in / height,
                     hspace=BLOCK_GAP_IN / block_h)
    grids = [GridSpecFromSubplotSpec(len(rows), len(ratios), subplot_spec=outer[i],
                                     width_ratios=ratios,
                                     wspace=0.3 if has_dots else (0.45 if print_block else 0.35)
                                     if plain else 0.55,
                                     hspace=0.3 if plain else 0.35)
             for i in range(len(blocks))]
    gs_of = {c[0]: (grids[i], j, b) for i, b in enumerate(blocks) for j, c in enumerate(b)}
    last_line = max((i for i, k in enumerate(kinds) if k == 'line'), default=None)
    top_axes, summary, drawn = {}, [], set()
    # Line rows share one y range across tasks: {row: [(ax, first of its suite)]}.
    shared = {}
    for r, (kind, rk) in enumerate(rows):
        label, hib = (LINES if kind == 'line' else ROWS)[rk]
        for key, suite, sub, env, task, pert in cols:
            gs, c, block = gs_of[key]
            ax = fig.add_subplot(gs[r, slot[key]])
            d = data.get(key) or {}
            if kind == 'line':
                got = draw_line_panel(ax, (d.get('series') or {}).get(rk) or {}, methods,
                                      log=rk in LOG_LINES, symlog=SYMLOG_LINES.get(rk),
                                      transitions=narrow)
                drawn |= got
                filled = bool(got)
                if filled and rk not in UNSHARED_LINES and key not in OWN_SCALE.get(rk, ()):
                    shared.setdefault(r, []).append(
                        (ax, c == 0 or (block[c - 1][1] != suite and not narrow)))
                if r != last_line:
                    ax.set_xticklabels([])
                elif plain or c == len(block) // 2:
                    ax.set_xlabel('Task', labelpad=1)
            else:
                cell = {m: v for m, v in (d.get(rk) or {}).items() if m in methods}
                filled = any(np.isfinite(np.asarray(v, float)).any() for v in cell.values())
                _, points = draw_panel(ax, cell if filled else {}, methods, rk, hib, fs,
                                       labels=c == 0)
                summary += [(label.replace('\n', ' '), task, pert, *p) for p in points]
                if filled:
                    if rk in LOG_DOTS:
                        ax.set_xscale('log')
                    elif rk in NONNEGATIVE_ROWS:
                        right = ax.get_xlim()[1]
                        ax.set_xlim(0, right if right > 1e-9 else 1)
                    if rk not in LOG_DOTS:
                        ax.xaxis.set_major_locator(MaxNLocator(nbins=3, min_n_ticks=2))
                        ax.xaxis.set_major_formatter(FuncFormatter(_tick))
                    ax.tick_params(axis='x', pad=1.5)
                else:
                    ax.set_xticks([])
                    ax.spines['bottom'].set_color('0.85')
            if not filled:
                ax.set_yticks([])
                note = ('ReLU: cannot\nsaturate' if rk == 'saturated' and key not in TANH_COLUMNS
                        else 'n/a')
                ax.text(0.5, 0.5, note, transform=ax.transAxes, ha='center',
                        va='center', color=pmo.MUTED, fontsize=fs - 1)
            if c == 0:
                # At a fixed distance from the page edge, so line rows (tick
                # numbers) and dot rows (method names) line up.
                if print_block:
                    label = WRAP_LABEL.get(rk, label)
                ax.annotate(label + (' ↑' if hib else ' ↓'),
                            xy=((0.12 if plain else 0.17) / width, 0.5),
                            xycoords=('figure fraction', 'axes fraction'),
                            rotation=90, ha='center', va='center')
            if r == 0 and plain:
                title = panel_title(suite, task, pert)
                if print_block:
                    title = title.replace(', ', ',\n').replace('MiniGrid ', 'MiniGrid\n')
                ax.set_title(title, pad=3, linespacing=1.1)
                top_axes[key] = ax
            elif r == 0:
                ax.set_title(task, pad=fs + 3)
                ax.annotate(pert, (0.5, 1), xycoords='axes fraction', xytext=(0, 2),
                            textcoords='offset points', ha='center', va='bottom',
                            color=pmo.MUTED, fontsize=fs - 0.5)
                top_axes[key] = ax
    for r, panels in shared.items():
        top = max(ax.get_ylim()[1] for ax, _ in panels)
        bottom = (min(ax.get_ylim()[0] for ax, _ in panels)
                  if rows[r][1] in LOG_LINES else 0)
        for ax, first in panels:
            ax.set_ylim(bottom, top)
            if not first:
                ax.tick_params(labelleft=False)
    pt = 1 / 72 / height
    groups = []
    for c in cols:
        if groups and groups[-1][0] == c[1]:
            groups[-1][1].append(c[0])
        else:
            groups.append((c[1], [c[0]]))
    for suite, members in ([] if plain else groups):
        first, last = (top_axes[k].get_position() for k in (members[0], members[-1]))
        y = first.y1 + (2 * fs + 9) * pt
        fig.add_artist(Line2D([first.x0, last.x1], [y, y], lw=0.6, color='0.3'))
        fig.text((first.x0 + last.x1) / 2, y + 1.5 * pt, suite, ha='center',
                 va='bottom', fontsize=fs + 0.5)
    if has_lines:
        order = [m for m in methods if m in drawn]
        fig.legend([Line2D([], [], color=lp.METHOD_STYLE[m]['color'], lw=1.6) for m in order],
                   [lp.METHOD_STYLE[m]['label'] for m in order], loc='upper center',
                   ncol=min(len(order), legend_cols), frameon=False,
                   fontsize=fs - 1.5 if narrow else fs,   # beside a caption: below the tick size
                   bbox_to_anchor=(0.5, 1.0),
                   handlelength=1.0 if narrow else 1.6, handletextpad=0.4 if narrow else 0.8,
                   columnspacing=0.8 if narrow else 1.2)
    for ext in ('pdf', 'png'):
        fig.savefig(out_path(stem).with_suffix(f'.{ext}'), dpi=300)
    plt.close(fig)
    print(f'wrote {out_path(stem)}.pdf/.png ({width:.2f} x {height:.2f} in)')
    return summary


def cell_gaps(cols, data):
    """Per task: (label, suite, width RL / NE, generalist NE - RL, learned),
    family means of per-method means; `learned` is the LOWER of the two
    families' fraction of checkpoints that learnt the shown sub-task."""
    out = []
    for key, suite, sub, env, task, pert in cols:
        d = data.get(key) or {}
        fam = {}
        for rk in ('width', 'generalist', 'learned'):
            for m, v in (d.get(rk) or {}).items():
                a = np.asarray(v, float)
                a = a[np.isfinite(a)]
                if a.size:
                    fam.setdefault((rk, lp.FAMILY.get(m)), []).append(a.mean())
        need = [('width', 'ne'), ('width', 'rl'), ('generalist', 'ne'), ('generalist', 'rl')]
        if all(k in fam for k in need):
            learned = min((np.mean(fam[('learned', f)]) if ('learned', f) in fam else np.nan)
                          for f in ('ne', 'rl'))
            out.append((f'{task}\n{pert}', suite,
                        np.mean(fam[('width', 'rl')]) / np.mean(fam[('width', 'ne')]),
                        np.mean(fam[('generalist', 'ne')]) - np.mean(fam[('generalist', 'rl')]),
                        learned))
    return out


def draw_scatter(stem, gaps, args):
    from scipy.stats import spearmanr
    fs = args.font_size + 1
    fig, ax = plt.subplots(figsize=(3.2, 2.6))
    colours = {'gymnax': '#4a6fa5', 'Brax': '#c0504d', 'MiniGrid': '#6b8e23',
               'Kinetix': '#8064a2'}
    # Hollow: a task where one family learns the sub-task it is shown in fewer
    # than LEARNED_MIN of the checkpoints, so its generalist gap is mostly
    # that family failing to learn, not a choice between kinds of solution.
    # Drawn, not dropped, and the correlation is reported both ways.
    for label, suite, x, y, learned in gaps:
        c = colours.get(suite, '0.3')
        filled = np.isfinite(learned) and learned >= LEARNED_MIN
        ax.scatter([x], [y], s=14, zorder=3, color=c if filled else 'white',
                   edgecolors=c, linewidths=0.9)
    ax.set_xscale('log')
    # Labels: a suite with one task is named by the suite; the rest by task
    # and perturbation. Each takes the first offset whose box does not overlap
    # a label already placed.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    placed = []
    offsets = [((4, 2), 'left'), ((4, -8), 'left'), ((-4, 2), 'right'),
               ((-4, -8), 'right'), ((4, 9), 'left'), ((-4, 9), 'right')]
    for label, suite, x, y, _learned in sorted(gaps, key=lambda g: -g[3]):
        text = suite if suite in ('MiniGrid', 'Kinetix') else label.replace('\n', ', ')
        for (dx, dy), ha in offsets:
            ann = ax.annotate(text, (x, y), xytext=(dx, dy), textcoords='offset points',
                              fontsize=fs - 3, color='0.35', ha=ha)
            box = ann.get_window_extent(renderer)
            if not any(box.overlaps(b) for b in placed):
                break
            ann.remove()
        else:
            ann = ax.annotate(text, (x, y), xytext=offsets[0][0],
                              textcoords='offset points', fontsize=fs - 3, color='0.35')
            box = ann.get_window_extent(renderer)
        placed.append(box)
    present = [s for s in colours if any(g[1] == s for g in gaps)]
    ax.legend(handles=[Line2D([], [], ls='', marker='o', ms=4, color=colours[s], label=s)
                       for s in present],
              loc='upper left', fontsize=fs - 3, frameon=False, handletextpad=0.2)
    ax.axhline(0, color='0.6', lw=0.5)
    ax.axvline(1, color='0.6', lw=0.5)
    ax.set_xlabel('Action change under weight noise, RL / NE')
    ax.set_ylabel('Generalist checkpoints, NE − RL')
    if len(gaps) >= 4:
        rho, p = spearmanr([g[2] for g in gaps], [g[3] for g in gaps])
        ax.set_title(f'Spearman ρ = {rho:+.2f} (p = {p:.2g}, {len(gaps)} tasks)',
                     fontsize=fs - 1)
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(color='0.92', lw=0.5, zorder=0)
    fig.tight_layout()
    for ext in ('pdf', 'png'):
        fig.savefig(OUT / f'{stem}.{ext}', dpi=300)
    plt.close(fig)
    print(f'wrote {OUT / stem}.pdf/.png')


def extract_final(cols):
    """Read the final figure's columns from the passes and write FINAL_DATA:
    `<column>|<row>|<arm>` -> trials x checkpoints (NaN-padded) in the .npz;
    kept arms, pass files and the extraction date in the .json."""
    import datetime
    kept_by_tree = json.loads((FINAL / 'data' / 'continual_main.json').read_text())['kept']
    rows = list(dict.fromkeys(k for f in FINAL_FIGURES for _, k in FIGURES[f][1]))
    arrays, meta = {}, {'kept': {}, 'passes': {}, 'rows': rows}
    for col in cols:
        key, sub = col[0], col[2]
        kept = kept_by_tree[FINAL_TREE[sub]]
        results = FINAL_RESULTS.get(sub, f'{sub}/results')
        d = load_column(col, results, kept) or {}
        meta['kept'][key] = kept
        meta['passes'][key] = [f'{results}/{p}' for p in (
            'centroid/plasticity_checkpoints.json', 'centroid/curvature_width.json',
            'centroid_pooled/plasticity_checkpoints.json')]
        for rk in rows:
            for m, curves in ((d.get('series') or {}).get(rk) or {}).items():
                T = max(len(v) for v in curves)
                arrays[f'{key}|{rk}|{m}'] = np.stack(
                    [np.pad(v, (0, T - len(v)), constant_values=np.nan) for v in curves])
        print(f'{key:18s} kept {"/".join(map(str, kept))}: '
              + '; '.join(f'{rk} ' + ' '.join(f'{m}:{len(c)}' for m, c in
                                               sorted(((d.get("series") or {}).get(rk) or {}).items()))
                          for rk in rows))
    meta['extracted'] = datetime.date.today().isoformat()
    FINAL_DATA.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(f'{FINAL_DATA}.npz', **arrays)
    pathlib.Path(f'{FINAL_DATA}.json').write_text(json.dumps(meta, indent=1) + '\n')
    print(f'wrote {FINAL_DATA}.npz, {FINAL_DATA}.json')


def load_final():
    """`{column: {'series': {row: {arm: [curve a trial]}}}}` from FINAL_DATA."""
    if not pathlib.Path(f'{FINAL_DATA}.npz').exists():
        sys.exit(f'no {FINAL_DATA}.npz: run with --extract first')
    data = {}
    with np.load(f'{FINAL_DATA}.npz') as npz:
        for name in npz.files:
            key, rk, m = name.split('|')
            data.setdefault(key, {'series': {}})['series'].setdefault(rk, {})[m] = [
                t[: np.flatnonzero(np.isfinite(t))[-1] + 1] if np.isfinite(t).any() else t
                for t in npz[name]]
    return data


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--set', default='two', choices=list(pmo.SETS))
    ap.add_argument('--figure', nargs='+', default=None, choices=list(FIGURES),
                    help='default: every figure; with --set main, the final figures')
    ap.add_argument('--width', type=float, default=None)
    ap.add_argument('--row-height', type=float, default=ROW_HEIGHT_IN)
    ap.add_argument('--font-size', type=float, default=7.0)   # print size: the final figures go in at their width
    ap.add_argument('--extract', action='store_true',
                    help='--set main --figure main: re-read the passes into the saved data first')
    args = ap.parse_args()
    if args.figure is None:
        args.figure = list(FINAL_FIGURES) if args.set == 'main' else list(FIGURES)
    # One entry a pair: `pbt` is PBT-PPO, whichever N the column kept.
    lp.METHOD_STYLE['pbt'] = {**lp.METHOD_STYLE['pbt'], 'label': 'PBT-PPO'}
    plt.rcParams.update({
        'font.size': args.font_size, 'axes.titlesize': args.font_size + 0.5,
        'xtick.labelsize': args.font_size - 1, 'ytick.labelsize': args.font_size - 0.5,
        'axes.linewidth': 0.5, 'xtick.major.width': 0.5, 'xtick.major.size': 2,
    })
    (OUT / 'main').mkdir(parents=True, exist_ok=True)
    suffix = SUFFIX.get(args.set, f'_{args.set}')
    cols = [c for c in pmo.SETS[args.set][1] if c[0] not in DROPPED]
    if args.set == 'main' and set(args.figure) <= set(FINAL_FIGURES):
        # The final figures: continual_main's arms, from the saved data only.
        if args.extract:
            extract_final(cols)
        data = load_final()
        union = {m for d in data.values() for rows in d['series'].values() for m in rows}
        for name in args.figure:
            stem, row_keys = FIGURES[name]
            fig_cols = [c for c in cols if c[0] in FIGURE_COLUMNS.get(name, {c[0]})]
            fig_args = args
            if name in SIDE_CAPTION and args.width is None:
                w, rh, fs = SIDE_CAPTION[name]
                fig_args = argparse.Namespace(**{**vars(args), 'width': w, 'font_size': fs,
                                                 'row_height': rh})
            with plt.rc_context({'font.size': fig_args.font_size,
                                 'axes.titlesize': fig_args.font_size + 0.5,
                                 'xtick.labelsize': fig_args.font_size - 1,
                                 'ytick.labelsize': fig_args.font_size - 0.5}):
                draw(stem + suffix, fig_cols, row_keys, data,
                     [m for m in lp.METHOD_ORDER if m in union], fig_args)
        return 0
    if args.extract:
        sys.exit('--extract builds the final figures only: --set main --figure '
                 + ' '.join(FINAL_FIGURES))
    data = {}
    for col in cols:
        data[col[0]] = load_column(col)
        got = {k: sorted(v) for k, v in (data[col[0]] or {}).items() if v and k != 'series'}
        print(f'{col[0]:18s} ' + ('; '.join(f'{k}: {len(v)} arms' for k, v in got.items())
                                  if got else 'nothing'))
    union = {m for d in data.values() if d for k, rows in d.items() if k != 'series'
             for m in rows}
    methods = [m for m in lp.METHOD_ORDER if m in union]
    md = [f'# plasticity_overview{suffix}', '',
          'Mean over trials [95% bootstrap CI], n, NE-vs-RL mark. See the '
          'docstring of `scripts/analysis/plot_plasticity_overview.py` for each row.', '',
          '| Row | Task | Perturbation | Method | mean | lo | hi | n | mark |',
          '|---|---|---|---|---|---|---|---|---|']
    for name in args.figure:
        stem, row_keys = FIGURES[name]
        fig_cols = [c for c in cols if c[0] in FIGURE_COLUMNS.get(name, {c[0]})]
        if name != 'specialists':
            for col in PLASTIC_EXTRA.get(args.set, []):
                if col[0] not in data:
                    data[col[0]] = load_column(col)
                fig_cols.append(col)
        for row, task, pert, m, mean, lo, hi, n, mark in draw(
                stem + suffix, fig_cols, row_keys, data, methods, args):
            md.append(f'| {row} | {task} | {pert} | {lp.METHOD_STYLE.get(m, {}).get("label", m)} '
                      f'| {mean:.3g} | {lo:.3g} | {hi:.3g} | {n} | {mark} |')
        if name == 'specialists':
            extra = SCATTER_EXTRA.get(args.set, [])
            for col in extra:
                if col[0] not in data:
                    data[col[0]] = load_column(col)
            gaps = cell_gaps(list(cols) + extra, data)
            draw_scatter(stem + '_scatter' + suffix, gaps, args)
            md += ['', f'Scatter: hollow when the lower family learns the shown sub-task in '
                   f'fewer than {LEARNED_MIN:g} of its checkpoints.', '',
                   '| Task | Suite | width RL / NE | generalist NE - RL | learned (lower family) |',
                   '|---|---|---|---|---|']
            md += [f'| {g[0].replace(chr(10), ", ")} | {g[1]} | {g[2]:.2f} | {g[3]:+.2f} '
                   f'| {g[4]:.2f} |' for g in gaps]
    # The table has dot rows only; a lines-only run leaves it alone.
    if any(kind == 'dot' for n in args.figure for kind, _ in FIGURES[n][1]):
        (OUT / f'plasticity_overview{suffix}.md').write_text('\n'.join(md) + '\n')
        print(f'wrote {OUT}/plasticity_overview{suffix}.md')
    return 0


if __name__ == '__main__':
    sys.exit(main())
