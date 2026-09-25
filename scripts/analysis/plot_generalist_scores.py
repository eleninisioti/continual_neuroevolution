"""Generalist or switching specialist, without a solved threshold: every
two-sub-task family, each method as one point (score on the sub-task it was
just trained on, score on the one before it). A final paper figure.

    # re-read the runs (only when they change)
    .venv/bin/python scripts/analysis/plot_generalist_scores.py --extract
    # redraw from visuals/final/data/
    .venv/bin/python scripts/analysis/plot_generalist_scores.py

    -> paper/visuals/final/generalist_scores_centroid.{pdf,png,md}
       paper/visuals/final/data/generalist_scores_centroid.json

The runs are read through the paper's symlink trees (GRID), one root a family:
gymnax `{noise,physics,actions}_2task`, MiniGrid, and HalfCheetah
`{noise,physics,actions}_2task`. The HalfCheetah noise cell is offset 0.5, the
width continual_main's 10-sub-task panel uses, with the RL arms at the ant PPO
shape as there. The HalfCheetah friction cell has only the repro2-shape RL
runs, and no PBT.

Scores for the centroid saved at the end of each phase:

    gymnax, MiniGrid   `evaluation.json` (source.studies.evaluate_continual,
                       100 fresh episodes): `returns` on the sub-task just
                       trained, `prev_returns` on the previous one, as
                       generalist_checkpoints.py reads them
    HalfCheetah        the training records: `centroid_task<k>` at the last
                       generation of each phase, as plot_stability_plasticity.py
                       reads them (the CLUSTER runs were never post-evaluated)

Arms: ES is NES (CLAUDE.md: NES on every suite but Kinetix); PBT-PPO is
whichever of N=8 and N=2 has the higher Cum. elite over the family's cells
(es_arm.pick), as in continual_main. No DNS. MiniGrid keeps only the
checkpoints that end an 8x8 phase (plot_generalist_outcomes.ONE_DIRECTION), one
point a method: its rooms are nested, so the other direction is a generalist
for every method.

Per trial, the post-switch checkpoints are averaged into one (shown, previous)
pair of returns; the point is the mean over trials with a 95% percentile-
bootstrap interval on each axis, the small marks the trials. On the dashed
diagonal an agent scores as well on the previous sub-task as on the one it
was trained on; the vertical distance below it is what the switch cost.

Each method is classed by where its mean lands, relative to the untrained
network's return (FLOOR, as in plot_stability_plasticity.py):

    learned      shown - untrained >= LEARNED x (best - untrained), `best` the
                 highest mean shown return any method reaches in the panel
    kept         (previous - untrained) / (shown - untrained)
    frozen       see below                         white marker with an x
    generalist   learned and kept >= KEPT          filled marker
    switching    learned and kept <  KEPT          hollow (white) marker
    not learned  the rest                          grey-filled marker, faded CI

KEPT = 0.9 and LEARNED = 0.5 (--kept, --learned). The dotted line is kept =
KEPT and the grey band is "not learned".

A frozen specialist only ever solves one sub-task. Pooling hides it: it is
(good, bad) at the end of one sub-task's phases and (bad, good) at the end of
the other's, and the average sits on the diagonal (ES on Acrobot and PPO on
MountainCar under action reversal would read as generalists). So the class is
decided on the two directions apart, the checkpoints ending a sub-task-1 phase
and those ending a sub-task-2 phase, each against its own learned bar: frozen
when one direction did not learn while the sub-task it keeps (its previous)
is above the bar, and the other learned. The point drawn is still the pooled
one. MiniGrid has one direction, so it cannot be frozen.

MiniGrid is drawn in its own row (Physics), and a separate column right of
the grid holds a threshold-free summary, performance profiles as in rliable.
Since 2026-09-24 the grid is the appendix figure (`_grid` stem: the two-task
points moved into the trade-off figure) and the main-text figure is the two
profiles alone, side by side. Per trial and panel, a score is
rescaled to (score - untrained) / (best - untrained), `best` the highest mean
shown return of any method and direction in the panel, and taken in the
trial's worse direction:

    learns   the shown score, worse direction: low for a method that did not
             learn and for a frozen specialist
    keeps    the previous score, worse direction: low as well for a
             switching specialist; high only for a generalist

A curve is the fraction of runs scoring at least tau, each panel weighted
equally (over the panels the method has runs in), with a 95% band from
resampling trials within panels. The area under it is the mean score (the
summary table). A generalist is high on both; a switching specialist high on
learns only; a frozen specialist drops on learns where it froze.
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
from matplotlib.colors import to_rgba                       # noqa: E402
from matplotlib.gridspec import GridSpec                    # noqa: E402
from matplotlib.lines import Line2D                         # noqa: E402
from matplotlib.patches import Patch                        # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import es_arm                                               # noqa: E402
import generalist_checkpoints as gc                         # noqa: E402
import plot_generalist_outcomes as pgo                      # noqa: E402
from plot_continual_lineplots import PROJECT, FINAL, complete_arms  # noqa: E402
from plot_stability_plasticity import FLOOR                 # noqa: E402

METHOD_STYLE = pgo.METHOD_STYLE
COLUMNS, ARMS, NE_ARMS = pgo.COLUMNS, pgo.ARMS, pgo.NE_ARMS
MUTED = pgo.MUTED
DATA = FINAL / 'data' / 'generalist_scores_centroid.json'
PROFILES = FINAL / 'data' / 'generalist_profiles.json'   # the drawn profiles, for Figure 2
STEM = FINAL / 'generalist_scores_centroid'          # the two profiles (main text)
STEM_GRID = FINAL / 'generalist_scores_centroid_grid'  # the 3 x 5 grid + profiles (appendix)
ES = 'nes'
READ = ['ga', ES, 'ppo', 'trac', 'redo', 'cchain', *es_arm.PBT_ARMS]
# (row label, [(data root, cell, what the switch changes) or None] a column),
# columns as pgo.COLUMNS. Rows are what the switch touches; values are the run
# configs' (noise width, physics multiplier, task.options).
GRID = [
    ('Noise', [
        ('paper/gymnax/data/noise_2task', 'CartPole_v1_sigma0.5', 'offset 0.5'),
        ('paper/gymnax/data/noise_2task', 'Acrobot_v1_sigma0.5', 'offset 0.5'),
        ('paper/gymnax/data/noise_2task', 'MountainCar_v0_sigma0.05', 'offset 0.05'),
        ('paper/mjx/cheetah/data/noise_2task', 'cheetah_noise', 'offset 0.5'),
        None,
    ]),
    ('Physics', [
        ('paper/gymnax/data/physics_2task', 'CartPole_v1_sigma1.0', 'pole length ×3'),
        ('paper/gymnax/data/physics_2task', 'Acrobot_v1_sigma1.0', 'link mass ×1.15'),
        ('paper/gymnax/data/physics_2task', 'MountainCar_v0_sigma1.0', 'gravity ×1.5'),
        ('paper/mjx/cheetah/data/physics_2task', 'cheetah_friction', 'ground friction'),
        ('paper/minigrid/data', 'MiniGrid_8x8_16x16', 'room 16x16 → 8x8'),
    ]),
    ('Action reversal', [
        ('paper/gymnax/data/actions_2task', 'CartPole_v1_sigma1.0', 'order reversed'),
        ('paper/gymnax/data/actions_2task', 'Acrobot_v1_sigma1.0', 'order reversed'),
        ('paper/gymnax/data/actions_2task', 'MountainCar_v0_sigma1.0', 'order reversed'),
        # Continuous controls: the sub-task negates the action (envs/mjx.py).
        ('paper/mjx/cheetah/data/actions_2task', 'cheetah_action', 'sign flipped'),
        None,
    ]),
]
DIAGONAL = '0.72'
WEAK_BAND = '0.94'
# The label of a method's mean (see the docstring) is the marker's fill:
# colour, white or grey. A faded filled marker read as a generalist.
NOT_LEARNED_FILL = '0.82'
MARKER = {'generalist': {'mec': 'white', 'mew': 0.5},
          'switching': {'mfc': 'white', 'mew': 1.1},
          'frozen': {'mfc': 'white', 'mew': 1.1},
          'not learned': {'mfc': NOT_LEARNED_FILL, 'mew': 1.1}}
# CI arms of a method that did not learn, as a fraction of full opacity.
NOT_LEARNED_CI = 0.4
N_BOOT = 2000
CLASSES = ('generalist', 'switching', 'frozen', 'not learned')
CLASS_NAME = {'generalist': 'Generalist', 'switching': 'Switching specialist',
              'frozen': 'Frozen specialist', 'not learned': 'Not learned'}
PROFILE = (('learns', 0, 'Current task', 'just trained'),
           ('keeps', 1, 'Previous task', 'previous'))
TAU = np.linspace(0, 1, 101)
N_BOOT_PROFILE = 500


def from_records(trial):
    """(shown, previous) a phase from the training records; previous is None
    on the first phase."""
    records = json.loads((trial / 'training_metrics.json').read_text())
    tasks = [r['task'] for r in records]
    ends = [i for i in range(len(tasks) - 1) if tasks[i + 1] != tasks[i]] + [len(tasks) - 1]
    shown = [float(records[e][f'centroid_task{tasks[e]}']) for e in ends]
    prev = [None] + [float(records[e][f'centroid_task{tasks[p]}'])
                     for p, e in zip(ends[:-1], ends[1:])]
    return shown, prev


def from_evaluation(trial):
    """(shown, previous) a phase from evaluation.json, or None without one."""
    path = trial / 'evaluation.json'
    if not path.exists():
        return None
    blob = json.loads(path.read_text())
    src = next(s for s in gc.SOURCES['centroid'] if s in blob['agent_sources'])
    own, prev, fallback = gc.per_checkpoint(
        [e for e in blob['per_task'] if e['source'] == src])
    assert not fallback, f'{path} has no prev_returns; evaluate again'
    return own.tolist(), [None if np.isnan(v) else float(v) for v in prev]


def extract():
    trees = {}
    for _, cells in GRID:
        for spec in cells:
            if spec:
                trees.setdefault(spec[0], []).append(spec[1])
    meta = {'panels': {}, 'pbt_cum_elite': {}, 'sources': {},
            'extracted': datetime.date.today().isoformat()}
    for tree, cells in trees.items():
        root = PROJECT / tree
        arms = [m for m in complete_arms(root, cells) if m in READ]
        pbt = [a for a in es_arm.PBT_ARMS if a in arms]
        if len(pbt) == 2:
            cum = es_arm.load(root, 'continual', cells, arms=es_arm.PBT_ARMS)
            pbt = [es_arm.pick(cum, es_arm.PBT_ARMS)[0]]
            meta['pbt_cum_elite'][tree] = {
                env: {a: float(np.mean(v)) for a, v in by.items()} for env, by in cum.items()}
        kept = {'es': ES, 'pbt': pbt[0] if pbt else None}
        reader = from_records if tree.startswith('paper/mjx') else from_evaluation
        for cell in cells:
            panel = {'kept': kept, 'arms': {},
                     'scores': 'training records' if reader is from_records
                     else 'evaluation.json'}
            for m in arms:
                if m in es_arm.PBT_ARMS and m != kept['pbt']:
                    continue
                row = {ES: 'es', kept['pbt']: 'pbt'}.get(m, m)
                trials = []
                for trial in sorted((root / 'continual' / m / cell).glob('trial_*'),
                                    key=lambda p: int(p.name.split('_')[1])):
                    if not (trial / 'training_metrics.json').exists():
                        continue
                    got = reader(trial)
                    if got is None:
                        sys.exit(f'{trial} has no evaluation.json: run '
                                 'source.studies.evaluate_continual on it')
                    trials.append({'shown': got[0], 'previous': got[1],
                                   'run': str(trial.resolve().relative_to(PROJECT))})
                if trials:
                    panel['arms'][row] = {'arm': m, 'trials': trials}
            meta['panels'][f'{tree}|{cell}'] = panel
            print(f'{tree:38s} {cell:26s} ' + ' '.join(
                f'{r}={len(v["trials"])}' for r, v in panel['arms'].items()))
        for arm_dir in sorted((root / 'continual').iterdir()):
            if arm_dir.is_symlink():
                meta['sources'][f'{tree}/continual/{arm_dir.name}'] = str(
                    arm_dir.resolve().relative_to(PROJECT))
            else:     # an arm linked cell by cell
                for c in sorted(arm_dir.iterdir()):
                    meta['sources'][f'{tree}/continual/{arm_dir.name}/{c.name}'] = str(
                        c.resolve().relative_to(PROJECT))
    DATA.parent.mkdir(parents=True, exist_ok=True)
    DATA.write_text(json.dumps(meta, indent=1) + '\n')
    print(f'wrote {DATA}')


def trial_scores(trials, cell, parity=None):
    """(n, 2): per trial, the mean (shown, previous) over its post-switch
    checkpoints (one direction only on MiniGrid), or over those that end a
    phase of `parity` (sub-task = phase mod 2); None for a direction MiniGrid
    drops."""
    keep = pgo.ONE_DIRECTION.get(cell)
    if keep is not None and parity not in (None, keep):
        return None
    want = keep if parity is None else parity
    out = []
    for t in trials:
        pairs = [(s, p) for i, (s, p) in enumerate(zip(t['shown'], t['previous']))
                 if p is not None and (want is None or i % 2 == want)]
        out.append(np.mean(pairs, axis=0))
    return np.array(out).reshape(-1, 2)


def boot(v, rng):
    means = rng.choice(v, (N_BOOT, v.size)).mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(v.mean()), float(lo), float(hi)


def outcome(xm, ym, floor, best, kept=0.9, learned=0.5):
    """(learned share, kept share, label) of a mean (shown, previous) point."""
    got = (xm - floor) / (best - floor)
    keep = (ym - floor) / (xm - floor) if xm > floor else float('nan')
    return got, keep, ('not learned' if got < learned else
                       'generalist' if keep >= kept else 'switching')


def classify(panel, cell, kept=0.9, learned=0.5):
    """`{method: class}` of a saved panel, the figure's rule: from the two
    directions apart, `frozen` when one direction did not learn while the
    sub-task it keeps is learned and the other direction learned; otherwise
    `not learned`, `generalist` or `switching` from the pooled means
    (`outcome`). plot_stability_plasticity reads it too: a frozen specialist
    has nothing to forget, so it cannot be ringed there."""
    floor = FLOOR[cell.split('_sigma')[0]]
    pooled = {m: trial_scores(v['trials'], cell) for m, v in panel['arms'].items()}
    split = {(m, d): trial_scores(v['trials'], cell, d)
             for m, v in panel['arms'].items() for d in (0, 1)}
    best = max(s[:, 0].mean() for s in pooled.values())
    best_dir = max(s[:, 0].mean() for s in split.values() if s is not None)
    bar_dir = floor + learned * (best_dir - floor)
    out = {}
    for m in panel['arms']:
        dirs = [split[m, d].mean(axis=0) for d in (0, 1) if split[m, d] is not None]
        missed = [d for d in dirs if d[0] < bar_dir]
        frozen = len(dirs) == 2 and len(missed) == 1 and missed[0][1] >= bar_dir
        xm, ym = pooled[m].mean(axis=0)
        out[m] = 'frozen' if frozen else outcome(xm, ym, floor, best, kept, learned)[2]
    return out


def profile(panels, tau):
    """Fraction of runs >= each tau, the panels (per-trial arrays) weighted equally."""
    return np.mean([(v[:, None] >= tau).mean(axis=0) for v in panels], axis=0)


def summary(axes, runs, rng, curves=None):
    """The column right of the grid: performance profiles of the rescaled worse-
    direction score on the sub-task just trained (top) and on the previous
    one (bottom). Returns the mean scores for the table; `curves`, if given,
    receives every drawn curve (`{key: {method: {mean, lo, hi, panels}}}`)."""
    means = {}
    for ax, (key, col_i, title, xlabel) in zip(axes, PROFILE):
        for m in ARMS:
            panels = [v[:, col_i] for v in runs[m]]
            if not panels:
                continue
            col = METHOD_STYLE[m]['color']
            ne = m in NE_ARMS
            reps = np.array([profile([v[rng.integers(0, v.size, v.size)] for v in panels], TAU)
                             for _ in range(N_BOOT_PROFILE)])
            lo, hi = np.percentile(reps, [2.5, 97.5], axis=0)
            mean = profile(panels, TAU)
            ax.fill_between(TAU, lo, hi, color=col, alpha=0.15, lw=0, zorder=1)
            ax.plot(TAU, mean, color=col, lw=1.3 if ne else 0.9, zorder=3 if ne else 2)
            if curves is not None:
                curves.setdefault(key, {})[m] = {
                    'mean': mean.tolist(), 'lo': lo.tolist(), 'hi': hi.tolist(),
                    'panels': len(panels)}
            means.setdefault(m, {})[key] = (float(np.mean([np.clip(v, 0, 1).mean()
                                                           for v in panels])),
                                            len(panels))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
        ax.set_title(title, pad=3)   # what each profile scores is the caption's
        ax.set_xlabel('τ (rescaled score)', labelpad=1)
        ax.set_box_aspect(1)
        ax.set_ylabel('runs ≥ τ', labelpad=1)
        ax.xaxis.set_major_locator(plt.MultipleLocator(0.5))
        ax.yaxis.set_major_locator(plt.MultipleLocator(0.5))
        ax.spines[['top', 'right']].set_visible(False)
        ax.tick_params(pad=1.5)
    return means


def label_of(m):
    return 'ES' if m == 'es' else 'PBT-PPO' if m == 'pbt' else METHOD_STYLE[m]['label']


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--extract', action='store_true',
                    help='read the runs and write the data file first')
    ap.add_argument('--width', type=float, default=5.5)   # \textwidth: sizes are print sizes
    ap.add_argument('--profile-layout', choices=('row', 'column'), default='row',
                    help='the standalone profiles side by side with a legend, or stacked '
                         'without one (the paper draws them in Figure 2 from '
                         'data/generalist_profiles.json since 2026-09-24)')
    ap.add_argument('--profile-width', type=float, default=None,
                    help='width in inches of the profiles-only figure (default 1.55 '
                         'stacked, 3.4 in a row)')
    ap.add_argument('--font-size', type=float, default=7.0)
    ap.add_argument('--kept', type=float, default=0.9,
                    help='generalist: keeps at least this share of its gain on the previous sub-task')
    ap.add_argument('--learned', type=float, default=0.5,
                    help="learned: reaches this share of the panel best's gain over untrained")
    args = ap.parse_args()
    if args.extract:
        extract()
    if not DATA.exists():
        sys.exit(f'no {DATA}: run with --extract first')
    meta = json.loads(DATA.read_text())

    fs = args.font_size
    plt.rcParams.update({
        'font.size': fs, 'axes.titlesize': fs + 0.5, 'xtick.labelsize': fs - 1,
        'ytick.labelsize': fs - 1, 'axes.linewidth': 0.5,
        'xtick.major.width': 0.5, 'ytick.major.width': 0.5,
        'xtick.major.size': 2, 'ytick.major.size': 2,
    })
    rng = np.random.default_rng(0)
    width = args.width
    ncol = len(COLUMNS)
    # The scatter grid, then a gap, then the profiles in a column of their own
    # (under MiniGrid they read as MiniGrid panels).
    left_in, right_in, top_in, bottom_in = 0.6, 0.12, 0.42, 0.35
    gap_in = 0.42                  # grid -> profiles: their y label and a margin
    profile_w = 1.35               # a profile's width, in scatter panels
    wspace, hspace = 0.6, 0.7      # room for the tick labels and the titles
    panel_in = ((width - left_in - right_in - gap_in)
                / (ncol + profile_w + (ncol - 1) * wspace))
    height = len(GRID) * panel_in * (1 + hspace) - hspace * panel_in + top_in + bottom_in
    grid_right_in = left_in + panel_in * (ncol + (ncol - 1) * wspace)
    fig = plt.figure(figsize=(width, height))
    gs = GridSpec(len(GRID), ncol, figure=fig,
                  left=left_in / width, right=grid_right_in / width,
                  top=1 - top_in / height, bottom=bottom_in / height,
                  wspace=wspace, hspace=hspace)
    gs_profile = GridSpec(len(PROFILE), 1, figure=fig,
                          left=1 - (right_in + profile_w * panel_in) / width,
                          right=1 - right_in / width,
                          top=1 - top_in / height, bottom=bottom_in / height, hspace=0.5)
    md = ['# generalist_scores_centroid', '',
          'Per method: mean over trials of the post-switch centroid checkpoints\' return on the '
          'sub-task just trained (shown) and on the previous one, 95% bootstrap CI. '
          f'kept = (previous - untrained) / (shown - untrained); generalist = learned '
          f'(>= {args.learned:g} of the panel best\'s gain) and kept >= {args.kept:g}. '
          'learned by direction = the same share for the checkpoints ending a sub-task-1 / '
          'sub-task-2 phase; frozen = one direction below the learned bar while its previous '
          'sub-task is above it and the other direction learned. '
          'ES = NES; PBT-PPO = the N with the higher Cum. elite. '
          'See the docstring of `scripts/analysis/plot_generalist_scores.py`. '
          f'Data extracted {meta["extracted"]}.', '',
          '| Row | Task | Change | Method | Arm | n | Scores | untrained | shown | previous '
          '| learned | kept | learned by direction | class |',
          '|' + '---|' * 14]
    slots = {(r, c): (r, c, spec)
             for r, (_, cells) in enumerate(GRID) for c, spec in enumerate(cells) if spec}
    profile_axes = [fig.add_subplot(gs_profile[i, 0]) for i in range(len(PROFILE))]
    runs = {m: [] for m in ARMS}   # per method, a panel: (trials, 2) worse-direction scores
    for (dr, dc), (r, c, spec) in sorted(slots.items(), key=lambda kv: kv[1][:2]):
        row_label = GRID[r][0]
        ax = fig.add_subplot(gs[dr, dc])
        tree, cell, change = spec
        body = COLUMNS[c]
        ax.set_title(body, pad=3)   # the change is the row's and the caption's
        ax.spines[['top', 'right']].set_visible(False)
        if dc == 0:
            ax.annotate(row_label, xy=(0.08 / width, 0.5),
                        xycoords=('figure fraction', 'axes fraction'),
                        rotation=90, ha='center', va='center',
                        fontsize=fs + 0.5)
        panel = meta['panels'].get(f'{tree}|{cell}')
        if not panel or not panel['arms']:
            ax.text(0.5, 0.5, 'no runs', transform=ax.transAxes,
                    ha='center', va='center', color=MUTED)
            continue
        pooled = {m: trial_scores(v['trials'], cell)
                  for m, v in panel['arms'].items()}
        split = {(m, d): trial_scores(v['trials'], cell, d)
                 for m, v in panel['arms'].items() for d in (0, 1)}
        floor = FLOOR[cell.split('_sigma')[0]]
        best = max(s[:, 0].mean() for s in pooled.values())
        bar = floor + args.learned * (best - floor)
        best_dir = max(s[:, 0].mean() for s in split.values() if s is not None)
        classes = classify(panel, cell, args.kept, args.learned)
        lo_all, hi_all = np.inf, -np.inf
        for m in ARMS:
            if m not in pooled:
                continue
            x, y = pooled[m][:, 0], pooled[m][:, 1]
            (xm, xl, xh), (ym, yl, yh) = boot(x, rng), boot(y, rng)
            learned = (xm - floor) / (best - floor)
            kept = (ym - floor) / (xm - floor) if xm > floor else float('nan')
            dirs = [split[m, d].mean(axis=0) for d in (0, 1) if split[m, d] is not None]
            klass = classes[m]
            frozen = klass == 'frozen'
            both = np.stack([split[m, d] for d in (0, 1) if split[m, d] is not None])
            runs[m].append(((both - floor) / (best_dir - floor)).min(axis=0))
            ne = m in NE_ARMS
            col = METHOD_STYLE[m]['color']
            ms = 5.5 if ne else 4.5
            ax.scatter(x, y, s=4, color=col, alpha=0.3, lw=0, zorder=2)
            ax.errorbar([xm], [ym], xerr=[[xm - xl], [xh - xm]],
                        yerr=[[ym - yl], [yh - ym]], fmt='o' if ne else 's',
                        ms=ms, color=col,
                        ecolor=to_rgba(col, NOT_LEARNED_CI if klass == 'not learned' else 1),
                        **MARKER[klass], elinewidth=0.8, capsize=0,
                        zorder=4 if ne else 3, clip_on=False)
            if frozen:
                ax.plot([xm], [ym], ls='', marker='x', ms=ms * 0.55, color=col,
                        mew=1.0, zorder=5 if ne else 4.5, clip_on=False)
            lo_all = min(lo_all, xl, yl, x.min(), y.min())
            hi_all = max(hi_all, xh, yh, x.max(), y.max())
            arm = panel['arms'][m]['arm']
            per_dir = ' / '.join(f'{(d[0] - floor) / (best_dir - floor):.2f}' for d in dirs)
            md.append(f'| {row_label} | {body} | {change} | {label_of(m)} | {arm} '
                      f'| {len(x)} | {panel["scores"]} | {floor:g} '
                      f'| {xm:.4g} [{xl:.4g}, {xh:.4g}] | {ym:.4g} [{yl:.4g}, {yh:.4g}] '
                      f'| {learned:.2f} | {kept:.2f} | {per_dir} | {klass} |')
            print(f'{cell:24s} {arm:7s} n={len(x):2d} shown {xm:8.4g} prev {ym:8.4g} '
                  f'learned {learned:4.2f} ({per_dir}) kept {kept:4.2f} {klass}')
        # One square frame a panel so the diagonal is at 45 degrees,
        # fitted to the trials: on Acrobot the untrained -500 is far
        # below every method, and a frame from it squeezes them into a dot.
        # The pad keeps most markers inside; clip_on=False above draws the
        # rest whole over the spine instead of cutting them.
        pad = 0.08 * (hi_all - lo_all)
        lim = (lo_all - pad, hi_all + pad)
        ax.axvspan(lim[0], bar, color=WEAK_BAND, lw=0, zorder=0)
        ax.plot(lim, lim, color=DIAGONAL, lw=0.7, ls='--', zorder=1)
        ax.plot(lim, [floor + args.kept * (v - floor) for v in lim],
                color=DIAGONAL, lw=0.7, ls=':', zorder=1)
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_aspect('equal', adjustable='box')
        ax.xaxis.set_major_locator(plt.MaxNLocator(2))
        ax.yaxis.set_major_locator(plt.MaxNLocator(3))
        if max(abs(v) for v in lim) >= 1000:   # HalfCheetah: 2000 4000 collide
            for axis in (ax.xaxis, ax.yaxis):
                axis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v / 1000:g}k'))
        ax.tick_params(pad=1.5)
    curves = {}
    means = summary(profile_axes, runs, np.random.default_rng(1), curves)
    PROFILES.write_text(json.dumps({
        'tau': TAU.tolist(), 'profiles': curves,
        'titles': {key: title for key, _i, title, _x in PROFILE},
        'xlabel': 'τ (rescaled score)', 'ylabel': 'runs ≥ τ',
        'note': 'Written by plot_generalist_scores.py; read by plot_continual_combined.py '
                'for the bottom row of Figure 2. Per profile and method: the fraction of '
                'runs scoring at least tau (panels weighted equally) with its 95% '
                'bootstrap band over runs within panels.',
        'extracted': meta['extracted']}, indent=1) + '\n')
    md += ['', '## Summary', '',
           'Mean over panels (each panel the mean over trials) of the rescaled worse-direction '
           'score clipped to [0, 1]: the area under each profile.', '',
           '| Method | learns | keeps | panels |', '|---|---|---|---|']
    for m in ARMS:
        if m in means:
            md.append(f'| {label_of(m)} | {means[m]["learns"][0]:.2f} '
                      f'| {means[m]["keeps"][0]:.2f} | {means[m]["learns"][1]} |')
    handles = [Line2D([], [], ls='', marker='o' if m in NE_ARMS else 's',
                      color=METHOD_STYLE[m]['color'], mec='white', mew=0.5,
                      ms=5.5 if m in NE_ARMS else 4.5, label=label_of(m))
               for m in ARMS]
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, 1.0),
               ncol=len(handles), frameon=False, fontsize=fs - 0.5, handletextpad=0.3,
               columnspacing=1.2)
    fig.text((left_in + grid_right_in) / 2 / width, 0.1 / height,
             'Return on the task just trained',
             ha='center', va='bottom', fontsize=fs)
    fig.text(0.24 / width, 0.5, 'Return on the previous task',
             rotation=90, ha='center', va='center', fontsize=fs)
    STEM.parent.mkdir(parents=True, exist_ok=True)
    for ext in ('pdf', 'png'):
        fig.savefig(STEM_GRID.with_suffix(f'.{ext}'), dpi=300)
    plt.close(fig)
    print(f'wrote {STEM_GRID}.pdf/.png ({width:.2f} x {height:.2f} in)')
    # The main-text figure: the profiles alone, side by side (the grid's
    # two-task points are in the trade-off figure since 2026-09-24).
    stacked = args.profile_layout == 'column'
    pw = args.profile_width or (1.55 if stacked else 3.4)
    if stacked:
        left_in, right_in, top_in, bottom_in, mid_in = 0.36, 0.06, 0.16, 0.34, 0.5
        panel_in = pw - left_in - right_in
        pheight = 2 * panel_in + mid_in + top_in + bottom_in
        pfig = plt.figure(figsize=(pw, pheight))
        pgs = GridSpec(2, 1, figure=pfig, left=left_in / pw, right=1 - right_in / pw,
                       top=1 - top_in / pheight, bottom=bottom_in / pheight,
                       hspace=mid_in / panel_in)
        paxes = [pfig.add_subplot(pgs[i, 0]) for i in range(len(PROFILE))]
        summary(paxes, runs, np.random.default_rng(1))
    else:
        left_in, right_in, top_in, bottom_in, mid_in = 0.4, 0.08, 0.5, 0.36, 0.5
        panel_in = (pw - left_in - right_in - mid_in) / 2
        pheight = panel_in + top_in + bottom_in
        pfig = plt.figure(figsize=(pw, pheight))
        pgs = GridSpec(1, 2, figure=pfig, left=left_in / pw, right=1 - right_in / pw,
                       top=1 - top_in / pheight, bottom=bottom_in / pheight,
                       wspace=mid_in / panel_in)
        paxes = [pfig.add_subplot(pgs[0, i]) for i in range(len(PROFILE))]
        summary(paxes, runs, np.random.default_rng(1))
        pfig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, 1.0),
                    ncol=4, frameon=False, fontsize=fs - 0.5, handletextpad=0.3,
                    columnspacing=1.0, labelspacing=0.2)
    for ext in ('pdf', 'png'):
        pfig.savefig(STEM.with_suffix(f'.{ext}'), dpi=300)
    plt.close(pfig)
    STEM.with_suffix('.md').write_text('\n'.join(md) + '\n')
    print(f'wrote {STEM}.pdf/.png/.md ({pw:.2f} x {pheight:.2f} in)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
