"""The metrics figure across suites, one column per task grouped under its
suite; rows are the metrics figure's rows. Three figures:

    main   THE SAME panels as the main-text lineplot figure    -> metrics_overview_main_<agent>
           (plot_continual_lineplots.py, `continual_main`):
           gymnax noise (10 sub-tasks) and action reversal,
           MiniGrid, Kinetix, cheetah noise (10) and actions
    more   the families with MORE than two sub-tasks          -> metrics_overview_<agent>
    two    the same families at TWO sub-tasks, with            -> metrics_overview_2task_<agent>
           MiniGrid in Kinetix's place (its only schedule)

each written to `projects/iclr_2027/paper/visuals/` (`main/` for the
centroid `main` figure, the one the paper includes) as `.pdf`, `.png` and a
`.md` of every point, beside the other paper visuals (`continual_main`,
`stability_plasticity`, `plasticity_overview`), which the `main` set draws the
same ten tasks and the same arms as.

    .venv/bin/python scripts/analysis/plot_metrics_overview.py
    .venv/bin/python scripts/analysis/plot_metrics_overview.py --set main --agent elite
    .venv/bin/python scripts/analysis/plot_metrics_overview.py --set two --agent elite \\
        --columns cartpole-noise acrobot-noise mountaincar-noise cheetah-noise minigrid

Read from the `metrics_<agent>_values.json` that `make_metrics_figure.py`
writes into each paper directory (built by `finish_iclr.sh <family>`, named
per column below), so no run tree is reloaded and every panel is drawn by the
per-family figure's own `draw_panel`: same points, same CIs, same marks.

Each column keeps its own x axis: the tasks' returns are on unrelated scales,
and the per-family figure is what the numbers are read from. A method is one
line across the whole figure; a method a family did not run leaves its line
empty in those columns. A column whose JSON is missing (family not built yet)
is drawn empty and named on stdout, so the grid does not reflow when it lands;
a row a family cannot fill (BD on Kinetix, or a post-hoc pass not run yet) is
`n/a` there.

ES/NES and PBT-PPO N=8/N=2 are each ONE method drawn once, the arm with the
higher Cum. elite over the column's family (`keep_one_arm`), so the figure has
the same rows as the lineplot figures and their legends agree.

Ten or eleven columns do not fit the 5.5 in text width: the figure is 0.75 in a
column (about 8.4-9.2 in, a rotated page) unless `--width` says otherwise.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt                            # noqa: E402
from matplotlib.gridspec import GridSpec                   # noqa: E402
from matplotlib.lines import Line2D                        # noqa: E402
from matplotlib.ticker import FuncFormatter, MaxNLocator   # noqa: E402

REPO = pathlib.Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO / 'scripts'))
import make_lineplot as lp                                 # noqa: E402
from make_metrics_figure import ROWS, draw_panel           # noqa: E402
import es_arm                                              # noqa: E402
from es_arm import dropped_arm                             # noqa: E402

PAPER = REPO / 'projects/iclr_2027/paper'
OUT = PAPER / 'visuals'
# The figures the paper includes go to visuals/main.
IN_PAPER = {('metrics_overview_main', 'centroid')}


def out_dir(stem, agent):
    return OUT / 'main' if (stem, agent) in IN_PAPER else OUT


GYMNAX_ENVS = (('CartPole', 'v1'), ('Acrobot', 'v1'), ('MountainCar', 'v0'))
# (paper sub-directory under gymnax/, the column's perturbation label).
PERTURBATIONS = (('noise', 'noise'), ('physics', 'physics'))
ACTIONS = (('actions', 'action reversal'),)


def _gymnax(dest, perts=PERTURBATIONS):
    """The three gymnax columns of every perturbation in `perts` at schedule
    `dest`; the column id is `<env>-<sub-directory>`, unique across sets."""
    return [(f'{env.lower()}-{sub}', 'gymnax', f'gymnax/{sub}/{dest}',
             f'{env}_{ver}', env, label)
            for sub, label in perts
            for env, ver in GYMNAX_ENVS]


# set -> (output stem, [(id, suite, paper directory, env key in the JSON, task,
# perturbation)]), in drawing order; consecutive columns of one suite share a
# header. The finish_iclr.sh family that builds each directory is on its line.
SETS = {
    # The main-text figure: the panels of `continual_main`, MiniGrid and
    # Kinetix (its last column) last.
    'main': ('metrics_overview_main',
             _gymnax('10task', PERTURBATIONS[:1])                            # noise
             + _gymnax('2task', ACTIONS) + [                                 # actions
        ('cheetah-noise',   'Brax', 'mjx/cheetah/noise/10task', 'cheetah_noise',    'HalfCheetah', 'noise'),        # cheetah_noise05_t10
        ('cheetah-actions', 'Brax', 'mjx/cheetah/actions/2task', 'cheetah_action',  'HalfCheetah', 'action reversal'),  # cheetah_action
        ('minigrid',    'MiniGrid', 'minigrid/minigrid',        'MiniGrid_8x8_16x16', '8x8 / 16x16', 'room size'),  # minigrid
        ('kinetix',      'Kinetix', 'kinetix/kinetix',          'Kinetix20',        '20 levels',   'level chain'),  # kinetix
    ]),
    'more': ('metrics_overview', _gymnax('10task') + [                          # noise, physics
        ('cheetah-noise',   'Brax', 'mjx/cheetah/noise/10task',   'cheetah_noise',    'HalfCheetah', 'noise'),     # cheetah_noise05_t10
        ('cheetah-physics', 'Brax', 'mjx/cheetah/physics/10task', 'cheetah_friction', 'HalfCheetah', 'friction'),  # cheetah_friction
        ('ant-noise',       'Brax', 'mjx/ant/noise/10task_warmup', 'ant_noise',       'Ant',         'noise'),     # ant_noise025_t10
        ('ant-physics',     'Brax', 'mjx/ant/physics/10task',     'ant_friction',     'Ant',         'friction'),  # ant
        ('kinetix',      'Kinetix', 'kinetix/kinetix',            'Kinetix20',        '20 levels',   'level chain'),  # kinetix
    ]),
    'two': ('metrics_overview_2task', _gymnax('2task') + [                      # noise_2task, physics_2task
        ('cheetah-noise',   'Brax', 'mjx/cheetah/noise/2task',    'cheetah_noise',    'HalfCheetah', 'noise'),     # cheetah_noise025
        ('cheetah-physics', 'Brax', 'mjx/cheetah/physics/2task',  'cheetah_friction', 'HalfCheetah', 'friction'),  # cheetah_friction_2task
        ('ant-noise',       'Brax', 'mjx/ant/noise/2task_warmup', 'ant_noise',        'Ant',         'noise'),     # ant_noise025
        ('ant-physics',     'Brax', 'mjx/ant/physics/2task_warmup', 'ant_friction',   'Ant',         'friction'),  # ant_friction_2task
        ('minigrid',    'MiniGrid', 'minigrid/minigrid',          'MiniGrid_8x8_16x16', '8x8 / 16x16', 'room size'),  # minigrid
    ]),
}
# Sets drawn like continual_main: one bold "Task, perturbation" title a panel,
# evenly spaced columns, no suite headers, the significance note left to the
# caption.
PLAIN = {'metrics_overview_main'}
# The paper reports no novelty arm (finish_iclr.sh THE ARMS). Dropped here too,
# so a paper directory drawn before that decision cannot put one back.
NOT_REPORTED = {'dns_gaussian', 'dns', 'ga_isoline'}
# (set, agent) -> the rows that figure draws when `--rows` is not given; any
# other figure draws every row some column has. The main-text centroid figure
# is performance and transfer only (2026-09-16): the F and BD it would add are
# in stability_plasticity, which draws the same ten tasks.
SET_ROWS = {('main', 'centroid'): ['cum', 'ZT']}
PAIRS = (es_arm.ARMS, es_arm.PBT_ARMS)          # ('es', 'nes'), ('pbt', 'pbt2')
PAIR_ARMS = {a for pair in PAIRS for a in pair}
MUTED = '#6b6a65'
LEFT_IN = 0.92          # method labels and the row label
COLUMN_IN = 0.75
PLAIN_COLUMN_IN = 1.45
TEXT_WIDTH_IN = 5.5     # ICLR


def _compact(x, _pos):
    return f'{x / 1e3:g}k' if abs(x) >= 1e3 else f'{x:g}'


def panel_title(suite, task, pert):
    """The panel title `plot_continual_lineplots.PANELS` gives the same task."""
    if suite == 'MiniGrid':
        return f'MiniGrid {task}'
    if suite == 'Kinetix':
        return f'Kinetix, {task}'
    return f'{task}, {pert}'


def _cum_elite(sub):
    """`{env: {arm: [Cum. elite a trial]}}` over the pair arms of a built
    paper directory: the column es_arm.py picks on, read from the ELITE JSON
    whichever agent is being drawn, so the two agents' figures keep the same
    arm."""
    path = PAPER / sub / 'metrics_elite_values.json'
    if not path.exists():
        return {}
    row = next((r for r in json.loads(path.read_text())['rows']
                if r['key'] == 'cum'), None)
    return {env: {a: v for a, v in by_method.items() if a in PAIR_ARMS}
            for env, by_method in (row or {}).get('values', {}).items()}


def keep_one_arm(sub, present):
    """`({arm: canonical name or None}, [kept arm a pair])` for one paper
    directory. ES vs NES and PBT-PPO N=8 vs N=2 are one method at two
    settings, so a column draws ONE of each pair -- the higher Cum. elite
    summed over that family's cells (es_arm.pick) -- filed under the pair's
    first name, `es` and `pbt`, exactly as plot_continual_lineplots.py and
    plot_noncontinual_solve.py do. One row a method, and the same arm as the
    family's own figures: the ES pair reads the `es_arm.json` finish_iclr.sh
    wrote at build time, so a column cannot disagree with the directory it
    came from; the PBT pair is not filed, so it is picked here. `present` (the
    JSON's methods) breaks the tie when neither is available."""
    per_cell = _cum_elite(sub)
    remap, kept = {}, []
    for pair in PAIRS:
        arm = None
        if pair == es_arm.ARMS:
            dropped = dropped_arm(PAPER / sub)
            arm = next((a for a in pair if a != dropped), None) if dropped else None
        if arm is None:
            arm = es_arm.pick(per_cell, pair)[0]
        if arm is None:
            arm = next((a for a in pair if a in present), None)
        kept.append(arm)
        remap.update({a: (pair[0] if a == arm else None) for a in pair})
    return remap, kept


def draw(stem, cols, agent, rows, args):
    """One figure: `cols` at `agent`, written to `<stem>_<agent>.pdf/.png`
    under `OUT`. Returns one `(row label, task, perturbation, method, mean, lo,
    hi, n, mark)` a point drawn, for the markdown."""
    fs = args.font_size
    loaded, remap = {}, {}
    for key, _suite, sub, env, *_ in cols:
        path = PAPER / sub / f'metrics_{agent}_values.json'
        loaded[key] = json.loads(path.read_text()) if path.exists() else None
        remap[key], kept = keep_one_arm(sub, loaded[key]['methods'] if loaded[key] else [])
        print(f'{key:20s} ' + (f'{sub} ({", ".join(loaded[key]["methods"])})'
                               if loaded[key] else f'missing {path.relative_to(REPO)}')
              + f', keeps {" and ".join(str(k) for k in kept)}')
    present = [d for d in loaded.values() if d]
    if not present:
        print('nothing built')
        return []
    labels = {}
    for d in present:
        for r in d['rows']:
            labels.setdefault(r['key'], (r['label'], r['higher_is_better']))
    row_keys = [k for k in rows if k in labels]
    methods = args.methods
    if not methods:
        union = {remap[key].get(m, m) for key, d in loaded.items() if d
                 for m in d['methods']} - NOT_REPORTED - {None}
        methods = ([m for m in lp.METHOD_ORDER if m in union]
                   + sorted(union - set(lp.METHOD_ORDER)))
    plain = stem in PLAIN
    # A plain figure's one-line titles need the plasticity line figures' columns.
    width = args.width or (LEFT_IN + PLAIN_COLUMN_IN * len(cols) if plain
                           else TEXT_WIDTH_IN if len(cols) <= 5
                           else LEFT_IN + COLUMN_IN * len(cols))
    # A narrow spacer column between suites.
    ratios, slot = [], {}
    for i, c in enumerate(cols):
        if i and c[1] != cols[i - 1][1] and not plain:
            ratios.append(0.2)
        slot[c[0]] = len(ratios)
        ratios.append(1.0)
    top_in, bottom_in = (0.22, 0.2) if plain else (0.48, 0.27)
    height = args.row_height * len(row_keys) + top_in + bottom_in
    fig = plt.figure(figsize=(width, height))
    gs = GridSpec(len(row_keys), len(ratios), figure=fig, width_ratios=ratios,
                  left=LEFT_IN / width, right=1 - 0.12 / width,
                  top=1 - top_in / height, bottom=bottom_in / height,
                  wspace=0.3, hspace=0.35)
    top_axes, table = {}, []
    for r, rk in enumerate(row_keys):
        label, hib = labels[rk]
        for c, (key, suite, sub, env, task, pert) in enumerate(cols):
            ax = fig.add_subplot(gs[r, slot[key]])
            d = loaded[key]
            row = next((x for x in d['rows'] if x['key'] == rk), None) if d else None
            # Only the drawn methods, each pair under its canonical name: the
            # marks test every method in the cell.
            cell = {remap[key].get(m, m): v
                    for m, v in ((row or {}).get('values', {}).get(env) or {}).items()
                    if remap[key].get(m, m) in methods}
            filled = any(cell.values())
            _tests, points = draw_panel(ax, cell if filled else {}, methods, rk,
                                        hib, fs, labels=c == 0)
            table += [(label, task, pert, *pt) for pt in points]
            if filled:
                ax.xaxis.set_major_locator(MaxNLocator(nbins=3, min_n_ticks=2))
                ax.xaxis.set_major_formatter(FuncFormatter(_compact))
                ax.tick_params(axis='x', pad=1.5)
            else:
                ax.set_xticks([])
                ax.spines['bottom'].set_color('0.85')
                ax.text(0.5, 0.5, 'not built' if d is None else 'n/a',
                        transform=ax.transAxes, ha='center', va='center',
                        color=MUTED, fontsize=fs - 1)
            if c == 0:
                ax.annotate(label + (' ↓' if not hib else ''), xy=(0, 0.5),
                            xycoords='axes fraction', xytext=(-54, 0),
                            textcoords='offset points', rotation=90,
                            ha='center', va='center', fontweight='bold')
            if r == 0 and plain:
                ax.set_title(panel_title(suite, task, pert), fontweight='bold', pad=3)
                top_axes[key] = ax
            elif r == 0:
                ax.set_title(task, fontweight='bold', pad=fs + 3)
                ax.annotate(pert, (0.5, 1), xycoords='axes fraction',
                            xytext=(0, 2), textcoords='offset points',
                            ha='center', va='bottom', color=MUTED,
                            fontsize=fs - 0.5)
                top_axes[key] = ax
    # Suite headers: a rule over the suite's columns, the name above it.
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
                 va='bottom', fontweight='bold', fontsize=fs + 0.5)
    lower = any(not labels[k][1] for k in row_keys)
    if not plain:
        fig.text(0.5, 2 * pt, ('↓ lower is better.  ' if lower else '')
                 + 'Marks: better than every method of '
                 'the other family (one-sided Mann-Whitney U, Holm; * p<.05, ** p<.01, '
                 '*** p<.001).', ha='center', va='bottom', color=MUTED, fontsize=fs - 1)
    out = out_dir(stem, agent) / f'{stem}_{agent}'
    for ext in ('pdf', 'png'):
        fig.savefig(out.with_suffix(f'.{ext}'), dpi=300)
    plt.close(fig)
    print(f'wrote {out}.pdf, {out}.png  ({width:.2f} x {height:.2f} in'
          + (', wider than the text' if width > TEXT_WIDTH_IN else '') + ')')
    return table


def write_markdown(stem, agent, table):
    """`<stem>_<agent>.md`: every point of the figure, in the shape
    plot_plasticity_overview.py writes its own."""
    md = [f'# {stem}_{agent}', '',
          f'The metrics figure of the {agent} agent across suites, built by '
          '`scripts/analysis/plot_metrics_overview.py`; the tasks are its '
          '`SETS` and every number is the one the family\'s own '
          f'`metrics_{agent}` figure draws. Mean over trials [95% bootstrap '
          'CI], n, NE-vs-RL mark (`*` p<.05, `**` p<.01, `***` p<.001: better '
          'than EVERY method of the other family, one-sided Mann-Whitney U, '
          'Holm). See the docstring of `scripts/make_metrics_figure.py` for '
          'each row.', '',
          '| Row | Task | Perturbation | Method | mean | lo | hi | n | mark |',
          '|---|---|---|---|---|---|---|---|---|']
    for row, task, pert, m, mean, lo, hi, n, mark in table:
        md.append(f'| {row} | {task} | {pert} | {lp.METHOD_STYLE.get(m, {}).get("label", m)} '
                  f'| {mean:.3g} | {lo:.3g} | {hi:.3g} | {n} | {mark} |')
    path = out_dir(stem, agent) / f'{stem}_{agent}.md'
    path.write_text('\n'.join(md) + '\n')
    print(f'wrote {path}')


def main() -> int:
    ids = sorted({c[0] for _, cols in SETS.values() for c in cols})
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--set', nargs='+', default=list(SETS), choices=list(SETS))
    ap.add_argument('--agent', nargs='+', default=['centroid', 'elite'],
                    choices=['centroid', 'elite'])
    ap.add_argument('--columns', nargs='+', default=None, choices=ids,
                    help='draw only these columns, in the set\'s order (default: all)')
    ap.add_argument('--rows', nargs='+', default=None, choices=list(ROWS),
                    help='default: SET_ROWS for that figure, else every row some column has')
    ap.add_argument('--methods', nargs='+', default=None,
                    help='default: every method some column has')
    ap.add_argument('--width', type=float, default=None,
                    help=f'inches (default: {TEXT_WIDTH_IN} up to five columns, '
                         f'else {COLUMN_IN} a column)')
    ap.add_argument('--row-height', type=float, default=0.95)
    ap.add_argument('--font-size', type=float, default=6.0)
    args = ap.parse_args()

    # One entry a pair: `pbt` is PBT-PPO, whichever N the column kept.
    lp.METHOD_STYLE['pbt'] = {**lp.METHOD_STYLE['pbt'], 'label': 'PBT-PPO'}
    plt.rcParams.update({
        'font.size': args.font_size, 'axes.titlesize': args.font_size + 0.5,
        'xtick.labelsize': args.font_size - 1, 'ytick.labelsize': args.font_size - 0.5,
        'axes.linewidth': 0.5, 'xtick.major.width': 0.5, 'xtick.major.size': 2,
    })
    (OUT / 'main').mkdir(parents=True, exist_ok=True)
    for name in args.set:
        stem, cols = SETS[name]
        cols = [c for c in cols if not args.columns or c[0] in args.columns]
        for agent in args.agent:
            print(f'--- {name} / {agent}')
            rows = args.rows or SET_ROWS.get((name, agent)) or list(ROWS)
            table = draw(stem, cols, agent, rows, args)
            if table:
                write_markdown(stem, agent, table)
    return 0


if __name__ == '__main__':
    sys.exit(main())
