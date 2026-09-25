"""The appendix metrics figure: every performance metric the paper reports for
the ten continual tasks of the main text, one row a task, one column a metric.

    .venv/bin/python scripts/analysis/plot_metrics_appendix.py             # from the saved data
    .venv/bin/python scripts/analysis/plot_metrics_appendix.py --extract   # re-read the runs first

    -> paper/visuals/final/appendix/metrics_continual.{pdf,png}   the figure
       paper/visuals/final/appendix/metrics_continual.md          every point, with definitions
       paper/visuals/final/data/metrics_continual.json            one value a trial it is drawn from
    (paper = projects/iclr_2027/paper)

    .venv/bin/python scripts/analysis/plot_metrics_appendix.py --main
    -> paper/visuals/final/metrics_main.{pdf,png,md}   Cum. centroid and ZT only,
       on the grid of the trade-off figure (plot_continual_combined
       .TRADEOFF_ROWS: multiple tasks, then two tasks under physics and
       action reversal; the two-task noise row is left out, MAIN_SKIP_ROWS),
       one column an environment, so every other panel of Figure 2 has its
       Cum. and ZT panels here, side by side in its column.

Built in two steps, like the other final figures: `--extract` reads the runs
through the symlink trees under paper/<suite>/data; without it only the saved
data is read. The tasks, trees and arms are continual_main's
(plot_continual_lineplots.PANELS, `main`): ES = NES except on Kinetix (OpenES,
`es`, plain OpenES since 2026-09-24), PBT-PPO = N=8 or N=2 by the higher Cum. elite. Both pairs are
filed under `es` and `pbt`.

Columns, all for the CENTROID unless named otherwise (make_metrics_figure.py's
docstring has the long form):

    Cum. centroid  area under the centroid training curve / 1000
                   (make_lineplot.metric_table, the lineplot's own number)
    Cum. elite     the same for the best-performing agent (the RL arms' only
                   agent is both)
    FT             (extracted, not drawn since 2026-09-21: the paper never
                   defines it) forward transfer against the method's OWN stationary run
                   (<suite>/data/noncontinual). Where the sub-tasks are
                   different cells (MiniGrid's two rooms, Kinetix's twenty
                   levels) each phase is scored against the stationary run of
                   ITS sub-task (ft_per_phase), over a window as long as the
                   phase, from that run's start
    ZT             zero-shot transfer: each sub-task's final agent on the next
                   sub-task, meaned over switches. From the `evaluate` pass
                   (evaluation.json), except HalfCheetah, whose CLUSTER runs
                   were never evaluated: there it is the training records'
                   `centroid_task<next>` at the last generation of each phase,
                   for every arm, so the panel has one source
    F              forgetting, lower is better
    BD             (extracted, not drawn since 2026-09-22) behavioural divergence
                   at the switch, lower is better, in [0, 1]
                   F and BD are read from data/stability_plasticity.json,
                   so they are the numbers that figure draws (run its
                   --extract first); BD is n/a on HalfCheetah and Kinetix

Each panel is make_metrics_figure.draw_panel: mean over trials with a 95%
bootstrap CI, and the NE-vs-RL mark (better than every method of the other
family, one-sided Mann-Whitney U, Holm).
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import pathlib
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt                            # noqa: E402
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec  # noqa: E402
from matplotlib.ticker import FuncFormatter                # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import plot_continual_lineplots as pcl                     # noqa: E402
import plot_stability_plasticity as spp                    # noqa: E402
from make_metrics_figure import draw_panel                 # noqa: E402
from source.metrics.continual_metrics import window_mean   # noqa: E402

lp, PROJECT, REPO, es_arm = pcl.lp, pcl.PROJECT, spp.REPO, spp.es_arm
STEM = 'metrics_continual'
OUT = pcl.FINAL / 'appendix'
DATA = pcl.FINAL / 'data' / f'{STEM}.json'
# (key, column title, higher is better), left to right.
METRICS = [('cum_centroid', 'Cum. centroid', True),
           ('cum_elite', 'Cum. elite', True),
           ('zt', 'ZT', True),
           ('F', 'F', False)]
# BD is still extracted but not drawn since 2026-09-22: it ranks the methods as F
# does (median Spearman 0.82 over the nine settings that have it) and is n/a on Kinetix.
# Trees whose ZT comes from the training records (module docstring).
ZT_FROM_RECORDS = {'paper/mjx/cheetah/data/noise_10task',
                   'paper/mjx/cheetah/data/noise_2task',
                   'paper/mjx/cheetah/data/actions_2task'}
# The panels extracted: continual_main's ten and the two-task trade-off ones.
FIGS = ('main', 'tradeoff')
# Column headers: (group spanning its columns, the column's own title).
GROUPS = {'noise': 'Observation noise', 'action reversal': 'Action reversal'}
HEADERS = {'MiniGrid 8x8 / 16x16': ('MiniGrid', '8x8 / 16x16'),
           'Kinetix, 20 levels': ('Kinetix', '20 levels')}


def header(title):
    """`(group, column title)` of a panel title such as 'CartPole, noise'."""
    if title in HEADERS:
        return HEADERS[title]
    env, change = title.split(', ')
    return GROUPS[change], env


def column_order(panels):
    """The noise columns, then the action-reversal ones, then the rest."""
    rank = {g: i for i, g in enumerate(GROUPS.values())}
    return sorted(panels, key=lambda p: rank.get(header(p[0])[0], len(rank)))
PAIRS = (es_arm.ARMS, es_arm.PBT_ARMS)          # ('es', 'nes'), ('pbt', 'pbt2')
MUTED = '#6b6a65'
LEFT_IN, TOP_IN, BOTTOM_IN = 0.85, 0.36, 0.28
TEXT_WIDTH_IN = 5.5     # ICLR


def ref_root(tree):
    """The directory whose `noncontinual/` holds the tree's stationary runs."""
    root = PROJECT / tree
    return root if (root / 'noncontinual').is_dir() else root.parent


def canonical(arm):
    """A pair's arm filed under the pair's first name, as continual_main does."""
    return next((pair[0] for pair in PAIRS if arm in pair), arm)


def curve_rows(tree, cells, arms, metric):
    """`{env: {'cum_max', 'ft'}}` from make_lineplot at `metric`, and the report."""
    args = lp.parse_args([str(PROJECT / tree), '--phase', 'continual', '--cells', *cells,
                          '--metric', metric, '--methods', *arms,
                          '--ref-root', str(ref_root(tree)), '--out', '-'])
    rep = lp.load_report(args)
    rows, _ = lp.metric_table(rep.data, rep.pop_data, rep.ref_data,
                              rep.per_gen, rep.edges, args)
    return rows, rep


def phase_cells(tree, cell, arm):
    """The stationary cell of each phase's sub-task, from the run's config."""
    cfg = json.loads(next((PROJECT / tree / 'continual' / arm / cell).glob('trial_*/config.json'))
                     .read_text())
    opts = cfg['task']['options']
    if 'levels' in opts:
        names = [f'Kinetix_{lv}' for lv in opts['levels']]
    elif 'envs' in opts:
        names = [f'MiniGrid_{e.split("-")[-1]}' for e in opts['envs']]
    else:               # one environment throughout: no per-phase reference
        return []
    return [names[t] for t in cfg['task_sequence']]


def ft_per_phase(rep, env, arm, refs):
    """`[FT a trial]`: mean over phases of the phase's window mean minus the
    stationary run of that phase's sub-task over as long a window."""
    scale = rep.per_gen or 1
    x, curves = rep.data[env][arm]
    edges = np.asarray(rep.edges, dtype=float) / scale
    if not refs or len(edges) - 1 != len(refs) or any(arm not in rep.ref_data.get(r, {}) for r in refs):
        return []
    base = []
    for r, (lo, hi) in zip(refs, zip(edges[:-1], edges[1:])):
        rx, rcurves = rep.ref_data[r][arm]
        base.append(np.nanmean([window_mean(rx / scale, c, 0.0, hi - lo) for c in rcurves]))
    out = []
    for c in curves:
        v = [window_mean(x / scale, c, lo, hi) - b
             for b, (lo, hi) in zip(base, zip(edges[:-1], edges[1:]))]
        v = [u for u in v if np.isfinite(u)]
        if v:
            out.append(float(np.mean(v)))
    return out


def zt_from_records(trial):
    """Mean over switches of the phase-end record on the NEXT sub-task."""
    records = json.loads((trial / 'training_metrics.json').read_text())
    tasks = np.array([r['task'] for r in records])
    ends = list(np.flatnonzero(tasks[1:] != tasks[:-1])) + [len(tasks) - 1]
    return float(np.mean([records[a][f'centroid_task{tasks[b]}']
                          for a, b in zip(ends[:-1], ends[1:])]))


def zero_shot(tree, cell, arm):
    """`[ZT a trial]` for one arm of one panel (module docstring for the source)."""
    cell_dir = PROJECT / tree / 'continual' / arm / cell
    if tree in ZT_FROM_RECORDS:
        return [zt_from_records(t) for t in sorted(cell_dir.glob('trial_*'))
                if (t / 'training_metrics.json').exists()]
    return list(lp.load_zero_shot(cell_dir, lp.agent_sources_for(cell_dir, 'centroid')).values())


def extract():
    """Write DATA: `{panels: {<tree>|<cell>: {arm: {metric: [per trial]}}}}`
    plus the arms kept, the ZT source a tree and the extraction date."""
    os.chdir(REPO)
    if not spp.DATA.exists():
        sys.exit(f'no {spp.DATA}: run plot_stability_plasticity.py --extract first')
    posthoc = json.loads(spp.DATA.read_text())['panels']
    meta = {'arms': {}, 'zt_source': {}, 'panels': {}}
    for tree, cells in pcl._cells(lambda f: f in FIGS).items():
        arms = spp.reported_arms(PROJECT / tree, cells,
                                 es_kept='es' if tree.startswith('paper/kinetix') else 'nes')
        meta['arms'][tree] = arms
        meta['zt_source'][tree] = ('training records' if tree in ZT_FROM_RECORDS
                                   else 'evaluate pass (evaluation.json)')
        centroid, crep = curve_rows(tree, cells, arms, 'centroid')
        elite, _ = curve_rows(tree, cells, arms, 'elite_eval')
        for cell in cells:
            env, key = cell.split('_sigma')[0], pcl._key(tree, cell)
            trials = posthoc.get(key, {})
            panel = {}
            for arm in arms:
                panel[canonical(arm)] = {
                    'cum_centroid': centroid[env]['cum_max'].get(arm, []),
                    'cum_elite': elite[env]['cum_max'].get(arm, []),
                    'ft': (centroid[env]['ft'].get(arm)
                           or ft_per_phase(crep, env, arm, phase_cells(tree, cell, arm))),
                    'zt': zero_shot(tree, cell, arm),
                    'F': [t['F'] for t in trials.get(arm, [])],
                    'BD': [t['BD'] for t in trials.get(arm, []) if t['BD'] is not None],
                }
                empty = [m for m, v in panel[canonical(arm)].items() if not v]
                if empty:
                    print(f'note: {key} {arm}: no {", ".join(empty)}')
            meta['panels'][key] = panel
    meta['extracted'] = datetime.date.today().isoformat()
    DATA.parent.mkdir(parents=True, exist_ok=True)
    DATA.write_text(json.dumps(meta, indent=1) + '\n')
    print(f'wrote {DATA}')


def _compact(x, _pos):
    return f'{x / 1e3:g}k' if abs(x) >= 1e3 else f'{x:g}'


def two_ticks(ax, key):
    """Two round ticks a fifth in from each end: ten columns leave no room
    for a third, and a locator's two may sit side by side."""
    if key == 'BD':
        return [0, 1]
    lo, hi = ax.get_xlim()
    q = 10 ** np.floor(np.log10((hi - lo) / 4))
    return [np.round((lo + 0.2 * (hi - lo)) / q) * q,
            np.round((hi - 0.2 * (hi - lo)) / q) * q]


def draw(panels, methods, args):
    """One row a metric, one column a task. Returns the markdown's points."""
    METRICS = args.metrics
    fs = args.font_size
    width = args.width
    height = args.row_height * len(METRICS) + TOP_IN + BOTTOM_IN
    fig = plt.figure(figsize=(width, height))
    gs = GridSpec(len(METRICS), len(panels), figure=fig,
                  left=LEFT_IN / width, right=1 - 0.12 / width,
                  top=1 - TOP_IN / height, bottom=BOTTOM_IN / height,
                  wspace=0.3, hspace=0.3)
    table = []
    for c, (title, cells) in enumerate(panels):
        for r, (key, label, hib) in enumerate(METRICS):
            ax = fig.add_subplot(gs[r, c])
            cell = {m: v[key] for m, v in cells.items() if m in methods and v[key]}
            _tests, points = draw_panel(ax, cell, methods, 'BD' if key == 'BD' else key,
                                        hib, fs, labels=c == 0)
            table += [(title, label, *p) for p in points]
            if cell:
                ax.set_xticks(two_ticks(ax, key))
                ax.xaxis.set_major_formatter(FuncFormatter(_compact))
                ax.tick_params(axis='x', pad=1.5)
            else:
                ax.set_xticks([])
                ax.spines['bottom'].set_color('0.85')
                ax.text(0.5, 0.5, 'n/a', transform=ax.transAxes, ha='center',
                        va='center', color=MUTED, fontsize=fs - 1)
            if r == 0:
                ax.set_title(header(title)[1], pad=3, fontsize=fs - 0.5)
            if c == 0:
                ax.annotate(label + ('' if hib else ' ↓'), xy=(0, 0.5),
                            xycoords='axes fraction', xytext=(-38, 0),
                            textcoords='offset points', rotation=90,
                            ha='center', va='center')
    # Group header over its columns, with a rule under it.
    groups = {}
    for c, (title, _cells) in enumerate(panels):
        groups.setdefault(header(title)[0], []).append(c)
    y = 1 - (TOP_IN - 0.13) / height
    for group, cols in groups.items():
        x0 = fig.axes[cols[0] * len(METRICS)].get_position().x0
        x1 = fig.axes[cols[-1] * len(METRICS)].get_position().x1
        fig.text((x0 + x1) / 2, y + 0.02 / height, group, ha='center', va='bottom',
                 fontsize=fs + 0.5)
        fig.add_artist(plt.Line2D([x0, x1], [y, y], transform=fig.transFigure,
                                  color='0.4', lw=0.5))
    if not all(hib for *_, hib in METRICS):
        fig.text(0.5, 1.5 / 72 / height, '↓ lower is better.', ha='center',
                 va='bottom', color=MUTED, fontsize=fs - 1)
    out, stem = args.out, args.stem
    out.mkdir(parents=True, exist_ok=True)
    for ext in ('pdf', 'png'):
        fig.savefig(out / f'{stem}.{ext}', dpi=300)
    plt.close(fig)
    print(f'wrote {out / stem}.pdf, .png  ({width:.2f} x {height:.2f} in)')
    return table


GROUP_CHANGE = {'ten': 'noise', 'noise': 'noise', 'physics': 'physics',
                'actions': 'action reversal'}
GROUP_RULE = '0.3'


# Rows of Figure 2 that Figure 3 leaves out (two tasks under noise, 2026-09-24).
MAIN_SKIP_ROWS = ('noise',)


def draw_main(meta, methods, args):
    """`--main`: Cum. centroid and ZT on the grid of Figure 2
    (plot_continual_combined.TRADEOFF_ROWS): one row a row of that figure
    except MAIN_SKIP_ROWS,
    one column an environment, and in each column the two metrics side by
    side (Cum. centroid left, ZT right, named in the first row's titles).
    Rows are labelled and bracketed at the left as there. The method names are
    a legend, not tick labels, so a row is short. Returns the markdown's points."""
    import plot_continual_combined as pcc
    METRICS = args.metrics
    fs, width = args.font_size, args.width
    rows = [r for r in pcc.TRADEOFF_ROWS if r[2] != 'profiles' and r[0] not in MAIN_SKIP_ROWS]
    ncol = max(len(r[2]) for r in rows)
    nm = len(METRICS)
    titles = {pcl._key(t, c): title for _f, t, c, title in pcl.PANELS}
    left_in, right_in, env_gap, pair_gap = 0.5, 0.12, 0.26, 0.15
    legend_in, row_h, tick_in, group_gap, bottom_in = 0.22, args.row_height, 0.11, 0.1, 0.02
    title_in = 0.24 if nm > 1 else 0.13          # env over the metric names in row 0
    panel_w = (width - left_in - right_in - (ncol - 1) * env_gap
               - ncol * (nm - 1) * pair_gap) / (ncol * nm)
    env_w = nm * panel_w + (nm - 1) * pair_gap
    height = legend_in + len(rows) * (title_in + row_h + tick_in) + (len(rows) - 1) * group_gap + bottom_in
    fig = plt.figure(figsize=(width, height))
    table, used = [], set()
    y = 1 - legend_in / height
    for g, (gkey, group, specs) in enumerate(rows):
        top = y - title_in / height
        bot = top - row_h / height
        outer = GridSpec(1, ncol, figure=fig, left=left_in / width, right=1 - right_in / width,
                         top=top, bottom=bot, wspace=env_gap / env_w)
        for c, spec in enumerate(specs):
            if spec is None:
                continue
            key = pcl._key(*spec)
            cells = meta['panels'].get(key, {})
            title = titles[key].replace(', two tasks', '')
            env = title.split(',')[0].split(' ')[0]     # 'MiniGrid 8x8 / 16x16' -> 'MiniGrid'
            inner = GridSpecFromSubplotSpec(1, nm, subplot_spec=outer[0, c],
                                            wspace=pair_gap / panel_w)
            x0 = left_in / width + c * (env_w + env_gap) / width
            fig.text(x0 + env_w / 2 / width, top + (0.03 + (0.11 if g == 0 else 0)) / height,
                     env, ha='center', va='bottom', fontsize=fs)
            for k, (mkey, label, hib) in enumerate(METRICS):
                ax = fig.add_subplot(inner[0, k])
                cell = {m: v[mkey] for m, v in cells.items() if m in methods and v[mkey]}
                used |= set(cell)
                _tests, points = draw_panel(ax, cell, methods, mkey, hib, fs, labels=False)
                table += [(title, label, *p) for p in points]
                if cell:
                    ax.set_xticks(two_ticks(ax, mkey))
                    ax.xaxis.set_major_formatter(FuncFormatter(_compact))
                    ax.tick_params(axis='x', pad=1.5)
                else:
                    ax.set_xticks([])
                    ax.spines['bottom'].set_color('0.85')
                    ax.text(0.5, 0.5, 'n/a', transform=ax.transAxes, ha='center',
                            va='center', color=MUTED, fontsize=fs - 1)
                if g == 0:
                    ax.set_title(label + ('' if hib else ' ↓'), pad=2, fontsize=fs - 0.5,
                                 color='0.25')
        # The group's label and bracket at the far left, over its row.
        x_rule = 0.3 / width
        fig.add_artist(plt.Line2D([x_rule, x_rule], [bot, top], lw=0.6, color=GROUP_RULE,
                                  transform=fig.transFigure))
        fig.text(0.1 / width, (top + bot) / 2, f'{group}\n{GROUP_CHANGE[gkey]}', rotation=90,
                 ha='center', va='center', fontsize=fs, linespacing=1.1)
        if g < len(rows) - 1:
            y_sep = bot - (tick_in + group_gap / 2) / height
            fig.add_artist(plt.Line2D([x_rule, 1 - right_in / width], [y_sep, y_sep], lw=0.6,
                                      color=GROUP_RULE, transform=fig.transFigure))
        y = bot - (tick_in + group_gap) / height
    legend = [m for m in methods if m in used]
    handles = [plt.Line2D([], [], ls='', marker='o' if lp.FAMILY.get(m) == 'ne' else 's',
                          color=lp.METHOD_STYLE[m]['color'], mec='white', mew=0.4,
                          ms=4.5 if lp.FAMILY.get(m) == 'ne' else 3.8) for m in legend]
    fig.legend(handles, [lp.METHOD_STYLE[m]['label'] for m in legend], loc='upper center',
               ncol=len(handles), frameon=False, bbox_to_anchor=(0.5, 1 - 0.01 / height),
               fontsize=fs - 0.5, handletextpad=0.2, columnspacing=1.0)
    if not all(hib for *_, hib in METRICS):
        fig.text(1 - right_in / width, 0.5 / 72 / height, '↓ lower is better.', ha='right',
                 va='bottom', color=MUTED, fontsize=fs - 1)
    out, stem = args.out, args.stem
    out.mkdir(parents=True, exist_ok=True)
    for ext in ('pdf', 'png'):
        fig.savefig(out / f'{stem}.{ext}', dpi=300)
    plt.close(fig)
    print(f'wrote {out / stem}.pdf, .png  ({width:.2f} x {height:.2f} in)')
    return table


def write_markdown(meta, table, out=OUT, stem=STEM):
    label = lambda m: lp.METHOD_STYLE.get(m, {}).get('label', m)  # noqa: E731
    md = [f'# {stem}', '',
          'Every performance metric of the ten continual tasks of the main text, for the '
          'centroid. Built by `scripts/analysis/plot_metrics_appendix.py` from '
          f'`data/{STEM}.json` (extracted {meta["extracted"]}); its docstring defines '
          'each column. Mean over trials [95% bootstrap CI], n, and the NE-vs-RL mark '
          '(`*` p<.05, `**` p<.01, `***` p<.001: better than EVERY method of the other '
          'family, one-sided Mann-Whitney U, Holm). ES is NES except on Kinetix '
          '(OpenES); PBT-PPO is the N kept below.', '',
          '| Tree | Arms | ZT from |', '|---|---|---|']
    md += [f'| `{t}` | {" ".join(a)} | {meta["zt_source"][t]} |' for t, a in meta['arms'].items()]
    md += ['', '| Task | Metric | Method | mean | lo | hi | n | mark |',
           '|---|---|---|---|---|---|---|---|']
    md += [f'| {task} | {metric} | {label(m)} | {mean:.3g} | {lo:.3g} | {hi:.3g} | {n} | {mark} |'
           for task, metric, m, mean, lo, hi, n, mark in table]
    (out / f'{stem}.md').write_text('\n'.join(md) + '\n')
    print(f'wrote {out / stem}.md')


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--extract', action='store_true',
                    help='re-read the runs into the saved data first')
    ap.add_argument('--width', type=float, default=None,
                    help='inches (default the text width)')
    ap.add_argument('--row-height', type=float, default=None,
                    help='inches a metric row (default 0.9, 0.36 with --main)')
    ap.add_argument('--font-size', type=float, default=5.5)
    ap.add_argument('--main', action='store_true',
                    help='the main-text subset (Cum. centroid, ZT) as metrics_main')
    args = ap.parse_args()
    if args.row_height is None:
        args.row_height = 0.36 if args.main else 0.9
    if args.width is None:
        args.width = TEXT_WIDTH_IN
    main_keys = ('cum_centroid', 'zt')
    args.metrics = [m for m in METRICS if not args.main or m[0] in main_keys]
    args.out, args.stem = (pcl.FINAL, 'metrics_main') if args.main else (OUT, STEM)
    if args.extract:
        extract()
    if not DATA.exists():
        sys.exit(f'no {DATA}: run with --extract first')
    meta = json.loads(DATA.read_text())

    lp.METHOD_STYLE['pbt'] = {**lp.METHOD_STYLE['pbt'], 'label': 'PBT-PPO'}
    plt.rcParams.update({
        'font.size': args.font_size, 'axes.titlesize': args.font_size + 0.5,
        'xtick.labelsize': args.font_size - 1, 'ytick.labelsize': args.font_size - 0.5,
        'axes.linewidth': 0.5, 'xtick.major.width': 0.5, 'xtick.major.size': 2,
    })
    panels = column_order([(title, meta['panels'][pcl._key(tree, cell)])
                           for fig, tree, cell, title in pcl.PANELS if fig == 'main'])
    if args.main:
        panels += [(title, meta['panels'][pcl._key(tree, cell)])
                   for fig, tree, cell, title in pcl.PANELS if fig == 'tradeoff']
    union = {m for _, cells in panels for m in cells}
    methods = [m for m in lp.METHOD_ORDER if m in union]
    table = draw_main(meta, methods, args) if args.main else draw(panels, methods, args)
    write_markdown(meta, table, args.out, args.stem)
    return 0


if __name__ == '__main__':
    sys.exit(main())
