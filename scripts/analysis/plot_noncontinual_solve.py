"""The stationary (noncontinual) centroid lineplots of every task the paper
reports, and the LaTeX tables that go with them: how much reward each method
collected and how quickly it converged -- neither needs a solved threshold.

    .venv/bin/python scripts/analysis/plot_noncontinual_solve.py            # from the saved data
    .venv/bin/python scripts/analysis/plot_noncontinual_solve.py --extract  # re-read the runs first

    -> paper/visuals/final/noncontinual_solve.{pdf,png}          gymnax, MiniGrid, Brax,
                                                                 then one panel a Kinetix level
       paper/visuals/final/noncontinual_solve.tex                the table (booktabs, xcolor),
                                                                 one row a task, Kinetix levels last
       paper/visuals/final/noncontinual_solve.md                 both tables, with definitions
    (paper = projects/iclr_2027/paper)

Two steps. `--extract` reads the stationary runs through the symlink trees
paper/<suite>/data and saves every curve the figures draw -- per seed, raw,
on the generation axis -- with the Cum. elite scores to
paper/visuals/final/data/noncontinual_solve.{npz,json}. Without it the script
reads only those two files, so the figures and tables rebuild from the paper
directory alone (the run trees are ~110 GB, the extract a few MB).

The figures keep the style of the NeurIPS rebuttal's stationary gymnax plot
(scripts/outdated/neurips_2026_rebuttal/make_figures.py `noncontinual_gymnax`):
one legend row on top, generations on x, plus a light grid. That plot drew
`best_fitness`, the best individual scored on the episodes it was selected on;
these draw the CENTROID, the curve `make_lineplot.py --phase noncontinual
--metric centroid` draws, loaded through its `load_report` (same runs, arms,
rolling median and bootstrap band). The two differ most for a GA whose archive
never consolidates: the plain GA's best individual on MountainCar read ~-95 and
the mean of its weights -500 in half the seeds, which is why the MountainCar GA
is `ga_focus_explore` since 2026-09-15
(projects/iclr_2027/runs_ga_focus_mountaincar/README.md).

MiniGrid is drawn over its first 250 of 4000 generations: every arm has
converged by then, and the smoothing window is 1% of the DRAWN records so the
cut does not blur the rise it zooms in on. Kinetix is twenty levels with one
stationary run each, so it has a block of the figure and a table of its own,
one panel and one row a level.

    Cum.    the area under the (unsmoothed) curve, reward x generations /
            1000, mean over trials: the noncontinual table's `Cum. centroid`,
            computed by the same call.
    Conv.   the generation at which the smoothed curve first covers CONVERGE
            (95%) of its own rise, from its first record to its mean over the
            last TAIL (5%) of the budget; median over trials. Scale-free and
            relative to the trial's own final level, so it needs no threshold
            -- read it with Cum.: a trial that converges early to a poor level
            has a small Conv. and a small Cum. A trial that never rises converges
            at its first record. Resolution is one record (RL logs per update,
            ~5 generations on Kinetix).
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

REPO = pathlib.Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO / 'scripts'))
import make_lineplot as lp                                 # noqa: E402
import es_arm                                              # noqa: E402

TEXT_WIDTH_IN = 5.5                                        # ICLR \linewidth
# Drawn large and shrunk in print like Figure 2 (plot_continual_combined.py:
# 16.85 in drawn, printed at 0.82\linewidth, x0.268), with its rcParams, so
# text, curves and grid print at the same size in both figures.
PRINT_SCALE = 0.82 * TEXT_WIDTH_IN / 16.85
DRAW_WIDTH_IN = TEXT_WIDTH_IN / PRINT_SCALE

PROJECT = REPO / 'projects/iclr_2027'
OUT = PROJECT / 'paper/visuals/final/noncontinual_solve'
DATA = OUT.parent / 'data' / OUT.name       # .npz curves, .json the rest

# (stationary tree, cell, title, last generation drawn -- None = the whole
# budget), in drawing order. Each tree is the suite's paper/<suite>/data: symlinks
# to the stationary runs finish_iclr.sh reads (README.md there says which).
COLUMNS = [
    ('paper/gymnax/data',     'CartPole_v1',    'CartPole',       None),
    ('paper/gymnax/data',     'Acrobot_v1',     'Acrobot',        None),
    ('paper/gymnax/data',     'MountainCar_v0', 'MountainCar',    None),
    ('paper/minigrid/data',   'MiniGrid_8x8',   'MiniGrid 8x8',   250),
    ('paper/minigrid/data',   'MiniGrid_16x16', 'MiniGrid 16x16', 250),
    ('paper/mjx/cheetah/data', 'cheetah',        'HalfCheetah',    None),
]
KINETIX = 'paper/kinetix/data'
# finish_iclr.sh THE ARMS: no novelty arm. ES and NES are one method, "ES", and
# PBT-PPO N=8 and N=2 are one method too: each figure (and its table) draws ONE
# of each pair. The ES arm is fixed by the paper: NES everywhere, the latest
# OpenES (`es`, plain OpenES since 2026-09-24) on Kinetix. The PBT arm is the one with the higher
# stationary Cum. elite summed over the figure's panels (es_arm.pick).
ARMS = ['ga', 'es', 'nes', 'ppo', 'trac', 'redo', 'cchain', 'pbt', 'pbt2']
ES_KEPT = {'main': 'nes', 'kinetix': 'es'}
TAIL = 0.05             # a trial's final level: its mean over this last fraction
CONVERGE = 0.95         # Conv.: first generation past this fraction of the rise
# The LaTeX table's suite group and short row name of each task.
TEX_HEAD = {'CartPole': ('gymnax', 'CartPole'), 'Acrobot': ('gymnax', 'Acrobot'),
            'MountainCar': ('gymnax', 'MountainCar'),
            'MiniGrid 8x8': ('MiniGrid', r'8$\times$8'),
            'MiniGrid 16x16': ('MiniGrid', r'16$\times$16'),
            'HalfCheetah': ('Brax', 'HalfCheetah')}
# Method names short enough to head a column.
TEX_METHOD = {'trac': 'TRAC', 'redo': 'ReDo', 'pbt': 'PBT'}
# Which variant an `es` / `pbt` row is, for the captions.
VARIANT = {'es': 'OpenES', 'nes': 'NES', 'pbt': '$N{=}8$', 'pbt2': '$N{=}2$'}
FALLBACK_MARK = r'$^\dagger$'


def _kept_note(kept, tex=True):
    es, pbt = (VARIANT.get(k, k) for k in kept)
    note = (f'ES is {es}; PBT-PPO is the better of $N{{=}}8$ and $N{{=}}2$ by '
            f'stationary Cum. elite ({pbt}).')
    return note if tex else note.replace('$N{=}', 'N=').replace('$', '')


def load_tree(tree):
    """`(data, per_gen)` for one stationary tree, via make_lineplot."""
    root = PROJECT / tree
    present = {p.name for p in (root / 'noncontinual').iterdir() if p.is_dir()}
    methods = [m for m in ARMS if m in present]
    args = lp.parse_args([str(root), '--phase', 'noncontinual', '--metric',
                          'centroid', '--methods', *methods, '--out', '-'])
    rep = lp.load_report(args)
    return rep.data, rep.per_gen


def _key(tree, cell):
    return f'{tree}|{cell}'


def extract():
    """Read every panel's runs and write DATA.npz (`<tree>|<cell>|<arm>|gens`
    and `...|curves`, seeds x records, unsmoothed) and DATA.json (Cum. elite a trial for the ES and PBT pairs, the runs each arm resolves to)."""
    wanted = {}
    for tree, cell, *_ in COLUMNS:
        wanted.setdefault(tree, []).append(cell)
    wanted[KINETIX] = None                          # every Kinetix level
    arrays = {}
    meta = {'elite_cum': {'es': {}, 'pbt': {}}, 'sources': {}}
    for tree, cells in wanted.items():
        data, per_gen = load_tree(tree)
        cells = cells or [c for c in data if c.startswith('Kinetix_')]
        for cell in cells:
            for m, (x, curves) in data[cell].items():
                arrays[f'{_key(tree, cell)}|{m}|gens'] = x / per_gen if per_gen else x
                arrays[f'{_key(tree, cell)}|{m}|curves'] = np.asarray(curves, dtype=np.float32)
        for name, arms in (('es', es_arm.ARMS), ('pbt', es_arm.PBT_ARMS)):
            for c, v in es_arm.load(PROJECT / tree, 'noncontinual', arms=arms).items():
                if c in cells:
                    meta['elite_cum'][name][_key(tree, c)] = v
        for arm_dir in sorted((PROJECT / tree / 'noncontinual').iterdir()):
            if arm_dir.name in ARMS:
                meta['sources'][f'{tree}/noncontinual/{arm_dir.name}'] = str(
                    arm_dir.resolve().relative_to(PROJECT))
    meta['extracted'] = datetime.date.today().isoformat()
    DATA.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(f'{DATA}.npz', **arrays)
    pathlib.Path(f'{DATA}.json').write_text(json.dumps(meta, indent=1) + '\n')
    print(f'wrote {DATA}.npz, {DATA}.json')


def load_data():
    """`({(tree, cell): {arm: (gens, curves)}}, meta)` from DATA."""
    if not pathlib.Path(f'{DATA}.npz').exists():
        sys.exit(f'no {DATA}.npz: run with --extract first')
    curves = {}
    with np.load(f'{DATA}.npz') as npz:
        for name in npz.files:
            tree, cell, m, field = name.split('|')
            if field == 'gens':
                curves.setdefault((tree, cell), {})[m] = (
                    npz[name], npz[f'{tree}|{cell}|{m}|curves'].astype(float))
    return curves, json.loads(pathlib.Path(f'{DATA}.json').read_text())


def converged_at(sm, gens):
    """Per trial, the first generation at which the smoothed curve covers
    CONVERGE of its rise from its first record to its mean over the last TAIL
    of the budget -- its first record where it never rises."""
    k = max(1, int(round(TAIL * sm.shape[-1])))
    start, final = sm[:, :1], sm[:, -k:].mean(axis=-1, keepdims=True)
    reached = sm >= start + CONVERGE * np.maximum(final - start, 0)
    return gens[reached.argmax(axis=-1)]


def build_column(arms, xmax):
    """`{'curves': {m: (gens, per_seed)}, 'rows': {m: stats}}` for one cell.
    The smoothing is make_lineplot's rolling median over 1% of the records
    DRAWN, so a cut axis is not smoothed at the full run's scale; Conv. reads
    the whole budget."""
    col = {'curves': {}, 'rows': {}}
    for m, (gens, curves) in arms.items():
        drawn = gens <= (xmax if xmax is not None else gens[-1])
        sm = lp.smooth(curves, max(int(drawn.sum() * 0.01) | 1, 1))
        col['curves'][m] = (gens[drawn], sm[:, drawn])
        col['rows'][m] = dict(n=len(curves), conv=_median(converged_at(sm, gens)),
                              cum=np.mean([lp.cumulative_reward(gens, c, gens[-1]) / 1e3
                                           for c in curves]))
    return col


def keep_one_arm(cols, per_cell, arms, kept=None):
    """Drop, from every column in `cols`, the arm of the pair `arms` (ES/NES or
    PBT N=8/N=2) that is not `kept` -- by default the one with the lower
    stationary Cum. elite summed over every cell in `per_cell` (es_arm.pick).
    ONE arm for the whole figure and its table: the paper treats each pair as
    one method. A column where `kept` was not run keeps the other arm instead,
    marked in `col['fallback']` (HalfCheetah has PBT N=8 only). Returns the
    kept arm."""
    if kept is None:
        kept, _, _ = es_arm.pick(per_cell, arms)
    for col in cols:
        use = kept if kept in col['rows'] else next(
            (a for a in arms if a in col['rows']), kept)
        if use != kept:
            col.setdefault('fallback', set()).add(arms[0])
        for key in ('curves', 'rows'):
            for arm in arms:
                if use and arm != use:
                    col[key].pop(arm, None)
            # Filed under the pair's first name, one legend entry a pair, as in
            # plot_continual_lineplots.py: `es` is "ES", `pbt` "PBT-PPO".
            if use and use != arms[0] and use in col[key]:
                col[key][arms[0]] = col[key].pop(use)
    return kept


def _median(values):
    return float(np.median(values)) if values.size else None


def _gens(v):
    return '' if v is None else f'{v:.0f}'


def _cum(v):
    a = abs(v)
    return f'{v:.0f}' if a >= 100 else f'{v:.1f}' if a >= 10 else \
        f'{v:.2f}' if a >= 1 else f'{v:.3f}'


def draw_curves(ax, col, methods, xmax=None):
    """One panel: task switches dashed, then each of `methods` in `col` as its
    mean with a bootstrap CI band."""
    for b in col.get('edges', ()):
        ax.axvline(b, color='0.7', lw=0.5, ls='--', zorder=0)
    for m in methods:
        if m not in col['curves']:
            continue
        gens, per_seed = col['curves'][m]
        mid, lo, hi = lp.bootstrap_ci(per_seed)
        style = lp.METHOD_STYLE[m]
        ax.plot(gens, mid, color=style['color'], lw=1.4)
        ax.fill_between(gens, lo, hi, color=style['color'], alpha=0.2, lw=0)
    ax.set_xlim(0, xmax if xmax is not None else max(
        g[-1] for g, _ in col['curves'].values()))
    ax.grid(True, color='0.9', lw=0.6)
    ax.set_axisbelow(True)


def draw(blocks, methods, stem, legend_ncol=None):
    """`blocks` = [(heading, panels, ncols, panel)], stacked into one figure
    with one legend, to `<stem>.pdf/.png`. `panels` = [(title, col, xmax)],
    `ncols` a row, `panel` = (width, height) in inches, `heading` (or None) a
    line above the block. `col['edges']`, where a column has them, are task
    switches in generations, drawn dashed (plot_continual_lineplots.py).
    `legend_ncol` wraps the legend (None = one row)."""
    drawn = [m for m in methods
             if any(m in col['curves'] for _, panels, _, _ in blocks for _, col, _ in panels)]
    heights = [p[1] * -(-len(panels) // ncols) + (0.2 / PRINT_SCALE if heading else 0)
               for heading, panels, ncols, p in blocks]
    width = max(p[0] * ncols for _, _, ncols, p in blocks)
    height = sum(heights)
    fig = plt.figure(figsize=(width, height), layout='constrained')
    subfigs = fig.subfigures(len(blocks), 1, height_ratios=heights, squeeze=False)[:, 0]
    for sub, (heading, panels, ncols, _p) in zip(subfigs, blocks):
        nrows = -(-len(panels) // ncols)
        axes = sub.subplots(nrows, ncols, squeeze=False)
        flat = [ax for row in axes for ax in row]
        for ax in flat[len(panels):]:
            ax.set_visible(False)
        for i, (ax, (title, col, xmax)) in enumerate(zip(flat, panels)):
            draw_curves(ax, col, drawn, xmax)
            ax.set_title(title)
            if i >= len(panels) - ncols:                   # bottom row only
                ax.set_xlabel('Generations')
            ax.spines[['top', 'right']].set_visible(False)
        # A block whose panels share one x range (Kinetix) labels its ticks once.
        if len({ax.get_xlim() for ax in flat[:len(panels)]}) == 1:
            for ax in flat[:len(panels) - ncols]:
                ax.tick_params(labelbottom=False)
        sub.supylabel('Episodic reward', fontsize=plt.rcParams['font.size'])
        if heading:
            sub.suptitle(heading, fontsize='large')
    handles = [plt.Line2D([], [], color=lp.METHOD_STYLE[m]['color'], lw=1.6)
               for m in drawn]
    fig.legend(handles, [lp.METHOD_STYLE[m]['label'] for m in drawn],
               loc='outside upper center', ncol=legend_ncol or len(drawn), frameon=False,
               handlelength=2.2, handletextpad=0.4, columnspacing=1.3)
    for ext in ('pdf', 'png'):
        fig.savefig(f'{stem}.{ext}', bbox_inches='tight', dpi=300)
    plt.close(fig)
    print(f'wrote {stem}.pdf, {stem}.png')


DEFINITIONS = (
    r'Each cell is the cumulative return, the area under the centroid curve '
    r'(reward $\times$ generations / 1000, mean over trials), with in grey the '
    r'generation at which the smoothed curve first covers 95\% of its rise from '
    r'its first value to its mean over the last 5\% of training (median over '
    r'trials); neither needs a solved threshold.')


def _tex_cell(r, fallback=False):
    if r is None:
        return '--'
    mark = FALLBACK_MARK if fallback else ''
    return rf'{_cum(r["cum"])}{mark}\,\textcolor{{gray}}{{\scriptsize {_gens(r["conv"])}}}'


def _fallback_note(kept):
    other = {'pbt': 'pbt2', 'pbt2': 'pbt'}.get(kept)
    return (rf'{FALLBACK_MARK}PBT-PPO {VARIANT[other]}: no {VARIANT[kept]} run.'
            if other else '')


def _trials_note(columns):
    """`Trials: ...` for the columns whose methods ran different trial counts."""
    notes = []
    for name, col in columns:
        ns = {}
        for m, r in sorted(col['rows'].items(), key=lambda mr: lp.METHOD_ORDER.index(mr[0])):
            ns.setdefault(r['n'], []).append(lp.METHOD_STYLE[m]['label'])
        if len(ns) > 1:
            notes.append(f'{name} ' + ', '.join(
                f'{n} for {"/".join(labels)}' for n, labels in sorted(ns.items(), reverse=True)))
    return ('Trials: ' + '; '.join(notes) + '.') if notes else ''


def write_tex(cols, methods, levels, kept_main, kept_kx):
    """`<OUT>.tex`: one table, methods across and one row a task, grouped by
    suite -- the twenty Kinetix levels last."""
    groups = []
    for _tree, _cell, title, _xmax, col in cols:
        suite, name = TEX_HEAD[title]
        if not groups or groups[-1][0] != suite:
            groups.append((suite, []))
        groups[-1][1].append((name, col))
    groups.append(('Kinetix', [(title.split()[0], col) for title, col in levels]))  # h0 ... h19
    rows = [r for _, rs in groups for r in rs]
    heads = [m for m in methods if any(m in col['rows'] for _, col in rows)]
    fell = any(c[4].get('fallback') for c in cols)
    es = [VARIANT[k] for k in (kept_main[0], kept_kx[0])]
    pbt = [VARIANT[k] for k in (kept_main[1], kept_kx[1])]
    caption = ' '.join(filter(None, (
        r'Stationary tasks. ' + DEFINITIONS,
        f'ES is {es[0]} ({es[1]} on Kinetix); PBT-PPO is the better of $N{{=}}8$ and '
        f'$N{{=}}2$ by stationary Cum. elite, picked separately for Kinetix '
        f'({pbt[0]}; {pbt[1]} on Kinetix).',
        _fallback_note(kept_main[1]) if fell else '',
        _trials_note([(TEX_HEAD[c[2]][1], c[4]) for c in cols]),
        r'Kinetix: one stationary run a level. --: not run.')))
    n = len(heads) + 1
    lines = ['% Built by scripts/analysis/plot_noncontinual_solve.py -- rerun it, do not edit.',
             r'% Needs \usepackage{booktabs} and \usepackage{xcolor}.',
             r'\begin{table}[t]', r'\centering', rf'\caption{{{caption}}}',
             r'\label{tab:noncontinual_solve}', r'\footnotesize',
             r'\setlength{\tabcolsep}{3pt}',
             r'\begin{tabular}{l' + 'r' * len(heads) + '}', r'\toprule',
             'Task & ' + ' & '.join(TEX_METHOD.get(m, lp.METHOD_STYLE[m]['label'])
                                   for m in heads) + r' \\']
    for suite, rs in groups:
        lines += [r'\midrule', rf'\multicolumn{{{n}}}{{l}}{{\textit{{{suite}}}}} \\']
        for name, col in rs:
            lines.append(name + ' & ' + ' & '.join(
                _tex_cell(col['rows'].get(m), m in col.get('fallback', ()))
                for m in heads) + r' \\')
    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    path = pathlib.Path(f'{OUT}.tex')
    path.write_text('\n'.join(lines) + '\n')
    print(f'wrote {path}')


def write_markdown(cols, methods, levels, kept_main, kept_kx):
    main = [m for m in methods if any(m in c[4]['rows'] for c in cols)]
    head = ['Method']
    for _tree, _cell, title, _xmax, _col in cols:
        head += [f'{title} Cum.', f'{title} Conv.']
    lines = ['# noncontinual_solve', '',
             'How much reward each method collects on the STATIONARY version of each '
             'task, and how quickly it converges. Built by '
             '`scripts/analysis/plot_noncontinual_solve.py`, with the figure and the '
             'LaTeX tables beside this file, from the curves in '
             '`data/noncontinual_solve.{npz,json}` (`--extract` refreshes them from '
             'the runs). No column needs a solved threshold.', '',
             '| Column | What it is |', '|---|---|',
             '| **Cum.** | Area under the unsmoothed `centroid` curve, reward x '
             'generations / 1000, mean over trials -- the `Cum. centroid` of '
             '`noncontinual_centroid_table.md`, by the same call. |',
             f'| **Conv.** | The generation at which the smoothed curve (rolling median '
             f'over 1% of the drawn records) first covers {CONVERGE:.0%} of its rise '
             f'from its first record to its mean over the last {TAIL:.0%} of the '
             'budget, median over trials. A trial that never rises converges at its '
             'first record, so read it with Cum. RL is on the same '
             'generation-equivalent axis. |',
             '', '| Task | Tree (noncontinual) | Generations drawn |', '|---|---|---|']
    for tree, _cell, title, xmax, col in cols:
        total = max(g[-1] for g, _ in col['curves'].values())
        drawn = f'{xmax} of {total:.0f}' if xmax is not None else f'{total:.0f}'
        lines.append(f"| {title} | `{tree}` | {drawn} |")
    kx_total = max(g[-1] for g, _ in levels[0][1]['curves'].values())
    lines += [f"| Kinetix, each level | `{KINETIX}` | {kx_total:.0f} |", '',
              'The MountainCar GA is `ga_focus_explore` '
              '(`runs_ga_focus_mountaincar/README.md`).', '',
              'HalfCheetah RL arms are the ant PPO shape only (`paper/mjx/cheetah/data/README.md`); '
              'the previous-shape runs are in `noncontinual_prevppo/` and not drawn.', '']
    trials = _trials_note([(c[2], c[4]) for c in cols])
    fell = any(c[4].get('fallback') for c in cols)
    lines += ['Main table: ' + _kept_note(kept_main, tex=False)
              + (' ' + _fallback_note(kept_main[1]).replace(FALLBACK_MARK, '(dagger) ')
                 .replace('$N{=}', 'N=').replace('$', '') if fell else '')
              + (' ' + trials if trials else '')
              + ' Kinetix: ' + _kept_note(kept_kx, tex=False), '',
              '| ' + ' | '.join(head) + ' |', '|' + '---|' * len(head)]
    for m in main:
        row = [lp.METHOD_STYLE[m]['label']]
        for *_, col in cols:
            r = col['rows'].get(m)
            dagger = ' (dagger)' if m in col.get('fallback', ()) else ''
            row += [_cum(r['cum']) + dagger, _gens(r['conv'])] if r else ['--', '']
        lines.append('| ' + ' | '.join(row) + ' |')
    kx = [m for m in methods if any(m in col['rows'] for _, col in levels)]
    lines += ['', '## Kinetix, per level', '', 'Cell: Cum. · Conv.', '',
              '| Level | ' + ' | '.join(lp.METHOD_STYLE[m]['label'] for m in kx) + ' |',
              '|' + '---|' * (len(kx) + 1)]
    for title, col in levels:
        cells = []
        for m in kx:
            r = col['rows'].get(m)
            cells.append(f"{_cum(r['cum'])} · {_gens(r['conv'])}" if r else '--')
        lines.append(f'| {title} | ' + ' | '.join(cells) + ' |')
    pathlib.Path(f'{OUT}.md').write_text('\n'.join(lines) + '\n')
    print(f'wrote {OUT}.md')


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--extract', action='store_true',
                    help='re-read the runs under paper/<suite>/data into the '
                         'saved data before drawing')
    ap.add_argument('--ncols', type=int, default=3,
                    help='panels a row, main figure (six panels: two full rows)')
    ap.add_argument('--kinetix-ncols', type=int, default=5,
                    help='panels a row, Kinetix block')
    args = ap.parse_args()
    # Figure 2's sizes, at DRAW_WIDTH_IN; the figure goes in at \linewidth.
    plt.rcParams.update({'font.size': 19, 'xtick.labelsize': 16, 'ytick.labelsize': 16})

    if args.extract:
        extract()
    curves, meta = load_data()
    cols = [(tree, cell, title, xmax, build_column(curves[tree, cell], xmax))
            for tree, cell, title, xmax in COLUMNS]

    def kept_pair(columns, keys, figure):
        es = keep_one_arm(columns, None, es_arm.ARMS, ES_KEPT[figure])
        pbt = keep_one_arm(columns, {k: v for k, v in meta['elite_cum']['pbt'].items()
                                     if k in keys}, es_arm.PBT_ARMS)
        print(f'{figure} figure keeps {es} and {pbt}')
        return [es, pbt]

    kept_main = kept_pair([c[4] for c in cols],
                          {_key(tree, cell) for tree, cell, *_ in COLUMNS}, 'main')
    for _tree, _cell, title, _xmax, col in cols:
        print(f'{title:15s} '
              + '  '.join(f"{m}={_cum(r['cum'])}@{_gens(r['conv'])}(n={r['n']})"
                          for m, r in sorted(col['rows'].items())))
    # Every Kinetix level on its own, in the levels' order (make_lineplot.ENV_TITLES).
    levels = [(lp.ENV_TITLES[e], build_column(curves[KINETIX, e], None))
              for e in lp.ENV_TITLES if e.startswith('Kinetix_') and (KINETIX, e) in curves]
    kept_kx = kept_pair([col for _, col in levels],
                        {_key(tree, cell) for tree, cell in curves if tree == KINETIX},
                        'kinetix')
    lp.METHOD_STYLE['pbt'] = {**lp.METHOD_STYLE['pbt'], 'label': 'PBT-PPO'}
    union = {m for *_, col in cols for m in col['rows']} | \
            {m for _, col in levels for m in col['rows']}
    methods = [m for m in lp.METHOD_ORDER if m in union]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    draw([(None, [(title, col, xmax) for _tree, _cell, title, xmax, col in cols],
           args.ncols, (DRAW_WIDTH_IN / args.ncols, 1.55 / PRINT_SCALE)),
          ('Kinetix', [(title.split()[0], col, None) for title, col in levels],  # h0 ... h19
           args.kinetix_ncols, (DRAW_WIDTH_IN / args.kinetix_ncols, 0.85 / PRINT_SCALE))],
         methods, OUT)
    write_tex(cols, methods, levels, kept_main, kept_kx)
    write_markdown(cols, methods, levels, kept_main, kept_kx)
    return 0


if __name__ == '__main__':
    sys.exit(main())
