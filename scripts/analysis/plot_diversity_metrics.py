"""The diversity study as a metrics figure: one column per gymnax task (noise,
then action reversal), one row per metric, one line per NE method, drawn by
the metrics figure's own panel code (make_metrics_figure.draw_panel).

    .venv/bin/python scripts/analysis/plot_diversity_metrics.py --extract   # re-read the runs first
    .venv/bin/python scripts/analysis/plot_diversity_metrics.py [--figure main|appendix] [--agent elite|centroid]

    -> projects/iclr_2027/paper/visuals/final/diversity_metrics[_appendix]_elite.{pdf,png,md}
       (the centroid variants go to paper/visuals/, they are not in the paper)

`--figure main` (the default) has the noise columns (10 sub-tasks); `appendix`
has every other family in FIGURES (gymnax action reversal so far).

Built in two steps, like the other final figures: `--extract` reads the runs
through paper/diversity/data/<family>/continual/{nes,ga,dns_gaussian}/<cell>
(cell-level symlinks, see its README) and the forgetting passes, and writes
both agents' per-seed numbers to visuals/final/data/diversity_metrics.json;
the plot step reads only that file.

Rows, raw values (each column keeps its own x axis, as in the metrics figure):

    Cum. return            the agent's training curve averaged over the run
                           (`elite_eval_fitness` / `centroid_fitness`)
    Cum. (centroid)        elite figure only: the centroid's, beside it
    F                      forgetting of the same agent, from the forgetting
                           pass (plot_diversity_tradeoff.divergence)
    Behavioural diversity  `bd_probe_disagreement`, run mean
    Genomic diversity      `bd_genomic_diversity`, run mean (log axis)

The two diversity rows describe the population, so they are the same under
both agents. Methods are ES, GA and GA + Novelty only: RL has no population
to measure. The marks are GA + Novelty against GA (two-sided Mann-Whitney U,
Holm over the columns of a row, within one figure; * p<.05, ** p<.01, *** p<.001), placed on
GA + Novelty's line with its sign (+ higher than GA, - lower); with ten seeds a
side and three columns the smallest Holm p is about 5e-4 -- the metrics figure's NE-vs-RL marks have no second
family here. Runs: plot_diversity_plasticity.py's (the GA + Novelty re-runs, the
plain GA on MountainCar); ES is NES, as in every final gymnax figure.
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
from matplotlib.gridspec import GridSpec                   # noqa: E402
from matplotlib.ticker import FuncFormatter, MaxNLocator   # noqa: E402
from scipy.stats import mannwhitneyu                       # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import plot_diversity_plasticity as pdp                    # noqa: E402
import plot_diversity_tradeoff as pdt                      # noqa: E402
import plot_metrics_overview as pmo                        # noqa: E402
from make_metrics_figure import draw_panel                 # noqa: E402

psp, lp, PROJECT, REPO = pdp.psp, pdp.lp, pdp.PROJECT, pdp.REPO
STEM = 'diversity_metrics'
RIGHT_IN = 0.3            # room for the last column's marks
FINAL = PROJECT / 'paper/visuals/final'
DATA = FINAL / 'data' / f'{STEM}.json'
RUNS = PROJECT / 'paper/diversity/data'
# A family is one data root under RUNS (symlinks, see its README), its cells
# by env, and where its forgetting values come from.
def _paper_family(fam, root, sub):
    """The two paper families: cells from plot_diversity_plasticity, F from
    the paper pass (`sub` under paper/) overlaid by the diversity passes."""
    return dict(fam=fam, root=root, sub=sub,
                cells={env: cell_of(fam, env) for env in pdp.ENVS},
                forget=lambda agent: pdt.divergence(fam, sub, agent))


def kept_divergence(root, sub, agent):
    """`{run_dir: F}` for a `kept_*` root: the paper pass of the matching
    family (`sub`, may be None) overlaid by every pass under
    paper/diversity/results/<root>/<agent>/*."""
    out = {}
    if sub:
        out.update({rd: v['F'] for rd, v in
                    lp.load_divergence(PROJECT / 'paper' / sub / 'results' / agent).items()})
    for d in sorted((PROJECT / 'paper/diversity/results' / root / agent).glob('*')):
        out.update({rd: v['F'] for rd, v in lp.load_divergence(d).items()})
    return out


def _kept_family(fam, root, cells, sub=None):
    return dict(fam=fam, root=root, sub=sub, cells=cells,
                forget=lambda agent: kept_divergence(root, sub, agent))


# The families each figure shows, in column order. `main`/`appendix` are the
# paper's two families. `kept` (2026-09-18) is every cell where a switch
# floors the GA's population or the paper reports the pair, with every
# GA + Novelty column on the top-tier selection rule
# (source/algorithms/ne/dns.py): MountainCar noise is the sigma-0.5 collapse
# cell (0.1 never floors the GA), and `action map` is DeepSea 12
# (gymnax_classic.DeepSeaEnv), the sparse grid whose sub-task is a per-cell
# action reversal.
FIGURES = {
    'main': [('noise', 'noise_10task', 'gymnax/noise/10task')],
    'appendix': [('action reversal', 'actions_2task', 'gymnax/actions/2task')],
    'kept': [
        _kept_family('noise', 'kept_noise',
                     {'CartPole_v1': 'CartPole_v1_sigma1.0', 'Acrobot_v1': 'Acrobot_v1_sigma1.0',
                      'MountainCar_v0': 'MountainCar_v0_sigma0.5'}, 'gymnax/noise/10task'),
        _kept_family('action reversal', 'kept_actions',
                     {e: f'{e}_sigma1.0' for e in pdp.ENVS}, 'gymnax/actions/2task'),
        _kept_family('action map', 'kept_deepsea',
                     {'DeepSea12_bsuite': 'DeepSea12_bsuite_sigma1.0'}),
    ],
}
PAPER_FIGURES = ('main', 'appendix')      # share one data file; `kept` has its own


def families(figure):
    if figure in PAPER_FIGURES:
        return [_paper_family(*f) for fig in PAPER_FIGURES for f in FIGURES[fig]]
    return FIGURES[figure]


def data_path(figure):
    return DATA if figure in PAPER_FIGURES else FINAL / 'data' / f'{STEM}_{figure}.json'
# The arm directory each method is read from; ES is NES (CLAUDE.md).
ARM_DIR = {'es': 'nes', 'ga': 'ga', 'dns_gaussian': 'dns_gaussian'}
METHODS = list(pdp.NE_ARMS)
# (key, label, higher is better, log axis)
ROWS = [('cum', 'Cum. return', True, False),
        ('cum_centroid', 'Cum. (centroid)', True, False),
        ('F', 'F', False, False),
        ('bdiv', 'Behav. diversity', True, False),
        ('gdiv', 'Genomic diversity', True, True)]
COLUMNS = {'bdiv': 'bd_probe_disagreement', 'gdiv': 'bd_genomic_diversity'}


def read(trial, score):
    records = json.loads((trial / 'training_metrics.json').read_text())
    y = np.array([np.nan if r.get(score) is None else float(r[score]) for r in records])
    c = np.array([np.nan if r.get('centroid_fitness') is None
                  else float(r['centroid_fitness']) for r in records])
    out = {'cum': float(np.trapz(y, dx=1) / max(len(y) - 1, 1)),
           'cum_centroid': float(np.trapz(c, dx=1) / max(len(c) - 1, 1))}
    for k, col in COLUMNS.items():
        v = [float(r[col]) for r in records if r.get(col) is not None]
        out[k] = float(np.mean(v)) if v else np.nan
    return out


def cell_of(fam, env):
    return next(c for f, _t, _r, cells, _s in pdp.FAMILIES if f == fam
                for e, c in cells.items() if e == env)


def trials(root, m, cell):
    d = RUNS / root / 'continual' / ARM_DIR[m] / cell
    return sorted(p for p in d.glob('trial_*') if (p / 'training_metrics.json').exists())


def source(root, m, cell):
    """The repo-relative cell directory the link names, symlinks above it kept:
    the forgetting passes key their runs by that path."""
    link = RUNS / root / 'continual' / ARM_DIR[m] / cell
    return pathlib.Path(os.path.normpath(link.parent / os.readlink(link))).relative_to(REPO)


def extract(figure):
    """Write the figure's data file: `{agent: {'<family>|<env>': {row: {method: [per seed]}}}}`,
    plus the run directory of every per-seed value."""
    os.chdir(REPO)
    meta = {'rows': {}, 'runs': {}, 'links': {}}
    for agent in sorted(pdp.SCORE):
        score = pdp.SCORE[agent]
        panels = meta['rows'][agent] = {}
        for family in families(figure):
            fam, root = family['fam'], family['root']
            forget = family['forget'](agent)
            for env, cell in family['cells'].items():
                panel = panels[f'{fam}|{env}'] = {k: {} for k, *_ in ROWS}
                for m in METHODS:
                    src = source(root, m, cell)
                    meta['links'][f'{root}/continual/{ARM_DIR[m]}/{cell}'] = str(src)
                    runs = meta['runs'].setdefault(f'{fam}|{env}|{m}', [])
                    ts = trials(root, m, cell)
                    if len(ts) != pdp.N_TRIALS:
                        print(f'WARNING: {fam} {env} {m}: {len(ts)} trials')
                    for t in ts:
                        run = str(src / t.name)
                        r = read(t, score)
                        r['F'] = forget.get(run, np.nan)
                        if not np.isfinite(r['F']):
                            print(f'WARNING: {agent} {run}: no F')
                        for k, *_ in ROWS:
                            panel[k].setdefault(m, []).append(
                                None if not np.isfinite(r[k]) else r[k])
                        if agent == 'elite':
                            runs.append(run)
    meta['extracted'] = datetime.date.today().isoformat()
    out = data_path(figure)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(meta, indent=1) + '\n')
    print(f'wrote {out}')


def load(agent, figure):
    """`{(fam, env): {row: {method: array per seed}}}` from DATA."""
    path = data_path(figure)
    if not path.exists():
        sys.exit(f'no {path}: run with --extract first')
    rows = json.loads(path.read_text())['rows'][agent]
    return {tuple(key.split('|')): {k: {m: np.array([np.nan if v is None else v for v in vs])
                                        for m, vs in by_m.items()}
                                    for k, by_m in panel.items()}
            for key, panel in rows.items()}


def stars(p):
    return '***' if p < 1e-3 else '**' if p < 1e-2 else '*' if p < 0.05 else ''


def holm(ps):
    order = np.argsort(ps)
    adj, run = np.empty(len(ps)), 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - rank) * ps[i]))
        adj[i] = run
    return adj


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--agent', choices=sorted(pdp.SCORE), default='elite')
    ap.add_argument('--font_size', type=float, default=7.0)
    ap.add_argument('--figure', choices=sorted(FIGURES), default='main')
    ap.add_argument('--rows', default=None,
                    help='comma-separated row keys to draw (default: all); e.g. '
                         'cum,cum_centroid,F,bdiv for the paper')
    ap.add_argument('--col_width', type=float, default=None,
                    help='inches per task column (default: fit \\textwidth, at most 1.05)')
    ap.add_argument('--envs', default=None,
                    help='comma-separated env keys to keep as columns (default: all of '
                         'the figure); e.g. MountainCar_v0,DeepSea12_bsuite for the main text')
    ap.add_argument('--nbins', type=int, default=3,
                    help='x ticks per panel (3; 2 for narrow columns)')
    ap.add_argument('--suffix', default='',
                    help='appended to the output stem, e.g. _paper')
    ap.add_argument('--extract', action='store_true',
                    help='re-read the runs into visuals/final/data first')
    args = ap.parse_args()
    if args.extract:
        extract(args.figure)
    os.chdir(REPO)
    fs = args.font_size
    plt.rcParams.update({'font.size': fs, 'xtick.labelsize': fs - 1})
    data = load(args.agent, args.figure)
    rows = [r for r in ROWS if not (r[0] == 'cum_centroid' and args.agent == 'centroid')]
    if args.rows:
        keep = args.rows.split(',')
        rows = [r for r in rows if r[0] in keep]
    if args.agent == 'elite':
        rows[0] = ('cum', 'Cum. (elite)', True, False)
    cols = [(f['fam'], env) for f in families(args.figure) for env in f['cells']]
    if args.figure in PAPER_FIGURES:
        cols = [c for c in cols if c[0] == FIGURES[args.figure][0][0]]
    if args.envs:
        keep = args.envs.split(',')
        cols = [c for c in cols if c[1] in keep]

    # GA + Novelty vs GA, Holm over a row's columns.
    tests = {}
    for key, *_ in rows:
        ps, where = [], []
        for c in cols:
            a = np.asarray(data[c][key].get('dns_gaussian', []), float)
            b = np.asarray(data[c][key].get('ga', []), float)
            a, b = a[np.isfinite(a)], b[np.isfinite(b)]
            if a.size > 2 and b.size > 2:
                ps.append(mannwhitneyu(a, b).pvalue)
                where.append((c, a.mean() - b.mean()))
        for (c, diff), p, q in zip(where, ps, holm(np.array(ps)) if ps else []):
            tests[key, c] = (diff, p, q)

    # Default: the columns share \textwidth, so the figure goes in at its own
    # size and --font_size is the size in print.
    col_width = args.col_width or min(1.05, (pmo.TEXT_WIDTH_IN - pmo.LEFT_IN - RIGHT_IN) / len(cols))
    width = pmo.LEFT_IN + col_width * len(cols) + RIGHT_IN
    top_in, bottom_in, row_h = 0.5, 0.35, 0.95
    height = row_h * len(rows) + top_in + bottom_in
    fig = plt.figure(figsize=(width, height))
    gs = GridSpec(len(rows), len(cols), figure=fig, left=pmo.LEFT_IN / width,
                  right=1 - RIGHT_IN / width, top=1 - top_in / height,
                  bottom=bottom_in / height, wspace=0.45, hspace=0.45)
    table = []
    for r, (key, label, hib, log) in enumerate(rows):
        for c, (fam, env) in enumerate(cols):
            ax = fig.add_subplot(gs[r, c])
            cell = {m: v for m, v in data[fam, env][key].items()
                    if np.isfinite(v).any()}
            _t, points = draw_panel(ax, cell, METHODS, 'none', hib, fs, labels=c == 0)
            # draw_panel's cross-family marks are empty with NE alone; ours:
            test = tests.get((key, (fam, env)))
            if test and stars(test[2]):
                pt = next(p for p in points if p[0] == 'dns_gaussian')
                ax.annotate(('+' if test[0] > 0 else '\u2212') + stars(test[2]), (pt[3], len(METHODS) - 1 - METHODS.index('dns_gaussian')),
                            xytext=(2, 0), textcoords='offset points', va='center',
                            ha='left', fontsize=fs - 1, color=lp.METHOD_STYLE['dns_gaussian']['color'],
                            annotation_clip=False)
            table += [(label, fam, env, *p, test) for p in points]
            if log:
                ax.set_xscale('log')
            else:
                ax.xaxis.set_major_locator(MaxNLocator(nbins=args.nbins if col_width > 0.8 else 2,
                                                       min_n_ticks=2))
                ax.xaxis.set_major_formatter(FuncFormatter(pmo._compact))
            ax.tick_params(axis='x', pad=1.5)
            if c == 0:
                ax.annotate(label + ('' if hib else ' ↓'), xy=(0, 0.5),
                            xycoords='axes fraction', xytext=(-54, 0),
                            textcoords='offset points', rotation=90,
                            ha='center', va='center')
            if r == 0:
                # Sized for a 0.62 in column: "MountainCar" beside "CartPole".
                ax.set_title(lp.ENV_TITLES[env], pad=fs + 2.5,
                             fontsize=fs - 0.5)
                ax.annotate(fam.replace('action reversal', 'reversal'), (0.5, 1),
                            xycoords='axes fraction', xytext=(0, 2),
                            textcoords='offset points', ha='center', va='bottom',
                            color=pmo.MUTED, fontsize=fs - 1)
    # No footer: the marks are described in the caption and in the .md table.
    name = STEM + ('' if args.figure == 'main' else f'_{args.figure}')
    stem = f"{FINAL if args.agent == 'elite' else PROJECT / 'paper/visuals'}/{name}_{args.agent}{args.suffix}"
    for ext in ('pdf', 'png'):
        fig.savefig(f'{stem}.{ext}', dpi=300)
    plt.close(fig)
    print(f'wrote {stem}.pdf, {stem}.png  ({width:.1f} x {height:.1f} in)')

    lines = [f'# {pathlib.Path(stem).name}', '',
             f'Built by `scripts/analysis/plot_diversity_metrics.py`; agent: {args.agent} '
             '(Cum. return and F only; the diversity rows describe the population). Mean '
             '[95% bootstrap CI], raw values. ES is NES. Data: `visuals/final/data/diversity_metrics.json` '
             f"(extracted {json.loads(data_path(args.figure).read_text())['extracted']}); runs: `paper/diversity/data/`.", '',
             '| Row | Task | Method | n | mean [CI] | GA + Novelty - GA | p | Holm p |',
             '|---|---|---|---|---|---|---|---|']
    for label, fam, env, m, mean, lo, hi, n, _mark, test in table:
        extra = (f'{test[0]:+.3g} | {test[1]:.2g} | {test[2]:.2g}'
                 if test and m == 'dns_gaussian' else ' | | ')
        lines.append(f'| {label} | {lp.ENV_TITLES[env]}, {fam} | {psp._label(m)} | {n} | '
                     f'{mean:.3g} [{lo:.3g}, {hi:.3g}] | {extra} |')
    pathlib.Path(f'{stem}.md').write_text('\n'.join(lines) + '\n')
    print(f'wrote {stem}.md')
    return 0


if __name__ == '__main__':
    sys.exit(main())
