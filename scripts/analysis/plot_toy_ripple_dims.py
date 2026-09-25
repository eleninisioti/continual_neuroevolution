"""Appendix figure: the GA's advantage at crossing local optima (toy_local_optima)
disappears once the landscape is rugged along many coordinates, and ES keeps
finding the generalist, only more slowly.

    # re-read the sweeps (only when they change)
    JAX_PLATFORMS=cpu .venv/bin/python scripts/analysis/plot_toy_ripple_dims.py --extract
    # redraw from visuals/final/data/
    .venv/bin/python scripts/analysis/plot_toy_ripple_dims.py

    -> paper/visuals/final/appendix/toy_ripple_dims.{pdf,png,md,tex}
       paper/visuals/final/data/toy_ripple_dims.json
    (paper = projects/iclr_2027/paper)

The runs are the `ripdim_k<k>` stage of scripts/train/queue_toy.sh: the
`rugged` toy of toy_local_optima (two sub-tasks alternating every 100
generations for 1000, start on sub-task A's side, pop 64, 24 seeds) with
d = 128 parameters and the ripple digging local optima along the first k of
them, k = 2 .. 128. Every coordinate carries the same barrier, and coordinates
3 .. k have their best value at 0, where every run already starts. So a larger
k adds no barrier the search has to cross; what it adds is directions in which
a mutation is COSTLY. At k = 2 this is toy_local_optima's local-optima row with
126 null coordinates added (rugged_d386 shows null coordinates alone change
nothing).

Why the GA fails: an isotropic child at width sigma pays, on every rugged
coordinate, an expected a/4 * (1 - exp(-(2 pi sigma / lambda)^2 / 2)), so its
cost grows linearly in k whether or not those coordinates needed changing.
Truncation selection needs a child at least as good as its parent; once even
the best of a generation's 64 children loses more than the whole generalist
score, it can only keep what it has. ES never needs a good sample, only a
ranking correlated with the smoothed gradient, and the per-sample cost is
nearly a constant offset in that ranking: ES slows down, it does not stop.
Panel (d) is that cost, computed from the landscape (not the runs), at
sub-task A's first specialist peak, where the GA sits when it has to cross.

Reported agent: the CENTROID (the ES mean; the mean of the GA's elite archive),
as in toy_local_optima. Its ripple-free generalist score min(A, B) is recorded
every 10 generations (`cen_g`); a seed is a generalist when that reaches 98% of
the 0.70 peak, found = at any record, held = at the last. Two fixed widths,
sigma 0.2 (toy_local_optima's) and 0.4 (the widest swept); both arms at both.
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

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import plot_continual_lineplots as pcl                     # noqa: E402

lp = pcl.lp
REPO = pathlib.Path(__file__).resolve().parents[2]
RUNS = REPO / 'projects/iclr_2027/runs_toy'
STEM = 'toy_ripple_dims'
OUT = pcl.FINAL / 'appendix'
DATA = pcl.FINAL / 'data' / STEM
ARMS = {'es': 'nes', 'ga': 'ga'}           # figure name -> sweep method
KS = (2, 4, 8, 16, 32, 64, 128)
SIGMAS = (0.2, 0.4)
SIGMA_DASH = {0.2: '-', 0.4: (0, (3, 1.5))}
PANEL_LEVELS = (0.4, 1.6)                  # the two outcome panels
FIND_LEVEL = 0.4                           # the level every arm finds at
CHILDREN = 64                              # one generation's evaluations
DRAWS = 2000                               # generations drawn per cost point
AMP_GREY = {0.4: '0.65', 0.8: '0.45', 1.6: '0.1'}
TEXT_WIDTH_IN = 5.5                        # ICLR


def extract():
    import jax.numpy as jnp
    from jax import random
    from source.envs import toy_landscapes as tl

    meta = dict(extracted=datetime.date.today().isoformat(), ks=list(KS),
                sigmas=list(SIGMAS), cells={}, cost={})
    config = None
    for k in KS:
        run = RUNS / f'ripdim_k{k}'
        res = json.loads((run / 'results.json').read_text())
        curves = np.load(run / 'curves.npz')
        cfg = res['config']
        assert res['landscape_options']['rugged_dims'] == k, run
        mine = {c: cfg[c] for c in ('num_params', 'pop_size', 'num_seeds',
                                    'num_generations', 'task_interval',
                                    'threshold', 'record_stride', 'levels')}
        assert config in (None, mine), (run, mine)
        config = mine
        threshold = cfg['threshold'] * res['peak']
        gens = np.asarray(res['record_gens']) + 1
        for arm, method in ARMS.items():
            for sigma in SIGMAS:
                for level in cfg['levels']:
                    key = f'{method}|{sigma:g}|{level:g}'
                    hit = curves[f'cen_g|{key}'] >= threshold
                    found = hit.any(1)
                    first = gens[hit.argmax(1)]
                    meta['cells'][f'{arm}|{k}|{key}'] = dict(
                        n=int(len(hit)), found=int(found.sum()),
                        held=int(hit[:, -1].sum()),
                        held_seeds=[int(v) for v in hit[:, -1]],
                        first_median=(float(np.median(first[found]))
                                      if found.any() else None))
    meta['config'] = config
    meta['source'] = str((RUNS / 'ripdim_k<k>').relative_to(REPO))

    # The cost of one generation of children, from the landscape alone.
    task = jnp.asarray(0)
    for level in config['levels']:
        for sigma in SIGMAS:
            for k in KS:
                land = tl.rugged(rugged_dims=k)
                parent = jnp.zeros(config['num_params']).at[:2].set(
                    jnp.asarray(tl.SPECIALISTS[0]))
                noise = sigma * random.normal(
                    random.key(k), (DRAWS, CHILDREN, config['num_params']))
                kids = np.asarray(land.task_score(parent + noise, task, level))
                base = float(land.task_score(parent, task, level))
                lost = base - kids
                a = level * k / 4 * (1 - np.exp(-(2 * np.pi * sigma / 0.6) ** 2 / 2))
                meta['cost'][f'{k}|{sigma:g}|{level:g}'] = dict(
                    parent=base, mean=float(lost.mean()),
                    best_median=float(np.median(lost.min(1))),
                    best_lo=float(np.quantile(lost.min(1), 0.1)),
                    best_hi=float(np.quantile(lost.min(1), 0.9)),
                    analytic_mean=float(a))
    DATA.parent.mkdir(parents=True, exist_ok=True)
    DATA.with_suffix('.json').write_text(json.dumps(meta, indent=1) + '\n')
    print(f'wrote {DATA}.json')


def style(arm):
    return lp.METHOD_STYLE[arm]['color'], lp.METHOD_STYLE[arm]['label']


def log2_axis(ax):
    ax.set_xscale('log', base=2)
    ax.set_xticks(KS)
    ax.set_xticklabels([str(k) for k in KS])
    ax.minorticks_off()
    ax.set_xlim(KS[0] / 1.3, KS[-1] * 1.3)
    ax.set_xlabel('Rugged coordinates $k$ (of $d$ = 128)', labelpad=1)


def draw_held(ax, meta, level):
    from source.metrics.continual_metrics import bootstrap_ci
    for i, arm in enumerate(ARMS):
        colour, _ = style(arm)
        for sigma in SIGMAS:
            seeds = np.array([meta['cells'][f'{arm}|{k}|{ARMS[arm]}|{sigma:g}|{level:g}']
                              ['held_seeds'] for k in KS], float).T
            mean, lo, hi = bootstrap_ci(seeds)
            x = np.array(KS) * 2 ** ((i - 0.5) * 0.12)
            ax.errorbar(x, mean, yerr=[mean - lo, hi - mean], color=colour,
                        ls=SIGMA_DASH[sigma], lw=1.0, marker='o', ms=2.6,
                        mec='0.15', mew=0.35, elinewidth=0.5, capsize=0,
                        zorder=3)
    log2_axis(ax)
    ax.set_ylim(-0.05, 1.08)
    ax.set_yticks([0, 0.5, 1])
    ax.set_ylabel('Seeds generalist at end', labelpad=1)
    ax.set_title(f'Generalist centroid, $a$ = {level:g}', pad=3)


def draw_find(ax, meta):
    total = meta['config']['num_generations']
    for arm in ARMS:
        colour, _ = style(arm)
        for sigma in SIGMAS:
            cells = [meta['cells'][f'{arm}|{k}|{ARMS[arm]}|{sigma:g}|{FIND_LEVEL:g}']
                     for k in KS]
            # A median over a quarter of the seeds or fewer says little.
            y = [c['first_median'] if c['found'] > c['n'] / 4 else np.nan
                 for c in cells]
            ax.plot(KS, y, color=colour, ls=SIGMA_DASH[sigma], lw=1.0,
                    marker='o', ms=2.6, mec='0.15', mew=0.35, zorder=3)
    log2_axis(ax)
    ax.set_yscale('log')
    ax.set_ylim(60, total)
    ax.set_yticks([100, 300, 1000])
    ax.set_yticklabels(['100', '300', '1000'])
    ax.minorticks_off()
    ax.set_ylabel('Generations to find', labelpad=1)
    ax.set_title(f'Search time ($a$ = {FIND_LEVEL:g})', pad=3)


def draw_cost(ax, meta):
    peak = 0.70
    ax.axhline(peak, color='0.5', lw=0.6, ls=':', zorder=1)
    ax.text(KS[-1] * 1.2, peak / 1.2, 'generalist score', ha='right', va='top',
            fontsize=plt.rcParams['font.size'] - 1, color='0.4')
    for level in PANEL_LEVELS:
        for sigma in SIGMAS:
            c = [meta['cost'][f'{k}|{sigma:g}|{level:g}'] for k in KS]
            y = np.array([v['best_median'] for v in c])
            lo = np.array([v['best_lo'] for v in c])
            hi = np.array([v['best_hi'] for v in c])
            ax.fill_between(KS, np.maximum(lo, 1e-3), np.maximum(hi, 1e-3),
                            color=AMP_GREY[level], alpha=0.18, lw=0, zorder=2)
            ax.plot(KS, np.maximum(y, 1e-3), color=AMP_GREY[level],
                    ls=SIGMA_DASH[sigma], lw=1.0, marker='o', ms=2.2, zorder=3)
    log2_axis(ax)
    ax.set_yscale('log')
    ax.set_ylim(3e-3, 80)
    ax.minorticks_off()
    ax.set_ylabel('Loss of the best child', labelpad=1)
    ax.set_title('Cost of one generation of mutations', pad=3)


def draw(meta, args):
    fig, axes = plt.subplots(2, 2, figsize=(args.width, args.height),
                             gridspec_kw=dict(left=0.09, right=0.98, top=0.86,
                                              bottom=0.11, wspace=0.28,
                                              hspace=0.62))
    axes = axes.ravel()
    for ax, level in zip(axes, PANEL_LEVELS):
        draw_held(ax, meta, level)
    draw_find(axes[2], meta)
    draw_cost(axes[3], meta)
    for letter, ax in zip('abcd', axes):
        ax.tick_params(pad=1.5)
        ax.spines[['top', 'right']].set_visible(False)
        ax.text(-0.14, 1.06, f'({letter})', transform=ax.transAxes,
                fontweight='bold', va='bottom')
    handles = [Line2D([], [], color=style(arm)[0], lw=1.5, label=style(arm)[1])
               for arm in ARMS]
    handles += [Line2D([], [], color='0.3', lw=1.0, ls=SIGMA_DASH[s],
                       label=f'$\\sigma$ = {s:g}') for s in SIGMAS]
    handles += [Line2D([], [], color=AMP_GREY[a], lw=1.5, label=f'$a$ = {a:g} (d)')
                for a in PANEL_LEVELS]
    fig.legend(handles=handles, loc='upper center', ncol=len(handles),
               frameon=False, bbox_to_anchor=(0.5, 1.0), handlelength=2.0,
               columnspacing=1.0, handletextpad=0.4)
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ('pdf', 'png'):
        fig.savefig(OUT / f'{STEM}.{ext}', dpi=300)
    plt.close(fig)
    print(f'wrote {OUT / STEM}.pdf, .png  ({args.width:.2f} x {args.height:.2f} in)')


def fmt(v, spec='.0f'):
    return 'n/a' if v is None else format(v, spec)


def tables(meta):
    cfg = meta['config']
    arms = list(ARMS)
    head = ['a', 'k'] + [f'{style(a)[1]} σ {s:g}' for a in arms for s in SIGMAS] + \
           [f'best-child loss σ {s:g}' for s in SIGMAS]
    body = []
    for level in cfg['levels']:
        for k in KS:
            cells = [f'{level:g}', str(k)]
            for arm in arms:
                for sigma in SIGMAS:
                    c = meta['cells'][f'{arm}|{k}|{ARMS[arm]}|{sigma:g}|{level:g}']
                    cells.append(f"{c['found']}/{c['held']}/{c['n']} "
                                 f"({fmt(c['first_median'])})")
            cells += [f"{meta['cost'][f'{k}|{s:g}|{level:g}']['best_median']:.2f}"
                      for s in SIGMAS]
            body.append(cells)
    md = [f'# {STEM}', '',
          f"Extracted {meta['extracted']} from `{meta['source']}` "
          f"(`scripts/train/queue_toy.sh`, stage ripdim). Built by "
          f"`scripts/analysis/plot_toy_ripple_dims.py`.", '',
          f"The `rugged` toy with d = {cfg['num_params']} and the ripple along the "
          f"first k coordinates. Two sub-tasks alternate every {cfg['task_interval']} "
          f"generations for {cfg['num_generations']}; pop {cfg['pop_size']}, "
          f"{cfg['num_seeds']} seeds. ES = NES, GA with archive re-scoring.",
          'Every outcome is for the CENTROID (the ES mean; the mean of the GA elite '
          'archive), whose ripple-free generalist score is recorded every '
          f"{cfg['record_stride']} generations; a generalist scores at least 98% of the "
          '0.70 peak.', '',
          '- Arm cells: found / held / seeds (median generations to the first '
          'recorded crossing, over the seeds that found one).',
          f'- best-child loss: median over {DRAWS} draws of the score lost by the best '
          f'of {CHILDREN} isotropic children of sub-task A\'s first specialist peak '
          '(its score minus theirs), from the landscape, not the runs.', '',
          '| ' + ' | '.join(head) + ' |', '|' + '---|' * len(head)]
    md += ['| ' + ' | '.join(r) + ' |' for r in body]
    (OUT / f'{STEM}.md').write_text('\n'.join(md) + '\n')

    tex = [r'\begin{tabular}{rr|' + 'cc' * len(arms) + '|cc}', r'\toprule',
           ' & & ' + ' & '.join(r'\multicolumn{2}{c}{' + style(a)[1] + '}'
                                for a in arms)
           + r' & \multicolumn{2}{c}{Best-child loss} \\',
           '$a$ & $k$ & ' + ' & '.join([f'$\\sigma$={s:g}' for s in SIGMAS]
                                       * (len(arms) + 1)) + r' \\']
    last = None
    for r in body:
        if r[0] != last:
            tex.append(r'\midrule')
        held = [c.split(' ')[0].split('/') for c in r[2:2 + 2 * len(arms)]]
        cells = ['' if r[0] == last else r[0], r[1]] + \
                [f'{h}/{n}' for _f, h, n in held] + r[2 + 2 * len(arms):]
        last = r[0]
        tex.append(' & '.join(cells) + r' \\')
    tex += [r'\bottomrule', r'\end{tabular}']
    (OUT / f'{STEM}.tex').write_text('\n'.join(tex) + '\n')
    print(f'wrote {OUT / STEM}.md, .tex')


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--extract', action='store_true',
                    help='re-read the sweeps into the saved data first')
    ap.add_argument('--width', type=float, default=TEXT_WIDTH_IN)
    ap.add_argument('--height', type=float, default=3.35)
    ap.add_argument('--font-size', type=float, default=6.5)
    args = ap.parse_args()
    if args.extract:
        extract()
    meta = json.loads(DATA.with_suffix('.json').read_text())
    plt.rcParams.update({
        'font.size': args.font_size, 'axes.titlesize': args.font_size + 0.5,
        'axes.labelsize': args.font_size, 'legend.fontsize': args.font_size,
        'xtick.labelsize': args.font_size - 0.5, 'ytick.labelsize': args.font_size - 0.5,
        'axes.linewidth': 0.5, 'xtick.major.width': 0.5, 'ytick.major.width': 0.5,
        'xtick.major.size': 2, 'ytick.major.size': 2,
    })
    draw(meta, args)
    tables(meta)
    return 0


if __name__ == '__main__':
    sys.exit(main())
