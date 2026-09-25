"""The manifold grid read across runs: population size against stiff dimensions.

    .venv/bin/python scripts/make_toy_manifold_figure.py projects/iclr_2027/runs_toy

Reads every `manifold_c<c>_n<pop>/` that `scripts/train/queue_toy.sh` wrote
(one sweep per stiff-dimension count c and population size) and writes, into
the root:

    manifold_summary.{png,pdf}   one row per arm:
        found      share of seeds whose incumbent became a generalist, against
                   population size, one line per c (each run's best sigma)
        E - C      incumbent minus centroid generalist score at the end,
                   against population size, one line per c
        q          share of the incumbent's gaussian children that stay as
                   good, against sigma sqrt(c) / tube width, every cell of the
                   grid, coloured by population size -- if q is set by the
                   mutation width against the tube, the colours overlap
        wide       population directions wider than wide_factor sigma^2,
                   against the true number of directions that change nothing,
                   every cell, coloured by population size
    manifold_gap.{png,pdf}       GA only: E - C as a population x sigma grid,
                                 one panel per c -- whether the centroid
                                 tracking the elite follows population size or
                                 sigma
    manifold_summary.md          the numbers behind both

The three questions this grid is for (2026-09-13): does a large enough
population find the generalist through the ripple; does its centroid then
track its elite; and does the spread of the population count the directions
that change nothing.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import pathlib
import re
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from make_toy_figures import STYLE, WIDTH, house_style, save, tidy   # noqa: E402

import matplotlib                                                    # noqa: E402
matplotlib.use('Agg')
import matplotlib.pyplot as plt                                      # noqa: E402

METHODS = ['ga', 'ga_isoline', 'dns_gaussian']


def load(root):
    """`{(c, pop): record}` for every finished manifold run under `root`."""
    runs = {}
    for path in glob.glob(os.path.join(root, 'manifold_c*_n*', 'results.json')):
        m = re.search(r'manifold_c(\d+)_n(\d+)', path)
        runs[(int(m.group(1)), int(m.group(2)))] = json.load(open(path))
    return runs


def _mean(values):
    v = np.asarray([np.nan if x is None else x for x in values], dtype=float)
    return float(np.nanmean(v)) if np.isfinite(v).any() else np.nan


def cells(runs, level):
    """One row per (c, pop, method, sigma) at `level`."""
    rows = []
    for (c, pop), rec in runs.items():
        cfg = rec['config']
        width = rec['landscape_options']['tube_width']
        free = cfg['num_params'] - 2 - c
        for cell in rec['cells']:
            if abs(cell['level'] - level) > 1e-9:
                continue
            elite = _mean(cell['final_generalist'])
            rows.append(dict(
                c=c, pop=pop, method=cell['method'], sigma=cell['sigma'],
                best=rec['best_sigma'][cell['method']] == cell['sigma'],
                found=cell['found'], held=cell['held'], elite=elite,
                centroid=cell.get('end_cen_g'), member=cell.get('end_mem_g'),
                gap=(elite - cell['end_cen_g']
                     if cell.get('end_cen_g') is not None else np.nan),
                q=cell.get('end_q'), wide=cell.get('end_wide'), free=free,
                archive=pop // 2,
                reach=cell['sigma'] * np.sqrt(max(c, 1)) / width))
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('root')
    p.add_argument('--level', type=float, default=1.6,
                   help='ruggedness amplitude to read (default: the hardest)')
    args = p.parse_args()
    house_style()
    runs = load(args.root)
    if not runs:
        raise SystemExit(f'no manifold_c*_n* runs under {args.root}')
    rows = cells(runs, args.level)
    cs = sorted({r['c'] for r in rows})
    pops = sorted({r['pop'] for r in rows})
    c_colors = plt.cm.viridis(np.linspace(0.1, 0.85, len(cs)))
    p_colors = dict(zip(pops, plt.cm.plasma(np.linspace(0.1, 0.8, len(pops)))))

    fig, axes = plt.subplots(len(METHODS), 4, squeeze=False,
                             figsize=(WIDTH, 1.6 * len(METHODS)),
                             constrained_layout=True)
    for row, m in enumerate(METHODS):
        mine = [r for r in rows if r['method'] == m]
        for c, col in zip(cs, c_colors):
            best = sorted([r for r in mine if r['c'] == c and r['best']],
                          key=lambda r: r['pop'])
            if not best:
                continue
            x = [r['pop'] for r in best]
            axes[row, 0].plot(x, [r['found'] for r in best], 'o-', color=col,
                              ms=2.5, lw=0.9, label=f'c = {c}')
            axes[row, 1].plot(x, [r['gap'] for r in best], 'o-', color=col,
                              ms=2.5, lw=0.9)
        for r in mine:
            if r['q'] is not None:
                axes[row, 2].scatter(r['reach'], r['q'], s=6,
                                     color=p_colors[r['pop']], lw=0)
            if r['wide'] is not None:
                axes[row, 3].scatter(r['free'], r['wide'], s=6,
                                     color=p_colors[r['pop']], lw=0)
        lim = max(r['free'] for r in rows)
        axes[row, 3].plot([0, lim], [0, lim], color='0.5', lw=0.6, ls='--')
        for ax in axes[row, :2]:
            ax.set_xscale('log', base=2)
        axes[row, 2].set_xscale('log')
        axes[row, 1].axhline(0, color='0.3', lw=0.4)
        axes[row, 0].set_ylabel(f'{STYLE.get(m, {}).get("label", m)}\nfound')
        for ax in axes[row]:
            tidy(ax)
    axes[0, 0].legend(frameon=False, fontsize=5)
    for pop in pops:
        axes[0, 3].scatter([], [], color=p_colors[pop], s=6, label=f'pop {pop}')
    axes[0, 3].legend(frameon=False, fontsize=5)
    titles = ['generalist found', 'incumbent $-$ centroid score',
              'q: children that stay as good', 'wide directions']
    xlabels = ['population size', 'population size',
               '$\\sigma\\sqrt{c}$ / tube width', 'directions that change nothing']
    for col in range(4):
        axes[0, col].set_title(titles[col])
        axes[-1, col].set_xlabel(xlabels[col])
    fig.suptitle(f'manifold grid, ruggedness amplitude {args.level:g}, '
                 'each run\'s best $\\sigma$ in the first two columns, every '
                 'cell in the last two', fontsize=6)
    save(fig, os.path.join(args.root, 'manifold_summary'))

    # GA: E - C as population x sigma, one panel per c.
    sigmas = sorted({r['sigma'] for r in rows})
    fig, axes = plt.subplots(1, len(cs), squeeze=False,
                             figsize=(WIDTH, 1.9), constrained_layout=True)
    for ax, c in zip(axes[0], cs):
        z = np.full((len(sigmas), len(pops)), np.nan)
        for r in rows:
            if r['method'] == 'ga' and r['c'] == c:
                z[sigmas.index(r['sigma']), pops.index(r['pop'])] = r['gap']
        im = ax.imshow(z, cmap='magma_r', vmin=0, vmax=0.7, aspect='auto',
                       origin='lower')
        for i in range(len(sigmas)):
            for j in range(len(pops)):
                if np.isfinite(z[i, j]):
                    ax.text(j, i, f'{z[i, j]:.2f}', ha='center', va='center',
                            fontsize=4.5, color='white' if z[i, j] > 0.35
                            else 'black')
        ax.set_xticks(range(len(pops)), [str(x) for x in pops])
        ax.set_yticks(range(len(sigmas)), [f'{s:g}' for s in sigmas])
        ax.set_title(f'c = {c}')
        ax.set_xlabel('population size')
    axes[0, 0].set_ylabel('$\\sigma$')
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.8,
                 label='GA: incumbent $-$ centroid')
    save(fig, os.path.join(args.root, 'manifold_gap'))

    lines = [f'# manifold grid, ruggedness amplitude {args.level:g}', '',
             '| c | pop | method | σ | best σ | found | held | elite | centroid '
             '| member mean | E − C | q | wide | free dims |',
             '|---|---|---|---|---|---|---|---|---|---|---|---|---|---|']
    for r in sorted(rows, key=lambda r: (r['method'], r['c'], r['pop'],
                                         r['sigma'])):
        def f(v, d=2):
            return '—' if v is None or not np.isfinite(v) else f'{v:.{d}f}'
        lines.append(f"| {r['c']} | {r['pop']} | {r['method']} | {r['sigma']:g} "
                     f"| {'✓' if r['best'] else ''} | {r['found']:.2f} "
                     f"| {r['held']:.2f} | {f(r['elite'])} | {f(r['centroid'])} "
                     f"| {f(r['member'])} | {f(r['gap'])} | {f(r['q'])} "
                     f"| {f(r['wide'], 0)} | {r['free']} |")
    out = os.path.join(args.root, 'manifold_summary.md')
    pathlib.Path(out).write_text('\n'.join(lines) + '\n')
    print(f'wrote {out}')


if __name__ == '__main__':
    main()
