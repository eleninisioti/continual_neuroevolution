"""The toy-landscape figures and table, from what `source/studies/toy/sweep.py` wrote.

    .venv/bin/python scripts/make_toy_figures.py projects/iclr_2027/runs_toy/rugged

Reads `results.json` and `curves.npz` (and any `project_*.npz`) in the run
directory, searches nothing, and writes beside them:

    curves_generalist.{png,pdf}   incumbent's generalist score, one panel per
                                  report level, mean and 95% bootstrap CI
    curves_subtask.{png,pdf}      its score on the sub-task being run
                                  (switching toys only)
    curves_seeds.png              every seed, one panel per (level, arm)
    grid.png                      sigma x level: found / held / cover_held
    landscape_<level>.png         the surface over the first two coordinates,
                                  every seed's incumbent path, seed 0's
                                  centroid path dashed
    centroid.{png,pdf}            population arms: centroid minus mean member
                                  score, incumbent minus centroid score,
                                  |centroid - incumbent| / (sigma sqrt d), and
                                  the minority-basin share
    table.md                      each arm at its best sigma, per level
    projection_<cell>.png         for each `--project` snapshot file: a PCA of
                                  the final population, the plane through
                                  incumbent, centroid and the farthest member,
                                  and the score along those two lines

Palette, labels and font sizes are `make_lineplot.py`'s, so an arm is the same
colour here as in the paper's lineplots; the toy-only controls have their own.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from make_lineplot import METHOD_ORDER, METHOD_STYLE          # noqa: E402
from source.envs import toy_landscapes                        # noqa: E402
from source.metrics.continual_metrics import bootstrap_ci     # noqa: E402

import matplotlib                                             # noqa: E402
matplotlib.use('Agg')
import matplotlib.pyplot as plt                               # noqa: E402

TOY_STYLE = {
    'dns_corrected': {'label': 'GA + Novelty (Iso+LineDD, line $\\sigma$ 0.05)',
                      'color': '#1F5C8F'},
    'ga_stale': {'label': 'GA (stale archive)', 'color': '#2E7A2B'},
    'ga_track': {'label': 'GA ($\\sigma$ tracks centroid)', 'color': '#B5179E'},
    'ga_success': {'label': 'GA ($\\sigma$ from offspring success)',
                   'color': '#F72585'},
    'ga_subspace': {'label': 'GA (iso + spread widths)', 'color': '#3A0CA3'},
    'ga_merge_noise': {'label': 'GA (rank noise when centroid lags)',
                       'color': '#E76F51'},
    'ga_merge_pc': {'label': 'GA (PC recombination when centroid lags)',
                    'color': '#2A9D8F'},
    'ga_survive': {'label': 'GA (survival-adapted $\\sigma$)',
                   'color': '#7209B7'},
    'nes_mu': {'label': 'NES, 8 centroids', 'color': '#9467BD'},
    'nes_mu_select': {'label': 'NES, 8 centroids + selection',
                      'color': '#6A3D9A'},
    'nes_adam': {'label': 'NES + Adam', 'color': '#F5D97A'},
    'es_sgd': {'label': 'ES + SGD', 'color': '#8C6A0A'},
    'es_nomom': {'label': 'ES + Adam, no momentum', 'color': '#1F6F6F'},
    'nes_adam_nomom': {'label': 'NES + Adam, no momentum', 'color': '#79C2C2'},
}
STYLE = {**TOY_STYLE, **METHOD_STYLE}
ORDER = METHOD_ORDER + list(TOY_STYLE)
# Distribution arms: the centroid IS the incumbent, so the centroid figure and
# columns say nothing about them.
DISTRIBUTION = {'es', 'nes', 'nes_adam', 'es_sgd', 'es_nomom', 'nes_adam_nomom'}
WIDTH = 6.9   # inches: 17.5 cm, the lineplot's full width


def label(m):
    return STYLE.get(m, {}).get('label', m)


def color(m):
    return STYLE.get(m, {}).get('color')


def house_style(fs=7.0):
    plt.rcParams.update({
        'font.size': fs, 'axes.labelsize': fs, 'axes.titlesize': fs + 1,
        'xtick.labelsize': fs - 0.5, 'ytick.labelsize': fs - 0.5,
        'legend.fontsize': fs, 'axes.linewidth': 0.6,
        'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
        'xtick.major.size': 2.5, 'ytick.major.size': 2.5,
    })


def tidy(ax):
    ax.spines[['top', 'right']].set_visible(False)
    ax.margins(x=0)


def save(fig, stem, pdf=True):
    fig.savefig(f'{stem}.png', dpi=200, bbox_inches='tight')
    if pdf:
        fig.savefig(f'{stem}.pdf', bbox_inches='tight')
    plt.close(fig)
    print(f'wrote {stem}.png' + (' + .pdf' if pdf else ''))


class Run:
    """One sweep directory."""

    def __init__(self, root):
        self.root = pathlib.Path(root)
        self.record = json.loads((self.root / 'results.json').read_text())
        self.npz = np.load(self.root / 'curves.npz')
        self.cfg = self.record['config']
        self.land = toy_landscapes.get(self.record['landscape'],
                                       **self.record['landscape_options'])
        self.best = {m: float(s) for m, s in self.record['best_sigma'].items()}
        self.threshold = self.cfg['threshold'] * self.land.peak
        self.cells = {(c['method'], float(c['sigma']), float(c['level'])): c
                      for c in self.record['cells']}
        self.gens = np.arange(self.cfg['num_generations'])
        self.record_gens = np.asarray(self.record['record_gens'])

    def methods(self, requested=None):
        have = set(self.best)
        order = requested or ORDER + sorted(have - set(ORDER))
        return [m for m in order if m in have]

    def series(self, kind, method, level, sigma=None):
        s = self.best[method] if sigma is None else sigma
        key = f'{kind}|{method}|{s:g}|{level:g}'
        return self.npz[key] if key in self.npz.files else None

    def boundaries(self, ax):
        if self.land.num_tasks < 2:
            return
        for b in range(self.cfg['task_interval'], self.cfg['num_generations'],
                       self.cfg['task_interval']):
            ax.axvline(b, color='0.6', lw=0.45, ls='--', zorder=0)

    def level_title(self, level):
        return f'{self.land.level_label} = {level:g}'


def band(y, floor=None):
    """Mean and 95% bootstrap CI over seeds, on the seeds that stayed finite.

    A diverging arm (dns at line_sigma 0.5) produces inf/NaN; clipping it to
    the panel's floor keeps it on the figure as a line along the bottom
    rather than a hole, which is what it is -- a search that left the surface.
    """
    y = np.asarray(y, dtype=np.float64)
    if floor is not None:
        y = np.where(np.isfinite(y), np.maximum(y, floor), floor)
    rows = np.isfinite(y).all(axis=1)
    if rows.sum() == 0:
        return None
    y = y[rows]
    if len(y) < 2:
        return y[0], y[0], y[0]
    return bootstrap_ci(y)


def fig_curves(run, methods, levels, kind, stem, ylabel):
    lo_lim, hi_lim = -0.05 * run.land.peak, run.land.peak * 1.08
    fig, axes = plt.subplots(1, len(levels), squeeze=False, sharey=True,
                             figsize=(WIDTH, 1.9), constrained_layout=True)
    handles = {}
    for ax, level in zip(axes[0], levels):
        run.boundaries(ax)
        for m in methods:
            y = run.series(kind, m, level)
            b = None if y is None else band(y, floor=lo_lim)
            if b is None:
                continue
            mid, lo, hi = b
            handles[m], = ax.plot(run.gens, mid, color=color(m), lw=1.0,
                                  zorder=3)
            ax.fill_between(run.gens, lo, hi, color=color(m), alpha=0.18,
                            lw=0, zorder=2)
        if kind == 'g':
            ax.axhline(run.threshold, color='black', ls=':', lw=0.8)
        ax.set_ylim(lo_lim, hi_lim)
        ax.set_title(run.level_title(level), fontweight='bold')
        ax.set_xlabel('Generations')
        tidy(ax)
    axes[0, 0].set_ylabel(ylabel)
    fig.legend(list(handles.values()),
               [f'{label(m)} ($\\sigma$ {run.best[m]:g})' for m in handles],
               loc='outside upper center', ncol=min(len(handles), 4),
               frameon=False)
    save(fig, stem)


def fig_seeds(run, methods, levels, stem):
    lo_lim, hi_lim = -0.05 * run.land.peak, run.land.peak * 1.08
    fig, axes = plt.subplots(len(levels), len(methods), squeeze=False,
                             sharex=True, sharey=True,
                             figsize=(1.35 * len(methods), 1.2 * len(levels)),
                             constrained_layout=True)
    for row, level in enumerate(levels):
        for col, m in enumerate(methods):
            ax = axes[row, col]
            run.boundaries(ax)
            y = run.series('g', m, level)
            if y is not None:
                y = np.where(np.isfinite(y), np.maximum(y, lo_lim), lo_lim)
                ax.plot(run.gens, y.T, color=color(m), lw=0.4, alpha=0.5)
            ax.axhline(run.threshold, color='black', ls=':', lw=0.6)
            ax.set_ylim(lo_lim, hi_lim)
            tidy(ax)
            if row == 0:
                ax.set_title(label(m), fontsize=6)
            if col == 0:
                ax.set_ylabel(f'{run.land.level_name} = {level:g}')
    save(fig, stem, pdf=False)


def fig_grid(run, methods, stem):
    keys = [('found', 'incumbent was a generalist'),
            ('held', 'still was at the end'),
            ('cover_held', 'some member still was')]
    levels, sigmas = run.cfg['levels'], run.cfg['sigmas']
    fig, axes = plt.subplots(len(methods), len(keys), squeeze=False,
                             figsize=(WIDTH, 0.75 * len(methods) + 0.4),
                             constrained_layout=True)
    for row, m in enumerate(methods):
        for col, (key, title) in enumerate(keys):
            ax = axes[row, col]
            z = np.array([[run.cells[(m, float(s), float(x))][key]
                           for x in levels] for s in sigmas])
            ax.imshow(z, vmin=0, vmax=1, cmap='RdYlGn', aspect='auto',
                      origin='lower')
            for i in range(len(sigmas)):
                for j in range(len(levels)):
                    ax.text(j, i, f'{z[i, j]:.2f}', ha='center', va='center',
                            fontsize=4.5)
            ax.set_xticks(range(len(levels)), [f'{x:g}' for x in levels])
            ax.set_yticks(range(len(sigmas)), [f'{s:g}' for s in sigmas])
            if col == 0:
                ax.set_ylabel(f'{label(m)}\n$\\sigma$', fontsize=5.5)
            if row == 0:
                ax.set_title(title)
            if row == len(methods) - 1:
                ax.set_xlabel(run.land.level_label)
            else:
                ax.set_xticklabels([])
    save(fig, stem, pdf=False)


def clip_to_window(window, path):
    """NaN outside the window, so a path that leaves the frame stops there
    rather than being joined across the panel."""
    (x0, x1), (y0, y1) = window
    out = np.array(path, dtype=float)
    inside = (np.isfinite(out).all(axis=1) & (out[:, 0] >= x0)
              & (out[:, 0] <= x1) & (out[:, 1] >= y0) & (out[:, 1] <= y1))
    out[~inside] = np.nan
    return out


def break_jumps(window, path, frac=0.08):
    """`clip_to_window`, and also broken wherever one generation moved more
    than `frac` of the window's width. A population arm's incumbent is its
    best member, and when a different member takes over the incumbent JUMPS;
    a line joining the two is a move no genome made."""
    (x0, x1), _ = window
    out = clip_to_window(window, path)
    step = np.linalg.norm(np.diff(np.asarray(path, dtype=float), axis=0),
                          axis=1)
    out[1:][~(step <= frac * (x1 - x0))] = np.nan
    return out


def fig_landscapes(run, methods, levels):
    land = run.land
    (x0, x1), (y0, y1) = land.window
    gx, gy = np.meshgrid(np.linspace(x0, x1, 360), np.linspace(y0, y1, 240))
    pts = toy_landscapes.plane(land, gx, gy)
    switching = land.num_tasks > 1
    shades = np.linspace(0.0, land.peak, 25)
    for level in levels:
        drawn = [m for m in methods if run.series('path', m, level) is not None]
        if not drawn:
            continue
        target = np.asarray(land.generalist_score(pts, level))
        fields = ([np.asarray(land.task_score(pts, 0, level)),
                   np.asarray(land.task_score(pts, 1, level)), target]
                  if switching else [target])
        titles = (['sub-task 0', 'sub-task 1', 'min(0, 1): what is graded']
                  if switching else
                  [f'score over $\\theta_1, \\theta_2$ '
                   f'(of d = {run.cfg["num_params"]})'])
        ncol = len(fields)
        fig, axes = plt.subplots(
            len(drawn), ncol, squeeze=False, sharex=True, sharey=True,
            figsize=(WIDTH if switching else 2.6,
                     (1.25 if switching else 2.2) * len(drawn)),
            constrained_layout=True)
        for row, m in enumerate(drawn):
            path = run.series('path', m, level)
            cpath = run.series('cpath', m, level)
            for col in range(ncol):
                ax = axes[row, col]
                im = ax.contourf(gx, gy, fields[col], levels=shades,
                                 cmap='viridis', extend='min')
                ax.contour(gx, gy, target, levels=[run.threshold],
                           colors='white', linewidths=0.8, linestyles='--')
                for seed in path:
                    vis = break_jumps(land.window, seed)
                    ax.plot(vis[:, 0], vis[:, 1], color=color(m), lw=0.4,
                            alpha=0.3)
                    end = clip_to_window(land.window, seed[-1:])[0]
                    if np.isfinite(end).all():
                        ax.plot(*end, 'o', color=color(m), ms=2,
                                mec='white', mew=0.3)
                vis = break_jumps(land.window, path[0])
                ax.plot(vis[:, 0], vis[:, 1], color='black', lw=0.7)
                if m not in DISTRIBUTION and cpath is not None:
                    cvis = break_jumps(land.window, cpath[0])
                    ax.plot(cvis[:, 0], cvis[:, 1], color='white', lw=0.45,
                            ls='--', alpha=0.8)
                    end = clip_to_window(land.window, cpath[0][-1:])[0]
                    if np.isfinite(end).all():
                        ax.plot(*end, 'D', color='white', ms=2.5,
                                mec='black', mew=0.4)
                for mx, my, mk in land.markers:
                    ax.plot(mx, my, mk, color='white' if mk != 'x' else '#222',
                            ms=4, mec='black', mew=0.5)
                ax.set_xlim(x0, x1)
                ax.set_ylim(y0, y1)
                if row == 0:
                    ax.set_title(titles[col])
                if col == 0:
                    ax.set_ylabel(f'{label(m)}\n$\\sigma$ {run.best[m]:g}',
                                  fontsize=5.5)
                if row == len(drawn) - 1:
                    ax.set_xlabel('$\\theta_1$')
        fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.6, label='score')
        fig.suptitle(f'{run.level_title(level)}: coloured, every seed\'s '
                     'incumbent; black, seed 0; white dashed, seed 0\'s '
                     'centroid', fontsize=6)
        save(fig, str(run.root / f'landscape_{land.level_name}{level:g}'),
             pdf=False)


def fig_centroid(run, methods, levels, stem):
    methods = [m for m in methods if m not in DISTRIBUTION]
    if not methods:
        return
    d = run.cfg['num_params']
    cols = [('centroid $-$ mean member score', 'jensen'),
            ('incumbent $-$ centroid score', 'gap'),
            ('$|c - e| \\,/\\, \\sigma\\sqrt{d}$', 'dist'),
            ('share outside the most common basin', 'minority')]
    idx = run.record_gens
    fig, axes = plt.subplots(len(levels), len(cols), squeeze=False,
                             sharex=True, figsize=(WIDTH, 1.45 * len(levels)),
                             constrained_layout=True)
    handles = {}
    for row, level in enumerate(levels):
        for col, (title, what) in enumerate(cols):
            ax = axes[row, col]
            run.boundaries(ax)
            for m in methods:
                cen = run.series('cen_g', m, level)
                if cen is None:
                    continue
                if what == 'jensen':
                    y = cen - run.series('mem_g', m, level)
                elif what == 'gap':
                    y = run.series('g', m, level)[:, idx] - cen
                elif what == 'dist':
                    y = run.series('dist', m, level) / (run.best[m] * np.sqrt(d))
                else:
                    y = run.series('minority', m, level)
                b = band(y)
                if b is None:
                    continue
                mid, lo, hi = b
                handles[m], = ax.plot(idx, mid, color=color(m), lw=0.9)
                ax.fill_between(idx, lo, hi, color=color(m), alpha=0.18, lw=0)
            if what in ('jensen', 'gap'):
                ax.axhline(0, color='0.3', lw=0.5)
            if what == 'dist':
                ax.set_yscale('symlog', linthresh=0.1)
            tidy(ax)
            if row == 0:
                ax.set_title(title)
            if col == 0:
                ax.set_ylabel(f'{run.land.level_name} = {level:g}')
            if row == len(levels) - 1:
                ax.set_xlabel('Generations')
    fig.legend(list(handles.values()), [label(m) for m in handles],
               loc='outside upper center', ncol=min(len(handles), 4),
               frameon=False)
    save(fig, stem)


def fig_population(run, methods, levels, stem):
    """How well a solution reproduces, and how many directions the population
    spreads along -- against the number of directions that change nothing."""
    methods = [m for m in methods if m not in DISTRIBUTION
               and any(run.series('q', m, x) is not None for x in levels)]
    if not methods:
        return
    free = run.cfg['num_params'] - run.land.relevant
    idx = run.record_gens
    cols = [('q: children that stay as good', 'q'),
            ('participation ratio of the spread', 'pr'),
            (f'directions wider than {run.cfg.get("wide_factor", 4):g}'
             ' $\\sigma^2$', 'wide')]
    # The mutation width over time, where a run recorded it (it only moves
    # under ga_track, so it is the figure that shows that rule working).
    if any(run.series('sigma', m, x) is not None for m in methods
           for x in levels):
        cols.append(('mutation width $\\sigma$', 'sigma'))
    if any(run.series('sigma_sub', m, x) is not None
           and np.isfinite(run.series('sigma_sub', m, x)).any()
           for m in methods for x in levels):
        cols.append(('width along the spread $\\sigma_{sub}$', 'sigma_sub'))
    if any(run.series('merge', m, x) is not None
           and np.isfinite(run.series('merge', m, x)).any()
           for m in methods for x in levels):
        cols.append(('merge level (rank noise / recombination share)', 'merge'))
    fig, axes = plt.subplots(len(levels), len(cols), squeeze=False,
                             sharex=True, figsize=(WIDTH, 1.45 * len(levels)),
                             constrained_layout=True)
    handles = {}
    for row, level in enumerate(levels):
        for col, (title, kind) in enumerate(cols):
            ax = axes[row, col]
            run.boundaries(ax)
            for m in methods:
                y = run.series(kind, m, level)
                b = None if y is None else band(y)
                if b is None:
                    continue
                mid, lo, hi = b
                handles[m], = ax.plot(idx, mid, color=color(m), lw=0.9)
                ax.fill_between(idx, lo, hi, color=color(m), alpha=0.18, lw=0)
            if kind == 'wide':
                ax.axhline(free, color='black', ls=':', lw=0.8)
                ax.text(idx[-1], free, 'directions that change nothing',
                        va='bottom', ha='right', fontsize=5)
            if kind == 'q':
                ax.set_ylim(-0.03, 1.03)
            tidy(ax)
            if row == 0:
                ax.set_title(title)
            if col == 0:
                ax.set_ylabel(f'{run.land.level_name} = {level:g}')
            if row == len(levels) - 1:
                ax.set_xlabel('Generations')
    fig.legend(list(handles.values()), [label(m) for m in handles],
               loc='outside upper center', ncol=min(len(handles), 4),
               frameon=False)
    save(fig, stem)


def write_table(run, methods, levels, path):
    def fmt(v, digits=2):
        if v is None or not np.isfinite(v):
            return '—'
        # dns and ga_isoline at line_sigma 0.5 run their weights away; a
        # 30-digit number in a table column says that less clearly than this.
        return 'diverged' if abs(v) > 1e4 else f'{v:.{digits}f}'

    cfg = run.cfg
    lines = [f'# {run.land.name}: {cfg["num_seeds"]} seeds, '
             f'{cfg["num_generations"]} generations, pop {cfg["pop_size"]}, '
             f'd = {cfg["num_params"]}', '',
             'Each arm at its best sigma. Found / Held: fraction of seeds whose '
             'incumbent was / still is a generalist. C − M: centroid score '
             'minus mean member score at the end (negative means the '
             'population spans more than one basin). E − C: incumbent minus '
             'centroid score. |c − e| / σ√d: centroid-incumbent distance in '
             'units of one mutation. Minority: share of members outside the '
             'most common basin (— where the landscape has a single basin). '
             '"diverged": the population\'s weights ran away (|value| > 1e4).',
             '']
    # Runs made before q and the spread spectrum were recorded have neither.
    extra = any(c.get('end_q') is not None for c in run.cells.values())
    if extra:
        lines[-2] += (' q: share of the incumbent\'s gaussian children that '
                      'stay as good. Wide: population directions wider than '
                      f'{cfg.get("wide_factor", 4):g} σ², against '
                      f'{cfg["num_params"] - run.land.relevant} directions '
                      'that change nothing.')
    for level in levels:
        lines += [f'## {run.level_title(level)}', '',
                  '| Method | σ | Found | Held | Retention | Gens to find | '
                  'C − M | E − C | \\|c − e\\| / σ√d | Minority |'
                  + (' q | Wide |' if extra else ''),
                  '|---|---|---|---|---|---|---|---|---|---|'
                  + ('---|---|' if extra else '')]
        for m in methods:
            c = run.cells.get((m, run.best[m], float(level)))
            if c is None:
                continue
            first = [f for f in c['first'] if f < cfg['num_generations']]
            elite = np.nanmean([np.nan if v is None else v
                                for v in c['final_generalist']])
            cen, mem = c['end_cen_g'], c['end_mem_g']
            pop = m not in DISTRIBUTION and cen is not None
            lines.append(
                f'| {label(m)} | {run.best[m]:g} | {c["found"]:.2f} | '
                f'{c["held"]:.2f} | {fmt(c["retention"])} | '
                f'{fmt(np.mean(first) if first else None, 0)} | '
                f'{fmt(cen - mem, 3) if pop and mem is not None else "—"} | '
                f'{fmt(elite - cen, 3) if pop else "—"} | '
                f'{fmt(c["end_dist"] / (run.best[m] * np.sqrt(cfg["num_params"]))) if pop and c["end_dist"] is not None else "—"} | '
                f'{fmt(c["end_minority"]) if pop else "—"} |'
                + (f' {fmt(c.get("end_q")) if pop else "—"} | '
                   f'{fmt(c.get("end_wide"), 0) if pop else "—"} |'
                   if extra else ''))
        lines.append('')
    path.write_text('\n'.join(lines))
    print(f'wrote {path}')


def fig_projection(npz_path):
    """What a d-dimensional population looks like, three ways.

    PCA of the final population shows its shape; the plane through incumbent,
    centroid and the member farthest from the incumbent is the one slice that
    must contain the valley if the population spans two basins; the line
    profiles are that statement in one dimension.
    """
    import jax.numpy as jnp
    z = np.load(npz_path)
    meta = json.loads(str(z['meta']))
    land = toy_landscapes.get(meta['landscape'], **meta['landscape_options'])
    level = meta['level']
    gens = z['gens']
    seeds = z['members'].shape[0]
    fig, axes = plt.subplots(seeds, 3, squeeze=False,
                             figsize=(WIDTH, 2.1 * seeds),
                             constrained_layout=True)
    for s in range(seeds):
        members = z['members'][s, -1].astype(np.float64)
        elite = z['elite'][s, -1].astype(np.float64)
        cen = z['centroid'][s, -1].astype(np.float64)
        score = z['member_score'][s, -1]
        if not (np.isfinite(members).all() and np.isfinite(elite).all()):
            for ax in axes[s]:
                ax.text(0.5, 0.5, 'non-finite population', ha='center',
                        transform=ax.transAxes)
            continue

        # (a) PCA of the final population, with the centroid's and the
        # incumbent's trajectories projected onto the same axes.
        ax = axes[s, 0]
        _, _, vt = np.linalg.svd(members - cen, full_matrices=False)
        basis = vt[:2]
        proj = (members - cen) @ basis.T
        sc = ax.scatter(proj[:, 0], proj[:, 1], c=score, s=4, cmap='viridis',
                        vmin=0, vmax=land.peak, lw=0)
        ctraj = (z['centroid'][s] - cen) @ basis.T
        etraj = (z['elite'][s] - cen) @ basis.T
        ax.plot(ctraj[:, 0], ctraj[:, 1], color='black', ls='--', lw=0.7,
                label='centroid')
        ax.plot(etraj[:, 0], etraj[:, 1], color='#E8504F', lw=0.7,
                label='incumbent')
        ax.plot(0, 0, 'D', color='white', mec='black', ms=3.5)
        ax.plot(*((elite - cen) @ basis.T), '*', color='#E8504F', ms=6,
                mec='black', mew=0.4)
        ax.set_xlabel('PC 1')
        ax.set_ylabel(f'seed {s}\nPC 2')
        if s == 0:
            ax.set_title(f'final population (gen {gens[-1]}), PCA')
            ax.legend(frameon=False, fontsize=5)
        fig.colorbar(sc, ax=ax, shrink=0.8, label='member score')

        # (b) the plane through incumbent, centroid and the farthest member.
        ax = axes[s, 1]
        far = members[np.argmax(np.linalg.norm(members - elite, axis=1))]
        u1 = cen - elite
        if np.linalg.norm(u1) < 1e-9:
            u1 = far - elite
        u1 /= max(np.linalg.norm(u1), 1e-12)
        u2 = far - elite - ((far - elite) @ u1) * u1
        u2 /= max(np.linalg.norm(u2), 1e-12)
        coords = (members - elite) @ np.stack([u1, u2]).T
        marks = {'incumbent': np.zeros(2),
                 'centroid': np.array([(cen - elite) @ u1, (cen - elite) @ u2]),
                 'farthest': np.array([(far - elite) @ u1, (far - elite) @ u2])}
        allc = np.concatenate([coords, np.stack(list(marks.values()))])
        span = np.ptp(allc, axis=0).max() * 0.35 + 1e-3
        lo, hi = allc.min(0) - span, allc.max(0) + span
        a, b = np.meshgrid(np.linspace(lo[0], hi[0], 150),
                           np.linspace(lo[1], hi[1], 150))
        pts = elite + a[..., None] * u1 + b[..., None] * u2
        field = np.asarray(land.generalist_score(jnp.asarray(pts), level))
        im = ax.contourf(a, b, field, levels=np.linspace(0, land.peak, 21),
                         cmap='viridis', extend='min')
        ax.scatter(coords[:, 0], coords[:, 1], s=2, color='white', alpha=0.5,
                   lw=0)
        for name, xy, mk, col in (('incumbent', marks['incumbent'], '*', '#E8504F'),
                                  ('centroid', marks['centroid'], 'D', 'white'),
                                  ('farthest', marks['farthest'], 'o', '#8FB8E8')):
            ax.plot(*xy, mk, color=col, mec='black', mew=0.4, ms=5, label=name)
        ax.set_xlabel('toward centroid')
        ax.set_ylabel('toward farthest member')
        if s == 0:
            ax.set_title('plane: incumbent, centroid, farthest member')
            ax.legend(frameon=False, fontsize=5, loc='lower right')
        fig.colorbar(im, ax=ax, shrink=0.8, label='score on the plane')

        # (c) the score along the two lines out of the incumbent.
        ax = axes[s, 2]
        t = np.linspace(-0.25, 1.25, 301)
        for target, name, col in ((cen, 'incumbent → centroid', 'black'),
                                  (far, 'incumbent → farthest', '#3B8FD4')):
            line = elite + t[:, None] * (target - elite)
            ax.plot(t, np.asarray(land.generalist_score(jnp.asarray(line),
                                                        level)),
                    color=col, lw=0.9, label=name)
        for v in (0, 1):
            ax.axvline(v, color='0.6', lw=0.45, ls='--')
        ax.set_ylim(-0.03, land.peak * 1.05)
        ax.set_xlabel('position along the line (0 = incumbent, 1 = end)')
        ax.set_ylabel('score')
        tidy(ax)
        if s == 0:
            ax.set_title('score along the lines')
            ax.legend(frameon=False, fontsize=5)
    fig.suptitle(f'{label(meta["method"])}, $\\sigma$ {meta["sigma"]:g}, '
                 f'{land.level_label} = {level:g}, d = {meta["num_params"]}, '
                 f'pop {meta["pop_size"]}', fontsize=7)
    save(fig, str(pathlib.Path(npz_path).with_suffix('')).replace(
        'project_', 'projection_'), pdf=False)


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('run_dir')
    p.add_argument('--methods', nargs='+', default=None)
    p.add_argument('--levels', type=float, nargs='+', default=None,
                   help='default: the report levels the sweep recorded')
    p.add_argument('--font-size', type=float, default=7.0)
    args = p.parse_args()
    house_style(args.font_size)

    run = Run(args.run_dir)
    methods = run.methods(args.methods)
    levels = args.levels or run.cfg['report_levels']
    root = run.root
    fig_curves(run, methods, levels, 'g', str(root / 'curves_generalist'),
               'Generalist score' if run.land.num_tasks > 1 else 'Score')
    if run.land.num_tasks > 1:
        fig_curves(run, methods, levels, 'train', str(root / 'curves_subtask'),
                   'Score on current sub-task')
    fig_seeds(run, methods, levels, str(root / 'curves_seeds'))
    fig_grid(run, methods, str(root / 'grid'))
    fig_landscapes(run, methods, levels)
    fig_centroid(run, methods, levels, str(root / 'centroid'))
    fig_population(run, methods, levels, str(root / 'population'))
    write_table(run, methods, run.cfg['levels'], root / 'table.md')
    for npz_path in sorted(root.glob('project_*.npz')):
        fig_projection(npz_path)


if __name__ == '__main__':
    main()
