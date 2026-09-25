"""Appendix figure: does the width of the basin ES settles in depend on its
sigma, and does that depend on the landscape?

    # re-read the sweeps and measure the landscape around the centroids
    JAX_PLATFORMS=cpu .venv/bin/python scripts/analysis/plot_toy_sigma_basin.py --extract
    # redraw from visuals/final/data/
    .venv/bin/python scripts/analysis/plot_toy_sigma_basin.py

    -> paper/visuals/final/appendix/toy_sigma_basin.{pdf,png,md,tex}
       paper/visuals/final/data/toy_sigma_basin.{json,npz}
    (paper = projects/iclr_2027/paper)

The runs are `scripts/train/queue_toy.sh`'s `sigma_*` stages: ES (NES) and
the GA at sigma 0.01 .. 0.8 on three two-sub-task toys (`source/envs/
toy_landscapes.py`), 24 seeds, pop 64, a switch every 100 generations for
1000, the full centroid saved every 10 generations (`--record_centroid`):

    smooth  no local optima; the generalist region is a plateau of
            half-width h. Level: h.
    rugged  local optima of ONE width: a ripple of amplitude a digs pits of
            wavelength 0.6 into peaks that are all 1.1 wide. Level: a.
    spikes  basins of TWO widths: a wide shared hill (0.70, width 1.1) and,
            for each sub-task, a narrow specialist spike (width 0.2, height s)
            0.6 from its top, added so that the hill top is a maximum of
            neither sub-task alone. Level: s.

THE MEASURE is the paper's basin width (`curvature_width.py`) transposed to a
landscape, where the return IS the score: around the centroid x, the share of
isotropic gaussian perturbations of radius r whose score on the sub-task just
trained differs from x's by more than 2% of the peak (any change counts, as a
changed greedy action does there), on a log grid of r. The basin width is the
radius at which that share crosses one half (log-interpolated); `change_0.1`
is the share at r = 0.1, the paper's radius. Measured at the end of each of
the LAST FIVE sub-tasks (the paper's `late`) and averaged per seed; the figure
draws the geometric mean over seeds with a 95% bootstrap CI. The same measure
at the landscape's own optima (the generalist peak, the specialist) gives the
reference lines: a centroid ON a peak has that peak's width, so anything the
curve does beyond the references is position, not landscape.

Reported agent: the CENTROID (the ES mean; the mean of the GA's elite
archive), as in every toy figure. A marker is filled where at least half the
seeds' centroids are generalists (min(A, B) >= 98% of 0.70) at the end.

Left column: both sub-tasks at the drawn level, the generalist region
(dashed), and the ES centroid at the end of the run for every seed, coloured
by sigma. Middle / right: basin width against sigma for ES / the GA, one line
per level, with the landscape's reference widths dotted.
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
import plot_toy_local_optima as ptl                        # noqa: E402
sys.path.insert(0, str(ptl.REPO))
from source.envs import toy_landscapes as tl               # noqa: E402

lp = ptl.lp
REPO = ptl.REPO
RUNS = ptl.RUNS
STEM = 'toy_sigma_basin'
OUT = ptl.OUT
DATA = ptl.pcl.FINAL / 'data' / STEM
ARMS = {'es': 'nes', 'ga': 'ga'}           # figure name -> sweep method
# (landscape, row title, level drawn, level axis label, level symbol)
ROWS = [('smooth', 'Smooth', 0.1, 'Shared region $h$', 'h'),
        ('rugged', 'Local optima', 0.8, 'Barrier depth $a$', 'a'),
        ('spikes', 'Sharp specialist', 0.3, 'Specialist height $s$', 's')]
RADII = np.logspace(-3, 1, 81)             # includes 0.01, 0.1, 1
PAPER_RADIUS = 0.1
CHANGE_TOL = 0.02                          # of the peak, the sweep's q_tolerance
SAMPLES = 1024
LATE = 5                                   # sub-tasks averaged, as the paper
PROBE_SEED = 11
TEXT_WIDTH_IN = ptl.TEXT_WIDTH_IN
TARGET, TASK_COLOURS = ptl.TARGET, ptl.TASK_COLOURS
SIGMA_CMAP = ('YlOrBr', 0.35, 0.95)        # the ES ends, by sigma
LEVEL_CMAP = ('Blues', 0.45, 0.95)         # the width curves, by level


def crossing(change, radii, at=0.5):
    """Radius at which `change` (R, n), non-decreasing-ish in r, first
    reaches `at`, log-interpolated; the largest radius where it never does."""
    R, n = change.shape
    hit = change >= at
    first = np.where(hit.any(0), hit.argmax(0), R - 1)
    lo = np.clip(first - 1, 0, R - 1)
    c_lo, c_hi = change[lo, np.arange(n)], change[first, np.arange(n)]
    frac = np.where(c_hi > c_lo, (at - c_lo) / np.maximum(c_hi - c_lo, 1e-9), 1.0)
    frac = np.clip(np.where(first == 0, 1.0, frac), 0, 1)
    logr = np.log(radii[lo]) + frac * (np.log(radii[first]) - np.log(radii[lo]))
    return np.exp(logr), ~hit.any(0)


def basin(land, pts, task, level, key):
    """`(change (R, n), width (n,), censored (n,))` around `pts` (n, d) on
    `task` at `level`, with the same perturbation directions at every radius."""
    import jax
    import jax.numpy as jnp
    from jax import random
    pts = jnp.asarray(pts, dtype=jnp.float32)
    eps = random.normal(key, (SAMPLES, pts.shape[-1]), dtype=jnp.float32)
    f0 = land.task_score(pts, task, level)
    tol = CHANGE_TOL * land.peak

    def at_r(r):
        f = land.task_score(pts[:, None, :] + r * eps[None], task, level)
        return jnp.mean(jnp.abs(f - f0[:, None]) > tol, axis=1)

    change = np.asarray(jax.vmap(at_r)(jnp.asarray(RADII, dtype=jnp.float32)))
    width, censored = crossing(change, RADII)
    return change, width, censored


def specialist_of(land, name, level):
    """Where the landscape's own specialist of sub-task A sits, if it has one."""
    import jax.numpy as jnp
    if name == 'rugged':
        return np.array([-tl.START_X, 0.0])
    if name == 'spikes':
        xs = np.linspace(-1.5, -0.05, 2901)
        pts = jnp.asarray(np.stack([xs, np.zeros_like(xs)], -1), dtype=jnp.float32)
        f = np.asarray(land.task_score(pts, 0, level))
        return np.array([xs[f.argmax()], 0.0])
    return None                        # smooth: the cap region is unbounded


def smoothed_threshold(land, level, target_radius):
    """spikes: the smallest sigma at which the maximum of the sub-task
    smoothed by an isotropic gaussian of width sigma (closed form for a sum
    of gaussians in d = 2) lies inside the generalist region."""
    h, W = tl.RUGGED_SHARED, tl.RUGGED_WIDTH
    w, c = land.options['spike_width'], land.options['spike_offset']
    xs = np.linspace(-1.5, 0.5, 4001)
    for sigma in np.logspace(-2, 0, 201):
        Wh, wh = np.sqrt(W ** 2 + sigma ** 2), np.sqrt(w ** 2 + sigma ** 2)
        f = (h * W ** 2 / Wh ** 2 * np.exp(-xs ** 2 / (2 * Wh ** 2))
             + level * w ** 2 / wh ** 2 * np.exp(-(xs + c) ** 2 / (2 * wh ** 2)))
        if abs(xs[f.argmax()]) <= target_radius:
            return float(sigma)
    return None


def extract():
    import jax.numpy as jnp
    from jax import random

    from source.metrics.continual_metrics import bootstrap_ci

    key = random.key(PROBE_SEED)
    arrays = {}
    meta = dict(extracted=datetime.date.today().isoformat(), radii=RADII.tolist(),
                paper_radius=PAPER_RADIUS, change_tol=CHANGE_TOL, samples=SAMPLES,
                late=LATE, specialists=tl.SPECIALISTS.tolist(), rows={})
    for name, _title, drawn, _label, _sym in ROWS:
        run = RUNS / f'sigma_{name}'
        res = json.loads((run / 'results.json').read_text())
        curves = np.load(run / 'curves.npz')
        cfg = res['config']
        land = tl.get(name, **res['landscape_options'])
        threshold = cfg['threshold'] * land.peak
        levels = [float(x) for x in cfg['levels']]
        sigmas = [float(s) for s in cfg['sigmas']]
        stride, T, G = cfg['record_stride'], cfg['task_interval'], cfg['num_generations']
        phases = G // T
        late = list(range(phases - LATE, phases))
        # record index of the last generation of phase i, and its sub-task
        ends = [(i + 1) * T // stride - 1 for i in late]
        tasks = [i % land.num_tasks for i in late]
        row = dict(source=str(run.relative_to(REPO)), levels=levels, sigmas=sigmas,
                   drawn=drawn, peak=land.peak, threshold=threshold,
                   window=land.window, late_phases=late,
                   config={k: cfg[k] for k in ('num_generations', 'task_interval',
                                               'pop_size', 'num_seeds', 'threshold',
                                               'num_params')},
                   arms={}, reference={})
        # The landscape's own widths, on sub-task A, at every level.
        for level in levels:
            refs = {'generalist': np.zeros(2)}
            spec = specialist_of(land, name, level)
            if spec is not None:
                refs['specialist'] = spec
            out = {}
            for what, pt in refs.items():
                _c, wd, cens = basin(land, pt[None], 0, level, key)
                out[what] = dict(point=pt.tolist(), width=float(wd[0]),
                                 censored=bool(cens[0]))
            if name == 'spikes':
                # the generalist region's radius along the axis
                xs = np.linspace(0, 1, 2001)
                g = np.asarray(land.generalist_score(
                    jnp.asarray(np.stack([xs, 0 * xs], -1)), level))
                radius = float(xs[g >= threshold].max())
                out['smoothed_threshold_sigma'] = smoothed_threshold(land, level, radius)
                out['generalist_radius'] = radius
            row['reference'][f'{level:g}'] = out
        for arm, method in ARMS.items():
            per = {}
            for sigma in sigmas:
                for level in levels:
                    k = f'{method}|{sigma:g}|{level:g}'
                    cen = curves[f'cen|{k}']                 # (seeds, records, d)
                    cen_g = curves[f'cen_g|{k}']             # (seeds, records)
                    assert np.allclose(
                        np.asarray(land.generalist_score(jnp.asarray(cen), level)),
                        cen_g, atol=1e-4), k
                    width = np.zeros((len(cen), len(ends)))
                    change = np.zeros((len(cen), len(ends)))
                    censored = np.zeros((len(cen), len(ends)), bool)
                    for j, (e, task) in enumerate(zip(ends, tasks)):
                        ch, wd, cs = basin(land, cen[:, e], task, level, key)
                        width[:, j], censored[:, j] = wd, cs
                        change[:, j] = ch[np.searchsorted(RADII, PAPER_RADIUS)]
                    logw = np.log10(width).mean(1)           # per seed, over late
                    mean, lo, hi = bootstrap_ci(logw)
                    held = cen_g[:, -1] >= threshold
                    late_gen = (cen_g[:, ends] >= threshold).mean()
                    per[f'{sigma:g}|{level:g}'] = dict(
                        width_geomean=float(10 ** mean), width_lo=float(10 ** lo),
                        width_hi=float(10 ** hi),
                        width_final=float(10 ** np.log10(width[:, -1]).mean()),
                        width_per_seed=[float(v) for v in 10 ** logw],
                        change_paper_radius=float(change.mean()),
                        censored=int(censored.any(1).sum()),
                        held=int(held.sum()), n=int(len(held)),
                        late_generalist=float(late_gen),
                        end_x=float(np.abs(cen[:, -1, 0]).mean()))
                    if level == drawn and arm == 'es':
                        arrays[f'ends|{name}|{sigma:g}'] = cen[:, -1, :2].astype(np.float32)
            row['arms'][arm] = dict(method=method, cells=per)
        (x0, x1), (y0, y1) = land.window
        gx, gy = np.meshgrid(np.linspace(x0, x1, 561), np.linspace(y0, y1, 441))
        pts = tl.plane(land, gx, gy)
        base = 0.0 if name == 'rugged' else drawn
        for task in range(2):
            arrays[f'surface|{name}|{task}'] = np.asarray(
                land.task_score(pts, jnp.asarray(task), base)).astype(np.float32)
        if name == 'rugged':
            step = land.options['wavelength']
            ks = np.arange(np.ceil(x0 / step), np.floor(x1 / step) + 1) * step
            ls = np.arange(np.ceil(y0 / step), np.floor(y1 / step) + 1) * step
            lx, ly = np.meshgrid(ks, ls)
            lat = jnp.asarray(np.stack([lx.ravel(), ly.ravel()], -1), dtype=jnp.float32)
            height = np.stack([np.asarray(land.task_score(lat, jnp.asarray(t), drawn))
                               for t in range(2)], -1)
            keep = height.max(1) > 0.01
            arrays[f'optima|{name}'] = np.concatenate(
                [np.asarray(lat)[keep], height[keep]], 1).astype(np.float32)
        arrays[f'shared|{name}'] = np.asarray(
            land.generalist_score(pts, drawn)).astype(np.float32)
        meta['rows'][name] = row
        print(f'{name}: measured')
    DATA.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(DATA.with_suffix('.npz'), **arrays)
    DATA.with_suffix('.json').write_text(json.dumps(meta, indent=1) + '\n')
    print(f'wrote {DATA}.json, .npz')


def ramp(spec, n):
    cmap, lo, hi = spec
    return [plt.get_cmap(cmap)(v) for v in np.linspace(lo, hi, n)]


def draw_landscape(ax, name, row, arrays, fs):
    specialists = tl.SPECIALISTS if name == 'rugged' else None
    ptl.draw_background(ax, name, row, arrays, specialists)
    if name == 'spikes':
        c = tl.SPIKE_OFFSET
        for task, sign in enumerate((-1, 1)):
            ax.scatter([sign * c], [0], marker='x', s=12, lw=0.9,
                       color=TASK_COLOURS[task], zorder=4,
                       path_effects=[ptl.pe.withStroke(linewidth=1.8, foreground='white')])
    sigmas = row['sigmas']
    colours = ramp(SIGMA_CMAP, len(sigmas))
    for sigma, colour in zip(sigmas, colours):
        p = arrays[f'ends|{name}|{sigma:g}']
        ax.scatter(p[:, 0], p[:, 1], s=7, color=colour, edgecolor='0.15', lw=0.3,
                   zorder=6, alpha=0.9)
    ax.plot(-tl.START_X, 0, marker='s', ms=3, color='white', mec='0.15',
            mew=0.5, zorder=6)


def draw_width(ax, name, row, arm, fs):
    levels, sigmas = row['levels'], row['sigmas']
    cells = row['arms'][arm]['cells']
    colours = ramp(LEVEL_CMAP, len(levels))
    for level, colour in zip(levels, colours):
        mean = [cells[f'{s:g}|{level:g}']['width_geomean'] for s in sigmas]
        lo = [cells[f'{s:g}|{level:g}']['width_lo'] for s in sigmas]
        hi = [cells[f'{s:g}|{level:g}']['width_hi'] for s in sigmas]
        held = [cells[f'{s:g}|{level:g}']['held'] >= cells[f'{s:g}|{level:g}']['n'] / 2
                for s in sigmas]
        # A censored cell: a centroid on the rugged toy's zero floor, where no
        # perturbation changes anything and the width is a bound, not a value.
        cens = [cells[f'{s:g}|{level:g}']['censored'] > 0 for s in sigmas]
        ax.fill_between(sigmas, lo, hi, color=colour, alpha=0.2, lw=0, zorder=2)
        ax.plot(sigmas, mean, color=colour, lw=1.1, zorder=3)
        for s, m, h, c in zip(sigmas, mean, held, cens):
            ax.plot(s, m, '^' if c else 'o', ms=3.8 if c else 3.4,
                    color=colour if h else 'white', mec=colour, mew=0.9, zorder=4)
    ref = row['reference'][f"{row['drawn']:g}"]
    for what, ls_ in (('generalist', (0, (1, 1.5))), ('specialist', (0, (3, 1.5)))):
        if what in ref:
            ax.axhline(ref[what]['width'], color='0.45', lw=0.7, ls=ls_, zorder=1)
    if name == 'spikes' and arm == 'es' and ref.get('smoothed_threshold_sigma'):
        ax.axvline(ref['smoothed_threshold_sigma'], color=TARGET, lw=0.6,
                   ls='--', zorder=1)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(sigmas[0] / 1.5, sigmas[-1] * 1.5)
    ax.set_xticks(sigmas)
    ax.set_xticklabels([f'{s:g}' for s in sigmas], rotation=45, ha='right',
                       rotation_mode='anchor', fontsize=fs - 1.5)
    ax.minorticks_off()
    ax.set_xlabel(r'Search width $\sigma$', labelpad=1)
    ax.grid(True, axis='y', color='0.92', lw=0.5, zorder=0)


def draw(meta, arrays, args):
    fs = args.font_size
    fig = plt.figure(figsize=(args.width, args.height))
    gs = ptl.GridSpec(len(ROWS), 3, figure=fig, width_ratios=[1.1, 1.1, 1.1],
                      left=0.12, right=0.985, top=0.885, bottom=0.07,
                      wspace=0.4, hspace=0.55)
    ylims = {}
    for r, (name, title, drawn, label, sym) in enumerate(ROWS):
        row = meta['rows'][name]
        ax = fig.add_subplot(gs[r, 0])
        draw_landscape(ax, name, row, arrays, fs)
        ax.annotate(f'{title}\n(${sym}$ = {drawn:g})', xy=(0, 0.5),
                    xycoords='axes fraction', xytext=(-36, 0),
                    textcoords='offset points', rotation=90, ha='center',
                    va='center', fontweight='bold', linespacing=1.1)
        if r == 0:
            ax.set_title('ES centroid at the end, by $\\sigma$', fontweight='bold', pad=3)
        panels = []
        for c, arm in enumerate(ARMS, start=1):
            ax = fig.add_subplot(gs[r, c])
            draw_width(ax, name, row, arm, fs)
            panels.append(ax)
            if c == 1:
                ax.set_ylabel('Basin width (radius)', labelpad=1)
            if r == 0:
                ax.set_title(f'{ptl.style(arm)[1]}: basin width',
                             fontweight='bold', pad=3)
            # one legend of levels per row, in the ES panel
            if c == 1:
                colours = ramp(LEVEL_CMAP, len(row['levels']))
                handles = [Line2D([], [], color=col, lw=1.2, marker='o', ms=3,
                                  label=f'${sym}$ = {lv:g}')
                           for lv, col in zip(row['levels'], colours)]
                ax.legend(handles=handles, loc='upper left', frameon=False,
                          fontsize=fs - 1, handlelength=1.3, borderaxespad=0.3,
                          labelspacing=0.25, handletextpad=0.4)
        lo = min(ax.get_ylim()[0] for ax in panels)
        hi = max(ax.get_ylim()[1] for ax in panels)
        for ax in panels:
            ax.set_ylim(lo, hi)
        ylims[name] = (lo, hi)
    for ax in fig.axes:
        ax.tick_params(pad=1.5)
        if not ax.get_images():
            ax.spines[['top', 'right']].set_visible(False)
    sig = meta['rows'][ROWS[0][0]]['sigmas']
    colours = ramp(SIGMA_CMAP, len(sig))
    handles = [Line2D([], [], ls='none', marker='o', ms=3.5, color=col, mec='0.15',
                      mew=0.3, label=f'$\\sigma$ = {s:g}') for s, col in zip(sig, colours)]
    handles += [
        Line2D([], [], color=TARGET, lw=0.7, ls='--', label='generalist region'),
        Line2D([], [], color='0.45', lw=0.8, ls=(0, (1, 1.5)), label='width of generalist peak'),
        Line2D([], [], color='0.45', lw=0.8, ls=(0, (3, 1.5)), label='width of specialist peak'),
        Line2D([], [], color=TARGET, lw=0.6, ls='--', label=r'predicted $\sigma^*$'),
        Line2D([], [], ls='none', marker='o', ms=3.4, color='0.3', mec='0.3', mew=0.9,
               label='half+ seeds generalist'),
        Line2D([], [], ls='none', marker='o', ms=3.4, color='white', mec='0.3', mew=0.9,
               label='fewer'),
        Line2D([], [], ls='none', marker='^', ms=3.8, color='white', mec='0.3', mew=0.9,
               label='score floor (bound)'),
    ]
    fig.legend(handles=handles, loc='upper center', ncol=5, frameon=False,
               bbox_to_anchor=(0.5, 1.0), handlelength=1.4, columnspacing=0.8,
               handletextpad=0.3, fontsize=fs - 0.5)
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ('pdf', 'png'):
        fig.savefig(OUT / f'{STEM}.{ext}', dpi=300)
    plt.close(fig)
    print(f'wrote {OUT / STEM}.pdf, .png  ({args.width:.2f} x {args.height:.2f} in)')


def tables(meta):
    arms = list(ARMS)
    md = [f'# {STEM}', '',
          f"Extracted {meta['extracted']} from "
          + ', '.join(f"`{meta['rows'][n]['source']}`" for n, *_ in ROWS)
          + " (`scripts/train/queue_toy.sh`, stage sigma). Built by "
          "`scripts/analysis/plot_toy_sigma_basin.py`.", '',
          'Two sub-tasks alternate every 100 generations for 1000; pop 64, 24 seeds, '
          'd = 2. ES = NES, GA with archive re-scoring, both at every sigma. '
          'Every number is for the CENTROID (the ES mean; the mean of the GA elite '
          'archive).', '',
          f"- width: the radius of isotropic gaussian perturbation at which half the "
          f"perturbed copies of the centroid change their score on the sub-task just "
          f"trained by more than {meta['change_tol']:.0%} of the peak ({meta['samples']} "
          f"draws, common across radii); measured at the end of each of the last "
          f"{meta['late']} sub-tasks, geometric mean over them and over seeds "
          f"[95% bootstrap CI over seeds]. Censored cells (a seed whose share never "
          f"crossed one half by r = {RADII[-1]:g}) are counted at that radius.",
          f"- change@{meta['paper_radius']:g}: the share that changes at r = "
          f"{meta['paper_radius']:g}, the paper's relative radius, mean over seeds and "
          f"the late sub-tasks (the paper's `width` column is this quantity; lower = wider).",
          '- held: seeds whose centroid is a generalist (min(A, B) >= 98% of 0.70) at '
          'the last generation.',
          '- reference: the same width measured AT the landscape\'s generalist peak '
          '(the origin) and at sub-task A\'s specialist, on sub-task A.', '']
    for name, title, drawn, _label, sym in ROWS:
        row = meta['rows'][name]
        md += [f'## {title} (`{name}`)', '']
        for level in row['levels']:
            ref = row['reference'][f'{level:g}']
            parts = [f"generalist {ref['generalist']['width']:.3f}"]
            if 'specialist' in ref:
                parts.append(f"specialist {ref['specialist']['width']:.3f} at "
                             f"x = {ref['specialist']['point'][0]:.2f}")
            if ref.get('smoothed_threshold_sigma'):
                parts.append(f"smoothed maximum enters the generalist region (radius "
                             f"{ref['generalist_radius']:.2f}) at sigma = "
                             f"{ref['smoothed_threshold_sigma']:.2f}")
            md.append(f'- reference widths at ${sym}$ = {level:g}: ' + '; '.join(parts))
        md.append('')
        head = ['sigma', 'level'] + [f'{ptl.style(a)[1]} {c}' for a in arms
                                     for c in ('width [CI]', f"change@{meta['paper_radius']:g}",
                                               'held', 'censored')]
        md += ['| ' + ' | '.join(head) + ' |', '|' + '---|' * len(head)]
        for sigma in row['sigmas']:
            for level in row['levels']:
                cells = [f'{sigma:g}', f'{sym} = {level:g}']
                for arm in arms:
                    c = row['arms'][arm]['cells'][f'{sigma:g}|{level:g}']
                    cells += [f"{c['width_geomean']:.3f} [{c['width_lo']:.3f}, {c['width_hi']:.3f}]",
                              f"{c['change_paper_radius']:.2f}", f"{c['held']}/{c['n']}",
                              str(c['censored'])]
                md.append('| ' + ' | '.join(cells) + ' |')
        md.append('')
    (OUT / f'{STEM}.md').write_text('\n'.join(md) + '\n')

    # LaTeX: one block per landscape, rows sigma, columns level x arm, cell
    # "width (held)".
    tex = []
    for name, title, _drawn, _label, sym in ROWS:
        row = meta['rows'][name]
        levels = row['levels']
        tex += [r'\begin{tabular}{l' + '|cc' * len(levels) + '}', r'\toprule',
                r'\multicolumn{' + str(1 + 2 * len(levels)) + r'}{l}{\textbf{' + title
                + r'}}\\',
                r'$\sigma$ & ' + ' & '.join(r'\multicolumn{2}{c}{$' + f'{sym} = {lv:g}' + '$}'
                                            for lv in levels) + r' \\',
                ' & ' + ' & '.join('ES & GA' for _ in levels) + r' \\', r'\midrule']
        ref = ' & '.join(
            f"\\multicolumn{{2}}{{c}}{{{row['reference'][f'{lv:g}']['generalist']['width']:.2f}"
            + (f" / {row['reference'][f'{lv:g}']['specialist']['width']:.2f}"
               if 'specialist' in row['reference'][f'{lv:g}'] else '') + '}'
            for lv in levels)
        tex.append(r'peak & ' + ref + r' \\')
        tex.append(r'\midrule')
        for sigma in row['sigmas']:
            cells = [f'{sigma:g}']
            for lv in levels:
                for arm in arms:
                    c = row['arms'][arm]['cells'][f'{sigma:g}|{lv:g}']
                    cells.append(f"{c['width_geomean']:.2f} ({c['held']})")
            tex.append(' & '.join(cells) + r' \\')
        tex += [r'\bottomrule', r'\end{tabular}', '']
    (OUT / f'{STEM}.tex').write_text('\n'.join(tex))
    print(f'wrote {OUT / STEM}.md, .tex')


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--extract', action='store_true',
                    help='re-read the sweeps and re-measure the basins first')
    ap.add_argument('--width', type=float, default=TEXT_WIDTH_IN)
    ap.add_argument('--height', type=float, default=5.3)
    ap.add_argument('--font-size', type=float, default=7.0)
    args = ap.parse_args()
    if args.extract:
        extract()
    meta = json.loads(DATA.with_suffix('.json').read_text())
    arrays = dict(np.load(DATA.with_suffix('.npz')))
    plt.rcParams.update({
        'font.size': args.font_size, 'axes.titlesize': args.font_size + 0.5,
        'axes.labelsize': args.font_size, 'legend.fontsize': args.font_size,
        'xtick.labelsize': args.font_size - 0.5, 'ytick.labelsize': args.font_size - 0.5,
        'axes.linewidth': 0.5, 'xtick.major.width': 0.5, 'ytick.major.width': 0.5,
        'xtick.major.size': 2, 'ytick.major.size': 2,
    })
    draw(meta, arrays, args)
    tables(meta)
    return 0


if __name__ == '__main__':
    sys.exit(main())
