"""Appendix figure: on two toy landscapes, ES keeps a generalist the GA drifts
out of when the landscape is smooth, and the GA crosses local optima that
stop ES when it is rugged.

    # re-read the sweeps (only when they change)
    JAX_PLATFORMS=cpu .venv/bin/python scripts/analysis/plot_toy_local_optima.py --extract
    # redraw from visuals/final/data/
    .venv/bin/python scripts/analysis/plot_toy_local_optima.py
    # d = 2 against d = 128 (needs both saved extracts)
    .venv/bin/python scripts/analysis/plot_toy_local_optima.py --compare-dims --height 1.9

    -> paper/visuals/final/appendix/toy_local_optima.{pdf,png,md,tex}
       paper/visuals/final/data/toy_local_optima.{json,npz}
    (paper = projects/iclr_2027/paper)

The runs are the two switching toys of the generalists report, re-run with
source/studies/toy/sweep.py (which records the centroid every generation):

    smooth  projects/iclr_2027/runs_toy/smooth (report section A). Each
            sub-task is a plane tilted toward its own side and capped at 0.70;
            both are at the cap on the lens |x| + y^2 <= h, the shared region.
            No local optima (the surfaces are concave). Level: h.
    rugged  projects/iclr_2027/runs_toy/rugged (report section B). Three
            specialist peaks (0.65) per sub-task, B's mirroring A's, and one
            shared generalist peak (0.70) at the origin. A periodic term
            splits the plane into basins of width 0.6, each a local optimum,
            with barriers of depth a/2 a coordinate; no peak moves.
            Level: a.

Two sub-tasks alternate every 100 generations for 1000; every run starts at
(-1.8, 0), on sub-task A's side. 24 seeds, pop 64.

Reported agent: the CENTROID, the coordinate-wise mean of the population
(`population_mean`: the ES mean, the mean of the GA's elite archive). Its
generalist score is min(A, B) (ripple-free on `rugged`), recomputed here from
the saved centroid path at every generation and checked against the sweep's
own every-10-generation record. A seed is a generalist when that reaches 98%
of the 0.70 peak.

Arms: ES is NES (CLAUDE.md); the GA re-scores its archive every generation.
Each at its best sigma of that sweep (the sweep's own choice).

Third column: share of seeds generalist at the end against sigma, at the
drawn level, from the wider sigma sweeps runs_toy/sigma_{smooth,rugged}
(GA and NES only, sigma 0.01 to 0.8; d = 2 only).
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
import matplotlib.patheffects as pe                        # noqa: E402
import matplotlib.pyplot as plt                            # noqa: E402
from matplotlib.colors import to_rgb                        # noqa: E402
from matplotlib.gridspec import GridSpec                   # noqa: E402
from matplotlib.lines import Line2D                        # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import plot_continual_lineplots as pcl                     # noqa: E402

lp = pcl.lp
REPO = pathlib.Path(__file__).resolve().parents[2]
RUNS = REPO / 'projects/iclr_2027/runs_toy'
STEM = 'toy_local_optima'
OUT = pcl.FINAL / 'appendix'
DATA = pcl.FINAL / 'data' / STEM
RUN_SUFFIX = ''                            # '_d128' under --dims 128
# figure name -> sweep method, in the paper's one method order
# (make_lineplot.METHOD_ORDER: GA before ES) since 2026-09-21.
ARMS = {'ga': 'ga', 'es': 'nes'}
# (landscape, row title, level drawn, level axis label, level symbol)
ROWS = [('smooth', 'Smooth', 0.1, 'Shared region $h$', 'h'),
        ('rugged', 'Local optima', 1.6, 'Barrier depth $a$', 'a')]
SEED = 0                                   # the trajectory drawn, not chosen
TARGET = '#c0392b'
TASK_COLOURS = ('#3f6fb0', '#c2562f')     # sub-task A, sub-task B
TEXT_WIDTH_IN = 5.5                        # ICLR
# One window for both rows, so the two landscape panels are the same size.
WINDOW = ((-2.8, 2.8), (-2.2, 2.2))
# The smooth row's panel zooms on the shared region at the same aspect, so the
# switching (|x| ~ 0.5 against a region of half-width 0.1) is legible.
ZOOM = ((-1.27, 1.27), (-1.0, 1.0))
LAST_SWITCHES = 4                          # sub-task ends drawn on the smooth row


def configure(dims):
    """Point STEM / DATA / the run directories at the d = `dims` re-run."""
    global STEM, DATA, RUN_SUFFIX, ROWS
    if dims != 2:
        STEM = f'toy_local_optima_d{dims}'
        DATA = pcl.FINAL / 'data' / STEM
        RUN_SUFFIX = f'_d{dims}'
        # At a = 1.6 nothing crosses in d = 128; draw the largest level ES
        # still holds at the figure's width, where the two arms differ.
        ROWS = [ROWS[0], ('rugged', 'Local optima', 0.3, *ROWS[1][3:])]


def extract():
    import jax.numpy as jnp
    from source.envs import toy_landscapes as tl
    from source.metrics.continual_metrics import bootstrap_ci

    arrays = {}
    meta = dict(extracted=datetime.date.today().isoformat(), seed=SEED,
                specialists=tl.SPECIALISTS.tolist(), rows={})
    for name, _title, drawn, _label, _sym in ROWS:
        run = RUNS / f'{name}{RUN_SUFFIX}'
        res = json.loads((run / 'results.json').read_text())
        curves = np.load(run / 'curves.npz')
        cfg = res['config']
        land = tl.get(name, **res['landscape_options'])
        threshold = cfg['threshold'] * land.peak
        levels = [float(x) for x in cfg['levels']]
        row = dict(source=str(run.relative_to(REPO)), levels=levels, drawn=drawn,
                   peak=land.peak, threshold=threshold, window=WINDOW,
                   config={k: cfg[k] for k in ('num_generations', 'task_interval',
                                               'pop_size', 'num_seeds', 'threshold')},
                   num_params=cfg['num_params'], arms={}, sigma_table={})
        stride = cfg['record_stride']
        # Every swept width, from the centroid record (every `stride`), plus
        # the wider widths of <run>_wide where it exists (d = 128).
        swept = [(curves, float(x)) for x in cfg['sigmas']]
        wide = run.with_name(run.name + '_wide')
        if (wide / 'curves.npz').exists():
            wcurves = np.load(wide / 'curves.npz')
            wcfg = json.loads((wide / 'results.json').read_text())['config']
            swept += [(wcurves, float(x)) for x in wcfg['sigmas']]
            row['source'] += f', {wide.relative_to(REPO)}'
        for arm, method in ARMS.items():
            sigma = res['best_sigma'][method]
            table = {}
            for src, s_ in swept:
                table[f'{s_:g}'] = {}
                for x in levels:
                    h = src[f'cen_g|{method}|{s_:g}|{x:g}'] >= threshold
                    first = (h.argmax(1) + 1) * stride
                    table[f'{s_:g}'][f'{x:g}'] = dict(
                        found=int(h.any(1).sum()), held=int(h[:, -1].sum()),
                        n=int(len(h)),
                        first_median=(float(np.median(first[h.any(1)]))
                                      if h.any() else None))
            row['sigma_table'][arm] = table
            per = {}
            for level in levels:
                key = f'{method}|{sigma:g}|{level:g}'
                cpath = curves[f'cpath|{key}']
                g = np.asarray(land.generalist_score(jnp.asarray(cpath), level))
                # The sweep scored the same centroid every `stride` generations.
                assert np.allclose(g[:, stride - 1::stride], curves[f'cen_g|{key}'],
                                   atol=1e-4), key
                hit = g >= threshold
                found = hit.any(1)
                first = hit.argmax(1)
                after = np.arange(hit.shape[1]) >= first[:, None]
                kept = (hit & after).sum(1) / after.sum(1)
                per[f'{level:g}'] = dict(
                    n=int(len(g)), found=int(found.sum()), held=int(hit[:, -1].sum()),
                    retention=float(kept[found].mean()) if found.any() else None,
                    first_median=float(np.median(first[found])) if found.any() else None,
                    final=[float(v) for v in g[:, -1]])
                if level == drawn:
                    mean, lo, hi = bootstrap_ci(g)
                    arrays[f'curve|{name}|{arm}'] = np.stack([mean, lo, hi]).astype(np.float32)
                    arrays[f'path|{name}|{arm}'] = cpath[SEED].astype(np.float32)
            row['arms'][arm] = dict(method=method, sigma=sigma, levels=per)
        (x0, x1), (y0, y1) = WINDOW
        gx, gy = np.meshgrid(np.linspace(x0, x1, 561), np.linspace(y0, y1, 441))
        pts = tl.plane(land, gx, gy)
        # The background is the ripple-free pair; the ripple's local optima are
        # drawn on top as points, or at a = 1.6 the two sub-tasks' shapes are lost.
        base = 0.0 if name == 'rugged' else drawn
        for task in range(2):
            arrays[f'surface|{name}|{task}'] = np.asarray(
                land.task_score(pts, jnp.asarray(task), base)).astype(np.float32)
        if name == 'rugged':
            # Every lattice point of the ripple is a local optimum of the rippled
            # sub-tasks; kept where either sub-task still scores above 0.
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
        sweep = RUNS / f'sigma_{name}'
        if RUN_SUFFIX == '' and (sweep / 'results.json').exists():
            sres = json.loads((sweep / 'results.json').read_text())
            sthr = sres['config']['threshold'] * sres['peak']
            row['sigma_source'] = str(sweep.relative_to(REPO))
            row['sigma_sweep'] = {}
            for arm, method in ARMS.items():
                cells = {f"{c['sigma']:g}": c for c in sres['cells']
                         if c['method'] == method and c['level'] == drawn}
                row['sigma_sweep'][arm] = {
                    k: dict(n=len(c['final_generalist']),
                            held=int((np.asarray(c['final_generalist']) >= sthr).sum()))
                    for k, c in sorted(cells.items(), key=lambda kv: float(kv[0]))}
        meta['rows'][name] = row
    DATA.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(DATA.with_suffix('.npz'), **arrays)
    DATA.with_suffix('.json').write_text(json.dumps(meta, indent=1) + '\n')
    print(f'wrote {DATA}.json, .npz')


def style(arm):
    return lp.METHOD_STYLE[arm]['color'], lp.METHOD_STYLE[arm]['label']


def draw_background(ax, name, row, arrays, specialists=None):
    """The landscape panel without any run on it: both sub-tasks as inks, the
    generalist region, the ripple's optima and the specialist peaks.

    `row` needs `window`, `peak` and `threshold`; `arrays` the
    `surface|<name>|<task>` pair, `shared|<name>` and, optionally,
    `optima|<name>` and (with `specialists`) the rugged toy's peak positions.
    `plot_toy_sigma_basin.py` draws its landscapes with this too.
    """
    (x0, x1), (y0, y1) = row['window']
    extent = (x0, x1, y0, y1)
    # Each sub-task is one ink on white paper (subtractive mix), so the region
    # both score well on comes out as the dark overlap of the two.
    ink = 0.0
    for task, colour in enumerate(TASK_COLOURS):
        level = np.clip(arrays[f'surface|{name}|{task}'] / row['peak'], 0, 1) ** 1.5
        ink = ink + level[..., None] * (1 - np.array(to_rgb(colour)))
    ax.imshow(np.clip(1 - 0.85 * ink, 0, 1), origin='lower', extent=extent,
              interpolation='bilinear', aspect='equal', rasterized=True)
    for task, (x, ha) in enumerate(((0.04, 'left'), (0.96, 'right'))):
        ax.text(x, 0.94, 'AB'[task], transform=ax.transAxes, ha=ha, va='top',
                fontweight='bold', color=TASK_COLOURS[task], zorder=9,
                path_effects=[pe.withStroke(linewidth=1.5, foreground='white')])
    # The generalist target: where min(A, B) clears the threshold.
    ax.contour(arrays[f'shared|{name}'], levels=[row['threshold']], origin='lower',
               extent=extent, colors=TARGET, linewidths=0.9, linestyles='--', zorder=8)
    if f'optima|{name}' in arrays:
        opt = arrays[f'optima|{name}']
        ax.scatter(opt[:, 0], opt[:, 1], s=0.6 + 5 * opt[:, 2:].max(1) / row['peak'], alpha=0.85,
                   facecolor='white', edgecolor='0.25', lw=0.3, zorder=4)
    if specialists is not None:
        spec = np.array(specialists)
        for task, sign in enumerate((1, -1)):
            ax.scatter(sign * spec[:, 0], spec[:, 1], marker='x', s=12, lw=0.9,
                       color=TASK_COLOURS[task], zorder=4,
                       path_effects=[pe.withStroke(linewidth=1.8, foreground='white')])
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_xticks([-2, 0, 2])
    ax.set_yticks([-2, 0, 2] if y1 > 2 else [-1, 0, 1])
    ax.set_xlabel(r'$\theta_1$', labelpad=1)
    ax.set_ylabel(r'$\theta_2$', labelpad=0)


def draw_switches(ax, name, meta, arrays):
    """The centroid at the end of every sub-task, as the letter of that
    sub-task, joined in order from the start: where the centroid switches
    between the sub-tasks' sides, the line zigzags across the shared region."""
    cfg = meta['rows'][name]['config']
    step = cfg['task_interval']
    for arm in ARMS:
        colour, _ = style(arm)
        p = arrays[f'path|{name}|{arm}']
        # The last few sub-tasks only: all ten tangle into a knot, and the
        # start lies outside the zoomed panel. Sub-task k ends at k * step.
        count = cfg['num_generations'] // step
        first = count - LAST_SWITCHES + 1
        ends = p[first * step - 1::step]
        line = ends
        ax.plot(line[:, 0], line[:, 1], color=colour, lw=0.6, alpha=0.9, zorder=5,
                path_effects=[pe.Stroke(linewidth=1.3, foreground='0.2', alpha=0.4),
                              pe.Normal()])
        for k, (x, y) in enumerate(ends):
            ax.text(x, y, 'AB'[(first - 1 + k) % 2], color=colour, fontsize=plt.rcParams['font.size'] - 1,
                    fontweight='bold', ha='center', va='center', zorder=7,
                    path_effects=[pe.withStroke(linewidth=1.4, foreground='0.15')])


def draw_surface(ax, name, meta, arrays):
    row = meta['rows'][name]
    draw_background(ax, name, row, arrays,
                    meta['specialists'] if name == 'rugged' else None)
    start = arrays[f'path|{name}|es'][0]
    ax.plot(*start, marker='s', ms=3, color='white', mec='0.15', mew=0.5, zorder=6)
    if name == 'smooth':
        draw_switches(ax, name, meta, arrays)
        (x0, x1), (y0, y1) = ZOOM
        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)
        ax.set_xticks([-1, 0, 1])
        ax.set_yticks([-1, 0, 1])
        return
    for arm in ARMS:
        colour, _ = style(arm)
        p = arrays[f'path|{name}|{arm}']
        ax.plot(p[:, 0], p[:, 1], color=colour, lw=0.8, alpha=0.95, zorder=5,
                solid_joinstyle='round',
                path_effects=[pe.Stroke(linewidth=1.6, foreground='0.2', alpha=0.5),
                              pe.Normal()])
        ax.plot(*p[-1], 'o', ms=3.2, color=colour, mec='0.15', mew=0.5, zorder=7)


def draw_sweep(ax, name, label, meta):
    row = meta['rows'][name]
    levels = row['levels']
    x = np.arange(len(levels))
    i = levels.index(row['drawn'])
    ax.axvspan(i - 0.3, i + 0.3, color='0.92', lw=0, zorder=0)
    for k, (arm, spec) in enumerate(row['arms'].items()):
        colour, _ = style(arm)
        held = [spec['levels'][f'{a:g}']['held'] / spec['levels'][f'{a:g}']['n']
                for a in levels]
        ax.plot(x + (k - 0.5) * 0.1, held, '-o', color=colour, lw=1.1, ms=3,
                mec='0.15', mew=0.4, zorder=3)
    ax.set_xticks(x)
    # Every other level labelled: seven fit the axis only as ticks.
    ax.set_xticklabels([f'{a:g}' if i % 2 == 0 or len(levels) < 5 else ''
                        for i, a in enumerate(levels)])
    ax.set_xlim(-0.5, len(levels) - 0.5)
    ax.set_ylim(-0.05, 1.08)
    ax.set_yticks([0, 0.5, 1])
    ax.set_ylabel('In shared region', labelpad=1)
    ax.set_xlabel(label, labelpad=1)


def draw_sigma(ax, name, meta):
    """Share of seeds generalist at the end against sigma, at the drawn level;
    the sigma the other panels use is shaded."""
    row = meta['rows'][name]
    sweep = row['sigma_sweep']
    sigmas = list(next(iter(sweep.values())))
    x = np.arange(len(sigmas))
    for arm in sweep:
        used = f"{row['arms'][arm]['sigma']:g}"
        if used in sigmas:
            i = sigmas.index(used)
            ax.axvspan(i - 0.3, i + 0.3, color='0.92', lw=0, zorder=0)
    for k, (arm, cells) in enumerate(sweep.items()):
        colour, _ = style(arm)
        held = [cells[s_]['held'] / cells[s_]['n'] for s_ in sigmas]
        ax.plot(x + (k - 0.5) * 0.1, held, '-o', color=colour, lw=1.1, ms=3,
                mec='0.15', mew=0.4, zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels([f'{s_:g}' if i % 2 == 0 else '' for i, s_ in
                        enumerate(float(v) for v in sigmas)])
    ax.set_xlim(-0.5, len(sigmas) - 0.5)
    ax.set_ylim(-0.05, 1.08)
    ax.set_yticks([0, 0.5, 1])
    ax.set_xlabel(r'Search width $\sigma$', labelpad=1)


def draw(meta, arrays, args):
    width, height = args.width, args.height
    fig = plt.figure(figsize=(width, height))
    with_sigma = all('sigma_sweep' in meta['rows'][n] for n, *_ in ROWS)
    # No centroid-over-time column since 2026-09-21: it showed one sigma only.
    gs = GridSpec(2, 3 if with_sigma else 2, figure=fig,
                  width_ratios=[1.1, 1.0, 1.0][:3 if with_sigma else 2],
                  left=0.115, right=0.99, top=0.88, bottom=0.1,
                  wspace=0.5, hspace=0.55)
    for r, (name, title, drawn, label, sym) in enumerate(ROWS):
        ax = fig.add_subplot(gs[r, 0])
        draw_surface(ax, name, meta, arrays)
        ax.annotate(f'{title}\n(${sym}$ = {drawn:g})', xy=(0, 0.5),
                    xycoords='axes fraction', xytext=(-30, 0),
                    textcoords='offset points', rotation=90, ha='center',
                    va='center', linespacing=1.1)
        if r == 0:
            ax.set_title('Tasks A and B', pad=3)
        ax = fig.add_subplot(gs[r, 1])
        draw_sweep(ax, name, label, meta)
        if r == 0:
            ax.set_title('Across levels', pad=3)
        if with_sigma:
            ax = fig.add_subplot(gs[r, 2])
            draw_sigma(ax, name, meta)
            if r == 0:
                ax.set_title(r'Across $\sigma$', pad=3)
    for ax in fig.axes:
        ax.tick_params(pad=1.5)
        if not ax.get_images():
            ax.spines[['top', 'right']].set_visible(False)
    handles = [Line2D([], [], color=style(arm)[0], lw=1.5, label=style(arm)[1]) for arm in ARMS]
    handles += [
        Line2D([], [], color=TARGET, lw=0.7, ls='--', label='shared region'),
        Line2D([], [], ls='none', marker='x', ms=4, color=TASK_COLOURS[0], mew=0.9,
               label='A peaks'),
        Line2D([], [], ls='none', marker='x', ms=4, color=TASK_COLOURS[1], mew=0.9,
               label='B peaks'),
        Line2D([], [], ls='none', marker='o', ms=2.5, color='white', mec='0.2',
               mew=0.35, label='local optima'),
        Line2D([], [], ls='none', marker='s', ms=3, color='white', mec='0.15',
               mew=0.5, label='start'),
    ]
    fig.legend(handles=handles, loc='upper center', ncol=len(handles),
               frameon=False, bbox_to_anchor=(0.5, 1.0), handlelength=1.4,
               columnspacing=0.9, handletextpad=0.3)
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ('pdf', 'png'):
        fig.savefig(OUT / f'{STEM}.{ext}', dpi=300)
    plt.close(fig)
    print(f'wrote {OUT / STEM}.pdf, .png  ({width:.2f} x {height:.2f} in)')


def draw_dims(args):
    """Both landscapes at d = 2 against d = 128: share of seeds generalist at
    the end at every level, one line per (method, d). Each arm at its sweep's
    best sigma, which is the same in both dimensions."""
    metas = {d: json.loads((pcl.FINAL / 'data' / f"toy_local_optima{'' if d == 2 else f'_d{d}'}.json")
                           .read_text()) for d in (2, 128)}
    fig, axes = plt.subplots(1, 2, figsize=(args.width, args.height))
    fig.subplots_adjust(left=0.08, right=0.99, top=0.8, bottom=0.2, wspace=0.25)
    dash = {2: '-', 128: (0, (3, 1.5))}
    for ax, (name, title, _drawn, label, _sym) in zip(axes, ROWS):
        levels = metas[2]['rows'][name]['levels']
        assert levels == metas[128]['rows'][name]['levels'], name
        x = np.arange(len(levels))
        for k, (arm, d) in enumerate([(a, d) for a in ARMS for d in (2, 128)]):
            spec = metas[d]['rows'][name]['arms'][arm]
            held = [spec['levels'][f'{v:g}']['held'] / spec['levels'][f'{v:g}']['n']
                    for v in levels]
            colour, _ = style(arm)
            ax.plot(x + (k - 1.5) * 0.06, held, ls=dash[d], marker='o', color=colour,
                    lw=1.1, ms=3.2, mec=colour, mew=0.8,
                    mfc=colour if d == 2 else 'white', zorder=3)
        ax.set_xticks(x)
        ax.set_xticklabels([f'{v:g}' for v in levels])
        ax.set_xlim(-0.4, len(levels) - 0.6)
        ax.set_ylim(-0.05, 1.08)
        ax.set_yticks([0, 0.5, 1])
        ax.set_xlabel(label, labelpad=1)
        ax.set_title(title, pad=3)
        ax.spines[['top', 'right']].set_visible(False)
        ax.tick_params(pad=1.5)
    axes[0].set_ylabel('In shared region', labelpad=1)
    handles = [Line2D([], [], color=style(arm)[0], lw=1.5, label=style(arm)[1]) for arm in ARMS]
    handles += [Line2D([], [], color='0.3', lw=1.1, ls=dash[d], marker='o', ms=3.2,
                       mec='0.3', mfc='0.3' if d == 2 else 'white', label=f'$d$ = {d}')
                for d in (2, 128)]
    fig.legend(handles=handles, loc='upper center', ncol=len(handles), frameon=False,
               bbox_to_anchor=(0.5, 1.0), handlelength=2.2, columnspacing=1.2)
    stem = 'toy_local_optima_dims'
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ('pdf', 'png'):
        fig.savefig(OUT / f'{stem}.{ext}', dpi=300)
    plt.close(fig)
    print(f'wrote {OUT / stem}.pdf, .png  ({args.width:.2f} x {args.height:.2f} in)')


def fmt(v, spec='.2f'):
    return 'n/a' if v is None else format(v, spec)


def tables(meta):
    arms = list(ARMS)
    head = ['Landscape', 'Level'] + [f'{style(a)[1]} {c}' for a in arms
                                     for c in ('found', 'held', 'retention', 'gens to find')]
    body = []
    for name, title, _drawn, _label, sym in ROWS:
        row = meta['rows'][name]
        for level in row['levels']:
            cells = [title, f'{sym} = {level:g}']
            for arm in arms:
                r = row['arms'][arm]['levels'][f'{level:g}']
                cells += [f"{r['found']}/{r['n']}", f"{r['held']}/{r['n']}",
                          fmt(r['retention']), fmt(r['first_median'], '.0f')]
            body.append(cells)
    sources = ', '.join(f"`{meta['rows'][n]['source']}`" for n, *_ in ROWS)
    sigmas = '; '.join(
        f'{t}: ' + ', '.join(f"{style(a)[1]} sigma {meta['rows'][n]['arms'][a]['sigma']}"
                             for a in arms)
        for n, t, *_ in ROWS)
    md = [f'# {STEM}', '',
          f"Extracted {meta['extracted']} from {sources} (report sections A and B, "
          f"re-run with `source/studies/toy/sweep.py`). Built by "
          f"`scripts/analysis/plot_toy_local_optima.py`.", '',
          'Two sub-tasks alternate every 100 generations for 1000. Pop 64, 24 seeds. '
          f"ES = NES, GA with archive re-scoring, at each sweep's best sigma ({sigmas}).",
          'Every number is for the CENTROID (the ES mean; the mean of the GA elite '
          'archive). Its generalist score is min(A, B); a generalist scores at least '
          '98% of the 0.70 peak.', '',
          '- found: seeds whose centroid was ever a generalist. held: seeds whose centroid '
          'was a generalist at the last generation.',
          '- retention: mean share of the generations after the first crossing that were '
          'still generalist, over the seeds that found one.',
          '- gens to find: median first crossing over those seeds.', '',
          f"Left: both sub-tasks (A blue, B red, dark where both score well; ripple-free "
          f"on the local-optima row, whose ripple optima are the dots, sized by height), "
          f"the generalist region (dashed), and seed {meta['seed']}'s centroid "
          f"path. Middle: mean and 95% bootstrap CI over seeds. Right: held share at every "
          f"level (the drawn level shaded).", '',
          '| ' + ' | '.join(head) + ' |', '|' + '---|' * len(head)]
    md += ['| ' + ' | '.join(r) + ' |' for r in body]
    (OUT / f'{STEM}.md').write_text('\n'.join(md) + '\n')
    tex = [r'\begin{tabular}{ll' + '|cccc' * len(arms) + '}', r'\toprule',
           ' & & ' + ' & '.join(r'\multicolumn{4}{c}{' + style(a)[1] + '}'
                                for a in arms) + r' \\',
           'Landscape & Level & '
           + ' & '.join(['Found', 'Held', 'Ret.', 'Gens'] * len(arms)) + r' \\']
    last = None
    for r in body:
        if r[0] != last:
            tex.append(r'\midrule')
        cells = ['' if r[0] == last else r[0], '$' + r[1] + '$'] + r[2:]
        last = r[0]
        tex.append(' & '.join(cells).replace('n/a', '--') + r' \\')
    tex += [r'\bottomrule', r'\end{tabular}']
    (OUT / f'{STEM}.tex').write_text('\n'.join(tex) + '\n')
    print(f'wrote {OUT / STEM}.md, .tex')


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--extract', action='store_true',
                    help='re-read the sweeps into the saved data first')
    ap.add_argument('--width', type=float, default=TEXT_WIDTH_IN)
    ap.add_argument('--height', type=float, default=3.35)
    ap.add_argument('--font-size', type=float, default=7.0)
    ap.add_argument('--dims', type=int, default=2,
                    help='128: the re-run in d = 128 (runs_toy/<landscape>_d128)')
    ap.add_argument('--compare-dims', action='store_true',
                    help='draw d = 2 against d = 128 from the saved data '
                         '(-> toy_local_optima_dims)')
    args = ap.parse_args()
    configure(args.dims)
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
    if args.compare_dims:
        draw_dims(args)
        return 0
    draw(meta, arrays, args)
    tables(meta)
    return 0


if __name__ == '__main__':
    sys.exit(main())
