"""Appendix E: every metric against the SWITCH INTERVAL, at a fixed budget.

    .venv/bin/python scripts/analysis/plot_frequency.py
    .venv/bin/python scripts/analysis/plot_frequency.py --agent elite --metrics cum F ZT

One column an environment, one row a metric, x the number of generations a
sub-task lasts. The three points on each x axis are three run trees that
differ in NOTHING but that interval -- same 4000-generation budget, same ten
sub-tasks (the draw is seeded by the trial and nests), same cells and arms:

    50   projects/iclr_2027/paper/gymnax/noise/interval50    80 phases
    200  projects/iclr_2027/paper/gymnax/noise/10task        20 phases  (the paper's)
    400  projects/iclr_2027/paper/gymnax/noise/interval400   10 phases

so a line that slopes is the protocol's doing and not the method's budget.
Built by `finish_iclr.sh freq50` / `freq400` / `noise`; this script only reads
the `metrics_<agent>_values.json` each of those writes, exactly as
plot_metrics_overview.py does, so the points, the CIs and the marks are the
per-family figure's own.

ES/NES and PBT-PPO N=8/N=2 are ONE method each. The arm kept is the one the
PAPER'S family kept (`noise/10task`), not a per-interval pick: the appendix
has to discuss the same ES the main text does, and an interval that flipped
the pair would be comparing two different methods along one line.

`F` IS THE SAME MEASUREMENT AT EVERY INTERVAL, although at 400 no task is
revisited. With ten distinct tasks the tables report `final_forgetting`
(behavioural_divergence.py): the end-of-run agent on every earlier phase
against the agent just trained there, which needs no revisit. Only the
two-task families report `switch_forgetting` instead. An earlier version of
this figure shaded the 400 column as if F changed meaning there; it does not.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt                            # noqa: E402
from matplotlib.lines import Line2D                        # noqa: E402
from matplotlib.ticker import NullFormatter                # noqa: E402

REPO = pathlib.Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO / 'scripts'))
sys.path.insert(0, str(REPO / 'scripts' / 'analysis'))
import make_lineplot as lp                                 # noqa: E402
from plot_metrics_overview import keep_one_arm             # noqa: E402
sys.path.insert(0, str(REPO))
from source.metrics.continual_metrics import bootstrap_ci  # noqa: E402

PAPER = REPO / 'projects/iclr_2027/paper/gymnax'
OUT = REPO / 'projects/iclr_2027/paper/visuals/final/appendix'

# (interval, paper subdirectory). The 200 point IS the paper's noise family.
INTERVALS = [(50, 'noise/interval50'), (200, 'noise/10task'), (400, 'noise/interval400')]
MAIN_SUB = 'noise/10task'          # whose ES/PBT arm choice the appendix follows
MIN_CI_SEEDS = 3                   # fewer seeds: draw the seeds, not a CI

CELLS = [('CartPole_v1', 'CartPole'), ('Acrobot_v1', 'Acrobot'),
         ('MountainCar_v0', 'MountainCar')]


def load(agent):
    """`{interval: values JSON or None}`, and the arm remap to apply to all."""
    loaded = {}
    for interval, sub in INTERVALS:
        path = PAPER / sub / f'metrics_{agent}_values.json'
        loaded[interval] = json.loads(path.read_text()) if path.exists() else None
        print(f'  {interval:>4} gens  ' + (sub if loaded[interval]
                                           else f'MISSING {path.relative_to(REPO)}'))
    present = next((d for d in loaded.values() if d), None)
    remap, kept = keep_one_arm(MAIN_SUB, present['methods'] if present else [])
    print(f'  arms kept: {" and ".join(str(k) for k in kept)} (from {MAIN_SUB})')
    return loaded, remap


def series(loaded, remap, key, cell):
    """`{method: {interval: (mean, lo, hi, n)}}` for one metric and cell."""
    out = {}
    for interval, data in loaded.items():
        if not data:
            continue
        row = next((r for r in data['rows'] if r['key'] == key), None)
        if not row or cell not in row['values']:
            continue
        for arm, vals in row['values'][cell].items():
            method = remap.get(arm, arm)
            if method is None:                 # the dropped half of a pair
                continue
            v = np.asarray([x for x in vals if np.isfinite(x)])
            if v.size:
                out.setdefault(method, {})[interval] = (*bootstrap_ci(v), v.size, v)
    return out


def marker_of(method):
    """Circle for neuroevolution, square for gradient-based: the marker
    rule of Figure 2 (plot_continual_combined.py)."""
    return 'o' if lp.FAMILY.get(method) == 'ne' else 's'


def label_of(row_key, loaded):
    for data in loaded.values():
        if data:
            row = next((r for r in data['rows'] if r['key'] == row_key), None)
            if row:
                return row['label']
    return row_key


def draw(loaded, remap, metrics, args):
    """The figure. Returns one `(metric, cell, method, interval, mean, lo, hi,
    n)` a point drawn, for the markdown."""
    fs = args.font_size
    xs = [i for i, _ in INTERVALS]
    nrow, ncol = len(metrics), len(CELLS)
    fig, axes = plt.subplots(nrow, ncol, squeeze=False,
                             figsize=(args.width, args.row_height * nrow + 0.55))
    points, methods_drawn = [], []
    for r, key in enumerate(metrics):
        for c, (cell, cell_label) in enumerate(CELLS):
            ax = axes[r][c]
            data = series(loaded, remap, key, cell)
            for method, per_interval in data.items():
                style = lp.METHOD_STYLE.get(method, {})
                colour = style.get('color')
                pts = [(x, *per_interval[x]) for x in xs if x in per_interval]
                if not pts:
                    continue
                if method not in methods_drawn:
                    methods_drawn.append(method)
                x = [p[0] for p in pts]
                mean = [p[1] for p in pts]
                ax.plot(x, mean, '-', marker=marker_of(method), color=colour,
                        lw=0.9, ms=2.4, mec='white', mew=0.35, zorder=3)
                for xi, mi, lo, hi, n, vals in pts:
                    if n >= MIN_CI_SEEDS:
                        ax.plot([xi, xi], [lo, hi], color=colour, lw=0.8,
                                solid_capstyle='butt', alpha=0.85, zorder=2)
                    else:
                        # A bootstrap CI over one or two seeds only spans
                        # them and claims more than it knows: draw the
                        # seeds themselves.
                        ax.plot([xi] * len(vals), vals, 'o', ms=1.6, mfc='none',
                                mec=colour, mew=0.5, alpha=0.9, zorder=2)
                    points.append((key, cell_label, method, xi, mi, lo, hi, n))
            ax.set_xscale('log')
            ax.set_xticks(xs)
            ax.set_xticklabels([str(x) for x in xs])
            # A log axis labels its minor ticks too ("3 x 10^2"), which on
            # three hand-placed points reads as a fourth interval.
            ax.xaxis.set_minor_formatter(NullFormatter())
            ax.set_xlim(min(xs) / 1.35, max(xs) * 1.35)
            ax.tick_params(length=2, width=0.5)
            for side in ('top', 'right'):
                ax.spines[side].set_visible(False)
            if r == 0:
                ax.set_title(cell_label, pad=3, fontsize=fs + 0.5)
            if c == 0:
                ax.set_ylabel(label_of(key, loaded), fontsize=fs)
            if r == nrow - 1:
                ax.set_xlabel('generations per task', fontsize=fs)
    # The paper's one method order (make_lineplot.METHOD_ORDER), as in every
    # other figure; the data's own order is alphabetical.
    methods_drawn = ([m for m in lp.METHOD_ORDER if m in methods_drawn]
                     + [m for m in methods_drawn if m not in lp.METHOD_ORDER])
    handles = [Line2D([], [], color=lp.METHOD_STYLE.get(m, {}).get('color'),
                      marker=marker_of(m), ms=3.0, lw=0.9, mec='white', mew=0.35,
                      label=lp.METHOD_STYLE.get(m, {}).get('label', m))
               for m in methods_drawn]
    fig.legend(handles=handles, loc='upper center', ncol=min(len(handles), 8),
               frameon=False, fontsize=fs - 0.5,
               bbox_to_anchor=(0.5, 1.0), handlelength=1.4,
               columnspacing=1.1, handletextpad=0.4)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ('pdf', 'png'):
        fig.savefig(OUT / f'{args.stem}.{ext}', dpi=300)
    print(f'wrote {(OUT / args.stem).relative_to(REPO)}.pdf/.png')
    return points


def write_markdown(points, loaded, agent, stem):
    """Every point drawn, as the table the caption's numbers come from."""
    lines = [f'# Switch interval, {agent} agent', '',
             'Mean over seeds [95% percentile-bootstrap CI] (n seeds). '
             'One tree an interval, same budget and tasks, seeds matched '
             'across methods at each interval.', '']
    missing = [str(i) for i, d in loaded.items() if not d]
    if missing:
        lines += [f'**Not built yet: interval {", ".join(missing)}.**', '']
    by_metric = {}
    for key, cell, method, interval, mean, lo, hi, n in points:
        by_metric.setdefault((key, cell), {}).setdefault(method, {})[interval] = (mean, lo, hi, n)
    xs = [i for i, _ in INTERVALS]
    for (key, cell), per_method in by_metric.items():
        lines += [f'## {key} — {cell}', '',
                  '| method | ' + ' | '.join(f'{x} gens' for x in xs) + ' |',
                  '|---' * (len(xs) + 1) + '|']
        order = ([m for m in lp.METHOD_ORDER if m in per_method]
                 + [m for m in per_method if m not in lp.METHOD_ORDER])
        for method in order:
            per_interval = per_method[method]
            cells = []
            for x in xs:
                if x in per_interval:
                    mean, lo, hi, n = per_interval[x]
                    cells.append(f'{mean:.1f} [{lo:.1f}, {hi:.1f}] ({n})')
                else:
                    cells.append('—')
            label = lp.METHOD_STYLE.get(method, {}).get('label', method)
            lines.append(f'| {label} | ' + ' | '.join(cells) + ' |')
        lines.append('')
    path = OUT / f'{stem}.md'
    path.write_text('\n'.join(lines) + '\n')
    print(f'wrote {path.relative_to(REPO)}')


# THE STABILITY-PLASTICITY VERSION (`--plane`), the one in the paper. Forgetting
# alone misleads here: a method that learns little at some task length has
# little to forget, so F falls for the wrong reason. The LA-F plane of Figure 2
# reads both at once. Every piece is Figure 2's own (plot_stability_plasticity):
# the loader, the rescaling, the panel with its RL-dominated region, the
# equal-LA-F lines and the tested ring. What differs is the grid -- one row a
# task length -- and the scale: LA and F are rescaled per environment on ONE
# scale shared by the three lengths (the best LA of any method at ANY length),
# because per-length rescaling, as Figure 2 does per family, would erase
# exactly the between-length differences this figure is for.
PLANE_TREES = {   # task length: (paper directory of its forgetting pass, run tree)
    50: ('gymnax/noise/interval50', 'runs_freq_matched/interval50/gymnax'),
    200: ('gymnax/noise/10task', 'paper/gymnax/data/noise_10task'),
    400: ('gymnax/noise/interval400', 'runs_freq_matched/interval400/gymnax'),
}


def draw_plane(stem):
    import os
    import plot_stability_plasticity as psp
    os.chdir(REPO)          # load_divergence finds runs by repo-relative path
    ref_tree = PLANE_TREES[200][1]
    cells = [c for c in psp.pcl._cells(lambda f: f == 'main')[ref_tree]]
    # The paper family's arms (ES = NES, its PBT pick), used at every length.
    arms = psp.reported_arms(psp.PROJECT / ref_tree, cells, es_kept='nes')
    print(f'plane: arms {" ".join(arms)}')
    envs = [c.split('_sigma')[0] for c in cells]
    merged = {env: {} for env in envs}           # {env: {(method, length): arrays}}
    for length, (sub, tree) in PLANE_TREES.items():
        loaded = psp.load_pass(sub, tree, cells, arms)
        for cell, env in zip(cells, envs):
            for m, trials in psp.as_arrays(loaded.get(cell, {})).items():
                merged[env][m, length] = trials
    scaled_all, _ = psp.normalise(merged, 'F')   # one scale a env, all lengths
    panels = []
    for row, length in enumerate(PLANE_TREES):
        for col, env in enumerate(envs):
            scaled = {env: {m: v for (m, l), v in scaled_all[env].items() if l == length}}
            points = psp.points_for(scaled, [env], 'F')
            title = f'{lp.ENV_TITLES.get(env, env)}, {length} gens'
            panels.append((row, col, title, points))
            s = psp.summary(points)
            n = {m: p['per_env']['x'][0].size for m, p in points.items()}
            print(f'  {title:28s} n={min(n.values()) if n else 0}-{max(n.values()) if n else 0} '
                  f'ringed {s["ringed"] if s else None}')
    methods = [m for m in lp.METHOD_ORDER if m in arms]
    plt.rcParams.update({'font.size': 9})            # Figure 2's own
    psp.draw(panels, [], (len(PLANE_TREES), len(envs)), str(OUT / stem), 'F', methods,
             axes_in=(1.55, 1.2), legend_rows=2)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--agent', default='centroid', choices=['centroid', 'elite'])
    ap.add_argument('--metrics', nargs='+', default=['cum', 'F'],
                    help='metric keys, as in metrics_<agent>_values.json '
                         '(cum, ft, F, BD, ZT)')
    ap.add_argument('--width', type=float, default=5.5, help='\\linewidth, in')
    ap.add_argument('--row-height', type=float, default=1.15)
    ap.add_argument('--font-size', type=float, default=7.0)
    ap.add_argument('--stem', default=None)
    ap.add_argument('--plane', action='store_true',
                    help='the LA-F stability-plasticity grid (the paper figure), '
                         'written to frequency_plane_<agent>')
    args = ap.parse_args()
    if args.plane:
        lp.METHOD_STYLE['pbt'] = {**lp.METHOD_STYLE['pbt'], 'label': 'PBT-PPO'}
        draw_plane(args.stem or f'frequency_plane_{args.agent}')
        return 0
    args.stem = args.stem or f'frequency_{args.agent}'
    plt.rcParams.update({
        'font.size': args.font_size, 'axes.titlesize': args.font_size + 0.5,
        'xtick.labelsize': args.font_size - 1, 'ytick.labelsize': args.font_size - 0.5,
        'axes.labelsize': args.font_size,
        'axes.linewidth': 0.5, 'xtick.major.width': 0.5, 'xtick.major.size': 2,
        'ytick.major.width': 0.5, 'ytick.major.size': 2,
    })
    # The paper's name for the PBT arm, as every other final figure sets it.
    lp.METHOD_STYLE['pbt'] = {**lp.METHOD_STYLE['pbt'], 'label': 'PBT-PPO'}
    print(f'switch-interval figure, {args.agent} agent')
    loaded, remap = load(args.agent)
    if not any(loaded.values()):
        sys.exit('no family built yet: run finish_iclr.sh noise / freq50 / freq400')
    points = draw(loaded, remap, args.metrics, args)
    write_markdown(points, loaded, args.agent, args.stem)
    return 0


if __name__ == '__main__':
    sys.exit(main())
