"""Population diversity of the population methods (GA, ES, PBT-PPO) across the
ten continual tasks of the main text: an appendix figure.

    .venv/bin/python scripts/analysis/plot_population_diversity.py             # from the saved data
    .venv/bin/python scripts/analysis/plot_population_diversity.py --extract   # re-read the runs first
    .venv/bin/python scripts/analysis/plot_population_diversity.py --main      # the one-row main-text figure

    -> paper/visuals/final/appendix/population_diversity_continual.{pdf,png}
       paper/visuals/final/population_diversity_main.{pdf,png}   --main: MAIN_ROW only
       paper/visuals/final/appendix/population_diversity_continual.md   the runs and trial counts
       paper/visuals/final/data/population_diversity_continual.{npz,json}
    (paper = projects/iclr_2027/paper)

Built in two steps, like the other final figures: `--extract` reads the runs
through the symlink tree paper/diversity/data/population/<family>/continual/
<arm>/<cell> (README.md there says where every link points and why);
without it only the saved data is read. The tasks are continual_main's and so
are the runs, except where those runs log no diversity: the MiniGrid GA/NES
and the HalfCheetah NES/GA runs are the diversity re-runs
(submit_population_diversity.sh, queue_cheetah_div_nes.sh). ES = NES except on
Kinetix (plain OpenES, `es`), PBT-PPO = the N continual_main keeps. HalfCheetah
action reversal has no PBT-PPO (continual_main has none).

One row a task, one column a quantity; each panel is the mean over trials with
a 95% bootstrap band, against sub-task (training progress scaled so every run
spans its twenty phases). Columns, all from the per-generation training
records (`source/metrics/population_diversity.py`):

    Fitness s.d.        spread of the members' training fitness
                        (`bd_fitness_std`), in the task's own units.
    Behavioural         mean pairwise argmax disagreement of the members on a
    diversity           batch of probe states (`bd_behavioural_diversity`, or
                        `bd_probe_disagreement` in the gymnax NE trainers --
                        the same statistic on that trainer's probe batch). On
                        HalfCheetah it is the mean per-actuator action
                        distance instead, in [0, 2].
    Genomic diversity   mean pairwise L2 between members over sqrt(2 P), P the
    (per weight)        parameter count: the members' standard deviation per
                        weight, on the same scale as the weights themselves.

A column a run does not log is n/a: Kinetix GA/ES log no behavioural column
(the Kinetix NE trace keeps step features, not frames, so there is no probe
batch to disagree on). An ES population is the centroid plus sigma-noise, so
its genomic diversity is sigma by construction. The weight RMS of the saved
agent is the plasticity figure's weight row; it used to be a column here, with
the per-weight s.d. over it (dropped 2026-09-19: both are the genomic column
against a quantity the figure is not about).
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
from matplotlib.gridspec import GridSpec                   # noqa: E402
from matplotlib.lines import Line2D                        # noqa: E402
from matplotlib.ticker import (FuncFormatter, LogLocator, MaxNLocator,  # noqa: E402
                               NullLocator)

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import plot_continual_lineplots as pcl                     # noqa: E402
from plot_metrics_overview import _compact                 # noqa: E402
from source.metrics.continual_metrics import bootstrap_ci  # noqa: E402

lp, PROJECT = pcl.lp, pcl.PROJECT
STEM = 'population_diversity_continual'
OUT = pcl.FINAL / 'appendix'
DATA = pcl.FINAL / 'data' / STEM
LINKS = PROJECT / 'paper/diversity/data/population'
# (family under LINKS, cell, title), top to bottom: continual_main's panels in
# the paper's column order.
PANELS = [
    ('noise_10task', 'CartPole_v1_sigma1.0', 'CartPole, noise'),
    ('noise_10task', 'Acrobot_v1_sigma1.0', 'Acrobot, noise'),
    ('noise_10task', 'MountainCar_v0_sigma0.1', 'MountainCar, noise'),
    ('actions_2task', 'CartPole_v1_sigma1.0', 'CartPole, action reversal'),
    ('actions_2task', 'Acrobot_v1_sigma1.0', 'Acrobot, action reversal'),
    ('actions_2task', 'MountainCar_v0_sigma1.0', 'MountainCar, action reversal'),
    # DeepSea 12, the action-map family (fig:novelty's collapse cell, not one
    # of continual_main's): probe_deepsea GA/ES and the PBT-PPO run of
    # 2026-09-20 (scripts/train/queue_iclr_deepsea_pbt.sh).
    ('deepsea', 'DeepSea12_bsuite_sigma1.0', 'DeepSea 12, action map'),
    ('cheetah_noise_10task', 'cheetah_noise', 'HalfCheetah, noise'),
    ('cheetah_actions_2task', 'cheetah_action', 'HalfCheetah, action reversal'),
    ('minigrid', 'MiniGrid_8x8_16x16', 'MiniGrid, 8x8 / 16x16'),
    ('kinetix', 'Kinetix20', 'Kinetix, 20 levels'),
]
# The row the main text shows (--main); the appendix figure has them all.
# CartPole under action reversal: the GA is the most plastic and the most
# diverse, ES the least diverse with the least forgetting, PBT-PPO between
# them on both counts (2026-09-19).
MAIN_ROW = ('actions_2task', 'CartPole_v1_sigma1.0', 'CartPole, action reversal')
# Link name -> the method it is drawn as: one ES and one PBT-PPO entry.
ARM_METHOD = {'ga': 'ga', 'nes': 'es', 'es': 'es', 'pbt': 'pbt', 'pbt2': 'pbt'}
METHODS = ('ga', 'es', 'pbt')

# key: (column title, record columns in preference order, shared y down the column, log y)
COLUMNS = {
    'fitness_std': ('Fitness s.d.', ('bd_fitness_std',), False, True),
    'behaviour':   ('Behavioural\ndiversity',
                    ('bd_behavioural_diversity', 'bd_probe_disagreement'), True, False),
    'genomic':     ('Genomic diversity\n(per weight)', ('bd_genomic_diversity',), True, True),
}
BINS = 200
PHASES = 20
MUTED = '#6b6a65'
LEFT_IN, TOP_IN, BOTTOM_IN = 1.12, 0.55, 0.3
TEXT_WIDTH_IN = 5.5     # ICLR


def trial_curves(trial_dir, columns):
    """`{column: binned record curve}` and P, the parameter count."""
    records = json.loads((trial_dir / 'training_metrics.json').read_text())
    with np.load(trial_dir / 'checkpoints.npz', allow_pickle=True) as ck:
        key = next(k for k in ('centroid', 'final', 'finalgen') if k in ck.files)
        num_params = int(ck[key].shape[-1])
    # At most one bin a record, so a short run (60 PBT records on the
    # cheetah) is not mostly empty bins.
    bins = min(BINS, len(records))
    x = (np.arange(len(records)) + 0.5) / len(records)
    idx = np.minimum((x * bins).astype(int), bins - 1)
    out = {}
    for col in columns:
        v = np.array([np.nan if r.get(col) is None else float(r[col]) for r in records])
        if not np.isfinite(v).any():
            continue
        sums = np.bincount(idx, np.nan_to_num(v), bins)
        counts = np.bincount(idx, np.isfinite(v), bins)
        with np.errstate(invalid='ignore', divide='ignore'):
            out[col] = sums / counts
    return out, num_params


def load_trial(trial_dir):
    """`{column key: curve}` for one finished trial."""
    curves, P = trial_curves(trial_dir, {c for spec in COLUMNS.values() for c in spec[1]})
    out = {}
    for key, (_, columns, _, _) in COLUMNS.items():
        got = next((curves[c] for c in columns if c in curves), None)
        if got is not None:
            out[key] = got / np.sqrt(2 * P) if key == 'genomic' else got
    return out


def extract():
    """Write DATA.npz (`<family>|<cell>|<method>|<column>`: trials x points,
    cut to the shortest trial) and DATA.json (the run each link resolves to and
    the trials read)."""
    arrays, meta = {}, {'links': {}, 'trials': {}}
    for family, cell, _title in PANELS:
        for link in sorted((LINKS / family / 'continual').glob(f'*/{cell}')):
            arm = link.parent.name
            method = ARM_METHOD[arm]
            key = f'{family}|{cell}|{method}'
            meta['links'][f'{family}/continual/{arm}/{cell}'] = str(
                pathlib.Path(link.resolve()).relative_to(PROJECT))
            trials = sorted(t for t in link.glob('trial_*')
                            if (t / 'training_metrics.json').exists()
                            and (t / 'checkpoints.npz').exists())
            meta['trials'][key] = [t.name for t in trials]
            if not trials:
                print(f'WARNING: {key}: no finished trial under {link}')
                continue
            per = [load_trial(t) for t in trials]
            for col in COLUMNS:
                got = [p[col] for p in per if col in p]
                if got:
                    n = min(len(g) for g in got)
                    arrays[f'{key}|{col}'] = np.stack([g[:n] for g in got]).astype(np.float32)
            print(f'{key:55s} {len(trials)} trials, '
                  + ', '.join(c for c in COLUMNS if f'{key}|{c}' in arrays))
    meta['extracted'] = datetime.date.today().isoformat()
    DATA.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(f'{DATA}.npz', **arrays)
    pathlib.Path(f'{DATA}.json').write_text(json.dumps(meta, indent=1) + '\n')
    print(f'wrote {DATA}.npz, {DATA}.json')


def _tick(x, pos):
    """`_compact`, with tiny values as 4e-4 so they fit."""
    if x and abs(x) < 1e-2:
        mant, exp = f'{x:.0e}'.split('e')
        return f'{mant}e{int(exp)}'
    return _compact(x, pos)


def panel(ax, series, log):
    """One (task, column) panel from `{method: trials x points}`."""
    drawn = set()
    for m in METHODS:
        arr = series.get(m)
        if arr is None:
            continue
        arr = arr.astype(float)
        if log:
            arr = np.where(arr > 0, arr, np.nan)
        ok = np.isfinite(arr).any(axis=0)
        if not ok.any():
            continue
        n = arr.shape[1]
        mean, lo, hi = (np.full(n, np.nan) for _ in range(3))
        for t in np.flatnonzero(ok):
            v = arr[:, t][np.isfinite(arr[:, t])]
            mean[t], lo[t], hi[t] = bootstrap_ci(v)
        # Records span phases 0..20.
        x = (np.arange(n) + 0.5) / n * PHASES
        colour = lp.METHOD_STYLE[m]['color']
        ax.plot(x, mean, color=colour, lw=0.8)
        ax.fill_between(x, lo, hi, color=colour, alpha=0.18, lw=0)
        drawn.add(m)
    ax.grid(True, color='0.92', lw=0.5)
    ax.spines[['top', 'right']].set_visible(False)
    ax.tick_params(length=2, pad=1.5)
    ax.set_xticks([1, 10, 20])
    if log and drawn:
        ax.set_yscale('log')
        # Decades only: minor ticks read as a black bar at this size.
        ax.yaxis.set_major_locator(LogLocator(numticks=3))
        ax.yaxis.set_minor_locator(NullLocator())
    elif drawn:
        ax.yaxis.set_major_locator(MaxNLocator(nbins=3))
        ax.yaxis.set_major_formatter(FuncFormatter(_tick))
        ax.set_ylim(bottom=0)
    return drawn


def draw(data, args, panels=PANELS, out=OUT / STEM):
    fs = args.font_size
    width = args.width
    height = args.row_height * len(panels) + TOP_IN + BOTTOM_IN
    fig = plt.figure(figsize=(width, height))
    gs = GridSpec(len(panels), len(COLUMNS), figure=fig,
                  left=LEFT_IN / width, right=1 - 0.08 / width,
                  top=1 - TOP_IN / height, bottom=BOTTOM_IN / height,
                  wspace=0.3, hspace=0.3)
    drawn, filled = set(), {c: [] for c in COLUMNS}
    for r, (family, cell, title) in enumerate(panels):
        for c, (col, (label, _, _, log)) in enumerate(COLUMNS.items()):
            ax = fig.add_subplot(gs[r, c])
            series = {m: data[f'{family}|{cell}|{m}|{col}'] for m in METHODS
                      if f'{family}|{cell}|{m}|{col}' in data}
            got = panel(ax, series, log)
            drawn |= got
            if got:
                filled[col].append(ax)
            else:
                ax.set_yticks([])
                ax.text(0.5, 0.5, 'n/a', transform=ax.transAxes, ha='center',
                        va='center', color=MUTED, fontsize=fs - 1)
            if r < len(panels) - 1:
                ax.set_xticklabels([])
            else:
                ax.set_xlabel('Task', labelpad=1)
            if r == 0:
                ax.set_title(label, pad=3)
            if c == 0:
                # Horizontal: rotated, a two-line name is longer than a row.
                ax.annotate(title.replace(', action reversal', ', reversal').replace(', ', ',\n'),
                            xy=(0.04 / width, 0.5),
                            xycoords=('figure fraction', 'axes fraction'),
                            ha='left', va='center', linespacing=1.1)
    for col, (_, _, share, _) in COLUMNS.items():
        axes = filled[col]
        if share and axes:
            lo = min(ax.get_ylim()[0] for ax in axes)
            hi = max(ax.get_ylim()[1] for ax in axes)
            for ax in axes:
                ax.set_ylim(lo, hi)
    order = [m for m in METHODS if m in drawn]
    fig.legend([Line2D([], [], color=lp.METHOD_STYLE[m]['color'], lw=1.6) for m in order],
               [lp.METHOD_STYLE[m]['label'] for m in order], loc='upper center',
               ncol=len(order), frameon=False, fontsize=fs, bbox_to_anchor=(0.5, 1.0),
               handlelength=1.6, columnspacing=1.2)
    out.parent.mkdir(parents=True, exist_ok=True)
    for ext in ('pdf', 'png'):
        fig.savefig(f'{out}.{ext}', dpi=300)
    plt.close(fig)
    print(f'wrote {out}.pdf, .png ({width:.2f} x {height:.2f} in)')


def write_markdown(meta):
    label = lambda m: lp.METHOD_STYLE[m]['label']  # noqa: E731
    md = [f'# {STEM}', '',
          'Population diversity of GA, ES and PBT-PPO on the ten continual tasks of the '
          'main text plus DeepSea 12. Built by `scripts/analysis/plot_population_diversity.py` from '
          f'`data/{STEM}.{{npz,json}}` (extracted {meta["extracted"]}); its docstring '
          'defines each column. The runs are linked in '
          '`paper/diversity/data/population/` (README there).', '',
          '| Task | ' + ' | '.join(label(m) for m in METHODS) + ' |',
          '|---|' + '---|' * len(METHODS)]
    for family, cell, title in PANELS:
        md.append(f'| {title} | ' + ' | '.join(
            str(len(meta['trials'].get(f'{family}|{cell}|{m}', []))) or '--'
            for m in METHODS) + ' |')
    md += ['', 'Trials read per task and method (0: not run or not finished).', '',
           '| Link | Resolves to |', '|---|---|']
    md += [f'| `{k}` | `{v}` |' for k, v in meta['links'].items()]
    (OUT / f'{STEM}.md').write_text('\n'.join(md) + '\n')
    print(f'wrote {OUT / STEM}.md')


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--extract', action='store_true',
                    help='re-read the runs into the saved data first')
    ap.add_argument('--main', action='store_true',
                    help='draw MAIN_ROW alone into paper/visuals/final (main text)')
    ap.add_argument('--width', type=float, default=TEXT_WIDTH_IN)
    ap.add_argument('--row-height', type=float, default=0.62)
    ap.add_argument('--font-size', type=float, default=7.0)   # print size, at \linewidth
    args = ap.parse_args()
    if args.extract:
        extract()
    if not pathlib.Path(f'{DATA}.npz').exists():
        sys.exit(f'no {DATA}.npz: run with --extract first')
    meta = json.loads(pathlib.Path(f'{DATA}.json').read_text())
    lp.METHOD_STYLE['pbt'] = {**lp.METHOD_STYLE['pbt'], 'label': 'PBT-PPO'}
    plt.rcParams.update({
        'font.size': args.font_size, 'axes.titlesize': args.font_size + 0.5,
        'xtick.labelsize': args.font_size - 1, 'ytick.labelsize': args.font_size - 1,
        'axes.linewidth': 0.5, 'xtick.major.width': 0.5, 'xtick.major.size': 2,
    })
    with np.load(f'{DATA}.npz') as npz:
        data = {k: npz[k] for k in npz.files}
    if args.main:
        draw(data, args, [MAIN_ROW], pcl.FINAL / 'population_diversity_main')
    else:
        draw(data, args)
        write_markdown(meta)
    return 0


if __name__ == '__main__':
    sys.exit(main())
