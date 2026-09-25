"""Appendix: the effect of hyperparameters, the UPDATE-COUNT half. Figures:

  hparam_sweep        gymnax observation noise: PPO epochs / minibatches /
                      rollout length / learning rate and ES population /
                      learning rate against the cumulative return, one row a
                      task, one column an axis
  hparam_minibatches  PPO's minibatch count (the number of gradient steps per
                      sub-task with the same data and the same K passes) on
                      EVERY family it was run on: gymnax noise, gymnax action
                      reversal, MiniGrid 8x8/16x16, HalfCheetah noise and
                      action reversal; one panel a cell
  hparam_lr           PPO's learning rate, the same way, on the families that
                      ran it (gymnax noise, HalfCheetah)

    .venv/bin/python scripts/analysis/plot_hparam_updates.py             # from the saved data
    .venv/bin/python scripts/analysis/plot_hparam_updates.py --extract   # re-read the runs first

    -> paper/visuals/final/appendix/hparam_{sweep,minibatches,lr}{,_final}.{pdf,png}
       paper/visuals/final/appendix/hparam_sweep.md                    every point of all
       paper/visuals/final/data/hparam_sweep.json                      one value a trial
    (paper = projects/iclr_2027/paper)
    (plot_hparam_sigma.py draws the OTHER hyperparameter figure, hparam_sigma:
     the GA / ES search width and PPO lr / entropy on the stability-plasticity
     plane, from runs_hparam/gymnax/. The two do not share a run or a file.)

The runs are scripts/train/queue_iclr_hparam_updates.sh's (HalfCheetah:
scripts/train/cluster/submit_cheetah_minibatches.sh, shipped home): one tree a
setting under projects/iclr_2027/runs_hparam/<setting>/<suite>, each changing
ONE value of the reported configuration at the reported budget. The reported
setting of each family is a symlink tree `runs_hparam/reported[_<family>]`
onto trials 1-5 of the reported runs (the same seeds), built by `--extract`.
A setting with no finished trial yet is left out of the figure and listed as
pending in the markdown, so the figures can be drawn while the sweeps run.

Cum. is the area under the centroid training curve / 1000 on the clock of the
family's REPORTED generation-equivalents (FAMILIES[...]['gen_steps'] env
steps a generation), whatever the setting's own update or generation count,
so every point in a panel integrates the same wall of environment steps.
`final` is the mean over phases of the last 10% of each phase
(plot_continual_lineplots.TAIL), per trial. Forgetting is not read: it needs
the post-hoc evaluation pass.
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

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent.parent
for p in (REPO, REPO / 'scripts', HERE):
    sys.path.insert(0, str(p))
import make_lineplot as lp                                 # noqa: E402
import plot_continual_lineplots as pcl                     # noqa: E402
from make_metrics_figure import bootstrap_ci               # noqa: E402
from source.metrics.continual_metrics import cumulative_reward, window_mean  # noqa: E402

PROJECT = REPO / 'projects' / 'iclr_2027'
RUNS = PROJECT / 'runs_hparam'
REPORTED_TRIALS = 5
OUT = pcl.FINAL / 'appendix'
STEM = 'hparam_sweep'
DATA = pcl.FINAL / 'data' / f'{STEM}.json'

TEXT_WIDTH_IN = 5.5
MUTED = '#6b6a65'
FAMILY_TITLE = {'ppo': 'PPO', 'nes': 'ES'}
METRICS = {'cum': 'Cum. centroid', 'final': 'End-of-sub-task return',
           'la': 'Learning accuracy LA', 'F': 'Forgetting F'}
# The metrics each figure is drawn for. LA and F exist only where the
# forgetting pass ran (PPO, the cross-family trees), so the gymnax-noise
# figure, which has ES columns, keeps to the two curve metrics.
SWEEP_METRICS = ('cum', 'final')
CROSS_METRICS = ('cum', 'final', 'la', 'F')

# The cross-family sweeps: figure stem, x label, tick format.
SWEEPS = {
    'minibatches': dict(stem='hparam_minibatches', xlabel='minibatches $M$', fmt='int'),
    'lr': dict(stem='hparam_lr', xlabel=r'learning rate $\alpha$', fmt='mult'),
}


def _mb(prefix, Ms, reported):
    return [(f'{prefix}{M}', M) for M in Ms] + [(None, reported)]


def _lr(prefix):
    return [(f'{prefix}0.1x', 0.1), (None, 1), (f'{prefix}10x', 10)]


# The families. `reported` is the tree the reported runs live in (trials 1-5
# are linked as runs_hparam/<link>/<suite>), `gen_steps` the env steps of one
# reported generation there (the Cum. clock: pop x evals x episode length, or
# for the RL-only trees the same number as U x N x L / G), `sweeps` the
# settings of each cross-family axis as (setting directory, x), with None
# standing for the reported point.
FAMILIES = {
    'noise': dict(
        suite='gymnax', reported=PROJECT / 'runs_centroid' / 'gymnax',
        cells=['CartPole_v1_sigma1.0', 'Acrobot_v1_sigma1.0', 'MountainCar_v0_sigma0.1'],
        titles={'CartPole_v1_sigma1.0': 'CartPole, noise',
                'Acrobot_v1_sigma1.0': 'Acrobot, noise',
                'MountainCar_v0_sigma0.1': 'MountainCar, noise'},
        gen_steps=512 * 3 * 500, link='reported', arms=('ppo', 'nes'),
        sweeps={'minibatches': _mb('ppo_minibatches', (4, 8, 128), 32),
                'lr': _lr('ppo_lr')}),
    'actions': dict(
        suite='gymnax', reported=PROJECT / 'runs_actions' / 'gymnax',
        cells=['CartPole_v1_sigma1.0', 'Acrobot_v1_sigma1.0', 'MountainCar_v0_sigma1.0'],
        titles={'CartPole_v1_sigma1.0': 'CartPole, action reversal',
                'Acrobot_v1_sigma1.0': 'Acrobot, action reversal',
                'MountainCar_v0_sigma1.0': 'MountainCar, action reversal'},
        gen_steps=512 * 3 * 500, link='reported_actions', arms=('ppo',),
        sweeps={'minibatches': _mb('actions_ppo_minibatches', (4, 8, 128), 32)}),
    'minigrid': dict(
        suite='minigrid', reported=PROJECT / 'runs_centroid' / 'minigrid',
        cells=['MiniGrid_8x8_16x16'],
        titles={'MiniGrid_8x8_16x16': 'MiniGrid 8x8 / 16x16'},
        gen_steps=2048 * 50 * 3072 // 200, link='reported_minigrid', arms=('ppo',),
        sweeps={'minibatches': _mb('minigrid_ppo_minibatches', (2, 4, 64), 16)}),
    'cheetah_noise': dict(
        suite='mjx',
        reported=PROJECT / 'cluster_2026-09-16' / 'runs_mjx_noise05_t10_antshape' / 'mjx',
        cells=['cheetah_noise'], titles={'cheetah_noise': 'HalfCheetah, noise'},
        gen_steps=512 * 20 * 2400 // 16, link='reported_cheetah_noise', arms=('ppo',),
        sweeps={'minibatches': _mb('cheetah_noise_ppo_minibatches', (4, 8, 128), 32),
                'lr': _lr('cheetah_noise_ppo_lr')}),
    'cheetah_actions': dict(
        suite='mjx',
        reported=PROJECT / 'cluster_2026-09-16' / 'runs_mjx_action_antshape' / 'mjx',
        cells=['cheetah_action'], titles={'cheetah_action': 'HalfCheetah, action reversal'},
        gen_steps=512 * 20 * 2400 // 16, link='reported_cheetah_actions', arms=('ppo',),
        sweeps={'minibatches': _mb('cheetah_actions_ppo_minibatches', (4, 8, 128), 32),
                'lr': _lr('cheetah_actions_ppo_lr')}),
}

# The gymnax-noise figure: (arm, key, panel title, [(setting, x)], tick format).
# `reported` is the noise family's link tree; a multiplier axis's x is the
# multiplier of the reported value.
# ES columns first: the paper's one method order (make_lineplot.METHOD_ORDER,
# NE before RL, as Figure 2) since 2026-09-21.
AXES = [
    ('nes', 'pop', 'population $P$',
     [('nes_pop128', 128), ('reported', 512), ('nes_pop2048', 2048)], 'int'),
    ('nes', 'lr', r'learning rate $\alpha$',
     [('nes_lr0.25x', 0.25), ('reported', 1), ('nes_lr4x', 4)], 'mult'),

    ('ppo', 'epochs', 'epochs $K$',
     [('ppo_epochs1', 1), ('ppo_epochs3', 3), ('reported', 10), ('ppo_epochs30', 30)], 'int'),
    ('ppo', 'minibatches', 'minibatches $M$',
     [('ppo_minibatches4', 4), ('ppo_minibatches8', 8), ('reported', 32),
      ('ppo_minibatches128', 128)], 'int'),
    ('ppo', 'rollout', 'rollout length $L$',
     [('ppo_rollout10', 10), ('reported', 50), ('ppo_rollout250', 250)], 'int'),
    ('ppo', 'lr', r'learning rate $\alpha$',
     [('ppo_lr0.1x', 0.1), ('reported', 1), ('ppo_lr10x', 10)], 'mult'),
]


# ----------------------------------------------------------------------------
# extract
# ----------------------------------------------------------------------------

def ensure_reported_tree(fam):
    """runs_hparam/<link>/<suite>: trials 1-5 of the family's reported runs."""
    for arm in fam['arms']:
        for cell in fam['cells']:
            src_cell = fam['reported'] / 'continual' / arm / cell
            dst_cell = RUNS / fam['link'] / fam['suite'] / 'continual' / arm / cell
            dst_cell.mkdir(parents=True, exist_ok=True)
            for k in range(1, REPORTED_TRIALS + 1):
                src, dst = src_cell / f'trial_{k}', dst_cell / f'trial_{k}'
                if not (src / 'training_metrics.json').exists():
                    print(f'WARNING: reported {arm}/{cell}/trial_{k} has no results under '
                          f'{fam["reported"]}')
                    continue
                if dst.is_symlink() or dst.exists():
                    continue
                os.symlink(os.path.relpath(src, dst.parent), dst)


def finished_cells(root, arm, cells):
    return [c for c in cells
            if any((root / 'continual' / arm / c).glob('trial_*/training_metrics.json'))]


def per_trial_metrics(rep, env, gen_steps):
    """`{'cum': [...], 'final': [...]}` for one env of a report, per trial."""
    x, curves = rep.data[env][next(iter(rep.data[env]))]
    xg = np.asarray(x, dtype=float) / gen_steps
    edges = np.asarray(rep.edges, dtype=float) / gen_steps
    cum, final = [], []
    for c in np.asarray(curves, dtype=float):
        cum.append(cumulative_reward(xg, c, xg[-1]) / 1e3)
        per_phase = []
        for a, b in zip(edges[:-1], edges[1:]):
            v = window_mean(xg, c, b - pcl.TAIL * (b - a), b + 1e-9)
            if not np.isfinite(v):             # a phase shorter than one record
                v = float(c[xg <= b][-1])
            per_phase.append(v)
        final.append(float(np.mean(per_phase)))
    return {'cum': cum, 'final': final}


def load_plane(results_dir, cells):
    """`{cell: {'la': [...], 'F': [...]}}` for the CENTROID from one forgetting
    pass (finish_hparam_updates_diverge.sh): LA = mean_i R[i][i] of the reward
    matrix, F from the pass's summary. {} while the pass has not run."""
    results_dir = pathlib.Path(results_dir)
    npz = results_dir / 'behavioural_divergence.npz'
    if not npz.exists():
        return {}
    divergence = lp.load_divergence(results_dir)
    out = {}
    with np.load(npz) as archive:
        for i, raw in enumerate(archive['index']):
            rec = json.loads(str(raw))
            run_dir = str(pathlib.Path(rec['run_dir']))
            cell = run_dir.rsplit('/', 2)[1]
            if cell not in cells or rec.get('source') not in lp.ZT_SOURCES['centroid']:
                continue
            la = float(np.mean(np.diag(archive[f'run{i}_reward'])))
            F = divergence.get(run_dir, {}).get('F')
            if np.isfinite(la) and F is not None:
                out.setdefault(cell, {'la': [], 'F': []})
                out[cell]['la'].append(la)
                out[cell]['F'].append(float(F))
    return out


def load_point(fam, setting, arm):
    """`{cell: {'cum', 'final', 'la', 'F'}}` for one setting tree of a family,
    finished cells only; {} while nothing has finished. LA and F come from
    the forgetting pass and are absent until it has run on that tree."""
    root = RUNS / setting / fam['suite']
    cells = finished_cells(root, arm, fam['cells']) if root.is_dir() else []
    if not cells:
        return {}
    args = lp.parse_args([str(root), '--phase', 'continual', '--cells', *cells,
                          '--metric', 'centroid', '--methods', arm, '--out', '-'])
    # `load_report` exits the process when a tree holds runs none of whose
    # columns it can resolve -- a shipped-but-unmigrated shared-runner tree
    # (scripts/analysis/migrate_shared_runner_columns.py). That is a reason to
    # leave one setting out of the figure, not to take the whole figure down,
    # since a sweep is read while trees are still arriving.
    try:
        rep = lp.load_report(args)
    except SystemExit as exc:
        print(f'    ! {root}: {exc}; treated as pending '
              '(un-migrated shared-runner tree?)')
        return {}
    # The loader keys on the env name without its sigma tag; map back to cells.
    out = {}
    for cell in cells:
        env = cell.split('_sigma')[0]
        if env in rep.data:
            out[cell] = per_trial_metrics(rep, env, fam['gen_steps'])
    if arm == 'ppo':
        plane = load_plane(PROJECT / 'paper' / fam['suite'] / 'hparam_updates' / 'results'
                           / setting / 'centroid', fam['cells'])
        for cell, v in plane.items():
            out.setdefault(cell, {}).update(v)
    return out


def extract():
    for fam in FAMILIES.values():
        ensure_reported_tree(fam)
    out = {'extracted': datetime.date.today().isoformat(), 'tail': pcl.TAIL,
           'families': {k: {'cells': f['cells'], 'titles': f['titles'],
                            'gen_steps': f['gen_steps']}
                        for k, f in FAMILIES.items()},
           'axes': [], 'sweeps': {k: [] for k in SWEEPS}}
    noise = FAMILIES['noise']
    for arm, key, title, points, fmt in AXES:
        axis = {'arm': arm, 'key': key, 'title': title, 'fmt': fmt, 'points': []}
        for setting, xval in points:
            envs = load_point(noise, setting, arm)
            point = {'setting': setting, 'x': xval,
                     'run': f'projects/iclr_2027/runs_hparam/{setting}/gymnax',
                     'envs': {c.split('_sigma')[0]: v for c, v in envs.items()}}
            print(f'{arm:4s} {key:7s} {setting:16s} '
                  + (', '.join(f'{e}: n={len(v["cum"])}' for e, v in point['envs'].items())
                     or 'pending'))
            axis['points'].append(point)
        out['axes'].append(axis)
    for name, fam in FAMILIES.items():
        for key, settings in fam['sweeps'].items():
            entry = {'family': name, 'points': []}
            for setting, xval in sorted(settings, key=lambda s: s[1]):
                reported = setting is None
                setting = fam['link'] if reported else setting
                cells = load_point(fam, setting, 'ppo')
                entry['points'].append({
                    'setting': setting, 'x': xval, 'reported': reported,
                    'run': f'projects/iclr_2027/runs_hparam/{setting}/{fam["suite"]}',
                    'cells': cells})
                print(f'{name:16s} {key:12s} x={xval:<6g} {setting:34s} '
                      + (', '.join(f'{c}: n={len(v["cum"])}' for c, v in cells.items())
                         or 'pending'))
            out['sweeps'][key].append(entry)
    DATA.parent.mkdir(parents=True, exist_ok=True)
    DATA.write_text(json.dumps(out, indent=1))
    print(f'wrote {DATA}')


# ----------------------------------------------------------------------------
# draw
# ----------------------------------------------------------------------------

def tick_label(x, fmt):
    return f'×{x:g}' if fmt == 'mult' else f'{x:g}'


def draw_points(ax, points, colour, fmt, table_row):
    """One panel: `points` = [(x, values, reported)], returns the y extent."""
    xs, means, lo_all, hi_all = [], [], np.inf, -np.inf
    for x, v, reported in points:
        v = np.asarray(v, dtype=float)
        v = v[np.isfinite(v)]
        if not v.size:
            continue
        mean, lo, hi = bootstrap_ci(v)
        xs.append(x); means.append(mean)
        lo_all, hi_all = min(lo_all, lo), max(hi_all, hi)
        table_row(x, mean, lo, hi, int(v.size))
        ax.plot([x] * 2, [lo, hi], color=colour, lw=1.0, solid_capstyle='butt', zorder=2)
        ax.plot([x], [mean], 'o', color=colour, ms=4.0 if reported else 3.0,
                mec='black' if reported else 'white', mew=0.5 if reported else 0.4, zorder=3)
    if len(xs) > 1:
        order = np.argsort(xs)
        ax.plot(np.asarray(xs)[order], np.asarray(means)[order], color=colour,
                lw=0.8, alpha=0.6, zorder=1)
    all_x = [x for x, _, _ in points]
    rep_x = next((x for x, _, r in points if r), None)
    if rep_x is not None:
        ax.axvline(rep_x, color='0.6', lw=0.5, ls=':', zorder=0)
    ax.set_xscale('log')
    ax.set_xlim(min(all_x) / 1.8, max(all_x) * 1.8)
    ax.set_xticks(all_x)
    ax.set_xticklabels([tick_label(x, fmt) for x in all_x])
    ax.minorticks_off()
    ax.tick_params(axis='x', pad=1.5)
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='y', color='0.9', lw=0.5, zorder=0)
    if not xs:
        ax.text(0.5, 0.5, 'pending', transform=ax.transAxes, ha='center',
                va='center', color=MUTED, fontsize=plt.rcParams['font.size'] - 1)
    return lo_all, hi_all


def footer(fig, height, metric, fs):
    fig.text(0.5, 1.5 / 72 / height,
             f'{METRICS[metric]}; mean over seeds with 95% bootstrap CI; '
             'the outlined marker and dotted line are the reported setting.',
             ha='center', va='bottom', color=MUTED, fontsize=fs - 1)


def save(fig, stem, width, height):
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ('pdf', 'png'):
        fig.savefig(OUT / f'{stem}.{ext}', dpi=300)
    plt.close(fig)
    print(f'wrote {OUT / stem}.pdf, .png  ({width:.2f} x {height:.2f} in)')


def _colour(arm):
    """The paper reports NES as ES: one method, ES's colour, as every other
    figure (plot_stability_plasticity.ES_COLOUR). METHOD_STYLE's own `nes`
    entry is a darker gold that appears nowhere else in the paper."""
    return lp.METHOD_STYLE['es' if arm == 'nes' else arm]['color']


def draw_sweep(meta, metric, args):
    """The gymnax-noise figure: one row a task, one column an axis."""
    fs = args.font_size
    envs = [c.split('_sigma')[0] for c in meta['families']['noise']['cells']]
    # Columns grouped in the paper's method order (NE before RL, as Figure 2),
    # not the saved extract's; stable, so each family keeps its own axis order.
    rank = {m: i for i, m in enumerate(lp.METHOD_ORDER)}
    axes = sorted(meta['axes'], key=lambda a: rank.get(a['arm'], len(rank)))
    width = args.width
    top_in, bottom_in, left_in = 0.42, 0.3, 0.5
    height = args.row_height * len(envs) + top_in + bottom_in
    fig = plt.figure(figsize=(width, height))
    gs = GridSpec(len(envs), len(axes), figure=fig,
                  left=left_in / width, right=1 - 0.08 / width,
                  top=1 - top_in / height, bottom=bottom_in / height,
                  wspace=0.45, hspace=0.35)
    table = []
    row_axes = {r: [] for r in range(len(envs))}
    row_ext = {r: [np.inf, -np.inf] for r in range(len(envs))}
    for c, axis in enumerate(axes):
        colour = _colour(axis['arm'])
        for r, env in enumerate(envs):
            ax = fig.add_subplot(gs[r, c])
            row_axes[r].append(ax)
            points = [(p['x'], p['envs'].get(env, {}).get(metric, []), p['setting'] == 'reported')
                      for p in axis['points']]
            lo, hi = draw_points(
                ax, points, colour, axis['fmt'],
                lambda x, mean, lo, hi, n, _a=axis, _e=env: table.append(
                    (_a['arm'], _a['key'], x, lp.ENV_TITLES.get(_e, _e) + ', noise',
                     metric, mean, lo, hi, n)))
            row_ext[r] = [min(row_ext[r][0], lo), max(row_ext[r][1], hi)]
            if r == 0:
                ax.set_title(axis['title'], pad=3, fontsize=fs)
            if c == 0:
                ax.set_ylabel(lp.ENV_TITLES.get(env, env), labelpad=4)
    # One y range a task across both families: the return is the same
    # quantity in every panel of a row, and a setting that breaks a method is
    # then read against the reported one at a glance.
    for r, (lo, hi) in row_ext.items():
        if not np.isfinite(lo):
            continue
        pad = 0.08 * (hi - lo) if hi > lo else 1.0
        for i, ax in enumerate(row_axes[r]):
            ax.set_ylim(lo - pad, hi + pad)
            if i > 0:
                ax.set_yticklabels([])
    # Family header over its columns, with a rule under it.
    groups = {}
    for c, axis in enumerate(axes):
        groups.setdefault(axis['arm'], []).append(c)
    y = 1 - (top_in - 0.14) / height
    for arm, cols in groups.items():
        x0 = fig.axes[cols[0] * len(envs)].get_position().x0
        x1 = fig.axes[cols[-1] * len(envs)].get_position().x1
        fig.text((x0 + x1) / 2, y + 0.02 / height, FAMILY_TITLE[arm], ha='center',
                 va='bottom', fontsize=fs + 0.5, color=_colour(arm))
        fig.add_artist(plt.Line2D([x0, x1], [y, y], transform=fig.transFigure,
                                  color='0.4', lw=0.5))
    footer(fig, height, metric, fs)
    save(fig, STEM if metric == 'cum' else f'{STEM}_{metric}', width, height)
    return table


def draw_cross_family(meta, key, metric, args):
    """One cross-family figure (`key` in SWEEPS): one panel a cell, PPO only."""
    fs = args.font_size
    spec = SWEEPS[key]
    entries = {e['family']: e['points'] for e in meta['sweeps'].get(key, [])}
    panels = [(name, cell) for name, fam in meta['families'].items()
              if name in entries for cell in fam['cells']]
    if not panels:
        return []
    ncols = 3
    nrows = (len(panels) + ncols - 1) // ncols
    width = args.width
    top_in, bottom_in, left_in = 0.25, 0.45, 0.45   # room for the x label and the footer
    height = args.row_height * 1.15 * nrows + top_in + bottom_in
    fig = plt.figure(figsize=(width, height))
    gs = GridSpec(nrows, ncols, figure=fig,
                  left=left_in / width, right=1 - 0.08 / width,
                  top=1 - top_in / height, bottom=bottom_in / height,
                  wspace=0.4, hspace=0.6)
    colour = lp.METHOD_STYLE['ppo']['color']
    table = []
    for i, (name, cell) in enumerate(panels):
        ax = fig.add_subplot(gs[i // ncols, i % ncols])
        title = meta['families'][name]['titles'][cell]
        points = [(p['x'], p['cells'].get(cell, {}).get(metric, []), p['reported'])
                  for p in entries[name]]
        draw_points(ax, points, colour, spec['fmt'],
                    lambda x, mean, lo, hi, n, _t=title: table.append(
                        ('ppo', key, x, _t, metric, mean, lo, hi, n)))
        ax.set_title(title, pad=3, fontsize=fs)
        if i % ncols == 0:
            ax.set_ylabel(METRICS[metric], labelpad=4)
        if i // ncols == nrows - 1 or i + ncols >= len(panels):
            ax.set_xlabel(spec['xlabel'], labelpad=2)
    footer(fig, height, metric, fs)
    save(fig, spec['stem'] if metric == 'cum' else f'{spec["stem"]}_{metric}', width, height)
    return table


def draw_plane(meta, key, args):
    """The trade-off picture for one cross-family sweep: one panel a cell,
    learning accuracy LA (x) against forgetting F (y) of the centroid, the
    settings joined in order of x with the reported one ringed and every
    point labelled by its value. A path that runs down-left is a trade-off
    (less forgetting bought with lower accuracy); a point up-left of the
    reported one is simply worse."""
    fs = args.font_size
    spec = SWEEPS[key]
    entries = {e['family']: e['points'] for e in meta['sweeps'].get(key, [])}
    panels = [(name, cell) for name, fam in meta['families'].items()
              if name in entries for cell in fam['cells']
              if any(cell in p['cells'] and 'la' in p['cells'][cell] for p in entries[name])]
    if not panels:
        return []
    ncols = 3
    nrows = (len(panels) + ncols - 1) // ncols
    width = args.width
    top_in, bottom_in, left_in = 0.25, 0.45, 0.5
    height = args.row_height * 1.25 * nrows + top_in + bottom_in
    fig = plt.figure(figsize=(width, height))
    gs = GridSpec(nrows, ncols, figure=fig,
                  left=left_in / width, right=1 - 0.08 / width,
                  top=1 - top_in / height, bottom=bottom_in / height,
                  wspace=0.45, hspace=0.65)
    colour = lp.METHOD_STYLE['ppo']['color']
    table = []
    for i, (name, cell) in enumerate(panels):
        ax = fig.add_subplot(gs[i // ncols, i % ncols])
        title = meta['families'][name]['titles'][cell]
        path = []
        for p in sorted(entries[name], key=lambda p: p['x']):
            v = p['cells'].get(cell, {})
            la, F = (np.asarray(v.get(k, []), dtype=float) for k in ('la', 'F'))
            la, F = la[np.isfinite(la)], F[np.isfinite(F)]
            if not la.size or not F.size:
                continue
            (lm, llo, lhi), (fm, flo, fhi) = bootstrap_ci(la), bootstrap_ci(F)
            path.append((p['x'], lm, fm, p['reported']))
            table.append(('ppo', key + ' plane', p['x'], title, 'la', lm, llo, lhi, int(la.size)))
            table.append(('ppo', key + ' plane', p['x'], title, 'F', fm, flo, fhi, int(F.size)))
            ax.plot([llo, lhi], [fm, fm], color=colour, lw=0.6, alpha=0.5, zorder=1)
            ax.plot([lm, lm], [flo, fhi], color=colour, lw=0.6, alpha=0.5, zorder=1)
            ax.plot([lm], [fm], 'o', color=colour, ms=4.0 if p['reported'] else 3.0,
                    mec='black' if p['reported'] else 'white',
                    mew=0.5 if p['reported'] else 0.4, zorder=3)
            ax.annotate(tick_label(p['x'], spec['fmt']), (lm, fm), xytext=(3, 3),
                        textcoords='offset points', fontsize=fs - 1.5, color=MUTED)
        if len(path) > 1:
            ax.plot([q[1] for q in path], [q[2] for q in path], color=colour, lw=0.8,
                    alpha=0.6, zorder=2)
        ax.set_title(title, pad=3, fontsize=fs)
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(color='0.9', lw=0.5, zorder=0)
        ax.margins(0.2)
        if i % ncols == 0:
            ax.set_ylabel('forgetting F ↓', labelpad=4)
        if i // ncols == nrows - 1 or i + ncols >= len(panels):
            ax.set_xlabel('learning accuracy LA ↑', labelpad=2)
    fig.text(0.5, 1.5 / 72 / height,
             f'Centroid; mean over seeds with 95% CI bars; labels are the {spec["xlabel"]}; '
             'the outlined marker is the reported setting.',
             ha='center', va='bottom', color=MUTED, fontsize=fs - 1)
    save(fig, f'{spec["stem"]}_plane', width, height)
    return table


def write_markdown(meta, tables):
    md = [f'# {STEM}', '',
          'Effect of hyperparameters, the update-count half. Built by '
          f'`scripts/analysis/plot_hparam_updates.py` from `data/{STEM}.json` '
          f'(extracted {meta["extracted"]}); its docstring defines the columns. '
          'Every setting changes one value of the reported configuration; the reported '
          'point of each family is trials 1-5 of its reported runs. Mean over trials '
          '[95% bootstrap CI], n.', '',
          '## Run trees', '', '| Setting | Run tree | Status |', '|---|---|---|']
    for axis in meta['axes']:
        for p in axis['points']:
            n = {e: len(v['cum']) for e, v in p['envs'].items()}
            md.append(f'| `{p["setting"]}` (noise, {axis["arm"]} {axis["key"]} = {p["x"]:g}) | '
                      f'`{p["run"]}` | {n if n else "pending"} |')
    for key, entries in meta['sweeps'].items():
        for entry in entries:
            for p in entry['points']:
                n = {c: len(v['cum']) for c, v in p['cells'].items()}
                md.append(f'| `{p["setting"]}` ({entry["family"]}, ppo {key} = {p["x"]:g}) | '
                          f'`{p["run"]}` | {n if n else "pending"} |')
    md += ['', '## Points', '',
           '| Family | Axis | x | Task | Metric | mean | lo | hi | n |',
           '|---|---|---|---|---|---|---|---|---|']
    seen = set()
    for table in tables:
        for row in table:
            if row in seen:                       # the noise cells appear in two figures
                continue
            seen.add(row)
            arm, key, x, task, metric, mean, lo, hi, n = row
            md.append(f'| {FAMILY_TITLE[arm]} | {key} | {x:g} | {task} | {METRICS[metric]} | '
                      f'{mean:.3g} | {lo:.3g} | {hi:.3g} | {n} |')
    (OUT / f'{STEM}.md').write_text('\n'.join(md) + '\n')
    print(f'wrote {OUT / STEM}.md')


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--extract', action='store_true',
                    help='re-read the runs into the saved data first')
    ap.add_argument('--metric', default='all', choices=[*METRICS, 'all'],
                    help='one metric, or all: the noise figure for the curve metrics, '
                         'the cross-family figures for every metric, plus the LA/F planes')
    ap.add_argument('--width', type=float, default=TEXT_WIDTH_IN)
    ap.add_argument('--row-height', type=float, default=0.8)
    ap.add_argument('--font-size', type=float, default=5.5)
    args = ap.parse_args()
    if args.extract:
        extract()
    if not DATA.exists():
        sys.exit(f'no {DATA}: run with --extract first')
    meta = json.loads(DATA.read_text())
    if 'sweeps' not in meta:
        sys.exit(f'{DATA} predates the cross-family figures: run with --extract')
    plt.rcParams.update({
        'font.size': args.font_size, 'axes.titlesize': args.font_size + 0.5,
        'xtick.labelsize': args.font_size - 1, 'ytick.labelsize': args.font_size - 0.5,
        'axes.labelsize': args.font_size,
        'axes.linewidth': 0.5, 'xtick.major.width': 0.5, 'xtick.major.size': 2,
        'ytick.major.width': 0.5, 'ytick.major.size': 2,
    })
    wanted = list(METRICS) if args.metric == 'all' else [args.metric]
    tables = []
    for m in wanted:
        if m in SWEEP_METRICS:
            tables.append(draw_sweep(meta, m, args))
        if m in CROSS_METRICS:
            for key in SWEEPS:
                tables.append(draw_cross_family(meta, key, m, args))
    if args.metric == 'all':
        for key in SWEEPS:
            tables.append(draw_plane(meta, key, args))
    write_markdown(meta, tables)
    return 0


if __name__ == '__main__':
    sys.exit(main())
