"""Appendix: the effect of hyperparameters on the continual comparison.

    .venv/bin/python scripts/analysis/plot_hparam_updates.py             # from the saved data
    .venv/bin/python scripts/analysis/plot_hparam_updates.py --extract   # re-read the passes first

    -> paper/visuals/final/appendix/hparam_sigma.{pdf,png}   the stability-plasticity plane, one
                                                             column a task; a PPO row only when
                                                             ppo_lr*/ppo_ent* arms exist
       paper/visuals/final/appendix/hparam_sigma_curves.*    LA and F against the multiplier
       paper/visuals/final/appendix/hparam_sigma.{md,tex}    every number
       paper/visuals/final/data/hparam_sigma.json            per-trial LA, F and Cum. it is drawn from
    (hparam_sweep.* is the OTHER hyperparameter figure: PPO epochs / rollout length /
     learning rate and ES population / width / learning rate against cumulative
     return, drawn from runs_hparam/<setting>/; this script does not touch it)
    (paper = projects/iclr_2027/paper)

THE QUESTION. The paper compares every method at one setting (Appendix tables
hyper_ne / hyper_rl). scripts/train/queue_iclr_hparam.sh moves ONE
hyperparameter at a time over two orders of magnitude around that setting --
x0.1, x0.3, x3, x10 -- with everything else the reported run's: the GA's
mutation width, NES's search width, PPO's learning rate and PPO's entropy
coefficient, on the three classic-control cells of the main continual figure
(CartPole / Acrobot at offset 1.0, MountainCar at 0.1), five trials each. The
reported value is the reported run itself (runs_centroid, ten trials), read
from the paper's own forgetting pass.

THE AXES are plot_stability_plasticity.py's, for the CENTROID: learning
accuracy LA = mean_i R[i][i] (plasticity) and forgetting F (stability), both
from the reward matrix of the forgetting pass, rescaled per environment by the
REPORTED family's scale -- (LA - untrained) / (best - untrained), with `best`
the best reported mean LA in that cell -- so a sweep point sits on the same
axes as the paper's figure and 1.0 on x is the paper's best method. A point
above 1 or below 0 is possible and means what it says.

THE FIGURE. One row a family (observation noise, action reversal, MiniGrid;
FAMILIES), one panel a cell: the GA's and ES's sweep as a path through the
plane, the reported setting ringed, the other four points labelled by their
multiplier; the reported RL arms as hollow grey references. The conclusions
hold when the GA's path stays right of (more plastic than) and below (more
forgetting than) ES's whatever the width. The noise cells have five seeds a
width, action reversal and MiniGrid three (2026-09-20).

The curves figure is the same data against the multiplier, one line a sweep,
for reading which values work at all.
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
import plot_stability_plasticity as psp                    # noqa: E402

lp, pcl, PROJECT, REPO = psp.lp, psp.pcl, psp.PROJECT, psp.REPO
FINAL = psp.FINAL
OUT = FINAL / 'appendix'
STEM = 'hparam_sigma'      # not hparam_sweep: that name is plot_hparam_updates.py's figure
DATA = FINAL / 'data' / f'{STEM}.json'

# THE FAMILIES (2026-09-20: action reversal and MiniGrid joined the noise
# cells). Per family: the reported family's forgetting pass (the reported
# arms' LA and F at the reported settings) and its arm list's key in the
# stability figure's saved data; the sweep tree and the pass
# finish_hparam_sigma.sh runs on it; the cells, their titles, and the
# reported GA / ES widths the multipliers are read against.
FAMILIES = {
    'noise': dict(
        reported=PROJECT / 'paper/gymnax/noise/10task/results/centroid',
        reported_tree='paper/gymnax/data/noise_10task',
        root=PROJECT / 'runs_hparam/gymnax',
        results=PROJECT / 'paper/gymnax/hparam/results/centroid',
        cells=('CartPole_v1_sigma1.0', 'Acrobot_v1_sigma1.0', 'MountainCar_v0_sigma0.1'),
        titles=('CartPole, noise', 'Acrobot, noise', 'MountainCar, noise'),
        default={'ga_sigma': 0.5, 'nes_sigma': 0.1}),
    'actions': dict(
        reported=PROJECT / 'paper/gymnax/actions/2task/results/centroid',
        reported_tree='paper/gymnax/data/actions_2task',
        root=PROJECT / 'runs_hparam_actions/gymnax',
        results=PROJECT / 'paper/gymnax/hparam_actions/results/centroid',
        cells=('CartPole_v1_sigma1.0', 'Acrobot_v1_sigma1.0', 'MountainCar_v0_sigma1.0'),
        titles=('CartPole, action reversal', 'Acrobot, action reversal',
                'MountainCar, action reversal'),
        default={'ga_sigma': 0.5, 'nes_sigma': 0.1}),
    'minigrid': dict(
        reported=PROJECT / 'paper/minigrid/minigrid/results/centroid',
        reported_tree='paper/minigrid/data',
        root=PROJECT / 'runs_hparam_minigrid/minigrid',
        results=PROJECT / 'paper/minigrid/hparam/results/centroid',
        cells=('MiniGrid_8x8_16x16',),
        titles=('MiniGrid 8x8 / 16x16',),
        default={'ga_sigma': 0.01, 'nes_sigma': 0.1}),
}
# A panel is (family, cell); the figure lays the families out one row each.
PANELS = [(f, c, t) for f, spec in FAMILIES.items() for c, t in zip(spec['cells'], spec['titles'])]
TITLES = {(f, c): t for f, c, t in PANELS}

# The four sweeps: (reported arm, arm prefix, label, the reported value per
# cell). The multiplier a sweep arm carries is value / reported.
# The four sweeps: (reported arm, label, style). The PPO two are defined for
# completeness: their arms were never queued (the other half of the appendix
# sweeps PPO), so they draw nothing.
SWEEPS = {
    'ga_sigma': dict(arm='ga', label=r'GA, mutation width $\sigma$', style='ga'),
    'nes_sigma': dict(arm='nes', label=r'ES, search width $\sigma$', style='es'),
    'ppo_lr': dict(arm='ppo', label=r'PPO, learning rate $\alpha$', style='ppo'),
    'ppo_ent': dict(arm='ppo', label=r'PPO, entropy coefficient $\beta$', style='ppo'),
}
PPO_LR = {'CartPole_v1_sigma1.0': 3e-4, 'Acrobot_v1_sigma1.0': 1e-4,
          'MountainCar_v0_sigma0.1': 3e-4, 'MountainCar_v0_sigma1.0': 3e-4}


def default_of(key, family, cell):
    spec = FAMILIES[family]['default']
    if key in spec:
        return spec[key]
    return PPO_LR.get(cell, 3e-4) if key == 'ppo_lr' else 0.01


ROWS = (('Neuroevolution: the search width', ('ga_sigma', 'nes_sigma')),
        ('PPO: learning rate and entropy coefficient', ('ppo_lr', 'ppo_ent')))
MULTS = (0.1, 0.3, 1.0, 3.0, 10.0)
N_BOOT = 2000


# ----------------------------------------------------------------------------
# extract
# ----------------------------------------------------------------------------

def load_plane(results_dir, keep, cells):
    """`{cell: {arm: [{'la', 'F', 'cum', 'run'}]}}` for the CENTROID from one
    forgetting pass, `arm` the run's method directory; `keep(arm)` filters."""
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
            arm_dir, cell, _trial = run_dir.rsplit('/', 2)
            arm = arm_dir.rsplit('/', 1)[-1]
            if cell not in cells or not keep(arm) \
                    or rec.get('source') not in lp.ZT_SOURCES['centroid']:
                continue
            la = float(np.mean(np.diag(archive[f'run{i}_reward'])))
            summary = divergence.get(run_dir, {})
            if np.isfinite(la) and summary.get('F') is not None:
                out.setdefault(cell, {}).setdefault(arm, []).append(
                    {'la': la, 'F': float(summary['F']),
                     'cum': psp.curve_mean(REPO / run_dir), 'run': run_dir})
    return out


def reported_arms(family):
    """The arms the paper reports for `family`, from the stability figure's
    saved data (one ES arm, one PBT arm), else the family's es_arm.json."""
    spec = FAMILIES[family]
    saved = FINAL / 'data' / 'stability_plasticity.json'
    if saved.exists():
        arms = json.loads(saved.read_text()).get('arms', {}).get(spec['reported_tree'])
        if arms:
            return list(arms)
    es_json = spec['reported'].parent.parent / 'es_arm.json'
    kept = json.loads(es_json.read_text()).get('kept') if es_json.exists() else None
    return ['ga', kept or 'nes', 'ppo', 'trac', 'redo', 'cchain', 'pbt']


def sweep_of(arm):
    """`(sweep key, value)` for a sweep arm such as `ga_sigma0.15`, else None."""
    for key in SWEEPS:
        if arm.startswith(key) and len(arm) > len(key):
            try:
                return key, float(arm[len(key):])
            except ValueError:
                return None
    return None


def extract():
    meta = {'reported_arms': {}, 'reported': {}, 'sweep': {},
            'extracted': datetime.date.today().isoformat()}
    for family, spec in FAMILIES.items():
        arms = reported_arms(family)
        reported = load_plane(spec['reported'], lambda a: a in arms, spec['cells'])
        sweep = load_plane(spec['results'], lambda a: sweep_of(a) is not None, spec['cells'])
        # What is on disk against what the pass has scored: an arm still
        # training, or scored on fewer trials than it has, is named here.
        for arm_dir in sorted((spec['root'] / 'continual').glob('*')):
            for cell in spec['cells']:
                done = len(list((arm_dir / cell).glob('trial_*/training_metrics.json')))
                scored = len(sweep.get(cell, {}).get(arm_dir.name, []))
                if done and scored < done:
                    print(f'note: {family} {arm_dir.name} {cell}: {done} finished, {scored} scored')
        meta['reported_arms'][family] = arms
        meta['reported'][family] = reported
        meta['sweep'][family] = sweep
    DATA.parent.mkdir(parents=True, exist_ok=True)
    DATA.write_text(json.dumps(meta, indent=1) + '\n')
    print(f'wrote {DATA}')


# ----------------------------------------------------------------------------
# the numbers
# ----------------------------------------------------------------------------

def scale_for(reported, cell):
    """`(floor, best)` of the reported family in `cell`: the paper's axes."""
    env = cell.split('_sigma')[0]
    best = max(float(np.mean([t['la'] for t in trials]))
               for trials in reported[cell].values())
    return psp.FLOOR[env], best


def rescale(trials, floor, best):
    span = best - floor
    la = np.array([t['la'] for t in trials], dtype=float)
    f = np.array([t['F'] for t in trials], dtype=float)
    return (la - floor) / span, f / span


def ci(values):
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return (np.nan, np.nan, np.nan)
    rng = np.random.default_rng(0)
    boots = rng.choice(values, (N_BOOT, values.size)).mean(axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(values.mean()), float(lo), float(hi)


def points(meta):
    """`{(family, cell): {sweep: {mult: {'x', 'y', 's', 'cum': (mean, lo, hi), 'n', 'value'}}}}`
    plus `{(family, cell): {arm: point}}` of the reported arms, all on the reported scale."""
    out, refs = {}, {}
    for family, cell, _title in PANELS:
        reported = meta['reported'].get(family, {})
        sweep = meta['sweep'].get(family, {})
        if cell not in reported:
            continue
        floor, best = scale_for(reported, cell)
        panel = (family, cell)
        refs[panel] = {}
        for arm, trials in reported[cell].items():
            x, y = rescale(trials, floor, best)
            refs[panel][arm] = {'x': ci(x), 'y': ci(y), 's': ci(x - y),
                                'cum': ci([t['cum'] for t in trials]), 'n': len(trials)}
        out[panel] = {}
        for key, spec in SWEEPS.items():
            by_mult = {}
            default = default_of(key, family, cell)
            if spec['arm'] in reported[cell]:
                by_mult[1.0] = {**refs[panel][spec['arm']], 'value': default}
            for arm, trials in sweep.get(cell, {}).items():
                parsed = sweep_of(arm)
                if not parsed or parsed[0] != key:
                    continue
                value = parsed[1]
                mult = value / default
                x, y = rescale(trials, floor, best)
                by_mult[float(f'{mult:.3g}')] = {
                    'x': ci(x), 'y': ci(y), 's': ci(x - y),
                    'cum': ci([t['cum'] for t in trials]), 'n': len(trials), 'value': value}
            out[panel][key] = dict(sorted(by_mult.items()))
    return out, refs


# ----------------------------------------------------------------------------
# figures
# ----------------------------------------------------------------------------

def _colour(style):
    return lp.METHOD_STYLE[style]['color']


def _errbar(ax, p, colour, marker, size, fill=True, z=3, alpha=1.0):
    ax.errorbar(p['x'][0], p['y'][0],
                xerr=[[p['x'][0] - p['x'][1]], [p['x'][2] - p['x'][0]]],
                yerr=[[p['y'][0] - p['y'][1]], [p['y'][2] - p['y'][0]]],
                fmt='none', ecolor=colour, elinewidth=0.6, capsize=0, alpha=0.5 * alpha, zorder=z)
    ax.plot(p['x'][0], p['y'][0], marker=marker, ms=size, mew=0.9,
            color=colour, mfc=colour if fill else 'white', alpha=alpha, zorder=z + 1, ls='none')


ROW_TITLES = {'noise': 'observation noise', 'actions': 'action reversal', 'minigrid': 'MiniGrid'}


def draw_plane(pts, refs, stem):
    """One row a family, one panel a cell: the GA's and ES's width paths."""
    keys = ROWS[0][1]
    rows = [f for f in FAMILIES if any(p[0] == f for p in pts)]
    ncols = max(len(FAMILIES[f]['cells']) for f in rows)
    fig, axes = plt.subplots(len(rows), ncols, figsize=(1.85 * ncols + 0.6, len(rows) * 1.8 + 0.7),
                             squeeze=False)
    for r, family in enumerate(rows):
        ref_family = 'rl'
        cells = FAMILIES[family]['cells']
        for c in range(ncols):
            ax = axes[r, c]
            if c >= len(cells) or (family, cells[c]) not in pts:
                ax.set_axis_off()
                continue
            cell = (family, cells[c])
            # the other family's reported arms, hollow grey
            for arm, p in refs[cell].items():
                if lp.FAMILY.get(arm) != ref_family:
                    continue
                _errbar(ax, p, '0.55', lp.METHOD_STYLE.get(arm, {}).get('marker', 'o'),
                        4, fill=False, z=1, alpha=0.9)
                ax.annotate(lp.METHOD_STYLE.get(arm, {}).get('label', arm),
                            (p['x'][0], p['y'][0]), xytext=(3, 3),
                            textcoords='offset points', fontsize=5, color='0.45')
            for key in keys:
                spec = SWEEPS[key]
                colour = _colour(spec['style'])
                by_mult = pts[cell].get(key, {})
                if not by_mult:
                    continue
                ls = '--' if key == 'ppo_ent' else '-'
                xs = [p['x'][0] for p in by_mult.values()]
                ys = [p['y'][0] for p in by_mult.values()]
                ax.plot(xs, ys, ls=ls, lw=0.8, color=colour, alpha=0.7, zorder=2)
                # Widths that land on the same point (typically the origin:
                # an ES width that never learns has LA = F = 0) get ONE label
                # listing them, instead of labels printed over each other.
                labels = {}
                for mult, p in by_mult.items():
                    default = abs(mult - 1.0) < 1e-6
                    _errbar(ax, p, colour, 'o', 5.5 if default else 4, z=4)
                    if default:
                        ax.plot(p['x'][0], p['y'][0], 'o', ms=9, mfc='none', mec=colour,
                                mew=0.9, zorder=5)
                    else:
                        at = (round(p['x'][0], 2), round(p['y'][0], 2))
                        labels.setdefault(at, []).append(f'{mult:g}x')
                for (x, y), names in labels.items():
                    ax.annotate(', '.join(names), (x, y), xytext=(3, -6),
                                textcoords='offset points', fontsize=5, color=colour)
            ax.axhline(0, color='0.85', lw=0.5, zorder=0)
            ax.set_xlim(-0.05, 1.15)
            # Forgetting can be NEGATIVE: an agent that learns slowly keeps
            # improving on old sub-tasks after their phase ends (ES at 10x on
            # Acrobot, F = -0.46). The axis extends to show it rather than
            # clip it; the shared floor keeps panels comparable otherwise.
            lows = [q['y'][1] for key in keys for q in pts[cell].get(key, {}).values()]
            lows += [q['y'][1] for q in refs[cell].values()]
            ax.set_ylim(min(-0.05, min(lows) - 0.08) if lows else -0.05, 1.15)
            ax.grid(True, lw=0.3, alpha=0.4)
            ax.tick_params(labelsize=6)
            ax.set_title(TITLES[cell], fontsize=7.5)
            if r == len(rows) - 1 or c >= len(FAMILIES[rows[r + 1]]['cells']):
                ax.set_xlabel('Learning accuracy', fontsize=7)
            if c == 0:
                ax.set_ylabel('Forgetting', fontsize=7)
    handles = []
    for key in keys:
        spec = SWEEPS[key]
        handles.append(Line2D([], [], color=_colour(spec['style']), marker='o', ms=4,
                              ls='-', lw=0.8, label=spec['label']))
    handles.append(Line2D([], [], color='0.55', marker='o', ms=4, mfc='white', ls='none',
                          label='reported arms of the other family'))
    handles.append(Line2D([], [], color='0.3', marker='o', ms=8, mfc='none', ls='none',
                          label='reported setting (1x)'))
    fig.legend(handles=handles, loc='lower center', ncol=3, fontsize=6, frameon=False,
               bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0.03, 0.09 / len(rows) * 1.0 + 0.02, 1, 1))
    for ext in ('pdf', 'png'):
        fig.savefig(OUT / f'{stem}.{ext}', dpi=200, bbox_inches='tight')
    plt.close(fig)


def draw_curves(pts, stem):
    """LA and F against the multiplier: which values work at all. One column
    a panel of the plane figure, LA over F."""
    panels = [p for p in PANELS if (p[0], p[1]) in pts]
    ncols = len(panels)
    fig, axes = plt.subplots(2, ncols, figsize=(1.55 * ncols + 0.6, 2 * 1.5 + 0.7),
                             squeeze=False, sharex=True)
    for c, (family, cell_name, _t) in enumerate(panels):
        cell = (family, cell_name)
        for r, (axis, name) in enumerate((('x', 'Learning accuracy'), ('y', 'Forgetting'))):
            ax = axes[r, c]
            for key, spec in SWEEPS.items():
                by_mult = pts[cell].get(key, {})
                if len(by_mult) < 2:
                    continue
                mults = np.array(list(by_mult))
                mean = np.array([p[axis][0] for p in by_mult.values()])
                lo = np.array([p[axis][1] for p in by_mult.values()])
                hi = np.array([p[axis][2] for p in by_mult.values()])
                colour = _colour(spec['style'])
                ls = '--' if key == 'ppo_ent' else '-'
                ax.plot(mults, mean, ls=ls, lw=0.9, marker='o', ms=3, color=colour,
                        label=spec['label'])
                ax.fill_between(mults, lo, hi, color=colour, alpha=0.15, lw=0)
            ax.set_xscale('log')
            ax.set_xticks(MULTS)
            ax.set_xticklabels([f'{m:g}x' for m in MULTS])
            ax.axvline(1.0, color='0.8', lw=0.5, zorder=0)
            ax.grid(True, lw=0.3, alpha=0.4)
            ax.tick_params(labelsize=6)
            if r == 0:
                ax.set_title(TITLES[cell], fontsize=6.5)
            if r == 1:
                ax.set_xlabel('x reported', fontsize=7)
            if c == 0:
                ax.set_ylabel(name, fontsize=7)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=2, fontsize=6, frameon=False,
               bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    for ext in ('pdf', 'png'):
        fig.savefig(OUT / f'{stem}_curves.{ext}', dpi=200, bbox_inches='tight')
    plt.close(fig)


# ----------------------------------------------------------------------------
# tables
# ----------------------------------------------------------------------------

def _fmt(p, digits=2):
    return f'{p[0]:.{digits}f} [{p[1]:.{digits}f}, {p[2]:.{digits}f}]'


def write_tables(pts, refs, stem):
    lines = [f'# {STEM}', '',
             'Centroid learning accuracy (LA, plasticity) and forgetting (F, stability) on the '
             'axes of the stability-plasticity figure, rescaled per cell by the reported '
             'family (0 = untrained network, 1 = the best reported mean LA); LA - F is the '
             'performance kept. Cum. is the mean of the centroid curve over the run (return '
             'units). Mean and 95% bootstrap interval over trials; the 1x row is the '
             'reported run (ten trials), the others the sweep (five).', '']
    tex = ['\\begin{tabular}{llrrrrr}', '\\toprule',
           'Task & Sweep & $\\times$ & value & LA & F & LA $-$ F \\\\', '\\midrule']
    for family, cell_name, _t in PANELS:
        cell = (family, cell_name)
        if cell not in pts:
            continue
        lines += [f'## {TITLES[cell]}', '',
                  '| sweep | x | value | n | LA | F | LA - F | Cum. |', '|---|---|---|---|---|---|---|---|']
        for key, spec in SWEEPS.items():
            for mult, p in pts[cell].get(key, {}).items():
                lines.append(f"| {spec['label']} | {mult:g} | {p['value']:g} | {p['n']} | "
                             f"{_fmt(p['x'])} | {_fmt(p['y'])} | {_fmt(p['s'])} | {_fmt(p['cum'], 0)} |")
                tex.append(f"{TITLES[cell]} & {spec['label']} & {mult:g} & {p['value']:g} & "
                           f"{p['x'][0]:.2f} & {p['y'][0]:.2f} & {p['s'][0]:.2f} \\\\")
        lines += ['', 'Reported arms (references):', '',
                  '| arm | n | LA | F | LA - F |', '|---|---|---|---|---|']
        for arm, p in refs[cell].items():
            lines.append(f"| {lp.METHOD_STYLE.get(arm, {}).get('label', arm)} | {p['n']} | "
                         f"{_fmt(p['x'])} | {_fmt(p['y'])} | {_fmt(p['s'])} |")
        lines.append('')
        tex.append('\\midrule')
    tex[-1] = '\\bottomrule'
    tex.append('\\end{tabular}')
    (OUT / f'{stem}.md').write_text('\n'.join(lines) + '\n')
    (OUT / f'{stem}.tex').write_text('\n'.join(tex) + '\n')


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--families', nargs='+', choices=list(FAMILIES), default=None,
                    help='draw only these families (rows); the saved data keeps all of them. '
                         'The four-hour version of D.3 is --families noise actions, because '
                         'a MiniGrid ES trial takes ~370 min even on a GH200 (2026-09-21)')
    ap.add_argument('--extract', action='store_true',
                    help='re-read the forgetting passes into the saved data first')
    args = ap.parse_args()
    if args.extract:
        extract()
    if not DATA.exists():
        sys.exit(f'no {DATA}: run with --extract first')
    meta = json.loads(DATA.read_text())
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.size': 7})
    # The paper's name for the PBT arm, as every other final figure sets it
    # (METHOD_STYLE's own label is 'PBT-PPO (N=8)').
    lp.METHOD_STYLE['pbt'] = {**lp.METHOD_STYLE['pbt'], 'label': 'PBT-PPO'}
    pts, refs = points(meta)
    if args.families:
        keep = set(args.families)
        pts = {k: v for k, v in pts.items() if k[0] in keep}
        refs = {k: v for k, v in refs.items() if k[0] in keep}
    for family, cell_name, _t in PANELS:
        cell = (family, cell_name)
        have = {k: list(v) for k, v in pts.get(cell, {}).items() if v}
        print(f'{TITLES[cell]:28s} ' + '  '.join(f'{k}: {[f"{m:g}" for m in ms]}'
                                                 for k, ms in have.items()))
    draw_plane(pts, refs, STEM)
    draw_curves(pts, STEM)
    write_tables(pts, refs, STEM)
    print(f'wrote {OUT / STEM}.{{pdf,png,md,tex}} and {OUT / STEM}_curves.*')
    return 0


if __name__ == '__main__':
    sys.exit(main())
