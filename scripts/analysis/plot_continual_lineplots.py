"""The continual lineplots of every family the paper reports, in the style of
the stationary figure (scripts/analysis/plot_noncontinual_solve.py): the
main-text figure, and for the appendix one figure a suite, gymnax split by the
number of sub-tasks.

    .venv/bin/python scripts/analysis/plot_continual_lineplots.py             # main, from the saved data
    .venv/bin/python scripts/analysis/plot_continual_lineplots.py --extract   # re-read the runs first
    .venv/bin/python scripts/analysis/plot_continual_lineplots.py --appendix  # the appendix figures

    -> paper/visuals/final/continual_main.{pdf,png}   gymnax noise (10) and action reversal, MiniGrid,
                                                      Kinetix, cheetah noise (10) and action reversal
       paper/visuals/final/continual_main.md          each panel's runs and phase-end scores
       paper/visuals/final/data/continual_main.{npz,json}   what the figure is drawn from
    (paper = projects/iclr_2027/paper)

The main figure is built in two steps, as plot_noncontinual_solve.py's:
`--extract` reads the runs through the symlink trees under paper/<suite>/data
(`<family>/continual/<arm>`, README.md beside each) and saves the per-seed
curves, task switches and kept arms; without it only the saved data is read.
The cheetah RL arms there are the ant-PPO-shape runs.

--appendix draws the rest straight from the run trees (not yet moved to final):

       projects/iclr_2027/paper/visuals/continual_gymnax_10task.{pdf,png}  noise, physics
       projects/iclr_2027/paper/visuals/continual_gymnax_2task.{pdf,png}   noise, physics, action reversal
       projects/iclr_2027/paper/visuals/continual_cheetah.{pdf,png}        noise, friction (2 and 10), action reversal
       projects/iclr_2027/paper/visuals/continual_minigrid.{pdf,png}
       projects/iclr_2027/paper/visuals/continual_kinetix.{pdf,png}
       projects/iclr_2027/paper/visuals/continual_lineplots.md             each panel's tree and phase-end scores

A panel is one (family, environment): the CENTROID curve that
`make_lineplot.py --phase continual --metric centroid` draws, loaded through
its `load_report` (same runs, arms, superseded-arm checks, rolling median over
1% of the records, bootstrap band), with a dashed line at every task switch
from its `phase_edges`.

A family draws ONE of OpenES/NES and ONE of PBT-PPO N=8/N=2: the paper treats
each pair as one method, so each is ONE legend entry, `ES` and `PBT-PPO`
(the markdown's Kept column says which variant). In the main figure ES is NES,
except on Kinetix where it is plain OpenES (`es`); elsewhere, and for
PBT everywhere, the variant is the one with the higher Cum. elite over the
family's continual cells (es_arm.py). Since 2026-09-24 the ES variant is
es_arm.kept_for(tree): NES everywhere except Kinetix (es_arm.KEPT_BY_SUITE),
the same choice as before, now in one place.
An arm with a trial still running is left out of its family until every trial
it started has finished, rather than drawn from its first seeds.

    Phase end  the unsmoothed curve's mean over the last 10% of every phase,
               averaged over the phases, then over seeds: how much of each
               sub-task a method has by the time the task changes. Needs no
               threshold; read it against the cheetah's do-nothing floor, ~677.
"""

from __future__ import annotations

import argparse
import datetime
import json
import pathlib
import sys
import time
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt                            # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import plot_noncontinual_solve as ncs                      # noqa: E402

lp, PROJECT = ncs.lp, ncs.PROJECT
OUT = PROJECT / 'paper/visuals'
FINAL = OUT / 'final'
DATA = FINAL / 'data' / 'continual_main'     # .npz curves, .json the rest

# (figure, tree, cell, title), in drawing order. Each block is one
# finish_iclr.sh family, named in its comment, read from that family's ROOT. A
# main-text panel is listed again under its appendix figure.
PANELS = [
    # main text, two rows of five: the noise row (`noise`,
    # `cheetah_noise05_t10`, `minigrid`) over the action-reversal row
    # (`actions`, `cheetah_action`, `kinetix`); MiniGrid and Kinetix, the last
    # column, only share a column. Read from the paper's own symlink trees.
    ('main', 'paper/gymnax/data/noise_10task', 'CartPole_v1_sigma1.0', 'CartPole, noise'),
    ('main', 'paper/gymnax/data/noise_10task', 'Acrobot_v1_sigma1.0', 'Acrobot, noise'),
    ('main', 'paper/gymnax/data/noise_10task', 'MountainCar_v0_sigma0.1', 'MountainCar, noise'),
    ('main', 'paper/mjx/cheetah/data/noise_10task', 'cheetah_noise', 'HalfCheetah, noise'),
    ('main', 'paper/minigrid/data', 'MiniGrid_8x8_16x16', 'MiniGrid 8x8 / 16x16'),
    ('main', 'paper/gymnax/data/actions_2task', 'CartPole_v1_sigma1.0', 'CartPole, action reversal'),
    ('main', 'paper/gymnax/data/actions_2task', 'Acrobot_v1_sigma1.0', 'Acrobot, action reversal'),
    ('main', 'paper/gymnax/data/actions_2task', 'MountainCar_v0_sigma1.0', 'MountainCar, action reversal'),
    ('main', 'paper/mjx/cheetah/data/actions_2task', 'cheetah_action', 'HalfCheetah, action reversal'),
    ('main', 'paper/kinetix/data', 'Kinetix20', 'Kinetix, 20 levels'),

    # `tradeoff`: the two-sub-task noise and physics families, read by
    # plot_stability_plasticity.py for Figure 2 (plot_continual_combined.py
    # --part tradeoff) beside the main panels. Not a lineplot figure.
    ('tradeoff', 'paper/gymnax/data/noise_2task', 'CartPole_v1_sigma0.5', 'CartPole, noise, two tasks'),
    ('tradeoff', 'paper/gymnax/data/noise_2task', 'Acrobot_v1_sigma0.5', 'Acrobot, noise, two tasks'),
    ('tradeoff', 'paper/gymnax/data/noise_2task', 'MountainCar_v0_sigma0.05', 'MountainCar, noise, two tasks'),
    ('tradeoff', 'paper/mjx/cheetah/data/noise_2task', 'cheetah_noise', 'HalfCheetah, noise, two tasks'),
    ('tradeoff', 'paper/gymnax/data/physics_2task', 'CartPole_v1_sigma1.0', 'CartPole, physics, two tasks'),
    ('tradeoff', 'paper/gymnax/data/physics_2task', 'Acrobot_v1_sigma1.0', 'Acrobot, physics, two tasks'),
    ('tradeoff', 'paper/gymnax/data/physics_2task', 'MountainCar_v0_sigma1.0', 'MountainCar, physics, two tasks'),
    ('tradeoff', 'paper/mjx/cheetah/data/physics_2task', 'cheetah_friction', 'HalfCheetah, friction, two tasks'),

    # `noise`
    ('gymnax_10task', 'runs_centroid/gymnax', 'CartPole_v1_sigma1.0', 'CartPole, noise'),
    ('gymnax_10task', 'runs_centroid/gymnax', 'Acrobot_v1_sigma1.0', 'Acrobot, noise'),
    ('gymnax_10task', 'runs_centroid/gymnax', 'MountainCar_v0_sigma0.1', 'MountainCar, noise'),
    # `physics`
    ('gymnax_10task', 'runs_param_centroid/gymnax', 'CartPole_v1_sigma1.0', 'CartPole, physics'),
    ('gymnax_10task', 'runs_param_centroid/gymnax', 'Acrobot_v1_sigma1.0', 'Acrobot, physics'),
    ('gymnax_10task', 'runs_param_centroid/gymnax', 'MountainCar_v0_sigma1.0', 'MountainCar, physics'),

    # `noise_2task`
    ('gymnax_2task', 'runs_noise_2task/gymnax', 'CartPole_v1_sigma0.5', 'CartPole, noise'),
    ('gymnax_2task', 'runs_noise_2task/gymnax', 'Acrobot_v1_sigma0.5', 'Acrobot, noise'),
    ('gymnax_2task', 'runs_noise_2task/gymnax', 'MountainCar_v0_sigma0.05', 'MountainCar, noise'),
    # `physics_2task`
    ('gymnax_2task', 'runs_param_2task/gymnax', 'CartPole_v1_sigma1.0', 'CartPole, physics'),
    ('gymnax_2task', 'runs_param_2task/gymnax', 'Acrobot_v1_sigma1.0', 'Acrobot, physics'),
    ('gymnax_2task', 'runs_param_2task/gymnax', 'MountainCar_v0_sigma1.0', 'MountainCar, physics'),
    # `actions`
    ('gymnax_2task', 'runs_actions/gymnax', 'CartPole_v1_sigma1.0', 'CartPole, action reversal'),
    ('gymnax_2task', 'runs_actions/gymnax', 'Acrobot_v1_sigma1.0', 'Acrobot, action reversal'),
    ('gymnax_2task', 'runs_actions/gymnax', 'MountainCar_v0_sigma1.0', 'MountainCar, action reversal'),

    # `cheetah_noise025`, `cheetah_friction_2task`, `cheetah_action`
    ('cheetah', 'runs_mjx_noise025/mjx', 'cheetah_noise', 'Noise, 2 sub-tasks'),
    ('cheetah', 'runs_mjx_2task/mjx', 'cheetah_friction', 'Friction, 2 sub-tasks'),
    ('cheetah', 'runs_mjx_action/mjx', 'cheetah_action', 'Action reversal'),
    # `cheetah_noise05_t10`, `cheetah_friction`
    ('cheetah', 'runs_mjx_noise05_t10/mjx', 'cheetah_noise', 'Noise, 10 sub-tasks'),
    ('cheetah', 'runs_mjx/mjx', 'cheetah_friction', 'Friction, 10 sub-tasks'),

    # `minigrid`, `kinetix`
    ('minigrid', 'runs_centroid/minigrid', 'MiniGrid_8x8_16x16', 'MiniGrid 8x8 / 16x16'),
    ('kinetix', 'runs_kinetix_paper/kinetix', 'Kinetix20', 'Kinetix, 20 levels'),
]
# Per figure: panels a row, (width, height) of a panel in inches, legend columns
# (None = one row). The one-panel figures are drawn wide for their 20 phases.
LAYOUT = {'main': (5, (3.2, 2.6), None),
          'gymnax_10task': (3, (3.2, 2.6), None),
          'gymnax_2task': (3, (3.2, 2.6), None),
          'cheetah': (3, (3.2, 2.6), None),
          'minigrid': (1, (6.5, 2.4), 4),
          'kinetix': (1, (6.5, 2.4), 4)}
TAIL = 0.10             # Phase end reads the last this fraction of every phase
# The variant behind a pair's one legend entry, for the markdown.
KEPT = {'es': 'OpenES', 'nes': 'NES', 'pbt': 'PBT N=8', 'pbt2': 'PBT N=2'}


RUNNING = 3600         # s: a trial with no results whose train.log moved this recently


def complete_arms(root, cells):
    """ncs.ARMS with finished trials in `cells` and none still training. A trial
    counts as training when it has no results and its train.log was written in
    the last RUNNING seconds; an older one is a dead run (trac Acrobot trial_6
    in runs_centroid, 2026-09-09) that make_lineplot skips anyway."""
    arms, now = [], time.time()
    for m in ncs.ARMS:
        trials = [t for c in cells for t in (root / 'continual' / m / c).glob('trial_*')]
        done = [t for t in trials if (t / 'training_metrics.json').exists()]
        running = [t for t in trials if t not in done and (t / 'train.log').exists()
                   and now - (t / 'train.log').stat().st_mtime < RUNNING]
        if done and not running:
            arms.append(m)
        elif running:
            print(f'note: leaving out {m} under {root}: {len(done)} trials finished, '
                  f'{len(running)} still training')
    return arms


def load_tree(tree, cells, es_kept=None):
    """`(data, per_gen, edges_in_generations, [kept ES arm, kept PBT arm], cum)`
    for `cells` of one continual tree. Each tree here is one finish_iclr.sh
    family, so ES vs NES (unless `es_kept` fixes it) and PBT N=8 vs N=2 are each
    es_arm.py's pick over these cells (higher Cum. elite); the losers are
    dropped from `data` and the kept arm is filed under its pair's first name
    (`es`, `pbt`), one legend entry a pair."""
    root = PROJECT / tree
    arms = complete_arms(root, cells)
    args = lp.parse_args([str(root), '--phase', 'continual', '--cells', *cells,
                          '--metric', 'centroid', '--methods', *arms, '--out', '-'])
    rep = lp.load_report(args)
    pairs = (ncs.es_arm.ARMS, ncs.es_arm.PBT_ARMS)
    cum = ncs.es_arm.load(root, 'continual', cells, arms=sum(pairs, ()))
    kept = [ncs.es_arm.pick(cum, pair)[0] if set(pair) <= set(arms)
            else next((a for a in pair if a in arms), None) for pair in pairs]
    if es_kept:
        assert es_kept in arms, f'{es_kept} has no finished runs under {root}'
        kept[0] = es_kept
    for by_method in rep.data.values():
        for pair, k in zip(pairs, kept):
            for arm in pair:
                if arm != k:
                    by_method.pop(arm, None)
            if k and k != pair[0] and k in by_method:
                by_method[pair[0]] = by_method.pop(k)
    per_gen = rep.per_gen or 1.0
    return rep.data, per_gen, np.asarray(rep.edges, dtype=float) / per_gen, kept, cum


def phase_end(gens, curves, edges):
    """Mean over seeds of the per-phase mean of the last TAIL of each phase."""
    per_phase = []
    for a, b in zip(edges[:-1], edges[1:]):
        window = (gens > b - TAIL * (b - a)) & (gens <= b)
        if not window.any():                   # a phase shorter than one record
            window = gens == gens[gens <= b].max()
        per_phase.append(curves[:, window].mean(axis=1))
    return float(np.mean(per_phase))


def build_column(arms, edges):
    """`{'curves', 'edges', 'end'}` for one panel from `{arm: (generations,
    seeds x records)}`, smoothed as make_lineplot does."""
    col = {'curves': {}, 'edges': edges[1:-1], 'end': {}}
    for m, (gens, curves) in arms.items():
        col['curves'][m] = (gens, lp.smooth(curves, max(int(curves.shape[1] * 0.01) | 1, 1)))
        col['end'][m] = phase_end(gens, curves, edges)
    return col


def write_markdown(figures, methods, path):
    lines = ['# continual_lineplots', '',
             'The continual centroid curves. Built by '
             '`scripts/analysis/plot_continual_lineplots.py`; change the panels in its '
             '`PANELS` table.', '',
             f'**Phase end**: the curve\'s mean over the last {TAIL:.0%} of every phase, '
             'averaged over phases, then seeds. Cheetah do-nothing floor ~677. The ES '
             'and PBT-PPO columns are the variant in Kept (module docstring for the '
             'rule). `--`: arm not drawn (not run, or still training).', '']
    for fig, panels in figures.items():
        drawn = [m for m in methods if any(m in col['curves'] for *_, col in panels)]
        lines += [f'## {fig}', '',
                  '| Panel | Tree | Cell | Kept | ' + ' | '.join(lp.METHOD_STYLE[m]['label']
                                                            for m in drawn) + ' |',
                  '|' + '---|' * (len(drawn) + 4)]
        for tree, cell, title, col in panels:
            cells = [ncs._cum(col['end'][m]) if m in col['end'] else '--' for m in drawn]
            lines.append(f'| {title} | `{tree}` | `{cell}` | '
                         + ', '.join(KEPT[k] for k in col['kept']) + ' | '
                         + ' | '.join(cells) + ' |')
        lines.append('')
    path.write_text('\n'.join(lines) + '\n')
    print(f'wrote {path}')


def _key(tree, cell):
    return f'{tree}|{cell}'


def _cells(fig_filter):
    cells = defaultdict(list)
    for fig, tree, cell, _title in PANELS:
        if fig_filter(fig) and cell not in cells[tree]:
            cells[tree].append(cell)
    return cells


def extract():
    """Read the main figure's runs and write DATA.npz (`<tree>|<cell>|<arm>|gens`
    and `...|curves`, unsmoothed, kept arms only, filed under `es`/`pbt`) and
    DATA.json (task switches and kept arms a tree, Cum. elite a trial, the runs
    each arm resolves to)."""
    arrays = {}
    meta = {'edges': {}, 'kept': {}, 'elite_cum': {}, 'sources': {}}
    for tree, cells in _cells(lambda f: f == 'main').items():
        data, per_gen, edges, kept, cum = load_tree(
            tree, cells, es_kept=ncs.es_arm.kept_for(tree))
        meta['edges'][tree], meta['kept'][tree] = edges.tolist(), kept
        for cell in cells:
            meta['elite_cum'][_key(tree, cell)] = cum.get(cell.split('_sigma')[0], {})
            for m, (x, curves) in data[cell.split('_sigma')[0]].items():
                arrays[f'{_key(tree, cell)}|{m}|gens'] = x / per_gen
                arrays[f'{_key(tree, cell)}|{m}|curves'] = np.asarray(curves, dtype=np.float32)
        for arm_dir in sorted((PROJECT / tree / 'continual').iterdir()):
            meta['sources'][f'{tree}/continual/{arm_dir.name}'] = str(
                arm_dir.resolve().relative_to(PROJECT))
    meta['extracted'] = datetime.date.today().isoformat()
    DATA.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(f'{DATA}.npz', **arrays)
    pathlib.Path(f'{DATA}.json').write_text(json.dumps(meta, indent=1) + '\n')
    print(f'wrote {DATA}.npz, {DATA}.json')


def load_main():
    """`[(tree, cell, title, col)]` for the main figure, from DATA."""
    if not pathlib.Path(f'{DATA}.npz').exists():
        sys.exit(f'no {DATA}.npz: run with --extract first')
    meta = json.loads(pathlib.Path(f'{DATA}.json').read_text())
    arms = defaultdict(dict)
    with np.load(f'{DATA}.npz') as npz:
        for name in npz.files:
            tree, cell, m, field = name.split('|')
            if field == 'gens':
                arms[tree, cell][m] = (npz[name],
                                       npz[f'{tree}|{cell}|{m}|curves'].astype(float))
    panels = []
    for fig, tree, cell, title in PANELS:
        if fig == 'main':
            col = build_column(arms[tree, cell], np.asarray(meta['edges'][tree]))
            col['kept'] = [k for k in meta['kept'][tree] if k]
            panels.append((tree, cell, title, col))
    return panels


def load_appendix():
    """`{figure: [(tree, cell, title, col)]}` for the appendix, from the runs."""
    loaded = {tree: load_tree(tree, cs) for tree, cs in _cells(lambda f: f != 'main').items()}
    figures = defaultdict(list)
    for fig, tree, cell, title in PANELS:
        if fig == 'main':
            continue
        data, per_gen, edges, kept, _ = loaded[tree]
        env = cell.split('_sigma')[0]
        col = build_column({m: (x / per_gen, c) for m, (x, c) in data[env].items()}, edges)
        col['kept'] = [k for k in kept if k]
        figures[fig].append((tree, cell, title, col))
    return figures


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--extract', action='store_true',
                    help='re-read the main figure\'s runs into the saved data first')
    ap.add_argument('--appendix', action='store_true',
                    help='draw the appendix figures (from the run trees) instead')
    args = ap.parse_args()
    plt.rcParams.update({'font.size': 9})
    # `pbt` stands for either PBT size here (load_tree); `es` is already `ES`.
    lp.METHOD_STYLE['pbt'] = {**lp.METHOD_STYLE['pbt'], 'label': 'PBT-PPO'}
    if args.appendix:
        figures, out, md = load_appendix(), OUT, OUT / 'continual_lineplots.md'
    else:
        if args.extract:
            extract()
        figures, out, md = {'main': load_main()}, FINAL, FINAL / 'continual_main.md'
    for fig, panels in figures.items():
        for tree, cell, title, col in panels:
            print(f'{fig:14s} {title:30s} kept {"/".join(col["kept"])}  phase end  '
                  + '  '.join(f'{m}={ncs._cum(v)}' for m, v in col['end'].items()))
    union = {m for panels in figures.values() for *_, col in panels for m in col['curves']}
    methods = [m for m in lp.METHOD_ORDER if m in union]
    out.mkdir(parents=True, exist_ok=True)
    for fig, panels in figures.items():
        ncols, panel, legend_ncol = LAYOUT[fig]
        # One block a figure (ncs.draw's blocks API, 2026-09-19).
        ncs.draw([(None, [(title, col, None) for _tree, _cell, title, col in panels],
                   ncols, panel)],
                 methods, out / f'continual_{fig}', legend_ncol)
    write_markdown(figures, methods, md)
    return 0


if __name__ == '__main__':
    sys.exit(main())
