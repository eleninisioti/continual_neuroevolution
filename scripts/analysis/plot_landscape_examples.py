"""The four behaviours of the intro schematic, one return-landscape slice each,
on a two-sub-task panel where all four occur (--cell: CartPole, observation
offset 0.5, since 2026-09-20; MountainCar, offset 0.05, before). An appendix paper figure, the chosen-example companion of
`plot_landscape_slices.py` (same slice and units, flat regions instead of its ink overlay; --zoom widens every
panel's window by one factor, 1.5 by default, so the plateaus do not fill them). A main-paper figure: a one-line
key of the regions and markers since 2026-09-21, and since 2026-09-20 each panel carries its role (ROLE_LABEL) under it,
so the caption only defines the three names.

    # slice the chosen checkpoints (GPU)
    .venv/bin/python scripts/analysis/plot_landscape_examples.py --compute --gpus 3
    # collect and draw
    .venv/bin/python scripts/analysis/plot_landscape_examples.py --extract

    -> paper/<tree>/results/centroid/landscape_examples/<cell>.json, <cell>_<row>.npz
       paper/visuals/final/data/landscape_examples_centroid.{json,npz}
       paper/visuals/final/appendix/landscape_examples_centroid.{pdf,png,md}

Every post-switch checkpoint of every trial is classed on the generalist
figure's scale: generalist (learned >= 50% of the best gain, keeps >= 90% of it
on the previous sub-task), switching (learned, does not keep), stuck (not
learned, but >= 50% on the previous sub-task) or neither. Each method is shown
in one role (ROLES; the continual RL column is ReDo-PPO since 2026-09-25, PBT-PPO
before); its example is, among its checkpoints of that role from
phase LATE on whose run had solved the previous sub-task before the switch, the
one nearest their mean
(shown, previous): a typical instance. A stuck example must also have solved
the new sub-task earlier in the run; its x axis runs from the checkpoint before
the switch to that earlier solution (x = 1), since a stuck run barely
moves; that solution is shown by its own region, not a marker. The y axis is
ls.REL_BETA times the weight norm (slice_plane rel_beta).
The .md table gives how often each method is in its role (the text quotes it).
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt                            # noqa: E402
import matplotlib.patheffects as pe                        # noqa: E402

import plot_landscape_slices as pls                         # noqa: E402

pgs, pgo, plm, ls = pls.pgs, pls.pgo, pls.plm, pls.ls
PROJECT, FINAL = pls.PROJECT, pls.FINAL
PASS = 'results/centroid/landscape_examples'
TREE = 'paper/gymnax/data/noise_2task'
# --cell picks the panel; CartPole (2026-09-20) is Figure 1, MountainCar the
# 2026-09-19 version. Outputs of a non-default cell carry its name.
CELLS = {'cartpole': 'CartPole_v1_sigma0.5', 'mountaincar': 'MountainCar_v0_sigma0.05',
         'acrobot': 'Acrobot_v1_sigma0.5'}
# Per-cell roles where the default does not fit (2026-09-25 draft): on Acrobot
# the GA keeps both tasks at 92% of switches, and TRAC-PPO is the variant that
# un-sticks PPO (6 of 180 tasks failed against PPO's 51).
CELL_ROLES = {'acrobot': {'es': 'generalist', 'ga': 'generalist', 'ppo': 'stuck',
                          'trac': 'switching'}}
CELL = NAME = STEM = DATA = None


WHICH = None


def configure(which, all_methods=False):
    global CELL, NAME, STEM, DATA, ROLES, WHICH
    WHICH = which
    CELL = CELLS[which]
    if which in CELL_ROLES:
        ROLES = CELL_ROLES[which]
    if all_methods:
        ROLES = ROLES_ALL
    NAME = ('landscape_examples_centroid' + ('' if which == 'cartpole' else f'_{which}')
            + ('_all' if all_methods else ''))
    STEM = FINAL / 'appendix' / NAME
    DATA = FINAL / 'data' / NAME
# Figure 1's roles. Four columns since 2026-09-20: ES, GA, PPO and ONE continual
# PPO variant standing for the four. ReDo-PPO since 2026-09-25: the most plastic
# RL method (median LA 0.94 over the 18 settings, PPO 0.84) whose basin is NOT
# wider than PPO's (x1.15, n.s., basin_width_return), so its panel shows what the
# continual variants do: restore plasticity (it re-initialises dormant units, so
# the network stays trainable) and switch, in a basin as narrow as PPO's. PBT-PPO
# from 2026-09-21 (the widest RL basin, x2.0, which undercut that message);
# TRAC-PPO on 2026-09-20. The appendix landscapes figure has every method.
# --all draws the seven-column version under the _all suffix.
ROLES = {'es': 'generalist', 'ga': 'switching', 'ppo': 'stuck', 'redo': 'switching'}
ROLES_ALL = {**ROLES, 'pbt': 'switching', 'cchain': 'switching', 'trac': 'switching'}
TITLE = {'pbt': 'PBT-PPO'}          # the paper's name everywhere else
# The role goes under its panel, so the caption only has to define the three
# names (2026-09-20); two lines each, so the first line aligns across panels.
# Trade-off wording, not generalist/specialist (the paper drops the terms, 2026-09-22).
# One line since 2026-09-25: the (stable)/(plastic) glosses were dropped.
ROLE_LABEL = {'generalist': 'keeps both', 'switching': 'switches', 'stuck': 'stuck'}
PANEL_LABEL = {}
# Movement arrow: a green (Dark2) no region uses (light blue / orange / dark grey),
# with a white halo (2026-09-25; violet #6a3d9a before, too dull on dark grey; magenta tried).
ARROW = '#1b9e77'
# Only checkpoints from this phase on are candidates (the last five of the 20,
# the window basin_width_main pools): the basin of PBT-PPO is 2.4x PPO's at its
# first checkpoint and 1.45x late (basin_width_evolution), so an early example
# misstates the converged width.
LATE = 15
# Pinned examples (2026-09-25): PPO and ReDo-PPO at the SAME switch of the same
# trial (same seed, 55), one where PPO is stuck and ReDo-PPO relearns. At task 15
# of trial 7 PPO stays at return 9 for all 150 updates with 56% dormant units
# and zero entropy; ReDo-PPO starts from 10 with 9% dormant and reaches 500.
# Chosen from every stuck PPO switch on the panel (at 12 of the 14 ReDo-PPO
# relearns at the same point); the pair shows the mechanism, not a typical one.
PICK = {'cartpole': {'ppo': (7, 15), 'redo': (7, 15)}}
# Flat regions, markers and key: pls.regions / BEFORE / AFTER / key_handles
# (2026-09-21), shared with the appendix landscapes figure.


def census():
    """(floor, best, {row: dict(run, k, shown, previous, share, n)})."""
    panel = json.loads(pgs.DATA.read_text())['panels'][f'{TREE}|{CELL}']
    floor = pls.FLOOR[CELL.split('_sigma')[0]]
    best = max(pgs.trial_scores(v['trials'], CELL)[:, 0].mean()
               for v in panel['arms'].values())
    out = {}
    for row, want in ROLES.items():
        v = panel['arms'][row]
        pts, total = [], 0
        for t in v['trials']:
            th = None
            for k, (s, p) in enumerate(zip(t['shown'], t['previous'])):
                if p is None:
                    continue
                total += 1
                if pls.role(s, p, floor, best) != want or not pls.learned(t, k - 1, floor, best):
                    continue
                if k < LATE:
                    continue
                toward = None
                if want == 'stuck':
                    toward = next((j for j in range(k - 2, -1, -2)
                                   if pls.learned(t, j, floor, best)), None)
                    if toward is None:
                        continue
                if th is None:
                    z = np.load(PROJECT / t['run'] / 'checkpoints.npz')
                    th = z[next(s_ for s_ in pls.SOURCES if s_ in z.files)]
                if pls.axis_phase(th, k) is not None:
                    pts.append((t['run'], k, s, p, toward))
        pin = PICK.get(WHICH, {}).get(row)
        if pin is not None:
            pts = [q for q in pts if (int(q[0].rsplit('_', 1)[1]), q[1]) == pin]
            if not pts:
                sys.exit(f'{row}: pinned switch {pin} is not a {want} candidate')
        centre = np.mean([q[2:4] for q in pts], axis=0)
        run, k, s, p, toward = min(pts, key=lambda q: (np.hypot(q[2] - centre[0], q[3] - centre[1]),
                                                       -q[1]))
        n_role = sum(pls.role(s_, p_, floor, best) == want for t in v['trials']
                     for s_, p_ in zip(t['shown'], t['previous']) if p_ is not None)
        out[row] = dict(arm=v['arm'], role=want, run=run, k=k, shown=s, previous=p, toward=toward,
                        share=n_role / total, n=total, trials=len(v['trials']),
                        trials_with=sum(any(pls.role(s_, p_, floor, best) == want
                                            for s_, p_ in zip(t['shown'], t['previous'])
                                            if p_ is not None) for t in v['trials']))
    return floor, float(best), out


def suffix(zoom):
    return '' if zoom == 1 else f'_z{zoom:g}'


def compute(force, zoom):
    floor, best, picks = census()
    _, _, n, episodes, chunk = ls.spec(TREE, CELL)
    out = PROJECT / TREE / PASS
    out.mkdir(parents=True, exist_ok=True)
    jpath = out / f'{CELL}.json'
    blob = json.loads(jpath.read_text()) if jpath.exists() else {}
    blob.update(tree=TREE, cell=CELL, grid=n, episodes=episodes, window=ls.WINDOW)
    blob.setdefault('rows', {})
    for row, c in picks.items():
        done = blob['rows'].get(row, {})
        if (not force and (out / f'{CELL}_{row}{suffix(zoom)}.npz').exists()
                and (done.get('run'), done.get('k'), done.get('rel_beta'), done.get('toward'),
                     done.get('zoom', 1))
                    == (c['run'], c['k'], ls.REL_BETA, c['toward'], zoom)):
            print(f'{row}: done, skipped', flush=True)
            continue
        d = PROJECT / c['run']
        z = np.load(d / 'checkpoints.npz')
        src = next(s for s in pls.SOURCES if s in z.files)
        j = pls.axis_phase(z[src], c['k'])
        # Stuck: the x axis runs to the run's own earlier solution of the new
        # sub-task (x = 1), since a stuck run barely moves over the phase.
        axis = (c['k'] - 1, c['toward']) if c['toward'] is not None else (j - 1, j)
        trial = int(d.name.rsplit('_', 1)[1])
        al, be, grids, at, drift = ls.slice_plane(d, src, c['k'] - 1, trial,
                                                  n, episodes, chunk, axis=axis,
                                                  rel_beta=ls.REL_BETA, zoom=zoom)
        np.savez(out / f'{CELL}_{row}{suffix(zoom)}.npz', alphas=al, betas=be,
                 previous=grids['at_risk'], shown=grids['trained_next'])
        blob['rows'][row] = dict(arm=c['arm'], run=c['run'], k=c['k'], source=src,
                                 axis_phase=j, drift=drift, rel_beta=ls.REL_BETA,
                                 toward=c['toward'], zoom=zoom, next_xy=at['next'],
                                 previous_at=at['at_risk'], shown_at=at['trained_next'])
        jpath.write_text(json.dumps(blob, indent=1) + '\n')
        print(f'{row}: {c["run"]} k={c["k"]} ({c["role"]}); record shown {c["shown"]:.4g} '
              f'prev {c["previous"]:.4g}, slice at k shown {at["trained_next"][1]:.4g} '
              f'prev {at["at_risk"][1]:.4g}', flush=True)


def extract(zoom):
    floor, best, picks = census()
    jpath = PROJECT / TREE / PASS / f'{CELL}.json'
    blob = json.loads(jpath.read_text())
    meta = dict(extracted=datetime.date.today().isoformat(), tree=TREE, cell=CELL,
                floor=floor, best=best, zoom=zoom, rows={})
    arrays = {}
    for row, c in picks.items():
        got = blob['rows'].get(row)
        if got is None or (got['run'], got['k'], got.get('zoom', 1)) != (c['run'], c['k'], zoom):
            sys.exit(f'{jpath}: {row} not sliced at {c["run"]} k={c["k"]} zoom {zoom}; run --compute')
        z = np.load(jpath.parent / f'{CELL}_{row}{suffix(zoom)}.npz')
        for name in ('previous', 'shown'):
            arrays[f'{row}/{name}'] = z[name].astype(np.float32)
        arrays[f'{row}/alphas'], arrays[f'{row}/betas'] = z['alphas'], z['betas']
        meta['rows'][row] = {**c, **{k: got.get(k) for k in ('axis_phase', 'drift', 'next_xy')}}
    DATA.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(DATA.with_suffix('.npz'), **arrays)
    DATA.with_suffix('.json').write_text(json.dumps(meta, indent=1) + '\n')
    print(f'wrote {DATA}.json/.npz')


def plot(width, fs):
    meta = json.loads(DATA.with_suffix('.json').read_text())
    arrays = np.load(DATA.with_suffix('.npz'))
    plt.rcParams.update({'font.size': fs, 'axes.linewidth': 0.5})
    scale = meta['best'] - meta['floor']
    rows = [m for m in pls.ORDER if m in ROLES]         # ES first, then the paper's order
    left_in, right_in, top_in, gap_in = 0.04, 0.04, 0.2, 0.08
    key_in = 0.2                                       # one-line key under everything
    bottom_in = key_in + 0.06 + 1.25 * (fs - 0.5) / 72   # one line of role label
    cell = (width - left_in - right_in - gap_in * (len(rows) - 1)) / len(rows)
    height = top_in + bottom_in + cell * 0.75
    fig = plt.figure(figsize=(width, height))
    md = [f'# {NAME}', '', f'{TREE} {CELL}; see the docstring of '
          '`scripts/analysis/plot_landscape_examples.py`. '
          f'Data extracted {meta["extracted"]}.', '',
          '| Method | role | share of switches | trials with one | run | k | shown / previous |',
          '|' + '---|' * 7]
    for c, m in enumerate(rows):
        r = meta['rows'][m]
        ax = fig.add_axes([(left_in + c * (cell + gap_in)) / width, bottom_in / height,
                           cell / width, cell * 0.75 / height])
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_linewidth(0.4)
            s.set_color('0.65')
        al, be = arrays[f'{m}/alphas'], arrays[f'{m}/betas']
        prev = (arrays[f'{m}/previous'] - meta['floor']) / scale
        shown = (arrays[f'{m}/shown'] - meta['floor']) / scale
        ax.imshow(pls.regions(prev, shown), origin='lower',
                  extent=[al[0], al[-1], be[0], be[-1]], aspect='auto',
                  interpolation='nearest')
        still = r['axis_phase'] != r['k']
        after = r['next_xy'] or ((0, 0) if still else (1, 0))
        # Movement arrow (2026-09-25): the run's move over the task, circle to
        # diamond, curved for legibility only (the plane holds the two endpoints,
        # not the path). A stuck run does not move and gets none. No return
        # arrow: the next checkpoint lies off the plane (1.1-1.4 steps for the
        # GA and ReDo-PPO) and does not project back onto the circle.
        if r['role'] != 'stuck':
            ax.annotate('', xy=tuple(after), xytext=(0, 0),
                        arrowprops=dict(arrowstyle='-|>,head_length=0.38,head_width=0.19',
                                        lw=0.8, color=ARROW, shrinkA=4.0, shrinkB=2.6,
                                        connectionstyle='arc3,rad=-0.38',
                                        path_effects=[pe.withStroke(linewidth=1.7,
                                                                    foreground='white')]),
                        zorder=4)
        ax.plot(0, 0, **pls.BEFORE, zorder=5)
        ax.plot(*after, **pls.AFTER, zorder=6)
        # A stuck example's earlier solution of the new sub-task sits at (1, 0);
        # its orange region marks it, no star (2026-09-20).
        ax.set_xlim(al[0], al[-1])
        ax.set_ylim(be[0], be[-1])
        ax.set_title(TITLE.get(m, pgs.label_of(m)), fontsize=fs, pad=3)
        # On a common baseline: xlabel aligns the text box top, which moves
        # with the letters' ascenders (2026-09-25).
        ax.text(0.5, -(3 + fs) / 72 / (cell * 0.75), PANEL_LABEL.get(m, ROLE_LABEL[r['role']]),
                transform=ax.transAxes, ha='center', va='baseline', fontsize=fs - 0.5,
                color='0.25')
        md.append(f'| {pgs.label_of(m)} | {r["role"]} | {r["share"]:.3f} (of {r["n"]}) '
                  f'| {r["trials_with"]}/{r["trials"]} | {r["run"]} | {r["k"]} '
                  f'| {r["shown"]:.4g} / {r["previous"]:.4g} |')
    handles = pls.key_handles()
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(0.5, 0), ncol=len(handles),
               frameon=False, fontsize=fs - 0.5, handlelength=1.0, handleheight=0.8,
               handletextpad=0.4, columnspacing=1.1, borderaxespad=0.1)
    STEM.parent.mkdir(parents=True, exist_ok=True)
    for ext in ('pdf', 'png'):
        fig.savefig(STEM.with_suffix(f'.{ext}'), dpi=300)
    plt.close(fig)
    STEM.with_suffix('.md').write_text('\n'.join(md) + '\n')
    print(f'wrote {STEM}.pdf/.png/.md ({width:.2f} x {height:.2f} in)')


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--compute', action='store_true', help='slice the examples (GPU)')
    ap.add_argument('--extract', action='store_true', help='collect the slices')
    ap.add_argument('--gpus', default='0')
    ap.add_argument('--force', action='store_true')
    ap.add_argument('--width', type=float, default=5.5)
    ap.add_argument('--font-size', type=float, default=7.0)
    ap.add_argument('--cell', choices=list(CELLS), default='cartpole')
    ap.add_argument('--all', action='store_true', help='all seven methods (ROLES_ALL)')
    ap.add_argument('--zoom', type=float, default=1.5,
                    help='window factor about the plane centre, same for every panel '
                         '(1 = the appendix slices; 1.5 since 2026-09-20)')
    args = ap.parse_args()
    configure(args.cell, args.all)
    if args.compute:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
        os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
        sys.path.insert(0, str(pls.REPO))
        os.chdir(pls.REPO)
        compute(args.force, args.zoom)
        return 0
    if args.extract:
        extract(args.zoom)
    if not DATA.with_suffix('.json').exists():
        sys.exit(f'no {DATA}.json: run with --extract first')
    plot(args.width, args.font_size)
    return 0


if __name__ == '__main__':
    sys.exit(main())
