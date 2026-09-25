"""The return landscape around one switch, for every method on every panel of
the final `generalist_scores_centroid` figure. An appendix paper figure.

    # slice the runs (GPU; only when the runs or the generalist data change)
    .venv/bin/python scripts/analysis/plot_landscape_slices.py --compute --gpus 0
    # collect the slices
    .venv/bin/python scripts/analysis/plot_landscape_slices.py --extract
    # redraw from visuals/final/data/
    .venv/bin/python scripts/analysis/plot_landscape_slices.py

    -> paper/<data root>/results/centroid/landscape_final/<cell>.json, <cell>_<row>.npz
       paper/visuals/final/data/landscape_slices_centroid.{json,npz}
       paper/visuals/final/appendix/landscape_slices_centroid.{pdf,png,md}

Panels, runs, trials and arms are generalist_scores_centroid's (its data file,
so run its extract first): ES = NES, the PBT-PPO it kept. Rows are methods,
columns the panels, grouped by what the switch changes.

The slice is the plane through the centroid checkpoint k-1 (circle),
spanned by the run's own movement over phase k (to checkpoint k, diamond, at
x = 1) and a random orthogonal direction of the same length
(`landscape_slices.slice_plane`), window -1..2 x -1.5..1.5 drifts. Every grid
point is a policy scored with common random numbers on sub-task k-1 (pink, the
previous sub-task) and on sub-task k (the sub-task just trained). The
diamond is therefore one checkpoint of the generalist figure.

Colours are on the generalist figure's scale: a return is rescaled to
(return - untrained) / (best - untrained), `untrained` the panel's FLOOR and
`best` the highest mean shown return of any method in the panel. Since
2026-09-21 the regions are flat (`regions`): blue where the previous sub-task
reaches LEARNED (half) of the best gain, orange the new one, dark both, white
neither; before that, a threshold-free multiplied ink overlay (pink x teal =
indigo) with contours at KEPT.

Checkpoint, the same rule for every method: the post-switch checkpoint whose
(shown, previous) is nearest, in rescaled units, the method's point in the
generalist figure (MiniGrid: only checkpoints ending an 8x8 phase), ties to
the later one; a checkpoint the search did not move from is skipped. A frozen
specialist is instead drawn in the direction it never learns: the checkpoints
ending that sub-task's phases, nearest that direction's mean. Where the run
did not move over phase k (ES at zero fitness variance under action
reversal), the x axis is its movement over the last phase it moved in and
both checkpoints sit at the origin. The corner names
the method's class in the generalist figure (the rule is mirrored in `chosen`;
`--extract` warns where the figure's table disagrees).
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
from matplotlib.colors import to_rgb                        # noqa: E402
from matplotlib.gridspec import GridSpec                    # noqa: E402
from matplotlib.lines import Line2D                         # noqa: E402
from matplotlib.patches import Patch                        # noqa: E402

REPO = pathlib.Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO / 'scripts'))
sys.path.insert(0, str(REPO / 'scripts/analysis'))
import landscape_slices as ls                               # noqa: E402
import plot_generalist_outcomes as pgo                      # noqa: E402
import plot_generalist_scores as pgs                        # noqa: E402
import plot_landscape_main as plm                           # noqa: E402
from plot_stability_plasticity import FLOOR                 # noqa: E402

PROJECT, FINAL = pgs.PROJECT, pgs.FINAL
NAME = 'landscape_slices_centroid'
STEM = FINAL / 'appendix' / NAME
DATA = FINAL / 'data' / NAME
PASS = 'results/centroid/landscape_final'
KEPT, LEARNED = 0.9, 0.5
SOURCES = ('centroid', 'final')          # generalist_checkpoints.SOURCES['centroid']
INK, MUTED = plm.INK, plm.MUTED
CORNER = {'generalist': 'Keeps both', 'switching': 'Switches',
          'frozen': 'Frozen', 'not learned': 'Not learned'}
# Episodes a grid point, where landscape_slices.spec's (4 MiniGrid, 2 HalfCheetah)
# left the slice at checkpoint k visibly off the generalist figure's score.
EPISODES = {'MiniGrid': 64, 'cheetah': 8}
# The paper's method order (make_lineplot.METHOD_ORDER, Figure 2's), except ES
# before GA: Figure 1 and this grid lead with the generalist (user, 2026-09-22).
import make_lineplot as lp                                  # noqa: E402
ORDER = ['es'] + [m for m in lp.METHOD_ORDER
                  if m in ('ga', 'ppo', 'trac', 'redo', 'cchain', 'pbt')]
SPACER = 0.35                 # gap between groups, in cells
ASPECT = 0.75                 # tall layout: cell height / width, to fit a page
# tanh hidden layers: the scale of the weights is part of the policy, so a slice
# whose extent is set by the run's own movement is not comparable across methods
# (the paper's fig:basin argument); left out unless --tanh.
TANH_BODIES = {'HalfCheetah'}


def panel_list():
    """[(group, body, change, data root, cell)] in the generalist figure's order."""
    return [(group, pgo.COLUMNS[c], p[2], p[0], p[1])
            for group, row in pgs.GRID for c, p in enumerate(row) if p]


def moved(th, k):
    return np.linalg.norm(th[k] - th[k - 1]) > 1e-6 * np.linalg.norm(th[k - 1])


def axis_phase(th, k):
    """The last phase j <= k over which the run moved (x axis = th[j] - th[j-1]),
    or None. j < k for a search that stood still over phase k (ES at zero
    fitness variance)."""
    return next((j for j in range(k, 0, -1) if moved(th, j)), None)


def role(s, p, floor, best):
    """One checkpoint's (shown, previous) on the generalist figure's scale:
    generalist, switching, stuck (new sub-task not learned, previous kept) or
    neither."""
    got, kept = (s - floor) / (best - floor), (p - floor) / (best - floor)
    if got >= LEARNED:
        return 'generalist' if (p - floor) / (s - floor) >= KEPT else 'switching'
    return 'stuck' if kept >= LEARNED else 'neither'


def learned(t, j, floor, best):
    """Checkpoint j of trial t solves the sub-task it was trained on."""
    return j >= 0 and (t['shown'][j] - floor) / (best - floor) >= LEARNED


# The generalist figure's class -> the checkpoint role that illustrates it.
ROLE_OF = {'generalist': 'generalist', 'switching': 'switching', 'frozen': 'stuck'}


def choose(trials, parities, target, floor, best, want=None):
    """(trial index into `trials`, k, toward): the post-switch checkpoint with
    k mod 2 in `parities` nearest `target` in rescaled units, ties to the later
    one. With `want`, only checkpoints of that role whose run solved the
    previous sub-task before the switch (all of them if there are none); a
    stuck one also needs an earlier solution of the new sub-task, `toward`."""
    best_key, pick, loose = None, None, None
    for strict in ((True, False) if want else (False,)):
        for i, t in enumerate(trials):
            th = None
            for k, (s, p) in enumerate(zip(t['shown'], t['previous'])):
                if p is None or k % 2 not in parities:
                    continue
                toward = None
                if strict:
                    if role(s, p, floor, best) != want or not learned(t, k - 1, floor, best):
                        continue
                    if want == 'stuck':
                        toward = next((j for j in range(k - 2, -1, -2)
                                       if learned(t, j, floor, best)), None)
                        if toward is None:
                            continue
                if th is None:
                    z = np.load(PROJECT / t['run'] / 'checkpoints.npz')
                    th = z[next(s_ for s_ in SOURCES if s_ in z.files)]
                if axis_phase(th, k) is None and toward is None:
                    continue         # never moved up to k: no drift axis
                d = np.hypot(s - target[0], p - target[1]) / (best - floor)
                key = (round(float(d), 9), -k)
                if best_key is None or key < best_key:
                    best_key, pick = key, (i, k, toward)
        if pick is not None:
            return pick
    return pick


def chosen(tree, cell):
    """(floor, best, {row: dict(run, k, class, ...)}) for one panel, from the
    generalist data, with the generalist figure's class (mirrors
    plot_generalist_scores.main; `extract` checks the two agree)."""
    panel = json.loads(pgs.DATA.read_text())['panels'][f'{tree}|{cell}']
    keep = pgo.ONE_DIRECTION.get(cell)
    pooled = {m: pgs.trial_scores(v['trials'], cell) for m, v in panel['arms'].items()}
    split = {(m, d): pgs.trial_scores(v['trials'], cell, d)
             for m, v in panel['arms'].items() for d in (0, 1)}
    floor = FLOOR[cell.split('_sigma')[0]]
    best = max(s[:, 0].mean() for s in pooled.values())
    best_dir = max(s[:, 0].mean() for s in split.values() if s is not None)
    bar_dir = floor + LEARNED * (best_dir - floor)
    out = {}
    for row, v in panel['arms'].items():
        xm, ym = pooled[row].mean(axis=0)
        learned, kept, label = pgs.outcome(xm, ym, floor, best, KEPT, LEARNED)
        dirs = {d: split[row, d].mean(axis=0) for d in (0, 1) if split[row, d] is not None}
        missed = [d for d, q in dirs.items() if q[0] < bar_dir]
        frozen = len(dirs) == 2 and len(missed) == 1 and dirs[missed[0]][1] >= bar_dir
        klass = 'frozen' if frozen else label
        # A frozen method is drawn in the direction it never learns; the rest
        # at their pooled point.
        parities = (missed[0],) if frozen else (0, 1) if keep is None else (keep,)
        target = dirs[missed[0]] if frozen else (xm, ym)
        i, k, toward = choose(v['trials'], parities, target, floor, best, ROLE_OF.get(klass))
        t = v['trials'][i]
        out[row] = dict(arm=v['arm'], run=t['run'], k=k, toward=toward,
                        shown=t['shown'][k], previous=t['previous'][k],
                        target=[float(target[0]), float(target[1])],
                        learned=float(learned), kept=float(kept), klass=klass)
    return floor, best, out


def compute(rows, panels, force):
    for _, body, change, tree, cell in panel_list():
        if panels and cell not in panels and f'{tree}|{cell}' not in panels:
            continue
        floor, best, picks = chosen(tree, cell)
        _, _, n, episodes, chunk = ls.spec(tree, cell)
        episodes = next((e for pre, e in EPISODES.items() if cell.startswith(pre)), episodes)
        out = PROJECT / tree / PASS
        out.mkdir(parents=True, exist_ok=True)
        jpath = out / f'{cell}.json'
        blob = json.loads(jpath.read_text()) if jpath.exists() else {}
        blob.update(tree=tree, cell=cell, grid=n, episodes=episodes, window=ls.WINDOW)
        blob.setdefault('rows', {})
        for row, c in picks.items():
            if rows and row not in rows:
                continue
            done = blob['rows'].get(row, {})
            if (not force and (out / f'{cell}_{row}.npz').exists()
                    and (done.get('run'), done.get('k'), done.get('rel_beta'), done.get('toward'))
                    == (c['run'], c['k'], ls.REL_BETA, c['toward'])):
                print(f'{tree} {cell} {row}: done, skipped', flush=True)
                continue
            d = PROJECT / c['run']
            z = np.load(d / 'checkpoints.npz')
            src = next(s for s in SOURCES if s in z.files)
            j = axis_phase(z[src], c['k'])
            # Stuck: the x axis runs to the run's own earlier solution of the new
            # sub-task (x = 1, star), since a stuck run barely moves over the phase.
            axis = (c['k'] - 1, c['toward']) if c['toward'] is not None else (j - 1, j)
            trial = int(d.name.rsplit('_', 1)[1])
            al, be, grids, at, drift = ls.slice_plane(d, src, c['k'] - 1, trial,
                                                      n, episodes, chunk, axis=axis,
                                                  rel_beta=ls.REL_BETA)
            np.savez(out / f'{cell}_{row}.npz', alphas=al, betas=be,
                     previous=grids['at_risk'], shown=grids['trained_next'])
            blob['rows'][row] = dict(arm=c['arm'], run=c['run'], k=c['k'], source=src,
                                     axis_phase=j, drift=drift, rel_beta=ls.REL_BETA,
                                     toward=c['toward'], next_xy=at['next'],
                                     previous_at=at['at_risk'], shown_at=at['trained_next'])
            jpath.write_text(json.dumps(blob, indent=1) + '\n')
            print(f'{tree} {cell} {row}: {c["run"]} k={c["k"]} ({c["klass"]}); '
                  f'record shown {c["shown"]:.4g} prev {c["previous"]:.4g}, '
                  f'slice at k shown {at["trained_next"][1]:.4g} '
                  f'prev {at["at_risk"][1]:.4g}', flush=True)


def generalist_classes():
    """{(body, change, method label): class} from the generalist figure's table."""
    path = pgs.STEM.with_suffix('.md')
    if not path.exists():
        return {}
    lines = [ln.split('|')[1:-1] for ln in path.read_text().splitlines() if ln.startswith('|')]
    head = [h.strip() for h in lines[0]]
    if 'class' not in head:
        return {}
    col = {h: head.index(h) for h in ('Task', 'Change', 'Method', 'class')}
    return {(r[col['Task']].strip(), r[col['Change']].strip(), r[col['Method']].strip()):
            r[col['class']].strip() for r in lines[2:] if len(r) == len(head)}


def extract(tanh):
    meta = {'panels': [], 'extracted': datetime.date.today().isoformat(),
            'generalist_scores_extracted': json.loads(pgs.DATA.read_text())['extracted']}
    arrays = {}
    drawn = generalist_classes()
    for group, body, change, tree, cell in panel_list():
        if body in TANH_BODIES and not tanh:
            continue
        i = len(meta['panels'])          # index among the panels kept
        floor, best, picks = chosen(tree, cell)
        jpath = PROJECT / tree / PASS / f'{cell}.json'
        blob = json.loads(jpath.read_text()) if jpath.exists() else {'rows': {}}
        panel = dict(group=group, body=body, change=change, tree=tree, cell=cell,
                     floor=floor, best=float(best), pass_file=str(jpath.relative_to(PROJECT)),
                     grid=blob.get('grid'), episodes=blob.get('episodes'), rows={})
        for row, c in picks.items():
            got = blob['rows'].get(row)
            if got is None:
                sys.exit(f'{jpath}: no {row} slice; run --compute')
            assert (got['run'], got['k']) == (c['run'], c['k']), (
                f'{jpath}: {row} sliced {got["run"]} k={got["k"]}, the generalist data '
                f'now picks {c["run"]} k={c["k"]}; run --compute')
            z = np.load(jpath.parent / f'{cell}_{row}.npz')
            for name in ('previous', 'shown'):
                arrays[f'{i}/{row}/{name}'] = z[name].astype(np.float32)
            arrays[f'{i}/alphas'], arrays[f'{i}/betas'] = z['alphas'], z['betas']
            panel['rows'][row] = {**c, **{k: got.get(k) for k in
                                          ('source', 'axis_phase', 'drift', 'previous_at', 'shown_at',
                                           'next_xy')}}
        for row, c in picks.items():
            theirs = drawn.get((body, change, pgs.label_of(row)))
            if theirs is not None and theirs != c['klass']:
                print(f'WARNING {body}, {change}, {row}: class {c["klass"]} here, '
                      f'{theirs} in {pgs.STEM.name}.md (stale table or a changed rule)')
        meta['panels'].append(panel)
        print(f'{body + ", " + change:32s} ' + ' '.join(
            f'{r}:k{v["k"]}' for r, v in panel['rows'].items()))
    DATA.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(DATA.with_suffix('.npz'), **arrays)
    DATA.with_suffix('.json').write_text(json.dumps(meta, indent=1) + '\n')
    print(f'wrote {DATA}.json/.npz')


# Flat regions (2026-09-21, a reviewer of Figure 1 found the continuous two-ink
# overlay hard to read: partial returns mixed into shades no caption named).
# A point is in a sub-task's region when its rescaled return reaches LEARNED
# (half the best gain); blue/orange stay apart under colour blindness and the
# overlap is a neutral dark, not a third hue. No contours, no family markers.
REGION_PREV, REGION_SHOWN, REGION_BOTH, REGION_NONE = '#9ecae1', '#fdae6b', '#404040', '#ffffff'
# Told apart by shape, since a filled dot vanishes on the dark overlap.
BEFORE = dict(marker='o', ls='', ms=5.4, mfc='white', mec='black', mew=0.6)
AFTER = dict(marker='D', ls='', ms=3.0, mfc='black', mec='white', mew=0.6)


def regions(prev, shown):
    """Scaled returns (0 untrained, 1 best) -> RGB of the four flat regions."""
    p, n = prev >= LEARNED, shown >= LEARNED
    img = np.empty(p.shape + (3,))
    for mask, colour in ((~p & ~n, REGION_NONE), (p & ~n, REGION_PREV),
                         (~p & n, REGION_SHOWN), (p & n, REGION_BOTH)):
        img[mask] = to_rgb(colour)
    return img


def key_handles(before='before the switch', after='after'):
    """The regions and the two checkpoints, for fig.legend."""
    return [Patch(fc=REGION_PREV, ec='none', label='solves the previous task'),
            Patch(fc=REGION_SHOWN, ec='none', label='solves the new task'),
            Patch(fc=REGION_BOTH, ec='none', label='solves both'),
            Line2D([], [], **BEFORE, label=before),
            Line2D([], [], **AFTER, label=after)]


def plot(width, fs, layout, tanh):
    """layout 'tall': methods are columns and panels rows, upright on a page at
    \\linewidth; 'wide': methods are rows, for a page turned sideways."""
    meta = json.loads(DATA.with_suffix('.json').read_text())
    arrays = np.load(DATA.with_suffix('.npz'))
    plt.rcParams.update({'font.size': fs, 'axes.linewidth': 0.5})
    panels = meta['panels']
    arms = [m for m in ORDER if any(m in p['rows'] for p in panels)]
    tall = layout == 'tall'
    drawn = [i for i, p in enumerate(panels) if tanh or p['body'] not in TANH_BODIES]
    ratios, col_of = [], {}
    for j, i in enumerate(drawn):
        if j and panels[i]['group'] != panels[drawn[j - 1]]['group']:
            ratios.append(SPACER)
        col_of[i] = len(ratios)
        ratios.append(1)
    if tall:
        left_in, right_in, top_in, bottom_in = 1.05, 0.04, 0.62, 0.04
        cell_w = (width - left_in - right_in) / (len(arms) + 0.06 * (len(arms) - 1))
        cell_h = cell_w * ASPECT
        height = top_in + bottom_in + cell_h * (sum(ratios) + 0.06 * (len(ratios) - 1))
        fig = plt.figure(figsize=(width, height))
        gs = GridSpec(len(ratios), len(arms), figure=fig, left=left_in / width,
                      right=1 - right_in / width, top=1 - top_in / height,
                      bottom=bottom_in / height, wspace=0.06, hspace=0.06,
                      height_ratios=ratios)
    else:
        left_in, right_in, top_in, bottom_in = 0.75, 0.1, 1.12, 0.08
        cell_in = (width - left_in - right_in) / (sum(ratios) + 0.08 * (len(ratios) - 1))
        height = top_in + bottom_in + len(arms) * cell_in * 1.08
        fig = plt.figure(figsize=(width, height))
        gs = GridSpec(len(arms), len(ratios), figure=fig, left=left_in / width,
                      right=1 - right_in / width, top=1 - top_in / height,
                      bottom=bottom_in / height, wspace=0.08, hspace=0.08,
                      width_ratios=ratios)
    span, titles, heads = {}, [], []
    md = [f'# {NAME}', '',
          'Per panel and method: the checkpoint sliced (nearest the method\'s mean in '
          '`generalist_scores_centroid`), its training-record/evaluation scores, and the '
          'slice\'s own scores at checkpoints k-1 and k. '
          'See the docstring of `scripts/analysis/plot_landscape_slices.py`. '
          f'Data extracted {meta["extracted"]}.', '',
          '| Row | Task | Change | Method | Run | k | class | shown / previous (data) '
          '| slice at k: shown / previous | slice at k-1: shown / previous | drift (x axis) |',
          '|' + '---|' * 11]
    for i, p in enumerate(panels):
        if i not in col_of:
            continue
        al, be = arrays[f'{i}/alphas'], arrays[f'{i}/betas']
        scale = p['best'] - p['floor']
        for r, m in enumerate(arms):
            ax = fig.add_subplot(gs[col_of[i], r] if tall else gs[r, col_of[i]])
            ax.set_xticks([])
            ax.set_yticks([])
            for s in ax.spines.values():
                s.set_linewidth(0.4)
                s.set_color('0.65')
            panel_title = f'{p["body"]}\n{plm.wrap(p["change"], 13).rstrip()}'
            if tall:
                if r == 0:
                    ax.set_ylabel(panel_title, rotation=0, ha='right', va='center',
                                  fontsize=fs - 1, labelpad=4, linespacing=1.15)
                    span.setdefault(p['group'], [ax, ax])[1] = ax
                    titles.append(ax.yaxis.label)
                if i == drawn[0]:
                    ax.set_title(pgs.label_of(m), fontsize=fs, pad=3)
                    heads.append((ax, m))
            else:
                if r == 0:
                    ax.set_title(panel_title, fontsize=fs - 1, pad=2.5, linespacing=1.15)
                    span.setdefault(p['group'], [ax, ax])[1] = ax
                    titles.append(ax.title)
                if i == drawn[0]:
                    ax.set_ylabel(pgs.label_of(m), rotation=0, ha='right', va='center',
                                  fontsize=fs, labelpad=10)
            row = p['rows'].get(m)
            if row is None:
                ax.set_facecolor('white')
                ax.text(0.5, 0.5, 'not run', transform=ax.transAxes, ha='center',
                        va='center', color=MUTED, fontsize=fs - 1)
                continue
            prev = (arrays[f'{i}/{m}/previous'] - p['floor']) / scale
            shown = (arrays[f'{i}/{m}/shown'] - p['floor']) / scale
            ax.imshow(regions(prev, shown), origin='lower',
                      extent=[al[0], al[-1], be[0], be[-1]], aspect='auto',
                      interpolation='nearest')
            ax.plot(0, 0, **BEFORE, zorder=5)
            still = row['axis_phase'] != row['k']
            ax.plot(*(row.get('next_xy') or ((0, 0) if still else (1, 0))), **AFTER, zorder=6)
            if row.get('toward') is not None:
                ax.plot(1, 0, '*', ms=4.5, mfc=REGION_SHOWN, mec='black', mew=0.4, zorder=5)
            ax.set_xlim(al[0], al[-1])
            ax.set_ylim(be[0], be[-1])
            ax.text(0.04, 0.95, CORNER[row['klass']], transform=ax.transAxes, ha='left',
                    va='top', linespacing=1.0, fontsize=fs - 2, color=INK,
                    bbox=dict(boxstyle='round,pad=0.12', fc='white', ec='none'))
            md.append(
                f'| {p["group"]} | {p["body"]} | {p["change"]} | {pgs.label_of(m)} '
                f'| {row["run"]} | {row["k"]} | {row["klass"]} '
                f'| {row["shown"]:.4g} / {row["previous"]:.4g} '
                f'| {row["shown_at"][1]:.4g} / {row["previous_at"][1]:.4g} '
                f'| {row["shown_at"][0]:.4g} / {row["previous_at"][0]:.4g} '
                f'| {row["drift"]:.3g}{" (did not move)" if still else ""} |')
    fig.canvas.draw()
    if tall:
        x_group = min(t.get_window_extent().x0 for t in titles) / fig.dpi / width - 0.05 / width
        for group, (a, b) in span.items():
            pa, pb = a.get_position(), b.get_position()
            fig.text(x_group, (pa.y1 + pb.y0) / 2, group, ha='right', va='center',
                     rotation=90, fontsize=fs + 0.5)
    else:
        y_group = max(t.get_window_extent().y1 for t in titles) / fig.dpi / height + 0.06 / height
        for group, (a, b) in span.items():
            pa, pb = a.get_position(), b.get_position()
            fig.text((pa.x0 + pb.x1) / 2, y_group, group, ha='center',
                     va='bottom', fontsize=fs + 0.5)
    x_key = 0.04 / width if tall else left_in / width
    handles = key_handles('checkpoint before the switch', 'checkpoint after the next task')
    if any(r.get('toward') is not None for p in panels for r in p['rows'].values()):
        handles.append(Line2D([], [], ls='', marker='*', ms=4.5, mfc=REGION_SHOWN, mec='black',
                              mew=0.4, label='its earlier solution (stuck runs)'))
    # Regions on the first line, checkpoints on the second: one line overruns
    # the width at print size.
    for line, (y, hs) in enumerate(((0.06, handles[:3]), (0.24, handles[3:]))):
        fig.add_artist(fig.legend(
            handles=hs, loc='upper left', bbox_to_anchor=(x_key, 1 - y / height),
            ncol=len(hs), frameon=False, fontsize=fs - 0.5, handlelength=1.0,
            handleheight=0.8, handletextpad=0.4, columnspacing=1.1, borderaxespad=0))
    STEM.parent.mkdir(parents=True, exist_ok=True)
    for ext in ('pdf', 'png'):
        fig.savefig(STEM.with_suffix(f'.{ext}'), dpi=300)
    plt.close(fig)
    STEM.with_suffix('.md').write_text('\n'.join(md) + '\n')
    print(f'wrote {STEM}.pdf/.png/.md ({width:.2f} x {height:.2f} in)')


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--compute', action='store_true', help='slice the runs (GPU)')
    ap.add_argument('--extract', action='store_true', help='collect the slices')
    ap.add_argument('--gpus', default='0')
    ap.add_argument('--rows', nargs='+', help='--compute: only these method rows')
    ap.add_argument('--panels', nargs='+', help='--compute: only these cells (or tree|cell)')
    ap.add_argument('--force', action='store_true')
    ap.add_argument('--layout', choices=('tall', 'wide'), default='tall',
                    help='tall: methods as columns, upright at \\linewidth; '
                         'wide: methods as rows, for a sideways page')
    ap.add_argument('--tanh', action='store_true',
                    help='also draw the HalfCheetah (tanh) panels')
    ap.add_argument('--width', type=float, help='inches (default 5.5 tall, 10 wide)')
    ap.add_argument('--font-size', type=float,
                    help='default 7 tall (printed at 1:1), 8 wide (rotated to 0.95\\textheight, x0.86)')
    args = ap.parse_args()
    if args.compute:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
        os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
        sys.path.insert(0, str(REPO))
        os.chdir(REPO)
        compute(args.rows, args.panels, args.force)
        return 0
    if args.extract:
        extract(args.tanh)
    if not DATA.with_suffix('.json').exists():
        sys.exit(f'no {DATA}.json: run with --extract first')
    tall = args.layout == 'tall'
    plot(args.width or (5.5 if tall else 10.0), args.font_size or (7.0 if tall else 8.0),
         args.layout, args.tanh)
    return 0


if __name__ == '__main__':
    sys.exit(main())
