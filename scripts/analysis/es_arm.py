"""Which ES-family arm a paper figure reports: `es` (OpenES) or `nes`.

The two are one method at two settings, so the paper draws ONE of them in
every figure and table -- the one with the higher Cum. elite (the table's
performance column: the area under the elite curve, mean over trials) on the
runs that figure is about:

    a family directory (finish_iclr.sh)  its continual cells; a stationary-only
                                         family, its stationary cells
    a stationary panel                   that task's stationary run

Over several cells the margins are summed, (ES - NES) / max(|ES|, |NES|) a
cell, which is free of each task's reward scale. Not a count of cells won: on
the gymnax physics family CartPole and Acrobot are saturated ties (1995.5 vs
1995.1) and a count lets them outvote a real gap on MountainCar. An exact tie
keeps NES. An arm with no runs loses to one that has.

PBT-PPO at N=8 (`pbt`) and N=2 (`pbt2`) are one method at two settings in the
same way; `pick(..., PBT_ARMS)` chooses between them by the same rule
(plot_continual_lineplots.py). The CLI and `es_arm.json` are the ES pair's.

    .venv/bin/python scripts/analysis/es_arm.py <root> --phase continual \\
        --cells CartPole_v1_sigma1.0 Acrobot_v1_sigma1.0 --out <paper dir>/es_arm.json

prints the kept arm as its last stdout line and writes the per-cell numbers to
`--out`, which the cross-suite figures read (`dropped_arm`) so a column drawn
from that directory hides the same arm the directory's own figures hide.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import pathlib
import sys

import numpy as np

REPO = pathlib.Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO / 'scripts'))
import make_lineplot as lp                                 # noqa: E402

ARMS = ('es', 'nes')
PBT_ARMS = ('pbt', 'pbt2')

# THE PAPER'S CHOICE, per suite: `nes` (z-scored fitness + SGD) is ES on
# gymnax, MiniGrid and the MJX bodies; `es` (OpenES: centered ranks + Adam) on
# Kinetix. Decided by the user, not by Cum. elite. On 2026-09-24 gymnax was
# tried on OpenES because NES cannot solve stationary MountainCar at any
# ordinary setting (runs_hparam/nes_mcar_tune: 30 width/step/population
# settings, none holds the goal) while OpenES does on 7/10 seeds -- but OpenES
# is worse than NES in the continual cells (Acrobot noise, CartPole reversal;
# 55% dead phases on MountainCar noise, the whole population at -500 after a
# switch), so the user kept NES and accepted the MountainCar limitation.
# `pick` and the CLI still exist for the PBT pair and the per-cell margins.
KEPT_BY_SUITE = {'gymnax': 'nes', 'kinetix': 'es', 'minigrid': 'nes', 'mjx': 'nes', 'cheetah': 'nes', 'ant': 'nes'}


def kept_for(tree):
    """The reported ES-family arm for a run or paper tree, by its suite.

    `tree` is any path that names the suite somewhere in it: a paper data
    tree (`paper/gymnax/data/noise_10task`, `paper/mjx/cheetah/data/...`),
    a run tree (`runs_centroid/gymnax`, `runs_mjx_noise05_t10/mjx`,
    `runs_kinetix_ep128_ev1/kinetix`) or a family directory.
    """
    parts = [q.lower() for q in str(tree).replace('\\', '/').split('/')]
    for suite, arm in KEPT_BY_SUITE.items():
        if suite in parts or any(q.startswith(f'runs_{suite}') for q in parts):
            return arm
    raise ValueError(f'kept_for: no suite in {tree!r}; add it to KEPT_BY_SUITE')
FILENAME = 'es_arm.json'


def scores(data, per_gen, arms=ARMS):
    """`{env: {arm: [Cum. elite a trial]}}` for `arms`, from make_lineplot
    curves, as the table computes the column (`metric_table`)."""
    out = {}
    for env, by_method in data.items():
        for arm in arms:
            if arm in by_method:
                x, curves = by_method[arm]
                xg = x / per_gen if per_gen else x
                out.setdefault(env, {})[arm] = [
                    lp.cumulative_reward(xg, c, xg[-1]) / 1e3 for c in curves]
    return out


def pick(per_cell, arms=ARMS):
    """`(kept arm or None, summed margin, {env: {arm: mean}})` under the rule
    above, between the two `arms` (any other arm in `per_cell` is ignored); the
    margin is positive when the first is ahead, and a tie keeps the second."""
    first, second = arms
    means = {env: {a: float(np.mean(v)) for a, v in by_arm.items() if a in arms}
             for env, by_arm in per_cell.items()}
    present = {a for m in means.values() for a in m}
    if len(present) < 2:
        return (present.pop() if present else None), 0.0, means
    margin = sum((m[first] - m[second]) / max(abs(m[first]), abs(m[second]))
                 for m in means.values() if len(m) == 2 and m[first] != m[second])
    return (first if margin > 0 else second), margin, means


def load(root, phase, cells=None, arms=ARMS):
    """Cum. elite of `arms` under `root/phase`. `cells` restricts the
    continual cells; the stationary cells are every one without a sigma, as
    make_lineplot's reference loader reads them."""
    root = pathlib.Path(root)
    cell_filter = (lp.cell_selector(cells, None)[0] if phase != 'noncontinual'
                   else lambda c: '_sigma' not in c)
    # collect() narrates to stdout; the kept arm must be the last line there.
    # The ES pair is always read too: only an NE budget gives `per_gen`, and
    # without it an RL pair (PBT_ARMS) stays on the env-step clock, where
    # cumulative_reward's one-point-a-unit grid is ~5e8 points a trial.
    with contextlib.redirect_stdout(sys.stderr):
        data, _, _, per_gen, _, _ = lp.collect(
            root, phase, cell_filter, 'elite_eval', 'generalist', None,
            methods=tuple(dict.fromkeys((*arms, *ARMS))))
    return scores(data, per_gen, arms)


def dropped_arm(paper_dir):
    """The ES-family arm a built paper directory does NOT report, or None when
    it has no `es_arm.json` (built before the choice) or runs only one arm."""
    path = pathlib.Path(paper_dir) / FILENAME
    if not path.exists():
        return None
    kept = json.loads(path.read_text()).get('kept')
    return next((a for a in ARMS if a != kept), None) if kept else None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('root')
    ap.add_argument('--phase', default='continual')
    ap.add_argument('--cells', nargs='*', default=None)
    ap.add_argument('--out', default=None, help=f'write the numbers here ({FILENAME})')
    args = ap.parse_args()

    kept, margin, means = pick(load(args.root, args.phase, args.cells))
    for env, m in sorted(means.items()):
        print(f'{env:30s} ' + '  '.join(f'{a}={v:.3f}' for a, v in sorted(m.items())),
              file=sys.stderr)
    print(f'kept: {kept}  (summed margin ES - NES {margin:+.3f})', file=sys.stderr)
    if args.out:
        pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(args.out).write_text(json.dumps(
            {'kept': kept, 'criterion': 'Cum. elite, mean over trials; summed '
             '(ES - NES) / max(|ES|, |NES|) over the cells', 'margin': margin,
             'root': str(args.root), 'phase': args.phase, 'cum_elite': means},
            indent=1) + '\n')
    print(kept or '')
    return 0


if __name__ == '__main__':
    sys.exit(main())
