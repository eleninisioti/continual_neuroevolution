"""Which PBT-PPO arm a paper figure reports: N=8 (`pbt`) or N=2 (`pbt2`).

The paper runs one ES arm, `es`, on every benchmark. PBT-PPO at N=8 and N=2
are one method at two settings, so the paper draws ONE of them in every
figure and table -- the one with the higher Cum. elite (the table's
performance column: the area under the elite curve, mean over trials) on the
runs that figure is about (a family's continual cells; a stationary panel's
stationary run).

Over several cells the margins are summed, (A - B) / max(|A|, |B|) a cell,
which is free of each task's reward scale. Not a count of cells won: saturated
ties would otherwise outvote a real gap elsewhere. An exact tie keeps the
second arm. An arm with no runs loses to one that has.

    .venv/bin/python scripts/analysis/es_arm.py <root> --phase continual \\
        --cells CartPole_v1_sigma1.0 Acrobot_v1_sigma1.0

prints the kept arm as its last stdout line.
"""

from __future__ import annotations

import argparse
import contextlib
import pathlib
import sys

import numpy as np

REPO = pathlib.Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO / 'scripts' / 'plotting'))
import make_lineplot as lp                                 # noqa: E402

ES = 'es'                     # the one ES arm, on every benchmark
PBT_ARMS = ('pbt', 'pbt2')


def scores(data, per_gen, arms=PBT_ARMS):
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


def pick(per_cell, arms=PBT_ARMS):
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


def load(root, phase, cells=None, arms=PBT_ARMS):
    """Cum. elite of `arms` under `root/phase`. `cells` restricts the
    continual cells; the stationary cells are every one without a sigma, as
    make_lineplot's reference loader reads them."""
    root = pathlib.Path(root)
    cell_filter = (lp.cell_selector(cells, None)[0] if phase != 'noncontinual'
                   else lambda c: '_sigma' not in c)
    # collect() narrates to stdout; the kept arm must be the last line there.
    # The ES arm is always read too: only an NE budget gives `per_gen`, and
    # without it an RL pair stays on the env-step clock, where
    # cumulative_reward's one-point-a-unit grid is ~5e8 points a trial.
    with contextlib.redirect_stdout(sys.stderr):
        data, _, _, per_gen, _, _ = lp.collect(
            root, phase, cell_filter, 'elite_eval', 'generalist', None,
            methods=tuple(dict.fromkeys((*arms, ES))))
    return scores(data, per_gen, arms)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('root')
    ap.add_argument('--phase', default='continual')
    ap.add_argument('--cells', nargs='*', default=None)
    args = ap.parse_args()

    kept, margin, means = pick(load(args.root, args.phase, args.cells))
    for env, m in sorted(means.items()):
        print(f'{env:30s} ' + '  '.join(f'{a}={v:.3f}' for a, v in sorted(m.items())),
              file=sys.stderr)
    print(f'kept: {kept}  (summed margin N=8 - N=2 {margin:+.3f})', file=sys.stderr)
    print(kept or '')
    return 0


if __name__ == '__main__':
    sys.exit(main())
