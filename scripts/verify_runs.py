"""Is a run tree fit to put in a paper figure?

Answers three questions about a directory of runs, per (method, environment)
cell, and refuses to be vague about any of them:

  complete     every trial has the artifacts a figure reads, they parse, and
               every trial in a cell has the same number of records. A cell
               with 9 trials where its neighbours have 10 is a cell that
               silently lost a run.

  compute-matched   every method in a cell saw the SAME number of environment
               steps. CLAUDE.md rule (c). NE spends
               `generations x pop_size x num_evals x episode_length`; PPO
               spends `num_updates x num_envs x num_steps`, and the two are
               only comparable once both are written in steps.

  aligned      every method met its task boundaries at the same step. Also
               rule (c): a method that switches at a different point in its
               budget is answering a different question, and two such curves
               do not belong on one axis.

It reads only `results.json` and `training_metrics.json`, never a checkpoint,
so it is fast and needs no JAX.

    .venv/bin/python scripts/verify_runs.py <run-tree> [--phase continual]

Exit 0 when every cell passes, 1 otherwise. What it CANNOT check is whether a
method is the method it claims to be -- a mis-set coefficient produces a
perfectly well-formed run. Those are tracked separately; see the notes in
`projects/iclr_2027/README.md`.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from collections import defaultdict

def env_of(cell_name: str) -> str:
    """`CartPole_v1_sigma1.0` -> `CartPole_v1`."""
    return cell_name.split('_sigma')[0]


ASSUMED_EPISODE_LENGTH: int | None = None


def episode_length(config: dict, results: dict):
    """The episode length the run actually used, or None.

    Read from the run, never from a table of environment defaults. The first
    version of this script hardcoded gymnax's own caps and reported every
    MountainCar cell as un-compute-matched, because these runs set
    `episode_length` to 500 there rather than gymnax's 200. A checker that
    invents a constant does not check the data, it checks the constant.

    The continual runs record it; the noncontinual ones do not, so their NE
    budget cannot be turned into steps from the run alone. There the answer is
    None and the cell is reported as unmeasurable, unless the caller states the
    assumption with `--assume-episode-length`, which prints it in the header so
    the claim travels with the result.
    """
    for src in (config, results):
        if src.get('episode_length') is not None:
            return int(src['episode_length'])
    return ASSUMED_EPISODE_LENGTH


def steps_of(config: dict, results: dict):
    """Total environment steps a run spent, or None if it cannot be derived.

    Two spellings, because NE and RL count different things:
      NE   generations x population x evaluations x episode length
      RL   `num_timesteps`, which the RL trainers already record in steps
    """
    if 'num_timesteps' in config:
        return int(config['num_timesteps']), 'rl'
    gens = config.get('num_generations') or results.get('num_generations')
    pop = config.get('pop_size') or results.get('pop_size')
    evals = config.get('num_evals') or results.get('num_evals')
    ep = episode_length(config, results)
    if None in (gens, pop, evals, ep):
        return None, 'ne'
    return int(gens) * int(pop) * int(evals) * int(ep), 'ne'


def boundary_every(config: dict, results: dict, key: str = 'task_interval'):
    """Steps between task switches, or None. The unit of `task_interval`
    differs by family -- generations for NE, updates for RL -- so it only
    becomes comparable once multiplied out. `key='task_warmup'` is the same
    arithmetic for the warm-up that holds sub-task 0 before the first switch."""
    interval = config.get(key) or results.get(key)
    if interval is None:
        return None
    if 'num_timesteps' in config:
        per = int(config['num_envs']) * int(config['num_steps'])
    else:
        pop = config.get('pop_size') or results.get('pop_size')
        evals = config.get('num_evals') or results.get('num_evals')
        ep = episode_length(config, results)
        if None in (pop, evals, ep):
            return None
        per = int(pop) * int(evals) * int(ep)
    return int(interval) * per


def scan(root: pathlib.Path, phase: str):
    """`{(method, cell): [per-trial dicts]}` for one phase of a run tree."""
    out = defaultdict(list)
    base = root / phase
    if not base.is_dir():
        sys.exit(f"no such phase directory: {base}")
    for method_dir in sorted(p for p in base.iterdir() if p.is_dir()):
        for cell_dir in sorted(p for p in method_dir.iterdir() if p.is_dir()):
            for trial_dir in sorted(cell_dir.glob('trial_*')):
                rec = {'trial': trial_dir.name, 'path': trial_dir}
                # The continual trainers write `results.json` with the config
                # nested inside; the noncontinual ones write a flat
                # `config.json` and no results file. Either is a complete
                # description of what the run was asked to do.
                for name in ('results.json', 'config.json'):
                    path = trial_dir / name
                    if path.exists():
                        try:
                            rec['results'] = json.loads(path.read_text())
                        except Exception as exc:        # noqa: BLE001
                            rec['error'] = f'{name}: {type(exc).__name__}'
                        break
                else:
                    rec['error'] = 'no results.json or config.json'
                if 'results' not in rec:
                    out[(method_dir.name, cell_dir.name)].append(rec)
                    continue
                try:
                    metrics = json.loads(
                        (trial_dir / 'training_metrics.json').read_text())
                    rec['n_records'] = len(metrics)
                    rec['keys'] = set(metrics[0]) if metrics else set()
                except Exception as exc:                # noqa: BLE001
                    rec['error'] = f'training_metrics.json: {type(exc).__name__}'
                out[(method_dir.name, cell_dir.name)].append(rec)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('root', help='run tree, e.g. projects/iclr_2027/runs/gymnax')
    ap.add_argument('--phase', default='continual',
                    help="'continual' or 'noncontinual' (default: continual)")
    ap.add_argument('--quiet', action='store_true',
                    help='only report problems')
    ap.add_argument('--assume-episode-length', type=int, default=None,
                    help='use this episode length where a run did not record '
                         'one (the noncontinual trees). Stated rather than '
                         'guessed: it is printed with the results.')
    args = ap.parse_args()

    global ASSUMED_EPISODE_LENGTH
    ASSUMED_EPISODE_LENGTH = args.assume_episode_length

    cells = scan(pathlib.Path(args.root), args.phase)
    if not cells:
        sys.exit(f'no runs found under {args.root}/{args.phase}')

    problems: list[str] = []

    # ---- per cell: completeness
    by_env: dict[str, dict[str, tuple] ] = defaultdict(dict)
    for (method, cell), trials in sorted(cells.items()):
        broken = [t for t in trials if 'error' in t]
        for t in broken:
            problems.append(f"{method}/{cell}/{t['trial']}: {t['error']}")
        good = [t for t in trials if 'error' not in t]
        counts = {t.get('n_records') for t in good}
        if len(counts) > 1:
            problems.append(
                f"{method}/{cell}: trials disagree on record count {sorted(c for c in counts if c)}")
        if good:
            res = good[0]['results']
            # `results.json` nests the run's settings under `config`; a flat
            # `config.json` IS them.
            cfg = res.get('config') or res
            steps, family = steps_of(cfg, res)
            bound = boundary_every(cfg, res)
            # The first switch, checked on its own: a warm-up leaves the
            # interval alone, so two arms with different warm-ups pass the
            # interval check and still meet every switch at a different step.
            first_sw = boundary_every(cfg, res, 'task_warmup') or bound
            by_env[cell][method] = (steps, bound, len(good), family, first_sw)

    # ---- per environment: compute matching and boundary alignment
    print(f"\n{args.root}  [{args.phase}]")
    if args.assume_episode_length:
        print(f"  (assuming episode_length={args.assume_episode_length} where "
              f"a run did not record one)")
    print()
    for cell, methods in sorted(by_env.items()):
        step_set = {v[0] for v in methods.values() if v[0] is not None}
        bound_set = {v[1] for v in methods.values() if v[1] is not None}
        trial_set = {v[2] for v in methods.values()}
        ok_steps = len(step_set) <= 1
        ok_bound = len(bound_set) <= 1
        ok_trials = len(trial_set) <= 1
        first_set = {v[4] for v in methods.values() if v[4] is not None}
        ok_first = len(first_set) <= 1
        mark = 'ok ' if (ok_steps and ok_bound and ok_first and ok_trials) else 'BAD'
        steps = next(iter(step_set)) if ok_steps and step_set else None
        bound = next(iter(bound_set)) if ok_bound and bound_set else None
        n = next(iter(trial_set)) if ok_trials else None
        first_sw = next(iter(first_set)) if ok_first and first_set else None
        if not args.quiet or mark == 'BAD':
            print(f"  [{mark}] {cell:26s} {len(methods)} methods  "
                  f"{'x'.join([str(n)]) if n else '?'} trials  "
                  f"steps={steps:.3e}" if steps else
                  f"  [{mark}] {cell:26s} {len(methods)} methods")
            if bound:
                print(f"        boundary every {bound:.3e} steps "
                      f"({(1 + (steps - (first_sw or bound)) // bound) if steps else '?'} sub-tasks)")
            if first_sw and bound and first_sw != bound:
                print(f"        first switch at {first_sw:.3e} steps (task_warmup)")
        if not step_set:
            problems.append(f"{cell}: budget not measurable -- no run records "
                            f"episode_length; pass --assume-episode-length")
        if not ok_steps:
            problems.append(f"{cell}: methods are NOT compute-matched -- " +
                            ", ".join(f"{m}={v[0]:.3e}" for m, v in sorted(methods.items())
                                      if v[0] is not None))
        if not ok_bound:
            problems.append(f"{cell}: task boundaries NOT aligned -- " +
                            ", ".join(f"{m}={v[1]:.3e}" for m, v in sorted(methods.items())
                                      if v[1] is not None))
        if not ok_first:
            problems.append(f"{cell}: first task switch NOT aligned (task_warmup) -- " +
                            ", ".join(f"{m}={v[4]:.3e}" for m, v in sorted(methods.items())
                                      if v[4] is not None))
        if not ok_trials:
            problems.append(f"{cell}: uneven trial counts -- " +
                            ", ".join(f"{m}={v[2]}" for m, v in sorted(methods.items())))

    print()
    if problems:
        print(f"{len(problems)} problem(s):\n")
        for p in problems:
            print("  " + p)
        return 1
    print(f"clean: {len(by_env)} cells, "
          f"{sum(len(m) for m in by_env.values())} (method, cell) pairs")
    return 0


if __name__ == '__main__':
    sys.exit(main())
