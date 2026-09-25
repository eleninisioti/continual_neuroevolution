"""Run one MiniGrid trial: one (method, cell, trial), one process, one GPU.

    python source/studies/minigrid/cli.py --env MiniGrid_8x8_16x16 \
        --method nes --trial 1 --seed 42 --gpus 0 \
        --output_dir projects/iclr_2027/runs_centroid/minigrid/continual/nes/MiniGrid_8x8_16x16/trial_1

The flag names are not a choice: they are exactly what
`scripts/train/run_experiments.sh:run_condition` passes, so the MiniGrid
blocks reuse that function rather than reimplementing the output layout, the
"already done" skip and the failed-trial bookkeeping. `--env` is therefore a
CELL name, which is what a directory in the run tree is named after.

WHAT THIS FILE IS AND IS NOT
----------------------------
Argument plumbing. It holds no training loop, no searcher and no environment:
`settings.py` says what an arm is, `source/envs/minigrid.py` is the body, and
the two suite-generic runners under `source/studies/generalists/` do the work.
If something here starts to look like an algorithm, it is in the wrong file.

THE OUTPUT LAYOUT is the gymnax tree's, because the figure scripts walk it:

    <root>/minigrid/<continual|noncontinual>/<method>/<cell>/trial_<n>/

and every trial holds the files a gymnax trial holds -- `results.json`,
`config.json`, `training_metrics.json`, and `checkpoints.npz` carrying one
saved network per sub-task phase under the `centroid` key the centroid
plasticity figure reads.

NO ARM IS TOLD WHERE THE BOUNDARIES ARE (CLAUDE.md (d)). Nothing here passes a
boundary signal and both runners default to taking none: C-CHAIN's
`cchain_reset_on_switch` is off, and the GA and DNS re-evaluate their stored
population every generation rather than only at a transition. `--oracle`
builds the deliberately-informed control arm and is the only way to turn any
of it on.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Invoked as a SCRIPT by run_experiments.sh, so `source` is importable only via
# this insert -- the same three-deep walk every trainer under
# `source/studies/<suite>/` does.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(SCRIPT_DIR)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# Both before jax and before anything that imports it: the device mask is read
# at import time and a later assignment is silently ignored.
from source.utils.runtime import Tee, select_gpus          # noqa: E402
from source.studies.minigrid import settings as S          # noqa: E402


def build_parser():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    # `--env` is a CELL. run_experiments.sh calls it that for every suite and
    # names the run directory after it; on this body a cell is a room pair
    # plus a schedule rather than a single environment.
    p.add_argument('--env', required=True, choices=sorted(S.CELLS))
    p.add_argument('--method', required=True, choices=list(S.ARMS))
    p.add_argument('--trial', type=int, default=1)
    p.add_argument('--seed', type=int, default=None,
                   help='default: the trial index')
    p.add_argument('--output_dir', required=True)
    p.add_argument('--gpus', default=None,
                   help='CUDA_VISIBLE_DEVICES value; read before jax loads')
    p.add_argument('--wandb_project', default=None,
                   help='accepted for run_condition\'s sake; these runners do '
                        'not log to wandb')
    # Budget overrides, for a smoke test. They land in the run's config, so a
    # short run cannot later be mistaken for a real one.
    p.add_argument('--num_generations', type=int, default=None)
    p.add_argument('--num_updates', type=int, default=None)
    p.add_argument('--task_interval', type=int, default=None,
                   help="in the arm's own units: generations for NE, "
                        'updates for RL')
    p.add_argument('--eval_episodes', type=int, default=S.EVAL_EPISODES)
    p.add_argument('--num_evals', type=int, default=S.NE_NUM_EVALS)
    p.add_argument('--pop_size', type=int, default=S.NE_POP_SIZE)
    p.add_argument('--ppo_override', nargs='*', default=None,
                   metavar='KEY=VALUE',
                   help='entries of PPO_CONFIGS to replace, e.g. '
                        'num_envs=32 for a smoke test on a CPU. RL arms only, '
                        'and every override lands in the run config, so a '
                        'shrunk run cannot pass for a real one.')
    p.add_argument('--ne_override', nargs='*', default=None,
                   metavar='KEY=VALUE',
                   help='entries of the NE arm (settings.NE_ARMS) to replace, '
                        'e.g. sigma=0.03 for the hyperparameter appendix. NE arms '
                        'only; the override lands in the run config like the arm.')
    p.add_argument('--oracle', action='store_true',
                   help='give the arm its task boundaries -- the deliberately '
                        'informed control. OFF for every reported run; see '
                        'CLAUDE.md (d).')
    p.add_argument('--dry_run', action='store_true',
                   help='print the resolved settings and exit without '
                        'importing jax')
    return p


def resolve(args):
    """`(phase, kwargs common to both runners)`. No side effects."""
    schedule, rooms = S.CELLS[args.env]
    phase = 'continual' if args.env in S.CONTINUAL_CELLS else 'noncontinual'
    common = dict(
        env_name=S.ENV_NAME,
        schedule=schedule,
        num_tasks=S.NUM_TASKS,
        # The rooms this cell is built from, in order. `task0`/`task1` then
        # pin the schedule to one of them, but the ENVIRONMENT is still built
        # from the pair -- so the observation encoding, the episode scan
        # length and the policy are identical in all three cells, and a
        # stationary run is the switching run minus the switch.
        task_options={'envs': ','.join(rooms)},
        eval_episodes=args.eval_episodes,
        seed=args.trial if args.seed is None else args.seed,
        trial=args.trial,
        output_dir=args.output_dir,
        # A row of `noise_vectors` is an environment index on this suite, not
        # an observation offset. Passed explicitly so the run's config records
        # that a sub-task here is not a perturbation.
        noise_range=0.0,
    )
    return phase, common


def main():
    args = build_parser().parse_args()
    phase, common = resolve(args)
    select_gpus()
    matched_steps = S.check()
    fam = S.family(args.method)

    print('=' * 70)
    print(f'MiniGrid | {args.method} ({fam}) | {args.env} | {phase} '
          f'| trial {args.trial}')
    print(f'  rooms          : {S.CELLS[args.env][1]}')
    print(f'  schedule       : {common["schedule"]}, {S.NUM_PHASES} phases')
    print(f'  matched budget : {matched_steps:.3e} environment steps')
    print(f'  seed           : {common["seed"]}')
    print(f'  boundary info  : {"ORACLE" if args.oracle else "none"}')
    print(f'  output         : {args.output_dir}')
    print('=' * 70, flush=True)
    if args.dry_run:
        return 0

    os.makedirs(args.output_dir, exist_ok=True)
    # The run keeps its own log beside its artifacts, as the gymnax trainers
    # do, so a finished run is self-contained once the launcher's per-job log
    # has been cleaned up.
    sys.stdout = Tee(os.path.join(args.output_dir, 'train.log'))

    if fam == 'ne':
        from source.studies.generalists.train_nes import run_nes
        arm = dict(S.NE_ARMS[args.method])
        for item in args.ne_override or []:
            key, _, value = item.partition('=')
            arm[key] = float(value) if key != 'method' else value
        result = run_nes(
            num_generations=args.num_generations or S.NE_GENERATIONS,
            task_interval=args.task_interval or S.NE_TASK_INTERVAL,
            pop_size=args.pop_size,
            num_evals=args.num_evals,
            **arm, **common)
    else:
        from source.studies.generalists.train_ppo import pbt_kwargs, run_ppo
        overrides = {}
        for item in args.ppo_override or []:
            key, _, value = item.partition('=')
            for cast in (int, float):
                try:
                    overrides[key] = cast(value)
                    break
                except ValueError:
                    continue
            else:
                overrides[key] = value
        result = run_ppo(
            # `pbt` / `pbt2` are one method at two population sizes, and
            # `pbt_weights` / `pbt2_weights` the same without explore.
            **pbt_kwargs(args.method),
            overrides=overrides or None,
            num_updates=args.num_updates or S.RL_UPDATES,
            task_interval=args.task_interval or S.RL_TASK_INTERVAL,
            # CLAUDE.md (d): off unless --oracle. The runner's own default is
            # already off; passed explicitly so every MiniGrid run's config
            # records which it was rather than leaving a reader to know what
            # the default was on the day it ran.
            cchain_reset_on_switch=bool(args.oracle),
            **common)

    # `config.json` beside the `results.json` the runner wrote. `load_config`
    # in the figure scripts takes whichever it reaches first and reads
    # `blob['config'] or blob`, so the two must agree; this is that same dict.
    with open(os.path.join(args.output_dir, 'config.json'), 'w') as f:
        json.dump(result['config'], f, indent=2)

    print(f'\nDone. {args.output_dir}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
