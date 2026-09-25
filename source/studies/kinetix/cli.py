"""Run one Kinetix trial: one (method, cell, trial), one process, one GPU.

    python source/studies/kinetix/cli.py --env Kinetix-h0_unicycle \
        --method ga --trial 1 --seed 1 --gpus 0 \
        --output_dir projects/iclr_2027/runs_kinetix/kinetix/noncontinual/ga/Kinetix-h0_unicycle/trial_1

The flag names are not a choice: they are exactly what
`scripts/train/run_experiments.sh:run_condition` passes, so the Kinetix blocks
reuse that function rather than reimplementing the output layout, the "already
done" skip and the failed-trial bookkeeping. `--env` is therefore a CELL name,
which is what a directory in the run tree is named after: `Kinetix-<level>` for
a stationary run and `Kinetix20` for the continual chain.

WHAT THIS FILE IS AND IS NOT
----------------------------
Argument plumbing. It holds no training loop, no searcher and no environment:
`settings.py` says what an arm is, `source/envs/kinetix.py` is the body, and
the two suite-generic runners under `source/studies/generalists/` do the work.
The eighteen old trainers beside this file are the PREVIOUS codebase and are
not imported.

THE OUTPUT LAYOUT is the gymnax tree's, because the figure scripts walk it:

    <root>/kinetix/<continual|noncontinual>/<method>/<cell>/trial_<n>/

and every trial holds what a gymnax trial holds -- `results.json`,
`config.json`, `training_metrics.json`, and `checkpoints.npz` carrying one
saved network per phase under the `centroid` key.

NO ARM IS TOLD WHERE THE BOUNDARIES ARE (CLAUDE.md (d)). Nothing here passes a
boundary signal and both runners default to taking none: C-CHAIN's
`cchain_reset_on_switch` is off, and the GA and DNS re-evaluate their stored
population every generation rather than only at a transition. `--oracle` is
the deliberately-informed control arm and the only way to turn any of it on.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(SCRIPT_DIR)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# Both before jax and before anything that imports it: the device mask is read
# at import time and a later assignment is silently ignored. `settings` imports
# `source/envs/kinetix_levels.py`, which is jax-free for exactly this reason.
from source.utils.runtime import Tee, select_gpus          # noqa: E402
from source.studies.kinetix import settings as S           # noqa: E402


def build_parser():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--env', required=True, choices=sorted(S.CELLS))
    p.add_argument('--method', required=True, choices=sorted(S.ARMS))
    p.add_argument('--trial', type=int, default=1)
    p.add_argument('--seed', type=int, default=None,
                   help='default: the trial index')
    p.add_argument('--output_dir', required=True)
    p.add_argument('--gpus', default=None,
                   help='CUDA_VISIBLE_DEVICES value; read before jax loads')
    p.add_argument('--wandb_project', default=None,
                   help="accepted for run_condition's sake; these runners do "
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
    p.add_argument('--episode_length', type=int, default=None,
                   help='override the per-episode scan length; the suite '
                        'table (source/envs/kinetix.py) is the default. '
                        'NOTE it scales the NE budget but NOT the RL one, '
                        'which is updates x envs x steps -- scale '
                        '--num_updates too or the comparison stops being '
                        'matched.')
    p.add_argument('--pop_size', type=int, default=S.NE_POP_SIZE)
    p.add_argument('--observation', default='pixels',
                   choices=['pixels', 'entity'],
                   help='pixels: the 125x125x3 frame and the conv policy '
                        '(every run before 2026-09-23). entity: Kinetix\'s '
                        'symbolic-entity observation and its transformer '
                        '(source/envs/kinetix.py). The same arms, budget and '
                        'cells either way; keep the two in separate trees.')
    p.add_argument('--dns_descriptor', default=None,
                   choices=['handcrafted', 'aurora'],
                   help="override the DNS arms' descriptor. Default is "
                        "settings.py's `handcrafted` -- the six-channel duty "
                        'factor; `aurora` learns one online from the 13-value '
                        'per-step feature vector.')
    p.add_argument('--ppo_override', nargs='*', default=None,
                   metavar='KEY=VALUE',
                   help='entries of PPO_CONFIGS to replace, e.g. num_envs=8 '
                        'for a smoke test. RL arms only, and every override '
                        'lands in the run config, so a shrunk run cannot pass '
                        'for a real one.')
    p.add_argument('--checkpoint_interval', type=int, default=None,
                   help='how often to append a parameter snapshot to '
                        "trajectory.npz, in the arm's own units. None is the "
                        "trainer's default -- every generation for NE, every "
                        'fifth record for RL. The continual RL run needs it: '
                        '32,000 updates at the default is 640 snapshots of '
                        'a 1.1M-parameter network, a 2.9 GB trajectory.npz and '
                        'a resume.pkl rewritten at that size at every segment '
                        'checkpoint. 1000 gives the 32 snapshots a stationary '
                        'RL run has, so the two are read the same way.')
    p.add_argument('--checkpoint_every', type=int, default=0,
                   help='save a boundary checkpoint every N generations '
                        '(the level length, 200, for the continual chain), '
                        'to <output_dir>/resume.pkl. A later run with the '
                        'same command resumes from it. 0 disables.')
    p.add_argument('--max_gens_this_run', type=int, default=0,
                   help='exit cleanly at the next checkpoint once this many '
                        'generations have run in THIS process -- so a segment '
                        'ends at a level boundary instead of under a SLURM '
                        'kill. 0 means run to the end.')
    p.add_argument('--oracle', action='store_true',
                   help='give the arm its task boundaries -- the deliberately '
                        'informed control. OFF for every reported run; see '
                        'CLAUDE.md (d).')
    p.add_argument('--dry_run', action='store_true',
                   help='print the resolved settings and exit without '
                        'importing jax')
    # Does one network solve several levels at once? `--joint K` with the
    # chain cell scores every member on the chain's first K levels every
    # generation and selects on `--objective` (default `capped`: the mean of
    # min(return, solved threshold), so only solving an unsolved level helps).
    # 200 generations per level by default; each generation costs K times a
    # stationary one, so this is an existence test, not a matched arm.
    p.add_argument('--joint', type=int, default=0,
                   help='search the chain cell\'s first K levels jointly')
    p.add_argument('--objective', default='capped',
                   choices=['capped', 'mean', 'min', 'worstk', 'capped_worstk'],
                   help='joint only: how the K per-level returns become one fitness')
    return p


def resolve(args):
    """`(phase, kwargs common to both runners)`. No side effects."""
    levels = S.CELLS[args.env]
    continual = args.env in S.CONTINUAL_CELLS
    if args.joint:
        if not continual or not 1 < args.joint <= len(levels):
            raise SystemExit(f'--joint K needs the chain cell and 1 < K <= '
                             f'{len(levels)}')
        levels = levels[:args.joint]
    common = dict(
        env_name=args.env,
        # The continual chain is round-robin over the twenty levels; a
        # stationary cell holds ONE level and `task0` pins every phase to it.
        schedule='joint' if args.joint else ('switch' if continual else 'task0'),
        num_tasks=len(levels),
        task_options={'levels': ','.join(levels),
                      **({'observation': args.observation}
                         if args.observation != 'pixels' else {})},
        eval_episodes=args.eval_episodes,
        seed=args.trial if args.seed is None else args.seed,
        trial=args.trial,
        output_dir=args.output_dir,
        # A row of `noise_vectors` is a LEVEL INDEX on this suite, not an
        # observation offset. Passed explicitly so the run's config records
        # that a sub-task here is not a perturbation.
        noise_range=0.0,
    )
    if args.joint:
        common['objective'] = args.objective
        return 'joint', common
    return ('continual' if continual else 'noncontinual'), common


def main():
    args = build_parser().parse_args()
    phase, common = resolve(args)
    continual = phase == 'continual'
    select_gpus()
    matched_steps = S.check(continual=continual)
    fam = S.family(args.method)

    print('=' * 70)
    print(f'Kinetix | {args.method} ({fam}) | {args.env} | {phase} '
          f'| trial {args.trial}')
    lv = common['task_options']['levels'].split(',')
    print(f'  levels         : {len(lv)} ({", ".join(lv[:3])}'
          f'{", ..." if len(lv) > 3 else ""})')
    print(f'  schedule       : {common["schedule"]}'
          + (f'  objective {common["objective"]}' if args.joint else ''))
    print(f'  matched budget : {matched_steps:.3e} environment steps')
    print(f'  observation    : {args.observation}')
    print(f'  seed           : {common["seed"]}')
    print(f'  boundary info  : {"ORACLE" if args.oracle else "none"}')
    print(f'  output         : {args.output_dir}')
    print('=' * 70, flush=True)
    if args.dry_run:
        return 0

    os.makedirs(args.output_dir, exist_ok=True)
    sys.stdout = Tee(os.path.join(args.output_dir, 'train.log'))

    if fam == 'ne':
        from source.studies.generalists.train_nes import run_nes
        arm = dict(S.NE_ARMS[args.method])
        if args.dns_descriptor is not None and 'searcher_kwargs' in arm:
            arm['searcher_kwargs'] = dict(arm['searcher_kwargs'],
                                          descriptor=args.dns_descriptor)
        default_gens = (S.GENERATIONS_PER_LEVEL * args.joint if args.joint
                        else S.NE_GENERATIONS_CONTINUAL if continual
                        else S.NE_GENERATIONS)
        default_interval = (S.NE_TASK_INTERVAL_CONTINUAL if continual
                            else S.NE_TASK_INTERVAL)
        result = run_nes(
            episode_length=args.episode_length,
            resume_path=(os.path.join(args.output_dir, 'resume.pkl')
                         if args.checkpoint_every else None),
            checkpoint_every=args.checkpoint_every,
            max_gens_this_run=args.max_gens_this_run,
            **({'checkpoint_interval': args.checkpoint_interval}
               if args.checkpoint_interval else {}),
            num_generations=args.num_generations or default_gens,
            task_interval=args.task_interval or default_interval,
            pop_size=args.pop_size,
            num_evals=args.num_evals,
            **arm, **common)
        if isinstance(result, dict) and result.get('resumed_segment'):
            # A segment that stopped at a level boundary has no artifacts to
            # report -- only the next segment (or the last) writes them. Exit
            # 0 here so a clean segment end is not logged as a traceback that
            # a real failure would hide behind.
            print(f"segment finished cleanly at generation "
                  f"{result['stopped_at']}; resume.pkl left for the next one")
            return 0
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
        default_updates = (S.RL_UPDATES_CONTINUAL if continual
                           else S.RL_UPDATES)
        default_interval = (S.RL_TASK_INTERVAL_CONTINUAL if continual
                            else S.RL_TASK_INTERVAL)
        result = run_ppo(
            episode_length=args.episode_length,
            resume_path=(os.path.join(args.output_dir, 'resume.pkl')
                         if args.checkpoint_every else None),
            checkpoint_every=args.checkpoint_every,
            max_updates_this_run=args.max_gens_this_run,
            **({'checkpoint_interval': args.checkpoint_interval}
               if args.checkpoint_interval else {}),
            # `pbt` / `pbt2` are one method at two population sizes, and
            # `pbt_weights` / `pbt2_weights` the same without explore.
            **pbt_kwargs(args.method),
            overrides=overrides or None,
            num_updates=args.num_updates or default_updates,
            task_interval=args.task_interval or default_interval,
            # CLAUDE.md (d): off unless --oracle. The runner's own default is
            # already off; passed explicitly so every Kinetix run's config
            # records which it was rather than leaving a reader to know what
            # the default was on the day it ran.
            cchain_reset_on_switch=bool(args.oracle),
            **common)
        if isinstance(result, dict) and result.get('resumed_segment'):
            # Same clean segment end as the NE branch: `--max_gens_this_run`
            # counts PPO updates here.
            print(f"segment finished cleanly at update "
                  f"{result['stopped_at']}; resume.pkl left for the next one")
            return 0

    with open(os.path.join(args.output_dir, 'config.json'), 'w') as f:
        json.dump(result['config'], f, indent=2)

    print(f'\nDone. {args.output_dir}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
