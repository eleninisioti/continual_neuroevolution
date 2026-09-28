"""Run one trial of any benchmark: one (suite, method, cell, trial), one process.

    python source/run.py --suite minigrid --env MiniGrid_8x8_16x16 \
        --method es --trial 1 --seed 42 --gpus 0 --output_dir runs/.../trial_1

Every benchmark goes through this file and the same two training loops:

    NE   source/runners/train_nes.py:run_nes   GA, ES, DNS
    RL   source/runners/train_ppo.py:run_ppo   PPO, TRAC, ReDo, C-CHAIN, PBT

What differs between benchmarks is only configuration, and that lives in
`source/configs/<suite>.yaml`: the cells (`--env`), the methods (`--method`),
their hyperparameters and the compute-matched budgets. The logic that reads it
is `source/utils/config.py`; a suite may add flags of its own through
`add_args`, everything else here is shared.

`--env` is a CELL: an environment plus a task schedule, which is what a run
directory is named after. The flag names are what
`scripts/train/run.sh:run_condition` passes.

NO METHOD IS TOLD WHERE THE TASK BOUNDARIES ARE. Nothing here passes a boundary
signal and both runners default to taking none. `--oracle` builds the
deliberately-informed control and is the only way to turn any of it on.

THE CONFIG INTERFACE is listed in `source/utils/config.py`.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# Before jax and before anything that imports it: the device mask is read at
# import time and a later assignment is silently ignored.
from source.utils.runtime import Tee, select_gpus          # noqa: E402
from source.utils import config as suite_config             # noqa: E402
from source.utils.config import parse_overrides            # noqa: E402

SUITES = tuple(suite_config.SUITES)


def apply_ne_overrides(arm, ne_override, searcher_override):
    """The arm with `--ne_override` / `--searcher_override` applied.

    A dotted key (`searcher_kwargs.elite_ratio=0.05`) reaches one level down.
    """
    arm = dict(arm)
    if 'searcher_kwargs' in arm:
        arm['searcher_kwargs'] = dict(arm['searcher_kwargs'])
    for key, value in parse_overrides(ne_override).items():
        if '.' in key:
            outer, inner = key.split('.', 1)
            arm[outer] = {**arm.get(outer, {}), inner: value}
        else:
            arm[key] = value
    searcher = parse_overrides(searcher_override)
    if searcher:
        arm['searcher_kwargs'] = {**arm.get('searcher_kwargs', {}), **searcher}
    return arm


def build_parser(cfg):
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--suite', required=True, choices=SUITES)
    p.add_argument('--env', required=True, choices=sorted(cfg.cells))
    p.add_argument('--method', required=True, choices=sorted(cfg.arms))
    p.add_argument('--trial', type=int, default=1)
    p.add_argument('--seed', type=int, default=None,
                   help='default: the trial index')
    p.add_argument('--output_dir', required=True)
    p.add_argument('--gpus', default=None,
                   help='CUDA_VISIBLE_DEVICES value; read before jax loads')
    p.add_argument('--wandb_project', default=None,
                   help="accepted for run_condition's sake; the runners do "
                        'not log to wandb')
    # Budget overrides, for a smoke test. They land in the run's config, so a
    # short run cannot later be mistaken for a real one.
    p.add_argument('--num_generations', type=int, default=None)
    p.add_argument('--num_updates', type=int, default=None)
    p.add_argument('--task_interval', type=int, default=None,
                   help="in the method's own units: generations for NE, "
                        'updates for RL')
    p.add_argument('--eval_episodes', type=int, default=cfg.eval_episodes)
    p.add_argument('--num_evals', type=int, default=cfg.ne_num_evals)
    p.add_argument('--pop_size', type=int, default=cfg.ne_pop_size)
    p.add_argument('--episode_length', type=int, default=None,
                   help="override the suite table's episode length. It "
                        'scales the NE budget but NOT the RL one (updates x '
                        'envs x steps); scale --num_updates too or the two '
                        'stop being matched.')
    p.add_argument('--ppo_override', nargs='*', default=None,
                   metavar='KEY=VALUE',
                   help='entries of PPO_CONFIGS to replace, e.g. num_envs=32 '
                        'for a smoke test. RL methods only; every override '
                        'lands in the run config.')
    p.add_argument('--ne_override', nargs='*', default=None,
                   metavar='KEY=VALUE',
                   help="entries of the NE method's settings to replace, e.g. "
                        'sigma=0.03, or a dotted key one level down, '
                        'searcher_kwargs.elite_ratio=0.05. NE methods only; '
                        'every override lands in the run config.')
    p.add_argument('--searcher_override', nargs='*', default=None,
                   metavar='KEY=VALUE',
                   help="entries of the NE method's `searcher_kwargs` to "
                        "replace -- GA's elite_ratio, DNS's k or "
                        'cross_over_rate.')
    p.add_argument('--checkpoint_interval', type=int, default=None,
                   help='how often to append a parameter snapshot to '
                        "trajectory.npz, in the method's own units. None is "
                        "the runner's default.")
    p.add_argument('--checkpoint_every', type=int, default=0,
                   help='save the whole training state every N generations '
                        '(NE) / updates (RL) to <output_dir>/resume.pkl; a '
                        'later run with the same command resumes from it. '
                        'Use a phase length. 0 disables.')
    p.add_argument('--max_gens_this_run', type=int, default=0,
                   help='exit cleanly at the next checkpoint once this many '
                        'generations / updates have run in THIS process. '
                        '0 means run to the end.')
    p.add_argument('--oracle', action='store_true',
                   help='give the method its task boundaries -- the '
                        'deliberately informed control. OFF for every '
                        'reported run.')
    p.add_argument('--dry_run', action='store_true',
                   help='print the resolved settings and exit before training')
    cfg.add_args(p)
    return p


def budget_overrides(cfg, args, fam):
    """The command-line flags that move this run off the config's budget."""
    names = ['task_interval', 'episode_length']
    names += (['num_generations', 'pop_size', 'num_evals'] if fam == 'ne'
              else ['num_updates'])
    out = [n for n in names if getattr(args, n)
           and getattr(args, n) != {'pop_size': cfg.ne_pop_size,
                                    'num_evals': cfg.ne_num_evals}.get(n)]
    if fam == 'rl':
        out += [k for k in parse_overrides(args.ppo_override)
                if k in ('num_envs', 'num_steps', 'num_updates', 'task_interval')]
    return out


def load_config(argv):
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument('--suite', choices=SUITES)
    known, _ = pre.parse_known_args(argv)
    if known.suite is None:
        raise SystemExit(f'--suite is required: one of {", ".join(SUITES)}')
    return suite_config.load(known.suite)


def build_run(cfg, args):
    """`(family, phase, common, runner kwargs)`. No side effects."""
    fam = cfg.family(args.method)
    phase, common = cfg.resolve(args)
    common.update(
        eval_episodes=args.eval_episodes,
        seed=args.trial if args.seed is None else args.seed,
        trial=args.trial,
        output_dir=args.output_dir,
    )
    run = dict(
        resume_path=(os.path.join(args.output_dir, 'resume.pkl')
                     if args.checkpoint_every else None),
        checkpoint_every=args.checkpoint_every,
        **({'episode_length': args.episode_length}
           if args.episode_length else {}),
        **({'checkpoint_interval': args.checkpoint_interval}
           if args.checkpoint_interval else {}),
    )
    if fam == 'ne':
        arm = apply_ne_overrides(cfg.ne_method(args), args.ne_override,
                                 args.searcher_override)
        gens, interval = cfg.ne_budget(args, phase)
        kwargs = dict(
            num_generations=args.num_generations or gens,
            task_interval=args.task_interval or interval,
            pop_size=args.pop_size,
            num_evals=args.num_evals,
            max_gens_this_run=args.max_gens_this_run,
            **run,
            **cfg.ne_extra(args),
            **arm, **common)
    else:
        from source.algorithms.rl.pbt import pbt_kwargs
        updates, interval = cfg.rl_budget(args, phase)
        # The config's PPO settings for this environment on top of
        # PPO_CONFIGS, then the command line's on top of both.
        overrides = {**cfg.ppo_overrides(args),
                     **parse_overrides(args.ppo_override)}
        kwargs = dict(
            # `pbt` / `pbt2` are one method at two population sizes, and
            # `pbt_weights` / `pbt2_weights` the same without explore.
            **pbt_kwargs(args.method),
            overrides=overrides or None,
            num_updates=args.num_updates or updates,
            task_interval=args.task_interval or interval,
            # Off unless --oracle. The runner's own default is already off;
            # passed explicitly so every run's config records which it was.
            cchain_reset_on_switch=bool(args.oracle),
            max_updates_this_run=args.max_gens_this_run,
            **run,
            **cfg.rl_extra(args),
            **common)
    return fam, phase, common, kwargs


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    cfg = load_config(argv)
    args = build_parser(cfg).parse_args(argv)
    cfg.setup(args)
    select_gpus()
    fam, phase, common, kwargs = build_run(cfg, args)
    matched = cfg.matched_steps(args, phase)

    print('=' * 70)
    print(f'{args.suite} | {args.method} ({fam}) | {args.env} | {phase} '
          f'| trial {args.trial}')
    for line in cfg.describe(args, common):
        print(f'  {line}')
    print(f"  schedule       : {common['schedule']}, "
          f"{common['num_tasks']} sub-tasks")
    print(f'  matched budget : {matched:.3e} environment steps')
    overridden = budget_overrides(cfg, args, fam)
    if overridden:
        # `matched` is the config's budget; these change this run's and not
        # the other family's, so the run is no longer compute-matched.
        print(f"  NOT MATCHED    : budget overridden ({', '.join(overridden)})")
    if args.ppo_override and fam == 'rl':
        print(f'  ppo overrides  : {parse_overrides(args.ppo_override)}')
    if (args.ne_override or args.searcher_override) and fam == 'ne':
        print(f"  ne overrides   : {args.ne_override or []} "
              f"{args.searcher_override or []}")
    print(f"  seed           : {common['seed']}")
    print(f'  boundary info  : {"ORACLE" if args.oracle else "none"}')
    print(f'  output         : {args.output_dir}')
    print('=' * 70, flush=True)
    if args.dry_run:
        return 0

    os.makedirs(args.output_dir, exist_ok=True)
    # The run keeps its own log beside its artifacts, so a finished run is
    # self-contained once the launcher's per-job log has been cleaned up.
    sys.stdout = Tee(os.path.join(args.output_dir, 'train.log'))
    # `kill -USR1 <pid>` writes every Python thread's stack to
    # <output_dir>/stacks.txt, for a run that hangs where py-spy needs root.
    import faulthandler
    import signal
    faulthandler.register(
        signal.SIGUSR1, all_threads=True,
        file=open(os.path.join(args.output_dir, 'stacks.txt'), 'a'))

    if fam == 'ne':
        from source.runners.train_nes import run_nes as runner
    else:
        from source.runners.train_ppo import run_ppo as runner
    result = runner(**kwargs)

    if isinstance(result, dict) and result.get('resumed_segment'):
        # A segment that stopped at a phase boundary has no artifacts yet --
        # only the last segment writes them. Exit 0 so a clean segment end
        # does not read as a crash.
        print(f"segment finished cleanly at {result['stopped_at']}; "
              'resume.pkl left for the next one')
        return 0

    # `config.json` beside the `results.json` the runner wrote. The figure
    # scripts read whichever they reach first, so the two must agree; this is
    # that same dict.
    with open(os.path.join(args.output_dir, 'config.json'), 'w') as f:
        json.dump(result['config'], f, indent=2)

    print(f'\nDone. {args.output_dir}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
