"""Run one mjx trial: one (method, cell, trial), one process, one GPU.

    python source/studies/mjx/cli.py --env ant_friction \
        --method ga --trial 1 --seed 42 --gpus 0 \
        --output_dir projects/iclr_2027/runs_ant/mjx/continual/ga/ant_friction/trial_1

A CELL carries its body, so `--env cheetah_noise` is the same command on the
other body and no flag anywhere selects between them.

The flag names are not a choice: they are exactly what
`scripts/train/run_experiments.sh:run_condition` passes, so the mjx blocks
reuse that function rather than reimplementing the output layout, the "already
done" skip and the failed-trial bookkeeping. `--env` is therefore a CELL name,
which is what a directory in the run tree is named after.

WHAT THIS FILE IS AND IS NOT
----------------------------
Argument plumbing, and the twin of `source/studies/minigrid/cli.py`. It holds
no training loop, no searcher and no environment: `settings.py` says what an
arm is, `source/envs/mjx.py` and `source/envs/brax_ant.py` are the body, and
the two suite-generic runners under `source/studies/generalists/` do the work.
If something here starts to look like an algorithm, it is in the wrong file.

The trainers under `source/studies/brax/` are the OLD ant path -- they
produced `runs_repro2/brax`, they cannot save a centroid and their DNS has no
gaussian operator. Nothing here imports them. The cheetah's old path
(`source/studies/mujoco/`, `source/envs/mjx_cheetah.py`) was deleted on
2026-09-08 and does not exist on this checkout.

THE OUTPUT LAYOUT is the gymnax tree's, because the figure scripts walk it:

    <root>/mjx/<continual|noncontinual>/<method>/<cell>/trial_<n>/

and every trial holds the files a gymnax trial holds -- `results.json`,
`config.json`, `training_metrics.json`, and `checkpoints.npz` carrying one
saved network per phase under the `centroid` key the centroid plasticity
figure reads.

NO ARM IS TOLD WHERE THE BOUNDARIES ARE (CLAUDE.md (d)). Nothing here passes a
boundary signal and both runners default to taking none: C-CHAIN's
`cchain_reset_on_switch` is off, and the GA and DNS re-evaluate their stored
population every generation rather than only at a transition. `--oracle`
builds the deliberately-informed control arm and is the only way to turn any of
it on.
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
from source.studies.mjx import settings as S               # noqa: E402


def build_parser():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    # `--env` is a CELL. run_experiments.sh calls it that for every suite and
    # names the run directory after it; on this body a cell is a kind of
    # sub-task plus a schedule rather than a single environment.
    p.add_argument('--env', required=True, choices=sorted(S.CELLS))
    p.add_argument('--method', required=True, choices=list(S.ARMS))
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
    # Checkpoint-restart across SLURM jobs, the Kinetix CLI's two flags with
    # the same meaning: CLUSTER's QoS cap is shorter than the ant-shape PPO on
    # the cheetah. Recorded in the run config; the run itself is the same run.
    p.add_argument('--checkpoint_every', type=int, default=0,
                   help='save the whole training state every N generations '
                        '(NE) / updates (RL) to <output_dir>/resume.pkl; a '
                        'later run with the same command resumes from it. '
                        'Use a phase length. 0 disables.')
    p.add_argument('--max_gens_this_run', type=int, default=0,
                   help='exit cleanly at the next checkpoint once this many '
                        'generations / updates have run in THIS process, so a '
                        'segment ends at a phase boundary instead of under a '
                        'SLURM kill. 0 means run to the end.')
    p.add_argument('--observe_task', action='store_true',
                   help='CONTROL, not a reported arm: append the sub-task '
                        "vector to the policy's observation, so a memoryless "
                        'policy can represent behaviour that depends on which '
                        'sub-task it is in. See `source/envs/mjx.TaskSpec.augment`.')
    p.add_argument('--noise_range', type=float, default=None,
                   help="width of the observation offset; None is the cell's "
                        'own. Only an obs_noise cell reads it.')
    p.add_argument('--task_options', nargs='*', default=None,
                   metavar='KEY=VALUE',
                   help="override this cell's task settings: task_mod, "
                        'friction_order / friction_low / friction_high / '
                        'friction_default, target_speed (`none` selects '
                        "brax's stock unbounded reward). See "
                        '`source/envs/mjx.build_env`.')
    p.add_argument('--task_warmup', type=int, default=0,
                   help='make the FIRST phase this long, in the same units as '
                        '--task_interval, and every later phase '
                        '--task_interval. 0 is the uniform grid every run '
                        'before 2026-09-11 used. For a body whose search '
                        'needs more generations than one phase has: converge '
                        'on the first sub-task, then switch fast.')
    p.add_argument('--num_tasks', type=int, default=S.NUM_TASKS)
    p.add_argument('--eval_episodes', type=int, default=S.EVAL_EPISODES)
    p.add_argument('--num_evals', type=int, default=S.NE_NUM_EVALS)
    p.add_argument('--pop_size', type=int, default=S.NE_POP_SIZE)
    p.add_argument('--ppo_override', nargs='*', default=None,
                   metavar='KEY=VALUE',
                   help='entries of PPO_CONFIGS to replace, e.g. num_envs=32 '
                        'for a smoke test. RL arms only, and every override '
                        'lands in the run config, so a shrunk run cannot pass '
                        'for a real one.')
    p.add_argument('--searcher_override', nargs='*', default=None,
                   metavar='KEY=VALUE',
                   help="entries of this arm's `searcher_kwargs` to replace -- "
                        "GA's elite_ratio, DNS's k or cross_over_rate. These are "
                        'NESTED inside the arm, so --ne_override cannot reach '
                        'them: it updates the top level and a key placed there '
                        'would never reach the searcher.')
    p.add_argument('--ne_override', nargs='*', default=None,
                   metavar='KEY=VALUE',
                   help="entries of this body's NE arm settings to replace, "
                        'e.g. sigma=0.1 learning_rate=0.01, or a dotted key '
                        'for one level down, searcher_kwargs.elite_ratio=0.05. '
                        'NE arms only. '
                        'Every override lands in the run config, so a run made '
                        'at a swept width cannot pass for one at the reported '
                        'width -- which is what this flag is for: a sigma '
                        'probe writes its value into the config it is read '
                        'back from.')
    p.add_argument('--obs_norm', action='store_true',
                   help='whiten the observation before the policy reads it, '
                        'the way _MJX_PPO already does for the RL arms '
                        '(normalize_obs: True). The ant spans 38x in per-dim '
                        'observation std, so without this the two families '
                        'read differently-scaled inputs -- an asymmetry the '
                        'compute-match check cannot see. Lands in the run '
                        'config.')
    p.add_argument('--track_plasticity', action='store_true',
                   help='record the gymnax plasticity columns -- ne_centroid_* '
                        'and ne_elite_* dormancy, action churn and NTK rank -- '
                        'so this body produces the same figure rows as the '
                        'gymnax tree. An observer on its own RNG stream; OFF '
                        'by default so runs made before it are reproduced.')
    p.add_argument('--plasticity_interval', type=int, default=10,
                   help='generations between plasticity measurements.')
    p.add_argument('--oracle', action='store_true',
                   help='give the arm its task boundaries -- the deliberately '
                        'informed control. OFF for every reported run; see '
                        'CLAUDE.md (d).')
    p.add_argument('--dry_run', action='store_true',
                   help='print the resolved settings and exit without '
                        'importing jax')
    return p


def _overrides(items):
    """`KEY=VALUE` pairs to a dict, ints and floats cast, rest left as strings."""
    out = {}
    for item in items or []:
        key, _, value = item.partition('=')
        for cast in (int, float):
            try:
                out[key] = cast(value)
                break
            except ValueError:
                continue
        else:
            out[key] = value
    return out



def resolve(args):
    """`(phase, kwargs common to both runners)`. No side effects."""
    _body, schedule, _task_mod, _extra = S.CELLS[args.env]
    phase = 'continual' if args.env in S.CONTINUAL_CELLS else 'noncontinual'
    common = dict(
        # The cell carries the body; nothing here branches on which.
        env_name=S.env_name(args.env),
        schedule=schedule,
        num_tasks=args.num_tasks,
        task_warmup=args.task_warmup,
        # What a sub-task IS on this cell -- an observation offset or a ground
        # friction multiplier -- plus the reward's target speed. The suite
        # reads it; nothing here branches on it.
        task_options={**S.task_options(args.env),
                      **({'observe_task': True} if args.observe_task else {}),
                      **_overrides(args.task_options)},
        # 0 on the friction cells, so a finished run's config does not record
        # an offset width it never drew.
        noise_range=(S.noise_range(args.env) if args.noise_range is None
                     else float(args.noise_range)),
        eval_episodes=args.eval_episodes,
        seed=args.trial if args.seed is None else args.seed,
        trial=args.trial,
        output_dir=args.output_dir,
    )
    return phase, common


def main():
    args = build_parser().parse_args()
    # Before build_env runs: source/envs/mjx.py reads NE_OBS_NORM when it
    # constructs the TaskSpec, so this has to be set ahead of any env import.
    # NE arms only. build_env reads this to measure the whitening statistics;
    # the RL arms never whiten (their normaliser is folded into the weights),
    # so on those the flag is ignored rather than left to compute stats that
    # nothing reads.
    if getattr(args, 'obs_norm', False) and S.family(args.method) == 'ne':
        os.environ['NE_OBS_NORM'] = '1'
    phase, common = resolve(args)
    select_gpus()
    matched_steps = S.check(args.env)
    fam = S.family(args.method)
    body, schedule, task_mod, _extra = S.CELLS[args.env]
    ppo_overrides = _overrides(args.ppo_override)
    ne_overrides = _overrides(args.ne_override)

    # THE EFFECTIVE BUDGET, recomputed from what this process will actually
    # run. `S.check` compares the two families' DEFAULTS; it cannot see a
    # --num_updates or a --ppo_override num_envs=..., and those are exactly how
    # a probe or a smoke test silently ends up on a different wall of
    # environment steps from the arm it is plotted against (CLAUDE.md (c)).
    # Reported rather than enforced: a deliberately short smoke run is
    # legitimate and lands in the config, so this prints the ratio and leaves
    # the reader to see it.
    if fam == 'ne':
        eff = ((args.num_generations or S.NE_GENERATIONS) * args.pop_size
               * args.num_evals * S.EPISODE_LENGTH)
        phases = ((args.num_generations or S.NE_GENERATIONS)
                  // (args.task_interval or S.NE_TASK_INTERVAL))
    else:
        from source.studies.generalists.train_ppo import PPO_CONFIGS
        hp = dict(PPO_CONFIGS[S.env_name(args.env)])
        hp.update(ppo_overrides)
        updates = args.num_updates or hp['num_updates']
        eff = updates * hp['num_envs'] * hp['num_steps']
        phases = updates // (args.task_interval or hp['task_interval'])

    print('=' * 70)
    print(f'{body} | {args.method} ({fam}) | {args.env} | {phase} '
          f'| trial {args.trial}')
    print(f'  sub-task       : {task_mod}'
          + (f", sigma {common['noise_range']}" if task_mod == 'obs_noise'
             else f', log-uniform {S.FRICTION_RANGE}'))
    print(f'  schedule       : {schedule}, {S.NUM_PHASES} phases over '
          f"{common['num_tasks']} sub-tasks")
    _ts = common['task_options']['target_speed']
    print('  reward         : ' + ('brax stock forward velocity'
                                   if str(_ts).lower() == 'none'
                                   else f'speed tracking at {_ts} m/s'))
    print(f'  matched budget : {matched_steps:.3e} environment steps')
    print(f'  this run       : {eff:.3e} steps over {phases} phases '
          f'({100 * eff / matched_steps:.1f}% of matched)')
    if ppo_overrides:
        print(f'  ppo overrides  : {ppo_overrides}')
    if ne_overrides:
        print(f'  ne overrides   : {ne_overrides}')
    print(f"  seed           : {common['seed']}")
    print(f'  obs whitening  : {"ON" if getattr(args, "obs_norm", False) else "off"}')
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
    # `kill -USR1 <pid>` writes every Python thread's stack to
    # <output_dir>/stacks.txt. Long ant-shape PPO runs on the home server have
    # twice hung after their last update, asleep in futex_wait, and py-spy
    # needs root there; this is the only way to see where.
    import faulthandler
    import signal
    faulthandler.register(
        signal.SIGUSR1, all_threads=True,
        file=open(os.path.join(args.output_dir, 'stacks.txt'), 'a'))

    resume = dict(resume_path=(os.path.join(args.output_dir, 'resume.pkl')
                               if args.checkpoint_every else None),
                  checkpoint_every=args.checkpoint_every)
    if fam == 'ne':
        from source.studies.generalists.train_nes import run_nes
        # This BODY's widths -- sigma does not transfer between bodies -- then
        # whatever --ne_override says on top.
        arm = S.ne_arm(args.env, args.method)
        arm.update(ne_overrides)
        # `ne_arm` already returned a fresh copy of the nested dict, so this
        # cannot leak into another run in the same process.
        searcher_overrides = _overrides(args.searcher_override)
        if searcher_overrides:
            arm['searcher_kwargs'] = {**arm.get('searcher_kwargs', {}),
                                      **searcher_overrides}
            print(f'  searcher over. : {searcher_overrides}')
        result = run_nes(
            # tanh / continuous: both mjx bodies use ContinuousMLPPolicy.
            obs_norm=bool(getattr(args, 'obs_norm', False)),
            track_plasticity=args.track_plasticity,
            plasticity_interval=args.plasticity_interval,
            plasticity_activation='tanh', plasticity_continuous=True,
            num_generations=args.num_generations or S.NE_GENERATIONS,
            task_interval=args.task_interval or S.NE_TASK_INTERVAL,
            pop_size=args.pop_size,
            num_evals=args.num_evals,
            max_gens_this_run=args.max_gens_this_run, **resume,
            **arm, **common)
    else:
        from source.studies.generalists.train_ppo import pbt_kwargs, run_ppo
        result = run_ppo(
            # `pbt` / `pbt2` are one method at two population sizes, and
            # `pbt_weights` / `pbt2_weights` the same without explore.
            **pbt_kwargs(args.method),
            overrides=ppo_overrides or None,
            num_updates=args.num_updates or S.RL_UPDATES,
            task_interval=args.task_interval or S.RL_TASK_INTERVAL,
            # CLAUDE.md (d): off unless --oracle. The runner's own default is
            # already off; passed explicitly so every mjx run's config records
            # which it was rather than leaving a reader to know what the
            # default was on the day it ran.
            cchain_reset_on_switch=bool(args.oracle),
            max_updates_this_run=args.max_gens_this_run, **resume,
            **common)

    if isinstance(result, dict) and result.get('resumed_segment'):
        # A segment that stopped at a phase boundary has no artifacts yet --
        # only the last segment writes them. Exit 0 so a clean segment end
        # does not read as a crash.
        print(f"segment finished cleanly at {result['stopped_at']}; "
              'resume.pkl left for the next one')
        return 0

    # `config.json` beside the `results.json` the runner wrote. `load_config`
    # in the figure scripts takes whichever it reaches first and reads
    # `blob['config'] or blob`, so the two must agree; this is that same dict.
    with open(os.path.join(args.output_dir, 'config.json'), 'w') as f:
        json.dump(result['config'], f, indent=2)

    print(f'\nDone. {args.output_dir}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
