"""Benchmark configs: `source/configs/<suite>.yaml` and the logic around it.

A YAML file says what an experiment on one benchmark IS: its cells (an
environment plus a task schedule), its methods and their hyperparameters, and
the compute-matched budgets of the two families. This module loads one and
answers what `source/run.py` asks of it:

    cells, arms, continual_cells, reported_arms   the valid --env / --method values
    eval_episodes, ne_pop_size, ne_num_evals      parser defaults
    family(arm) -> 'ne' | 'rl'
    add_args(parser)                              the suite's own flags
    setup(args)                                   runs before jax is imported
    resolve(args) -> (phase, kwargs common to both runners)
    matched_steps(args, phase) -> the compute-matched budget, after checking it
    ne_method(args) -> the NE method's run_nes keyword arguments
    ne_budget(args, phase), rl_budget(args, phase) -> (length, task_interval)
    ne_extra(args), rl_extra(args)                further runner keyword arguments
    ppo_overrides(args)                           PPO_CONFIGS entries to replace
    describe(args, common) -> extra header lines

    gymnax     CartPole, Acrobot, MountainCar
    minigrid   two EmptyRandom rooms
    kinetix    the twenty hand-designed Kinetix levels
    mjx        the HalfCheetah body

NO JAX at import: `scripts/train/run.sh` and `source/run.py` load a config
before `CUDA_VISIBLE_DEVICES` is set. The budget checks import the runners
lazily.
"""

from __future__ import annotations

import copy
import os

import yaml

from source.envs import kinetix_levels

CONFIG_DIR = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), 'configs')


def parse_overrides(items):
    """`KEY=VALUE` pairs to a dict; ints and floats cast, the rest strings."""
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


def load(suite):
    """The config for one suite, e.g. `load('gymnax')`."""
    if suite not in SUITES:
        raise SystemExit(f'unknown suite {suite!r}; have {", ".join(SUITES)}')
    with open(os.path.join(CONFIG_DIR, f'{suite}.yaml')) as f:
        return SUITES[suite](yaml.safe_load(f))


def _phase(continual):
    return 'continual' if continual else 'noncontinual'


class Suite:
    """What every suite shares. A subclass builds `cells` and `continual_cells`
    and overrides whatever its benchmark does differently."""

    title = None

    def __init__(self, raw):
        self.raw = raw
        self.ne_arms = raw.get('ne_arms', {})
        self.rl_arms = tuple(raw['rl_arms'])
        self.reported_arms = tuple(raw['reported_arms'])
        self.eval_episodes = raw['eval_episodes']
        self.ne_pop_size = raw['ne_pop_size']
        self.ne_num_evals = raw['ne_num_evals']
        self.episode_length = raw['episode_length']

    @property
    def arms(self):
        return tuple(self.ne_arms) + self.rl_arms

    def family(self, arm):
        if arm in self.ne_arms:
            return 'ne'
        if arm in self.rl_arms:
            return 'rl'
        raise ValueError(f'unknown arm {arm!r}; have {sorted(self.arms)}')

    def add_args(self, p):
        pass

    def setup(self, args):
        pass

    def ne_method(self, args):
        # A fresh copy: YAML anchors share one dict between arms.
        return copy.deepcopy(self.ne_arms[args.method])

    def ne_extra(self, args):
        return {}

    def rl_extra(self, args):
        # Updates between ReDo passes. 50 on every body: the interval the
        # original per-method gymnax and Kinetix PPO trainers used. The
        # runner's own default (1000) gave a stationary Kinetix run one pass.
        return dict(redo_interval=self.raw['redo_interval'])

    def ppo_overrides(self, args):
        return {}

    def describe(self, args, common):
        return []

    # The fixed phase grid minigrid and mjx use: `num_phases` equal phases of
    # `ne_generations` / `rl_updates`, the RL shape owned by PPO_CONFIGS.

    def ne_budget(self, args, phase):
        gens = self.raw['ne_generations']
        return gens, gens // self.raw['num_phases']

    def rl_budget(self, args, phase):
        updates = self.raw['rl_updates']
        return updates, updates // self.raw['num_phases']

    def check_against_ppo(self, env_name):
        """Assert the two families really are matched; the NE step budget.

        A silent mismatch here is the failure that matters most: both arms
        finish, both write curves, and the figure compares a method against
        another method with more environment steps or a different number of
        task changes. Cheap to check, invisible if it is ever wrong.
        """
        from source.runners.train_ppo import PPO_CONFIGS

        hp = PPO_CONFIGS[env_name]
        num_phases = self.raw['num_phases']
        gens, ne_interval = self.ne_budget(None, None)
        ne_steps = gens * self.ne_pop_size * self.ne_num_evals * self.episode_length
        rl_steps = hp['num_updates'] * hp['num_envs'] * hp['num_steps']
        problems = []
        if ne_steps != rl_steps:
            problems.append(f'budget: NE {ne_steps:.3e} steps vs RL {rl_steps:.3e}')
        if hp['num_updates'] != self.raw['rl_updates']:
            problems.append(f"updates: PPO_CONFIGS {hp['num_updates']} vs "
                            f"settings {self.raw['rl_updates']}")
        if hp['num_updates'] // hp['task_interval'] != num_phases:
            problems.append(
                f"phases: RL {hp['num_updates'] // hp['task_interval']} vs "
                f'{num_phases}')
        if gens // ne_interval != num_phases:
            problems.append(f'phases: NE {gens // ne_interval} vs {num_phases}')
        if problems:
            raise SystemExit(f'{self.title} study settings are inconsistent:\n  '
                             + '\n  '.join(problems))
        return ne_steps


# ---------------------------------------------------------------------------
# gymnax
# ---------------------------------------------------------------------------

class Gymnax(Suite):
    title = 'gymnax'

    def __init__(self, raw):
        super().__init__(raw)
        # cell -> (env, noise sigma or None for a stationary cell)
        self.cells = {env: (env, None) for env in raw['envs']}
        self.cells.update({f'{env}_sigma{s}': (env, s)
                           for env in raw['envs'] for s in raw['sigmas']})
        self.continual_cells = tuple(c for c, (_e, s) in self.cells.items()
                                     if s is not None)

    def add_args(self, p):
        p.add_argument('--num_phases', type=int, default=self.raw['num_phases'],
                       help='continual cells: how many phases the run has')
        p.add_argument('--num_tasks', type=int, default=None,
                       help='continual cells: how many DISTINCT sub-tasks the '
                            'phases cycle through; default one per phase. '
                            '--num_phases 20 --num_tasks 10 visits each twice, '
                            'which is what makes forgetting measurable.')
        p.add_argument('--task_type', default='noise',
                       choices=['noise', 'param', 'actions'],
                       help='what a sub-task is: an observation offset (the '
                            "cell's sigma), a physics multiplier, or a reversed "
                            'action map')
        p.add_argument('--param_name', default=None,
                       help='--task_type param: the physics group to rescale; '
                            "default the environment's own")
        p.add_argument('--param_range', nargs=2, type=float, default=None,
                       metavar=('LOW', 'HIGH'),
                       help='--task_type param: the multiplier range; default '
                            "the environment's own")

    def resolve(self, args):
        env, sigma = self.cells[args.env]
        if sigma is None:
            return 'noncontinual', dict(env_name=env, schedule='task0',
                                        num_tasks=1, noise_range=0.0)
        task_options = {}
        if args.task_type == 'param':
            task_options = {'task_mod': 'physics',
                            'physics_mult_range': (
                                ','.join(str(v) for v in args.param_range)
                                if args.param_range else 'default')}
            if args.param_name:
                task_options['physics_param'] = args.param_name
        elif args.task_type == 'actions':
            task_options = {'task_mod': 'actions'}
        common = dict(env_name=env, schedule='switch',
                      num_tasks=args.num_tasks or args.num_phases,
                      noise_range=float(sigma))
        if task_options:
            common['task_options'] = task_options
        return 'continual', common

    # The budget: RL updates are derived from the NE generations.

    def _ne_generations(self, args, phase):
        if phase == 'continual':
            return args.num_phases * self.raw['ne_task_interval_continual']
        return self.raw['ne_generations']

    def _num_phases(self, args, phase):
        return args.num_phases if phase == 'continual' else self.raw['num_phases']

    def _ne_steps(self, gens):
        return gens * self.ne_pop_size * self.ne_num_evals * self.episode_length

    def _rl_updates(self, gens):
        """PPO updates matched to `gens` generations at the reported population."""
        steps = self._ne_steps(gens)
        per_update = self.raw['rl_num_envs'] * self.raw['rl_num_steps']
        if steps % per_update:
            raise SystemExit(f'{steps} NE steps is not a whole number of PPO '
                             f'updates of {per_update} steps')
        return steps // per_update

    def matched_steps(self, args, phase):
        # The updates are derived from this file's `rl_num_envs` x
        # `rl_num_steps`; the runner steps with PPO_CONFIGS'. One number,
        # checked here rather than kept equal by hand.
        from source.runners.train_ppo import PPO_CONFIGS
        env = self.cells[args.env][0]
        hp = {**PPO_CONFIGS[env], **self.ppo_overrides(args)}
        shape = (hp['num_envs'], hp['num_steps'])
        if shape != (self.raw['rl_num_envs'], self.raw['rl_num_steps']):
            raise SystemExit(
                f'gymnax study settings are inconsistent: PPO_CONFIGS[{env!r}] '
                f'steps {shape[0]} x {shape[1]}, configs/gymnax.yaml '
                f"{self.raw['rl_num_envs']} x {self.raw['rl_num_steps']}")
        gens = self._ne_generations(args, phase)
        self._rl_updates(gens)           # raises if the two cannot be matched
        return self._ne_steps(gens)

    def ne_budget(self, args, phase):
        gens = self._ne_generations(args, phase)
        return gens, gens // self._num_phases(args, phase)

    def rl_budget(self, args, phase):
        updates = self._rl_updates(self._ne_generations(args, phase))
        return updates, updates // self._num_phases(args, phase)

    def ne_method(self, args):
        # A per-environment arm replaces the default one whole.
        env = self.cells[args.env][0]
        by_env = self.raw.get('ne_by_env', {}).get(env, {})
        if args.method in by_env:
            return copy.deepcopy(by_env[args.method])
        return super().ne_method(args)

    def ne_extra(self, args):
        # The plasticity columns the figures read (ne_centroid_* / ne_elite_*),
        # measured for the ReLU MLP with a categorical head that gymnax uses.
        return dict(track_plasticity=True, plasticity_activation='relu',
                    plasticity_continuous=False)

    def ppo_overrides(self, args):
        env = self.cells[args.env][0]
        return {**self.raw['ppo'], **self.raw['ppo_by_env'].get(env, {})}

    def describe(self, args, common):
        env, sigma = self.cells[args.env]
        lines = [f'environment    : {env}']
        if sigma is not None:
            lines.append(f'sub-task       : {args.task_type}'
                         + (f', sigma {sigma}' if args.task_type == 'noise' else ''))
            lines.append(f'phases         : {args.num_phases}')
        return lines


# ---------------------------------------------------------------------------
# MiniGrid
# ---------------------------------------------------------------------------

class MiniGrid(Suite):
    title = 'MiniGrid'

    def __init__(self, raw):
        super().__init__(raw)
        # cell -> schedule over the rooms
        self.cells = dict(raw['cells'])
        self.continual_cells = tuple(c for c, s in self.cells.items()
                                     if s == 'switch')

    def resolve(self, args):
        common = dict(
            env_name=self.raw['env_name'],
            schedule=self.cells[args.env],
            num_tasks=self.raw['num_tasks'],
            # The rooms this cell is built from, in order. `task0`/`task1` then
            # pin the schedule to one of them, but the ENVIRONMENT is still
            # built from the pair -- so the observation encoding, the episode
            # scan length and the policy are identical in all three cells, and
            # a stationary run is the switching run minus the switch.
            task_options={'envs': ','.join(self.raw['rooms'])},
            # A row of `noise_vectors` is an environment index on this suite,
            # not an observation offset. Passed explicitly so the run's config
            # records that a sub-task here is not a perturbation.
            noise_range=0.0,
        )
        return _phase(args.env in self.continual_cells), common

    def matched_steps(self, args, phase):
        return self.check_against_ppo(self.raw['env_name'])

    def describe(self, args, common):
        return [f"rooms          : {tuple(self.raw['rooms'])}",
                f"phases         : {self.raw['num_phases']}"]


# ---------------------------------------------------------------------------
# Kinetix
# ---------------------------------------------------------------------------

class Kinetix(Suite):
    title = 'Kinetix'

    def __init__(self, raw):
        super().__init__(raw)
        # cell -> the levels it visits; the level list is kinetix_levels'.
        self.cells = dict(kinetix_levels.CELLS)
        self.continual_cells = (kinetix_levels.CELL_ALL,)
        self.num_tasks = len(kinetix_levels.LEVELS)

    def add_args(self, p):
        p.add_argument('--dns_descriptor', default=None,
                       choices=['handcrafted', 'aurora'],
                       help="override the DNS arms' descriptor. Default is "
                            '`handcrafted` -- the six-channel duty factor; '
                            '`aurora` learns one online from the 13-value '
                            'per-step feature vector.')
        p.add_argument('--joint', type=int, default=0,
                       help="search the chain cell's first K levels jointly")
        p.add_argument('--objective', default='capped',
                       choices=['capped', 'mean', 'min', 'worstk', 'capped_worstk'],
                       help='joint only: how the K per-level returns become one fitness')

    def resolve(self, args):
        levels = self.cells[args.env]
        continual = args.env in self.continual_cells
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
            task_options={'levels': ','.join(levels)},
            # A row of `noise_vectors` is a LEVEL INDEX on this suite, not an
            # observation offset. Passed explicitly so the run's config records
            # that a sub-task here is not a perturbation.
            noise_range=0.0,
        )
        if args.joint:
            if self.family(args.method) != 'ne':
                raise SystemExit('--joint is an NE-only control: run_ppo has no '
                                 'joint objective')
            return 'joint', common
        return _phase(continual), common

    def ne_extra(self, args):
        return {'objective': args.objective} if args.joint else {}

    def ne_method(self, args):
        arm = super().ne_method(args)
        if args.dns_descriptor is not None and 'searcher_kwargs' in arm:
            arm['searcher_kwargs']['descriptor'] = args.dns_descriptor
        return arm

    # The budget: a stationary cell is one level's generations cut into
    # `noncontinual_phases`; the chain is one level's generations per level.

    def ne_budget(self, args, phase):
        per_level = self.raw['generations_per_level']
        stationary_interval = per_level // self.raw['noncontinual_phases']
        if args.joint:
            # A joint generation scores every genome on all K levels, so it
            # costs K generations of one level: K levels' budget -- what the
            # chain spends on them -- is one level's generations.
            return per_level, stationary_interval
        if phase == 'continual':
            return per_level * self.num_tasks, per_level
        return per_level, stationary_interval

    def rl_budget(self, args, phase):
        updates = self.raw['rl_updates']
        if phase == 'continual':
            return updates * self.num_tasks, updates
        return updates, updates // self.raw['noncontinual_phases']

    def matched_steps(self, args, phase):
        """The compute match, recomputed. Raises if the two families have drifted
        apart; a comparison that is not matched is not a comparison and a
        comment saying it is matched is not a check."""
        # Per level, whatever the cell: one stationary cell's budget.
        gens = self.raw['generations_per_level']
        ne_interval = gens // self.raw['noncontinual_phases']
        updates = self.raw['rl_updates']
        rl_interval = updates // self.raw['noncontinual_phases']
        ne_steps = gens * self.ne_pop_size * self.ne_num_evals * self.episode_length
        rl_steps = updates * self.raw['rl_num_envs'] * self.raw['rl_num_steps']
        if ne_steps != rl_steps:
            raise AssertionError(
                f'Kinetix budgets have drifted: NE {ne_steps:.4e} environment '
                f'steps against RL {rl_steps:.4e}')
        if gens // ne_interval != updates // rl_interval:
            raise AssertionError(
                f'Kinetix phase grids have drifted: NE {gens // ne_interval} '
                f'phases against RL {updates // rl_interval}; both families '
                'must meet a checkpoint at the same point on the shared wall of '
                'steps')
        batch = self.raw['eval_batch_size']
        if self.ne_pop_size % batch:
            raise AssertionError(
                f'population {self.ne_pop_size} is not divisible by the Kinetix '
                f'eval_batch_size ({batch}); the chunked scoring pass needs it to be')
        # The suite table is what `build_env` reads; the YAML is what the queue
        # scripts and the arithmetic above read. One number, checked here
        # rather than kept equal by hand (lazy import: the suite pulls in jax).
        from source.envs.kinetix import ENV_CONFIGS
        table_batch = int(ENV_CONFIGS[kinetix_levels.CELL_ALL]['eval_batch_size'])
        if table_batch != batch:
            raise AssertionError(
                f'source/envs/kinetix.py has eval_batch_size {table_batch} '
                f'against configs/kinetix.yaml {batch}')
        table = int(ENV_CONFIGS[kinetix_levels.CELL_ALL]['episode_length'])
        if table != self.episode_length:
            raise AssertionError(
                f'source/envs/kinetix.py has episode_length {table} against '
                f'configs/kinetix.yaml episode_length {self.episode_length}; the '
                'budget is computed from the wrong number')
        if phase == 'joint':
            return ne_steps * args.joint
        return ne_steps * self.num_tasks if phase == 'continual' else ne_steps

    def describe(self, args, common):
        lv = common['task_options']['levels'].split(',')
        lines = [f'levels         : {len(lv)} ({", ".join(lv[:3])}'
                 f'{", ..." if len(lv) > 3 else ""})']
        if args.joint:
            lines.append(f'objective      : {args.objective}')
        return lines


# ---------------------------------------------------------------------------
# mjx
# ---------------------------------------------------------------------------

class Mjx(Suite):
    title = 'mjx'

    def __init__(self, raw):
        super().__init__(raw)
        self.bodies = raw['bodies']
        # The NE arm names; their settings are per body.
        self.ne_arms = {arm: None for body in self.bodies.values()
                        for arm in body['ne_arms']}
        # cell -> (body, schedule, task_mod, extra task options)
        low, high = raw['friction_range']
        self.cells = {}
        for body, spec in self.bodies.items():
            friction = {'friction_order': raw['friction_order'],
                        'friction_low': spec.get('friction_low', low),
                        'friction_high': high}
            for suffix, cell in raw['cells'].items():
                extra = {**(friction if cell.get('friction') else {}),
                         **cell.get('options', {})}
                self.cells[f'{body}{suffix}'] = (body, cell['schedule'],
                                                 cell['task_mod'], extra)
        self.continual_cells = tuple(c for c, v in self.cells.items()
                                     if v[1] != 'task0')

    def _body(self, cell):
        return self.bodies[self.cells[cell][0]]

    def add_args(self, p):
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
                       help='make the FIRST phase this many NE GENERATIONS long, '
                            'and every later phase the usual length. An RL arm '
                            'gets the same share of the budget in updates. 0 '
                            'is the uniform grid.')
        p.add_argument('--num_tasks', type=int, default=self.raw['num_tasks'])
        p.add_argument('--obs_norm', action='store_true',
                       help='whiten the observation before the NE policy reads '
                            'it, the way _MJX_PPO already does for the RL arms '
                            '(normalize_obs: True). Lands in the run config.')
        p.add_argument('--track_plasticity', action='store_true',
                       help='record the plasticity columns -- ne_centroid_* and '
                            'ne_elite_* dormancy, action churn and NTK rank. An '
                            'observer on its own RNG stream; OFF by default so '
                            'runs made before it are reproduced.')
        p.add_argument('--plasticity_interval', type=int, default=10,
                       help='generations between plasticity measurements.')

    def setup(self, args):
        # Before any env import: source/envs/mjx.py reads NE_OBS_NORM when it
        # constructs the TaskSpec. NE arms only -- the RL arms never whiten this
        # way (their normaliser is folded into the weights).
        if args.obs_norm and self.family(args.method) == 'ne':
            os.environ['NE_OBS_NORM'] = '1'

    def resolve(self, args):
        _body, schedule, task_mod, extra = self.cells[args.env]
        body = self._body(args.env)
        # The cell's own offset width; 0 on the cells whose sub-task is not an
        # offset, so a finished run's config does not record a width it never
        # drew.
        own_noise = body['noise_range'] if task_mod == 'obs_noise' else 0.0
        common = dict(
            # The cell carries the body; nothing here branches on which.
            env_name=body['env_name'],
            schedule=schedule,
            num_tasks=args.num_tasks,
            task_warmup=self._warmup(args),
            # What a sub-task IS on this cell -- an observation offset or a
            # ground friction multiplier -- plus the reward's target speed. The
            # suite reads it; nothing here branches on it.
            task_options={'task_mod': task_mod,
                          'target_speed': body['target_speed'], **extra,
                          **({'observe_task': True} if args.observe_task else {}),
                          **parse_overrides(args.task_options)},
            noise_range=(own_noise if args.noise_range is None
                         else float(args.noise_range)),
        )
        return _phase(args.env in self.continual_cells), common

    def _warmup(self, args):
        """`--task_warmup` in the method's own units. Given in generations, so
        an RL arm's first phase ends at the same environment step as NE's."""
        if not args.task_warmup or self.family(args.method) == 'ne':
            return args.task_warmup
        _gens, ne_interval = self.ne_budget(args, None)
        _updates, rl_interval = self.rl_budget(args, None)
        if rl_interval % ne_interval:
            raise SystemExit(f'--task_warmup: {rl_interval} updates a phase is not '
                             f'a whole multiple of {ne_interval} generations')
        return args.task_warmup * (rl_interval // ne_interval)

    def matched_steps(self, args, phase):
        return self.check_against_ppo(self._body(args.env)['env_name'])

    def ne_method(self, args):
        # This BODY's widths -- sigma does not transfer between bodies.
        return copy.deepcopy(self._body(args.env)['ne_arms'][args.method])

    def ne_extra(self, args):
        return dict(obs_norm=bool(args.obs_norm),
                    track_plasticity=args.track_plasticity,
                    plasticity_interval=args.plasticity_interval,
                    # tanh / continuous: the body uses ContinuousMLPPolicy.
                    plasticity_activation='tanh', plasticity_continuous=True)

    def describe(self, args, common):
        body, _schedule, task_mod, _extra = self.cells[args.env]
        ts = common['task_options']['target_speed']
        return [f'body           : {body}',
                'sub-task       : ' + (
                    f"{task_mod}, sigma {common['noise_range']}"
                    if task_mod == 'obs_noise'
                    else f"{task_mod}, log-uniform {tuple(self.raw['friction_range'])}"),
                'reward         : ' + ('brax stock forward velocity'
                                       if str(ts).lower() == 'none'
                                       else f'speed tracking at {ts} m/s'),
                f"phases         : {self.raw['num_phases']}",
                f'obs whitening  : {"ON" if args.obs_norm else "off"}']


SUITES = {'gymnax': Gymnax, 'minigrid': MiniGrid, 'kinetix': Kinetix, 'mjx': Mjx}
