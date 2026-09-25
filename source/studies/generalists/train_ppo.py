"""Train PPO on the same sub-tasks, the same schedules, the same budget as NES.

This writes exactly the artifacts ``train_nes.py`` writes -- ``results.json``,
``training_metrics.json``, ``trajectory.npz`` with the same keys -- so every
analysis script runs on a PPO tag unchanged and the two methods land on the
same figures. That is not left to the two files agreeing by hand: the schedule
and the artifacts both come from ``source/studies/generalists/common.py``, which
neither trainer owns. Three further things have to hold for the comparison to
mean anything, and all three are enforced here:

Same search space
    PPO's actor is the policy NES searches -- ``MLPPolicy(16, 16)`` on gymnax,
    ``ContinuousMLPPolicy(128, 128)`` on the mjx bodies -- so a PPO checkpoint
    and an NES centroid are points in one parameter space and can be plotted on
    one landscape. On the mjx bodies the actor adds a Gaussian's log-std and
    normalises its observations; both are folded out before a checkpoint or a
    score is taken (``source/studies/generalists/actors.py`` says how), so what is
    stored is the evolved policy's parameter vector. The critic is bigger and
    is *not* part of that space; it is discarded from ``trajectory.npz``.

Same fitness
    The logged score is the policy evaluated DETERMINISTICALLY -- ``argmax``
    actions on gymnax, the Gaussian's mean on the mjx bodies -- on a fixed
    number of episodes with the identical ``make_scoring_fn`` NES is scored
    with. PPO *trains* by sampling, which is a real difference between the
    methods, but it must not also be a difference in the ruler. Reporting PPO
    on its own stochastic-policy return would confound "learned less" with
    "was measured differently".

Same budget
    On gymnax 3.072e9 environment steps, matched to NES's 4000 generations x
    512 population x 3 evaluations x 500 steps, and split into the same twenty
    phases: 1500 PPO updates per phase against 200 NES generations. On the ant
    4.9152e8, matched to 320 generations x 512 x 3 x 1000 -- this repo's own
    ant budget -- as 2400 updates a phase against 16 generations. Both are in
    ``PPO_CONFIGS``.

The `joint` schedule collects a rollout on *each* sub-task each update and
trains on the concatenation, which is the direct analogue of NES scoring one
population on both.

Since 2026-09-05 the trainer runs on the mjx suite as well, through the same
``suites.py`` interface ``train_nes`` uses: the environment, the sub-task
vectors, the vectorised reset/step and the scoring function all come from the
suite, and the action head from ``actors.py``. A gymnax run is bit-identical
to one made before that change (checked on ppo, trac, redo and cchain).

Per-update plasticity diagnostics
    Since 2026-08-27 each record can also carry churn and dormancy, measured
    across ONE PPO update on a frozen probe batch -- the C-CHAIN estimator and
    the ReDo criterion, both imported from `source/metrics/plasticity.py`
    and `source/algorithms/rl/redo.py` rather than reimplemented, so an
    RL row here is the same measurement as an RL row in the benchmarking paper.
    Runs finished before that date do not have these columns; the keys are
    written only on the updates where they were measured, so a reader must not
    assume every record has them. See `--churn_interval`. On a continuous
    body the observer is `action_disagreement_continuous` and the method's own
    estimator the action-mean MSE, logged as `rl_churn_mse` rather than
    `rl_churn_ce`.

Records on the mjx bodies are SPARSE. A gymnax update is 102,400 steps and a
held-out evaluation is 16 x 500, so scoring every update is cheap; an ant
update is 10,240 steps and the evaluation 16 x 1000, so scoring every update
would cost more than training. ``eval_interval`` scores and records every N
updates, always including the last update of every phase, and the analysis
reads a record's ``generation`` rather than its index. Checkpoints follow the
records.
"""

from __future__ import annotations

import argparse
import os
import time
from functools import partial
from types import SimpleNamespace

import sys

# Runnable as a script (`python source/studies/generalists/train_ppo.py`, the
# way scripts/train/run_experiments.sh launches every trainer): the repo root
# goes on sys.path, as the suite CLIs do for themselves.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


def _select_gpu_early(argv):
    """``--gpus N`` must take effect BEFORE jax initialises, which happens
    at import below (actors.py evaluates a jnp constant). run_experiments.sh
    launches four of these per node, one per card; without this every one of
    them opened card 0 and three of four died of CUDA_ERROR_OUT_OF_MEMORY at
    import (CLUSTER, 2026-09-13). The suite CLIs do the same in select_gpus()."""
    for i, a in enumerate(argv):
        if a in ('--gpu', '--gpus') and i + 1 < len(argv):
            os.environ['CUDA_VISIBLE_DEVICES'] = argv[i + 1]
        elif a.startswith('--gpu=') or a.startswith('--gpus='):
            os.environ['CUDA_VISIBLE_DEVICES'] = a.split('=', 1)[1]


if __name__ == '__main__':
    _select_gpu_early(sys.argv)

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training.train_state import TrainState
from jax import random

from source.studies.generalists import actors
from source.studies.generalists.common import (
    SCHEDULES, make_phase_grid, make_task_sequence, record_centroid_scores, record_scores,
    summarise_records, write_run)
from source.envs.registry import ENV_NAMES, get_suite, suite_for
from source.algorithms.networks import ACTIVATIONS, ValueNetwork, get_flat_params
from source.metrics.population_diversity import (diversity_columns,
                                                 make_pairwise_behaviour_fn)
from source.algorithms.rl import ppo as ppo_lib
from source.algorithms.rl import redo as redo_lib
from source.algorithms.rl.redo import criterion_for_activation
from source.utils.run_artifacts import save_checkpoints

# The RL arm and the three continual-RL baselines the benchmarking paper runs.
# They are the SAME implementations that paper uses -- TRAC through
# `start_trac`, ReDo through `run_redo_pass`, C-CHAIN through
# `source/studies/gymnax/cchain.py` -- wired onto this study's sub-task construction so
# an RL row here is comparable with an NE row here AND with the paper's own
# numbers.
RL_METHODS = ('ppo', 'trac', 'redo', 'cchain', 'pbt')


#: Arm-name suffix for PBT without explore: the loser copies the winner's
#: weights and keeps the hyperparameters it started with (`--pbt_mode
#: weights_only`, Jaderberg et al. Sect. 4.1.2). `pbt_weights` / `pbt2_weights`
#: are the ablation of `pbt` / `pbt2` (2026-09-19); every earlier PBT run on
#: disk is mode `full`, whatever `source/algorithms/rl/pbt.py`'s
#: DEFAULT_PBT_MODE says -- the runner's own default is `full`.
PBT_WEIGHTS_SUFFIX = '_weights'
PBT_HP_SUFFIX = '_hp'            # mode hp_only: explore without exploit (2026-09-25)


def pbt_arm(name):
    """``(runner method, pbt_pop_size, pbt_mode)`` for an ARM name.

    `pbt` is the N = 8 population and `pbt<N>` the same method at N members
    (`pbt2`, 2026-09-13): two compute-matched population sizes, one method,
    one run directory per size so the two conditions are never averaged
    together. A `_weights` suffix (PBT_WEIGHTS_SUFFIX) is the same population
    in mode `weights_only`, a `_hp` suffix (PBT_HP_SUFFIX) the same population
    in mode `hp_only`; without either the mode is `full`. Any other arm is
    its own method and has no population (None, None).
    """
    mode = 'full'
    if name.endswith(PBT_WEIGHTS_SUFFIX):
        name, mode = name[:-len(PBT_WEIGHTS_SUFFIX)], 'weights_only'
    elif name.endswith(PBT_HP_SUFFIX):
        name, mode = name[:-len(PBT_HP_SUFFIX)], 'hp_only'
    if name == 'pbt':
        return 'pbt', 8, mode
    if name.startswith('pbt') and name[3:].isdigit():
        return 'pbt', int(name[3:]), mode
    return name, None, None


def pbt_kwargs(name):
    """The `run_ppo` keyword arguments an ARM name stands for: `method`, plus
    `pbt_pop_size` and `pbt_mode` when the arm is a PBT population. The suite
    CLIs splat this, so an arm name is decoded in one place."""
    method, pop_size, mode = pbt_arm(name)
    if pop_size is None:
        return {'method': method}
    return {'method': method, 'pbt_pop_size': pop_size, 'pbt_mode': mode}
# `pbt` is a POPULATION of the plain PPO learner over PPO's own budget --
# see the PBT block in `run_ppo`; the exploit/explore rule is
# `source/algorithms/rl/pbt.py`.


# Budget and hyperparameters carried over from the earlier study's gymnax PPO
# trainer, including the step matching in the comment above. steps_per_update =
# num_envs * num_steps = 2048 * 50 = 102,400, so 5000 updates is 512M steps and
# a 500-update phase is 51.2M -- the same as 200 NES generations.
#
# The C-CHAIN controller's settings are per environment because the two action
# spaces put the churn term on scales five orders of magnitude apart: the
# categorical reference's 10000 / warmup 10 / window 50 / start 1 / floor 1,
# and the continuous reference's 0.05 / 50 / 100 / start 100 / no floor. See
# `source/studies/brax/my_brax/cchain.py` for the second set.
_GYMNAX_CHAIN = dict(chain_target_rel_scale=10000.0, chain_warmup_updates=10,
                     chain_coef_window=50, chain_initial_coef=1.0,
                     chain_floor=1.0)
_MJX_CHAIN = dict(chain_target_rel_scale=0.05, chain_warmup_updates=50,
                  chain_coef_window=100, chain_initial_coef=100.0,
                  chain_floor=0.0)

_GYMNAX_PPO = {
    "num_updates": 5000,
    "task_interval": 500,
    "num_envs": 2048,
    "num_steps": 50,
    "num_epochs": 10,
    "num_minibatches": 32,
    "gae_lambda": 0.95,
    "ent_coef": 1e-2,
    "clip_eps": 0.2,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
    "value_hidden_dims": (128, 128, 128),
    "normalize_obs": False,
    "reward_scale": 1.0,
    "eval_interval": 1,
    **_GYMNAX_CHAIN,
}

# The mjx bodies: this repo's own ant PPO hyperparameters
# (`runs_repro2/brax/continual/ppo/ant/trial_1/config.json` -- lr 3e-4,
# entropy 0.01, discount 0.97, clip 0.3, reward scaling 10, observation
# normalisation, 512 envs), on the budget that matches the NE arms' 16
# generations x 512 x 3 x 1000 a phase: 2400 updates x 512 envs x 20 steps =
# 24,576,000 steps a phase, twenty phases. The paper's brax loop takes 5-step
# unrolls and 320 gradient steps per 2,560 transitions; a 20-step window with
# 32 minibatches x 10 epochs is one gradient step per 32 transitions, the
# Python-loop shape this trainer has always had. The critic is the shared
# relu ValueNetwork at the paper's width. `eval_interval` 150 gives a phase
# the 16 records an NE phase has generations.
_MJX_PPO = {
    "num_updates": 48000,
    "task_interval": 2400,
    "num_envs": 512,
    "num_steps": 20,
    "num_epochs": 10,
    "num_minibatches": 32,
    "gamma": 0.97,
    "gae_lambda": 0.95,
    "learning_rate": 3e-4,
    "ent_coef": 1e-2,
    "clip_eps": 0.3,
    "vf_coef": 0.5,
    "max_grad_norm": 1.0,
    "value_hidden_dims": (128, 128),
    "normalize_obs": True,
    "reward_scale": 10.0,
    "eval_interval": 150,
    "log_std_init": -0.5,
    **_MJX_CHAIN,
}

# PPO on a gridded, binary-plane observation behind the conv policy
# (`GridConvPolicy`). The update's shape is the gymnax one (2048 x 50) rather
# than the 64 x 128 of PureJaxRL / gymnax-blines, because at this budget the
# small shape is 750,000 updates a run; the per-sample settings are those
# references' -- lr 5e-4, 4 epochs, gamma 0.99, GAE 0.95, clip 0.2, entropy
# 0.01, value 0.5, grad norm 0.5 -- with 16 minibatches of 6400 (theirs: 8 of
# 1024). Observations are binary planes, so no normalisation. A held-out
# evaluation is 16 episodes of up to 1000 sequential steps, about an update's
# worth of time, so a record every 10 updates. The C-CHAIN controller is the
# categorical reference's, as on gymnax.
#
# These values were set for a MinAtar suite that was dropped on 2026-09-08;
# MiniGrid inherits them below and is the only body that uses them.
_CONV_PPO = {
    "num_updates": 60000,
    "task_interval": 3000,
    "num_envs": 2048,
    "num_steps": 50,
    "num_epochs": 4,
    "num_minibatches": 16,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "learning_rate": 5e-4,
    "ent_coef": 1e-2,
    "clip_eps": 0.2,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
    "value_hidden_dims": (128, 128, 128),
    "normalize_obs": False,
    "reward_scale": 1.0,
    "eval_interval": 10,
    **_GYMNAX_CHAIN,
}

# MiniGrid, since 2026-09-07 (`source/envs/minigrid.py`). The per-sample
# settings and batch shape above, on the same CNN; the budget matches the NE arms'
# 4000 generations x 512 x 3 x 1024 (the scan length) = 6.29e9 steps:
# 61,440 updates x 2048 envs x 50 steps, 3072 updates a phase against 200
# generations. The observation is binary one-hot planes, so no
# normalisation. Not tuned on this body: the xminigrid baselines' PPO (lr
# 1e-3, 1 epoch, 8192 x 16 with a recurrent actor) is a different shape,
# and this is a placeholder until a run on the body says otherwise.
_MINIGRID_PPO = {
    **_CONV_PPO,
    "num_updates": 61440,
    "task_interval": 3072,
}

# Kinetix, since 2026-09-09 (`source/envs/kinetix.py`). Everything PPO learns
# FROM is the vendored `third_party/kinetix/kinetix_config_pixels.yaml`, i.e.
# what Kinetix itself tuned on this body: lr 5e-5, gamma 0.995, gae_lambda 0.9,
# 8 epochs, 32 minibatches, clip 0.2, ent_coef 0.01, vf_coef 0.5, grad norm
# 1.0, and no observation normalisation (pixels are already in [0, 1]).
#
# The BATCH SHAPE is not theirs and is not a hyperparameter: 128 environments x
# 64 steps rather than their 16 x 1000, because rendering 125x125x3 frames is
# what costs on this body and 16 parallel environments leave the card idle.
# 1600 updates x 128 x 64 = 1.311e7 steps a level, which is the NE arms'
# 200 generations x 512 x 1 x 128 exactly -- `settings.check()` asserts it,
# against a 24x gap in the previous codebase's kinetix table. (9600 until
# 2026-09-13, matched to a nominal NE budget of 3 rollouts x 256 steps that the
# deterministic level never used; see settings.py.)
#
# `value_hidden_dims` is the same MLP every other body's critic is, over the
# flat 46,876-value observation, where Kinetix's own PPO shares a convolutional
# trunk between actor and critic. That difference was the suspect when PPO here
# was training on NaN; it was not the cause, and with the cause fixed (the
# multi-discrete head's -inf padding, see actors.py) this critic solves
# h0_unicycle from scratch in 819k environment steps -- half of what the old
# trainer's shared-trunk runs took. It is left as it is on that evidence.
#
# The C-CHAIN controller is the categorical reference's, as on gymnax: this
# body's head is multi-discrete, whose churn is that same cross-entropy summed
# over six independent categoricals, so it sits on the gymnax scale.
_KINETIX_PPO = {
    "num_updates": 1600,
    "task_interval": 160,
    "num_envs": 128,
    "num_steps": 64,
    "num_epochs": 8,
    "num_minibatches": 32,
    "gamma": 0.995,
    "gae_lambda": 0.9,
    "learning_rate": 5e-5,
    "ent_coef": 1e-2,
    "clip_eps": 0.2,
    "vf_coef": 0.5,
    "max_grad_norm": 1.0,
    "value_hidden_dims": (128,) * 5,
    "normalize_obs": False,
    "reward_scale": 1.0,
    "eval_interval": 50,
    **_GYMNAX_CHAIN,
}


PPO_CONFIGS = {
    "CartPole-v1": {**_GYMNAX_PPO, "gamma": 0.95, "learning_rate": 3e-4},
    "Acrobot-v1": {**_GYMNAX_PPO, "gamma": 0.99, "learning_rate": 1e-4},
    "MountainCar-v0": {**_GYMNAX_PPO, "gamma": 0.99, "learning_rate": 1e-4},
    # The same settings for both bodies. Only the ant has been run (section
    # G); CheetahRun's entry is here so the RL arms CAN be queued on it, and
    # says nothing about whether these values suit it.
    "ant": dict(_MJX_PPO),
    "CheetahRun": dict(_MJX_PPO),
    "MiniGrid": dict(_MINIGRID_PPO),
    "MiniGrid-L1024": dict(_MINIGRID_PPO),
}

# One entry per Kinetix cell -- the twenty stationary levels and the continual
# chain -- all the same settings, because a cell differs from another only in
# WHICH levels the schedule can reach. The per-cell `num_updates` and
# `task_interval` a continual run needs are passed by
# `source/studies/kinetix/cli.py`, which is where the phase grid lives; these
# are the stationary defaults.
from source.envs.kinetix_levels import CELLS as _KINETIX_CELLS   # noqa: E402

PPO_CONFIGS.update({name: dict(_KINETIX_PPO) for name in _KINETIX_CELLS})

# DeepSea<N>-bsuite (gymnax_classic.DeepSeaEnv, the action-map family): the
# gymnax RL trainer's entry, CartPole's settings at gamma 0.99
# (source/studies/gymnax/train_RL_gymnax_continual.py), so the PBT arm's
# members are the PPO already reported on it (probe_deepsea, 2026-09-19). The
# episode is N steps; the launcher passes the compute-matched budget.
PPO_CONFIGS.update({f"DeepSea{n}-bsuite": {**_GYMNAX_PPO, "gamma": 0.99,
                                           "learning_rate": 3e-4}
                    for n in (8, 10, 12, 14, 16, 20)})


def make_rollout_fn(env_step, head, actor, value_net, hp, offset_fn):
    """Return ``rollout(policy_params, value_params, carry, task, stats)``.

    ``carry`` is ``(obs, state, key)`` and is threaded across updates, so the
    environments are never reset at an update boundary -- a rollout is a window
    over episodes that keep running, which is what the GAE bootstrap assumes.

    The sub-task enters in two places and only two: ``offset_fn(task)`` is
    added to the observation before the networks read it (the sub-task vector
    on gymnax and under obs_noise, 0 for a friction sub-task), and ``env_step``
    runs the environment under the sub-task's physics (a no-op on gymnax). The
    returns remain comparable with NES's.

    ``stats`` are the observation normaliser's running statistics, or None.
    Inputs are normalised with the statistics as they stood when the policy
    ACTED, and the statistics are updated with this window's observations for
    the next one, so the stored log-probabilities and the update's forward
    pass see the same inputs.
    """
    num_steps = hp['num_steps']
    reward_scale = float(hp.get('reward_scale', 1.0))

    def rollout(policy_params, value_params, carry, task, stats):
        offset = offset_fn(task)

        def inputs(obs):
            shifted = obs + offset
            return shifted, (actors.normalize(shifted, stats)
                             if stats is not None else shifted)

        def env_step_fn(carry, _):
            obs, state, key = carry
            shifted, inp = inputs(obs)
            logits = actor.apply(policy_params, inp)
            values = value_net.apply(value_params, inp)

            key, action_key, step_key = random.split(key, 3)
            actions = jax.vmap(head.sample, in_axes=(0, 0))(
                random.split(action_key, obs.shape[0]), logits)
            log_probs = jax.vmap(head.log_prob)(logits, actions)

            next_obs, next_state, reward, done = env_step(
                step_key, state, actions, task)
            if reward_scale != 1.0:
                reward = reward * reward_scale

            transition = {'obs': inp, 'actions': actions,
                          'log_probs': log_probs, 'values': values,
                          'rewards': reward, 'dones': done.astype(jnp.float32)}
            if stats is not None:
                # Only when there is a normaliser to update. An extra scan
                # output changes XLA's fusion of the rollout and with it the
                # float32 rounding, and a gymnax run has to stay bit-identical
                # to the runs on disk.
                transition['raw_obs'] = shifted
            return (next_obs, next_state, key), transition

        (obs, state, key), traj = jax.lax.scan(
            env_step_fn, carry, None, length=num_steps)

        # V(s_T) for the bootstrap, on the state the window stopped on.
        last_value = value_net.apply(value_params, inputs(obs)[1])

        # Explicit wrapper rather than functools.partial: vmap passes its
        # arguments positionally, and last_value is the sixth parameter of
        # gae_advantages, so a partial over the keyword arguments would receive
        # it as `gamma`.
        def gae(rewards, values, dones, bootstrap):
            return ppo_lib.gae_advantages(
                rewards, values, dones, gamma=hp['gamma'],
                gae_lambda=hp['gae_lambda'], last_value=bootstrap)

        advantages = jax.vmap(gae, in_axes=(1, 1, 1, 0), out_axes=1)(
            traj['rewards'], traj['values'], traj['dones'], last_value)
        returns = advantages + traj['values']

        flat = lambda x: x.reshape((-1,) + x.shape[2:])
        batch = {'obs': flat(traj['obs']), 'actions': flat(traj['actions']),
                 'log_probs': flat(traj['log_probs']),
                 'advantages': flat(advantages), 'returns': flat(returns)}
        if stats is not None:
            stats = actors.update_norm_stats(stats, traj['raw_obs'])
        return (obs, state, key), batch, stats

    return rollout


def make_update_fn(actor, value_net, hp, head, joint_tx=None):
    """Return ``update(policy_state, value_state, batch, key, opt_state=None)``.

    ``num_epochs`` passes over the batch, reshuffled each epoch and split into
    ``num_minibatches``. Advantages are normalised per minibatch, the usual PPO
    convention.

    With ``joint_tx`` the two parameter sets are updated by ONE optax
    transformation over both, threading ``opt_state`` through. That is only
    needed for TRAC, whose tuner is a handful of scalars over whatever pytree it
    is handed -- give it the policy alone and it is tuning half the model. Every
    other optimiser here is per-parameter, so the two paths are arithmetically
    identical for them. See `train_step_joint` in
    `source/algorithms/rl/ppo.py`.
    """
    num_minibatches = hp['num_minibatches']
    if joint_tx is not None:
        step_joint = partial(ppo_lib.train_step_joint, actor, value_net,
                             hp['clip_eps'], hp['vf_coef'], joint_tx)
    step = partial(ppo_lib.train_step, actor, value_net, hp['clip_eps'],
                   hp['vf_coef'])
    dist = dict(log_prob_fn=head.log_prob, entropy_fn=head.entropy)

    def update(policy_state, value_state, batch, key, opt_state=None,
               ent_coef=None):
        # A per-member entropy coefficient under PBT, the config's
        # otherwise. None keeps the Python constant every other method
        # compiles in, so their traces are unchanged.
        ec = hp['ent_coef'] if ent_coef is None else ent_coef
        batch_size = batch['obs'].shape[0]
        minibatch_size = batch_size // num_minibatches

        def epoch(carry, epoch_key):
            policy_state, value_state, opt_state = carry
            perm = random.permutation(epoch_key, batch_size)
            shuffled = jax.tree.map(lambda x: x[perm], batch)
            minibatches = jax.tree.map(
                lambda x: x.reshape((num_minibatches, minibatch_size)
                                    + x.shape[1:]), shuffled)

            def minibatch(carry, mb):
                policy_state, value_state, opt_state = carry
                mb = dict(mb)
                mb['advantages'] = ((mb['advantages'] - mb['advantages'].mean())
                                    / (mb['advantages'].std() + 1e-8))
                if joint_tx is not None:
                    (policy_state, value_state, opt_state, loss,
                     metrics) = step_joint(policy_state, value_state,
                                           opt_state, mb, ec,
                                           **dist)
                else:
                    policy_state, value_state, loss, metrics = step(
                        policy_state, value_state, mb, ec, **dist)
                return (policy_state, value_state, opt_state), (loss, metrics)

            carry, out = jax.lax.scan(
                minibatch, (policy_state, value_state, opt_state), minibatches)
            return carry, out

        carry, (losses, metrics) = jax.lax.scan(
            epoch, (policy_state, value_state, opt_state),
            random.split(key, hp['num_epochs']))
        policy_state, value_state, opt_state = carry
        return policy_state, value_state, opt_state, {
            'loss': jnp.mean(losses),
            'entropy': jnp.mean(metrics['entropy']),
            'approx_kl': jnp.mean(metrics['approx_kl']),
        }

    return update


def run_ppo(env_name='CartPole-v1', method='ppo', schedule='switch',
            num_updates=None,
            task_interval=None, task_warmup=0, noise_range=None, num_tasks=2,
            first_task_clean=True, pool_size=None, pair_repeats=5,
            eval_episodes=16, seed=42, trial=1,
            output_dir=None, log_interval=100, checkpoint_interval=5,
            overrides=None, redo_interval=1000, redo_tau=0.025,
            chain_target_rel_scale=None, chain_warmup_updates=None,
            chain_coef_window=None, cchain_reset_on_switch=False,
            churn_interval=10, num_probe_states=512, eval_interval=None,
            task_options=None,
            resume_path=None, checkpoint_every=0, max_updates_this_run=0,
            episode_length=None,
            pbt_pop_size=8, pbt_interval=10, pbt_exploit_fraction=None,
            pbt_perturb_factor=None, pbt_mode='full'):
    if schedule not in SCHEDULES:
        raise ValueError(f"schedule must be one of {SCHEDULES}, got {schedule!r}")
    if method not in RL_METHODS:
        raise ValueError(f'method must be one of {RL_METHODS}, got {method!r}')
    # Which task family this environment belongs to. `suites.py` is the whole
    # of the difference between gymnax and mjx as far as this file is
    # concerned, and `actors.py` the whole of the difference between a
    # categorical and a Gaussian policy.
    suite_name = suite_for(env_name)
    suite = get_suite(suite_name)
    cfg = suite.env_configs[env_name]
    # None means "this environment's own default", the same convention run_nes
    # follows, so `train_all.py` can pass one value through to either trainer.
    if noise_range is None:
        noise_range = cfg.get('noise_range', 1.0)
    hp = dict(PPO_CONFIGS[env_name])
    hp.update(overrides or {})
    # The C-CHAIN controller's settings, and how often to evaluate: the
    # caller's if given, else the environment's own (see PPO_CONFIGS).
    for key_, value in (('chain_target_rel_scale', chain_target_rel_scale),
                        ('chain_warmup_updates', chain_warmup_updates),
                        ('chain_coef_window', chain_coef_window),
                        ('eval_interval', eval_interval)):
        if value is not None:
            hp[key_] = value
    eval_interval = int(hp.get('eval_interval', 1) or 1)
    num_updates = num_updates or hp['num_updates']
    task_interval = task_interval or hp['task_interval']
    # An explicit episode_length overrides the suite table, so a run can be
    # made at a shorter scan without editing the table. None keeps the
    # recorded value, so unflagged runs are unchanged.
    episode_length = episode_length or cfg['episode_length']
    normalize_obs = bool(hp.get('normalize_obs', False))

    env, env_params, obs_dim, action_dim = suite.make_env(
        env_name, episode_length, task_options)
    # `action_dims` is None on every suite whose action is one categorical or
    # one continuous vector, and (3, 3, 3, 3, 2, 2) on kinetix, where the
    # multi-discrete head is built from it. See actors.head_for.
    head = actors.head_for(suite_name, suite.action_dims(env))

    key = random.key(seed)
    key, policy_key, value_key = random.split(key, 3)
    arch = dict(cfg.get('arch', {}))       # see train_nes
    policy, param_template, num_params = suite.build_policy(
        policy_key, obs_dim, action_dim, cfg['hidden_dims'], **arch)
    # The network PPO trains: the evolved policy itself on gymnax, and on the
    # mjx bodies the same network as the mean of a Gaussian.
    actor = actors.build_actor(suite_name, obs_dim, action_dim,
                               cfg['hidden_dims'], hp.get('log_std_init', -0.5),
                               policy)
    value_net = ValueNetwork(hidden_dims=tuple(hp['value_hidden_dims']))
    value_params = value_net.init(value_key, jnp.zeros((obs_dim,)))

    # `adam_b1` is Adam's momentum coefficient, optax's 0.9 unless a run sets
    # it. The generalists study's section A found retention under switching is
    # decided by the optimizer's MOMENTUM TAIL and not by its per-coordinate
    # normalisation (the `beta1 = 0` arms there); `--adam_b1 0` is the same
    # ablation on the RL arm, so the question "is RL's retention failure the
    # optimizer too?" is one flag rather than a second trainer. `.get` so a
    # config written before the key existed reads as 0.9, which is what it ran.
    adam_b1 = float(hp.get('adam_b1', 0.9))
    tx = lambda: optax.chain(optax.clip_by_global_norm(hp['max_grad_norm']),
                             optax.adam(hp['learning_rate'], b1=adam_b1))
    # TRAC takes no hyperparameters -- that is the point of it -- and wraps the
    # base optimiser over BOTH parameter sets at once. `start_trac` is the same
    # wrapper the paper's gymnax, brax, mujoco and kinetix RL trainers use, so
    # an arm here and an arm there are the same method.
    joint_tx = None
    if method == 'trac':
        from trac_optimizer.experimental.jax.trac import start_trac
        joint_tx = start_trac(tx())
    policy_state = TrainState.create(apply_fn=actor.apply,
                                     params=actor.init(policy_key,
                                                       jnp.zeros((obs_dim,))),
                                     tx=tx())
    value_state = TrainState.create(apply_fn=value_net.apply,
                                    params=value_params, tx=tx())
    # The same pytree shape `train_step_joint` rebuilds each step -- a dict
    # keyed 'policy'/'value', not a tuple -- or TRAC's init and its update
    # disagree about what they are tuning.
    joint_opt_state = (
        joint_tx.init({'policy': policy_state.params,
                       'value': value_state.params})
        if joint_tx is not None else None)

    # One row per sub-task: an observation offset, or on the ant a friction
    # multiplier. Same draw, same name in the artifacts, as train_nes.
    noise_vectors = suite.task_vectors(env_params, trial, num_tasks, obs_dim,
                                       noise_range, first_task_clean)
    offset_fn = lambda task: suite.obs_offset(env_params, task)

    # The ruler, identical to NES's: deterministic policy, fixed episode
    # count. Not PPO's own stochastic-policy return.
    eval_score = suite.make_scoring_fn(env, env_params, policy, param_template,
                                       episode_length, eval_episodes)

    env_reset, env_step = suite.rl_env_fns(env, env_params, hp['num_envs'])
    # Jitted: run eagerly, a 512-environment MJX reset is hundreds of separate
    # op compiles at startup, and the carry it produces then differs in weak
    # typing from one produced inside `step_one`, which recompiled that step
    # once more per sub-task (measured: 6 compiles of step_one in a 30-update
    # run, 91 s, where 2 are needed).
    env_reset = jax.jit(env_reset)
    rollout = make_rollout_fn(env_step, head, actor, value_net, hp, offset_fn)
    update = make_update_fn(actor, value_net, hp, head, joint_tx)

    # C-CHAIN replaces the SGD-epoch loop rather than wrapping it: the policy
    # is pulled towards the policy from one gradient step ago, evaluated on an
    # independently drawn minibatch of the same rollout. Imported from the
    # paper's port of the reference implementation so the two studies run one
    # C-CHAIN, not two. The head supplies the churn it regularises -- the
    # cross-entropy on gymnax, the action-mean MSE on a continuous body -- and
    # the controller's scale, start and floor come from the env's config.
    chain_epochs = chain_state = chain_ctrl = None
    if method == 'cchain':
        from source.studies.gymnax.cchain import (ChainCoefController,
                                          init_chain_state,
                                          make_chain_sgd_epochs)
        chain_epochs = make_chain_sgd_epochs(
            actor, value_net, ppo_lib.compute_ppo_loss, hp['clip_eps'],
            hp['vf_coef'], hp['num_epochs'], hp['num_minibatches'],
            hp['num_envs'] * hp['num_steps'],
            churn_fn=(None if head is actors.CATEGORICAL
                      else head.chain_churn),
            log_prob_fn=(None if head is actors.CATEGORICAL
                         else head.log_prob),
            entropy_fn=(None if head is actors.CATEGORICAL
                        else head.entropy),
            # As make_update_fn does for every other arm here.
            normalize_minibatch_advantages=True)
        chain_state = init_chain_state(policy_state, value_state)
        chain_ctrl = ChainCoefController(hp['chain_target_rel_scale'],
                                         hp['chain_warmup_updates'],
                                         hp['chain_coef_window'],
                                         initial_coef=hp['chain_initial_coef'],
                                         floor=hp['chain_floor'])

    # ReDo needs an `args`-shaped object: the shared pass reads its knobs off
    # one, and duplicating its signature here would be a second place for the
    # defaults to drift.
    redo_args = SimpleNamespace(redo_targets='both', redo_tau=redo_tau,
                                redo_batch_size=hp['num_envs'],
                                redo_keep_adam_count=False)
    # `run_redo_pass` reads the layer counts off `hp`, under names this study's
    # PPO_CONFIGS does not use: the policy's widths live in ENV_CONFIGS (the NE
    # arms share that policy) and only the value net's are in `hp`. Supplying
    # the alias here keeps the shared pass the shared pass.
    hp = dict(hp, policy_hidden_dims=cfg['hidden_dims'])
    # Which activation the dormancy criterion is for. relu on gymnax, tanh on
    # the mjx bodies; the shared pass and the live diagnostics both read it.
    policy_activation = head.policy_activation
    redo_activations = (dict(policy_activation=policy_activation)
                        if head is not actors.CATEGORICAL else {})
    policy_activation_fn = ACTIVATIONS[policy_activation]
    policy_criterion = criterion_for_activation(policy_activation)
    # A policy that is not a Dense chain (the conv policy) reports its own
    # hidden layers; the dormancy probe and ReDo take them from it rather than
    # replaying a Dense walk that cannot see a convolution. None on the MLP
    # policies, where both diagnostics are bit-unchanged.
    policy_layers = getattr(policy, 'LAYERS', None)
    policy_activations_fn = getattr(policy, 'hidden_activations', None)
    # Likewise a policy whose first Dense also reads inputs that are not
    # hidden units (the Kinetix pixel policy's global-info scalar) says so;
    # None on every other network.
    _extra = getattr(policy, 'extra_fan_in', None)
    policy_extra_fan_in = _extra() if _extra is not None else None

    # What is stored and what is scored: the actor's parameters as a point in
    # the NE space -- the log-std dropped, the observation normaliser folded
    # into the first layer. On gymnax both are the identity and the
    # arithmetic is exactly `get_flat_params(policy_state.params)`.
    def ne_params(params, stats):
        params = head.mean_params(params)
        if stats is not None:
            params = actors.fold_normalizer(params, stats)
        return params

    @jax.jit
    def step_one(policy_state, value_state, opt_state, carry, task, key, stats,
                 ent_coef=None):
        roll_key, update_key = random.split(key)
        carry, batch, stats = rollout(policy_state.params, value_state.params,
                                      carry, task, stats)
        policy_state, value_state, opt_state, metrics = update(
            policy_state, value_state, batch, update_key, opt_state, ent_coef)
        return (policy_state, value_state, opt_state, carry, metrics,
                batch['returns'].mean(), batch, stats)

    @jax.jit
    def step_joint(policy_state, value_state, opt_state, carries, all_tasks,
                   key, stats):
        """One update on the concatenation of a rollout from every sub-task."""
        roll_key, update_key = random.split(key)
        new_carries, batches = [], []
        for t in range(all_tasks.shape[0]):
            carry, batch, stats = rollout(policy_state.params,
                                          value_state.params,
                                          carries[t], all_tasks[t], stats)
            new_carries.append(carry)
            batches.append(batch)
        merged = jax.tree.map(lambda *xs: jnp.concatenate(xs, axis=0), *batches)
        policy_state, value_state, opt_state, metrics = update(
            policy_state, value_state, merged, update_key, opt_state)
        return (policy_state, value_state, opt_state, tuple(new_carries),
                metrics, merged['returns'].mean(), merged, stats)

    @jax.jit
    def rollout_only(policy_params, value_params, carry, task, key, stats):
        """C-CHAIN takes its rollout here and owns the epoch loop itself."""
        return rollout(policy_params, value_params, carry, task, stats)

    @jax.jit
    def eval_policy(key, params, stats, all_tasks):
        flat = get_flat_params(ne_params(params, stats))
        keys = random.split(key, all_tasks.shape[0])
        return flat, jax.vmap(lambda k, nv: eval_score(flat[None, :], k, nv)[0])(
            keys, all_tasks)

    @jax.jit
    def eval_flat(key, flat, all_tasks):
        """The same ruler on a point given as a flat NE vector (the PBT
        population's weight mean)."""
        keys = random.split(key, all_tasks.shape[0])
        return jax.vmap(lambda k, nv: eval_score(flat[None, :], k, nv)[0])(
            keys, all_tasks)

    # PBT's behavioural diversity: the members' outputs on the frozen probe
    # states churn is measured on, paired up by the head's own action
    # distance (`source/metrics/population_diversity.py`). A Gaussian head
    # emits [mean, log_std]; the mean half is the action compared.
    behaviour_distance = make_pairwise_behaviour_fn(head.name,
                                                    suite.action_dims(env))
    member_outputs = jax.jit(lambda params_list, probe: jnp.stack(
        [actor.apply(p, probe)[..., :action_dim] if head.name == 'gaussian'
         else actor.apply(p, probe) for p in params_list]))

    key, *reset_keys = random.split(key, num_tasks + 1)
    carries = tuple(
        (lambda o, s: (o, s, reset_keys[t]))(
            *env_reset(reset_keys[t], noise_vectors[t]))
        for t in range(num_tasks))
    norm_stats = actors.init_norm_stats(obs_dim) if normalize_obs else None

    # ---- PBT: a population of PPO learners over PPO's OWN budget ----------
    #
    # `pbt_pop_size` members, each a complete PPO learner: its own actor,
    # critic, Adam state, environments, normaliser, learning rate and
    # entropy coefficient. The loop below is the same loop: iteration
    # `step` advances member `step % N` by one update, so the run spends
    # exactly PPO's `num_updates x num_envs x num_steps` environment steps,
    # meets every phase boundary at the same step, and the budget needs no
    # rounding for any N (CLAUDE.md (c)). Every `pbt_interval` updates PER
    # MEMBER the bottom `exploit_fraction` copy a top member's weights and
    # take its hyperparameters perturbed (Jaderberg et al. 2017) -- on a
    # fixed update clock, never at a sub-task boundary (CLAUDE.md (d)). The
    # rule (who copies whom, the perturbation and its bounds, the optimizer
    # reset) is `source/algorithms/rl/pbt.py`.
    #
    # Reported like the NE arms: `incumbent_*` is the ELITE -- the member
    # with the best training return, the selection's own number, as the
    # GA's incumbent is archive[0] -- re-scored on fresh keys; `centroid_*`
    # is the score of the members' WEIGHT MEAN on the same keys. For a
    # handful of independently trained networks that is a collapse
    # diagnostic, as it is for the GA and DNS, not a performance number.
    pbt = None
    if method == 'pbt':
        from source.algorithms.rl import pbt as pbt_lib
        N = int(pbt_pop_size)
        if N < 2:
            raise ValueError(f'pbt needs a population of at least 2, got {N}')
        exploit_fraction = (pbt_lib.DEFAULT_EXPLOIT_FRACTION
                            if pbt_exploit_fraction is None
                            else float(pbt_exploit_fraction))
        perturb_factor = (pbt_lib.DEFAULT_PERTURB_FACTOR
                          if pbt_perturb_factor is None
                          else float(pbt_perturb_factor))
        copy_weights, perturb_hp = pbt_lib.mode_flags(pbt_mode)
        bounds = pbt_lib.HYPERPARAM_BOUNDS
        # The learning rate lives in the optimizer STATE
        # (`inject_hyperparams`), so a member's is set without rebuilding
        # its optimizer, and every member shares this ONE optax object: a
        # distinct object per member would be a distinct TrainState treedef
        # and `step_one` would recompile per member and per perturbation.
        pbt_tx = optax.chain(
            optax.clip_by_global_norm(hp['max_grad_norm']),
            optax.inject_hyperparams(optax.adam)(
                learning_rate=float(hp['learning_rate']), b1=adam_b1))
        # The initial spread, the old gymnax trainer's convention: lr and
        # ent_coef at x U(0.5, 1.5) of the config's, weights from each
        # member's own init key. One split off the run's key stream.
        key, pop_key = random.split(key)
        member_keys = random.split(pop_key, N)
        spread = np.random.default_rng(seed * 31 + N)
        members = []
        for i in range(N):
            p_key, v_key, r_key = random.split(member_keys[i], 3)
            lr = float(np.clip(hp['learning_rate'] * spread.uniform(0.5, 1.5),
                               *bounds['learning_rate']))
            ec = float(np.clip(hp['ent_coef'] * spread.uniform(0.5, 1.5),
                               *bounds['ent_coef']))
            m_reset = random.split(r_key, num_tasks)
            members.append(dict(
                policy=_with_lr(TrainState.create(
                    apply_fn=actor.apply,
                    params=actor.init(p_key, jnp.zeros((obs_dim,))),
                    tx=pbt_tx), lr),
                value=_with_lr(TrainState.create(
                    apply_fn=value_net.apply,
                    params=value_net.init(v_key, jnp.zeros((obs_dim,))),
                    tx=pbt_tx), lr),
                lr=lr, ent_coef=ec,
                carries=tuple(
                    (lambda o, s: (o, s, m_reset[t]))(
                        *env_reset(m_reset[t], noise_vectors[t]))
                    for t in range(num_tasks)),
                stats=(actors.init_norm_stats(obs_dim) if normalize_obs
                       else None),
                last_return=float('nan'), window=[]))
        pbt = dict(N=N, members=members, exploits=0)

        def pbt_exploit_explore(key):
            """One PBT step over the whole population; how many copied."""
            means = [float(np.mean(m['window'])) if m['window'] else -np.inf
                     for m in members]
            order = np.argsort(means)                       # worst first
            moves = pbt_lib.pbt_moves(order, key, exploit_fraction)
            for loser, winner in moves:
                w, l = members[winner], members[loser]
                hyper = {'learning_rate': w['lr'], 'ent_coef': w['ent_coef']}
                if perturb_hp:
                    key, h_key = random.split(key)
                    hyper = pbt_lib.perturb_hyperparams(hyper, h_key,
                                                        perturb_factor)
                lr = hyper['learning_rate']
                if copy_weights:
                    # The winner's weights AND its normaliser (the first
                    # layer was trained against those statistics), a fresh
                    # optimizer: `OPTIMIZER_ON_EXPLOIT = 'reset'`.
                    l['policy'] = _with_lr(TrainState.create(
                        apply_fn=actor.apply, params=w['policy'].params,
                        tx=pbt_tx), lr)
                    l['value'] = _with_lr(TrainState.create(
                        apply_fn=value_net.apply, params=w['value'].params,
                        tx=pbt_tx), lr)
                    l['stats'] = w['stats']
                else:
                    l['policy'] = _with_lr(l['policy'], lr)
                    l['value'] = _with_lr(l['value'], lr)
                l['lr'], l['ent_coef'] = lr, hyper['ent_coef']
            for m in members:
                m['window'] = []
            return len(moves)

    # The identical schedule generator NES uses, so a PPO run and an NES run at
    # the same trial face the same sub-task sequence phase for phase.
    phase_of_step, num_phases = make_phase_grid(num_updates, task_interval,
                                                task_warmup)
    task_sequence = make_task_sequence(schedule, num_phases, num_tasks, trial,
                                       pool_size, pair_repeats)
    # The last update of a phase; see the same helper in run_nes.
    phase_end = lambda s: (s + 1 >= num_updates
                           or phase_of_step[s + 1] != phase_of_step[s])

    # --- per-update plasticity diagnostics --------------------------------
    #
    # A pure observer: its RNG is its own stream, seeded from the run's seed
    # rather than split off the training key, so a run with diagnostics on and
    # the same run with them off are the same run. Nothing computed here is
    # read back by the search.
    probe_key = random.key(seed + 1_000_003)
    probe_obs = None
    num_policy_hidden = len(cfg['hidden_dims'])

    @jax.jit
    def churn_diagnostics(prev_p, cur_p, prev_v, cur_v, probe):
        """Churn across one PPO update, on the frozen probe states.

        Both estimators are the published ones, imported: the cross-method
        observer every method in this repo reports (argmax disagreement, or
        the bounded action change on a continuous body) and the method's own
        (C-CHAIN's cross-entropy H(pi_before, pi_after), or its action-mean
        MSE). Value churn is the reference's too -- it logs `value_churn`
        beside `policy_churn` -- and is the MSE between the critic's
        predictions before and after the update.
        """
        before, after = actor.apply(prev_p, probe), actor.apply(cur_p, probe)
        return {
            'rl_churn': head.churn(before, after),
            head.own_churn_key: head.own_churn(before, after),
            'value_churn': jnp.mean(jnp.square(
                value_net.apply(prev_v, probe) - value_net.apply(cur_v, probe))),
        }

    records, checkpoints, checkpoint_updates = [], [], []
    # The deployed policy at the end of each sub-task PHASE, written to
    # `checkpoints.npz` for the plasticity figures. A gradient method carries
    # ONE network, so there is no finalgen/incumbent/centroid distinction to
    # make and `final` is the only key -- exactly what
    # source/studies/gymnax/train_RL_gymnax_continual.py writes, so an RL row
    # of a plasticity table is the same object in both studies.
    phase_agents, phase_tasks = [], []
    phase_centroids = []            # PBT: the weight mean at each phase end
    start = time.time()

    task_start_step, previous_task = 0, None
    # --- checkpoint-restart across SLURM jobs ------------------------------
    #
    # The contract run_nes has, for the same reason: a 12 h job cap and a
    # Kinetix continual run of 192,000 updates. Every `checkpoint_every`
    # updates the WHOLE training state -- both TrainStates, TRAC's joint
    # optimiser, C-CHAIN's reference networks and controller, the per-sub-task
    # env carries, the observation normaliser, the RNG, the frozen churn probe
    # and every record and checkpoint written so far -- goes to `resume_path`;
    # after `max_updates_this_run` updates the run returns early and the next
    # job continues from the file. Nothing about the run is read from the
    # clock, so a resumed run and an uninterrupted one are the same run.
    start_step, elapsed_before = 0, 0.0
    if resume_path and os.path.exists(resume_path):
        import pickle
        with open(resume_path, 'rb') as f:
            ck = pickle.load(f)
        start_step = int(ck['step'])
        policy_state = policy_state.replace(
            params=_device_tree(ck['policy_params']),
            opt_state=_device_tree(ck['policy_opt_state']),
            step=int(ck['policy_step']))
        value_state = value_state.replace(
            params=_device_tree(ck['value_params']),
            opt_state=_device_tree(ck['value_opt_state']),
            step=int(ck['value_step']))
        joint_opt_state = _device_tree(ck['joint_opt_state'])
        chain_state = _device_tree(ck['chain_state'])
        chain_ctrl = ck['chain_ctrl'] if chain_ctrl is not None else None
        carries = tuple(_device_tree(c) for c in ck['carries'])
        norm_stats = _device_tree(ck['norm_stats'])
        key = _device_tree(ck['key'])
        probe_obs = _device_tree(ck['probe_obs'])
        records, checkpoints = ck['records'], ck['checkpoints']
        checkpoint_updates = ck['checkpoint_updates']
        phase_agents, phase_tasks = ck['phase_agents'], ck['phase_tasks']
        phase_centroids = ck.get('phase_centroids', [])
        if pbt is not None:
            for m, saved in zip(pbt['members'], ck['pbt_members']):
                m['policy'] = m['policy'].replace(
                    params=_device_tree(saved['policy_params']),
                    opt_state=_device_tree(saved['policy_opt_state']),
                    step=int(saved['policy_step']))
                m['value'] = m['value'].replace(
                    params=_device_tree(saved['value_params']),
                    opt_state=_device_tree(saved['value_opt_state']),
                    step=int(saved['value_step']))
                m['carries'] = tuple(_device_tree(c) for c in saved['carries'])
                m['stats'] = _device_tree(saved['stats'])
                for k_ in ('lr', 'ent_coef', 'last_return', 'window'):
                    m[k_] = saved[k_]
            pbt['exploits'] = int(ck['pbt_exploits'])
        task_start_step = ck['task_start_step']
        previous_task = ck['previous_task']
        elapsed_before = float(ck['elapsed_before'])
        print(f'  RESUMED from {resume_path} at update {start_step} '
              f'({len(records)} records, {elapsed_before/3600:.2f} h '
              'already spent)', flush=True)
    updates_this_run = 0
    for step in range(start_step, num_updates):
        if (checkpoint_every and step > start_step
                and step % checkpoint_every == 0):
            from source.studies.generalists.train_nes import _resume_save
            _resume_save(resume_path, dict(
                step=step,
                policy_params=_host_tree(policy_state.params),
                policy_opt_state=_host_tree(policy_state.opt_state),
                policy_step=int(policy_state.step),
                value_params=_host_tree(value_state.params),
                value_opt_state=_host_tree(value_state.opt_state),
                value_step=int(value_state.step),
                joint_opt_state=_host_tree(joint_opt_state),
                chain_state=_host_tree(chain_state),
                chain_ctrl=chain_ctrl,
                carries=[_host_tree(c) for c in carries],
                norm_stats=_host_tree(norm_stats),
                key=_host_tree(key),
                probe_obs=_host_tree(probe_obs),
                records=records, checkpoints=checkpoints,
                checkpoint_updates=checkpoint_updates,
                phase_agents=phase_agents, phase_tasks=phase_tasks,
                phase_centroids=phase_centroids,
                **({'pbt_members': [dict(
                        policy_params=_host_tree(m['policy'].params),
                        policy_opt_state=_host_tree(m['policy'].opt_state),
                        policy_step=int(m['policy'].step),
                        value_params=_host_tree(m['value'].params),
                        value_opt_state=_host_tree(m['value'].opt_state),
                        value_step=int(m['value'].step),
                        carries=[_host_tree(c) for c in m['carries']],
                        stats=_host_tree(m['stats']), lr=m['lr'],
                        ent_coef=m['ent_coef'], last_return=m['last_return'],
                        window=list(m['window'])) for m in pbt['members']],
                    'pbt_exploits': pbt['exploits']}
                   if pbt is not None else {}),
                task_start_step=task_start_step, previous_task=previous_task,
                elapsed_before=elapsed_before + (time.time() - start)))
            print(f'  checkpoint: update {step} -> {resume_path}', flush=True)
            if max_updates_this_run and updates_this_run >= max_updates_this_run:
                print(f'  SEGMENT DONE: {updates_this_run} updates this run, '
                      f'stopping at update {step}; resume.pkl left for the '
                      'next one', flush=True)
                return {'resumed_segment': True, 'stopped_at': step}
        updates_this_run += 1
        task_idx = int(task_sequence[phase_of_step[step]])
        if task_idx != previous_task:
            # OFF BY DEFAULT, which is BOTH the fair comparison and the
            # published algorithm. Tang et al. 2025 say C-CHAIN "does not need
            # to be aware of task switches", Algorithm 1 has no reset, and the
            # appendix's lambda_pi mechanism is a plain running ratio -- but
            # the released code resets the coefficient, its loss history and
            # its warmup gate at every boundary in three of its four suites
            # (crl_gym_classic_control, crl_procgen, crl_dmc).
            # The flag exists to reproduce that implementation, not to be the
            # default: no other method in this comparison is ever told a switch
            # happened. Same flag and default as the paper's trainers; see
            # `--cchain_reset_on_switch` in
            # source/studies/gymnax/train_RL_gymnax_continual.py.
            if (chain_ctrl is not None and previous_task is not None
                    and cchain_reset_on_switch):
                chain_ctrl.reset()
                task_start_step = step
            previous_task = task_idx
        key, step_key, eval_key = random.split(key, 3)

        # Under PBT this iteration is one member's update: its learner state
        # and environments stand in for the run-level ones below and are
        # written back after the step.
        member = None
        if pbt is not None:
            member = pbt['members'][step % pbt['N']]
            policy_state, value_state = member['policy'], member['value']
            carries, norm_stats = member['carries'], member['stats']

        # Captured before the update, so churn is measured across exactly one
        # of them -- the same delta the paper's gymnax RL trainer uses, which
        # is what lets the two columns be read against each other.
        prev_policy_params, prev_value_params = (policy_state.params,
                                                 value_state.params)

        if schedule == 'joint':
            (policy_state, value_state, joint_opt_state, carries, metrics,
             train_return, batch, norm_stats) = step_joint(
                 policy_state, value_state, joint_opt_state, carries,
                 noise_vectors, step_key, norm_stats)
        elif method == 'cchain':
            # C-CHAIN owns the epoch loop, so the rollout is taken here and
            # handed to it with the churn coefficient the controller has
            # settled on.
            roll_key, sgd_key = random.split(step_key)
            carry_list = list(carries)
            carry_list[task_idx], batch, norm_stats = rollout_only(
                policy_state.params, value_state.params, carries[task_idx],
                noise_vectors[task_idx], roll_key, norm_stats)
            carries = tuple(carry_list)
            (policy_state, value_state, chain_state, _loss,
             metrics) = chain_epochs(policy_state, value_state, batch,
                                     hp['ent_coef'], chain_ctrl.coef,
                                     chain_state, sgd_key)
            # The warmup gate is the second boundary signal and is gated with
            # the first: with the reset off, `task_start_step` stays 0, so this
            # is the global update index and the controller is never re-armed.
            chain_ctrl.update(metrics['chain_p_loss'],
                              metrics['chain_p_reg_loss'],
                              step - task_start_step)
            train_return = batch['returns'].mean()
        else:
            carry_list = list(carries)
            (policy_state, value_state, joint_opt_state, carry_list[task_idx],
             metrics, train_return, batch, norm_stats) = step_one(
                 policy_state, value_state, joint_opt_state,
                 carries[task_idx], noise_vectors[task_idx], step_key,
                 norm_stats,
                 None if member is None else jnp.float32(member['ent_coef']))
            carries = tuple(carry_list)

        # ReDo: recycle the units that have gone dormant, every
        # `redo_interval` updates, on observations from the rollout just taken.
        if method == 'redo' and (step + 1) % redo_interval == 0:
            key, redo_key = random.split(key)
            obs = batch['obs'][:redo_args.redo_batch_size]
            policy_state, value_state, _stats = ppo_lib.run_redo_pass(
                policy_state, value_state, obs, redo_key, redo_args, hp,
                policy_layers=policy_layers,
                policy_activations_fn=policy_activations_fn,
                policy_extra_fan_in=policy_extra_fan_in,
                **redo_activations)

        if member is not None:
            member['policy'], member['value'] = policy_state, value_state
            member['carries'], member['stats'] = carries, norm_stats
            member['last_return'] = float(train_return)
            member['window'].append(float(train_return))
            # Every member has had `pbt_interval` updates: exploit/explore.
            # A fixed clock in updates, never a sub-task boundary.
            if (step + 1) % (int(pbt_interval) * pbt['N']) == 0:
                key, pbt_key = random.split(key)
                pbt['exploits'] += pbt_exploit_explore(pbt_key)

        # Frozen on the first rollout and never refreshed: churn has to measure
        # the policy moving, not the observation distribution moving under it,
        # and a probe that follows the current sub-task would confound the two
        # at every switch. Same reasoning, and the same freeze point, as the
        # paper's gymnax RL trainer.
        if probe_obs is None and churn_interval:
            flat_obs = batch['obs'].reshape(-1, batch['obs'].shape[-1])
            idx = random.choice(probe_key, flat_obs.shape[0],
                                (min(num_probe_states, flat_obs.shape[0]),),
                                replace=False)
            probe_obs = flat_obs[idx]

        # A record every `eval_interval` updates, plus the last update of every
        # phase and of the run. On gymnax that is every update.
        last = step == num_updates - 1
        recording = (step % eval_interval == 0 or phase_end(step) or last)
        if not recording:
            continue

        if member is None:
            flat_params, per_task = eval_policy(eval_key, policy_state.params,
                                                norm_stats, noise_vectors)
            per_task = np.asarray(per_task)
            centroid_flat = centroid_per_task = None
            diversity = {}
            train_mean = train_max = float(train_return)
            dormancy_params = policy_state.params
        else:
            # The elite by the selection's own number (best last training
            # return), re-scored on fresh keys; the weight mean on the SAME
            # keys, so the gap between the two curves is the genome alone
            # -- the convention train_nes uses for incumbent vs centroid.
            ms = pbt['members']
            returns = np.asarray([m['last_return'] for m in ms])
            elite = (ms[int(np.nanargmax(returns))]
                     if np.isfinite(returns).any() else ms[0])
            flat_params, per_task = eval_policy(
                eval_key, elite['policy'].params, elite['stats'], noise_vectors)
            per_task = np.asarray(per_task)
            flats = np.stack([np.asarray(get_flat_params(
                ne_params(m['policy'].params, m['stats']))) for m in ms])
            centroid_flat = flats.mean(axis=0)
            centroid_per_task = np.asarray(eval_flat(
                eval_key, jnp.asarray(centroid_flat), noise_vectors))
            outputs = (member_outputs([m['policy'].params for m in ms], probe_obs)
                       if probe_obs is not None else None)
            diversity = diversity_columns(flats, returns, outputs,
                                          behaviour_distance)
            train_mean = float(np.nanmean(returns))
            train_max = float(np.nanmax(returns))
            dormancy_params = elite['policy'].params
        record = {
            'generation': step,
            'task': int(task_idx),
            'train_fitness_mean': train_mean,
            'train_fitness_max': train_max,
            'entropy': float(metrics['entropy']),
            'approx_kl': float(metrics['approx_kl']),
            **diversity,
        }

        if churn_interval and step % churn_interval == 0:
            stats = churn_diagnostics(prev_policy_params, policy_state.params,
                                      prev_value_params, value_state.params,
                                      probe_obs)
            record.update({k: float(v) for k, v in stats.items()})
            # Dormancy on the SAME probe states as churn, with the ReDo
            # criterion the analysis scripts use, so the live column and the
            # post-hoc one are the same measurement.
            dormant = redo_lib.dormant_stats(
                dormancy_params, probe_obs, num_policy_hidden,
                tau=redo_tau, activation_fn=policy_activation_fn,
                criterion=policy_criterion,
                activations_fn=policy_activations_fn)
            record['dormant_fraction'] = float(dormant['dormant_fraction'])
            record['zero_fraction'] = float(dormant['zero_fraction'])

        if member is None:
            record_centroid_scores(record, per_task)
        else:
            record_scores(record, per_task, prefix='incumbent')
            # ELITE = the best-performing agent (2026-09-13): for a PBT
            # population that IS the incumbent above.
            record_scores(record, per_task, prefix='elite')
            record_centroid_scores(record, centroid_per_task)
            record['pbt_exploits'] = int(pbt['exploits'])
        records.append(record)

        if step % checkpoint_interval == 0 or last:
            checkpoints.append(np.asarray(flat_params))
            checkpoint_updates.append(step)

        # `recording` above already guarantees the last update of every phase
        # is evaluated, so `flat_params` here IS the phase's closing policy.
        if phase_end(step) or last:
            phase_agents.append(np.asarray(flat_params))
            phase_tasks.append(int(task_idx))
            if member is not None:
                phase_centroids.append(np.asarray(centroid_flat))

        if step % log_interval == 0 or last:
            per_task_str = ' '.join(f"t{t}={per_task[t]:7.1f}"
                                    for t in range(num_tasks))
            print(f"  update {step:5d} task={task_idx:2d} "
                  f"H={record['entropy']:5.3f} {per_task_str} "
                  f"gen'ist={per_task.min():7.1f}", flush=True)

    elapsed = elapsed_before + (time.time() - start)
    threshold = cfg['solved_threshold']
    steps_per_update = hp['num_envs'] * hp['num_steps']

    result = {
        'config': {
            'env': env_name, 'method': method, 'schedule': schedule,
            'num_generations': num_updates, 'task_interval': task_interval,
            # 0 unless the run had a long first phase; see `make_phase_grid`.
            'task_warmup': int(task_warmup or 0),
            'noise_range': noise_range, 'num_tasks': num_tasks,
            'first_task_clean': first_task_clean,
            'pool_size': pool_size, 'pair_repeats': pair_repeats,
            'task_sequence': task_sequence.tolist(),
            'eval_episodes': eval_episodes, 'seed': seed, 'trial': trial,
            'hidden_dims': list(cfg['hidden_dims']),
            'episode_length': episode_length, 'num_params': num_params,
            'solved_threshold': threshold,
            **({'arch': arch} if arch else {}),
            # Present so the NES analysis code, which reads them, does not have
            # to special-case a PPO tag. PPO has no search distribution.
            'churn_interval': churn_interval,
            'num_probe_states': num_probe_states,
            # Recorded so a cchain tag says which coefficient it ran at; runs
            # made before 2026-08-29 have no entry and were at 0.1. The
            # correct value is per action space -- see PPO_CONFIGS.
            **({k: hp[k] for k in ('chain_target_rel_scale',
                                   'chain_warmup_updates', 'chain_coef_window',
                                   'chain_initial_coef', 'chain_floor')}
               if method == 'cchain' else {}),
            # Whether C-CHAIN was told where the sub-task boundaries are.
            # False since 2026-09-08 and for every run made after it; runs
            # with no entry were made when the reset was unconditional.
            **({'cchain_reset_on_switch': bool(cchain_reset_on_switch)}
               if method == 'cchain' else {}),
            # True since 2026-09-17: C-CHAIN normalises each minibatch's
            # advantages as PPO does here. Runs with no entry trained on raw
            # advantages (make_chain_sgd_epochs).
            **({'chain_minibatch_adv_norm': True} if method == 'cchain' else {}),
            'pop_size': hp['num_envs'], 'sigma': float('nan'),
            # The budget in steps, the field `scripts/verify_runs.py` and
            # the lineplot price an RL run by. Population or not: under
            # PBT the members share this budget, they do not multiply it.
            'num_timesteps': num_updates * hp['num_envs'] * hp['num_steps']
                             * (num_tasks if schedule == 'joint' else 1),
            **({'pbt_pop_size': pbt['N'], 'pbt_interval': int(pbt_interval),
                'pbt_exploit_fraction': exploit_fraction,
                'pbt_perturb_factor': perturb_factor, 'pbt_mode': pbt_mode,
                'pbt_exploits': int(pbt['exploits']),
                'elite_convention': 'best_member'}
               if pbt is not None else {}),
            'learning_rate': hp['learning_rate'], 'optimizer': 'adam',
            # Adam's momentum coefficient. Since 2026-09-05; a run without
            # the key ran at optax's 0.9.
            'adam_b1': adam_b1,
            'shaping': 'n/a', 'sigma_lr': 0.0, 'num_evals': 1,
            **{k: hp[k] for k in ('num_envs', 'num_steps', 'num_epochs',
                                  'num_minibatches', 'gamma', 'ent_coef',
                                  'clip_eps')},
            # Since 2026-09-05. Absent on a gymnax run made before, where all
            # of these were the values written here for gymnax.
            'head': head.name, 'normalize_obs': normalize_obs,
            'reward_scale': float(hp.get('reward_scale', 1.0)),
            'eval_interval': eval_interval,
            'checkpoint_every': int(checkpoint_every or 0),
            'value_hidden_dims': list(hp['value_hidden_dims']),
            **({'log_std_init': float(hp.get('log_std_init', -0.5))}
               if head is not actors.CATEGORICAL else {}),
            **({'task': env_params.describe()}
               if hasattr(env_params, 'describe') else {}),
        },
        'noise_vectors': np.asarray(noise_vectors).tolist(),
        'elapsed_seconds': elapsed,
        **summarise_records(records, num_tasks, threshold),
        'env_steps': num_updates * steps_per_update
                     * (num_tasks if schedule == 'joint' else 1),
    }

    if output_dir:
        write_run(output_dir, result, records,
                  centroids=np.stack(checkpoints),
                  generations=np.asarray(checkpoint_updates),
                  noise_vectors=np.asarray(noise_vectors))
        # Beside it, never through `save_eval_artifacts`: that writes a
        # `results.json` of its own and would clobber the one above.
        if pbt is None:
            save_checkpoints(
                output_dir,
                noise_vectors=[np.asarray(noise_vectors[t]) for t in phase_tasks],
                final=phase_agents)
        else:
            # A population has the NE arms' three networks: the elite is
            # both the best member and what the search hands back.
            save_checkpoints(
                output_dir,
                noise_vectors=[np.asarray(noise_vectors[t]) for t in phase_tasks],
                finalgen=phase_agents, incumbent=phase_agents,
                centroid=phase_centroids)
    if resume_path and os.path.exists(resume_path):
        os.remove(resume_path)          # a finished trial carries no stale checkpoint

    return result


def _with_lr(state, lr):
    """``state`` with its injected learning rate set to ``lr``.

    The PBT optimizer is ``chain(clip, inject_hyperparams(adam))``, so the
    opt_state is ``(clip_state, InjectHyperparamsState)`` and the rate is a
    leaf of the second -- settable without rebuilding the optimizer, which
    is what lets every member share one optax object.
    """
    clip_state, injected = state.opt_state
    injected = injected._replace(hyperparams={
        **injected.hyperparams,
        'learning_rate': jnp.asarray(lr, dtype=jnp.float32)})
    return state.replace(opt_state=(clip_state, injected))


class _HostKey:
    """A PRNG key on its way through pickle: typed key arrays are not numpy."""
    def __init__(self, data):
        self.data = data


def _host_tree(x):
    """jax -> numpy on every leaf of a pytree, PRNG keys as raw key data.

    run_nes's `_to_host` handles a key only at the top level; a PPO env carry
    holds one INSIDE a tuple, so this walks the tree. Python scalars pass
    through untouched, so a static field stays static on the way back.
    """
    def leaf(l):
        if isinstance(l, jax.Array):
            if jax.dtypes.issubdtype(l.dtype, jax.dtypes.prng_key):
                return _HostKey(np.asarray(random.key_data(l)))
            return np.asarray(l)
        return l
    return jax.tree_util.tree_map(leaf, x)


def _device_tree(x):
    def leaf(l):
        if isinstance(l, _HostKey):
            return random.wrap_key_data(jnp.asarray(l.data))
        return jnp.asarray(l) if isinstance(l, np.ndarray) else l
    return jax.tree_util.tree_map(leaf, x)


def build_parser():
    p = argparse.ArgumentParser(description='PPO on the study\'s sub-tasks')
    p.add_argument('--env', default='CartPole-v1',
                   choices=[e for e in ENV_NAMES if e in PPO_CONFIGS])
    p.add_argument('--method', default='ppo', choices=list(RL_METHODS))
    p.add_argument('--schedule', default='switch', choices=SCHEDULES)
    p.add_argument('--num_updates', type=int, default=None)
    p.add_argument('--task_interval', type=int, default=None)
    p.add_argument('--num_envs', type=int, default=None)
    p.add_argument('--num_steps', type=int, default=None)
    p.add_argument('--noise_range', type=float, default=None)
    p.add_argument('--num_tasks', type=int, default=2)
    p.add_argument('--eval_episodes', type=int, default=16)
    p.add_argument('--eval_interval', type=int, default=None,
                   help='score the policy and write a record every N updates; '
                        "None is the environment's own (1 on gymnax, 150 on "
                        'the mjx bodies). The last update of every phase is '
                        'always recorded.')
    p.add_argument('--task_options', nargs='*', default=None,
                   metavar='KEY=VALUE',
                   help="the suite's sub-task settings; see train_nes.py.")
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--trial', type=int, default=1)
    p.add_argument('--output_dir', default=None)
    p.add_argument('--log_interval', type=int, default=100)
    p.add_argument('--checkpoint_interval', type=int, default=5)
    p.add_argument('--cchain_reset_on_switch', type=int, default=0,
                   help="re-calibrate C-CHAIN's coefficient controller at "
                        'every sub-task boundary: clear the coefficient and '
                        'its loss history, and re-arm the warmup gate. OFF by '
                        'default, and the default is the point -- every other '
                        'method in this study is boundary-agnostic, so this '
                        'would hand C-CHAIN alone a signal about where the '
                        'switches are, and because the C-CHAIN paper itself '
                        'says the method "does not need to be aware of task '
                        'switches" and its Algorithm 1 has no reset. The '
                        'released reference code does reset anyway, in three '
                        'of its four suites, so this is kept as an option to '
                        'reproduce that. Matches --cchain_reset_on_switch in '
                        'the paper\'s trainers.')
    p.add_argument('--chain_target_rel_scale', type=float, default=None,
                   help="C-CHAIN's churn coefficient target: coef is "
                        'max(scale * |policy loss| / churn loss, floor). None '
                        "is the environment's own -- the reference's 10000 "
                        'for a categorical policy (source/studies/gymnax/cchain.py) and '
                        'its 0.05 for a continuous one '
                        '(source/studies/brax/my_brax/cchain.py), five orders of '
                        'magnitude apart because the churn term is a '
                        'cross-entropy in one and an action-mean MSE in the '
                        'other. The gymnax runs made before 2026-08-29 used '
                        '0.1, which pins coef at its floor of 1.0 on every '
                        'update (measured: 400/400) where the controller would '
                        'otherwise choose ~3500 -- so those runs are PPO plus a '
                        'unit-weight self-distillation term and not this '
                        'method. They are kept under the `cchain_coef1` tags.')
    p.add_argument('--chain_warmup_updates', type=int, default=None,
                   help='updates into a sub-task before the coefficient '
                        "controller is allowed to move off its start; None is "
                        "the environment's own (10 / 50)")
    p.add_argument('--chain_coef_window', type=int, default=None,
                   help='updates the controller averages the two losses over; '
                        "None is the environment's own (50 / 100)")
    p.add_argument('--churn_interval', type=int, default=10,
                   help='measure churn and dormancy every N updates; 0 turns '
                        'the diagnostics off. They are a pure observer, so a '
                        'run is the same run either way -- the only cost is '
                        'time. At 10 a 30,000-update run keeps 3,000 points, '
                        'which is more than any figure resolves.')
    p.add_argument('--num_probe_states', type=int, default=512,
                   help='size of the frozen probe batch churn and dormancy '
                        'are measured on. Frozen at the first rollout and '
                        'never refreshed.')
    p.add_argument('--pbt_pop_size', type=int, default=8,
                   help='PBT: members in the population. They SHARE the run\'s '
                        'update budget (member step %% N takes update step), '
                        'so the run costs what a PPO run costs.')
    p.add_argument('--pbt_interval', type=int, default=10,
                   help='PBT: updates per member between exploit/explore steps.')
    p.add_argument('--pbt_exploit_fraction', type=float, default=None,
                   help='PBT: bottom fraction that copies a top-fraction member '
                        '(default source/algorithms/rl/pbt.py, 0.2).')
    p.add_argument('--pbt_perturb_factor', type=float, default=None,
                   help='PBT: hyperparameters of a copied member are scaled by '
                        'U(1-f, 1+f) (default 0.2).')
    p.add_argument('--pbt_mode', default='full',
                   help="PBT: full | weights_only | hp_only (Jaderberg et al. "
                        "Sect. 4.1.2).")
    p.add_argument('--gpu', '--gpus', dest='gpu', default=None)
    p.add_argument('--wandb_project', default=None,
                   help='accepted for run_experiments.sh; wandb is not used here.')
    return p


def main():
    args = build_parser().parse_args()
    if args.gpu is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    from scripts.outdated.generalists.train.train_all import parse_task_options
    overrides = {k: v for k, v in (('num_envs', args.num_envs),
                                   ('num_steps', args.num_steps))
                 if v is not None}
    print(f"{args.method.upper()} | {args.env} | schedule={args.schedule} "
          f"| trial={args.trial}")
    run_ppo(env_name=args.env, method=args.method, schedule=args.schedule,
            num_updates=args.num_updates, task_interval=args.task_interval,
            noise_range=args.noise_range, num_tasks=args.num_tasks,
            eval_episodes=args.eval_episodes, seed=args.seed, trial=args.trial,
            output_dir=args.output_dir, log_interval=args.log_interval,
            checkpoint_interval=args.checkpoint_interval,
            chain_target_rel_scale=args.chain_target_rel_scale,
            chain_warmup_updates=args.chain_warmup_updates,
            chain_coef_window=args.chain_coef_window,
            cchain_reset_on_switch=bool(args.cchain_reset_on_switch),
            churn_interval=args.churn_interval,
            num_probe_states=args.num_probe_states, overrides=overrides,
            eval_interval=args.eval_interval,
            task_options=parse_task_options(args.task_options),
            pbt_pop_size=args.pbt_pop_size, pbt_interval=args.pbt_interval,
            pbt_exploit_fraction=args.pbt_exploit_fraction,
            pbt_perturb_factor=args.pbt_perturb_factor, pbt_mode=args.pbt_mode)


if __name__ == '__main__':
    main()
