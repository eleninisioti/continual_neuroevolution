"""The plasticity columns that a training curve cannot carry, from the saved agents.

    .venv/bin/python scripts/analysis/plasticity_checkpoints.py \\
        --runs_root projects/iclr_2027/runs/gymnax --phase continual --sigma 1.0 \\
        --out projects/iclr_2027/results/gymnax_continual/sigma1.0

Four of the paper's five plasticity panels are logged per record by the
trainers themselves (`training_metrics.json`: dormant fraction, churn, weight
magnitude, NTK rank) and `scripts/make_plasticity_figure.py` reads them
straight. Two are NOT functions of a scalar time series, because they need the
IDENTITY of a unit or of a parameter vector across time:

    weight mean            the SIGNED mean of the parameters, and their
                           variance about it. Not a curve column because only
                           some trainers logged it: the RL side logs
                           `*_weight_mean_abs`, a magnitude, which cannot show
                           a drift off centre, and ES and DNS logged no signed
                           mean at all. It is what TRAC's rescaling is supposed
                           to hold near its initialisation value, so a figure
                           that reports a magnitude instead is not reporting
                           the thing that method acts on.
    dormancy persistence   the same dormant FRACTION is produced by a network
                           whose dead units are dead for good and by one that
                           is merely sparse at any instant, using a different
                           handful of units each time. Only the first is
                           plasticity loss. Separating them needs the per-unit
                           mask at two points in time, not its mean.
    dormancy age           the same mask read as a duration: how many
                           consecutive sub-tasks the units dormant right now
                           have been dormant. The figure's row; the
                           persistence index is its chance-corrected control.
    parameter change       Juliani & Ash's "weight difference": how far the
                           agent MOVED between checkpoints, and how far it has
                           drifted from where it started. Both are distances
                           between two parameter vectors.

Plus one column that is logged per record but under the OTHER literature's
definition, and is recomputed here under this one:

    curvature              Lewandowski et al. attribute loss of plasticity to
                           the effective rank of the HESSIAN of the training
                           objective falling, and approximate it (their §4.1)
                           by the rank of the empirical Fisher G^T G, G the
                           matrix of per-example gradients. The live
                           `*_ntk_*` columns are Tang et al.'s NTK Gram, a
                           different matrix from the same family -- the mean
                           output rather than the log-likelihood of the chosen
                           action. Both are reported; where they disagree,
                           they disagree about which literature's claim the
                           runs support, and that is worth seeing.

This pass reads `checkpoints.npz` only. It never re-runs a search, it is
CPU-only, and it is safe to run beside training.

## Resolution, and what that costs

`checkpoints.npz` holds ONE saved agent per sub-task -- 20 points over a
3.072e9-step run, one every 200 NE generations or 1500 PPO updates. So:

  * persistence is measured in SUB-TASKS, not generations. "Always dormant"
    means dormant at the end of every sub-task, and the survival curve's lag
    is a lag in sub-tasks. A unit that goes quiet for ten generations and
    comes back is invisible here and reads as not dormant.
  * that makes every persistence number an UPPER bound on how transient
    dormancy is and a LOWER bound on how persistent it is: if a unit is not
    dormant across a 200-generation gap it is certainly not permanently dead.
  * `step_norm` is the distance moved over a whole sub-task, so it may be
    compared across methods (they spend the same steps per sub-task, CLAUDE.md
    rule (c)) but NOT read as a per-update step size.

Finer resolution needs runs saved with a checkpoint interval, which the
generalists study did (`scripts/outdated/generalists/analysis/dormancy_persistence.py`
reads those, at 5 generations) and this tree does not.

## The probe batch

`--probe` chooses the observations every dormancy and curvature column is
scored on. The default is `matched`, and the reason is that in this benchmark
a sub-task IS the observation distribution: under `obs_noise` the policy is fed
`x + v_k` with `v_k ~ N(0, sigma^2 I)`, and at sigma=1.0 that offset is 19x the
per-dimension spread of CartPole's first coordinate and 55x MountainCar's
second. A batch drawn on sub-task 0 and reused at sub-task 12 therefore scores
the network in an environment the agent is not in -- not "with some drift",
but on a region of input space essentially disjoint from the one it lives in.
Under `param` the body is rescaled instead, and the same argument applies more
weakly through the dynamics.

    matched   (default) sub-task t's own distribution at checkpoint t, which is
              the agent saved at the end of sub-task t. This is the ReDo
              reference's convention: it draws a fresh replay-buffer batch at
              every check (inspiration/redo/redo_dqn.py:170), so dormancy there
              is always relative to the data the agent is training on.
    pooled    an equal share of states from every sub-task, one fixed batch for
              every checkpoint. On-distribution in aggregate AND constant in
              time, which is what a persistence measure wants -- see below.
    subtask0  the historical batch: a random policy on the unperturbed body,
              reused at every checkpoint. Reproduces earlier outputs, and is
              kept as the control that shows how much the choice matters.

The sub-task sequence itself is read out of `checkpoints.npz`
(`noise_vectors`, `param_mults`), which the trainers write beside the agents,
so nothing about it is reconstructed or assumed.

In every mode the states come from a UNIFORM RANDOM policy, not from each
method's own visited states. The reference uses the agent's replay buffer,
which here would fold policy quality into the metric and make the arms
incomparable; a random-policy batch under the correct sub-task fixes the large
error without introducing that one.

## Persistence under a moving probe

`matched` moves the probe between checkpoints, so "the unit reactivated" could
in principle mean "we looked somewhere else". Here it does not: when the
sub-task IS the input distribution, a unit that is dormant under `v_5` and
active under `v_6` has genuinely reactivated, in the only sense that has
functional content. That is the plasticity claim, and it is what `matched`
measures.

`pooled` answers the stricter question -- is this unit dead as a FUNCTION,
across every distribution the run ever saw -- on states that do not move, so
unit identity is compared like for like. Run both: they are different claims,
and the persistence figure is worth drawing under each.

## The lineage caveat

For NES, OpenES and the RL arms the saved agent is a point that moves
continuously, so unit `i` at consecutive checkpoints is the same unit of the
same network. For GA and DNS the saved agent is an argmax over a population
and can jump lineage between sub-tasks, so low persistence there may be the
elite changing rather than a unit reactivating. Their numbers are computed and
are worth reading; they answer a slightly different question, and the figure
script says so.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import jax                                                   # noqa: E402
import jax.numpy as jnp                                      # noqa: E402
import gymnax                                                # noqa: E402

from source.algorithms.networks import ACTIVATIONS           # noqa: E402
from source.algorithms.rl import redo                        # noqa: E402
from source.algorithms.rl.ppo import GYMNAX_POLICY_ACTIVATION  # noqa: E402
from source.envs.run_context import RunContext, run_config as _merged_config  # noqa: E402
from source.envs.gymnax_classic import (                     # noqa: E402
    make_gymnax_env, wrap_actions,
    ENV_CONFIGS, apply_physics, build_policy, task_noise_vectors,
    unflatten_params)
from source.utils.task_sequence import (                     # noqa: E402
    GYMNAX_PHYSICS_TASKS, cycle_task_sequence, physics_mult_sequence)
from source.metrics.ntk import rank_stats                    # noqa: E402

# Which saved agent the columns are read off, in preference order, keyed by
# `--agent`. The NE trainers write `finalgen` (best member of the sub-task's
# final generation) and `incumbent`; the single-policy RL trainers write
# `final`. `elite` is the historical behaviour and the order
# `make_lineplot.ZT_SOURCES` uses, so every post-hoc column in the paper
# describes the same agent as the elite lineplot.
#
# `centroid` reads `centroid` instead, so the plasticity table describes the
# NETWORK THE CENTROID LINEPLOT SCORES -- `centroid_fitness` is the score of
# the coordinate-wise mean of the population's weights, and this is that mean.
# Per arm: `archive.mean(0)` for the GA, `population.mean(0)` for DNS,
# `es_state.mean` for ES/NES.
#
# `incumbent` is the FALLBACK and is a different object: what the optimizer
# would hand back, which is `archive[0]` for the GA (this trainer sets
# `ga_state.mean = archive[0]` every generation -- the best archive member, not
# any mean) and the repertoire best for DNS. It coincides with the centroid
# only for ES/NES. Runs written before the `centroid` key existed therefore
# give a GA or DNS row that is NOT a centroid row, and the loader says so.
# Two further caveats belong in the caption rather than being silently
# absorbed:
#
#   * the centroid of a GA archive or a novelty-selected repertoire is a
#     network the search never evaluated and would never hand back. Read those
#     rows as "what does averaging the population give", which is the same
#     thing `centroid_fitness` reports and the reason both are drawn.
#   * the RL arms have no population and write only `final`, so they fall
#     back to it under both modes and their rows are IDENTICAL in the two
#     figures. That is the same fallback `make_lineplot.METRIC_COLUMNS` makes
#     for `centroid`, so the two figures stay comparable.
AGENT_SOURCES = {
    'elite':    ('finalgen', 'final'),
    'centroid': ('centroid', 'incumbent', 'final'),
}

# Arms whose `incumbent` is NOT the network the centroid lineplot scores, so a
# fallback to it under `--agent centroid` silently plots a different network
# from the curve beside it.
#
#   ES, NES    `incumbent` IS `centroid`; both are `es_state.mean`.
#   GA         `incumbent` is `ga_state.mean`; `centroid_fitness` scores
#              `ga_state.archive.mean(axis=0)`, which is `centroid`.
#   DNS        `incumbent` is the repertoire BEST -- an argmax, not a mean --
#              while `centroid_fitness` scores the repertoire mean.
#   RL         no population; `final` under both, so the two figures agree.
#
# Runs made before the trainers started saving `centroid` have no such array
# and fall back; `check_agent_sources` names them rather than letting it pass.
INCUMBENT_IS_NOT_CENTROID = {'ga', 'ga_refresh', 'ga_reeval', 'ga_isoline',
                             'dns', 'dns_gaussian'}


def check_agent_sources(cells, agent):
    """Warnings for arms whose saved agent is not the one `agent` asked for."""
    if agent != 'centroid':
        return []
    out = []
    for cell, methods in sorted(cells.items()):
        for method, entry in sorted(methods.items()):
            if method not in INCUMBENT_IS_NOT_CENTROID:
                continue
            used = {t['agent_source'] for t in entry['trials'].values()}
            if used == {'centroid'}:
                continue
            out.append(
                f"{method}/{cell}: --agent centroid fell back to "
                f"{'/'.join(sorted(used))}, which for this arm is NOT the "
                f"network `centroid_fitness` scores. Its rows describe a "
                f"different agent from the centroid lineplot beside them. "
                f"These runs predate the trainers saving `centroid`; only a "
                f"re-run fixes it.")
    return out


# Methods whose saved agent is an argmax over a population and can change
# lineage between checkpoints. See "The lineage caveat" above.
LINEAGE_JUMPS = {'ga', 'ga_refresh', 'ga_reeval', 'ga_isoline', 'dns',
                 'dns_gaussian'}


PROBE_DESCRIPTIONS = {
    'matched': "uniform-random policy under SUB-TASK t's own environment, "
               "scored on the checkpoint saved at the end of sub-task t",
    'pooled':  'uniform-random policy, an equal share of states from every '
               'sub-task, one fixed batch for every checkpoint',
    'subtask0': 'uniform-random policy on sub-task 0, the unperturbed body, '
                'one fixed batch for every checkpoint',
}


def run_config(trial_dir):
    """The run's own config, or `None`. `results.json` is written by every arm."""
    path = pathlib.Path(trial_dir) / 'results.json'
    if not path.exists():
        return None
    try:
        blob = json.loads(path.read_text())
    except Exception:                                        # noqa: BLE001
        return None
    # The trainers write the interesting fields at the top level AND nest the
    # full argparse namespace under `config`; the top level wins where both
    # have a key, because that is the one every arm agrees on.
    return {**(blob.get('config') or {}),
            **{k: v for k, v in blob.items() if k != 'config'}}


def env_key(cell: str) -> str:
    """`CartPole_v1_sigma1.0` -> `CartPole-v1`, gymnax's own spelling."""
    return cell.split('_sigma')[0].replace('_v', '-v')


def rollout_observations(env_name, num_obs=512, seed=0, env_params=None):
    """A batch of observations under a UNIFORM RANDOM policy.

    Not any method's own visited states: the batch has to be neutral between
    the arms being compared, or a method's dormancy would be confounded with
    how good its policy is. `env_params` selects the body -- the stock one by
    default, a rescaled one for a `param` sub-task.

    Because the policy is uniform random it does not read the observation, so
    the state distribution it induces is independent of any observation
    offset. That is what lets the `obs_noise` family reuse one rollout for
    every sub-task and simply add the offset (`subtask_probes`).
    """
    env, base_params = make_gymnax_env(env_name)
    cfg = ENV_CONFIGS[env_name]
    if env_params is None:
        env_params = base_params
    env_params = env_params.replace(max_steps_in_episode=cfg['episode_length'])
    num_actions = env.action_space(env_params).n
    key = jax.random.key(seed)
    obs_list = []
    while len(obs_list) < num_obs:
        key, reset_key = jax.random.split(key)
        obs, state = env.reset(reset_key, env_params)
        for _ in range(cfg['episode_length']):
            obs_list.append(np.asarray(obs))
            if len(obs_list) >= num_obs:
                break
            key, act_key, step_key = jax.random.split(key, 3)
            action = jax.random.randint(act_key, (), 0, num_actions)
            obs, state, _, done, _ = env.step(step_key, state, action, env_params)
            if bool(done):
                break
    return jnp.asarray(np.stack(obs_list[:num_obs]))


def subtask_sequence(env_name, blob, cfg, num_tasks, obs_dim):
    """`(offsets, mults)` for one trial's sub-task sequence, as the run had them.

    Read straight out of `checkpoints.npz`, which every continual trainer
    writes alongside the agents: `noise_vectors` is `(num_tasks, obs_dim)`, the
    offset added to the observation before the policy reads it, and
    `param_mults` is `(num_tasks,)`, the multiplier on the body's physics. One
    of the two is trivial for any given run -- zeros under `param`, ones under
    `noise` -- and both are carried so this needs no branch on task type.

    Falling back to rebuilding them from `results.json` covers older runs that
    predate those arrays. It is exact where it applies, because the sequence is
    seeded from the trial index alone and deliberately not from the training
    RNG, but it is the fallback and not the path: the stored arrays are what
    the run actually used.

    This reads the sub-task schedule, which no TRAINER may see (CLAUDE.md (d)).
    That rule is about information reaching the search. This is a measurement
    pass over finished checkpoints that gains nothing for any method; it is the
    analysis knowing which environment an agent was in when it scores that
    agent in that environment.
    """
    offsets = mults = None
    if ('task_type' not in cfg and cfg.get('task')
            and 'noise_vectors' in getattr(blob, 'files', ())):
        # A shared-runner physics / actions run: the row is the sub-task
        # itself, not an offset. One resolver for every pass.
        from source.envs.gymnax_classic import saved_task_rows
        offsets, _bodies, mults = saved_task_rows(cfg, blob, None, env_name)
        offsets = np.asarray(offsets, dtype=np.float32)
    if offsets is None and 'noise_vectors' in getattr(blob, 'files', ()):
        offsets = np.asarray(blob['noise_vectors'], dtype=np.float32)
    if mults is None and 'param_mults' in getattr(blob, 'files', ()):
        mults = np.asarray(blob['param_mults'], dtype=np.float64)

    if offsets is None or mults is None:
        trial = int(cfg.get('trial', 1))
        period = int(cfg.get('task_period', 0) or 0)
        n = int(cfg.get('num_tasks', num_tasks))
        if cfg.get('task_type') == 'param' and mults is None:
            rng = (cfg.get('param_range')
                   or GYMNAX_PHYSICS_TASKS[env_name]['mult_range'])
            mults = np.asarray(physics_mult_sequence(trial, n, rng, period),
                               dtype=np.float64)
        if offsets is None:
            if cfg.get('task_type') == 'param':
                offsets = np.zeros((n, obs_dim), dtype=np.float32)
            else:
                seq = cycle_task_sequence(
                    list(task_noise_vectors(trial, n, obs_dim,
                                            float(cfg.get('noise_range', 1.0)))),
                    period)
                offsets = np.stack([np.asarray(v) for v in seq]).astype(np.float32)
        if mults is None:
            mults = np.ones(len(offsets), dtype=np.float64)

    if len(offsets) < num_tasks or len(mults) < num_tasks:
        return None, None
    return offsets[:num_tasks], mults[:num_tasks]


def subtask_probes(env_name, offsets, mults, num_obs, seed, mode, cache):
    """The probe batch(es) for one trial, under `--probe`.

    Returns `(T, num_obs, obs_dim)` for `matched` and `(num_obs, obs_dim)` for
    the two fixed modes. `cache` is a dict keyed by the rollout actually needed,
    so the random-policy rollouts are shared across methods and trials.

      subtask0   the historical batch: sub-task 0, the UNPERTURBED body, one
                 batch for every checkpoint. Kept as a control.
      matched    sub-task `t`'s own observation distribution at checkpoint `t`
                 -- the ReDo reference's convention, which resamples from the
                 current replay buffer at every check
                 (inspiration/redo/redo_dqn.py). The default.
      pooled     an equal share of states from EVERY sub-task, fixed across
                 time. On-distribution in aggregate and constant, which is
                 what a persistence measure wants: see the note below.
    """
    num_tasks = len(offsets)
    physics = not np.allclose(mults, 1.0)

    def batch(t):
        if physics:
            return np.asarray(_body_rollout(env_name, float(mults[t]), num_obs,
                                            seed, cache))
        return np.asarray(_base_rollout(env_name, num_obs, seed, cache)) + offsets[t]

    if mode == 'subtask0':
        return jnp.asarray(batch(0))
    if mode == 'matched':
        return jnp.asarray(np.stack([batch(t) for t in range(num_tasks)]))
    if mode == 'pooled':
        rows = np.concatenate([np.asarray(offsets, dtype=np.float64).reshape(num_tasks, -1),
                               np.asarray(mults, dtype=np.float64).reshape(num_tasks, -1)], 1)
        return jnp.asarray(pooled_probe(batch, rows, num_obs))
    raise ValueError(f'unknown probe mode {mode!r}')


def pooled_probe(batch, rows, num_obs):
    """`--probe pooled`: an equal share of `num_obs` from each DISTINCT sub-task.

    `rows[t]` identifies checkpoint t's sub-task and `batch(t)` is its states.
    Distinct, not every checkpoint: a two-sub-task schedule alternating over
    twenty checkpoints would otherwise pool ten copies of the same two slices
    and score the network on 50 states instead of 512.
    """
    _, first = np.unique(np.round(np.asarray(rows, dtype=np.float64), 9),
                         axis=0, return_index=True)
    ts = sorted(first)
    share = max(1, num_obs // len(ts))

    def spread(b):
        # Evenly through the batch: the gymnax batch is filled episode by
        # episode, so its first `share` states would be one early trajectory.
        b = np.asarray(b)
        return b[np.linspace(0, len(b) - 1, min(share, len(b))).astype(int)]
    return np.concatenate([spread(batch(t)) for t in ts])[:num_obs]


def _base_rollout(env_name, num_obs, seed, cache):
    key = ('base', env_name)
    if key not in cache:
        cache[key] = rollout_observations(env_name, num_obs, seed)
    return cache[key]


def _body_rollout(env_name, mult, num_obs, seed, cache):
    """One rollout per distinct body. Rounded so a repeated sub-task hits it."""
    key = ('body', env_name, round(mult, 9))
    if key not in cache:
        _, base_params = make_gymnax_env(env_name)
        spec = GYMNAX_PHYSICS_TASKS[env_name]
        cache[key] = rollout_observations(
            env_name, num_obs, seed,
            apply_physics(env_name, base_params, spec['param'], mult))
    return cache[key]


# A unit is SATURATED at a checkpoint when |output| > SAT_LEVEL with one sign
# on at least SAT_SHARE of the probe states: a constant +/-1, which passes
# (almost) no gradient to its incoming weights and only a bias to the next
# layer. A unit that flips sign between states is a working binary feature and
# is not counted. Reported beside the dormant (silent) fraction, not in it.
SAT_LEVEL = 0.99
SAT_SHARE = 0.99


def dormancy_masks(flat_checkpoints, template, num_hidden, activation_fn,
                   criterion, probe, tau, activations_fn=None):
    """`(T, units)` boolean: which hidden units are dormant at each checkpoint.

    The score, its layer normalisation and `tau` are ReDo's own, taken from
    `source/algorithms/rl/redo.py` rather than restated, so "dormant" means
    here exactly what it means in the live columns and in what ReDo recycles.
    `activations_fn(params, obs)` is the network reporting its own hidden
    layers (the MiniGrid conv policy); without it the MLP is walked by
    parameter name, as `redo.dormant_stats` does.
    """
    def scores(flat, obs):
        params = unflatten_params(flat, template)
        if activations_fn is not None:
            acts = activations_fn(params, obs)
        else:
            acts = redo.hidden_activations(params, obs, num_hidden,
                                           activation_fn=activation_fn)
        # Saturation: the share of states on which a unit sits beyond
        # +/-SAT_LEVEL, on whichever side it spends more time. Flattened over
        # every leading axis, as the dormancy score is.
        sat = []
        for a in acts:
            a = a.reshape(-1, a.shape[-1])
            sat.append(jnp.maximum((a > SAT_LEVEL).mean(0), (a < -SAT_LEVEL).mean(0)))
        return (jnp.concatenate(redo.score_layers(acts, criterion)),
                jnp.concatenate(sat))

    if probe.ndim == 3:
        # One probe per checkpoint (`--probe matched`): checkpoint `t` is the
        # agent at the end of sub-task `t`, so it is scored on sub-task `t`'s
        # own observations. The trailing checkpoints of a run with fewer
        # sub-tasks than probes are dropped by the caller, not padded.
        all_scores = jax.jit(jax.vmap(scores))(flat_checkpoints, probe[:len(flat_checkpoints)])
    else:
        all_scores = jax.jit(jax.vmap(scores, in_axes=(0, None)))(
            flat_checkpoints, probe)
    all_scores, all_sat = all_scores
    # Only a bounded activation saturates; a ReLU unit above 0.99 is just large.
    saturated = (np.asarray(all_sat >= SAT_SHARE)
                 if activation_fn is ACTIVATIONS['tanh'] else None)
    return (np.asarray(all_scores <= tau), np.asarray(np.isclose(all_scores, 0.0)),
            saturated)


def persistence(mask):
    """Persistence statistics of a `(T, units)` dormancy mask.

    `instant` is the ordinary dormant fraction, meaned over checkpoints -- the
    number every plasticity paper reports, recomputed here so the comparison
    with the rest of this dict is like for like.

    `survival[k-1]` is P(dormant at t+k | dormant at t), pooled over units and
    over every valid t. Its CHANCE LEVEL is `instant`: if the dormant set were
    redrawn independently at each checkpoint, a dormant unit would be dormant
    k checkpoints later exactly `instant` of the time. A survival curve lying
    on that line is functional sparsity; one near 1 is capacity that has left.

    `age_mean` is the mean length, in checkpoints, of a maximal run of
    consecutive dormant checkpoints for one unit -- the dormancy age. Episodes
    running off either end of the record are censored (they are at least as
    long as measured), and `age_censored` is the fraction of episodes that are,
    because a network whose units are dormant from checkpoint 0 to the end
    reports an age of T with every episode censored, and that is a different
    statement from an age of T that was observed to start and stop.
    """
    T, units = mask.shape
    instant = float(mask.mean())
    survival = []
    for k in range(1, T):
        base = mask[:T - k]
        if base.sum() == 0:
            survival.append(None)
        else:
            survival.append(float((mask[k:] & base).sum() / base.sum()))

    lengths, censored = [], 0
    for u in range(units):
        col = mask[:, u]
        t = 0
        while t < T:
            if not col[t]:
                t += 1
                continue
            start = t
            while t < T and col[t]:
                t += 1
            lengths.append(t - start)
            if start == 0 or t == T:
                censored += 1
    return {
        'instant': instant,
        'always': float((mask.all(axis=0)).mean()),
        'never': float((~mask.any(axis=0)).mean()),
        'survival': survival,
        'age_mean': float(np.mean(lengths)) if lengths else 0.0,
        'age_max': int(np.max(lengths)) if lengths else 0,
        'age_censored': float(censored / len(lengths)) if lengths else 0.0,
        'num_episodes': len(lengths),
    }


def persistence_index(mask):
    """Chance-corrected one-step persistence at each checkpoint: `(T,)`, NaN at 0.

    The `survival` curve in `persistence` pools every lag and every start, which
    makes it a figure of its own against lag and NOT a series that can be drawn
    on the same time axis as the dormant fraction. This is its lag-1 slice
    resolved in time, so the two can sit as adjacent rows of one figure:

        index_t = ( P(dormant at t | dormant at t-1) - p_t ) / ( 1 - p_t )

    with `p_t` the dormant fraction at `t`, which is exactly the chance level --
    if the dormant set at `t` were redrawn independently of `t-1`, a unit
    dormant at `t-1` would be dormant again with probability `p_t`. So 0 is
    functional sparsity (a different handful of units each time) and 1 is
    capacity that has left (the same units, still dormant). Negative values are
    possible and meaningful: the dormant set ANTI-correlates across the
    boundary, i.e. the units that go quiet are systematically the ones that
    were active.

    NaN where it is undefined: at `t=0`, where there is no predecessor; where
    nothing was dormant at `t-1`; and where `p_t = 1`, since a fully dormant
    network survives trivially and the correction divides by zero.
    """
    T = mask.shape[0]
    out = np.full(T, np.nan)
    for t in range(1, T):
        base = mask[t - 1]
        if not base.any():
            continue
        p = float(mask[t].mean())
        if p >= 1.0:
            continue
        surv = float((mask[t] & base).sum() / base.sum())
        out[t] = (surv - p) / (1.0 - p)
    return out


def dormant_age(mask):
    """Mean age of the units dormant at each checkpoint: `(T,)`, NaN where none is.

    The age of a dormant unit at checkpoint `t` is the number of consecutive
    checkpoints, ending at `t`, for which it has been dormant -- 1 for a unit
    that went quiet this sub-task, `t + 1` for one that has been quiet since
    the first checkpoint. The row averages that over the units dormant at `t`
    ONLY: averaging over every unit with 0 for the active ones multiplies the
    dormant fraction back in and says nothing the row above does not.

    It is the persistence index in the units a reader already has: "the
    dormant units have been dormant for N sub-tasks". A method that recycles
    or replaces its dormant units sits flat near 1; one whose units die and
    stay dead climbs the ceiling `t + 1`, which is the censoring bound -- the
    record starts at checkpoint 0, so no age can be longer than the record so
    far, and the early part of every curve is compressed against it. Unlike
    the index it is NOT chance-corrected: an arm with a larger dormant
    fraction accumulates longer ages by chance alone, which is why the
    `persistence` dict beside it is kept.
    """
    T, units = mask.shape
    age = np.zeros(units, dtype=int)
    out = np.full(T, np.nan)
    for t in range(T):
        age = np.where(mask[t], age + 1, 0)
        if mask[t].any():
            out[t] = float(age[mask[t]].mean())
    return out


def fisher_rank(flat_checkpoints, template, policy, probe, max_samples=128):
    """Effective rank of the empirical Fisher at each checkpoint, or `None`s.

    Lewandowski et al.'s approximation to the rank of the Hessian: G is the
    matrix of PER-EXAMPLE gradients and the quantity of interest is the rank of
    G^T G, which is the rank of the (n, n) Gram G G^T -- the smaller matrix,
    with the same nonzero spectrum. The per-example scalar is
    log pi(a*|s) at a* = argmax pi(.|s), the action the agent actually takes:
    both families act greedily here, and this is the empirical Fisher's own
    definition (the model's prediction standing in for a label).

    That differs from the live `*_ntk_*` columns, which use the MEAN OUTPUT as
    the scalar (Tang et al.'s Eq. 2). Same machinery, different function, and
    they are reported side by side rather than merged.

    NOTHING HERE MAY RAISE: `rank_stats` already refuses to, and this adds the
    guard for the gradients themselves. DNS's genotypes reach |w| ~ 1e14 on
    this tree, which overflows to non-finite logits and gives LAPACK a matrix
    it will not factor -- an observer that kills the analysis of the run it
    observes is worse than no observer.
    """
    def obs_at(t):
        """The states checkpoint `t` is differentiated on.

        Same probe the dormancy masks use, so the curvature row and the
        dormancy row describe the same network on the same states -- which is
        the whole reason they are drawn together.
        """
        return (probe[min(t, len(probe) - 1)][:max_samples] if probe.ndim == 3
                else probe[:max_samples])

    def logp_of_greedy(flat, x):
        logits = policy.apply(unflatten_params(flat, template), x[None])[0]
        return jax.nn.log_softmax(logits)[jnp.argmax(jax.lax.stop_gradient(logits))]

    def gram(flat, obs):
        # `flat` IS the parameter vector, so the gradient comes back flat and
        # there is nothing to ravel.
        grads = jax.vmap(lambda x: jax.grad(logp_of_greedy)(flat, x))(obs)
        return grads @ grads.T

    jgram = jax.jit(gram)
    out = []
    for t in range(flat_checkpoints.shape[0]):
        obs = obs_at(t)
        try:
            g = np.asarray(jgram(flat_checkpoints[t], obs))
        except Exception:                                    # noqa: BLE001
            out.append({'fisher_effective_rank': None, 'fisher_srank': None,
                        'fisher_trace': None, 'fisher_rank_ratio': None,
                        'fisher_num_probe': len(obs)})
            continue
        out.append(rank_stats(g, prefix='fisher'))
    return out


def argmax_shift(flat_checkpoints, template, policy, probe, max_samples=512,
                 continuous=False):
    """Fraction of probe states whose greedy action changed. One value per gap.

    `continuous`: the policy returns an action in [-1, 1] (the mjx bodies)
    rather than logits, and the value is the mean normalised L2 distance
    between the two actions over the probe -- in [0, 1] like the argmax
    fraction, 0 for identical behaviour, 1 for every actuator flipped end to
    end (`source/metrics/behavioral_divergence.normalized_action_distance`).

    The assumption-free counterpart of `calibrated_policy_kl`: both families
    act by `argmax` over the logits, so this compares the policies as they are
    actually deployed and needs no temperature to be chosen. Its cost is
    resolution -- it is quantised at 1/|probe| and blind to any change that
    does not cross a decision boundary, while counting a near-tie that flips as
    a full disagreement.

    Both networks of a pair are scored on checkpoint `t`'s probe, so the value
    is a difference between two policies and not between two batches.

    Returns `[None, s_1, ..., s_{T-1}]`, aligned with the checkpoint axis, with
    `None` wherever a genotype is not finite.
    """
    def greedy(flat, obs):
        logits = np.asarray(policy.apply(
            unflatten_params(jnp.asarray(flat), template), obs))
        if continuous:
            return np.nan_to_num(logits)
        return np.argmax(np.nan_to_num(logits, nan=-np.inf), axis=-1)

    def shift(a, b):
        if continuous:
            dim = a.shape[-1]
            return float(np.linalg.norm(a - b, axis=-1).mean() / (np.sqrt(dim) * 2.0))
        return float((a != b).mean())

    def obs_at(t):
        return (probe[min(t, len(probe) - 1)][:max_samples] if probe.ndim == 3
                else probe[:max_samples])

    out = [None]
    for t in range(1, flat_checkpoints.shape[0]):
        pair = flat_checkpoints[t - 1], flat_checkpoints[t]
        if not all(np.isfinite(np.asarray(f)).all() for f in pair):
            out.append(None)
            continue
        obs = obs_at(t)
        try:
            out.append(shift(greedy(pair[0], obs), greedy(pair[1], obs)))
        except Exception:                                    # noqa: BLE001
            out.append(None)
    return out


def calibrated_policy_kl(flat_checkpoints, template, policy, probe, num_actions,
                         entropy_frac=0.5, max_samples=512):
    """C-CHAIN's churn, made computable for the NE arms. One value per gap.

    C-CHAIN measures discrete churn as the cross-entropy H(pi_before, pi_after)
    between two successive policies on a fixed batch of states, and the gymnax
    RL trainers log exactly that per update (`policy_churn`). It cannot be read
    off an NE policy as it stands, for two separate reasons, and this function
    removes both rather than declaring the row RL-only:

      **the temperature is arbitrary.** NE fitness depends on `argmax(logits)`
      alone, so nothing anchors the SCALE of the logits: multiplying them by
      1000 is the same policy and gives a completely different softmax. The
      genotypes here make that concrete -- the same estimator applied to the
      raw logits gives a KL of ~1.6 for the RL arms and ~84 (GA) to ~203 (ES)
      for the NE ones, which is a fact about |w| and not about the policies.
      Fixed by CALIBRATING each network's temperature: solve for the T whose
      mean softmax entropy over the probe batch is `entropy_frac * log|A|`.
      That is scale-free by construction -- only the RELATIVE logit gaps
      survive -- and it is the minimum assumption that makes a distribution out
      of an argmax policy at all. It is an assumption, and the target is a
      knob: report it, and check a second value before quoting a number.

      **the cross-entropy has an entropy floor.** H(pi_b, pi_a) =
      H(pi_b) + KL(pi_b || pi_a), so a run whose policy entropy collapses
      reports a smaller churn for that reason alone and not because it moved
      less. On this tree the RL arms' own entropies span 0.014 (PPO) to 0.535
      (ReDo) nats, a 40x spread, which is most of the range of the churn column
      itself. So this reports the KL -- the half that is the policy MOVING.
      C-CHAIN's gradient is identical either way, the two differing by a
      constant in its loss. With both networks calibrated to the same entropy
      the cross-entropy is recoverable as `entropy_frac * log|A| + KL`.

    The clock is one SUB-TASK, not one update: `checkpoints.npz` holds one
    agent per sub-task, so this is the policy-space counterpart of `step_norm`
    (which is the same gap measured in parameter space) and NOT the same number
    as the per-update `policy_churn` column. Both are worth drawing and they
    must not be merged.

    Measured on checkpoint `t`'s probe -- the sub-task the LATER of the two
    networks was trained on -- which is the convention `fisher_rank` uses.

    Returns `[None, k_1, ..., k_{T-1}]`, aligned with the checkpoint axis and
    `None` wherever the pair could not be scored (a non-finite genotype, or a
    network whose logits are so flat that no temperature reaches the target).
    """
    target = float(entropy_frac) * float(np.log(num_actions))

    def logits_at(flat, obs):
        return np.asarray(policy.apply(unflatten_params(jnp.asarray(flat), template),
                                       obs), dtype=np.float64)

    def probs_at_target(logits):
        """Softmax at the temperature matching `target`, or None.

        Bisection on log T over 4e-18..1e26, in float64 and with the max
        subtracted, so a genotype at |w| ~ 1e14 neither overflows nor pins the
        search: mean entropy is monotone in T, which is what makes 80 halvings
        exact to machine precision.
        """
        L = logits - logits.max(axis=-1, keepdims=True)

        def mean_entropy(log_t):
            p = np.exp(L / np.exp(log_t))
            p /= p.sum(axis=-1, keepdims=True)
            return float(np.mean(-(p * np.log(np.clip(p, 1e-300, None))).sum(-1)))

        lo, hi = -40.0, 60.0
        if mean_entropy(hi) < target or mean_entropy(lo) > target:
            return None
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            if mean_entropy(mid) < target:
                lo = mid
            else:
                hi = mid
        p = np.exp(L / np.exp(0.5 * (lo + hi)))
        return p / p.sum(axis=-1, keepdims=True)

    def obs_at(t):
        return (probe[min(t, len(probe) - 1)][:max_samples] if probe.ndim == 3
                else probe[:max_samples])

    out = [None]
    for t in range(1, flat_checkpoints.shape[0]):
        pair = flat_checkpoints[t - 1], flat_checkpoints[t]
        if not all(np.isfinite(np.asarray(f)).all() for f in pair):
            out.append(None)
            continue
        obs = obs_at(t)
        try:
            before = probs_at_target(logits_at(pair[0], obs))
            after = probs_at_target(logits_at(pair[1], obs))
        except Exception:                                    # noqa: BLE001
            out.append(None)
            continue
        if before is None or after is None:
            out.append(None)
            continue
        kl = (before * (np.log(np.clip(before, 1e-300, None))
                        - np.log(np.clip(after, 1e-300, None)))).sum(-1)
        out.append(float(np.mean(kl)))
    return out


def analyse_trial(path, template, policy, num_hidden, activation_fn, criterion,
                  probe_fn, tau, with_fisher, sources=AGENT_SOURCES['elite'],
                  num_actions=None, entropy_frac=0.5, activations_fn=None,
                  continuous=False):
    """Every checkpoint-resolution column for one trial, or `None`.

    `probe_fn(blob, num_checkpoints, num_params) -> probe` is called once the
    archive is open, because under `--probe matched` the batch is per sub-task
    and the sub-task sequence is stored in the archive itself.
    """
    try:
        blob = np.load(path, allow_pickle=True)
    except Exception:                                        # noqa: BLE001
        return None
    source = next((s for s in sources if s in blob.files), None)
    if source is None:
        return None
    flat = np.asarray(blob[source], dtype=np.float32)
    if flat.ndim != 2 or flat.shape[0] < 2:
        return None

    # A checkpoint that overflowed is not a network. Recording it as such is
    # the point -- DNS diverges on this tree -- but it must not be scored as
    # if its dormancy or its distance meant anything.
    finite = np.isfinite(flat).all(axis=1)
    # SATURATION is the failure `finite` misses. A genotype that has run away
    # stops at float32's largest value rather than at inf, so every check for
    # non-finiteness passes while every parameter is pinned to the same number:
    # consecutive checkpoints are then identical and the step norm reads a
    # perfectly healthy 0. `ga_isoline` does exactly this on MountainCar, in
    # all ten trials. Counted here so a figure can name it instead of drawing
    # 3.4e38 as though it were a measurement.
    saturated = (np.abs(flat) >= 0.9 * np.finfo(np.float32).max).any(axis=1)
    probe = probe_fn(blob, int(flat.shape[0]), int(flat.shape[1]))
    if probe is None:
        return None
    dormant, zero, saturated_mask = dormancy_masks(jnp.asarray(flat), template, num_hidden,
                                   activation_fn, criterion, probe, tau,
                                   activations_fn)
    # Every distance in float64. The checkpoints are stored float32, and a
    # norm squares its argument: DNS's genotypes diverge past 1e19 on the
    # centroid tree, where the SQUARE overflows float32 and the norm comes back
    # `inf` -- a silently infinite step norm, not an error. float64 buys 19
    # more decades and costs nothing at 20 x 386 values.
    wide = flat.astype(np.float64)
    steps = np.linalg.norm(np.diff(wide, axis=0), axis=1)
    drift = np.linalg.norm(wide - wide[0], axis=1)
    rms = np.sqrt(np.mean(wide ** 2, axis=1))
    # The SIGNED mean and the variance about it. They decompose the norm --
    # rms**2 = mean**2 + var -- so a network whose weights are spreading and
    # one whose weights are drifting off centre have the same norm and
    # different (mean, var); initialisation puts the mean at ~0. Computed here
    # rather than read from `training_metrics.json` because only some of the NE
    # trainers ever logged `weight_mean` (not ES, not DNS) and none of the RL
    # ones did -- they log `*_weight_mean_abs`, which is a magnitude and cannot
    # show a drift off centre at all. From the saved agent, every arm has it.
    mean = wide.mean(axis=1)
    var = wide.var(axis=1)

    out = {
        'agent_source': source,
        'num_probe': int(probe.shape[-2]),
        'num_checkpoints': int(flat.shape[0]),
        'num_units': int(dormant.shape[1]),
        # What `action_shift` measures on this policy: the argmax fraction on
        # a logits head, the normalised action distance on a continuous one.
        'action_shift_kind': 'distance' if continuous else 'argmax',
        'num_finite_checkpoints': int(finite.sum()),
        'num_saturated_checkpoints': int(saturated.sum()),
        'dormant_fraction': dormant.mean(axis=1).tolist(),
        # `None` on an unbounded activation (ReLU), which cannot saturate.
        'saturated_fraction': (None if saturated_mask is None
                               else saturated_mask.mean(axis=1).tolist()),
        'zero_fraction': zero.mean(axis=1).tolist(),
        'dormant': persistence(dormant),
        'zero': persistence(zero),
        # From checkpoint 1: the index is undefined at 0, and dropping the
        # leading gap matches the `step_norm` convention the figure already
        # positions with `CHECKPOINT_ROWS[...][1]`.
        'persistence_index': [None if np.isnan(v) else float(v)
                              for v in persistence_index(dormant)[1:]],
        # Mean age, in sub-tasks, of the units dormant at each checkpoint.
        # `None` where nothing is dormant: an age over no units is not 0.
        'dormant_age': [None if np.isnan(v) else float(v)
                        for v in dormant_age(dormant)],
        'step_norm': steps.tolist(),
        'drift': drift.tolist(),
        'weight_rms': rms.tolist(),
        'weight_mean': mean.tolist(),
        'weight_var': var.tolist(),
    }
    if with_fisher:
        out['curvature'] = fisher_rank(jnp.asarray(flat), template, policy, probe)
    out['action_shift'] = argmax_shift(flat, template, policy, probe,
                                       continuous=continuous)
    # `policy_kl` softmaxes the output; on a continuous head that is not a
    # distribution and the column is left out.
    if num_actions and not continuous:
        out['policy_kl'] = calibrated_policy_kl(
            flat, template, policy, probe, num_actions, entropy_frac)
        out['policy_kl_entropy_frac'] = float(entropy_frac)
    return out


def pick_criterion(args, activation):
    """The dormancy score: `--criterion`, or ReDo's choice for the activation.

    `magnitude` on a tanh network counts only SILENT units (output ~0), not
    the ones saturated at +/-1 that `variability` also flags.
    """
    if args.criterion == 'auto':
        return redo.criterion_for_activation(activation)
    return args.criterion


def analyse_suite_cell(method, cell_dir, contexts, cells, skipped, args):
    """`analyse_trial` over one non-gymnax cell. False if no run could be rebuilt.

    The probe under `--probe matched` is the random-policy batch on sub-task
    t's own environment (`RunContext.probe`, the suite's
    `random_policy_observations`), one per distinct row of `noise_vectors`;
    `pooled` and `subtask0` are as for gymnax.
    """
    trials = {}
    for trial_dir in sorted(cell_dir.glob('trial_*')):
        ckpt = trial_dir / 'checkpoints.npz'
        path = trial_dir / 'results.json'
        if not (ckpt.exists() and path.exists()):
            continue
        try:
            cfg = _merged_config(json.loads(path.read_text()))
        except Exception:                                    # noqa: BLE001
            continue
        key = RunContext.cache_key(cfg, 1)
        if key not in contexts:
            contexts[key] = RunContext(cfg, episodes=1)
        ctx = contexts[key]
        activation_fn = ACTIVATIONS[ctx.activation]
        criterion = pick_criterion(args, ctx.activation)

        def probe_fn(blob, T, _n, _ctx=ctx):
            if 'noise_vectors' not in getattr(blob, 'files', ()):
                return None
            rows = np.asarray(blob['noise_vectors'])[:T]
            if len(rows) < T:
                return None
            batch = lambda t: np.asarray(_ctx.probe(rows[t], args.num_probe,  # noqa: E731
                                                    args.probe_seed))
            if args.probe == 'subtask0':
                return jnp.asarray(batch(0))
            if args.probe == 'matched':
                return jnp.asarray(np.stack([batch(t) for t in range(T)]))
            return jnp.asarray(pooled_probe(batch, rows.reshape(T, -1), args.num_probe))

        res = analyse_trial(ckpt, ctx.template, ctx.policy,
                            len(cfg['hidden_dims']), activation_fn, criterion,
                            probe_fn, args.tau, not args.no_curvature,
                            AGENT_SOURCES[args.agent],
                            num_actions=(ctx.num_actions if args.entropy_frac
                                         else None),
                            entropy_frac=args.entropy_frac,
                            activations_fn=ctx.hidden_activations_fn,
                            continuous=ctx.continuous)
        if res is None:
            skipped.append((method, cell_dir.name,
                            f'{trial_dir.name}: unreadable checkpoints'))
            continue
        trials[trial_dir.name] = res
    if not trials:
        return False
    cells.setdefault(cell_dir.name, {})[method] = {
        'lineage_jumps': method in LINEAGE_JUMPS,
        'trials': trials,
    }
    fell_back = sorted({t['agent_source'] for t in trials.values()}
                       - {AGENT_SOURCES[args.agent][0], 'final'})
    note = f'  [fell back to {"/".join(fell_back)}]' if fell_back else ''
    print(f'  {method:12s} {cell_dir.name:28s} {len(trials)} trials, '
          f'{trials[next(iter(trials))]["num_checkpoints"]} checkpoints{note}')
    return True


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--runs_root', default='projects/iclr_2027/runs/gymnax')
    ap.add_argument('--phase', default='continual')
    ap.add_argument('--sigma', default=None,
                    help='only cells ending `_sigma<S>`; omitted means the '
                         'cells with no sigma in their name')
    ap.add_argument('--cells', nargs='*', default=None,
                    help='explicit cell directory names, one per environment, '
                         'e.g. CartPole_v1_sigma1.0 Acrobot_v1_sigma1.0 '
                         'MountainCar_v0_sigma0.1 -- for a reported grid whose '
                         'noise differs by environment. Mutually exclusive '
                         'with --sigma. Pass the SAME list to '
                         'make_plasticity_figure.py.')
    ap.add_argument('--methods', nargs='*', default=None)
    ap.add_argument('--envs', nargs='*', default=None,
                    help='cell-name prefixes, e.g. CartPole_v1')
    ap.add_argument('--num_probe', type=int, default=512)
    ap.add_argument('--probe_seed', type=int, default=0)
    ap.add_argument('--probe', default='matched',
                    choices=['matched', 'pooled', 'subtask0'],
                    help="which observations the dormancy and curvature "
                         "columns are scored on. `matched` (default) uses "
                         "sub-task t's own distribution at checkpoint t, the "
                         "ReDo reference's convention -- it resamples the "
                         "replay buffer at every check. `pooled` mixes an "
                         "equal share of every sub-task and holds that fixed, "
                         "so unit identity is compared on identical states. "
                         "`subtask0` is the historical batch (the unperturbed "
                         "body, always) and reproduces earlier outputs.")
    ap.add_argument('--tau', type=float, default=redo.DEFAULT_TAU)
    ap.add_argument('--criterion', default='auto',
                    choices=['auto', redo.CRITERION_MAGNITUDE, redo.CRITERION_VARIABILITY],
                    help="dormancy score. `auto` is ReDo's per activation "
                         "(variability on tanh: silent AND saturated units); "
                         "`magnitude` counts silent units only, on every body.")
    ap.add_argument('--entropy-frac', type=float, default=0.5,
                    help="target mean softmax entropy for `policy_kl`, as a "
                         "fraction of log|A|. NE policies are argmax policies "
                         "whose temperature is arbitrary, so one has to be "
                         "chosen before C-CHAIN's churn is defined for them at "
                         "all; both networks in a pair are calibrated to it, "
                         "which also removes the entropy floor the raw "
                         "cross-entropy has on the RL side. Vary it and re-run "
                         "before quoting a number. 0 disables the column.")
    ap.add_argument('--no-curvature', action='store_true',
                    help='skip the empirical-Fisher rank, the only column here '
                         'that costs a jacobian per checkpoint')
    ap.add_argument('--agent', default='elite',
                    choices=sorted(AGENT_SOURCES),
                    help="which saved network per sub-task to measure. "
                         "`elite` (default) is the best member of the "
                         "sub-task's final generation, matching the "
                         "elite_eval lineplot. `centroid` is the coordinate-"
                         "wise mean of the population's weights -- the network "
                         "`centroid_fitness` scores, so the rows describe the "
                         "individual the centroid lineplot draws. Runs saved "
                         "before that key existed fall back to `incumbent`, "
                         "which is a DIFFERENT network for the GA and DNS; "
                         "see AGENT_SOURCES.")
    ap.add_argument('--out', required=True,
                    help='directory for plasticity_checkpoints.json')
    args = ap.parse_args()

    root = pathlib.Path(args.runs_root) / args.phase
    if not root.is_dir():
        sys.exit(f'no such directory: {root}')

    if args.cells and args.sigma:
        sys.exit('ERROR: pass --cells or --sigma, not both. --cells already '
                 'names the sigma of each environment.')
    wanted_cells = set(args.cells) if args.cells else None

    def wanted(cell):
        if wanted_cells is not None:
            if cell not in wanted_cells:
                return False
        elif args.sigma is None:
            if '_sigma' in cell:
                return False
        elif not cell.endswith(f'_sigma{args.sigma}'):
            return False
        return not args.envs or any(cell.startswith(e) for e in args.envs)

    probes, policies, obs_dims, num_actions = {}, {}, {}, {}
    # Non-gymnax cells: one `RunContext` per run shape, from a trial's own
    # config. The policy, its activation and its probe batches all come from
    # it, so a MiniGrid conv net and a gymnax MLP go through one `analyse_trial`.
    contexts: dict = {}
    cells: dict[str, dict] = {}
    skipped = []
    for method_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        method = method_dir.name
        if args.methods and method not in args.methods:
            continue
        for cell_dir in sorted(p for p in method_dir.iterdir() if p.is_dir()):
            if not wanted(cell_dir.name):
                continue
            name = env_key(cell_dir.name)
            if name not in ENV_CONFIGS:
                if not analyse_suite_cell(method, cell_dir, contexts, cells,
                                          skipped, args):
                    skipped.append((method, cell_dir.name,
                                    'no run with a config to rebuild it from'))
                continue
            activation_fn = ACTIVATIONS[GYMNAX_POLICY_ACTIVATION]
            criterion = pick_criterion(args, GYMNAX_POLICY_ACTIVATION)
            if name not in policies:
                env, env_params = make_gymnax_env(name)
                obs_dim = int(_base_rollout(name, args.num_probe,
                                            args.probe_seed, probes).shape[-1])
                obs_dims[name] = obs_dim
                policy, template, _ = build_policy(
                    jax.random.key(0), obs_dim,
                    int(env.action_space(env_params).n),
                    ENV_CONFIGS[name]['hidden_dims'])
                policies[name] = (policy, template,
                                  len(ENV_CONFIGS[name]['hidden_dims']))
                num_actions[name] = int(env.action_space(env_params).n)
            policy, template, num_hidden = policies[name]

            trials = {}
            for trial_dir in sorted(cell_dir.glob('trial_*')):
                ckpt = trial_dir / 'checkpoints.npz'
                if not ckpt.exists():
                    continue
                # Only the fallback path in `subtask_sequence` needs this;
                # runs that store their sub-task arrays ignore it.
                cfg = run_config(trial_dir) or {}

                def probe_fn(blob, T, _n, _name=name, _cfg=cfg,
                             _dim=obs_dims[name]):
                    offsets, mults = subtask_sequence(_name, blob, _cfg, T, _dim)
                    if offsets is None:
                        return None
                    return subtask_probes(_name, offsets, mults, args.num_probe,
                                          args.probe_seed, args.probe, probes)

                res = analyse_trial(ckpt, template, policy, num_hidden,
                                    activation_fn, criterion, probe_fn,
                                    args.tau, not args.no_curvature,
                                    AGENT_SOURCES[args.agent],
                                    num_actions=(num_actions[name]
                                                 if args.entropy_frac else None),
                                    entropy_frac=args.entropy_frac)
                if res is None:
                    skipped.append((method, cell_dir.name,
                                    f'{trial_dir.name}: unreadable checkpoints'))
                    continue
                trials[trial_dir.name] = res
            if not trials:
                continue
            cells.setdefault(cell_dir.name, {})[method] = {
                'lineage_jumps': method in LINEAGE_JUMPS,
                'trials': trials,
            }
            # A run written before the `centroid` key existed silently falls
            # back to `incumbent`, which for the GA and DNS is a DIFFERENT
            # network -- `archive[0]` and the repertoire best, not any mean.
            # Naming it here is the difference between a mixed figure and a
            # figure that lies about what it draws.
            # `final` is not a fallback: the RL arms have one policy and that
            # is its only name under either mode. Anything else IS one.
            fell_back = sorted({t['agent_source'] for t in trials.values()}
                               - {AGENT_SOURCES[args.agent][0], 'final'})
            note = f'  [fell back to {"/".join(fell_back)}]' if fell_back else ''
            print(f'  {method:12s} {cell_dir.name:28s} {len(trials)} trials, '
                  f'{trials[next(iter(trials))]["num_checkpoints"]} checkpoints'
                  f'{note}')

    if not cells:
        sys.exit(f'no runs with checkpoints matched under {root}')

    out_dir = pathlib.Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / 'plasticity_checkpoints.json'
    activations = sorted({ctx.activation for ctx in contexts.values()}
                         | ({GYMNAX_POLICY_ACTIVATION} if policies else set()))
    out_path.write_text(json.dumps({
        'meta': {
            'runs_root': str(args.runs_root),
            'phase': args.phase,
            'sigma': args.sigma,
            'cells': args.cells,
            'tau': args.tau,
            'criterion': [pick_criterion(args, a) for a in activations],
            'activation': activations,
            'num_probe': args.num_probe,
            'probe_seed': args.probe_seed,
            'probe': args.probe,
            'probe_description': PROBE_DESCRIPTIONS[args.probe],
            'agent': args.agent,
            'entropy_frac': args.entropy_frac,
            'policy_kl': (None if not args.entropy_frac else
                          'KL(pi_t-1 || pi_t) on checkpoint t\'s probe, both '
                          'temperatures calibrated to mean entropy '
                          f'{args.entropy_frac} x log|A|; one sub-task per '
                          'gap, NOT the per-update policy_churn column'),
            'agent_sources': list(AGENT_SOURCES[args.agent]),
            'curvature': (None if args.no_curvature else
                          'empirical Fisher of log pi(argmax|s), '
                          'Lewandowski et al. 2024 sec 4.1'),
            'resolution': 'one checkpoint per sub-task',
        },
        'cells': cells,
    }, indent=1))
    for problem in check_agent_sources(cells, args.agent):
        print(f'WARNING: {problem}')
    for method, cell, why in skipped:
        print(f'WARNING: skipped {method}/{cell}: {why}')
    print(f'wrote {out_path}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
