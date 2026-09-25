"""Train NES on a gymnax task or on a switching sequence of them.

Four schedules, all on the same compute budget so their curves can be put on
one axis:

    task0    every generation scored on sub-task 0 (the specialist control)
    task1    every generation scored on sub-task 1 (the other specialist)
    switch   the sub-task alternates every ``task_interval`` generations
    joint    every generation scored on the *mean* return over both sub-tasks

``switch`` is the condition under test -- the setting the toy rugged-landscape
result says should let the search cross between peaks. ``joint`` is not a
competitor to it but a reference: it optimises the generalist objective
directly, so if ``joint`` cannot find a genome that is good on both, then no
generalist basin is reachable from here and ``switch``'s failure would say
nothing about switching. The two specialists bound the other end -- what a run
that never sees the other task achieves on it.

Whatever the schedule, every generation the *centroid* is scored on both
sub-tasks with held-out keys. Those two numbers are the run's actual result;
the training fitness only says how the currently-scored task is going.
"""

from __future__ import annotations

import argparse
import os
import time

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
from jax import random

from source.envs.registry import ENV_NAMES, get_suite, suite_for
from source.utils.run_artifacts import save_checkpoints
from source.studies.generalists.ne import (
    DNS_METHODS, GA_METHODS, NE_METHODS, build_searcher,
)
from source.studies.generalists.common import (
    SCHEDULES, make_phase_grid, make_task_sequence, record_centroid_scores,
    record_scores, summarise_records, write_run)
from source.studies.generalists import actors
from source.metrics.population_diversity import (
    diversity_columns, make_pairwise_behaviour_fn, probe_subsample)


def aggregate_over_tasks(per_task, objective, threshold, worst_k=None):
    """Reduce ``(num_tasks, pop)`` per-sub-task scores to one fitness per member.

    Which reduction is used decides what "generalist" means, and the choice is
    not cosmetic -- these do not have the same optimum:

        mean     the average return. What `joint` has always used. It trades
                 BREADTH for DEPTH: on a task set whose returns are bimodal
                 (CartPole is 500 or ~9, nothing between) a member that solves
                 eight sub-tasks outright scores the same as one that is
                 mediocre on all ten, and pushing a solved sub-task from 490 to
                 500 counts exactly as much as rescuing a failed one from 9.
        capped   ``mean_t min(return_t, threshold)``. Once a sub-task is at
                 threshold it contributes nothing further, so the ONLY way to
                 improve is to solve a sub-task that is not yet solved. This is
                 "solve as many as possible" written as a fitness.
        min      the worst sub-task -- the generalist score itself. This is
                 the PERFECT-GENERALIST objective and it is the wrong one for
                 "solve as many as possible". On bimodal returns it actively
                 prefers mediocrity: a member scoring 300 on all ten sub-tasks
                 (breadth ZERO, since 300 < 475) beats one that solves eight
                 outright and fails two, 300.0 against 9.0. Use it only when
                 the goal really is to lift the worst case, and expect it to
                 walk away from sub-tasks it had already solved.
        worstk   the mean of the ``worst_k`` sub-tasks. Interpolates between
                 `min` (k=1) and `mean` (k=num_tasks), which is the usual way
                 to keep a worst-case objective from being all noise.

    `capped` and `worstk` compose into `capped_worstk`, but capping does NOT
    rescue the worst-case family from the trap above -- capped_worstk scores the
    same two members 164.3 against 300.0, still backwards. Capping bounds the
    reward for depth; it does not stop `min` from preferring a uniformly
    mediocre point to a partially solved one.

    So for "solve as many sub-tasks as possible" the objective is `capped`, and
    the check that it is the right one is that it ranks these correctly:

        solves 8, fails 2 at 9      -> 381.8
        solves 8, fails 2 at 400    -> 460.0   (closer to solving all ten)
        solves 9, fails 1 at 9      -> 428.4   (more sub-tasks solved wins)
        300 on all ten              -> 300.0   (breadth zero, ranked last)
    """
    if objective == 'mean':
        return per_task.mean(axis=0)
    if objective == 'min':
        return per_task.min(axis=0)
    # `capped` and `capped_worstk` are defined in terms of a solved threshold,
    # so they do not exist on a suite that has none -- CheetahRun and the ant.
    # Refused rather than silently substituted: an uncapped `worstk` is a
    # different objective, and a run recording `objective: capped` that did not
    # cap is unreadable afterwards.
    # `worstk` is above the threshold guard because it does not use one: it is
    # the mean of the k worst sub-tasks, capped by nothing. Same arithmetic as
    # before this was hoisted -- `top_k` of the negative is the k smallest.
    if objective == 'worstk':
        k = max(1, min(int(worst_k or 1), per_task.shape[0]))
        return (-jax.lax.top_k(-per_task.T, k)[0]).mean(axis=-1)
    if threshold is None:
        raise ValueError(
            f"objective {objective!r} is defined in terms of a solved "
            f"threshold and this environment has none; use 'mean', 'min' or "
            f"'worstk'")
    capped = jnp.minimum(per_task, threshold)
    if objective == 'capped':
        return capped.mean(axis=0)
    if objective == 'capped_worstk':
        k = max(1, min(int(worst_k or 1), capped.shape[0]))
        # top_k of the negative is the k smallest, i.e. the k worst sub-tasks.
        return (-jax.lax.top_k(-capped.T, k)[0]).mean(axis=-1)
    raise ValueError(f'unknown objective {objective!r}')


def searcher_sigma(state, es=None):
    """The searcher's current mutation scale, for logging.

    NES keeps per-coordinate log sigmas, GA a scalar, DNS none at all. Logged
    under one key so the training_metrics schema does not fork per method.

    Under the isoline operator the answer is the ISO width, not ``state.sigma``:
    `GAState.sigma` still exists there but nothing reads it, so reporting it
    would put the gaussian width of a run that never mutated gaussianly into
    the column a reader takes as "how far this search is stepping". The isoline
    step also has a second, larger term proportional to the archive's own
    spread, which no scalar here can express -- so read this as the FLOOR of
    the step, and take the spread-driven part from the population diagnostics.
    """
    # `ga_track` moves its gaussian width every generation, so the configured
    # value in `variation_params` would log a constant it stopped using.
    if getattr(es, 'sigma_rule', 'decay') in ('track', 'success', 'subspace',
                                              'merge') \
            and hasattr(state, 'sigma'):
        return float(state.sigma)
    # The operator's own width wins where there is one, which also fills this
    # column for DNS -- it has no `state.sigma` at all and used to log NaN.
    params = getattr(es, 'variation_params', None) if es is not None else None
    if params:
        for key in ('iso_sigma', 'sigma'):
            if key in params:
                return float(params[key])
    if hasattr(state, 'log_sigma'):
        return float(jnp.exp(jnp.mean(state.log_sigma)))
    if hasattr(state, 'sigma'):
        return float(state.sigma)
    return float('nan')


def searcher_resolved(es):
    """Method settings the searcher decided for itself, for the run config.

    Only what a default could silently change. Empty for searchers with no such
    setting, which is the honest answer rather than a padded dict.
    """
    out = {}
    for name in ('refresh', 'num_elites', 'num_offspring',
                 'repertoire_size', 'batch_size'):
        if hasattr(es, name):
            value = getattr(es, name)
            out[name] = bool(value) if name == 'refresh' else int(value)
    # Which operator bred the offspring, and at what widths. Recorded because
    # `ga` and `ga_isoline` (and `dns` and `dns_gaussian`) are the same class
    # with one string changed, so nothing else in results.json would say which
    # of them a directory holds.
    if hasattr(es, 'variation'):
        out['variation'] = str(es.variation)
        out.update({k: float(v)
                    for k, v in getattr(es, 'variation_params', {}).items()})
    return out


def _resume_save(path, payload):
    """Write the boundary checkpoint atomically: tmp file, then os.replace.

    A SLURM kill in the middle of the write would otherwise leave a truncated
    pickle that the next segment loads and dies on -- and that segment is the
    one whose whole job was to recover from exactly such a kill.
    """
    import pickle, os
    tmp = path + '.tmp'
    with open(tmp, 'wb') as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)


class _PRNGKeyData:
    """A typed PRNG key's raw data inside a pickled state. A plain class, so
    `tree_map` treats it as a leaf (a tuple or dict would be descended into)."""

    def __init__(self, data):
        self.data = data


def _to_host(x):
    """jax -> numpy for pickling. PRNG keys are TYPED arrays that np.asarray
    rejects, so they travel as raw key data: at the top level as
    ('__prng__', data), and INSIDE a state as `_PRNGKeyData` leaves. Only the
    top-level case was handled, and every `ga_focus_*` Kinetix20 chain -- whose
    MergeGAState carries a key -- died at its first checkpoint (2026-09-14)."""
    import jax, numpy as np

    def is_key(v):
        return (isinstance(v, jax.Array)
                and jax.dtypes.issubdtype(v.dtype, jax.dtypes.prng_key))

    if is_key(x):
        return ('__prng__', np.asarray(jax.random.key_data(x)))
    return jax.tree_util.tree_map(
        lambda v: (_PRNGKeyData(np.asarray(jax.random.key_data(v))) if is_key(v)
                   else np.asarray(v)), x)


def _from_host(x):
    import jax, jax.numpy as jnp
    if isinstance(x, tuple) and len(x) == 2 and isinstance(x[0], str) \
            and x[0] == '__prng__':
        return jax.random.wrap_key_data(jnp.asarray(x[1]))
    return jax.tree_util.tree_map(
        lambda v: (jax.random.wrap_key_data(jnp.asarray(v.data))
                   if isinstance(v, _PRNGKeyData) else jnp.asarray(v)),
        x, is_leaf=lambda v: isinstance(v, _PRNGKeyData))


def run_nes(env_name='CartPole-v1', schedule='switch', num_generations=2000,
            method='nes',
            task_interval=100, task_warmup=0,
            pop_size=512, sigma=0.1, learning_rate=0.05,
            optimizer='sgd', shaping='zscore', sigma_lr=0.0, num_evals=1,
            noise_range=None, num_tasks=2, first_task_clean=True,
            pool_size=None, pair_repeats=5, expand_every=25,
            eval_episodes=16, seed=42, trial=1, output_dir=None,
            log_interval=50, checkpoint_interval=1,
            track_population=False, population_interval=20,
            track_members=False, population_snapshot_interval=0,
            snapshot_members=128,
            objective='mean', worst_k=1, tasks_per_batch=None,
            searcher_kwargs=None, task_options=None,
            obs_norm=False,
            resume_path=None, checkpoint_every=0, max_gens_this_run=0,
            track_plasticity=False, plasticity_interval=10,
            plasticity_activation='tanh', plasticity_continuous=True,
            num_probe_states=512,
            track_diversity=True,
            episode_length=None):
    """Run one NES search and write its trajectory and metrics to disk.

    ``task_options`` is the suite's per-environment sub-task settings from the
    command line (``--task_options KEY=VALUE``): on the ant, the friction
    order and bounds and the reward's target speed. gymnax takes none.

    Returns the record dict that is also saved as ``results.json``.
    """
    if schedule not in SCHEDULES:
        raise ValueError(f"schedule must be one of {SCHEDULES}, got {schedule!r}")
    # Which task family this environment belongs to, and the module that knows
    # how to build it. `source/envs/registry.py` is the whole of the
    # difference between gymnax and mjx as far as this file is concerned; a
    # gymnax run through it is bit-identical to one through the inline code it
    # replaced, because the key stream below is untouched.
    suite = get_suite(suite_for(env_name))
    cfg = suite.env_configs[env_name]
    # An explicit episode_length overrides the suite table, so a run can be
    # made at a shorter scan without editing the table. None keeps the
    # recorded value, so unflagged runs are unchanged.
    episode_length = episode_length or cfg['episode_length']
    # Read here rather than just before the results block: the `capped` and
    # `worstk` objectives need it inside the jitted generation steps, which are
    # defined well above that point. It is None on a suite with no solved
    # threshold -- CheetahRun and the ant -- where those two objectives are not
    # available; `aggregate_over_tasks` is what refuses them.
    threshold = cfg['solved_threshold']
    hidden_dims = cfg['hidden_dims']
    # Per-environment default, so `--noise_range` only has to be passed when
    # departing from what that body's own runs were made at. Sigma does not
    # transfer between bodies; see source/envs/mjx.py.
    if noise_range is None:
        noise_range = cfg.get('noise_range', 1.0)

    env, env_params, obs_dim, action_dim = suite.make_env(
        env_name, episode_length, task_options)

    key = random.key(seed)
    key, init_key = random.split(key)
    # `arch` is whatever the environment's config says the policy takes
    # beyond its hidden widths -- the conv policy's conv width -- and is
    # empty everywhere else, where this call is what it always was.
    arch = dict(cfg.get('arch', {}))
    policy, param_template, num_params = suite.build_policy(
        init_key, obs_dim, action_dim, hidden_dims, **arch)

    # One row per sub-task. An observation offset everywhere except the ant,
    # where it is a friction multiplier -- the suite decides, and every
    # scoring function below takes a row and applies it (see tasks_mjx.py).
    # Still called `noise_vectors` here and in the artifacts, because every
    # reader of `trajectory.npz` and `results.json` knows the key by that name.
    noise_vectors = suite.task_vectors(
        env_params, trial, num_tasks, obs_dim, noise_range, first_task_clean)

    _wk = {'whiten': True} if obs_norm else {}
    score = suite.make_scoring_fn(env, env_params, policy, param_template,
                                  episode_length, num_evals, **_wk)
    # The centroid's report card. More episodes than training uses, because a
    # single point is cheap and this number is the result.
    eval_score = suite.make_scoring_fn(env, env_params, policy, param_template,
                                       episode_length, eval_episodes, **_wk)

    # One searcher interface for NES / OpenES / GA / DNS; see source/studies/generalists/ne.py.
    # Only DNS asks for behaviour descriptors, and only then is the more
    # expensive rollout used.
    # `searcher_kwargs` carries method-specific knobs (GA's elite_ratio, DNS's
    # iso/line sigmas) that have no counterpart in this signature. build_searcher
    # filters to what each searcher accepts, so passing all of them is safe.
    # AURORA is the reference's default descriptor for DNS and the one its
    # published gymnax numbers used: the descriptor is the latent code of an
    # LSTM auto-encoder trained online on the population's own observation
    # trajectories, so the descriptor SPACE moves during the run. The
    # hand-crafted alternative stays available (`descriptor='handcrafted'`)
    # because it is two interpretable numbers and a useful ablation, but it is
    # no longer what runs by default.
    kw = dict(searcher_kwargs or {})
    # `dns_gaussian` is DNS too -- it swaps the variation operator, not the
    # selection rule -- so it needs descriptors and AURORA like the rest.
    is_dns = method in DNS_METHODS
    descriptor = kw.pop('descriptor', 'aurora' if is_dns else None)
    traj_steps = int(kw.pop('traj_steps', 50))
    aurora_cfg = {k: kw.pop(f'aurora_{k}') for k in
                  ('latent_dim', 'lr', 'train_ratio', 'batch_size')
                  if f'aurora_{k}' in kw}
    use_aurora = is_dns and descriptor == 'aurora'
    latent_dim = int(aurora_cfg.get('latent_dim', 6))
    # What one row of `make_trajectory_scoring_fn`'s output is WIDE, which is
    # the observation width on every suite whose trajectories are observations
    # and 13 on kinetix, where they are per-step behaviour features instead --
    # an LSTM auto-encoder over 125x125x3 frames would cost more than the
    # search it serves. `Suite.traj_feature_dim` returns `obs_dim` unless the
    # suite says otherwise, so this is unchanged everywhere else.
    traj_dim = suite.traj_feature_dim(env_name, obs_dim)

    es = build_searcher(method, num_params, pop_size, sigma_init=sigma,
                        learning_rate=learning_rate, optimizer=optimizer,
                        shaping=shaping, sigma_lr=sigma_lr,
                        descriptor_dim=(latent_dim if use_aurora
                                        else suite.descriptor_dim(env_name)),
                        traj_steps=(min(traj_steps, episode_length)
                                    if use_aurora else 0),
                        obs_dim=traj_dim if use_aurora else 0,
                        **kw)

    aurora = aurora_state = None
    aurora_schedule = set()
    if use_aurora:
        from source.metrics.aurora import (AuroraDescriptors,
                                          aurora_training_schedule)
        score = suite.make_trajectory_scoring_fn(
            env, env_params, policy, param_template, episode_length, num_evals,
            traj_steps, **_wk)
        aurora = AuroraDescriptors(
            obs_size=traj_dim, traj_steps=min(traj_steps, episode_length),
            latent_dim=latent_dim,
            learning_rate=float(aurora_cfg.get('lr', 1e-3)),
            batch_size=int(aurora_cfg.get('batch_size', 128)))
        key, aurora_key = random.split(key)
        aurora_state = aurora.init(aurora_key)
        aurora_schedule = aurora_training_schedule(
            num_generations, int(aurora_cfg.get('train_ratio', 8)))
    elif es.needs_descriptors:
        score = suite.make_descriptor_scoring_fn(
            env, env_params, policy, param_template, episode_length, num_evals,
            env_name, **_wk)

    key, mean_key = random.split(key)
    # Start from the policy's own initialisation rather than zeros: a zero MLP
    # has zero gradient signal through the ReLUs and every perturbation of it
    # is symmetric, so the first generations would be wasted breaking that.
    # ---- plasticity, the gymnax columns on this body ----------------------
    # OFF BY DEFAULT so every run made before this existed is reproduced
    # bit-for-bit; the brax blocks turn it on. It is an OBSERVER -- its own RNG
    # stream, never fed back into the search -- which is the rule everything
    # under source/metrics/ is held to.
    #
    # Without it ant and cheetah have no ne_centroid_*/ne_elite_* columns at
    # all, so the per-generation half of the centroid plasticity figure cannot
    # be drawn for them while gymnax has it. `checkpoints.npz['centroid']` was
    # already saved on both, so only the curve was missing.
    plast = None
    if track_plasticity:
        from source.metrics import plasticity as _plast_mod
        from source.algorithms.networks import unflatten_params
        from source.algorithms.rl import redo as _redo_mod
        _ACT = {'tanh': jnp.tanh, 'relu': jax.nn.relu}
        _unflatten = lambda f: unflatten_params(f, param_template)
        plast = _plast_mod.NEPlasticityTracker(
            apply_flat=lambda f, o: policy.apply(_unflatten(f), o),
            unflatten=_unflatten,
            num_hidden=len(hidden_dims),
            activation_fn=_ACT[plasticity_activation],
            criterion=_redo_mod.criterion_for_activation(plasticity_activation),
            num_probe=num_probe_states,
            continuous=plasticity_continuous,
        )
        # The probe batch has to come from somewhere, and the default mjx
        # scoring function returns fitness only. This is the AURORA path's
        # trajectory scorer at num_evals=1, called ONCE at generation 0 on a
        # slice of the population -- cheap, and the observations are the ones
        # the search actually visited rather than a synthetic batch.
        plast_probe_score = suite.make_trajectory_scoring_fn(
            env, env_params, policy, param_template, episode_length, 1, **_wk)
        plasticity_key = random.key(seed + 3_000_000)

    # ---- population diversity: the three columns every population arm logs
    # `bd_genomic_diversity`, `bd_fitness_std`, `bd_behavioural_diversity`
    # (source/metrics/population_diversity.py; PBT writes the same three),
    # per generation, on the population the generation scored. An observer
    # on its own RNG stream: the search is the same run with it on or off.
    # The behavioural column needs a frozen probe batch of the policy's
    # INPUTS, taken once at generation 0 from the trajectory scorer exactly
    # as the plasticity tracker takes its own; a body whose trace does not
    # carry the policy input (Kinetix keeps 13 step features, not frames)
    # logs the other two only, and says so once.
    bd_distance = bd_outputs = bd_probe_score = bd_probe_obs = None
    bd_key = random.key(seed + 5_000_000)
    if track_diversity:
        from source.algorithms.networks import unflatten_params as _bd_unflat
        _bd_head = actors.head_for(suite_for(env_name), suite.action_dims(env))
        bd_distance = make_pairwise_behaviour_fn(_bd_head.name,
                                                 suite.action_dims(env))
        bd_outputs = jax.jit(lambda flats, probe: jax.vmap(
            lambda f: policy.apply(_bd_unflat(f, param_template), probe))(flats))
        bd_probe_score = suite.make_trajectory_scoring_fn(
            env, env_params, policy, param_template, episode_length, 1, **_wk)

    from source.algorithms.networks import get_flat_params
    key, searcher_key = random.split(key)
    state = es.init(searcher_key,
                    get_flat_params(policy.init(mean_key, jnp.zeros((obs_dim,)))))

    if use_aurora:
        # Evaluate the initial repertoire and fit the encoder to it before the
        # first generation, as the reference does. The reason that survives
        # `refresh` is the encoder: one that has never seen this environment's
        # observations produces descriptors of nothing in particular, and the
        # first selections would be run on those. The other reason applies to
        # `dns_stale` only -- there a repertoire starting at fitness -inf is
        # INVALID for dominated novelty and, with fewer offspring than members,
        # cannot be flushed in one generation, so the invalid members linger
        # and shrink the effective repertoire. Under `refresh` the repertoire
        # is re-scored on generation 0 anyway and the stored values below are
        # overwritten before anything reads them.
        key, init_eval_key, init_ae_key = random.split(key, 3)
        init_fitness, init_obs = score(state.repertoire, init_eval_key,
                                       noise_vectors[0])
        aurora_state, _ = aurora.train(init_ae_key, init_obs, aurora_state,
                                       iteration=0)
        state = state._replace(fitness=init_fitness, observations=init_obs,
                               descriptors=aurora.encode(init_obs, aurora_state))

    @jax.jit
    def generation_step_aurora(key, state, noise_vector, aurora_state):
        """DNS with a learned descriptor space.

        The encoder is a jit argument rather than a closure because it is
        retrained during the run: closing over it would bake generation 0's
        parameters into the compiled step and every later descriptor would come
        from a stale encoder, silently.
        """
        ask_key, score_key = random.split(key)
        offspring, _ = es.ask(ask_key, state)
        fitness, observations = score(offspring, score_key, noise_vector)
        descs = aurora.encode(observations, aurora_state)
        return es.tell(state, offspring, fitness, descs, observations), fitness

    @jax.jit
    def reencode_repertoire(state, aurora_state):
        """Re-descriptor the whole repertoire under a freshly trained encoder.

        Dominated novelty is a distance between descriptors, so mixing codes
        from two different encoders compares nothing. The reference re-encodes
        everything stored the moment it retrains, and so does this.
        """
        return es.reencode(
            state, aurora.encode(state.observations, aurora_state))

    @jax.jit
    def generation_step(key, state, noise_vector):
        ask_key, score_key = random.split(key)
        population, aux = es.ask(ask_key, state)
        if es.needs_descriptors:
            fitness, descs = score(population, score_key, noise_vector)
            return es.tell(state, population, fitness, descs), fitness
        fitness = score(population, score_key, noise_vector)
        # GA selects over the population it scored, so it needs the genomes;
        # NES needs the perturbations. `aux` carries whichever the searcher
        # returned, and GA is handed the population explicitly.
        carried = population if aux is None else aux
        return es.tell(state, carried, fitness), fitness

    @jax.jit
    def generation_step_joint(key, state, all_noise_vectors,
                              aurora_state=None):
        ask_key, score_key = random.split(key)
        population, aux = es.ask(ask_key, state)
        # Score the same population on every sub-task with the same key, so the
        # comparison between sub-tasks is not confounded by the resets, then
        # average. This *is* the generalist objective.
        if use_aurora:
            fits, obs = jax.lax.map(lambda nv: score(population, score_key, nv),
                                    all_noise_vectors)
            # Fitness is averaged over sub-tasks -- that is the generalist
            # objective -- but the trajectory is sub-task 0's rather than an
            # average of trajectories from different sub-tasks, which would
            # describe no episode that ever happened and is not something the
            # auto-encoder was trained to reconstruct.
            mean_fitness = fits.mean(axis=0)
            return (es.tell(state, population, mean_fitness,
                            aurora.encode(obs[0], aurora_state), obs[0]),
                    mean_fitness)
        if es.needs_descriptors:
            scored = jax.lax.map(lambda nv: score(population, score_key, nv),
                                 all_noise_vectors)
            mean_fitness = scored[0].mean(axis=0)
            return (es.tell(state, population, mean_fitness,
                            scored[1].mean(axis=0)), mean_fitness)
        # Mapped, not vmapped, over the sub-task rows: see `eval_member`.
        per_task = jax.lax.map(lambda nv: score(population, score_key, nv),
                               all_noise_vectors)
        mean_fitness = aggregate_over_tasks(per_task, objective, threshold,
                                            worst_k)
        carried = population if aux is None else aux
        return es.tell(state, carried, mean_fitness), mean_fitness

    @jax.jit
    def generation_step_batch(key, state, batch_noise_vectors):
        """Score every member on a SUBSET of sub-tasks, then aggregate.

        Sits between `switch` (one sub-task per generation) and `joint` (all of
        them). With `objective='mean'` the expected search gradient is the joint
        gradient at a fraction of the cost, which is ordinary mini-batching;
        with a breadth objective it is mini-batching on the quantity actually
        wanted. Cost per generation is proportional to the batch size, so k=2 on
        ten sub-tasks is a fifth of what `joint` spends per step.
        """
        ask_key, score_key = random.split(key)
        population, aux = es.ask(ask_key, state)
        per_task = jax.lax.map(lambda nv: score(population, score_key, nv),
                               batch_noise_vectors)
        fitness = aggregate_over_tasks(per_task, objective, threshold, worst_k)
        carried = population if aux is None else aux
        return es.tell(state, carried, fitness), fitness

    # ---- the population that a generation actually evaluated ---------------
    # Every `generation_step*` above opens with `ask_key, score_key =
    # random.split(key)` and then `es.ask(ask_key, state)`. Handed the SAME
    # step key and the SAME pre-step state, this reproduces that population
    # exactly -- so the snapshot is the cloud that was scored, not a fresh
    # sample from the same distribution. Nothing in the step functions has to
    # change to get it, and the sampling is cheap next to the rollouts.
    @jax.jit
    def population_of(key, state):
        ask_key, _ = random.split(key)
        population, _ = es.ask(ask_key, state)
        return population

    # ---- the best sampled member, which is what the reference reports -------
    # `source/train_ES_gymnax_continual.py` logs
    # `max(mean_fitness)` and checkpoints `population[argmax(mean_fitness)]`:
    # its "agent" is the best member of the generation, not the distribution
    # mean. This project reports the mean, which for NES and OpenES is a
    # different policy and is not the one the earlier study's curves are about.
    # Both are logged here so the gap between them is readable rather than a
    # choice made upstream of the figure.
    # Every "score these weights on every sub-task" below maps over the
    # sub-task rows (`jax.lax.map`) instead of vmapping over them. A vmap
    # shares the weights across the row axis, which makes XLA compute the
    # policy with different kernels than the generation step (where the
    # weights are batched along the population axis); on Kinetix's pixel
    # policy the logits moved by up to 0.066, and the chaotic physics turned
    # that into different outcomes -- a robust car_thrust solver scored 1.28
    # one level per call and -0.08 in the 20-row vmap, and the chain's
    # per-level records were wrong in both directions for every NE arm
    # (2026-09-14, claude_probe/kx_kernel_batch.py). Mapping gives each row
    # exactly the numerics a stationary run of that sub-task has.
    # For ONE vector over the rows the weights are instead repeated along the
    # row axis and vmapped together with it: the policy then runs with the
    # weights batched, as in the generation step, so every row gets the
    # one-sub-task-per-call numerics while the rows still run in parallel
    # (the per-generation record would otherwise be num_tasks sequential
    # evaluations; a repeat of one vector is cheap). Checked equal to one call
    # per sub-task on all twenty Kinetix levels.
    @jax.jit
    def eval_member(key, member, all_noise_vectors):
        keys = random.split(key, all_noise_vectors.shape[0])
        rows = jnp.repeat(member[None, :], all_noise_vectors.shape[0], axis=0)
        return jax.vmap(lambda w, k, nv: eval_score(w[None, :], k, nv)[0])(
            rows, keys, all_noise_vectors)

    @jax.jit
    def eval_centroid(key, mean, all_noise_vectors):
        keys = random.split(key, all_noise_vectors.shape[0])
        rows = jnp.repeat(mean[None, :], all_noise_vectors.shape[0], axis=0)
        return jax.vmap(lambda w, k, nv: eval_score(w[None, :], k, nv)[0])(
            rows, keys, all_noise_vectors)

    # ---- population coverage -------------------------------------------
    # Does the search hold ONE genome that is good at every sub-task, or a SET
    # of genomes each good at one? Both look like progress if you only watch
    # per-task performance, and they are opposite outcomes for this project:
    # the second is a population of specialists with no generalist in it.
    #
    # Scored on the persistent population -- GA's elite archive, DNS's
    # repertoire -- not on the transient sample a generation was bred from.
    # NES and OpenES have no such population (`has_population = False`); their
    # centroid is scored instead, which is the honest answer for a (1, lambda)
    # method and makes the specialist_gap identically zero for them by
    # construction rather than by measurement.
    @jax.jit
    def score_population_all_tasks(key, members, all_noise_vectors):
        keys = random.split(key, all_noise_vectors.shape[0])
        return jax.lax.map(lambda a: eval_score(members, a[0], a[1]),
                           (keys, all_noise_vectors))   # (num_tasks, num_members)

    def population_record(key, state, all_noise_vectors):
        members = es.population(state)
        scores = np.asarray(score_population_all_tasks(
            key, members, all_noise_vectors))            # (T, M)
        per_task_best = scores.max(axis=1)                # best specialist
        champions = scores.argmax(axis=1)                 # who each one is
        # The best SINGLE individual at being good everywhere.
        best_generalist = float(scores.min(axis=0).max())
        out = {
            'pop_size_scored': int(members.shape[0]),
            'pop_best_generalist': best_generalist,
            # How much the population as a whole beats its own best individual.
            # Large => the sub-tasks are covered by DIFFERENT genomes, i.e. a
            # population of specialists. Zero => one genome carries both.
            'specialist_gap': float(per_task_best.min() - best_generalist),
            'num_distinct_champions': int(len(set(champions.tolist()))),
        }
        for t in range(all_noise_vectors.shape[0]):
            out[f'pop_best_task{t}'] = float(per_task_best[t])
            out[f'champion_task{t}'] = int(champions[t])
        if members.shape[0] > 1:
            # Genomic spread, on a subsample: the full pairwise matrix is
            # 512x512x386 and this runs inside the training loop.
            idx = np.linspace(0, members.shape[0] - 1,
                              min(32, members.shape[0])).astype(int)
            sub = np.asarray(members)[idx]
            d = np.linalg.norm(sub[:, None, :] - sub[None, :, :], axis=-1)
            iu = np.triu_indices(len(idx), k=1)
            out['genomic_diversity'] = float(d[iu].mean())
            # Distance between the two sub-tasks' champions: 0 when the same
            # genome wins both, large when they are different solutions.
            out['champion_distance'] = float(np.linalg.norm(
                np.asarray(members)[champions[0]]
                - np.asarray(members)[champions[-1]]))
        else:
            out['genomic_diversity'] = 0.0
            out['champion_distance'] = 0.0
        return out, np.asarray(members)[champions]

    phase_of_gen, num_phases = make_phase_grid(num_generations, task_interval,
                                               task_warmup)
    task_sequence = make_task_sequence(schedule, num_phases, num_tasks, trial,
                                       pool_size, pair_repeats, expand_every)
    # The last generation of a phase, which is where the phase artifacts are
    # written. `task_interval` arithmetic cannot answer this once the first
    # phase has a different length.
    phase_end = lambda g: (g + 1 >= num_generations
                           or phase_of_gen[g + 1] != phase_of_gen[g])

    records = []
    centroids = []
    centroid_gens = []
    # One saved network per sub-task PHASE, for the centroid plasticity
    # figure. Written to `checkpoints.npz`, which is the only file
    # `scripts/analysis/plasticity_checkpoints.py` reads; `trajectory.npz`'s
    # `centroids` is a different thing -- it is `incumbent`, sampled on
    # `checkpoint_interval`, and on the GA and DNS `incumbent` is NOT the
    # network the centroid lineplot scores. See save_checkpoints.
    ckpt_finalgen = []
    ckpt_incumbent = []
    ckpt_centroid = []
    ckpt_phase_tasks = []
    population_snapshots = []
    population_snapshot_gens = []
    population_snapshot_tasks = []
    champion_genomes = []
    champion_gens = []
    start = time.time()

    # ---- checkpoint-restart, at LEVEL BOUNDARIES ---------------------------
    # For the runs a 12 h wall-clock cap cannot hold: the Kinetix continual
    # chain is 20 levels x 200 generations and measured ~96 min a level on
    # GH200, ~32 h end to end, and the runners had no resume -- a kill lost
    # the chain, because every artifact is written at the end. So a segment
    # saves EVERYTHING the loop carries at a boundary, exits cleanly after
    # `max_gens_this_run` generations, and the next SLURM job picks it up.
    #
    # What is saved is the loop-carried state and nothing derived: the RNG
    # key, the searcher state, AURORA's state for DNS, every accumulator the
    # finalisation block reads, and the elapsed time so far -- so the
    # results.json a resumed run writes is the one an unbroken run would
    # have written. Deleted on a clean finish, so a finished trial never
    # carries a stale one.
    start_gen = 0
    elapsed_before = 0.0
    if resume_path and os.path.exists(resume_path):
        import pickle
        with open(resume_path, 'rb') as f:
            ck = pickle.load(f)
        start_gen = int(ck['gen'])
        key = _from_host(ck['key'])
        state = _from_host(ck['state'])
        if ck.get('aurora_state') is not None:
            aurora_state = _from_host(ck['aurora_state'])
        records, centroids, centroid_gens = ck['records'], ck['centroids'], ck['centroid_gens']
        ckpt_finalgen, ckpt_incumbent = ck['ckpt_finalgen'], ck['ckpt_incumbent']
        ckpt_centroid, ckpt_phase_tasks = ck['ckpt_centroid'], ck['ckpt_phase_tasks']
        population_snapshots = ck['population_snapshots']
        population_snapshot_gens = ck['population_snapshot_gens']
        population_snapshot_tasks = ck['population_snapshot_tasks']
        champion_genomes, champion_gens = ck['champion_genomes'], ck['champion_gens']
        elapsed_before = float(ck.get('elapsed_before', 0.0))
        print(f'  RESUMED from {resume_path} at generation {start_gen} '
              f'({len(records)} records, {elapsed_before/3600:.2f} h already spent)',
              flush=True)
    gens_this_run = 0

    for gen in range(start_gen, num_generations):
        task_idx = int(task_sequence[phase_of_gen[gen]])

        # Save at a boundary BEFORE this generation runs, so `gen` is the
        # first generation the next segment must execute.
        if (checkpoint_every and gen > start_gen and gen % checkpoint_every == 0):
            _resume_save(resume_path, dict(
                gen=gen, key=_to_host(key), state=_to_host(state),
                aurora_state=(None if aurora_state is None else _to_host(aurora_state)),
                records=records, centroids=centroids, centroid_gens=centroid_gens,
                ckpt_finalgen=ckpt_finalgen, ckpt_incumbent=ckpt_incumbent,
                ckpt_centroid=ckpt_centroid, ckpt_phase_tasks=ckpt_phase_tasks,
                population_snapshots=population_snapshots,
                population_snapshot_gens=population_snapshot_gens,
                population_snapshot_tasks=population_snapshot_tasks,
                champion_genomes=champion_genomes, champion_gens=champion_gens,
                elapsed_before=elapsed_before + (time.time() - start),
            ))
            print(f'  checkpoint: generation {gen} -> {resume_path}', flush=True)
            if max_gens_this_run and gens_this_run >= max_gens_this_run:
                print(f'  SEGMENT DONE: {gens_this_run} generations this run, '
                      f'stopping cleanly at generation {gen} of {num_generations}',
                      flush=True)
                return {'resumed_segment': True, 'stopped_at': gen}
        gens_this_run += 1
        key, step_key, eval_key = random.split(key, 3)
        # Kept because `population_of` needs the state as it was BEFORE `tell`
        # advanced it; after the step it would sample the next generation.
        pre_step_state = state

        if use_aurora and schedule != 'joint':
            state, fitness = generation_step_aurora(
                step_key, state, noise_vectors[task_idx], aurora_state)
            # Retrain on the SURVIVORS' trajectories, then re-encode every
            # stored descriptor -- both after selection, in that order, as the
            # reference does. `gen + 1` because its schedule counts completed
            # generations while this loop counts the one just run.
            if (gen + 1) in aurora_schedule:
                key, ae_key = random.split(key)
                aurora_state, aurora_loss = aurora.train(
                    ae_key, state.observations, aurora_state, iteration=gen)
                state = reencode_repertoire(state, aurora_state)
        elif tasks_per_batch:
            # A fresh draw of sub-tasks every generation. Seeded from the
            # generation index and the trial so the sequence is reproducible
            # and identical across methods at a given trial, the same
            # convention `sampled` uses for its phase draw.
            pick = np.random.default_rng(int(trial) * 104729 + gen).choice(
                num_tasks, size=min(tasks_per_batch, num_tasks), replace=False)
            state, fitness = generation_step_batch(
                step_key, state, noise_vectors[np.asarray(pick)])
            task_idx = int(pick[0])
        elif schedule == 'joint':
            state, fitness = generation_step_joint(
                step_key, state, noise_vectors, aurora_state)
            if use_aurora and (gen + 1) in aurora_schedule:
                key, ae_key = random.split(key)
                aurora_state, _ = aurora.train(
                    ae_key, state.observations, aurora_state, iteration=gen)
                state = reencode_repertoire(state, aurora_state)
        else:
            state, fitness = generation_step(
                step_key, state, noise_vectors[task_idx])

        per_task = np.asarray(eval_centroid(eval_key, es.incumbent(state), noise_vectors))
        # The same protocol on the mean of the population's weights. On
        # NES/OpenES this is the identical point and the two curves coincide;
        # on GA and DNS it is a different genome and the gap is the reading.
        # Same key as the incumbent's evaluation, so the two are scored on the
        # identical episodes and their difference is the genome alone.
        popmean_per_task = np.asarray(
            eval_centroid(eval_key, es.population_mean(state), noise_vectors))
        record = {
            'generation': gen,
            'task': int(task_idx),
            'train_fitness_mean': float(jnp.mean(fitness)),
            'train_fitness_max': float(jnp.max(fitness)),
            'sigma': float(searcher_sigma(state, es)),
        }
        # WHICH NETWORK IS CALLED `centroid_*`. Until 2026-09-09 the INCUMBENT
        # went out under the `centroid` prefix and the population mean under
        # `popmean`, which is the exact trap this file's checkpoint block
        # documents and avoids ("Saving `incumbent` under the name `centroid`
        # is what made the gymnax centroid figure measure the elite"). It was
        # avoided for the checkpoints and not for the per-generation record.
        #
        # It matters because `ne_centroid_*` on the gymnax side IS the
        # coordinate-wise mean of the population, so one figure script reading
        # `centroid_*` across bodies compared a centroid on gymnax against a
        # best-elite on brax. Harmless for es/nes, where `incumbent` and
        # `population_mean` are the same point -- and wrong for exactly the two
        # arms the gaussian pair exists to separate, ga and dns_gaussian.
        #
        # So: `centroid_*` is the population mean, matching gymnax and matching
        # `ckpt_centroid` in this same loop, and the incumbent keeps its own
        # prefix rather than being dropped.
        record_scores(record, per_task, prefix='incumbent')
        record_centroid_scores(record, popmean_per_task)

        # The best member of this generation, on the same held-out protocol as
        # the centroid, so `best_generalist` and `centroid_generalist` differ
        # only in WHICH policy was scored.
        plast_now = (plast is not None
                     and (gen % plasticity_interval == 0
                          or gen == num_generations - 1))
        need_pop = (track_members or plast_now or track_diversity
                    or not getattr(es, 'has_population', True)
                    or (population_snapshot_interval
                        and gen % population_snapshot_interval == 0))
        population_now = (np.asarray(population_of(step_key, pre_step_state))
                          if need_pop else None)
        if track_diversity:
            if bd_probe_score is not None:
                # Once: the probe batch, or the reason there is none.
                try:
                    from source.metrics.behaviour_descriptors import \
                        collect_probe_states
                    _, _bd_obs = bd_probe_score(
                        jnp.asarray(population_now[:32]), bd_key,
                        noise_vectors[task_idx])
                    _probe = collect_probe_states(
                        _bd_obs, num_probe=num_probe_states, key=bd_key)
                    if int(_probe.shape[-1]) != int(obs_dim):
                        raise ValueError(
                            f'probe width {_probe.shape[-1]} is not the '
                            f'policy input {obs_dim}')
                    bd_probe_obs = _probe
                except Exception as exc:        # noqa: BLE001
                    # Kinetix: the trace keeps step features, not frames. The
                    # suite's random-policy frames on this sub-task stand in
                    # (the dormancy probe batch; a generation-0 population is
                    # close to a random policy anyway).
                    try:
                        bd_probe_obs = jnp.asarray(
                            suite.random_policy_observations(
                                env, env_params,
                                jnp.asarray(noise_vectors[task_idx]),
                                num_probe_states, seed + 5_000_000))
                        print(f'  diversity: trace has no policy input '
                              f'({type(exc).__name__}); behavioural probe = '
                              f'{bd_probe_obs.shape[0]} random-policy '
                              'observations', flush=True)
                    except Exception as exc2:   # noqa: BLE001
                        print(f'  diversity: no behavioural probe on this body '
                              f'({type(exc2).__name__}: {exc2}); logging the '
                              'genomic and fitness spread only', flush=True)
                bd_probe_score = None
            _bd_out = (bd_outputs(jnp.asarray(probe_subsample(population_now)),
                                  bd_probe_obs)
                       if bd_probe_obs is not None else None)
            record.update(diversity_columns(population_now, np.asarray(fitness),
                                            _bd_out, bd_distance))
        if track_members:
            # Derived from `eval_key` rather than split off `key`. Splitting the
            # main stream here would make a run with `--track_members` a
            # DIFFERENT run from one without it, and a diagnostic that changes
            # the search it measures is not a diagnostic -- the same rule
            # everything under `source/metrics/` is held to. `fold_in` is
            # deterministic and independent of the per-sub-task keys
            # `eval_centroid` splits out of the same parent.
            member_key = random.fold_in(eval_key, 1)
            best_idx = int(jnp.argmax(fitness))
            best_member = population_now[best_idx]
            best_task = np.asarray(eval_member(
                member_key, jnp.asarray(best_member), noise_vectors))
            for t in range(num_tasks):
                record[f'best_task{t}'] = float(best_task[t])
            record['best_generalist'] = float(best_task.min())
            record['best_mean_over_tasks'] = float(best_task.mean())
        # THE ELITE: the best-performing agent, always (decided 2026-09-13).
        # On GA and DNS that is the incumbent scored above (the best archive
        # member). On NES/OpenES the incumbent is the distribution MEAN, the
        # same point as the centroid, so the elite is the best member this
        # generation sampled, re-scored on held-out episodes -- what the gymnax
        # trainers' `elite_eval_fitness` always was. The member key is folded
        # off `eval_key` exactly as `--track_members` does, so the search is
        # the same run with or without this column.
        if getattr(es, 'has_population', True):
            elite_per_task = per_task
        elif track_members:
            elite_per_task = best_task
        else:
            elite_per_task = np.asarray(eval_member(
                random.fold_in(eval_key, 1),
                jnp.asarray(population_now[int(jnp.argmax(fitness))]),
                noise_vectors))
        record_scores(record, elite_per_task, prefix='elite')

        if plast_now:
            # `centroid=` is population_mean, the same vector `centroid_*`
            # above is scored on, so the fitness column and the plasticity
            # columns describe ONE network. `incumbent=` is the distribution's
            # own point, which is what the ES/NES churn column means.
            if not plast.started():
                _, _probe_obs = plast_probe_score(
                    jnp.asarray(population_now[:32]), plasticity_key,
                    noise_vectors[task_idx])
                plast.start(plasticity_key, _probe_obs)
            plasticity_key, _pk = random.split(plasticity_key)
            record.update(plast.update(
                _pk, jnp.asarray(population_now),
                jnp.asarray(population_now[int(jnp.argmax(fitness))]),
                incumbent=es.incumbent(state),
                centroid=es.population_mean(state)))

        records.append(record)

        # ---- per-phase checkpoints ----------------------------------------
        # Taken at the phase's LAST generation, which is the network carried
        # across the boundary and the one whose zero-shot return on the next
        # sub-task is the transfer measurement.
        #
        # `centroid` is `population_mean`, deliberately not `incumbent`: for
        # NES and OpenES the two are the same point, but for the GA it is the
        # elite archive's mean against its best member and for DNS the
        # repertoire's mean against its best. Saving `incumbent` under the
        # name `centroid` is what made the gymnax centroid figure measure the
        # elite; it is the same trap here and this is where it is avoided.
        if phase_end(gen):
            # `population_of` re-derives the generation from the PRE-step
            # state and the same key, so it is a pure function of what the
            # search already did -- calling it here consumes no randomness and
            # cannot move the run. It is skipped entirely except at these
            # `num_phases` generations.
            pop_at_end = (population_now if population_now is not None
                          else np.asarray(population_of(step_key,
                                                        pre_step_state)))
            ckpt_finalgen.append(np.asarray(pop_at_end[int(jnp.argmax(fitness))]))
            ckpt_incumbent.append(np.asarray(es.incumbent(state)))
            ckpt_centroid.append(np.asarray(es.population_mean(state)))
            ckpt_phase_tasks.append(int(task_idx))

        # Population snapshots for the landscape animation: the cloud the
        # search was carrying at that moment, subsampled to `snapshot_members`
        # because storing 512 x num_params every few generations is hundreds of
        # megabytes a run and the figure draws a scatter, not a census.
        if population_now is not None and population_snapshot_interval and (
                gen % population_snapshot_interval == 0
                or gen == num_generations - 1):
            take = population_now
            if snapshot_members and take.shape[0] > snapshot_members:
                sel = np.linspace(0, take.shape[0] - 1,
                                  snapshot_members).astype(int)
                take = take[sel]
            population_snapshots.append(take.astype(np.float32))
            population_snapshot_gens.append(gen)
            population_snapshot_tasks.append(int(task_idx))

        if track_population and (gen % population_interval == 0
                                 or gen == num_generations - 1):
            key, pop_key = random.split(key)
            pop_rec, champs = population_record(pop_key, state, noise_vectors)
            record.update(pop_rec)
            # Saved so the ANALYSIS can ask whether the specialist that wins
            # sub-task 0 on this visit is the same genome that won it last
            # visit, or a freshly rediscovered one. That is a question about
            # identity across time, so identity has to be stored, not summarised.
            champion_genomes.append(champs)
            champion_gens.append(gen)

        if gen % checkpoint_interval == 0 or gen == num_generations - 1:
            centroids.append(np.asarray(es.incumbent(state)))
            centroid_gens.append(gen)

        if gen % log_interval == 0 or gen == num_generations - 1:
            per_task_str = ' '.join(f"t{t}={per_task[t]:7.1f}"
                                    for t in range(num_tasks))
            print(f"  gen {gen:5d} task={task_idx:2d} "
                  f"train={record['train_fitness_mean']:7.1f} "
                  f"{per_task_str} gen'ist={per_task.min():7.1f}", flush=True)

    elapsed = elapsed_before + (time.time() - start)

    result = {
        'config': {
            'env': env_name, 'method': method, 'schedule': schedule,
            'num_generations': num_generations, 'task_interval': task_interval,
            # 0 unless the run had a long first phase; see `make_phase_grid`.
            'task_warmup': int(task_warmup or 0),
            'pop_size': pop_size, 'sigma': sigma,
            'learning_rate': learning_rate, 'optimizer': optimizer,
            'shaping': shaping, 'sigma_lr': sigma_lr, 'num_evals': num_evals,
            'noise_range': noise_range, 'num_tasks': num_tasks,
            'first_task_clean': first_task_clean,
            'pool_size': pool_size, 'pair_repeats': pair_repeats,
            'expand_every': expand_every,
            'searcher_kwargs': dict(searcher_kwargs or {}),
            # What build_searcher RESOLVED, not what the caller passed.
            # `refresh` has a default that changed on 2026-08-27, so a
            # run recording only the caller's kwargs cannot be read back
            # later: `{'elite_ratio': 0.5}` means two different
            # algorithms either side of that date.
            'searcher_resolved': searcher_resolved(es),
            'objective': objective, 'worst_k': worst_k,
            'tasks_per_batch': tasks_per_batch,
            'track_population': track_population,
            'population_interval': population_interval,
            'descriptor': descriptor, 'traj_steps': traj_steps,
            'aurora': dict(aurora_cfg) if use_aurora else None,

            'task_sequence': task_sequence.tolist(),
            'eval_episodes': eval_episodes, 'seed': seed, 'trial': trial,
            'track_members': bool(track_members),
            'population_snapshot_interval': int(population_snapshot_interval),
            'snapshot_members': int(snapshot_members),
            'hidden_dims': list(hidden_dims), 'episode_length': episode_length,
            'obs_norm': bool(obs_norm),
            # The whitening statistics this run's NE policy was trained on, so a
            # post-hoc rebuild reads them instead of re-measuring them
            # (re-measurement does not reproduce; source/envs/mjx.py).
            **({'obs_mean': np.asarray(env_params.obs_mean).tolist(),
                'obs_std': np.asarray(env_params.obs_std).tolist()}
               if getattr(env_params, 'obs_mean', None) is not None else {}),
            'checkpoint_every': int(checkpoint_every) if checkpoint_every else None,
            'track_plasticity': bool(track_plasticity),
            'track_diversity': bool(track_diversity),
            # `elite_*` = the best-performing agent (best sampled member on
            # NES/OpenES, archive best on GA/DNS). Absent on runs before
            # 2026-09-13, whose ES/NES elite curve is the distribution mean.
            'elite_convention': 'best_member',
            'plasticity_interval': int(plasticity_interval) if track_plasticity else None,
            'num_params': num_params, 'solved_threshold': threshold,
            **({'arch': arch} if arch else {}),
            # What a row of `noise_vectors` IS on this suite, and the options
            # it was drawn under. Absent on gymnax, where it is always an
            # observation offset and a run made before this key existed
            # means the same thing as one made after.
            **({'task': env_params.describe()}
               if hasattr(env_params, 'describe') else {}),
        },
        'noise_vectors': np.asarray(noise_vectors).tolist(),
        'elapsed_seconds': elapsed,
        **summarise_records(records, num_tasks, threshold),
        # `joint` scores every member on EVERY sub-task, so one generation costs
        # num_tasks times as many episodes as a schedule that scores one
        # sub-task. Counting that here (as train_ppo.py already did) rather than
        # reporting the single-sub-task figure for every arm: without the
        # factor, joint's runs claim the same budget as switch's while spending
        # num_tasks times more. It makes joint an upper bound on what is
        # reachable, not a compute-matched competitor -- which is the role it is
        # used in, but the number has to say so.
        # A batch of k sub-tasks costs k times a single-sub-task generation,
        # exactly as `joint` costs num_tasks times. Counting it here keeps
        # every arm's env_steps comparable.
        'env_steps': (num_generations * pop_size * num_evals * episode_length
                      * (num_tasks if schedule == 'joint'
                         else min(tasks_per_batch or 1, num_tasks))),
        'env_steps_per_generation': (pop_size * num_evals * episode_length
                                     * (num_tasks if schedule == 'joint' else 1)),
    }

    if output_dir:
        arrays = dict(centroids=np.stack(centroids),
                      generations=np.asarray(centroid_gens),
                      noise_vectors=np.asarray(noise_vectors))
        if population_snapshots:
            # (num_snapshots, snapshot_members, num_params), plus which
            # sub-task each snapshot was taken under -- the animation colours
            # the cloud by that.
            arrays['populations'] = np.stack(population_snapshots)
            arrays['population_generations'] = np.asarray(population_snapshot_gens)
            arrays['population_tasks'] = np.asarray(population_snapshot_tasks)
        if champion_genomes:
            # (num_measurements, num_tasks, num_params): who was best at each
            # sub-task, each time we looked.
            arrays['champions'] = np.stack(champion_genomes)
            arrays['champion_generations'] = np.asarray(champion_gens)
        write_run(output_dir, result, records, **arrays)
        # `checkpoints.npz`, alongside -- never through `save_eval_artifacts`,
        # which would overwrite the `results.json` `write_run` just wrote with
        # a schema that has no `task_sequence` and no suite `task` block.
        # `noise_vectors` here is the PER-PHASE sequence: a schedule that
        # revisits sub-tasks has more phases than sub-tasks, and a checkpoint
        # belongs to the phase it closed.
        save_checkpoints(
            output_dir,
            noise_vectors=[np.asarray(noise_vectors[t])
                           for t in ckpt_phase_tasks],
            finalgen=ckpt_finalgen, incumbent=ckpt_incumbent,
            centroid=ckpt_centroid)

    if resume_path and os.path.exists(resume_path):
        os.remove(resume_path)          # a finished trial carries no stale checkpoint
    return result


def build_parser():
    p = argparse.ArgumentParser(description='NES on gymnax, switching sub-tasks')
    p.add_argument('--env', default='CartPole-v1', choices=ENV_NAMES)
    # `run_nes` has run all four searchers since the ne.py interface landed,
    # but this parser could not select one, so the module was NES-only from the
    # command line while its function was not. scripts/outdated/generalists/train/train_all.py is still
    # the entry point for anything with trials and schedules; this is for a
    # single run.
    p.add_argument('--method', default='nes',
                   choices=list(NE_METHODS))
    p.add_argument('--searcher_kwargs', nargs='*', default=None,
                   metavar='KEY=VALUE',
                   help="method-specific knobs: GA's elite_ratio, DNS's "
                        'iso_sigma/line_sigma/k. Unlike train_all.py this '
                        'applies NO reference defaults -- what is passed is '
                        'what runs.')
    p.add_argument('--schedule', default='switch', choices=SCHEDULES)
    p.add_argument('--num_generations', type=int, default=2000)
    p.add_argument('--task_interval', type=int, default=100,
                   help='generations per sub-task under --schedule switch')
    p.add_argument('--pop_size', type=int, default=512)
    p.add_argument('--sigma', type=float, default=0.1)
    p.add_argument('--learning_rate', type=float, default=0.05)
    p.add_argument('--optimizer', default='sgd',
                   choices=['sgd', 'sgd_momentum', 'adam'])
    p.add_argument('--shaping', default='zscore',
                   choices=['zscore', 'centered_rank', 'raw'])
    p.add_argument('--sigma_lr', type=float, default=0.0,
                   help='>0 turns on the separable-NES per-coordinate sigma')
    p.add_argument('--num_evals', type=int, default=1)
    # None means "this environment's own default" -- 1.0 on gymnax, the sigma
    # that body's obs-noise runs were made at on mjx. Sigma does not transfer
    # between bodies, so there is no one number to default to.
    p.add_argument('--noise_range', type=float, default=None)
    p.add_argument('--num_tasks', type=int, default=2)
    p.add_argument('--perturb_first_task', action='store_true',
                   help='draw an offset for sub-task 0 too, instead of leaving '
                        'it as the unperturbed environment')
    p.add_argument('--eval_episodes', type=int, default=16)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--trial', type=int, default=1)
    p.add_argument('--output_dir', default=None)
    p.add_argument('--log_interval', type=int, default=50)
    p.add_argument('--checkpoint_interval', type=int, default=1)
    p.add_argument('--track_members', action='store_true',
                   help="also score the generation's BEST MEMBER on every "
                        'sub-task, held out, and log it as best_task*/'
                        'best_generalist. That is the quantity the earlier '
                        "study reports; the centroid columns are this "
                        "project's. Costs one extra eval batch per generation.")
    p.add_argument('--population_snapshot_interval', type=int, default=0,
                   help='store the evaluated population every N generations '
                        'in trajectory.npz, for the landscape animation. '
                        '0 disables it.')
    p.add_argument('--snapshot_members', type=int, default=128,
                   help='members kept per snapshot (evenly spaced). The '
                        'animation draws a scatter, not a census.')
    p.add_argument('--task_options', nargs='*', default=None,
                   metavar='KEY=VALUE',
                   help="the suite's sub-task settings for this environment. "
                        'On the ant: friction_order=cycle|random, '
                        'friction_low, friction_high, friction_default, '
                        'target_speed (or none). gymnax takes none.')
    p.add_argument('--gpu', default=None, help='CUDA_VISIBLE_DEVICES value')
    return p


def main():
    args = build_parser().parse_args()
    if args.gpu is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    from scripts.outdated.generalists.train.train_all import (parse_searcher_kwargs,
                                                     parse_task_options)
    print(f"{args.method.upper()} | {args.env} | schedule={args.schedule} "
          f"| trial={args.trial}")
    run_nes(env_name=args.env, method=args.method,
            searcher_kwargs=parse_searcher_kwargs(args.searcher_kwargs),
            schedule=args.schedule,
            num_generations=args.num_generations,
            task_interval=args.task_interval, pop_size=args.pop_size,
            sigma=args.sigma, learning_rate=args.learning_rate,
            optimizer=args.optimizer, shaping=args.shaping,
            sigma_lr=args.sigma_lr, num_evals=args.num_evals,
            noise_range=args.noise_range, num_tasks=args.num_tasks,
            first_task_clean=not args.perturb_first_task,
            eval_episodes=args.eval_episodes, seed=args.seed, trial=args.trial,
            output_dir=args.output_dir, log_interval=args.log_interval,
            checkpoint_interval=args.checkpoint_interval,
            track_members=args.track_members,
            population_snapshot_interval=args.population_snapshot_interval,
            snapshot_members=args.snapshot_members,
            task_options=parse_task_options(args.task_options))


if __name__ == '__main__':
    main()
