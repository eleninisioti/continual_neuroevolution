#!/usr/bin/env python
"""Post-hoc evaluation of finished continual brax-ant runs.

The brax counterpart of source/studies/mujoco/evaluate_continual.py, and a near-direct
port of it: the two suites save the same two checkpoint families and want the
same T x T sweep, so the arithmetic, the output payloads and the agent-selection
rule are that file's and are not restated here. What differs is only how a
sub-task is rebuilt, which is the whole of `build_env` below.

    reward[t][a] = mean return of the agent trained on sub-task `a`,
                   evaluated on sub-task `t`

  * `reward[t][t]`         -- the diagonal: `P`, and the success rate.
  * `reward[t+1][t]`       -- zero-shot transfer, `ZT`.
  * `reward[i][i] - reward[i][T-1]` meaned over i -- Continual World's
    forgetting (eq. 3), which make_figures.py reads as `final_forgetting`,
    and the action-disagreement matrix beside it gives `BD`.

Writing this unblocks four columns that print `--` in both ant tables today
(`P`, `ZT`, `F`, `BD`) with no re-training: every ant trial already carries its
12 per-sub-task checkpoints.

Rebuilding a sub-task
---------------------
Two things the mujoco script gets for free have to be recovered here, and both
are recovered from data rather than assumed:

*The task parameters* are read from each checkpoint, which records its own
`task_idx`, `friction_mult`, `target_speed` and `damaged_leg`. So sub-task t's
environment is built from sub-task t's checkpoint, whichever agent is being
scored in it.

*The observation offset* is the one piece not in any checkpoint or config.
`ObsOffsetWrapper` derives it from `(seed, task_idx)` alone, so it is
reproducible given the seed -- which every run does record -- and the sigma,
which only the setting directory name carries (`..._s2p0_...` -> 2.0). That
parse is explicit in `obs_sigma_of` and raises on an unrecognised
`*obsnoise*` directory rather than silently evaluating at sigma 0, which would
score every agent on an unperturbed task and quietly report no forgetting at
all.

Usage:
    python -m source.studies.brax.evaluate_continual --root projects/neurips_2026_rebuttal
    python -m source.studies.brax.evaluate_continual --root <dir> --episodes 20 --gpus 0
    python -m source.studies.brax.evaluate_continual --settings continual_obsnoise_s2p0_speed2_mjx_t12
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import pickle
import re
import sys
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# One dirname deeper since 2026-09-08: the trainers moved from
# `source/<suite>/` into `source/studies/<suite>/`. The runner
# invokes them as SCRIPTS, so `source` is importable only via
# this insert -- a short walk here is a ModuleNotFoundError at
# launch, not a subtle one.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(SCRIPT_DIR)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


def _get_gpu_arg():
    """`--gpus` read before jax is imported, which is when it must be set."""
    for i, a in enumerate(sys.argv):
        if a == "--gpus" and i + 1 < len(sys.argv):
            return sys.argv[i + 1]
        if a.startswith("--gpus="):
            return a.split("=", 1)[1]
    return None


_gpus = _get_gpu_arg()
if _gpus is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = _gpus

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from jax import flatten_util, random  # noqa: E402

from source.envs.brax_ant import create_env_with_damaged_leg
from source.algorithms.networks import ACTIVATIONS, POLICY_ARCH  # noqa: E402
from source.studies.brax.train_GA_ant import MLPPolicy  # noqa: E402

EVALUATION_JSON = "evaluation.json"
# Beside the ant figures, so `forgetting_cell` finds it through the same
# `results_dir` it is already given. Both ant tasks write here; records carry
# `run_dir` and `_load_divergence_records` matches on it, so the obs-noise
# figure never sees the direction runs or the reverse.
DIVERGENCE_JSON = os.path.join("results", "brax_continual",
                               "behavioural_divergence.json")

_TASK_PKL = re.compile(r"task_(\d+)_.*\.pkl$")
# `..._s2p0_...` -> 2.0. Only the obs-noise trees carry it.
_SIGMA_RE = re.compile(r"_s(\d+)p(\d+)")


def is_ne_checkpoint(ckpt):
    """Flat-vector NE checkpoint rather than a brax PPO one.

    Decided on the file's keys, not the directory name, for the reason the
    mujoco script documents at length: condition directories outlive the method
    names any dispatch table knows, and a name-based test sends an archived or
    renamed NE condition to the brax loader, which dies on `normalizer_params`
    partway through a sweep. The two key sets are disjoint.
    """
    return "flat_params" in ckpt and "param_template" in ckpt


def obs_sigma_of(setting):
    """The observation-noise sigma a setting directory declares.

    0.0 for the trees that carry no offset (the direction task, the stationary
    reference). Raises for a directory that says `obsnoise` but whose sigma
    cannot be read: evaluating those at 0.0 would rebuild every sub-task as the
    unperturbed task, which scores every agent on the one thing it cannot have
    forgotten and reports F ~ 0 for every method.
    """
    m = _SIGMA_RE.search(setting)
    if m:
        return float(f"{int(m.group(1))}.{m.group(2)}")
    if "obsnoise" in setting:
        raise ValueError(
            f"{setting}: directory says obsnoise but carries no _s<N>p<M> "
            "sigma tag; refusing to evaluate it as unperturbed")
    return 0.0


def obs_task_period_of(setting):
    """Sub-task revisit period, 0 for every tree except the `_t20` reruns.

    `ObsOffsetWrapper` folds this in before drawing the offset, so getting it
    wrong on a revisit tree gives sub-tasks 10 and 11 fresh offsets where the
    run saw repeats of 0 and 1.
    """
    return 10 if setting.endswith("_t20") else 0


def find_runs(root, settings=None, methods=None, max_trials=None):
    """[(trial_dir, [task pkls])] for every finished continual ant run.

    `methods` splits one tree across several processes: a 12x12 mjx sweep is
    ~400s per run and a full tree is hundreds of runs, so scoring it serially
    takes longer than the rest of the pipeline put together. Each process
    writes only into the trial directories it owns, so disjoint method sets
    cannot collide.
    """
    runs = []
    # `ant*`, not `ant`: the 24-sub-task friction root links its env directory
    # as `ant_t24` so make_figures' `brax_continual_config` can tell the two
    # sequence lengths apart. Matched literally, this pass silently found zero
    # runs there and every post-hoc column (P, ZT, F, BD) stayed `--` while the
    # tree looked complete.
    pattern = os.path.join(root, "brax", "*", "*", "ant*", "trial_*")
    for trial_dir in sorted(glob.glob(pattern)):
        setting = trial_dir.split(os.sep)[-4]
        if not setting.startswith(("continual", "reference")):
            continue
        if settings and setting not in settings:
            continue
        if methods and trial_dir.split(os.sep)[-3] not in methods:
            continue
        ckpt_dir = os.path.join(trial_dir, "checkpoints")
        if not os.path.isdir(ckpt_dir):
            continue
        # Unfinished runs are skipped: a partial matrix silently changes what
        # the mean over sub-tasks is a mean of.
        if not os.path.isfile(os.path.join(trial_dir, "training_metrics.json")):
            continue
        pkls = [p for p in sorted(glob.glob(os.path.join(ckpt_dir, "task_*.pkl")))
                if _TASK_PKL.search(os.path.basename(p))]
        if pkls:
            runs.append((trial_dir, pkls))

    if max_trials:
        # The post-hoc columns (P, ZT, F, BD) are a mean and a spread over
        # seeds; the trace columns keep every trial regardless, since they cost
        # nothing to load. A 12x12 mjx sweep is ~400s per run, so capping the
        # seeds here is the difference between hours and tens of minutes.
        # Lowest trial indices, so the choice is deterministic and the same
        # seeds are scored in every cell rather than whichever finished first.
        by_cell = {}
        for trial_dir, pkls in runs:
            by_cell.setdefault(os.path.dirname(trial_dir), []).append((trial_dir, pkls))
        runs = []
        for cell in sorted(by_cell):
            picked = sorted(by_cell[cell],
                            key=lambda tp: int(tp[0].rsplit("_", 1)[1]))[:max_trials]
            runs.extend(picked)
    return runs


# `  Friction sequence (random): [1.0, 0.5205, 0.0545, ...]` in a PBT train.log.
_FRICTION_SEQ_RE = re.compile(
    r"Friction sequence \((\w+)\):\s*\[([^\]]*)\]")


def friction_sequence_from_log(trial_dir):
    """`(order, [mult per sub-task])` from a run's own `train.log`, or None.

    Only `train_PBT_ant_continual.py` needs this. Every other ant trainer
    records `friction_mult` in each checkpoint, so `build_env` reads the
    sub-task's own parameters and this is never consulted.

    **Why the log rather than regenerating the sequence.** `frictions_from_args`
    is deterministic given the run's seed, so the draw could be recomputed --
    but that reads the sampling rule as it is *today* against runs trained under
    whatever it was then, which is the failure the module docstring rejects for
    the observation offset. The log line is the run's own record of the sequence
    it actually applied, so it has the same standing as a checkpoint field.
    """
    log = os.path.join(trial_dir, "train.log")
    if not os.path.isfile(log):
        return None
    with open(log, errors="replace") as f:
        for line in f:
            m = _FRICTION_SEQ_RE.search(line)
            if m:
                body = m.group(2).strip()
                if not body:
                    return m.group(1), []
                return m.group(1), [float(x) for x in body.split(",")]
    return None


def backfill_friction(trial_dir, agents):
    """Give PBT checkpoints the `friction_mult` their trainer did not save.

    `train_PBT_ant_continual.py` applies `friction_sequence[task_idx]` when it
    builds each sub-task (see its line 735) but writes only `multiplier` and a
    hardcoded `task_mod='obsnoise'` into the checkpoint -- the multiplier is the
    sub-task INDEX, not a friction value, and the label is legacy. So a friction
    PBT run's checkpoints carry no record of the friction they trained under.

    Left alone, `build_env`'s `ckpt.get("friction_mult", 1.0)` would build all
    12 sub-tasks as the unperturbed ant. Every agent would then be scored on the
    same environment it was trained on, F and BD would collapse toward zero, and
    nothing about the output would look wrong -- the same silent-success failure
    `obs_sigma_of` raises on rather than permits.

    So this raises rather than guessing when a run needs the fallback and the
    log cannot supply it.
    """
    if all("friction_mult" in ckpt for _, ckpt in agents):
        return agents
    found = friction_sequence_from_log(trial_dir)
    if found is None:
        raise ValueError(
            f"{trial_dir}: checkpoints carry no `friction_mult` and train.log "
            "has no `Friction sequence` line, so the sub-task friction cannot "
            "be recovered. Evaluating anyway would build every sub-task at "
            "friction 1.0 and silently report near-zero forgetting.")
    order, seq = found
    for idx, ckpt in agents:
        if "friction_mult" in ckpt:
            continue
        if idx >= len(seq):
            raise ValueError(
                f"{trial_dir}: train.log's friction sequence has {len(seq)} "
                f"entries but a checkpoint is sub-task {idx}.")
        ckpt["friction_mult"] = float(seq[idx])
    return agents


def load_agents(pkls, trial_dir=None):
    """[(task_idx, checkpoint)] ordered by sub-task."""
    agents = []
    for path in pkls:
        with open(path, "rb") as f:
            ckpt = pickle.load(f)
        idx = int(_TASK_PKL.search(os.path.basename(path)).group(1))
        agents.append((idx, ckpt))
    agents.sort(key=lambda a: a[0])
    # RL checkpoints only -- the NE ones are flat vectors with no task fields at
    # all and are rebuilt from their own embedded config elsewhere.
    if agents and not is_ne_checkpoint(agents[0][1]):
        agents = backfill_friction(trial_dir or os.path.dirname(
            os.path.dirname(pkls[0])), agents)
    return agents


def seed_of(trial_dir, agents):
    """The training seed, which is what the observation offset is drawn from.

    NE checkpoints carry it in their embedded config; the RL ones do not, so
    that family falls back to the run's `config.json`. Raises rather than
    defaulting -- a wrong seed rebuilds a different offset and the whole matrix
    is then measuring the wrong perturbation, which nothing downstream could
    detect.
    """
    cfg = agents[0][1].get("config") or {}
    if "seed" in cfg:
        return int(cfg["seed"])
    path = os.path.join(trial_dir, "config.json")
    if os.path.isfile(path):
        with open(path) as f:
            disk = json.load(f)
        if "seed" in disk:
            return int(disk["seed"])
    raise ValueError(f"{trial_dir}: no seed in checkpoint config or config.json")


def build_env(ckpt, setting, seed, episode_length, backend):
    """The environment of the sub-task `ckpt` was trained on.

    Every task parameter comes from that sub-task's own checkpoint, so this is
    the environment the run actually faced rather than one regenerated from a
    sequence rule that may since have changed.
    """
    return create_env_with_damaged_leg(
        ckpt.get("config", {}).get("env", "ant") if ckpt.get("config") else "ant",
        ckpt.get("damaged_leg"),
        episode_length=episode_length,
        friction_mult=float(ckpt.get("friction_mult", 1.0)),
        target_speed=ckpt.get("target_speed"),
        backend=backend,
        gravity_mult=float(ckpt.get("gravity_mult", 1.0) or 1.0),
        obs_noise_sigma=obs_sigma_of(setting),
        obs_noise_seed=seed,
        task_idx=int(ckpt["task_idx"]),
        obs_task_period=obs_task_period_of(setting),
    )


def make_ne_policy(agents, action_dim):
    """`(apply(flat, obs), [flat...])` for the flat-vector checkpoints.

    The agent argument is the flat vector, so every sub-task's agent shares one
    pytree structure and the rollout compiles once per environment rather than
    once per (agent, sub-task) pair.
    """
    template = agents[0][1]["param_template"]
    _, unravel = flatten_util.ravel_pytree(template)
    hidden = tuple(agents[0][1].get("config", {}).get("hidden_dims") or (128, 128))
    policy = MLPPolicy(hidden_dims=hidden, action_dim=action_dim)

    def apply(flat, obs):
        return policy.apply(unravel(flat), obs)

    return apply, [jnp.asarray(ckpt["flat_params"]) for _, ckpt in agents]


def make_rl_policy(agents, obs_dim, action_dim):
    """`(apply(agent, obs), [agent...])` for the brax PPO checkpoints.

    `deterministic=True` takes the mode of the policy distribution, matching
    training-time evaluation. Sampling instead would score a stochastic RL
    policy against a deterministic NE one.
    """
    from brax.training.acme import running_statistics
    from brax.training.agents.ppo import networks as ppo_networks

    cfg = agents[0][1].get("config", {}) or {}
    hidden = tuple(cfg.get("policy_hidden_sizes") or POLICY_ARCH['brax']['hidden_dims'])
    value_hidden = tuple(cfg.get("value_hidden_sizes") or hidden)
    # From the run, not from make_ppo_networks' default swish -- see the same
    # comment in source/studies/mujoco/evaluate_continual.py. Ant runs written before
    # `activation` was recorded were swish; they predate the switch to tanh and
    # have to be re-run, so the fallback is the current POLICY_ARCH value and a
    # stale checkpoint will read visibly wrong rather than silently plausible.
    activation = ACTIVATIONS[cfg.get("activation") or POLICY_ARCH['brax']['activation']]

    network = ppo_networks.make_ppo_networks(
        observation_size=obs_dim,
        action_size=action_dim,
        preprocess_observations_fn=running_statistics.normalize,
        policy_hidden_layer_sizes=hidden,
        value_hidden_layer_sizes=value_hidden,
        activation=activation,
    )
    make_policy = ppo_networks.make_inference_fn(network)

    def apply(agent, obs):
        action, _ = make_policy(agent, deterministic=True)(obs, random.key(0))
        return action

    return apply, [(ckpt["normalizer_params"], ckpt["policy_params"])
                   for _, ckpt in agents]



def make_action_fn(apply_fn):
    """Jitted `actions(agents, states) -> (num_agents, num_states, action_dim)`."""

    @jax.jit
    def actions(agents, states):
        per_agent = jax.vmap(apply_fn, in_axes=(None, 0))
        return jax.vmap(per_agent, in_axes=(0, None))(agents, states)

    return actions


def probe_states(obs, alive, num_states, rng):
    """`num_states` states the agent actually visited, from its trajectories.

    Post-termination steps are dropped: an ant that has fallen is frozen, and a
    probe set including those states would have every agent agree on them and
    drag the disagreement toward zero.
    """
    flat = np.asarray(obs).reshape(-1, np.asarray(obs).shape[-1])
    live = np.asarray(alive).reshape(-1) > 0
    flat = flat[live]
    if len(flat) == 0:
        return None
    if len(flat) > num_states:
        flat = flat[rng.choice(len(flat), size=num_states, replace=False)]
    return jnp.asarray(flat)


def disagreement_matrix(action_fn, stacked, states, action_range=2.0):
    """`d[i][j]`: how differently agent j acts from agent i on `states`.

    Mean per-actuator absolute action difference over the probe states, divided
    by the action range so it lands in [0, 1]. Identical in meaning for an
    evolved policy and a deterministically evaluated RL one, which is what lets
    NE and RL sit in one column. Same definition as the mujoco evaluator's, so
    the cheetah and ant BD columns are the same quantity.
    """
    acts = np.asarray(action_fn(stacked, states))       # (agents, states, dims)
    n = acts.shape[0]
    d = np.zeros((n, n))
    for i in range(n):
        d[i] = np.abs(acts - acts[i]).mean(axis=(1, 2)) / action_range
    return d


def consecutive(mat):
    """Mean of the sub-diagonal: how much each agent differs from the previous."""
    n = len(mat)
    return float(np.mean([mat[i + 1][i] for i in range(n - 1)])) if n > 1 else 0.0


def make_rollout(env, apply_fn, episode_length, episodes):
    """Jitted `rollout(stacked_agents, key) -> (returns, obs, alive)`.

    vmapped over agents and episodes together: the T x T sweep is T rollouts of
    T agents, and compiling per agent would dominate the run.
    """
    reset = jax.jit(env.reset)
    step = jax.jit(env.step)

    def one(agent, key):
        state = reset(key)

        def body(carry, _):
            state, total, alive = carry
            action = apply_fn(agent, state.obs)
            nxt = step(state, action)
            total = total + nxt.reward * alive
            alive = alive * (1.0 - nxt.done)
            return (nxt, total, alive), (state.obs, alive)

        (_, total, _), (obs, alive) = jax.lax.scan(
            body, (state, 0.0, 1.0), None, length=episode_length)
        return total, obs, alive

    def rollout(stacked, key):
        keys = random.split(key, episodes)
        per_agent = jax.vmap(lambda a: jax.vmap(lambda k: one(a, k))(keys))
        return per_agent(stacked)

    return jax.jit(rollout)


def stack(agents_params):
    """Agent list -> one pytree with a leading agent axis."""
    return jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *agents_params)


def evaluate_run(trial_dir, pkls, episodes, seed_offset, backend,
                 num_states=512):
    """The T x T matrix and both payloads for one run."""
    agents = load_agents(pkls, trial_dir)
    setting = trial_dir.split(os.sep)[-4]
    seed = seed_of(trial_dir, agents)
    cfg = agents[0][1].get("config") or {}
    episode_length = int(cfg.get("episode_length") or 1000)
    if "episode_length" not in cfg:
        path = os.path.join(trial_dir, "config.json")
        if os.path.isfile(path):
            with open(path) as f:
                episode_length = int(json.load(f).get("episode_length", 1000))

    probe = build_env(agents[0][1], setting, seed, episode_length, backend)
    obs_dim, action_dim = int(probe.observation_size), int(probe.action_size)

    if is_ne_checkpoint(agents[0][1]):
        apply_fn, params = make_ne_policy(agents, action_dim)
    else:
        apply_fn, params = make_rl_policy(agents, obs_dim, action_dim)
    stacked = stack(params)

    n = len(agents)
    rng = np.random.default_rng(seed_offset)
    probe_set = None
    reward = np.zeros((n, n), dtype=float)
    for t, (_, ckpt) in enumerate(agents):
        env = probe if t == 0 else build_env(ckpt, setting, seed,
                                             episode_length, backend)
        rollout = make_rollout(env, apply_fn, episode_length, episodes)
        returns, obs, alive = rollout(stacked, random.key(seed_offset + t))
        reward[t, :] = np.asarray(returns).mean(axis=1)
        if t == 0:
            # Probe states come from sub-task 0, where every
            # agent is evaluated on the same unperturbed task --
            # so the disagreement is between policies, not
            # between the states they happen to visit.
            probe_set = probe_states(np.asarray(obs)[0],
                                 np.asarray(alive)[0],
                                     num_states, rng)

    # The action-disagreement matrix, which is what `BD` is and what the
    # per-trial fallback in `_load_divergence_records` filters on
    # (DIVERGENCE_KEY = "consecutive_disagreement"). Without it both F and BD
    # stay blank however many runs are scored, because a record lacking that
    # key is skipped entirely -- which is exactly what happened on the first
    # pass of this evaluator.
    if probe_set is not None:
        action_fn = make_action_fn(apply_fn)
        disagreement = disagreement_matrix(action_fn, stacked, probe_set)
    else:
        disagreement = np.zeros((n, n))

    retention = reward - np.diag(reward)[:, None]
    last = n - 1
    forgotten = -retention[:last, last] if last > 0 else np.zeros(0)

    # Schema is `_load_continual_evaluations`'s, not one of this file's
    # choosing: it reads `per_task`, filters on `source` against
    # `agent_sources`, and takes zero-shot from `zero_shot_next_returns`. An
    # earlier version of this function used `tasks`/`zero_shot_returns` and the
    # loader died on KeyError: 'per_task' the first time a table was built from
    # it. Match gymnax's writer exactly.
    evaluation = {
        "env": "ant",
        "method": os.path.basename(os.path.dirname(os.path.dirname(trial_dir))),
        "trial": int(os.path.basename(trial_dir).split("_")[-1]),
        "episodes": episodes,
        "num_tasks": n,
        "pop_size": (agents[0][1].get("config") or {}).get("pop_size"),
        "agent_sources": ["final_generation_best"],
        "eval_seed": seed_offset,
        "setting": setting,
        "seed": seed,
        "per_task": [
            {
                "task_idx": t,
                # The diagonal: this sub-task's own agent on it. `P`, and the
                # success rate.
                "returns": [float(reward[t, t])],
                "source": "final_generation_best",
                # Sub-task t's agent on t+1, before any search on it. `ZT`.
                "zero_shot_next_returns": ([float(reward[t + 1, t])]
                                           if t + 1 < n else None),
            }
            for t in range(n)
        ],
    }
    off = ~np.eye(n, dtype=bool)
    summary = {
        "consecutive_disagreement": consecutive(disagreement),
        "drift_from_first": float(disagreement[0][last]) if n > 1 else 0.0,
        "final_forgetting": float(np.mean(forgotten)) if forgotten.size else None,
    }
    # Written into the trial's own evaluation.json as well as the aggregate.
    # The aggregate is only flushed when the whole sweep finishes, so a figure
    # built while it is still running would see nothing; the per-trial copy is
    # the source that cannot go stale, and is what the loader falls back to.
    evaluation["summary"] = summary
    evaluation["matrices"] = {"reward": reward.tolist(),
                              "retention": retention.tolist(),
                              "disagreement": disagreement.tolist()}

    divergence = {
        "method": evaluation["method"],
        "env": "ant",
        "pop_size": (agents[0][1].get("config") or {}).get("pop_size"),
        "trial": evaluation["trial"],
        "num_tasks": n,
        "source": setting,
        "run_dir": trial_dir,
        "matrices": evaluation["matrices"],
        "summary": summary,
    }
    return evaluation, divergence


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.join(
        REPO_ROOT, "projects", "neurips_2026_rebuttal"))
    ap.add_argument("--episodes", type=int, default=10,
                    help="evaluation episodes per (agent, sub-task) cell")
    ap.add_argument("--settings", nargs="+", default=None,
                    help="only these setting directories")
    ap.add_argument("--max-trials", dest="max_trials", type=int, default=None,
                    help="score only the N lowest-numbered trials per cell")
    ap.add_argument("--methods", nargs="+", default=None,
                    help="only these method directories -- use to split one "
                         "tree across several GPUs")
    ap.add_argument("--backend", default="mjx")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gpus", default=None)
    ap.add_argument("--overwrite", action="store_true",
                    help="re-score runs that already have an evaluation.json")
    args = ap.parse_args()

    runs = find_runs(args.root, args.settings, args.methods,
                     args.max_trials)
    print(f"{len(runs)} finished continual ant runs under {args.root}")

    records, done, failed = [], 0, []
    for trial_dir, pkls in runs:
        out = os.path.join(trial_dir, EVALUATION_JSON)
        if os.path.isfile(out) and not args.overwrite:
            with open(out) as f:
                pass
            print(f"  skip (scored)  {os.path.relpath(trial_dir, args.root)}")
            continue
        t0 = time.time()
        try:
            evaluation, divergence = evaluate_run(
                trial_dir, pkls, args.episodes, args.seed, args.backend)
        except Exception as exc:  # one bad run must not cost the sweep
            print(f"  FAIL {os.path.relpath(trial_dir, args.root)}: "
                  f"{type(exc).__name__}: {exc}")
            failed.append((trial_dir, repr(exc)))
            continue
        with open(out, "w") as f:
            json.dump(evaluation, f)
        records.append(divergence)
        done += 1
        print(f"  ok   {os.path.relpath(trial_dir, args.root)}  "
              f"({time.time() - t0:.0f}s, F={divergence['summary']['final_forgetting']})")

    if records:
        agg = os.path.join(args.root, DIVERGENCE_JSON)
        os.makedirs(os.path.dirname(agg), exist_ok=True)
        payload = {"config": {"episodes": args.episodes,
                              "source": "source.studies.brax.evaluate_continual"},
                   "runs": records}
        if os.path.isfile(agg):
            with open(agg) as f:
                old = json.load(f)
            keep = [r for r in old.get("runs", [])
                    if r.get("run_dir") not in {r2["run_dir"] for r2 in records}]
            payload["runs"] = keep + records
        with open(agg, "w") as f:
            json.dump(payload, f)
        print(f"\nwrote {len(records)} divergence records -> {agg}")

    print(f"\n{done} scored, {len(failed)} failed")
    for trial_dir, exc in failed:
        print(f"  {trial_dir}: {exc}")
    return 1 if failed and not done else 0


if __name__ == "__main__":
    sys.exit(main())
