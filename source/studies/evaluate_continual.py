#!/usr/bin/env python
"""Post-hoc evaluation of finished continual runs, on every suite.

Training saves, for each sub-task, the agent it ended with and the sub-task it
was trained on. This script scores those agents. Nothing here touches search,
so the number of evaluation episodes, the agent-selection rule and the metrics
can all change without re-running a single generation.

One evaluator for every body (CLAUDE.md (a)). A gymnax run is rebuilt the way
it always was -- `gymnax.make`, the classic-control MLP, and
`saved_task_params` resolving what an offset, a rescaled body or a reversed
action order did to the env params. Every other suite (MiniGrid, and brax and
kinetix once their modules supply the trace functions) goes through
`source/envs/run_context.RunContext`, which rebuilds the run from its own
recorded `task` block and scores an agent with the suite's own
`make_scoring_fn` -- the same ruler the training curve was drawn with.

It walks a project directory for runs (a ``results.json`` plus a
``checkpoints.npz``), rolls each saved agent out for ``--episodes`` fresh
episodes on its own sub-task, and writes ``evaluation.json`` next to the run
holding the raw per-episode returns. Metrics are then computed from those
returns by source/metrics/evaluation_metrics.py.

Zero-shot transfer is measured here too: each sub-task's agent is also scored on
the *next* sub-task, before any search has been done on it. It is the same agent
that sub-task scored for success rate, so measuring it costs one extra rollout
per sub-task and no extra checkpointing. Since 2026-09-10 each agent is ALSO
scored on the *previous* sub-task (`prev_returns`): agent t+1 on sub-task t is
the forgetting at that switch, `R[t][t] - R[t][t+1]`, and under a two-regime
alternation it is what tells a generalist from a switching specialist
(`scripts/analysis/generalist_checkpoints.py`).

Usage:
    python -m source.studies.evaluate_continual --root projects/iclr_2027/runs_centroid/minigrid/continual
    python -m source.studies.evaluate_continual --root <dir> --episodes 100 --gpus 0
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
import zlib

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# `source/studies/<this file>`: two levels up is the repo root. Invoked as a
# module (`python -m`) this insert is redundant; as a script it is what makes
# `source` importable.
REPO_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


def _get_gpu_arg():
    for i, arg in enumerate(sys.argv):
        if arg == "--gpus" and i + 1 < len(sys.argv):
            return sys.argv[i + 1]
    return None


_gpu_arg = _get_gpu_arg()
if _gpu_arg:
    os.environ["CUDA_VISIBLE_DEVICES"] = _gpu_arg
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp
import numpy as np
import gymnax
from jax import random

from source.envs.gymnax_classic import (
    make_gymnax_env, wrap_actions,
    FlipEnv, build_policy, gymnax_task_type, make_episode_fn, saved_task_rows)
from source.envs.run_context import RunContext, is_gymnax_run, run_config

# Which saved agent to score. "finalgen" is the best member of the sub-task's
# final generation, matching the definition the gymnax training scripts already
# use; "incumbent" is what the optimizer would hand back. Both are saved, so
# both are scored -- choosing between them is a reporting decision, not one
# that should require re-running anything.
#
# A run may override this in results.json under "agent_sources": the
# gradient-based methods (PPO, C-CHAIN, ...) carry a single agent per sub-task
# and declare ["final"], since there is no population to select from.
#
# "centroid" is the coordinate-wise mean of the population's weights -- THE
# network the paper's centroid tables report, and for ES/NES the incumbent
# itself. Scored whenever the checkpoint carries it; a run saved before the
# centroid was written simply lacks the entry. For a single-policy run
# `final` is the centroid, and the readers treat it as such.
AGENT_SOURCES = ("finalgen", "incumbent", "centroid")


def wanted_sources(cfg, ckpt_files):
    """The saved agents this run should be scored at.

    The run's own declaration (`agent_sources` in results.json -- the NE
    trainers write ["finalgen", "incumbent"], the gradient trainers
    ["final"]) plus the centroid whenever the checkpoint carries one: the
    trainers declared their sources before the centroid was saved beside
    them, and the declaration is a floor, not a ceiling.
    """
    declared = list(cfg.get("agent_sources") or AGENT_SOURCES)
    if "centroid" in ckpt_files and "centroid" not in declared:
        declared.append("centroid")
    wanted = tuple(s for s in declared if s in ckpt_files)
    # The shared runners' RL arms save `final` and declare nothing; a
    # single-policy run's `final` is its elite and its centroid alike.
    if not wanted and "final" in ckpt_files:
        wanted = ("final",)
    return wanted


def find_runs(root):
    """Every finished run under ``root``, as (results.json, checkpoints.npz)."""
    runs = []
    for results_path in sorted(glob.glob(os.path.join(root, "**", "results.json"),
                                         recursive=True)):
        ckpt_path = os.path.join(os.path.dirname(results_path), "checkpoints.npz")
        if os.path.exists(ckpt_path):
            runs.append((results_path, ckpt_path))
    return runs


def _gymnax_rollout(cfg, ckpt, episodes, cache):
    """`rollout(agent, key, task_index) -> (episodes,)` for a gymnax run."""
    num_tasks = int(ckpt["noise_vectors"].shape[0])
    # What a sub-task IS -- offset, rescaled body or reversed action order --
    # is resolved by `saved_task_rows` once the stock params exist below, for
    # a run from either trainer family.
    task_type = gymnax_task_type(cfg)
    env_name = cfg["env"]
    hidden_dims = tuple(cfg["hidden_dims"])
    episode_length = int(cfg["episode_length"])

    # Building the env and jitting the rollout dominates the cost of a single
    # run, so they are shared across every run with the same shape.
    # task_type is part of the key: an `actions` run needs the FlipEnv wrapper
    # around the same gymnax env, and the two must not share a jitted rollout.
    cache_key = ("gymnax", env_name, hidden_dims, episode_length, episodes, task_type)
    if cache_key not in cache:
        env, env_params = make_gymnax_env(env_name)
        env_params = env_params.replace(max_steps_in_episode=episode_length)
        obs, _ = env.reset(jax.random.key(0), env_params)
        obs_dim = obs.shape[-1]
        action_dim = env.action_space(env_params).n
        # AFTER the spaces are read, as in the trainers.
        if task_type == "actions":
            env = wrap_actions(env)
        policy, param_template, _ = build_policy(
            jax.random.key(0), obs_dim, action_dim, hidden_dims
        )
        episode = make_episode_fn(env, policy, param_template, episode_length)

        # `env_params` is an ARGUMENT, not a closure constant: a param sub-task
        # is a different body, and the whole point of gymnax's EnvParams being
        # a pytree is that it can be swapped without a recompile.
        @jax.jit
        def rollout(flat_params, key, noise_vector, task_env_params):
            keys = random.split(key, episodes)
            return jax.vmap(episode, in_axes=(None, 0, None, None))(
                flat_params, keys, noise_vector, task_env_params
            )

        cache[cache_key] = (rollout, env_params)
    rollout, stock_params = cache[cache_key]
    offsets, bodies, _mults = saved_task_rows(cfg, ckpt, stock_params, env_name)
    if len(bodies) < num_tasks:
        raise ValueError(f"{env_name}: {len(bodies)} saved bodies for "
                         f"{num_tasks} sub-tasks")
    offsets = jnp.asarray(offsets)

    def on_task(agent, key, t):
        return rollout(agent, key, offsets[t], bodies[t])
    return on_task


def _suite_rollout(cfg, ckpt, episodes, cache):
    """The same, through `RunContext`, for every suite the shared runners drive.

    Batched over the checkpoint: `score(agents, key, ts)` rolls agent i on
    sub-task `ts[i]` for every i in one jitted call, on the same episode
    seeds. The per-agent loop the gymnax path runs -- 20 agents x 3 shifts x
    up to 3 sources, 174 launches of a 1000-step scan per run -- took three
    to fifteen minutes per MJX run on a shared card (the ant, 2026-09-11),
    because an MJX step is launch-bound and a hundred episodes do not fill
    the GPU. Three calls per source do the same work in a fraction of it.
    """
    noise_vectors = jnp.asarray(ckpt["noise_vectors"])
    key = RunContext.cache_key(cfg, episodes)
    if key not in cache:
        cache[key] = RunContext(cfg, episodes)
    ctx = cache[key]

    def score(agents, key, ts):
        return ctx.returns_own_tasks(agents, key, noise_vectors[jnp.asarray(ts)])
    return score


def evaluate_run(results_path, ckpt_path, episodes, seed, cache):
    """Score one run's saved agents and return the evaluation record."""
    with open(results_path) as f:
        cfg = run_config(json.load(f))

    ckpt = np.load(ckpt_path)
    # One saved agent per sub-task PHASE, and `noise_vectors` is the matching
    # per-phase sequence (`save_checkpoints`). The gymnax trainers' `num_tasks`
    # is that phase count; the shared runners' is the number of DISTINCT
    # sub-tasks (2 on MiniGrid, over 20 phases), so the checkpoint decides.
    num_tasks = int(ckpt["noise_vectors"].shape[0])
    env_name = cfg["env"]
    agent_sources = wanted_sources(cfg, ckpt.files)
    gymnax = is_gymnax_run(cfg)
    if gymnax:
        on_task = _gymnax_rollout(cfg, ckpt, episodes, cache)
    else:
        score = _suite_rollout(cfg, ckpt, episodes, cache)

    # Evaluation keys derive from the run's identity, not from the training RNG
    # chain, so a re-evaluation is reproducible and is not correlated with the
    # episodes the agent was selected on.
    #
    # crc32 of the identity, NOT hash(). Python salts hash() of a str per
    # process unless PYTHONHASHSEED is set, so the previous version drew a
    # different evaluation key on every invocation -- which is precisely what
    # the sentence above says it does not do. Two runs of this script over the
    # same tree gave two sets of ZT and S numbers, and nothing said so.
    identity = repr((cfg["method"], env_name, cfg.get("pop_size"),
                     cfg["trial"], seed)).encode()
    key = jax.random.key(zlib.crc32(identity) % (2 ** 31))

    record = {
        "method": cfg["method"],
        "env": env_name,
        "pop_size": cfg.get("pop_size"),
        "trial": cfg["trial"],
        "num_tasks": num_tasks,
        "episodes": episodes,
        "eval_seed": seed,
        "agent_sources": list(agent_sources),
        "per_task": [],
    }

    for source in agent_sources:
        agents = jnp.asarray(ckpt[source])
        if agents.shape[0] != num_tasks:
            raise ValueError(
                f"{ckpt_path}: {source} has {agents.shape[0]} agents but the run "
                f"reports {num_tasks} sub-tasks"
            )
        if not gymnax:
            # Three batched calls per source: every agent on its own
            # sub-task, on the next one, on the previous one. The shifted
            # sequences are rolled rather than sliced so all three calls
            # share one compiled shape; the wrapped-around row (agent T-1 on
            # sub-task 0, agent 0 on sub-task T-1) is computed and dropped.
            key, k_task, k_zero, k_prev = random.split(key, 4)
            ts = np.arange(num_tasks)
            own = np.asarray(score(agents, k_task, ts))
            nxt = np.asarray(score(agents, k_zero, np.roll(ts, -1)))
            prv = np.asarray(score(agents, k_prev, np.roll(ts, 1)))
            for t in range(num_tasks):
                entry = {
                    "task_idx": t,
                    "source": source,
                    "returns": [float(x) for x in own[t]],
                }
                if t + 1 < num_tasks:
                    entry["zero_shot_next_returns"] = [float(x) for x in nxt[t]]
                if t > 0:
                    entry["prev_returns"] = [float(x) for x in prv[t]]
                record["per_task"].append(entry)
            continue
        for t in range(num_tasks):
            key, k_task, k_zero, k_prev = random.split(key, 4)
            returns = np.asarray(on_task(agents[t], k_task, t))
            entry = {
                "task_idx": t,
                "source": source,
                "returns": [float(x) for x in returns],
            }
            # Zero-shot transfer: this sub-task's agent, scored on the *next*
            # sub-task before any search on it. Reconstructible here only
            # because the agents and the task sequence were both saved.
            if t + 1 < num_tasks:
                nxt = np.asarray(on_task(agents[t], k_zero, t + 1))
                entry["zero_shot_next_returns"] = [float(x) for x in nxt]
            # And on the *previous* one: what this agent still scores on the
            # sub-task it was trained on before this one. Agent t on sub-task
            # t-1 is the reward side of forgetting at switch t-1.
            if t > 0:
                prev = np.asarray(on_task(agents[t], k_prev, t - 1))
                entry["prev_returns"] = [float(x) for x in prev]
            record["per_task"].append(entry)

    return record


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True,
                    help="Project directory to walk for finished runs")
    ap.add_argument("--episodes", type=int, default=100,
                    help="Fresh evaluation episodes per saved agent")
    ap.add_argument("--seed", type=int, default=0,
                    help="Evaluation seed; changing it re-draws the evaluation "
                         "episodes without touching training")
    ap.add_argument("--gpus", type=str, default=None)
    ap.add_argument("--force", action="store_true",
                    help="Re-evaluate runs that already have evaluation.json")
    args = ap.parse_args()

    runs = find_runs(args.root)
    if not runs:
        raise SystemExit(f"No runs with checkpoints found under {args.root}")

    pending = []
    for results_path, ckpt_path in runs:
        out_path = os.path.join(os.path.dirname(results_path), "evaluation.json")
        if not args.force and os.path.exists(out_path):
            with open(out_path) as f:
                existing = json.load(f)
            # Current only if it also scored every source the checkpoint
            # carries: an evaluation.json written before `centroid` joined
            # AGENT_SOURCES is stale for the centroid tables, not current.
            with open(results_path) as f, np.load(ckpt_path) as data:
                wanted = wanted_sources(run_config(json.load(f)), data.files)
            if existing.get("episodes") == args.episodes and \
                    existing.get("eval_seed") == args.seed and \
                    set(wanted) <= set(existing.get("agent_sources") or []):
                continue
        pending.append((results_path, ckpt_path, out_path))

    print(f"Evaluating {len(pending)} run(s) at {args.episodes} episodes "
          f"({len(runs) - len(pending)} already current)")

    cache = {}
    start = time.time()
    for i, (results_path, ckpt_path, out_path) in enumerate(pending, 1):
        record = evaluate_run(results_path, ckpt_path, args.episodes, args.seed, cache)
        with open(out_path, "w") as f:
            json.dump(record, f)
        if i % 20 == 0 or i == len(pending):
            print(f"  [{i}/{len(pending)}] {time.time() - start:.0f}s", flush=True)

    print(f"Done in {(time.time() - start) / 60:.1f} min")


if __name__ == "__main__":
    main()
