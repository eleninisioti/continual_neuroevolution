#!/usr/bin/env python
"""Forgetting as behavioural divergence between sub-tasks (continual runs, every suite).

A continual run hands us one agent per sub-task. Reward curves say whether the
agent is still competent; they do not say whether it is still doing the same
thing. This script measures the second question directly, for every ordered
pair of sub-tasks, and reports it as a heatmap:

    row i    the sub-task whose states and observation shift define the test
    col j    the agent being tested on them

The diagonal is zero by construction, the first sub-diagonal (i, i+1) is the
divergence at each task switch, and row 0 read left to right is the cumulative
drift away from the agent that solved sub-task 0.

Four matrices, in the order they should be read:

* `disagreement` -- the headline. The fraction of states in D_i on which agent
  j takes a different action from agent i. Bounded in [0, 1], free of any
  temperature choice, and identical in meaning for a deterministic evolved
  policy and a greedily evaluated RL one, which is what lets NE and RL numbers
  sit in the same figure. This is BD_tau from docs/behavioral_divergence.md,
  evaluated at every pair rather than only at consecutive ones.

* `occupancy_shift` -- how far agent j's state visitation on sub-task i moved
  from agent i's, as an MMD under an RBF kernel (random Fourier features,
  AutoQD-style). A policy can keep `disagreement` low on D_i and still stop
  going where agent i went; this is the part of forgetting that the action
  metric cannot see.

* `retention` -- reward[i][j] - reward[i][i], the return agent j gets on
  sub-task i relative to the agent that was trained on it. This is forgetting
  as the continual-learning literature defines it, and it is what the two
  behavioural matrices are validated against: a divergence measure is only
  interesting if it predicts the performance the run actually loses.

* `reward` -- the raw return behind `retention`, plotted alongside it because
  the two answer different questions. `retention` is normalised by each row's
  own diagonal, so a row where nothing was ever learnt looks as retentive as a
  row that was learnt and kept; only the raw matrix shows that the whole row
  sits at the floor. Its scale is per env, so read it down a column and never
  against another env's panel.

`js_action` and `kl_action` (mean Jensen-Shannon / KL between softmaxed logits
over D_i) are recorded as diagnostics but kept out of the headline figure. The
gymnax policies are deterministic: softmaxing their logits at temperature 1
gives a divergence that scales with the magnitude of the weights, which is
unconstrained under evolution and differs systematically from what gradient
training produces, so these two are comparable within a method and not across.
Where a distributional number is wanted, JS is the one to quote -- symmetric,
bounded by log 2, finite under disjoint support -- and `occupancy_shift` is the
distributional divergence that *is* cross-method comparable.

Reads any run written in the continual format of
`source/studies/evaluate_continual.py` (a `results.json` beside a
`checkpoints.npz`), so it works on the NE runs and on the PPO/C-CHAIN ones
without being told which is which. A gymnax run is rebuilt by `EnvContext`
below, with `saved_task_params` resolving what each sub-task did to the env;
a run on any other suite (MiniGrid) goes through
`source/envs/run_context.RunContext`, which reads the run's own `task` block.
The two expose the same three rollouts, and everything from the state sets
on is shared. On MiniGrid the states in D_i are the raw 7x7 views and the
occupancy feature is the agent's grid position; on gymnax both are the
observation.

Outputs (under <results>/continual/):
    behavioural_divergence.npz     every matrix of every run
    behavioural_divergence.json    per-run summary numbers
    behavioural_divergence.md      tables, incl. BD-vs-forgetting correlation
    bd_heatmaps_<env>.pdf          the heatmaps, averaged over trials
    bd_consecutive.pdf             divergence per task switch, and from task 0
    bd_vs_forgetting.pdf           does divergence predict the reward lost?

Usage:
    python scripts/neurips_2026_rebuttal/behavioural_divergence_analysis.py \
        --runs_root projects/gymnax/ne_popsweep --pop_sizes 128 --gpus 0
    python scripts/neurips_2026_rebuttal/behavioural_divergence_analysis.py \
        --envs CartPole_v1 --methods ga es --num_tasks 10
"""

import argparse
import glob
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# One dirname deeper since 2026-09-08: this file moved from
# `scripts/` into `scripts/outdated/`.
def _find_repo_root(start):
    """Walk up from `start` until the directory holding `pyproject.toml`.

    Not `dirname(dirname(...))`: that hardcodes how deep this file sits, and
    every time a script moved between `scripts/`, `scripts/outdated/` and
    `scripts/analysis/` the count went stale and the failure was a
    ModuleNotFoundError or a runs_root pointing inside `scripts/`. A marker
    search is correct wherever the file lives.
    """
    path = os.path.abspath(start)
    while True:
        parent = os.path.dirname(path)
        if os.path.exists(os.path.join(path, 'pyproject.toml')):
            return path
        if parent == path:
            raise RuntimeError(f'no pyproject.toml above {start}')
        path = parent


REPO_ROOT = _find_repo_root(__file__)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


def _set_gpus():
    for i, arg in enumerate(sys.argv):
        if arg == '--gpus' and i + 1 < len(sys.argv):
            os.environ['CUDA_VISIBLE_DEVICES'] = sys.argv[i + 1]


_set_gpus()
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
from jax import random  # noqa: E402
import gymnax  # noqa: E402

from source.envs.gymnax_classic import (  # noqa: E402
    make_gymnax_env, wrap_actions,
    FlipEnv, build_policy, gymnax_task_type, saved_task_rows, unflatten_params)
from source.envs.run_context import (  # noqa: E402
    RunContext, is_gymnax_run, live_states, run_config)
from source.metrics.behavioral_divergence import (  # noqa: E402
    disagreement_rate,
    mean_js_categorical,
    mean_kl_categorical,
    normalized_action_distance,
    rff_feature_space,
    subsample_states,
)

RUNS_ROOT = os.path.join(REPO_ROOT, "projects", "neurips_2026_rebuttal")

METHOD_STYLE = {
    "ga":     {"label": "GA",           "color": "#4CBB47"},
    "dns":    {"label": "GA + Novelty", "color": "#3B8FD4"},
    "es":     {"label": "ES",           "color": "#F0C33C"},
    "ppo":    {"label": "PPO",          "color": "#E8504F"},
    "trac":   {"label": "TRAC-PPO",     "color": "#F08C4B"},
    "redo":   {"label": "ReDo-PPO",     "color": "#8B4A2B"},
    "cchain": {"label": "C-CHAIN",      "color": "#7B5EA7"},
}

ENV_TITLES = {
    "CartPole-v1":    "Cart Pole",
    "Acrobot-v1":     "Acrobot",
    "MountainCar-v0": "Mountain Car",
}

# Which saved agent stands for "the policy at the end of sub-task tau". The
# first of these that a run carries is used, so NE runs (finalgen/incumbent)
# and gradient runs (final) are handled without a per-method switch.
SOURCE_PREFERENCE = ("finalgen", "final", "incumbent")

# Matrices, in reading order, with the colour scale each one needs: `zero` for
# magnitudes (a single-hue ramp from 0), `diverging` for signed quantities
# (symmetric about 0), `range` for a raw return, which has no meaningful zero
# and is negative throughout on Acrobot and Mountaincar.
MATRICES = [
    ("disagreement",    "Action disagreement", "Blues",   "zero"),
    ("occupancy_shift", "State-visitation MMD", "Purples", "zero"),
    ("retention",       "Reward vs. own agent", "RdBu",    "diverging"),
    ("reward",          "Return",               "Greens",  "range"),
]

plt.rcParams.update({
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "legend.frameon": False,
    "figure.dpi": 150,
})


def env_dir_name(env_name):
    return env_name.replace("-", "_")


# ============================================================================
# Run discovery
# ============================================================================

def find_runs(root, methods, envs, pop_sizes, trials, agent_source=None):
    """Every continual run under `root` that the filters accept.

    A run is a `results.json` next to a `checkpoints.npz` holding the sub-task
    sequence; anything else under the tree (the non-continual runs, the
    aggregate result files) has no `noise_vectors` and is skipped.

    Anything under an `archive_*` directory is skipped too. Those hold runs kept
    for the record but superseded -- `archive_sigma1.0/` is the flat-sigma set
    the per-env sigmas replaced -- and they are *not* comparable with current
    runs. They carry the same method, env and trial fields, so pooling them
    silently doubles a cell's trial count instead of failing.
    """
    runs = []
    for results_path in sorted(glob.glob(os.path.join(root, "**", "results.json"),
                                         recursive=True)):
        run_dir = os.path.dirname(results_path)
        if any(part.startswith("archive_")
               for part in os.path.relpath(run_dir, root).split(os.sep)):
            continue
        ckpt_path = os.path.join(run_dir, "checkpoints.npz")
        if not os.path.exists(ckpt_path):
            continue
        with open(results_path) as f:
            cfg = run_config(json.load(f))
        if "num_tasks" not in cfg or int(cfg.get("num_tasks", 0)) < 2:
            continue

        # The ARM is the method directory's name: on the shared runners the
        # config records the searcher (`openes` under `es/`), and every other
        # figure script names an arm by its directory.
        arm = os.path.basename(os.path.dirname(os.path.dirname(run_dir)))
        if methods and arm not in methods and cfg["method"] not in methods:
            continue
        # Matched against the *directory*, not against the env id in the config.
        # The continual tree tags the observation-noise sigma into the
        # directory name (`CartPole_v1_sigma2.0/`) and nothing in results.json
        # carries it, so filtering on the config alone cannot separate the
        # sigmas -- it would silently pool three different experiments. The
        # un-tagged name still matches every sigma, which is what a caller
        # asking for `CartPole_v1` means.
        if envs:
            run_env_dir = os.path.basename(os.path.dirname(run_dir))
            if run_env_dir not in envs and env_dir_name(cfg["env"]) not in envs:
                continue
        if pop_sizes and cfg.get("pop_size") not in pop_sizes:
            continue
        if trials and int(cfg["trial"]) not in trials:
            continue

        with np.load(ckpt_path) as data:
            names = set(data.files)
        if "noise_vectors" not in names:
            continue
        preference = ((agent_source,) + SOURCE_PREFERENCE if agent_source
                      else SOURCE_PREFERENCE)
        source = next((s for s in preference if s in names), None)
        if source is None:
            print(f"  ! {run_dir}: no agents among {sorted(names)}, skipping")
            continue
        if agent_source and source != agent_source and source != "final":
            # Asked for one agent and given another: that is a different
            # quantity, not a near miss, so say so rather than quietly
            # measuring the fallback. Gradient runs only ever carry `final`,
            # which IS their centroid and their elite alike, so that one is
            # not a fallback and is not reported.
            print(f"  ! {run_dir}: no '{agent_source}' agent, using '{source}'")

        runs.append({"cfg": cfg, "dir": run_dir, "ckpt": ckpt_path,
                     "source": source, "arm": arm})
    return runs


def load_agents(run, num_tasks, stock_params):
    """(agents, noise_vectors, task params) truncated to the sub-tasks analysed.

    The task sequence is a prefix-stable stream (see `task_noise_vectors`), so
    truncating a 20-sub-task run to 10 gives exactly the 10-sub-task
    experiment rather than a different one. The third element is one env
    params pytree per sub-task -- the body and action order that sub-task was
    trained under -- from `saved_task_params`; the offset stays in `noise`.
    """
    with np.load(run["ckpt"]) as data:
        agents = np.asarray(data[run["source"]], dtype=np.float32)
        noise = np.asarray(data["noise_vectors"], dtype=np.float32)
        # Only gymnax resolves a sub-task into an offset plus env params; on
        # the other suites the row of `noise_vectors` IS the sub-task and the
        # context takes it directly.
        if stock_params is not None:
            noise, bodies, _mults = saved_task_rows(run["cfg"], data, stock_params)
            noise = np.asarray(noise, dtype=np.float32)
        else:
            bodies = [None] * len(noise)
    t = min(num_tasks, agents.shape[0], noise.shape[0], len(bodies))
    return agents[:t], noise[:t], bodies[:t]


def stack_params(bodies):
    """Per-sub-task params pytrees -> one pytree with a leading sub-task axis."""
    if bodies[0] is None:
        return None
    return jax.tree_util.tree_map(lambda *xs: jnp.stack([jnp.asarray(x) for x in xs]),
                                  *bodies)


# ============================================================================
# Rollouts
# ============================================================================

class EnvContext:
    """Everything needed to roll a saved agent out on a saved sub-task.

    Built once per (env, architecture, episode budget) and reused across runs:
    the jit compilation dominates the cost of a single run.
    """

    def __init__(self, env_name, hidden_dims, episode_length, episodes,
                 task_type='noise'):
        env, env_params = make_gymnax_env(env_name)
        env_params = env_params.replace(max_steps_in_episode=episode_length)
        obs, _ = env.reset(random.key(0), env_params)
        self.obs_dim = int(obs.shape[-1])
        self.num_actions = int(env.action_space(env_params).n)
        self.continuous = False
        self.episode_length = episode_length
        self.episodes = episodes
        # The stock body. What a sub-task did to it -- nothing, a physics
        # rescaling or a reversed action order -- arrives per call as
        # `task_params` (see `saved_task_params`), so the jitted rollout is
        # compiled once per family and never bakes a sub-task in.
        self.stock_params = env_params
        self.task_type = task_type
        if task_type == 'actions':
            env = wrap_actions(env)      # AFTER the spaces are read, as in training

        policy, param_template, _ = build_policy(
            random.key(0), self.obs_dim, self.num_actions, hidden_dims)

        def rollout(flat_params, key, noise_vector, task_params):
            """One episode: the states acted on, which of them are real, the return.

            The scan always runs the full episode so the shape stays fixed;
            `alive` marks the steps before the first termination, so the states
            after it (a frozen final state, repeated) are excluded downstream.
            """
            params = unflatten_params(flat_params, param_template)
            obs, state = env.reset(key, task_params)

            def step_fn(carry, _):
                obs, state, total, done_flag, key = carry
                logits = policy.apply(params, obs + noise_vector)
                action = jnp.argmax(logits)
                key, step_key = random.split(key)
                next_obs, next_state, reward, done, _ = env.step(
                    step_key, state, action, task_params)
                total = total + reward * (1.0 - done_flag)
                emitted = (obs, 1.0 - done_flag)
                done_flag = jnp.maximum(done_flag, done.astype(jnp.float32))
                return (next_obs, next_state, total, done_flag, key), emitted

            (_, _, total, _, _), (obs_seq, alive) = jax.lax.scan(
                step_fn, (obs, state, 0.0, 0.0, key), None, length=episode_length)
            return obs_seq, alive, total

        episodes_of = jax.vmap(rollout, in_axes=(None, 0, None, None))

        # Both rollouts return `(states, occupancy, alive, returns)`, the
        # contract `RunContext` also keeps: on gymnax the state a policy acts
        # on and the feature its visitation is measured in are the same
        # observation, so it is returned twice.
        @jax.jit
        def roll_on_task(agents, key, noise_vector, task_params):
            """Every agent on one sub-task, on the *same* episode seeds.

            Sharing the seeds across agents means a difference between two rows
            is a difference between the policies, not between the initial
            states they happened to draw.
            """
            keys = random.split(key, episodes)
            obs, alive, total = jax.vmap(episodes_of, in_axes=(0, None, None, None))(
                agents, keys, noise_vector, task_params)
            return obs, obs, alive, total

        @jax.jit
        def roll_own_tasks(agents, key, noise_vectors, task_params):
            """Agent i on sub-task i, for every i: the diagonal of the matrix.

            `task_params` is the per-sub-task params STACKED along a leading
            axis, so the vmap pairs agent i with body i.
            """
            keys = random.split(key, episodes)
            obs, alive, total = jax.vmap(episodes_of, in_axes=(0, None, 0, 0))(
                agents, keys, noise_vectors, task_params)
            return obs, obs, alive, total

        @jax.jit
        def logits_of(agents, obs_batch, noise_vector):
            """(A, P) agents applied to (N, obs_dim) raw states, under sub-task
            `noise_vector`'s observation shift -> (A, N, num_actions)."""
            shifted = obs_batch + noise_vector

            def one(flat_params):
                return policy.apply(unflatten_params(flat_params, param_template),
                                    shifted)
            return jax.vmap(one)(agents)

        self.roll_on_task = roll_on_task
        self.roll_own_tasks = roll_own_tasks
        self.logits_of = logits_of


class SuiteContext:
    """`EnvContext`'s interface over `RunContext`, for every non-gymnax suite.

    A sub-task is the row of `noise_vectors` itself and there are no per-
    sub-task env params, so `stock_params` is None and the `bodies` the
    shared code threads through are ignored.
    """

    def __init__(self, cfg, episodes):
        self.ctx = RunContext(cfg, episodes)
        self.stock_params = None
        self.obs_dim = self.ctx.obs_dim
        self.num_actions = self.ctx.num_actions
        self.continuous = self.ctx.continuous
        self.episode_length = self.ctx.episode_length
        self.episodes = episodes
        self.task_type = cfg.get('task', {}).get('task_mod', 'env')

    def roll_on_task(self, agents, key, task, _body=None):
        return self.ctx.roll_on_task(agents, key, task)

    def roll_own_tasks(self, agents, key, tasks, _bodies=None):
        return self.ctx.roll_own_tasks(agents, key, tasks)

    def roll_cross(self, agents, keys, tasks, _bodies=None):
        return self.ctx.roll_cross(agents, keys, tasks)

    def logits_of(self, agents, states, _task=None):
        # Padded to a multiple of 256 states: D_i is smaller than
        # `--num_states` whenever an agent solves its room in a few steps
        # (every MiniGrid agent), and a jit keyed on the exact batch size
        # recompiled the conv policy for every distinct |D_i| -- up to 20
        # per run, 1,600 over a sweep, which was the whole running time.
        states = np.asarray(states)
        n = len(states)
        padded = -(-n // 256) * 256
        if padded > n:
            pad = np.repeat(states[:1], padded - n, axis=0)
            states = np.concatenate([states, pad], axis=0)
        return np.asarray(self.ctx.logits_of(agents, jnp.asarray(states)))[:, :n]


def make_context(cfg, episodes):
    """The right context for a run, keyed so runs of one shape share a jit."""
    if is_gymnax_run(cfg):
        key = ("gymnax", cfg["env"], tuple(cfg["hidden_dims"]),
               int(cfg["episode_length"]), gymnax_task_type(cfg))
        build = lambda: EnvContext(cfg["env"], tuple(cfg["hidden_dims"]),  # noqa: E731
                                   int(cfg["episode_length"]), episodes,
                                   gymnax_task_type(cfg))
    else:
        key = RunContext.cache_key(cfg, episodes)
        build = lambda: SuiteContext(cfg, episodes)                     # noqa: E731
    return key, build


def run_key(cfg, salt):
    """A key derived from the run's identity, not from its training RNG.

    Evaluation episodes are then reproducible from `results.json` alone and are
    uncorrelated with the episodes the agents were selected on.
    """
    base = abs(hash((cfg["method"], cfg["env"], cfg.get("pop_size"),
                     cfg["trial"], salt)))
    return random.key(base % (2 ** 31))


# ============================================================================
# The state sets D_i
# ============================================================================

STATE_BLOWUP = 1e3      # |input| beyond this is a diverging simulation, as in mjx.OBS_BLOWUP


def collect_state_sets(ctx, cfg, agents, noise, bodies, num_states, seed):
    """D_i for every sub-task: raw states visited by agent i on sub-task i.

    Raw, i.e. before the sub-task's observation shift is added, so that the same
    set can be replayed under any sub-task's shift. Subsampled to a fixed size
    because a policy that falls over after 40 steps would otherwise contribute
    40 states where one that survives 500 contributes 500, and the two
    divergences would be averages over differently sized, differently
    distributed sets.
    """
    states, occ, alive, _ = ctx.roll_own_tasks(
        agents, run_key(cfg, seed), jnp.asarray(noise), stack_params(bodies))
    states, occ, alive = np.asarray(states), np.asarray(occ), np.asarray(alive)

    sets, occ_sets, counts = [], [], []
    dropped = 0
    for i in range(len(agents)):
        visited = live_states(states[i], alive[i])
        # A diverged simulation is not a state. MJX has no NaN protection
        # (source/envs/mjx._reset_blown_up): under some ground-friction
        # multipliers the cheetah's contact solver diverges mid-episode,
        # `done` never fires, and the trace carries inf / NaN policy inputs
        # from there to the end of the episode. One such row makes every
        # action distance on D_i NaN, and the cheetah friction family lost
        # BD in 9 of 10 trials on every arm to it (2026-09-11). The test is
        # the trainer's: finite and below OBS_BLOWUP in magnitude.
        flat = visited.reshape(len(visited), -1)
        ok = np.all(np.isfinite(flat) & (np.abs(flat) < STATE_BLOWUP), axis=-1)
        dropped += int((~ok).sum())
        visited = visited[ok]
        counts.append(int(len(visited)))
        sets.append(subsample_states(visited, num_states, seed=seed))
        # The occupancy feature of the same visits, for fitting the shared
        # feature space. Same seed, same indices, so the two sets are one
        # sample of the same episodes.
        occ_sets.append(subsample_states(live_states(occ[i], alive[i])[ok],
                                         num_states, seed=seed))
    if dropped:
        print(f"    dropped {dropped} diverged-simulation states from the D_i")
    return sets, occ_sets, counts


# ============================================================================
# The matrices
# ============================================================================

_BD_UNDEFINED = []


def action_matrices(ctx, agents, state_sets, noise):
    """`disagreement`, `js_action`, `kl_action`: how differently agents *act*.

    Row i is measured on D_i under sub-task i's observation shift, so the
    question each row asks is the forgetting-relevant one -- would this agent
    still do what agent i did, in the situations agent i actually met?
    """
    n = len(agents)
    disagreement = np.zeros((n, n))
    js = np.zeros((n, n))
    kl = np.zeros((n, n))

    # A continuous-action policy (the mjx bodies) returns the executed action
    # in [-1, 1], so D(s) is the normalised distance between the two actions,
    # in [0, 1] like the discrete disagreement and equally temperature-free;
    # the two softmax diagnostics have no meaning there and are left NaN.
    continuous = getattr(ctx, "continuous", False)
    for i in range(n):
        try:
            logits = np.asarray(ctx.logits_of(jnp.asarray(agents),
                                              jnp.asarray(state_sets[i]),
                                              jnp.asarray(noise[i])))
        except NotImplementedError as err:
            # A suite whose trace cannot be fed back to the policy (Kinetix
            # keeps step features, not frames): BD is undefined there and left
            # NaN, which the tables print as `--`. F comes from the rollout
            # matrices and is unaffected.
            if not _BD_UNDEFINED:
                print(f"  BD left undefined on this suite: {err}")
                _BD_UNDEFINED.append(True)
            disagreement[:] = js[:] = kl[:] = float("nan")
            break
        for j in range(n):
            if i == j:
                continue
            if continuous:
                disagreement[i, j] = normalized_action_distance(logits[i],
                                                                logits[j])
                js[i, j] = kl[i, j] = float("nan")
                continue
            disagreement[i, j] = disagreement_rate(logits[i], logits[j])
            js[i, j] = mean_js_categorical(logits[i], logits[j])
            kl[i, j] = mean_kl_categorical(logits[i], logits[j])

    return {"disagreement": disagreement, "js_action": js, "kl_action": kl}


def make_feature_fn(space):
    """Mean random-Fourier feature per agent, computed on device.

    `lax.map` rather than a vmap over the agent axis: the intermediate is
    (episodes, steps, rff_dim) per agent, which is small one agent at a time and
    hundreds of megabytes for twenty at once.
    """
    mean = jnp.asarray(space["mean"], dtype=jnp.float32)
    std = jnp.asarray(space["std"], dtype=jnp.float32)
    W = jnp.asarray(space["W"], dtype=jnp.float32)
    b = jnp.asarray(space["b"], dtype=jnp.float32)
    scale = float(np.sqrt(2.0 / space["rff_dim"]))

    @jax.jit
    def mean_features(obs, alive):
        def per_agent(pair):
            o, a = pair
            feats = scale * jnp.cos(((o - mean) / std) @ W + b)
            return (feats * a[..., None]).sum(axis=(0, 1)) / jnp.maximum(a.sum(), 1.0)
        return jax.lax.map(per_agent, (obs, alive))

    return mean_features


def rollout_matrices(ctx, cfg, agents, noise, bodies, mean_features, seed):
    """`reward`, `retention` and `occupancy_shift`, from one T x T rollout sweep.

    Row i rolls every agent on sub-task i. That single sweep gives the return
    each agent gets there (the forgetting ground truth) and the visitation
    distribution each one induces there (whose distance from agent i's is the
    occupancy shift), so the two matrices cost one pass rather than two.
    """
    n = len(agents)
    reward = np.zeros((n, n))
    occupancy = np.zeros((n, n))
    agents_d = jnp.asarray(agents)

    if hasattr(ctx, "roll_cross"):
        # The suite contexts roll the whole T x T sweep in ONE call, under
        # the same per-sub-task keys the loop below would use. On the MJX
        # bodies twenty serial calls of a 1000-step scan were the whole cost
        # of this pass (2026-09-11: no run finished in an hour on a card
        # shared eight ways); the sweep batched is one launch sequence.
        keys = jnp.stack([run_key(cfg, seed + 1000 + i) for i in range(n)])
        occ_all, alive_all, ret_all = ctx.roll_cross(
            agents_d, keys, jnp.asarray(noise), bodies)
        for i in range(n):
            reward[i] = np.asarray(ret_all[i]).mean(axis=1)
            phi = np.asarray(mean_features(occ_all[i], alive_all[i]))
            occupancy[i] = np.linalg.norm(phi - phi[i], axis=-1)
        retention = reward - reward.diagonal()[:, None]
        return {"reward": reward, "retention": retention,
                "occupancy_shift": occupancy}

    for i in range(n):
        _states, occ, alive, returns = ctx.roll_on_task(
            agents_d, run_key(cfg, seed + 1000 + i), jnp.asarray(noise[i]),
            bodies[i])
        reward[i] = np.asarray(returns).mean(axis=1)
        phi = np.asarray(mean_features(occ, alive))
        occupancy[i] = np.linalg.norm(phi - phi[i], axis=-1)

    retention = reward - reward.diagonal()[:, None]
    return {"reward": reward, "retention": retention,
            "occupancy_shift": occupancy}


# ============================================================================
# Summaries
# ============================================================================

def spearman(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    if len(a) < 3:
        return float("nan")
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean()
    rb -= rb.mean()
    denom = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / denom) if denom > 0 else float("nan")


def consecutive(matrix):
    """The first sub-diagonal: what happened at each task switch."""
    return np.array([matrix[i, i + 1] for i in range(len(matrix) - 1)])


def summarise_run(matrices, state_counts):
    """The few numbers per run that the tables and the correlation need."""
    n = len(matrices["disagreement"])
    last = n - 1
    off = ~np.eye(n, dtype=bool)
    # `final_forgetting` is the literature's F (Lopez-Paz & Ranzato 2017's
    # backward transfer, negated; Continual World's F): the END-of-sequence
    # agent on every earlier sub-task, against the agent that had just been
    # trained there. The paper tables report it when a run visits more than
    # two distinct sub-tasks, and `switch_forgetting` when it alternates two
    # (`make_lineplot.load_divergence`, since 2026-09-11).
    forgotten = -matrices["retention"][:last, last]     # reward lost by the end
    # `switch_forgetting` is what ONE switch costs, at every switch: sub-task
    # i's own agent against the agent one sub-task later, on sub-task i. The
    # mirror of zero-shot transfer (agent i on sub-task i+1 BEFORE training
    # there). Under a two-regime alternation the final agent only measures
    # forgetting of the regime it was NOT just trained on, so there this is
    # the F the tables report.
    return {
        "min_states": int(min(state_counts)),
        "consecutive_disagreement": float(np.mean(consecutive(matrices["disagreement"]))),
        "consecutive_occupancy": float(np.mean(consecutive(matrices["occupancy_shift"]))),
        "drift_from_first": float(matrices["disagreement"][0, last]),
        "switch_forgetting": float(np.mean(-consecutive(matrices["retention"]))),
        "final_forgetting": float(np.mean(forgotten)),
        "rho_disagreement_forgetting": spearman(
            matrices["disagreement"][off], -matrices["retention"][off]),
        "rho_occupancy_forgetting": spearman(
            matrices["occupancy_shift"][off], -matrices["retention"][off]),
    }


def group_key(record):
    return (record["env"], record["method"], record["pop_size"])


def group_records(records):
    """Runs grouped into conditions, each holding its trials' matrices."""
    groups = {}
    for record in records:
        groups.setdefault(group_key(record), []).append(record)
    return groups


def mean_matrix(trials, name):
    """Trial-mean matrix, truncated to the shortest run in the condition."""
    n = min(len(t["matrices"][name]) for t in trials)
    return np.mean([np.asarray(t["matrices"][name])[:n, :n] for t in trials], axis=0)


# ============================================================================
# Figures
# ============================================================================

def condition_label(env, method, pop_size, show_pop):
    label = METHOD_STYLE.get(method, {}).get("label", method.upper())
    return f"{label} (N={pop_size})" if show_pop and pop_size else label


def heatmap_figure(groups, env, output_path, show_pop):
    """One column per condition, one row per matrix, averaged over trials."""
    conditions = [k for k in groups if k[0] == env]
    if not conditions:
        return
    conditions.sort(key=lambda k: (list(METHOD_STYLE).index(k[1])
                                   if k[1] in METHOD_STYLE else 99, k[2] or 0))

    ncols, nrows = len(conditions), len(MATRICES)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.1 * ncols, 3.0 * nrows),
                             squeeze=False)

    for row, (name, title, cmap, scale) in enumerate(MATRICES):
        mats = [mean_matrix(groups[k], name) for k in conditions]
        if scale == "diverging":
            lim = max(np.abs(m).max() for m in mats) or 1.0
            vmin, vmax = -lim, lim
        elif scale == "range":
            vmin = min(m.min() for m in mats)
            vmax = max(m.max() for m in mats)
            if vmax <= vmin:
                vmax = vmin + 1.0
        else:
            vmin, vmax = 0.0, max(m.max() for m in mats) or 1.0

        for col, (key, mat) in enumerate(zip(conditions, mats)):
            ax = axes[row][col]
            im = ax.imshow(mat, cmap=cmap, vmin=vmin, vmax=vmax,
                           interpolation="nearest")
            n = len(mat)
            if row == 0:
                ax.set_title(condition_label(*key, show_pop), fontweight="bold")
            if col == 0:
                ax.set_ylabel(f"{title}\n\nsub-task i (states)")
            if row == nrows - 1:
                ax.set_xlabel("agent from sub-task j")
            ticks = list(range(0, n, max(1, n // 10)))
            ax.set_xticks(ticks)
            ax.set_yticks(ticks)
            ax.tick_params(labelsize=7)
            # Annotating every cell is unreadable past ~10 sub-tasks, and the
            # colourbar carries the magnitude anyway.
            if n <= 8:
                # Reward is in the hundreds where the two behavioural matrices
                # are fractions, so the cell text follows the spread of the row
                # rather than its top value -- a return that runs -500 to -86
                # still wants no decimals.
                fmt = "{:.0f}" if (vmax - vmin) >= 10 else "{:.2f}"
                for i in range(n):
                    for j in range(n):
                        v = mat[i, j]
                        # Diverging cells are pale in the middle and saturated at
                        # both ends; the other scales only darken upwards.
                        on_dark = (abs(v) > 0.6 * vmax if scale == "diverging"
                                   else v > vmin + 0.6 * (vmax - vmin))
                        ax.text(j, i, fmt.format(v), ha="center", va="center",
                                fontsize=6,
                                color="white" if on_dark else "black")
            if col == ncols - 1:
                fig.colorbar(im, ax=axes[row].tolist(), shrink=0.85, pad=0.02)

    fig.suptitle(f"{ENV_TITLES.get(env, env)}: behavioural divergence between "
                 f"sub-tasks", fontweight="bold")
    fig.savefig(output_path, bbox_inches="tight")
    fig.savefig(output_path.replace(".pdf", ".png"), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def _band(ax, x, curves, color, label):
    """Mean over trials with a 95% CI band."""
    stacked = np.stack(curves)
    mean = stacked.mean(axis=0)
    if len(stacked) > 1:
        half = 1.96 * stacked.std(axis=0, ddof=1) / np.sqrt(len(stacked))
        ax.fill_between(x, mean - half, mean + half, color=color, alpha=0.18,
                        linewidth=0)
    ax.plot(x, mean, color=color, linewidth=1.8, label=label)


def consecutive_figure(groups, envs, output_path, show_pop):
    """Divergence at each task switch (top) and away from sub-task 0 (bottom)."""
    envs = [e for e in envs if any(k[0] == e for k in groups)]
    if not envs:
        return
    fig, axes = plt.subplots(2, len(envs), figsize=(3.3 * len(envs), 5.0),
                             squeeze=False, sharex="col")

    for col, env in enumerate(envs):
        for key in sorted((k for k in groups if k[0] == env),
                          key=lambda k: (list(METHOD_STYLE).index(k[1])
                                         if k[1] in METHOD_STYLE else 99, k[2] or 0)):
            trials = groups[key]
            color = METHOD_STYLE.get(key[1], {}).get("color", "#777777")
            label = condition_label(*key, show_pop)

            n = min(len(t["matrices"]["disagreement"]) for t in trials)
            switch = [consecutive(np.asarray(t["matrices"]["disagreement"])[:n, :n])
                      for t in trials]
            drift = [np.asarray(t["matrices"]["disagreement"])[0, 1:n] for t in trials]
            _band(axes[0][col], np.arange(1, n), switch, color, label)
            _band(axes[1][col], np.arange(1, n), drift, color, label)

        axes[0][col].set_title(ENV_TITLES.get(env, env), fontweight="bold")
        axes[1][col].set_xlabel("sub-task $\\tau$")
        for ax in (axes[0][col], axes[1][col]):
            ax.spines[["top", "right"]].set_visible(False)
            ax.set_ylim(0, None)
            ax.grid(alpha=0.25, linewidth=0.5)

    axes[0][0].set_ylabel("divergence at the switch\n$\\tau-1 \\rightarrow \\tau$")
    axes[1][0].set_ylabel("divergence from\nthe sub-task 0 agent")
    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        # Above the panels, not on them: with seven methods the legend wraps to
        # two rows and would otherwise land on the first row's titles.
        fig.legend(handles, labels, loc="upper center", ncol=min(len(labels), 4),
                   bbox_to_anchor=(0.5, 1.13))
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    fig.savefig(output_path.replace(".pdf", ".png"), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def validation_figure(groups, envs, output_path, show_pop):
    """Divergence against the reward it costs, over every off-diagonal pair.

    The point of the figure is that the behavioural matrices are not a
    free-standing curiosity: if they do not track the return the run gives up on
    a sub-task, they are not measuring forgetting.
    """
    envs = [e for e in envs if any(k[0] == e for k in groups)]
    if not envs:
        return
    fig, axes = plt.subplots(1, len(envs), figsize=(3.3 * len(envs), 3.0),
                             squeeze=False)

    for col, env in enumerate(envs):
        ax = axes[0][col]
        xs, ys = [], []
        for key in sorted(k for k in groups if k[0] == env):
            color = METHOD_STYLE.get(key[1], {}).get("color", "#777777")
            for trial in groups[key]:
                bd = np.asarray(trial["matrices"]["disagreement"])
                lost = -np.asarray(trial["matrices"]["retention"])
                off = ~np.eye(len(bd), dtype=bool)
                ax.scatter(bd[off], lost[off], s=6, alpha=0.35, color=color,
                           linewidths=0)
                xs.append(bd[off])
                ys.append(lost[off])
            ax.scatter([], [], s=18, color=color,
                       label=condition_label(*key, show_pop))
        rho = spearman(np.concatenate(xs), np.concatenate(ys)) if xs else float("nan")
        ax.set_title(f"{ENV_TITLES.get(env, env)}  ($\\rho$ = {rho:+.2f})",
                     fontweight="bold")
        ax.set_xlabel("action disagreement")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(alpha=0.25, linewidth=0.5)
        if col == 0:
            ax.set_ylabel("reward lost on sub-task $i$")
        ax.legend(fontsize=7, loc="lower right")

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    fig.savefig(output_path.replace(".pdf", ".png"), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


# ============================================================================
# Entry point
# ============================================================================

SUMMARY_COLUMNS = [
    ("consecutive_disagreement", "BD at the switch"),
    ("drift_from_first", "BD from sub-task 0"),
    ("consecutive_occupancy", "Occupancy MMD at the switch"),
    ("switch_forgetting", "Reward lost at the switch"),
    ("final_forgetting", "Reward lost by the end"),
    ("rho_disagreement_forgetting", "rho(BD, reward lost)"),
    ("rho_occupancy_forgetting", "rho(MMD, reward lost)"),
    ("min_states", "min &#124;D_i&#124;"),
]


def write_tables(groups, path, show_pop, args):
    lines = [
        "# Forgetting as behavioural divergence between sub-tasks",
        "",
        "`BD` is the action disagreement rate between the agent of one sub-task",
        "and the agent of another, measured on the states the earlier agent",
        "visited, under the earlier sub-task's observation shift. `Occupancy MMD`",
        "is the shift in state visitation over the same pair. `Reward lost by the",
        "end` is how much return the final agent gives up on the earlier",
        "sub-tasks, averaged over them -- forgetting as the reward curve sees it.",
        "",
        "The last two columns are the ones that decide whether the behavioural",
        "measures are worth reporting: they rank every ordered sub-task pair of a",
        "run by divergence and by reward lost, and correlate the two rankings.",
        "",
        f"Estimated from {args.episodes} episodes per (agent, sub-task) and "
        f"|D_i| = {args.num_states} states; means over trials, +- s.e.m.",
        "",
        "| Task | Method | Trials | " + " | ".join(c[1] for c in SUMMARY_COLUMNS) + " |",
        "|---" * (len(SUMMARY_COLUMNS) + 3) + "|",
    ]

    for key in sorted(groups, key=lambda k: (k[0], list(METHOD_STYLE).index(k[1])
                                             if k[1] in METHOD_STYLE else 99, k[2] or 0)):
        trials = groups[key]
        cells = []
        for name, _ in SUMMARY_COLUMNS:
            values = np.array([t["summary"][name] for t in trials], dtype=float)
            values = values[np.isfinite(values)]
            if len(values) == 0:
                cells.append("--")
                continue
            if name == "min_states":
                # The worst-estimated row of the condition; averaging a minimum
                # over trials would hide exactly the case this column is for.
                cells.append(f"{int(values.min())}")
                continue
            # Counts and returns run into the hundreds where the divergences are
            # fractions; three decimals on the former is noise.
            digits = 0 if np.abs(values).max() >= 10 else 3
            if len(values) == 1:
                cells.append(f"{values[0]:.{digits}f}")
            else:
                sem = values.std(ddof=1) / np.sqrt(len(values))
                cells.append(f"{values.mean():.{digits}f} ± {sem:.{digits}f}")
        lines.append(f"| {ENV_TITLES.get(key[0], key[0])} "
                     f"| {condition_label(*key, show_pop)} | {len(trials)} | "
                     + " | ".join(cells) + " |")

    lines += ["", "## Diagnostics kept out of the figures", "",
              "`js_action` and `kl_action` are in the .npz. They are comparable",
              "within a method and not across: the evolved policies are",
              "deterministic, so softmaxing their logits at temperature 1 gives a",
              "divergence that scales with the magnitude of the evolved weights.",
              ""]

    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f"Saved: {path}")


def write_outputs(records, results_dir, args):
    """The .npz, the .json, the tables and every figure, from the records.

    Split out so that `--figures_only` can redraw from a finished .npz. The
    matrices cost a full rollout sweep over every run to produce and nothing
    about a figure changes them, so iterating on a plot should not mean
    re-measuring 200 runs.
    """
    groups = group_records(records)
    # A population size in the label is only informative when the runs differ
    # in it. The gradient methods carry None (one agent, no population), which
    # is not a second value in that sense.
    show_pop = len({k[2] for k in groups if k[2] is not None}) > 1
    envs = sorted({k[0] for k in groups})

    npz_path = os.path.join(results_dir, "behavioural_divergence.npz")
    payload = {}
    for idx, record in enumerate(records):
        for name, matrix in record["matrices"].items():
            payload[f"run{idx}_{name}"] = np.asarray(matrix, dtype=np.float32)
    np.savez_compressed(
        npz_path,
        index=np.array([json.dumps({k: v for k, v in r.items()
                                    if k != "matrices"}) for r in records]),
        **payload)
    print(f"\nSaved: {npz_path}")

    json_path = os.path.join(results_dir, "behavioural_divergence.json")
    with open(json_path, "w") as f:
        json.dump({"config": vars(args),
                   "runs": [{k: v for k, v in r.items() if k != "matrices"}
                            for r in records]}, f, indent=2)
    print(f"Saved: {json_path}")

    write_tables(groups, os.path.join(results_dir, "behavioural_divergence.md"),
                 show_pop, args)

    for env in envs:
        heatmap_figure(groups, env, os.path.join(
            results_dir, f"bd_heatmaps_{env_dir_name(env)}.pdf"), show_pop)
    consecutive_figure(groups, envs, os.path.join(
        results_dir, "bd_consecutive.pdf"), show_pop)
    validation_figure(groups, envs, os.path.join(
        results_dir, "bd_vs_forgetting.pdf"), show_pop)


def load_records(results_dir):
    """Rebuild the records of a previous run from its .npz."""
    path = os.path.join(results_dir, "behavioural_divergence.npz")
    if not os.path.exists(path):
        raise SystemExit(f"--figures_only needs {path}, which does not exist")

    records = []
    with np.load(path, allow_pickle=False) as data:
        for idx, meta in enumerate(data["index"]):
            record = json.loads(str(meta))
            record["matrices"] = {
                name: data[f"run{idx}_{name}"]
                for _, name in ((k, k.split("_", 1)[1]) for k in data.files
                                if k.startswith(f"run{idx}_"))
            }
            records.append(record)
    print(f"Loaded {len(records)} run(s) from {path}")
    return records


def main():
    parser = argparse.ArgumentParser(
        description="Behavioural divergence between the sub-task agents of a "
                    "continual run, as heatmaps over every sub-task pair")
    parser.add_argument("--runs_root", default=RUNS_ROOT,
                        help="Directory walked for continual runs "
                             "(results.json + checkpoints.npz)")
    parser.add_argument("--results_dir", default=None)
    parser.add_argument("--methods", nargs="+", default=None)
    parser.add_argument("--envs", nargs="+", default=None,
                        help="Env directory names, e.g. CartPole_v1")
    parser.add_argument("--agent_source", default=None,
                        choices=("finalgen", "incumbent", "final", "centroid"),
                        help="Which saved agent stands for the policy at the "
                             "end of a sub-task. Default follows "
                             "SOURCE_PREFERENCE, which takes `finalgen` -- the "
                             "population's best member. Pass `incumbent` for "
                             "its mean agent (elite mean for GA, distribution "
                             "mean for ES); that is a different measurement, "
                             "so give it its own --results_dir.")
    parser.add_argument("--pop_sizes", nargs="+", type=int, default=None)
    parser.add_argument("--trials", nargs="+", type=int, default=None)
    parser.add_argument("--num_tasks", type=int, default=10,
                        help="Sub-tasks analysed; longer runs are truncated to "
                             "their first N, which is the same experiment")
    parser.add_argument("--episodes", type=int, default=20,
                        help="Evaluation episodes per (agent, sub-task)")
    parser.add_argument("--num_states", type=int, default=2000,
                        help="|D_i| after subsampling")
    parser.add_argument("--rff_dim", type=int, default=512,
                        help="Random Fourier features for the occupancy MMD")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gpus", default=None, help="CUDA_VISIBLE_DEVICES")
    parser.add_argument("--figures_only", action="store_true",
                        help="Redraw the tables and figures from the .npz of a "
                             "previous run instead of re-measuring every run")
    parser.add_argument("--merge", nargs="+", default=None, metavar="DIR",
                        help="Concatenate the records of these results "
                             "directories (each a finished .npz, e.g. one "
                             "--methods shard per GPU) and write the joint "
                             "outputs to --results_dir. Nothing is "
                             "re-measured: an MJX sweep over a cell is hours "
                             "on one card and minutes across eight, and the "
                             "records are per run, so sharding by method and "
                             "merging is the same sweep in a different order.")
    args = parser.parse_args()

    results_dir = args.results_dir or os.path.join(RUNS_ROOT, "results",
                                                   "gymnax_continual")
    os.makedirs(results_dir, exist_ok=True)

    if args.figures_only:
        write_outputs(load_records(results_dir), results_dir, args)
        return
    if args.merge:
        records = [r for d in args.merge for r in load_records(d)]
        seen = {}
        for r in records:
            seen.setdefault(r["run_dir"], 0)
            seen[r["run_dir"]] += 1
        dupes = sorted(k for k, n in seen.items() if n > 1)
        if dupes:
            raise SystemExit(f"--merge: {len(dupes)} run(s) appear in more than "
                             f"one shard, e.g. {dupes[0]}")
        write_outputs(records, results_dir, args)
        return

    runs = find_runs(args.runs_root, set(args.methods or []), set(args.envs or []),
                     set(args.pop_sizes or []), set(args.trials or []),
                     agent_source=args.agent_source)
    if not runs:
        raise SystemExit(f"No continual runs found under {args.runs_root}")

    by_env = {}
    for run in runs:
        by_env.setdefault(run["cfg"]["env"], []).append(run)
    print(f"{len(runs)} run(s) over {len(by_env)} task(s)")

    records = []
    for env_name, env_runs in sorted(by_env.items()):
        print(f"\n=== {env_name} ({len(env_runs)} runs) ===")
        contexts = {}
        loaded = []

        # Pass 1: D_i for every run. Pooling these is what makes the occupancy
        # feature map shared -- fitted per task, on every method's states, and
        # then reused unchanged, so it cannot flatter whichever method it saw
        # most of.
        for run in env_runs:
            cfg = run["cfg"]
            # task_type is part of the key: an `actions` run needs the
            # FlipEnv wrapper and must not share a jitted rollout with the
            # stock env.
            ctx_key, build = make_context(cfg, args.episodes)
            if ctx_key not in contexts:
                contexts[ctx_key] = build()
            ctx = contexts[ctx_key]
            agents, noise, bodies = load_agents(run, args.num_tasks,
                                                ctx.stock_params)

            sets, occ_sets, counts = collect_state_sets(
                ctx, cfg, agents, noise, bodies, args.num_states, args.seed)
            short = [c for c in counts if c < args.num_states]
            if short:
                # An agent that fails immediately on its own sub-task visits a
                # handful of near-identical states, and its row of the matrix is
                # an average over those few states rather than over |D_i|. The
                # counts are carried through to the outputs so such rows can be
                # recognised rather than read as if they were as well estimated
                # as the rest.
                print(f"  ! {os.path.relpath(run['dir'], args.runs_root)}: "
                      f"{len(short)}/{len(counts)} sub-tasks gave fewer than "
                      f"{args.num_states} states (min {min(short)})")
            loaded.append((run, ctx, agents, noise, bodies, sets, occ_sets,
                           counts))

        space = rff_feature_space(
            np.concatenate([s for *_, occ_sets, _ in loaded for s in occ_sets]),
            seed=args.seed, rff_dim=args.rff_dim)
        print(f"  shared occupancy space: D={args.rff_dim}, "
              f"sigma={space['sigma']:.4f}")
        mean_features = make_feature_fn(space)

        # Pass 2: the matrices.
        for run, ctx, agents, noise, bodies, sets, _occ, counts in loaded:
            cfg = run["cfg"]
            matrices = action_matrices(ctx, agents, sets, noise)
            matrices.update(rollout_matrices(ctx, cfg, agents, noise, bodies,
                                             mean_features, args.seed))
            record = {
                "method": run["arm"],
                "env": env_name,
                "pop_size": cfg.get("pop_size"),
                "trial": int(cfg["trial"]),
                "num_tasks": int(len(agents)),
                "source": run["source"],
                "run_dir": os.path.relpath(run["dir"], REPO_ROOT),
                "states_available": counts,
                "matrices": {k: v.tolist() for k, v in matrices.items()},
                "summary": summarise_run(matrices, counts),
            }
            records.append(record)
            s = record["summary"]
            print(f"  {run['arm']:>6} N={str(cfg.get('pop_size')):>4} "
                  f"trial {cfg['trial']:>2} | BD switch {s['consecutive_disagreement']:.3f} "
                  f"| BD from 0 {s['drift_from_first']:.3f} "
                  f"| lost/switch {s['switch_forgetting']:.1f} "
                  f"| lost by end {s['final_forgetting']:.1f} "
                  f"| rho {s['rho_disagreement_forgetting']:+.2f}")

    write_outputs(records, results_dir, args)


if __name__ == "__main__":
    main()
