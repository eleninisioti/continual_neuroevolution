"""Behaviour descriptors for the gymnax and mujoco control tasks, and the
population diversity statistics computed from them.

Why more than one descriptor
----------------------------
"How diverse is this population?" has no single answer: every diversity number
is a *choice of behaviour space*, and different choices disagree. Parameter
distance is the weakest of them -- two genomes can differ everywhere and act
identically -- and the action distribution on a fixed probe batch is better but
blind to policies that differ only on a small set of critical states. So this
module computes several descriptor families side by side, from the same
rollouts, and every trainer logs all of them:

| family        | what it is                                             | space it lives in | comparable across methods? |
|---------------|--------------------------------------------------------|-------------------|----------------------------|
| `handcrafted` | expert descriptor per task (where the agent ends up,    | fixed, [0, 1]^d   | yes                        |
|               | how high it swings, how much it moves)                  |                   |                            |
| `occupancy`   | discretised state-visitation histogram over the two     | fixed simplex     | yes                        |
|               | interpretable state dimensions                          |                   |                            |
| `action_freq` | proportion of time each discrete action is used         | fixed simplex     | yes                        |
| `action_usage`| mean absolute torque per actuator (continuous tasks)    | fixed, [0, 1]^nu  | yes                        |
| `aurora`      | latent code of an auto-encoder trained online on the    | learned, drifts   | no (see below)             |
|               | observation trajectories                                |                   |                            |
| `probe`       | action distribution on a frozen batch of probe states   | fixed simplex     | yes                        |
| genomic       | distance in parameter space (computed by the trainers)  | parameter space   | not behavioural            |

`handcrafted` is the quality-diversity convention (Cully et al., *Robots that
can adapt like animals*, Nature 2015): a short, expert-designed vector that
names the aspect of behaviour we care about. `occupancy` is the state-occupancy
view (Hedayatian & Nikolaidis 2026; Fraschini et al. 2026) projected onto the
two state dimensions that carry the task semantics -- a policy's occupancy
measure is what its behaviour *is*, so two policies that visit the same states
in the same proportions are behaviourally identical no matter what their
parameters or action logits look like. `action_freq` is the discrete-action
analogue of the duty-factor style descriptors used in QD. `aurora` is the
unsupervised alternative (Grillotti & Cully, IEEE TEC 2022), and `probe` is the
action-distribution proxy (Pacchiano et al., ICML 2020) the paper used on its
own.

The AURORA caveat: its latent space is learned per run, so its *scale* is not
shared between two runs and `bd_aurora_diversity` must not be compared across
methods or seeds directly. Within a run it is a valid trend. For cross-method
comparison, re-encode the saved behaviour snapshots with one encoder trained on
the pooled trajectories -- that is what
`scripts/neurips_2026_rebuttal/behaviour_diversity_analysis.py` does.

Not every family exists for every task
--------------------------------------
A task declares which families it supports through its `EnvBehaviourSpec`, and
the ones it leaves out are simply absent from the metrics dict rather than
filled with a placeholder -- a missing row in the figure is honest, an
incomparable number silently plotted next to comparable ones is not.

* `occupancy` needs two interpretable state dimensions to bin over. Tasks that
  set `coords_fn=None` (CheetahRun) skip it.
* `action_freq` is defined only for discrete action spaces. Continuous-action
  tasks get `action_usage` instead, and the two are *different measures* under
  different keys: JS distance between action histograms vs Euclidean distance
  between mean-|torque| vectors. They must not be plotted as one row.
* `probe` likewise splits: discrete policies are compared by the JS divergence
  of their softmax over a frozen probe batch (`bd_probe_js`), continuous ones by
  the Euclidean distance between the action vectors they emit on it
  (`bd_probe_action_dist`).

Units
-----
* `handcrafted`, `aurora`, `action_usage`: mean pairwise Euclidean distance.
  `handcrafted` and `action_usage` are normalised to [0, 1] per dimension first,
  so their scale is task-independent.
* `occupancy`, `action_freq`: mean pairwise Jensen-Shannon *distance*
  (sqrt of the base-2 divergence), in [0, 1].
* `probe` (discrete): mean pairwise Jensen-Shannon *divergence* in nats, the
  same formula as `source/studies/gymnax/pbt_diagnostics.py`, so the numbers stay
  comparable with what the paper already reports; plus the fraction of probe
  states on which two policies pick different greedy actions.
* `probe` (continuous): mean pairwise Euclidean distance between action vectors
  on the probe batch, plus the mean per-actuator absolute difference.

All pairwise statistics sub-sample the population (`max_n`, default 128) so the
cost does not grow with population size.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Callable

import jax
import jax.numpy as jnp

from source.metrics.probe import probe_flatten, probe_index, probe_size
import numpy as np


Array = Any


# ============================================================================
# Per-task descriptor definitions
# ============================================================================


def _masked_mean(values, weights):
    """Mean of `values` over the valid steps (weights are 0/1)."""
    total = jnp.sum(weights)
    return jnp.sum(values * weights) / jnp.maximum(total, 1.0)


def _masked_max(values, weights):
    return jnp.max(jnp.where(weights > 0, values, -jnp.inf))


def _last_valid(obs, weights):
    """Observation at the last valid step of the episode."""
    last = jnp.maximum(jnp.sum(weights).astype(jnp.int32) - 1, 0)
    return jnp.take(obs, last, axis=0)


def _cartpole_descriptor(obs, weights, aux):
    """Where the cart sits, how much it moves, how far the pole leans.

    None of the three is the return, so this stays a behaviour descriptor
    rather than a re-labelled fitness.
    """
    del aux
    return jnp.stack([
        _masked_mean(obs[:, 0], weights),          # mean cart position
        _masked_mean(jnp.abs(obs[:, 1]), weights),  # mean |cart velocity|
        _masked_mean(jnp.abs(obs[:, 2]), weights),  # mean |pole angle|
    ])


def _mountaincar_descriptor(obs, weights, aux):
    """Classic location descriptor: how far up the hill the car gets."""
    del aux
    return jnp.stack([
        _last_valid(obs, weights)[0],               # final position
        _masked_max(obs[:, 0], weights),            # furthest position reached
        _masked_mean(jnp.abs(obs[:, 1]), weights),  # mean |velocity|
    ])


def _acrobot_height(obs):
    """Height of the tip: -cos(t1) - cos(t1 + t2), from the cos/sin obs."""
    cos1, sin1, cos2, sin2 = obs[:, 0], obs[:, 1], obs[:, 2], obs[:, 3]
    cos12 = cos1 * cos2 - sin1 * sin2
    return -cos1 - cos12


def _acrobot_descriptor(obs, weights, aux):
    """How high the arm swings and how fast the first joint spins."""
    del aux
    height = _acrobot_height(obs)
    return jnp.stack([
        _masked_max(height, weights),
        _masked_mean(height, weights),
        _masked_mean(jnp.abs(obs[:, 4]), weights),
    ])


def _cheetah_descriptor(obs, weights, aux):
    """Duty factor of each foot: the fraction of the episode it is on the ground.

    The standard quality-diversity descriptor for a legged robot (Cully et al.
    2015 use exactly this for the hexapod; QDax's `halfcheetah_uni` uses the
    two-foot version). It describes *gait* -- a bounding gait, a crawl and a
    front-foot drag have visibly different duty factors -- while saying nothing
    about how fast the cheetah goes, which matters here because CheetahRun's
    reward is forward speed alone. A descriptor built on `obs[8]` (the root x
    velocity) would be the return under another name.

    `aux["foot_contact"]` is (T, 2) in {0, 1} for (back foot, front foot); the
    trainers derive it geometrically, see `foot_contact_from_geoms`.
    """
    del obs
    contact = aux["foot_contact"]
    return jnp.stack([
        _masked_mean(contact[:, 0], weights),
        _masked_mean(contact[:, 1], weights),
    ])


def _ant_descriptor(obs, weights, aux):
    """Duty factor of each of the ant's four feet.

    The same descriptor as `_cheetah_descriptor`, one dimension per leg. Cully
    et al. (2015) use exactly this for their six-legged robot, and it is what
    QDax's `ant_uni` measures: a trot, a pace and a three-legged limp are
    different gaits with different duty-factor vectors, and none of the four
    numbers is the forward velocity the ant is rewarded for.

    `aux["foot_contact"]` is (T, 4) in {0, 1}, ordered as
    `source/studies/brax/behaviour_brax.py:ANT_FOOT_GEOMS`. The four legs are named
    after the geoms in brax's ant.xml rather than by compass direction, because
    the xml's own names do not agree with the geometry: `right_ankle` sits at
    (-x, +y) and `back_leg` at (-x, -y).
    """
    del obs
    contact = aux["foot_contact"]
    return jnp.stack([_masked_mean(contact[:, i], weights) for i in range(4)])


def _cartpole_coords(obs):
    return jnp.stack([obs[:, 0], obs[:, 2]], axis=-1)     # cart position, pole angle


def _mountaincar_coords(obs):
    return obs[:, :2]                                      # position, velocity


def _deepsea_coords(obs):
    """(T, 2) row and column of a flat one-hot DeepSea observation, each
    scaled to [0, 1] by the grid size so one spec serves every size."""
    n = int(round(float(np.sqrt(obs.shape[-1]))))
    idx = jnp.argmax(obs, axis=-1)
    return jnp.stack([idx // n, idx % n], axis=-1) / max(n - 1, 1)


def _deepsea_descriptor(obs, weights, aux):
    """Where the descent went: final column, deepest column, mean column."""
    del aux
    col = _deepsea_coords(obs)[:, 1]
    return jnp.stack([
        _last_valid(col[:, None], weights)[0],
        _masked_max(col, weights),
        _masked_mean(col, weights),
    ])


def _acrobot_coords(obs):
    theta1 = jnp.arctan2(obs[:, 1], obs[:, 0])
    theta2 = jnp.arctan2(obs[:, 3], obs[:, 2])
    return jnp.stack([theta1, theta2], axis=-1)


@dataclasses.dataclass(frozen=True)
class EnvBehaviourSpec:
    """Everything task-specific about the hand-designed descriptor families.

    Attributes:
        handcrafted_fn: (obs, weights, aux) -> (d,) raw expert descriptor. `aux`
            carries per-step signals that are not in the observation (CheetahRun
            passes foot contact there); tasks that need none ignore it.
        handcrafted_names/low/high: names and normalisation range of its dims.
        action_kind: "discrete" or "continuous". Selects the action-family
            descriptor and the probe-state measure; see the module docstring.
        coords_fn: (obs) -> (T, 2) the two state dimensions the occupancy
            histogram is built over, or None to skip the occupancy family.
        occupancy_names/low/high: names and range of those two dimensions.
            Unused when coords_fn is None.
    """

    handcrafted_fn: Callable[[Array, Array, Any], Array]
    handcrafted_names: tuple
    handcrafted_low: tuple
    handcrafted_high: tuple
    action_kind: str = "discrete"
    coords_fn: Callable[[Array], Array] | None = None
    occupancy_names: tuple = ()
    occupancy_low: tuple = ()
    occupancy_high: tuple = ()

    @property
    def has_occupancy(self):
        return self.coords_fn is not None


# Ranges are the reachable ranges of the gymnax envs. Where the observation
# space itself is unbounded (the two CartPole velocities) the range is the
# practically reachable one; values outside are clipped, which only affects the
# occupancy binning at the extremes.
ENV_BEHAVIOUR_SPECS = {
    "CartPole-v1": EnvBehaviourSpec(
        handcrafted_fn=_cartpole_descriptor,
        handcrafted_names=("cart_pos_mean", "cart_speed_mean", "pole_angle_absmean"),
        handcrafted_low=(-2.4, 0.0, 0.0),
        handcrafted_high=(2.4, 3.0, 0.21),
        coords_fn=_cartpole_coords,
        occupancy_names=("cart_pos", "pole_angle"),
        occupancy_low=(-2.4, -0.21),
        occupancy_high=(2.4, 0.21),
    ),
    "MountainCar-v0": EnvBehaviourSpec(
        handcrafted_fn=_mountaincar_descriptor,
        handcrafted_names=("pos_final", "pos_max", "speed_mean"),
        handcrafted_low=(-1.2, -1.2, 0.0),
        handcrafted_high=(0.6, 0.6, 0.07),
        coords_fn=_mountaincar_coords,
        occupancy_names=("position", "velocity"),
        occupancy_low=(-1.2, -0.07),
        occupancy_high=(0.6, 0.07),
    ),
    "Acrobot-v1": EnvBehaviourSpec(
        handcrafted_fn=_acrobot_descriptor,
        handcrafted_names=("tip_height_max", "tip_height_mean", "joint1_speed_mean"),
        handcrafted_low=(-2.0, -2.0, 0.0),
        handcrafted_high=(2.0, 2.0, 4.0 * np.pi),
        coords_fn=_acrobot_coords,
        occupancy_names=("theta1", "theta2"),
        occupancy_low=(-np.pi, -np.pi),
        occupancy_high=(np.pi, np.pi),
    ),
    # CheetahRun (mujoco_playground / MJX). Already a fraction of the episode,
    # so the [0, 1] normalisation is the identity and the numbers are readable
    # as duty factors. No occupancy family: the two state dimensions that would
    # carry the task semantics here are the root velocity (which is the reward)
    # and the torso pitch, and a histogram over those describes speed rather
    # than gait -- see the module docstring on families a task may omit.
    # Keyed by the body so `get_spec`'s prefix match serves every grid size.
    "DeepSea": EnvBehaviourSpec(
        handcrafted_fn=_deepsea_descriptor,
        handcrafted_names=("col_final", "col_max", "col_mean"),
        handcrafted_low=(0.0, 0.0, 0.0),
        handcrafted_high=(1.0, 1.0, 1.0),
        coords_fn=_deepsea_coords,
        occupancy_names=("row", "column"),
        occupancy_low=(0.0, 0.0),
        occupancy_high=(1.0, 1.0),
    ),
    "CheetahRun": EnvBehaviourSpec(
        handcrafted_fn=_cheetah_descriptor,
        handcrafted_names=("bfoot_contact_frac", "ffoot_contact_frac"),
        handcrafted_low=(0.0, 0.0),
        handcrafted_high=(1.0, 1.0),
        action_kind="continuous",
        coords_fn=None,
    ),
    # The MJX planar walker, keyed by BODY rather than by task: the Continual
    # Walker chain runs WalkerStand, WalkerWalk and WalkerRun on one XML, and the
    # diversity tracker is built once for the whole sequence, so a per-task key
    # could not name it. `_cheetah_descriptor` is reused unchanged -- it is the
    # generic two-foot duty factor and the walker has exactly two feet
    # (`left_foot`, `right_foot`; see FOOT_GEOMS in
    # source/envs/mjx_cheetah.py for the geom names and the ordering).
    #
    # Duty factor is the right descriptor here for the same reason it is on the
    # cheetah, and more so: all three Walker rewards are functions of horizontal
    # speed and torso height, so any descriptor built on those would be the
    # return re-labelled -- and on a chain that varies the target speed it would
    # be the return re-labelled DIFFERENTLY per sub-task, which is worse.
    "Walker": EnvBehaviourSpec(
        handcrafted_fn=_cheetah_descriptor,
        handcrafted_names=("left_foot_contact_frac", "right_foot_contact_frac"),
        handcrafted_low=(0.0, 0.0),
        handcrafted_high=(1.0, 1.0),
        action_kind="continuous",
        coords_fn=None,
    ),
    # Brax `ant`, the other continuous-control task. Same descriptor family as
    # CheetahRun with four legs instead of two, and no occupancy family for the
    # same reason: ant's reward is forward velocity, so a histogram over the
    # torso's position or speed would be the return re-labelled. The ant's own
    # observation does not even contain the torso x/y -- brax drops qpos[:2] --
    # so there is nothing to bin over without reaching into the simulator.
    "ant": EnvBehaviourSpec(
        handcrafted_fn=_ant_descriptor,
        handcrafted_names=("left_foot_contact_frac", "right_foot_contact_frac",
                           "third_foot_contact_frac", "fourth_foot_contact_frac"),
        handcrafted_low=(0.0, 0.0, 0.0, 0.0),
        handcrafted_high=(1.0, 1.0, 1.0, 1.0),
        action_kind="continuous",
        coords_fn=None,
    ),
}


# ----------------------------------------------------------------------------
# Foot contact for the MJX cheetah
# ----------------------------------------------------------------------------

# Height below which a foot capsule counts as touching the ground, in metres.
# The MJX state exposes no public contact array (`data.contact` is absent; the
# arrays live on the private `data._impl`), so contact is decided geometrically
# from the capsule's lowest point. Verified against the simulator: with zero
# torques the cheetah rests on both feet and this test returns 1.0 for each,
# while under random torques the feet lift and the duty factors separate.
FOOT_CONTACT_HEIGHT = 0.005


def foot_contact_from_geoms(geom_xpos, geom_xmat, geom_ids, radii, half_lengths,
                            threshold=FOOT_CONTACT_HEIGHT):
    """Which of the named foot capsules are on the ground, as (n_feet,) 0/1.

    The lowest point of a capsule is its centre minus the radius and minus the
    vertical extent of its long axis (the geom's local z), which is what
    `geom_xmat[:, 2, 2]` gives once the frame is in world coordinates. Taking
    the centre height alone would be wrong: the two cheetah feet are different
    capsules at different heights, so no single centre threshold separates
    "standing" from "in flight" for both.

    Args:
        geom_xpos: (ngeom, 3) geom centres in world coordinates.
        geom_xmat: (ngeom, 3, 3) geom rotation matrices.
        geom_ids: indices of the foot geoms.
        radii, half_lengths: capsule size per foot, same order as geom_ids.
    """
    axis_z = jnp.abs(geom_xmat[geom_ids, 2, 2])
    bottom = geom_xpos[geom_ids, 2] - (half_lengths * axis_z + radii)
    return (bottom < threshold).astype(jnp.float32)


def get_spec(env_name):
    try:
        return ENV_BEHAVIOUR_SPECS[env_name]
    except KeyError:
        pass
    # Body-keyed fallback. The 'Walker' spec is stored under the BODY because
    # the Continual Walker chain runs three tasks on one XML and the tracker is
    # built once for the sequence -- but a friction sweep on a single walker
    # task constructs the tracker with the task name ('WalkerRun'), which must
    # resolve to the same body spec. Longest matching prefix wins, so an exact
    # task-specific key always takes precedence over a body key.
    matches = [k for k in ENV_BEHAVIOUR_SPECS if env_name.startswith(k)]
    if matches:
        return ENV_BEHAVIOUR_SPECS[max(matches, key=len)]
    raise ValueError(
        f"No behaviour descriptor defined for {env_name!r}. Add one to "
        "ENV_BEHAVIOUR_SPECS in source/metrics/behaviour_descriptors.py; "
        "the hand-designed descriptors are deliberately task-specific."
    )


# ============================================================================
# Per-rollout descriptors (traced, called inside the scoring functions)
# ============================================================================


@dataclasses.dataclass(frozen=True)
class BehaviourConfig:
    """Options for the descriptors computed during a rollout."""

    env_name: str
    num_actions: int
    occupancy_bins: int = 12


def occupancy_descriptor(obs, weights, cfg):
    """Normalised 2-D state-visitation histogram of one episode.

    A Monte-Carlo estimate of the policy's (undiscounted) occupancy measure,
    projected onto the two state dimensions the task is about. Steps after
    termination carry zero weight, so a policy that dies early is described by
    the states it actually visited rather than by a frozen final state.
    """
    spec = get_spec(cfg.env_name)
    bins = cfg.occupancy_bins
    coords = spec.coords_fn(obs)                              # (T, 2)
    low = jnp.asarray(spec.occupancy_low)
    high = jnp.asarray(spec.occupancy_high)

    scaled = (coords - low) / jnp.maximum(high - low, 1e-8)
    idx = jnp.clip((scaled * bins).astype(jnp.int32), 0, bins - 1)
    flat_idx = idx[:, 0] * bins + idx[:, 1]

    counts = jnp.zeros(bins * bins).at[flat_idx].add(weights)
    return counts / jnp.maximum(jnp.sum(counts), 1e-8)


def handcrafted_descriptor(obs, weights, cfg, aux=None):
    """Expert descriptor of one episode, normalised to [0, 1] per dimension."""
    spec = get_spec(cfg.env_name)
    raw = spec.handcrafted_fn(obs, weights, aux)
    low = jnp.asarray(spec.handcrafted_low)
    high = jnp.asarray(spec.handcrafted_high)
    return jnp.clip((raw - low) / jnp.maximum(high - low, 1e-8), 0.0, 1.0)


def action_frequency_descriptor(actions, weights, cfg):
    """Proportion of the episode spent on each discrete action."""
    counts = jnp.zeros(cfg.num_actions).at[actions].add(weights)
    return counts / jnp.maximum(jnp.sum(counts), 1e-8)


def action_usage_descriptor(actions, weights, cfg):
    """Mean |torque| per actuator over the episode, the continuous analogue.

    Actions are the policy's tanh output, so each component is already in
    [-1, 1] and the mean absolute value lands in [0, 1] without further
    normalisation. Magnitude rather than signed mean, because a joint that
    oscillates hard and a joint that is never driven are the behaviours we want
    told apart, and their signed means are both near zero.
    """
    del cfg
    total = jnp.maximum(jnp.sum(weights), 1.0)
    return jnp.sum(jnp.abs(actions) * weights[:, None], axis=0) / total


def rollout_behaviour(obs, actions, weights, cfg, aux=None):
    """All per-rollout descriptors the task supports, at once.

    Which families appear depends on the task's `EnvBehaviourSpec`: a task
    without occupancy coordinates has no "occupancy" key, and the action family
    is "action_freq" for discrete action spaces and "action_usage" for
    continuous ones. Callers must not assume a fixed key set.

    Args:
        obs: (T, obs_dim) observations of one episode. Steps past termination
            may hold any value; `weights` masks them out.
        actions: (T,) int actions taken (discrete) or (T, nu) torques
            (continuous).
        weights: (T,) 1.0 while the episode is running, 0.0 after it ends.
        cfg: BehaviourConfig.
        aux: optional dict of extra per-step signals for the expert descriptor,
            e.g. {"foot_contact": (T, 2)} for CheetahRun.

    Returns:
        dict of descriptor name -> flat array.
    """
    spec = get_spec(cfg.env_name)
    out = {
        "handcrafted": handcrafted_descriptor(obs, weights, cfg, aux),
        "episode_steps": jnp.sum(weights),
    }
    if spec.has_occupancy:
        out["occupancy"] = occupancy_descriptor(obs, weights, cfg)
    if spec.action_kind == "continuous":
        out["action_usage"] = action_usage_descriptor(actions, weights, cfg)
    else:
        out["action_freq"] = action_frequency_descriptor(actions, weights, cfg)
    return out


def average_over_evals(behaviour, pop_size, num_evals):
    """Average per-rollout descriptors over the repeats of each individual.

    The scoring functions evaluate every individual `num_evals` times; a single
    stochastic rollout is a noisy sample of the policy's behaviour, and the
    diversity of noise is not behavioural diversity. Averaging the histograms
    is exactly the mixture over rollouts, i.e. a better occupancy estimate.
    """
    return jax.tree.map(
        lambda x: jnp.mean(x.reshape((pop_size, num_evals) + x.shape[1:]), axis=1),
        behaviour,
    )


# ============================================================================
# Population diversity statistics (host side, numpy)
# ============================================================================


def _subsample(x, max_n, seed):
    x = np.asarray(x, dtype=np.float64).reshape(len(x), -1)
    if len(x) > max_n:
        idx = np.random.default_rng(seed).choice(len(x), max_n, replace=False)
        x = x[idx]
    return x


def mean_pairwise_euclidean(x, max_n=128, seed=0):
    """Mean pairwise Euclidean distance over a sub-sample of the population."""
    from scipy.spatial.distance import pdist

    x = _subsample(x, max_n, seed)
    if len(x) < 2:
        return 0.0
    return float(pdist(x, metric="euclidean").mean())


def mean_pairwise_js_distance(p, max_n=128, seed=0):
    """Mean pairwise Jensen-Shannon distance between distributions, in [0, 1].

    The distance (square root of the base-2 divergence) rather than the
    divergence, because it is a true metric, so "mean pairwise distance" means
    the same thing here as it does for the Euclidean families.
    """
    p = _subsample(p, max_n, seed)
    if len(p) < 2:
        return 0.0
    p = p / np.maximum(p.sum(axis=1, keepdims=True), 1e-12)

    eps = 1e-12
    n = len(p)
    iu, ju = np.triu_indices(n, k=1)
    total, chunk = 0.0, 4096
    for start in range(0, len(iu), chunk):
        a = p[iu[start:start + chunk]]
        b = p[ju[start:start + chunk]]
        m = 0.5 * (a + b)
        kl_a = np.sum(a * (np.log2(a + eps) - np.log2(m + eps)), axis=1)
        kl_b = np.sum(b * (np.log2(b + eps) - np.log2(m + eps)), axis=1)
        total += np.sum(np.sqrt(np.clip(0.5 * (kl_a + kl_b), 0.0, 1.0)))
    return float(total / len(iu))


def grid_coverage(x01, bins=8):
    """Fraction of reachable descriptor cells the population occupies.

    The QD coverage measure: discretise the (already [0, 1]-normalised)
    descriptor space into `bins` per dimension and count occupied cells. The
    count is divided by `min(pop_size, num_cells)` so that 1.0 means "every
    individual sits in its own cell", which keeps the number readable for
    populations smaller than the grid.
    """
    x = np.asarray(x01, dtype=np.float64).reshape(len(x01), -1)
    if len(x) == 0:
        return 0.0
    idx = np.clip((x * bins).astype(np.int64), 0, bins - 1)
    cells = {tuple(row) for row in idx}
    return float(len(cells) / min(len(x), bins ** x.shape[1]))


def population_diversity(behaviour, max_n=128, seed=0, coverage_bins=8):
    """Diversity of one population, one number per descriptor family.

    Args:
        behaviour: dict as returned by `rollout_behaviour` / `average_over_evals`,
            with a leading population axis. Device arrays are fine.
        max_n: cap on the number of individuals compared pairwise.

    Returns:
        dict of metric name -> float, ready to drop into wandb / the metrics json.
    """
    b = {k: np.asarray(jax.device_get(v)) for k, v in behaviour.items()}
    out = {
        "bd_handcrafted_diversity": mean_pairwise_euclidean(b["handcrafted"], max_n, seed),
        "bd_handcrafted_coverage": grid_coverage(b["handcrafted"], coverage_bins),
    }
    # Keyed off what the task actually produced -- see `rollout_behaviour`.
    if "occupancy" in b:
        out["bd_occupancy_diversity"] = mean_pairwise_js_distance(b["occupancy"], max_n, seed)
    if "action_freq" in b:
        out["bd_action_freq_diversity"] = mean_pairwise_js_distance(b["action_freq"], max_n, seed)
    if "action_usage" in b:
        out["bd_action_usage_diversity"] = mean_pairwise_euclidean(b["action_usage"], max_n, seed)
    if "episode_steps" in b:
        out["bd_episode_steps_std"] = float(np.std(b["episode_steps"]))
    return out


# ============================================================================
# Probe-state action distribution (the measure the paper already used)
# ============================================================================


def collect_probe_states(observations, num_probe=512, key=None, batch_dims=None):
    """A batch of states to score every policy on, drawn from visited states.

    Called once, on the initial population's trajectories, and then frozen for
    the whole run: the point of this measure is that the *same* states are
    scored at every generation, which is also its weakness -- it says nothing
    about states this batch happens to miss.

    `batch_dims` is for structured observations. The default (None) keeps the
    original behaviour: `observations` is one array whose last axis is the
    observation and every leading axis is batch. The scheduling suite, dropped
    2026-09-08, had an observation that was a NamedTuple of arrays with
    different trailing shapes, so it
    passes `batch_dims=2` to say "the first two axes are (step, env), whatever
    each leaf looks like after that". Both paths draw the same indices from the
    same key, so this is a strictly wider signature, not a change.
    """
    if batch_dims is None:
        obs = jnp.asarray(observations)
        flat = obs.reshape(-1, obs.shape[-1])
        n = min(num_probe, flat.shape[0])
        if key is None:
            stride = max(1, flat.shape[0] // n)
            return flat[::stride][:n]
        idx = jax.random.choice(key, flat.shape[0], shape=(n,), replace=False)
        return flat[idx]

    flat = probe_flatten(observations, batch_dims=batch_dims)
    total = probe_size(flat)
    n = min(num_probe, total)
    if key is None:
        stride = max(1, total // n)
        return probe_index(flat, jnp.arange(0, total, stride)[:n])
    idx = jax.random.choice(key, total, shape=(n,), replace=False)
    return probe_index(flat, idx)


def make_probe_diversity_fn(apply_flat, max_sample=128, action_kind="discrete"):
    """Build the jitted probe-state diversity statistic.

    Args:
        apply_flat: (flat_params, obs_batch) -> logits (discrete) or actions
            (continuous), i.e. the trainer's policy applied to a flat genome.
        max_sample: cap on the number of individuals compared pairwise.
        action_kind: "discrete" uses the softmax Jensen-Shannon divergence the
            paper already reports; "continuous" compares the emitted action
            vectors directly, since there is no distribution over actions to
            take a divergence between.
    """
    if action_kind == "continuous":
        return _make_continuous_probe_fn(apply_flat, max_sample)

    def _stats(flat_genotypes, probe_obs, key):
        n = min(max_sample, flat_genotypes.shape[0])
        idx = jax.random.choice(key, flat_genotypes.shape[0], shape=(n,), replace=False)
        genotypes = flat_genotypes[idx]

        logits = jax.vmap(lambda p: apply_flat(p, probe_obs))(genotypes)
        log_probs = jax.nn.log_softmax(logits, axis=-1)          # (n, batch, A)
        probs = jnp.exp(log_probs)

        actions = jnp.argmax(logits, axis=-1)                    # (n, batch)
        mask = jnp.triu(jnp.ones((n, n)), k=1)
        disagree = (actions[:, None, :] != actions[None, :, :]).astype(jnp.float32)
        disagreement = jnp.sum(disagree.mean(axis=-1) * mask) / jnp.sum(mask)

        p_i, p_j = probs[:, None], probs[None, :]
        m = 0.5 * (p_i + p_j)
        log_m = jnp.log(m + 1e-12)
        kl_i = jnp.sum(p_i * (jnp.log(p_i + 1e-12) - log_m), axis=-1)
        kl_j = jnp.sum(p_j * (jnp.log(p_j + 1e-12) - log_m), axis=-1)
        js = 0.5 * (kl_i + kl_j)                                 # (n, n, batch)
        mean_js = jnp.sum(js.mean(axis=-1) * mask) / jnp.sum(mask)

        entropy = -jnp.sum(probs * log_probs, axis=-1)
        return mean_js, disagreement, jnp.mean(entropy)

    jitted = jax.jit(_stats)

    def probe_diversity(flat_genotypes, probe_obs, key):
        js, disagreement, entropy = jitted(flat_genotypes, probe_obs, key)
        return {
            "bd_probe_js": float(js),
            "bd_probe_disagreement": float(disagreement),
            "bd_probe_policy_entropy": float(entropy),
        }

    return probe_diversity


def _make_continuous_probe_fn(apply_flat, max_sample=128):
    """Probe-state measure for deterministic continuous-control policies.

    The discrete version asks how differently two policies *distribute* their
    action mass on a fixed batch of states. A tanh policy emits one action
    vector per state and no distribution, so the analogue is the distance
    between those vectors: `bd_probe_action_dist` is the mean pairwise Euclidean
    distance over the probe batch, and `bd_probe_disagreement` the mean
    per-actuator absolute difference, which is on the same [0, 2] scale as the
    action range whatever the number of actuators.

    Not comparable in level with `bd_probe_js` -- different units, different
    measure. It answers the same question for a task where the other cannot.
    """

    def _stats(flat_genotypes, probe_obs, key):
        n = min(max_sample, flat_genotypes.shape[0])
        idx = jax.random.choice(key, flat_genotypes.shape[0], shape=(n,), replace=False)
        genotypes = flat_genotypes[idx]

        actions = jax.vmap(lambda p: apply_flat(p, probe_obs))(genotypes)  # (n, batch, nu)
        mask = jnp.triu(jnp.ones((n, n)), k=1)
        denom = jnp.maximum(jnp.sum(mask), 1.0)

        diff = actions[:, None] - actions[None, :]                        # (n, n, batch, nu)
        dist = jnp.sqrt(jnp.sum(diff ** 2, axis=-1) + 1e-12).mean(axis=-1)
        absdiff = jnp.abs(diff).mean(axis=(-1, -2))

        return (jnp.sum(dist * mask) / denom,
                jnp.sum(absdiff * mask) / denom,
                jnp.mean(jnp.std(actions, axis=0)))

    jitted = jax.jit(_stats)

    def probe_diversity(flat_genotypes, probe_obs, key):
        dist, disagreement, spread = jitted(flat_genotypes, probe_obs, key)
        return {
            "bd_probe_action_dist": float(dist),
            "bd_probe_disagreement": float(disagreement),
            "bd_probe_action_std": float(spread),
        }

    return probe_diversity


# ============================================================================
# AURORA as an observer
# ============================================================================


class AuroraTracker:
    """AURORA descriptors used only for measurement, never for selection.

    DNS already trains an AURORA encoder because its selection needs one; GA
    and ES do not, so they train this one alongside the search on the same
    schedule. It is a pure observer: nothing it returns is fed back into the
    algorithm, so adding it cannot change what a run does.
    """

    def __init__(self, obs_size, traj_steps, num_generations, latent_dim=6,
                 learning_rate=1e-3, batch_size=128, train_ratio=8):
        from source.metrics.aurora import AuroraDescriptors, aurora_training_schedule

        self.aurora = AuroraDescriptors(
            obs_size=obs_size, traj_steps=traj_steps, latent_dim=latent_dim,
            learning_rate=learning_rate, batch_size=batch_size,
        )
        self.schedule = aurora_training_schedule(num_generations, train_ratio)
        self.state = None
        self.loss = float("nan")

    def init(self, key, observations=None):
        """Random encoder, then one training pass if trajectories are given."""
        key, init_key = jax.random.split(key)
        self.state = self.aurora.init(init_key)
        if observations is not None:
            self.state, self.loss = self.aurora.train(
                key, observations, self.state, iteration=0)
        return self.state

    def maybe_train(self, key, generation, observations):
        """Retrain on AURORA's schedule; returns True if it retrained."""
        if (generation + 1) not in self.schedule:
            return False
        self.state, self.loss = self.aurora.train(
            key, observations, self.state, iteration=generation)
        return True

    def diversity(self, observations, max_n=128, seed=0):
        latent = self.aurora.encode(jnp.asarray(observations), self.state)
        return {
            "bd_aurora_diversity": mean_pairwise_euclidean(
                jax.device_get(latent), max_n, seed),
            "bd_aurora_loss": float(self.loss),
        }
