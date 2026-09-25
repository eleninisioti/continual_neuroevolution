"""Behavioural descriptors for Kinetix.

Why this file exists
--------------------
DNS on Kinetix was running with no behavioural signal at all. Two different
placeholders were in the tree, and neither is a behaviour:

* ``experiments/dns.py`` used ``PixelsObservation.global_info`` padded to two
  dims. That field is ``[state.gravity[1] / 10.0]`` -- a **constant**, identical
  for every individual on a level and unchanged through the episode. Every
  pairwise descriptor distance is therefore 0, novelty is 0 for the whole
  population, and DNS selection degenerates to ranking on fitness alone, i.e.
  to plain GA.
* ``experiments/dns_continual.py`` used the first two entries of the flattened
  pixel observation -- two channel values of one corner pixel, essentially the
  background colour.

This module provides the real thing, mirroring what
``source/metrics/behaviour_descriptors.py`` does for gymnax and mujoco.

The descriptors
---------------
``duty_factor`` (dim 6, the default hand-designed descriptor)
    Fraction of the episode each actuator binding is active: 4 motor bindings
    and 2 thruster bindings for ``env_size=m``. This is the Kinetix analogue of
    the foot duty factor Cully et al. (Nature 2015) use for legged robots, and
    it is what ``_cheetah_descriptor`` / ``_ant_descriptor`` compute for the
    mujoco tasks.

    It is the right primary descriptor here for one reason the others cannot
    match: **it means the same thing on all 20 sub-tasks**. The number of
    bindings is fixed by ``env_size``, not by the level, whereas the levels
    differ in geometry, in how many shapes they contain and in what the
    morphology even is. In the continual setting a descriptor built on
    coordinates measures the level as much as the policy.

    Action semantics come from ``convert_multi_discrete_actions``: motors take
    ``{0,1,2} -> {0, +1, -1}`` and thrusters ``{0,1} -> {0,1}``, so **0 is off
    for every channel** and "active" is uniformly ``action != 0``.

``goal_trajectory`` (dim 5, hand-designed, within-task only)
    Where the controlled (green, role 1) shape ends up, how close it got to the
    target (blue, role 2), and how far it travelled. The Kinetix analogue of
    MountainCar's "how far up the hill". Interpretable and directly
    task-relevant, but level-relative -- do not compare it across sub-tasks.

``trajectory_features`` (dim 13 per step, the AURORA input)
    The per-step feature sequence handed to the auto-encoder. Deliberately
    *not* the pixel observation: an LSTM auto-encoder over 8192-dim frames
    would dominate the cost of the whole run and would mostly encode rendering,
    not behaviour. Instead: green position and velocity, blue position, the
    green-blue distance, and the six action channels.

Roles come from the reward rule in ``environment/env.py``: the goal fires when a
role-1 shape touches a role-2 shape (``r1 * r2 == 2``) and fails against role 3.
Green and blue positions are taken as role-masked centroids over polygons and
circles together, so the descriptor has a fixed size no matter how many shapes a
level has or which primitive carries the role.
"""

import jax
import jax.numpy as jnp


ROLE_GREEN = 1  # the shape that must reach the target
ROLE_BLUE = 2   # the target
ROLE_RED = 3    # touching this fails the episode


def _role_centroid(state, role):
    """Mean position of the shapes carrying `role`, over polygons and circles.

    Role-masked and active-masked so the result is a fixed (2,) vector for any
    level. Falls back to zeros when a level has no shape with that role.
    """
    pos = jnp.concatenate([state.polygon.position, state.circle.position], axis=0)
    roles = jnp.concatenate([state.polygon_shape_roles, state.circle_shape_roles], axis=0)
    active = jnp.concatenate([state.polygon.active, state.circle.active], axis=0)

    mask = ((roles == role) & active).astype(pos.dtype)
    total = jnp.sum(mask)
    return jnp.sum(pos * mask[:, None], axis=0) / jnp.maximum(total, 1.0)


def _role_velocity(state, role):
    vel = jnp.concatenate([state.polygon.velocity, state.circle.velocity], axis=0)
    roles = jnp.concatenate([state.polygon_shape_roles, state.circle_shape_roles], axis=0)
    active = jnp.concatenate([state.polygon.active, state.circle.active], axis=0)

    mask = ((roles == role) & active).astype(vel.dtype)
    total = jnp.sum(mask)
    return jnp.sum(vel * mask[:, None], axis=0) / jnp.maximum(total, 1.0)


def step_features(state, action, num_motor_bindings):
    """(13,) per-step feature vector: the AURORA input and the raw material for
    the hand-designed descriptors.

    Layout: green xy (2), green velocity (2), blue xy (2), green-blue distance
    (1), action channels (6).
    """
    green = _role_centroid(state, ROLE_GREEN)
    green_v = _role_velocity(state, ROLE_GREEN)
    blue = _role_centroid(state, ROLE_BLUE)
    dist = jnp.linalg.norm(green - blue, keepdims=True)
    # Raw action indices, not the mapped torques: the point is which channel is
    # being driven, and the mapping is a fixed relabelling.
    act = jnp.asarray(action, dtype=jnp.float32)
    return jnp.concatenate([green, green_v, blue, dist, act])


def duty_factor(actions, valid, num_motor_bindings, num_thruster_bindings):
    """(n_bindings,) fraction of the valid episode each binding is active.

    `actions` is (T, n_bindings) of raw multi-discrete indices, `valid` is (T,)
    with 1 for steps inside the episode. Action 0 is off for both motors and
    thrusters, so this is one expression for all channels.
    """
    n = num_motor_bindings + num_thruster_bindings
    actions = jnp.asarray(actions)[:, :n]
    w = jnp.asarray(valid, dtype=jnp.float32)
    active = (actions != 0).astype(jnp.float32)
    return jnp.sum(active * w[:, None], axis=0) / jnp.maximum(jnp.sum(w), 1.0)


def goal_trajectory(features, valid):
    """(5,) where the agent ended up and how it got there.

    From the `step_features` sequence: final green xy, minimum green-blue
    distance reached, mean speed, and total path length.
    """
    w = jnp.asarray(valid, dtype=jnp.float32)
    n = jnp.maximum(jnp.sum(w), 1.0)

    green = features[:, 0:2]
    green_v = features[:, 2:4]
    dist = features[:, 6]

    # Position at the last valid step.
    last = jnp.maximum(jnp.sum(w).astype(jnp.int32) - 1, 0)
    final_xy = green[last]

    # Invalid steps must not win the min, so push them up.
    min_dist = jnp.min(jnp.where(w > 0, dist, jnp.inf))
    min_dist = jnp.where(jnp.isfinite(min_dist), min_dist, 0.0)

    speed = jnp.linalg.norm(green_v, axis=-1)
    mean_speed = jnp.sum(speed * w) / n

    steps = jnp.linalg.norm(jnp.diff(green, axis=0), axis=-1)
    path_len = jnp.sum(steps * w[1:])

    return jnp.concatenate([
        final_xy,
        jnp.array([min_dist]),
        jnp.array([mean_speed]),
        jnp.array([path_len]),
    ])


def descriptor_from_rollout(features, actions, valid, num_motor_bindings,
                            num_thruster_bindings, kind="duty_factor"):
    """Dispatch used by the trainers. `kind` is the --descriptor flag."""
    if kind == "duty_factor":
        return duty_factor(actions, valid, num_motor_bindings, num_thruster_bindings)
    if kind == "goal_trajectory":
        return goal_trajectory(features, valid)
    if kind == "combined":
        return jnp.concatenate([
            duty_factor(actions, valid, num_motor_bindings, num_thruster_bindings),
            goal_trajectory(features, valid),
        ])
    raise ValueError(
        f"Unknown descriptor kind {kind!r}; expected one of "
        "'duty_factor', 'goal_trajectory', 'combined' "
        "(AURORA is not built here -- it consumes `features` directly)."
    )


def subsample(features, valid, traj_steps):
    """(traj_steps, feat) subsample of a rollout, for the AURORA encoder.

    Spread over the *valid* part of the episode rather than the padded array,
    so episodes of different lengths give comparable sequences.
    """
    length = jnp.maximum(jnp.sum(jnp.asarray(valid, jnp.int32)), 1)
    idx = (jnp.linspace(0.0, 1.0, traj_steps) * (length - 1)).astype(jnp.int32)
    return features[idx]
