"""Indexing helpers for a frozen probe batch.

Every plasticity diagnostic scores policies on the same held-out batch of
visited states. For gymnax/brax/kinetix that batch is a single `(n, obs_dim)`
array and plain slicing is enough. The scheduling suite, dropped
2026-09-08, had an observation that was a NamedTuple of arrays with different
trailing shapes (`ops_durations` is
(n, J, O), `machines_remaining_times` is (n, M)), so `probe_obs[:k]` and
`probe_obs.shape[0]` stop working.

These four helpers are the whole difference. They are `jax.tree` operations, so
a bare array -- a pytree with one leaf -- goes through them unchanged and every
existing caller keeps its numbers bit-exactly. Use them instead of indexing a
probe batch directly.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


def probe_size(probe_obs) -> int:
    """Number of states in the batch, i.e. the leading axis of any leaf."""
    leaves = jax.tree_util.tree_leaves(probe_obs)
    if not leaves:
        return 0
    return int(leaves[0].shape[0])


def probe_take(probe_obs, n: int):
    """First `n` states."""
    return jax.tree_util.tree_map(lambda a: a[:n], probe_obs)


def probe_index(probe_obs, idx):
    """States at `idx` (an integer array)."""
    return jax.tree_util.tree_map(lambda a: a[idx], probe_obs)


def probe_expand(probe_state):
    """One state -> a batch of one, i.e. add the leading axis back."""
    return jax.tree_util.tree_map(lambda a: a[None, ...], probe_state)


def probe_flatten(observations, batch_dims: int = 2):
    """Collapse `batch_dims` leading axes of a rollout buffer into one.

    `observations` comes off a scan as (num_steps, num_envs, ...) per leaf; the
    probe batch wants (num_steps * num_envs, ...).
    """
    return jax.tree_util.tree_map(
        lambda a: jnp.asarray(a).reshape((-1,) + a.shape[batch_dims:]), observations
    )
