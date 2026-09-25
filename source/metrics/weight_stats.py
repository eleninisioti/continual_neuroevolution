"""Weight statistics, the third plasticity signal alongside dormancy and churn.

Dormant-neuron counts say how many units have stopped responding and churn says
how much the policy's outputs move per update. Neither sees the parameters
themselves, and the failure mode they miss is the well-documented one: weight
norms that grow without bound over a long non-stationary run, so the network
becomes progressively harder to move and loses plasticity while every unit is
still nominally active. That is why CLAUDE.md asks for all three.

Nothing in this repository measured this before -- not gymnax, not brax, not
mujoco, not kinetix -- so these keys are new everywhere rather than a brax
catch-up, and the same function serves the NE and RL sides so the two trees
carry identical names.

The keys are deliberately flat and prefixed rather than nested, because
`training_metrics.json` is a list of flat records and every reader in
`scripts/compare.py` indexes it by string key.

NE and RL need slightly different things and both are here:

  `weight_stats`             one parameter set -- an RL policy/value pytree, or
                             a single NE genome.
  `population_weight_stats`  a (pop_size, n_params) NE population, which also
                             reports the spread of per-genome norms. That spread
                             is a parameter-space diversity measure and is NOT a
                             substitute for the behavioural `bd_*` descriptors:
                             two genomes at the same distance from the origin can
                             be anywhere on that sphere.
"""

import numpy as np

# What counts as "effectively zero". Weights below this contribute nothing to
# the forward pass at the scales these policies operate on, and the fraction of
# them is how a network that has collapsed onto a few active paths shows up in
# the parameter statistics rather than the activation ones.
NEAR_ZERO = 1e-3

# The statistics `weight_stats` produces, without their prefix. Exported so a
# reader can build the full column names without hard-coding them a second
# time -- source/metrics/rl_diagnostics.py builds its weight mapping
# from this list, so adding a statistic below adds it to the RL trees too.
STAT_SUFFIXES = ('l2_norm', 'rms', 'mean_abs', 'max_abs', 'frac_near_zero')


def _as_flat_array(params):
    """A 1-D float64 numpy array of every scalar in `params`.

    Accepts a numpy/jax array or an arbitrarily nested dict/list/tuple pytree,
    so the NE side (a flat genome vector) and the RL side (a flax parameter
    dict) can call the same function. float64 because these are reductions over
    ~10^5 values that are then differenced across a long run: accumulating them
    in float32 loses the drift being measured.
    """
    if isinstance(params, dict):
        leaves = [_as_flat_array(v) for v in params.values()]
    elif isinstance(params, (list, tuple)):
        leaves = [_as_flat_array(v) for v in params]
    else:
        return np.asarray(params, dtype=np.float64).ravel()
    if not leaves:
        return np.zeros(0, dtype=np.float64)
    return np.concatenate(leaves)


def weight_stats(params, prefix="weight"):
    """Summary statistics of one parameter set, as a flat dict of floats.

    Returns l2_norm, rms, mean, var, mean_abs, max_abs and frac_near_zero.
    `rms` as well as `l2_norm` because the norm alone is not comparable between
    the policy and the value network -- they have different parameter counts,
    and a critic with 7x the parameters has a larger norm at identical scale.
    `rms` divides that out; `l2_norm` is kept because it is what the plasticity
    literature reports.

    `mean` is SIGNED and `var` is about that mean, which is what makes them say
    something `rms` and `mean_abs` cannot: rms**2 = mean**2 + var, so a network
    whose weights are spreading out and one whose weights are drifting off
    centre have the same rms and different (mean, var). Initialisation puts the
    mean at ~0, so a mean that walks away from zero is a real asymmetry in the
    parameter cloud rather than growth.
    """
    flat = _as_flat_array(params)
    if flat.size == 0:
        return {}
    abs_flat = np.abs(flat)
    return {
        f"{prefix}_l2_norm": float(np.sqrt(np.sum(flat * flat))),
        f"{prefix}_rms": float(np.sqrt(np.mean(flat * flat))),
        f"{prefix}_mean": float(np.mean(flat)),
        f"{prefix}_var": float(np.var(flat)),
        f"{prefix}_mean_abs": float(np.mean(abs_flat)),
        f"{prefix}_max_abs": float(np.max(abs_flat)),
        f"{prefix}_frac_near_zero": float(np.mean(abs_flat < NEAR_ZERO)),
    }


def population_weight_stats(population, prefix="weight"):
    """`weight_stats` over every weight in a population, plus per-genome spread.

    `population` is (pop_size, n_params), the layout every NE trainer here keeps
    its population in. The pooled statistics answer "how large are this
    population's weights"; `genome_l2_mean` / `genome_l2_std` answer "and how
    much do its members differ in scale", which is the quantity that collapses
    when a GA converges.
    """
    pop = np.asarray(population, dtype=np.float64)
    if pop.size == 0:
        return {}
    stats = weight_stats(pop, prefix=prefix)
    if pop.ndim == 2:
        norms = np.linalg.norm(pop, axis=1)
        stats[f"{prefix}_genome_l2_mean"] = float(np.mean(norms))
        stats[f"{prefix}_genome_l2_std"] = float(np.std(norms))
    return stats


# ---------------------------------------------------------------------------
# In-graph variant, for trainers whose population never reaches the host
# ---------------------------------------------------------------------------
#
# kinetix runs its generations inside a `jax.lax.scan`, so the population exists
# only as a traced device array: the numpy functions above cannot see it, and
# pulling 512 x 1.1e6 floats to the host every generation to measure a norm is
# not a trade worth making.
#
# These mirror the numpy definitions EXACTLY -- same NEAR_ZERO, same keys, same
# arithmetic -- so a kinetix column and a brax column mean the same thing. They
# live here rather than in the kinetix trainer for that reason: one definition,
# two execution paths, which is what CLAUDE.md's "single part of the code
# responsible" requires. Change a formula above and change it here.
#
# The returned values are traced scalars, not floats; the caller converts them
# when it converts the rest of its metrics.

def weight_stats_jax(flat, prefix="weight"):
    """`weight_stats`, on a jax array, safe to call inside jit/scan.

    `flat` is any shape; it is reduced over every element, matching the numpy
    version's `.ravel()`. No float64 promotion here -- jax defaults to float32
    and enabling x64 for one diagnostic would change every other computation in
    the trainer. The reductions are over ~1e6 values rather than the numpy
    path's ~1e5, so the precision note in `_as_flat_array` applies more, not
    less; treat these columns as trend indicators rather than exact drift.
    """
    import jax.numpy as jnp
    x = jnp.asarray(flat).ravel()
    abs_x = jnp.abs(x)
    return {
        f"{prefix}_l2_norm": jnp.sqrt(jnp.sum(x * x)),
        f"{prefix}_rms": jnp.sqrt(jnp.mean(x * x)),
        f"{prefix}_mean": jnp.mean(x),
        f"{prefix}_var": jnp.var(x),
        f"{prefix}_mean_abs": jnp.mean(abs_x),
        f"{prefix}_max_abs": jnp.max(abs_x),
        f"{prefix}_frac_near_zero": jnp.mean((abs_x < NEAR_ZERO).astype(x.dtype)),
    }


def population_weight_stats_jax(population, prefix="weight"):
    """`population_weight_stats`, on a (pop_size, n_params) jax array."""
    import jax.numpy as jnp
    pop = jnp.asarray(population)
    stats = weight_stats_jax(pop, prefix=prefix)
    norms = jnp.linalg.norm(pop, axis=1)
    stats[f"{prefix}_genome_l2_mean"] = jnp.mean(norms)
    stats[f"{prefix}_genome_l2_std"] = jnp.std(norms)
    return stats
