"""Dominated Novelty Search: selection, variation, and the diversity measures.

DNS -- the paper's "GA + Novelty" arm -- replaces the GA's fitness-ranked
truncation with a rank on *dominated novelty*: an individual's mean distance in
descriptor space to the k nearest individuals that beat it on fitness. A
low-fitness individual survives when nothing similar to it is better, which is
what keeps the population spread out instead of converging.

The fittest individuals have no fitter neighbour and score NaN, which sorts
highest under a descending argsort -- so elitism falls out of the ranking rather
than being a special case around it.

`isoline_variation` is the crossover DNS pairs with -- but only by default.
The operator is a CHOICE, not part of the method: it lives in
`source/algorithms/ne/variation.py`, shared with the GA, and either method can
run with either operator. That exists because DNS and the GA differed in the
operator as well as the selection rule, so a DNS-over-GA gap confounded
"novelty selection helps" with "recombination helps"; `--variation gaussian`
here and `--variation isoline` on the GA are the arms that separate them.

The descriptor space is either hand-designed (`handcrafted_descriptors`) or
learned online by AURORA (`source/metrics/aurora.py`), which is the default on
tasks with no established descriptor.

An exact port of QDax `dns_repertoire.py`, and identical between the stationary
and continual DNS trainers.
"""

import jax
import jax.numpy as jnp
import numpy as np
from jax import random

from source.algorithms.ne.variation import isoline_variation as _isoline


def _compute_dominated_novelty(fitness, descriptor, k, normalize=False):
    """Compute dominated novelty — exact port of QDax dns_repertoire.py.

    For each individual, dominated novelty is the mean distance in descriptor
    space to the k nearest neighbors that have fitness >= its own.
    Returns NaN for the fittest individuals (no fitter neighbors exist).
    NaN sorts highest in descending argsort, so they are always kept.

    `normalize` z-scores the descriptors per dimension across the pool before
    the distances are taken, which makes the ranking invariant to how the
    descriptor space happens to be scaled. Off by default (the reference does
    not do it, and its descriptor spaces are bounded and commensurate anyway);
    worth having for AURORA, whose six latent dimensions are on arbitrary and
    unequal scales, so an unnormalised Euclidean distance is dominated by
    whichever dimension the encoder happened to give the largest range.
    Note this is a whitening, not a bound: it fixes anisotropy, it does not on
    its own stop a runaway individual from being the farthest point. Pair it
    with a bounded descriptor rather than treating it as a substitute.
    """
    n = fitness.shape[0]
    if n <= 1:
        return jnp.full(n, jnp.nan)
    k = min(k, n - 1)
    valid = fitness != -jnp.inf

    if normalize:
        mean = jnp.nanmean(descriptor, axis=0, keepdims=True)
        std = jnp.nanstd(descriptor, axis=0, keepdims=True)
        descriptor = (descriptor - mean) / jnp.where(std > 1e-8, std, 1.0)

    # Neighbor mask excluding self
    neighbor = valid[:, None] & valid[None, :]
    neighbor = neighbor & ~jnp.eye(n, dtype=bool)

    # Fitter-or-equal mask: fitter[i,j] = True if fitness[j] >= fitness[i]
    fitter = fitness[:, None] <= fitness[None, :]
    fitter = jnp.where(neighbor, fitter, False)

    # Pairwise distances
    distance = jnp.linalg.norm(
        descriptor[:, None, :] - descriptor[None, :, :], axis=-1
    )
    distance = jnp.where(neighbor, distance, jnp.inf)

    # Distances to fitter neighbors only
    distance_fitter = jnp.where(fitter, distance, jnp.inf)

    # Dominated novelty: mean distance to k nearest fitter neighbors
    values_fit, indices_fit = jax.vmap(
        lambda x: jax.lax.top_k(-x, k)
    )(distance_fitter)
    dominated_novelty = jnp.mean(
        -values_fit,
        axis=-1,
        where=jnp.take_along_axis(fitter, indices_fit, axis=-1),
    )

    return dominated_novelty


def dns_selection(
    genotypes, fitnesses, descriptors, observations,
    new_genotypes, new_fitnesses, new_descriptors, new_observations,
    population_size, k, normalize=False,
):
    """DNS selection — follows QDax DominatedNoveltyRepertoire.add().

    Combines parents and offspring, computes dominated novelty on the combined
    pool, and keeps the top population_size individuals by dominated novelty.

    The observation trajectories travel with the survivors so that the AURORA
    encoder can be retrained on them and every stored descriptor recomputed
    whenever the encoder changes (as in AURORA.train()). Same contract as
    train_DNS_gymnax.py.
    """
    # Combine candidates
    combined_genotypes = jnp.concatenate([genotypes, new_genotypes], axis=0)
    combined_fitnesses = jnp.concatenate([fitnesses, new_fitnesses], axis=0)
    combined_descriptors = jnp.concatenate([descriptors, new_descriptors], axis=0)
    # None under --descriptor handcrafted, where nothing reads trajectories.
    combined_observations = (
        None if observations is None
        else jnp.concatenate([observations, new_observations], axis=0))

    # Compute dominated novelty on combined pool
    dominated_novelty = _compute_dominated_novelty(
        combined_fitnesses, combined_descriptors, k, normalize=normalize
    )

    valid = combined_fitnesses != -jnp.inf
    # THE TOP TIER IS KEPT AS A BLOCK (2026-09-18). The reference protects the
    # unique fittest individual (NaN novelty, sorted first) and nothing else:
    # members that TIE at the top score their novelty against each other, and
    # when they share a descriptor -- every DeepSea solver walks the one
    # solving path, every CartPole solver scores 500 -- that novelty is 0,
    # they rank below every non-solver, and are all dropped in the same
    # generation; a solution survived only while it was unique. Here every
    # member with no STRICTLY fitter neighbour ranks above the rest, ordered
    # among themselves by the same `<=` novelty, so tied solvers are kept
    # (the most novel ones if the tier is larger than the population) and a
    # population that ties everywhere -- the floor after a switch -- is still
    # ranked by novelty, exactly as before. Strict `<` inside the novelty
    # itself was tried first and is wrong: on that floor every member is
    # "fittest", every score is NaN, and selection degenerates to keeping the
    # newest offspring.
    n = combined_fitnesses.shape[0]
    strictly_fitter = combined_fitnesses[:, None] < combined_fitnesses[None, :]
    strictly_fitter = strictly_fitter & valid[None, :] & ~jnp.eye(n, dtype=bool)
    top_tier = valid & ~jnp.any(strictly_fitter, axis=1)
    rank_novelty = jnp.where(jnp.isnan(dominated_novelty), jnp.inf, dominated_novelty)
    meta_fitness = jnp.where(valid, rank_novelty, -jnp.inf)
    # lexicographic (tier, novelty): a stable sort on novelty, then on tier
    order = jnp.argsort(meta_fitness, stable=True)[::-1]
    indices = order[jnp.argsort(~top_tier[order], stable=True)]
    survivor_indices = indices[:population_size]

    return (
        combined_genotypes[survivor_indices],
        combined_fitnesses[survivor_indices],
        combined_descriptors[survivor_indices],
        None if combined_observations is None else combined_observations[survivor_indices],
        dominated_novelty[survivor_indices],
    )


def isoline_variation(genotypes, key, iso_sigma=0.005, line_sigma=0.05,
                      batch_size=256, return_parents=False):
    """Iso+Line-DD variation, in the argument order the DNS trainers call it.

    The operator itself lives in `source/algorithms/ne/variation.py`, shared
    with the GA so that the two methods can be crossed with either operator and
    the novelty-selection claim can be separated from the recombination one.
    This wrapper only reorders arguments: every DNS trainer here calls it
    positionally as `(population, key, iso_sigma, line_sigma, batch_size)`,
    which is not the `(genotypes, key, num_offspring, ...)` shape every other
    operator has.

    The sigma defaults are the DNS reference's
    (inspiration/DNS/Dominated-Novelty-Search/configs/algo/{me,aurora}.yaml).
    They were 0.05/0.5 until 2026-07-29, ten times too large -- see
    docs/dns_cheetah_diagnosis.md and the shared module's docstring.
    """
    return _isoline(genotypes, key, batch_size, iso_sigma=iso_sigma,
                    line_sigma=line_sigma, return_parents=return_parents)


def handcrafted_descriptors(observations, env_name):
    """Hand-designed descriptors, from the last valid observation of the episode.

    Only used with --descriptor handcrafted; the default is the unsupervised
    AURORA encoding of the whole trajectory.
    """
    last_obs = observations[:, -1, :]
    if 'CartPole' in env_name:
        return jnp.stack([last_obs[:, 0], last_obs[:, 2]], axis=-1)  # cart pos, pole angle
    elif 'MountainCar' in env_name:
        return last_obs  # position and velocity
    elif 'Acrobot' in env_name:
        return last_obs[:, :2]  # cos, sin of first joint
    return last_obs[:, :2]


def compute_fitness_diversity(fitnesses):
    return float(jnp.std(fitnesses))


def compute_descriptor_diversity(descriptors):
    """Mean pairwise distance in (learned) descriptor space."""
    n = descriptors.shape[0]
    if n < 2:
        return 0.0
    distances = jnp.linalg.norm(
        descriptors[:, None, :] - descriptors[None, :, :], axis=-1
    )
    mask = jnp.triu(jnp.ones((n, n)), k=1)
    return float(jnp.sum(distances * mask) / jnp.sum(mask))


def compute_genomic_diversity(genotypes, sample_size=32):
    """Compute genomic diversity using sampling to avoid OOM."""
    pop_size = genotypes.shape[0]
    if pop_size < 2:
        return 0.0
    
    actual_sample = min(sample_size, pop_size)
    indices = np.random.choice(pop_size, size=actual_sample, replace=False)
    sampled = genotypes[indices]
    
    sampled_np = np.array(sampled)
    g_min = np.min(sampled_np, axis=0, keepdims=True)
    g_max = np.max(sampled_np, axis=0, keepdims=True)
    g_range = np.maximum(g_max - g_min, 1e-8)
    norm_genotypes = (sampled_np - g_min) / g_range
    
    diffs = norm_genotypes[:, None, :] - norm_genotypes[None, :, :]
    distances = np.linalg.norm(diffs, axis=-1)
    
    mask = np.triu(np.ones((actual_sample, actual_sample)), k=1)
    mean_dist = np.sum(distances * mask) / np.sum(mask)
    
    return float(mean_dist)
