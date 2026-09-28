"""Dominated Novelty Search, the paper's "GA + Novelty" arm.

DNS replaces the GA's fitness-ranked truncation with a rank on *dominated
novelty*: an individual's mean distance in descriptor space to the k nearest
individuals that are at least as fit. A low-fitness individual survives when
nothing similar to it is better, which is what keeps the population spread out
instead of converging. The fittest individuals have no fitter neighbour and
score NaN, which sorts highest under a descending argsort -- so elitism falls
out of the ranking rather than being a special case around it.

Two methods of one class, differing only in the variation operator
(`source/algorithms/ne/variation.py`):

    dns            Iso+LineDD, DNS as published.
    dns_gaussian   the GA's gaussian mutation, so `ga` vs `dns_gaussian`
                   differs in the selection rule alone.

The descriptor space is either hand-designed (per suite, see `source/envs`)
or learned online by AURORA (`source/metrics/aurora.py`).

The dominated-novelty computation is an exact port of QDax
`dns_repertoire.py`.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import random

from source.algorithms.ne.variation import (
    GAUSSIAN, ISOLINE, VARIATIONS, resolve_params, vary,
)


def dominated_novelty(fitness, descriptor, k, normalize=False):
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


class DNSState(NamedTuple):
    repertoire: jnp.ndarray    # (repertoire_size, num_params)
    fitness: jnp.ndarray       # (repertoire_size,) MAXIMISED
    descriptors: jnp.ndarray   # (repertoire_size, descriptor_dim)
    generation: jnp.ndarray
    # (repertoire_size, traj_steps, obs_dim) under AURORA, else a (P, 0, 0)
    # placeholder. The trajectories travel with the survivors because the
    # AURORA encoder is retrained during the run and EVERY stored descriptor
    # has to be recomputed with the new encoder -- a descriptor encoded by a
    # superseded encoder is not comparable with a fresh one.
    observations: jnp.ndarray = jnp.zeros((0, 0, 0))


class DNSSearcher:
    """Dominated Novelty Search. ``method='dns'`` or ``'dns_gaussian'``.

    Offspring come from ``variation`` (Iso+LineDD by default); survivors are
    the repertoire_size individuals of highest dominated novelty, the fittest
    (NaN) always first.

    The repertoire is RE-SCORED every generation, for the same reason as the
    GA's archive (`source/algorithms/ne/ga.py`): under a switching schedule a
    stored fitness and a stored descriptor were measured on a different
    sub-task, and "who dominates whom" would compare numbers from two tasks.
    ``population_size`` is therefore the EVALUATION BUDGET per generation and
    ``repertoire_ratio`` splits it: at 512 / 0.5 the repertoire is 256 and 256
    offspring are bred from it, matching the other methods' 512 evaluations.

    The iso/line sigmas default to the reference's corrected values
    (0.005 / 0.05).
    """

    needs_descriptors = True
    # Recorded in the run config (`train_nes.searcher_resolved`).
    refresh = True

    def __init__(self, num_params, population_size, descriptor_dim,
                 iso_sigma=0.005, line_sigma=0.05, k=3, init_scale=0.1,
                 normalize_descriptors=False, traj_steps=0, obs_dim=0,
                 repertoire_ratio=0.5, init_around_mean=True,
                 variation=ISOLINE, sigma_init=0.1, cross_over_rate=0.0):
        self.num_params = int(num_params)
        self.population_size = int(population_size)
        self.descriptor_dim = int(descriptor_dim)
        if variation not in VARIATIONS:
            raise ValueError(f'variation must be one of {VARIATIONS}, '
                             f'not {variation!r}')
        self.variation = variation
        self.variation_params = (
            resolve_params(ISOLINE, iso_sigma=iso_sigma,
                           line_sigma=line_sigma)
            if variation == ISOLINE else
            resolve_params(GAUSSIAN, sigma=sigma_init,
                           cross_over_rate=cross_over_rate))
        self.k = int(k)
        self.init_scale = float(init_scale)
        self.normalize_descriptors = bool(normalize_descriptors)
        # Where the repertoire starts: jittered copies of the seed policy, or
        # the reference's `N(0, init_scale)` -- as `GASearcher.init_around_mean`.
        self.init_around_mean = bool(init_around_mean)
        self.repertoire_size = max(
            1, int(self.population_size * float(repertoire_ratio)))
        self.num_offspring = self.population_size - self.repertoire_size
        if self.num_offspring < 1:
            raise ValueError('DNS needs repertoire_ratio < 1')
        # The name the run configs carry for the offspring count.
        self.batch_size = self.num_offspring
        self.traj_steps = int(traj_steps)
        self.obs_dim = int(obs_dim)

    def init(self, key, mean):
        r = self.repertoire_size
        jitter = random.normal(key, (r, self.num_params))
        return DNSState(
            repertoire=((mean[None, :] if self.init_around_mean else 0.0)
                        + self.init_scale * jitter),
            fitness=jnp.full((r,), -jnp.inf),
            descriptors=jnp.zeros((r, self.descriptor_dim)),
            generation=jnp.asarray(0, dtype=jnp.int32),
            observations=jnp.zeros((r, self.traj_steps, self.obs_dim)))

    def ask(self, key, state):
        # Parents are drawn from the WHOLE repertoire (see `_draw_parents` in
        # variation.py). Offspring first, then the repertoire to be re-scored.
        offspring = vary(self.variation, state.repertoire, key,
                         self.num_offspring, self.variation_params)
        return jnp.concatenate([offspring, state.repertoire], axis=0), None

    def tell(self, state, aux, fitness, descriptors=None, observations=None):
        if descriptors is None:
            raise ValueError('DNS needs behaviour descriptors')
        novelty = dominated_novelty(fitness, descriptors, self.k,
                                    self.normalize_descriptors)
        valid = fitness != -jnp.inf
        meta = jnp.where(valid, novelty, -jnp.inf)
        # NaN (the fittest, having no fitter neighbour) sorts first descending.
        keep = jnp.argsort(meta)[::-1][:self.repertoire_size]
        new = state._replace(repertoire=aux[keep], fitness=fitness[keep],
                             descriptors=descriptors[keep],
                             generation=state.generation + 1)
        if observations is not None:
            new = new._replace(observations=observations[keep])
        return new

    def reencode(self, state, descriptors):
        """Replace every stored descriptor, after the encoder was retrained.

        Separate from ``tell`` because it happens on AURORA's schedule rather
        than every generation, and because the encoder lives in the runner.
        """
        return state._replace(descriptors=descriptors)

    def incumbent(self, state):
        """The highest-fitness member. The repertoire is deliberately diverse,
        so its average is not a policy."""
        return state.repertoire[jnp.argmax(state.fitness)]

    def population_mean(self, state):
        """The coordinate-wise mean of the repertoire -- the same caveat as
        `GASearcher.population_mean`, and stronger: the repertoire is selected
        for behavioural spread."""
        return jnp.mean(state.repertoire, axis=0)

    # A persistent population, and one selected for spread.
    has_population = True

    def population(self, state):
        return state.repertoire
