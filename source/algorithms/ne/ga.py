"""The genetic algorithm arm: truncation selection with gaussian mutation.

(mu + lambda) elitism in the style of Such et al. (2017): an archive of the
best ``elite_ratio * population_size`` genomes, offspring bred from it by
gaussian mutation (optionally with uniform crossover), and the next archive
chosen as the fittest of offspring and archive together.

The archive is RE-SCORED every generation: it rides along in the evaluated
batch, so every genome `tell` ranks was scored on the same sub-task in the
same generation. Keeping each member's stored fitness instead -- the
reference's behaviour, and right for a stationary task -- is wrong under a
switching schedule: an archive of sub-task A specialists carrying their
sub-task A scores cannot be displaced by anything evaluated on sub-task B, so
the GA would simply stop at the switch. Re-scoring costs ``num_elites``
evaluations, taken out of the offspring budget, so the GA spends
``population_size`` evaluations per generation like every other method.

Two searchers, one per setting the paper runs:

    GASearcher        ``method='ga'``: fixed mutation width. gymnax, MiniGrid,
                      MJX.
    FocusGASearcher   ``method='ga_focus'``: the GA with the mutation width
                      and the breeding pool set by how well the archive's
                      centroid scores against the archive. Kinetix.

Fitness is MAXIMISED at this interface and negated internally, where the
archive is kept sorted ascending by loss.
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
from jax import random

from source.algorithms.ne.variation import GAUSSIAN, resolve_params, vary


class GAState(NamedTuple):
    archive: jnp.ndarray      # (num_elites, num_params), best first
    fitness: jnp.ndarray      # (num_elites,) MINIMISED internally
    sigma: jnp.ndarray        # scalar mutation width
    generation: jnp.ndarray


class GASearcher:
    """Truncation-selection GA with gaussian mutation. ``method='ga'``.

    Offspring come first in the evaluated batch and the sort is stable, so a
    child that ties an archive member's score replaces it.
    """

    needs_descriptors = False
    # Recorded in the run config (`train_nes.searcher_resolved`).
    refresh = True
    variation = GAUSSIAN

    def __init__(self, num_params, population_size, elite_ratio=0.5,
                 sigma_init=0.1, cross_over_rate=0.0, init_scale=0.1,
                 init_around_mean=True):
        self.num_params = int(num_params)
        self.population_size = int(population_size)
        self.num_elites = max(1, int(population_size * elite_ratio))
        self.num_offspring = self.population_size - self.num_elites
        if self.num_offspring < 1:
            raise ValueError('the GA needs elite_ratio < 1')
        self.sigma_init = float(sigma_init)
        self.init_scale = float(init_scale)
        # Where the archive starts. True: jittered copies of the seed policy,
        # the study's convention so every arm begins at one point. False: the
        # reference's init, `N(0, init_scale)` on every weight with no seed
        # policy. On the ant the two are not interchangeable: the seed policy's
        # actions are twice as large, every jittered copy falls within ~30
        # steps, while 4% of an N(0, 0.1) population stands for the whole
        # episode.
        self.init_around_mean = bool(init_around_mean)
        self.variation_params = resolve_params(
            GAUSSIAN, sigma=sigma_init, cross_over_rate=cross_over_rate)

    def init(self, key, mean):
        jitter = random.normal(key, (self.num_elites, self.num_params))
        archive = ((mean[None, :] if self.init_around_mean else 0.0)
                   + self.init_scale * jitter)
        return GAState(
            archive=archive,
            fitness=jnp.full((self.num_elites,), jnp.inf),
            sigma=jnp.asarray(self.sigma_init, dtype=jnp.float32),
            generation=jnp.asarray(0, dtype=jnp.int32))

    def ask(self, key, state):
        params = dict(self.variation_params, sigma=state.sigma)
        x = vary(GAUSSIAN, state.archive, key, self.num_offspring, params)
        # Offspring, then the archive to be re-scored.
        return jnp.concatenate([x, state.archive], axis=0), None

    def tell(self, state, aux, fitness, descriptors=None):
        loss = -fitness
        order = jnp.argsort(loss)[:self.num_elites]
        return state._replace(archive=aux[order], fitness=loss[order],
                              generation=state.generation + 1)

    def incumbent(self, state):
        """The best archive member. A GA has no mean; the mean of an elite
        archive is not itself a policy that was ever evaluated."""
        return state.archive[0]

    def population_mean(self, state):
        """The coordinate-wise mean of the elite archive.

        NOT a policy this search ever evaluated, and not one it would return:
        `incumbent` is. It exists so a figure can ask whether the archive has
        collapsed onto one solution -- if it has, scoring the mean and scoring
        the best agree. Two networks that compute the same function can differ
        by a permutation of their hidden units, and the mean of such a pair is
        generally much worse than both, so a low value here is evidence of a
        spread archive, not of a bad search.
        """
        return jnp.mean(state.archive, axis=0)

    # The elite archive IS a persistent population: a genome that is best on
    # sub-task 0 can sit in it alongside one that is best on sub-task 1.
    has_population = True

    def population(self, state):
        return state.archive


class FocusGAState(NamedTuple):
    archive: jnp.ndarray      # (num_elites, num_params), best first
    fitness: jnp.ndarray      # (num_elites,) MINIMISED internally
    sigma: jnp.ndarray        # the gaussian width, in [sigma_min, sigma_init]
    focus: jnp.ndarray        # share of the archive, best first, bred from
    generation: jnp.ndarray


class FocusGASearcher(GASearcher):
    """The GA that consolidates its archive onto one solution.
    ``method='ga_focus'``, the Kinetix GA.

    Why: on Kinetix a GA's archive fills with many unrelated solvers of the
    current level, and their centroid solves nothing. Each generation the
    centroid is scored beside the archive (one evaluation, taken out of the
    offspring budget) and ``p`` is the share of the new archive it scores at
    least as well as. While ``p`` is below ``track_target``:

      sigma    shrinks, ``sigma <- clip(sigma * exp(sigma_rate (p - target)),
               sigma_min, sigma_init)``, so children stay on their parent's
               solution;
      focus    shrinks, ``focus <- clip(focus * exp(-focus_rate (target - p)),
               1 / num_elites, 1)``, and offspring are bred only from the best
               ``ceil(focus * num_elites)`` members, so the archive fills with
               the children of a few solvers.

    Both grow back once the centroid tracks the archive.

    ``explore_fraction`` of the offspring are always bred from the whole
    archive at ``sigma_init``: after the level changes, a population
    consolidated at a tiny sigma cannot search the new one, and the explorers
    do. Nothing here is told where a level boundary is.
    """

    def __init__(self, num_params, population_size, elite_ratio=0.5,
                 sigma_init=0.1, cross_over_rate=0.0, init_scale=0.1,
                 init_around_mean=True, focus_rate=0.3, sigma_rate=0.1,
                 track_target=0.5, sigma_min=None, explore_fraction=0.0):
        super().__init__(num_params, population_size, elite_ratio=elite_ratio,
                         sigma_init=sigma_init, cross_over_rate=cross_over_rate,
                         init_scale=init_scale,
                         init_around_mean=init_around_mean)
        self.num_offspring -= 1          # one evaluation for the centroid
        if self.num_offspring < 1:
            raise ValueError('ga_focus needs room for one centroid evaluation')
        self.focus_rate = float(focus_rate)
        self.sigma_rate = float(sigma_rate)
        self.track_target = float(track_target)
        self.sigma_min = (self.sigma_init / 100.0 if sigma_min is None
                          else float(sigma_min))
        self.focus_min = 1.0 / self.num_elites
        self.num_explore = int(round(float(explore_fraction)
                                     * self.num_offspring))

    # The width moves every generation, so the logged sigma is the state's.
    adapts_sigma = True

    def init(self, key, mean):
        # The split is kept so the archive starts from the same draw as in
        # the paper's runs.
        k_init, _ = random.split(key)
        s = super().init(k_init, mean)
        return FocusGAState(archive=s.archive, fitness=s.fitness,
                            sigma=s.sigma,
                            focus=jnp.asarray(1.0, dtype=jnp.float32),
                            generation=s.generation)

    def ask(self, key, state):
        if self.num_explore:
            key, k_explore = random.split(key)
        k_mut, _ = random.split(key)
        # The gaussian operator with both parents drawn from the best
        # ceil(focus * archive) members; the archive is stored best first.
        k_a, k_b, k_mate, k_eps = random.split(k_mut, 4)
        n = state.archive.shape[0]
        m = self.num_offspring
        pool = jnp.maximum(1.0, jnp.ceil(state.focus * n))
        ia = jnp.minimum((random.uniform(k_a, (m,)) * pool).astype(jnp.int32),
                         n - 1)
        ib = jnp.minimum((random.uniform(k_b, (m,)) * pool).astype(jnp.int32),
                         n - 1)
        take_b = (random.uniform(k_mate, (m, self.num_params))
                  > self.variation_params['cross_over_rate'])
        x = (state.archive[ia] * (1 - take_b) + state.archive[ib] * take_b
             + state.sigma * random.normal(k_eps, (m, self.num_params)))
        if self.num_explore:
            explorers = vary(GAUSSIAN, state.archive, k_explore,
                             self.num_explore,
                             dict(self.variation_params, sigma=self.sigma_init))
            x = x.at[:self.num_explore].set(explorers)
        centroid = jnp.mean(state.archive, axis=0, keepdims=True)
        # Offspring, the archive to be re-scored, then the centroid.
        return jnp.concatenate([x, state.archive, centroid], axis=0), None

    def tell(self, state, aux, fitness, descriptors=None):
        centroid_fitness, fitness = fitness[-1], fitness[:-1]
        evaluated = aux[:-1]
        # Ranked on the standardized score rather than the raw one; the order
        # is the same up to float rounding, and this is the arithmetic the
        # paper's runs used.
        rank_score = (fitness - jnp.mean(fitness)) / (jnp.std(fitness) + 1e-12)
        chosen = jnp.argsort(-rank_score)[:self.num_elites]
        chosen = chosen[jnp.argsort(-fitness[chosen])]     # best first
        kept = fitness[chosen]
        share = jnp.mean(centroid_fitness >= kept)
        sigma = jnp.clip(state.sigma * jnp.exp(self.sigma_rate
                                               * (share - self.track_target)),
                         self.sigma_min, self.sigma_init)
        focus = jnp.clip(state.focus * jnp.exp(-self.focus_rate
                                               * (self.track_target - share)),
                         self.focus_min, 1.0)
        return FocusGAState(archive=evaluated[chosen], fitness=-kept,
                            sigma=sigma, focus=focus,
                            generation=state.generation + 1)
