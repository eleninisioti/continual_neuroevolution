"""SimpleGA -- the genetic algorithm arm. The only GA in the codebase.

Truncation selection on fitness, then variation of the survivors:
(mu+lambda) elitism, where `mu = max(1, elite_ratio * popsize)` genomes are kept
in an archive, `popsize` offspring are bred from it each generation, and `tell`
keeps the best `mu` of the two sets combined. A good genome is therefore never
lost.

The variation operator is a CHOICE, not part of the method: `variation`
selects gaussian mutation (this GA as published, the default) or DNS's
Iso+LineDD. Both live in `source/algorithms/ne/variation.py`. This exists so
that a GA/DNS comparison can hold the operator fixed and vary only the
selection rule -- otherwise "novelty selection helps" is confounded with
"recombination helps", which is what the two arms differed by until
2026-09-08.

Standalone by design -- no evosax, no QDax, nothing that knows what an
environment is. The caller passes flat genotypes and their fitnesses.

## What this replaced (2026-08-06)

Four code paths ran three different algorithms:

  gymnax            this file, via `--ga_version kinetix`
  kinetix           `experiments/simple_ga_elitist.py`, the same algorithm
                    reimplemented on the evosax v2 base class
  mujoco cheetah    `evosax.algorithms.SimpleGA`, which has **no elitism at
                    all** -- its `tell` overwrites the archive with the
                    offspring. A plain generational GA, not the method the
                    paper describes.
  brax ant          the same evosax GA, plus a manual patch that copied the
                    previous generation's top genomes over the first slots of
                    `ask`'s output. That preserved genomes but not their
                    fitness, and re-evaluated them, so an elite could still be
                    lost. Removed: elitism is in `tell` now, where it belongs.

Cheetah and ant therefore change behaviour and are re-run. Kinetix keeps the
same algorithm but not the same random numbers -- it initialised the archive
from `uniform(-1, 1)` rather than `normal * 0.1`, and its crossover mask
compared `<` where this one compares `>` (with `cross_over_rate = 0.0` both are
"take one uniformly drawn parent", so the search is identical in distribution
but not trajectory-by-trajectory). See docs/unify_implementations.md.

Sigma -- the gaussian operator's width -- follows one mechanism,
multiplicative decay (`sigma_decay` per generation, floored at `sigma_limit`),
which covers every schedule the four trainers used. Under `isoline` there is no
such width and a `sigma_decay != 1.0` is rejected rather than ignored. The one thing it cannot express is ant's per-sub-task restart,
so that is `reset_sigma`, called explicitly at a task boundary by the caller
that wants it -- visible in the training loop rather than hidden in a lambda.
"""
from typing import Callable, Optional, Tuple
import jax
import jax.numpy as jnp
import chex
from flax import struct
from functools import partial

from source.algorithms.ne.variation import (
    GAUSSIAN, ISOLINE, VARIATIONS, resolve_params, vary,
)


def exp_decay(value: float, decay: float, limit: float) -> float:
    """Exponential decay with a limit."""
    return jnp.maximum(value * decay, limit)


@struct.dataclass
class EvoState:
    mean: chex.Array
    archive: chex.Array
    fitness: chex.Array
    sigma: float
    best_member: chex.Array
    best_fitness: float = jnp.finfo(jnp.float32).max
    gen_counter: int = 0


@struct.dataclass
class EvoParams:
    cross_over_rate: float = 0.0
    sigma_init: float = 0.07
    sigma_decay: float = 1.0
    sigma_limit: float = 0.0001
    init_scale: float = 0.1
    clip_min: float = -jnp.finfo(jnp.float32).max
    clip_max: float = jnp.finfo(jnp.float32).max


class SimpleGA:
    """Simple Genetic Algorithm (Such et al., 2017)
    Reference: https://arxiv.org/abs/1712.06567
    Inspired by: https://github.com/hardmaru/estool/blob/master/es.py

    Fitness is **minimised**, as in evosax. Every caller maximises a return, so
    every caller passes `-fitness` to `tell`.
    """

    def __init__(
        self,
        popsize: int,
        num_dims: int,
        elite_ratio: float = 0.5,
        sigma_init: float = 0.1,
        sigma_decay: float = 1.0,
        sigma_limit: float = 0.0001,
        init_scale: float = 0.1,
        cross_over_rate: float = 0.0,
        variation: str = GAUSSIAN,
        iso_sigma: Optional[float] = None,
        line_sigma: Optional[float] = None,
    ):
        self.popsize = popsize
        self.num_dims = num_dims
        self.elite_ratio = elite_ratio
        self.elite_popsize = max(1, int(self.popsize * self.elite_ratio))
        self.strategy_name = "SimpleGA"

        # Set core kwargs es_params
        self.sigma_init = sigma_init
        self.sigma_decay = sigma_decay
        self.sigma_limit = sigma_limit
        self.init_scale = init_scale
        self.cross_over_rate = cross_over_rate

        # Which operator breeds the offspring. `gaussian` is this method as
        # published (Such et al., 2017) and the default; `isoline` is DNS's
        # Iso+LineDD, available here so that the GA/DNS comparison can hold the
        # operator fixed and vary only the selection rule. See
        # source/algorithms/ne/variation.py.
        if variation not in VARIATIONS:
            raise ValueError(f"variation must be one of {VARIATIONS}, "
                             f"not {variation!r}")
        self.variation = variation
        if variation == GAUSSIAN:
            self.variation_params = resolve_params(
                GAUSSIAN, sigma=sigma_init, cross_over_rate=cross_over_rate)
        else:
            # `sigma_decay` has nothing to decay under Iso+LineDD -- the
            # operator's widths are the iso/line pair, and DNS never decays
            # them. Silently ignoring a decay the caller asked for would make
            # a GA/isoline arm quietly not the schedule its config records, so
            # this is an error rather than a no-op.
            if sigma_decay != 1.0:
                raise ValueError(
                    "sigma_decay applies to the gaussian operator's width; "
                    "the isoline operator has no decaying width. Pass "
                    "sigma_decay=1.0, or decay iso_sigma/line_sigma "
                    "explicitly if that is what you want.")
            self.variation_params = resolve_params(
                ISOLINE, iso_sigma=iso_sigma, line_sigma=line_sigma)

    @property
    def num_elites(self) -> int:
        """Archive size. Alias for `elite_popsize`, the name evosax uses."""
        return self.elite_popsize

    @property
    def default_params(self) -> EvoParams:
        """Return default parameters of evolution strategy."""
        return EvoParams(
            sigma_init=self.sigma_init,
            sigma_decay=self.sigma_decay,
            sigma_limit=self.sigma_limit,
            init_scale=self.init_scale,
            cross_over_rate=self.cross_over_rate,
        )

    def init(
        self,
        rng: chex.PRNGKey,
        params: Optional[EvoParams] = None,
        init_archive: Optional[chex.Array] = None,
    ) -> EvoState:
        """Initialize the evolution strategy.

        The archive holds `elite_popsize` genomes, not `popsize`: it is the
        surviving elite, and `ask` breeds `popsize` offspring from it. Passing a
        population of the wrong size was the easiest mistake to make against the
        old evosax API, so the archive is drawn here by default. `init_archive`
        overrides it for a warm start, and must be `(elite_popsize, num_dims)`.
        """
        if params is None:
            params = self.default_params

        if init_archive is None:
            # Normal initialization (like evosax) instead of uniform.
            initialization = (
                jax.random.normal(rng, (self.elite_popsize, self.num_dims))
                * params.init_scale
            )
        else:
            initialization = jnp.asarray(init_archive)
            if initialization.shape != (self.elite_popsize, self.num_dims):
                raise ValueError(
                    f"init_archive must be "
                    f"({self.elite_popsize}, {self.num_dims}), the archive "
                    f"size, not {initialization.shape}. The archive is "
                    f"elite_ratio * popsize genomes, not popsize."
                )

        state = EvoState(
            mean=initialization.mean(axis=0),
            archive=initialization,
            fitness=jnp.zeros(self.elite_popsize) + jnp.finfo(jnp.float32).max,
            sigma=params.sigma_init,
            best_member=initialization.mean(axis=0),
        )
        return state

    @partial(jax.jit, static_argnums=(0,))
    def _breed(self, rng, state, params):
        """One generation of offspring, with the genome each one descends from.

        The body of `ask`. Split out so `ask_with_parents` can report the
        pairing without a second copy of this key schedule -- two copies would
        desynchronise the moment the schedule changed, and the pairing would
        then be silently wrong rather than obviously broken.

        Which of the two drawn parents is the base differs by operator and is
        the non-obvious one in both cases; `source/algorithms/ne/variation.py`
        works it out and returns it, so nothing here has to know.
        """
        variation_params = dict(self.variation_params)
        if self.variation == GAUSSIAN:
            # The one parameter that moves during a run: `state.sigma` carries
            # the decay schedule, so it overrides the width the operator was
            # configured with rather than the other way round.
            variation_params['sigma'] = state.sigma

        x, parents = vary(self.variation, state.archive, rng, self.popsize,
                          variation_params, return_parents=True)

        # Clip to bounds
        x = jnp.clip(x, params.clip_min, params.clip_max)

        return x, parents

    def ask(
        self,
        rng: chex.PRNGKey,
        state: EvoState,
        params: Optional[EvoParams] = None,
    ) -> Tuple[chex.Array, EvoState]:
        """Ask for new parameter candidates to evaluate next."""
        if params is None:
            params = self.default_params
        x, _ = self._breed(rng, state, params)
        return x, state

    def ask_with_parents(
        self,
        rng: chex.PRNGKey,
        state: EvoState,
        params: Optional[EvoParams] = None,
    ):
        """`ask`, plus the genome each offspring was bred from.

        Returns `(x, state, parents)` with `parents[i]` the archive member that
        offspring `x[i]` descends from. The plasticity churn column needs this
        pairing: it measures the effect of ONE update on ONE network, which
        elite-to-elite cannot do because the elite changes lineage. See
        `pairwise_churn` in `source/metrics/plasticity.py`.

        Identical offspring to `ask` for the same key -- both go through
        `_breed`, so this is a strictly additional return and switching a
        trainer from one to the other does not change its search.

        WHICH PARENT, given there are two, depends on the operator and is the
        non-obvious one either way -- `b` (the second draw) under gaussian
        mutation, `x1` (the first) under Iso+LineDD. The reasoning for both is
        in `source/algorithms/ne/variation.py`, which returns the base parent
        so nothing here has to re-derive it.

        Under gaussian mutation the pairing was checked by comparing
        `std(offspring - parent)` against `state.sigma`: it reads 0.166 against
        a sigma of 0.100 with the wrong parent, and exactly sigma with the
        right one. Keep that assertion if you touch this.

        Raise `cross_over_rate` above 0, or run the isoline operator at a
        `line_sigma` large enough to land the offspring nearer `x2` than `x1`,
        and "the parent" stops being well defined -- this column then needs
        rethinking rather than relabelling.
        """
        if params is None:
            params = self.default_params
        x, parents = self._breed(rng, state, params)
        return x, state, parents

    @partial(jax.jit, static_argnums=(0,))
    def tell(
        self,
        x: chex.Array,
        fitness: chex.Array,
        state: EvoState,
        params: Optional[EvoParams] = None,
    ) -> EvoState:
        """Tell performance data for strategy state update.

        `fitness` is minimised. Takes no random key: selection is deterministic.
        """
        if params is None:
            params = self.default_params

        # Combine current elite and recent generation info
        fitness_combined = jnp.concatenate([fitness, state.fitness])
        solution = jnp.concatenate([x, state.archive])

        # Select top elite from total archive info
        idx = jnp.argsort(fitness_combined)[0 : self.elite_popsize]
        fitness_new = fitness_combined[idx]
        archive = solution[idx]

        # Update mutation epsilon - multiplicative decay
        sigma = exp_decay(state.sigma, params.sigma_decay, params.sigma_limit)

        # Set mean to best member seen so far
        improved = fitness_new[0] < state.best_fitness
        best_member = jax.lax.select(improved, archive[0], state.best_member)
        best_fitness = jax.lax.select(improved, fitness_new[0], state.best_fitness)

        return state.replace(
            fitness=fitness_new,
            archive=archive,
            sigma=sigma,
            mean=best_member,
            best_member=best_member,
            best_fitness=best_fitness,
            gen_counter=state.gen_counter + 1,
        )

    def reset_sigma(
        self, state: EvoState, sigma: Optional[float] = None
    ) -> EvoState:
        """Restart the mutation schedule, keeping the archive.

        For continual settings that want generation-0 mutation noise back at
        each sub-task boundary. `sigma=None` restores `sigma_init`.
        """
        return state.replace(
            sigma=self.sigma_init if sigma is None else sigma
        )

    @staticmethod
    def decay_for(sigma_init: float, sigma_final: float, num_generations: int) -> float:
        """Per-generation `sigma_decay` that takes `sigma_init` to `sigma_final`.

        The trainers all spelled this out inline as
        `(final / init) ** (1 / (gens - 1))`; it is here so that "decay to
        sigma_final over the run" means one thing everywhere.
        """
        if sigma_final is None or sigma_final == sigma_init:
            return 1.0
        return float((sigma_final / sigma_init) ** (1.0 / max(1, num_generations - 1)))

    def elite_mean(self, state: EvoState) -> chex.Array:
        """Mean of the elite archive -- the incumbent the search carries forward."""
        return state.archive.mean(axis=0)
