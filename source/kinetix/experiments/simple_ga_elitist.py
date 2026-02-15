"""Simple Genetic Algorithm with (μ+λ) Elitist Archive (Such et al., 2017).

Like evosax v2 SimpleGA but maintains a persistent elite archive across
generations. New candidates compete with the archive, and only improvements
survive — good solutions are never lost.

[1] https://arxiv.org/abs/1712.06567
"""

from collections.abc import Callable
from functools import partial

import jax
import jax.numpy as jnp
import optax
from flax import struct

from evosax.core.fitness_shaping import identity_fitness_shaping_fn
from evosax.types import Fitness, Population, Solution
from evosax.algorithms.population_based.base import (
    Params as BaseParams,
    PopulationBasedAlgorithm,
    State as BaseState,
    metrics_fn,
)


@struct.dataclass
class ElitistState(BaseState):
    population: Population      # full population (popsize) — used by base class init
    fitness: Fitness             # full population fitness
    archive: Population          # elite archive (num_elites x num_dims)
    archive_fitness: Fitness     # elite archive fitness (num_elites,)
    std: jax.Array


@struct.dataclass
class ElitistParams(BaseParams):
    crossover_rate: float


class SimpleGA_Elitist(PopulationBasedAlgorithm):
    """Simple GA with (μ+λ) elitist archive selection.
    
    Maintains a persistent archive of elite solutions. Each generation,
    new offspring are combined with the archive and only the top elite_ratio
    fraction survives. This ensures good solutions are never lost.
    """

    def __init__(
        self,
        population_size: int,
        solution: Solution,
        std_schedule: Callable = optax.constant_schedule(1.0),
        fitness_shaping_fn: Callable = identity_fitness_shaping_fn,
        metrics_fn: Callable = metrics_fn,
    ):
        """Initialize Simple GA with Elitism."""
        super().__init__(population_size, solution, fitness_shaping_fn, metrics_fn)

        self.elite_ratio = 0.5

        # std schedule
        self.std_schedule = std_schedule

    @property
    def _default_params(self) -> ElitistParams:
        return ElitistParams(crossover_rate=0.0)

    def _init(self, key: jax.Array, params: ElitistParams) -> ElitistState:
        state = ElitistState(
            # population/fitness are required by PopulationBasedAlgorithm.init()
            population=jnp.full((self.population_size, self.num_dims), jnp.nan),
            fitness=jnp.full((self.population_size,), jnp.inf),
            # The elite archive — sized to num_elites
            archive=jnp.full((self.num_elites, self.num_dims), jnp.nan),
            archive_fitness=jnp.full((self.num_elites,), jnp.inf),
            std=self.std_schedule(0),
            best_solution=jnp.full((self.num_dims,), jnp.nan),
            best_fitness=jnp.inf,
            generation_counter=0,
        )
        return state

    @partial(jax.jit, static_argnames=("self",))
    def init(
        self,
        key: jax.Array,
        population: Population,
        fitness: Fitness,
        params: ElitistParams,
    ) -> ElitistState:
        """Initialize and seed the elite archive from the initial population."""
        # Call parent init (sets population & fitness on state)
        state = super().init(key, population, fitness, params)
        # Seed archive with the top num_elites from the initial population
        idx = jnp.argsort(state.fitness)[:self.num_elites]
        archive = state.population[idx]
        archive_fitness = state.fitness[idx]
        state = state.replace(archive=archive, archive_fitness=archive_fitness)
        return state

    def _ask(
        self,
        key: jax.Array,
        state: ElitistState,
        params: ElitistParams,
    ) -> tuple[Population, ElitistState]:
        # Parents are selected uniformly from the elite archive
        archive = state.archive
        elite_ids = jnp.arange(self.num_elites)

        key_crossover, key_mutation, key_a, key_b = jax.random.split(key, 4)
        key_crossover = jax.random.split(key_crossover, self.population_size)
        key_mutation = jax.random.split(key_mutation, self.population_size)

        # Select parents uniformly from archive
        idx_a = jax.random.choice(key_a, elite_ids, (self.population_size,))
        idx_b = jax.random.choice(key_b, elite_ids, (self.population_size,))
        parents_1 = archive[idx_a]
        parents_2 = archive[idx_b]

        # Crossover
        population = jax.vmap(crossover, in_axes=(0, 0, 0, None))(
            key_crossover, parents_1, parents_2, params.crossover_rate
        )

        # Mutation
        population = jax.vmap(mutation, in_axes=(0, 0, None))(
            key_mutation, population, state.std
        )

        return population, state

    def _tell(
        self,
        key: jax.Array,
        population: Population,
        fitness: Fitness,
        state: ElitistState,
        params: ElitistParams,
    ) -> ElitistState:
        # (μ+λ) selection: combine new population with archive, keep top
        combined_pop = jnp.concatenate([population, state.archive], axis=0)
        combined_fit = jnp.concatenate([fitness, state.archive_fitness], axis=0)

        # Select top num_elites
        idx = jnp.argsort(combined_fit)[:self.num_elites]
        new_archive = combined_pop[idx]
        new_archive_fitness = combined_fit[idx]

        return state.replace(
            population=population,
            fitness=fitness,
            archive=new_archive,
            archive_fitness=new_archive_fitness,
            std=self.std_schedule(state.generation_counter),
        )


def crossover(
    key: jax.Array, parent_1: Solution, parent_2: Solution, crossover_rate: float
) -> Solution:
    """Crossover between two parents."""
    mask = jax.random.uniform(key, parent_1.shape) > crossover_rate
    return parent_1 * (1 - mask) + parent_2 * mask


def mutation(key: jax.Array, solution: Solution, std: jax.Array) -> Solution:
    """Mutation of a solution."""
    return solution + std * jax.random.normal(key, solution.shape)
