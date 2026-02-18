"""Simple Genetic Algorithm with (μ+λ) Elitism (Such et al., 2017).

Like the evosax v1 SimpleGA: the stored population includes the elite.
Each generation, ``popsize`` offspring are produced from the top
``elite_ratio`` fraction (parents).  In ``tell``, offspring and the
current population are merged and only the top ``popsize`` survive.
Good solutions are never lost, and total stored memory is a single
(popsize × num_dims) array — same as vanilla SimpleGA.

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
class State(BaseState):
    population: Population      # (popsize, num_dims) — elites live at the top
    fitness: Fitness             # (popsize,)
    std: jax.Array


@struct.dataclass
class Params(BaseParams):
    crossover_rate: float


class SimpleGA_Elitist(PopulationBasedAlgorithm):
    """Simple GA with (μ+λ) elitist selection.

    The stored population *is* the archive. The top ``num_elites``
    individuals (sorted by fitness) act as parents for the next
    generation.  After evaluation, offspring compete with the current
    population and only the best ``popsize`` survive.  Memory usage
    is identical to vanilla SimpleGA — a single (popsize × num_dims)
    array.
    """

    def __init__(
        self,
        population_size: int,
        solution: Solution,
        std_schedule: Callable = optax.constant_schedule(1.0),
        fitness_shaping_fn: Callable = identity_fitness_shaping_fn,
        metrics_fn: Callable = metrics_fn,
        elite_ratio: float = 0.5,
    ):
        """Initialize Simple GA with Elitism."""
        super().__init__(population_size, solution, fitness_shaping_fn, metrics_fn)

        self.elite_ratio = elite_ratio

        # std schedule
        self.std_schedule = std_schedule

    @property
    def _default_params(self) -> Params:
        return Params(crossover_rate=0.0)

    def _init(self, key: jax.Array, params: Params) -> State:
        return State(
            population=jnp.full((self.population_size, self.num_dims), jnp.nan),
            fitness=jnp.full((self.population_size,), jnp.inf),
            std=self.std_schedule(0),
            best_solution=jnp.full((self.num_dims,), jnp.nan),
            best_fitness=jnp.inf,
            generation_counter=0,
        )

    def _ask(
        self,
        key: jax.Array,
        state: State,
        params: Params,
    ) -> tuple[Population, State]:
        # Sort population by fitness — top num_elites are the parents
        idx = jnp.argsort(state.fitness)
        sorted_pop = state.population[idx]
        elite_ids = jnp.arange(self.num_elites)

        key_crossover, key_mutation, key_a, key_b = jax.random.split(key, 4)
        key_crossover = jax.random.split(key_crossover, self.population_size)
        key_mutation = jax.random.split(key_mutation, self.population_size)

        # Select two parents uniformly from the elite portion
        idx_a = jax.random.choice(key_a, elite_ids, (self.population_size,))
        idx_b = jax.random.choice(key_b, elite_ids, (self.population_size,))
        parents_1 = sorted_pop[idx_a]
        parents_2 = sorted_pop[idx_b]

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
        state: State,
        params: Params,
    ) -> State:
        # (μ+λ) selection: merge offspring with current population, keep top popsize
        combined_pop = jnp.concatenate([population, state.population], axis=0)
        combined_fit = jnp.concatenate([fitness, state.fitness], axis=0)

        # Select top popsize
        idx = jnp.argsort(combined_fit)[:self.population_size]
        new_pop = combined_pop[idx]
        new_fit = combined_fit[idx]

        return state.replace(
            population=new_pop,
            fitness=new_fit,
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
