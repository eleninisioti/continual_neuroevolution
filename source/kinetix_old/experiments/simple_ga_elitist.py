"""Simple Genetic Algorithm with (μ+λ) Elitism (Such et al., 2017).

Matches the behaviour of the custom evosax-v1 SimpleGA in
``inspiration/for_GA``:

* An **elite archive** of size ``popsize × elite_ratio`` is stored.
* ``ask`` generates ``popsize`` offspring from the archive via crossover
  + mutation.
* ``tell`` merges the ``popsize`` offspring with the current archive and
  keeps only the top ``num_elites`` by fitness.

This means the selection pressure is (popsize + num_elites) → num_elites,
exactly like the inspiration implementation.

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
    population: Population      # (num_elites, num_dims) — the elite archive
    fitness: Fitness             # (num_elites,)
    std: jax.Array


@struct.dataclass
class Params(BaseParams):
    crossover_rate: float


class SimpleGA_Elitist(PopulationBasedAlgorithm):
    """Simple GA with (μ+λ) elitist selection.

    Stores an elite archive of ``num_elites`` individuals.  Each
    generation, ``popsize`` offspring are bred from the archive.  In
    ``tell``, offspring + archive are merged and only the top
    ``num_elites`` survive — matching the inspiration codebase exactly.
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
            population=jnp.full((self.num_elites, self.num_dims), jnp.nan),
            fitness=jnp.full((self.num_elites,), jnp.inf),
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
        # Sort archive by fitness — all num_elites are potential parents
        idx = jnp.argsort(state.fitness)
        sorted_pop = state.population[idx]
        elite_ids = jnp.arange(self.num_elites)

        key_crossover, key_mutation, key_a, key_b = jax.random.split(key, 4)
        key_crossover = jax.random.split(key_crossover, self.population_size)
        key_mutation = jax.random.split(key_mutation, self.population_size)

        # Select two parents uniformly from the archive
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
        # (μ+λ) selection: merge offspring with archive, keep top num_elites
        combined_pop = jnp.concatenate([population, state.population], axis=0)
        combined_fit = jnp.concatenate([fitness, state.fitness], axis=0)

        # Select top num_elites
        idx = jnp.argsort(combined_fit)[:self.num_elites]
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
    mask = jax.random.uniform(key, parent_1.shape) < crossover_rate
    return parent_1 * (1 - mask) + parent_2 * mask


def mutation(key: jax.Array, solution: Solution, std: jax.Array) -> Solution:
    """Mutation of a solution."""
    return solution + std * jax.random.normal(key, solution.shape)
