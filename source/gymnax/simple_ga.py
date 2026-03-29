"""
Standalone SimpleGA implementation for gymnax.
Based on the kinetix evosax SimpleGA but with no external dependencies.
"""
from typing import Tuple, Optional
import jax
import jax.numpy as jnp
import chex
from flax import struct
from functools import partial


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
    init_min: float = -1.0
    init_max: float = 1.0
    clip_min: float = -jnp.finfo(jnp.float32).max
    clip_max: float = jnp.finfo(jnp.float32).max


def single_mate(
    rng: chex.PRNGKey, a: chex.Array, b: chex.Array, cross_over_rate: float
) -> chex.Array:
    """Only cross-over dims for x% of all dims."""
    idx = jax.random.uniform(rng, (a.shape[0],)) > cross_over_rate
    cross_over_candidate = a * (1 - idx) + b * idx
    return cross_over_candidate


class SimpleGA:
    """Simple Genetic Algorithm (Such et al., 2017)
    Reference: https://arxiv.org/abs/1712.06567
    Inspired by: https://github.com/hardmaru/estool/blob/master/es.py
    """

    def __init__(
        self,
        popsize: int,
        num_dims: int,
        elite_ratio: float = 0.5,
        sigma_init: float = 0.1,
        sigma_decay: float = 1.0,
        sigma_limit: float = 0.01,
        init_min: float = -1.0,
        init_max: float = 1.0,
        cross_over_rate: float = 0.0,
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
        self.init_min = init_min
        self.init_max = init_max
        self.cross_over_rate = cross_over_rate

    @property
    def default_params(self) -> EvoParams:
        """Return default parameters of evolution strategy."""
        return EvoParams(
            sigma_init=self.sigma_init,
            sigma_decay=self.sigma_decay,
            sigma_limit=self.sigma_limit,
            init_min=self.init_min,
            init_max=self.init_max,
            cross_over_rate=self.cross_over_rate,
        )

    def initialize(
        self,
        rng: chex.PRNGKey,
        params: Optional[EvoParams] = None,
    ) -> EvoState:
        """Initialize the evolution strategy."""
        if params is None:
            params = self.default_params

        # Use normal initialization (like evosax) instead of uniform
        initialization = jax.random.normal(
            rng,
            (self.elite_popsize, self.num_dims),
        ) * 0.1
        
        state = EvoState(
            mean=initialization.mean(axis=0),
            archive=initialization,
            fitness=jnp.zeros(self.elite_popsize) + jnp.finfo(jnp.float32).max,
            sigma=params.sigma_init,
            best_member=initialization.mean(axis=0),
        )
        return state

    @partial(jax.jit, static_argnums=(0,))
    def ask(
        self,
        rng: chex.PRNGKey,
        state: EvoState,
        params: Optional[EvoParams] = None,
    ) -> Tuple[chex.Array, EvoState]:
        """Ask for new parameter candidates to evaluate next."""
        if params is None:
            params = self.default_params

        rng, rng_eps, rng_idx_a, rng_idx_b = jax.random.split(rng, 4)
        rng_mate = jax.random.split(rng, self.popsize)
        
        epsilon = (
            jax.random.normal(rng_eps, (self.popsize, self.num_dims))
            * state.sigma
        )
        
        elite_ids = jnp.arange(self.elite_popsize)
        idx_a = jax.random.choice(rng_idx_a, elite_ids, (self.popsize,))
        idx_b = jax.random.choice(rng_idx_b, elite_ids, (self.popsize,))
        members_a = state.archive[idx_a]
        members_b = state.archive[idx_b]
        
        x = jax.vmap(single_mate, in_axes=(0, 0, 0, None))(
            rng_mate, members_a, members_b, self.cross_over_rate
        )
        x += epsilon
        
        # Clip to bounds
        x = jnp.clip(x, params.clip_min, params.clip_max)
        
        return x, state

    @partial(jax.jit, static_argnums=(0,))
    def tell(
        self,
        x: chex.Array,
        fitness: chex.Array,
        state: EvoState,
        params: Optional[EvoParams] = None,
    ) -> EvoState:
        """Tell performance data for strategy state update."""
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
