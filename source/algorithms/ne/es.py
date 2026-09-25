"""The distribution-based NE arm: one implementation, NES and OpenES as settings.

One centroid, a Gaussian search distribution around it, and a step along the
*search gradient*: sample perturbations, score them, move the centroid along the
fitness-weighted average perturbation. NES and OpenES are the same search with
two knobs turned, so they are one code path here and not two:

    shaping     'zscore' standardizes, (f - mean) / std, keeping *how much*
                better a perturbation was. 'centered_rank' keeps only the order
                and throws the magnitudes away. Under a task switch that is the
                whole story: ranks cannot tell "one perturbation escaped the
                basin" from "one was slightly better than its neighbours", so
                the step after a switch is the same size as any other step.
    optimizer   'sgd' carries no state across a switch. 'adam' has a
                per-coordinate second-moment estimate that is a long-run
                average, and a task switch invalidates it -- the optimizer keeps
                normalising by statistics collected on a landscape that no
                longer exists.

    NES     = zscore        + sgd     (the textbook (1, lambda) search gradient)
    OpenES  = centered_rank + adam    (evosax `Open_ES`, bit-for-bit -- see
                                       `scripts/check_es.py`, check 2)

Both are hypotheses about plasticity, which is what both studies in this repo
measure, so making them one class with two settings is not tidying: it is what
makes an `openes` vs `nes` row in a table the difference between those two
choices and nothing else.

## What is in here

    shape_fitness, es_grads     the arithmetic. ~15 lines, and the ONLY copy.
    ES                          the plain interface: init(mean) / ask -> (pop,
                                eps) / tell(state, eps, fitness), fitness
                                MAXIMISED. Used by `source/studies/generalists/ne.py`'s
                                searcher adapters, next to GA and DNS, and by
                                the toy-landscape scripts.
    MultiES                     M centroids sharing one evaluation budget, a
                                (mu, lambda) at the distribution level. vmaps
                                the core, so it is the same search M times over.
    EvosaxES                    the evosax `DistributionBasedAlgorithm`
                                interface over the same core, so a trainer built
                                around evosax swaps `Open_ES` for this and
                                changes nothing else. Holds no arithmetic.

`ES` and `EvosaxES` are two *interfaces*, not two implementations: neither one
computes a gradient, they both call `es_grads`. The split exists because their
callers genuinely differ -- the gymnax trainers drive a sharded evosax object
with `default_params` / `best_solution` / `_unravel_solution`, while the
generalists searcher protocol is keyless in `tell` and hands GA, DNS and ES the
same shape. Collapsing those two into one call signature would mean rewriting
one caller to pretend it is the other.

## sigma, and the three things it does at once

`sigma` is the standard deviation of the search distribution: `ask` returns
`mean + sigma * eps` with `eps ~ N(0, I)`. It is stored per coordinate (as
`log_sigma`) so the fixed and adapted cases have one shape, and stays at
`sigma_init` everywhere unless `sigma_lr > 0` turns on the separable-NES update
for it. Off by default, so NES and OpenES differ only in shaping and step rule.

1. **Sampling radius.** A population member sits at distance about
   `sigma * sqrt(d)` from the centroid -- the norm of a d-dimensional Gaussian
   concentrates. At d=386 and sigma=0.1 that is 1.96, which is not a small
   distance in this repo's parameter space.

2. **Smoothing bandwidth.** The search does not ascend `f`. It ascends
   `E_eps[f(theta + sigma*eps)]`, i.e. `f` convolved with a Gaussian of width
   sigma, so structure thinner than sigma is averaged away and the search cannot
   see, or stand on, a ridge narrower than its own sampling radius. This is why
   sigma behaves as an *inverse ruggedness* knob.

3. **Step size**, through the `1/sigma` in `es_grads`. With standardized fitness
   the utilities are scale-free, so the estimated gradient has magnitude
   ~`1/sigma` and the centroid moves about `learning_rate / sigma` per
   generation. Lowering sigma at a fixed learning rate therefore *lengthens* the
   step -- the opposite of the intuition that a smaller search radius means
   smaller moves.

Because of (3), `sigma` and `learning_rate` are not independent knobs, and a
sweep that varies one without the other is measuring step size. Any sweep over
sigma here holds `lr / sigma` fixed for that reason, and says so where it does.

## Sign convention

The core functions take fitness **MAXIMISED** and return **ascent** directions,
because every caller in this repo maximises a return. `EvosaxES` negates once at
its boundary, since evosax minimises.

Both shapings are monotone, so it does not matter mathematically whether that
negation happens before or after `shape_fitness`. It matters in float32: under
'centered_rank', `ranks/(n-1) - 0.5` computed on `-f` and negated computed on
`f` round apart by about 3e-8. `EvosaxES` therefore shapes the minimised fitness
and negates the utilities, which is the order that hands `shape_fitness` exactly
the input evosax hands its own shaping function -- and is what makes the
`Open_ES` reproduction bit-for-bit rather than merely close.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import NamedTuple

import jax
import jax.numpy as jnp
import optax
from flax import struct
from jax import random

from evosax.algorithms.distribution_based.base import (
    DistributionBasedAlgorithm,
    Params as BaseParams,
    State as BaseState,
    metrics_fn,
)
from evosax.core.fitness_shaping import identity_fitness_shaping_fn
from evosax.types import Fitness, Population, Solution

SHAPINGS = ('zscore', 'centered_rank', 'raw')
OPTIMIZERS = ('sgd', 'sgd_momentum', 'adam')


# ---------------------------------------------------------------------------
# The core. Everything else in this file is an interface over these two.
# ---------------------------------------------------------------------------

def shape_fitness(fitness, shaping):
    """Raw fitness (MAXIMISED) -> utilities (MAXIMISED), along the last axis.

    Batched: with a leading axis this shapes each row independently, which is
    what `MultiES` needs -- each centroid's utilities come from its OWN
    sub-population, so it is a genuinely separate search rather than one search
    with a fancier proposal.
    """
    if shaping == 'raw':
        return fitness
    if shaping == 'zscore':
        # True standardization wherever there is a spread to standardize, and
        # an explicit zero where there is not. The flat case is not exotic here:
        # a whole population of MountainCar episodes timing out at -200, or of
        # CartPole episodes at the 500 cap, is one generation of no signal, and
        # the step must be 0.
        #
        # NOT `jax.nn.standardize` (what evosax's `standardize_fitness_shaping_fn`
        # calls), which is `(f - mean) * rsqrt(var + 1e-8)` over a variance it
        # computes as `E[f^2] - E[f]^2`. On a constant population that
        # cancellation can land slightly NEGATIVE -- at f = 0.55 it does -- and
        # `rsqrt` of a negative is nan, which then eats the mean and every
        # generation after it. Where the cancellation happens to give exactly 0
        # (f = -200, f = 500) it is merely a damped step rather than the unit
        # variance the name promises.
        #
        # `1e-8` is an ABSOLUTE tolerance on a quantity that scales with |f|,
        # and `jnp.mean` of n identical float32s is not always that float32: at
        # a flat population of 0.7 the mean is one ulp out, `std` comes back at
        # 1.2e-7, the guard misses and the utilities are O(1) rounding noise.
        # It fires at 500, -200, 0.5, 1.0 and 100 -- every cap this repo's
        # environments actually saturate at -- and misses at 0.7, 0.55, 0.35.
        # Harmless where it misses, because `ask` samples antithetically: a
        # +/- pair on a flat population gets one fitness, hence one utility,
        # and its two contributions to `es_grads` cancel exactly. Measured on
        # section A's landscape (cap 0.7), the resulting step is 1.7e-10 and
        # the centroid drifts 0.0000 over 1000 plateau generations. Left as is
        # rather than made relative: the values on disk were produced by this
        # arithmetic, and a relative tolerance would be a change to the search
        # under finished runs for no measured difference.
        centered = fitness - jnp.mean(fitness, axis=-1, keepdims=True)
        std = jnp.std(fitness, axis=-1, keepdims=True)
        return jnp.where(std > 1e-8, centered / (std + 1e-8), 0.0)
    if shaping == 'centered_rank':
        # Ranks in [-0.5, 0.5], magnitude discarded. `rankdata` averages ties,
        # which is what evosax's `centered_rank_fitness_shaping_fn` does and so
        # what `Open_ES` did for every OpenES number in this repo. Ordinal ranks
        # (argsort of argsort) would break ties arbitrarily instead, and ties
        # are not rare here -- a whole population of CartPole episodes hitting
        # the return cap is one.
        ranks = jax.scipy.stats.rankdata(fitness, axis=-1) - 1.0
        return ranks / (fitness.shape[-1] - 1) - 0.5
    raise ValueError(f'unknown shaping {shaping!r}')


def es_grads(eps, utility, sigma, population_size):
    """Search gradients of `E_eps[f(mean + sigma*eps)]`, as ASCENT directions.

    `eps` is (..., population_size, num_params) in units of sigma, `utility` is
    (..., population_size) and already shaped, `sigma` is (..., num_params).
    Returns `(grad_mean, grad_log_sigma)`, both (..., num_params).

    The `1/sigma` in `grad_mean` is the chain rule, not a normalisation: it is
    what makes this an estimate of the gradient rather than of the finite
    difference. Note its consequence for step size -- module docstring, item 3.

    `grad_log_sigma` is the separable-NES update for the search width:
    `E[u * (eps^2 - 1)]`, so a coordinate whose large perturbations scored well
    wants a wider search. It is the part of NES that `Open_ES` has no
    counterpart for at all, and it is a no-op wherever `sigma_lr == 0`.
    """
    grad_mean = jnp.einsum('...p,...pd->...d', utility, eps) / (
        population_size * sigma)
    grad_log_sigma = jnp.einsum('...p,...pd->...d', utility,
                                eps ** 2 - 1.0) / population_size
    return grad_mean, grad_log_sigma


def build_optimizer(optimizer, learning_rate, momentum=0.9):
    """Name -> optax transformation. The step rule half of the NES/OpenES pair.

    `momentum` is Adam's `b1` as well as SGD's, and 0.9 is optax's own default
    for both -- so every caller that does not pass it gets exactly the
    optimizer it got before this argument reached Adam. It is threaded through
    for one reason: Adam differs from SGD in TWO things, per-coordinate
    normalisation and a momentum tail that keeps stepping after the gradient
    has gone to zero, and `momentum=0.0` is the only way to ask which of the
    two a result is about. `scripts/outdated/generalists/analysis/toy_shared_landscape.py
    --describe` does exactly that.
    """
    if optimizer == 'sgd':
        # Plain gradient ascent, no momentum and no per-coordinate scaling.
        return optax.sgd(learning_rate=learning_rate)
    if optimizer == 'sgd_momentum':
        return optax.sgd(learning_rate=learning_rate, momentum=momentum)
    if optimizer == 'adam':
        return optax.adam(learning_rate=learning_rate, b1=momentum)
    raise ValueError(f'unknown optimizer {optimizer!r}')


# ---------------------------------------------------------------------------
# The plain interface
# ---------------------------------------------------------------------------

class ESState(NamedTuple):
    """Everything the search carries between generations.

    `sigma` is per-coordinate so the fixed and adaptive cases have one shape;
    with `sigma_lr == 0` every entry stays at its initial value.
    """
    mean: jnp.ndarray        # (num_params,) the centroid
    log_sigma: jnp.ndarray   # (num_params,) log of the per-coordinate std
    opt_state: optax.OptState
    generation: jnp.ndarray  # scalar int


class ES:
    """A (1, lambda) evolution strategy: NES or OpenES, by `shaping`/`optimizer`.

    Use as `ask` -> evaluate -> `tell`. `ask` returns the population *and* the
    raw perturbations, because `tell` needs them and recovering them by
    subtraction would divide by sigma a second time.

    Sampling is antithetic: perturbations come in +/- pairs, so the estimator is
    exactly unbiased for the linear part of the landscape and the population is
    symmetric about the centroid. `population_size` must therefore be even.

    Fitness is MAXIMISED at this interface.
    """

    def __init__(self, num_params, population_size, sigma_init=0.1,
                 learning_rate=0.05, optimizer='sgd', shaping='zscore',
                 sigma_lr=0.0, momentum=0.9):
        if population_size % 2 != 0:
            raise ValueError(
                f'population_size must be even for antithetic sampling, got '
                f'{population_size}')
        if shaping not in SHAPINGS:
            raise ValueError(f'unknown shaping {shaping!r}')

        self.num_params = int(num_params)
        self.population_size = int(population_size)
        self.half_pop = self.population_size // 2
        self.sigma_init = float(sigma_init)
        self.learning_rate = float(learning_rate)
        self.shaping = shaping
        self.sigma_lr = float(sigma_lr)
        self.optimizer = build_optimizer(optimizer, learning_rate, momentum)

    # -- lifecycle ---------------------------------------------------------

    def init(self, mean):
        mean = jnp.asarray(mean, dtype=jnp.float32)
        log_sigma = jnp.full((self.num_params,), jnp.log(self.sigma_init),
                             dtype=jnp.float32)
        return ESState(
            mean=mean,
            log_sigma=log_sigma,
            opt_state=self.optimizer.init(mean),
            generation=jnp.asarray(0, dtype=jnp.int32),
        )

    def ask(self, key, state):
        """Return `(population, eps)`, both `(population_size, num_params)`.

        `eps` is in units of sigma: `population = mean + sigma * eps`.
        """
        half = random.normal(key, (self.half_pop, self.num_params))
        eps = jnp.concatenate([half, -half], axis=0)
        sigma = jnp.exp(state.log_sigma)
        population = state.mean[None, :] + sigma[None, :] * eps
        return population, eps

    def tell(self, state, eps, fitness):
        """One search-gradient ascent step on `fitness` (higher is better)."""
        utility = shape_fitness(fitness, self.shaping)
        sigma = jnp.exp(state.log_sigma)
        grad_mean, grad_log_sigma = es_grads(eps, utility, sigma,
                                             self.population_size)

        # optax minimises, so hand it the negative of an ascent direction.
        updates, opt_state = self.optimizer.update(-grad_mean, state.opt_state,
                                                   state.mean)
        mean = optax.apply_updates(state.mean, updates)
        log_sigma = state.log_sigma + self.sigma_lr * grad_log_sigma

        return state._replace(mean=mean, log_sigma=log_sigma,
                              opt_state=opt_state,
                              generation=state.generation + 1)


def sigma_scalar(state):
    """The one number to log when sigma is not being adapted per coordinate."""
    return float(jnp.exp(jnp.mean(state.log_sigma)))


# ---------------------------------------------------------------------------
# M centroids out of one budget
# ---------------------------------------------------------------------------

class MultiESState(NamedTuple):
    """M stacked `ESState`s plus the scores selection would need."""
    es: ESState               # every leaf carries a leading (M,) axis
    score: jnp.ndarray        # (M,) mean fitness of each centroid's last sample
    generation: jnp.ndarray   # scalar int
    # Re-seeding needs randomness and `tell` is keyless in the shared searcher
    # interface (`source/studies/generalists/ne.py`), so the stream is carried in state.
    key: jnp.ndarray


def _tree_select(mask, chosen, fallback):
    """`where(mask, chosen, fallback)` over a pytree with a leading M axis."""
    def pick(a, b):
        m = mask.reshape(mask.shape + (1,) * (a.ndim - 1))
        return jnp.where(m, a, b)
    return jax.tree.map(pick, chosen, fallback)


class MultiES:
    """`num_centroids` ES searches sharing one per-generation budget.

    Plain ES is (1, lambda) -- one centroid, one Gaussian cloud, one search
    gradient. On a landscape whose peaks are separated by a valley deeper than
    the cloud is wide, that single centroid can only ever be in one basin, and
    which basin is decided early and by chance. This runs `M` independent ES
    searches at once out of the SAME per-generation evaluation budget -- each
    centroid gets `population_size // M` samples -- so the search covers M
    basins instead of one.

    That is the whole idea, and it is not free. Two costs, both worth stating
    because they are what the experiment measures:

        fewer samples each  A centroid with 8 samples estimates its search
                            gradient from 8 numbers instead of 64. `shape_fitness`
                            runs over that sub-population only, so each centroid
                            is a genuinely separate ES, noise and all.
        covering != finding A portfolio of specialists is not a generalist. If
                            one centroid sits on sub-task A's peak and another on
                            B's, the *set* is good at both and every individual
                            member is good at one. `incumbent` returns a single
                            vector for exactly this reason; `centroids` returns
                            all M so the two can be scored separately and the
                            difference reported rather than hidden.

    Coupling
    --------

    `coupling='none'` is M independent searches sharing a budget -- the honest
    baseline, and already a real change: it is parallel restarts, the oldest
    answer there is to local optima.

    `coupling='select'` adds selection *between* centroids: every
    `restart_interval` generations the worst `restart_frac` of them are re-seeded
    near the best one (fresh optimizer state included, because an Adam moment
    estimate from the abandoned basin is meaningless in the new one). This is
    what makes it a (mu, lambda) rather than a portfolio, and it is the setting
    that can lose the generalist -- a centroid that has found the shared basin
    but is momentarily behind on the current sub-task is exactly the kind of
    centroid selection deletes.

    Centroids are ranked by the **mean fitness of their own sub-population in
    the current generation**. Every centroid is scored on the same sub-task in
    the same generation, so that comparison is fair without any extra
    evaluations. A running average would not be: it would compare a score from
    before a task switch with one from after it.
    """

    def __init__(self, num_params, population_size, num_centroids=8,
                 coupling='none', restart_interval=50, restart_frac=0.25,
                 restart_scale=0.5, init_scale=0.0, **es_kwargs):
        if coupling not in ('none', 'select'):
            raise ValueError(f'unknown coupling {coupling!r}')
        per = population_size // int(num_centroids)
        if per < 2 or per % 2 != 0:
            raise ValueError(
                f'population_size // num_centroids must be even and >= 2, got '
                f'{population_size} // {num_centroids} = {per}')

        self.num_params = int(num_params)
        self.population_size = int(num_centroids) * per
        self.num_centroids = int(num_centroids)
        self.per_centroid = per
        self.coupling = coupling
        self.restart_interval = int(restart_interval)
        self.num_restart = max(1, int(round(self.num_centroids * restart_frac)))
        self.restart_scale = float(restart_scale)
        # How far apart the centroids start. 0.0 puts them all on the same seed
        # point, which makes 'none' a pure test of whether sampling noise alone
        # separates them; > 0 spreads them over the initial region on purpose.
        self.init_scale = float(init_scale)
        self.es = ES(num_params=num_params, population_size=per, **es_kwargs)

    # -- lifecycle ---------------------------------------------------------

    def init(self, key, mean):
        mean = jnp.asarray(mean, dtype=jnp.float32)
        jitter = random.normal(key, (self.num_centroids, self.num_params))
        means = mean[None, :] + self.init_scale * jitter
        return MultiESState(
            es=jax.vmap(self.es.init)(means),
            score=jnp.full((self.num_centroids,), -jnp.inf),
            generation=jnp.asarray(0, dtype=jnp.int32),
            key=key)

    def ask(self, key, state):
        """`(population, eps)` flattened to `(population_size, num_params)`.

        Members are laid out centroid-major, so `tell` recovers which centroid
        scored what by reshaping. The caller evaluates one flat batch and never
        has to know the search has more than one centre.
        """
        keys = random.split(key, self.num_centroids)
        population, eps = jax.vmap(self.es.ask)(keys, state.es)
        return (population.reshape(self.population_size, self.num_params),
                eps.reshape(self.population_size, self.num_params))

    def tell(self, state, eps, fitness):
        """One ES step per centroid, then (under 'select') centroid selection."""
        shape = (self.num_centroids, self.per_centroid)
        eps = eps.reshape(shape + (self.num_params,))
        fitness = fitness.reshape(shape)

        es = jax.vmap(self.es.tell)(state.es, eps, fitness)
        score = jnp.mean(fitness, axis=1)
        state = state._replace(es=es, score=score,
                               generation=state.generation + 1)
        if self.coupling == 'none':
            return state
        return self._select(state)

    def _select(self, state):
        """Re-seed the worst centroids near the best, on the restart schedule."""
        key, restart_key = random.split(state.key)
        due = ((state.generation % self.restart_interval == 0)
               & (state.generation > 0))

        # rank 0 = best. `is_worst` is the tail of that order.
        rank = jnp.argsort(jnp.argsort(-state.score))
        is_worst = due & (rank >= self.num_centroids - self.num_restart)

        best = state.es.mean[jnp.argmax(state.score)]
        jitter = random.normal(restart_key,
                               (self.num_centroids, self.num_params))
        fresh = jax.vmap(self.es.init)(best[None, :] + self.restart_scale * jitter)

        # The whole ES state is replaced, not just the mean: a re-seeded
        # centroid must not inherit the optimizer moments of the basin it left.
        return state._replace(
            es=_tree_select(is_worst, fresh, state.es),
            # -inf would make a just-restarted centroid the next one deleted
            # before it has been scored, so give it the score it inherits.
            score=jnp.where(is_worst, jnp.max(state.score), state.score),
            key=key)

    # -- what the run hands you --------------------------------------------

    def incumbent(self, state):
        """The single best-scoring centroid -- one vector, as every method gives."""
        return state.es.mean[jnp.argmax(state.score)]

    def centroids(self, state):
        return state.es.mean


# ---------------------------------------------------------------------------
# The evosax interface over the same core
# ---------------------------------------------------------------------------

@struct.dataclass
class EvosaxESState(BaseState):
    mean: jax.Array
    log_std: jax.Array          # (num_dims,) -- `log_sigma` under evosax names
    opt_state: optax.OptState


@struct.dataclass
class EvosaxESParams(BaseParams):
    pass


class EvosaxES(DistributionBasedAlgorithm):
    """`ES` behind the evosax v2 interface. Holds no arithmetic of its own.

    Written against the same base class as `Open_ES` so that a trainer swaps one
    for the other by changing the constructor call and nothing else: same
    `init`/`ask`/`tell` signatures, same `state.mean`, same
    `state.best_solution`, same metrics. That is what the gymnax, mujoco, brax
    and kinetix trainers drive, and they also shard this state across devices,
    which is why the evosax shape is not negotiable for them.

    With `shaping='centered_rank', optimizer='adam'` this reproduces `Open_ES`
    bit for bit -- `scripts/check_es.py` asserts `max|dmean| == 0`, and that is
    what licenses `--algo openes` pointing here instead of at evosax.

    ## Why shaping is done in `_tell` rather than by the base class

    evosax's base `tell` applies `fitness_shaping_fn` and then calls `_tell`.
    Routing shaping through there would mean a second implementation of it --
    evosax's callables instead of `shape_fitness` -- so the base is given
    `identity_fitness_shaping_fn` and `_tell` shapes with the core instead. This
    changes nothing observable: the base computes `best_solution`,
    `best_fitness` and the metrics from the RAW fitness *before* shaping, so
    those are untouched, and `_tell` is the only consumer of the shaped values.

    ## std

    `std` is stored per coordinate, as `log_std`, rather than as the scalar
    `Open_ES` reads from an `std_schedule`. With `std_lr = 0.0` -- the default,
    and what every run in this repo uses -- every coordinate stays at `std_init`
    for the whole run and the two are numerically identical.

    Fitness is MINIMISED here, as evosax requires and as `ga.py` does. Every
    caller maximises a return and passes `-fitness` to `tell`. The core is the
    other way round, so `_tell` negates once at this boundary; `shape_fitness`
    is antisymmetric, so that costs nothing.
    """

    def __init__(
        self,
        population_size: int,
        solution: Solution,
        std_init: float = 0.1,
        std_lr: float = 0.0,
        use_antithetic_sampling: bool = True,
        optimizer: optax.GradientTransformation | str = 'sgd',
        learning_rate: float = 0.05,
        shaping: str = 'zscore',
        metrics_fn: Callable = metrics_fn,
    ):
        """Initialize the strategy. `optimizer` takes a name or an optax object."""
        if use_antithetic_sampling:
            assert population_size % 2 == 0, (
                'Population size must be even for antithetic sampling.'
            )
        if shaping not in SHAPINGS:
            raise ValueError(f'unknown shaping {shaping!r}')
        super().__init__(population_size, solution,
                         identity_fitness_shaping_fn, metrics_fn)

        self.std_init = std_init
        self.std_lr = std_lr
        self.shaping = shaping
        self.optimizer = (build_optimizer(optimizer, learning_rate)
                          if isinstance(optimizer, str) else optimizer)
        self.use_antithetic_sampling = use_antithetic_sampling

    @property
    def _default_params(self) -> EvosaxESParams:
        return EvosaxESParams()

    def _init(self, key: jax.Array, params: EvosaxESParams) -> EvosaxESState:
        return EvosaxESState(
            mean=jnp.full((self.num_dims,), jnp.nan),
            log_std=jnp.full((self.num_dims,), jnp.log(self.std_init)),
            opt_state=self.optimizer.init(jnp.zeros(self.num_dims)),
            best_solution=jnp.full((self.num_dims,), jnp.nan),
            best_fitness=jnp.inf,
            generation_counter=0,
        )

    def _ask(
        self,
        key: jax.Array,
        state: EvosaxESState,
        params: EvosaxESParams,
    ) -> tuple[Population, EvosaxESState]:
        if self.use_antithetic_sampling:
            z_plus = jax.random.normal(key, (self.population_size // 2, self.num_dims))
            z = jnp.concatenate([z_plus, -z_plus])
        else:
            z = jax.random.normal(key, (self.population_size, self.num_dims))
        population = state.mean + jnp.exp(state.log_std) * z
        return population, state

    def _tell(
        self,
        key: jax.Array,
        population: Population,
        fitness: Fitness,
        state: EvosaxESState,
        params: EvosaxESParams,
    ) -> EvosaxESState:
        # `fitness` arrives RAW and MINIMISED (see the class docstring); the core
        # maximises. Shape FIRST and negate the utilities, rather than shaping
        # `-fitness`: both are the same in exact arithmetic, but only this order
        # hands `shape_fitness` the identical input evosax hands its own shaping
        # function, so `Open_ES` is reproduced to the bit. Shaping `-fitness`
        # instead costs ~3e-8 under 'centered_rank', where `ranks/(n-1) - 0.5`
        # and `0.5 - ranks/(n-1)` round apart in float32. Negating a float is
        # exact, so this direction loses nothing.
        utility = -shape_fitness(fitness, self.shaping)
        std = jnp.exp(state.log_std)

        # Recovering z by subtraction -- rather than threading it out of `ask` --
        # is exactly what `Open_ES` does, and is exact here because `ask` and
        # `tell` see the same mean.
        z = (population - state.mean) / std
        grad_mean, grad_log_std = es_grads(z, utility, std, self.population_size)

        # optax minimises, so hand it the negative of an ascent direction.
        updates, opt_state = self.optimizer.update(-grad_mean, state.opt_state,
                                                   state.mean)
        mean = optax.apply_updates(state.mean, updates)
        log_std = state.log_std + self.std_lr * grad_log_std

        return state.replace(mean=mean, log_std=log_std, opt_state=opt_state)

    def get_std(self, state: EvosaxESState) -> jax.Array:
        """The per-coordinate search width. Scalar-valued while std_lr == 0."""
        return jnp.exp(state.log_std)
