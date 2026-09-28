"""The distribution-based NE arm: one evolution strategy, `ES`.

One centroid, a Gaussian search distribution around it, and a step along the
*search gradient*: sample perturbations, score them, move the centroid along the
fitness-weighted average perturbation. Two settings choose the variant, and
they are the only things an ES run is configured by beyond sigma and the
learning rate:

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

    zscore        + sgd     the textbook (1, lambda) search gradient
    centered_rank + adam    OpenAI-ES (Salimans et al., 2017), as in evosax
                            `Open_ES`

Both settings are hypotheses about plasticity, so they are arguments to one
class rather than two classes: two ES runs that differ in them differ in
exactly those choices and nothing else.

## What is in here

    shape_fitness, es_grads     the arithmetic. ~15 lines, and the ONLY copy.
    build_optimizer             optimizer name -> optax transformation.
    ES                          init(mean) / ask -> (pop, eps) /
                                tell(state, eps, fitness), fitness MAXIMISED.
    ESSearcher                  `ES` behind the searcher interface shared
                                with GA and DNS (`searchers.py`).

## sigma, and the three things it does at once

`sigma` is the standard deviation of the search distribution: `ask` returns
`mean + sigma * eps` with `eps ~ N(0, I)`. It is stored per coordinate (as
`log_sigma`) so the fixed and adapted cases have one shape, and stays at
`sigma_init` everywhere unless `sigma_lr > 0` turns on the separable-NES update
for it. Off by default.

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
because every caller in this repo maximises a return.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import optax
from jax import random

SHAPINGS = ('zscore', 'centered_rank', 'raw')
OPTIMIZERS = ('sgd', 'sgd_momentum', 'adam')


# ---------------------------------------------------------------------------
# The core. Everything else in this file is an interface over these two.
# ---------------------------------------------------------------------------

def shape_fitness(fitness, shaping):
    """Raw fitness (MAXIMISED) -> utilities (MAXIMISED), along the last axis.

    With a leading axis this shapes each row independently.
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
        # what `Open_ES` does. Ordinal ranks
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
    wants a wider search. It is a no-op wherever `sigma_lr == 0`.
    """
    grad_mean = jnp.einsum('...p,...pd->...d', utility, eps) / (
        population_size * sigma)
    grad_log_sigma = jnp.einsum('...p,...pd->...d', utility,
                                eps ** 2 - 1.0) / population_size
    return grad_mean, grad_log_sigma


def build_optimizer(optimizer, learning_rate, momentum=0.9):
    """Name -> optax transformation: the step rule, one of `OPTIMIZERS`.

    `momentum` is Adam's `b1` as well as SGD's, and 0.9 is optax's own default
    for both -- so every caller that does not pass it gets exactly the
    optimizer it got before this argument reached Adam. It is threaded through
    for one reason: Adam differs from SGD in TWO things, per-coordinate
    normalisation and a momentum tail that keeps stepping after the gradient
    has gone to zero, and `momentum=0.0` is the only way to ask which of the
    two a result is about.
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
    """A (1, lambda) evolution strategy; the variant is `shaping` + `optimizer`.

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
        if optimizer not in OPTIMIZERS:
            raise ValueError(f'unknown optimizer {optimizer!r}')

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


class ESSearcher:
    """`ES` behind the searcher interface (`source/algorithms/ne/searchers.py`).
    ``method='es'``; the variant is ``shaping`` + ``optimizer``.
    """

    needs_descriptors = False

    def __init__(self, num_params, population_size, sigma_init=0.1,
                 learning_rate=0.05, optimizer='sgd', shaping='zscore',
                 sigma_lr=0.0, momentum=0.9):
        self.es = ES(num_params=num_params, population_size=population_size,
                     sigma_init=sigma_init, learning_rate=learning_rate,
                     optimizer=optimizer, shaping=shaping, sigma_lr=sigma_lr,
                     momentum=momentum)
        self.population_size = population_size

    def init(self, key, mean):
        return self.es.init(mean)

    def ask(self, key, state):
        return self.es.ask(key, state)

    def tell(self, state, aux, fitness, descriptors=None):
        return self.es.tell(state, aux, fitness)

    def incumbent(self, state):
        """The distribution mean. No population member represents the search
        better than its centre."""
        return state.mean

    def population_mean(self, state):
        """Identical to ``incumbent``: for a distribution-based search the two
        ARE the same point. Here so the runner needs no branch."""
        return state.mean

    # A (1, lambda) strategy has no persistent population: each generation is
    # sampled fresh around the one centroid and discarded, so there is no set
    # of genomes that could hold one specialist per sub-task. If ES scores well
    # on two sub-tasks it is because ONE genome is good at both.
    has_population = False

    def population(self, state):
        return state.mean[None, :]
