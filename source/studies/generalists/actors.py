"""The two action heads the study's PPO trainer can put on the evolved policy.

``train_ppo.py`` holds PPO to the same search space as the NE arms: its
actor's deterministic part IS the policy the searchers evolve, so a PPO
checkpoint and an NES centroid are points in one parameter space and are
scored by one scoring function. What PPO adds on top -- a distribution to
sample from and take gradients through -- differs by action space, and this
module is where that difference lives so the trainer holds no branch on it:

    categorical   gymnax. The evolved ``MLPPolicy`` emits logits; the actor IS
                  that network and the head samples its categorical. Every
                  function here is the one the trainer already used.
    gaussian      the mjx bodies. The evolved ``ContinuousMLPPolicy`` emits a
                  tanh-squashed action, which becomes the MEAN of a diagonal
                  Gaussian with one learned state-independent log-std per
                  actuator. ``GaussianMLPActor`` is that network: the same
                  ``Dense_0 .. Dense_n`` trunk and output as the evolved
                  policy, parameter names and all, plus a ``log_std`` vector.
                  Dropping ``log_std`` from its parameters gives EXACTLY the
                  evolved policy's parameter tree, so the deterministic action
                  ``tanh(mean)`` scored by the NE scoring function is the mean
                  of the distribution PPO trained -- the continuous analogue
                  of scoring the categorical's argmax.

The deterministic evaluation is the ruler both arms are measured with, and it
must not also be a difference in what PPO optimises. The Gaussian's sample is
NOT re-squashed: ``tanh`` bounds the mean to the actuator range and the
environment clamps the sample to it, so the log-probability is the plain
Gaussian one and the deterministic policy is the mean, unchanged. brax's own
PPO uses a tanh-squashed Normal whose deterministic action is
``tanh(loc)``; the difference is only in how exploration noise past the
range is handled, and it is what keeps the mean network identical to the
evolved policy.

## Observation normalisation, folded

The paper's ant PPO normalises observations with running statistics and the
NE arms do not, so a normalising PPO's parameters are not in the NE space as
they stand. But the normalisation is affine and so is the first layer, so
``fold_normalizer`` rewrites ``Dense_0`` to absorb it:

    W' = W / std        b' = b - (mean / std) @ W

and the folded network on raw observations equals the trained network on
normalised ones. Every checkpoint and every evaluation goes through the fold,
so PPO's points on a figure are points in the NE space and its scores come
from the identical scoring function. With no normaliser the fold is the
identity and is not applied at all, so a gymnax run is arithmetically the run
it was before this module existed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from jax import random

from source.algorithms.rl import ppo as ppo_lib
from source.metrics import plasticity

LOG_2PI = float(jnp.log(2.0 * jnp.pi))


@dataclass(frozen=True)
class ActionHead:
    """What PPO needs from a policy's output that the searchers do not.

    ``logits`` is whatever the actor network emits: categorical logits, or the
    Gaussian's ``[mean, log_std]`` concatenated along the last axis. All
    functions act on ONE timestep's output; the trainer vmaps them.
    """
    name: str
    sample: Callable            # (key, logits) -> action
    log_prob: Callable          # (logits, action) -> scalar
    entropy: Callable           # (logits) -> scalar
    churn: Callable             # (logits_before, logits_after) -> scalar, the
                                # cross-method OBSERVER, in [0, 1]
    own_churn: Callable         # the method's own estimator, the one C-CHAIN
                                # regularises: cross-entropy or action-mean MSE
    own_churn_key: str          # the record key it is logged under
    chain_churn: Callable       # (ref_logits, cur_logits) -> per-sample churn,
                                # for the C-CHAIN regulariser
    mean_params: Callable       # actor params -> the evolved policy's params
    policy_activation: str      # hidden activation, for ReDo's criterion


# ---- categorical: the gymnax head, every function the trainer already used --

def _categorical_own_churn(before, after):
    return jnp.mean(plasticity.policy_churn_cross_entropy(before, after))


CATEGORICAL = ActionHead(
    name='categorical',
    sample=ppo_lib.categorical_sample,
    log_prob=ppo_lib.categorical_log_prob,
    entropy=ppo_lib.categorical_entropy,
    churn=plasticity.action_disagreement,
    own_churn=_categorical_own_churn,
    own_churn_key='rl_churn_ce',
    chain_churn=plasticity.policy_churn_cross_entropy,
    mean_params=lambda params: params,
    policy_activation='relu',
)


# ---- gaussian: the mjx head -------------------------------------------------

class GaussianMLPActor(nn.Module):
    """``ContinuousMLPPolicy`` as the mean of a diagonal Gaussian.

    Same trunk, same output squash, same flax parameter names (``Dense_i``) as
    the evolved policy, plus one ``log_std`` vector, state-independent. Emits
    ``[tanh(mean), log_std]`` along the last axis.
    """
    hidden_dims: tuple = (128, 128)
    action_dim: int = 6
    log_std_init: float = -0.5

    @nn.compact
    def __call__(self, x):
        for dim in self.hidden_dims:
            x = nn.Dense(dim)(x)
            x = nn.tanh(x)
        mean = nn.tanh(nn.Dense(self.action_dim)(x))
        log_std = self.param(
            'log_std',
            lambda key, shape: jnp.full(shape, self.log_std_init, jnp.float32),
            (self.action_dim,))
        return jnp.concatenate(
            [mean, jnp.broadcast_to(log_std, mean.shape)], axis=-1)


def _split(logits):
    action_dim = logits.shape[-1] // 2
    return logits[..., :action_dim], logits[..., action_dim:]


def gaussian_sample(key, logits):
    mean, log_std = _split(logits)
    return mean + jnp.exp(log_std) * jax.random.normal(key, mean.shape)


def gaussian_log_prob(logits, action):
    mean, log_std = _split(logits)
    z = (action - mean) * jnp.exp(-log_std)
    return jnp.sum(-0.5 * jnp.square(z) - log_std - 0.5 * LOG_2PI, axis=-1)


def gaussian_entropy(logits):
    _, log_std = _split(logits)
    return jnp.sum(log_std + 0.5 * (LOG_2PI + 1.0), axis=-1)


def gaussian_chain_churn(ref_logits, cur_logits):
    """C-CHAIN's continuous churn, per sample: MSE between action MEANS.

    The reference's DMC variant regularises the mean and leaves the scale
    alone -- regularising it would fight the entropy bonus. Per sample here,
    so it has the shape ``chain_policy_churn`` has; the epoch loop averages.
    """
    ref_mean, _ = _split(ref_logits)
    cur_mean, _ = _split(cur_logits)
    return jnp.mean(jnp.square(cur_mean - ref_mean), axis=-1)


def _gaussian_churn(before, after):
    return plasticity.action_disagreement_continuous(_split(before)[0],
                                                     _split(after)[0])


def _gaussian_own_churn(before, after):
    return plasticity.action_churn_mse(_split(before)[0], _split(after)[0])


def gaussian_mean_params(params):
    """The evolved policy's parameter tree: the actor's minus ``log_std``."""
    inner = {k: v for k, v in params['params'].items() if k != 'log_std'}
    return {**params, 'params': inner}


def gaussian_head():
    return ActionHead(
        name='gaussian',
        sample=gaussian_sample,
        log_prob=gaussian_log_prob,
        entropy=gaussian_entropy,
        churn=_gaussian_churn,
        own_churn=_gaussian_own_churn,
        own_churn_key='rl_churn_mse',
        chain_churn=gaussian_chain_churn,
        mean_params=gaussian_mean_params,
        policy_activation='tanh',
    )


# ---- multi-discrete: the kinetix head ---------------------------------------
#
# Kinetix's action is SIX independent categoricals -- one per motor binding
# (3 choices: reverse / off / forward) and one per thruster binding (2: off /
# on) -- which the network emits as one flat vector of 16 logits, exactly as
# `kinetix.models.action_spaces.MultiDiscreteActionDistribution` splits it.
#
# This is a new ACTION SPACE, not a new PPO. Every function below is the
# categorical one summed or vmapped over the six sub-distributions: the joint
# is a product of independents, so its log-prob is a sum, its entropy is a sum,
# and its greedy action is the per-distribution argmax. Nothing in
# `source/algorithms/rl/ppo.py`, in the rollout or in the update changes.
#
# The dims are a property of the environment's static parameters (the medium
# Kinetix env size: 4 motor bindings, 2 thruster bindings), so the head is
# BUILT from them rather than being a constant -- `head_for` reads them off the
# suite. Padding the ragged [3,3,3,3,2,2] into a (6, 3) block and masking the
# unused column with -inf keeps every operation one vectorised call; the mask
# is what stops a thruster ever sampling the third option.

# The padding value for the ragged block. FINITE, and that is the whole point:
# -inf gives the right entropy VALUE and a NaN entropy GRADIENT, because
# `jnp.where` differentiates both branches and `0 * -inf` is nan. PPO's loss
# carries `-ent_coef * entropy`, so with -inf here the first gradient step
# NaNs the actor, every logit becomes nan, and -- since a masked-out nan reads
# as 0 -- the run then reports a policy entropy of exactly 0.000 and a return
# at the floor, looking for all the world like an entropy collapse. It is not:
# it is this line. Measured and fixed 2026-09-09; `_multi_discrete_grads_finite`
# below is the regression test.
#
# exp(-1e9) is 0 in float32, so a padded choice has exactly zero probability
# and cannot be sampled, which is what -inf was there for.
_MD_PAD = -1e9


def _multi_discrete_split(dims):
    """`(offsets, width, mask)`: how a flat logit vector becomes a (n, w) block."""
    offsets = jnp.asarray(np.cumsum([0] + list(dims[:-1])), dtype=jnp.int32)
    width = int(max(dims))
    mask = jnp.asarray(
        np.arange(width)[None, :] < np.asarray(dims)[:, None], dtype=bool)
    return offsets, width, mask


def _multi_discrete_block(logits, dims):
    """Flat `(..., sum(dims))` logits -> `(..., n, max(dims))` and its mask.

    Padded entries are `_MD_PAD`, not -inf; see the note there.
    """
    offsets, width, mask = _multi_discrete_split(dims)
    cols = jnp.arange(width)
    # (n, w) gather indices into the flat vector; the padded entries read some
    # in-range logit and are then masked out, so the read is safe.
    idx = jnp.clip(offsets[:, None] + cols[None, :],
                   0, int(sum(dims)) - 1)
    block = jnp.take(logits, idx, axis=-1)
    return jnp.where(mask, block, _MD_PAD), mask


def multi_discrete_head(dims):
    """The `ActionHead` for a multi-discrete space of `dims` per dimension."""
    dims = tuple(int(d) for d in dims)

    def sample(key, logits):
        block, _ = _multi_discrete_block(logits, dims)
        keys = random.split(key, len(dims))
        return jax.vmap(lambda k, row: random.categorical(k, row))(
            keys, block)

    def log_prob(logits, action):
        block, _ = _multi_discrete_block(logits, dims)
        lp = jax.nn.log_softmax(block, axis=-1)
        return jnp.sum(jnp.take_along_axis(
            lp, action.astype(jnp.int32)[..., None], axis=-1).squeeze(-1),
            axis=-1)

    def entropy(logits):
        block, mask = _multi_discrete_block(logits, dims)
        lp = jax.nn.log_softmax(block, axis=-1)
        p = jnp.exp(lp)
        # Masked, not `isfinite`-guarded. An `isfinite` test would ALSO read a
        # NaN logit as a zero contribution, which is how the -inf padding hid
        # its own NaN gradient behind a plausible-looking entropy of 0.000.
        # This way a NaN policy reports NaN and is seen.
        return jnp.sum(jnp.where(mask, -p * lp, 0.0), axis=(-1, -2))

    def greedy(logits):
        return jnp.argmax(_multi_discrete_block(logits, dims)[0], axis=-1)

    def churn(before, after):
        """Fraction of (state, action-dimension) pairs whose greedy choice moved.

        The observer, on the same [0, 1] scale as `action_disagreement`: a
        joint-argmax test over 324 combinations would read ~1.0 whenever any
        one of six dimensions moved, and measure nothing.
        """
        return jnp.mean(greedy(before) != greedy(after)).astype(jnp.float32)

    def chain_churn(ref_logits, cur_logits):
        """Per-sample cross-entropy H(pi_ref, pi_cur), summed over dimensions.

        C-CHAIN's discrete estimator (`plasticity.policy_churn_cross_entropy`)
        applied to each sub-distribution and added, which is the cross-entropy
        of the joint because the joint is a product of independents.
        """
        ref, mask = _multi_discrete_block(
            jax.lax.stop_gradient(ref_logits), dims)
        cur, _ = _multi_discrete_block(cur_logits, dims)
        p = jax.nn.softmax(ref, axis=-1)
        lq = jax.nn.log_softmax(cur, axis=-1)
        return jnp.sum(jnp.where(mask, -p * lq, 0.0), axis=(-1, -2))

    return ActionHead(
        name='multi_discrete',
        sample=sample,
        log_prob=log_prob,
        entropy=entropy,
        churn=churn,
        own_churn=lambda before, after: jnp.mean(chain_churn(before, after)),
        own_churn_key='rl_churn_ce',
        chain_churn=chain_churn,
        mean_params=lambda params: params,
        policy_activation='tanh',
    )


def _multi_discrete_grads_finite(dims=(3, 3, 3, 3, 2, 2), seed=0):
    """Regression test for the -inf padding bug. Returns True, or raises.

    THE BUG THIS EXISTS FOR (2026-09-09). The ragged logit block was padded
    with -inf. Every VALUE it produced was correct -- the entropy of a fresh
    Kinetix policy read 5.7807, the multi-discrete maximum, to four decimals --
    and its GRADIENT was NaN in the columns belonging to the 2-choice
    distributions, because `jnp.where` differentiates the branch it did not
    take and `0 * -inf` is nan.

    PPO's loss carries `-ent_coef * entropy`, so the first gradient step NaN'd
    the whole actor. The symptom did not look like a NaN: the `isfinite` guard
    inside the old entropy read a NaN logit as a zero contribution, so the run
    reported `H=0.000` with the return at the floor, which reads as an entropy
    collapse. It survived a bisect over batch shape, a 10x learning rate and a
    1000x Adam epsilon -- all reporting `H=0.023` to three decimals, which is
    what finally gave it away, because no training effect is invariant to the
    learning rate.

    Run it from `scripts/check_imports.py` or by hand; it needs no GPU.
    """
    head = multi_discrete_head(dims)
    n = int(sum(dims))
    logits = random.normal(random.key(seed), (n,))
    action = jnp.zeros((len(dims),), jnp.int32)
    checks = {
        'entropy': jax.grad(head.entropy)(logits),
        'log_prob': jax.grad(lambda l: head.log_prob(l, action))(logits),
        'chain_churn': jax.grad(
            lambda l: jnp.sum(head.chain_churn(logits, l)))(logits),
    }
    for name, g in checks.items():
        if not bool(jnp.all(jnp.isfinite(g))):
            raise AssertionError(
                f'multi-discrete {name} gradient is not finite: {g}')
    # And a NaN policy must REPORT NaN rather than a plausible 0.0, or the next
    # occurrence of this hides the same way.
    if not bool(jnp.isnan(head.entropy(jnp.full((n,), jnp.nan)))):
        raise AssertionError(
            'entropy of a NaN policy reads as a number; a NaN actor would be '
            'reported as an entropy collapse again')
    return True


# Suites whose policy emits categorical logits: PPO's actor IS the evolved
# policy there. The mjx bodies are the continuous ones.
CATEGORICAL_SUITES = ('gymnax', 'minigrid')

# Suites whose policy emits logits but whose action is a VECTOR of independent
# categoricals. Kinetix is the only one; the head is built from the
# environment's own dims-per-distribution, so `head_for` needs the suite.
MULTI_DISCRETE_SUITES = ('kinetix',)

# Suites where PPO's actor IS the evolved policy object -- the logits suites.
# On the mjx bodies the actor is that policy as a Gaussian mean instead.
LOGITS_SUITES = CATEGORICAL_SUITES + MULTI_DISCRETE_SUITES


def head_for(suite_name, action_dims=None):
    """The head a suite's policy takes.

    Categorical on gymnax and MiniGrid, multi-discrete on kinetix, Gaussian on
    mjx. `action_dims` is the per-dimension choice count and is required by --
    and only by -- the multi-discrete head; the suite supplies it
    (`source/envs/kinetix.py:action_dims`), because it is a property of the
    environment's static parameters rather than a constant.
    """
    if suite_name in CATEGORICAL_SUITES:
        return CATEGORICAL
    if suite_name in MULTI_DISCRETE_SUITES:
        if action_dims is None:
            raise ValueError(
                f'suite {suite_name!r} is multi-discrete; head_for needs its '
                'action_dims')
        return multi_discrete_head(action_dims)
    return gaussian_head()


def build_actor(suite_name, obs_dim, action_dim, hidden_dims, log_std_init,
                policy):
    """The network PPO trains. On a logits suite it IS the evolved
    policy object."""
    if suite_name in LOGITS_SUITES:
        return policy
    return GaussianMLPActor(hidden_dims=tuple(hidden_dims),
                            action_dim=action_dim, log_std_init=log_std_init)


# ---- observation normalisation, and the fold that keeps PPO in the NE space --

def init_norm_stats(obs_dim):
    """Running mean / variance over observations, Welford's, as a pytree."""
    return {'count': jnp.zeros((), jnp.float32),
            'mean': jnp.zeros((obs_dim,), jnp.float32),
            'var': jnp.ones((obs_dim,), jnp.float32)}


def update_norm_stats(stats, obs):
    """Fold a batch of observations (any leading shape) into the statistics."""
    flat = obs.reshape((-1, obs.shape[-1]))
    n = jnp.asarray(flat.shape[0], jnp.float32)
    batch_mean = flat.mean(axis=0)
    batch_var = flat.var(axis=0)
    total = stats['count'] + n
    delta = batch_mean - stats['mean']
    mean = stats['mean'] + delta * n / total
    m_a = stats['var'] * stats['count']
    m_b = batch_var * n
    var = (m_a + m_b + jnp.square(delta) * stats['count'] * n / total) / total
    return {'count': total, 'mean': mean, 'var': var}


NORM_EPS = 1e-6


def norm_std(stats):
    return jnp.sqrt(stats['var'] + NORM_EPS)


def normalize(obs, stats):
    return (obs - stats['mean']) / norm_std(stats)


def fold_normalizer(params, stats, first_layer='Dense_0'):
    """Absorb ``normalize`` into the first Dense layer. See the module docstring.

    Exact to rounding: ``(x - m) / s @ W + b == x @ (W / s[:, None]) + (b - (m / s) @ W)``.
    """
    p = dict(params['params'])
    layer = dict(p[first_layer])
    std = norm_std(stats)
    kernel = layer['kernel'] / std[:, None]
    bias = layer['bias'] - (stats['mean'] / std) @ layer['kernel']
    p[first_layer] = {**layer, 'kernel': kernel, 'bias': bias}
    return {**params, 'params': p}
