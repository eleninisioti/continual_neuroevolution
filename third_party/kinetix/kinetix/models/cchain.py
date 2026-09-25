"""C-CHAIN: Continual Churn Approximated Reduction (Tang et al., ICML 2025)
for the Kinetix actor-critic.

Port of `source/studies/gymnax/cchain.py` -- itself a port of
`inspiration/C-CHAIN/crl_gym_classic_control/` -- onto this repo's Kinetix PPO,
so the Kinetix `cchain` column means the same thing as the gymnax and mujoco
ones.

The method adds one term to the PPO objective: the policy is pulled towards the
policy from one gradient step ago, evaluated on an *independently drawn*
minibatch of the same rollout. That suppresses the off-diagonal entries of the
empirical NTK, which the paper identifies as the mechanism behind plasticity
loss. The coefficient is not a fixed hyperparameter -- it is driven so the churn
term keeps a target relative scale against the policy loss.

Two things differ from the gymnax port, both forced by the Kinetix trainer:

1. **The coefficient controller is JAX state, not a Python object.** In gymnax
   the PPO update loop is Python, so `ChainCoefController` can keep growing
   lists and be read between updates. Kinetix runs the whole update loop inside
   `jax.lax.scan`, so the running means are kept in a fixed-size ring buffer
   threaded through the carry (`ChainCoefState`). The arithmetic is the same:
   `coef = max(target * mean|p_loss| / mean(p_reg_loss), 1)` over the last
   `window` updates, held at 1 until `warmup` updates have passed.

2. **The churn is summed over the factored action distribution.** Kinetix's
   default action space is MULTI_DISCRETE, represented as a list of independent
   `distrax.Categorical` sub-distributions. The reference's cross-entropy is
   defined for one categorical, so it is applied per factor and summed -- which
   is exactly how `MultiDiscreteActionDistribution` already defines `log_prob`
   and `entropy`, so the churn stays on the same scale as the policy loss the
   controller compares it against.

The reset-at-task-switch that `ChainCoefController.reset()` performs in gymnax
is implicit here: the Kinetix continual runner starts a fresh process per level,
so the counter and the buffers begin empty on every sub-task, and the
regularizer is inactive for the first two gradient steps of each one.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp


# Reference defaults (Tang et al. and the reference implementation), identical
# to the ones source/studies/gymnax/cchain.py uses.
DEFAULT_TARGET_REL_SCALE = 10000.0
DEFAULT_WARMUP_UPDATES = 10
DEFAULT_COEF_WINDOW = 50


def _logit_list(pi):
    """Per-factor logits of a Kinetix policy distribution.

    MULTI_DISCRETE holds a list of `distrax.Categorical`; DISCRETE is a single
    one. Continuous action types have no logits and are rejected rather than
    silently given a meaningless churn.
    """
    if hasattr(pi, "distributions"):
        return [d.logits for d in pi.distributions]
    if hasattr(pi, "logits"):
        return [pi.logits]
    raise TypeError(
        f"C-CHAIN needs a categorical policy to compute churn, got {type(pi).__name__}. "
        "The churn term is a cross-entropy over action logits and is not defined "
        "for continuous action distributions."
    )


def chain_policy_churn(ref_pi, cur_pi):
    """Cross-entropy H(pi_ref, pi_cur) per sample; gradients flow only through cur.

    The reference uses the cross-entropy rather than the KL. The two differ by
    the (constant) entropy of pi_ref, so the gradient is identical; the reported
    value is kept on the paper's scale because the coefficient controller is
    calibrated against it.
    """
    total = 0.0
    for ref_logits, cur_logits in zip(_logit_list(ref_pi), _logit_list(cur_pi)):
        ref_probs = jax.nn.softmax(jax.lax.stop_gradient(ref_logits))
        cur_log_probs = jax.nn.log_softmax(cur_logits)
        total = total + -jnp.sum(ref_probs * cur_log_probs, axis=-1)
    return total


def value_churn(ref_values, cur_values):
    """Mean squared change in predicted value. Logged, never penalised --
    the reference regularises the policy only."""
    return jnp.mean(jnp.square(cur_values - ref_values))


class ChainCoefState(NamedTuple):
    """Ring buffers backing the coefficient controller, plus the live coef."""
    p_loss_buf: jnp.ndarray      # (window,) |policy loss| history
    p_reg_buf: jnp.ndarray       # (window,) churn loss history
    n: jnp.ndarray               # updates seen in this task
    coef: jnp.ndarray            # current churn coefficient


def init_coef_state(window=DEFAULT_COEF_WINDOW,
                    target_rel_scale=DEFAULT_TARGET_REL_SCALE):
    return ChainCoefState(
        p_loss_buf=jnp.zeros((window,), dtype=jnp.float32),
        p_reg_buf=jnp.zeros((window,), dtype=jnp.float32),
        n=jnp.array(0, dtype=jnp.int32),
        coef=jnp.array(1.0 if target_rel_scale > 0 else 0.0, dtype=jnp.float32),
    )


def update_coef_state(state, p_loss, p_reg_loss,
                      target_rel_scale=DEFAULT_TARGET_REL_SCALE,
                      warmup_updates=DEFAULT_WARMUP_UPDATES):
    """One controller step. Equivalent to gymnax's ChainCoefController.update.

    Before the buffer has filled, the mean is taken over the entries actually
    written -- matching `hist[-window:]` on a list shorter than the window.
    """
    window = state.p_loss_buf.shape[0]
    idx = state.n % window
    p_buf = state.p_loss_buf.at[idx].set(jnp.abs(p_loss).astype(jnp.float32))
    r_buf = state.p_reg_buf.at[idx].set(jnp.asarray(p_reg_loss, jnp.float32))
    n = state.n + 1

    valid = jnp.minimum(n, window)
    mask = jnp.arange(window) < valid
    denom = jnp.maximum(valid, 1).astype(jnp.float32)
    mean_p = jnp.sum(jnp.where(mask, p_buf, 0.0)) / denom
    mean_r = jnp.sum(jnp.where(mask, r_buf, 0.0)) / denom

    tuned = jnp.maximum(target_rel_scale * mean_p / (mean_r + 1e-8), 1.0)
    coef = jnp.where(
        jnp.logical_and(target_rel_scale > 0, n >= warmup_updates),
        tuned, state.coef,
    )
    return ChainCoefState(p_loss_buf=p_buf, p_reg_buf=r_buf, n=n,
                          coef=coef.astype(jnp.float32))
