"""
C-CHAIN: Continual Churn Approximated Reduction (Tang et al., ICML 2025), for
the brax/MJX PPO in this package.

This is the continuous-control counterpart of `source/studies/gymnax/cchain.py`. The
method is the same -- one extra term pulls the policy towards the policy from
one gradient step ago, evaluated on an *independently shuffled* minibatch of
the same rollout, which suppresses the off-diagonal entries of the empirical
NTK that the paper identifies as the mechanism behind plasticity loss -- but
two things differ, both because the action space is continuous:

  * **The churn is an MSE between action means, not a cross-entropy.**
    CheetahRun's policy is a `NormalTanhDistribution`, so `logits` is
    `[loc, scale]` rather than a categorical logit vector. The reference
    implementation's DMC variant (`crl_dmc`) regularises the mean action, which
    is what `churn_loss` computes here. A cross-entropy over `logits` would be
    meaningless -- half of that vector is a standard deviation.
  * **The two live on completely different scales**, so the coefficient
    controller is calibrated differently: `target_rel_scale` is 0.05 here
    against 10000 for the gymnax categorical version. Passing the gymnax value
    would swamp the PPO objective.

Unlike the gymnax version's `ChainCoefController`, the controller state here is
a JAX pytree: brax's `training_step` runs under `jax.jit`, so the running
windows have to be traced arrays rather than Python lists.

`reset_for_new_task` is the continual part of C-CHAIN -- the coefficient and
its history are cleared at every task switch, so the regulariser recalibrates
against the new task's loss scale instead of the previous task's.
"""

from typing import Any

import jax
import jax.numpy as jnp
from flax import struct


@struct.dataclass
class ChainState:
  """Reference policy (one gradient step behind) plus the coefficient controller.

  Attributes:
    ref_policy_params: policy params as of the previous gradient step. The
      churn term is measured against these.
    coef: current churn coefficient, applied multiplicatively to the churn loss.
    grad_steps: gradient steps taken. The regulariser stays off until this
      reaches 2, because before then `ref_policy_params` is not genuinely one
      step behind.
    iterations: controller ticks *within the current task*. Reset by
      `reset_for_new_task`; compared against `warmup_iterations`.
    p_loss_hist: rolling window of recent policy losses (most recent last).
    p_reg_loss_hist: rolling window of recent churn losses (most recent last).
  """

  ref_policy_params: Any
  coef: jnp.ndarray
  grad_steps: jnp.ndarray
  iterations: jnp.ndarray
  p_loss_hist: jnp.ndarray
  p_reg_loss_hist: jnp.ndarray


# Coefficient before the controller takes over, from the reference's
# `cur_reg_coef = 100.0` (crl_dmc/crl_run_ppo_c_chain_dmc.py). It is *not* 1.0:
# the churn term for a continuous policy is an MSE between action means, which
# is orders of magnitude smaller than the policy loss, so a coefficient of 1
# would leave the regulariser doing nothing for the whole warmup.
INITIAL_COEF = 100.0


def init_chain_state(policy_params, window: int = 100) -> ChainState:
  """Initial state. `window` fixes the controller's averaging length for the run."""
  window = max(1, int(window))
  return ChainState(
      ref_policy_params=policy_params,
      coef=jnp.array(INITIAL_COEF, dtype=jnp.float32),
      grad_steps=jnp.array(0, dtype=jnp.int32),
      iterations=jnp.array(0, dtype=jnp.int32),
      p_loss_hist=jnp.zeros(window, dtype=jnp.float32),
      p_reg_loss_hist=jnp.zeros(window, dtype=jnp.float32),
  )


def reset_for_new_task(chain_state: ChainState) -> ChainState:
  """Clear the coefficient and its history at a task switch.

  `ref_policy_params` and `grad_steps` are deliberately *not* reset: the policy
  itself is continuous across the switch, so the previous step's params are
  still the correct reference and there is no reason to re-serve the two-step
  warmup. What must go is the loss history, which is on the old task's scale.
  """
  return chain_state.replace(
      coef=jnp.array(INITIAL_COEF, dtype=jnp.float32),
      iterations=jnp.array(0, dtype=jnp.int32),
      p_loss_hist=jnp.zeros_like(chain_state.p_loss_hist),
      p_reg_loss_hist=jnp.zeros_like(chain_state.p_reg_loss_hist),
  )


def churn_loss(ppo_network, normalizer_params, policy_params, ref_policy_params,
               obs) -> jnp.ndarray:
  """Mean squared difference between current and reference *action means*.

  Gradients flow only through `policy_params`; the reference is stopped, which
  is what makes this a pull towards the old policy rather than a mutual
  averaging of the two.

  `logits` from a `NormalTanhDistribution` policy is `[loc, scale]` along the
  last axis, so only the first half -- the pre-tanh mean action -- is compared.
  The scale is left alone on purpose: regularising it would fight the entropy
  bonus, which the reference implementation does not do.
  """
  apply = ppo_network.policy_network.apply
  cur_logits = apply(normalizer_params, policy_params, obs)
  ref_logits = jax.lax.stop_gradient(
      apply(normalizer_params, ref_policy_params, obs))

  # param_size == 2 * action_size for NormalTanh; fall back to the whole vector
  # for distributions whose params are not a (loc, scale) pair.
  action_size = cur_logits.shape[-1] // 2
  if 2 * action_size == cur_logits.shape[-1]:
    cur_logits = cur_logits[..., :action_size]
    ref_logits = ref_logits[..., :action_size]

  return jnp.mean(jnp.square(cur_logits - ref_logits))


def update_coefficient(chain_state: ChainState, policy_loss, reg_loss,
                       target_rel_scale: float, warmup_iterations: int,
                       coef_window: int) -> ChainState:
  """One controller tick, called once per PPO iteration.

  Drives the coefficient so the churn term sits at `target_rel_scale` times the
  policy loss, averaged over the last `coef_window` iterations:

      coef = target_rel_scale * mean|p_loss| / mean(p_reg_loss)

  This is the reference's *continuous-control* rule
  (crl_dmc/crl_run_ppo_c_chain_dmc.py), which differs from the categorical one
  in `source/studies/gymnax/cchain.py` in one important way: there is **no floor at 1**.
  With target_rel_scale=0.05 and an action-mean MSE for the churn term, the
  ratio here is routinely below 1, so clamping it would pin the coefficient at
  its floor forever and turn the auto-tuning off entirely. The floor only makes
  sense alongside the categorical variant's target_rel_scale of 10000.

  The reference also scales by the learning-rate annealing fraction `frac`.
  This trainer does not anneal the learning rate, so that factor is identically
  1 and is omitted.

  `target_rel_scale <= 0` disables the regulariser (coefficient pinned to 0),
  which is how a "C-CHAIN with no churn term" ablation is expressed.

  The window is only averaged over entries that have actually been written
  since the last reset -- otherwise the zeros a fresh buffer is padded with
  would drag `mean(p_reg_loss)` down and blow the coefficient up for the first
  `coef_window` iterations of every task.
  """
  window = chain_state.p_loss_hist.shape[0]

  def push(hist, value):
    return jnp.concatenate([hist[1:], jnp.reshape(value, (1,)).astype(hist.dtype)])

  p_loss_hist = push(chain_state.p_loss_hist, policy_loss)
  p_reg_loss_hist = push(chain_state.p_reg_loss_hist, reg_loss)
  iterations = chain_state.iterations + 1

  # Entries written since the last reset live at the END of the buffer.
  effective_window = min(window, max(1, int(coef_window)))
  valid = jnp.minimum(iterations, effective_window)
  positions = jnp.arange(window)
  mask = (positions >= (window - valid)).astype(jnp.float32)
  denom = jnp.maximum(jnp.sum(mask), 1.0)

  mean_p_loss = jnp.sum(jnp.abs(p_loss_hist) * mask) / denom
  mean_p_reg_loss = jnp.sum(p_reg_loss_hist * mask) / denom

  if target_rel_scale > 0:
    tuned = target_rel_scale * mean_p_loss / (mean_p_reg_loss + 1e-8)
    # Held at INITIAL_COEF until the task has had `warmup_iterations` ticks, so
    # the coefficient is not set from a handful of noisy early updates.
    coef = jnp.where(iterations >= warmup_iterations, tuned, chain_state.coef)
  else:
    coef = jnp.array(0.0, dtype=jnp.float32)

  return chain_state.replace(
      coef=coef.astype(jnp.float32),
      iterations=iterations,
      p_loss_hist=p_loss_hist,
      p_reg_loss_hist=p_reg_loss_hist,
  )


def add_chain_args(parser):
  """CLI flags for the mujoco trainers.

  The defaults are the reference implementation's *continuous-control* values,
  not the gymnax ones; see the module docstring for why they differ by five
  orders of magnitude.
  """
  parser.add_argument('--use_cchain', action='store_true', default=False,
                      help='Enable the C-CHAIN churn regulariser')
  parser.add_argument('--chain_target_rel_scale', type=float, default=0.05,
                      help='Target scale of the churn term relative to the policy '
                           'loss (0 disables the regulariser)')
  parser.add_argument('--chain_warmup_iterations', type=int, default=50,
                      help='PPO iterations into a task before the coefficient is tuned')
  parser.add_argument('--chain_coef_window', type=int, default=100,
                      help='Recent iterations averaged by the coefficient controller')
  parser.add_argument('--cchain_reset_on_switch', type=int, default=0,
                      help='Re-calibrate the coefficient controller at each '
                           'sub-task boundary (see reset_for_new_task). Off by '
                           'default: every other method in the comparison is '
                           'never told a switch happened, and this would hand '
                           'C-CHAIN alone that signal')
  return parser
