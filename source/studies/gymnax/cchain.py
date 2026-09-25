"""
C-CHAIN: Continual Churn Approximated Reduction (Tang et al., ICML 2025).

Port of inspiration/C-CHAIN/crl_gym_classic_control/control_train_aligned_c_chain.py
to the JAX/gymnax PPO used in this repo. Shared by train_RL_gymnax.py (single
task) and train_RL_gymnax_continual.py (task sequence).

The method adds one term to the PPO objective: the policy is pulled towards the
policy from one gradient step ago, evaluated on an *independently drawn*
minibatch of the same rollout (the "reference batch", disjoint in sampling from
the training batch). This suppresses the off-diagonal entries of the empirical
NTK matrix, which is what the paper identifies as the mechanism behind
plasticity loss.

The coefficient is not a fixed hyperparameter: it is driven so that the churn
term stays at a target relative scale w.r.t. the policy loss, and it is reset
whenever the task changes.
"""

import jax
import jax.numpy as jnp
from jax import random

from source.metrics import plasticity


def chain_policy_churn(ref_logits, cur_logits):
    """Cross-entropy H(pi_ref, pi_cur) per sample; gradients flow only through cur.

    ONE DEFINITION, in source/metrics/plasticity.py. It moved there when
    the churn column stopped being C-CHAIN-only: every gymnax RL method reports
    it now, and the stationary trainer imports this module behind a try/except,
    so a measure all four methods need cannot live behind that guard. Aliased
    rather than renamed so the regulariser's call sites below are untouched.
    """
    return plasticity.policy_churn_cross_entropy(ref_logits, cur_logits)


def init_chain_state(policy_state, value_state):
    """Reference networks (one gradient step behind) + gradient-step counter."""
    return (policy_state.params, value_state.params, jnp.array(0, dtype=jnp.int32))


class ChainCoefController:
    """Auto-tunes the churn coefficient from the relative loss scales.

    coef = max(target_rel_scale * mean|p_loss| / mean(p_reg_loss), 1), averaged
    over the last `window` updates and only applied once `warmup_updates` have
    passed within the current task. `reset()` is called at every task switch,
    which is the continual part of C-CHAIN.
    """

    def __init__(self, target_rel_scale=10000.0, warmup_updates=10, window=50,
                 initial_coef=1.0, floor=1.0):
        """`initial_coef` and `floor` are the categorical reference's 1.0 and
        1.0 by default. The continuous-control reference (crl_dmc) starts at
        100 and has NO floor, because its churn term is an MSE between action
        means, orders of magnitude below the policy loss, so a floor of 1
        would pin the coefficient forever -- see
        `source/studies/brax/my_brax/cchain.py`, whose controller this mirrors for the
        generalists study's continuous PPO. Defaults reproduce every gymnax run.
        """
        self.target_rel_scale = target_rel_scale
        self.warmup_updates = warmup_updates
        self.window = window
        self.initial_coef = float(initial_coef)
        self.floor = float(floor)
        self.reset()

    def reset(self):
        self.coef = self.initial_coef if self.target_rel_scale > 0 else 0.0
        self.p_loss_hist = []
        self.p_reg_loss_hist = []

    def update(self, p_loss, p_reg_loss, updates_in_task):
        self.p_loss_hist.append(float(p_loss))
        self.p_reg_loss_hist.append(float(p_reg_loss))
        if self.target_rel_scale > 0 and updates_in_task >= self.warmup_updates:
            w = self.window
            running_p_loss = float(jnp.mean(jnp.abs(jnp.array(self.p_loss_hist[-w:]))))
            running_p_reg_loss = float(jnp.mean(jnp.array(self.p_reg_loss_hist[-w:])))
            self.coef = max(
                self.target_rel_scale * running_p_loss / (running_p_reg_loss + 1e-8),
                self.floor,
            )
        return self.coef


def make_chain_sgd_epochs(
    policy_network,
    value_network,
    compute_ppo_loss,
    clip_eps,
    vf_coef,
    num_epochs,
    num_minibatches,
    batch_size,
    churn_fn=None,
    log_prob_fn=None,
    entropy_fn=None,
    normalize_minibatch_advantages=False,
):
    """Build the jitted C-CHAIN replacement for the plain PPO SGD-epoch loop.

    The returned function has the same role as the trainers' `jit_sgd_epochs`,
    but additionally threads `chain_state` (reference params + step counter)
    through the update and across calls.

    `churn_fn(ref_logits, cur_logits) -> per-sample churn` is the regulariser
    and the diagnostic; None is the categorical cross-entropy above. A
    continuous policy passes the reference's MSE between action means instead
    (`source/studies/brax/my_brax/cchain.py` says why a cross-entropy over `[mean,
    log_std]` would be meaningless), together with its own `log_prob_fn` /
    `entropy_fn` for `compute_ppo_loss`. All three default to the gymnax
    behaviour, bit for bit.

    `normalize_minibatch_advantages` standardises each training minibatch's
    advantages, as the shared runner's PPO update does
    (`source/studies/generalists/train_ppo.make_update_fn`). The gymnax
    trainers normalise the whole batch before calling this and leave it off.
    Until 2026-09-17 the shared runner did not pass it, so its C-CHAIN trained
    on raw advantages while its PPO did not.
    """
    minibatch_size = batch_size // num_minibatches
    churn_fn = chain_policy_churn if churn_fn is None else churn_fn

    def compute_chain_loss(policy_params, value_params, ref_policy_params,
                           batch, reg_batch, p_reg_coef, ent_coef):
        loss, metrics = compute_ppo_loss(
            policy_params, value_params, policy_network, value_network,
            batch, clip_eps, vf_coef, ent_coef,
            log_prob_fn=log_prob_fn, entropy_fn=entropy_fn,
        )

        reg_obs = reg_batch['obs']
        cur_logits = policy_network.apply(policy_params, reg_obs)
        ref_logits = policy_network.apply(ref_policy_params, reg_obs)
        p_reg_loss = jnp.mean(churn_fn(ref_logits, cur_logits))

        loss = loss + p_reg_coef * p_reg_loss

        # Policy loss on the scale the coefficient controller expects
        # (clipped surrogate minus entropy bonus, as in the reference code).
        metrics['chain_p_loss'] = metrics['pg_loss'] - ent_coef * metrics['entropy']
        metrics['chain_p_reg_loss'] = p_reg_loss
        return loss, metrics

    def train_step_chain(policy_state, value_state, ref_policy_params, ref_value_params,
                         batch, reg_batch, ref_batch, ent_coef, p_reg_coef):
        def loss_fn(policy_params, value_params):
            return compute_chain_loss(
                policy_params, value_params, ref_policy_params,
                batch, reg_batch, p_reg_coef, ent_coef,
            )

        (loss, metrics), (policy_grads, value_grads) = jax.value_and_grad(
            loss_fn, argnums=(0, 1), has_aux=True
        )(policy_state.params, value_state.params)

        # Churn diagnostics on a third, independently drawn minibatch (no gradient)
        ref_obs = ref_batch['obs']
        cur_ref_logits = policy_network.apply(policy_state.params, ref_obs)
        ref_ref_logits = policy_network.apply(ref_policy_params, ref_obs)
        metrics['policy_churn'] = jnp.mean(
            churn_fn(ref_ref_logits, cur_ref_logits))
        cur_ref_values = value_network.apply(value_state.params, ref_obs)
        ref_ref_values = value_network.apply(ref_value_params, ref_obs)
        metrics['value_churn'] = jnp.mean(jnp.square(cur_ref_values - ref_ref_values))

        policy_state = policy_state.apply_gradients(grads=policy_grads)
        value_state = value_state.apply_gradients(grads=value_grads)
        return policy_state, value_state, loss, metrics

    @jax.jit
    def sgd_epochs_chain(policy_state, value_state, flat_batch, ent_coef,
                         p_reg_coef, chain_state, key):
        def single_epoch(carry, _):
            policy_state, value_state, chain_state, key = carry
            key, train_key, reg_key, ref_key = random.split(key, 4)

            def make_minibatches(perm_key):
                perm = random.permutation(perm_key, batch_size)
                return {
                    k: v[perm].reshape(num_minibatches, minibatch_size, *v.shape[1:])
                    for k, v in flat_batch.items()
                }

            # Three independent shuffles: training, regularization, churn diagnostics
            mb_data = (make_minibatches(train_key),
                       make_minibatches(reg_key),
                       make_minibatches(ref_key))

            def single_minibatch(carry, minibatches):
                policy_state, value_state, chain_state = carry
                ref_policy_params, ref_value_params, step = chain_state
                train_mb, reg_mb, ref_mb = minibatches
                if normalize_minibatch_advantages:
                    adv = train_mb['advantages']
                    train_mb = dict(train_mb, advantages=(adv - adv.mean())
                                    / (adv.std() + 1e-8))

                # Regularizer is off until two gradient steps have been taken
                active = jnp.where(step >= 2, 1.0, 0.0)

                # Reference for the next step: the params before this update
                next_ref = (policy_state.params, value_state.params)

                policy_state, value_state, loss, metrics = train_step_chain(
                    policy_state, value_state,
                    ref_policy_params, ref_value_params,
                    train_mb, reg_mb, ref_mb,
                    ent_coef, p_reg_coef * active,
                )
                metrics['chain_p_reg_loss'] = metrics['chain_p_reg_loss'] * active

                chain_state = (next_ref[0], next_ref[1], step + 1)
                return (policy_state, value_state, chain_state), metrics

            (policy_state, value_state, chain_state), all_metrics = jax.lax.scan(
                single_minibatch, (policy_state, value_state, chain_state), mb_data,
                length=num_minibatches,
            )
            return (policy_state, value_state, chain_state, key), all_metrics

        (policy_state, value_state, chain_state, key), epoch_metrics = jax.lax.scan(
            single_epoch, (policy_state, value_state, chain_state, key), None,
            length=num_epochs,
        )

        last_metrics = jax.tree_util.tree_map(lambda x: x[-1, -1], epoch_metrics)
        last_loss = last_metrics['pg_loss'] + vf_coef * last_metrics['vf_loss']
        # Averages over this update's minibatches, used by the controller and logs
        for k in ['chain_p_loss', 'chain_p_reg_loss', 'policy_churn', 'value_churn']:
            last_metrics[k] = jnp.mean(epoch_metrics[k])

        return policy_state, value_state, chain_state, last_loss, last_metrics

    return sgd_epochs_chain


def add_chain_args(parser):
    """CLI flags shared by both trainers."""
    parser.add_argument('--chain_target_rel_scale', type=float, default=10000.0,
                        help='Target relative scale of policy loss vs churn loss '
                             '(0 disables the churn regularizer)')
    parser.add_argument('--chain_warmup_updates', type=int, default=10,
                        help='Updates into a task before the coefficient is auto-tuned')
    parser.add_argument('--chain_coef_window', type=int, default=50,
                        help='Number of recent updates averaged by the coefficient controller')
    return parser
