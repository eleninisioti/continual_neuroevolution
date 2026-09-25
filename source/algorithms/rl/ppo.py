"""The PPO implementation every RL method in the study shares.

PPO, ReDo-PPO, TRAC-PPO and C-CHAIN differ in what they do *around* the update,
not in the update itself, and the PBT trainer runs this same step for each
population member. One copy is what makes "the RL arm differs only in its
continual-learning mechanism" true by construction rather than by four files
happening to agree.

Every symbol here was verified identical between the stationary and continual
gymnax trainers before it was lifted, so importing it cannot move a number.

One of them belongs to a specific method rather than to PPO itself, and lives
here because it is a hook the shared update exposes:

  run_redo_pass  ReDo (Sokar et al., ICML 2023) -- re-initialises neurons whose
                 normalised activation score has collapsed. The mechanism is in
                 `source/algorithms/rl/redo.py`; this is the pass over the selected
                 networks.

TRAC has no hook here. It is an optimiser wrapper (`start_trac`), so it is
applied where the trainer builds its optax chain and is invisible to the update.
There used to be an `update_entropy_coef` in this file that `--method trac`
called; it was an adaptive entropy coefficient, which is not TRAC and not what
the brax, mujoco or kinetix `trac` arms ran. It is gone.

`collect_rollout` is deliberately NOT here: the stationary and continual
trainers hold the only two genuinely different versions of it, because the
continual one has to know where a sub-task boundary falls.
"""

import jax
import jax.numpy as jnp
import optax
from jax import random

from source.algorithms.rl import redo
from source.algorithms.networks import ACTIVATIONS, POLICY_ARCH

# The gymnax policy's hidden activation is pinned in POLICY_ARCH; the critic is
# not part of that contract (it has no NE counterpart) and `ValueNetwork` is
# relu by construction.
GYMNAX_POLICY_ACTIVATION = POLICY_ARCH['gymnax']['activation']
GYMNAX_VALUE_ACTIVATION = 'relu'


def categorical_sample(key, logits):
    """Sample from categorical distribution."""
    return jax.random.categorical(key, logits)


def categorical_log_prob(logits, action):
    """Log probability of action under categorical distribution."""
    log_probs = jax.nn.log_softmax(logits)
    return log_probs[action]


def categorical_entropy(logits):
    """Entropy of categorical distribution."""
    log_probs = jax.nn.log_softmax(logits)
    probs = jax.nn.softmax(logits)
    return -jnp.sum(probs * log_probs, axis=-1)


def gae_advantages(rewards, values, dones, gamma=0.99, gae_lambda=0.95, last_value=0.0):
    """Compute GAE advantages using jax.lax.scan (GPU-friendly).

    `last_value` is V(s_T), the value of the state the rollout stopped on. A
    rollout of num_steps is a *window* over an episode that is usually still
    running, so the return past its end is V(s_T), not 0: bootstrapping with 0
    tells the critic every rollout ended in an absorbing state worth nothing,
    which biases every advantage in the last steps of the window downward.
    Defaults to 0.0 so callers that have not been updated keep their old
    numbers bit-exactly rather than changing silently.
    """
    last_value = jnp.asarray(last_value, dtype=values.dtype).reshape(1)
    values_with_bootstrap = jnp.concatenate([values, last_value])

    def scan_fn(gae, t):
        # t goes from 0 to T-1 (reversed order via [::-1])
        delta = rewards[t] + gamma * values_with_bootstrap[t + 1] * (1 - dones[t]) - values[t]
        gae = delta + gamma * gae_lambda * (1 - dones[t]) * gae
        return gae, gae

    # Scan in reverse order
    indices = jnp.arange(len(rewards))[::-1]
    _, advantages_reversed = jax.lax.scan(scan_fn, 0.0, indices)

    return advantages_reversed[::-1]


def make_vec_env_fns(env, env_params, num_envs):
    """Create vectorized reset and step functions."""

    def vec_reset(key):
        keys = random.split(key, num_envs)
        return jax.vmap(lambda k: env.reset(k, env_params))(keys)

    def vec_step(keys, states, actions):
        return jax.vmap(lambda k, s, a: env.step(k, s, a, env_params))(keys, states, actions)

    return jax.jit(vec_reset), jax.jit(vec_step)


def compute_ppo_loss(
    policy_params,
    value_params,
    policy_network,
    value_network,
    batch,
    clip_eps=0.2,
    vf_coef=0.5,
    ent_coef=0.01,
    log_prob_fn=None,
    entropy_fn=None,
):
    """Compute PPO loss.

    `log_prob_fn` / `entropy_fn` act on ONE timestep's logits and default to the
    single-categorical pair, which is what every gymnax/brax/kinetix caller
    wants. The generalists study's Gaussian head passes its own pair
    (`source/studies/generalists/actors.py`). Both are vmapped over the batch
    here, so a network whose logits have extra axes needs no other change.
    """
    log_prob_fn = categorical_log_prob if log_prob_fn is None else log_prob_fn
    entropy_fn = categorical_entropy if entropy_fn is None else entropy_fn
    obs = batch['obs']
    actions = batch['actions']
    old_log_probs = batch['log_probs']
    advantages = batch['advantages']
    returns = batch['returns']

    # Forward pass
    logits = policy_network.apply(policy_params, obs)
    values = value_network.apply(value_params, obs)

    # Policy loss
    log_probs = jax.vmap(log_prob_fn)(logits, actions)
    ratio = jnp.exp(log_probs - old_log_probs)

    # Clipped surrogate objective
    pg_loss1 = -advantages * ratio
    pg_loss2 = -advantages * jnp.clip(ratio, 1 - clip_eps, 1 + clip_eps)
    pg_loss = jnp.maximum(pg_loss1, pg_loss2).mean()

    # Value loss
    vf_loss = 0.5 * jnp.square(values - returns).mean()

    # Entropy bonus
    entropy = jax.vmap(entropy_fn)(logits).mean()

    # Total loss
    loss = pg_loss + vf_coef * vf_loss - ent_coef * entropy

    return loss, {
        'pg_loss': pg_loss,
        'vf_loss': vf_loss,
        'entropy': entropy,
        'approx_kl': jnp.mean((ratio - 1) - jnp.log(ratio)),
    }


def train_step(
    policy_network,
    value_network,
    clip_eps,
    vf_coef,
    policy_state,
    value_state,
    batch,
    ent_coef,
    log_prob_fn=None,
    entropy_fn=None,
):
    """Perform one training step."""

    def loss_fn(policy_params, value_params):
        return compute_ppo_loss(
            policy_params, value_params,
            policy_network, value_network,
            batch, clip_eps, vf_coef, ent_coef,
            log_prob_fn=log_prob_fn, entropy_fn=entropy_fn,
        )

    # Compute gradients
    (loss, metrics), (policy_grads, value_grads) = jax.value_and_grad(
        loss_fn, argnums=(0, 1), has_aux=True
    )(policy_state.params, value_state.params)

    # Update
    policy_state = policy_state.apply_gradients(grads=policy_grads)
    value_state = value_state.apply_gradients(grads=value_grads)

    return policy_state, value_state, loss, metrics


def train_step_joint(
    policy_network,
    value_network,
    clip_eps,
    vf_coef,
    tx,
    policy_state,
    value_state,
    opt_state,
    batch,
    ent_coef,
    log_prob_fn=None,
    entropy_fn=None,
):
    """`train_step` with ONE optax transformation over both parameter sets.

    Identical arithmetic to `train_step` for any per-parameter optimiser -- the
    same gradients reach the same adam -- but it exists for TRAC, which is not
    per-parameter. TRAC's tuner is a handful of scalars over whatever pytree it
    is given, and giving it the policy alone is what breaks the gymnax `trac`
    arm: see the block in `source/studies/gymnax/train_RL_gymnax.py` where the joint
    optimiser is built. Every other suite is joint already because it has one
    actor-critic network and therefore one optimiser.

    `policy_state.tx` and `value_state.tx` are unused on this path; `opt_state`
    is the single state and the caller threads it through the epoch loop.
    """

    def loss_fn(policy_params, value_params):
        return compute_ppo_loss(
            policy_params, value_params,
            policy_network, value_network,
            batch, clip_eps, vf_coef, ent_coef,
            log_prob_fn=log_prob_fn, entropy_fn=entropy_fn,
        )

    (loss, metrics), (policy_grads, value_grads) = jax.value_and_grad(
        loss_fn, argnums=(0, 1), has_aux=True
    )(policy_state.params, value_state.params)

    params = {'policy': policy_state.params, 'value': value_state.params}
    grads = {'policy': policy_grads, 'value': value_grads}
    updates, opt_state = tx.update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)

    # `step` is advanced by hand because `apply_gradients` is what normally
    # does it, and it would also run the per-state optimiser we are bypassing.
    policy_state = policy_state.replace(
        params=new_params['policy'], step=policy_state.step + 1)
    value_state = value_state.replace(
        params=new_params['value'], step=value_state.step + 1)

    return policy_state, value_state, opt_state, loss, metrics


def run_redo_pass(policy_state, value_state, obs, key, args, hp,
                  policy_activation=None, value_activation=None,
                  policy_layers=None, policy_activations_fn=None,
                  policy_extra_fan_in=None):
    """One ReDo pass over the selected networks. Returns (policy, value, stats).

    The dormancy criterion is derived from each network's hidden activation
    rather than passed in -- see `source/algorithms/rl/redo.py`. Both gymnax networks are
    relu (`MLPPolicy` and `ValueNetwork` in core/networks.py), so both get the
    reference's magnitude test; a suite whose policy is tanh gets the
    variability test automatically.

    `policy_activation` / `value_activation` name the hidden activation each
    network was trained with, for a caller whose networks are not the gymnax
    ones -- the generalists study's PPO on the mjx bodies searches the tanh
    `ContinuousMLPPolicy`. None is the gymnax pair, so every existing caller
    is unchanged.

    `policy_layers` / `policy_activations_fn` are for a policy that is not a
    Dense chain (the grid conv policy); see
    `redo.apply_redo`. None is the Dense-chain walk, unchanged.
    `policy_extra_fan_in` is for a policy whose flattened conv map is joined
    by inputs that are not hidden units (the Kinetix pixel policy); also
    documented at `redo.apply_redo`.
    """
    targets = args.redo_targets
    stats = {}
    key, policy_key, value_key = random.split(key, 3)

    policy_activation = policy_activation or GYMNAX_POLICY_ACTIVATION
    value_activation = value_activation or GYMNAX_VALUE_ACTIVATION
    policy_criterion = redo.criterion_for_activation(policy_activation)
    value_criterion = redo.criterion_for_activation(value_activation)

    if targets in ('both', 'policy'):
        policy_state, stats['policy'] = redo.apply_redo(
            policy_state, obs, len(hp['policy_hidden_dims']), policy_key,
            tau=args.redo_tau,
            activation_fn=ACTIVATIONS[policy_activation],
            criterion=policy_criterion,
            reset_adam_count=not args.redo_keep_adam_count,
            layers=policy_layers, activations_fn=policy_activations_fn,
            extra_fan_in=policy_extra_fan_in,
        )
    if targets in ('both', 'value'):
        value_state, stats['value'] = redo.apply_redo(
            value_state, obs, len(hp['value_hidden_dims']), value_key,
            tau=args.redo_tau,
            activation_fn=ACTIVATIONS[value_activation],
            criterion=value_criterion,
            reset_adam_count=not args.redo_keep_adam_count,
        )
    return policy_state, value_state, stats


