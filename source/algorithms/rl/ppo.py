"""The PPO implementation every RL method in the study shares.

PPO, ReDo-PPO, TRAC-PPO and C-CHAIN differ in what they do *around* the update,
not in the update itself, and the PBT trainer runs this same step for each
population member.

`make_rollout_fn` collects one window of experience from a batch of
environments and computes GAE advantages; `make_update_fn` runs PPO's epochs
of minibatch updates over it. Both are independent of which suite the
environment comes from: the sub-task enters only through the `env_step` and
`offset_fn` the caller passes. The loop around them -- schedule, evaluation,
checkpoints, the continual-RL mechanisms and PBT -- is
`source/runners/train_ppo.py:run_ppo`.
"""

from functools import partial

import jax
import jax.numpy as jnp
import optax
from jax import random

from source.algorithms.rl import action_heads as actors
from source.algorithms.rl import redo
from source.algorithms.networks import ACTIVATIONS, POLICY_ARCH

# The gymnax policy's hidden activation is pinned in POLICY_ARCH; the critic is
# not part of that contract (it has no NE counterpart) and `ValueNetwork` is
# relu by construction.
GYMNAX_POLICY_ACTIVATION = POLICY_ARCH['gymnax']['activation']
GYMNAX_VALUE_ACTIVATION = 'relu'


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
    (`source/algorithms/rl/action_heads.py`). Both are vmapped over the batch
    here, so a network whose logits have extra axes needs no other change.
    """
    log_prob_fn = actors.categorical_log_prob if log_prob_fn is None else log_prob_fn
    entropy_fn = actors.categorical_entropy if entropy_fn is None else entropy_fn
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
    is given, and giving it the policy alone is what broke the gymnax `trac`
    arm (the old per-method gymnax trainer had to build a joint optimiser for
    it). Every other suite is joint already because it has one
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


def make_rollout_fn(env_step, head, actor, value_net, hp, offset_fn):
    """Return ``rollout(policy_params, value_params, carry, task, stats)``.

    ``carry`` is ``(obs, state, key)`` and is threaded across updates, so the
    environments are never reset at an update boundary -- a rollout is a window
    over episodes that keep running, which is what the GAE bootstrap assumes.

    The sub-task enters in two places and only two: ``offset_fn(task)`` is
    added to the observation before the networks read it (the sub-task vector
    on gymnax and under obs_noise, 0 for a friction sub-task), and ``env_step``
    runs the environment under the sub-task's physics (a no-op on gymnax). The
    returns remain comparable with NES's.

    ``stats`` are the observation normaliser's running statistics, or None.
    Inputs are normalised with the statistics as they stood when the policy
    ACTED, and the statistics are updated with this window's observations for
    the next one, so the stored log-probabilities and the update's forward
    pass see the same inputs.
    """
    num_steps = hp['num_steps']
    reward_scale = float(hp.get('reward_scale', 1.0))

    def rollout(policy_params, value_params, carry, task, stats):
        offset = offset_fn(task)

        def inputs(obs):
            shifted = obs + offset
            return shifted, (actors.normalize(shifted, stats)
                             if stats is not None else shifted)

        def env_step_fn(carry, _):
            obs, state, key = carry
            shifted, inp = inputs(obs)
            logits = actor.apply(policy_params, inp)
            values = value_net.apply(value_params, inp)

            key, action_key, step_key = random.split(key, 3)
            actions = jax.vmap(head.sample, in_axes=(0, 0))(
                random.split(action_key, obs.shape[0]), logits)
            log_probs = jax.vmap(head.log_prob)(logits, actions)

            next_obs, next_state, reward, done = env_step(
                step_key, state, actions, task)
            if reward_scale != 1.0:
                reward = reward * reward_scale

            transition = {'obs': inp, 'actions': actions,
                          'log_probs': log_probs, 'values': values,
                          'rewards': reward, 'dones': done.astype(jnp.float32)}
            if stats is not None:
                # Only when there is a normaliser to update. An extra scan
                # output changes XLA's fusion of the rollout and with it the
                # float32 rounding, and a gymnax run has to stay bit-identical
                # to the runs on disk.
                transition['raw_obs'] = shifted
            return (next_obs, next_state, key), transition

        (obs, state, key), traj = jax.lax.scan(
            env_step_fn, carry, None, length=num_steps)

        # V(s_T) for the bootstrap, on the state the window stopped on.
        last_value = value_net.apply(value_params, inputs(obs)[1])

        # Explicit wrapper rather than functools.partial: vmap passes its
        # arguments positionally, and last_value is the sixth parameter of
        # gae_advantages, so a partial over the keyword arguments would receive
        # it as `gamma`.
        def gae(rewards, values, dones, bootstrap):
            return gae_advantages(
                rewards, values, dones, gamma=hp['gamma'],
                gae_lambda=hp['gae_lambda'], last_value=bootstrap)

        advantages = jax.vmap(gae, in_axes=(1, 1, 1, 0), out_axes=1)(
            traj['rewards'], traj['values'], traj['dones'], last_value)
        returns = advantages + traj['values']

        flat = lambda x: x.reshape((-1,) + x.shape[2:])
        batch = {'obs': flat(traj['obs']), 'actions': flat(traj['actions']),
                 'log_probs': flat(traj['log_probs']),
                 'advantages': flat(advantages), 'returns': flat(returns)}
        if stats is not None:
            stats = actors.update_norm_stats(stats, traj['raw_obs'])
        return (obs, state, key), batch, stats

    return rollout


def make_update_fn(actor, value_net, hp, head, joint_tx=None):
    """Return ``update(policy_state, value_state, batch, key, opt_state=None)``.

    ``num_epochs`` passes over the batch, reshuffled each epoch and split into
    ``num_minibatches``. Advantages are normalised per minibatch, the usual PPO
    convention.

    With ``joint_tx`` the two parameter sets are updated by ONE optax
    transformation over both, threading ``opt_state`` through. That is only
    needed for TRAC, whose tuner is a handful of scalars over whatever pytree it
    is handed -- give it the policy alone and it is tuning half the model. Every
    other optimiser here is per-parameter, so the two paths are arithmetically
    identical for them. See `train_step_joint`.
    """
    num_minibatches = hp['num_minibatches']
    if joint_tx is not None:
        step_joint = partial(train_step_joint, actor, value_net,
                             hp['clip_eps'], hp['vf_coef'], joint_tx)
    step = partial(train_step, actor, value_net, hp['clip_eps'],
                   hp['vf_coef'])
    dist = dict(log_prob_fn=head.log_prob, entropy_fn=head.entropy)

    def update(policy_state, value_state, batch, key, opt_state=None,
               ent_coef=None):
        # A per-member entropy coefficient under PBT, the config's
        # otherwise. None keeps the Python constant every other method
        # compiles in, so their traces are unchanged.
        ec = hp['ent_coef'] if ent_coef is None else ent_coef
        batch_size = batch['obs'].shape[0]
        minibatch_size = batch_size // num_minibatches

        def epoch(carry, epoch_key):
            policy_state, value_state, opt_state = carry
            perm = random.permutation(epoch_key, batch_size)
            shuffled = jax.tree.map(lambda x: x[perm], batch)
            minibatches = jax.tree.map(
                lambda x: x.reshape((num_minibatches, minibatch_size)
                                    + x.shape[1:]), shuffled)

            def minibatch(carry, mb):
                policy_state, value_state, opt_state = carry
                mb = dict(mb)
                mb['advantages'] = ((mb['advantages'] - mb['advantages'].mean())
                                    / (mb['advantages'].std() + 1e-8))
                if joint_tx is not None:
                    (policy_state, value_state, opt_state, loss,
                     metrics) = step_joint(policy_state, value_state,
                                           opt_state, mb, ec,
                                           **dist)
                else:
                    policy_state, value_state, loss, metrics = step(
                        policy_state, value_state, mb, ec, **dist)
                return (policy_state, value_state, opt_state), (loss, metrics)

            carry, out = jax.lax.scan(
                minibatch, (policy_state, value_state, opt_state), minibatches)
            return carry, out

        carry, (losses, metrics) = jax.lax.scan(
            epoch, (policy_state, value_state, opt_state),
            random.split(key, hp['num_epochs']))
        policy_state, value_state, opt_state = carry
        return policy_state, value_state, opt_state, {
            'loss': jnp.mean(losses),
            'entropy': jnp.mean(metrics['entropy']),
            'approx_kl': jnp.mean(metrics['approx_kl']),
        }

    return update
