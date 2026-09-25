"""
Custom PPO training function for continual learning.

This module provides a PPO training function that supports switching environments
mid-training while preserving the full training state (including optimizer state,
normalizer stats, etc.). This is essential for true continual learning where
we don't want to reset the optimizer between tasks.

Based on brax.training.agents.ppo.train but modified to support environment switching.
"""

import functools
import time
from typing import Any, Callable, List, Optional, Tuple, Union

from absl import logging
from brax import envs
from brax.training import acting
from brax.training import gradients
from brax.training import pmap
from brax.training import types
from brax.training.acme import running_statistics
from brax.training.acme import specs
from brax.training.agents.ppo import losses as ppo_losses
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.types import Params
from brax.training.types import PRNGKey
import flax
from flax import linen
import jax
import jax.numpy as jnp
import numpy as np
import optax

# Import ReDo support from my_brax
from source.studies.brax.my_brax import cchain as my_brax_cchain
from source.studies.brax.my_brax import networks as my_brax_networks
from source.studies.brax.my_brax import dormant as my_brax_dormant
from source.metrics.weight_stats import weight_stats
from source.metrics import ntk as ntk_metrics

InferenceParams = Tuple[running_statistics.NestedMeanStd, Params]
Metrics = types.Metrics

_PMAP_AXIS_NAME = 'i'


# ONE COPY, in source/studies/brax/my_brax/dormant.py. The tracker and the mask
# computation used to live here and nowhere else, which is why ppo_train.py --
# the STATIONARY core -- had no dormancy column at all. Aliased rather than
# renamed at the call sites so nothing else in this file changes.
DormantNeuronTracker = my_brax_dormant.DormantNeuronTracker
_compute_dormant_masks = my_brax_dormant.compute_dormant_masks


@flax.struct.dataclass
class TrainingState:
    """Contains training state for the learner."""
    optimizer_state: optax.OptState
    params: ppo_losses.PPONetworkParams
    normalizer_params: running_statistics.RunningStatisticsState
    env_steps: jnp.ndarray
    # C-CHAIN controller state, or None when --use_cchain is off. Carried in the
    # training state for the same reason the optimizer state is: it must survive
    # a task switch, except for the parts reset_chain_state deliberately clears.
    chain_state: Any = None


def _unpmap(v):
    return jax.tree_util.tree_map(lambda x: x[0], v)


def _apply_redo(
    training_state: TrainingState,
    ppo_network,
    sample_obs,
    local_key,
    local_devices_to_use: int,
    redo_tau: float = my_brax_networks.REDO_DEFAULT_TAU,
    policy_hidden_sizes: tuple = None,
    value_hidden_sizes: tuple = None,
) -> TrainingState:
    """One ReDo pass over the policy and value networks.

    Identical to the stationary trainer's; see `my_brax/ppo_train.py` for the
    commentary. All four steps of source/algorithms/rl/redo.py, including the Adam-moment
    reset this file used to skip with the comment "the paper suggests this is
    fine" -- the paper says the opposite.

    `policy_hidden_sizes` / `value_hidden_sizes` default to None, meaning "read
    them off the parameters". They used to default to brax's (256, 256), which
    is not what ant or cheetah run.

    Args:
        training_state: Current training state with network parameters
        ppo_network: PPO network with policy and value networks
        sample_obs: Sample observations to compute activations
        local_key: Random key for reinitialization
        local_devices_to_use: Number of local devices
        redo_tau: Threshold the networks were built with; logged, not applied here

    Returns:
        Updated training state with ReDo applied
    """
    params = _unpmap(training_state.params)
    normalizer_params = _unpmap(training_state.normalizer_params)
    optimizer_state = _unpmap(training_state.optimizer_state)

    sample_obs_flat = _unpmap(sample_obs)
    if isinstance(sample_obs_flat, dict):
        sample_obs_flat = sample_obs_flat.get('state', list(sample_obs_flat.values())[0])
    sample_obs_flat = sample_obs_flat.reshape(-1, sample_obs_flat.shape[-1])

    local_key, policy_rng, value_rng = jax.random.split(local_key, 3)

    policy_hidden_list = (
        list(policy_hidden_sizes) if policy_hidden_sizes is not None
        else my_brax_networks.hidden_layer_sizes_from_params(params.policy))
    value_hidden_list = (
        list(value_hidden_sizes) if value_hidden_sizes is not None
        else my_brax_networks.hidden_layer_sizes_from_params(params.value))

    policy_network = ppo_network.policy_network
    value_network = ppo_network.value_network
    for net, name in ((policy_network, 'policy'), (value_network, 'value')):
        if not hasattr(net, 'apply_with_dormant_masks'):
            raise ValueError(
                f'ReDo was requested but the {name} network cannot report dormant '
                f'neurons. Build the networks with my_brax.networks, not brax\'s.'
            )

    _, _, policy_masks = policy_network.apply_with_dormant_masks(
        normalizer_params, params.policy, sample_obs_flat
    )
    _, _, value_masks = value_network.apply_with_dormant_masks(
        normalizer_params, params.value, sample_obs_flat
    )

    new_policy_params = my_brax_networks.apply_redo_to_params(
        params.policy, policy_hidden_list, policy_masks, rng=policy_rng,
    )
    new_value_params = my_brax_networks.apply_redo_to_params(
        params.value, value_hidden_list, value_masks, rng=value_rng,
    )

    n_policy = int(sum(jnp.sum(m) for m in policy_masks))
    n_value = int(sum(jnp.sum(m) for m in value_masks))
    logging.info(
        f'ReDo (tau={redo_tau}): recycled {n_policy} policy and {n_value} value neurons'
    )

    new_params = ppo_losses.PPONetworkParams(
        policy=new_policy_params,
        value=new_value_params,
    )
    new_optimizer_state = my_brax_networks.reset_adam_moments_for_masks(
        optimizer_state,
        policy_hidden_list,
        value_hidden_list,
        policy_masks,
        value_masks,
    )

    devices = jax.local_devices()[:local_devices_to_use]
    new_params = jax.device_put_replicated(new_params, devices)
    new_optimizer_state = jax.device_put_replicated(new_optimizer_state, devices)

    return TrainingState(
        optimizer_state=new_optimizer_state,
        params=new_params,
        normalizer_params=training_state.normalizer_params,
        env_steps=training_state.env_steps,
        # ReDo and C-CHAIN are not run together by any sweep, but carrying the
        # controller through rather than dropping it keeps the state valid if
        # they ever are.
        chain_state=training_state.chain_state,
    )


def train_continual(
    # Environment factory that takes a multiplier and returns (train_env, eval_env)
    env_factory: Callable[[float], Tuple[envs.Env, envs.Env]],
    # List of task multipliers (gravity or friction) for each task
    task_multipliers: List[float],
    # Timesteps per task
    timesteps_per_task: int,
    # PPO hyperparameters
    num_envs: int = 2048,
    episode_length: int = 1000,
    action_repeat: int = 1,
    wrap_env_fn: Optional[Callable[[Any], Any]] = None,
    learning_rate: float = 3e-4,
    entropy_cost: float = 1e-2,
    discounting: float = 0.97,
    unroll_length: int = 10,
    batch_size: int = 256,
    num_minibatches: int = 32,
    num_updates_per_batch: int = 8,
    normalize_observations: bool = True,
    reward_scaling: float = 0.1,
    clipping_epsilon: float = 0.3,
    gae_lambda: float = 0.95,
    max_grad_norm: Optional[float] = 1.0,
    # Clip observations into [-obs_clip, obs_clip] before they reach the
    # observation normalizer and the PPO loss. None reproduces the pre-fix
    # behaviour exactly, which is what every run currently on disk was produced
    # under -- see the comment at the clip site in `training_step` for the
    # collapse this guards against. Left at None deliberately so existing
    # results stay reproducible; pass --obs_clip to enable.
    obs_clip: Optional[float] = None,
    # Skip any optimizer update whose gradients contain NaN/Inf, via
    # optax.apply_if_finite. 0 disables (pre-fix behaviour). This is the
    # actual guard against the collapse: clip_by_global_norm does NOT stop
    # NaN -- it propagates straight through Adam into the weights, which is
    # how ppo_obsclip trial_6 ended sub-task 1 with all 20,364 policy
    # parameters NaN while its observation normalizer was still healthy.
    nan_guard: int = 0,
    network_factory: types.NetworkFactory[ppo_networks.PPONetworks] = ppo_networks.make_ppo_networks,
    # Floor on the action std. Only takes effect on the my_brax network path
    # (track_dormant / use_redo), which is what every walker run here uses.
    # brax's default 0.001 is no floor at all; see make_ppo_networks.
    policy_min_std: float = 0.001,
    # 'tanh_normal' (brax default) or 'state_independent_normal' (C-CHAIN /
    # CleanRL: unbounded Gaussian, one learnable log-std shared across states,
    # sigma 1.0 at init). The latter is what makes entropy_cost 0 survivable --
    # see my_brax/networks.StateIndependentNormalDistribution.
    distribution_type: str = 'tanh_normal',
    seed: int = 0,
    num_eval_envs: int = 128,
    num_evals_per_task: int = 5,
    # Trac optimizer
    use_trac: bool = False,
    # ReDo (Reinitializing Dormant Neurons)
    use_redo: bool = False,
    redo_frequency: int = 10,  # Apply ReDo every N epochs
    redo_tau: float = my_brax_networks.REDO_DEFAULT_TAU,  # Dormancy threshold; see source/algorithms/rl/redo.py (fraction of layer mean)
    # Dormant neuron tracking
    track_dormant: bool = False,  # Track dormant neurons and their age
    dormant_tau: float = my_brax_networks.REDO_DEFAULT_TAU,  # Dormancy threshold; see source/algorithms/rl/redo.py
    # C-CHAIN (Tang et al., ICML 2025), continuous-control form. See
    # source/studies/brax/my_brax/cchain.py. The controller is re-tuned at every task switch, which
    # is the continual half of the method and has no analogue in the
    # non-continual trainer.
    use_cchain: bool = False,
    # Whether C-CHAIN's coefficient controller is re-calibrated at a sub-task
    # boundary. Off by default, which is what makes the comparison fair: every
    # other method here -- GA, ES, DNS, PPO, TRAC, ReDo -- is never told a switch
    # happened, and a controller that re-derives its coefficient exactly when the
    # loss scales shift is using information the others do not have. ReDo's
    # neuron resets are periodic (`redo_frequency` epochs) rather than aligned to
    # boundaries, so they are not this.
    cchain_reset_on_switch: bool = False,
    # C-CHAIN's ORACLE. crl_run_ppo_dmc_oracle.py differs from
    # crl_run_ppo_dmc.py by exactly one thing: it drops the `if agent is None`
    # guard, so a fresh network and a fresh Adam are built at the top of every
    # sub-task instead of once for the whole chain. Everything else -- budget,
    # rollout, evaluation -- is identical. That is the comparison their Table 3
    # rests on: on Continual Walker vanilla scores 305.2 against the oracle's
    # 396.0, i.e. carrying the agent across the chain is WORSE than restarting
    # it, which is the plasticity claim. Without this arm a reproduction can
    # show a curve going down but cannot show the gap that makes it a result.
    #
    # The normalizer is reset with the network on purpose: C-CHAIN build a fresh
    # env with a fresh NormalizeObservation wrapper per sub-task, so their
    # observation statistics restart too, in the oracle AND the vanilla run.
    reset_agent_per_task: bool = False,
    chain_target_rel_scale: float = 0.05,
    chain_warmup_iterations: int = 50,
    chain_coef_window: int = 100,
    # Callbacks
    progress_fn: Callable[[int, int, float, Metrics], None] = lambda *args: None,
    # Checkpoint callback: called at end of each task with (task_idx, params_dict)
    checkpoint_fn: Callable[[int, dict], None] = lambda *args: None,
    # Generation checkpoint callback: called at specific generations for KL divergence analysis
    # Signature: (generation, params_dict) -> None
    generation_checkpoint_fn: Callable[[int, dict], None] = lambda *args: None,
    # GIF callback: called at end of each task to save evaluation GIFs
    # Signature: (task_idx, multiplier, inference_fn, env, params) -> None
    gif_callback_fn: Optional[Callable] = None,
):
    """
    PPO training with continual learning support.
    
    This function trains across multiple environments (tasks) while preserving
    the full training state between tasks - including optimizer state, 
    normalizer parameters, and network weights.
    
    Args:
        env_factory: Function that takes task multiplier and returns (train_env, eval_env)
        task_multipliers: List of multipliers (gravity or friction), one per task
        timesteps_per_task: Number of timesteps to train on each task
        use_trac: If True, wrap optimizer with TRAC for adaptive learning rates
        use_redo: If True, apply ReDo (Reinitializing Dormant Neurons) periodically.
            This technique from "The Dormant Neuron Phenomenon" paper reinitializes
            neurons that have become dormant (low activation) during training.
        redo_frequency: Apply ReDo every N epochs (only used if use_redo=True)
        redo_tau: Threshold for dormant neuron detection as fraction of layer mean
            activation (only used if use_redo=True)
        ... (other args same as brax ppo.train)
        progress_fn: Callback with signature (global_step, task_idx, multiplier, metrics)
    
    Returns:
        Tuple of (make_policy function, final params, final metrics)
    """
    num_tasks = len(task_multipliers)
    
    # Device setup
    process_count = jax.process_count()
    process_id = jax.process_index()
    local_device_count = jax.local_device_count()
    local_devices_to_use = local_device_count
    device_count = local_devices_to_use * process_count
    
    logging.info(
        'Device count: %d, process count: %d (id %d), local device count: %d',
        jax.device_count(), process_count, process_id, local_device_count,
    )
    
    assert num_envs % device_count == 0
    assert batch_size * num_minibatches % num_envs == 0
    
    # Steps per training iteration
    env_step_per_training_step = batch_size * unroll_length * num_minibatches * action_repeat

    # One epoch = one evaluation, so a task is exactly num_evals_per_task epochs
    # and the work is spread across them. This used to be the other way round --
    # one training step per epoch, with the epoch count derived from the step
    # budget -- which made two things wrong as soon as the budget was not
    # num_evals_per_task training steps:
    #
    #   * the run evaluated once per training step (625 times per task at the
    #     rebuttal budget, not the 100 requested), and
    #   * `generation`, computed below as task_idx * num_evals_per_task + it,
    #     ran past the next task's starting value, so the per-task generation
    #     ranges overlapped in wandb.
    #
    # Both are fixed by making num_evals_per_task the epoch count, which is also
    # what the comment on the training loop always claimed.
    training_steps_per_task = int(np.ceil(timesteps_per_task / env_step_per_training_step))
    num_training_epochs = max(1, min(num_evals_per_task, training_steps_per_task))
    num_training_steps_per_epoch = max(
        1, int(round(training_steps_per_task / num_training_epochs)))

    actual_steps_per_task = (num_training_epochs * num_training_steps_per_epoch
                             * env_step_per_training_step)
    # print, not logging.info: absl's default verbosity drops INFO, so anything
    # sent there is invisible in the sweep's per-condition logs. What budget a
    # run actually got is not something to have to reconstruct afterwards.
    print(
        f'Continual PPO: {num_tasks} tasks, {timesteps_per_task:,} requested steps/task, '
        f'{actual_steps_per_task:,} actual steps/task ({num_training_epochs} epochs x '
        f'{num_training_steps_per_epoch} training steps x {env_step_per_training_step:,} steps)',
        flush=True,
    )
    # The budget is only hit exactly when num_evals_per_task divides the number
    # of training steps in a task. It is not silently rounded away: the sweep
    # matches NE and RL on environment steps, so a few percent here is a few
    # percent of the comparison.
    if abs(actual_steps_per_task - timesteps_per_task) > 0.01 * timesteps_per_task:
        print(
            f'WARNING: actual steps/task ({actual_steps_per_task:,}) differs from '
            f'requested ({timesteps_per_task:,}) by more than 1%. '
            f'{training_steps_per_task} training steps per task do not divide into '
            f'{num_evals_per_task} evaluations; pick a num_evals_per_task that '
            f'divides it to hit the budget exactly.',
            flush=True,
        )
    
    # Random keys
    key = jax.random.PRNGKey(seed)
    global_key, local_key = jax.random.split(key)
    local_key = jax.random.fold_in(local_key, process_id)
    local_key, key_env, eval_key = jax.random.split(local_key, 3)
    key_policy, key_value = jax.random.split(global_key)
    
    # Create first environment to get dimensions
    first_env, first_eval_env = env_factory(task_multipliers[0])
    
    # Wrap environment
    def wrap_env(env, num_envs_to_wrap, key_envs):
        if wrap_env_fn is not None:
            return wrap_env_fn(
                env,
                episode_length=episode_length,
                action_repeat=action_repeat,
            )
        else:
            from brax import envs as brax_envs
            return brax_envs.training.wrap(
                env,
                episode_length=episode_length,
                action_repeat=action_repeat,
            )
    
    wrapped_env = wrap_env(first_env, num_envs, key_env)
    
    # Initial reset to get observation shape
    key_envs = jax.random.split(key_env, num_envs // process_count)
    key_envs = jnp.reshape(key_envs, (local_devices_to_use, -1) + key_envs.shape[1:])
    reset_fn = jax.pmap(wrapped_env.reset, axis_name=_PMAP_AXIS_NAME)
    env_state = reset_fn(key_envs)
    
    obs_shape = jax.tree_util.tree_map(lambda x: x.shape[2:], env_state.obs)
    
    # Create networks
    normalize = lambda x, y: x
    if normalize_observations:
        normalize = running_statistics.normalize
    
    # Extract hidden layer sizes from network_factory if available
    policy_hidden_sizes = (256, 256)
    value_hidden_sizes = (256, 256)
    activation_fn = linen.swish  # Default brax activation
    if hasattr(network_factory, 'keywords'):
        policy_hidden_sizes = network_factory.keywords.get('policy_hidden_layer_sizes', (256, 256))
        value_hidden_sizes = network_factory.keywords.get('value_hidden_layer_sizes', (256, 256))
        activation_fn = network_factory.keywords.get('activation', linen.swish)
    
    # Use custom network factory with dormant detection when tracking is enabled
    if track_dormant or use_redo:
        # Use our custom networks with dormant neuron detection
        print(f'[NETWORK DEBUG] Using my_brax_networks.make_ppo_networks for dormant tracking', flush=True)
        print(f'[NETWORK DEBUG] policy_hidden_sizes: {policy_hidden_sizes}, value_hidden_sizes: {value_hidden_sizes}', flush=True)
        ppo_network = my_brax_networks.make_ppo_networks(
            obs_shape, wrapped_env.action_size,
            preprocess_observations_fn=normalize,
            policy_hidden_layer_sizes=policy_hidden_sizes,
            value_hidden_layer_sizes=value_hidden_sizes,
            activation=activation_fn,
            policy_min_std=policy_min_std,
            distribution_type=distribution_type,
            # The threshold is applied inside the forward pass that scores the
            # neurons, so it has to be bound on the network. ReDo uses it to
            # decide what to recycle; --track_dormant uses it to decide what to
            # report, and the two must be the same number.
            redo_tau=redo_tau if use_redo else dormant_tau,
        )
        print(f'[NETWORK DEBUG] ppo_network.policy_network type: {type(ppo_network.policy_network)}', flush=True)
        print(f'[NETWORK DEBUG] has apply_with_activation_stats: {hasattr(ppo_network.policy_network, "apply_with_activation_stats")}', flush=True)
    else:
        # network_factory is brax's make_ppo_networks, which has no min_std
        # parameter, so a non-default policy_min_std would be silently dropped
        # here. It once was: a whole std-floor sweep ran without --track_dormant
        # and returned four bit-identical results, which read as "the floor does
        # nothing" rather than "the floor was never applied". Fail loudly.
        if policy_min_std != 0.001:
            raise ValueError(
                f'policy_min_std={policy_min_std} requires the my_brax network '
                'path: pass --track_dormant (or --use_redo). brax\'s own '
                'make_ppo_networks takes no min_std and would ignore it.')
        ppo_network = network_factory(
            obs_shape, wrapped_env.action_size, preprocess_observations_fn=normalize
        )
    make_policy = ppo_networks.make_inference_fn(ppo_network)
    
    # Optimizer
    if max_grad_norm is not None:
        optimizer = optax.chain(
            optax.clip_by_global_norm(max_grad_norm),
            optax.adam(learning_rate=learning_rate),
        )
    else:
        optimizer = optax.adam(learning_rate=learning_rate)
    
    # Wrap optimizer with Trac if enabled
    if use_trac:
        from trac_optimizer.experimental.jax.trac import start_trac
        optimizer = start_trac(optimizer)
        logging.info('Using TRAC optimizer wrapper')

    # Outermost, so it also guards TRAC's own state.
    if nan_guard and nan_guard > 0:
        optimizer = optax.apply_if_finite(optimizer, nan_guard)
        logging.info('NaN guard active: skipping non-finite updates '
                     f'(max_consecutive_errors={nan_guard})')
    
    # Loss function
    loss_fn = functools.partial(
        ppo_losses.compute_ppo_loss,
        ppo_network=ppo_network,
        entropy_cost=entropy_cost,
        discounting=discounting,
        reward_scaling=reward_scaling,
        gae_lambda=gae_lambda,
        clipping_epsilon=clipping_epsilon,
        normalize_advantage=True,
    )
    
    loss_and_pgrad_fn = gradients.loss_and_pgrad(
        loss_fn, pmap_axis_name=_PMAP_AXIS_NAME, has_aux=True
    )

    if use_cchain:
        # Ported from source/studies/brax/my_brax/ppo_train.py, which carries the commentary on why
        # each piece is shaped this way. Kept on its own code path rather than
        # branched inside minibatch_step/sgd_step, so a plain PPO, TRAC or ReDo
        # run traces exactly the code it did before.
        def chain_loss_fn(params, normalizer_params, data, reg_obs,
                          ref_policy_params, chain_coef, key):
            total_loss, metrics = loss_fn(params, normalizer_params, data, key)
            reg_loss = my_brax_cchain.churn_loss(
                ppo_network, normalizer_params, params.policy, ref_policy_params,
                reg_obs)
            metrics['chain_reg_loss'] = reg_loss
            metrics['chain_coef'] = chain_coef
            return total_loss + chain_coef * reg_loss, metrics

        chain_loss_and_pgrad_fn = gradients.loss_and_pgrad(
            chain_loss_fn, pmap_axis_name=_PMAP_AXIS_NAME, has_aux=True
        )

    # Training step functions
    def minibatch_step(carry, data, normalizer_params):
        # `prev_params` is carried so that after the scans finish it holds the
        # parameters from exactly ONE gradient step back -- the reference the
        # churn column is measured against, for every method and not just
        # C-CHAIN (which gets the same thing from chain_state.ref_policy_params).
        # It is threaded rather than recomputed because the alternative is
        # stacking a parameter pytree per step as a scan output.
        optimizer_state, params, prev_params, key = carry
        key, key_loss = jax.random.split(key)
        (_, metrics), grads = loss_and_pgrad_fn(params, normalizer_params, data, key_loss)
        # Pass params as third argument for Trac optimizer compatibility
        params_update, optimizer_state = optimizer.update(grads, optimizer_state, params)
        new_params = optax.apply_updates(params, params_update)
        return (optimizer_state, new_params, params, key), metrics
    
    def sgd_step(carry, unused_t, data, normalizer_params):
        optimizer_state, params, prev_params, key = carry
        key, key_perm, key_grad = jax.random.split(key, 3)
        
        def convert_data(x):
            x = jax.random.permutation(key_perm, x)
            x = jnp.reshape(x, (num_minibatches, -1) + x.shape[1:])
            return x
        
        shuffled_data = jax.tree_util.tree_map(convert_data, data)
        (optimizer_state, params, prev_params, _), metrics = jax.lax.scan(
            functools.partial(minibatch_step, normalizer_params=normalizer_params),
            (optimizer_state, params, prev_params, key_grad),
            shuffled_data,
            length=num_minibatches,
        )
        return (optimizer_state, params, prev_params, key), metrics

    def minibatch_step_chain(carry, xs, normalizer_params):
        """minibatch_step plus the C-CHAIN churn term.

        `xs` is (training minibatch, regularisation observations); the two come
        from independent shuffles of the same rollout, which is what makes the
        churn term suppress off-diagonal NTK entries rather than just re-fitting
        the batch the policy loss already saw.
        """
        optimizer_state, params, chain_state, key = carry
        data, reg_obs = xs
        key, key_loss = jax.random.split(key)

        # The regulariser needs a reference that is genuinely one step behind,
        # so it stays off until two updates have happened. Folding it into the
        # coefficient rather than branching keeps the traced graph one shape.
        active = jnp.where(chain_state.grad_steps >= 2, 1.0, 0.0)
        chain_coef = chain_state.coef * active

        (_, metrics), grads = chain_loss_and_pgrad_fn(
            params, normalizer_params, data, reg_obs,
            chain_state.ref_policy_params, chain_coef, key_loss
        )

        # Reference for the *next* step: the params this update started from.
        next_ref_policy_params = params.policy

        params_update, optimizer_state = optimizer.update(grads, optimizer_state, params)
        params = optax.apply_updates(params, params_update)

        chain_state = chain_state.replace(
            ref_policy_params=next_ref_policy_params,
            grad_steps=chain_state.grad_steps + 1,
        )
        # Reported (and fed to the controller) as 0 while inactive, so the
        # controller does not calibrate against a churn value never applied.
        metrics['chain_reg_loss'] = metrics['chain_reg_loss'] * active

        return (optimizer_state, params, chain_state, key), metrics

    def sgd_step_chain(carry, unused_t, data, normalizer_params):
        optimizer_state, params, chain_state, key = carry
        key, key_perm, key_perm_reg, key_grad = jax.random.split(key, 4)

        def convert_data(x):
            x = jax.random.permutation(key_perm, x)
            return jnp.reshape(x, (num_minibatches, -1) + x.shape[1:])

        # A second, independent shuffle. Only the observations are permuted: the
        # churn term needs nothing else from the transition.
        def convert_obs(x):
            x = jax.random.permutation(key_perm_reg, x)
            return jnp.reshape(x, (num_minibatches, -1) + x.shape[1:])

        shuffled_data = jax.tree_util.tree_map(convert_data, data)
        shuffled_reg_obs = jax.tree_util.tree_map(convert_obs, data.observation)

        (optimizer_state, params, chain_state, _), metrics = jax.lax.scan(
            functools.partial(minibatch_step_chain, normalizer_params=normalizer_params),
            (optimizer_state, params, chain_state, key_grad),
            (shuffled_data, shuffled_reg_obs),
            length=num_minibatches,
        )

        return (optimizer_state, params, chain_state, key), metrics

    def training_step(carry, unused_t, env, reset_fn):
        training_state, state, key = carry
        key_sgd, key_generate_unroll, new_key = jax.random.split(key, 3)
        
        policy = make_policy((
            training_state.normalizer_params,
            training_state.params.policy,
            training_state.params.value,
        ))
        
        def f(carry, unused_t):
            current_state, current_key = carry
            current_key, next_key = jax.random.split(current_key)
            next_state, data = acting.generate_unroll(
                env,
                current_state,
                policy,
                current_key,
                unroll_length,
                extra_fields=('truncation', 'episode_metrics', 'episode_done'),
            )
            return (next_state, next_key), data
        
        (state, _), data = jax.lax.scan(
            f,
            (state, key_generate_unroll),
            (),
            length=batch_size * num_minibatches // num_envs,
        )
        
        data = jax.tree_util.tree_map(lambda x: jnp.swapaxes(x, 1, 2), data)
        data = jax.tree_util.tree_map(lambda x: jnp.reshape(x, (-1,) + x.shape[2:]), data)

        # Guard the observation normalizer against a single blown-up state.
        #
        # This is the fix for the permanent collapses that killed 4 of 6 ppo
        # seeds (and the matching trials of trac/redo/cchain, which share a
        # seed's fate). Diagnosis, from the task checkpoints of ppo trial_1:
        # after the collapse the policy and value weights are HEALTHY
        # (max|w| 1.51, no NaN), while normalizer_params.mean has gone from 6.45
        # to 4.79e+07 and std to brax's std_max_value clamp of 1e6. The network
        # is fine and is being fed (obs - 4.8e7)/1e6, i.e. nothing. By sub-task
        # 10 the state is NaN outright, which is where that run's reward stops
        # being ~2-4 and becomes exactly 0.00.
        #
        # The trigger is a MuJoCo blow-up during training on a LOW-FRICTION
        # (x0.2) sub-task -- every collapse in the tree is at a x0.2 sub-task and
        # at epoch 1, right after the env is rebuilt. `running_statistics.update`
        # clamps std via std_max_value but leaves the MEAN unbounded, so one
        # outlier permanently poisons a running estimate whose count is already
        # ~1e8. Nothing can wash it out, which is why the failure is permanent
        # while the weights look untouched.
        #
        # Clipping here rather than at the policy input because this is where the
        # damage becomes persistent: `data.observation` feeds both the normalizer
        # state and the PPO loss. `nan_to_num` first so a NaN state is neutralised
        # rather than clipped to +limit.
        # `Transition` is a NamedTuple, so this is `._replace`, not `.replace`.
        # `next_observation` is clipped too: compute_ppo_loss reads it for the
        # value bootstrap, so leaving it raw would let the blown-up state back in
        # through the critic target even with `observation` clean.
        if obs_clip is not None:
            _sane = lambda x: jnp.clip(
                jnp.nan_to_num(x, nan=0.0, posinf=obs_clip, neginf=-obs_clip),
                -obs_clip, obs_clip)
            data = data._replace(
                observation=_sane(data.observation),
                next_observation=_sane(data.next_observation),
                # CheetahRun's per-step reward is in [0, 1]; a non-finite one can
                # only come from a diverged sim, and it would make the loss NaN
                # regardless of how clean the observations are.
                reward=jnp.nan_to_num(data.reward, nan=0.0,
                                      posinf=0.0, neginf=0.0),
            )

        normalizer_params = running_statistics.update(
            training_state.normalizer_params,
            data.observation,
            pmap_axis_name=_PMAP_AXIS_NAME,
        )

        if use_cchain:
            key_sgd, key_churn = jax.random.split(key_sgd)
            (optimizer_state, params, chain_state, _), metrics = jax.lax.scan(
                functools.partial(
                    sgd_step_chain, data=data, normalizer_params=normalizer_params
                ),
                (
                    training_state.optimizer_state,
                    training_state.params,
                    training_state.chain_state,
                    key_sgd,
                ),
                (),
                length=num_updates_per_batch,
            )

            # One controller tick per PPO iteration, from the last minibatch of
            # the last epoch -- metrics are stacked
            # (num_updates_per_batch, num_minibatches), so [-1, -1] picks it out.
            chain_state = my_brax_cchain.update_coefficient(
                chain_state,
                metrics['policy_loss'][-1, -1],
                metrics['chain_reg_loss'][-1, -1],
                chain_target_rel_scale,
                chain_warmup_iterations,
                chain_coef_window,
            )

            # Diagnostic only, on a third independent shuffle and outside the
            # gradient: the paper's reported "policy churn".
            churn_obs = jax.tree_util.tree_map(
                lambda x: jax.random.permutation(key_churn, x)[:batch_size],
                data.observation,
            )
            metrics = dict(metrics)
            metrics['policy_churn'] = my_brax_cchain.churn_loss(
                ppo_network, normalizer_params, params.policy,
                chain_state.ref_policy_params, churn_obs)
            metrics['chain_coef'] = chain_state.coef
        else:
            chain_state = training_state.chain_state
            (optimizer_state, params, prev_params, _), metrics = jax.lax.scan(
                functools.partial(sgd_step, data=data, normalizer_params=normalizer_params),
                (training_state.optimizer_state, training_state.params,
                 training_state.params, key_sgd),
                (),
                length=num_updates_per_batch,
            )

        # POLICY CHURN, FOR EVERY METHOD -- which is what the reference does.
        # `crl_run_ppo_dmc.py:318` logs the identical quantity for vanilla PPO
        # that `crl_run_ppo_c_chain_dmc.py:339` logs for C-CHAIN, so this is the
        # published practice rather than an extension of it: mean squared
        # difference of action means against the policy ONE GRADIENT STEP back.
        #
        # C-CHAIN takes its reference from chain_state, which is maintained by
        # the regulariser anyway; every other method takes it from `prev_params`
        # threaded out of the minibatch scan. Same delta, same estimator.
        #
        # A DETERMINISTIC slice of the rollout, not a fresh shuffle: splitting a
        # key here would consume the trainer's RNG and change every non-C-CHAIN
        # run, and a diagnostic may not do that. C-CHAIN keeps its own shuffled
        # sample above because that split is already part of its stream.
        if not use_cchain:
            churn_obs = jax.tree_util.tree_map(lambda x: x[:batch_size],
                                               data.observation)
            metrics = dict(metrics)
            metrics['policy_churn'] = my_brax_cchain.churn_loss(
                ppo_network, normalizer_params, params.policy,
                prev_params.policy, churn_obs)

        new_training_state = TrainingState(
            optimizer_state=optimizer_state,
            params=params,
            normalizer_params=normalizer_params,
            env_steps=training_state.env_steps + env_step_per_training_step,
            chain_state=chain_state,
        )

        return (new_training_state, state, new_key), metrics
    
    # Initialize training state
    init_params = ppo_losses.PPONetworkParams(
        policy=ppo_network.policy_network.init(key_policy),
        value=ppo_network.value_network.init(key_value),
    )
    
    obs_shape_spec = jax.tree_util.tree_map(
        lambda x: specs.Array(x.shape[-1:], jnp.dtype('float32')),
        env_state.obs
    )
    
    training_state = TrainingState(
        optimizer_state=optimizer.init(init_params),
        params=init_params,
        normalizer_params=running_statistics.init_state(obs_shape_spec),
        env_steps=jnp.array(0),
        chain_state=(
            my_brax_cchain.init_chain_state(
                init_params.policy, window=chain_coef_window)
            if use_cchain else None
        ),
    )
    
    training_state = jax.device_put_replicated(
        training_state, jax.local_devices()[:local_devices_to_use]
    )
    
    # Track global progress
    global_step = 0
    # brax accumulates env_steps inside the training state, where JAX's default
    # 32-bit integers wrap at 2**31 -- reached at sub-task 13 of a 30-sub-task run
    # (153.6M steps each). Casting to a Python int afterwards is too late, so the
    # raw counter is differenced and the wrap added back. Measured: task 12 read
    # 1,996,800,000 and task 13 read -2,144,567,296.
    _prev_raw_env_steps = 0
    training_walltime = 0
    all_metrics = {}
    
    # Initialize dormant neuron trackers if enabled
    policy_dormant_tracker = None
    value_dormant_tracker = None
    if track_dormant:
        # Use the hidden sizes extracted earlier (same as used for network creation)
        policy_sizes = list(policy_hidden_sizes)
        value_sizes = list(value_hidden_sizes)
        
        logging.info(f'Initializing dormant neuron trackers: policy={policy_sizes}, value={value_sizes}')
        print(f'[DORMANT DEBUG] Tracker initialized with policy={policy_sizes}, value={value_sizes}', flush=True)
        policy_dormant_tracker = DormantNeuronTracker(policy_sizes)
        value_dormant_tracker = DormantNeuronTracker(value_sizes)
    
    # Main continual learning loop
    for task_idx, multiplier in enumerate(task_multipliers):
        logging.info(f'Starting task {task_idx + 1}/{num_tasks}, multiplier={multiplier:.2f}')
        
        # Create new environments for this task
        train_env, eval_env = env_factory(multiplier)
        wrapped_train_env = wrap_env(train_env, num_envs, key_env)
        wrapped_eval_env = wrap_env(eval_env, num_eval_envs, eval_key)
        
        # Re-tune the C-CHAIN coefficient from scratch for the new task. The
        # controller sets `coef` from the running ratio of policy loss to churn
        # loss, and both scales move when the physics does; carrying the old
        # window across the boundary would hold the coefficient at the previous
        # task's calibration for `chain_coef_window` iterations. The networks
        # and the optimizer state are untouched -- only the controller's history
        # is cleared. Skipped at task 0, where there is nothing to forget.
        if use_cchain and cchain_reset_on_switch and task_idx > 0:
            training_state = training_state.replace(
                chain_state=jax.pmap(my_brax_cchain.reset_for_new_task)(
                    training_state.chain_state
                )
            )
            print(f'  Reset C-CHAIN coefficient controller for task {task_idx}',
                  flush=True)

        # The oracle arm: rebuild the network, the optimizer and the observation
        # normalizer from scratch, so this sub-task is trained by an agent that
        # has never seen the earlier ones. `env_steps` is carried rather than
        # zeroed -- it is the x-axis of every trace and the budget accounting,
        # not part of what the agent knows. Keyed off task_idx so each sub-task
        # gets its own initialisation rather than repeating task 0's.
        if reset_agent_per_task and task_idx > 0:
            reset_key = jax.random.fold_in(global_key, 10_000 + task_idx)
            reset_policy_key, reset_value_key = jax.random.split(reset_key)
            fresh_params = ppo_losses.PPONetworkParams(
                policy=ppo_network.policy_network.init(reset_policy_key),
                value=ppo_network.value_network.init(reset_value_key),
            )
            fresh_state = TrainingState(
                optimizer_state=optimizer.init(fresh_params),
                params=fresh_params,
                normalizer_params=running_statistics.init_state(obs_shape_spec),
                env_steps=_unpmap(training_state.env_steps),
                chain_state=(
                    my_brax_cchain.init_chain_state(
                        fresh_params.policy, window=chain_coef_window)
                    if use_cchain else None
                ),
            )
            training_state = jax.device_put_replicated(
                fresh_state, jax.local_devices()[:local_devices_to_use])
            print(f'  ORACLE: re-initialised network, optimizer and normalizer '
                  f'for task {task_idx}', flush=True)

        # Reset environment state (but keep training state!)
        key_envs = jax.random.split(jax.random.fold_in(key_env, task_idx), num_envs // process_count)
        key_envs = jnp.reshape(key_envs, (local_devices_to_use, -1) + key_envs.shape[1:])
        reset_fn = jax.pmap(wrapped_train_env.reset, axis_name=_PMAP_AXIS_NAME)
        env_state = reset_fn(key_envs)
        
        # Create training epoch function for this environment
        def training_epoch(training_state, state, key):
            training_step_fn = functools.partial(
                training_step, env=wrapped_train_env, reset_fn=reset_fn
            )
            (training_state, state, _), loss_metrics = jax.lax.scan(
                training_step_fn,
                (training_state, state, key),
                (),
                length=num_training_steps_per_epoch,
            )
            loss_metrics = jax.tree_util.tree_map(jnp.mean, loss_metrics)
            return training_state, state, loss_metrics
        
        training_epoch_pmap = jax.pmap(
            training_epoch,
            axis_name=_PMAP_AXIS_NAME,
            donate_argnums=(0, 1),
        )
        
        # Evaluator for this task
        evaluator = acting.Evaluator(
            wrapped_eval_env,
            functools.partial(make_policy, deterministic=True),
            num_eval_envs=num_eval_envs,
            episode_length=episode_length,
            action_repeat=action_repeat,
            key=jax.random.fold_in(eval_key, task_idx),
        )
        
        # Run initial eval for this task
        if process_id == 0 and num_evals_per_task > 1:
            params = _unpmap((
                training_state.normalizer_params,
                training_state.params.policy,
                training_state.params.value,
            ))
            metrics = evaluator.run_evaluation(params, training_metrics={})
            metrics['task'] = task_idx
            metrics['multiplier'] = multiplier
            # Pass generation number (0-indexed): task_idx * 100 + 0 for initial eval
            metrics['generation'] = task_idx * num_evals_per_task
            progress_fn(global_step, task_idx, multiplier, metrics)
        
        # Training loop for this task - do num_training_epochs (= num_evals_per_task = 100)
        for it in range(num_training_epochs):
            t = time.time()
            
            epoch_key, local_key = jax.random.split(local_key)
            epoch_keys = jax.random.split(epoch_key, local_devices_to_use)
            
            training_state, env_state, training_metrics = training_epoch_pmap(
                training_state, env_state, epoch_keys
            )
            
            _raw_env_steps = int(_unpmap(training_state.env_steps))
            _delta = _raw_env_steps - _prev_raw_env_steps
            if _delta < 0:                      # int32 wrapped since the last read
                _delta += 2 ** 32
            global_step += _delta
            _prev_raw_env_steps = _raw_env_steps
            current_step = global_step
            
            jax.tree_util.tree_map(lambda x: x.block_until_ready(), training_metrics)
            
            epoch_training_time = time.time() - t
            training_walltime += epoch_training_time
            
            if process_id == 0:
                params = _unpmap((
                    training_state.normalizer_params,
                    training_state.params.policy,
                    training_state.params.value,
                ))
                
                metrics = evaluator.run_evaluation(params, dict(training_metrics))
                metrics['training/walltime'] = training_walltime
                metrics['training/sps'] = (
                    num_training_steps_per_epoch * env_step_per_training_step
                ) / epoch_training_time
                metrics['task'] = task_idx
                metrics['multiplier'] = multiplier
                # Pass generation number: task_idx * 100 + (it + 1)
                # it+1 because we've completed epoch it (0-indexed)
                generation = task_idx * num_evals_per_task + it + 1
                metrics['generation'] = generation
                
                # Track dormant neurons if enabled
                if track_dormant and policy_dormant_tracker is not None:
                    if it == 0:
                        print(f'[DORMANT DEBUG] Computing dormant stats, policy_network type: {type(ppo_network.policy_network)}', flush=True)
                        print(f'[DORMANT DEBUG] has apply_with_activation_stats: {hasattr(ppo_network.policy_network, "apply_with_activation_stats")}', flush=True)
                    try:
                        # Get sample observations for dormant detection
                        sample_obs = _unpmap(env_state.obs)
                        unpmap_params = _unpmap(training_state.params)
                        unpmap_normalizer = _unpmap(training_state.normalizer_params)
                        
                        # Compute dormant masks and activation stats
                        policy_masks, value_masks, policy_frac, value_frac, policy_act_stats, value_act_stats = _compute_dormant_masks(
                            ppo_network,
                            unpmap_normalizer,
                            unpmap_params,
                            sample_obs,
                            tau=dormant_tau,
                        )
                        
                        # Update trackers
                        if policy_masks:
                            policy_dormant_tracker.update(policy_masks)
                            policy_stats = policy_dormant_tracker.get_stats()
                            for k, v in policy_stats.items():
                                metrics[f'policy/{k}'] = v
                        
                        if value_masks:
                            value_dormant_tracker.update(value_masks)
                            value_stats = value_dormant_tracker.get_stats()
                            for k, v in value_stats.items():
                                metrics[f'value/{k}'] = v
                        
                        # Also log raw dormant fractions
                        metrics['policy/dormant_fraction'] = policy_frac
                        metrics['value/dormant_fraction'] = value_frac
                        
                        # Log activation statistics for debugging
                        for k, v in policy_act_stats.items():
                            metrics[f'policy/activation/{k}'] = v
                        for k, v in value_act_stats.items():
                            metrics[f'value/activation/{k}'] = v
                        
                        if it == 0:
                            print(f'[DORMANT DEBUG] Successfully computed dormant stats: policy_frac={policy_frac:.4f}, value_frac={value_frac:.4f}', flush=True)
                            print(f'[DORMANT DEBUG] policy_act_stats keys: {list(policy_act_stats.keys())}', flush=True)
                        
                    except Exception as e:
                        import traceback
                        print(f'[DORMANT DEBUG] Failed to compute dormant stats: {e}', flush=True)
                        print(traceback.format_exc(), flush=True)
                        logging.warning(f'Failed to compute dormant stats: {e}')
                        logging.warning(traceback.format_exc())
                
                # Weight statistics, the third plasticity signal alongside
                # dormancy and churn. Computed here rather than in the trainer
                # because progress_fn is handed metrics, not parameters, and
                # `params` was unpmapped for the evaluation just above -- the
                # same reason the dormant stats are computed here. Same
                # function the NE ant trainers call, so the columns mean the
                # same thing on both sides of the comparison.
                # NTK effective rank -- C-CHAIN's own plasticity indicator, and
                # the CAUSE its argument assigns to churn (rank collapse ->
                # correlated gradients -> churn). Computed on the CURRENT
                # rollout's observations, which is what this trainer's dormancy
                # already uses; the NE ant trainers use their frozen probe
                # batch instead, so the two are on the same network but not the
                # same states. That asymmetry predates this column -- see
                # source/metrics/ntk.py.
                _ntk_obs = _unpmap(env_state.obs)
                if isinstance(_ntk_obs, dict):
                    _ntk_obs = _ntk_obs.get('state', list(_ntk_obs.values())[0])
                _ntk_obs = _ntk_obs.reshape(-1, _ntk_obs.shape[-1])
                metrics.update(ntk_metrics.ntk_rank_stats(
                    lambda p, o: ppo_network.policy_network.apply(params[0], p, o),
                    params[1], _ntk_obs, prefix='policy_ntk'))

                metrics.update(weight_stats(params[1], prefix='policy_weight'))
                metrics.update(weight_stats(params[2], prefix='value_weight'))

                progress_fn(global_step, task_idx, multiplier, metrics)
                all_metrics = metrics
                
                # Save generation checkpoint at the last generation before task switch
                # (e.g., gen 99, 199, 299, etc. for num_evals_per_task=100)
                # This is when it == num_evals_per_task - 1, so generation ends in 99, 199, etc.
                if it == num_training_epochs - 1:
                    generation_checkpoint_fn(generation, {
                        'normalizer_params': params[0],
                        'policy_params': params[1],
                        'value_params': params[2],
                        'task_idx': task_idx,
                        'multiplier': multiplier,
                        'generation': generation,
                        'global_step': global_step,
                    })
                
                # Apply ReDo if enabled and it's time
                global_epoch = task_idx * num_training_epochs + it + 1
                if use_redo and global_epoch % redo_frequency == 0:
                    logging.info(f'Applying ReDo at global epoch {global_epoch}')
                    training_state = _apply_redo(
                        training_state,
                        ppo_network,
                        env_state.obs,
                        local_key,
                        local_devices_to_use,
                        redo_tau,
                        # None => read the widths off the parameters, which
                        # cannot disagree with the network actually in use.
                        policy_hidden_sizes=None,
                        value_hidden_sizes=None,
                    )
        
        # Save checkpoint at end of task
        if process_id == 0:
            task_params = _unpmap((
                training_state.normalizer_params,
                training_state.params.policy,
                training_state.params.value,
            ))
            checkpoint_fn(task_idx, {
                'normalizer_params': task_params[0],
                'policy_params': task_params[1],
                'value_params': task_params[2],
                'task_idx': task_idx,
                'multiplier': multiplier,
                'global_step': global_step,
            })
            
            # Save GIFs at end of task (before switching to next task)
            if gif_callback_fn is not None:
                inference_fn = make_policy(task_params, deterministic=True)
                gif_callback_fn(task_idx, multiplier, inference_fn, train_env, task_params)
        
        logging.info(f'Task {task_idx + 1} complete, global_step={global_step}')
    
    # Return final policy and params.
    #
    # The replication check is advisory here rather than fatal. It compares the
    # per-device copies bitwise, and a policy that has collapsed to zero reward
    # can carry non-finite parameters whose copies stop matching -- which is
    # exactly what happened to PPO seed 42, killing a finished 30-sub-task run
    # after its last checkpoint and before its caller could write
    # training_metrics.json. The run's own failure must not also cost us the
    # record of it: the seeds that collapse are the seeds this experiment is
    # about.
    try:
        pmap.assert_is_replicated(training_state)
    except AssertionError:
        logging.warning(
            'training_state is not replicated across devices at the end of the '
            'run -- continuing anyway. Usually means the policy went non-finite; '
            'check the final rewards before trusting the returned params.')
    final_params = _unpmap((
        training_state.normalizer_params,
        training_state.params.policy,
        training_state.params.value,
    ))
    
    return make_policy, final_params, all_metrics