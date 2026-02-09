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
from my_brax import networks as my_brax_networks

InferenceParams = Tuple[running_statistics.NestedMeanStd, Params]
Metrics = types.Metrics

_PMAP_AXIS_NAME = 'i'


class DormantNeuronTracker:
    """Tracks dormant neurons and their age across training."""
    
    def __init__(self, layer_sizes: List[int]):
        """Initialize tracker with network layer sizes.
        
        Args:
            layer_sizes: List of hidden layer sizes (excluding output layer)
        """
        self.layer_sizes = layer_sizes
        # Age of each neuron: how many consecutive epochs it's been dormant
        # Dictionary: layer_idx -> array of ages per neuron
        self.dormant_ages = {i: np.zeros(size, dtype=np.int32) 
                            for i, size in enumerate(layer_sizes)}
        # Track which neurons are currently dormant
        self.currently_dormant = {i: np.zeros(size, dtype=bool) 
                                  for i, size in enumerate(layer_sizes)}
        # Total epochs tracked
        self.total_epochs = 0
    
    def update(self, dormant_masks: List[np.ndarray]):
        """Update dormant tracking with new dormant masks.
        
        Args:
            dormant_masks: List of boolean arrays, one per layer, 
                          True for dormant neurons
        """
        self.total_epochs += 1
        
        for layer_idx, mask in enumerate(dormant_masks):
            if layer_idx >= len(self.layer_sizes):
                continue
            
            mask_np = np.array(mask, dtype=bool)
            
            # Update ages: increment for dormant neurons, reset for active ones
            self.dormant_ages[layer_idx] = np.where(
                mask_np,
                self.dormant_ages[layer_idx] + 1,  # Increment if dormant
                0  # Reset to 0 if active
            )
            self.currently_dormant[layer_idx] = mask_np
    
    def get_stats(self) -> dict:
        """Get dormant neuron statistics.
        
        Returns:
            Dictionary with dormant neuron metrics
        """
        stats = {}
        
        total_dormant = 0
        total_neurons = 0
        all_ages = []
        
        for layer_idx, size in enumerate(self.layer_sizes):
            dormant_mask = self.currently_dormant[layer_idx]
            ages = self.dormant_ages[layer_idx]
            
            n_dormant = np.sum(dormant_mask)
            pct_dormant = 100.0 * n_dormant / size if size > 0 else 0.0
            
            stats[f'dormant/layer{layer_idx}/count'] = int(n_dormant)
            stats[f'dormant/layer{layer_idx}/percent'] = float(pct_dormant)
            
            if n_dormant > 0:
                dormant_ages = ages[dormant_mask]
                stats[f'dormant/layer{layer_idx}/age_mean'] = float(np.mean(dormant_ages))
                stats[f'dormant/layer{layer_idx}/age_max'] = int(np.max(dormant_ages))
                all_ages.extend(dormant_ages.tolist())
            else:
                stats[f'dormant/layer{layer_idx}/age_mean'] = 0.0
                stats[f'dormant/layer{layer_idx}/age_max'] = 0
            
            total_dormant += n_dormant
            total_neurons += size
        
        # Overall stats
        stats['dormant/total_count'] = int(total_dormant)
        stats['dormant/total_neurons'] = int(total_neurons)
        stats['dormant/total_percent'] = float(100.0 * total_dormant / total_neurons) if total_neurons > 0 else 0.0
        
        if all_ages:
            stats['dormant/age_mean'] = float(np.mean(all_ages))
            stats['dormant/age_max'] = int(np.max(all_ages))
            stats['dormant/age_median'] = float(np.median(all_ages))
        else:
            stats['dormant/age_mean'] = 0.0
            stats['dormant/age_max'] = 0
            stats['dormant/age_median'] = 0.0
        
        return stats


def _compute_dormant_masks(
    ppo_network,
    normalizer_params,
    params,
    sample_obs: jnp.ndarray,
    tau: float = 0.01,
) -> Tuple[List[np.ndarray], List[np.ndarray], float, float, dict, dict]:
    """Compute dormant neuron masks for policy and value networks.
    
    Args:
        ppo_network: PPO network with policy and value networks
        normalizer_params: Normalizer parameters
        params: Network parameters
        sample_obs: Sample observations for forward pass
        tau: Threshold for dormant detection (default 0.01 = 1%)
    
    Returns:
        Tuple of (policy_dormant_masks, value_dormant_masks, policy_frac, value_frac,
                  policy_activation_stats, value_activation_stats)
    """
    policy_masks = []
    value_masks = []
    policy_frac = 0.0
    value_frac = 0.0
    policy_activation_stats = {}
    value_activation_stats = {}
    
    # Flatten sample_obs to 2D if needed
    if isinstance(sample_obs, dict):
        sample_obs = sample_obs.get('state', list(sample_obs.values())[0])
    sample_obs_flat = sample_obs.reshape(-1, sample_obs.shape[-1])
    
    # Policy network
    policy_network = ppo_network.policy_network
    has_activation_stats = hasattr(policy_network, 'apply_with_activation_stats')
    has_dormant_indices = hasattr(policy_network, 'apply_with_dormant_indices')
    logging.debug(f'Policy network has apply_with_activation_stats: {has_activation_stats}')
    logging.debug(f'Policy network has apply_with_dormant_indices: {has_dormant_indices}')
    
    if has_activation_stats:
        try:
            _, policy_frac, dormant_indices, activation_stats = policy_network.apply_with_activation_stats(
                normalizer_params, params.policy, sample_obs_flat
            )
            policy_frac = float(policy_frac)
            # Convert indices to boolean masks
            for layer_indices in dormant_indices:
                layer_indices_np = np.array(layer_indices)
                # Indices are -1 for non-dormant, >= 0 for dormant
                mask = layer_indices_np >= 0
                policy_masks.append(mask)
            
            # Extract activation stats per layer
            for i, stats in enumerate(activation_stats):
                policy_activation_stats[f'layer{i}/mean'] = float(stats['layer_mean'])
                policy_activation_stats[f'layer{i}/std'] = float(stats['layer_std'])
                policy_activation_stats[f'layer{i}/min_neuron_mean'] = float(stats['min_neuron_mean'])
                policy_activation_stats[f'layer{i}/max_neuron_mean'] = float(stats['max_neuron_mean'])
                
        except Exception as e:
            logging.warning(f'Failed to compute policy dormant masks: {e}')
    elif hasattr(policy_network, 'apply_with_dormant_indices'):
        try:
            _, policy_frac, dormant_indices = policy_network.apply_with_dormant_indices(
                normalizer_params, params.policy, sample_obs_flat
            )
            policy_frac = float(policy_frac)
            for layer_indices in dormant_indices:
                layer_indices_np = np.array(layer_indices)
                mask = layer_indices_np >= 0
                policy_masks.append(mask)
        except Exception as e:
            logging.warning(f'Failed to compute policy dormant masks: {e}')
    
    # Value network
    value_network = ppo_network.value_network
    if hasattr(value_network, 'apply_with_activation_stats'):
        try:
            _, value_frac, dormant_indices, activation_stats = value_network.apply_with_activation_stats(
                normalizer_params, params.value, sample_obs_flat
            )
            value_frac = float(value_frac)
            # Convert indices to boolean masks
            for layer_indices in dormant_indices:
                layer_indices_np = np.array(layer_indices)
                mask = layer_indices_np >= 0
                value_masks.append(mask)
            
            # Extract activation stats per layer
            for i, stats in enumerate(activation_stats):
                value_activation_stats[f'layer{i}/mean'] = float(stats['layer_mean'])
                value_activation_stats[f'layer{i}/std'] = float(stats['layer_std'])
                value_activation_stats[f'layer{i}/min_neuron_mean'] = float(stats['min_neuron_mean'])
                value_activation_stats[f'layer{i}/max_neuron_mean'] = float(stats['max_neuron_mean'])
                
        except Exception as e:
            logging.warning(f'Failed to compute value dormant masks: {e}')
    elif hasattr(value_network, 'apply_with_dormant_indices'):
        try:
            _, value_frac, dormant_indices = value_network.apply_with_dormant_indices(
                normalizer_params, params.value, sample_obs_flat
            )
            value_frac = float(value_frac)
            for layer_indices in dormant_indices:
                layer_indices_np = np.array(layer_indices)
                mask = layer_indices_np >= 0
                value_masks.append(mask)
        except Exception as e:
            logging.warning(f'Failed to compute value dormant masks: {e}')
    
    return policy_masks, value_masks, policy_frac, value_frac, policy_activation_stats, value_activation_stats


@flax.struct.dataclass
class TrainingState:
    """Contains training state for the learner."""
    optimizer_state: optax.OptState
    params: ppo_losses.PPONetworkParams
    normalizer_params: running_statistics.RunningStatisticsState
    env_steps: jnp.ndarray


def _unpmap(v):
    return jax.tree_util.tree_map(lambda x: x[0], v)


def _apply_redo(
    training_state: TrainingState,
    ppo_network,
    sample_obs,
    local_key,
    local_devices_to_use: int,
    redo_tau: float = 0.01,
) -> TrainingState:
    """
    Apply ReDo (Reinitializing Dormant Neurons) to the policy and value networks.
    
    This function:
    1. Computes dormant neurons by running a forward pass on sample observations
    2. Reinitializes the incoming weights for dormant neurons
    3. Zeros out the outgoing weights from dormant neurons
    
    Args:
        training_state: Current training state with network parameters
        ppo_network: PPO network with policy and value networks
        sample_obs: Sample observations to compute activations
        local_key: Random key for reinitialization
        local_devices_to_use: Number of local devices
        redo_tau: Threshold for dormant neuron detection
        
    Returns:
        Updated training state with ReDo applied
    """
    # Unpmap the params to work with them
    params = _unpmap(training_state.params)
    normalizer_params = _unpmap(training_state.normalizer_params)
    
    # Get sample observations (flatten if needed)
    sample_obs_flat = _unpmap(sample_obs)
    if isinstance(sample_obs_flat, dict):
        sample_obs_flat = sample_obs_flat.get('state', list(sample_obs_flat.values())[0])
    # Flatten batch dimensions
    sample_obs_flat = sample_obs_flat.reshape(-1, sample_obs_flat.shape[-1])
    
    # Split keys for policy and value
    local_key, policy_rng, value_rng = jax.random.split(local_key, 3)
    
    # Get default hidden layer sizes (matches GA MLPPolicy for fair comparison)
    policy_hidden_sizes = [512, 256, 128]
    value_hidden_sizes = [512, 256, 128]
    
    # Apply ReDo to policy network if it supports dormant detection
    policy_network = ppo_network.policy_network
    if hasattr(policy_network, 'apply_with_dormant_indices'):
        try:
            # Get dormant indices by running forward pass
            _, _, policy_dormant_indices = policy_network.apply_with_dormant_indices(
                normalizer_params, params.policy, sample_obs_flat
            )
            
            # Apply ReDo to policy params
            new_policy_params = my_brax_networks.apply_redo_to_params(
                params.policy,
                policy_hidden_sizes,
                policy_dormant_indices,
                rng=policy_rng,
            )
            
            # Count how many neurons were reinitialized
            n_reinit = sum(
                jnp.sum(jnp.array(idx) >= 0) for idx in policy_dormant_indices
            )
            logging.info(f'ReDo: Reinitialized {n_reinit} dormant policy neurons')
        except Exception as e:
            logging.warning(f'Failed to apply ReDo to policy network: {e}')
            new_policy_params = params.policy
    else:
        new_policy_params = params.policy
    
    # Apply ReDo to value network if it supports dormant detection
    value_network = ppo_network.value_network
    if hasattr(value_network, 'apply_with_dormant_indices'):
        try:
            # Get dormant indices by running forward pass
            _, _, value_dormant_indices = value_network.apply_with_dormant_indices(
                normalizer_params, params.value, sample_obs_flat
            )
            
            # Apply ReDo to value params
            new_value_params = my_brax_networks.apply_redo_to_params(
                params.value,
                value_hidden_sizes,
                value_dormant_indices,
                rng=value_rng,
            )
            
            # Count how many neurons were reinitialized
            n_reinit = sum(
                jnp.sum(jnp.array(idx) >= 0) for idx in value_dormant_indices
            )
            logging.info(f'ReDo: Reinitialized {n_reinit} dormant value neurons')
        except Exception as e:
            logging.warning(f'Failed to apply ReDo to value network: {e}')
            new_value_params = params.value
    else:
        new_value_params = params.value
    
    # Create new params with ReDo applied
    new_params = ppo_losses.PPONetworkParams(
        policy=new_policy_params,
        value=new_value_params,
    )
    
    # Replicate back to devices
    new_params = jax.device_put_replicated(
        new_params, jax.local_devices()[:local_devices_to_use]
    )
    
    # Also need to replicate the optimizer state if we want to reset it for dormant neurons
    # For now, we keep the optimizer state as-is (the paper suggests this is fine)
    
    # Create new training state
    new_training_state = TrainingState(
        optimizer_state=training_state.optimizer_state,
        params=new_params,
        normalizer_params=training_state.normalizer_params,
        env_steps=training_state.env_steps,
    )
    
    return new_training_state


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
    network_factory: types.NetworkFactory[ppo_networks.PPONetworks] = ppo_networks.make_ppo_networks,
    seed: int = 0,
    num_eval_envs: int = 128,
    num_evals_per_task: int = 5,
    # Trac optimizer
    use_trac: bool = False,
    # ReDo (Reinitializing Dormant Neurons)
    use_redo: bool = False,
    redo_frequency: int = 10,  # Apply ReDo every N epochs
    redo_tau: float = 0.01,  # Threshold for dormant neuron detection (fraction of layer mean)
    # Dormant neuron tracking
    track_dormant: bool = False,  # Track dormant neurons and their age
    dormant_tau: float = 0.01,  # Threshold for dormant neuron detection
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
    # We want num_evals_per_task training epochs per task (each epoch = 1 generation equivalent)
    # The initial eval is just for logging before training starts, doesn't consume budget
    num_training_steps_per_epoch = np.ceil(
        timesteps_per_task / (num_evals_per_task * env_step_per_training_step)
    ).astype(int)
    # Do all num_evals_per_task as training epochs (initial eval is extra, before training)
    num_training_epochs = num_evals_per_task
    
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
        )
        print(f'[NETWORK DEBUG] ppo_network.policy_network type: {type(ppo_network.policy_network)}', flush=True)
        print(f'[NETWORK DEBUG] has apply_with_activation_stats: {hasattr(ppo_network.policy_network, "apply_with_activation_stats")}', flush=True)
    else:
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
    
    # Training step functions
    def minibatch_step(carry, data, normalizer_params):
        optimizer_state, params, key = carry
        key, key_loss = jax.random.split(key)
        (_, metrics), grads = loss_and_pgrad_fn(params, normalizer_params, data, key_loss)
        # Pass params as third argument for Trac optimizer compatibility
        params_update, optimizer_state = optimizer.update(grads, optimizer_state, params)
        params = optax.apply_updates(params, params_update)
        return (optimizer_state, params, key), metrics
    
    def sgd_step(carry, unused_t, data, normalizer_params):
        optimizer_state, params, key = carry
        key, key_perm, key_grad = jax.random.split(key, 3)
        
        def convert_data(x):
            x = jax.random.permutation(key_perm, x)
            x = jnp.reshape(x, (num_minibatches, -1) + x.shape[1:])
            return x
        
        shuffled_data = jax.tree_util.tree_map(convert_data, data)
        (optimizer_state, params, _), metrics = jax.lax.scan(
            functools.partial(minibatch_step, normalizer_params=normalizer_params),
            (optimizer_state, params, key_grad),
            shuffled_data,
            length=num_minibatches,
        )
        return (optimizer_state, params, key), metrics
    
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
        
        normalizer_params = running_statistics.update(
            training_state.normalizer_params,
            data.observation,
            pmap_axis_name=_PMAP_AXIS_NAME,
        )
        
        (optimizer_state, params, _), metrics = jax.lax.scan(
            functools.partial(sgd_step, data=data, normalizer_params=normalizer_params),
            (training_state.optimizer_state, training_state.params, key_sgd),
            (),
            length=num_updates_per_batch,
        )
        
        new_training_state = TrainingState(
            optimizer_state=optimizer_state,
            params=params,
            normalizer_params=normalizer_params,
            env_steps=training_state.env_steps + env_step_per_training_step,
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
    )
    
    training_state = jax.device_put_replicated(
        training_state, jax.local_devices()[:local_devices_to_use]
    )
    
    # Track global progress
    global_step = 0
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
            
            current_step = int(_unpmap(training_state.env_steps))
            global_step = current_step  # Keep track of global steps
            
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
    
    # Return final policy and params
    pmap.assert_is_replicated(training_state)
    final_params = _unpmap((
        training_state.normalizer_params,
        training_state.params.policy,
        training_state.params.value,
    ))
    
    return make_policy, final_params, all_metrics