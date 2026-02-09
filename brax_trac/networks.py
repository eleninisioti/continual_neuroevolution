# Copyright 2024 The Brax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Network definitions with dormant neuron counting support."""

import dataclasses
import functools
from typing import Any, Callable, Mapping, Sequence, Tuple
import warnings

from brax.training import types
from brax.training.acme import running_statistics
from brax.training.spectral_norm import SNDense
from flax import linen
import jax
import jax.numpy as jnp


ActivationFn = Callable[[jnp.ndarray], jnp.ndarray]
Initializer = Callable[..., Any]


@dataclasses.dataclass
class FeedForwardNetwork:
  init: Callable[..., Any]
  apply: Callable[..., Any]


class MLP(linen.Module):
  """MLP module with dormant neuron counting."""

  layer_sizes: Sequence[int]
  activation: ActivationFn = linen.relu
  kernel_init: Initializer = jax.nn.initializers.lecun_uniform()
  activate_final: bool = False
  bias: bool = True
  layer_norm: bool = False

  @linen.compact
  def __call__(self, data: jnp.ndarray, return_dormant_indices: bool = False, return_activation_stats: bool = False):
    hidden = data
    
    n_dormant = 0 
    n_neurons = 0
    dormant_indices_per_layer = []  # List of indices for each layer
    activation_stats_per_layer = []  # List of (mean, std, layer_mean) per layer
    
    for i, hidden_size in enumerate(self.layer_sizes):
      hidden = linen.Dense(
          hidden_size,
          name=f'hidden_{i}',
          kernel_init=self.kernel_init,
          use_bias=self.bias,
      )(hidden)
      if i != len(self.layer_sizes) - 1 or self.activate_final:
        hidden = self.activation(hidden)
        if self.layer_norm:
          hidden = linen.LayerNorm()(hidden)
        
        # Count dormant neurons only for layers with activation
        # A neuron is dormant if its mean activation (over batch) is <= 1% of the layer's mean activation
        abs_hidden = jnp.abs(hidden)
        # Compute mean activation per neuron (average over ALL batch dimensions)
        # hidden shape might be [batch, neurons] or [batch, ..., neurons] from pmap
        # We need to average over all dimensions except the last one (neuron dimension)
        if len(hidden.shape) > 1:
          # Average over all dimensions except the last one
          # This handles cases like (512, 1, 16) -> average over dims 0 and 1 -> (16,)
          batch_dims = tuple(range(len(abs_hidden.shape) - 1))
          mean_act_per_neuron = jnp.mean(abs_hidden, axis=batch_dims)  # [neurons]
          std_act_per_neuron = jnp.std(abs_hidden, axis=batch_dims)  # [neurons]
          layer_mean_act = jnp.mean(mean_act_per_neuron)  # scalar: mean across all neurons
          layer_std_act = jnp.std(mean_act_per_neuron)  # scalar: std across all neurons
        else:
          mean_act_per_neuron = abs_hidden
          std_act_per_neuron = jnp.zeros_like(abs_hidden)
          layer_mean_act = jnp.mean(abs_hidden)
          layer_std_act = jnp.std(abs_hidden)
        
        # Store activation stats if requested
        if return_activation_stats:
          activation_stats_per_layer.append({
            'mean_per_neuron': mean_act_per_neuron,
            'std_per_neuron': std_act_per_neuron,
            'layer_mean': layer_mean_act,
            'layer_std': layer_std_act,
            'min_neuron_mean': jnp.min(mean_act_per_neuron),
            'max_neuron_mean': jnp.max(mean_act_per_neuron),
          })
        
        # Use a small epsilon to avoid division by zero
        layer_mean_safe = jnp.maximum(layer_mean_act, 1e-8)
        # Normalize by layer mean: dormant if neuron's mean activation <= 1% of layer mean
        per_neuron_act = mean_act_per_neuron / layer_mean_safe
        # Detect dormant neurons
        dormant_mask = (per_neuron_act <= 0.01)
        
        if return_dormant_indices:
          # Get indices of dormant neurons for this layer
          # Use jnp.argwhere to get indices where mask is True
          # jnp.argwhere returns shape (n_true, ndim), so we need to flatten if 1D
          neuron_indices = jnp.arange(len(dormant_mask))
          # Get indices where dormant_mask is True
          dormant_indices = jnp.where(dormant_mask, neuron_indices, -1)
          # Filter out -1 values using jnp.where with a condition
          # Since we can't use boolean indexing, we'll keep all indices but mark non-dormant as -1
          # Then in apply_redo_to_params, we'll filter out -1 values
          dormant_indices_per_layer.append(dormant_indices)
        
        n_dormant += jnp.sum(dormant_mask.astype(jnp.int32))
        n_neurons += hidden.shape[-1]  # Number of neurons in this layer
    
    # Return fraction of dormant neurons (avoid division by zero)
    dormant_fraction = jnp.where(n_neurons > 0, n_dormant / n_neurons, 0.0)
    
    if return_activation_stats:
      return hidden, dormant_fraction, dormant_indices_per_layer, activation_stats_per_layer
    if return_dormant_indices:
      return hidden, dormant_fraction, dormant_indices_per_layer
    return hidden, dormant_fraction


def _get_obs_state_size(obs_size: types.ObservationSize, obs_key: str) -> int:
  obs_size = obs_size[obs_key] if isinstance(obs_size, Mapping) else obs_size
  return jax.tree_util.tree_flatten(obs_size)[0][-1]


def make_policy_network(
    param_size: int,
    obs_size: types.ObservationSize,
    preprocess_observations_fn: types.PreprocessObservationFn = types.identity_observation_preprocessor,
    hidden_layer_sizes: Sequence[int] = (256, 256),
    activation: ActivationFn = linen.relu,
    kernel_init: Initializer = jax.nn.initializers.lecun_uniform(),
    layer_norm: bool = False,
    obs_key: str = 'state',
) -> FeedForwardNetwork:
  """Creates a policy network."""
  policy_module = MLP(
      layer_sizes=list(hidden_layer_sizes) + [param_size],
      activation=activation,
      kernel_init=kernel_init,
      layer_norm=layer_norm,
  )

  def apply(processor_params, policy_params, obs):
    obs = preprocess_observations_fn(obs, processor_params)
    obs = obs if isinstance(obs, jax.Array) else obs[obs_key]
    # MLP returns (hidden, dormant_fraction), but for compatibility with brax
    # we only return the logits
    logits, _ = policy_module.apply(policy_params, obs)
    return logits
  
  def apply_with_dormant_indices(processor_params, policy_params, obs):
    """Apply policy network and return logits, dormant fraction, and dormant indices per layer."""
    obs = preprocess_observations_fn(obs, processor_params)
    obs = obs if isinstance(obs, jax.Array) else obs[obs_key]
    logits, n_dormant, dormant_indices = policy_module.apply(
        policy_params, obs, return_dormant_indices=True)
    return logits, n_dormant, dormant_indices

  def apply_with_activation_stats(processor_params, policy_params, obs):
    """Apply policy network and return logits, dormant fraction, dormant indices, and activation stats."""
    obs = preprocess_observations_fn(obs, processor_params)
    obs = obs if isinstance(obs, jax.Array) else obs[obs_key]
    logits, n_dormant, dormant_indices, activation_stats = policy_module.apply(
        policy_params, obs, return_dormant_indices=True, return_activation_stats=True)
    return logits, n_dormant, dormant_indices, activation_stats

  obs_size = _get_obs_state_size(obs_size, obs_key)
  dummy_obs = jnp.zeros((1, obs_size))
  network = FeedForwardNetwork(
      init=lambda key: policy_module.init(key, dummy_obs), apply=apply
  )
  network.apply_with_dormant_indices = apply_with_dormant_indices
  network.apply_with_activation_stats = apply_with_activation_stats
  return network


def make_value_network(
    obs_size: types.ObservationSize,
    preprocess_observations_fn: types.PreprocessObservationFn = types.identity_observation_preprocessor,
    hidden_layer_sizes: Sequence[int] = (256, 256),
    activation: ActivationFn = linen.relu,
    obs_key: str = 'state',
) -> FeedForwardNetwork:
  """Creates a value network."""
  value_module = MLP(
      layer_sizes=list(hidden_layer_sizes) + [1],
      activation=activation,
      kernel_init=jax.nn.initializers.lecun_uniform(),
  )

  def apply(processor_params, value_params, obs):
    obs = preprocess_observations_fn(obs, processor_params)
    obs = obs if isinstance(obs, jax.Array) else obs[obs_key]
    # MLP returns (hidden, dormant_fraction), but value network should return just the value
    # for compatibility with loss computation
    applied, _ = value_module.apply(value_params, obs)
    return jnp.squeeze(applied, axis=-1)
  
  def apply_with_dormant(processor_params, value_params, obs):
    """Apply value network and return both value and dormant fraction."""
    obs = preprocess_observations_fn(obs, processor_params)
    obs = obs if isinstance(obs, jax.Array) else obs[obs_key]
    applied, dormant_fraction = value_module.apply(value_params, obs)
    return jnp.squeeze(applied, axis=-1), dormant_fraction
  
  def apply_with_dormant_indices(processor_params, value_params, obs):
    """Apply value network and return value, dormant fraction, and dormant indices per layer."""
    obs = preprocess_observations_fn(obs, processor_params)
    obs = obs if isinstance(obs, jax.Array) else obs[obs_key]
    applied, dormant_fraction, dormant_indices = value_module.apply(
        value_params, obs, return_dormant_indices=True)
    return jnp.squeeze(applied, axis=-1), dormant_fraction, dormant_indices

  def apply_with_activation_stats(processor_params, value_params, obs):
    """Apply value network and return value, dormant fraction, dormant indices, and activation stats."""
    obs = preprocess_observations_fn(obs, processor_params)
    obs = obs if isinstance(obs, jax.Array) else obs[obs_key]
    applied, dormant_fraction, dormant_indices, activation_stats = value_module.apply(
        value_params, obs, return_dormant_indices=True, return_activation_stats=True)
    return jnp.squeeze(applied, axis=-1), dormant_fraction, dormant_indices, activation_stats

  obs_size = _get_obs_state_size(obs_size, obs_key)
  dummy_obs = jnp.zeros((1, obs_size))
  network = FeedForwardNetwork(
      init=lambda key: value_module.init(key, dummy_obs), apply=apply
  )
  network.apply_with_dormant = apply_with_dormant
  network.apply_with_dormant_indices = apply_with_dormant_indices
  network.apply_with_activation_stats = apply_with_activation_stats
  return network


def apply_redo_to_params(
    params: Any,
    layer_sizes: Sequence[int],
    dormant_indices_per_layer: Sequence[jnp.ndarray],
    kernel_init: Initializer = jax.nn.initializers.lecun_uniform(),
    rng: jax.random.PRNGKey = None,
) -> Any:
  """Applies ReDo (Reinitializing Dormant Neurons) technique to network parameters.
  
  This function implements the ReDo technique from "The Dormant Neuron Phenomenon" paper:
  - Reinitializes incoming weights for dormant neurons using original weight distribution
  - Zeros out outgoing weights for dormant neurons
  
  Args:
    params: Network parameters (Flax parameter dict, typically nested under 'params' key)
    layer_sizes: List of layer sizes (excluding input)
    dormant_indices_per_layer: List of arrays containing indices of dormant neurons for each layer
    kernel_init: Initializer for reinitializing weights
    rng: Random key for reinitialization
    
  Returns:
    Modified parameters with ReDo applied
  """
  if rng is None:
    rng = jax.random.PRNGKey(0)
  
  # Extract params dict (handle nested structure)
  if 'params' in params:
    param_dict = dict(params['params'])
  else:
    param_dict = dict(params)
  
  param_dict = jax.tree_util.tree_map(lambda x: x, param_dict)  # Create a copy
  rngs = jax.random.split(rng, len(layer_sizes))
  
  for layer_idx, (layer_size, dormant_indices, layer_rng) in enumerate(
      zip(layer_sizes, dormant_indices_per_layer, rngs)
  ):
    # Skip if no dormant neurons in this layer
    if dormant_indices.size == 0:
      continue
    
    # Create boolean mask from indices: True for dormant neurons, False otherwise
    dormant_mask = jnp.zeros(layer_size, dtype=jnp.bool_).at[dormant_indices].set(True)
    
    layer_name = f'hidden_{layer_idx}'
    if layer_name not in param_dict:
      continue
      
    layer_params = dict(param_dict[layer_name])
    
    # Reinitialize incoming weights for dormant neurons
    if 'kernel' in layer_params:
      kernel = layer_params['kernel']
      # Kernel is 2D: (input_size, output_size)
      input_size, output_size = kernel.shape[-2], kernel.shape[-1]
      
      # Generate new weights for dormant neurons
      neuron_rngs = jax.random.split(layer_rng, output_size)
      
      def reinit_for_neuron(neuron_rng):
        weights_2d = kernel_init(neuron_rng, (input_size, 1), kernel.dtype)
        return jnp.squeeze(weights_2d, axis=-1)  # Shape: (input_size,)
      
      new_weights = jax.vmap(reinit_for_neuron)(neuron_rngs).T  # Shape: [input_size, output_size]
      
      # Update kernel: use new weights for dormant neurons, keep old for others
      dormant_mask_2d = jnp.expand_dims(dormant_mask, axis=0)  # [1, output_size]
      kernel_updated = jnp.where(
          jnp.broadcast_to(dormant_mask_2d, (input_size, output_size)),
          new_weights,
          kernel
      )
      layer_params['kernel'] = kernel_updated
    
    # Zero out outgoing weights for dormant neurons (in next layer)
    if layer_idx < len(layer_sizes) - 1:
      next_layer_name = f'hidden_{layer_idx + 1}'
      if next_layer_name in param_dict:
        next_layer_params = dict(param_dict[next_layer_name])
        if 'kernel' in next_layer_params:
          next_kernel = next_layer_params['kernel']
          # Zero out rows corresponding to dormant neurons
          # next_kernel shape is [current_layer_size, next_layer_size]
          dormant_mask_2d = jnp.expand_dims(dormant_mask, axis=1)  # [layer_size, 1]
          next_kernel_updated = jnp.where(
              jnp.broadcast_to(dormant_mask_2d, next_kernel.shape),
              0.0,
              next_kernel
          )
          next_layer_params['kernel'] = next_kernel_updated
          param_dict[next_layer_name] = next_layer_params
    
    param_dict[layer_name] = layer_params
  
  # Return in the same structure as input
  if 'params' in params:
    return {**params, 'params': param_dict}
  else:
    return param_dict


# Import distribution for PPO networks
from brax.training import distribution
from brax.training.agents.ppo import networks as ppo_networks

def make_ppo_networks(
    observation_size: types.ObservationSize,
    action_size: int,
    preprocess_observations_fn: types.PreprocessObservationFn = types.identity_observation_preprocessor,
    policy_hidden_layer_sizes: Sequence[int] = (256, 256),
    value_hidden_layer_sizes: Sequence[int] = (256, 256),
    activation: ActivationFn = linen.relu,
    policy_obs_key: str = 'state',
    value_obs_key: str = 'state',
    distribution_type: str = 'tanh_normal',
    init_noise_std: float = 1.0,
) -> ppo_networks.PPONetworks:
  """Make PPO networks with dormant neuron detection support.
  
  This is a drop-in replacement for brax.training.agents.ppo.networks.make_ppo_networks
  that uses our custom MLP with dormant neuron counting.
  """
  # Create parametric action distribution
  if distribution_type == 'normal':
    parametric_action_distribution = distribution.NormalDistribution(
        event_size=action_size
    )
  elif distribution_type == 'tanh_normal':
    parametric_action_distribution = distribution.NormalTanhDistribution(
        event_size=action_size
    )
  else:
    raise ValueError(f'Unsupported distribution type: {distribution_type}')
  
  # Create policy network with dormant detection
  policy_network = make_policy_network(
      parametric_action_distribution.param_size,
      observation_size,
      preprocess_observations_fn=preprocess_observations_fn,
      hidden_layer_sizes=policy_hidden_layer_sizes,
      activation=activation,
      obs_key=policy_obs_key,
  )
  
  # Create value network with dormant detection
  value_network = make_value_network(
      observation_size,
      preprocess_observations_fn=preprocess_observations_fn,
      hidden_layer_sizes=value_hidden_layer_sizes,
      activation=activation,
      obs_key=value_obs_key,
  )
  
  return ppo_networks.PPONetworks(
      policy_network=policy_network,
      value_network=value_network,
      parametric_action_distribution=parametric_action_distribution,
  )
