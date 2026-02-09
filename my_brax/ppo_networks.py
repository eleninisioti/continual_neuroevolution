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

"""PPO networks using local networks module with dormant neuron counting."""

from typing import Callable, Sequence, Tuple

from brax.training import distribution
from my_brax import networks  # Use our local networks module
from brax.training import types
from brax.training.types import PRNGKey
from flax import linen
import flax
import jax
import jax.numpy as jnp


class CategoricalDistribution:
  """Categorical distribution for discrete action spaces.
  
  Outputs one logit per action, samples using categorical distribution,
  and takes argmax for deterministic mode.
  
  This is a simple implementation that doesn't inherit from brax's ParametricDistribution
  to avoid the abstract method requirements.
  """

  def __init__(self, num_actions: int):
    self._num_actions = num_actions
    self._param_size = num_actions

  @property
  def param_size(self) -> int:
    return self._param_size

  def sample_no_postprocessing(self, logits, key):
    """Sample action index from categorical distribution."""
    return jax.random.categorical(key, logits)

  def sample(self, logits, key):
    """Sample and postprocess (no postprocessing for discrete)."""
    return self.sample_no_postprocessing(logits, key)

  def log_prob(self, logits, actions):
    """Compute log probability of actions."""
    log_probs = jax.nn.log_softmax(logits)
    # Handle both batched and unbatched cases
    if actions.ndim == 0:
      return log_probs[actions]
    return jnp.take_along_axis(log_probs, actions[..., None].astype(jnp.int32), axis=-1).squeeze(-1)

  def postprocess(self, actions):
    """No postprocessing needed for discrete actions."""
    return actions

  def mode(self, logits):
    """Return argmax (most likely action)."""
    return jnp.argmax(logits, axis=-1)

  def entropy(self, logits, unused_actions):
    """Compute entropy of the distribution."""
    probs = jax.nn.softmax(logits)
    log_probs = jax.nn.log_softmax(logits)
    return -jnp.sum(probs * log_probs, axis=-1)


@flax.struct.dataclass
class PPONetworks:
  policy_network: networks.FeedForwardNetwork
  value_network: networks.FeedForwardNetwork
  parametric_action_distribution: distribution.ParametricDistribution


def make_inference_fn(ppo_networks: PPONetworks):
  """Creates params and inference function for the PPO agent."""

  def make_policy(
      params: types.Params, deterministic: bool = False
  ) -> types.Policy:
    policy_network = ppo_networks.policy_network
    parametric_action_distribution = ppo_networks.parametric_action_distribution

    def policy(
        observations: types.Observation, key_sample: PRNGKey
    ) -> Tuple[types.Action, types.Extra]:
      param_subset = (params[0], params[1])  # normalizer and policy params
      logits, n_dormant = policy_network.apply(*param_subset, observations)
      if deterministic:
        return ppo_networks.parametric_action_distribution.mode(logits), {}
      raw_actions = parametric_action_distribution.sample_no_postprocessing(
          logits, key_sample
      )
      log_prob = parametric_action_distribution.log_prob(logits, raw_actions)
      postprocessed_actions = parametric_action_distribution.postprocess(
          raw_actions
      )
      return postprocessed_actions, {
          'log_prob': log_prob,
          'raw_action': raw_actions,
          # Note: n_dormant is NOT included here because it's a scalar and would
          # break tree_map operations that expect batched tensors. Use the 
          # separate get_dormant_fraction function if needed.
      }

    return policy

  return make_policy


def make_ppo_networks(
    observation_size: int,
    action_size: int,
    preprocess_observations_fn: types.PreprocessObservationFn = types.identity_observation_preprocessor,
    policy_hidden_layer_sizes: Sequence[int] = (32,) * 4,
    value_hidden_layer_sizes: Sequence[int] = (256,) * 5,
    activation: networks.ActivationFn = linen.swish,
    policy_obs_key: str = 'state',
    value_obs_key: str = 'state',
) -> PPONetworks:
  """Make PPO networks with preprocessor."""
  parametric_action_distribution = distribution.NormalTanhDistribution(
      event_size=action_size
  )
  policy_network = networks.make_policy_network(
      parametric_action_distribution.param_size,
      observation_size,
      preprocess_observations_fn=preprocess_observations_fn,
      hidden_layer_sizes=policy_hidden_layer_sizes,
      activation=activation,
      obs_key=policy_obs_key,
  )
  value_network = networks.make_value_network(
      observation_size,
      preprocess_observations_fn=preprocess_observations_fn,
      hidden_layer_sizes=value_hidden_layer_sizes,
      activation=activation,
      obs_key=value_obs_key,
  )

  return PPONetworks(
      policy_network=policy_network,
      value_network=value_network,
      parametric_action_distribution=parametric_action_distribution,
  )


def make_ppo_networks_vision_discrete(
    observation_size,  # Mapping[str, Tuple[int, ...]] e.g., {"pixels/": (7, 7, 2)}
    action_size: int,
    preprocess_observations_fn: types.PreprocessObservationFn = types.identity_observation_preprocessor,
    policy_hidden_layer_sizes: Sequence[int] = (256, 256),
    value_hidden_layer_sizes: Sequence[int] = (256, 256),
    activation: Callable = linen.swish,
    normalise_channels: bool = False,
    policy_obs_key: str = "",
    value_obs_key: str = "",
) -> 'PPONetworks':
  """Make Vision PPO networks for discrete action spaces.
  
  Uses CategoricalDistribution - outputs one logit per action,
  samples from softmax, takes argmax for deterministic mode.
  """
  from brax.training.agents.ppo import networks_vision as brax_networks_vision
  
  parametric_action_distribution = CategoricalDistribution(
      num_actions=action_size
  )
  
  # Use brax's vision networks but with our discrete output size
  policy_network = brax_networks_vision.networks.make_policy_network_vision(
      observation_size=observation_size,
      output_size=parametric_action_distribution.param_size,  # = action_size (one logit per action)
      preprocess_observations_fn=preprocess_observations_fn,
      activation=activation,
      hidden_layer_sizes=policy_hidden_layer_sizes,
      state_obs_key=policy_obs_key,
      normalise_channels=normalise_channels,
  )
  
  value_network = brax_networks_vision.networks.make_value_network_vision(
      observation_size=observation_size,
      preprocess_observations_fn=preprocess_observations_fn,
      activation=activation,
      hidden_layer_sizes=value_hidden_layer_sizes,
      state_obs_key=value_obs_key,
      normalise_channels=normalise_channels,
  )
  
  return PPONetworks(
      policy_network=policy_network,
      value_network=value_network,
      parametric_action_distribution=parametric_action_distribution,
  )

def make_ppo_networks_discrete(
    observation_size: int,
    action_size: int,
    preprocess_observations_fn: types.PreprocessObservationFn = types.identity_observation_preprocessor,
    policy_hidden_layer_sizes: Sequence[int] = (32,) * 4,
    value_hidden_layer_sizes: Sequence[int] = (256,) * 5,
    activation: networks.ActivationFn = linen.swish,
    policy_obs_key: str = 'state',
    value_obs_key: str = 'state',
) -> PPONetworks:
  """Make PPO networks for discrete action spaces.
  
  Uses CategoricalDistribution - outputs one logit per action,
  samples from softmax, takes argmax for deterministic mode.
  """
  parametric_action_distribution = CategoricalDistribution(
      num_actions=action_size
  )
  policy_network = networks.make_policy_network(
      parametric_action_distribution.param_size,  # = action_size
      observation_size,
      preprocess_observations_fn=preprocess_observations_fn,
      hidden_layer_sizes=policy_hidden_layer_sizes,
      activation=activation,
      obs_key=policy_obs_key,
  )
  value_network = networks.make_value_network(
      observation_size,
      preprocess_observations_fn=preprocess_observations_fn,
      hidden_layer_sizes=value_hidden_layer_sizes,
      activation=activation,
      obs_key=value_obs_key,
  )

  return PPONetworks(
      policy_network=policy_network,
      value_network=value_network,
      parametric_action_distribution=parametric_action_distribution,
  )
