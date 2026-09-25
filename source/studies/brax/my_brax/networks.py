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

"""Network definitions with dormant neuron counting support.

The dormancy score and the threshold come from `source/algorithms/rl/redo.py`, the one
ReDo the whole study uses. Before that they were local to this file and wrong in
three separate ways, all of which invalidated every mujoco and brax `redo` run:

  * the threshold was the literal `0.01`, so `--redo_tau` never reached the
    test it documented;
  * the score was mean|activation|, the ReLU test, on tanh networks -- where the
    dead unit is the one saturated at +/-1, which has the *largest* possible
    mean|activation| and so could never be flagged; and
  * dormancy was passed around as an index array using `-1` for "not dormant",
    which `jnp.zeros(w).at[idx].set(True)` wrapped to index `w-1`, so the last
    neuron of every hidden layer was recycled on every pass no matter what.

Dormancy is a boolean mask everywhere here now.
"""

import dataclasses
import functools
from typing import Any, Callable, List, Mapping, Sequence, Tuple
import warnings

from brax.training import types
from brax.training.acme import running_statistics
from brax.training.spectral_norm import SNDense
from flax import linen
import jax
import jax.numpy as jnp
import optax

from source.algorithms.rl import redo as core_redo
from source.metrics import plasticity
from source.algorithms.rl.redo import DEFAULT_TAU as REDO_DEFAULT_TAU


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
  # ReDo's threshold, and how a neuron is scored against it. Both come from
  # source/algorithms/rl/redo.py so mujoco and brax count a dormant neuron exactly as
  # gymnax and kinetix do. `dormant_criterion=None` means "derive it from
  # `activation`", which is what every caller should do -- the ReLU test and the
  # tanh test are not interchangeable. See core/redo.py.
  redo_tau: float = REDO_DEFAULT_TAU
  dormant_criterion: str = None

  def _criterion(self):
    if self.dormant_criterion is not None:
      return self.dormant_criterion
    return core_redo.criterion_for_activation_fn(self.activation)

  @linen.compact
  def __call__(self, data: jnp.ndarray, return_dormant_masks: bool = False, return_activation_stats: bool = False):
    hidden = data
    criterion = self._criterion()

    n_dormant = 0
    n_neurons = 0
    dormant_masks_per_layer = []  # Boolean mask per hidden layer, True = dormant
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

        # Score every neuron in this layer. `hidden` may be [batch, neurons] or
        # [batch, ..., neurons] under pmap; every leading axis is batch, which
        # is what neuron_dormancy_score assumes.
        per_neuron_score = core_redo.neuron_dormancy_score(hidden, criterion)
        dormant_mask = per_neuron_score <= self.redo_tau

        if return_activation_stats:
          abs_hidden = jnp.abs(hidden)
          batch_dims = tuple(range(abs_hidden.ndim - 1))
          mean_act_per_neuron = (
              jnp.mean(abs_hidden, axis=batch_dims) if abs_hidden.ndim > 1 else abs_hidden)
          std_act_per_neuron = (
              jnp.std(abs_hidden, axis=batch_dims) if abs_hidden.ndim > 1
              else jnp.zeros_like(abs_hidden))
          activation_stats_per_layer.append({
            'mean_per_neuron': mean_act_per_neuron,
            'std_per_neuron': std_act_per_neuron,
            'layer_mean': jnp.mean(mean_act_per_neuron),
            'layer_std': jnp.std(mean_act_per_neuron),
            'min_neuron_mean': jnp.min(mean_act_per_neuron),
            'max_neuron_mean': jnp.max(mean_act_per_neuron),
            'min_neuron_score': jnp.min(per_neuron_score),
            'max_neuron_score': jnp.max(per_neuron_score),
          })

        if return_dormant_masks:
          dormant_masks_per_layer.append(dormant_mask)

        n_dormant += jnp.sum(dormant_mask.astype(jnp.int32))
        n_neurons += hidden.shape[-1]  # Number of neurons in this layer

    # Return fraction of dormant neurons (avoid division by zero)
    dormant_fraction = jnp.where(n_neurons > 0, n_dormant / n_neurons, 0.0)

    if return_activation_stats:
      return hidden, dormant_fraction, dormant_masks_per_layer, activation_stats_per_layer
    if return_dormant_masks:
      return hidden, dormant_fraction, dormant_masks_per_layer
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
    redo_tau: float = REDO_DEFAULT_TAU,
) -> FeedForwardNetwork:
  """Creates a policy network."""
  policy_module = MLP(
      layer_sizes=list(hidden_layer_sizes) + [param_size],
      activation=activation,
      kernel_init=kernel_init,
      layer_norm=layer_norm,
      redo_tau=redo_tau,
  )

  def apply(processor_params, policy_params, obs):
    obs = preprocess_observations_fn(obs, processor_params)
    obs = obs if isinstance(obs, jax.Array) else obs[obs_key]
    # MLP returns (hidden, dormant_fraction), but for compatibility with brax
    # we only return the logits
    logits, _ = policy_module.apply(policy_params, obs)
    return logits
  
  def apply_with_dormant_masks(processor_params, policy_params, obs):
    """Apply policy network and return logits, dormant fraction, and dormant masks per layer."""
    obs = preprocess_observations_fn(obs, processor_params)
    obs = obs if isinstance(obs, jax.Array) else obs[obs_key]
    logits, n_dormant, dormant_masks = policy_module.apply(
        policy_params, obs, return_dormant_masks=True)
    return logits, n_dormant, dormant_masks

  def apply_with_activation_stats(processor_params, policy_params, obs):
    """Apply policy network and return logits, dormant fraction, dormant masks, and activation stats."""
    obs = preprocess_observations_fn(obs, processor_params)
    obs = obs if isinstance(obs, jax.Array) else obs[obs_key]
    logits, n_dormant, dormant_masks, activation_stats = policy_module.apply(
        policy_params, obs, return_dormant_masks=True, return_activation_stats=True)
    return logits, n_dormant, dormant_masks, activation_stats

  obs_size = _get_obs_state_size(obs_size, obs_key)
  dummy_obs = jnp.zeros((1, obs_size))
  network = FeedForwardNetwork(
      init=lambda key: policy_module.init(key, dummy_obs), apply=apply
  )
  network.apply_with_dormant_masks = apply_with_dormant_masks
  network.apply_with_activation_stats = apply_with_activation_stats
  return network


def make_value_network(
    obs_size: types.ObservationSize,
    preprocess_observations_fn: types.PreprocessObservationFn = types.identity_observation_preprocessor,
    hidden_layer_sizes: Sequence[int] = (256, 256),
    activation: ActivationFn = linen.relu,
    obs_key: str = 'state',
    redo_tau: float = REDO_DEFAULT_TAU,
) -> FeedForwardNetwork:
  """Creates a value network."""
  value_module = MLP(
      layer_sizes=list(hidden_layer_sizes) + [1],
      activation=activation,
      kernel_init=jax.nn.initializers.lecun_uniform(),
      redo_tau=redo_tau,
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
  
  def apply_with_dormant_masks(processor_params, value_params, obs):
    """Apply value network and return value, dormant fraction, and dormant masks per layer."""
    obs = preprocess_observations_fn(obs, processor_params)
    obs = obs if isinstance(obs, jax.Array) else obs[obs_key]
    applied, dormant_fraction, dormant_masks = value_module.apply(
        value_params, obs, return_dormant_masks=True)
    return jnp.squeeze(applied, axis=-1), dormant_fraction, dormant_masks

  def apply_with_activation_stats(processor_params, value_params, obs):
    """Apply value network and return value, dormant fraction, dormant masks, and activation stats."""
    obs = preprocess_observations_fn(obs, processor_params)
    obs = obs if isinstance(obs, jax.Array) else obs[obs_key]
    applied, dormant_fraction, dormant_masks, activation_stats = value_module.apply(
        value_params, obs, return_dormant_masks=True, return_activation_stats=True)
    return jnp.squeeze(applied, axis=-1), dormant_fraction, dormant_masks, activation_stats

  obs_size = _get_obs_state_size(obs_size, obs_key)
  dummy_obs = jnp.zeros((1, obs_size))
  network = FeedForwardNetwork(
      init=lambda key: value_module.init(key, dummy_obs), apply=apply
  )
  network.apply_with_dormant = apply_with_dormant
  network.apply_with_dormant_masks = apply_with_dormant_masks
  network.apply_with_activation_stats = apply_with_activation_stats
  return network


def apply_redo_to_params(
    params: Any,
    layer_sizes: Sequence[int],
    dormant_masks_per_layer: Sequence[jnp.ndarray],
    kernel_init: Initializer = jax.nn.initializers.lecun_uniform(),
    rng: jax.random.PRNGKey = None,
) -> Any:
  """Applies ReDo (Reinitializing Dormant Neurons) technique to network parameters.

  Implements steps 3 of `source/algorithms/rl/redo.py`, on brax's `hidden_i` layer naming:
  - reinitialises incoming weights for dormant neurons from the layer initialiser
  - zeros their bias
  - zeros their outgoing weights, so the function computed is essentially
    unchanged at the moment of the reset

  Step 4, zeroing the matching Adam moments, is `reset_adam_moments_for_masks`
  below, and it is not optional -- see the reference's own comment on it. The
  caller must run both.

  Args:
    params: Network parameters (Flax parameter dict, typically nested under 'params' key)
    layer_sizes: List of layer sizes (excluding input)
    dormant_masks_per_layer: Boolean mask per hidden layer, True where dormant.
      This used to be an index array with `-1` marking "not dormant", which
      wrapped round to the last neuron and recycled it unconditionally.
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

  for layer_idx, (layer_size, dormant_mask, layer_rng) in enumerate(
      zip(layer_sizes, dormant_masks_per_layer, rngs)
  ):
    dormant_mask = jnp.asarray(dormant_mask, dtype=jnp.bool_)

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

    # Zero the bias of the recycled neurons. flax initialises Dense biases to
    # zero, so this is drawing the bias from its initialiser exactly as the
    # kernel is drawn from its own; the reference does the same. Skipping it
    # left a recycled unit with the bias that helped kill it.
    if 'bias' in layer_params:
      layer_params['bias'] = jnp.where(dormant_mask, 0.0, layer_params['bias'])

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


def hidden_layer_sizes_from_params(params: Any) -> List[int]:
  """Widths of the hidden layers of an MLP, read back off its parameters.

  `MLP` names its layers `hidden_0 .. hidden_{n-1}` with the last one being the
  output head, which ReDo never touches. Reading the widths from the params is
  what keeps ReDo honest about the network it is actually operating on: the
  call sites used to hardcode `[256, 256]`, which is brax's default and not what
  ant or cheetah run (both are 128, 128 -- see core/networks.POLICY_ARCH).
  """
  param_dict = params['params'] if 'params' in params else params
  sizes = []
  i = 0
  while f'hidden_{i}' in param_dict:
    sizes.append(int(param_dict[f'hidden_{i}']['kernel'].shape[-1]))
    i += 1
  return sizes[:-1]  # drop the output head


def _zero_moment_subtree(moment_subtree, layer_sizes, masks):
  """Zero the Adam moment entries of every weight `apply_redo_to_params` touched.

  `moment_subtree` mirrors one network's parameter dict, i.e. it has the
  `hidden_i` keys directly (or nested under 'params').
  """
  if 'params' in moment_subtree:
    inner = _zero_moment_subtree(moment_subtree['params'], layer_sizes, masks)
    return {**moment_subtree, 'params': inner}

  out = dict(moment_subtree)
  for layer_idx, mask in enumerate(masks):
    mask = jnp.asarray(mask, dtype=jnp.bool_)
    name = f'hidden_{layer_idx}'
    if name not in out:
      continue
    layer = dict(out[name])
    if 'kernel' in layer:
      layer['kernel'] = jnp.where(mask[None, :], 0.0, layer['kernel'])
    if 'bias' in layer:
      layer['bias'] = jnp.where(mask, 0.0, layer['bias'])
    out[name] = layer

    nxt_name = f'hidden_{layer_idx + 1}'
    if layer_idx < len(layer_sizes) - 1 and nxt_name in out:
      nxt = dict(out[nxt_name])
      if 'kernel' in nxt:
        nxt['kernel'] = jnp.where(mask[:, None], 0.0, nxt['kernel'])
      out[nxt_name] = nxt
  return out


def reset_adam_moments_for_masks(
    optimizer_state: Any,
    policy_layer_sizes: Sequence[int],
    value_layer_sizes: Sequence[int],
    policy_masks: Sequence[jnp.ndarray],
    value_masks: Sequence[jnp.ndarray],
    reset_count: bool = True,
) -> Any:
  """Step 4 of ReDo: clear the Adam state of every recycled weight.

  The brax path used to skip this entirely -- its comment read "Keep the
  optimizer state as-is (the paper suggests this is fine)", which the paper does
  not: `inspiration/redo/src/redo.py` says "Step count resets are key to the
  algorithm's performance". A recycled neuron that keeps the momentum and second
  moment accumulated while it was dying is pulled straight back towards the
  weights it was just rescued from.

  brax keeps one optimizer over `PPONetworkParams(policy=..., value=...)`, so the
  moment tree has those two branches and each mirrors its network's parameters.
  `reset_count` restarts optax's single global bias-correction counter, the
  closest available equivalent to the reference's per-tensor step reset (the
  reference resets the step of nearly every tensor anyway).
  """
  found = []

  def rewrite(moments):
    # `moments` is a PPONetworkParams-shaped pytree; touch both branches.
    policy = _zero_moment_subtree(moments.policy, policy_layer_sizes, policy_masks)
    value = _zero_moment_subtree(moments.value, value_layer_sizes, value_masks)
    return moments.replace(policy=policy, value=value)

  def walk(node):
    if isinstance(node, optax.ScaleByAdamState):
      found.append(True)
      return node._replace(
          mu=rewrite(node.mu),
          nu=rewrite(node.nu),
          count=jnp.zeros_like(node.count) if reset_count else node.count,
      )
    if isinstance(node, tuple) and hasattr(node, '_fields'):  # other NamedTuple states
      return type(node)(*[walk(c) for c in node])
    if isinstance(node, (list, tuple)):
      return type(node)(walk(c) for c in node)
    return node

  new_state = walk(optimizer_state)
  if not found:
    # Half-applying ReDo is the failure mode it is most sensitive to, so refuse
    # rather than silently skip. Matches source/algorithms/rl/redo.py.
    raise ValueError(
        'ReDo found no optax.adam state to reset. It only knows how to clear '
        'Adam moments; adapt reset_adam_moments_for_masks for other optimizers.'
    )
  return new_state


# Import distribution for PPO networks
from brax.training import distribution
from brax.training.agents.ppo import networks as ppo_networks

class StateIndependentNormalDistribution(distribution.ParametricDistribution):
  """C-CHAIN's / CleanRL's continuous policy: unbounded Gaussian, fixed std.

  brax's default NormalTanhDistribution PREDICTS a per-state scale and squashes
  the sample through tanh. CleanRL's ppo_continuous_action -- which C-CHAIN
  builds on -- instead has

      actor_logstd = nn.Parameter(torch.zeros(1, action_dim))

  i.e. ONE learnable log-std shared across every state, initialised at 0 so
  sigma = 1.0 on a [-1, 1] action space, with the sampled action then clipped by
  a gym ClipAction wrapper. That difference is why `ent_coef 0` is survivable
  for them and fatal for us: at sigma 1.0 a large fraction of their actions
  saturate the action bounds, so exploration persists with no entropy bonus,
  whereas brax's state-dependent squashed sigma collapses and the policy stops
  learning entirely (WalkerStand: their PPO peaks 762, ours 279 against a 138
  random baseline).

  Reproducing their entropy-0 result therefore needs this parameterisation, not
  just their entropy coefficient. brax's own NormalDistribution cannot be used:
  it declares param_size = event_size, leaving no scale output at all, and its
  create_dist unpacks the parameter array positionally.

  The network emits 2 * action_size logits, the second half of which is a
  broadcast learnable parameter rather than a function of the observation --
  see make_state_independent_std_policy_network.

  Actions are NOT squashed. MJX clamps ctrl to the actuator range, which is the
  same thing ClipAction does for them, and log_prob is taken on the unclipped
  sample exactly as in their code.
  """

  def __init__(self, event_size: int, min_std: float = 1e-3):
    super().__init__(
        param_size=2 * event_size,
        postprocessor=distribution.IdentityPostprocessor(),
        event_ndims=1,
        reparametrizable=True,
    )
    self._min_std = min_std

  def create_dist(self, parameters):
    loc, log_scale = jnp.split(parameters, 2, axis=-1)
    scale = jnp.exp(log_scale) + self._min_std
    return distribution._NormalDistribution(loc=loc, scale=scale)


class _StateIndependentStdMLP(linen.Module):
  """MLP mean head plus a single learnable log-std vector.

  log_std is a module parameter, not an output of the trunk, so it does not
  depend on the observation. Initialised to zeros => sigma 1.0, matching
  `nn.Parameter(torch.zeros(...))`.
  """

  layer_sizes: Sequence[int]
  action_size: int
  activation: ActivationFn = linen.relu
  kernel_init: Initializer = jax.nn.initializers.lecun_uniform()

  @linen.compact
  def __call__(self, obs):
    mean, dormant = MLP(
        layer_sizes=list(self.layer_sizes) + [self.action_size],
        activation=self.activation,
        kernel_init=self.kernel_init,
    )(obs)
    log_std = self.param('log_std', linen.initializers.zeros, (self.action_size,))
    log_std = jnp.broadcast_to(log_std, mean.shape)
    return jnp.concatenate([mean, log_std], axis=-1), dormant


def make_state_independent_std_policy_network(
    action_size: int,
    obs_size: types.ObservationSize,
    preprocess_observations_fn: types.PreprocessObservationFn = types.identity_observation_preprocessor,
    hidden_layer_sizes: Sequence[int] = (256, 256),
    activation: ActivationFn = linen.relu,
    kernel_init: Initializer = jax.nn.initializers.lecun_uniform(),
    obs_key: str = 'state',
) -> FeedForwardNetwork:
  """Policy network for StateIndependentNormalDistribution."""
  module = _StateIndependentStdMLP(
      layer_sizes=list(hidden_layer_sizes),
      action_size=action_size,
      activation=activation,
      kernel_init=kernel_init,
  )

  def apply(processor_params, policy_params, obs):
    obs = preprocess_observations_fn(obs, processor_params)
    obs = obs if isinstance(obs, jax.Array) else obs[obs_key]
    logits, _ = module.apply(policy_params, obs)
    return logits

  # Same resolution the other heads use: obs_size may be a dict spec, and the
  # policy consumes only the flat `obs_key` entry.
  flat_obs_size = _get_obs_state_size(obs_size, obs_key)
  dummy_obs = jnp.zeros((1, flat_obs_size))
  return FeedForwardNetwork(
      init=lambda key: module.init(key, dummy_obs), apply=apply)


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
    policy_min_std: float = 0.001,
    redo_tau: float = REDO_DEFAULT_TAU,
) -> ppo_networks.PPONetworks:
  """Make PPO networks with dormant neuron detection support.

  This is a drop-in replacement for brax.training.agents.ppo.networks.make_ppo_networks
  that uses our custom MLP with dormant neuron counting.

  policy_min_std floors the action std: brax parameterises it as
  softplus(head) + min_std, and its default 0.001 is no floor at all. With
  entropy_cost 0 nothing else opposes the std collapsing. The knob separates
  "the entropy BONUS was needed" from "any floor on exploration was needed" --
  see docs/cchain_walker_reproduction.md.
  """
  # Create parametric action distribution
  if distribution_type == 'normal':
    parametric_action_distribution = distribution.NormalDistribution(
        event_size=action_size
    )
  elif distribution_type == 'state_independent_normal':
    # C-CHAIN / CleanRL's parameterisation. Needs its own policy network too,
    # because the log-std is a module parameter rather than a trunk output.
    parametric_action_distribution = StateIndependentNormalDistribution(
        event_size=action_size
    )
    policy_network = make_state_independent_std_policy_network(
        action_size,
        observation_size,
        preprocess_observations_fn=preprocess_observations_fn,
        hidden_layer_sizes=policy_hidden_layer_sizes,
        activation=activation,
        obs_key=policy_obs_key,
    )
    value_network = make_value_network(
        observation_size,
        preprocess_observations_fn=preprocess_observations_fn,
        hidden_layer_sizes=value_hidden_layer_sizes,
        activation=activation,
        obs_key=value_obs_key,
        redo_tau=redo_tau,
    )
    return ppo_networks.PPONetworks(
        policy_network=policy_network,
        value_network=value_network,
        parametric_action_distribution=parametric_action_distribution,
    )
  elif distribution_type == 'tanh_normal':
    parametric_action_distribution = distribution.NormalTanhDistribution(
        event_size=action_size, min_std=policy_min_std
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
      redo_tau=redo_tau,
  )
  
  # Create value network with dormant detection
  value_network = make_value_network(
      observation_size,
      preprocess_observations_fn=preprocess_observations_fn,
      hidden_layer_sizes=value_hidden_layer_sizes,
      activation=activation,
      obs_key=value_obs_key,
      redo_tau=redo_tau,
  )
  
  return ppo_networks.PPONetworks(
      policy_network=policy_network,
      value_network=value_network,
      parametric_action_distribution=parametric_action_distribution,
  )


def policy_action_means(ppo_network, normalizer_params, policy_params, obs):
  """The policy's mean action on `obs`, with the distribution's scale dropped.

  `logits` from a `NormalTanhDistribution` are `[loc, scale]` concatenated on
  the last axis, so only the first half is the action. Same slicing as
  `my_brax/cchain.py:churn_loss`, and the same fallback for distributions whose
  parameters are not a (loc, scale) pair.
  """
  out = ppo_network.policy_network.apply(normalizer_params, policy_params, obs)
  action_size = out.shape[-1] // 2
  if 2 * action_size == out.shape[-1]:
    return out[..., :action_size]
  return out

