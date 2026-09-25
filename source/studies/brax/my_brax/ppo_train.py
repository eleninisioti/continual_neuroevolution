# Copyright 2025 The Brax Authors.
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

"""Proximal policy optimization training.

See: https://arxiv.org/pdf/1707.06347.pdf
"""

import functools
import time
from typing import Any, Callable, Mapping, Optional, Tuple, Union

from absl import logging
from brax import base
from brax import envs
from brax.training import acting
from brax.training import gradients
from brax.training import logger as metric_logger
from brax.training import pmap
from brax.training import types
from brax.training.acme import running_statistics
from brax.training.acme import specs
from brax.training.agents.ppo import checkpoint
from brax.training.agents.ppo import losses as ppo_losses
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import optimizer as ppo_optimizer
from brax.training.types import Params
from brax.training.types import PRNGKey
import flax
import jax
import jax.numpy as jnp
import numpy as np
import optax

# Import ReDo support from my_brax
from source.studies.brax.my_brax import networks as my_brax_networks
from source.studies.brax.my_brax import dormant as my_brax_dormant
from source.studies.brax.my_brax import cchain as my_brax_cchain
from source.metrics.weight_stats import weight_stats
from source.metrics import ntk as ntk_metrics

InferenceParams = Tuple[running_statistics.NestedMeanStd, Params]
Metrics = types.Metrics

_PMAP_AXIS_NAME = 'i'


@flax.struct.dataclass
class TrainingState:
  """Contains training state for the learner."""

  optimizer_state: optax.OptState
  params: ppo_losses.PPONetworkParams
  normalizer_params: running_statistics.RunningStatisticsState
  env_steps: types.UInt64
  # C-CHAIN reference networks + coefficient controller. None when the method
  # is off, which is an empty pytree and so costs nothing in the pmap/scan.
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
) -> TrainingState:
    """One ReDo pass over the policy and value networks.

    All four steps, in the order source/algorithms/rl/redo.py defines them: score the
    neurons, mask the dormant ones, resample their incoming weights and zero
    their bias and outgoing weights, then clear the matching Adam moments.

    `redo_tau` is used by the *network*, not here: the threshold is an `MLP`
    attribute, bound when `network_factory` built the networks, so it is applied
    inside the forward pass that produces the masks. It is accepted here so the
    trainer can assert the two agree rather than silently diverge.

    Nothing here is wrapped in try/except. A ReDo pass that fails is a run that
    trained plain PPO under the label "redo"; the previous version logged a
    warning and returned the parameters untouched, which is how a mask bug
    survived unnoticed.

    Args:
        training_state: Current training state with network parameters
        ppo_network: PPO network with policy and value networks
        sample_obs: Sample observations to compute activations
        local_key: Random key for reinitialization
        local_devices_to_use: Number of local devices
        redo_tau: Threshold the networks were built with; checked, not applied

    Returns:
        Updated training state with ReDo applied
    """
    # Unpmap the params to work with them
    params = _unpmap(training_state.params)
    normalizer_params = _unpmap(training_state.normalizer_params)
    optimizer_state = _unpmap(training_state.optimizer_state)

    # Get sample observations (flatten if needed)
    sample_obs_flat = _unpmap(sample_obs)
    if isinstance(sample_obs_flat, dict):
        sample_obs_flat = sample_obs_flat.get('state', list(sample_obs_flat.values())[0])
    # Flatten batch dimensions
    sample_obs_flat = sample_obs_flat.reshape(-1, sample_obs_flat.shape[-1])

    # Split keys for policy and value
    local_key, policy_rng, value_rng = jax.random.split(local_key, 3)

    # Read the real widths off the parameters rather than assuming brax's
    # default (256, 256) -- ant and cheetah are (128, 128).
    policy_hidden_sizes = my_brax_networks.hidden_layer_sizes_from_params(params.policy)
    value_hidden_sizes = my_brax_networks.hidden_layer_sizes_from_params(params.value)

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
        params.policy, policy_hidden_sizes, policy_masks, rng=policy_rng,
    )
    new_value_params = my_brax_networks.apply_redo_to_params(
        params.value, value_hidden_sizes, value_masks, rng=value_rng,
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

    # Step 4: clear the Adam state of every recycled weight. Skipping this was
    # the single largest way the brax ReDo differed from the reference.
    new_optimizer_state = my_brax_networks.reset_adam_moments_for_masks(
        optimizer_state,
        policy_hidden_sizes,
        value_hidden_sizes,
        policy_masks,
        value_masks,
    )

    # Replicate back to devices
    devices = jax.local_devices()[:local_devices_to_use]
    new_params = jax.device_put_replicated(new_params, devices)
    new_optimizer_state = jax.device_put_replicated(new_optimizer_state, devices)

    return training_state.replace(
        params=new_params, optimizer_state=new_optimizer_state
    )


def _strip_weak_type(tree):
  # brax user code is sometimes ambiguous about weak_type.  in order to
  # avoid extra jit recompilations we strip all weak types from user input
  def f(leaf):
    leaf = jnp.asarray(leaf)
    return jnp.astype(leaf, leaf.dtype)

  return jax.tree_util.tree_map(f, tree)


def _validate_madrona_args(
    madrona_backend: bool,
    num_envs: int,
    num_eval_envs: int,
    action_repeat: int,
    eval_env: Optional[envs.Env] = None,
):
  """Validates arguments for Madrona-MJX."""
  if madrona_backend:
    if eval_env:
      raise ValueError("Madrona-MJX doesn't support multiple env instances")
    if num_eval_envs != num_envs:
      raise ValueError('Madrona-MJX requires a fixed batch size')
    if action_repeat != 1:
      raise ValueError(
          "Implement action_repeat using PipelineEnv's _n_frames to avoid"
          ' unnecessary rendering!'
      )


def _maybe_wrap_env(
    env: envs.Env,
    wrap_env: bool,
    num_envs: int,
    episode_length: Optional[int],
    action_repeat: int,
    device_count: int,
    key_env: PRNGKey,
    wrap_env_fn: Optional[Callable[[Any], Any]] = None,
    randomization_fn: Optional[
        Callable[[base.System, jnp.ndarray], Tuple[base.System, base.System]]
    ] = None,
):
  """Wraps the environment for training/eval if wrap_env is True."""
  if not wrap_env:
    return env
  if episode_length is None:
    raise ValueError('episode_length must be specified in ppo.train')
  v_randomization_fn = None
  if randomization_fn is not None:
    randomization_batch_size = num_envs // device_count
    # all devices gets the same randomization rng
    randomization_rng = jax.random.split(key_env, randomization_batch_size)
    v_randomization_fn = functools.partial(
        randomization_fn, rng=randomization_rng
    )
  if wrap_env_fn is not None:
    wrap_for_training = wrap_env_fn
  else:
    wrap_for_training = envs.training.wrap
  env = wrap_for_training(
      env,
      episode_length=episode_length,
      action_repeat=action_repeat,
      randomization_fn=v_randomization_fn,
  )  # pytype: disable=wrong-keyword-args
  return env


def _random_translate_pixels(
    obs: Mapping[str, jax.Array], key: PRNGKey
) -> Mapping[str, jax.Array]:
  """Apply random translations to B x T x ... pixel observations.

  The same shift is applied across the unroll_length (T) dimension.

  Args:
    obs: a dictionary of observations
    key: a PRNGKey

  Returns:
    A dictionary of observations with translated pixels
  """

  @jax.vmap
  def rt_all_views(
      ub_obs: Mapping[str, jax.Array], key: PRNGKey
  ) -> Mapping[str, jax.Array]:
    # Expects dictionary of unbatched observations.
    def rt_view(
        img: jax.Array, padding: int, key: PRNGKey
    ) -> jax.Array:  # TxHxWxC
      # Randomly translates a set of pixel inputs.
      # Adapted from
      # https://github.com/ikostrikov/jaxrl/blob/main/jaxrl/agents/drq/augmentations.py
      crop_from = jax.random.randint(key, (2,), 0, 2 * padding + 1)
      zero = jnp.zeros((1,), dtype=jnp.int32)
      crop_from = jnp.concatenate([zero, crop_from, zero])
      padded_img = jnp.pad(
          img,
          ((0, 0), (padding, padding), (padding, padding), (0, 0)),
          mode='edge',
      )
      return jax.lax.dynamic_slice(padded_img, crop_from, img.shape)

    out = {}
    for k_view, v_view in ub_obs.items():
      if k_view.startswith('pixels/'):
        key, key_shift = jax.random.split(key)
        out[k_view] = rt_view(v_view, 4, key_shift)
    return {**ub_obs, **out}

  bdim = next(iter(obs.items()), None)[1].shape[0]
  keys = jax.random.split(key, bdim)
  obs = rt_all_views(obs, keys)
  return obs


def _remove_pixels(
    obs: Union[jnp.ndarray, Mapping[str, jax.Array]],
) -> Union[jnp.ndarray, Mapping[str, jax.Array]]:
  """Removes pixel observations from the observation dict."""
  if not isinstance(obs, Mapping):
    return obs
  return {k: v for k, v in obs.items() if not k.startswith('pixels/')}


def train(
    environment: envs.Env,
    num_timesteps: int,
    max_devices_per_host: Optional[int] = None,
    # high-level control flow
    wrap_env: bool = True,
    madrona_backend: bool = False,
    augment_pixels: bool = False,
    # environment wrapper
    num_envs: int = 1,
    episode_length: Optional[int] = None,
    action_repeat: int = 1,
    wrap_env_fn: Optional[Callable[[Any], Any]] = None,
    randomization_fn: Optional[
        Callable[[base.System, jnp.ndarray], Tuple[base.System, base.System]]
    ] = None,
    # ppo params
    learning_rate: float = 1e-4,
    entropy_cost: float = 1e-4,
    discounting: float = 0.9,
    unroll_length: int = 10,
    batch_size: int = 32,
    num_minibatches: int = 16,
    num_updates_per_batch: int = 2,
    num_resets_per_eval: int = 0,
    normalize_observations: bool = False,
    normalize_observations_std_eps: float = 0.0,
    normalize_observations_mode: str = "welford",
    reward_scaling: float = 1.0,
    clipping_epsilon: float = 0.3,
    clipping_epsilon_value: float | None = None,
    gae_lambda: float = 0.95,
    max_grad_norm: Optional[float] = None,
    normalize_advantage: bool = True,
    vf_loss_coefficient: float = 0.5,
    bootstrap_on_timeout: bool = False,
    desired_kl: float = 0.01,
    learning_rate_schedule: Optional[
        Union[str, ppo_optimizer.LRSchedule]
    ] = None,
    network_factory: types.NetworkFactory[
        ppo_networks.PPONetworks
    ] = ppo_networks.make_ppo_networks,
    seed: int = 0,
    use_pmap_on_reset: bool = True,
    # eval
    num_evals: int = 1,
    eval_env: Optional[envs.Env] = None,
    num_eval_envs: int = 128,
    deterministic_eval: bool = False,
    # training metrics
    log_training_metrics: bool = False,
    training_metrics_steps: Optional[int] = None,
    # callbacks
    progress_fn: Callable[[int, Metrics], None] = lambda *args: None,
    policy_params_fn: Callable[..., None] = lambda *args: None,
    # checkpointing
    save_checkpoint_path: Optional[str] = None,
    restore_checkpoint_path: Optional[str] = None,
    restore_params: Optional[Any] = None,
    restore_value_fn: bool = True,
    run_evals: bool = True,
    use_trac: bool = False,
    # ReDo (Reinitializing Dormant Neurons)
    use_redo: bool = False,
    redo_frequency: int = 10,  # Apply ReDo every N epochs
    # Measure dormancy every evaluation and log it. Independent of ReDo:
    # ReDo RECYCLES dormant units, this only COUNTS them, and every method
    # needs the count or the plasticity figures have one populated column.
    # The continual core has had this since it was written; this one had
    # nothing, so every stationary ant/cheetah run logged no dormancy at all.
    track_dormant: bool = False,
    dormant_tau: float = my_brax_networks.REDO_DEFAULT_TAU,
    redo_tau: float = my_brax_networks.REDO_DEFAULT_TAU,  # Dormancy threshold; see source/algorithms/rl/redo.py
    # C-CHAIN (Continual Churn Approximated Reduction)
    use_cchain: bool = False,
    chain_target_rel_scale: float = 0.05,
    chain_warmup_iterations: int = 50,
    chain_coef_window: int = 100,
):
  """PPO training.

  Args:
    environment: the environment to train
    num_timesteps: the total number of environment steps to use during training
    max_devices_per_host: maximum number of chips to use per host process
    wrap_env: If True, wrap the environment for training. Otherwise use the
      environment as is.
    madrona_backend: whether to use Madrona backend for training
    augment_pixels: whether to add image augmentation to pixel inputs
    num_envs: the number of parallel environments to use for rollouts
      NOTE: `num_envs` must be divisible by the total number of chips since each
        chip gets `num_envs // total_number_of_chips` environments to roll out
      NOTE: `batch_size * num_minibatches` must be divisible by `num_envs` since
        data generated by `num_envs` parallel envs gets used for gradient
        updates over `num_minibatches` of data, where each minibatch has a
        leading dimension of `batch_size`
    episode_length: the length of an environment episode
    action_repeat: the number of timesteps to repeat an action
    wrap_env_fn: a custom function that wraps the environment for training. If
      not specified, the environment is wrapped with the default training
      wrapper.
    randomization_fn: a user-defined callback function that generates randomized
      environments
    learning_rate: learning rate for ppo loss
    entropy_cost: entropy reward for ppo loss, higher values increase entropy of
      the policy
    discounting: discounting rate
    unroll_length: the number of timesteps to unroll in each environment. The
      PPO loss is computed over `unroll_length` timesteps
    batch_size: the batch size for each minibatch SGD step
    num_minibatches: the number of times to run the SGD step, each with a
      different minibatch with leading dimension of `batch_size`
    num_updates_per_batch: the number of times to run the gradient update over
      all minibatches before doing a new environment rollout
    num_resets_per_eval: the number of environment resets to run between each
      eval. The environment resets occur on the host
    normalize_observations: whether to normalize observations
    normalize_observations_std_eps: small value added to the standard deviation
      for obs normalization to improve numerical stability
    normalize_observations_mode: method to use for running statistics, welford
      is the default, but ema is more numerically stable for long training runs
    reward_scaling: float scaling for reward
    clipping_epsilon: clipping epsilon for PPO loss
    clipping_epsilon_value: Value function loss clipping epsilon
    gae_lambda: General advantage estimation lambda
    max_grad_norm: gradient clipping norm value. If None, no clipping is done
    normalize_advantage: whether to normalize advantage estimate
    vf_loss_coefficient: Coefficient for value function loss.
    bootstrap_on_timeout: if True, bootstrap value on time_out steps using
      reward += gamma * V(s) * time_out. Environments should set
      state.info['time_out'] = 1.0 and done=True for steps where the episode ends
      due to a time_out.
    desired_kl: Desired KL divergence for adaptive KL divergence learning rate
      schedule.
    learning_rate_schedule: Learning rate schedule for the optimizer.
    network_factory: function that generates networks for policy and value
      functions
    seed: random seed
    num_evals: the number of evals to run during the entire training run.
      Increasing the number of evals increases total training time
    eval_env: an optional environment for eval only, defaults to `environment`
    num_eval_envs: the number of envs to use for evluation. Each env will run 1
      episode, and all envs run in parallel during eval.
    deterministic_eval: whether to run the eval with a deterministic policy
    log_training_metrics: whether to log training metrics and callback to
      progress_fn
    training_metrics_steps: the number of environment steps between logging
      training metrics
    progress_fn: a user-defined callback function for reporting/plotting metrics
    policy_params_fn: a user-defined callback function that can be used for
      saving custom policy checkpoints or creating policy rollouts and videos
    save_checkpoint_path: the path used to save checkpoints. If None, no
      checkpoints are saved.
    restore_checkpoint_path: the path used to restore previous model params
    restore_params: raw network parameters to restore the TrainingState from.
      These override `restore_checkpoint_path`. These paramaters can be obtained
      from the return values of ppo.train().
    restore_value_fn: whether to restore the value function from the checkpoint
      or use a random initialization
    run_evals: if True, use the evaluator num_eval times to collect distinct
      eval rollouts. If False, num_eval_envs and eval_env are ignored.
      progress_fn is then expected to use training_metrics.
    use_pmap_on_reset: default to True. if True, use pmap instead of vmap for
      env.reset across devices.

  Returns:
    Tuple of (make_policy function, network params, metrics)
  """
  assert batch_size * num_minibatches % num_envs == 0
  _validate_madrona_args(
      madrona_backend, num_envs, num_eval_envs, action_repeat, eval_env
  )

  xt = time.time()

  process_count = jax.process_count()
  process_id = jax.process_index()
  local_device_count = jax.local_device_count()
  local_devices_to_use = local_device_count
  if max_devices_per_host:
    local_devices_to_use = min(local_devices_to_use, max_devices_per_host)
  logging.info(
      'Device count: %d, process count: %d (id %d), local device count: %d, '
      'devices to be used count: %d',
      jax.device_count(),
      process_count,
      process_id,
      local_device_count,
      local_devices_to_use,
  )
  device_count = local_devices_to_use * process_count

  # The number of environment steps executed for every training step.
  env_step_per_training_step = (
      batch_size * unroll_length * num_minibatches * action_repeat
  )
  num_evals_after_init = max(num_evals - 1, 1)
  # The number of training_step calls per training_epoch call.
  # equals to ceil(num_timesteps / (num_evals * env_step_per_training_step *
  #                                 num_resets_per_eval))
  num_training_steps_per_epoch = np.ceil(
      num_timesteps
      / (
          num_evals_after_init
          * env_step_per_training_step
          * max(num_resets_per_eval, 1)
      )
  ).astype(int)

  key = jax.random.PRNGKey(seed)
  global_key, local_key = jax.random.split(key)
  del key
  local_key = jax.random.fold_in(local_key, process_id)
  local_key, key_env, eval_key = jax.random.split(local_key, 3)
  # key_networks should be global, so that networks are initialized the same
  # way for different processes.
  key_policy, key_value = jax.random.split(global_key)
  del global_key

  assert num_envs % device_count == 0

  env = _maybe_wrap_env(
      environment,
      wrap_env,
      num_envs,
      episode_length,
      action_repeat,
      device_count,
      key_env,
      wrap_env_fn,
      randomization_fn,
  )

  def reset_fn_donated_env_state(env_state_donated, key_envs):
    return env.reset(key_envs)

  key_envs = jax.random.split(key_env, num_envs // process_count)
  key_envs = jnp.reshape(
      key_envs, (local_devices_to_use, -1) + key_envs.shape[1:]
  )
  if local_devices_to_use > 1 or use_pmap_on_reset:
    reset_fn_ = jax.pmap(env.reset, axis_name=_PMAP_AXIS_NAME)
    env_state = reset_fn_(key_envs)
    reset_fn = jax.pmap(
        reset_fn_donated_env_state,
        axis_name=_PMAP_AXIS_NAME,
        donate_argnums=(0,),
    )
  else:
    reset_fn_ = jax.jit(jax.vmap(env.reset))
    env_state = reset_fn_(key_envs)
    reset_fn = jax.jit(
        reset_fn_donated_env_state, donate_argnums=(0,), keep_unused=True
    )

  # Discard the batch axes over devices and envs.
  obs_shape = jax.tree_util.tree_map(lambda x: x.shape[2:], env_state.obs)

  normalize = lambda x, y: x
  if normalize_observations:
    normalize = running_statistics.normalize
  # ReDo's threshold lives on the network (it is applied inside the forward pass
  # that scores the neurons), so it has to be bound here. Passed only when ReDo
  # is on, so every other run builds exactly the network it did before.
  # The threshold is applied inside the forward pass that scores the neurons,
  # so it is bound on the network. ReDo uses it to decide what to recycle and
  # --track_dormant uses it to decide what to report; they must be the same
  # number, which is why measuring without ReDo still binds it.
  if use_redo:
    _net_kwargs = {'redo_tau': redo_tau}
  elif track_dormant:
    _net_kwargs = {'redo_tau': dormant_tau}
  else:
    _net_kwargs = {}
  ppo_network = network_factory(
      obs_shape, env.action_size, preprocess_observations_fn=normalize,
      **_net_kwargs
  )
  make_policy = ppo_networks.make_inference_fn(
      ppo_network,
      compute_value=bootstrap_on_timeout or clipping_epsilon_value is not None,
  )

  # Optimizer.
  base_optimizer = optax.adam(learning_rate=learning_rate)
  lr_schedule = learning_rate_schedule or ppo_optimizer.LRSchedule.NONE
  lr_schedule = ppo_optimizer.LRSchedule(lr_schedule)
  lr_is_adaptive_kl = lr_schedule == ppo_optimizer.LRSchedule.ADAPTIVE_KL
  if lr_is_adaptive_kl:
    base_optimizer = optax.inject_hyperparams(optax.adam)(
        learning_rate=learning_rate
    )
  if max_grad_norm is not None:
    # TODO(btaba): Move gradient clipping to `training/gradients.py`.
    optimizer = optax.chain(
        optax.clip_by_global_norm(max_grad_norm),
        base_optimizer,
    )
  else:
    optimizer = base_optimizer

  # Wrap optimizer with Trac if enabled
  if use_trac:
    from trac_optimizer.experimental.jax.trac import start_trac
    optimizer = start_trac(optimizer)
    logging.info('Using TRAC optimizer wrapper')

  loss_fn = functools.partial(
      ppo_losses.compute_ppo_loss,
      ppo_network=ppo_network,
      entropy_cost=entropy_cost,
      discounting=discounting,
      reward_scaling=reward_scaling,
      gae_lambda=gae_lambda,
      clipping_epsilon=clipping_epsilon,
      normalize_advantage=normalize_advantage,
      vf_coefficient=vf_loss_coefficient,
      clipping_epsilon_value=clipping_epsilon_value,
  )

  loss_and_pgrad_fn = gradients.loss_and_pgrad(
      loss_fn, pmap_axis_name=_PMAP_AXIS_NAME, has_aux=True
  )

  if use_cchain:
    # The churn regulariser needs a second, independently shuffled minibatch of
    # observations and the reference policy params, so it cannot reuse
    # loss_and_pgrad_fn's signature. Everything C-CHAIN is kept on its own code
    # path below rather than branching inside minibatch_step/sgd_step, so a
    # plain PPO, TRAC or ReDo run traces exactly the code it did before.
    if augment_pixels:
      raise NotImplementedError(
          'use_cchain does not support augment_pixels: the churn term would be '
          'computed on differently augmented observations than the policy loss.')
    if lr_is_adaptive_kl:
      raise NotImplementedError(
          'use_cchain does not support adaptive-KL learning rates.')

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

  steps_between_logging = training_metrics_steps or env_step_per_training_step
  metrics_aggregator = metric_logger.EpisodeMetricsLogger(
      steps_between_logging=steps_between_logging,
      progress_fn=progress_fn,
  )

  def minibatch_step(
      carry,
      data: types.Transition,
      normalizer_params: running_statistics.RunningStatisticsState,
  ):
    # `prev_params` is carried so that after the scans finish it holds the
    # parameters exactly ONE GRADIENT STEP back -- the reference the churn
    # column is measured against, for every method and not just C-CHAIN.
    optimizer_state, params, prev_params, key = carry
    key, key_loss = jax.random.split(key)
    (_, metrics), grads = loss_and_pgrad_fn(
        params, normalizer_params, data, key_loss
    )

    if lr_is_adaptive_kl:
      kl_mean = metrics['kl_mean']
      kl_mean = jax.lax.pmean(kl_mean, axis_name=_PMAP_AXIS_NAME)
      optimizer_state, lr = ppo_optimizer.adaptive_kl_learning_rate(
          optimizer_state, kl_mean, desired_kl
      )
    else:
      lr = jnp.array(learning_rate)
    metrics['learning_rate'] = lr

    # apply gradients (pass params as third argument for Trac optimizer compatibility)
    params_update, optimizer_state = optimizer.update(grads, optimizer_state, params)
    new_params = optax.apply_updates(params, params_update)

    return (optimizer_state, new_params, params, key), metrics

  def sgd_step(
      carry,
      unused_t,
      data: types.Transition,
      normalizer_params: running_statistics.RunningStatisticsState,
  ):
    optimizer_state, params, prev_params, key = carry
    key, key_perm, key_grad = jax.random.split(key, 3)

    if augment_pixels:
      key, key_rt = jax.random.split(key)
      r_translate = functools.partial(_random_translate_pixels, key=key_rt)
      data = types.Transition(
          observation=r_translate(data.observation),  # pytype: disable=wrong-arg-types
          action=data.action,
          reward=data.reward,
          discount=data.discount,
          next_observation=r_translate(data.next_observation),  # pytype: disable=wrong-arg-types
          extras=data.extras,
      )

    def convert_data(x: jnp.ndarray):
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

  def minibatch_step_chain(
      carry,
      xs,
      normalizer_params: running_statistics.RunningStatisticsState,
  ):
    """minibatch_step plus the C-CHAIN churn term.

    `xs` is (training minibatch, regularisation observations); the two come
    from independent shuffles of the same rollout, which is what makes the
    churn term suppress off-diagonal NTK entries rather than just re-fitting
    the batch the policy loss already saw.
    """
    optimizer_state, params, chain_state, key = carry
    data, reg_obs = xs
    key, key_loss = jax.random.split(key)

    # The regulariser needs a reference that is genuinely one step behind, so
    # it stays off until two updates have happened. Folding it into the
    # coefficient rather than branching keeps the traced graph one shape.
    active = jnp.where(chain_state.grad_steps >= 2, 1.0, 0.0)
    chain_coef = chain_state.coef * active

    (_, metrics), grads = chain_loss_and_pgrad_fn(
        params, normalizer_params, data, reg_obs,
        chain_state.ref_policy_params, chain_coef, key_loss
    )

    metrics['learning_rate'] = jnp.array(learning_rate)

    # Reference for the *next* step: the params this update started from.
    next_ref_policy_params = params.policy

    params_update, optimizer_state = optimizer.update(grads, optimizer_state, params)
    params = optax.apply_updates(params, params_update)

    chain_state = chain_state.replace(
        ref_policy_params=next_ref_policy_params,
        grad_steps=chain_state.grad_steps + 1,
    )
    # Reported (and fed to the controller) as 0 while inactive, matching the
    # reference's `reg_loss = 0` branch -- otherwise the controller would
    # calibrate against a churn value that was never actually applied.
    metrics['chain_reg_loss'] = metrics['chain_reg_loss'] * active

    return (optimizer_state, params, chain_state, key), metrics

  def sgd_step_chain(
      carry,
      unused_t,
      data: types.Transition,
      normalizer_params: running_statistics.RunningStatisticsState,
  ):
    optimizer_state, params, chain_state, key = carry
    key, key_perm, key_perm_reg, key_grad = jax.random.split(key, 4)

    def convert_data(x: jnp.ndarray):
      x = jax.random.permutation(key_perm, x)
      return jnp.reshape(x, (num_minibatches, -1) + x.shape[1:])

    # A second, independent shuffle. Only the observations are permuted: the
    # churn term needs nothing else from the transition, and shuffling the full
    # Transition a second time would double this scan's memory for no reason.
    def convert_obs(x: jnp.ndarray):
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

  def training_step(
      carry: Tuple[TrainingState, envs.State, PRNGKey], unused_t
  ) -> Tuple[Tuple[TrainingState, envs.State, PRNGKey], Metrics]:
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
      extra_fields = ['truncation', 'episode_metrics', 'episode_done']
      if bootstrap_on_timeout:
        extra_fields.append('time_out')
      next_state, data = acting.generate_unroll(
          env,
          current_state,
          policy,
          current_key,
          unroll_length,
          extra_fields=tuple(extra_fields),
      )
      return (next_state, next_key), data

    (state, _), data = jax.lax.scan(
        f,
        (state, key_generate_unroll),
        (),
        length=batch_size * num_minibatches // num_envs,
    )
    # Have leading dimensions (batch_size * num_minibatches, unroll_length)
    data = jax.tree_util.tree_map(lambda x: jnp.swapaxes(x, 1, 2), data)
    data = jax.tree_util.tree_map(
        lambda x: jnp.reshape(x, (-1,) + x.shape[2:]), data
    )
    assert data.discount.shape[1:] == (unroll_length,)

    if bootstrap_on_timeout:  # bootstrap reward on timeout
      time_out = data.extras['state_extras']['time_out']
      value = data.extras['policy_extras']['value']
      data = types.Transition(
          observation=data.observation,
          action=data.action,
          reward=data.reward + discounting * time_out * value,
          discount=data.discount,
          next_observation=data.next_observation,
          extras=data.extras,
      )

    normalizer_params = training_state.normalizer_params
    if not lr_is_adaptive_kl:
      # Update normalization params before SGD for backwards compatibility.
      normalizer_params = running_statistics.update(
          normalizer_params,
          _remove_pixels(data.observation),
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

      # One controller tick per PPO iteration, from the last minibatch of the
      # last epoch. The reference updates its coefficient once per `iteration`
      # from whatever pg_loss/reg_loss were left in scope after the epoch loop,
      # which is exactly that minibatch; metrics here are stacked
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
      # gradient: the paper's reported "policy churn". It is the same quantity
      # the regulariser minimises, measured on data the update did not use.
      churn_obs = jax.tree_util.tree_map(
          lambda x: jax.random.permutation(key_churn, x)[: batch_size], data.observation
      )
      metrics = dict(metrics)
      metrics['policy_churn'] = my_brax_cchain.churn_loss(
          ppo_network, normalizer_params, params.policy,
          chain_state.ref_policy_params, churn_obs)
      metrics['chain_coef'] = chain_state.coef
    else:
      chain_state = training_state.chain_state
      (optimizer_state, params, prev_params, _), metrics = jax.lax.scan(
          functools.partial(
              sgd_step, data=data, normalizer_params=normalizer_params
          ),
          (training_state.optimizer_state, training_state.params,
           training_state.params, key_sgd),
          (),
          length=num_updates_per_batch,
      )

    if lr_is_adaptive_kl:
      # For adaptive KL, normalization params should be updated after SGD s.t.
      # old distribution outputs are valid for KL computation.
      normalizer_params = running_statistics.update(
          normalizer_params,
          _remove_pixels(data.observation),
          pmap_axis_name=_PMAP_AXIS_NAME,
      )

    # POLICY CHURN, FOR EVERY METHOD -- see the note in ppo_continual_train.py.
    # The reference logs the identical quantity for vanilla PPO
    # (crl_run_ppo_dmc.py:318) that it logs for C-CHAIN
    # (crl_run_ppo_c_chain_dmc.py:339): squared difference of action means
    # against the policy ONE GRADIENT STEP back. C-CHAIN takes that reference
    # from chain_state; everything else takes it from `prev_params`.
    #
    # A deterministic slice, not a fresh shuffle: a key split here would consume
    # the trainer's RNG and change every non-C-CHAIN run.
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

    if log_training_metrics:  # log unroll metrics
      jax.debug.callback(
          metrics_aggregator.update_episode_metrics,
          data.extras['state_extras']['episode_metrics'],
          data.extras['state_extras']['episode_done'],
          metrics,
      )

    return (new_training_state, state, new_key), metrics

  def training_epoch(
      training_state: TrainingState, state: envs.State, key: PRNGKey
  ) -> Tuple[TrainingState, envs.State, Metrics]:
    (training_state, state, _), loss_metrics = jax.lax.scan(
        training_step,
        (training_state, state, key),
        (),
        length=num_training_steps_per_epoch,
    )
    loss_metrics = jax.tree_util.tree_map(jnp.mean, loss_metrics)
    return training_state, state, loss_metrics

  training_epoch = jax.pmap(
      training_epoch,
      axis_name=_PMAP_AXIS_NAME,
      donate_argnums=(
          0,
          1,
      ),
  )

  # Note that this is NOT a pure jittable method.
  def training_epoch_with_timing(
      training_state: TrainingState, env_state: envs.State, key: PRNGKey
  ) -> Tuple[TrainingState, envs.State, Metrics]:
    nonlocal training_walltime
    t = time.time()
    training_state, env_state = _strip_weak_type((training_state, env_state))
    result = training_epoch(training_state, env_state, key)
    training_state, env_state, metrics = _strip_weak_type(result)

    metrics = jax.tree_util.tree_map(jnp.mean, metrics)
    jax.tree_util.tree_map(lambda x: x.block_until_ready(), metrics)

    epoch_training_time = time.time() - t
    training_walltime += epoch_training_time
    sps = (
        num_training_steps_per_epoch
        * env_step_per_training_step
        * max(num_resets_per_eval, 1)
    ) / epoch_training_time
    metrics = {
        'training/sps': sps,
        'training/walltime': training_walltime,
        **{f'training/{name}': value for name, value in metrics.items()},
    }
    return training_state, env_state, metrics  # pytype: disable=bad-return-type  # py311-upgrade

  # Initialize model params and training state.
  init_params = ppo_losses.PPONetworkParams(
      policy=ppo_network.policy_network.init(key_policy),
      value=ppo_network.value_network.init(key_value),
  )

  obs_shape = jax.tree_util.tree_map(
      lambda x: specs.Array(x.shape[-1:], jnp.dtype('float32')), env_state.obs
  )
  training_state = TrainingState(  # pytype: disable=wrong-arg-types  # jax-ndarray
      optimizer_state=optimizer.init(init_params),  # pytype: disable=wrong-arg-types  # numpy-scalars
      params=init_params,
      normalizer_params=running_statistics.init_state(
          _remove_pixels(obs_shape),
          std_eps=normalize_observations_std_eps,
          mode=normalize_observations_mode,
      ),
      env_steps=types.UInt64(hi=0, lo=0),
      chain_state=(
          my_brax_cchain.init_chain_state(
              init_params.policy, window=chain_coef_window)
          if use_cchain else None
      ),
  )

  if restore_checkpoint_path is not None:
    params = checkpoint.load(restore_checkpoint_path)
    value_params = params[2] if restore_value_fn else init_params.value
    training_state = training_state.replace(
        normalizer_params=params[0],
        params=training_state.params.replace(
            policy=params[1], value=value_params
        ),
    )

  if restore_params is not None:
    logging.info('Restoring TrainingState from `restore_params`.')
    value_params = restore_params[2] if restore_value_fn else init_params.value
    training_state = training_state.replace(
        normalizer_params=restore_params[0],
        params=training_state.params.replace(
            policy=restore_params[1], value=value_params
        ),
    )

  if num_timesteps == 0:
    return (
        make_policy,
        (
            training_state.normalizer_params,
            training_state.params.policy,
            training_state.params.value,
        ),
        {},
    )

  training_state = jax.device_put_replicated(
      training_state, jax.local_devices()[:local_devices_to_use]
  )

  eval_env = _maybe_wrap_env(
      eval_env or environment,
      wrap_env,
      num_eval_envs,
      episode_length,
      action_repeat,
      device_count=1,  # eval on the host only
      key_env=eval_key,
      wrap_env_fn=wrap_env_fn,
      randomization_fn=randomization_fn,
  )
  evaluator = acting.Evaluator(
      eval_env,
      functools.partial(make_policy, deterministic=deterministic_eval),
      num_eval_envs=num_eval_envs,
      episode_length=episode_length,
      action_repeat=action_repeat,
      key=eval_key,
  )

  training_metrics = {}
  # Hidden widths come from the factory's bound kwargs, which is where the
  # trainers set them; the tracker needs one age array per hidden layer.
  policy_dormant_tracker = value_dormant_tracker = None
  if track_dormant:
    _fk = getattr(network_factory, 'keywords', {}) or {}
    policy_dormant_tracker, value_dormant_tracker = my_brax_dormant.make_trackers(
        _fk.get('policy_hidden_layer_sizes', (32,) * 4),
        _fk.get('value_hidden_layer_sizes', (256,) * 5),
    )
  training_walltime = 0
  current_step = 0

  # Run initial eval
  metrics = {}
  if process_id == 0 and num_evals > 1 and run_evals:
    metrics = evaluator.run_evaluation(
        _unpmap((
            training_state.normalizer_params,
            training_state.params.policy,
            training_state.params.value,
        )),
        training_metrics={},
    )
    logging.info(metrics)
    progress_fn(0, metrics)

  # Run initial policy_params_fn.
  params = _unpmap((
      training_state.normalizer_params,
      training_state.params.policy,
      training_state.params.value,
  ))
  policy_params_fn(current_step, make_policy, params)

  for it in range(num_evals_after_init):
    logging.info('starting iteration %s %s', it, time.time() - xt)

    for _ in range(max(num_resets_per_eval, 1)):
      # optimization
      epoch_key, local_key = jax.random.split(local_key)
      epoch_keys = jax.random.split(epoch_key, local_devices_to_use)
      (training_state, env_state, training_metrics) = (
          training_epoch_with_timing(training_state, env_state, epoch_keys)
      )
      current_step = int(_unpmap(training_state.env_steps))

      # Apply ReDo (Reinitializing Dormant Neurons) periodically
      if use_redo and (it + 1) % redo_frequency == 0:
        local_key, redo_key = jax.random.split(local_key)
        # Deliberately not guarded: a ReDo pass that silently fails leaves a run
        # labelled "redo" that trained plain PPO. Let it raise.
        training_state = _apply_redo(
            training_state,
            ppo_network,
            env_state.obs,
            redo_key,
            local_devices_to_use,
            redo_tau=redo_tau,
        )
        logging.info(f'Applied ReDo at iteration {it + 1}')

      key_envs = jax.vmap(
          lambda x, s: jax.random.split(x[0], s), in_axes=(0, None)
      )(key_envs, key_envs.shape[1])
      # TODO(brax-team): move extra reset logic to the AutoResetWrapper.
      if num_resets_per_eval > 0:
        env_state = reset_fn(env_state, key_envs)

    if process_id != 0:
      continue

    # Process id == 0.
    params = _unpmap((
        training_state.normalizer_params,
        training_state.params.policy,
        training_state.params.value,
    ))

    policy_params_fn(current_step, make_policy, params)

    if save_checkpoint_path is not None:
      ckpt_config = checkpoint.network_config(
          observation_size=obs_shape,
          action_size=env.action_size,
          normalize_observations=normalize_observations,
          network_factory=network_factory,
      )
      checkpoint.save(
          save_checkpoint_path, current_step, params, ckpt_config
      )

    if num_evals > 0:
      metrics = training_metrics
      if run_evals:
        metrics = evaluator.run_evaluation(
            params,
            training_metrics,
        )
      # Weight statistics and NTK effective rank, the same two the continual
      # core computes at the same point and by the same functions. Here rather
      # than in each trainer's policy_params_fn because `ppo_train` serves the
      # stationary ant AND the stationary cheetah: one copy, and the two suites
      # cannot drift. See source/metrics/{weight_stats,ntk}.py.
      metrics.update(weight_stats(params[1], prefix='policy_weight'))
      metrics.update(weight_stats(params[2], prefix='value_weight'))
      _ntk_obs = _unpmap(env_state.obs)
      if isinstance(_ntk_obs, dict):
        _ntk_obs = _ntk_obs.get('state', list(_ntk_obs.values())[0])
      _ntk_obs = _ntk_obs.reshape(-1, _ntk_obs.shape[-1])
      metrics.update(ntk_metrics.ntk_rank_stats(
          lambda pp, o: ppo_network.policy_network.apply(params[0], pp, o),
          params[1], _ntk_obs, prefix='policy_ntk'))

      if track_dormant:
        # On the observations the policy is actually seeing this iteration, not
        # a fixed probe batch: this is the RL-side measure and it matches what
        # the continual core does, so the two trees' columns mean one thing.
        metrics = dict(metrics)
        my_brax_dormant.add_dormancy_metrics(
            metrics, ppo_network,
            _unpmap(training_state.normalizer_params),
            _unpmap(training_state.params),
            _unpmap(env_state.obs),
            policy_dormant_tracker, value_dormant_tracker,
            tau=dormant_tau if not use_redo else redo_tau,
        )
      logging.info(metrics)
      progress_fn(current_step, metrics)

  total_steps = current_step
  if not total_steps >= num_timesteps:
    raise AssertionError(
        f'Total steps {total_steps} is less than `num_timesteps`='
        f' {num_timesteps}.'
    )

  # If there was no mistakes the training_state should still be identical on all
  # devices.
  pmap.assert_is_replicated(training_state)
  params = _unpmap((
      training_state.normalizer_params,
      training_state.params.policy,
      training_state.params.value,
  ))
  logging.info('total steps: %s', total_steps)
  pmap.synchronize_hosts()
  return (make_policy, params, metrics)
