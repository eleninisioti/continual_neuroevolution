import os
import sys
import time
import json
import pickle
from typing import Any, NamedTuple

import hydra
import jax
import jax.numpy as jnp
from kinetix.environment import make_reset_fn_from_config
import numpy as np
import optax
import imageio

# Helper class for teeing stdout to a log file
class Tee:
    def __init__(self, filepath):
        self.file = open(filepath, 'w')
        self.stdout = sys.stdout
    def write(self, data):
        self.file.write(data)
        self.stdout.write(data)
    def flush(self):
        self.file.flush()
        self.stdout.flush()
    def close(self):
        self.file.close()

# Get repository root for project structure
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Optional TRAC optimizer import
try:
    from trac_optimizer.experimental.jax.trac import start_trac
    TRAC_AVAILABLE = True
except ImportError:
    TRAC_AVAILABLE = False
from flax.serialization import to_state_dict
from flax.training.train_state import TrainState
from omegaconf import OmegaConf

import wandb
from kinetix.environment import LogWrapper, PixelObservations
from kinetix.environment.env import make_kinetix_env
from kinetix.models import ScannedRNN, make_network_from_config
from kinetix.models.redo import (
    DEFAULT_TAU as REDO_DEFAULT_TAU,
    apply_redo,
    collect_scores,
)
from kinetix.models.cchain import (
    DEFAULT_COEF_WINDOW,
    DEFAULT_TARGET_REL_SCALE,
    DEFAULT_WARMUP_UPDATES,
    chain_policy_churn,
    init_coef_state,
    update_coef_state,
)
from kinetix.render.renderer_pixels import make_render_pixels
from kinetix.util import (
    general_eval,
    generate_params_from_config,
    load_evaluation_levels,
    get_video_frequency,
    init_wandb,
    load_train_state_from_wandb_artifact_path,
    normalise_config,
    save_model,
)

os.environ["WANDB_DISABLE_SERVICE"] = "True"


class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: Any
    info: jnp.ndarray


def make_train(config, env_params, static_env_params, init_params=None):
    """Build the PPO train function.

    `init_params` seeds the network with existing weights instead of a fresh
    init. Only PBT uses it -- an agent that loses the exploit step is restarted
    from the winner's weights, and doing that in-process avoids a round trip
    through wandb artifacts. Left None, behaviour is exactly as before.
    """
    config["num_updates"] = config["total_timesteps"] // config["num_steps"] // config["num_train_envs"]
    print(config["num_minibatches"])
    config["minibatch_size"] = config["num_train_envs"] * config["num_steps"] // config["num_minibatches"]
    #print("config", config)
    #()

    print(config["num_updates"])
    def make_env(reset_fn, static_env_params):
        return LogWrapper(
            make_kinetix_env(config["action_type"], config["observation_type"], reset_fn, env_params, static_env_params)
        )

    eval_levels, eval_static_env_params = load_evaluation_levels(config["eval_levels"], static_env_params_override=static_env_params)
    env = make_env(make_reset_fn_from_config(config, env_params, static_env_params), static_env_params)
    eval_env = make_env(None, static_env_params)  # Use training static_env_params to ensure observation size matches

    def linear_schedule(count):
        frac = 1.0 - (count // (config["num_minibatches"] * config["update_epochs"])) / config["num_updates"]
        return config["lr"] * frac

    def linear_warmup_cosine_decay_schedule(count):
        frac = (count // (config["num_minibatches"] * config["update_epochs"])) / config[
            "num_updates"
        ]  # between 0 and 1
        delta = config["peak_lr"] - config["initial_lr"]
        frac_diff_max = 1.0 - config["warmup_frac"]
        frac_cosine = (frac - config["warmup_frac"]) / frac_diff_max

        return jax.lax.select(
            frac < config["warmup_frac"],
            config["initial_lr"] + delta * frac / config["warmup_frac"],
            config["peak_lr"] * jnp.maximum(0.0, 0.5 * (1.0 + jnp.cos(jnp.pi * ((frac_cosine) % 1.0)))),
        )

    time_start = time.time()
    use_redo = config.get("use_redo", False)
    monitor_dormant = config.get("monitor_dormant", False)
    use_trac = config.get("use_trac", False)
    # ReDo knobs, named and defaulted as in the gymnax trainer so the two
    # benchmarks' redo runs are configured the same way.
    redo_interval = int(config.get("redo_interval", 50))
    redo_batch_size = int(config.get("redo_batch_size", 512))
    redo_targets = config.get("redo_targets", "both")
    # Total hidden units across both stacks; the denominator for the reset
    # fraction, and a compile-time constant so both lax.cond branches agree.
    _n_hidden_neurons = 2 * config["fc_layer_depth"] * config["fc_layer_width"]
    # C-CHAIN knobs, same names and defaults as add_chain_args in the gymnax port.
    use_cchain = config.get("use_cchain", False)
    chain_target_rel_scale = float(config.get("chain_target_rel_scale", DEFAULT_TARGET_REL_SCALE))
    chain_warmup_updates = int(config.get("chain_warmup_updates", DEFAULT_WARMUP_UPDATES))
    chain_coef_window = int(config.get("chain_coef_window", DEFAULT_COEF_WINDOW))

    def train(rng):
        last_time = time.time()
        # INIT NETWORK
        network = make_network_from_config(env, env_params, config)
        
        # Helper function to handle network.apply - network always returns 4 values now
        def apply_network(params, hstate, ac_in):
            result = network.apply(params, hstate, ac_in)
            hstate, pi, value, dormant_counts = result
            # Only use dormant_counts if monitoring is enabled
            if monitor_dormant or use_redo:
                return hstate, pi, value, dormant_counts
            else:
                return hstate, pi, value, None
        rng, _rng = jax.random.split(rng)
        obsv, env_state = jax.vmap(env.reset, (0, None))(jax.random.split(_rng, config["num_train_envs"]), env_params)
        dones = jnp.zeros((config["num_train_envs"]), dtype=jnp.bool_)
        rng, _rng = jax.random.split(rng)
        init_hstate = ScannedRNN.initialize_carry(config["num_train_envs"])
        init_x = jax.tree.map(lambda x: x[None, ...], (obsv, dones))
        network_params = network.init(_rng, init_hstate, init_x)
        if init_params is not None:
            # Shapes must match the freshly-initialised tree, so a mismatch is
            # a bug rather than something to paper over.
            jax.tree.map(lambda a, b: None, network_params, init_params)
            network_params = init_params

        param_count = sum(x.size for x in jax.tree_util.tree_leaves(network_params))
        obs_size = sum(x.size for x in jax.tree_util.tree_leaves(obsv)) // config["num_train_envs"]

        print("Number of parameters", param_count, "size of obs: ", obs_size)
        if config["anneal_lr"]:
            tx = optax.chain(
                optax.clip_by_global_norm(config["max_grad_norm"]),
                optax.adam(learning_rate=linear_schedule, eps=1e-5),
            )
        elif config["warmup_lr"]:
            tx = optax.chain(
                optax.clip_by_global_norm(config["max_grad_norm"]),
                optax.adamw(learning_rate=linear_warmup_cosine_decay_schedule, eps=1e-5),
            )
        else:
            tx = optax.chain(
                optax.clip_by_global_norm(config["max_grad_norm"]),
                optax.adam(config["lr"], eps=1e-5),
            )
        
        # Wrap optimizer with TRAC if enabled
        if use_trac:
            if not TRAC_AVAILABLE:
                raise ImportError("TRAC optimizer requested but trac-optimizer package is not installed. Install with: pip install trac-optimizer")
            print("Using TRAC optimizer wrapper")
            tx = start_trac(tx)
        
        train_state = TrainState.create(
            apply_fn=network.apply,
            params=network_params,
            tx=tx,
        )
        print("load_from_checkpoint", config["load_from_checkpoint"])
        if config["load_from_checkpoint"] != None:
            print("LOADING from", config["load_from_checkpoint"], "with only params =", config["load_only_params"])
            #train_state = load_train_state_from_wandb_artifact_path(
            #    train_state, config["load_from_checkpoint"], load_only_params=config["load_only_params"]
            #)
            
            train_state = load_train_state_from_wandb_artifact_path(
                train_state, config["load_from_checkpoint"], load_only_params=False
            )
        # INIT ENV
        rng, _rng = jax.random.split(rng)
        obsv, env_state = jax.vmap(env.reset, (0, None))(jax.random.split(_rng, config["num_train_envs"]), env_params)
        init_hstate = ScannedRNN.initialize_carry(config["num_train_envs"])
        render_static_env_params = eval_env.static_env_params.replace(downscale=4)
        pixel_renderer = jax.jit(make_render_pixels(env_params, render_static_env_params))
        pixel_render_fn = lambda x: pixel_renderer(x) / 255.0

        def _vmapped_eval_step(runner_state, rng):
            def _single_eval_step(rng):
                return general_eval(
                    rng,
                    eval_env,
                    env_params,
                    runner_state[0],
                    eval_levels,
                    env_params.max_timesteps,
                    config["num_eval_levels"],
                    keep_states=True,
                    return_trajectories=True,
                )

            (states, returns, done_idxs, episode_lengths, eval_infos), (eval_dones, eval_rewards) = jax.vmap(
                _single_eval_step
            )(jax.random.split(rng, config["eval_num_attempts"]))
            mask = jnp.arange(env_params.max_timesteps)[None, ..., None] < episode_lengths[:, None, :]
            eval_solves = (eval_infos["returned_episode_solved"] * eval_dones * mask).sum(axis=1) / jnp.maximum(
                1, (eval_dones * mask).sum(axis=1)
            )
            states_to_plot = jax.tree.map(lambda x: x[0], states)

            return (
                states_to_plot,
                done_idxs[0],
                returns[0],
                returns.mean(axis=0),
                episode_lengths.mean(axis=0),
                eval_solves.mean(axis=0),
            )

        # TRAIN LOOP
        def _update_step(carry, unused):
            # C-CHAIN state has to survive across updates (the reference policy
            # is one gradient step back, and the coefficient controller averages
            # over a window of updates), so it rides in the scan carry beside
            # runner_state rather than inside it -- runner_state is unpacked as a
            # 7-tuple in five places and widening it would touch all of them.
            runner_state, chain_carry = carry
            chain_ref, chain_step, coef_state = chain_carry
            chain_coef = coef_state.coef
            # COLLECT TRAJECTORIES
            def _env_step(runner_state_and_counts, unused):
                runner_state, policy_dormant_sum, value_dormant_sum, n_counts = runner_state_and_counts
                (
                    train_state,
                    env_state,
                    last_obs,
                    last_done,
                    hstate,
                    rng,
                    update_step,
                ) = runner_state

                # SELECT ACTION
                rng, _rng = jax.random.split(rng)
                ac_in = (jax.tree.map(lambda x: x[np.newaxis, :], last_obs), last_done[np.newaxis, :])
                hstate, pi, value, dormant_counts = apply_network(train_state.params, hstate, ac_in)
                action = pi.sample(seed=_rng)
                log_prob = pi.log_prob(action)
                value, action, log_prob = (
                    value.squeeze(0),
                    action.squeeze(0),
                    log_prob.squeeze(0),
                )
                
                # Accumulate dormant counts if monitoring or ReDo is enabled
                if monitor_dormant or use_redo:
                    policy_dormant_frac, value_dormant_frac = dormant_counts
                    # Convert to JAX arrays and average over batch dimension if needed
                    policy_dormant_frac = jnp.asarray(policy_dormant_frac)
                    value_dormant_frac = jnp.asarray(value_dormant_frac)
                    if policy_dormant_frac.ndim > 0:
                        policy_dormant_frac = jnp.mean(policy_dormant_frac)
                    if value_dormant_frac.ndim > 0:
                        value_dormant_frac = jnp.mean(value_dormant_frac)
                    policy_dormant_sum = policy_dormant_sum + policy_dormant_frac
                    value_dormant_sum = value_dormant_sum + value_dormant_frac
                    n_counts = n_counts + 1

                # STEP ENV
                rng, _rng = jax.random.split(rng)
                obsv, env_state, reward, done, info = jax.vmap(env.step, in_axes=(0, 0, 0, None))(
                    jax.random.split(_rng, config["num_train_envs"]), env_state, action, env_params
                )
                transition = Transition(last_done, action, value, reward, log_prob, last_obs, info)
                runner_state = (
                    train_state,
                    env_state,
                    obsv,
                    done,
                    hstate,
                    rng,
                    update_step,
                )
                return (runner_state, policy_dormant_sum, value_dormant_sum, n_counts), transition

            initial_hstate = runner_state[-3]
            # Initialize dormant count accumulators
            init_policy_dormant = 0.0
            init_value_dormant = 0.0
            init_n_counts = 0
            runner_state_and_counts = (runner_state, init_policy_dormant, init_value_dormant, init_n_counts)
            (runner_state, policy_dormant_sum, value_dormant_sum, n_counts), traj_batch = jax.lax.scan(
                _env_step, runner_state_and_counts, None, config["num_steps"]
            )

            # CALCULATE ADVANTAGE
            (
                train_state,
                env_state,
                last_obs,
                last_done,
                hstate,
                rng,
                update_step,
            ) = runner_state
            ac_in = (jax.tree.map(lambda x: x[np.newaxis, :], last_obs), last_done[np.newaxis, :])
            _, _, last_val, _ = apply_network(train_state.params, hstate, ac_in)
            last_val = last_val.squeeze(0)

            def _calculate_gae(traj_batch, last_val, last_done):
                def _get_advantages(carry, transition):
                    gae, next_value, next_done = carry
                    done, value, reward = (
                        transition.done,
                        transition.value,
                        transition.reward,
                    )
                    delta = reward + config["gamma"] * next_value * (1 - next_done) - value
                    gae = delta + config["gamma"] * config["gae_lambda"] * (1 - next_done) * gae
                    return (gae, value, done), gae

                _, advantages = jax.lax.scan(
                    _get_advantages,
                    (jnp.zeros_like(last_val), last_val, last_done),
                    traj_batch,
                    reverse=True,
                    unroll=16,
                )
                return advantages, advantages + traj_batch.value

            advantages, targets = _calculate_gae(traj_batch, last_val, last_done)

            # UPDATE NETWORK
            def _update_epoch(update_state, unused):
                def _update_minbatch(carry, batch_info):
                    # With C-CHAIN the carry also holds the reference params
                    # (the policy one gradient step back) and the step counter;
                    # without it the carry is just the train state, exactly as
                    # before.
                    if use_cchain:
                        train_state, chain_ref, chain_step = carry
                        (init_hstate, traj_batch, advantages, targets), reg_info = batch_info
                        reg_hstate, reg_traj, _, _ = reg_info
                    else:
                        train_state = carry
                        init_hstate, traj_batch, advantages, targets = batch_info

                    def _loss_fn(params, init_hstate, traj_batch, gae, targets):
                        # RERUN NETWORK
                        _, pi, value, _ = apply_network(params, init_hstate[0], (traj_batch.obs, traj_batch.done))
                        log_prob = pi.log_prob(traj_batch.action)

                        # CALCULATE VALUE LOSS
                        value_pred_clipped = traj_batch.value + (value - traj_batch.value).clip(
                            -config["clip_eps"], config["clip_eps"]
                        )
                        value_losses = jnp.square(value - targets)
                        value_losses_clipped = jnp.square(value_pred_clipped - targets)
                        value_loss = 0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()

                        # CALCULATE ACTOR LOSS
                        ratio = jnp.exp(log_prob - traj_batch.log_prob)
                        gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                        loss_actor1 = ratio * gae
                        loss_actor2 = (
                            jnp.clip(
                                ratio,
                                1.0 - config["clip_eps"],
                                1.0 + config["clip_eps"],
                            )
                            * gae
                        )
                        loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
                        loss_actor = loss_actor.mean()
                        entropy = pi.entropy().mean()

                        total_loss = loss_actor + config["vf_coef"] * value_loss - config["ent_coef"] * entropy

                        if use_cchain:
                            # Pull the policy towards the one from a gradient
                            # step ago, measured on an INDEPENDENTLY shuffled
                            # minibatch of the same rollout. Suppressing the
                            # off-diagonal NTK entries this way is the whole
                            # mechanism -- using the training minibatch instead
                            # would not.
                            _, cur_pi, _, _ = apply_network(
                                params, reg_hstate[0], (reg_traj.obs, reg_traj.done))
                            _, ref_pi, _, _ = apply_network(
                                chain_ref, reg_hstate[0], (reg_traj.obs, reg_traj.done))
                            p_reg_loss = jnp.mean(chain_policy_churn(ref_pi, cur_pi))
                            # Inactive until two gradient steps have been taken,
                            # matching the reference's `len(his_policy_list) > 2`.
                            active = jnp.where(chain_step >= 2, 1.0, 0.0)
                            total_loss = total_loss + chain_coef * active * p_reg_loss
                            return total_loss, (value_loss, loss_actor, entropy,
                                                p_reg_loss * active)

                        return total_loss, (value_loss, loss_actor, entropy)

                    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                    total_loss, grads = grad_fn(train_state.params, init_hstate, traj_batch, advantages, targets)

                    if use_cchain:
                        # The reference for the next step is this step's params,
                        # captured before the update is applied.
                        next_ref = train_state.params
                        train_state = train_state.apply_gradients(grads=grads)
                        return (train_state, next_ref, chain_step + 1), total_loss

                    train_state = train_state.apply_gradients(grads=grads)
                    return train_state, total_loss

                (
                    train_state,
                    init_hstate,
                    traj_batch,
                    advantages,
                    targets,
                    rng,
                    chain_ref,
                    chain_step,
                ) = update_state
                batch = (init_hstate, traj_batch, advantages, targets)

                def _split_minibatches(perm_key):
                    """Shuffle over envs and cut into minibatches.

                    The permutation is over the env axis rather than over
                    timesteps because the policy is recurrent -- a minibatch has
                    to be whole trajectories, not scattered transitions.
                    """
                    perm = jax.random.permutation(perm_key, config["num_train_envs"])
                    shuffled = jax.tree.map(lambda x: jnp.take(x, perm, axis=1), batch)
                    return jax.tree.map(
                        lambda x: jnp.swapaxes(
                            jnp.reshape(
                                x,
                                [x.shape[0], config["num_minibatches"], -1] + list(x.shape[2:]),
                            ),
                            1,
                            0,
                        ),
                        shuffled,
                    )

                if use_cchain:
                    # Two independent shuffles: one to train on, one to measure
                    # churn on. The reference draws the regularisation batch
                    # from its own DataLoader with an independent shuffle, which
                    # is what makes the churn term an off-diagonal NTK estimate
                    # rather than a self-consistency penalty.
                    rng, _train_key, _reg_key = jax.random.split(rng, 3)
                    minibatches = (_split_minibatches(_train_key),
                                   _split_minibatches(_reg_key))
                    (train_state, chain_ref, chain_step), total_loss = jax.lax.scan(
                        _update_minbatch, (train_state, chain_ref, chain_step), minibatches
                    )
                else:
                    rng, _rng = jax.random.split(rng)
                    minibatches = _split_minibatches(_rng)
                    train_state, total_loss = jax.lax.scan(_update_minbatch, train_state, minibatches)

                update_state = (
                    train_state,
                    init_hstate,
                    traj_batch,
                    advantages,
                    targets,
                    rng,
                    chain_ref,
                    chain_step,
                )
                return update_state, total_loss

            init_hstate = initial_hstate[None, :]  # TBH
            update_state = (
                train_state,
                init_hstate,
                traj_batch,
                advantages,
                targets,
                rng,
                chain_ref,
                chain_step,
            )
            update_state, loss_info = jax.lax.scan(_update_epoch, update_state, None, config["update_epochs"])
            train_state = update_state[0]
            metric = jax.tree.map(
                lambda x: (x * traj_batch.info["returned_episode"]).sum() / traj_batch.info["returned_episode"].sum(),
                traj_batch.info,
            )
            rng = update_state[5]
            chain_ref, chain_step = update_state[6], update_state[7]

            if use_cchain:
                # loss_info aux is (value_loss, loss_actor, entropy, p_reg_loss)
                # with C-CHAIN; drive the coefficient from this update's means,
                # on the same scale the reference calibrates against (clipped
                # surrogate minus the entropy bonus).
                _, (_vl, _la, _ent, _preg) = loss_info
                _chain_p_loss = jnp.mean(_la) - config["ent_coef"] * jnp.mean(_ent)
                _chain_p_reg = jnp.mean(_preg)
                coef_state = update_coef_state(
                    coef_state, _chain_p_loss, _chain_p_reg,
                    target_rel_scale=chain_target_rel_scale,
                    warmup_updates=chain_warmup_updates,
                )
            else:
                _chain_p_reg = jnp.array(0.0, jnp.float32)

            # ---- ReDo: actually recycle the dormant neurons --------------
            # `use_redo` used only to switch the *counter* on; the network was
            # never modified, so the redo variant trained an identical net to
            # vanilla PPO. This applies the real thing every `redo_interval`
            # updates: resample dormant neurons' incoming weights, zero their
            # outgoing weights and their Adam moments (kinetix/models/redo.py).
            if use_redo:
                rng, _redo_key, _batch_key = jax.random.split(rng, 3)

                # Score on a fixed-size sample of the trajectory rather than all
                # of it: the full batch is num_steps * num_train_envs pixel
                # frames through the CNN, which is far more than the estimate
                # needs. Matches the gymnax trainer's --redo_batch_size 512.
                _steps = max(1, redo_batch_size // config["num_train_envs"])
                _steps = min(_steps, config["num_steps"])
                _idx = jax.random.choice(
                    _batch_key, config["num_steps"], (_steps,), replace=False
                )
                _obs_b = jax.tree.map(lambda x: x[_idx], traj_batch.obs)
                _done_b = traj_batch.done[_idx]

                def _do_redo(ts):
                    _, _mutated = network.apply(
                        ts.params, initial_hstate, (_obs_b, _done_b),
                        mutable=["intermediates"],
                    )
                    _ps, _vs = collect_scores(_mutated)
                    return apply_redo(
                        ts, _ps, _vs, _redo_key,
                        fc_layer_depth=config["fc_layer_depth"],
                        tau=config.get("redo_tau", REDO_DEFAULT_TAU),
                        targets=redo_targets,
                    )

                def _skip_redo(ts):
                    return ts, {
                        "redo_policy_reset": jnp.array(0, jnp.int32),
                        "redo_value_reset": jnp.array(0, jnp.int32),
                        "redo_total_neurons": jnp.array(_n_hidden_neurons, jnp.int32),
                    }

                train_state, redo_stats = jax.lax.cond(
                    ((update_step + 1) % redo_interval) == 0,
                    _do_redo, _skip_redo, train_state,
                )
            else:
                redo_stats = {
                    "redo_policy_reset": jnp.array(0, jnp.int32),
                    "redo_value_reset": jnp.array(0, jnp.int32),
                    "redo_total_neurons": jnp.array(_n_hidden_neurons, jnp.int32),
                }

            # Calculate average dormant neuron fractions (if monitoring or ReDo is enabled)
            if monitor_dormant or use_redo:
                policy_dormant_avg = jnp.where(n_counts > 0, policy_dormant_sum / n_counts, 0.0)
                value_dormant_avg = jnp.where(n_counts > 0, value_dormant_sum / n_counts, 0.0)
            else:
                policy_dormant_avg = 0.0
                value_dormant_avg = 0.0

            if config["use_wandb"]:

                def _fake_video():
                    return jnp.zeros(
                        (
                            env_params.max_timesteps,
                            config["num_eval_levels"],
                            *PixelObservations(env_params, render_static_env_params)
                            .observation_space(env_params)
                            .shape,
                        )
                    )

                def _real_eval(rng, update_step):
                    vid_frequency = get_video_frequency(config, update_step)
                    rng, _rng = jax.random.split(rng)
                    to_log_videos = _vmapped_eval_step(runner_state, _rng)
                    should_log_videos = update_step % vid_frequency == 0
                    first = jax.lax.cond(
                        should_log_videos,
                        lambda: jax.vmap(jax.vmap(pixel_render_fn))(to_log_videos[0].env_state),
                        lambda: (_fake_video()),
                    )
                    return (first, should_log_videos, True, *to_log_videos[1:])

                def _fake_eval(rng, update_step):

                    return (
                        _fake_video(),
                        False,
                        False,
                        jnp.zeros((config["num_eval_levels"],), jnp.int32),  # lengths
                        jnp.zeros((config["num_eval_levels"],), jnp.float32),  # returns for video
                        jnp.zeros((config["num_eval_levels"],), jnp.float32),  # returns avg
                        jnp.zeros((config["num_eval_levels"],), jnp.float32),  # ep lengths avg
                        jnp.zeros((config["num_eval_levels"],), jnp.float32),  # solve avg
                    )

                rng, _rng = jax.random.split(rng)
                to_log_videos = jax.lax.cond(
                    update_step % config["eval_freq"] == 0, _real_eval, _fake_eval, _rng, update_step
                )

                def callback(metric, raw_info, loss_info, update_step, to_log_videos, policy_dormant_avg, value_dormant_avg):
                    nonlocal last_time
                    time_now = time.time()
                    delta_time = time_now - last_time
                    last_time = time_now
                    dones = raw_info["returned_episode"]
                    to_log = {
                        "episode_return": (raw_info["returned_episode_returns"] * dones).sum()
                        / jnp.maximum(1, dones.sum()),
                        "episode_solved": (raw_info["returned_episode_solved"] * dones).sum()
                        / jnp.maximum(1, dones.sum()),
                        "episode_length": (raw_info["returned_episode_lengths"] * dones).sum()
                        / jnp.maximum(1, dones.sum()),
                        "num_completed_episodes": dones.sum(),
                    }
                    # Add dormant neuron metrics if monitoring or ReDo is enabled
                    if monitor_dormant or use_redo:
                        to_log["redo/policy_dormant_fraction"] = float(policy_dormant_avg)
                        to_log["redo/value_dormant_fraction"] = float(value_dormant_avg)
                    to_log["timing/num_updates"] = update_step
                    to_log["timing/num_env_steps"] = (
                        int(update_step) * int(config["num_steps"]) * int(config["num_train_envs"])
                    )
                    to_log["timing/sps"] = (config["num_steps"] * config["num_train_envs"]) / delta_time
                    to_log["timing/sps_agg"] = (to_log["timing/num_env_steps"]) / (time_now - time_start)
                    (
                        obs_vid,
                        should_log_videos,
                        should_log_eval,
                        idx_vid,
                        eval_return_vid,
                        eval_return_mean,
                        eval_eplen_mean,
                        eval_solverate_mean,
                    ) = to_log_videos

                    if should_log_eval:
                        to_log["eval/mean_eval_return"] = eval_return_mean.mean()
                        to_log["eval/mean_eval_eplen"] = eval_eplen_mean.mean()
                        to_log["eval/mean_eval_solve"] = eval_solverate_mean.mean()
                        for i, eval_name in enumerate(config["eval_levels"]):
                            return_on_video = eval_return_vid[i]
                            to_log[f"eval_video/return_{eval_name}"] = return_on_video
                            to_log[f"eval_video/len_{eval_name}"] = idx_vid[i]
                            to_log[f"eval_avg/return_{eval_name}"] = eval_return_mean[i]
                            to_log[f"eval_avg/solve_rate_{eval_name}"] = eval_solverate_mean[i]

                    if should_log_videos:
                        for i, eval_name in enumerate(config["eval_levels"]):
                            obs_to_use = obs_vid[: idx_vid[i], i]
                            obs_to_use = np.asarray(obs_to_use).transpose(0, 3, 2, 1)[:, :, ::-1, :]
                            to_log[f"media/eval_video_{eval_name}"] = wandb.Video(
                                (obs_to_use * 255).astype(np.uint8), fps=15
                            )

                    wandb.log(to_log)

                jax.debug.callback(callback, metric, traj_batch.info, loss_info, update_step, to_log_videos, policy_dormant_avg, value_dormant_avg)

            runner_state = (
                train_state,
                env_state,
                last_obs,
                last_done,
                hstate,
                rng,
                update_step + 1,
            )
            # Carry the dormancy fractions out of the scan as well. They used to
            # exist only inside the wandb callback above, so a run left no
            # dormancy series on disk and the ReDo-vs-PPO panel could not be
            # drawn from the project tree. Both are scalars per update, so this
            # stacks to (num_updates,) alongside `metric`.
            if use_cchain:
                _total_loss, (_value_loss, _loss_actor, _entropy, _) = loss_info
            else:
                _total_loss, (_value_loss, _loss_actor, _entropy) = loss_info
            aux = {
                "policy_dormant_fraction": jnp.asarray(policy_dormant_avg, dtype=jnp.float32),
                "value_dormant_fraction": jnp.asarray(value_dormant_avg, dtype=jnp.float32),
                # loss_info is (update_epochs, num_minibatches); average both so
                # each update contributes one number, as the gymnax trainer does.
                "total_loss": jnp.mean(_total_loss),
                "vf_loss": jnp.mean(_value_loss),
                "pg_loss": jnp.mean(_loss_actor),
                "entropy": jnp.mean(_entropy),
                # How many neurons ReDo actually recycled this update. Zero on
                # every non-interval update, and zero throughout for the other
                # variants -- which is what makes it a usable check that redo
                # is doing something and vanilla is not.
                "redo_policy_reset": redo_stats["redo_policy_reset"],
                "redo_value_reset": redo_stats["redo_value_reset"],
                # C-CHAIN diagnostics; constant zero / one for the other variants.
                "chain_p_reg_loss": jnp.asarray(_chain_p_reg, dtype=jnp.float32),
                "chain_p_reg_coef": jnp.asarray(coef_state.coef, dtype=jnp.float32),
            }
            return (runner_state, (chain_ref, chain_step, coef_state)), (metric, aux)

        rng, _rng = jax.random.split(rng)
        runner_state = (
            train_state,
            env_state,
            obsv,
            jnp.zeros((config["num_train_envs"]), dtype=bool),
            init_hstate,
            _rng,
            0,
        )
        # The reference params start as the initial params; `chain_step` gates
        # the regulariser off for the first two gradient steps regardless, so
        # this initial value is never actually used as a target.
        chain_carry = (
            train_state.params,
            jnp.array(0, dtype=jnp.int32),
            init_coef_state(chain_coef_window, chain_target_rel_scale),
        )
        (runner_state, chain_carry), (metric, aux) = jax.lax.scan(
            _update_step, (runner_state, chain_carry), None, config["num_updates"]
        )
        return {"runner_state": runner_state, "metric": metric, "aux": aux}

    return train


def _variant_name(config):
    """Method name for this run, used for both the output directory and the
    filenames inside it.

    One function rather than the two divergent copies this used to have, which
    disagreed on the vanilla case ("ppo" for the directory, "vanilla" for the
    checkpoint). The names match the gymnax tree so the figure code can read
    both benchmarks with one method list.
    """
    if config.get("use_cchain", False):
        return "cchain"
    if config.get("use_trac", False):
        return "trac"
    if config.get("use_redo", False):
        return "redo"
    return "ppo"


@hydra.main(version_base=None, config_path="../configs", config_name="ppo")
def main(config):
    temp = OmegaConf.to_container(config)

    config = normalise_config(OmegaConf.to_container(config), "PPO")
    
    continual = config.get("continual", False)
    
    # ---- interaction budget -------------------------------------------------
    # This block used to be three unconditional assignments that DISCARDED
    # whatever the config said. `learning/ppo-base.yaml` still carries
    # `total_timesteps: 1073741824`, and it never had the slightest effect --
    # every kinetix RL level ran 50000 * num_train_envs = 800,000 steps because
    # of the line below, and setting total_timesteps anywhere in a config or on
    # the CLI was silently ignored.
    #
    # That made the kinetix RL arm impossible to compute-match against the NE
    # arm without editing this file, which is exactly what compute-matching the
    # two families requires. The budget is now config-driven, and the DEFAULTS
    # REPRODUCE THE HISTORICAL NUMBERS EXACTLY: pass nothing and this is still
    # 800,000 steps over 50 updates of 1000-step rollouts.
    #
    #   +total_timesteps_override=16000000   raise the budget
    #   +num_steps_override=1000             rollout length, if it must change
    _HISTORICAL_STEPS_PER_ENV = 50000
    _HISTORICAL_NUM_UPDATES = 50
    # 50000*N / (50*N) == 1000 for any N, so the historical rollout length is
    # independent of num_train_envs.
    _HISTORICAL_ROLLOUT = _HISTORICAL_STEPS_PER_ENV // _HISTORICAL_NUM_UPDATES

    _budget = config.get("total_timesteps_override")
    config["total_timesteps"] = int(
        _budget if _budget else _HISTORICAL_STEPS_PER_ENV * config["num_train_envs"]
    )

    config["num_minibatches"] = config["num_minibatches"] * 4

    # num_steps is the ROLLOUT LENGTH, and it is pinned rather than derived.
    # Deriving it as total_timesteps / (50 * num_train_envs) forced exactly 50
    # updates at every budget, so a larger budget bought LONGER ROLLOUTS rather
    # than MORE GRADIENT STEPS -- the opposite of what "give PPO more compute"
    # should mean, and it would have made a compute-matched comparison measure
    # the wrong thing. Pinned, num_updates grows with the budget instead
    # (see make_train, which divides total_timesteps by num_steps).
    _rollout = config.get("num_steps_override")
    config["num_steps"] = int(_rollout if _rollout else _HISTORICAL_ROLLOUT)
    env_params, static_env_params = generate_params_from_config(config)
    config["env_params"] = to_state_dict(env_params)
    config["static_env_params"] = to_state_dict(static_env_params)
    
    # Project-style saving setup (similar to gymnax structure)
    # Set up output directory and logging BEFORE training starts
    # Try both top-level and misc. prefix since normalise_config may flatten
    project_dir = config.get("project_dir") or config.get("misc", {}).get("project_dir")
    trial_idx = config.get("trial_idx") or config.get("misc", {}).get("trial_idx", 1) or 1
    output_dir = None
    tee_logger = None
    
    if project_dir:
        variant = _variant_name(config)

        # Get environment name
        train_levels_list = config.get("train_levels_list", [])
        if train_levels_list:
            env_name = train_levels_list[0].split("/")[-1].replace(".json", "")
        else:
            env_name = "unknown"
        
        # Determine if continual or vanilla (non-continual)
        is_continual = config.get("continual", False)
        training_mode = "continual" if is_continual else "vanilla"
        
        # Create output directory structure: project_dir/{mode}/{variant}/{env}/trial_{n}/
        output_dir = os.path.join(project_dir, training_mode, variant, env_name, f"trial_{trial_idx}")
        os.makedirs(output_dir, exist_ok=True)
        gifs_dir = os.path.join(output_dir, "gifs")
        os.makedirs(gifs_dir, exist_ok=True)
        
        # Set up logging to train.log
        log_file = os.path.join(output_dir, "train.log")
        tee_logger = Tee(log_file)
        sys.stdout = tee_logger
        
        print(f"\n=== Kinetix PPO Training ===")
        print(f"  Mode: {training_mode}")
        print(f"  Variant: {variant}")
        print(f"  Environment: {env_name}")
        print(f"  Trial: {trial_idx}")
        print(f"  Seed: {config.get('seed', 0)}")
        print(f"  Output directory: {output_dir}")
        print(f"  Log file: {log_file}")
        print(f"==============================\n")

    # Modify run_name to include the training level, seed, and optimizer info (TRAC/REDO)
    if config["use_wandb"]:
        # Extract the environment name from train_levels_list
        train_levels_list = config.get("train_levels_list", [])
        if train_levels_list:
            # Get the first environment name (remove path and .json extension)
            env_name = train_levels_list[0].split("/")[-1].replace(".json", "")
            # Add trial index if provided
        trial_idx_wandb = config.get("trial_idx") or config.get("misc", {}).get("trial_idx", "")
        trial_suffix = f"_trial{trial_idx_wandb}" if trial_idx_wandb else ""
        seed = config.get("seed", 0)
        
        # Add TRAC and REDO info to run_name for checkpoint identification
        use_trac = config.get("use_trac", False)
        use_redo = config.get("use_redo", False)
        optimizer_suffix = f"_trac{int(use_trac)}_redo{int(use_redo)}"
        
        config["run_name"] = f"{config['run_name']}_{env_name}" + "_continual_" + str(continual) + trial_suffix + f"_seed{seed}" + optimizer_suffix

        # Keep the wandb artifact name under wandb's 128-character limit.
        #
        # save_dict() builds the artifact as
        #   f"{run_name}-{random_hash}-{timesteps}-checkpoint"
        # and silently truncates it to 128 if it is longer. The continual runner
        # then asks for the FULL, untruncated name, so the fetch 404s and the
        # next sub-task dies with "artifact membership ... not found" -- the
        # checkpoint chain breaks and the run is lost.
        #
        # It bit h19 (which loads h18_thrust_right_very_easy's checkpoint, the
        # longest level name) on trac, redo and cchain, but not on ppo: the
        # random hash varies in digit count, so whether the name crosses 128 is
        # luck. Capping it here removes the coin flip.
        #
        # The tail is kept rather than the head: the constant
        # "Kinetix-Pixels-MultiDiscrete-..." prefix carries no information,
        # while the tail holds env, trial, seed and variant, which are what make
        # the name unique.
        _suffix_budget = 25  # "-" + up to 10-digit hash + "-" + "0B" + "-checkpoint"
        _max_run_name = 128 - _suffix_budget
        if len(config["run_name"]) > _max_run_name:
            config["run_name"] = config["run_name"][-_max_run_name:]
        
        run = init_wandb(config, "PPO")
    
    print(f"Starting training...")
    start_time = time.time()

    rng = jax.random.PRNGKey(config["seed"])
    rng, _rng = jax.random.split(rng)
    train_jit = jax.jit(make_train(config, env_params, static_env_params))

    out = train_jit(_rng)
    
    total_time = time.time() - start_time
    print(f"\nTraining complete! Time: {total_time:.1f}s")

    if config["save_policy"]:
        train_state = jax.tree.map(lambda x: x, out["runner_state"][0])
        save_model(train_state, config["total_timesteps"], config, save_to_wandb=config["use_wandb"])
    
    # Project-style saving - save checkpoint, metrics, and GIFs
    if output_dir:
        # Get environment name (may have been set already)
        train_levels_list = config.get("train_levels_list", [])
        if train_levels_list:
            env_name = train_levels_list[0].split("/")[-1].replace(".json", "")
        else:
            env_name = "unknown"
        
        # Variant already set at the beginning -- recomputed with the same rule
        # so the checkpoint name matches the directory it is written into. This
        # branch used to say "vanilla" where the directory said "ppo", which put
        # vanilla_<env>_best.pkl inside .../ppo/ and wrote variant="vanilla" into
        # a training_metrics.json under .../ppo/.
        variant = _variant_name(config)
        use_trac = config.get("use_trac", False)
        use_redo = config.get("use_redo", False)

        gifs_dir = os.path.join(output_dir, "gifs")
        
        print(f"\n=== Saving outputs ===")
        
        # Save model checkpoint
        train_state = jax.tree.map(lambda x: x, out["runner_state"][0])
        ckpt_path = os.path.join(output_dir, f"{variant}_{env_name}_best.pkl")
        with open(ckpt_path, 'wb') as f:
            pickle.dump({
                'params': train_state.params,
                'opt_state': train_state.opt_state,
                'step': int(train_state.step),
                'seed': config["seed"],
                'use_trac': use_trac,
                'use_redo': use_redo,
            }, f)
        print(f"  Saved checkpoint: {ckpt_path}")
        
        # Save the per-update training history.
        #
        # This used to collapse every series to a `<key>_final` / `<key>_mean`
        # pair, which threw the learning curve away -- there was no way to plot
        # reward against timesteps from the project tree. `out["metric"]` is a
        # lax.scan stack of shape (num_updates,) per key, so the whole history
        # was always there; only the writer discarded it.
        #
        # Written as a LIST of per-update dicts, the same shape as
        # train_RL_gymnax_continual.py's training_metrics.json, so one figure
        # loader reads both benchmarks. `mean_reward` carries the name gymnax
        # uses; the Kinetix-native key is kept beside it.
        metrics = out.get("metric", {}) or {}
        aux = out.get("aux", {}) or {}

        def _series(source, key):
            if key not in source:
                return None
            arr = np.asarray(source[key]).reshape(-1)
            return arr if arr.size == config["num_updates"] else None

        returns = _series(metrics, "returned_episode_returns")
        lengths = _series(metrics, "returned_episode_lengths")
        solved = _series(metrics, "returned_episode_solved")
        p_dormant = _series(aux, "policy_dormant_fraction")
        v_dormant = _series(aux, "value_dormant_fraction")
        p_reset = _series(aux, "redo_policy_reset")
        v_reset = _series(aux, "redo_value_reset")
        chain_reg = _series(aux, "chain_p_reg_loss")
        chain_coef_s = _series(aux, "chain_p_reg_coef")
        entropy = _series(aux, "entropy")
        pg_loss = _series(aux, "pg_loss")
        vf_loss = _series(aux, "vf_loss")
        total_loss = _series(aux, "total_loss")

        steps_per_update = int(config["num_steps"]) * int(config["num_train_envs"])

        def _at(arr, i):
            return float(arr[i]) if arr is not None else None

        training_metrics = []
        for i in range(int(config["num_updates"])):
            training_metrics.append({
                "update": i + 1,
                "timestep": (i + 1) * steps_per_update,
                # The sub-task index is not known inside a single process: the
                # continual run is 20 chained processes, one level each. The
                # collector fills `task` in when it stitches them together.
                "task": None,
                "mean_reward": _at(returns, i),
                "episode_length": _at(lengths, i),
                "solve_rate": _at(solved, i),
                "returned_episode_returns": _at(returns, i),
                "entropy": _at(entropy, i),
                "pg_loss": _at(pg_loss, i),
                "vf_loss": _at(vf_loss, i),
                "total_loss": _at(total_loss, i),
                "policy_dormant_frac": _at(p_dormant, i),
                "value_dormant_frac": _at(v_dormant, i),
                "redo_policy_reset": _at(p_reset, i),
                "redo_value_reset": _at(v_reset, i),
                # Named as the gymnax trainer names them, so one figure loader
                # reads the C-CHAIN diagnostics from either benchmark.
                "chain_p_reg_loss": _at(chain_reg, i),
                "chain_p_reg_coef": _at(chain_coef_s, i),
            })

        metrics_path = os.path.join(output_dir, "training_metrics.json")
        with open(metrics_path, 'w') as f:
            json.dump(training_metrics, f, indent=2)
        print(f"  Saved metrics: {metrics_path} ({len(training_metrics)} updates)")

        # Run-level facts that used to share training_metrics.json with the
        # (now per-update) history. The collector reads this to build the
        # run's results.json.
        run_info = {
            "total_timesteps": int(config["total_timesteps"]),
            "num_updates": int(config["num_updates"]),
            "num_steps": int(config["num_steps"]),
            "num_train_envs": int(config["num_train_envs"]),
            "steps_per_update": steps_per_update,
            "seed": int(config["seed"]),
            "variant": variant,
            "env_name": env_name,
            "trial_idx": int(trial_idx),
            "continual": bool(config.get("continual", False)),
            "use_trac": bool(use_trac),
            "use_redo": bool(use_redo),
            "use_cchain": bool(config.get("use_cchain", False)),
            "load_from_checkpoint": config.get("load_from_checkpoint"),
            "final_mean_reward": _at(returns, -1) if returns is not None else None,
            "final_solve_rate": _at(solved, -1) if solved is not None else None,
            "elapsed_seconds": total_time,
        }
        with open(os.path.join(output_dir, "run_info.json"), 'w') as f:
            json.dump(run_info, f, indent=2)
        print(f"  Saved run info: {os.path.join(output_dir, 'run_info.json')}")
        
        # Run 10 eval trajectories and save GIFs
        print("\n  Generating 10 eval trajectory GIFs...")
        
        # Get the trained policy and create eval environment
        env_params_loaded, static_env_params_loaded = generate_params_from_config(config)
        eval_env = make_kinetix_env(
            observation_type=config["observation_type"],
            action_type=config["action_type"],
            reset_fn=make_reset_fn_from_config(config, env_params_loaded, static_env_params_loaded),
            static_env_params=static_env_params_loaded,
        )
        
        # Renderer for GIFs
        render_static_env_params = eval_env.static_env_params.replace(downscale=4)
        pixel_renderer = jax.jit(make_render_pixels(env_params_loaded, render_static_env_params))
        
        # Create network once and jit the inference functions
        network = make_network_from_config(eval_env, env_params_loaded, config)
        
        @jax.jit
        def get_action(params, hstate, obs, done, rng):
            """Get action using RNN network with proper input formatting."""
            # Format input: (obs, done) with leading sequence dimension
            x = jax.tree.map(lambda x: x[None, ...], (obs, done))
            # Apply network
            new_hstate, pi, _, _ = network.apply(params, hstate, x)
            # Sample action
            action = pi.sample(seed=rng).squeeze(0)
            return action, new_hstate
        
        @jax.jit
        def env_step(rng, state, action):
            return eval_env.step(rng, state, action, env_params_loaded)
        
        @jax.jit
        def env_reset(rng):
            return eval_env.reset(rng, env_params_loaded)
        
        for eval_idx in range(10):
            print(f"    Eval trajectory {eval_idx}...", end="")
            eval_rng = jax.random.PRNGKey(config["seed"] * 100 + eval_idx)
            
            # Reset environment
            obs, env_state = env_reset(eval_rng)
            
            # Initialize hidden state for single env
            hstate = ScannedRNN.initialize_carry(1)
            done = jnp.zeros(1, dtype=jnp.bool_)
            
            frames = []
            total_reward = 0.0
            
            # Run episode
            for step in range(env_params_loaded.max_timesteps):
                # Render frame (transpose and flip to correct orientation)
                frame = np.array(pixel_renderer(env_state))
                frame = frame.transpose(1, 0, 2)[::-1].astype(np.uint8)
                frames.append(frame)
                
                # Get action from trained policy (jitted)
                eval_rng, action_rng = jax.random.split(eval_rng)
                # Add batch dimension to obs for single env
                obs_batched = jax.tree.map(lambda x: x[None, ...], obs)
                action, hstate = get_action(train_state.params, hstate, obs_batched, done, action_rng)
                # Remove batch dimension from action
                action = action.squeeze(0) if hasattr(action, 'squeeze') else action[0]
                
                # Step environment (jitted)
                eval_rng, step_rng = jax.random.split(eval_rng)
                obs, env_state, reward, step_done, info = env_step(step_rng, env_state, action)
                total_reward += float(reward)
                
                if bool(step_done):
                    # Render final frame
                    frame = np.array(pixel_renderer(env_state))
                    frame = frame.transpose(1, 0, 2)[::-1].astype(np.uint8)
                    frames.append(frame)
                    break
            
            # Save GIF
            gif_path = os.path.join(gifs_dir, f"eval_{eval_idx:02d}_reward{int(total_reward)}.gif")
            imageio.mimsave(gif_path, frames, fps=15, loop=0)
            print(f" reward={total_reward:.1f}, saved to {gif_path}")
        
        print(f"\n=== Project saving complete! ===")
        
        # Close the Tee logger
        if tee_logger:
            sys.stdout = tee_logger.stdout
            tee_logger.close()


if __name__ == "__main__":
    main()
