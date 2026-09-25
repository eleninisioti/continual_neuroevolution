"""Policy and value networks, and the flat-vector view of their parameters.

The NE methods search in a flat parameter vector and the RL methods keep a
pytree, so both representations are needed and `get_flat_params` /
`unflatten_params` are the bridge between them.

`MLPPolicy` is the discrete-action policy. `PolicyNetwork` is an alias of it,
not a copy: gymnax's NE and RL trainers must search the *same* architecture or
the comparison measures capacity rather than method, and two identical class
definitions are two things that can drift. `ContinuousMLPPolicy` is the
cheetah/ant counterpart. `ValueNetwork` is PPO's critic and is deliberately NOT
matched to the policy -- it has no NE counterpart, so there is nothing to match
it to, and shrinking it would handicap PPO for no comparability gain.

`POLICY_ARCH` below is the single place the per-suite policy architecture is
written down. Read `scripts/check_architectures.py` for what enforces it.
"""

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from jax import flatten_util


# The policy architecture every method in a suite must use.
#
# Fairness across methods is per (suite, env): when a figure puts GA, ES, DNS,
# PPO, ReDo, TRAC, C-CHAIN and PBT-PPO on one axis, they must all be searching
# the same policy. Across suites the architectures differ because the tasks do
# (discrete gymnax logits against continuous cheetah/ant torques against
# kinetix's permutation-invariant encoder), and nothing compares across suites.
#
# `activation` is the HIDDEN activation. The output layer differs by action
# space, not by method: gymnax emits logits, cheetah/ant tanh-squash to the
# actuator range.
#
# Changing an entry here invalidates every run of that suite -- it is a method
# change, not a knob. See CLAUDE.md.
POLICY_ARCH = {
    'gymnax':  {'hidden_dims': (16, 16),   'activation': 'relu'},
    'mujoco':  {'hidden_dims': (128, 128), 'activation': 'tanh'},
    # brax is 'swish' as of 2026-08-11, and the reason is measured rather than
    # aesthetic: PPO CANNOT LEARN THE ANT ON TANH AT THE SMALL-BATCH SHAPE.
    #
    # Same trainer, same seed, same env (friction pinned 1.0, speed target
    # 2.0), same 512/16/32/10/5 rollout, only the activation differing:
    #
    #     swish   end of sub-task 0  4868.6   sub-task 1  3967.4
    #     tanh    end of sub-task 0  1157.5   sub-task 1   475.4
    #
    # against 4110.1 / 2857.6 for the reported run this reproduces. Tanh does
    # not merely learn worse, it falls from 1157 to 475 across a boundary where
    # NOTHING CHANGES -- a within-task collapse, not forgetting. It is fine at
    # brax's big-batch shape (4780), so what fails is the interaction of tanh
    # with 10 update epochs over 80-transition minibatches.
    #
    # This entry was 'tanh' from the day `--activation` became a flag, chosen so
    # PPO would search the same network as GA/ES/DNS. Every run in
    # `projects/neurips_2026_rebuttal/runs/` predates that flag and therefore
    # ran make_ppo_networks' swish default on the RL side while NE ran tanh --
    # so the reported trees are internally MISMATCHED, and no single value here
    # reproduces both halves of them. Choosing swish keeps the fairness property
    # this table exists for and reproduces the RL half; the NE arms are re-run
    # under it and are new numbers, not a reproduction.
    'brax':    {'hidden_dims': (128, 128), 'activation': 'swish'},
    # kinetix is fc_layer_depth x fc_layer_width from configs/model/model-base.yaml,
    # shared by the NE and PPO trainers; see check_architectures.py.
    'kinetix': {'hidden_dims': (128,) * 5, 'activation': 'tanh'},
    # MiniGrid, since 2026-09-07 (`source/envs/minigrid.py`): the
    # `GridConvPolicy` of Young & Tian 2019 -- one 3x3 valid convolution,
    # relu, one Dense, relu, then the head -- on the one-hot planes of
    # xminigrid's 7x7 view, at conv 4 / Dense 64 (7,758 parameters).
    # `hidden_dims` records the Dense widths; the convolution is
    # `GridConvPolicy.conv_features`, a fixed part of the architecture
    # rather than a knob. Shared by the NE and RL arms, like every entry here.

    'minigrid': {'hidden_dims': (64,), 'activation': 'relu'},
}

ACTIVATIONS = {'relu': nn.relu, 'tanh': nn.tanh, 'swish': nn.swish}


def policy_arch(suite):
    """(hidden_dims, activation_fn) for a suite. Raises on an unknown suite."""
    cfg = POLICY_ARCH[suite]
    return cfg['hidden_dims'], ACTIVATIONS[cfg['activation']]


class MLPPolicy(nn.Module):
    """MLP policy that outputs logits for discrete action selection."""
    hidden_dims: tuple = (16, 16)
    action_dim: int = 2

    @nn.compact
    def __call__(self, x):
        for dim in self.hidden_dims:
            x = nn.Dense(dim)(x)
            x = nn.relu(x)
        x = nn.Dense(self.action_dim)(x)
        return x  # logits


class ContinuousMLPPolicy(nn.Module):
    """Deterministic continuous-action policy, tanh hidden and tanh output.

    The cheetah and ant NE policy. It was defined identically and separately in
    six trainers; it lives here so the RL side has one thing to match.
    """
    hidden_dims: tuple = (128, 128)
    action_dim: int = 6

    @nn.compact
    def __call__(self, x):
        for dim in self.hidden_dims:
            x = nn.Dense(dim)(x)
            x = nn.tanh(x)
        x = nn.Dense(self.action_dim)(x)
        x = nn.tanh(x)
        return x


class GridConvPolicy(nn.Module):
    """A small conv net as a discrete-action policy on a FLAT grid observation.

    Young & Tian 2019's network: Conv(16, 3x3, valid) -> relu -> flatten ->
    Dense(128) -> relu -> Dense(actions). It is the MiniGrid suite's policy,
    run at conv 4 / Dense 64 on the one-hot planes of xminigrid's 7x7 view. Everything that consumes an
    observation in this repo -- the NE scoring functions, PPO's rollout, the
    observation-normaliser, the churn and dormancy probes, AURORA's
    trajectories -- takes a VECTOR, so the observation is carried flattened
    (``height * width * channels``) and this module reshapes it back to the
    image on the way in. The reshape is free and it is what lets a run on a
    gridded observation go through both trainers unchanged.

    Flax names the layers ``Conv_0``, ``Dense_0``, ``Dense_1``. ``LAYERS``
    lists them in forward order, hidden ones then the head, and
    ``hidden_activations`` returns the hidden ones' post-activation outputs;
    together they are what the dormancy diagnostics and ReDo walk, since the
    Dense-chain replay in `redo.hidden_activations` cannot walk a convolution.
    """
    obs_shape: tuple = (10, 10, 10)
    conv_features: int = 16
    hidden_dims: tuple = (128,)
    action_dim: int = 6

    LAYERS = ('Conv_0', 'Dense_0', 'Dense_1')

    @nn.compact
    def __call__(self, x):
        x = x.reshape(x.shape[:-1] + tuple(self.obs_shape))
        x = nn.relu(nn.Conv(self.conv_features, (3, 3), padding='VALID')(x))
        self.sow('intermediates', 'conv', x)
        # (..., 8, 8, 16) -> (..., 1024), channel fastest: the outgoing rows
        # of Dense_0 for conv channel c are every 16th row from c, which is
        # what `redo._expand_mask` relies on.
        x = x.reshape(x.shape[:-3] + (-1,))
        for dim in self.hidden_dims:
            x = nn.relu(nn.Dense(dim)(x))
            self.sow('intermediates', 'dense', x)
        return nn.Dense(self.action_dim)(x)

    def hidden_activations(self, params, x):
        """``[conv, dense]`` post-activation outputs, for the dormancy probes."""
        _, state = self.apply(params, x, mutable=['intermediates'])
        # `sow` collects every value stored under one name into a tuple, so
        # `dense` holds one entry per hidden Dense, in order.
        inter = state['intermediates']
        return [inter['conv'][0]] + list(inter['dense'])


def get_flat_params(params):
    flat_params, _ = flatten_util.ravel_pytree(params)
    return flat_params


def unflatten_params(flat_params, param_template):
    _, unravel_fn = flatten_util.ravel_pytree(param_template)
    return unravel_fn(flat_params)


def create_policy_network(key, obs_dim, action_dim, hidden_dims=(16, 16)):
    policy = MLPPolicy(hidden_dims=hidden_dims, action_dim=action_dim)
    dummy_obs = jnp.zeros((obs_dim,))
    params = policy.init(key, dummy_obs)
    return policy, params


def create_grid_conv_policy_network(key, obs_shape, action_dim,
                                  hidden_dims=(128,), conv_features=16):
    policy = GridConvPolicy(obs_shape=tuple(obs_shape),
                              conv_features=conv_features,
                              hidden_dims=tuple(hidden_dims),
                              action_dim=action_dim)
    dummy_obs = jnp.zeros((int(jnp.prod(jnp.asarray(obs_shape))),))
    params = policy.init(key, dummy_obs)
    return policy, params


def create_continuous_policy_network(key, obs_dim, action_dim,
                                     hidden_dims=(128, 128)):
    policy = ContinuousMLPPolicy(hidden_dims=hidden_dims, action_dim=action_dim)
    dummy_obs = jnp.zeros((obs_dim,))
    params = policy.init(key, dummy_obs)
    return policy, params


# PPO's actor on gymnax is the evolved policy, not a copy of it. It was a
# separate class with a byte-identical body; aliasing removes the only way the
# two could come to differ. Flax names parameters Dense_0.. from the module
# body, not the class, so this is bit-identical to the class it replaces.
PolicyNetwork = MLPPolicy


class ValueNetwork(nn.Module):
    """Value network for state value estimation."""
    hidden_dims: tuple = (128, 128, 128)

    @nn.compact
    def __call__(self, x):
        for dim in self.hidden_dims:
            x = nn.Dense(dim)(x)
            x = nn.relu(x)
        value = nn.Dense(1)(x)
        return value.squeeze(-1)


class KinetixPixelsPolicy(nn.Module):
    """Kinetix's ``ActorOnlyPixelsRNN`` as a STATELESS flat-observation policy.

    The Kinetix study evolved `kinetix.models.actor_critic.ActorOnlyPixelsRNN`
    at ``recurrent_model: false`` -- 1,128,256 parameters, and the config the
    2026-08 GA runs solved all twenty hand-designed levels under. This is that
    same network with the two things the shared runners cannot take removed,
    and NOTHING else:

    * **the carry.** With ``recurrent=False`` Kinetix's module threads `hidden`
      straight through untouched (`GeneralActorOnlyRNN.__call__`), so a policy
      built from it is already stateless and the `(hidden, x)` signature is
      dead weight. Dropping it is what lets `policy.apply(params, obs)` -- the
      one call every scoring function, PPO rollout, churn probe and dormancy
      probe in this repo makes -- work here as it does on every other body.
    * **the time/batch axes.** Kinetix's network is written for `(T, B, ...)`;
      the callers here pass one observation.

    The layers, their order, their widths, their initialisers and their
    activation are Kinetix's, so the parameter count is 1,128,256 exactly --
    which is the check `source/envs/kinetix.py` asserts at build time. The
    observation arrives FLAT, as `GridConvPolicy`'s does and for the same
    reason: `image.ravel()` then `global_info`, split and reshaped here.

    Emits ``sum(dims_per_distribution)`` logits (16 at the medium env size:
    four motor bindings x 3, two thruster bindings x 2), which
    `source/studies/generalists/actors.py:MULTI_DISCRETE` reads as six
    independent categoricals. That head is the ONLY thing PPO needed for this
    body; it is a new action space, not a second PPO.
    """
    image_shape: tuple = (125, 125, 3)
    global_info_dim: int = 1
    hidden_dims: tuple = (128,) * 5
    action_dim: int = 16

    LAYERS = ('Conv_0', 'Conv_1', 'Dense_0', 'Dense_1', 'Dense_2', 'Dense_3',
              'Dense_4', 'Dense_5')

    @nn.compact
    def __call__(self, x):
        # Plain Python arithmetic on a static field: `jnp.prod` of it would be
        # a traced array inside a scan, and `int()` of that raises.
        n_image = 1
        for dim in self.image_shape:
            n_image *= int(dim)
        image = x[..., :n_image].reshape(x.shape[:-1] + tuple(self.image_shape))
        global_info = x[..., n_image:]

        image = nn.relu(nn.Conv(features=16, kernel_size=(8, 8),
                                strides=(4, 4))(image))
        self.sow('intermediates', 'conv', image)
        image = nn.relu(nn.Conv(features=32, kernel_size=(4, 4),
                                strides=(2, 2))(image))
        self.sow('intermediates', 'conv', image)
        # (..., 16, 16, 32) -> (..., 8192), channel fastest, as GridConvPolicy.
        h = image.reshape(image.shape[:-3] + (-1,))
        h = jnp.concatenate([h, global_info], axis=-1)

        for dim in self.hidden_dims:
            h = nn.Dense(dim, kernel_init=nn.initializers.orthogonal(2.0 ** 0.5),
                         bias_init=nn.initializers.constant(0.0))(h)
            h = nn.tanh(h)
            self.sow('intermediates', 'dense', h)
        return nn.Dense(self.action_dim,
                        kernel_init=nn.initializers.orthogonal(0.01),
                        bias_init=nn.initializers.constant(0.0))(h)

    def hidden_activations(self, params, x):
        """``[conv0, conv1, dense0..dense4]``, for the dormancy probes."""
        _, state = self.apply(params, x, mutable=['intermediates'])
        inter = state['intermediates']
        return list(inter['conv']) + list(inter['dense'])

    def extra_fan_in(self):
        """Inputs of a layer that are no hidden unit's output, for ReDo.

        ``Dense_0`` reads the 8192 flattened conv features AND the
        ``global_info`` scalars appended after them, so its fan-in is not a
        multiple of ``Conv_1``'s 32 channels; `redo._expand_mask` pads the
        channel mask with this many unmasked trailing inputs.
        """
        return {'Dense_0': int(self.global_info_dim)}


def create_kinetix_pixels_policy_network(key, image_shape, global_info_dim,
                                         action_dim, hidden_dims=(128,) * 5):
    policy = KinetixPixelsPolicy(image_shape=tuple(image_shape),
                                 global_info_dim=int(global_info_dim),
                                 hidden_dims=tuple(hidden_dims),
                                 action_dim=int(action_dim))
    obs_dim = int(np.prod(image_shape)) + int(global_info_dim)
    params = policy.init(key, jnp.zeros((obs_dim,)))
    return policy, params


class KinetixTransformerPolicy(nn.Module):
    """Kinetix's ``ActorCriticTransformer``, actor half, over a FLAT entity
    observation.

    Kinetix's second network (Matthews et al. 2025, `configs/model/model-
    transformer.yaml`, the ``symbolic_entity`` observation): every circle and
    polygon is a token, encoded by a per-type Dense; two gated self-attention
    layers run over the shape tokens plus a constant dummy token, with
    joints and thrusters passing messages into the shapes they connect; the
    scene embedding is the masked mean of the tokens; and the actor head on top
    is the same ``fc_layer_depth x fc_layer_width`` tanh MLP the pixel policy
    has. The attention stack is Kinetix's own `Transformer` module, imported,
    not copied; the encoders, the dummy token, the mask handling and the head
    are `ActorCriticTransformer.__call__` line for line, with the critic half,
    the carry and the (T, B) axes removed for the same reasons as
    `KinetixPixelsPolicy` (see its docstring).

    The observation arrives FLAT: ``entity_layout`` is the ``(name, shape)``
    list of `EntityObservation`'s fields in the order the suite concatenated
    them (`source/envs/kinetix.py:ENTITY_LAYOUT`). Masks come back as ``> 0.5``
    and indexes as rounded int32 -- they were exact small integers as floats.

    ``LAYERS`` and ``hidden_activations`` expose the actor HEAD only
    (``Dense_0 .. Dense_{depth}``): that is the Dense chain ReDo can recycle
    and the dormancy probes read, as on every MLP body. The attention trunk has
    no per-unit notion ReDo defines, so it is measured by the weight and churn
    rows and left alone by recycling.
    """
    entity_layout: tuple
    hidden_dims: tuple = (128,) * 5
    action_dim: int = 16
    encoder_size: int = 128           # transformer_encoder_size
    num_heads: int = 8
    qkv_features: int = 16            # transformer_size
    num_layers: int = 2               # transformer_depth

    @property
    def LAYERS(self):
        return tuple(f'Dense_{i}' for i in range(len(self.hidden_dims) + 1))

    def _unflatten(self, x):
        out, i = {}, 0
        for name, shape in self.entity_layout:
            n = 1
            for d in shape:
                n *= int(d)
            out[name] = x[..., i:i + n].reshape(x.shape[:-1] + tuple(shape))
            i += n
        for name in ('circle_mask', 'polygon_mask', 'joint_mask',
                     'thruster_mask', 'attention_mask'):
            out[name] = out[name] > 0.5
        for name in ('joint_indexes', 'thruster_indexes'):
            out[name] = jnp.round(out[name]).astype(jnp.int32)
        return out

    @nn.compact
    def __call__(self, x):
        from flax.linen.initializers import constant, orthogonal
        from kinetix.models.transformer_model import Transformer

        batch_shape = x.shape[:-1]
        # Kinetix's modules are written for (T, B, ...): one step, flat batch.
        obs = self._unflatten(x.reshape((1, -1, x.shape[-1])))
        act = nn.tanh

        def encoder(features, entity_id, concat, name):
            width = self.encoder_size - (1 if concat else 0)
            emb = act(nn.Dense(width, kernel_init=orthogonal(np.sqrt(2)),
                               bias_init=constant(0.0), name=name)(features))
            if concat:
                # Kinetix's `id_1h`: one extra column holding the type id.
                emb = jnp.concatenate(
                    [emb, jnp.full(emb.shape[:-1] + (1,), float(entity_id))],
                    axis=-1)
            return emb

        circle_enc = encoder(obs['circles'], 0, True, 'enc_circle')
        polygon_enc = encoder(obs['polygons'], 1, True, 'enc_polygon')
        joint_enc = encoder(obs['joints'], -1, False, 'enc_joint')
        thruster_enc = encoder(obs['thrusters'], -1, False, 'enc_thruster')

        shape_enc = jnp.concatenate([polygon_enc, circle_enc], axis=2)
        shape_mask = jnp.concatenate([obs['polygon_mask'],
                                      obs['circle_mask']], axis=2)

        # aggregate_mode 'dummy_and_mean': a constant dummy token that every
        # active shape attends to and from, then the masked mean.
        T, B, _, K = circle_enc.shape
        shape_enc = jnp.concatenate([jnp.ones((T, B, 1, K)), shape_enc], axis=2)
        shape_mask = jnp.concatenate([jnp.ones((T, B, 1), dtype=bool),
                                      shape_mask], axis=2)
        attn = obs['attention_mask']
        n = attn.shape[-1]
        overall = (jnp.ones((T, B, attn.shape[2], n + 1, n + 1), dtype=bool)
                   .at[:, :, :, 1:, 1:].set(attn))

        def mask_out_inactives(active, matrix):
            return matrix & active[:, None] & active[None, :]

        overall = jax.vmap(jax.vmap(mask_out_inactives))(shape_mask, overall)

        emb = Transformer(num_layers=self.num_layers, num_heads=self.num_heads,
                          qkv_features=self.qkv_features,
                          encoder_size=self.encoder_size, gating=True,
                          gating_bias=0.0, name='transformer')(
            shape_enc,
            jnp.repeat(overall, repeats=self.num_heads // overall.shape[2],
                       axis=2),
            joint_enc, obs['joint_mask'], obs['joint_indexes'] + 1,
            thruster_enc, obs['thruster_mask'], obs['thruster_indexes'] + 1)
        h = jnp.mean(emb, axis=2, where=shape_mask[..., None])[0]   # (B, K)

        for i, dim in enumerate(self.hidden_dims):
            h = act(nn.Dense(dim, kernel_init=orthogonal(np.sqrt(2)),
                             bias_init=constant(0.0), name=f'Dense_{i}')(h))
            self.sow('intermediates', 'dense', h)
        logits = nn.Dense(self.action_dim, kernel_init=orthogonal(0.01),
                          bias_init=constant(0.0),
                          name=f'Dense_{len(self.hidden_dims)}')(h)
        return logits.reshape(batch_shape + (self.action_dim,))

    def hidden_activations(self, params, x):
        """``[dense0 .. dense{depth-1}]`` of the actor head, (batch, width)."""
        _, state = self.apply(params, x, mutable=['intermediates'])
        return list(state['intermediates']['dense'])


def create_kinetix_transformer_policy_network(key, entity_layout, action_dim,
                                              hidden_dims=(128,) * 5):
    policy = KinetixTransformerPolicy(entity_layout=tuple(
        (name, tuple(shape)) for name, shape in entity_layout),
        hidden_dims=tuple(hidden_dims), action_dim=int(action_dim))
    obs_dim = sum(int(np.prod(shape)) for _, shape in entity_layout)
    params = policy.init(key, jnp.zeros((obs_dim,)))
    return policy, params
