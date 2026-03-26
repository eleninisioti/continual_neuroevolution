import functools
from typing import List, Sequence

import distrax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from flax.linen.initializers import constant, orthogonal

from kinetix.environment.utils import ActionType
from kinetix.models.action_spaces import HybridActionDistribution, MultiDiscreteActionDistribution


class ScannedRNN(nn.Module):
    @functools.partial(
        nn.scan,
        variable_broadcast="params",
        in_axes=0,
        out_axes=0,
        split_rngs={"params": False},
    )
    @nn.compact
    def __call__(self, carry, x):
        """Applies the module."""
        rnn_state = carry
        ins, resets = x
        rnn_state = jnp.where(
            resets[:, np.newaxis],
            self.initialize_carry(ins.shape[0], 256),
            rnn_state,
        )
        new_rnn_state, y = nn.GRUCell(features=256)(rnn_state, ins)
        return new_rnn_state, y

    @staticmethod
    def initialize_carry(batch_size, hidden_size=256):
        # Use a dummy key since the default state init fn is just zeros.
        cell = nn.GRUCell(features=256)
        return cell.initialize_carry(jax.random.PRNGKey(0), (batch_size, hidden_size))


class GeneralActorCriticRNN(nn.Module):
    action_dim: Sequence[int]
    fc_layer_depth: int
    fc_layer_width: int
    action_type: ActionType
    hybrid_action_continuous_dim: int
    multi_discrete_number_of_dims_per_distribution: List[int]
    add_generator_embedding: bool = False
    generator_embedding_number_of_timesteps: int = 10
    recurrent: bool = False
    count_dormant: bool = False  # Enable dormant neuron counting for ReDo-PPO
    redo_tau: float = 0.01  # Threshold for dormant neuron detection (as fraction of layer mean)

    # Given an embedding, return the action/values, since this is shared across all models.
    @nn.compact
    def __call__(self, hidden, obs, embedding, dones, activation):

        if self.add_generator_embedding:
            raise NotImplementedError()

        if self.recurrent:
            rnn_in = (embedding, dones)
            hidden, embedding = ScannedRNN()(hidden, rnn_in)

        actor_mean = embedding
        critic = embedding
        actor_mean_last = embedding
        
        # Track dormant neurons for policy and value networks
        policy_n_dormant = 0
        policy_n_neurons = 0
        value_n_dormant = 0
        value_n_neurons = 0
        
        for _ in range(self.fc_layer_depth):
            # Policy network forward pass
            actor_mean = nn.Dense(
                self.fc_layer_width,
                kernel_init=orthogonal(np.sqrt(2)),
                bias_init=constant(0.0),
            )(actor_mean)
            actor_mean = activation(actor_mean)
            
            # Count dormant neurons in policy network if enabled
            if self.count_dormant:
                abs_act = jnp.abs(actor_mean)
                # Average over all dimensions except the last (neuron dimension)
                if len(abs_act.shape) > 1:
                    batch_dims = tuple(range(len(abs_act.shape) - 1))
                    mean_act_per_neuron = jnp.mean(abs_act, axis=batch_dims)
                    layer_mean_act = jnp.mean(mean_act_per_neuron)
                else:
                    mean_act_per_neuron = abs_act
                    layer_mean_act = jnp.mean(abs_act)
                layer_mean_safe = jnp.maximum(layer_mean_act, 1e-8)
                per_neuron_act = mean_act_per_neuron / layer_mean_safe
                dormant_mask = (per_neuron_act <= self.redo_tau)
                policy_n_dormant += jnp.sum(dormant_mask.astype(jnp.int32))
                policy_n_neurons += actor_mean.shape[-1]

            # Value network forward pass
            critic = nn.Dense(
                self.fc_layer_width,
                kernel_init=orthogonal(np.sqrt(2)),
                bias_init=constant(0.0),
            )(critic)
            critic = activation(critic)
            
            # Count dormant neurons in value network if enabled
            if self.count_dormant:
                abs_crit = jnp.abs(critic)
                if len(abs_crit.shape) > 1:
                    batch_dims = tuple(range(len(abs_crit.shape) - 1))
                    mean_act_per_neuron = jnp.mean(abs_crit, axis=batch_dims)
                    layer_mean_act = jnp.mean(mean_act_per_neuron)
                else:
                    mean_act_per_neuron = abs_crit
                    layer_mean_act = jnp.mean(abs_crit)
                layer_mean_safe = jnp.maximum(layer_mean_act, 1e-8)
                per_neuron_act = mean_act_per_neuron / layer_mean_safe
                dormant_mask = (per_neuron_act <= self.redo_tau)
                value_n_dormant += jnp.sum(dormant_mask.astype(jnp.int32))
                value_n_neurons += critic.shape[-1]

        actor_mean_last = actor_mean
        actor_mean = nn.Dense(self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0))(actor_mean)
        if self.action_type == ActionType.DISCRETE:
            pi = distrax.Categorical(logits=actor_mean)
        elif self.action_type == ActionType.CONTINUOUS:
            actor_logtstd = self.param("log_std", nn.initializers.zeros, (self.action_dim,))
            pi = distrax.MultivariateNormalDiag(actor_mean, jnp.exp(actor_logtstd))
        elif self.action_type == ActionType.MULTI_DISCRETE:
            pi = MultiDiscreteActionDistribution(actor_mean, self.multi_discrete_number_of_dims_per_distribution)
        else:
            actor_mean_continuous = nn.Dense(
                self.hybrid_action_continuous_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
            )(actor_mean_last)
            actor_mean_sigma = jnp.exp(
                nn.Dense(self.hybrid_action_continuous_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0))(
                    actor_mean_last
                )
            )
            pi = HybridActionDistribution(actor_mean, actor_mean_continuous, actor_mean_sigma)

        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(critic)
        value = jnp.squeeze(critic, axis=-1)
        
        # Always return dormant counts for consistency (set to 0.0 when not counting)
        if self.count_dormant:
            policy_dormant_fraction = jnp.where(
                policy_n_neurons > 0, 
                policy_n_dormant / policy_n_neurons, 
                0.0
            )
            value_dormant_fraction = jnp.where(
                value_n_neurons > 0,
                value_n_dormant / value_n_neurons,
                0.0
            )
        else:
            policy_dormant_fraction = 0.0
            value_dormant_fraction = 0.0
        
        return hidden, pi, value, (policy_dormant_fraction, value_dormant_fraction)


class ActorCriticPixelsRNN(nn.Module):

    action_dim: Sequence[int]
    fc_layer_depth: int
    fc_layer_width: int
    action_mode: str
    hybrid_action_continuous_dim: int
    multi_discrete_number_of_dims_per_distribution: List[int]
    activation: str
    add_generator_embedding: bool = False
    generator_embedding_number_of_timesteps: int = 10
    recurrent: bool = True
    count_dormant: bool = False
    redo_tau: float = 0.01

    @nn.compact
    def __call__(self, hidden, x, **kwargs):
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh
        og_obs, dones = x

        if self.add_generator_embedding:
            obs = og_obs.obs
        else:
            obs = og_obs

        image = obs.image
        global_info = obs.global_info

        x = nn.Conv(features=16, kernel_size=(8, 8), strides=(4, 4))(image)
        x = nn.relu(x)
        x = nn.Conv(features=32, kernel_size=(4, 4), strides=(2, 2))(x)
        x = nn.relu(x)
        embedding = x.reshape(x.shape[0], x.shape[1], -1)

        embedding = jnp.concatenate([embedding, global_info], axis=-1)

        return GeneralActorCriticRNN(
            action_dim=self.action_dim,
            fc_layer_depth=self.fc_layer_depth,
            fc_layer_width=self.fc_layer_width,
            action_type=self.action_mode,
            hybrid_action_continuous_dim=self.hybrid_action_continuous_dim,
            multi_discrete_number_of_dims_per_distribution=self.multi_discrete_number_of_dims_per_distribution,
            add_generator_embedding=self.add_generator_embedding,
            generator_embedding_number_of_timesteps=self.generator_embedding_number_of_timesteps,
            recurrent=self.recurrent,
        )(hidden, og_obs, embedding, dones, activation)

    @staticmethod
    def initialize_carry(batch_size, hidden_size=256):
        return ScannedRNN.initialize_carry(batch_size, hidden_size)


class GeneralActorOnlyRNN(nn.Module):
    """Actor-only version of GeneralActorCriticRNN - no critic network for GA."""
    action_dim: Sequence[int]
    fc_layer_depth: int
    fc_layer_width: int
    action_type: str
    hybrid_action_continuous_dim: int
    multi_discrete_number_of_dims_per_distribution: List[int]
    add_generator_embedding: bool
    generator_embedding_number_of_timesteps: int
    recurrent: bool = False

    @nn.compact
    def __call__(self, hidden, og_obs, embedding, dones, activation):
        if self.add_generator_embedding:
            gen_timestep = og_obs.env_state.timestep
            gen_timestep = gen_timestep / self.generator_embedding_number_of_timesteps
            gen_timestep = jnp.expand_dims(gen_timestep, axis=-1)
            embedding = jnp.concatenate([embedding, gen_timestep], axis=-1)

        if self.recurrent:
            rnn_in = (embedding, dones)
            hidden, embedding = ScannedRNN()(hidden, rnn_in)

        actor_mean = embedding
        for i in range(self.fc_layer_depth):
            actor_mean = nn.Dense(
                self.fc_layer_width,
                kernel_init=orthogonal(np.sqrt(2)),
                bias_init=constant(0.0),
            )(actor_mean)
            actor_mean = activation(actor_mean)

        actor_mean_last = actor_mean
        actor_mean = nn.Dense(self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0))(actor_mean)
        if self.action_type == ActionType.DISCRETE:
            pi = distrax.Categorical(logits=actor_mean)
        elif self.action_type == ActionType.CONTINUOUS:
            actor_logtstd = self.param("log_std", nn.initializers.zeros, (self.action_dim,))
            pi = distrax.MultivariateNormalDiag(actor_mean, jnp.exp(actor_logtstd))
        elif self.action_type == ActionType.MULTI_DISCRETE:
            pi = MultiDiscreteActionDistribution(actor_mean, self.multi_discrete_number_of_dims_per_distribution)
        else:
            actor_mean_continuous = nn.Dense(
                self.hybrid_action_continuous_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
            )(actor_mean_last)
            actor_mean_sigma = jnp.exp(
                nn.Dense(self.hybrid_action_continuous_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0))(
                    actor_mean_last
                )
            )
            pi = HybridActionDistribution(actor_mean, actor_mean_continuous, actor_mean_sigma)

        return hidden, pi


class ActorOnlyPixelsRNN(nn.Module):
    """Actor-only version of ActorCriticPixelsRNN - for GA (no critic needed)."""
    action_dim: Sequence[int]
    fc_layer_depth: int
    fc_layer_width: int
    action_mode: str
    hybrid_action_continuous_dim: int
    multi_discrete_number_of_dims_per_distribution: List[int]
    activation: str
    add_generator_embedding: bool = False
    generator_embedding_number_of_timesteps: int = 10
    recurrent: bool = False
    count_dormant: bool = False  # Unused, but needed for API compatibility
    redo_tau: float = 0.01  # Unused, but needed for API compatibility

    @nn.compact
    def __call__(self, hidden, x, **kwargs):
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh
        og_obs, dones = x

        if self.add_generator_embedding:
            obs = og_obs.obs
        else:
            obs = og_obs

        image = obs.image
        global_info = obs.global_info

        x = nn.Conv(features=16, kernel_size=(8, 8), strides=(4, 4))(image)
        x = nn.relu(x)
        x = nn.Conv(features=32, kernel_size=(4, 4), strides=(2, 2))(x)
        x = nn.relu(x)
        embedding = x.reshape(x.shape[0], x.shape[1], -1)

        embedding = jnp.concatenate([embedding, global_info], axis=-1)

        return GeneralActorOnlyRNN(
            action_dim=self.action_dim,
            fc_layer_depth=self.fc_layer_depth,
            fc_layer_width=self.fc_layer_width,
            action_type=self.action_mode,
            hybrid_action_continuous_dim=self.hybrid_action_continuous_dim,
            multi_discrete_number_of_dims_per_distribution=self.multi_discrete_number_of_dims_per_distribution,
            add_generator_embedding=self.add_generator_embedding,
            generator_embedding_number_of_timesteps=self.generator_embedding_number_of_timesteps,
            recurrent=self.recurrent,
        )(hidden, og_obs, embedding, dones, activation)

    @staticmethod
    def initialize_carry(batch_size, hidden_size=256):
        return ScannedRNN.initialize_carry(batch_size, hidden_size)


class ActorCriticSymbolicRNN(nn.Module):
    action_dim: Sequence[int]
    fc_layer_width: int
    action_mode: str
    hybrid_action_continuous_dim: int
    multi_discrete_number_of_dims_per_distribution: List[int]
    fc_layer_depth: int
    activation: str
    add_generator_embedding: bool = False
    generator_embedding_number_of_timesteps: int = 10
    recurrent: bool = True
    count_dormant: bool = False
    redo_tau: float = 0.01

    @nn.compact
    def __call__(self, hidden, x):
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh

        og_obs, dones = x
        if self.add_generator_embedding:
            obs = og_obs.obs
        else:
            obs = og_obs

        embedding = nn.Dense(
            self.fc_layer_width,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(obs)
        embedding = nn.relu(embedding)

        return GeneralActorCriticRNN(
            action_dim=self.action_dim,
            fc_layer_depth=self.fc_layer_depth,
            fc_layer_width=self.fc_layer_width,
            action_type=self.action_mode,
            hybrid_action_continuous_dim=self.hybrid_action_continuous_dim,
            multi_discrete_number_of_dims_per_distribution=self.multi_discrete_number_of_dims_per_distribution,
            add_generator_embedding=self.add_generator_embedding,
            generator_embedding_number_of_timesteps=self.generator_embedding_number_of_timesteps,
            recurrent=self.recurrent,
            count_dormant=self.count_dormant,
            redo_tau=self.redo_tau,
        )(hidden, og_obs, embedding, dones, activation)

    @staticmethod
    def initialize_carry(batch_size, hidden_size=256):
        return ScannedRNN.initialize_carry(batch_size, hidden_size)


class MultiHeadDense(nn.Module):
    num_heads: int  # Number of heads
    out_dim: int  # Output dimension for each head
    kernel_init: nn.initializers.Initializer
    bias_init: nn.initializers.Initializer

    @nn.compact
    def __call__(self, x):
        # x has shape (...., num_features, feature_dim)
        num_features, feature_dim = x.shape[-2:]

        # Initialize a Dense layer for each head, stacked as (num_heads, feature_dim, out_dim)
        dense_kernels = self.param("dense_kernels", self.kernel_init, (self.num_heads, feature_dim, self.out_dim))
        dense_biases = self.param("dense_biases", self.bias_init, (self.num_heads, self.out_dim))

        # Apply the dense layer to each head by broadcasting and matrix multiplying
        x_expanded = jnp.expand_dims(x, axis=-2)  # Shape: (..., num_features, 1, feature_dim)
        output = jnp.einsum("...fhd,hdo->...fho", x_expanded, dense_kernels) + dense_biases
        output = nn.relu(output)  # Shape: (..., num_features, num_heads, out_dim)
        output = output.sum(axis=-3)  # Shape=(..., num_heads, out_dim)

        output = output.reshape((*output.shape[:-2], self.num_heads * self.out_dim))  # Shape=(..., num_heads * out_dim)
        return output


class ActorCriticPermutationInvariantSymbolicRNN(nn.Module):
    action_dim: Sequence[int]
    symbolic_embedding_dim: int
    fc_layer_width: int
    action_mode: str
    multi_discrete_number_of_dims_per_distribution: List[int]
    hybrid_action_continuous_dim: int
    fc_layer_depth: int
    activation: str
    recurrent: bool
    add_generator_embedding: bool = False
    include_actions_and_rewards: bool = False
    permutation_invariant: bool = True
    num_heads: int = None
    preprocess_separately: bool = False
    encoder_size: int = 64
    count_dormant: bool = False
    redo_tau: float = 0.01

    @nn.compact
    def __call__(self, hidden, x):
        # print(f"ActorCriticSymbolicRNN\n")
        if self.activation == "relu":
            activation_fn = nn.relu
        elif self.activation == "tanh":
            activation_fn = nn.tanh
        else:
            raise ValueError(f"Unknown activation function: {self.activation}")

        og_obs, dones = x
        if self.add_generator_embedding:
            obs = og_obs.obs
        else:
            obs = og_obs

        if self.permutation_invariant:
            assert (
                self.symbolic_embedding_dim % self.num_heads == 0
            ), f"{self.symbolic_embedding_dim=} must be divisible by {self.num_heads=}"

            if self.preprocess_separately:

                def _single_encoder(features, entity_id):
                    num_to_remove = 4
                    embedding = activation_fn(
                        nn.Dense(
                            self.encoder_size - num_to_remove,
                            kernel_init=orthogonal(np.sqrt(2)),
                            bias_init=constant(0.0),
                        )(features)
                    )
                    id_1h = jnp.zeros((*embedding.shape[:3], 4)).at[:, :, :, entity_id].set(1)
                    return jnp.concatenate([embedding, id_1h], axis=-1)

                circle_encodings = _single_encoder(obs.circles, 0)
                polygon_encodings = _single_encoder(obs.polygons, 1)
                joint_encodings = _single_encoder(obs.joints, 2)
                thruster_encodings = _single_encoder(obs.thrusters, 3)

                all_encodings = jnp.concatenate(
                    [polygon_encodings, circle_encodings, joint_encodings, thruster_encodings], axis=2
                )
                all_mask = jnp.concatenate(
                    [obs.polygon_mask, obs.circle_mask, obs.joint_mask, obs.thruster_mask], axis=2
                )

                def mask(features, mask):
                    return jnp.where(mask[:, None], features, jnp.zeros_like(features))

                obs = jax.vmap(jax.vmap(mask))(all_encodings, all_mask)

            dim_per_head = self.symbolic_embedding_dim // self.num_heads
            obs_embedding = MultiHeadDense(
                num_heads=self.num_heads,
                out_dim=dim_per_head,
                kernel_init=orthogonal(np.sqrt(2)),
                bias_init=constant(0.0),
            )(obs)
            embedding = obs_embedding
        embedding = activation_fn(embedding)

        return GeneralActorCriticRNN(
            action_dim=self.action_dim,
            fc_layer_depth=self.fc_layer_depth,
            fc_layer_width=self.fc_layer_width,
            action_type=self.action_mode,
            hybrid_action_continuous_dim=self.hybrid_action_continuous_dim,
            multi_discrete_number_of_dims_per_distribution=self.multi_discrete_number_of_dims_per_distribution,
            recurrent=self.recurrent,
            count_dormant=self.count_dormant,
            redo_tau=self.redo_tau,
        )(hidden, og_obs, embedding, dones, activation_fn)

    @staticmethod
    def initialize_carry(batch_size, hidden_dim):
        return ScannedRNN.initialize_carry(batch_size, hidden_dim)
