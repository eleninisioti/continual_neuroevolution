"""AURORA-style unsupervised behavioural descriptors.

The DNS paper ("Dominated Novelty Search: Rethinking Local Competition in
Quality-Diversity", GECCO'25) evaluates DNS either with *hand-designed*
behaviour descriptors (e.g. final xy position, feet contact proportions) or
with *unsupervised* descriptors learned online with AURORA, where the
descriptor is the latent code of an auto-encoder trained on the observation
trajectories collected during evaluation.

This module is a port of the AURORA machinery used by the reference
implementation (``inspiration/DNS/Dominated-Novelty-Search``):

* ``qdax/core/neuroevolution/networks/seq2seq_networks.py`` — the LSTM
  sequence-to-sequence auto-encoder (encoder final cell state = descriptor).
* ``qdax/utils/train_seq2seq.py`` — periodic training of that auto-encoder on
  the observations stored alongside the population, with per-dimension
  observation normalisation.
* ``qdax/tasks/environments/bd_extractors.py`` — ``get_aurora_encoding``.

The only structural difference is that the population here is DNS's fixed-size
population rather than a QDax repertoire, so the auto-encoder is trained on the
observation trajectories of the current population.

Usage sketch::

    aurora = AuroraDescriptors(obs_size=obs_dim, traj_steps=10, latent_dim=6)
    aurora_state = aurora.init(key)                       # random encoder
    aurora_state, loss = aurora.train(key, observations, aurora_state, 0)
    descriptors = aurora.encode(observations, aurora_state)

``observations`` always has shape ``(num_individuals, traj_steps, obs_size)``
and holds the *last valid* observation after episode termination (matching
QDax's ``last_valid_observations``).

The seq2seq networks and the training step below are adapted from the Flax
seq2seq example (Copyright 2022 The Flax Authors, Apache-2.0) via QDax.
"""

from __future__ import annotations

import functools
from typing import Any

import flax.struct
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import linen as nn
from flax.training import train_state

Array = Any
PRNGKey = Any


# ============================================================================
# LSTM seq2seq auto-encoder (port of qdax seq2seq_networks.py)
# ============================================================================


class EncoderLSTM(nn.Module):
    """EncoderLSTM Module wrapped in a lifted scan transform."""

    @functools.partial(
        nn.scan,
        variable_broadcast="params",
        in_axes=1,
        out_axes=1,
        split_rngs={"params": False},
    )
    @nn.compact
    def __call__(self, carry: tuple[Array, Array], x: Array):
        """Applies the module."""
        lstm_state, is_eos = carry
        features = lstm_state[0].shape[-1]
        new_lstm_state, y = nn.LSTMCell(features)(lstm_state, x)

        def select_carried_state(new_state: Array, old_state: Array) -> Array:
            return jnp.where(is_eos[:, np.newaxis], old_state, new_state)

        # LSTM state is a tuple (c, h).
        carried_lstm_state = tuple(
            select_carried_state(*s) for s in zip(new_lstm_state, lstm_state)
        )

        return (carried_lstm_state, is_eos), y

    @staticmethod
    def initialize_carry(batch_size: int, hidden_size: int) -> tuple[Array, Array]:
        # Use a dummy key since the default state init fn is just zeros.
        return nn.LSTMCell(hidden_size, parent=None).initialize_carry(
            jax.random.key(0), (batch_size, hidden_size)
        )


class Encoder(nn.Module):
    """LSTM encoder, returning state after finding the EOS token in the input."""

    hidden_size: int

    @nn.compact
    def __call__(self, inputs: Array) -> Array:
        batch_size = inputs.shape[0]
        lstm = EncoderLSTM(name="encoder_lstm")
        init_lstm_state = lstm.initialize_carry(batch_size, self.hidden_size)

        init_is_eos = jnp.zeros(batch_size, dtype=bool)
        init_carry = (init_lstm_state, init_is_eos)
        (final_state, _), _ = lstm(init_carry, inputs)

        return final_state


class DecoderLSTM(nn.Module):
    """DecoderLSTM Module wrapped in a lifted scan transform."""

    teacher_force: bool
    obs_size: int

    @functools.partial(
        nn.scan,
        variable_broadcast="params",
        in_axes=1,
        out_axes=1,
        split_rngs={"params": False, "lstm": True},
    )
    @nn.compact
    def __call__(self, carry: tuple[Array, Array], x: Array) -> Array:
        """Applies the DecoderLSTM model."""
        lstm_state, last_prediction = carry
        if not self.teacher_force:
            x = last_prediction

        features = lstm_state[0].shape[-1]
        new_lstm_state, y = nn.LSTMCell(features)(lstm_state, x)

        logits = nn.Dense(features=self.obs_size)(y)

        return (lstm_state, logits), (logits, logits)


class Decoder(nn.Module):
    """LSTM decoder."""

    teacher_force: bool
    obs_size: int

    @nn.compact
    def __call__(self, inputs: Array, init_state: Any) -> tuple[Array, Array]:
        lstm = DecoderLSTM(teacher_force=self.teacher_force, obs_size=self.obs_size)
        init_carry = (init_state, inputs[:, 0])
        _, (logits, predictions) = lstm(init_carry, inputs)
        return logits, predictions


class Seq2seq(nn.Module):
    """Sequence-to-sequence class using encoder/decoder architecture.

    `bounded_descriptor` selects which half of the encoder's final LSTM carry
    becomes the descriptor. QDax (and so `False`, the default) uses the cell
    state `c`, which has no bound and grows with the magnitude of the input
    trajectory. That is fine in QDax's own settings but not here: on CheetahRun
    it lets a diverging genotype win dominated novelty on descriptor magnitude
    alone, which is the failure documented in docs/dns_cheetah_diagnosis.md.
    `True` returns the hidden state `h = o * tanh(c)` instead, which lies in
    (-1, 1)^latent_dim and so cannot run away -- the property every descriptor
    space the DNS release was validated on happens to have.
    """

    teacher_force: bool
    hidden_size: int
    obs_size: int
    bounded_descriptor: bool = False

    def setup(self) -> None:
        self.encoder = Encoder(hidden_size=self.hidden_size)
        self.decoder = Decoder(teacher_force=self.teacher_force, obs_size=self.obs_size)

    @nn.compact
    def __call__(
        self, encoder_inputs: Array, decoder_inputs: Array
    ) -> tuple[Array, Array]:
        init_decoder_state = self.encoder(encoder_inputs)
        logits, predictions = self.decoder(decoder_inputs, init_decoder_state)
        return logits, predictions

    def encode(self, encoder_inputs: Array) -> Array:
        init_decoder_state = self.encoder(encoder_inputs)
        cell_state, hidden_state = init_decoder_state
        return hidden_state if self.bounded_descriptor else cell_state


# ============================================================================
# Training (port of qdax train_seq2seq.py)
# ============================================================================


class AuroraState(flax.struct.PyTreeNode):
    """Everything needed to encode observations into descriptors.

    Args:
        model_params: parameters of the seq2seq auto-encoder.
        mean_observations: per-dimension mean used to normalise observations.
        std_observations: per-dimension std used to normalise observations.
    """

    model_params: Any
    mean_observations: jnp.ndarray
    std_observations: jnp.ndarray


@jax.jit
def _train_step(
    state: train_state.TrainState,
    batch: Array,
    lstm_random_key: PRNGKey,
) -> tuple[train_state.TrainState, jnp.ndarray]:
    """One auto-encoder gradient step on a batch of trajectories."""
    lstm_key = jax.random.fold_in(lstm_random_key, state.step)
    dropout_key, lstm_key = jax.random.split(lstm_key, 2)

    # Shift input by one to avoid leakage
    batch_decoder = jnp.roll(batch, shift=1, axis=1)

    # Large number as zero token
    batch_decoder = batch_decoder.at[:, 0, :].set(-1000)

    def loss_fn(params):
        logits, _ = state.apply_fn(
            {"params": params},
            batch,
            batch_decoder,
            rngs={"lstm": lstm_key, "dropout": dropout_key},
        )

        def mean_squared_error(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
            return jnp.inner(y - x, y - x) / x.shape[-1]

        res = jax.vmap(mean_squared_error)(
            jnp.reshape(logits.at[:, :-1, ...].get(), (logits.shape[0], -1)),
            jnp.reshape(
                batch_decoder.at[:, 1:, ...].get(), (batch_decoder.shape[0], -1)
            ),
        )
        loss = jnp.mean(res, axis=0)
        return loss, logits

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss_val, _logits), grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)

    return state, loss_val


class AuroraDescriptors:
    """Learned (unsupervised) behaviour descriptors for DNS.

    The descriptor of an individual is the final cell state of an LSTM encoder
    applied to its (sub-sampled, normalised) observation trajectory. The
    auto-encoder is retrained periodically on the observations of the current
    population, so the descriptor space changes during evolution — which is
    exactly the setting DNS is designed for (no descriptor bounds required).
    """

    def __init__(
        self,
        obs_size: int,
        traj_steps: int,
        latent_dim: int = 6,
        learning_rate: float = 1e-3,
        batch_size: int = 128,
        bounded_descriptor: bool = False,
    ) -> None:
        self.obs_size = obs_size
        self.traj_steps = traj_steps
        self.latent_dim = latent_dim
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.bounded_descriptor = bounded_descriptor

        # teacher_force=True: the decoder is only used for training, the
        # descriptor comes from the encoder alone. bounded_descriptor changes
        # only which half of the encoder carry is read out -- the auto-encoder
        # trained is identical either way, so the reconstruction losses of a
        # bounded and an unbounded run are directly comparable.
        self.model = Seq2seq(
            teacher_force=True, hidden_size=latent_dim, obs_size=obs_size,
            bounded_descriptor=bounded_descriptor,
        )

        self._encode_jit = jax.jit(
            lambda params, obs: self.model.apply(
                {"params": params}, obs, method=self.model.encode
            )
        )

    def init(self, key: PRNGKey) -> AuroraState:
        """Random encoder, identity normalisation."""
        key1, key2, key3 = jax.random.split(key, 3)
        encoder_input_shape = (1, self.traj_steps, self.obs_size)
        variables = self.model.init(
            {"params": key1, "lstm": key2, "dropout": key3},
            jnp.ones(encoder_input_shape, jnp.float32),
            jnp.ones(encoder_input_shape, jnp.float32),
        )
        return AuroraState(
            model_params=variables["params"],
            mean_observations=jnp.zeros((self.obs_size,)),
            std_observations=jnp.ones((self.obs_size,)),
        )

    def encode(self, observations: jnp.ndarray, state: AuroraState) -> jnp.ndarray:
        """Encode observation trajectories into descriptors.

        Args:
            observations: (num_individuals, traj_steps, obs_size)
            state: current auto-encoder state.

        Returns:
            descriptors of shape (num_individuals, latent_dim)
        """
        normalized = (
            observations - state.mean_observations
        ) / state.std_observations
        return self._encode_jit(state.model_params, normalized)

    def train(
        self,
        key: PRNGKey,
        observations: jnp.ndarray,
        state: AuroraState,
        iteration: int = 0,
        num_epochs: int | None = None,
        verbose: bool = False,
    ) -> tuple[AuroraState, float]:
        """Train the auto-encoder on the population's observation trajectories.

        Mirrors ``lstm_ae_train``: observations are normalised per dimension,
        the encoder/decoder are trained to reconstruct the (shifted)
        trajectory, and the normalisation statistics are stored with the new
        parameters so that later encodings use the same scaling.
        """
        if num_epochs is None:
            # Same schedule as the reference: cheaper re-training once the
            # descriptor space has roughly settled.
            num_epochs = 100 if iteration <= 100 else 25

        # Normalisation statistics over the whole population of trajectories.
        mean_obs = jnp.nanmean(observations, axis=(0, 1))
        std_obs = jnp.nanstd(observations, axis=(0, 1))
        # Dimensions with zero variance are mapped to 0 rather than dividing
        # by zero (the reference replaces the zeros with inf).
        std_obs = jnp.where(std_obs == 0, jnp.inf, std_obs)

        dataset = (observations - mean_obs) / std_obs
        dataset = jnp.nan_to_num(dataset, nan=0.0, posinf=0.0, neginf=0.0)

        num_samples = dataset.shape[0]
        batch_size = min(self.batch_size, num_samples)
        steps_per_epoch = max(1, num_samples // batch_size)

        tx = optax.adam(self.learning_rate)
        train_st = train_state.TrainState.create(
            apply_fn=self.model.apply, params=state.model_params, tx=tx
        )

        loss_val = 0.0
        for epoch in range(num_epochs):
            key, shuffle_key, step_key = jax.random.split(key, 3)
            shuffled = jax.random.permutation(shuffle_key, dataset, axis=0)

            for i in range(steps_per_epoch):
                batch = shuffled[i * batch_size : (i + 1) * batch_size]
                if batch.shape[0] < batch_size:
                    continue
                train_st, loss_val = _train_step(train_st, batch, step_key)

            if verbose and (epoch + 1) % 25 == 0:
                print(f"    AE epoch {epoch + 1}/{num_epochs}, loss: {loss_val:.4f}")

        new_state = AuroraState(
            model_params=train_st.params,
            mean_observations=mean_obs,
            std_observations=std_obs,
        )
        return new_state, float(loss_val)


# ============================================================================
# Training schedule
# ============================================================================


def aurora_training_schedule(num_generations: int, train_ratio: int = 8) -> set:
    """Generations at which the auto-encoder is retrained.

    AURORA retrains the encoder with decreasing frequency as evolution
    progresses (the descriptor space should stabilise over time). Following the
    reference implementation's ``train_ratio``, retraining happens at
    generations ``train_ratio * cumsum(1, 2, 3, ...)``, i.e. 8, 24, 48, 80, ...
    for the default ratio of 8.
    """
    schedule = set()
    gen, step = 0, 1
    while gen < num_generations:
        gen += train_ratio * step
        step += 1
        schedule.add(gen)
    return schedule


def subsample_indices(episode_length: int, traj_steps: int) -> jnp.ndarray:
    """Evenly spaced time indices used to sub-sample an episode trajectory."""
    traj_steps = min(traj_steps, episode_length)
    return jnp.linspace(0, episode_length - 1, traj_steps).astype(jnp.int32)


def episode_relative_indices(valid: jnp.ndarray, traj_steps: int) -> jnp.ndarray:
    """Time indices spread over the part of a rollout that actually happened.

    `subsample_indices` spreads its samples over the episode *cap*, so on tasks
    that terminate early most of the sampled steps land after termination and
    are the frozen final state repeated. A 67-step Acrobot episode sampled at
    linspace(0, 499, 10) yields two real states and eight copies of the last
    one, which makes the trajectory describe termination rather than behaviour.

    Spreading the same number of samples over `valid` instead keeps every
    sample inside the episode. Episodes shorter than `traj_steps` repeat states,
    but those repeats are real states rather than padding.

    valid: (episode_length,) 1.0 while the episode is still running.
    """
    num_valid = jnp.maximum(jnp.sum(valid).astype(jnp.int32), 1)
    frac = jnp.linspace(0.0, 1.0, traj_steps)
    idx = (frac * (num_valid - 1).astype(jnp.float32)).astype(jnp.int32)
    return jnp.minimum(idx, num_valid - 1)
