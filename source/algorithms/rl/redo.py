"""ReDo -- Recycling Dormant Neurons (Sokar et al., ICML 2023).

The one ReDo every suite uses. Port of `inspiration/redo/src/redo.py`, the
reference implementation released with the paper, from PyTorch to JAX/optax.

There used to be three of these -- `source/studies/gymnax/redo.py`,
`third_party/kinetix/kinetix/models/redo.py` and the `_apply_redo` inside
`source/studies/brax/my_brax/ppo_train.py` -- and they did not agree on what a dormant
neuron is, on whether the threshold was configurable, or on whether the
optimizer state was reset. See `docs/unify_implementations.md` Sect. 8.

## What ReDo does, in the order this module does it

1. Score every hidden neuron on a batch of observations, normalised by the
   layer's mean score so one threshold works for layers of different widths
   (`neuron_dormancy_score`). This normalisation is the reference's, taken in
   turn from Dopamine's weight recycler.
2. Neurons scoring <= tau are dormant (`dormant_masks`). `tau=0` degenerates to
   "exactly zero", the strict dead-unit count the paper also reports and which
   this module logs alongside the real count.
3. Resample their incoming weights from the layer's initialiser, zero their
   bias, and zero their outgoing weights, so the network's function is
   essentially unchanged at the moment of the reset (`_reinit_params`).
4. Zero the Adam moments of every weight touched, and restart bias correction
   (`_reset_adam_state`). The reference's own comment on this is "Step count
   resets are key to the algorithm's performance" -- it is not optional, and
   the brax path skipping it was one of the ways the suites disagreed.

## The criterion, and why it depends on the activation

The reference is a ReLU network throughout, and scores a neuron by its mean
absolute activation: a dead ReLU outputs exactly 0. That test does not transfer
to tanh. Under tanh the useless unit is the *constant* one, and the usual way a
tanh unit goes constant is saturation at +/-1 -- which has the LARGEST possible
mean|activation|, so `magnitude` scores it near 1.0 and never flags it, even
though tanh'(+/-1) ~ 0 means no gradient flows through it and it is exactly as
dead as a dead ReLU.

So the score is the spread rather than the level when the network is tanh:

  "magnitude"    mean|act| / layer mean. The reference's test, and the right
                 one for ReLU.
  "variability"  std(act) / layer mean std. Catches both saturated units and
                 stuck-at-zero units. For a ReLU network the two nearly
                 coincide -- a dead ReLU has zero variance as well as zero mean
                 -- so the two suites stay comparable.

`criterion_for_activation` derives which one from the suite's activation, so
this is not a knob anyone has to remember to set: it follows `POLICY_ARCH` in
`source/algorithms/networks.py`, and gymnax (relu) gets magnitude while mujoco, brax
and kinetix (tanh) get variability. Stated plainly because it matters for
reading the numbers: `tau` is the reference's 0.025, calibrated on the
magnitude score. It is reused for variability because both scores are
layer-mean-normalised and therefore dimensionless, but it was not separately
calibrated there.
"""

import jax
import jax.numpy as jnp
import optax


# The threshold and the check interval below are the reference defaults
# (inspiration/redo/src/config.py: redo_tau=0.025, redo_check_interval=1000
# gradient steps). One PPO "update" in the gymnax trainers is
# num_epochs * num_minibatches gradient steps, so their default of every 50
# updates lands in the same ballpark.
DEFAULT_TAU = 0.025

CRITERION_MAGNITUDE = 'magnitude'
CRITERION_VARIABILITY = 'variability'

# Hidden activation -> criterion. See the module docstring for why this is
# derived rather than configured.
_CRITERION_FOR_ACTIVATION = {
    'relu': CRITERION_MAGNITUDE,
    'tanh': CRITERION_VARIABILITY,
    'swish': CRITERION_VARIABILITY,
}


def criterion_for_activation(activation):
    """Dormancy criterion for a hidden activation, by name ('relu', 'tanh', ...).

    Raises on an unknown activation rather than defaulting, because silently
    picking the ReLU test for a saturating activation is precisely the failure
    this module exists to remove.
    """
    try:
        return _CRITERION_FOR_ACTIVATION[activation]
    except KeyError:
        raise ValueError(
            f"No ReDo dormancy criterion defined for activation {activation!r}. "
            f"Known: {sorted(_CRITERION_FOR_ACTIVATION)}. Add it to "
            f"_CRITERION_FOR_ACTIVATION with a reason, do not guess."
        ) from None


def criterion_for_activation_fn(activation_fn):
    """Same, for callers that hold the activation callable rather than its name.

    brax builds its networks from an `ActivationFn`, not from a string. The
    callables in `core.networks.ACTIVATIONS` are the flax/jax ones, so identity
    against them resolves the name.
    """
    from source.algorithms.networks import ACTIVATIONS

    for name, fn in ACTIVATIONS.items():
        if fn is activation_fn:
            return criterion_for_activation(name)
    raise ValueError(
        f"Cannot map activation {activation_fn!r} to a ReDo dormancy criterion. "
        f"Add it to core.networks.ACTIVATIONS and _CRITERION_FOR_ACTIVATION."
    )


def neuron_dormancy_score(act, criterion):
    """Per-neuron score along the last axis of `act`, normalised by the layer mean.

    `act` is (..., width); every leading axis is batch. The normalisation is
    what makes `tau` independent of the layer width, and it is the reference's:
    divide by the mean of the per-neuron scores.
    """
    if act.ndim > 1:
        batch_dims = tuple(range(act.ndim - 1))
        if criterion == CRITERION_VARIABILITY:
            per_neuron = jnp.std(act, axis=batch_dims)
        else:
            per_neuron = jnp.mean(jnp.abs(act), axis=batch_dims)
    else:
        # No batch axis: spread is undefined, so fall back to magnitude rather
        # than reporting every neuron as dormant.
        per_neuron = jnp.abs(act)

    return per_neuron / (per_neuron.mean() + 1e-9)


def hidden_activations(params, obs, num_hidden, activation_fn=jax.nn.relu,
                       prefix='Dense_'):
    """Post-activation outputs of each hidden layer of an MLP.

    Replays the forward pass from the parameter dict alone, which works for any
    `[Dense -> activation] * n` trunk followed by an output Dense. `prefix` is
    the flax layer-name prefix: 'Dense_' for the gymnax networks,
    'hidden_' for brax's.

    `activation_fn` must be the activation the network was *trained* with. It
    used to be hardcoded to relu here, which was right for gymnax and wrong for
    everyone else -- the same class of bug as the evaluators rebuilding brax
    policies with swish (see source/studies/brax/evaluate_continual.py).
    """
    p = params['params']
    x = obs
    acts = []
    for i in range(num_hidden):
        layer = p[f'{prefix}{i}']
        x = activation_fn(x @ layer['kernel'] + layer['bias'])
        acts.append(x)
    return acts


def dormant_masks(scores, tau):
    """Boolean mask per hidden layer, True where the neuron is dormant.

    `scores` are the layer-normalised per-neuron scores from
    `neuron_dormancy_score`. `tau=0` degenerates to "exactly zero", the strict
    dead-neuron count the paper also reports.
    """
    if tau > 0.0:
        return [s <= tau for s in scores]
    return [jnp.isclose(s, 0.0) for s in scores]


def score_layers(activations, criterion):
    return [neuron_dormancy_score(a, criterion) for a in activations]


def _stats_from_scores(scores, masks, tau):
    zero_masks = masks if tau == 0.0 else dormant_masks(scores, 0.0)

    per_layer = [(int(jnp.sum(m)), int(m.size)) for m in masks]
    total = sum(n for n, _ in per_layer)
    total_neurons = sum(sz for _, sz in per_layer)
    zero_total = sum(int(jnp.sum(m)) for m in zero_masks)

    return {
        'per_layer': per_layer,
        'dormant_count': total,
        'total_neurons': total_neurons,
        'dormant_fraction': total / max(total_neurons, 1),
        'zero_count': zero_total,
        'zero_fraction': zero_total / max(total_neurons, 1),
    }


def dormant_stats_from_activations(acts, tau=DEFAULT_TAU,
                                   criterion=CRITERION_MAGNITUDE):
    """`dormant_stats` for a network that hands over its own activations.

    `acts` is a list of (batch, width) post-activation arrays, one per hidden
    layer. Everything after this point -- the per-neuron score, the tau
    threshold, the reported counts -- is identical to the MLP path, so a
    non-MLP policy lands in the same column as every other method. Only the way
    the activations are obtained differs.
    """
    scores = score_layers(acts, criterion)
    return _stats_from_scores(scores, dormant_masks(scores, tau), tau)


def dormant_stats(params, obs, num_hidden, tau=DEFAULT_TAU,
                  activation_fn=jax.nn.relu, criterion=CRITERION_MAGNITUDE,
                  prefix='Dense_', activations_fn=None):
    """Dormant-neuron diagnostics; pure measurement, nothing is modified.

    Reports both the tau-threshold count (the neurons ReDo would recycle) and
    the tau=0 count (neurons that are strictly dead), as the reference logs.

    `activations_fn(params, obs) -> [act_0, ..., act_k]` overrides the replayed
    MLP forward pass for architectures `hidden_activations` cannot walk -- the
    scheduling policy, dropped 2026-09-08, had attention and per-entity
    encoders rather than a Dense chain, and the hook is kept for the next such.
    When it is given, `num_hidden`, `activation_fn` and `prefix` are unused: the
    network is reporting its own hidden layers rather than having them inferred
    from the parameter names.
    """
    if activations_fn is not None:
        acts = activations_fn(params, obs)
    else:
        acts = hidden_activations(params, obs, num_hidden, activation_fn, prefix)
    return dormant_stats_from_activations(acts, tau=tau, criterion=criterion)


def _layer_names(num_hidden, prefix, layers):
    """The hidden layers plus the output layer, in forward order.

    ``layers`` names them outright for a network that is not a Dense chain
    (the conv policy's ``Conv_0, Dense_0, Dense_1``); otherwise they are
    ``prefix0 .. prefix{num_hidden}``, which is every caller before 2026-09-06.
    """
    if layers is not None:
        return list(layers)
    return [f'{prefix}{i}' for i in range(num_hidden + 1)]


def _expand_mask(mask, next_kernel, extra_fan_in=0):
    """A hidden layer's dormancy mask as a mask over the NEXT layer's fan-in.

    Identity for Dense -> Dense. For a convolution whose output is flattened
    into a Dense, the next layer's fan-in is ``height * width * channels`` with
    the channel fastest (``x.reshape(..., -1)`` on an NHWC activation), so a
    channel's mask is tiled over every spatial position. ``extra_fan_in``
    inputs of the next layer are not any hidden unit's output (the Kinetix
    policy appends its global-info scalar after the flattened conv map); they
    come last and are never masked. Refuses anything else rather than
    guessing a layout.
    """
    fan_in = next_kernel.shape[-2]
    if fan_in == mask.shape[0]:
        return mask
    hidden_in = fan_in - int(extra_fan_in)
    if hidden_in > 0 and hidden_in % mask.shape[0] == 0:
        out = jnp.tile(mask, hidden_in // mask.shape[0])
        if extra_fan_in:
            out = jnp.concatenate(
                [out, jnp.zeros((int(extra_fan_in),), dtype=out.dtype)])
        return out
    raise ValueError(f'cannot map a {mask.shape[0]}-unit mask onto a layer '
                     f'with fan-in {fan_in} (extra_fan_in={extra_fan_in})')


def _reinit_params(params, masks, key, prefix='Dense_', layers=None,
                   extra_fan_in=None):
    """Resample incoming weights of dormant neurons, zero their outgoing ones."""
    p = jax.tree_util.tree_map(lambda x: x, params['params'])
    names = _layer_names(len(masks), prefix, layers)
    extra_fan_in = extra_fan_in or {}
    # flax.linen.Dense initialises kernels with lecun_normal and biases with
    # zeros; drawing the replacements from the same distribution is what the
    # paper means by "reinitialise", and it keeps the reset units on the same
    # scale as the ones that never went dormant. flax's Conv uses the same
    # initialiser, and the kernel's unit axis is the last one for both, so
    # the (..., width) broadcast below covers a (kh, kw, fan_in, width)
    # kernel as it does a (fan_in, width) one.
    kernel_init = jax.nn.initializers.lecun_normal()
    keys = jax.random.split(key, len(masks))

    for i, (mask, layer_key) in enumerate(zip(masks, keys)):
        layer = dict(p[names[i]])
        kernel = layer['kernel']                      # (..., fan_in, width)
        fresh = kernel_init(layer_key, kernel.shape, kernel.dtype)
        layer['kernel'] = jnp.where(mask, fresh, kernel)
        layer['bias'] = jnp.where(mask, 0.0, layer['bias'])
        p[names[i]] = layer

        # Outgoing weights -> 0, so the network's function is unchanged at the
        # moment of the reset (exactly so for strictly dead units; for tau > 0 a
        # dormant unit still carries a small activation, so the output moves a
        # little). The next layer's bias is deliberately left alone: zeroing it
        # would itself create dormant neurons.
        #
        # Note the layers are visited in order, so if a neuron in layer i+1 is
        # itself dormant, resampling its incoming row overwrites the zeros just
        # written here. That is also what the reference implementation does, and
        # it is harmless: that neuron's own outgoing weights get zeroed in turn,
        # so there is still no path from a recycled unit to the output.
        nxt = dict(p[names[i + 1]])
        out_mask = _expand_mask(mask, nxt['kernel'],
                                extra_fan_in.get(names[i + 1], 0))
        nxt['kernel'] = jnp.where(out_mask[:, None], 0.0, nxt['kernel'])
        p[names[i + 1]] = nxt

    return {**params, 'params': p}


def _zero_moments(moments, masks, prefix='Dense_', layers=None,
                  extra_fan_in=None):
    """Zero the Adam moment entries belonging to the recycled weights."""
    m = jax.tree_util.tree_map(lambda x: x, moments['params'])
    names = _layer_names(len(masks), prefix, layers)
    extra_fan_in = extra_fan_in or {}
    for i, mask in enumerate(masks):
        layer = dict(m[names[i]])
        layer['kernel'] = jnp.where(mask, 0.0, layer['kernel'])
        layer['bias'] = jnp.where(mask, 0.0, layer['bias'])
        m[names[i]] = layer

        nxt = dict(m[names[i + 1]])
        out_mask = _expand_mask(mask, nxt['kernel'],
                                extra_fan_in.get(names[i + 1], 0))
        nxt['kernel'] = jnp.where(out_mask[:, None], 0.0, nxt['kernel'])
        m[names[i + 1]] = nxt
    return {**moments, 'params': m}


def _reset_adam_state(opt_state, masks, reset_count=True, prefix='Dense_',
                      layers=None, extra_fan_in=None):
    """Rewrite every ScaleByAdamState in an optax state tree.

    The reference zeroes the per-parameter Adam step count as well. optax keeps
    a single count for the whole tree, so `reset_count` restarts bias correction
    globally rather than per-tensor -- the closest available equivalent, and the
    reference resets the step of nearly every tensor anyway.
    """
    found = []

    def walk(node):
        if isinstance(node, optax.ScaleByAdamState):
            found.append(True)
            return node._replace(
                mu=_zero_moments(node.mu, masks, prefix, layers, extra_fan_in),
                nu=_zero_moments(node.nu, masks, prefix, layers, extra_fan_in),
                count=jnp.zeros_like(node.count) if reset_count else node.count,
            )
        if isinstance(node, tuple) and hasattr(node, '_fields'):  # other NamedTuple states
            return type(node)(*[walk(c) for c in node])
        if isinstance(node, (list, tuple)):
            return type(node)(walk(c) for c in node)
        return node

    new_state = walk(opt_state)
    if not found:
        # Leaving stale moments on a recycled neuron is the failure mode ReDo is
        # most sensitive to, so refuse to run rather than silently half-apply.
        raise ValueError(
            'ReDo found no optax.adam state to reset. It only knows how to clear '
            'Adam moments; adapt _reset_adam_state for other optimizers.'
        )
    return new_state


def apply_redo(state, obs, num_hidden, key, tau=DEFAULT_TAU,
               activation_fn=jax.nn.relu, criterion=CRITERION_MAGNITUDE,
               prefix='Dense_', reset_adam_count=True, layers=None,
               activations_fn=None, extra_fan_in=None):
    """Run one ReDo pass over a flax TrainState. Returns (new_state, stats).

    `state.tx` is assumed to contain optax.adam; any other transformation in the
    chain is left untouched.

    `layers` and `activations_fn` are for a network that is not a Dense chain,
    since 2026-09-06 (the grid conv policy): `layers` names the hidden
    layers and then the output layer in forward order, and `activations_fn`
    is the network's own `hidden_activations`, as for `dormant_stats`. A
    hidden unit of a convolution is a channel: its incoming kernel slice is
    resampled and its outgoing rows in the flattened next layer are zeroed
    (`_expand_mask`), which is the reference's treatment of conv layers.
    Both None is every caller before that date, bit-unchanged.

    `extra_fan_in` maps a layer name to how many of its inputs are NOT a
    hidden unit's output (the Kinetix policy's `Dense_0` takes the flattened
    conv map plus one global-info scalar); those trailing inputs are left
    alone when the previous layer's outgoing rows are zeroed. None: no layer
    has any, which is every network before the Kinetix one.
    """
    if activations_fn is not None:
        acts = activations_fn(state.params, obs)
    else:
        acts = hidden_activations(state.params, obs, num_hidden, activation_fn,
                                  prefix)
    scores = score_layers(acts, criterion)
    masks = dormant_masks(scores, tau)
    stats = _stats_from_scores(scores, masks, tau)

    if stats['dormant_count'] == 0:
        return state, stats

    return state.replace(
        params=_reinit_params(state.params, masks, key, prefix, layers,
                              extra_fan_in),
        opt_state=_reset_adam_state(state.opt_state, masks, reset_adam_count,
                                    prefix, layers, extra_fan_in),
    ), stats
