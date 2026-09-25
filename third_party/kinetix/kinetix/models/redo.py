"""ReDo: Recycling Dormant Neurons (Sokar et al., ICML 2023, arXiv:2302.12902)
for the Kinetix actor-critic.

Why this file exists
--------------------
Kinetix already had a `use_redo` flag, but it only ever *counted* dormant
neurons -- there was no resampling anywhere in the tree, so `use_redo=True`
trained exactly the same network as `use_redo=False` and the "ReDo" column was
vanilla PPO under another name. This module is the missing half.

It is a port of `source/studies/gymnax/redo.py` (itself a port of
`inspiration/redo/src/redo.py`) onto this repo's Kinetix network, and keeps the
same four steps:

  1. score each hidden neuron by its mean absolute post-activation over a batch,
     normalised by the layer mean so the threshold is width-independent;
  2. neurons scoring <= tau are dormant: resample their incoming weights and
     zero their bias;
  3. zero their outgoing weights, so applying ReDo does not change the function
     the network computes at that moment;
  4. zero the matching Adam moments (and the step count).

Steps 3 and 4 are what make ReDo work rather than merely perturb the network.

What differs from the gymnax version
------------------------------------
The gymnax networks are plain `[Dense -> relu] * n` MLPs, so `Dense_i` is
hidden layer `i` and the forward pass can be replayed from the parameter dict.
The Kinetix network cannot be replayed that cheaply -- the embedding comes out
of a two-layer CNN over pixels -- so the per-neuron scores are taken from the
network's own forward pass via `sow` (see `actor_critic.py`) instead of being
recomputed here.

The layer layout also differs. `GeneralActorCriticRNN` builds the policy and
value stacks in one loop, alternating, so with `fc_layer_depth = d` the modules
are named:

    Dense_0, Dense_2, Dense_4, ...   policy hidden layers   (d of them)
    Dense_1, Dense_3, Dense_5, ...   value  hidden layers   (d of them)
    Dense_{2d}                       policy head   (never reset)
    Dense_{2d+1}                     value  head   (never reset)

so policy hidden layer i is `Dense_{2i}` and its outgoing weights live in
`Dense_{2i+2}` -- which is the *next policy layer*, two indices along, not the
next module. Getting that stride wrong would zero the value stack's incoming
weights instead, which is why it is spelled out here.

Heads are never reset, matching the reference (there the final activations are
Q-values; here they are the action logits and the value scalar).

Everything runs inside `jax.lax.scan`, so this is written with `lax.cond` and
masked `jnp.where` updates rather than the reference's Python control flow.
"""

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.linen.initializers import orthogonal


# One threshold for the whole study, defined in source/algorithms/rl/redo.py: the
# reference default (inspiration/redo/src/config.py: redo_tau=0.025). The Kinetix
# configs used to say 0.01, which made the `redo` column here a different
# measurement from the gymnax one.
from source.algorithms.rl.redo import DEFAULT_TAU  # noqa: F401

# Module that holds the fully-connected stacks. The Conv layers that produce the
# embedding sit outside it and are not touched -- ReDo is defined on the dense
# hidden layers.
TRUNK = "GeneralActorCriticRNN_0"


def layer_layout(fc_layer_depth):
    """(policy, value) lists of (hidden_layer, next_layer) Dense name pairs.

    `next_layer` is where the hidden layer's outgoing weights live, which is the
    following layer *of the same stack* (stride 2), or that stack's head for the
    last hidden layer.
    """
    policy, value = [], []
    for i in range(fc_layer_depth):
        cur_p, cur_v = f"Dense_{2 * i}", f"Dense_{2 * i + 1}"
        if i + 1 < fc_layer_depth:
            nxt_p, nxt_v = f"Dense_{2 * i + 2}", f"Dense_{2 * i + 3}"
        else:
            nxt_p, nxt_v = f"Dense_{2 * fc_layer_depth}", f"Dense_{2 * fc_layer_depth + 1}"
        policy.append((cur_p, nxt_p))
        value.append((cur_v, nxt_v))
    return policy, value


def collect_scores(intermediates):
    """Per-layer neuron scores sown by the network, as (policy, value) lists.

    `sow` appends one entry per call under the owning module's path, so the
    scores arrive already ordered by layer. Each entry is a 1-tuple holding a
    (width,) array.
    """
    node = intermediates
    for key in ("intermediates", TRUNK):
        if isinstance(node, dict) and key in node:
            node = node[key]

    def series(name):
        if name not in node:
            return []
        entry = node[name]
        # sow with the default 'append' reduction stores a tuple of values.
        return [jnp.asarray(x) for x in (entry if isinstance(entry, tuple) else (entry,))]

    return series("policy_neuron_score"), series("value_neuron_score")


def _masks_from_scores(scores, tau):
    """Boolean dormancy mask per layer. Scores are already layer-mean-normalised."""
    return [s <= tau for s in scores]


def _reinit_stack(params, pairs, masks, key):
    """Resample dormant neurons' incoming weights, zero their outgoing ones.

    Mirrors `_reinit_params` in source/studies/gymnax/redo.py, but draws replacements
    from `orthogonal(sqrt(2))` because that is what these Dense layers are built
    with (gymnax's use flax's default lecun_normal). Resampling from a different
    distribution than the layer was initialised with would put recycled units on
    a different scale from the ones that never went dormant.
    """
    if not masks:
        return params
    init = orthogonal(np.sqrt(2))
    keys = jax.random.split(key, len(masks))

    for (cur, nxt), mask, lk in zip(pairs, masks, keys):
        layer = dict(params[cur])
        kernel = layer["kernel"]                      # (fan_in, width)
        fresh = init(lk, kernel.shape, kernel.dtype)
        layer["kernel"] = jnp.where(mask[None, :], fresh, kernel)
        layer["bias"] = jnp.where(mask, 0.0, layer["bias"])
        params[cur] = layer

        # Outgoing weights -> 0. The next layer's bias is deliberately left
        # alone; zeroing it would itself create dormant neurons.
        following = dict(params[nxt])
        following["kernel"] = jnp.where(mask[:, None], 0.0, following["kernel"])
        params[nxt] = following

    return params


def _zero_moment_stack(moments, pairs, masks):
    """Zero the Adam moment entries belonging to the weights just recycled."""
    if not masks:
        return moments
    for (cur, nxt), mask in zip(pairs, masks):
        layer = dict(moments[cur])
        layer["kernel"] = jnp.where(mask[None, :], 0.0, layer["kernel"])
        layer["bias"] = jnp.where(mask, 0.0, layer["bias"])
        moments[cur] = layer

        following = dict(moments[nxt])
        following["kernel"] = jnp.where(mask[:, None], 0.0, following["kernel"])
        moments[nxt] = following
    return moments


def _apply_to_trunk(tree, fn):
    """Run `fn` over the trunk sub-dict of a params/moments tree, functionally."""
    outer = dict(tree)
    inner = dict(outer["params"])
    trunk = dict(inner[TRUNK])
    inner[TRUNK] = fn(trunk)
    outer["params"] = inner
    return outer


def _reset_adam(opt_state, policy_pairs, value_pairs, p_masks, v_masks,
                reset_count=True):
    """Rewrite every ScaleByAdamState in the optax state tree.

    Refuses to silently do nothing if no Adam state is present: stale moments on
    a recycled neuron is the failure mode ReDo is most sensitive to.
    """
    found = []

    def zero(moments):
        return _apply_to_trunk(
            moments,
            lambda trunk: _zero_moment_stack(
                _zero_moment_stack(trunk, policy_pairs, p_masks),
                value_pairs, v_masks),
        )

    def walk(node):
        if isinstance(node, optax.ScaleByAdamState):
            found.append(True)
            return node._replace(
                mu=zero(node.mu),
                nu=zero(node.nu),
                count=jnp.zeros_like(node.count) if reset_count else node.count,
            )
        if isinstance(node, tuple) and hasattr(node, "_fields"):
            return type(node)(*[walk(c) for c in node])
        if isinstance(node, (list, tuple)):
            return type(node)(walk(c) for c in node)
        return node

    new_state = walk(opt_state)
    if not found:
        raise ValueError(
            "ReDo found no optax.adam state to reset. It only knows how to clear "
            "Adam moments; adapt _reset_adam for other optimizers."
        )
    return new_state


def apply_redo(train_state, policy_scores, value_scores, key,
               fc_layer_depth, tau=DEFAULT_TAU, targets="both",
               reset_adam_count=True):
    """One ReDo pass. Returns (new_train_state, stats).

    `policy_scores` / `value_scores` are the per-layer neuron scores from
    `collect_scores`. `targets` selects which stacks are recycled
    ('policy' | 'value' | 'both'), matching --redo_targets in the gymnax trainer.

    Traceable: no Python branch depends on a traced value, so this can be called
    from inside `lax.cond` within the update scan.
    """
    policy_pairs, value_pairs = layer_layout(fc_layer_depth)

    p_masks = _masks_from_scores(policy_scores, tau) if targets in ("policy", "both") else []
    v_masks = _masks_from_scores(value_scores, tau) if targets in ("value", "both") else []

    # Pair lists must line up with the mask lists actually used.
    p_pairs = policy_pairs[:len(p_masks)]
    v_pairs = value_pairs[:len(v_masks)]

    k_p, k_v = jax.random.split(key)
    params = _apply_to_trunk(
        train_state.params,
        lambda trunk: _reinit_stack(
            _reinit_stack(trunk, p_pairs, p_masks, k_p),
            v_pairs, v_masks, k_v),
    )
    opt_state = _reset_adam(train_state.opt_state, p_pairs, v_pairs,
                            p_masks, v_masks, reset_adam_count)

    n_policy = sum(jnp.sum(m.astype(jnp.int32)) for m in p_masks) if p_masks else jnp.array(0)
    n_value = sum(jnp.sum(m.astype(jnp.int32)) for m in v_masks) if v_masks else jnp.array(0)
    total = sum(m.size for m in p_masks) + sum(m.size for m in v_masks)
    stats = {
        "redo_policy_reset": jnp.asarray(n_policy, dtype=jnp.int32),
        "redo_value_reset": jnp.asarray(n_value, dtype=jnp.int32),
        "redo_total_neurons": jnp.asarray(max(total, 1), dtype=jnp.int32),
    }
    return train_state.replace(params=params, opt_state=opt_state), stats
