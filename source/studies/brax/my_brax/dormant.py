"""Dormant-neuron tracking for the brax PPO cores.

`ppo_continual_train.py` carried the tracker, the mask computation and the
per-iteration logging inline, and `ppo_train.py` carried none of them -- so
`--use_redo` on the STATIONARY ant and cheetah produced runs with no dormancy
column at all, while the continual ones had a full one. Every plasticity figure
built off the stationary trees read an empty series. This module is the one
copy; both cores import it.

It lives under `my_brax/` rather than `source/` because it is specific to
these networks: the scoring happens inside `apply_with_activation_stats`, which
only `my_brax.networks` provides. `source/algorithms/rl/redo.py` is the suite-agnostic
definition of what "dormant" means and is what the NE side
(`source/metrics/plasticity.py`) uses; the two agree on the criterion.
"""

from typing import List, Tuple

import jax.numpy as jnp
import numpy as np

from source.studies.brax.my_brax import networks as my_brax_networks


class DormantNeuronTracker:
    """Dormant neurons and how long each has been dormant, across training.

    The age is the point of the class -- a fraction alone cannot distinguish a
    network where a different 5% goes quiet each iteration from one where the
    same 5% has been dead since iteration 3, and only the second is loss of
    plasticity.
    """

    def __init__(self, layer_sizes: List[int]):
        """`layer_sizes` is the hidden layer widths, excluding the output."""
        self.layer_sizes = layer_sizes
        # layer_idx -> per-neuron count of consecutive dormant epochs
        self.dormant_ages = {i: np.zeros(size, dtype=np.int32)
                             for i, size in enumerate(layer_sizes)}
        self.currently_dormant = {i: np.zeros(size, dtype=bool)
                                  for i, size in enumerate(layer_sizes)}
        self.total_epochs = 0

    def update(self, dormant_masks: List[np.ndarray]):
        """One epoch's masks: True where the neuron is dormant."""
        self.total_epochs += 1
        for layer_idx, mask in enumerate(dormant_masks):
            if layer_idx not in self.dormant_ages:
                continue
            mask = np.asarray(mask, dtype=bool)
            # Still dormant -> age grows; woken or never dormant -> reset to 0.
            self.dormant_ages[layer_idx] = np.where(
                mask, self.dormant_ages[layer_idx] + 1, 0)
            self.currently_dormant[layer_idx] = mask

    def get_stats(self) -> dict:
        stats = {}
        total_dormant = 0
        total_neurons = 0
        all_ages = []

        for layer_idx, size in enumerate(self.layer_sizes):
            dormant_mask = self.currently_dormant[layer_idx]
            ages = self.dormant_ages[layer_idx]

            n_dormant = int(np.sum(dormant_mask))
            stats[f'dormant/layer{layer_idx}/count'] = n_dormant
            stats[f'dormant/layer{layer_idx}/percent'] = (
                float(100.0 * n_dormant / size) if size > 0 else 0.0)

            if n_dormant > 0:
                dormant_ages = ages[dormant_mask]
                stats[f'dormant/layer{layer_idx}/age_mean'] = float(np.mean(dormant_ages))
                stats[f'dormant/layer{layer_idx}/age_max'] = int(np.max(dormant_ages))
                all_ages.extend(dormant_ages.tolist())
            else:
                stats[f'dormant/layer{layer_idx}/age_mean'] = 0.0
                stats[f'dormant/layer{layer_idx}/age_max'] = 0

            total_dormant += n_dormant
            total_neurons += size

        stats['dormant/total_count'] = int(total_dormant)
        stats['dormant/total_neurons'] = int(total_neurons)
        stats['dormant/total_percent'] = (
            float(100.0 * total_dormant / total_neurons) if total_neurons > 0 else 0.0)

        if all_ages:
            stats['dormant/age_mean'] = float(np.mean(all_ages))
            stats['dormant/age_max'] = int(np.max(all_ages))
            stats['dormant/age_median'] = float(np.median(all_ages))
        else:
            stats['dormant/age_mean'] = 0.0
            stats['dormant/age_max'] = 0
            stats['dormant/age_median'] = 0.0
        return stats


def compute_dormant_masks(
    ppo_network,
    normalizer_params,
    params,
    sample_obs: jnp.ndarray,
    tau: float = my_brax_networks.REDO_DEFAULT_TAU,
) -> Tuple[List[np.ndarray], List[np.ndarray], float, float, dict, dict]:
    """Dormant masks and activation stats for the policy and value networks.

    Pure measurement -- `_apply_redo` recomputes its own masks. `tau` is applied
    inside the network (an `MLP` attribute bound by the network factory), so it
    is accepted here for the log line only.

    The networks return boolean masks. They used to return index arrays with
    `-1` marking "not dormant", decoded here with `>= 0` -- correct -- but the
    same arrays went to `apply_redo_to_params`, which decoded them with
    `.at[idx].set(True)`, and `-1` wrapped round to the last neuron. See
    source/studies/brax/my_brax/networks.py.
    """
    if isinstance(sample_obs, dict):
        sample_obs = sample_obs.get('state', list(sample_obs.values())[0])
    sample_obs_flat = sample_obs.reshape(-1, sample_obs.shape[-1])

    def one_network(network, net_params, name):
        if not hasattr(network, 'apply_with_activation_stats'):
            raise ValueError(
                f'Dormancy was requested but the {name} network cannot report it. '
                f"Build the networks with my_brax.networks, not brax's."
            )
        _, frac, masks, act_stats = network.apply_with_activation_stats(
            normalizer_params, net_params, sample_obs_flat
        )
        masks = [np.asarray(m, dtype=bool) for m in masks]
        stats = {}
        for i, st in enumerate(act_stats):
            stats[f'layer{i}/mean'] = float(st['layer_mean'])
            stats[f'layer{i}/std'] = float(st['layer_std'])
            stats[f'layer{i}/min_neuron_mean'] = float(st['min_neuron_mean'])
            stats[f'layer{i}/max_neuron_mean'] = float(st['max_neuron_mean'])
            stats[f'layer{i}/min_neuron_score'] = float(st['min_neuron_score'])
            stats[f'layer{i}/max_neuron_score'] = float(st['max_neuron_score'])
        return masks, float(frac), stats

    policy_masks, policy_frac, policy_activation_stats = one_network(
        ppo_network.policy_network, params.policy, 'policy')
    value_masks, value_frac, value_activation_stats = one_network(
        ppo_network.value_network, params.value, 'value')

    return (policy_masks, value_masks, policy_frac, value_frac,
            policy_activation_stats, value_activation_stats)


def add_dormancy_metrics(metrics, ppo_network, normalizer_params, params,
                         sample_obs, policy_tracker, value_tracker,
                         tau=my_brax_networks.REDO_DEFAULT_TAU):
    """Measure dormancy and write it into `metrics` under the logged names.

    The whole per-iteration block, so the two cores share it rather than each
    keeping their own 40 lines. Mutates and returns `metrics`.

    Never raises: a failed measurement must not kill a training run that is
    otherwise fine. It records `policy/dormant_error` instead, which is visible
    in the run rather than only in a log line -- the previous version swallowed
    the exception into a print, and a sweep could finish with an empty dormancy
    column and no record of why.
    """
    try:
        (policy_masks, value_masks, policy_frac, value_frac,
         policy_act_stats, value_act_stats) = compute_dormant_masks(
            ppo_network, normalizer_params, params, sample_obs, tau=tau)

        if policy_masks and policy_tracker is not None:
            policy_tracker.update(policy_masks)
            for k, v in policy_tracker.get_stats().items():
                metrics[f'policy/{k}'] = v
        if value_masks and value_tracker is not None:
            value_tracker.update(value_masks)
            for k, v in value_tracker.get_stats().items():
                metrics[f'value/{k}'] = v

        metrics['policy/dormant_fraction'] = policy_frac
        metrics['value/dormant_fraction'] = value_frac
        for k, v in policy_act_stats.items():
            metrics[f'policy/activation/{k}'] = v
        for k, v in value_act_stats.items():
            metrics[f'value/activation/{k}'] = v
    except Exception as exc:  # pragma: no cover - diagnostics only
        metrics['policy/dormant_error'] = f'{type(exc).__name__}: {exc}'
    return metrics


def make_trackers(policy_hidden_sizes, value_hidden_sizes):
    """A (policy, value) tracker pair for the given hidden widths."""
    return (DormantNeuronTracker(list(policy_hidden_sizes)),
            DormantNeuronTracker(list(value_hidden_sizes)))
