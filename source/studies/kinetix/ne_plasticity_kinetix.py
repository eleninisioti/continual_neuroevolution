"""Plasticity and diversity metrics for the kinetix NE trainers.

One definition, imported by ga/dns/es_continual.py, because CLAUDE.md asks for
"a single part of the code responsible to calculate these metrics that is
shared/interfaced from all methods and tasks" -- and three copies of a churn
estimator is exactly how two of them quietly stop agreeing.

WHY THIS IS NOT source/metrics/
--------------------------------------
The shared modules assume a feed-forward MLP whose layers are named `Dense_*`
and an `apply(params, obs) -> logits` signature. kinetix's actor is a
convolutional network over pixel observations wrapped in ScannedRNN, called as
`apply(params, hstate, (obs, dones))`, and it publishes its own per-neuron
dormancy scores through flax `sow`. So the *formulas* come from the shared
modules (imported below where they exist) while the *plumbing* has to be
kinetix-specific.

OBSERVER DISCIPLINE
-------------------
Everything here is a pure observer: it consumes a frozen probe batch drawn from
a private RNG stream and never writes back into the search. A run with these
columns and one without are the same run. Verified for dormancy: the network's
logits and hidden state are bit-identical with counting on and off.
"""

import jax
import jax.numpy as jnp

from source.metrics.behaviour_descriptors import mean_pairwise_euclidean


def make_ne_metrics(network, reshaper, ScannedRNN, *, dormancy_sample=64,
                    churn_sample=64, diversity_sample=32):
    """Build the metric functions for one trainer's network and reshaper.

    A factory rather than free functions because every one of these needs the
    network, the ParameterReshaper and ScannedRNN, and threading three objects
    through six call sites in three trainers is how they drift apart.
    """

    def _probe_inputs(probe_obs):
        n_probe = jax.tree_util.tree_leaves(probe_obs)[0].shape[0]
        dones = jnp.zeros((n_probe, 1), dtype=jnp.bool_)
        hstate0 = ScannedRNN.initialize_carry(1)
        obs_in = jax.tree.map(lambda x: x[:, None, ...], probe_obs)
        return obs_in, dones, hstate0

    def _greedy(flat, obs_in, dones, hstate0):
        """Greedy action. MultiDiscreteActionDistribution has no `mode`, but it
        holds one distrax.Categorical per action dimension, so the greedy action
        is the argmax of each sub-distribution's logits."""
        params = reshaper.reshape_single(flat)
        _, pi = network.apply(params, hstate0, (obs_in, dones))
        return jnp.stack([jnp.argmax(d.logits, axis=-1)
                          for d in pi.distributions], axis=-1)

    def population_dormancy(flat_sample, probe_obs, redo_tau):
        """Mean dormant / zero fraction over a sample of the population."""
        obs_in, dones, hstate0 = _probe_inputs(probe_obs)

        def _one(flat):
            params = reshaper.reshape_single(flat)
            _, state = network.apply(params, hstate0, (obs_in, dones),
                                     mutable=["intermediates"])
            leaves = jax.tree_util.tree_leaves(state["intermediates"])
            scores = jnp.concatenate([jnp.ravel(l) for l in leaves])
            return (jnp.mean(scores <= redo_tau).astype(jnp.float32),
                    jnp.mean(scores == 0.0).astype(jnp.float32))

        fr, zf = jax.vmap(_one)(flat_sample[:dormancy_sample])
        return {"pop_dormant_fraction": jnp.mean(fr),
                "pop_zero_fraction": jnp.mean(zf)}

    def population_churn(parents, offspring, probe_obs):
        """Parent -> offspring argmax disagreement: the PRIMARY churn column.

        CLAUDE.md is explicit that NE churn pairs each offspring with the genome
        it was bred from, not consecutive elites -- the elite changes lineage
        between generations, so elite-to-elite is not one network before and
        after one update.
        """
        obs_in, dones, hstate0 = _probe_inputs(probe_obs)
        g = lambda f: _greedy(f, obs_in, dones, hstate0)
        a = jax.vmap(g)(parents[:churn_sample])
        b = jax.vmap(g)(offspring[:churn_sample])
        return {"ne_pop_churn_action": jnp.mean((a != b).astype(jnp.float32))}

    def elite_churn(prev_elite, cur_elite, probe_obs):
        """Elite(t-1) -> elite(t). The SECONDARY column, kept for comparison."""
        obs_in, dones, hstate0 = _probe_inputs(probe_obs)
        g = lambda f: _greedy(f, obs_in, dones, hstate0)
        return {"ne_elite_churn_action":
                jnp.mean((g(prev_elite) != g(cur_elite)).astype(jnp.float32))}

    def population_diversity(flat_sample, fitnesses, ep_lengths, probe_obs):
        """The environment-agnostic bd_* descriptors.

        Handcrafted, occupancy and action-frequency descriptors are deliberately
        absent: they need a notion of "where the agent was" that is specific to
        each environment and kinetix has no such definition. AURORA fills that
        role instead. What remains needs only genomes, fitnesses and the probe
        batch, so it carries over unchanged.

        Formulas follow behaviour_descriptors.py exactly (upper-triangular
        pairwise means, 1e-12 log guards). MultiDiscrete is the one departure:
        each quantity is computed per action dimension and averaged, which
        reduces to the shared definition when there is one dimension.
        """
        flat_sample = flat_sample[:diversity_sample]
        n = flat_sample.shape[0]
        obs_in, dones, hstate0 = _probe_inputs(probe_obs)

        def _logits(flat):
            params = reshaper.reshape_single(flat)
            _, pi = network.apply(params, hstate0, (obs_in, dones))
            return [d.logits for d in pi.distributions]

        per_dim = jax.vmap(_logits)(flat_sample)
        mask = jnp.triu(jnp.ones((n, n)), k=1)
        denom = jnp.sum(mask)

        dis_acc = js_acc = ent_acc = 0.0
        for lg in per_dim:
            lp = jax.nn.log_softmax(lg, axis=-1)
            pr = jnp.exp(lp)
            act = jnp.argmax(lg, axis=-1)
            d = (act[:, None, ...] != act[None, ...]).astype(jnp.float32)
            dis_acc += jnp.sum(d.reshape(n, n, -1).mean(axis=-1) * mask) / denom
            p_i, p_j = pr[:, None], pr[None, :]
            m = 0.5 * (p_i + p_j)
            log_m = jnp.log(m + 1e-12)
            kl_i = jnp.sum(p_i * (jnp.log(p_i + 1e-12) - log_m), axis=-1)
            kl_j = jnp.sum(p_j * (jnp.log(p_j + 1e-12) - log_m), axis=-1)
            js_acc += jnp.sum((0.5 * (kl_i + kl_j)).reshape(n, n, -1).mean(axis=-1)
                              * mask) / denom
            ent_acc += jnp.mean(-jnp.sum(pr * lp, axis=-1))
        k = float(len(per_dim))

        diff = flat_sample[:, None, :] - flat_sample[None, :, :]
        gd = jnp.sum(jnp.sqrt(jnp.sum(diff ** 2, axis=-1) + 1e-12) * mask) / denom

        return {"bd_probe_disagreement": dis_acc / k,
                "bd_probe_js": js_acc / k,
                "bd_probe_policy_entropy": ent_acc / k,
                "bd_genomic_diversity": gd,
                "bd_fitness_std": jnp.std(fitnesses),
                "bd_episode_steps_std": jnp.std(ep_lengths)}

    return {"dormancy": population_dormancy, "churn": population_churn,
            "elite_churn": elite_churn, "diversity": population_diversity,
            "samples": {"dormancy": dormancy_sample, "churn": churn_sample,
                        "diversity": diversity_sample}}


def aurora_metrics(aurora, aurora_state, traj, loss_last, seed, max_n=32):
    """bd_aurora_diversity / bd_aurora_loss from an encoded population.

    Host-side: encoding and auto-encoder training are numpy/optax work that
    cannot live inside a lax.scan. The latent space is PER-RUN, so
    bd_aurora_diversity is comparable within a run and never across runs.
    """
    desc = aurora.encode(traj, aurora_state)
    out = {"bd_aurora_diversity": mean_pairwise_euclidean(
        jax.device_get(desc), max_n=max_n, seed=seed)}
    if loss_last is not None:
        out["bd_aurora_loss"] = float(loss_last)
    return out
