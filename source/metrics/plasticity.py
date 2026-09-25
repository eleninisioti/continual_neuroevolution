"""Plasticity diagnostics reported for every method, NE and RL alike.

Two quantities, measured the same way on both sides of the NE/RL divide so a
single table can carry all eight methods:

  **dormancy**  the fraction of hidden units that have stopped responding.
                Defined and scored in `source/algorithms/rl/redo.py` -- this module only
                lifts it to a population. ReDo's own recycling uses the same
                score, so "how dormant is this network" means one thing here.

  **churn**     how much one update moved the policy's output on a frozen batch
                of states. Reported for every method, which is what the
                reference does -- see below.

## The measure is per suite; the pairing is per family

Tang et al. measure churn with a different estimator per action space, and this
module follows them rather than inventing a single one:

  discrete (gymnax)     cross-entropy H(pi_before, pi_after)
                        `crl_procgen/vis_train_procgen_c_chain.py:192`
  continuous (ant)      mean squared difference of action means
                        `crl_dmc/crl_run_ppo_c_chain_dmc.py:339`

Both are logged for EVERY method, not just C-CHAIN: the reference's own
vanilla-PPO scripts log the identical quantity (`crl_run_ppo_dmc.py:318`), so
churn-for-all-methods is its practice and not an extension of it.

The one deliberate departure: the gymnax **NE** policies report
`action_disagreement` -- the fraction of probe states whose argmax action
changed -- and not the cross-entropy. They are deterministic argmax policies, so
a softmax over their logits has an arbitrary temperature and a cross-entropy
over it would be scale-meaningless. The RL side of gymnax still reports the
cross-entropy, because its policies really are categorical distributions.

## What counts as "one update"

  RL      one gradient step. The reference keeps a buffer of historical policies
          and compares against `[-2]`; here C-CHAIN reads its own
          `chain_state.ref_policy_params` and the other methods read a
          `prev_params` threaded out of the minibatch scan.

  NE      one application of the variation operator: `ne_churn` pairs each
          offspring with the genome it was bred from, averaged over the
          population (`pairwise_churn`). Elite-to-elite is kept as a SECONDARY
          column, `ne_elite_churn_action`, because the elite changes lineage
          between generations and so is not one network before and after one
          update -- read `pairwise_churn`'s docstring before using either.

          Open_ES has no parent/offspring relation at all, so its `ne_churn` is
          `mean_t -> mean_{t+1}`, the update its search actually performs.

## The frozen probe batch


Churn is measured on a fixed batch of states collected once, at the start of the
run, and then frozen -- the same convention (and the same helper) as the
behavioural-diversity probes in `source/metrics/behaviour_descriptors.py`.
Measuring on each update's own rollout instead would confound "the policy
changed" with "the state distribution moved", which in the continual settings is
precisely the thing being manipulated.

Its weakness is the mirror of its strength, and matters in the continual blocks:
states frozen under sub-task 0 drift off-distribution as the sub-tasks change,
so late-run churn is measured somewhere the policy no longer lives. That is a
deliberate trade for comparability across time, and it is the same trade the
diversity probes already make.

## These are observers and must stay observers

Every function here is pure measurement. None consumes the trainer's RNG stream:
callers pass a key derived from a separate stream (`seed + N_000_000`, the
convention `track_diversity` already uses), so a run with diagnostics on and one
with them off are the same run, bit for bit. Verify that before believing any
number produced alongside them.
"""

import jax
import jax.numpy as jnp

from source.algorithms.rl import redo as core_redo
from source.metrics import ntk as core_ntk


def greedy_actions(logits):
    """Chosen action per state. The NE and RL policies both act this way here."""
    return jnp.argmax(logits, axis=-1)


def action_disagreement(logits_a, logits_b):
    """Fraction of states whose greedy action differs between two policies.

    Both arguments are (batch, num_actions) logits evaluated on the SAME states.
    Returns a scalar in [0, 1]: 0 = the two policies act identically here, 1 =
    they never agree.
    """
    return jnp.mean(greedy_actions(logits_a) != greedy_actions(logits_b)).astype(jnp.float32)


def action_disagreement_continuous(actions_a, actions_b, action_range=2.0):
    """Continuous-action counterpart of `action_disagreement`.

    Both arguments are (batch, action_dim) ALREADY-SQUASHED actions -- not
    logits -- evaluated on the same states. Returns the mean absolute change per
    actuator divided by `action_range`, so a policy that moves every actuator
    from one end of its range to the other scores 1.0 and an unchanged policy
    scores 0.0: the same [0, 1] reading as the discrete version.

    Why this and not the argmax test: cheetah's and ant's NE policies are
    deterministic tanh-squashed vectors, so there is no `argmax` to disagree on,
    and every pair of distinct continuous actions would count as a full
    disagreement -- churn would read ~1.0 always and measure nothing.

    Why this and not the raw MSE `my_brax/cchain.py` uses: that is the C-CHAIN
    regulariser's own control signal, it is unbounded, and squaring makes it
    dominated by whichever actuator moved most. This is the cross-method
    OBSERVER, and it has to sit on the same scale as the discrete column to be
    read in the same table. The repo already draws exactly this distinction on
    the RL side -- see the `chain/policy_churn` note at the top of this module.

    `action_range` is 2.0 for a tanh output in [-1, 1], which is every
    continuous policy here (`ContinuousMLPPolicy` in source/algorithms/networks.py).

    NOTE: comparable across methods WITHIN a suite, which is what every figure
    needs; not comparable against the discrete column across suites, and
    nothing compares across suites (see POLICY_ARCH in source/algorithms/networks.py).
    """
    diff = jnp.abs(jnp.asarray(actions_a) - jnp.asarray(actions_b))
    return jnp.mean(diff).astype(jnp.float32) / action_range


def action_churn_mse(actions_a, actions_b):
    """C-CHAIN's own continuous churn: mean squared difference of action means.

    This is the published definition, verbatim. The reference computes

        policy_churn = ((cur_ref_action_means - ref_action_means) ** 2).mean()

    on the DeepMind Control suite -- see
    `inspiration/C-CHAIN/crl_dmc/crl_run_ppo_c_chain_dmc.py:339`, and note that
    `crl_run_ppo_dmc.py:318` logs the identical quantity for VANILLA PPO. So
    churn-for-every-method is the reference's own practice, not an extension of
    it, and this is the function every continuous-control method here reports.

    Both arguments are (batch, action_dim) action means -- the distribution's
    scale half already dropped -- evaluated on the same states.

    Unbounded, unlike `action_disagreement_continuous`, and dominated by
    whichever actuator moved most. That is a real cost and it is paid on
    purpose: matching the published estimator matters more than bounding it,
    because nothing compares across suites anyway (see POLICY_ARCH) and within
    a suite every method is measured identically.
    """
    diff = jnp.asarray(actions_a) - jnp.asarray(actions_b)
    return jnp.mean(jnp.square(diff)).astype(jnp.float32)


def policy_churn_cross_entropy(logits_before, logits_after):
    """C-CHAIN's discrete churn: H(pi_before, pi_after), per sample.

    The published estimator for discrete control --
    `inspiration/C-CHAIN/crl_procgen/vis_train_procgen_c_chain.py:192` -- and
    its vanilla-PPO script logs the identical quantity, which is why every
    gymnax RL method reports it and not only C-CHAIN.

    Cross-entropy rather than KL, matching the reference: the two differ by the
    constant entropy of pi_before, so the gradient is identical and the reported
    value stays on the scale the coefficient controller is calibrated against.
    `stop_gradient` on the reference so this is a pull towards the old policy
    when it is used as a loss, rather than a mutual averaging of the two.

    Lives here, not in `source/studies/gymnax/cchain.py`, because it is now a measure
    every method needs: the stationary gymnax trainer imports cchain behind a
    try/except so a checkout without it can still run PPO, and a churn column
    for all four methods cannot sit behind that guard.

    NOT USABLE FOR THE NE POLICIES. It needs pi to be a distribution, and the
    NE policies here are deterministic argmax -- see the note at the top of this
    module. They report `action_disagreement` instead.
    """
    ref_probs = jax.nn.softmax(jax.lax.stop_gradient(logits_before))
    cur_log_probs = jax.nn.log_softmax(logits_after)
    return -jnp.sum(ref_probs * cur_log_probs, axis=-1)


def churn_action_disagreement(apply_fn, params_before, params_after, probe_obs,
                              continuous=False, action_range=2.0):
    """Churn between two successive versions of one policy.

    `apply_fn(params, obs_batch) -> logits` (discrete) or `-> actions`
    (continuous). For RL, `params_before` is the policy as of the previous
    update; for NE, the previous generation's elite.
    """
    out_before = apply_fn(params_before, probe_obs)
    out_after = apply_fn(params_after, probe_obs)
    if continuous:
        return float(action_churn_mse(out_before, out_after))
    return float(action_disagreement(out_before, out_after))


def pairwise_churn(apply_flat, parents, offspring, probe_obs,
                   continuous=False, max_sample=128):
    """Mean churn over (parent, offspring) pairs -- the NE update's own churn.

    THIS, NOT ELITE-TO-ELITE, IS THE ANALOGUE OF THE PAPER'S CHURN.
    Tang et al. measure the change in ONE network's outputs, on states outside
    the update's batch, caused by ONE learning update. Three properties:

      (a) same network before and after
      (b) exactly one update
      (c) evaluated off-batch

    Elite-to-elite satisfies (b) and (c) but breaks (a): `SimpleGA.tell` re-sorts
    a combined archive, so the elite at generation t+1 is frequently not a
    descendant of the elite at t -- it is a different genome from a different
    lineage, and the "churn" is then a jump between two unrelated networks
    rather than the effect of an update on one.

    A (parent, offspring) pair satisfies all three: the variation operator IS
    the update, the offspring descends from that parent by construction, and the
    probe batch was frozen at generation 0. Averaging over the population is the
    population analogue of RL's per-update scalar.

    WHAT IT MEASURES, AND HOW IT DIFFERS FROM THE RL COLUMN. This is (variation
    magnitude) x (local sensitivity of the network). Since the operator's sigma
    is fixed, drift over a run is a sensitivity signal -- a network that has
    lost plasticity moves LESS per unit parameter change. That is a genuine
    plasticity measure, but it isolates the VARIATION half of the NE update and
    excludes SELECTION, which is the data-driven half. An RL gradient step
    contains both. The two columns are therefore close analogues, not the same
    quantity, and the elite column is kept beside this one for that reason.

    `parents` and `offspring` are (n, num_dims) flat genomes in correspondence:
    `offspring[i]` was produced from `parents[i]`. Capped at `max_sample` pairs,
    like the other population measures here.
    """
    n = int(min(max_sample, offspring.shape[0]))
    par = jnp.asarray(parents)[:n]
    off = jnp.asarray(offspring)[:n]
    out_par = jax.vmap(lambda g: apply_flat(g, probe_obs))(par)
    out_off = jax.vmap(lambda g: apply_flat(g, probe_obs))(off)
    if continuous:
        per_pair = jax.vmap(action_churn_mse)(out_par, out_off)
    else:
        per_pair = jax.vmap(action_disagreement)(out_par, out_off)
    return float(jnp.mean(per_pair))


def dormancy(params, probe_obs, num_hidden, activation_fn, criterion,
             tau=core_redo.DEFAULT_TAU, prefix='Dense_', activations_fn=None):
    """Dormant-unit diagnostics for one network. Thin wrapper over core.redo.

    Returned dict is core.redo's: dormant_fraction, dormant_count, zero_fraction,
    total_neurons, per_layer. Scored with the same criterion ReDo recycles on, so
    the `redo` column's dormancy is directly comparable to every other method's.
    """
    return core_redo.dormant_stats(
        params, probe_obs, num_hidden, tau=tau,
        activation_fn=activation_fn, criterion=criterion, prefix=prefix,
        activations_fn=activations_fn,
    )


def population_dormancy(unflatten, flat_population, probe_obs, num_hidden,
                        activation_fn, criterion, tau=core_redo.DEFAULT_TAU,
                        max_sample=64, prefix='Dense_', activations_fn=None):
    """Dormancy across an NE population: the mean over individuals.

    `unflatten(flat_genome) -> params`. Capped at `max_sample` individuals
    because this is a Python loop over `dormant_stats`, which is host-side; the
    cap is a cost control, and 64 of 512 is enough to place the mean to well
    under the between-generation variation.

    Returns the mean dormant and zero fractions, plus the sample size actually
    used so a reader can tell how noisy the estimate is.
    """
    n = int(min(max_sample, flat_population.shape[0]))
    fracs = []
    zeros = []
    for i in range(n):
        st = core_redo.dormant_stats(
            unflatten(flat_population[i]), probe_obs, num_hidden, tau=tau,
            activation_fn=activation_fn, criterion=criterion, prefix=prefix,
            activations_fn=activations_fn,
        )
        fracs.append(st['dormant_fraction'])
        zeros.append(st['zero_fraction'])
    return {
        'pop_dormant_fraction': float(sum(fracs) / max(n, 1)),
        'pop_zero_fraction': float(sum(zeros) / max(n, 1)),
        'pop_dormant_sample': n,
    }


class NEPlasticityTracker:
    """Churn and dormancy for a population-based method, per generation.

    Gives the NE trainers the two columns the RL trainers report, measured the
    same way so GA/ES/DNS and PPO/TRAC/ReDo/C-CHAIN can share a table:

      ne_elite_churn_action   fraction of probe states where the greedy action
                              of the BEST genome changed since last generation.
                              The direct analogue of the RL churn, which is the
                              change in the policy you would actually deploy.
      ne_pop_churn_action     mean pairwise action disagreement WITHIN the
                              current population. This is behavioural diversity,
                              not change over time -- a different question, and
                              one the RL methods cannot answer at all because
                              they have a single policy. Reported because a
                              population can hold its elite steady while
                              churning internally, and the two together say
                              which is happening.
      ne_elite_dormant_fraction / ne_pop_dormant_fraction
                              ReDo's dormancy score on the elite, and averaged
                              over a sample of the population.
      ne_centroid_*           the same churn, dormancy and NTK columns for the
                              CENTROID -- the network the centroid lineplot
                              scores. Present only when the caller passes
                              `centroid=`; see `update`. The elite and the
                              centroid are different networks in every NE arm
                              (in ES/NES the elite is a sampled offspring, in
                              GA/DNS an argmax that changes lineage), so a
                              figure drawn beside the centroid curve wants
                              these and not the elite ones.

    Genomes are not in correspondence across generations -- selection reorders
    them -- so there is no meaningful per-individual churn. The elite is the one
    genome with a stable identity from one generation to the next, which is why
    the time-series measure is defined on it.

    Like the diversity tracker this is a pure observer: it is handed a key from a
    private stream and never touches the search's.
    """

    def __init__(self, apply_flat, unflatten, num_hidden, activation_fn,
                 criterion, tau=core_redo.DEFAULT_TAU, num_probe=512,
                 max_pop_sample=64, max_pairwise=128, prefix='Dense_',
                 continuous=False, action_range=2.0, max_ntk_probe=128,
                 activations_fn=None, probe_batch_dims=None):
        """`continuous=True` for the cheetah/ant policies.

        The churn measure is the only thing it changes -- dormancy is scored on
        hidden activations and does not care what the output layer means. It
        defaults to False so the gymnax trainers, which are mid-sweep, keep
        exactly the numbers they have.
        """
        self.apply_flat = apply_flat
        self.unflatten = unflatten
        self.num_hidden = num_hidden
        self.activation_fn = activation_fn
        self.criterion = criterion
        self.tau = tau
        self.num_probe = num_probe
        self.max_pop_sample = max_pop_sample
        self.max_pairwise = max_pairwise
        self.prefix = prefix
        self.continuous = continuous
        self.action_range = action_range
        self.max_ntk_probe = max_ntk_probe
        # Non-MLP policies report their own hidden activations and their own
        # probe-batch shape; see `dormant_stats` and `collect_probe_states`.
        self.activations_fn = activations_fn
        self.probe_batch_dims = probe_batch_dims
        self.probe_obs = None
        self._prev_elite = None
        self._prev_incumbent = None
        self._prev_centroid = None

    def started(self):
        return self.probe_obs is not None

    def start(self, key, observations):
        """Freeze the probe batch from the initial population's trajectories."""
        from source.metrics.behaviour_descriptors import collect_probe_states
        self.probe_obs = collect_probe_states(
            observations, num_probe=self.num_probe, key=key,
            batch_dims=self.probe_batch_dims)

    def _pop_pairwise_churn(self, key, flat_population):
        n = int(min(self.max_pairwise, flat_population.shape[0]))
        idx = jax.random.choice(key, flat_population.shape[0], shape=(n,), replace=False)
        genomes = jnp.asarray(flat_population)[idx]
        outputs = jax.vmap(lambda g: self.apply_flat(g, self.probe_obs))(genomes)
        mask = jnp.triu(jnp.ones((n, n)), k=1)
        if self.continuous:
            # MSE per pair, the same estimator `action_churn_mse` uses for the
            # churn columns. It was mean |da| / action_range until the churn
            # columns moved to the published MSE, which left this on a
            # different scale from `ne_churn` and `ne_elite_churn_action` in the
            # same record -- 0.18 against 1e-3 on the ant, three columns and two
            # units. Whatever the estimator, all three move together.
            diff = outputs[:, None] - outputs[None, :]
            pair = jnp.square(diff).reshape(n, n, -1).mean(axis=-1)
        else:
            actions = greedy_actions(outputs)                 # (n, batch, ...)
            disagree = (actions[:, None] != actions[None, :]).astype(jnp.float32)
            # Reshape before the mean so this also handles a policy whose action
            # is a VECTOR: the scheduling policy (dropped 2026-09-08) emitted
            # one job per machine, so
            # `actions` is (n, batch, num_machines) and averaging over the last
            # axis alone would leave a per-state matrix instead of a scalar per
            # pair. Identical arithmetic for the (n, batch) case.
            pair = disagree.reshape(disagree.shape[0], disagree.shape[1], -1).mean(axis=-1)
        return float(jnp.sum(pair * mask) / jnp.sum(mask))

    def update(self, key, flat_population, elite_flat, parents=None,
               incumbent=None, offspring=None, centroid=None):
        """One generation's diagnostics. Returns {} until `start` has been called.

        `parents` is (n, num_dims) in correspondence with the offspring:
        `parents[i]` is the genome offspring `i` was bred from. When given,
        `ne_churn` is the primary column -- the paper's churn, measured across
        one update of one network (see `pairwise_churn`).

        `offspring` defaults to `flat_population`, which is right for the GA:
        `ask` returns the offspring and `tell` only rewrites the archive, so the
        population still IS this generation's offspring when the metrics are
        logged. It is NOT right for DNS, where `dns_selection` has already
        merged parents and offspring and kept the top `population_size` of the
        pool -- pairing those survivors against `x1` would compare genomes that
        are not in correspondence at all. The DNS trainers therefore pass the
        pre-selection `offspring` explicitly, and the pairing stays honest.

        `incumbent` is for the distribution-based searches, which have no
        parent/offspring relation at all: Open_ES samples `mean + sigma*eps`, so
        its one update is `mean_t -> mean_{t+1}` and the caller passes today's
        mean here. That is the closest NE analogue to a gradient step, and it is
        why ES gets `ne_churn` without any parent bookkeeping.

        Callers may pass neither, in which case only the elite column appears.

        `centroid` is THE NETWORK THE CENTROID LINEPLOT SCORES -- the same
        vector the trainer hands to `centroid_fitness`, and not the optimizer's
        incumbent, which for the GA and DNS is a different object again. Pass
        it and every diagnostic below gets a `ne_centroid_*` twin, so the
        centroid plasticity figure describes the individual its lineplot draws
        instead of falling back on the elite. Per arm that vector is

            ES / NES   `es_state.mean`, which is also `incumbent`
            GA         `ga_state.archive.mean(axis=0)`, NOT `archive[0]`
            DNS        `population.mean(axis=0)`

        and the last two are networks the search never evaluated, which is the
        point of the row: it says whether averaging the archive still gives a
        network, not how well the method does.
        """
        if self.probe_obs is None:
            return {}

        out = {}

        # THE PRIMARY CHURN COLUMN. Same network before and after one update, so
        # it is the analogue of the RL per-update churn; see `pairwise_churn`
        # for why elite-to-elite is not.
        if parents is not None:
            kids = flat_population if offspring is None else offspring
            out['ne_churn'] = pairwise_churn(
                self.apply_flat, parents, kids, self.probe_obs,
                continuous=self.continuous, max_sample=self.max_pairwise)
            out['ne_churn_kind'] = 'parent_offspring'
        elif incumbent is not None:
            # mean_t -> mean_{t+1}. None on the first generation for the same
            # reason the elite column is: there is no previous mean.
            if self._prev_incumbent is None:
                out['ne_churn'] = None
            else:
                out['ne_churn'] = churn_action_disagreement(
                    self.apply_flat, self._prev_incumbent, incumbent,
                    self.probe_obs, continuous=self.continuous,
                    action_range=self.action_range)
            out['ne_churn_kind'] = 'incumbent_step'
            self._prev_incumbent = jnp.asarray(incumbent)
        else:
            out['ne_churn'] = None
            out['ne_churn_kind'] = None

        # SECONDARY: how much the policy you would actually DEPLOY moved. Kept
        # beside `ne_churn` because it answers a question the primary column
        # does not -- but it is not the paper's churn, because the elite can
        # change lineage between generations.
        #
        # Undefined at the first generation, since there is no previous elite.
        # Reported as None rather than 0.0 -- "no measurement" and "the policy
        # did not move" are different claims.
        if self._prev_elite is None:
            out['ne_elite_churn_action'] = None
        else:
            out['ne_elite_churn_action'] = churn_action_disagreement(
                self.apply_flat, self._prev_elite, elite_flat, self.probe_obs,
                continuous=self.continuous, action_range=self.action_range)
        self._prev_elite = jnp.asarray(elite_flat)

        out['ne_pop_churn_action'] = self._pop_pairwise_churn(key, flat_population)

        elite_stats = core_redo.dormant_stats(
            self.unflatten(elite_flat), self.probe_obs, self.num_hidden,
            tau=self.tau, activation_fn=self.activation_fn,
            criterion=self.criterion, prefix=self.prefix,
            activations_fn=self.activations_fn)
        out['ne_elite_dormant_fraction'] = elite_stats['dormant_fraction']
        out['ne_elite_zero_fraction'] = elite_stats['zero_fraction']

        # NTK effective rank of the elite -- C-CHAIN's own plasticity indicator,
        # and the CAUSE its argument assigns to the churn above (rank collapse
        # -> correlated gradients -> churn). Measured on the elite only: a
        # jacobian per individual across the whole population would dominate the
        # run, unlike the dormancy below which is cheap enough to sample.
        # See source/metrics/ntk.py.
        out.update(core_ntk.ntk_rank_stats(
            self.apply_flat, elite_flat, self.probe_obs,
            max_samples=self.max_ntk_probe, prefix='ne_elite_ntk'))

        # The centroid's twin of the three columns above. Same probe batch,
        # same tau, same estimator -- the only difference is which network.
        # Skipped entirely when the caller passes no centroid, so the arms that
        # predate this keep exactly the columns they had.
        if centroid is not None:
            centroid = jnp.asarray(centroid)
            if self._prev_centroid is None:
                out['ne_centroid_churn_action'] = None
            else:
                out['ne_centroid_churn_action'] = churn_action_disagreement(
                    self.apply_flat, self._prev_centroid, centroid,
                    self.probe_obs, continuous=self.continuous,
                    action_range=self.action_range)
            self._prev_centroid = centroid
            centroid_stats = core_redo.dormant_stats(
                self.unflatten(centroid), self.probe_obs, self.num_hidden,
                tau=self.tau, activation_fn=self.activation_fn,
                criterion=self.criterion, prefix=self.prefix,
                activations_fn=self.activations_fn)
            out['ne_centroid_dormant_fraction'] = centroid_stats['dormant_fraction']
            out['ne_centroid_zero_fraction'] = centroid_stats['zero_fraction']
            out.update(core_ntk.ntk_rank_stats(
                self.apply_flat, centroid, self.probe_obs,
                max_samples=self.max_ntk_probe, prefix='ne_centroid_ntk'))

        out.update(population_dormancy(
            self.unflatten, flat_population, self.probe_obs, self.num_hidden,
            self.activation_fn, self.criterion, tau=self.tau,
            max_sample=self.max_pop_sample, prefix=self.prefix,
            activations_fn=self.activations_fn))
        return out
