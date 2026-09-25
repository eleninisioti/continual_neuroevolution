"""The neuroevolution methods, behind one ask/tell interface.

`source/studies/generalists/train_nes.py` runs the schedule, evaluates, and writes the
artifacts; none of that is method-specific. What differs between NES, OpenES,
GA and DNS is only how a generation is proposed and how the survivors are
chosen, so that is all these classes hold. One runner, four searchers, and the
comparison cannot drift into being a comparison of four training loops.

Every searcher exposes:

    init(key, mean)                -> state
    ask(key, state)                -> (population, aux)
    tell(state, aux, fitness, descriptors=None) -> state
    incumbent(state)               -> one flat parameter vector

``fitness`` is always the return, MAXIMISED, whatever the underlying reference
implementation wanted -- SimpleGA minimises internally and is handed ``-fitness``
here rather than making every caller remember that.

``incumbent`` is the single agent a run would hand you, and it is what the
generalist score is measured on. It differs by method and the choice is not
cosmetic:

    NES / OpenES   the distribution mean. There is no population member that
                   represents the search better than its centre.
    GA             the best archive member. A GA has no mean; the mean of an
                   elite archive is not itself a policy that was ever evaluated.
    DNS            the highest-fitness member of the repertoire. The repertoire
                   is deliberately diverse, so its "average" is meaningless, and
                   novelty is a selection pressure rather than an output.

``needs_descriptors`` says whether ``tell`` requires behaviour descriptors. Only
DNS does, and the runner only pays for computing them when it is asked to.

## The GA/DNS operator ablation

GA and DNS are the same (mu + lambda) loop and differed in TWO things, not one:
what survivors are ranked by (fitness against dominated novelty) and how
offspring are produced (gaussian mutation of one parent against Iso+LineDD
recombination of two). A DNS-over-GA gap therefore could not be attributed to
novelty selection -- recombination is an equally good explanation, and no run
separated them.

Since 2026-09-08 the operator is a parameter of both classes, so the two axes
cross:

                        gaussian          isoline
    fitness             ga                ga_isoline
    dominated novelty   dns_gaussian      dns

The diagonal is the pair of methods as published; the off-diagonal is the
ablation. `ga_isoline` vs `dns` isolates the selection rule at a fixed
Iso+LineDD operator, `ga` vs `dns_gaussian` isolates it at a fixed gaussian
one, and only if both point the same way does "the diversity pressure is what
does it" survive.

A third difference, the operator WIDTHS, does not cross the same way -- see
`source/algorithms/ne/variation.py`. The crossed arms take each width from the
method that operator belongs to (`inspiration_matched.settings`), so
`ga_isoline` breeds at DNS's iso/line and `dns_gaussian` mutates at the GA's
sigma.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import random

from source.algorithms.ne.es import ES, MultiES
from source.algorithms.ne.dns import (
    _compute_dominated_novelty as dominated_novelty,  # noqa: F401
)
from source.algorithms.ne.variation import (
    GAUSSIAN, ISOLINE, VARIATIONS, resolve_params, vary,
)

# The two variation operators are shared with the GA/DNS trainers rather than
# reimplemented, and -- since 2026-09-08 -- either method can use either one.
# Until then the GA could only mutate a single parent and DNS could only
# recombine two, so a DNS-over-GA gap confounded novelty SELECTION with
# recombination. `variation='isoline'` on the GA and `variation='gaussian'` on
# DNS complete the 2x2 that separates them; see that module's docstring.

# Dominated novelty is the benchmarking paper's DNS selection criterion,
# imported rather than re-derived so the two studies rank by the same
# number. The shared version additionally guards n <= 1, which this study
# never hits (its populations are never that small) and which changes
# nothing for any population it does see.


# ---------------------------------------------------------------------------
# NES / OpenES
# ---------------------------------------------------------------------------

class ESSearcher:
    """Adapter putting `source/algorithms/ne/es.py` behind this interface.

    `nes` and `openes` are the SAME class with two arguments changed -- see that
    module. The paper's gymnax trainers drive the same core through
    `EvosaxES`, so an `openes` number here and an `openes` number there come
    from one implementation.
    """

    needs_descriptors = False

    def __init__(self, num_params, population_size, **kwargs):
        allowed = ('sigma_init', 'learning_rate', 'optimizer', 'shaping',
                   'sigma_lr', 'momentum')
        self.es = ES(num_params=num_params, population_size=population_size,
                     **{k: v for k, v in kwargs.items() if k in allowed})
        self.population_size = population_size

    def init(self, key, mean):
        return self.es.init(mean)

    def ask(self, key, state):
        return self.es.ask(key, state)

    def tell(self, state, aux, fitness, descriptors=None):
        return self.es.tell(state, aux, fitness)

    def incumbent(self, state):
        return state.mean

    def population_mean(self, state):
        """The mean of the search distribution -- identical to ``incumbent``.

        Here for the protocol, so the trainer can score "the mean of the
        population's weights" on every method without a branch. For a
        distribution-based search the two ARE the same point, so the extra
        curve costs one evaluation and says nothing new; it is on GA and DNS
        that they come apart.
        """
        return state.mean

    # A (1, lambda) strategy has no persistent population. Each generation is
    # sampled fresh around the one centroid and discarded, so there is no set of
    # genomes that could hold one specialist per sub-task -- the whole
    # distribution has to sit somewhere, and that somewhere is a single point.
    # Returning the centroid alone says exactly that, and makes the
    # population-coverage analysis structurally honest: NES cannot cover two
    # sub-tasks with two different individuals, so if its generalist score is
    # high it is because ONE genome is good at both.
    has_population = False

    def population(self, state):
        return state.mean[None, :]


class AdaptiveESState(NamedTuple):
    es: object                  # the underlying ES's state (mean, base log_sigma, optimizer)
    elite: jnp.ndarray          # (num_params,) the best sample kept, re-scored every generation
    elite_fitness: jnp.ndarray  # its score this generation; -inf before the first sample
    scale: jnp.ndarray          # multiplies sigma and the mean's step, in [scale_min, 1]
    hold: jnp.ndarray           # generations left at scale 1 after a detected collapse
    anchored: jnp.ndarray       # a jump has put the mean on a kept sample (no collapse since)
    proposal: jnp.ndarray = None      # accept_tol only: the step waiting to be scored
    has_proposal: jnp.ndarray = None  # accept_tol only: False before the first step

    @property
    def mean(self):
        return self.es.mean

    @property
    def log_sigma(self):
        return self.es.log_sigma + jnp.log(self.scale)


class AdaptiveESSearcher(ESSearcher):
    """OpenES / NES whose MEAN is meant to hold a solution until the task
    changes. ``method='openes_adaptive'`` (2026-09-14).

    Why (the Kinetix20 chain, runs_kinetix_ep128_ev1): OpenES's mean solved
    16.4/20 levels at the switch, its best final-generation sample 18.6. In
    most missed levels a sample had solved the level during the phase -- often
    at its first generation -- and the mean either never got there (a rank
    step over 512 samples barely moves for one solver) or drifted off.

    Two of the ``population_size`` evaluations score the mean and a kept
    sample, so the budget is unchanged; the ES itself samples the other
    ``population_size - 2`` antithetically.

      keep     the best sample seen replaces the kept one when it scores
               higher this generation (both on the current task).
      jump     ``jump_kappa > 0``: when the kept sample beats the mean by more
               than kappa x the samples' score spread, the mean moves to it and
               the optimizer state restarts -- rare during a smooth climb (the
               best of hundreds of samples sits ~3 spreads above the mean), the
               common case when one sample hits a narrow solution.
      settle   ``settle_rate > 0``: once a jump has ANCHORED the mean on a kept
               sample, the share p of samples the mean matches or beats moves
               a common scale of sigma and of the mean's step,
               ``scale <- clip(scale * exp(-rate (p - target)), scale_min, 1)``:
               a mean that beats nearly all its samples sits on something
               narrower than the cloud. Un-anchored, the scale relaxes back to
               1. Why gated: ungated, a mean at a LOCAL optimum also beats its
               samples, the cloud shrank and ES lost the smoothing that climbs
               out (toys, 2026-09-14: rugged and 8-coordinate ripple generalist
               found 1.00 -> 0.00); a jump is the evidence of a narrow solution
               that a local optimum does not give.
      reopen   ``restart_drop > 0``: the kept sample is a fixed point, so under
               a deterministic evaluation its score changes only when the task
               does. When it falls by more than ``restart_drop`` x its last
               |score|, the scale returns to 1 for ``restart_hold``
               generations. Nothing is told where a boundary is.
      hold     ``accept_tol`` not None: the mean moves only to LAST
               generation's proposed step, scored this generation, and only
               if it scored no worse than the mean minus accept_tol x the
               samples' spread; otherwise the mean stays and the next step is
               proposed from it. Two more evaluations (the proposal and a spare
               that keeps the ES's sample count even). Why (GH200 single-level
               test, 2026-09-14): with jump alone the mean alternated 1010...
               between the kept solver and one ES step off it on 10/12 runs --
               solving in exactly half the generations, so at the end by coin
               flip -- and the gated settle never shrank the step enough
               (sigma 0.004-0.019).
    """

    def __init__(self, num_params, population_size, jump_kappa=0.0,
                 settle_rate=0.0, settle_target=0.9, scale_min=1e-3,
                 restart_drop=0.0, restart_hold=50, accept_tol=None, **kwargs):
        self.accept_tol = None if accept_tol is None else float(accept_tol)
        self.num_extra = 2 if self.accept_tol is None else 4
        if population_size < self.num_extra + 2 or population_size % 2:
            raise ValueError('openes_adaptive needs an even population with '
                             'room for its extra evaluations')
        super().__init__(num_params, population_size - self.num_extra, **kwargs)
        self.population_size = int(population_size)
        self.jump_kappa = float(jump_kappa)
        self.settle_rate = float(settle_rate)
        self.settle_target = float(settle_target)
        self.scale_min = float(scale_min)
        self.restart_drop = float(restart_drop)
        self.restart_hold = int(restart_hold)

    def init(self, key, mean):
        s = self.es.init(mean)
        return AdaptiveESState(es=s, elite=s.mean,
                               elite_fitness=jnp.asarray(-jnp.inf, jnp.float32),
                               scale=jnp.asarray(1.0, jnp.float32),
                               hold=jnp.asarray(0, jnp.int32),
                               anchored=jnp.asarray(False),
                               proposal=s.mean, has_proposal=jnp.asarray(False))

    def ask(self, key, state):
        samples, eps = self.es.ask(key, state.es._replace(log_sigma=state.log_sigma))
        rows = [samples, state.es.mean[None], state.elite[None]]
        if self.accept_tol is not None:
            rows += [state.proposal[None], state.es.mean[None]]
        return jnp.concatenate(rows, axis=0), eps

    def tell(self, state, aux, fitness, descriptors=None):
        n = self.num_extra
        eps, f = aux, fitness[:-n]
        f_mean, f_elite = fitness[-n], fitness[-n + 1]
        old = state.es.mean
        stepped = self.es.tell(state.es._replace(log_sigma=state.log_sigma), eps, f)
        delta = state.scale * (stepped.mean - old)
        proposal = old + delta
        if self.accept_tol is not None:
            f_prop = fitness[-n + 2]
            accept = state.has_proposal & (
                f_prop >= f_mean - self.accept_tol * jnp.std(f))
            centre = jnp.where(accept, state.proposal, old)

        had_elite = jnp.isfinite(state.elite_fitness)
        elite_now = jnp.where(had_elite, f_elite, -jnp.inf)
        best = jnp.argmax(f)
        take = f[best] > elite_now
        elite = jnp.where(take, old + jnp.exp(state.log_sigma) * eps[best], state.elite)
        elite_fitness = jnp.where(take, f[best], elite_now)

        mean = proposal if self.accept_tol is None else centre
        opt_state, anchored = stepped.opt_state, state.anchored
        if self.jump_kappa > 0:
            jump = elite_fitness - f_mean > self.jump_kappa * jnp.std(f) + 1e-8
            mean = jnp.where(jump, elite, mean)
            fresh = self.es.optimizer.init(mean)
            opt_state = jax.tree.map(lambda a, b: jnp.where(jump, a, b),
                                     fresh, stepped.opt_state)
            anchored = anchored | jump

        scale, hold = state.scale, state.hold
        collapsed = jnp.asarray(False)
        if self.restart_drop > 0:
            collapsed = had_elite & (
                f_elite < state.elite_fitness
                - self.restart_drop * (jnp.abs(state.elite_fitness) + 1e-8))
            hold = jnp.where(collapsed, self.restart_hold, jnp.maximum(hold - 1, 0))
            anchored = anchored & ~collapsed
        if self.settle_rate > 0:
            p = jnp.mean(f_mean >= f)
            settled = jnp.clip(scale * jnp.exp(-self.settle_rate
                                               * (p - self.settle_target)),
                               self.scale_min, 1.0)
            relaxed = jnp.minimum(1.0, scale * jnp.exp(self.settle_rate))
            scale = jnp.where(anchored, settled, relaxed)
        scale = jnp.where(hold > 0, 1.0, scale)

        es = stepped._replace(mean=mean, log_sigma=state.es.log_sigma,
                              opt_state=opt_state)
        next_proposal, has_proposal = state.proposal, state.has_proposal
        if self.accept_tol is not None:
            next_proposal, has_proposal = mean + delta, jnp.asarray(True)
        return AdaptiveESState(es=es, elite=elite, elite_fitness=elite_fitness,
                               scale=scale, hold=hold, anchored=anchored,
                               proposal=next_proposal, has_proposal=has_proposal)


# ---------------------------------------------------------------------------
# NES with mu centroids
# ---------------------------------------------------------------------------

class MultiESSearcher:
    """Adapter putting `MultiES` (`source/algorithms/ne/es.py`) behind it.

    ``incumbent`` is one centroid, not the set: a portfolio that covers both
    sub-tasks with two different specialists has not found a generalist, and
    scoring the set would let it claim it had. ``population`` returns all the
    centroids so the coverage claim can be made separately and labelled as
    such.
    """

    needs_descriptors = False

    def __init__(self, num_params, population_size, **kwargs):
        allowed = ('sigma_init', 'learning_rate', 'optimizer', 'shaping',
                   'sigma_lr', 'momentum', 'num_centroids', 'coupling',
                   'restart_interval', 'restart_frac', 'restart_scale',
                   'init_scale')
        self.es = MultiES(num_params=num_params,
                          population_size=population_size,
                          **{k: v for k, v in kwargs.items() if k in allowed})
        self.population_size = self.es.population_size

    def init(self, key, mean):
        return self.es.init(key, mean)

    def ask(self, key, state):
        return self.es.ask(key, state)

    def tell(self, state, aux, fitness, descriptors=None):
        return self.es.tell(state, aux, fitness)

    def incumbent(self, state):
        return self.es.incumbent(state)

    def population_mean(self, state):
        """The mean over the M centroids.

        Not any one centroid, and not a policy the search ever evaluated when
        M > 1 -- the same caveat GA's carries.
        """
        return jnp.mean(self.es.centroids(state), axis=0)

    has_population = True

    def population(self, state):
        return self.es.centroids(state)


# ---------------------------------------------------------------------------
# Genetic algorithm
# ---------------------------------------------------------------------------

class GAState(NamedTuple):
    archive: jnp.ndarray      # (num_elites, num_params), best first
    fitness: jnp.ndarray      # (num_elites,) MINIMISED internally
    sigma: jnp.ndarray        # scalar mutation scale
    generation: jnp.ndarray


class GASearcher:
    """Truncation-selection GA with gaussian mutation (Such et al., 2017).

    Ported from `source/algorithms/ne/ga.py` (SimpleGA). Two details
    from that file are load-bearing and are kept:

      - Selection is over the union of the new generation AND the surviving
        archive, so an elite is never lost to a bad draw.
      - With ``cross_over_rate = 0`` (the default there and here) an offspring
        is exactly one archive member plus gaussian noise. The crossover path
        exists but is off; turning it on makes "the parent" ill-defined.

    ``variation='isoline'`` replaces the gaussian operator with DNS's
    Iso+LineDD and changes nothing else -- selection is still fitness
    truncation. That arm is ``method='ga_isoline'``; see the module docstring.

    Fitness is maximised at this interface and negated internally, because the
    reference sorts ascending.
    """

    needs_descriptors = False

    def __init__(self, num_params, population_size, elite_ratio=0.5,
                 sigma_init=0.1, sigma_decay=1.0, sigma_limit=1e-4,
                 cross_over_rate=0.0, init_scale=0.1, refresh=True,
                 init_around_mean=True, variation=GAUSSIAN,
                 iso_sigma=None, line_sigma=None, sigma_rule='decay',
                 track_rate=0.1, track_target=None, sigma_min=None,
                 ties='children'):
        self.num_params = int(num_params)
        # Who wins a tied score under `refresh`. `children` (the default): the
        # offspring come first in the evaluated batch and the sort is stable,
        # so a child that merely equals its parent replaces it -- on a plateau
        # that is a random walk of the whole archive. `parents`
        # (`method='ga_keep'`): a child must strictly beat a member to take its
        # place, so a plateau freezes the archive. Why (smooth toy, 2026-09-14):
        # the GA found the generalist in every seed and lost it by exactly that
        # walk, more with larger sigma and longer sub-task intervals, while
        # NES, whose mean does not move on a plateau, kept it.
        if ties not in ('children', 'parents'):
            raise ValueError(f"ties must be 'children' or 'parents', not {ties!r}")
        self.ties = ties
        # Where the archive starts. True: jittered copies of the seed policy
        # (the flax initialisation NES starts from), the study's convention so
        # every arm begins at one point. False: the paper's `SimpleGA` init,
        # `N(0, init_scale)` on every weight with no seed policy at all. On
        # the ant the two are not interchangeable: the seed policy's actions
        # are twice as large, every jittered copy falls within ~30 steps, and
        # a GA at sigma 0.01 never finds the upright posture from there --
        # while 4% of an N(0, 0.1) population stands for the whole episode
        # (~1460, near the 1678 an all-zero policy scores under the
        # speed-tracking reward). Measured 2026-09-06; see CLAUDE.md,
        # section G.
        self.init_around_mean = bool(init_around_mean)
        self.population_size = int(population_size)
        self.num_elites = max(1, int(population_size * elite_ratio))
        self.sigma_init = float(sigma_init)
        self.sigma_decay = float(sigma_decay)
        self.sigma_limit = float(sigma_limit)
        self.cross_over_rate = float(cross_over_rate)
        self.init_scale = float(init_scale)
        # `gaussian` is this GA as published and the default. `isoline` swaps
        # in DNS's operator and changes NOTHING else -- selection is still
        # fitness truncation -- which is the arm that isolates DNS's selection
        # rule from its recombination. Reach for `method='ga_isoline'` rather
        # than setting this by hand; see `build_searcher`.
        if variation not in VARIATIONS:
            raise ValueError(f'variation must be one of {VARIATIONS}, '
                             f'not {variation!r}')
        self.variation = variation
        if variation == ISOLINE and float(sigma_decay) != 1.0:
            # Nothing to decay: the isoline widths are constants, as they are
            # in DNS. Ignoring the request would leave the run's config
            # claiming a schedule the search never ran.
            raise ValueError(
                'sigma_decay is the gaussian width\'s schedule; the isoline '
                'operator has no decaying width. Pass sigma_decay=1.0.')
        self.variation_params = (
            resolve_params(GAUSSIAN, sigma=sigma_init,
                           cross_over_rate=cross_over_rate)
            if variation == GAUSSIAN else
            resolve_params(ISOLINE, iso_sigma=iso_sigma,
                           line_sigma=line_sigma))
        # `refresh` re-evaluates the archive every generation instead of
        # trusting the fitness it was stored with. DEFAULT ON since 2026-08-27,
        # because off is wrong for every schedule this study runs.
        #
        # The reference keeps the stored value -- correct for a stationary task
        # and the reason its GA is elitist at all -- but under a SWITCHING
        # schedule that value was measured on a different sub-task, and
        # comparing it with an offspring's score on the current one is
        # comparing two different numbers. The consequence is not subtle: an
        # archive full of sub-task A specialists carrying their sub-task A
        # scores cannot be displaced by anything evaluated on sub-task B, so
        # the GA simply stops at the switch, and its flat curve is bookkeeping
        # rather than a statement about search.
        #
        # Refreshing costs `num_elites` evaluations, taken OUT of the offspring
        # budget rather than added to it, so both GA arms spend the same number
        # of evaluations per generation as every other method. What it pays
        # instead is half the offspring, and weaker elitism: an elite has to
        # re-win against a fresh noisy estimate every generation, so a good
        # genome can be evicted by one unlucky draw.
        #
        # `method='ga_stale'` restores the old behaviour. It is what the
        # `*_gastale_*` run tags were produced with, and what the toy-landscape
        # panel contrasts against.
        self.refresh = bool(refresh)
        self.num_offspring = (self.population_size - self.num_elites
                              if self.refresh else self.population_size)
        if self.refresh and self.num_offspring < 1:
            raise ValueError('refresh needs elite_ratio < 1')
        # How the gaussian width moves. `decay` is the fixed schedule above
        # (constant at the default sigma_decay = 1.0) and the default, so
        # every existing arm is unchanged. `track` is `method='ga_track'`: each
        # generation the archive's centroid is scored beside the archive, as
        # a signal only -- it never enters selection -- and sigma moves by
        #     sigma <- sigma * exp(track_rate * (p - track_target))
        # where p is the share of the new archive the centroid scores at least
        # as well as. A centroid below the median elite means the archive is
        # spread over solutions its mean is not one of -- on Kinetix, unrelated
        # solvers whose children do not inherit -- so sigma shrinks until
        # children stay on their parent's solution; a centroid as good as the
        # archive lets it grow back, never past sigma_init. A rank rather than
        # a score difference, so one rate works on every return scale. The
        # centroid's evaluation comes out of the offspring budget, so the arm
        # spends the same evaluations as `ga`; nothing in it knows where a
        # sub-task boundary is (CLAUDE.md d).
        #
        # `success` is `method='ga_success'`: the same update with p the share
        # of this generation's OFFSPRING that score at least as well as the
        # median of the new archive (target 0.2, the 1/5 rule's). It asks the
        # centroid rule's underlying question directly -- do children keep
        # their parents' quality? -- and costs no evaluation. It exists
        # because the centroid rule has a confound, seen on the rugged toy
        # (2026-09-13): a population climbing a convex slope has its centroid
        # behind most elites even though its children inherit fine, so
        # `track` shrank sigma mid-climb and stalled.
        if sigma_rule not in ('decay', 'track', 'success'):
            raise ValueError(f"sigma_rule must be 'decay', 'track' or "
                             f"'success', not {sigma_rule!r}")
        self.sigma_rule = sigma_rule
        self.track_rate = float(track_rate)
        self.track_target = (
            float(track_target) if track_target is not None
            else 0.2 if sigma_rule == 'success' else 0.5)
        self.sigma_min = (self.sigma_init / 100.0 if sigma_min is None
                          else float(sigma_min))
        if sigma_rule in ('track', 'success'):
            if variation != GAUSSIAN or not self.refresh \
                    or float(sigma_decay) != 1.0:
                raise ValueError(
                    f"sigma_rule={sigma_rule!r} adapts the gaussian width "
                    'against a freshly scored archive: it needs '
                    'variation=gaussian, refresh=True and sigma_decay=1.0')
        if sigma_rule == 'track':
            self.num_offspring -= 1
            if self.num_offspring < 1:
                raise ValueError('track needs room for one centroid evaluation')

    def init(self, key, mean):
        # The archive starts as jittered copies of the seed policy, so
        # generation 0 is a local search around the same initialisation NES
        # starts from rather than from scratch.
        jitter = random.normal(key, (self.num_elites, self.num_params))
        archive = ((mean[None, :] if self.init_around_mean else 0.0)
                   + self.init_scale * jitter)
        return GAState(
            archive=archive,
            fitness=jnp.full((self.num_elites,), jnp.inf),
            sigma=jnp.asarray(self.sigma_init, dtype=jnp.float32),
            generation=jnp.asarray(0, dtype=jnp.int32))

    def ask(self, key, state):
        params = dict(self.variation_params)
        if self.variation == GAUSSIAN:
            # The one width that moves during a run: `state.sigma` carries the
            # decay schedule, so it overrides the configured value.
            params['sigma'] = state.sigma
        x = vary(self.variation, state.archive, key, self.num_offspring,
                 params)
        # Under `refresh` the archive rides along in the evaluated batch, so
        # every genome `tell` ranks was scored on the SAME sub-task this
        # generation and no stored fitness is carried across a switch.
        if self.refresh:
            x = jnp.concatenate([x, state.archive], axis=0)
        if self.sigma_rule == 'track':
            # Last row: the centroid, scored for the sigma rule only.
            x = jnp.concatenate([x, jnp.mean(state.archive, axis=0,
                                             keepdims=True)], axis=0)
        return x, None

    def tell(self, state, aux, fitness, descriptors=None):
        evaluated = aux_population(aux, state)
        if self.sigma_rule == 'track':
            centroid_fitness, fitness = fitness[-1], fitness[:-1]
            evaluated = evaluated[:-1]
        loss = -fitness                      # the reference minimises
        if self.refresh:
            combined, combined_loss = evaluated, loss
        else:
            combined_loss = jnp.concatenate([loss, state.fitness])
            combined = jnp.concatenate([evaluated, state.archive])
        if self.ties == 'parents' and self.refresh:
            # The archive sits after the offspring in the batch (see `ask`).
            child = jnp.arange(combined_loss.shape[0]) < self.num_offspring
            order = jnp.lexsort((child, combined_loss))[:self.num_elites]
        else:
            order = jnp.argsort(combined_loss)[:self.num_elites]
        if self.sigma_rule in ('track', 'success'):
            kept = -combined_loss[order]
            if self.sigma_rule == 'track':
                share = jnp.mean(centroid_fitness >= kept)
            else:
                # Offspring come first in the evaluated batch (see `ask`).
                share = jnp.mean(fitness[:self.num_offspring]
                                 >= jnp.median(kept))
            sigma = jnp.clip(
                state.sigma * jnp.exp(self.track_rate
                                      * (share - self.track_target)),
                self.sigma_min, self.sigma_init)
        else:
            sigma = jnp.maximum(state.sigma * self.sigma_decay,
                                self.sigma_limit)
        return state._replace(archive=combined[order],
                              fitness=combined_loss[order],
                              sigma=sigma,
                              generation=state.generation + 1)

    def incumbent(self, state):
        return state.archive[0]

    def population_mean(self, state):
        """The coordinate-wise mean of the elite archive.

        NOT a policy this search ever evaluated, and not one it would return:
        `incumbent` is the best elite and that is what the run's result is
        read from. This exists so a figure can ask whether the archive has
        collapsed onto one solution -- if it has, scoring the mean and scoring
        the best agree; where they disagree, the archive is still spread and
        the mean is averaging genuinely different networks.

        That averaging is not innocent. Two networks that compute the same
        function can differ by a permutation of their hidden units, and the
        mean of such a pair is neither of them and is generally much worse than
        both. So a low value here is evidence of a spread archive, not of a bad
        search, and it must not be read as this method's performance.
        """
        return jnp.mean(state.archive, axis=0)

    # The elite archive IS a persistent population: `tell` selects over the
    # union of offspring and survivors, so a genome that is best on sub-task 0
    # can sit in it indefinitely alongside one that is best on sub-task 1.
    has_population = True

    def population(self, state):
        return state.archive


def aux_population(aux, state):
    """GA's ``tell`` needs the population it scored; ``ask`` returned it as x.

    The runner threads ``aux`` from ask to tell, so GA stores the population
    there. Kept as a named function so the indirection is visible rather than
    looking like a bug at the call site.
    """
    return aux


class SubspaceGAState(NamedTuple):
    archive: jnp.ndarray      # (num_elites, num_params), best first
    fitness: jnp.ndarray      # (num_elites,) MINIMISED internally
    sigma: jnp.ndarray        # isotropic mutation width
    sigma_sub: jnp.ndarray    # width along the archive's own spread
    generation: jnp.ndarray


class SubspaceGASearcher(GASearcher):
    """The refreshed gaussian GA with a second mutation channel and two widths.

    ``method='ga_subspace'`` (2026-09-13). Offspring come in two groups:

      iso   parent + sigma * N(0, I): the GA's own mutation.
      sub   parent + sigma_sub * D^T z / sqrt(A - 1), with D the archive's
            deviations from its centroid and z ~ N(0, I_A): a draw from the
            archive's empirical covariance, i.e. a step along the directions
            the population currently spans.

    Each width moves by the share of its group's children that SURVIVE into
    the next archive,
        width <- width * exp(adapt_rate * (survived - survive_target)),
    so it is a rank signal (no scale), costs no evaluation, and asks the
    question the centroid analysis landed on: do children keep their parents'
    quality? Where the solution set is thin in some directions (Kinetix, the
    manifold toy) isotropic children fall off it and sigma shrinks, while
    children along the population's spread -- directions that did not change
    the score -- keep surviving and keep sigma_sub. Where the archive is split
    over two basins its main spread IS the line between them; children along
    it land in the valley and die, which is what lets a recombination-like
    step pull a split archive onto one basin. sigma never exceeds its start;
    sigma_sub is bounded above so the spread channel cannot inflate the
    archive without limit (Iso+LineDD at line_sigma 0.5 did).

    ``subspace_fraction=0`` keeps only the iso group: the survival rule
    alone, the ablation that says what the spread channel adds.

    Always refreshed and gaussian; spends ``population_size`` evaluations per
    generation like every arm; told nothing about sub-task boundaries.
    """

    def __init__(self, num_params, population_size, elite_ratio=0.5,
                 sigma_init=0.1, sigma_sub_init=0.5, subspace_fraction=0.5,
                 adapt_rate=0.1, survive_target=0.25, sigma_min=None,
                 sigma_sub_min=1e-3, sigma_sub_max=2.0, init_scale=0.1,
                 init_around_mean=True):
        super().__init__(num_params, population_size, elite_ratio=elite_ratio,
                         sigma_init=sigma_init, init_scale=init_scale,
                         refresh=True, init_around_mean=init_around_mean)
        self.sigma_rule = 'subspace'
        self.num_sub = int(round(self.num_offspring * float(subspace_fraction)))
        self.num_iso = self.num_offspring - self.num_sub
        if self.num_iso < 1:
            raise ValueError('subspace_fraction must leave some iso offspring')
        self.sigma_sub_init = float(sigma_sub_init)
        self.adapt_rate = float(adapt_rate)
        self.survive_target = float(survive_target)
        self.sigma_min = (self.sigma_init / 100.0 if sigma_min is None
                          else float(sigma_min))
        self.sigma_sub_min = float(sigma_sub_min)
        self.sigma_sub_max = float(sigma_sub_max)

    def init(self, key, mean):
        s = super().init(key, mean)
        return SubspaceGAState(
            archive=s.archive, fitness=s.fitness, sigma=s.sigma,
            sigma_sub=jnp.asarray(self.sigma_sub_init, dtype=jnp.float32),
            generation=s.generation)

    def ask(self, key, state):
        k_pi, k_iso, k_ps, k_sub = random.split(key, 4)
        archive = state.archive
        n = archive.shape[0]
        parents = archive[random.randint(k_pi, (self.num_iso,), 0, n)]
        groups = [parents + state.sigma * random.normal(
            k_iso, (self.num_iso, self.num_params))]
        if self.num_sub:
            deviations = archive - jnp.mean(archive, axis=0, keepdims=True)
            z = random.normal(k_sub, (self.num_sub, n))
            parents = archive[random.randint(k_ps, (self.num_sub,), 0, n)]
            groups.append(parents + state.sigma_sub * (z @ deviations)
                          / jnp.sqrt(max(n - 1, 1)))
        # iso offspring, sub offspring, then the archive to be re-scored.
        return jnp.concatenate(groups + [archive], axis=0), None

    def tell(self, state, aux, fitness, descriptors=None):
        loss = -fitness
        order = jnp.argsort(loss)[:self.num_elites]
        survived = jnp.zeros(loss.shape[0], dtype=bool).at[order].set(True)

        def move(width, group, low, high):
            share = jnp.mean(group)
            return jnp.clip(width * jnp.exp(self.adapt_rate
                                            * (share - self.survive_target)),
                            low, high)

        sigma = move(state.sigma, survived[:self.num_iso], self.sigma_min,
                     self.sigma_init)
        sigma_sub = (move(state.sigma_sub,
                          survived[self.num_iso:self.num_offspring],
                          self.sigma_sub_min, self.sigma_sub_max)
                     if self.num_sub else state.sigma_sub)
        return SubspaceGAState(archive=aux[order], fitness=loss[order],
                               sigma=sigma, sigma_sub=sigma_sub,
                               generation=state.generation + 1)


class MergeGAState(NamedTuple):
    archive: jnp.ndarray      # (num_elites, num_params), best TRUE score first
    fitness: jnp.ndarray      # (num_elites,) MINIMISED internally, true scores
    sigma: jnp.ndarray        # the GA's fixed gaussian width
    merge: jnp.ndarray        # tau (noise) or phi (pc), in [0, merge_max]
    key: jnp.ndarray          # for the selection noise and the recombination
    generation: jnp.ndarray
    # ga_path only (zeros otherwise): the evolution path of the archive's
    # centroid, in units of one generation's random-drift step, and the gain
    # of the push offspring get along it.
    path: jnp.ndarray = None
    path_gain: jnp.ndarray = None
    # ga_focus only (1.0 otherwise): the share of the archive, best first,
    # that offspring are bred from.
    focus: jnp.ndarray = None
    # restart only (0 otherwise): generations left in which sigma and focus
    # are held at their starting values after a detected collapse.
    hold: jnp.ndarray = None


class MergeGASearcher(GASearcher):
    """The refreshed gaussian GA that merges a split archive when its centroid
    stops tracking its elite. ``method='ga_merge'`` (2026-09-13).

    Why (the wells toy, same day): with two equally good basins a GA's archive
    splits and FREEZES -- children rarely beat their parents, the same members
    keep winning, nothing moves the shares -- and its centroid sits in the
    valley. A small sigma only prevents the split (it turns the climb into a
    race one side wins); it does not undo one. Two things did: evaluation noise
    (who survives becomes partly random, so the shares drift until one basin
    fixes) and recombination whose children land between two parents (pairs
    from opposite basins produce valley children that die, which costs the
    minority basin more than the majority).

    Each generation the centroid is scored beside the archive (one evaluation,
    taken out of the offspring budget) and p is the share of the new archive it
    scores at least as well as. The merge level moves by
        merge <- clip(merge + merge_rate * (track_target - p), 0, merge_max)
    so it rises while the centroid sits below the median elite and decays to 0
    once it does not. What the level does is ``merge``:

      noise  selection ranks z-scored offspring+archive scores plus
             merge * N(0, 1): the simple version, drift on demand.
      pc     a share `merge` of offspring are p1 + u * P (p2 - p1) + sigma * eps,
             u ~ U(0, 1), P the projection onto the archive's top `num_pcs`
             principal directions: recombination confined to the directions
             the population is spread along -- the axis between its clusters
             when it is split -- and kept out of the many directions a line
             step would otherwise inflate (Iso+LineDD at line_sigma 0.5 ran its
             weights away doing exactly that).

    sigma is fixed, so a difference from `ga` is the merge mechanism alone.
    Nothing here knows where a sub-task boundary is.
    """

    def __init__(self, num_params, population_size, elite_ratio=0.5,
                 sigma_init=0.1, merge='noise', merge_rate=0.02,
                 merge_max=None, track_target=0.5, num_pcs=2, init_scale=0.1,
                 init_around_mean=True, cross_over_rate=0.0, sigma_rate=0.0,
                 path_rate=0.0, path_fraction=0.5, path_decay=0.1,
                 path_gain_init=0.5, path_gain_max=3.0, focus_rate=0.0,
                 focus_min=None, focus_survival=False, sigma_min=None,
                 explore_fraction=0.0, restart_share=0.0, restart_drop=0.1,
                 restart_hold=50):
        # `cross_over_rate` passes through to the gaussian operator, so this
        # arm can match a body's GA exactly (Kinetix's runs at 0.2).
        super().__init__(num_params, population_size, elite_ratio=elite_ratio,
                         sigma_init=sigma_init, init_scale=init_scale,
                         refresh=True, init_around_mean=init_around_mean,
                         cross_over_rate=cross_over_rate)
        if merge not in ('noise', 'pc'):
            raise ValueError(f"merge must be 'noise' or 'pc', not {merge!r}")
        self.sigma_rule = 'merge'
        self.merge = merge
        self.merge_rate = float(merge_rate)
        self.merge_max = float(merge_max if merge_max is not None
                               else (1.0 if merge == 'noise' else 0.5))
        self.track_target = float(track_target)
        self.num_pcs = int(num_pcs)
        # `sigma_rate > 0` (`method='ga_merge_track'`) also moves the gaussian
        # width from the SAME centroid signal, as ga_track does: sigma shrinks
        # while the centroid lags and grows back (never past sigma_init). On
        # the min-wells toy the two rules win in different regimes -- rank
        # noise with few split coordinates, the sigma shrink with many
        # interacting ones -- so this arm carries both.
        self.sigma_rate = float(sigma_rate)
        # The floor. sigma_init / 100 is too coarse for Kinetix: around a
        # consolidated elite (1.1M weights) children still solve at sigma 1e-4
        # (88-97%) but only 0-12% at 5e-3, so at that floor focused children
        # lose to old unrelated solvers and the archive never consolidates
        # (probe of the ga_focus_purge runs, 2026-09-14).
        self.sigma_min = (self.sigma_init / 100.0 if sigma_min is None
                          else float(sigma_min))
        # `path_rate > 0` (`method='ga_path'`): moving along the manifold. The
        # archive's centroid displacement each generation, divided by the
        # displacement random drift alone would give (sigma / sqrt(archive)),
        # is folded into an evolution path as CMA-ES folds its mean steps:
        #     path <- (1 - c) path + sqrt(c (2 - c)) * displacement / drift
        # Under pure drift -- the directions that change nothing -- successive
        # steps are uncorrelated and |path| stays near sqrt(d); along a
        # consistent slope they add up. The first `path_fraction` of the
        # offspring get parent + sigma*eps + u*gain*drift*sqrt(d)*path_dir
        # (u ~ U(0, 1), path_dir the path's UNIT direction), so a push is at
        # most gain * sigma * sqrt(d / archive) long whatever the path's length.
        # The gain moves by exp(path_rate * (s_push - s_plain)), the survival
        # share of pushed offspring minus that of unpushed ones: it grows only
        # while pushing along the path breeds survivors more often than not
        # pushing. The first version grew the gain with the path's length
        # instead, and since the push itself moves the centroid that was a
        # feedback loop: on the smooth toy the weights ran to 1e9
        # (2026-09-13).
        self.path_rate = float(path_rate)
        self.path_fraction = float(path_fraction)
        self.path_decay = float(path_decay)
        self.path_gain_init = float(path_gain_init)
        self.path_gain_max = float(path_gain_max)
        self.num_offspring -= 1          # one evaluation for the centroid
        self.num_push = (int(round(self.path_fraction * self.num_offspring))
                         if self.path_rate > 0 else 0)
        # `focus_rate > 0` (`method='ga_focus'`): consolidation through the
        # choice of PARENTS. While the centroid lags, offspring are bred only
        # from the best `focus` share of the archive, and that share shrinks
        #     focus <- clip(focus * exp(-focus_rate * (target - p)), min, 1)
        # toward a single parent; once the centroid tracks it grows back to the
        # whole archive. Why (Kinetix pilot, 2026-09-14): the archive there
        # holds many unrelated solver lineages, and merging them by drift
        # takes about as many generations as the archive has members -- more
        # than the 200-generation budget -- while shrinking sigma only freezes
        # them. Breeding from a few parents replaces the unrelated lineages
        # with the children of those few within a handful of generations
        # (children that tie the stored score come first in the sort, so they
        # displace equal-scoring old members), and with sigma small those
        # children inherit the solution, so the centroid lands near them.
        self.focus_rate = float(focus_rate)
        self.focus_min = (1.0 / self.num_elites if focus_min is None
                          else float(focus_min))
        # `focus_survival` (`method='ga_focus_purge'`): the focus also decides
        # who may SURVIVE. Old archive members outside the focused pool cannot
        # be selected again, so the next archive is the pool plus the best of
        # its children. Why (Kinetix, home pilot 2026-09-14): with breeding
        # focus alone the old unrelated solvers stayed -- Kinetix returns
        # differ slightly between solutions, so a focused child scoring a
        # hair below an old solver never displaced it, and the centroid kept
        # averaging unrelated solutions even at sigma 0.005.
        self.focus_survival = bool(focus_survival)
        # `explore_fraction > 0`: that share of the offspring are ALWAYS bred
        # from the whole archive at sigma_init, whatever the focus and sigma.
        # Why (the Kinetix20 chain, 2026-09-14): the focus consolidates a level
        # at sigma ~1e-4, and after the next level begins every member fails
        # alike, so the centroid rarely beats 90% of them and the rule keeps
        # shrinking sigma -- a population that cannot search the new level.
        # Explorers search it anyway; one that solves outranks the failing
        # consolidated members, becomes the focused parent, and its small-sigma
        # children consolidate around it. While the centroid tracks, explorers
        # fail and never displace a solver. No boundary is detected.
        self.explore_fraction = float(explore_fraction)
        self.num_explore = int(round(self.explore_fraction * self.num_offspring))
        # `restart_share > 0`: search again when the archive's own scores
        # collapse. The archive is re-scored every generation, so a member
        # scores what it scored last generation unless the sub-task changed
        # (Kinetix is deterministic; a noisy body needs `restart_drop` above its
        # noise). When more than `restart_share` of the members score more than
        # `restart_drop` x the mean |stored score| below their stored score,
        # sigma and focus go back to their starting values and stay there for
        # `restart_hold` generations -- long enough for the gaussian GA to find
        # a solver before the tracking rule starts shrinking again. Nothing is
        # told where a boundary is; the collapse is measured, not given.
        self.restart_share = float(restart_share)
        self.restart_drop = float(restart_drop)
        self.restart_hold = int(restart_hold)
        if self.num_offspring < 1:
            raise ValueError('merge needs room for one centroid evaluation')

    def init(self, key, mean):
        k_init, k_state = random.split(key)
        s = super().init(k_init, mean)
        return MergeGAState(archive=s.archive, fitness=s.fitness, sigma=s.sigma,
                            merge=jnp.asarray(0.0, dtype=jnp.float32),
                            key=k_state, generation=s.generation,
                            path=jnp.zeros(self.num_params, dtype=jnp.float32),
                            path_gain=jnp.asarray(self.path_gain_init,
                                                  dtype=jnp.float32),
                            focus=jnp.asarray(1.0, dtype=jnp.float32),
                            hold=jnp.asarray(0, dtype=jnp.int32))

    def ask(self, key, state):
        if self.path_rate > 0:
            # Split only when the path is on, so every other merge arm draws
            # exactly the random numbers it drew before this option existed.
            key, k_path = random.split(key)
        if self.num_explore:
            key, k_explore = random.split(key)
        k_mut, k_pc = random.split(key)
        params = dict(self.variation_params, sigma=state.sigma)
        if self.focus_rate > 0:
            # The gaussian operator with both parents drawn from the best
            # ceil(focus * archive) members; the archive is stored best
            # true score first. Same crossover and noise as `vary`.
            k_a, k_b, k_mate, k_eps = random.split(k_mut, 4)
            n = state.archive.shape[0]
            pool = jnp.maximum(1.0, jnp.ceil(state.focus * n))
            ia = jnp.minimum((random.uniform(k_a, (self.num_offspring,)) * pool)
                             .astype(jnp.int32), n - 1)
            ib = jnp.minimum((random.uniform(k_b, (self.num_offspring,)) * pool)
                             .astype(jnp.int32), n - 1)
            take_b = (random.uniform(k_mate, (self.num_offspring,
                                              self.num_params))
                      > params['cross_over_rate'])
            x = (state.archive[ia] * (1 - take_b) + state.archive[ib] * take_b
                 + state.sigma * random.normal(
                     k_eps, (self.num_offspring, self.num_params)))
        else:
            x = vary(GAUSSIAN, state.archive, k_mut, self.num_offspring, params)
        if self.num_explore:
            explorers = vary(GAUSSIAN, state.archive, k_explore, self.num_explore,
                             dict(self.variation_params, sigma=self.sigma_init))
            x = x.at[:self.num_explore].set(explorers)
        if self.merge == 'pc':
            archive = state.archive
            n = archive.shape[0]
            k_a, k_b, k_u, k_eps, k_use = random.split(k_pc, 5)
            centred = archive - jnp.mean(archive, axis=0, keepdims=True)
            basis = jnp.linalg.svd(centred, full_matrices=False)[2][:self.num_pcs]
            p1 = archive[random.randint(k_a, (self.num_offspring,), 0, n)]
            p2 = archive[random.randint(k_b, (self.num_offspring,), 0, n)]
            u = random.uniform(k_u, (self.num_offspring, 1))
            step = ((p2 - p1) @ basis.T) @ basis
            line = p1 + u * step + state.sigma * random.normal(
                k_eps, (self.num_offspring, self.num_params))
            use = random.uniform(k_use, (self.num_offspring, 1)) < state.merge
            x = jnp.where(use, line, x)
        if self.num_push:
            drift = state.sigma / jnp.sqrt(float(self.num_elites))
            direction = state.path / (jnp.linalg.norm(state.path) + 1e-12)
            u = random.uniform(k_path, (self.num_push, 1))
            push = (u * state.path_gain * drift
                    * jnp.sqrt(float(self.num_params)) * direction[None, :])
            x = x.at[:self.num_push].add(push)
        centroid = jnp.mean(state.archive, axis=0, keepdims=True)
        return jnp.concatenate([x, state.archive, centroid], axis=0), None

    def tell(self, state, aux, fitness, descriptors=None):
        centroid_fitness, fitness = fitness[-1], fitness[:-1]
        evaluated = aux[:-1]
        key, k_noise = random.split(state.key)
        rank_score = (fitness - jnp.mean(fitness)) / (jnp.std(fitness) + 1e-12)
        if self.merge == 'noise':
            rank_score = rank_score + state.merge * random.normal(
                k_noise, fitness.shape)
        if self.focus_rate > 0 and self.focus_survival:
            # The old archive sits after the offspring in the batch, best
            # first; members ranked outside the focused pool are barred.
            n_old = state.archive.shape[0]
            pool = jnp.maximum(1.0, jnp.ceil(state.focus * n_old))
            start = self.num_offspring
            barred = jnp.arange(n_old) >= pool
            old = rank_score[start:start + n_old]
            rank_score = rank_score.at[start:start + n_old].set(
                jnp.where(barred, -jnp.inf, old))
        chosen = jnp.argsort(-rank_score)[:self.num_elites]
        # Stored best-true-score first, so `incumbent` stays the real elite.
        chosen = chosen[jnp.argsort(-fitness[chosen])]
        kept = fitness[chosen]
        share = jnp.mean(centroid_fitness >= kept)
        merge = jnp.clip(state.merge + self.merge_rate
                         * (self.track_target - share), 0.0, self.merge_max)
        sigma = state.sigma
        if self.sigma_rate > 0:
            sigma = jnp.clip(sigma * jnp.exp(self.sigma_rate
                                             * (share - self.track_target)),
                             self.sigma_min, self.sigma_init)
        archive = evaluated[chosen]
        path, path_gain = state.path, state.path_gain
        if self.path_rate > 0:
            drift = state.sigma / jnp.sqrt(float(self.num_elites))
            step = (jnp.mean(archive, axis=0)
                    - jnp.mean(state.archive, axis=0)) / drift
            c = self.path_decay
            path = (1.0 - c) * path + jnp.sqrt(c * (2.0 - c)) * step
            survived = jnp.zeros(evaluated.shape[0], dtype=bool).at[chosen].set(True)
            s_push = jnp.mean(survived[:self.num_push])
            s_plain = jnp.mean(survived[self.num_push:self.num_offspring])
            path_gain = jnp.clip(path_gain * jnp.exp(self.path_rate
                                                     * (s_push - s_plain)),
                                 0.01, self.path_gain_max)
        focus = state.focus
        if self.focus_rate > 0:
            focus = jnp.clip(focus * jnp.exp(-self.focus_rate
                                             * (self.track_target - share)),
                             self.focus_min, 1.0)
        hold = state.hold
        if self.restart_share > 0:
            # The old archive sits after the offspring in the batch, in the
            # order its stored scores are kept (-state.fitness; -inf at init).
            n_old = state.archive.shape[0]
            stored = -state.fitness
            now = fitness[self.num_offspring:self.num_offspring + n_old]
            scale = jnp.mean(jnp.abs(jnp.where(jnp.isfinite(stored), stored, 0.0)))
            collapsed = jnp.mean(now < stored - self.restart_drop * (scale + 1e-8))
            hold = jnp.where(collapsed > self.restart_share, self.restart_hold,
                             jnp.maximum(hold - 1, 0))
            sigma = jnp.where(hold > 0, self.sigma_init, sigma)
            focus = jnp.where(hold > 0, 1.0, focus)
        return MergeGAState(archive=archive, fitness=-kept,
                            sigma=sigma, merge=merge, key=key,
                            generation=state.generation + 1,
                            path=path, path_gain=path_gain, focus=focus,
                            hold=hold)


# ---------------------------------------------------------------------------
# Dominated Novelty Search
# ---------------------------------------------------------------------------

class DNSState(NamedTuple):
    repertoire: jnp.ndarray    # (population_size, num_params)
    fitness: jnp.ndarray       # (population_size,) MAXIMISED
    descriptors: jnp.ndarray   # (population_size, descriptor_dim)
    generation: jnp.ndarray
    # (population_size, traj_steps, obs_dim) under AURORA, else a (P, 0, 0)
    # placeholder. The trajectories travel with the survivors because the
    # AURORA encoder is retrained during the run and EVERY stored descriptor
    # has to be recomputed with the new encoder -- a descriptor encoded by a
    # superseded encoder is not comparable with a fresh one, and dominated
    # novelty is a distance between exactly those. The reference does the same
    # (`combined_observations` in its `dns_selection`); dropping it is what
    # made this port hand-crafted-descriptor only.
    observations: jnp.ndarray = jnp.zeros((0, 0, 0))


class DNSSearcher:
    """Dominated Novelty Search, ported from `source/ne/dns.py`.

    Offspring come from the Iso+Line-DD operator (``variation='gaussian'``
    swaps in the GA's single-parent mutation instead and changes nothing else,
    which is ``method='dns_gaussian'`` -- see the module docstring); survivors
    are chosen by *dominated novelty* -- for each individual, the mean descriptor-space
    distance to its k nearest neighbours that are at least as fit. The fittest
    individuals have no fitter neighbours, get NaN, and NaN sorts first in a
    descending argsort, so elites are always kept. Novelty only ever breaks ties
    among the non-elite.

    The iso/line sigmas are the reference's corrected values (0.005 / 0.05).
    The reference notes the earlier 0.05 / 0.5 inflated population variance by
    1.5x per generation and diverged in an unbounded descriptor space.

    ``refresh`` re-evaluates the whole repertoire every generation, exactly as
    GASearcher's does and for the same reason -- see that docstring and
    docs/generalists/dns.md. DNS carried the problem in a worse form than the
    GA: a member kept its stored fitness AND its stored descriptor (and, under
    AURORA, the trajectory both are derived from) from whichever generation it
    entered on, so after a switch the `fitness[j] >= fitness[i]` mask that
    defines "who dominates whom" compared scores from two different sub-tasks,
    the distances mixed descriptors of two different sub-tasks, and
    ``incumbent`` -- the argmax over that mixed column, and the genome every
    reported centroid curve is measured on -- was chosen by numbers that were
    not comparable.

    The split differs from the GA's only because DNS keeps its whole
    population rather than an elite fraction of it, so there is no room to
    re-evaluate it and still breed ``population_size`` offspring. Under
    ``refresh`` ``population_size`` is the EVALUATION BUDGET per generation and
    ``repertoire_ratio`` splits it: at the study's 512 / 0.5 the repertoire is
    256 and 256 offspring are bred from it, so the arm still spends 512
    evaluations a generation and stays matched to GA, OpenES, NES and PPO. The
    cost is half the diversity reservoir, which is a real loss for the one
    method whose point is spread -- and the reason it is a ratio rather than
    hard-coded.
    """

    needs_descriptors = True

    def __init__(self, num_params, population_size, descriptor_dim,
                 iso_sigma=0.005, line_sigma=0.05, k=3, init_scale=0.1,
                 normalize_descriptors=False, batch_size=None,
                 traj_steps=0, obs_dim=0, refresh=True, repertoire_ratio=0.5,
                 init_around_mean=True, variation=ISOLINE, sigma_init=0.1,
                 cross_over_rate=0.0):
        self.num_params = int(num_params)
        self.population_size = int(population_size)
        self.descriptor_dim = int(descriptor_dim)
        self.iso_sigma = float(iso_sigma)
        self.line_sigma = float(line_sigma)
        # `isoline` is DNS as published and the default. `gaussian` swaps in
        # the GA's operator and changes NOTHING else -- selection is still
        # dominated novelty -- which is the other half of the 2x2 that
        # separates novelty selection from recombination. Reach for
        # `method='dns_gaussian'` rather than setting this by hand.
        #
        # There is no decay schedule on either operator here: DNS's widths are
        # constants in the reference, and a `sigma_init` that decayed under
        # `gaussian` but not under `isoline` would put a third difference back
        # into the arm this option exists to remove.
        if variation not in VARIATIONS:
            raise ValueError(f'variation must be one of {VARIATIONS}, '
                             f'not {variation!r}')
        self.variation = variation
        self.variation_params = (
            resolve_params(ISOLINE, iso_sigma=iso_sigma,
                           line_sigma=line_sigma)
            if variation == ISOLINE else
            resolve_params(GAUSSIAN, sigma=sigma_init,
                           cross_over_rate=cross_over_rate))
        self.k = int(k)
        self.init_scale = float(init_scale)
        self.normalize_descriptors = bool(normalize_descriptors)
        # DEFAULT ON since 2026-08-27, matching `ga`. `method='dns_stale'`
        # restores the reference's carry-the-stored-value behaviour; it is what
        # the `*_dnsstale_*` run tags were produced with.
        # Where the repertoire starts: jittered copies of the seed policy (the
        # study's default), or the paper's `N(0, init_scale)` with no seed --
        # the same choice, for the same reason, as GASearcher.init_around_mean.
        self.init_around_mean = bool(init_around_mean)
        self.refresh = bool(refresh)
        if self.refresh:
            # Two knobs cannot both own the same split. `batch_size` is the
            # stale path's knob (how few offspring the reference breeds from a
            # full repertoire); under refresh the offspring count falls out of
            # the ratio, so passing both is a contradiction, not a preference.
            if batch_size is not None:
                raise ValueError(
                    'batch_size is the dns_stale knob; under refresh the '
                    'offspring/repertoire split is set by repertoire_ratio')
            self.repertoire_size = max(
                1, int(self.population_size * float(repertoire_ratio)))
            self.num_offspring = self.population_size - self.repertoire_size
            if self.num_offspring < 1:
                raise ValueError('refresh needs repertoire_ratio < 1')
        else:
            self.repertoire_size = self.population_size
            # Offspring per generation. The reference breeds `batch_size` = 256
            # from a 512 repertoire, NOT one offspring per member: parents keep
            # their stored fitness and only the offspring are evaluated, so DNS
            # spends HALF the environment steps per generation that GA and
            # OpenES do at the same population size. Defaulting to
            # population_size keeps this study's arms matched on environment
            # steps; pass batch_size=256 to reproduce the reference's asymmetry
            # instead. Either way the choice is recorded, which it was not
            # before.
            self.num_offspring = int(batch_size or population_size)
        # The name the run configs already carry, kept pointing at the same
        # quantity it always meant: how many offspring an `ask` produces.
        self.batch_size = self.num_offspring
        self.traj_steps = int(traj_steps)
        self.obs_dim = int(obs_dim)

    def init(self, key, mean):
        r = self.repertoire_size
        jitter = random.normal(key, (r, self.num_params))
        return DNSState(
            repertoire=((mean[None, :] if self.init_around_mean else 0.0)
                        + self.init_scale * jitter),
            fitness=jnp.full((r,), -jnp.inf),
            descriptors=jnp.zeros((r, self.descriptor_dim)),
            generation=jnp.asarray(0, dtype=jnp.int32),
            observations=jnp.zeros((r, self.traj_steps, self.obs_dim)))

    def ask(self, key, state):
        # `num_offspring` offspring, but parents are drawn from the WHOLE
        # repertoire: the two sizes differ whenever there are fewer offspring
        # than members, and drawing in [0, num_offspring) would breed every
        # generation from only the first few members and leave the rest of the
        # repertoire to stagnate. Both operators draw that way; see
        # `_draw_parents` in source/algorithms/ne/variation.py.
        offspring = vary(self.variation, state.repertoire, key,
                         self.num_offspring, self.variation_params)
        # Under `refresh` the repertoire rides along in the evaluated batch, so
        # every genome `tell` ranks -- and every descriptor it measures a
        # distance between -- comes from THIS generation's sub-task. Offspring
        # first, then the repertoire, the same order GASearcher uses.
        if self.refresh:
            offspring = jnp.concatenate([offspring, state.repertoire], axis=0)
        return offspring, None

    def tell(self, state, aux, fitness, descriptors=None, observations=None):
        if descriptors is None:
            raise ValueError('DNS needs behaviour descriptors')
        if self.refresh:
            # `aux` already IS repertoire + offspring, all freshly scored.
            genotypes, fitnesses, descs = aux, fitness, descriptors
            obs = observations
        else:
            genotypes = jnp.concatenate([state.repertoire, aux], axis=0)
            fitnesses = jnp.concatenate([state.fitness, fitness], axis=0)
            descs = jnp.concatenate([state.descriptors, descriptors], axis=0)
            obs = (None if observations is None else
                   jnp.concatenate([state.observations, observations], axis=0))

        novelty = dominated_novelty(fitnesses, descs, self.k,
                                    self.normalize_descriptors)
        valid = fitnesses != -jnp.inf
        meta = jnp.where(valid, novelty, -jnp.inf)
        # NaN (the fittest, having no fitter neighbour) sorts first descending.
        keep = jnp.argsort(meta)[::-1][:self.repertoire_size]
        new = state._replace(repertoire=genotypes[keep],
                             fitness=fitnesses[keep],
                             descriptors=descs[keep],
                             generation=state.generation + 1)
        if obs is not None:
            new = new._replace(observations=obs[keep])
        return new

    def reencode(self, state, descriptors):
        """Replace every stored descriptor, after the encoder was retrained.

        Separate from ``tell`` because it happens on AURORA's schedule rather
        than every generation, and because the encoder lives in the runner: the
        searcher owns which trajectories survived, the runner owns what they
        currently encode to.
        """
        return state._replace(descriptors=descriptors)

    def incumbent(self, state):
        return state.repertoire[jnp.argmax(state.fitness)]

    def population_mean(self, state):
        """The coordinate-wise mean of the repertoire.

        The same caveat as GA's, and stronger: DNS keeps its repertoire
        BEHAVIOURALLY DIVERSE on purpose, so its members are the least likely
        in this study to be permutations of one solution. Expect the mean to
        score far below `incumbent` here, and expect that to be a fact about
        novelty search rather than about continual learning.
        """
        return jnp.mean(state.repertoire, axis=0)

    # The repertoire is a persistent population and, unlike GA's, one that is
    # selected for spread: dominated novelty keeps an individual alive when
    # nothing similar to it is fitter. If any method holds a set of specialists
    # rather than a generalist, this is the one designed to.
    has_population = True

    def population(self, state):
        return state.repertoire


# ---------------------------------------------------------------------------

# The method names each searcher answers to. `ga_isoline` and `dns_gaussian`
# are the operator ablation: the four names below cross {fitness truncation,
# dominated novelty} with {gaussian mutation, Iso+LineDD}, which is the 2x2
# that separates "novelty selection helps" from "recombination helps".
#
#                         gaussian          isoline
#   fitness truncation    ga                ga_isoline
#   dominated novelty     dns_gaussian      dns
#
# They are names rather than a bare `variation=` kwarg because a run tree is
# keyed by method: two arms that differ in their search must not land in the
# same directory and be averaged together.
GA_METHODS = ('ga', 'ga_fresh', 'ga_stale', 'ga_keep', 'ga_isoline', 'ga_track',
              'ga_success')
DNS_METHODS = ('dns', 'dns_fresh', 'dns_stale', 'dns_gaussian')
# What `train_nes.py --method` offers. `nes_mu` / `nes_mu_select` are built by
# `build_searcher` but stay off the CLI, as they were before this list existed.
NE_METHODS = ('nes', 'openes', 'openes_adaptive') + GA_METHODS + DNS_METHODS + (
    'ga_subspace', 'ga_merge_noise', 'ga_merge_pc', 'ga_merge_track', 'ga_path',
    'ga_focus', 'ga_focus_purge')


def build_searcher(method, num_params, population_size, descriptor_dim=2,
                   **kwargs):
    """One place that knows which settings define which method."""
    if method in ('nes', 'openes'):
        return ESSearcher(num_params, population_size, **kwargs)
    if method == 'openes_adaptive':
        return AdaptiveESSearcher(num_params, population_size, **kwargs)
    if method in ('nes_mu', 'nes_mu_select'):
        # One class, two settings -- the coupling is the comparison, so it must
        # not become two implementations.
        kwargs = dict(kwargs)
        kwargs.setdefault('coupling',
                          'select' if method == 'nes_mu_select' else 'none')
        return MultiESSearcher(num_params, population_size, **kwargs)
    if method in ('ga_merge_noise', 'ga_merge_pc', 'ga_merge_track', 'ga_path',
                  'ga_focus', 'ga_focus_purge'):
        merge_kwargs = {k: v for k, v in kwargs.items()
                        if k in ('elite_ratio', 'sigma_init', 'merge_rate',
                                 'merge_max', 'track_target', 'num_pcs',
                                 'init_scale', 'init_around_mean',
                                 'cross_over_rate', 'sigma_rate', 'path_rate',
                                 'path_fraction', 'path_decay',
                                 'path_gain_init', 'path_gain_max',
                                 'focus_rate', 'focus_min', 'focus_survival',
                                 'sigma_min', 'explore_fraction',
                                 'restart_share', 'restart_drop',
                                 'restart_hold')}
        if method == 'ga_focus_purge':
            merge_kwargs.setdefault('focus_survival', True)
        if method in ('ga_focus', 'ga_focus_purge'):
            # Parent focus plus the sigma shrink; no rank noise unless asked
            # (merge_rate), since the focus is the consolidation mechanism.
            merge_kwargs.setdefault('focus_rate', 0.3)
            merge_kwargs.setdefault('sigma_rate', 0.1)
            merge_kwargs.setdefault('merge_rate', 0.0)
        if 'sigma' in kwargs:
            merge_kwargs['sigma_init'] = kwargs['sigma']
        if method in ('ga_merge_track', 'ga_path'):
            merge_kwargs.setdefault('sigma_rate', 0.1)
        if method == 'ga_path':
            # ga_merge_track's rules plus the push along the centroid's path.
            merge_kwargs.setdefault('merge_rate', 0.1)
            merge_kwargs.setdefault('merge_max', 2.0)
            merge_kwargs.setdefault('path_rate', 1.0)
        return MergeGASearcher(num_params, population_size,
                               merge='pc' if method == 'ga_merge_pc'
                               else 'noise', **merge_kwargs)
    if method == 'ga_subspace':
        sub_kwargs = {k: v for k, v in kwargs.items()
                      if k in ('elite_ratio', 'sigma_init', 'sigma_sub_init',
                               'subspace_fraction', 'adapt_rate',
                               'survive_target', 'sigma_min', 'sigma_sub_min',
                               'sigma_sub_max', 'init_scale',
                               'init_around_mean')}
        if 'sigma' in kwargs:
            sub_kwargs['sigma_init'] = kwargs['sigma']
        return SubspaceGASearcher(num_params, population_size, **sub_kwargs)
    if method in GA_METHODS:
        ga_kwargs = {k: v for k, v in kwargs.items()
                     if k in ('elite_ratio', 'sigma_init', 'sigma_decay',
                              'sigma_limit', 'cross_over_rate', 'init_scale',
                              'init_around_mean', 'iso_sigma', 'line_sigma',
                              'variation', 'sigma_rule', 'track_rate',
                              'track_target', 'sigma_min', 'ties')}
        # `ga_keep`: the refreshed gaussian GA where a tie keeps the parent.
        if method == 'ga_keep':
            ga_kwargs.setdefault('ties', 'parents')
        # `ga` refreshes; `ga_stale` is the reference's carry-the-stored-value
        # behaviour, kept so the old runs can be reproduced and so the toy
        # landscape can show the difference. `ga_fresh` is an alias for `ga`,
        # kept because scripts and run tags already use the name.
        ga_kwargs['refresh'] = kwargs.get('refresh', method != 'ga_stale')
        # `ga_isoline` is the GA breeding with DNS's Iso+LineDD and selecting
        # on fitness exactly as `ga` does -- the ablation arm that says how
        # much of a DNS/GA gap is the operator rather than the selection rule.
        # The method name sets it so a run tag records the operator; passing
        # `variation=` directly still works and wins.
        ga_kwargs.setdefault('variation',
                             ISOLINE if method == 'ga_isoline' else GAUSSIAN)
        # `ga_track` / `ga_success`: the gaussian GA with sigma set by how
        # well the archive's centroid, or its offspring, score against the
        # archive; see GASearcher.__init__.
        ga_kwargs.setdefault('sigma_rule', {'ga_track': 'track',
                                            'ga_success': 'success'}
                             .get(method, 'decay'))
        # `sigma` is the name every caller already uses for a mutation scale;
        # accept it as an alias so one tuner grid covers NES and GA.
        if 'sigma' in kwargs:
            ga_kwargs['sigma_init'] = kwargs['sigma']
        return GASearcher(num_params, population_size, **ga_kwargs)
    if method in DNS_METHODS:
        dns_kwargs = {k: v for k, v in kwargs.items()
                      if k in ('iso_sigma', 'line_sigma', 'k',
                               'normalize_descriptors', 'batch_size',
                               'repertoire_ratio', 'traj_steps', 'obs_dim',
                               'init_scale', 'init_around_mean', 'variation',
                               'sigma_init', 'cross_over_rate')}
        # Same split as GA: `dns` re-evaluates its repertoire every generation,
        # `dns_stale` is the reference's carry-the-stored-value behaviour, kept
        # so the pre-2026-08-27 runs can be reproduced, and `dns_fresh` is an
        # alias for `dns`.
        dns_kwargs['refresh'] = kwargs.get('refresh', method != 'dns_stale')
        # The mirror of `ga_isoline`: dominated-novelty selection over
        # offspring bred by the GA's gaussian mutation.
        dns_kwargs.setdefault('variation',
                              GAUSSIAN if method == 'dns_gaussian' else ISOLINE)
        if 'sigma' in kwargs:
            dns_kwargs['sigma_init'] = kwargs['sigma']
        return DNSSearcher(num_params, population_size, descriptor_dim,
                           **dns_kwargs)
    raise ValueError(f'unknown NE method {method!r}')
