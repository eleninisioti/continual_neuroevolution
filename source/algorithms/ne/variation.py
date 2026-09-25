"""The variation operators the population-based methods breed with.

The GA and DNS are the same (mu + lambda) loop under two independent
substitutions: *how offspring are produced* (this file) and *what survivors are
ranked by* (fitness for the GA, dominated novelty for DNS -- see
`source/algorithms/ne/dns.py`). Until 2026-09-08 those two axes were welded
together: the GA could only mutate a single parent with isotropic gaussian
noise and DNS could only use Iso+LineDD, so "DNS beats the GA" was a statement
about novelty selection AND recombination AND the operator's width all at once,
and no run in the tree separated them.

They are separated here. Both operators live behind one signature

    op(genotypes, key, num_offspring, ..., return_parents=False)
        -> offspring                        (num_offspring, num_params)
        -> (offspring, base_parents)        if return_parents

so a searcher picks one by name and nothing else about it changes. That makes
the 2x2 -- GA/gaussian, GA/isoline, DNS/gaussian, DNS/isoline -- four runs of
one implementation rather than four implementations, and the diversity claim
can be read off the selection axis with the operator held fixed.

## base_parents, and why it is not the same column for both

`return_parents` reports, for each offspring, the genome it descends from. The
plasticity churn metric needs it: it measures the effect of ONE update on ONE
network, so it must compare an offspring against its own base, not against an
arbitrary archive member (`pairwise_churn` in `source/metrics/plasticity.py`).

Which of the two drawn parents that is differs by operator, and in both cases
it is the non-obvious one:

  gaussian  `b`, the SECOND draw. `_mate` computes `a * (1 - take_b) +
            b * take_b` with `take_b = uniform(...) > cross_over_rate`, and
            `cross_over_rate` defaults to 0, so `take_b` is True everywhere and
            the offspring is `b` plus noise. Reporting `a` would compare each
            offspring against an unrelated genome and inflate churn by the
            archive's own spread.
  isoline   `x1`, the FIRST draw. The operator perturbs `x1` by `iso` and then
            slides it toward `x2`; `x2` only supplies a direction.

Raise `cross_over_rate` above 0, or `line_sigma` far enough that the offspring
lands nearer `x2` than `x1`, and "the parent" stops being well defined --
the churn column then needs rethinking rather than relabelling.

## Widths are not comparable across operators

`sigma` (gaussian) and `iso_sigma` (isoline) are both the width of an isotropic
term, but the isoline operator adds a second, much larger displacement along
`x2 - x1` whose scale is set by the population's own spread. The reference
values differ by two orders of magnitude for exactly that reason (GA 0.1-0.5
against iso 0.005 / line 0.05). Swapping the operator and keeping the number is
therefore not a controlled ablation: hold the operator's own reference widths,
or tune both, and say which was done.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import random


GAUSSIAN = 'gaussian'
ISOLINE = 'isoline'
VARIATIONS = (GAUSSIAN, ISOLINE)

# The reference defaults for each operator, per its own paper.
#   gaussian  Such et al. (2017); the widths every GA trainer here already used
#   isoline   QDax `mutation_operators.py` / the DNS configs
#             (inspiration/DNS/Dominated-Novelty-Search/configs/algo/{me,aurora}.yaml)
DEFAULTS = {
    GAUSSIAN: {'sigma': 0.1, 'cross_over_rate': 0.0},
    ISOLINE: {'iso_sigma': 0.005, 'line_sigma': 0.05},
}


def resolve_params(name, **overrides):
    """Fill an operator's defaults, rejecting a knob it does not have.

    Silently dropping `iso_sigma` on a gaussian run is how an ablation ends up
    reporting the wrong operator, so an unknown key is an error here rather
    than a filter. `None` means "not set" and falls back to the default, so a
    caller can pass every knob it has an argparse flag for.
    """
    if name not in DEFAULTS:
        raise ValueError(f'unknown variation operator {name!r}; '
                         f'expected one of {VARIATIONS}')
    params = dict(DEFAULTS[name])
    for key, value in overrides.items():
        if value is None:
            continue
        if key not in params:
            raise ValueError(
                f'{key!r} is not a parameter of the {name!r} operator '
                f'(it takes {sorted(params)}). Passing it would be silently '
                f'ignored and the run would not be the ablation it claims.')
        params[key] = float(value)
    return params


def _draw_parents(key, pop_size, num_offspring):
    """Two independent uniform draws with replacement, one operator's worth.

    Shared so that the GA and DNS breed from the archive the same way and the
    only thing the operator name changes is what is done with the pair. The
    draw is over the WHOLE archive, not over the first `num_offspring` of it:
    the two sizes differ whenever fewer offspring are bred than there are
    members, and drawing in `[0, num_offspring)` would leave the rest of the
    archive to stagnate.
    """
    k_a, k_b = random.split(key, 2)
    idx_a = random.randint(k_a, (num_offspring,), 0, pop_size)
    idx_b = random.randint(k_b, (num_offspring,), 0, pop_size)
    return idx_a, idx_b


def gaussian_variation(genotypes, key, num_offspring, sigma=0.1,
                       cross_over_rate=0.0, return_parents=False):
    """SimpleGA's operator: one drawn parent plus isotropic gaussian noise.

    With `cross_over_rate = 0` (the default everywhere in this repo) the
    crossover step is a no-op and an offspring is exactly one archive member
    plus `N(0, sigma^2 I)`. The path exists because the reference has it.
    """
    num_offspring = int(num_offspring)
    pop_size, num_params = genotypes.shape
    k_idx, k_eps, k_mate = random.split(key, 3)
    idx_a, idx_b = _draw_parents(k_idx, pop_size, num_offspring)
    a, b = genotypes[idx_a], genotypes[idx_b]

    def mate(k, x, y):
        take_y = random.uniform(k, (num_params,)) > cross_over_rate
        return x * (1 - take_y) + y * take_y

    offspring = jax.vmap(mate)(random.split(k_mate, num_offspring), a, b)
    offspring = offspring + random.normal(
        k_eps, (num_offspring, num_params)) * sigma

    if return_parents:
        # `b`, not `a` -- see the module docstring.
        return offspring, b
    return offspring


def isoline_variation(genotypes, key, num_offspring, iso_sigma=0.005,
                      line_sigma=0.05, return_parents=False):
    """Iso+LineDD (Vassiliades & Mouret, GECCO 2018), as QDax implements it.

    `line_noise` is ONE scalar per individual drawn from a normal, not a
    per-parameter uniform: the offspring is displaced along the whole line
    joining its two parents, which is what makes this a recombination operator
    rather than an anisotropic mutation. Getting that wrong was a real bug in
    this repo's history.

    The operator multiplies population variance by `1 + 2 * line_sigma^2` per
    generation. Fitness selection contracts that; dominated novelty in an
    UNBOUNDED descriptor space rewards the outliers instead, so at the old
    `line_sigma = 0.5` the genotypes diverged -- see
    docs/dns_cheetah_diagnosis.md. The defaults here are the reference's
    corrected 0.005 / 0.05.
    """
    num_offspring = int(num_offspring)
    pop_size, num_params = genotypes.shape
    k_idx, k_line, k_iso = random.split(key, 3)
    idx_1, idx_2 = _draw_parents(k_idx, pop_size, num_offspring)
    x1, x2 = genotypes[idx_1], genotypes[idx_2]

    line = random.normal(k_line, (num_offspring,)) * line_sigma
    iso = random.normal(k_iso, (num_offspring, num_params)) * iso_sigma
    offspring = (x1 + iso) + (x2 - x1) * line[:, None]

    if return_parents:
        # `x1`, the iso base -- see the module docstring.
        return offspring, x1
    return offspring


_OPERATORS = {GAUSSIAN: gaussian_variation, ISOLINE: isoline_variation}


def vary(name, genotypes, key, num_offspring, params, return_parents=False):
    """Apply operator `name` with the parameters `resolve_params` produced.

    The one call site a searcher needs, so that switching operators is a string
    in a config and not a branch in every trainer.
    """
    if name not in _OPERATORS:
        raise ValueError(f'unknown variation operator {name!r}; '
                         f'expected one of {VARIATIONS}')
    return _OPERATORS[name](genotypes, key, num_offspring,
                            return_parents=return_parents, **params)
