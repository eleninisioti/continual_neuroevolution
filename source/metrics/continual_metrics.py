"""The continual metrics the paper reports, as arithmetic on training curves.

Extracted from `scripts/outdated/compare.py` on 2026-09-08 so the figure script
and the table script cannot come to define the same column differently -- the
failure this repo keeps guarding against. Every definition below is that
file's, unchanged; what is new is only that it lives somewhere importable.

    Cum. max    the area under the training curve over the whole sequence, in
                reward x generations. Integrated rather than summed, because NE
                logs once per generation and RL once per update, so the same
                budget gives one family four times the points. It prices in
                every drop at a switch and every recovery after one: a method
                that solves each sub-task *eventually* is charged for the
                generations it spent getting there.
    Cum. mean   the same integral of the population-average curve. The
                single-policy RL methods have no population, so it is undefined
                for them rather than equal to their Cum. max.
    FT          forward transfer: the mean over sub-tasks of the continual
                curve's window mean, minus the same method's own STATIONARY
                run over an equal window. 0 means a sub-task was learnt as well
                as the stationary task; negative is what the continual setting
                cost. Needs no threshold.

## What is NOT here, and why

Forgetting, behavioural divergence and zero-shot transfer are not functions of
a training curve. They are measured post hoc, by re-rolling every sub-task's
saved agent against every sub-task (`evaluate_continual.py`), and live in the
`evaluation.json` / divergence records that pass writes. A tree without that
pass has no F, BD or ZT, and the honest report is a blank rather than a number
derived from something else.

## FT is Continual World's numerator, not its FT

Wolczyk et al. (2021, sec. 4.1) define `FT_i = (AUC_i - AUC_i^b) / (1 - AUC_i^b)`.
The denominator normalises by the headroom above the stationary run and needs
that run to be some way off ceiling. Ours are not -- on Acrobot and Cartpole
every method solves the stationary task within a few dozen generations, so
`AUC^b` runs 0.95-0.99, `1 - AUC^b` is a rounding error, and the ratio lands at
-37 where the numerator says -0.52. That is the case their footnote 6 sets
aside as ill-defined. What is left needs no threshold and is in the
environment's own reward units, deliberately -- so is forgetting, so the two
can be read against each other.

**Read FT beside an absolute column, never alone.** It subtracts each method's
OWN stationary reference, so a method that learns the stationary task badly has
little left to lose and scores well for the wrong reason.
"""

from __future__ import annotations

import numpy as np

# `trapezoid` is the numpy>=2 spelling; `trapz` the <2 one, removed in 2.0.
# Bound once here rather than guarded at each call site.
_TRAPEZOID = getattr(np, 'trapezoid', None) or np.trapz

__all__ = ['cumulative_reward', 'window_mean', 'forward_transfer_trials',
           'holm', 'mann_whitney_tests', 'mann_whitney_marks',
           'bootstrap_ci', 'SIG_LEVELS']


def bootstrap_ci(values, level=0.95, n_boot=2000, seed=0, chunk=200):
    """`(mean, lo, hi)` of the seed-mean by percentile bootstrap.

    ``values`` is `(n_seeds,)` or `(n_seeds, T)`; the statistic is the mean
    over the first axis and the interval is the percentile one, resampling
    seeds with replacement. The figures report the MEAN, not the median,
    because the per-seed outcomes here are bimodal -- a PPO seed on CartPole
    scores 500 on a sub-task or 9 -- and the mean is the only summary that
    still encodes how many seeds failed. A median flips to the ceiling at 6
    of 10, and an interquartile band hides the failed seeds under it. The
    bootstrap CI says where the expected return is, given this many seeds,
    and is narrow enough to draw where a +-sd band spans the axis.

    Fixed ``seed`` so the same runs draw the same band twice. Chunked so a
    `(10, 4000)` curve at 2000 resamples does not allocate 640 MB at once.
    A single seed has no interval and gets `lo == hi == mean`.
    """
    v = np.asarray(values, dtype=float)
    squeeze = v.ndim == 1
    if squeeze:
        v = v[:, None]
    n = v.shape[0]
    mean = v.mean(axis=0)
    if n < 2:
        out = (mean, mean.copy(), mean.copy())
        return tuple(o[0] if squeeze else o for o in out)
    rng = np.random.default_rng(seed)
    boots = np.empty((n_boot, v.shape[1]))
    for start in range(0, n_boot, chunk):
        stop = min(start + chunk, n_boot)
        idx = rng.integers(0, n, size=(stop - start, n))
        boots[start:stop] = v[idx].mean(axis=1)
    alpha = (1.0 - level) / 2.0
    lo, hi = np.percentile(boots, [100 * alpha, 100 * (1 - alpha)], axis=0)
    out = (mean, lo, hi)
    return tuple(o[0] if squeeze else o for o in out)


def cumulative_reward(x, y, x_hi, x_lo=0.0):
    """Area under a training curve over ``[x_lo, x_hi]``, in reward x x-units.

    `x` and `x_hi` must be on the same clock; the paper's tables use
    generation-equivalents, so an RL run's updates are rescaled onto that axis
    before this is called. The curve is held at its first logged value before
    the first sample and at its last after the last, which only bites for RL,
    whose first evaluation lands a few generation-equivalents in.

    ``x_lo`` exists for the reading that wants sub-tasks 1..T-1 only: sub-task
    0 is the unperturbed environment learned from scratch, and including it
    credits fast initial learning rather than the ability to acquire a *new*
    sub-task.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size < 2 or x_hi <= x_lo:
        return float('nan')
    grid = np.linspace(float(x_lo), float(x_hi), max(int(x_hi - x_lo) + 1, 2))
    return float(_TRAPEZOID(np.interp(grid, x, y), grid))


def window_mean(x, y, start, stop):
    """Mean of the curve over ``[start, stop)``, in the environment's own units.

    NOT min-max scaled. FT is a difference of two windows of the same env and
    forgetting is a difference of two returns on the same env, so a
    ``(hi - lo)`` divisor would be a per-env constant that changes no ordering
    within a column and buys only cross-env comparability, which no table here
    uses. Leaving both unscaled is what lets them be read against each other.

    NaN when the window holds no logged point, so a caller can tell "the run
    never reached this window" from "the run scored zero here".
    """
    x = np.asarray(x, dtype=float)
    mask = (x >= start) & (x < stop)
    if not mask.any():
        return float('nan')
    return float(np.asarray(y, dtype=float)[mask].mean())


def forward_transfer_trials(ref, cont, edges):
    """Per-trial ``mean_i(AUC_i - AUC_i^b)``; empty where undefined.

    ``ref`` and ``cont`` are ``[(x, y), ...]`` per trial, already on the clock
    the caller means to window in, and ``edges`` are the phase edges
    ``[0, ..., total]`` on that same clock. The reference is the method's OWN
    stationary run over a window as long as phase ``i``, from its start -- per
    method on purpose, because this column answers what the continual setting
    cost *that* method rather than how it ranks against the others.

    Phases need not be equally long: under a warm-up the first holds sub-task
    0 for ten ordinary phases' worth and is scored against that much of the
    reference. With equal phases every ``AUC_i^b`` is the same window and this
    is the familiar ``mean_i(AUC_i) - AUC^b``.
    """
    if not ref or not cont:
        return np.array([])
    edges = np.asarray(edges, dtype=float)
    spans = list(zip(edges[:-1], edges[1:]))
    baseline = {}
    for lo, hi in spans:
        if hi - lo not in baseline:
            baseline[hi - lo] = np.nanmean([window_mean(x, y, 0.0, hi - lo)
                                            for x, y in ref])
    per_trial = []
    for x, y in cont:
        per_task = [window_mean(x, y, lo, hi) - baseline[hi - lo]
                    for lo, hi in spans]
        per_task = [v for v in per_task if np.isfinite(v)]
        if not per_task:
            continue
        per_trial.append(float(np.mean(per_task)))
    return np.array(per_trial, dtype=float)


def holm(pvalues):
    """Holm-Bonferroni adjusted p-values, in the input order.

    Holm rather than plain Bonferroni because the comparisons are one method
    against each rival in the other family, and Bonferroni at that width throws
    away real effects; Holm is uniformly more powerful and needs no extra
    assumption. Adjusted values are made monotone, as the method requires.
    """
    pvalues = np.asarray(pvalues, dtype=float)
    k = pvalues.size
    if k == 0:
        return pvalues
    order = np.argsort(pvalues)
    adjusted = np.empty(k, dtype=float)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (k - rank) * float(pvalues[idx]))
        adjusted[idx] = min(1.0, running)
    return adjusted


# Thresholds for the asterisks, loosest last. Applied to the WEAKEST of a
# method's comparisons, so a mark means the claim holds against every rival.
SIG_LEVELS = ((0.001, '***'), (0.01, '**'), (0.05, '*'))


def mann_whitney_tests(values_by_method, families, higher_is_better=True):
    """Every cross-family comparison behind `mann_whitney_marks`, in full.

    Returns ``{method: {'n', 'mark', 'worst_p_holm', 'comparisons'}}`` where
    each comparison is ``{'rival', 'n_rival', 'U', 'p', 'p_holm', 'ps'}``:
    ``U`` is scipy's statistic for ``method``'s sample, ``p`` the one-sided
    p-value in the direction the column is read, ``p_holm`` its Holm-adjusted
    value within ``method``'s comparisons, and ``ps`` the probability of
    superiority in that same direction -- the chance a random seed of
    ``method`` is better than a random seed of the rival, ties counted half,
    i.e. (1 + Cliff's delta) / 2.

    A method with fewer than two finite values, with no rival, or with a rival
    that has fewer than two gets no comparisons and no mark: the claim is
    "better than EVERY member of the other family", and it cannot be made
    against a rival that was not measured.
    """
    from scipy.stats import mannwhitneyu

    alt = 'greater' if higher_is_better else 'less'
    finite = {m: [float(v) for v in (vals if vals is not None else [])
                  if np.isfinite(v)]
              for m, vals in values_by_method.items()}
    out = {}
    for method, mine in finite.items():
        rivals = [m for m in finite
                  if families.get(m) not in (None, families.get(method))]
        rec = {'n': len(mine), 'mark': '', 'worst_p_holm': None,
               'comparisons': []}
        out[method] = rec
        if (len(mine) < 2 or not rivals
                or any(len(finite[r]) < 2 for r in rivals)):
            continue
        comps = []
        for rival in rivals:
            theirs = finite[rival]
            res = mannwhitneyu(mine, theirs, alternative=alt)
            frac = float(res.statistic) / (len(mine) * len(theirs))
            comps.append({'rival': rival, 'n_rival': len(theirs),
                          'U': float(res.statistic), 'p': float(res.pvalue),
                          'ps': frac if higher_is_better else 1.0 - frac})
        for comp, adjusted in zip(comps, holm([c['p'] for c in comps])):
            comp['p_holm'] = float(adjusted)
        rec['comparisons'] = comps
        rec['worst_p_holm'] = max(c['p_holm'] for c in comps)
        rec['mark'] = next((mark for level, mark in SIG_LEVELS
                            if rec['worst_p_holm'] < level), '')
    return out


def mann_whitney_marks(values_by_method, families, higher_is_better=True):
    """Significance mark per method: does it beat every member of the other family?

    ``values_by_method`` maps a method to its per-seed values;  ``families``
    maps a method to a family label (here 'ne' or 'rl'). A method is marked
    only if its WEAKEST comparison against the other family survives
    Holm-Bonferroni correction within its own set of comparisons -- so a mark
    means "better than all of them", never "better than one of them". The
    tests themselves are `mann_whitney_tests`.

    Returns ``{method: mark}`` with '' where nothing survives.
    """
    try:
        tests = mann_whitney_tests(values_by_method, families, higher_is_better)
    except ImportError:                                   # pragma: no cover
        return {m: '' for m in values_by_method}
    return {m: rec['mark'] for m, rec in tests.items()}
