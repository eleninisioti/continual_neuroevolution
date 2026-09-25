"""Toy fitness landscapes: task families whose parameter space can be drawn.

A toy landscape is an environment with no episode. A genome is a point, its
score on a sub-task is a number, and the surface can be evaluated on a grid --
so a figure of it is the surface itself rather than a slice through a policy
network's. They exist to test, where the answer can be seen, mechanisms the
gymnax / mjx / kinetix runs can only be argued about.

    smooth   section A of `docs/generalists/generalist_report.html`. Two
             tilted planes clipped at a cap; both sub-tasks are at the cap on
             the lens |x| + y^2 <= h. Concave, so there are no local optima.
             The level is h, the size of the shared region.
    rugged   section B. Three specialist peaks per sub-task, one generalist
             peak at the origin, and a ripple subtracted from both sub-tasks.
             The level is the ripple amplitude. The search is scored on the
             rugged surface and REPORTED on the ripple-free one, so the target
             does not shrink as the amplitude grows.
    wells    the centroid question (2026-09-13): why a GA's archive mean does
             not converge onto its elite. The first `relevant_dims` coordinates
             each have two gaussian wells at +-WELL_A and the score is their
             mean; every other coordinate does nothing. The level is how much
             lower the negative well is. At level 0 all 2^k optima tie, so
             truncation selection cannot choose between them and only drift
             can; with `num_params > relevant_dims` the population also spreads
             along directions that do not change the score. One sub-task.
    spikes   the basin-width question (2026-09-18): does the width of the basin
             ES settles in depend on its sigma? One wide hill at the origin
             that both sub-tasks share, plus, for each sub-task, a narrow
             specialist spike on its own shoulder of the hill, ADDED to it (not
             the max) so the generalist is never a maximum of either sub-task
             alone: a search that resolves the spike leaves the hill top for
             it at every switch, a search that smooths it over stays. The
             level is the spike's height s.

`smooth` and `rugged` are ported from
`scripts/outdated/generalists/analysis/toy_{shared,rugged}_landscape.py` with
the same constants and formulas, so a run here and the published sweeps score
the same point the same way.

Every score function takes the FULL parameter vector `(..., num_params)` and a
traced `task` and `level`, so a sweep over levels, seeds and generations is one
XLA program. Coordinates past `relevant` are null directions, as in a policy
network with many weights that barely matter.

Not in `source/envs/registry.py`: that interface is built around a policy
rolled out in an environment (observation width, action count, rollouts), and
a toy has none of it. The runner is `source/studies/toy/sweep.py`. As with the
rest of this package, nothing here imports a searcher, a trainer or a parser.
"""

from __future__ import annotations

from typing import Callable, NamedTuple

import jax.numpy as jnp
import numpy as np


class Landscape(NamedTuple):
    """One toy task family, as the runner and the figures need it."""

    name: str
    options: dict               # the factory arguments, recorded with a run
    level_name: str             # the knob as one word, for filenames
    level_label: str            # the knob as an axis label
    levels: tuple               # swept by default
    report_levels: tuple        # drawn by default
    peak: float                 # best generalist score; thresholds are fractions of it
    num_tasks: int              # 2: a switching schedule; 1: stationary
    relevant: int               # how many leading coordinates the score reads
    task_score: Callable        # (x, task, level) -> what the search is scored on
    generalist_score: Callable  # (x, level) -> what is reported
    start: Callable             # (num_params) -> where every run begins
    window: tuple               # ((x0, x1), (y0, y1)) for the first two coordinates
    markers: tuple = ()         # ((x, y, marker), ...) drawn on landscape panels
    rest: float = 0.0           # relevant coordinates past the first two, in a 2-D slice
    defaults: dict = {}         # runner settings this family needs, e.g. pop_size
    # (members (P, d)) -> the share of members outside the most common basin.
    # None where the surface has a single basin, so the question is empty.
    minority: Callable = None


# ---------------------------------------------------------------------------
# smooth (section A)
# ---------------------------------------------------------------------------

SMOOTH_CAP = 0.70
SMOOTH_SLOPE = 0.35
START_X = 1.8          # both switching toys start at (-START_X, 0)


def _smooth_task(x, task, h):
    side = jnp.where(task == 0, 1.0, -1.0)
    return SMOOTH_CAP - jnp.maximum(
        0.0, SMOOTH_SLOPE * (side * x[..., 0] + x[..., 1] ** 2 - h))


def _smooth_generalist(x, h):
    return SMOOTH_CAP - jnp.maximum(
        0.0, SMOOTH_SLOPE * (jnp.abs(x[..., 0]) + x[..., 1] ** 2 - h))


def _start_left(num_params):
    return jnp.zeros(num_params).at[0].set(-START_X)


def smooth():
    return Landscape(
        name='smooth', options={}, level_name='h',
        level_label='shared half-width $h$',
        levels=(0.1, 0.3, 1.0), report_levels=(0.1, 0.3, 1.0),
        peak=SMOOTH_CAP, num_tasks=2, relevant=2,
        task_score=_smooth_task, generalist_score=_smooth_generalist,
        start=_start_left, window=((-2.6, 2.6), (-1.2, 1.2)),
        markers=((-START_X, 0.0, 's'),),
        defaults=dict(num_generations=1000, pop_size=64, task_interval=100))


# ---------------------------------------------------------------------------
# rugged (section B)
# ---------------------------------------------------------------------------

RUGGED_WIDTH = 1.10     # every peak, specialist and generalist alike
RUGGED_PEAK = 0.65      # each specialist
RUGGED_SHARED = 0.70    # the one generalist peak, a little above them
RUGGED_CAP = 0.70
# Sub-task 0's specialists; sub-task 1's are these mirrored in x. Every
# coordinate is a multiple of the default wavelength, so the ripple digs pits
# between the peaks without moving or lowering any of them.
SPECIALISTS = np.array([[-1.8, 0.0], [-1.2, 1.2], [-1.2, -1.2]])


def rugged(wavelength=0.6, specialist_peak=RUGGED_PEAK, rugged_dims=2):
    """Two options, both added 2026-09-14 for the generalist question:

    specialist_peak  above RUGGED_SHARED makes the generalist a trade-off: on
                     each sub-task its specialists then beat it. At the default
                     0.65 it is the best point of both sub-tasks and nothing
                     selects against it.
    rugged_dims      how many coordinates the ripple digs pits along (the
                     first two carry the peaks; the rest have their best value
                     at 0). The same barrier per coordinate, so more of them is
                     local optima in more dimensions, not higher barriers.
    """
    k = 2.0 * np.pi / wavelength
    r = max(2, int(rugged_dims))
    cap = max(RUGGED_CAP, float(specialist_peak))

    def penalty(x, amplitude):
        # Zero on the lattice, at most `amplitude` / 2 per coordinate between;
        # a sum rather than a product so no line through the plane is free of it.
        return amplitude * jnp.sum(1.0 - jnp.cos(k * x[..., :r]), axis=-1) / 4.0

    def smooth_task(x, task):
        sign = jnp.where(task == 0, 1.0, -1.0)
        cx, cy = sign * SPECIALISTS[:, 0], SPECIALISTS[:, 1]
        d2 = ((x[..., 0][..., None] - cx) ** 2
              + (x[..., 1][..., None] - cy) ** 2)
        specialist = jnp.max(
            specialist_peak * jnp.exp(-d2 / (2 * RUGGED_WIDTH ** 2)), axis=-1)
        shared = RUGGED_SHARED * jnp.exp(
            -(x[..., 0] ** 2 + x[..., 1] ** 2) / (2 * RUGGED_WIDTH ** 2))
        return jnp.clip(jnp.maximum(specialist, shared), 0.0, cap)

    def task_score(x, task, amplitude):
        score = smooth_task(x, task) - penalty(x, amplitude)
        # Beyond two coordinates the summed penalty of the start's scatter
        # alone clips every member to exactly 0 (k = 32, scatter 0.05: all of
        # them), a plateau with nothing to select on; so only the two-coordinate
        # landscape keeps its floor at 0 (2026-09-14).
        return jnp.clip(score, 0.0, cap) if r == 2 else jnp.minimum(score, cap)

    def generalist_score(x, amplitude):
        # Ripple-free on purpose: the target must be the same region at every
        # amplitude, or the sweep confounds difficulty with target size.
        return jnp.minimum(smooth_task(x, 0), smooth_task(x, 1))

    def minority(members):
        # A basin of the rippled surface is one lattice cell of the ripple.
        cells = jnp.round(members[:, :2] / wavelength)
        same = jnp.all(cells[:, None, :] == cells[None, :, :], axis=-1)
        return 1.0 - jnp.max(jnp.mean(same, axis=1))

    markers = ((0.0, 0.0, '*'),) + tuple(
        (float(cx), float(cy), 'x')
        for cx, cy in np.concatenate([SPECIALISTS, SPECIALISTS * [-1.0, 1.0]]))
    return Landscape(
        name='rugged', options=dict(wavelength=float(wavelength),
                                    specialist_peak=float(specialist_peak),
                                    rugged_dims=r),
        level_name='amp', level_label='ruggedness amplitude',
        levels=(0.0, 0.1, 0.2, 0.3, 0.4, 0.8, 1.6), report_levels=(0.8, 1.6),
        peak=RUGGED_SHARED, num_tasks=2, relevant=r,
        task_score=task_score, generalist_score=generalist_score,
        start=_start_left, window=((-2.8, 2.8), (-2.2, 2.2)), markers=markers,
        defaults=dict(num_generations=1000, pop_size=64, task_interval=100),
        minority=minority)


# ---------------------------------------------------------------------------
# wells (the centroid question)
# ---------------------------------------------------------------------------

WELL_A = 1.0     # the two wells of every relevant coordinate sit at +-WELL_A
WELL_W = 0.4     # wide enough that the start (the origin) has a gradient


def wells(relevant_dims=2, combine='mean'):
    """`combine` is how the per-coordinate well scores make one score.

    mean  additive: every coordinate's basin choice is worth 1/k of the score,
          and averaging members from different corners only costs the split
          coordinates' share. With many coordinates the centroid even beats
          the elite, because averaging cancels mutation damage (2026-09-13).
    min   epistatic, the weakest coordinate decides: a genome is only as good
          as its worst-placed coordinate, so a child that knocks ANY coordinate
          off its well is ruined and a centroid that sits in the valley of any
          split coordinate fails -- the Kinetix signature (solvers whose
          children rarely inherit, a mean of solvers that does not solve).
    """
    k = int(relevant_dims)
    if combine not in ('mean', 'min'):
        raise ValueError(f"combine must be 'mean' or 'min', not {combine!r}")
    pool = jnp.mean if combine == 'mean' else jnp.min

    def score(x, level):
        u = x[..., :k]
        right = jnp.exp(-(u - WELL_A) ** 2 / (2 * WELL_W ** 2))
        left = (1.0 - level) * jnp.exp(-(u + WELL_A) ** 2 / (2 * WELL_W ** 2))
        return pool(jnp.maximum(right, left), axis=-1)

    def task_score(x, task, level):
        return score(x, level)

    def minority(members):
        # Every relevant coordinate is a two-basin locus: the minority share
        # of each, averaged over the loci. 0 once each locus has fixed.
        frac = jnp.mean(members[:, :k] > 0, axis=0)
        return jnp.mean(jnp.minimum(frac, 1.0 - frac))

    corners = tuple((sx * WELL_A, sy * WELL_A, 'x')
                    for sx in (-1, 1) for sy in (-1, 1))
    return Landscape(
        name='wells', options=dict(relevant_dims=k, combine=combine),
        level_name='delta',
        level_label='negative-well deficit $\\Delta$',
        levels=(0.0, 0.01, 0.05, 0.2), report_levels=(0.0, 0.05),
        peak=1.0, num_tasks=1, relevant=k,
        task_score=task_score, generalist_score=score,
        # The origin is the ridge between the wells: a population started
        # there straddles every coordinate's two basins.
        start=lambda num_params: jnp.zeros(num_params),
        window=((-2.0, 2.0), (-2.0, 2.0)), markers=corners, rest=WELL_A,
        defaults=dict(num_generations=2000, pop_size=256), minority=minority)


# ---------------------------------------------------------------------------
# spikes (the basin-width question)
# ---------------------------------------------------------------------------

SPIKE_OFFSET = 0.6     # each sub-task's spike sits at (-+SPIKE_OFFSET, 0)
SPIKE_WIDTH = 0.2      # against the hill's 1.1


def spikes(spike_width=SPIKE_WIDTH, spike_offset=SPIKE_OFFSET):
    """The hill of `rugged` (0.70, width 1.1) with one narrow spike per sub-task.

    Sub-task A scores hill(x) + spike(x - (-c, 0)); B mirrors the spike to
    (+c, 0). The spike is a gaussian of width w = 0.2 and height s, the level:
    on its own sub-task the specialist scores hill(c) + s = 0.60 + s against
    the generalist's 0.70 at the origin. The two are ADDED, not pooled by a
    max, and c = 0.6 is close enough that the spike's tail tilts the hill top
    (slope 0.025 at s = 0.15, 0.1 at s = 0.6): the origin is a maximum of
    NEITHER sub-task, so a search that resolves the spike leaves the hill
    top for it at every switch. The spike is a maximum of its sub-task while
    its steepest slope s / (w sqrt e) exceeds the hill's slope at c (0.30):
    from s ~ 0.1 up.

    Smoothing a gaussian of width w with an isotropic gaussian of width sigma
    (what ES climbs) leaves a gaussian of width sqrt(w^2 + sigma^2) whose
    height is scaled by w^2 / (w^2 + sigma^2) in two dimensions, so the spike
    fades from the smoothed surface once sigma passes w while the hill hardly
    changes. The smoothed surface has one maximum, which slides from the
    spike toward the origin as sigma grows; it enters the generalist region
    (min(A, B) >= 98% of 0.70, radius 0.22) near sigma = 0.25 at s = 0.15,
    0.4 at s = 0.3 and 0.5 at s = 0.6. So sigma, not the landscape alone,
    decides the width of the basin the search settles in.
    """
    w, c = float(spike_width), float(spike_offset)

    def hill(x):
        return RUGGED_SHARED * jnp.exp(
            -(x[..., 0] ** 2 + x[..., 1] ** 2) / (2 * RUGGED_WIDTH ** 2))

    def spike(x, task, height):
        cx = jnp.where(task == 0, -c, c)
        return height * jnp.exp(-((x[..., 0] - cx) ** 2 + x[..., 1] ** 2)
                                / (2 * w ** 2))

    def task_score(x, task, height):
        return hill(x) + spike(x, task, height)

    def generalist_score(x, height):
        return jnp.minimum(task_score(x, 0, height), task_score(x, 1, height))

    def minority(members):
        # Three basins: A's spike (x < -c/2), B's (x > c/2), the hill between.
        basin = jnp.digitize(members[:, 0], jnp.asarray([-c / 2, c / 2]))
        return 1.0 - jnp.max(jnp.mean(basin[:, None] == jnp.arange(3)[None],
                                      axis=0))

    markers = ((0.0, 0.0, '*'), (-c, 0.0, 'x'), (c, 0.0, 'x'))
    return Landscape(
        name='spikes', options=dict(spike_width=w, spike_offset=c),
        level_name='s', level_label='specialist height $s$',
        levels=(0.15, 0.3, 0.6), report_levels=(0.3,),
        peak=RUGGED_SHARED, num_tasks=2, relevant=2,
        task_score=task_score, generalist_score=generalist_score,
        start=_start_left, window=((-2.8, 2.8), (-2.2, 2.2)), markers=markers,
        defaults=dict(num_generations=1000, pop_size=64, task_interval=100),
        minority=minority)


# ---------------------------------------------------------------------------
# manifold (population size, manifold dimension, and the centroid)
# ---------------------------------------------------------------------------

def manifold(stiff_dims=0, tube_width=0.1, wavelength=0.6, manifold_base='rugged'):
    """`rugged` inside a tube: the solution set as a manifold of known dimension.

    `manifold_base='smooth'` (2026-09-22) puts the smooth landscape in the tube
    instead: the shared region of width h, now with `stiff_dims` directions in
    which every solution of both sub-tasks is narrow. The question is whether
    the search still holds the shared region when the curvature is spread
    over more directions, not only whether it crosses the ripple.

    Coordinates 0-1 are the rugged two-sub-task surface (specialists, one
    generalist, the ripple's local optima). The next `stiff_dims` must stay
    within about `tube_width` of 0 -- both sub-tasks' scores are multiplied by
    exp(-|y|^2 / 2 w^2) -- and every coordinate after them changes nothing.
    So around the generalist the set of equally good genomes has dimension
    `num_params - 2 - stiff_dims`, and the ratio of the mutation width to
    `tube_width` sets how often a good genome's child is still good.

    Axis-aligned on purpose: gaussian mutation, Iso+LineDD and a PCA of the
    population are all rotation-invariant, so a random rotation of the
    subspaces would change nothing the runner measures.
    """
    if manifold_base not in ('rugged', 'smooth'):
        raise ValueError(f"manifold_base must be 'rugged' or 'smooth', not {manifold_base!r}")
    base = rugged(wavelength) if manifold_base == 'rugged' else smooth()
    c, w = int(stiff_dims), float(tube_width)

    def tube(x):
        if c == 0:
            return 1.0
        return jnp.exp(-jnp.sum(x[..., 2:2 + c] ** 2, axis=-1) / (2 * w ** 2))

    def task_score(x, task, amplitude):
        return base.task_score(x, task, amplitude) * tube(x)

    def generalist_score(x, amplitude):
        return base.generalist_score(x, amplitude) * tube(x)

    # Every run starts ON the tube: the stiff coordinates of a population
    # jittered at the other toys' 0.05 / 0.1 per coordinate are exp(-c/2)-ish
    # off it, which at c = 48 is a surface of zeros with nothing to select on
    # (the first grid, 2026-09-13, found nothing at c >= 16 for that reason).
    # So the start and the searchers' initial scatter are a tenth of the tube
    # width; how far children then fall off is the mutation's business alone.
    levels = dict(levels=(0.0, 0.8, 1.6), report_levels=(0.0, 1.6),
                  level_label='ruggedness amplitude') if manifold_base == 'rugged' \
        else {}
    return base._replace(
        name='manifold',
        options=dict(stiff_dims=c, tube_width=w, wavelength=float(wavelength),
                     manifold_base=manifold_base),
        **levels, relevant=2 + c, task_score=task_score,
        generalist_score=generalist_score,
        defaults=dict(num_generations=1000, pop_size=64, task_interval=100,
                      start_jitter=w / 10, init_scale=w / 10))


FACTORIES = {'smooth': smooth, 'rugged': rugged, 'wells': wells,
             'spikes': spikes, 'manifold': manifold}
NAMES = tuple(FACTORIES)


def get(name, **options):
    """The landscape called `name`, built with only the options it takes."""
    import inspect
    factory = FACTORIES[name]
    accepted = inspect.signature(factory).parameters
    return factory(**{k: v for k, v in options.items()
                      if k in accepted and v is not None})


def plane(land, gx, gy):
    """Grid points `(ny, nx, max(2, relevant))` over the first two coordinates.

    Relevant coordinates past the first two are held at `land.rest` -- for the
    wells, at an optimum -- so the panel is the slice through a best point.
    """
    width = max(2, land.relevant)
    pts = np.full(gx.shape + (width,), land.rest, dtype=np.float32)
    pts[..., 0], pts[..., 1] = gx, gy
    return jnp.asarray(pts)
