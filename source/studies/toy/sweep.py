"""Every NE arm on a toy landscape: one (method, sigma) as one XLA program.

    .venv/bin/python -m source.studies.toy.sweep --landscape smooth
    .venv/bin/python -m source.studies.toy.sweep --landscape rugged
    .venv/bin/python -m source.studies.toy.sweep --landscape wells \\
        --num_params 400 --out_dir projects/iclr_2027/runs_toy/wells_d400
    .venv/bin/python -m source.studies.toy.sweep --landscape wells \\
        --num_params 400 --out_dir projects/iclr_2027/runs_toy/wells_d400 \\
        --project ga 0.05 0      # full-dimensional snapshots of one cell

then `.venv/bin/python scripts/make_toy_figures.py <out_dir>`. A sweep is 2-D
(or d-D) arithmetic vmapped over levels and seeds, so it is CPU work
(`JAX_PLATFORMS=cpu`) and minutes.

WHAT IT REPLACES
----------------
`scripts/outdated/generalists/analysis/toy_sweep.py` (section A's harness)
and the private copy of it inside `toy_rugged_landscape.py` (section B's).
The two had drifted: section B gave OpenES the SGD arm's learning rate, so
its ES moved 4-16x slower than its NES. There is one harness now and it uses
section A's step-matched rates (`method_kwargs`), so section B's ES rows are
expected to differ from the published ones; nothing else about the arms
changed, and the key stream per seed is the one both old harnesses used.

THE SEARCHERS are `source/studies/generalists/ne.py:build_searcher`, the same
objects the gymnax, mjx, minigrid and kinetix runners drive, with `num_params`
changed. (Those shared runners still sit under the generalists study; see
`source/studies/minigrid/settings.py` for why they have not moved.)

THE ARMS are named as the paper's run trees name them -- `es` is OpenES --
plus the toy-only controls:

    ga, ga_isoline, dns, dns_gaussian    the selection x operator 2x2
    es, nes                               the distribution pair
    dns_corrected                         dns at the reference's line_sigma 0.05
    ga_stale                              the archive keeps stored fitness
    nes_mu, nes_mu_select                 8 centroids, without / with selection
    nes_adam, es_sgd, es_nomom,           section A's optimizer x shaping
      nes_adam_nomom                      ablation (off by default)

`dns` and `ga_isoline` breed at `--line_sigma` (0.5, the gymnax arms' value);
`iso_sigma` and every gaussian width take the swept sigma.

COMPUTE MATCH (CLAUDE.md c): every arm evaluates `pop_size` genomes a
generation; GA and DNS re-score their archive out of that budget. NO ARM IS
TOLD WHERE A SWITCH IS (CLAUDE.md d): the sub-task only enters through the
scores.

`--record_centroid` adds the full centroid at every record, for every sigma
(`scripts/analysis/plot_toy_sigma_basin.py` measures the basin around it).

WHAT IS RECORDED, beyond the old harness's found / held / retention:
the CENTROID -- the coordinate-wise mean of the population's weights
(`population_mean`) -- every `record_stride` generations: its generalist
score, the mean score of the members, |centroid - incumbent|, the rms distance
of members to the centroid, and the share of members outside the most common
basin (`Landscape.minority`; NaN on `smooth`, which has one). A centroid that scores below the average member
means the population sits in more than one basin; on a single basin (or a
plateau) it scores at least as well as they do.
"""

from __future__ import annotations

import argparse
import json
import os
import time

import jax
import jax.numpy as jnp
import numpy as np
from jax import random

from source.envs import toy_landscapes
from source.studies.generalists.ne import build_searcher

# ES arms: (shaping, optimizer, momentum). One `ES` class at six settings.
ES_ARMS = {
    'nes':            ('zscore', 'sgd', None),
    'es':             ('centered_rank', 'adam', None),
    'nes_adam':       ('zscore', 'adam', None),
    'es_sgd':         ('centered_rank', 'sgd', None),
    'es_nomom':       ('centered_rank', 'adam', 0.0),
    'nes_adam_nomom': ('zscore', 'adam', 0.0),
}
# Arm -> the `build_searcher` method it is a setting of.
BASE = {'ga': 'ga', 'ga_stale': 'ga_stale', 'ga_keep': 'ga_keep',
        'ga_isoline': 'ga_isoline',
        'ga_track': 'ga_track', 'ga_success': 'ga_success',
        'ga_subspace': 'ga_subspace', 'ga_survive': 'ga_subspace',
        'ga_merge_noise': 'ga_merge_noise', 'ga_merge_pc': 'ga_merge_pc',
        'ga_merge_track': 'ga_merge_track', 'ga_focus_fine': 'ga_focus',
        'dns': 'dns', 'dns_corrected': 'dns', 'dns_gaussian': 'dns_gaussian',
        'nes_mu': 'nes_mu', 'nes_mu_select': 'nes_mu_select',
        **{m: 'nes' for m in ES_ARMS}}
DEFAULT_METHODS = ('ga', 'ga_isoline', 'dns', 'dns_gaussian', 'es', 'nes',
                   'dns_corrected', 'ga_stale', 'nes_mu', 'nes_mu_select')
# The per-`record_stride` series, in the order `run_cell` emits them.
#   q      share of gaussian children of the incumbent (at the arm's sigma)
#          scoring within q_tolerance of it: how well a solution reproduces
#   pr     participation ratio of the population's covariance spectrum
#   wide   directions whose population variance exceeds wide_factor * sigma^2,
#          the candidate count of directions that do not change the score
#   sigma  the searcher's current (isotropic) mutation width
#   sigma_sub  ga_subspace's width along the archive's spread (NaN elsewhere)
#   merge  ga_merge_*'s merge level: rank-noise std or recombination share
RECORD_KINDS = ('cen_g', 'mem_g', 'dist', 'spread', 'minority', 'q', 'pr',
                'wide', 'sigma', 'sigma_sub', 'merge')


def utility_scale(shaping, pop_size):
    """Rms of one shaped utility vector, which an SGD step is proportional to.

    z-scores are 1 by construction; centered ranks on [-0.5, 0.5] are
    sqrt((n + 1) / (12 (n - 1))) ~ 0.29, so an SGD arm on ranks needs its rate
    divided by this to take the same step. Adam normalises it away.
    """
    if shaping == 'zscore':
        return 1.0
    return float(np.sqrt((pop_size + 1) / (12.0 * (pop_size - 1))))


def method_kwargs(method, sigma, args):
    """Everything that distinguishes the arms, in one place.

    Distribution arms take a rate chosen so the step they ACTUALLY take is
    about `lr_scale` in parameter units: SGD on standardised utilities steps
    ~lr/sigma, Adam steps ~lr*sqrt(d) whatever the gradient.
    """
    if method in ES_ARMS:
        shaping, optimizer, momentum = ES_ARMS[method]
        if optimizer == 'sgd':
            rate = args.lr_scale * sigma / utility_scale(shaping, args.pop_size)
        else:
            rate = args.lr_scale / np.sqrt(args.num_params)
        kw = dict(sigma_init=sigma, learning_rate=rate, optimizer=optimizer,
                  shaping=shaping)
        if momentum is not None:
            kw['momentum'] = momentum
        return kw
    # `init_scale` is the searchers' own default (0.1) everywhere but the
    # manifold landscape, so every other toy builds exactly what it did.
    if method in ('ga', 'ga_stale', 'ga_keep'):
        return dict(sigma_init=sigma, elite_ratio=args.elite_ratio,
                    init_scale=args.init_scale)
    if method in ('ga_track', 'ga_success'):
        # sigma is the starting (and largest) width; the rule only shrinks it.
        return dict(sigma_init=sigma, elite_ratio=args.elite_ratio,
                    init_scale=args.init_scale, track_rate=args.track_rate,
                    track_target=(args.track_target if method == 'ga_track'
                                  else args.success_target))
    if method in ('ga_merge_noise', 'ga_merge_pc', 'ga_merge_track'):
        # The merge level (rank noise, or the share of principal-direction
        # recombination) moves with centroid tracking; under ga_merge_track
        # sigma also shrinks from the same signal.
        kw = dict(sigma_init=sigma, elite_ratio=args.elite_ratio,
                  init_scale=args.init_scale, merge_rate=args.merge_rate,
                  merge_max=args.merge_max)
        if method == 'ga_merge_track':
            kw['sigma_rate'] = args.sigma_rate
        return kw
    if method == 'ga_focus_fine':
        # Kinetix's ga_focus_fine: parents from the best `focus` share and
        # sigma shrinking while the centroid lags (target 0.9), floor at
        # sigma / 50000 (Kinetix: 0.5 -> 1e-5).
        return dict(sigma_init=sigma, elite_ratio=args.elite_ratio,
                    init_scale=args.init_scale, focus_rate=0.3, sigma_rate=0.1,
                    track_target=0.9, sigma_min=sigma / 50000)
    if method in ('ga_subspace', 'ga_survive'):
        # `ga_survive` is ga_subspace with no spread-channel offspring.
        return dict(sigma_init=sigma, elite_ratio=args.elite_ratio,
                    init_scale=args.init_scale, adapt_rate=args.track_rate,
                    survive_target=args.survive_target,
                    sigma_sub_init=args.sigma_sub_init,
                    subspace_fraction=(0.0 if method == 'ga_survive'
                                       else args.subspace_fraction))
    if method == 'ga_isoline':
        return dict(elite_ratio=args.elite_ratio, iso_sigma=sigma,
                    line_sigma=args.line_sigma, init_scale=args.init_scale)
    if method in ('dns', 'dns_corrected'):
        return dict(iso_sigma=sigma, k=args.dns_k, init_scale=args.init_scale,
                    line_sigma=(0.05 if method == 'dns_corrected'
                                else args.line_sigma))
    if method == 'dns_gaussian':
        return dict(sigma_init=sigma, k=args.dns_k, init_scale=args.init_scale)
    if method in ('nes_mu', 'nes_mu_select'):
        return dict(sigma_init=sigma, learning_rate=args.lr_scale * sigma,
                    optimizer='sgd', shaping='zscore',
                    num_centroids=args.num_centroids,
                    restart_interval=args.restart_interval,
                    init_scale=args.centroid_spread,
                    restart_scale=args.centroid_spread)
    raise ValueError(f'unknown arm {method!r}')


def build(method, sigma, args):
    return build_searcher(BASE[method], num_params=args.num_params,
                          population_size=args.pop_size, descriptor_dim=2,
                          **method_kwargs(method, sigma, args))


def schedule(land, args):
    return (jnp.arange(args.num_generations) // args.task_interval) \
        % land.num_tasks


def init_run(land, searcher, args, seed):
    """The start state and key for one seed -- the old harnesses' key stream."""
    key = random.key(seed)
    key, start_key, init_key = random.split(key, 3)
    start = land.start(args.num_params) + args.start_jitter * random.normal(
        start_key, (args.num_params,))
    return searcher.init(init_key, start), key


def make_step(land, searcher, level):
    """One generation: ask, score on the current sub-task, tell."""
    def step(carry, task):
        state, key = carry
        key, ask_key = random.split(key)
        population, aux = searcher.ask(ask_key, state)
        fitness = land.task_score(population, task, level)
        carried = population if aux is None else aux
        if searcher.needs_descriptors:
            # Genotypic novelty: a toy has no episode, so a genome's only
            # observable besides its score is its own position.
            state = searcher.tell(state, carried, fitness, population[:, :2])
        else:
            state = searcher.tell(state, carried, fitness)
        return (state, key), None
    return step


def retention_of(hit):
    """Mean fraction of generations still generalist after the first crossing.

    NaN for a seed that never crossed, so a `nanmean` over seeds averages only
    the seeds it is defined for.
    """
    found = hit.any(axis=-1)
    first = hit.argmax(axis=-1)
    after = np.arange(hit.shape[-1]) >= first[..., None]
    frac = np.where(after, hit, False).sum(-1) / np.maximum(after.sum(-1), 1)
    with np.errstate(invalid='ignore'):
        return np.where(found, frac, np.nan)


def run_cell(land, method, sigma, args):
    """One arm at one sigma, every level x every seed.

    Returns `(summary, series, trails)`, each a dict of arrays whose first two
    axes are `(num_levels, num_seeds)`.
    """
    searcher = build(method, sigma, args)
    stride = args.record_stride
    tasks = schedule(land, args).reshape(-1, stride)
    gens = jnp.arange(args.num_generations).reshape(-1, stride)
    # Its own stream, so measuring q never moves the search.
    probe_key = random.key(args.q_seed)

    def one_run(level, seed):
        state, key = init_run(land, searcher, args, seed)
        advance = make_step(land, searcher, level)

        def step(carry, task):
            carry, _ = advance(carry, task)
            state = carry[0]
            inc = searcher.incumbent(state)
            members = searcher.population(state)
            every = (land.generalist_score(inc, level),
                     land.task_score(inc, task, level),
                     jnp.max(land.generalist_score(members, level)),
                     inc[:2], searcher.population_mean(state)[:2])
            return carry, every

        def measure(state, task, gen):
            """The per-`record_stride` statistics, on a chunk's last state."""
            inc = searcher.incumbent(state)
            cen = searcher.population_mean(state)
            members = searcher.population(state)
            member_g = land.generalist_score(members, level)
            # The width the search is actually mutating at: the GA's state
            # carries it (and ga_track moves it); the others use the swept one.
            width = (jnp.mean(jnp.asarray(state.sigma))
                     if hasattr(state, 'sigma') else jnp.asarray(sigma))
            kids = inc[None] + width * random.normal(
                random.fold_in(random.fold_in(probe_key, seed), gen),
                (args.q_samples, inc.shape[0]))
            q = jnp.mean(land.task_score(kids, task, level)
                         >= land.task_score(inc, task, level)
                         - args.q_tolerance * land.peak)
            centred = members - cen
            lam = jnp.linalg.svd(centred, compute_uv=False) ** 2 / max(
                members.shape[0] - 1, 1)
            return (land.generalist_score(cen, level), jnp.mean(member_g),
                    jnp.linalg.norm(cen - inc),
                    jnp.sqrt(jnp.mean(jnp.sum(centred ** 2, -1))),
                    (jnp.nan if land.minority is None
                     else land.minority(members)),
                    q, jnp.sum(lam) ** 2 / jnp.sum(lam ** 2),
                    jnp.sum(lam > args.wide_factor * width ** 2).astype(
                        jnp.float32),
                    width.astype(jnp.float32),
                    (jnp.asarray(state.sigma_sub, dtype=jnp.float32)
                     if hasattr(state, 'sigma_sub')
                     else jnp.asarray(jnp.nan, dtype=jnp.float32)),
                    (jnp.asarray(state.merge, dtype=jnp.float32)
                     if hasattr(state, 'merge')
                     else jnp.asarray(jnp.nan, dtype=jnp.float32))), cen

        def chunk(carry, inp):
            chunk_tasks, chunk_gens = inp
            carry, every = jax.lax.scan(step, carry, chunk_tasks)
            stats, cen = measure(carry[0], chunk_tasks[-1], chunk_gens[-1])
            # The full centroid every `record_stride` is d floats a record
            # and grows with every sigma and level, so only on request.
            return carry, (every, stats,
                           cen if args.record_centroid else cen[:0])

        _, (every, thin, cen) = jax.lax.scan(chunk, (state, key), (tasks, gens))
        every = tuple(a.reshape((-1,) + a.shape[2:]) for a in every)
        return every, thin, cen

    run = jax.jit(jax.vmap(jax.vmap(one_run, in_axes=(None, 0)),
                           in_axes=(0, None)))
    (g, train, cover, path, cpath), thin, cen = run(
        jnp.asarray(args.levels, dtype=jnp.float32),
        jnp.arange(args.num_seeds))

    g, cover = np.asarray(g), np.asarray(cover)
    threshold = args.threshold * land.peak
    hit = g >= threshold
    found = hit.any(axis=-1)
    thin = {k: np.asarray(v, dtype=np.float32)
            for k, v in zip(RECORD_KINDS, thin)}
    summary = dict(
        found=found, held=g[..., -1] >= threshold,
        cover=(cover >= threshold).any(axis=-1),
        cover_held=cover[..., -1] >= threshold,
        first=np.where(found, hit.argmax(axis=-1), args.num_generations),
        final_g=g[..., -1], retention=retention_of(hit),
        **{f'end_{k}': v[..., -1] for k, v in thin.items()})
    series = dict(g=g.astype(np.float32),
                  train=np.asarray(train, dtype=np.float32), **thin)
    trails = dict(path=np.asarray(path, dtype=np.float32),
                  cpath=np.asarray(cpath, dtype=np.float32))
    if args.record_centroid:
        # (levels, seeds, records, d): the centroid at every record, in full.
        series['cen'] = np.asarray(cen, dtype=np.float32)
    return summary, series, trails


def _mean(values):
    """A JSON-safe mean over seeds: None where nothing is finite."""
    values = np.asarray(values, dtype=np.float64)
    finite = values[np.isfinite(values)]
    return float(finite.mean()) if finite.size else None


def sweep(land, args):
    grid, series, trails = {}, {}, {}
    for method in args.methods:
        for sigma in args.sigmas:
            t0 = time.time()
            summary, cell_series, cell_trails = run_cell(land, method, sigma,
                                                         args)
            for i, level in enumerate(args.levels):
                key = (method, sigma, level)
                grid[key] = {k: v[i] for k, v in summary.items()}
                series[key] = {k: v[i] for k, v in cell_series.items()}
                trails[key] = {k: v[i] for k, v in cell_trails.items()}
            print(f'  {method:<15} sigma {sigma:<5g} {time.time() - t0:6.1f}s',
                  flush=True)

    # Best sigma per arm: the highest mean discovery rate over every level,
    # ties broken on held, then cover, then the LARGER sigma -- an arm that
    # never finds anything must not land on whichever sigma is listed first.
    best_sigma = {}
    for method in args.methods:
        def rank(s, method=method):
            return tuple(np.mean([grid[(method, s, x)][k].mean()
                                  for x in args.levels])
                         for k in ('found', 'held', 'cover')) + (s,)
        best_sigma[method] = max(args.sigmas, key=rank)
    return grid, series, trails, best_sigma


def write(land, args, grid, series, trails, best_sigma):
    record = dict(
        landscape=land.name, landscape_options=land.options,
        level_name=land.level_name, level_label=land.level_label,
        peak=land.peak, num_tasks=land.num_tasks, config=vars(args),
        best_sigma=best_sigma,
        record_gens=list(range(args.record_stride - 1, args.num_generations,
                               args.record_stride)),
        cells=[dict(method=m, sigma=s, level=float(x),
                    found=float(r['found'].mean()),
                    held=float(r['held'].mean()),
                    cover=float(r['cover'].mean()),
                    cover_held=float(r['cover_held'].mean()),
                    retention=_mean(r['retention']),
                    first=[int(v) for v in r['first']],
                    final_generalist=[_mean([v]) for v in r['final_g']],
                    **{f'end_{k}': _mean(r[f'end_{k}']) for k in RECORD_KINDS})
               for (m, s, x), r in grid.items()])
    out_json = os.path.join(args.out_dir, 'results.json')
    with open(out_json, 'w') as f:
        json.dump(record, f, indent=1)
    print(f'wrote {out_json}')

    # Every per-generation series for every cell; the 2-D paths only at each
    # arm's best sigma, which is all the landscape panels draw.
    arrays = {f'{kind}|{m}|{s:g}|{x:g}': v
              for (m, s, x), cell in series.items() for kind, v in cell.items()}
    arrays.update({f'{kind}|{m}|{s:g}|{x:g}': v
                   for (m, s, x), cell in trails.items()
                   if s == best_sigma[m] for kind, v in cell.items()})
    out_npz = os.path.join(args.out_dir, 'curves.npz')
    np.savez_compressed(out_npz, **arrays)
    print(f'wrote {out_npz} ({os.path.getsize(out_npz) / 1e6:.1f} MB)')


def print_table(land, args, grid, best_sigma):
    print(f'\n{land.name}: {args.num_seeds} seeds, {args.num_generations} '
          f'generations, pop {args.pop_size}, d = {args.num_params}, switch '
          f'every {args.task_interval if land.num_tasks > 1 else "never"}\n')
    print(f"{'method':<15}{'sigma':>6}{'level':>7}{'found':>7}{'held':>6}"
          f"{'retain':>7}{'C-M':>8}{'E-C':>7}{'|c-e|':>9}{'minor':>7}"
          f"{'q':>6}{'wide':>6}")
    for method in args.methods:
        s = best_sigma[method]
        for x in args.levels:
            r = grid[(method, s, x)]
            ret = _mean(r['retention'])
            vals = [_mean(r['end_cen_g']), _mean(r['end_mem_g']),
                    _mean(r['final_g']), _mean(r['end_dist']),
                    _mean(r['end_minority']), _mean(r['end_q']),
                    _mean(r['end_wide'])]
            cen, mem, eli, dist, minor, q, wide = [
                np.nan if v is None else v for v in vals]
            print(f'{method:<15}{s:>6g}{x:>7g}{r["found"].mean():>7.2f}'
                  f'{r["held"].mean():>6.2f}'
                  f'{np.nan if ret is None else ret:>7.2f}'
                  f'{cen - mem:>8.3f}{eli - cen:>7.3f} {dist:>8.3g}'
                  f'{minor:>7.2f}{q:>6.2f}{wide:>6.0f}')
        print()
    print('C-M centroid score minus mean member score (< 0: several basins)\n'
          'E-C incumbent score minus centroid score\n'
          '|c-e| distance between centroid and incumbent\n'
          'minor share of members outside the most common basin\n'
          'q     share of the incumbent\'s gaussian children that stay as good\n'
          'wide  population directions wider than wide_factor * sigma^2')


def project(land, args):
    """Full-dimensional population snapshots for one (method, sigma, level).

    The sweep keeps only the first two coordinates of the incumbent and the
    centroid; a projection of a d-dimensional population needs every member's
    every coordinate. Snapshots are geometric in generation, so the early
    split and the late state are both there, and `--project_seeds` seeds are
    kept.
    """
    method, sigma, level = args.project[0], float(args.project[1]), \
        float(args.project[2])
    if method not in BASE:
        raise SystemExit(f'unknown arm {method!r}')
    searcher = build(method, sigma, args)
    gens = np.unique(np.geomspace(
        1, args.num_generations, args.num_snapshots).round().astype(int))
    tasks = np.asarray(schedule(land, args))

    @jax.jit
    def advance(state, key, chunk, level):
        (state, key), _ = jax.lax.scan(make_step(land, searcher, level),
                                       (state, key), chunk)
        return state, key

    rec = {k: [] for k in ('members', 'elite', 'centroid', 'member_score')}
    for seed in range(args.project_seeds):
        state, key = init_run(land, searcher, args, seed)
        done, per = 0, {k: [] for k in rec}
        for g in gens:
            state, key = advance(state, key, jnp.asarray(tasks[done:g]), level)
            done = g
            members = searcher.population(state)
            per['members'].append(np.asarray(members, np.float32))
            per['elite'].append(np.asarray(searcher.incumbent(state),
                                           np.float32))
            per['centroid'].append(np.asarray(searcher.population_mean(state),
                                              np.float32))
            per['member_score'].append(np.asarray(
                land.generalist_score(members, level), np.float32))
        for k in rec:
            rec[k].append(np.stack(per[k]))
    meta = dict(landscape=land.name, landscape_options=land.options,
                method=method, sigma=sigma, level=level,
                num_params=args.num_params, pop_size=args.pop_size,
                num_generations=args.num_generations)
    out = os.path.join(args.out_dir,
                       f'project_{method}_s{sigma:g}_{land.level_name}'
                       f'{level:g}.npz')
    np.savez_compressed(out, gens=gens, meta=json.dumps(meta),
                        **{k: np.stack(v) for k, v in rec.items()})
    print(f'wrote {out} ({os.path.getsize(out) / 1e6:.1f} MB)')


def build_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--landscape', required=True, choices=toy_landscapes.NAMES)
    p.add_argument('--levels', type=float, nargs='+', default=None,
                   help="the knob's values; default: the landscape's own")
    p.add_argument('--methods', nargs='+', default=list(DEFAULT_METHODS),
                   choices=sorted(BASE))
    p.add_argument('--sigmas', type=float, nargs='+',
                   default=[0.05, 0.1, 0.15, 0.2],
                   help='swept per arm; tables and figures use each arm at its '
                        'best sigma, the json keeps every one')
    p.add_argument('--lr_scale', type=float, default=0.05,
                   help='the parameter-space step the distribution arms are '
                        'matched to; keep it well under the smallest feature '
                        'of the landscape')
    p.add_argument('--num_params', type=int, default=2,
                   help='coordinates past the landscape\'s relevant ones are '
                        'null directions')
    p.add_argument('--num_generations', type=int, default=None)
    p.add_argument('--task_interval', type=int, default=None,
                   help='generations per sub-task phase (switching toys)')
    p.add_argument('--pop_size', type=int, default=None,
                   help='evaluations per generation, for every arm')
    p.add_argument('--num_seeds', type=int, default=24)
    p.add_argument('--start_jitter', type=float, default=None,
                   help='std of the start point\'s jitter (0.05; the manifold '
                        'landscape uses a tenth of its tube width)')
    p.add_argument('--init_scale', type=float, default=None,
                   help='GA/DNS initial scatter around the start (0.1, the '
                        'searchers\' own default; manifold: tube width / 10)')
    p.add_argument('--threshold', type=float, default=0.98,
                   help='fraction of the peak generalist score that counts '
                        'as a generalist')
    p.add_argument('--elite_ratio', type=float, default=0.5)
    p.add_argument('--line_sigma', type=float, default=0.5,
                   help='Iso+LineDD line width for dns and ga_isoline (the '
                        'gymnax arms\' value); dns_corrected uses 0.05')
    p.add_argument('--dns_k', type=int, default=3)
    p.add_argument('--track_rate', type=float, default=0.1,
                   help='ga_track: log-sigma step per generation per unit of '
                        '(centroid rank - target)')
    p.add_argument('--track_target', type=float, default=0.5,
                   help='ga_track: the share of the archive the centroid should '
                        'score at least as well as')
    p.add_argument('--merge_rate', type=float, default=0.02,
                   help='ga_merge_*: step of the merge level per generation per '
                        'unit of (0.5 - centroid rank)')
    p.add_argument('--merge_max', type=float, default=None,
                   help='ga_merge_*: ceiling of the merge level (noise: 1, '
                        'pc: 0.5)')
    p.add_argument('--sigma_rate', type=float, default=0.3,
                   help='ga_merge_track: log-sigma step per generation per unit '
                        'of (centroid rank - 0.5)')
    p.add_argument('--combine', default=None, choices=('mean', 'min'),
                   help='wells: additive (mean, default) or weakest-coordinate '
                        '(min) pooling of the per-coordinate wells')
    p.add_argument('--survive_target', type=float, default=0.25,
                   help='ga_subspace / ga_survive: the share of a group\'s '
                        'offspring that should survive into the archive')
    p.add_argument('--sigma_sub_init', type=float, default=0.5,
                   help='ga_subspace: starting width along the archive spread')
    p.add_argument('--subspace_fraction', type=float, default=0.5,
                   help='ga_subspace: share of offspring bred along the spread')
    p.add_argument('--success_target', type=float, default=0.2,
                   help='ga_success: the share of offspring that should score '
                        'at least as well as the median elite')
    p.add_argument('--num_centroids', type=int, default=8)
    p.add_argument('--restart_interval', type=int, default=50)
    p.add_argument('--centroid_spread', type=float, default=0.6)
    p.add_argument('--wavelength', type=float, default=None,
                   help='rugged: ripple wavelength (0.6)')
    p.add_argument('--specialist_peak', type=float, default=None,
                   help='rugged: specialist height (0.65; the generalist is 0.70)')
    p.add_argument('--rugged_dims', type=int, default=None,
                   help='rugged: coordinates the ripple digs pits along (2)')
    p.add_argument('--relevant_dims', type=int, default=None,
                   help='wells: how many coordinates have two wells (2)')
    p.add_argument('--stiff_dims', type=int, default=None,
                   help='manifold: coordinates that must stay near 0 (0)')
    p.add_argument('--tube_width', type=float, default=None,
                   help='manifold: how near, the gaussian width w (0.1)')
    p.add_argument('--manifold_base', default=None, choices=('rugged', 'smooth'),
                   help='manifold: the landscape inside the tube (rugged)')
    p.add_argument('--record_stride', type=int, default=10,
                   help='generations between centroid records')
    p.add_argument('--record_centroid', action='store_true',
                   help='also save the FULL centroid at every record, for '
                        'every sigma (curves.npz key cen|method|sigma|level, '
                        'shape (seeds, records, d)); the basin-width figure '
                        'measures the landscape around it')
    p.add_argument('--q_samples', type=int, default=64,
                   help='children of the incumbent sampled to measure q')
    p.add_argument('--q_tolerance', type=float, default=0.02,
                   help='a child "stays as good" within this fraction of the '
                        'peak score')
    p.add_argument('--q_seed', type=int, default=7)
    p.add_argument('--wide_factor', type=float, default=4.0,
                   help='a population direction counts as wide above this '
                        'many one-generation mutation variances')
    p.add_argument('--out_dir', default=None,
                   help='default: projects/iclr_2027/runs_toy/<landscape>')
    p.add_argument('--project', nargs=3, metavar=('METHOD', 'SIGMA', 'LEVEL'),
                   help='instead of the sweep, save full-dimensional population '
                        'snapshots of one cell')
    p.add_argument('--project_seeds', type=int, default=2)
    p.add_argument('--num_snapshots', type=int, default=20)
    return p


def resolve(args):
    land = toy_landscapes.get(args.landscape, wavelength=args.wavelength,
                              specialist_peak=args.specialist_peak,
                              rugged_dims=args.rugged_dims,
                              relevant_dims=args.relevant_dims,
                              combine=args.combine,
                              stiff_dims=args.stiff_dims,
                              tube_width=args.tube_width,
                              manifold_base=args.manifold_base)
    for key, value in land.defaults.items():
        if getattr(args, key) is None:
            setattr(args, key, value)
    if args.start_jitter is None:
        args.start_jitter = 0.05
    if args.init_scale is None:
        args.init_scale = 0.1
    if args.num_generations is None:
        args.num_generations = 1000
    if args.pop_size is None:
        args.pop_size = 64
    if args.task_interval is None:
        args.task_interval = args.num_generations
    if args.levels is None:
        args.levels = list(land.levels)
    args.report_levels = [x for x in land.report_levels if x in args.levels] \
        or list(args.levels)
    if args.out_dir is None:
        args.out_dir = os.path.join('projects', 'iclr_2027', 'runs_toy',
                                    land.name)
    if args.num_generations % args.record_stride:
        raise SystemExit('--num_generations must be a multiple of '
                         '--record_stride')
    if args.num_params < land.relevant:
        raise SystemExit(f'--num_params must be at least {land.relevant}')
    args.landscape_options = land.options
    return land


def main():
    args = build_parser().parse_args()
    land = resolve(args)
    os.makedirs(args.out_dir, exist_ok=True)
    if args.project:
        project(land, args)
        return
    grid, series, trails, best_sigma = sweep(land, args)
    print_table(land, args, grid, best_sigma)
    write(land, args, grid, series, trails, best_sigma)


if __name__ == '__main__':
    main()
