"""What the fitness landscape looks like around a GA's elite and its centroid.

    .venv/bin/python scripts/analysis/landscape_probe.py checkpoints \\
        projects/iclr_2027/runs_centroid/gymnax/continual --methods ga dns_gaussian
    .venv/bin/python scripts/analysis/landscape_probe.py populations \\
        projects/generalists/runs --methods ga dns
    .venv/bin/python scripts/analysis/landscape_probe.py plot

The question (2026-09-13). A GA's centroid -- the coordinate-wise mean of its
archive -- sits several mutation lengths from its elite on every body, yet on
CartPole and Acrobot it scores what the elite scores and on MountainCar and
Kinetix far less. Three measurements separate "the population is spread along
directions that do not change the score" from "the population sits in several
basins":

  line      the score along elite -> centroid, t in [-0.5, 1.5]. A dip between
            0 and 1 is a barrier between them; flat means one basin or a
            neutral direction.
  radial    the mean score at distance r from the elite along random
            directions, r in units of one mutation (sigma * sqrt(d)). Where it
            falls off is the width of the elite's peak at the scale the GA
            searches; the observed |centroid - elite| is marked.
  barriers  (populations only) for pairs of the best members,
            min(f_i, f_j) - f(midpoint). Members joined by a barrier below
            `--tau` of the score span are one basin; the number of groups is
            how many basins the population occupies, and centroid minus mean
            member score says whether averaging them costs anything.

`source/studies/toy/sweep.py` measures the same quantities on the wells
landscape, where the answer is known; this is the same reading taken on the
real bodies.

Every score uses FIXED evaluation seeds (common random numbers,
`make_fixed_seed_scoring_fn`'s convention), so two nearby points differ because
the genomes differ, not because they drew different resets.

Only gymnax so far. The shared-runner bodies go through
`source/envs/run_context.RunContext.returns_of` and are the next step, with
two caveats written down here so they are not rediscovered: a Kinetix probe
must run on GH200 (CLAUDE.md, "Kinetix post-hoc"), and a rebuilt cheetah
re-measures its whitening statistics (none were recorded before 2026-09-13),
which moves its level but not a comparison between nearby points.

Populations: no `projects/iclr_2027` run saved one. The `*_2task_movie` runs of
the generalists study did (128 members every 20 generations, the same shared
GA/DNS searchers), so `populations` reads those.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import pathlib
import sys
import time

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'scripts'))

DEFAULT_OUT = 'projects/iclr_2027/figures/landscape_probe'
ENV_ORDER = ['CartPole-v1', 'Acrobot-v1', 'MountainCar-v0']
LINE_T = np.linspace(-0.5, 1.5, 41)
E_IDX = int(np.argmin(np.abs(LINE_T)))          # t = 0, the elite
C_IDX = int(np.argmin(np.abs(LINE_T - 1.0)))    # t = 1, the centroid
RADII = np.geomspace(0.1, 10.0, 9)       # in units of sigma * sqrt(d)
NUM_DIRECTIONS = 8


class Scorer:
    """`scorer(genomes (N, P), offset, env_params) -> (N,)` on fixed seeds."""

    def __init__(self, env_name, hidden_dims, episode_length, episodes,
                 task_type='noise', eval_seed=0, max_chunk=4096):
        import gymnax
        import jax
        from jax import random

        from source.envs.gymnax_classic import (
    make_gymnax_env, wrap_actions,
            FlipEnv, build_policy, make_episode_fn)

        env, stock = make_gymnax_env(env_name)
        stock = stock.replace(max_steps_in_episode=episode_length)
        obs, _ = env.reset(random.key(0), stock)
        action_dim = env.action_space(stock).n
        if task_type == 'actions':        # after the spaces are read, as in training
            env = wrap_actions(env)
        _policy, _tmpl, self.num_params = build_policy(
            random.key(0), obs.shape[-1], action_dim, hidden_dims)
        episode = make_episode_fn(env, _policy, _tmpl, episode_length)
        keys = random.split(random.key(eval_seed), episodes)
        self.stock, self.max_chunk = stock, max_chunk

        @jax.jit
        def batch(genomes, offset, params):
            per = jax.vmap(lambda k: jax.vmap(
                episode, in_axes=(0, None, None, None))(
                    genomes, k, offset, params))(keys)
            return per.mean(axis=0)
        self._batch = batch

    def __call__(self, genomes, offset, params):
        import jax.numpy as jnp
        genomes = np.asarray(genomes, np.float32)
        out = []
        for i in range(0, len(genomes), self.max_chunk):
            part = genomes[i:i + self.max_chunk]
            # Padded to a power of two, so a handful of shapes are compiled
            # rather than one per call.
            size = min(self.max_chunk, 1 << max(6, (len(part) - 1).bit_length()))
            padded = np.concatenate(
                [part, np.repeat(part[-1:], size - len(part), axis=0)])
            out.append(np.asarray(self._batch(
                jnp.asarray(padded), jnp.asarray(offset, dtype=jnp.float32),
                params))[:len(part)])
        return np.concatenate(out)


class SuiteScorer:
    """The same call for a shared-runner body (mjx, kinetix), via RunContext.

    `task` is a row of the run's `noise_vectors`; the run's own `task` block
    says what it means. `returns_own_tasks` returns returns only, so no trace
    of visited states is kept, and the fixed key makes it common random
    numbers like the gymnax path.
    """

    stock = None

    def __init__(self, cfg, episodes, eval_seed=0):
        from jax import random

        from source.envs.run_context import RunContext
        from source.envs.registry import suite_for
        self.ctx = RunContext(cfg, episodes)
        self.key = random.key(eval_seed)
        # A Kinetix network is 1.1M weights and its rollout state is a physics
        # scene; the suite's own scoring functions batch 32 at a time.
        self.chunk = 16 if suite_for(cfg['env']) == 'kinetix' else 64

    def __call__(self, genomes, task, params=None):
        import jax.numpy as jnp
        genomes = np.asarray(genomes, np.float32)
        tasks = jnp.asarray(np.repeat(np.asarray(task, np.float32)[None],
                                      self.chunk, axis=0))
        out = []
        for i in range(0, len(genomes), self.chunk):
            part = genomes[i:i + self.chunk]
            padded = np.concatenate(
                [part, np.repeat(part[-1:], self.chunk - len(part), axis=0)])
            ret = self.ctx.returns_own_tasks(jnp.asarray(padded), self.key,
                                             tasks)
            out.append(np.asarray(ret).mean(axis=1)[:len(part)])
        return np.concatenate(out)


_SCORERS = {}


def suite_of(cfg):
    from source.envs.registry import suite_for
    return suite_for(cfg['env'])


def scorer_for(cfg, episodes):
    if suite_of(cfg) != 'gymnax':
        from source.envs.run_context import RunContext
        key = RunContext.cache_key(cfg, episodes)
        if key not in _SCORERS:
            _SCORERS[key] = SuiteScorer(cfg, episodes)
        return _SCORERS[key]
    key = (cfg['env'], tuple(cfg['hidden_dims']), int(cfg['episode_length']),
           episodes, cfg.get('task_type') or 'noise')
    if key not in _SCORERS:
        _SCORERS[key] = Scorer(*key)
    return _SCORERS[key]


def span_of(cfg):
    """What a score change is divided by, to put bodies on one axis.

    One episode length on gymnax (every return lies within it of 0) and on the
    mjx bodies (dm_control rewards are per-step in [0, 1]); 1 on Kinetix,
    whose returns sit in about [-1, 1.5].
    """
    return 1.0 if suite_of(cfg) == 'kinetix' else float(cfg['episode_length'])


def group_of(env):
    """Kinetix levels are drawn as one family; everything else by name."""
    return 'Kinetix (20 levels)' if env.startswith('Kinetix') else env


def mutation_scale(blob):
    """The run's gaussian mutation width, from whichever field its trainer used."""
    for src in (blob.get('config') or {}, blob):
        for k in ('mutation_std', 'sigma', 'sigma_init'):
            v = src.get(k)
            if isinstance(v, (int, float)) and v > 0:
                return float(v)
    return None


def probe_pair(scorer, elite, centroid, sigma, offset, params, seed=0,
               directions=None):
    """The line and radial profiles around one elite.

    `directions` (k, d), if given, are probed at the same radii as the random
    ones -- the population's own principal axes, so a body where the elite is
    sharp in random directions but flat along the population's spread shows it.
    """
    d = elite.size
    dist = float(np.linalg.norm(centroid - elite))
    unit = sigma * np.sqrt(d) if sigma else max(dist, 1e-9)
    rng = np.random.default_rng(seed)
    dirs = rng.standard_normal((NUM_DIRECTIONS, d))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    sets = [dirs] if directions is None else [dirs, directions]
    line = elite[None] + LINE_T[:, None] * (centroid - elite)[None]
    radial = [(elite[None, None] + (RADII[:, None, None] * unit) * s[None])
              .reshape(-1, d) for s in sets]
    scores = scorer(np.concatenate([line] + radial), offset, params)
    out = dict(line=scores[:len(LINE_T)].tolist(), unit=float(unit),
               dist=dist, sigma=sigma, d=int(d))
    start = len(LINE_T)
    for name, s in zip(('radial', 'radial_pc'), sets):
        n = len(RADII) * len(s)
        out[name] = scores[start:start + n].reshape(len(RADII), len(s)).tolist()
        start += n
    return out


def components(adjacent):
    """Connected-component labels of a boolean adjacency matrix."""
    n = len(adjacent)
    labels = -np.ones(n, dtype=int)
    for start in range(n):
        if labels[start] >= 0:
            continue
        stack, labels[start] = [start], start
        while stack:
            i = stack.pop()
            for j in np.flatnonzero(adjacent[i] & (labels < 0)):
                labels[j] = start
                stack.append(j)
    return labels


def cmd_checkpoints(args):
    from source.envs.gymnax_classic import saved_task_rows
    from source.envs.run_context import run_config

    rows, suites = [], set()
    for method in args.methods:
        pattern = os.path.join(args.root, method, '*', 'trial_*', 'results.json')
        for res in sorted(glob.glob(pattern)):
            trial = os.path.dirname(res)
            ck_path = os.path.join(trial, 'checkpoints.npz')
            if not os.path.exists(ck_path):
                continue
            ckpt = np.load(ck_path)
            if 'centroid' not in ckpt.files or 'incumbent' not in ckpt.files:
                continue
            blob = json.load(open(res))
            cfg = run_config(blob)
            t0 = time.time()
            suites.add(suite_of(cfg))
            scorer = scorer_for(cfg, args.episodes)
            if suite_of(cfg) == 'gymnax':
                offsets, bodies, _ = saved_task_rows(cfg, ckpt, scorer.stock)
            else:
                offsets = np.asarray(ckpt['noise_vectors'], dtype=np.float32)
                bodies = [None] * len(offsets)
            phases = (range(len(offsets)) if args.all_phases
                      else [len(offsets) - 1])
            for t in phases:
                e = ckpt['incumbent'][t].astype(np.float64)
                c = ckpt['centroid'][t].astype(np.float64)
                if not (np.isfinite(e).all() and np.isfinite(c).all()):
                    continue
                r = probe_pair(scorer, e, c, mutation_scale(blob), offsets[t],
                               bodies[t], seed=t)
                rows.append(dict(method=method, env=cfg['env'],
                                 group=group_of(cfg['env']),
                                 cell=os.path.basename(os.path.dirname(trial)),
                                 trial=os.path.basename(trial), phase=int(t),
                                 span=span_of(cfg), **r))
            print(f'{method:<14}{trial[-40:]:>42}  {time.time() - t0:5.1f}s',
                  flush=True)
    # One file per suite set, so a Kinetix pass does not overwrite gymnax's;
    # `plot` reads every probe_checkpoints_*.json in --out.
    write_json(args.out, f'probe_checkpoints_{"_".join(sorted(suites))}.json',
               rows)


def cmd_populations(args):
    from source.envs.run_context import run_config

    rows = []
    for path in sorted(glob.glob(os.path.join(
            args.root, '*_2task_movie', 'switch', 'trial_*', 'trajectory.npz'))):
        trial = os.path.dirname(path)
        method = trial.split(os.sep)[-3].split('_')[1]
        if method not in args.methods:
            continue
        blob = json.load(open(os.path.join(trial, 'results.json')))
        cfg = run_config(blob)
        env = cfg['env']
        scorer = scorer_for(cfg, args.episodes)
        z = np.load(path)
        pops, gens = z['populations'], z['population_generations']
        tasks, offsets = z['population_tasks'], z['noise_vectors']
        # Every classic-control return lies within one episode length of 0.
        span = float(cfg['episode_length'])
        picks = np.unique(np.linspace(0, len(gens) - 1,
                                      args.snapshots).round().astype(int))
        t0 = time.time()
        for i in picks:
            X = pops[i].astype(np.float64)
            offset = offsets[int(tasks[i])]
            row = dict(method=method, env=env, trial=os.path.basename(trial),
                       generation=int(gens[i]), task=int(tasks[i]))
            if not np.isfinite(X).all():
                rows.append(dict(row, finite=False))
                continue
            f = scorer(X, offset, scorer.stock)
            c = X.mean(axis=0)
            fc = float(scorer(c[None], offset, scorer.stock)[0])
            order = np.argsort(-f)[:args.members]
            Y, fy = X[order], f[order]
            iu, ju = np.triu_indices(len(Y), 1)
            fm = scorer((Y[iu] + Y[ju]) / 2, offset, scorer.stock)
            barrier = np.minimum(fy[iu], fy[ju]) - fm
            B = np.zeros((len(Y), len(Y)))
            B[iu, ju] = B[ju, iu] = barrier
            labels = components(B <= args.tau * span)
            sizes = np.bincount(np.unique(labels, return_inverse=True)[1])
            pair_dist = np.linalg.norm(Y[iu] - Y[ju], axis=1)
            _, sv, vt = np.linalg.svd(X - c, full_matrices=False)
            probe = probe_pair(scorer, Y[0], c, mutation_scale(blob), offset,
                               scorer.stock, seed=int(i),
                               directions=vt[:NUM_DIRECTIONS])
            rows.append(dict(
                row, finite=True, span=span, member_scores=f.tolist(),
                pc_variance=(sv[:NUM_DIRECTIONS] ** 2
                             / max((sv ** 2).sum(), 1e-12)).tolist(),
                centroid_score=fc,
                member_mean=float(f.mean()), elite_score=float(fy[0]),
                barrier=barrier.tolist(), pair_dist=pair_dist.tolist(),
                basins=int((sizes >= 2).sum()),
                largest_basin=float(sizes.max() / len(Y)),
                singletons=int((sizes == 1).sum()), **probe))
        print(f'{method:<6}{env:<16}{len(picks)} snapshots  '
              f'{time.time() - t0:6.1f}s', flush=True)
    write_json(args.out, 'probe_populations.json', rows)


def write_json(out, name, rows):
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, name)
    with open(path, 'w') as f:
        json.dump(rows, f)
    print(f'wrote {path} ({len(rows)} rows)')


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def _style():
    from make_lineplot import METHOD_STYLE
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fs = 7.0
    plt.rcParams.update({
        'font.size': fs, 'axes.labelsize': fs, 'axes.titlesize': fs + 1,
        'xtick.labelsize': fs - 0.5, 'ytick.labelsize': fs - 0.5,
        'legend.fontsize': fs, 'axes.linewidth': 0.6})
    return plt, METHOD_STYLE


def _band(ax, x, ys, colour, label=None):
    ys = np.asarray(ys, dtype=float)
    if ys.ndim == 1:
        ys = ys[None]
    mid = np.nanmean(ys, axis=0)
    lo, hi = np.nanpercentile(ys, 25, axis=0), np.nanpercentile(ys, 75, axis=0)
    line, = ax.plot(x, mid, color=colour, lw=1.0, label=label)
    ax.fill_between(x, lo, hi, color=colour, alpha=0.18, lw=0)
    return line


def plot_checkpoints(out):
    rows = []
    for path in sorted(glob.glob(os.path.join(out, 'probe_checkpoints_*.json'))):
        rows += json.load(open(path))
    if not rows:
        return
    for r in rows:
        r.setdefault('group', group_of(r['env']))
    plt, STYLE = _style()
    present = {r['group'] for r in rows}
    envs = [e for e in ENV_ORDER if e in present] + sorted(
        present - set(ENV_ORDER))
    methods = sorted({r['method'] for r in rows})
    fig, axes = plt.subplots(2, len(envs), squeeze=False,
                             figsize=(max(6.9, 2.3 * len(envs)), 3.4),
                             constrained_layout=True)
    for col, env in enumerate(envs):
        for m in methods:
            sel = [r for r in rows if r['group'] == env and r['method'] == m]
            if not sel:
                continue
            colour = STYLE.get(m, {}).get('color')
            name = STYLE.get(m, {}).get('label', m)
            line = [(np.asarray(r['line']) - r['line'][E_IDX]) / r['span']
                    for r in sel]
            _band(axes[0, col], LINE_T, line, colour, name)
            radial = [(np.mean(r['radial'], axis=1) - r['line'][E_IDX])
                      / r['span'] for r in sel]
            _band(axes[1, col], RADII, radial, colour)
            axes[1, col].axvline(np.median([r['dist'] / r['unit'] for r in sel]),
                                 color=colour, ls=':', lw=0.9)
        ax = axes[0, col]
        for v in (0, 1):
            ax.axvline(v, color='0.6', lw=0.45, ls='--')
        ax.axhline(0, color='0.3', lw=0.4)
        ax.set_title(env, fontweight='bold')
        ax.set_xlabel('elite (0) $\\rightarrow$ centroid (1)')
        ax = axes[1, col]
        ax.set_xscale('log')
        ax.axvline(1, color='0.6', lw=0.45, ls='--')
        ax.axhline(0, color='0.3', lw=0.4)
        ax.set_xlabel('distance from elite / $\\sigma\\sqrt{d}$')
        for a in axes[:, col]:
            a.spines[['top', 'right']].set_visible(False)
    axes[0, 0].set_ylabel('score $-$ elite score\n(normalised: episode '
                          'length; 1 on Kinetix)')
    axes[1, 0].set_ylabel('random directions:\nscore $-$ elite score')
    axes[0, 0].legend(frameon=False)
    fig.suptitle('Every saved phase of every trial; mean line, IQR band. '
                 'Dotted: median observed |centroid $-$ elite|.', fontsize=6)
    _save(fig, os.path.join(out, 'probe_checkpoints'))


def plot_populations(out):
    path = os.path.join(out, 'probe_populations.json')
    if not os.path.exists(path):
        return
    rows = [r for r in json.load(open(path)) if r.get('finite')]
    # A population whose weights have run away (dns at line_sigma 0.5 reaches
    # |w| ~ 1e13) cannot be probed at the mutation scale: every radius is a
    # rounding error against the weights. Dropped, and said so.
    diverged = sorted({(r['method'], r['env']) for r in rows
                       if r['dist'] / r['unit'] > 1e3})
    rows = [r for r in rows if (r['method'], r['env']) not in diverged]
    if diverged:
        print(f'not drawn, weights diverged: {diverged}')
    plt, STYLE = _style()
    envs = [e for e in ENV_ORDER if any(r['env'] == e for r in rows)]
    methods = sorted({r['method'] for r in rows})
    fig, axes = plt.subplots(5, len(envs), squeeze=False, figsize=(6.9, 7.6),
                             constrained_layout=True)
    for col, env in enumerate(envs):
        for m in methods:
            sel = sorted([r for r in rows if r['env'] == env
                          and r['method'] == m], key=lambda r: r['generation'])
            if not sel:
                continue
            span = sel[0]['span']
            colour = STYLE.get(m, {}).get('color')
            name = STYLE.get(m, {}).get('label', m)
            g = [r['generation'] for r in sel]
            axes[0, col].plot(g, [(r['centroid_score'] - r['member_mean']) / span
                                  for r in sel], 'o-', color=colour, ms=2.5,
                              lw=0.9, label=name)
            axes[0, col].plot(g, [(r['elite_score'] - r['centroid_score']) / span
                                  for r in sel], 's--', color=colour, ms=2,
                              lw=0.7)
            axes[1, col].plot(g, [r['basins'] + r['singletons'] for r in sel],
                              'o-', color=colour, ms=2.5, lw=0.9)
            axes[1, col].plot(g, [r['largest_basin'] * 10 for r in sel], ':',
                              color=colour, lw=0.8)
            last = sel[-1]
            axes[2, col].hist(np.asarray(last['barrier']) / span, bins=40,
                              color=colour, alpha=0.5, density=True)
            axes[3, col].plot(LINE_T, (np.asarray(last['line'])
                                       - last['line'][E_IDX]) / span,
                              color=colour, lw=1.0)
            # Late snapshots (second half of the run): random directions
            # solid, the population's own principal axes dashed.
            late = [r for r in sel if r['generation'] >= g[-1] / 2]
            for key, ls in (('radial', '-'), ('radial_pc', '--')):
                ys = [(np.mean(r[key], axis=1) - r['line'][E_IDX]) / span
                      for r in late if key in r]
                if ys:
                    axes[4, col].plot(RADII, np.mean(ys, axis=0), ls,
                                      color=colour, lw=1.0)
        axes[0, col].axhline(0, color='0.3', lw=0.4)
        axes[0, col].set_title(env, fontweight='bold')
        axes[0, col].set_xlabel('Generations')
        axes[1, col].set_xlabel('Generations')
        axes[2, col].set_xlabel('midpoint barrier, final snapshot')
        axes[3, col].set_xlabel('elite (0) $\\rightarrow$ centroid (1)')
        axes[4, col].set_xscale('log')
        axes[4, col].axhline(0, color='0.3', lw=0.4)
        axes[4, col].set_xlabel('distance from elite / $\\sigma\\sqrt{d}$')
        for v in (0, 1):
            axes[3, col].axvline(v, color='0.6', lw=0.45, ls='--')
        for a in axes[:, col]:
            a.spines[['top', 'right']].set_visible(False)
    axes[0, 0].set_ylabel('solid: centroid $-$ member mean\n'
                          'dashed: elite $-$ centroid')
    axes[1, 0].set_ylabel('basins among best members\n(dotted: largest share x10)')
    axes[2, 0].set_ylabel('density')
    axes[3, 0].set_ylabel('score $-$ elite score')
    axes[4, 0].set_ylabel('solid: random directions\n'
                          'dashed: population PCs')
    axes[0, 0].legend(frameon=False)
    _save(fig, os.path.join(out, 'probe_populations'))


def _save(fig, stem):
    import matplotlib.pyplot as plt
    fig.savefig(stem + '.png', dpi=200, bbox_inches='tight')
    fig.savefig(stem + '.pdf', bbox_inches='tight')
    plt.close(fig)
    print(f'wrote {stem}.png + .pdf')


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='cmd', required=True)
    for name in ('checkpoints', 'populations'):
        s = sub.add_parser(name)
        s.add_argument('root')
        s.add_argument('--methods', nargs='+', default=['ga', 'dns_gaussian']
                       if name == 'checkpoints' else ['ga', 'dns'])
        s.add_argument('--episodes', type=int, default=3,
                       help="evaluation episodes per point (training's num_evals)")
        s.add_argument('--out', default=DEFAULT_OUT)
        if name == 'checkpoints':
            s.add_argument('--all_phases', action='store_true',
                           help='probe every saved phase, not only the last')
        else:
            s.add_argument('--snapshots', type=int, default=6)
            s.add_argument('--members', type=int, default=64,
                           help='the best this many members enter the barrier '
                                'matrix (all pairs)')
            s.add_argument('--tau', type=float, default=0.05,
                           help='barrier below this fraction of the return span '
                                'joins two members into one basin')
    s = sub.add_parser('plot')
    s.add_argument('--out', default=DEFAULT_OUT)
    args = p.parse_args()

    if args.cmd == 'checkpoints':
        cmd_checkpoints(args)
        plot_checkpoints(args.out)
    elif args.cmd == 'populations':
        cmd_populations(args)
        plot_populations(args.out)
    else:
        plot_checkpoints(args.out)
        plot_populations(args.out)


if __name__ == '__main__':
    main()
