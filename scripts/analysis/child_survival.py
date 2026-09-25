"""How many of a saved agent's gaussian children stay as good: the toy's q on the real tasks.

    .venv/bin/python scripts/analysis/child_survival.py probe --panels cartpole_noise acrobot_noise \\
        mountaincar_noise --gpus 0
    .venv/bin/python scripts/analysis/child_survival.py probe --panels cheetah_noise --phases 1::2 --gpus 4
    .venv/bin/python scripts/analysis/child_survival.py table

The question (2026-09-17). On the rugged toy (`source/studies/toy/sweep.py`)
a GA stops finding the generalist when the ruggedness spans many coordinates,
because an isotropic gaussian child has to land well on every rugged coordinate
at once. The toy reads this as `q`: the share of the incumbent's gaussian
children, at the arm's sigma, whose score is at least the incumbent's minus
`q_tolerance` (0.02) x the landscape's peak. Its toy scores are deterministic,
so a child is compared with the incumbent's exact score.

The same reading on the paper's continual_main tasks
(`plot_continual_lineplots.PANELS`, `main`):

  parent    the saved agent at every plasticity checkpoint (end of each
            sub-task phase, `checkpoints.npz`): the GA's `incumbent`, the ES's
            `centroid` (the network ES perturbs). ES = NES except on Kinetix,
            where it is the OpenES arm (`es` -> es_hold), as CLAUDE.md says.
  width     the run's OWN mutation width at that checkpoint: the logged
            `sigma` column at the phase's last record where the run logs one,
            else the config (`mutation_std`, then `sigma`). This matters:
            the MountainCar and Kinetix GAs are `ga_focus`, whose width
            shrinks from its configured 0.5 to ~1e-5 within the first
            sub-task, while a quarter of their offspring (`explore_fraction`)
            are always bred at the configured 0.5. Those runs are probed at
            both widths, the second reported as "GA (explorer width)".
            `landscape_probe.mutation_scale` returns the configured 0.5 for
            them.
  children  N (64) unit gaussian directions, the SAME N for every width
            multiple (0.1, 0.25, 0.5, 1, 2), so the q-against-width curve is
            not re-drawn at each point. Multiple 0 is N copies of the parent:
            the noise control.
  scoring   every point, children and parent copies alike, on its OWN fresh
            episodes (keys from the run's identity, not the training chain),
            on the sub-task the checkpoint was trained on, with the reported
            curve's episode count (10; MiniGrid 16; Kinetix 1, its levels are
            deterministic). The parent's reference score is the mean of its N
            copies: children are compared with a re-scored parent, never with
            the stored fitness, which is the maximum of a noisy selection.
  q         share of children scoring >= reference - tol, tol = 0.02 x the
            panel's score span, the span being the one stability_plasticity
            rescales by (rliable style): best mean learning accuracy of any
            reported arm minus the untrained network's return (FLOOR). The
            0x row is q for exact copies, i.e. what evaluation noise alone
            gives; a child is only "as good" in a sense the noise can resolve
            when q at a multiple is compared against that row.
  up        share of children scoring strictly above the reference: the
            improvement rate. Under noise the 0x row reads ~0.5, the same
            caveat.
  bcl       best-child loss, the quantity that predicts GA failure on the
            finished toy (paper/visuals/final/appendix/toy_ripple_dims.md):
            (reference - the best of the N children) / span, N = 64 as on the
            toy (not the runs' 512-member generations). Positive = even the
            best child is worse than the parent. The 0x row is the same max
            over N noisy copies, so its (negative) value is the noise bias of
            a max; the toy's scores are exact and have none.
  width50   the width multiple at which the mean q curve falls to 0.5
            (log-linear interpolation; "<0.1" / ">2" when the curve does not
            cross inside the grid).

Means carry a 95% percentile bootstrap interval that resamples SEEDS with all
their checkpoints (the checkpoints of one seed are not independent).

Output: `results/child_survival/raw/<panel>__<arm>.json` (one row per
checkpoint and width, every score kept, so another tolerance is a re-read, not a
re-run), and `table` writes `child_survival.json` and `child_survival.md`.
`probe` skips checkpoints already in the raw file, so a killed pass resumes.

Compute: gymnax is minutes. The MJX (cheetah) and Kinetix passes are
launch-bound and collapse on a card shared with training
(memory: mjx-posthoc-needs-free-gpus); run them on idle cards, one arm per card
(`--arms`). Kinetix NE agents re-score differently on x86 and GH200
(docs/generalists/ga_centroid_tracking.md): on the home server the parent may
not reproduce its stored score, which is logged (`stored`) beside the
re-score; q compares children with the re-scored parent on the same hardware.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import time
import zlib


def _gpu_arg():
    for i, a in enumerate(sys.argv):
        if a == '--gpus' and i + 1 < len(sys.argv):
            return sys.argv[i + 1]
    return None


if _gpu_arg() is not None:
    os.environ['CUDA_VISIBLE_DEVICES'] = _gpu_arg()
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')

import numpy as np                                             # noqa: E402

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'scripts'))
sys.path.insert(0, str(REPO / 'scripts' / 'analysis'))
PROJECT = REPO / 'projects' / 'iclr_2027'
OUT = REPO / 'results' / 'child_survival'
SP_DATA = PROJECT / 'paper/visuals/final/data/stability_plasticity.json'

# (name, tree, cell, title): continual_main's panels (plot_continual_lineplots
# PANELS, `main`). The first six are the ones the toy prediction is about.
PANELS = [
    ('cartpole_noise', 'paper/gymnax/data/noise_10task', 'CartPole_v1_sigma1.0', 'CartPole, noise'),
    ('acrobot_noise', 'paper/gymnax/data/noise_10task', 'Acrobot_v1_sigma1.0', 'Acrobot, noise'),
    ('mountaincar_noise', 'paper/gymnax/data/noise_10task', 'MountainCar_v0_sigma0.1', 'MountainCar, noise'),
    ('cheetah_noise', 'paper/mjx/cheetah/data/noise_10task', 'cheetah_noise', 'HalfCheetah, noise'),
    ('cartpole_physics2', 'paper/gymnax/data/physics_2task', 'CartPole_v1_sigma1.0', 'CartPole, pole length'),
    ('acrobot_physics2', 'paper/gymnax/data/physics_2task', 'Acrobot_v1_sigma1.0', 'Acrobot, link mass'),
    ('mountaincar_physics2', 'paper/gymnax/data/physics_2task', 'MountainCar_v0_sigma1.0', 'MountainCar, gravity'),
    ('minigrid', 'paper/minigrid/data', 'MiniGrid_8x8_16x16', 'MiniGrid 8x8 / 16x16'),
    ('kinetix', 'paper/kinetix/data', 'Kinetix20', 'Kinetix, 20 levels'),
    ('cartpole_actions', 'paper/gymnax/data/actions_2task', 'CartPole_v1_sigma1.0', 'CartPole, action reversal'),
    ('acrobot_actions', 'paper/gymnax/data/actions_2task', 'Acrobot_v1_sigma1.0', 'Acrobot, action reversal'),
    ('mountaincar_actions', 'paper/gymnax/data/actions_2task', 'MountainCar_v0_sigma1.0', 'MountainCar, action reversal'),
    ('cheetah_actions', 'paper/mjx/cheetah/data/actions_2task', 'cheetah_action', 'HalfCheetah, action reversal'),
]
PANEL = {p[0]: p for p in PANELS}
MULTIPLES = (0.0, 0.1, 0.25, 0.5, 1.0, 2.0)
TOLERANCE = 0.02          # the toy's q_tolerance, as a share of the score span
N_BOOT = 2000


def es_dir(tree):
    return 'es' if tree.startswith('paper/kinetix') else 'nes'


def arm_dir(arm, tree):
    return es_dir(tree) if arm == 'es' else arm


# ---------------------------------------------------------------------------
# Run metadata
# ---------------------------------------------------------------------------

def phase_ends(records, num_phases):
    """Index of the last record of each phase, from the logged `task` column."""
    tasks = np.array([r.get('task', -1) for r in records])
    ends = list(np.flatnonzero(tasks[1:] != tasks[:-1])) + [len(tasks) - 1]
    if len(ends) != num_phases:
        # A run whose consecutive phases share a sub-task cannot be split by
        # the column; fall back to equal phases.
        ends = [int(round((t + 1) * len(records) / num_phases)) - 1
                for t in range(num_phases)]
    return ends


def widths(cfg, records, ends):
    """`[(base, sigma per phase, source)]`: the width(s) the run mutates at."""
    configured = None
    for k in ('mutation_std', 'sigma'):
        v = cfg.get(k)
        if isinstance(v, (int, float)) and v > 0:
            configured = float(v)
            break
    if records and 'sigma' in records[-1]:
        run = [float(records[e]['sigma']) for e in ends]
        source = 'logged sigma at phase end'
    else:
        run = [configured] * len(ends)
        source = 'config ' + ('mutation_std' if cfg.get('mutation_std') else 'sigma')
    out = [('run', run, source)]
    explore = (cfg.get('searcher_kwargs') or {}).get('explore_fraction') or 0.0
    if explore > 0:
        out.append(('explore', [configured] * len(ends),
                    f'config sigma (explore_fraction {explore})'))
    return out


def stored_score(records, end, source, task, gymnax_legacy):
    """The training record's own score of the parent on its sub-task, where logged."""
    if gymnax_legacy:
        return None
    key = f'{"incumbent" if source == "incumbent" else "centroid"}_task{task}'
    v = records[end].get(key)
    return None if v is None else float(v)


def panel_span(tree, cell):
    """(floor, best mean LA): the span stability_plasticity rescales by."""
    from plot_stability_plasticity import FLOOR
    meta = json.loads(SP_DATA.read_text())
    by_m = meta['panels'][f'{tree}|{cell}']
    best = max(float(np.mean([t['la'] for t in trials])) for trials in by_m.values())
    return FLOOR[cell.split('_sigma')[0]], best


# ---------------------------------------------------------------------------
# Scorers: `score(genomes (B, P), keys (B, 2), phase) -> (B, E)`, one fresh
# key per point.
# ---------------------------------------------------------------------------

_CACHE = {}


def gymnax_scorer(cfg, ckpt, episodes):
    import gymnax
    import jax
    import jax.numpy as jnp
    from jax import random

    from source.envs.gymnax_classic import (
    make_gymnax_env, wrap_actions,
        FlipEnv, build_policy, gymnax_task_type, make_episode_fn, saved_task_rows)

    task_type = gymnax_task_type(cfg)
    key = ('gymnax', cfg['env'], tuple(cfg['hidden_dims']),
           int(cfg['episode_length']), episodes, task_type)
    if key not in _CACHE:
        env, stock = make_gymnax_env(cfg['env'])
        stock = stock.replace(max_steps_in_episode=int(cfg['episode_length']))
        obs, _ = env.reset(random.key(0), stock)
        action_dim = env.action_space(stock).n
        if task_type == 'actions':       # after the spaces are read, as in training
            env = wrap_actions(env)
        policy, tmpl, _ = build_policy(random.key(0), obs.shape[-1], action_dim,
                                       tuple(cfg['hidden_dims']))
        episode = make_episode_fn(env, policy, tmpl, int(cfg['episode_length']))

        @jax.jit
        def batch(genomes, keys, offset, params):
            def one(g, k):
                return jax.vmap(episode, in_axes=(None, 0, None, None))(
                    g, random.split(k, episodes), offset, params)
            return jax.vmap(one)(genomes, keys)
        _CACHE[key] = (batch, stock, tmpl)
    batch, stock, tmpl = _CACHE[key]
    offsets, bodies, _ = saved_task_rows(cfg, ckpt, stock)

    def score(genomes, keys, t):
        return np.asarray(batch(jnp.asarray(genomes), keys,
                                jnp.asarray(offsets[t], dtype=jnp.float32), bodies[t]))
    score.leaf_sizes = [int(np.size(x)) for x in jax.tree_util.tree_leaves(tmpl)]
    return score, 1024


def suite_scorer(cfg, ckpt, episodes):
    import jax
    import jax.numpy as jnp

    from source.envs.registry import suite_for
    from source.envs.run_context import RunContext

    key = RunContext.cache_key(cfg, episodes)
    if key not in _CACHE:
        ctx = RunContext(cfg, episodes)
        batch = jax.jit(jax.vmap(ctx.returns_of, in_axes=(0, 0, None)))
        _CACHE.clear()          # one rebuilt MJX env at a time: they are large
        _CACHE[key] = (batch, [int(np.size(x))
                               for x in jax.tree_util.tree_leaves(ctx.template)])
    batch, leaf_sizes = _CACHE[key]
    rows = np.asarray(ckpt['noise_vectors'], dtype=np.float32)
    suite = suite_for(cfg['env'])
    chunk = {'kinetix': 16, 'mjx': 64}.get(suite, 128)

    def score(genomes, keys, t):
        return np.asarray(batch(jnp.asarray(genomes), keys, jnp.asarray(rows[t])))
    score.leaf_sizes = leaf_sizes
    return score, chunk


def default_episodes(cfg):
    from source.envs.registry import suite_for
    suite = suite_for(cfg['env'])
    if suite == 'kinetix':
        return 1
    if suite == 'minigrid':
        return int(cfg.get('eval_episodes') or 16)
    return int(cfg.get('eval_episodes') or cfg.get('report_episodes') or 10)


# ---------------------------------------------------------------------------
# probe
# ---------------------------------------------------------------------------

def probe_checkpoint(score, chunk, parent, t, sigmas, children, run_key,
                     multiples=MULTIPLES):
    """Scores `(copies (N,), {base: (M, N)})` for one checkpoint.

    Points: N parent copies, then for every base width and every non-zero
    multiple the parent plus multiple x width x the same N unit directions.
    """
    import jax
    from jax import random

    d = parent.size
    eps = np.random.default_rng(zlib.crc32(repr(run_key).encode())).standard_normal(
        (children, d)).astype(np.float32)
    plan = [('copies', 0.0, 0.0)]
    for base, sigma in sigmas.items():
        plan += [(base, m, sigma) for m in multiples if m > 0]
    total = len(plan) * children
    base_key = random.key(zlib.crc32(repr(run_key).encode()) % (2 ** 31))
    keys_all = jax.vmap(lambda i: random.fold_in(base_key, i))(np.arange(total))
    scores = np.empty(total, dtype=np.float64)
    buf, idx = [], []

    def flush():
        n = len(buf)
        g = np.stack(buf + [buf[-1]] * (chunk - n))
        k = keys_all[np.array(idx + [idx[-1]] * (chunk - n))]
        out = score(g, k, t)[:n]
        scores[idx] = out.mean(axis=1)
        buf.clear()
        idx.clear()

    i = 0
    for _name, m, sigma in plan:
        for j in range(children):
            buf.append(parent + np.float32(m * sigma) * eps[j] if m > 0 else parent)
            idx.append(i)
            i += 1
            if len(buf) == chunk:
                flush()
    if buf:
        flush()
    copies = scores[:children]
    out, start = {}, children
    for base in sigmas:
        n_m = sum(1 for m in multiples if m > 0)
        out[base] = scores[start:start + n_m * children].reshape(n_m, children)
        start += n_m * children
    return copies, out


def cmd_probe(args):
    from source.envs.run_context import is_gymnax_run, run_config

    OUT.joinpath('raw').mkdir(parents=True, exist_ok=True)
    for name in args.panels:
        _, tree, cell, title = PANEL[name]
        floor, best = panel_span(tree, cell)
        span = best - floor
        for arm in args.arms:
            adir = PROJECT / tree / 'continual' / arm_dir(arm, tree)
            source = 'incumbent' if arm == 'ga' else 'centroid'
            suffix = '' if args.shard is None else '__shard' + args.shard.replace('/', 'of')
            raw_path = OUT / 'raw' / f'{name}__{arm}{suffix}.json'
            rows = json.loads(raw_path.read_text()) if raw_path.exists() else []
            # Done in ANY raw file of this panel and arm, so shards and an
            # unsharded pass never repeat a checkpoint.
            done = {(r['trial'], r['phase'])
                    for path in OUT.joinpath('raw').glob(f'{name}__{arm}.json')
                    for r in json.loads(path.read_text())}
            done |= {(r['trial'], r['phase'])
                     for path in OUT.joinpath('raw').glob(f'{name}__{arm}__shard*.json')
                     for r in json.loads(path.read_text())}
            trials = sorted((adir / cell).glob('trial_*'),
                            key=lambda p: int(p.name.split('_')[1]))
            if args.trials:
                trials = trials[:args.trials]
            if args.shard is not None:
                i, n = map(int, args.shard.split('/'))
                trials = trials[i::n]
            for trial in trials:
                ck_path = trial / 'checkpoints.npz'
                if not ck_path.exists():
                    print(f'SKIP {trial}: no checkpoints.npz')
                    continue
                ckpt = np.load(ck_path)
                cfg = run_config(json.loads((trial / 'results.json').read_text()))
                records = json.loads((trial / 'training_metrics.json').read_text())
                agents = ckpt[source]
                T = len(agents)
                phases = list(range(T))[_slice(args.phases)]
                todo = [t for t in phases if (trial.name, t) not in done]
                if not todo:
                    continue
                ends = phase_ends(records, T)
                bases = widths(cfg, records, ends)
                episodes = args.episodes or default_episodes(cfg)
                gymnax = is_gymnax_run(cfg)
                score, chunk = (gymnax_scorer if gymnax else suite_scorer)(
                    cfg, ckpt, episodes)
                t0 = time.time()
                for t in todo:
                    parent = agents[t].astype(np.float32)
                    if not np.isfinite(parent).all():
                        print(f'SKIP {trial} phase {t}: non-finite parent')
                        continue
                    sigmas = {b: s[t] for b, s, _ in bases}
                    copies, kids = probe_checkpoint(
                        score, chunk, parent, t, sigmas, args.children,
                        (str(trial.resolve().relative_to(REPO)), t, args.seed),
                        args.multiples)
                    task = int(records[ends[t]].get('task', t))
                    for b, s, src in bases:
                        rows.append(dict(
                            panel=name, title=title, arm=arm, base=b,
                            method=cfg.get('method'), run=str(trial.resolve().relative_to(REPO)),
                            trial=trial.name, phase=t, generation=int(
                                records[ends[t]].get('generation', ends[t])),
                            source=source, sigma=s[t], sigma_source=src,
                            configured_sigma=cfg.get('mutation_std') or cfg.get('sigma'),
                            d=int(parent.size), rms=float(np.sqrt(np.mean(parent ** 2))),
                            episodes=episodes, children=args.children,
                            multiples=[m for m in args.multiples if m > 0],
                            floor=floor, best=best, span=span,
                            stored=stored_score(records, ends[t], source, task,
                                                'task_type' in cfg),
                            copies=copies.tolist(), kids=kids[b].tolist()))
                raw_path.write_text(json.dumps(rows))
                print(f'{name:<20}{arm:<4}{trial.name:<10}{len(todo):>3} ckpts '
                      f'{time.time() - t0:7.1f}s  sigma {[round(s[todo[-1]], 6) for _, s, _ in bases]}',
                      flush=True)


# ---------------------------------------------------------------------------
# curvature: the perturbation protocol of Liang et al. (2026, arXiv 2602.00170)
# ---------------------------------------------------------------------------
#
# Liang et al. do not compute a Hessian: they perturb the weights by sigma x
# a unit gaussian (M = 240 draws, one fixed ABSOLUTE sigma for every model),
# score every perturbed model, and read the geometry off the distribution of
# the return change -- chiefly the best-of-N improvement E[max_{i<=N} dR_i]
# against N. We draw the directions in antithetic pairs, which adds one
# quantity: to second order the symmetric difference
#     s = (R(th + sigma e) + R(th - sigma e)) / 2 - R(th) = -sigma^2/2 e'He
# has mean -sigma^2/2 tr H and variance sigma^4/2 tr H^2 for a gaussian e, so
#     d_eff = 2 mean(s)^2 / var(s) = (tr H)^2 / tr(H^2)
# is the number of directions the return is curved in (the participation
# ratio of the Hessian spectrum; 1 = one stiff direction, D = isotropic).
# It is a second-order reading: on a return that jumps (a level solved or
# not) it is only an index, and it is reported with the share of s that is
# exactly 0. Every method gets the same absolute sigma, as in the tanh basin
# widths (tab:basin_noise); the reported agent is scored (NE centroid, RL
# final), and the parent's score is its own re-score on the same hardware.

CURV_OUT = REPO / 'results' / 'curvature_probe'
CURV_SIGMAS = (0.005, 0.01, 0.02, 0.04)      # 0.02 = Kinetix ES's own sigma


def _direction(run_key, j, d):
    """Unit gaussian direction j, drawn on demand (240 x 1.1M floats is 1 GB)."""
    rng = np.random.default_rng([zlib.crc32(repr(run_key).encode()), j])
    return rng.standard_normal(d).astype(np.float32)


def curvature_checkpoint(score, chunk, parent, t, sigmas, pairs, copies, run_key):
    """`(copies (C,), plus (S, M), minus (S, M))` returns of one checkpoint."""
    import jax
    from jax import random

    d = parent.size
    plan = [(None, 0, 0)] * copies + [(s, j, sign) for s in range(len(sigmas))
                                     for j in range(pairs) for sign in (1, -1)]
    base_key = random.key(zlib.crc32(repr(run_key).encode()) % (2 ** 31))
    keys_all = jax.vmap(lambda i: random.fold_in(base_key, i))(np.arange(len(plan)))
    scores = np.empty(len(plan), dtype=np.float64)
    cache = {}
    buf, idx = [], []

    def flush():
        n = len(buf)
        g = np.stack(buf + [buf[-1]] * (chunk - n))
        k = keys_all[np.array(idx + [idx[-1]] * (chunk - n))]
        scores[idx] = score(g, k, t)[:n].mean(axis=1)
        buf.clear()
        idx.clear()

    for i, (s, j, sign) in enumerate(plan):
        if s is None:
            buf.append(parent)
        else:
            if j not in cache:
                cache.clear()
                cache[j] = _direction(run_key, j, d)
            buf.append(parent + np.float32(sign * sigmas[s]) * cache[j])
        idx.append(i)
        if len(buf) == chunk:
            flush()
    if buf:
        flush()
    pm = scores[copies:].reshape(len(sigmas), pairs, 2)
    return scores[:copies], pm[..., 0], pm[..., 1]


def cmd_curvature(args):
    from source.envs.run_context import is_gymnax_run, run_config

    CURV_OUT.joinpath('raw').mkdir(parents=True, exist_ok=True)
    for name in args.panels:
        _, tree, cell, title = PANEL[name]
        floor, best = panel_span(tree, cell)
        for arm in args.arms:
            # --tree probes an arm outside the paper's tree (plain ES on
            # Kinetix: runs_kinetix_ep128_ev1/kinetix, arm `es`), saved as `--label`.
            adir = (PROJECT / args.tree / 'continual' / arm if args.tree
                    else PROJECT / tree / 'continual' / arm_dir(arm, tree))
            arm = args.label or arm
            raw_path = CURV_OUT / 'raw' / f'{name}__{arm}.json'
            rows = json.loads(raw_path.read_text()) if raw_path.exists() else []
            done = {(r['trial'], r['phase']) for r in rows}
            trials = sorted((adir / cell).glob('trial_*'),
                            key=lambda p: int(p.name.split('_')[1]))[:args.trials]
            for trial in trials:
                ckpt = np.load(trial / 'checkpoints.npz')
                source = 'final' if 'final' in ckpt.files else 'centroid'
                agents = ckpt[source]
                T = len(agents)
                todo = [t for t in list(range(T))[_slice(args.phases)]
                        if (trial.name, t) not in done]
                if not todo:
                    continue
                cfg = run_config(json.loads((trial / 'results.json').read_text()))
                records = json.loads((trial / 'training_metrics.json').read_text())
                ends = phase_ends(records, T)
                episodes = args.episodes or default_episodes(cfg)
                score, chunk = (gymnax_scorer if is_gymnax_run(cfg) else suite_scorer)(
                    cfg, ckpt, episodes)
                for t in todo:
                    t0 = time.time()
                    parent = agents[t].astype(np.float32)
                    if not np.isfinite(parent).all():
                        print(f'SKIP {trial} phase {t}: non-finite parent')
                        continue
                    run = str(trial.resolve().relative_to(REPO))
                    copies, plus, minus = curvature_checkpoint(
                        score, chunk, parent, t, args.sigmas, args.pairs, args.copies,
                        (run, t, args.seed))
                    rows.append(dict(
                        panel=name, title=title, arm=arm, run=run, trial=trial.name,
                        phase=t, task=int(records[ends[t]].get('task', t)),
                        source=source, method=cfg.get('method'), d=int(parent.size),
                        rms=float(np.sqrt(np.mean(parent ** 2))), episodes=episodes,
                        floor=floor, best=best, sigmas=list(args.sigmas),
                        copies=copies.tolist(), plus=plus.tolist(), minus=minus.tolist()))
                    raw_path.write_text(json.dumps(rows))
                    print(f'{name:<10}{arm:<7}{trial.name:<9}phase {t:<3}'
                          f'{time.time() - t0:7.1f}s  parent {copies.mean():+.3f}', flush=True)


def curvature_stats(r, sigma_index, tol_frac=TOLERANCE):
    """One checkpoint at one sigma.

    fail    share of the 2M perturbed policies scoring below the re-scored
            parent - tol (tol = tol_frac x span): off the plateau.
    both    share of antithetic pairs whose two sides BOTH fail. A curved peak
            is symmetric (e'He is even in e), so there both = fail; a plateau
            whose edge is one flat face fails on one side only, both = 0.
    faces   k from both / fail^2 = 1 - 1/k, the value for k independent flat
            faces each crossed with the same small probability (a single
            face: 0; many: -> 1); inf when both >= fail^2.
    d_eff   2 mean(s)^2 / var(s) of the symmetric differences, the Hessian's
            participation ratio when the return is locally quadratic.
    best_N  Liang et al.'s best-of-N improvement / span, N = 1, 8, 32, all.
    """
    span = r['best'] - r['floor']
    ref = float(np.mean(r['copies']))
    plus = np.asarray(r['plus'][sigma_index])
    minus = np.asarray(r['minus'][sigma_index])
    tol = tol_frac * span
    fp, fm = plus < ref - tol, minus < ref - tol
    fail = float(np.mean(np.concatenate([fp, fm])))
    both = float(np.mean(fp & fm))
    ratio = both / fail ** 2 if fail > 0 else np.nan
    faces = 1.0 / (1.0 - ratio) if ratio == ratio and ratio < 1 else np.inf
    s = (plus + minus) / 2 - ref
    d_eff = 2 * s.mean() ** 2 / s.var() if s.var() > 0 else np.nan
    dr = np.concatenate([plus, minus]) - ref
    rng = np.random.default_rng(0)
    best = {n: float(np.mean([rng.choice(dr, n, replace=False).max()
                              for _ in range(200)]) / span)
            for n in (1, 8, 32, dr.size)}
    return dict(fail=fail, both=both, faces=faces, d_eff=float(d_eff),
                zero_s=float(np.mean(np.abs(s) < tol)), best=best,
                parent=(ref - r['floor']) / span)


def cmd_curvature_table(args):
    rows = []
    for path in sorted(CURV_OUT.joinpath('raw').glob('*.json')):
        rows += json.loads(path.read_text())
    groups, dropped = {}, {}
    for r in rows:
        # Solved checkpoints only, the basin-width rule (app:basin_measure):
        # falling off a level the agent never solved says nothing about a basin.
        solved = (np.mean(r['copies']) - r['floor']) / (r['best'] - r['floor']) >= 0.5
        (groups if solved else dropped).setdefault((r['panel'], r['arm']), []).append(r)
    lines = ['# Perturbation probe (Liang et al. 2026 protocol, antithetic)', '',
             'Solved checkpoints only (re-scored parent >= half the span); dropped: '
             + (', '.join(f'{a} {len(v)}' for (_, a), v in sorted(dropped.items())) or 'none')
             + '.', '',
             'Written by `scripts/analysis/child_survival.py curvature-table`; '
             'definitions in `curvature_stats`. Mean over checkpoints; n = trials / checkpoints. '
             'faces pooled: from the pooled fail and both of the arm.', '',
             '| Task | Arm | n | parent (scaled) | sigma | fail | both | both/fail^2 | faces (pooled) | '
             'd_eff | best-of-1 / 8 / 32 / all (x span) |', '|' + '---|' * 11]
    out = []
    for (panel, arm), sel in sorted(groups.items()):
        for i, sigma in enumerate(sel[0]['sigmas']):
            st = [curvature_stats(r, i) for r in sel]
            fail = np.mean([s['fail'] for s in st])
            both = np.mean([s['both'] for s in st])
            ratio = both / fail ** 2 if fail > 0 else np.nan
            faces = 1 / (1 - ratio) if ratio == ratio and ratio < 1 else np.inf
            best = {n: np.mean([list(s['best'].values())[j] for s in st])
                    for j, n in enumerate(('1', '8', '32', 'all'))}
            entry = dict(panel=panel, arm=arm, sigma=sigma,
                         trials=len({r['trial'] for r in sel}), checkpoints=len(sel),
                         parent=float(np.mean([s['parent'] for s in st])),
                         fail=float(fail), both=float(both), ratio=float(ratio),
                         faces=float(faces),
                         d_eff=float(np.nanmedian([s['d_eff'] for s in st])),
                         best={k: float(v) for k, v in best.items()})
            out.append(entry)
            lines.append(
                f"| {sel[0]['title']} | {arm} | {entry['trials']}/{entry['checkpoints']} | "
                f"{entry['parent']:.2f} | {sigma:g} | {fail:.2f} | {both:.3f} | {ratio:.2f} | "
                f"{faces:.1f} | {entry['d_eff']:.2f} | "
                + ' / '.join(f'{v:+.3f}' for v in best.values()) + ' |')
    CURV_OUT.joinpath('curvature_probe.json').write_text(json.dumps(out, indent=1) + '\n')
    CURV_OUT.joinpath('curvature_probe.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))


# ---------------------------------------------------------------------------
# overlap: does the next task's solution lie inside this task's basin?
# ---------------------------------------------------------------------------
#
# The retention argument is about two regions, not one: after training on A
# (checkpoint theta_A), is the region where A still works shared with the
# region where B, the next task, works? Around every end-of-task checkpoint
# (the reported agent: NE centroid, RL final), perturbations are scored on
# BOTH tasks, A (the checkpoint's) and B (the next checkpoint's):
#
#   fixed      N random directions at each radius, noise relative to each
#              parameter tensor's norm on ReLU networks (Figure 4's
#              convention; eps 0.03 / 0.1 / 0.3) and absolute on tanh ones
#              (HalfCheetah, Kinetix).
#   matched    N random directions with, in every tensor, f times the norm of
#              the step the run actually took to the next checkpoint,
#              Delta = theta_B - theta_A, f = 0.5 / 1: a random move of the
#              same size, layer by layer, as the real one.
#   actual     theta_A + f Delta itself (f = 1 is theta_B).
#
# `matched` against `actual` separates two reasons to forget A: a basin too
# narrow for a move of that size (random directions lose A too), and a move
# along A's stiff directions (random ones keep A, the actual one does not).

OVERLAP_PANELS = [
    ('cartpole_noise2', 'paper/gymnax/data/noise_2task', 'CartPole_v1_sigma0.5', 'CartPole, noise'),
    ('acrobot_noise2', 'paper/gymnax/data/noise_2task', 'Acrobot_v1_sigma0.5', 'Acrobot, noise'),
    ('mountaincar_noise2', 'paper/gymnax/data/noise_2task', 'MountainCar_v0_sigma0.05', 'MountainCar, noise'),
    ('cartpole_actions', 'paper/gymnax/data/actions_2task', 'CartPole_v1_sigma1.0', 'CartPole, action reversal'),
    ('acrobot_actions', 'paper/gymnax/data/actions_2task', 'Acrobot_v1_sigma1.0', 'Acrobot, action reversal'),
    ('mountaincar_actions', 'paper/gymnax/data/actions_2task', 'MountainCar_v0_sigma1.0', 'MountainCar, action reversal'),
    ('cartpole_physics2', 'paper/gymnax/data/physics_2task', 'CartPole_v1_sigma1.0', 'CartPole, pole length'),
    ('acrobot_physics2', 'paper/gymnax/data/physics_2task', 'Acrobot_v1_sigma1.0', 'Acrobot, link mass'),
    ('mountaincar_physics2', 'paper/gymnax/data/physics_2task', 'MountainCar_v0_sigma1.0', 'MountainCar, gravity'),
    ('minigrid', 'paper/minigrid/data', 'MiniGrid_8x8_16x16', 'MiniGrid 8x8 / 16x16'),
    ('kinetix', 'paper/kinetix/data', 'Kinetix20', 'Kinetix, 20 levels'),
    # Figure 2's noise settings: ten offsets, each visited twice; B = the next one.
    ('cartpole_noise10', 'paper/gymnax/data/noise_10task', 'CartPole_v1_sigma1.0', 'CartPole, 10 noise tasks'),
    ('acrobot_noise10', 'paper/gymnax/data/noise_10task', 'Acrobot_v1_sigma1.0', 'Acrobot, 10 noise tasks'),
    ('mountaincar_noise10', 'paper/gymnax/data/noise_10task', 'MountainCar_v0_sigma0.1', 'MountainCar, 10 noise tasks'),
    # The PBT mode probe (queue_pbt_modes_probe.sh, 2026-09-25): arms
    # pbt_weights / pbt_hp on the paper's noise two-task cells.
    ('cartpole_noise2_modes', 'probe_pbt_modes/gymnax', 'CartPole_v1_sigma0.5', 'CartPole, noise (PBT modes)'),
    ('acrobot_noise2_modes', 'probe_pbt_modes/gymnax', 'Acrobot_v1_sigma0.5', 'Acrobot, noise (PBT modes)'),
    # Stationary runs (--stage noncontinual): A = B = the one task, so only
    # `keep` means anything. Only PBT-PPO saved checkpoints there.
    ('cartpole_stationary', 'paper/gymnax/data', 'CartPole_v1', 'CartPole, no switches'),
    ('acrobot_stationary', 'paper/gymnax/data', 'Acrobot_v1', 'Acrobot, no switches'),
    ('mountaincar_stationary', 'paper/gymnax/data', 'MountainCar_v0', 'MountainCar, no switches'),
]
OVERLAP_PANEL = {p[0]: p for p in OVERLAP_PANELS}
OVERLAP_OUT = REPO / 'results' / 'overlap_probe'
RELATIVE_RADII = (0.03, 0.1, 0.3)
ABSOLUTE_RADII = (0.005, 0.02, 0.04)
STEP_FRACTIONS = (0.5, 1.0)
# direction: the basin along the method's own path. Multiples of the step the
# run took over the next task, Delta = theta_B - theta_A, applied to the step
# itself (`actual`, both signs, Rahn et al. 2023's alpha in [-3, 3] extended)
# and to random directions matched to it per tensor (`matched`), so that the
# multiple at which half the moves lose A can be read off both. Cheap on
# gymnax; ~3x the overlap probe on MiniGrid.
DIRECTION_FRACTIONS = (0.1, 0.2, 0.35, 0.5, 0.7, 1.0, 1.4, 2.0, 3.0, 5.0, 8.0)


def _per_tensor(vec, leaf_sizes):
    """Norm of every parameter tensor of flat `vec`, broadcast back to `vec`."""
    starts = np.concatenate([[0], np.cumsum(leaf_sizes)[:-1]])
    norms = np.sqrt(np.add.reduceat(vec.astype(np.float64) ** 2, starts))
    return np.repeat(norms, leaf_sizes)


def overlap_checkpoint(score, chunk, a, b, t, radii, relative, n, copies, run_key):
    """Scores on tasks t (A) and t + 1 (B) of every point around `a`."""
    leaf = score.leaf_sizes
    step = b - a
    a_norm, step_norm = _per_tensor(a, leaf), _per_tensor(step, leaf)
    points, labels = [a] * copies, [('copies', 0.0)] * copies
    for j in range(n):
        e = _direction(run_key, j, a.size)
        e_unit = e / np.maximum(_per_tensor(e, leaf), 1e-12)   # unit norm per tensor
        for r in radii:
            points.append(a + (r * a_norm * e_unit if relative else r * e).astype(np.float32))
            labels.append(('fixed', r))
        for f in STEP_FRACTIONS:
            points.append(a + (f * step_norm * e_unit).astype(np.float32))
            labels.append(('matched', f))
    for f in STEP_FRACTIONS:
        points += [a + np.float32(f) * step] * copies
        labels += [('actual', f)] * copies
    return _score_points(score, chunk, np.stack(points).astype(np.float32), labels, t, run_key)


def direction_checkpoint(score, chunk, a, b, t, radii, relative, n, copies, run_key):
    """Scores on A and B along the actual step (both signs, `copies` re-scores
    each) and along n random directions matched to it per tensor, at every
    multiple of DIRECTION_FRACTIONS. Same signature as overlap_checkpoint;
    `radii` and `relative` are unused."""
    leaf = score.leaf_sizes
    step = b - a
    step_norm = _per_tensor(step, leaf)
    points, labels = [a] * copies, [('copies', 0.0)] * copies
    for j in range(n):
        e = _direction(run_key, j, a.size)
        e_unit = e / np.maximum(_per_tensor(e, leaf), 1e-12)
        for f in DIRECTION_FRACTIONS:
            points.append(a + (f * step_norm * e_unit).astype(np.float32))
            labels.append(('matched', f))
    for f in DIRECTION_FRACTIONS:
        for sign in (1.0, -1.0):
            points += [a + np.float32(sign * f) * step] * copies
            labels += [('actual', sign * f)] * copies
    return _score_points(score, chunk, np.stack(points).astype(np.float32), labels, t, run_key)


def _score_points(score, chunk, points, labels, t, run_key):
    """{label: [[score on A, score on B], ...]} of `points` around a checkpoint of task t."""
    import jax
    from jax import random

    base_key = random.key(zlib.crc32(repr(run_key).encode()) % (2 ** 31))
    out = {}
    for task, name in ((t, 'A'), (t + 1, 'B')):
        keys = jax.vmap(lambda i: random.fold_in(base_key, i))(np.arange(len(points)))
        scores = np.empty(len(points))
        for i in range(0, len(points), chunk):
            g = points[i:i + chunk]
            pad = chunk - len(g)
            gk = keys[i:i + chunk]
            if pad:
                g = np.concatenate([g, np.repeat(g[-1:], pad, 0)])
                gk = jax.numpy.concatenate([gk, jax.numpy.repeat(gk[-1:], pad, 0)])
            scores[i:i + chunk] = score(g, gk, task)[:chunk - pad].mean(axis=1)
        out[name] = scores
    grouped = {}
    for (kind, x), sa, sb in zip(labels, out['A'], out['B']):
        grouped.setdefault(f'{kind}:{x:g}', []).append([sa, sb])
    return {k: np.asarray(v).tolist() for k, v in grouped.items()}


def cmd_overlap(args):
    from plot_stability_plasticity import FLOOR
    from source.envs.run_context import is_gymnax_run, run_config

    probe = direction_checkpoint if args.cmd == 'direction' else overlap_checkpoint
    out_root = REPO / 'results' / args.out
    out_root.joinpath('raw').mkdir(parents=True, exist_ok=True)
    for name in args.panels:
        _, tree, cell, title = OVERLAP_PANEL[name]
        for arm in args.arms:
            adir = PROJECT / tree / args.stage / arm_dir(arm, tree)
            # --source incumbent on pbt: its best single member instead of the
            # weight mean it reports, to test whether averaging makes the width.
            label = args.label or arm
            raw_path = out_root / 'raw' / f'{name}__{label}.json'
            rows = json.loads(raw_path.read_text()) if raw_path.exists() else []
            done = {(r['trial'], r['phase']) for r in rows}
            trials = sorted((adir / cell).glob('trial_*'),
                            key=lambda p: int(p.name.split('_')[1]))[:args.trials]
            if not trials:
                print(f'SKIP {name} {arm}: no trials under {adir / cell}')
            for trial in trials:
                if not (trial / 'checkpoints.npz').exists():
                    print(f'SKIP {trial}: no checkpoints.npz (unfinished run)')
                    continue
                ckpt = np.load(trial / 'checkpoints.npz')
                source = args.source or ('final' if 'final' in ckpt.files else 'centroid')
                agents = ckpt[source]
                T = len(agents)
                todo = [t for t in list(range(T - 1))[_slice(args.phases)]
                        if (trial.name, t) not in done]
                if not todo:
                    continue
                cfg = run_config(json.loads((trial / 'results.json').read_text()))
                gymnax = is_gymnax_run(cfg)
                score, chunk = (gymnax_scorer if gymnax else suite_scorer)(
                    cfg, ckpt, args.episodes or default_episodes(cfg))
                relative = not (name == 'kinetix' or 'cheetah' in name)
                radii = tuple(args.radii) if args.radii else (RELATIVE_RADII if relative else ABSOLUTE_RADII)
                t0 = time.time()
                for t in todo:
                    a, b = agents[t].astype(np.float32), agents[t + 1].astype(np.float32)
                    if not (np.isfinite(a).all() and np.isfinite(b).all()):
                        continue
                    run = str(trial.resolve().relative_to(REPO))
                    rows.append(dict(
                        panel=name, title=title, arm=label, run=run, trial=trial.name,
                        phase=t, source=source, relative=relative, radii=list(radii),
                        floor=FLOOR[cell.split('_sigma')[0]],
                        step_rel=float(np.linalg.norm(b - a) / np.linalg.norm(a)),
                        scores=probe(score, chunk, a, b, t, radii, relative,
                                                  args.directions, args.copies,
                                                  (run, t, args.seed))))
                raw_path.write_text(json.dumps(rows))
                print(f'{name:<20}{arm:<7}{trial.name:<9}{len(todo):>3} ckpts '
                      f'{time.time() - t0:7.1f}s', flush=True)


def overlap_stats(r, best, tol_frac, good):
    """One checkpoint: shares of each point set that keep A, are good on A /
    B / both (rescaled score >= `good`), and mean rescaled A and B scores."""
    span = best - r['floor']
    sc = {k: (np.asarray(v) - r['floor']) / span for k, v in r['scores'].items()}
    ref = sc['copies:0'][:, 0].mean()
    out = {'parent_A': ref, 'parent_B': sc['copies:0'][:, 1].mean()}
    for k, v in sc.items():
        out[k] = dict(keep=float(np.mean(v[:, 0] >= ref - tol_frac)),
                      good_A=float(np.mean(v[:, 0] >= good)),
                      good_B=float(np.mean(v[:, 1] >= good)),
                      both=float(np.mean((v[:, 0] >= good) & (v[:, 1] >= good))),
                      A=float(v[:, 0].mean()), B=float(v[:, 1].mean()))
    return out


def cmd_overlap_table(args):
    rows = []
    for path in sorted(OVERLAP_OUT.joinpath('raw').glob('*.json')):
        rows += json.loads(path.read_text())
    by_panel = {}
    for r in rows:
        by_panel.setdefault(r['panel'], []).append(r)
    order = {p[0]: i for i, p in enumerate(OVERLAP_PANELS)}
    lines = ['# Overlap probe', '',
             'Written by `scripts/analysis/child_survival.py overlap-table` (definitions in '
             '`cmd_overlap` / `overlap_stats`). Scores rescaled per panel: 0 = untrained '
             f'network, 1 = best mean parent score on its own task of any arm. good = >= {args.good}; '
             f'keep = >= parent - {args.tol}. Only checkpoints whose parent is good on A. '
             'Mean over checkpoints.', '']
    for panel in sorted(by_panel, key=order.get):
        sel = by_panel[panel]
        parents = {}
        for r in sel:
            c = np.asarray(r['scores']['copies:0'])[:, 0].mean()
            parents.setdefault(r['arm'], []).append(c)
        best = max(np.mean(v) for v in parents.values())
        keys = [f'fixed:{x:g}' for x in sel[0]['radii']] + \
               [f'matched:{f:g}' for f in STEP_FRACTIONS] + [f'actual:{f:g}' for f in STEP_FRACTIONS]
        lines += [f"## {sel[0]['title']}", '',
                  '| Arm | n | parent A / B | ' + ' | '.join(
                      f'{k}: keep, B, both' for k in keys) + ' |',
                  '|' + '---|' * (3 + len(keys))]
        for arm in sorted({r['arm'] for r in sel}):
            st = [overlap_stats(r, best, args.tol, args.good) for r in sel if r['arm'] == arm]
            st = [s for s in st if s['parent_A'] >= args.good]
            if not st:
                lines.append(f'| {arm} | 0 | | ' + ' | ' * len(keys))
                continue
            cells = [f"{np.mean([s[k]['keep'] for s in st]):.2f}, "
                     f"{np.mean([s[k]['good_B'] for s in st]):.2f}, "
                     f"{np.mean([s[k]['both'] for s in st]):.2f}" for k in keys]
            lines.append(f"| {arm} | {len(st)} | {np.mean([s['parent_A'] for s in st]):.2f} / "
                         f"{np.mean([s['parent_B'] for s in st]):.2f} | " + ' | '.join(cells) + ' |')
        lines.append('')
    OVERLAP_OUT.joinpath('overlap_probe.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))


FIG2_OF_PANEL = {        # overlap panel -> Figure 2's stability_plasticity key
    'cartpole_noise10': 'paper/gymnax/data/noise_10task|CartPole_v1_sigma1.0',
    'acrobot_noise10': 'paper/gymnax/data/noise_10task|Acrobot_v1_sigma1.0',
    'mountaincar_noise10': 'paper/gymnax/data/noise_10task|MountainCar_v0_sigma0.1',
    'minigrid': 'paper/minigrid/data|MiniGrid_8x8_16x16',
    'cartpole_actions': 'paper/gymnax/data/actions_2task|CartPole_v1_sigma1.0',
    'acrobot_actions': 'paper/gymnax/data/actions_2task|Acrobot_v1_sigma1.0',
    'mountaincar_actions': 'paper/gymnax/data/actions_2task|MountainCar_v0_sigma1.0',
}
NE_ARMS = {'es', 'nes', 'ga'}


def cmd_overlap_vs_tradeoff(args):
    """Does the shared basin predict Figure 2's LA - F? One point per setting
    and method: the mean over (seed-pooled) checkpoints of `both` at `--key`,
    against the method's mean rescaled LA - F. Frozen specialists are left out,
    as in Figure 2's ring rule."""
    from scipy.stats import spearmanr
    from plot_stability_plasticity import frozen_methods

    pts = overlap_points(args.key, args.good)
    lines = ['# Shared basin against the stability-plasticity trade-off', '',
                      f'Overlap = share of random moves ({args.key}) that are good (>= {args.good}) '
                      'on both the task just trained and the next; LA - F rescaled as in Figure 2. '
                      'Frozen specialists left out.', '',
                      '| Setting | Method | overlap | LA - F |', '|---|---|---|---|']
    lines += [f'| {p[0]} | {p[1]} | {p[2]:.2f} | {p[3]:.2f} |' for p in pts if not p[4]]
    pts = [p for p in pts if not p[4]]
    lines.append('')
    for name, sub in (('all', pts), ('NE', [p for p in pts if p[1] in NE_ARMS]),
                      ('RL', [p for p in pts if p[1] not in NE_ARMS])):
        if len(sub) > 3:
            r = spearmanr([p[2] for p in sub], [p[3] for p in sub])
            lines.append(f'- Spearman, {name}: rho {r.correlation:.2f}, p {r.pvalue:.3g}, n {len(sub)}')
    for panel in FIG2_OF_PANEL:
        sub = [p for p in pts if p[0] == panel]
        if len(sub) > 2:
            top_ov = max(sub, key=lambda p: p[2])[1]
            top_keep = max(sub, key=lambda p: p[3])[1]
            r = spearmanr([p[2] for p in sub], [p[3] for p in sub])
            lines.append(f'- {panel}: most overlap {top_ov}, best LA - F {top_keep}, '
                         f'rho {r.correlation:.2f} (n {len(sub)})')
    OVERLAP_OUT.joinpath('overlap_vs_tradeoff.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))


def overlap_points(key_name='matched:1', good=0.8):
    """`[(panel, arm, overlap, LA - F, frozen, n)]` for Figure 2's settings."""
    from plot_stability_plasticity import frozen_methods

    rows = []
    for path in sorted(OVERLAP_OUT.joinpath('raw').glob('*.json')):
        rows += json.loads(path.read_text())
    sp = json.loads(SP_DATA.read_text())['panels']
    pts = []
    for panel, key in FIG2_OF_PANEL.items():
        sel = [r for r in rows if r['panel'] == panel]
        if not sel or key not in sp:
            continue
        best = max(np.mean([np.asarray(r['scores']['copies:0'])[:, 0].mean()
                            for r in sel if r['arm'] == a]) for a in {r['arm'] for r in sel})
        by_m = sp[key]
        floor, _ = panel_span(*key.split('|'))
        la_best = max(float(np.mean([t['la'] for t in tr])) for tr in by_m.values())
        span = la_best - floor
        frozen = frozen_methods(key)
        for arm in sorted({r['arm'] for r in sel}):
            m = 'nes' if arm == 'es' else arm
            if m not in by_m:
                continue
            st = [overlap_stats(r, best, 0.05, good) for r in sel if r['arm'] == arm]
            st = [s for s in st if s['parent_A'] >= good]
            if not st:
                continue
            ov = float(np.mean([s[key_name]['both'] for s in st]))
            keep = float(np.mean([(t['la'] - floor - t['F']) / span for t in by_m[m]]))
            pts.append((panel, arm, ov, keep, m in frozen or arm in frozen, len(st)))
    return pts


def _slice(text):
    parts = [int(p) if p else None for p in text.split(':')]
    return slice(*parts) if len(parts) > 1 else slice(parts[0], parts[0] + 1)


# ---------------------------------------------------------------------------
# table
# ---------------------------------------------------------------------------

def per_checkpoint(r, tol_frac):
    """q, up per multiple (0 first) and the parent's reference score."""
    tol = tol_frac * r['span']
    copies = np.asarray(r['copies'])
    ref = float(copies.mean())
    kids = np.vstack([copies[None], np.asarray(r['kids'])])
    q = (kids >= ref - tol).mean(axis=1)
    up = (kids > ref).mean(axis=1)
    return q, up, ref


def best_child_loss(r):
    """(re-scored parent - best child) / span, per multiple (0 first)."""
    copies = np.asarray(r['copies'])
    kids = np.vstack([copies[None], np.asarray(r['kids'])])
    return (copies.mean() - kids.max(axis=1)) / r['span']


def width50(mults, q):
    """The multiple where the q curve (non-zero multiples) first falls below 0.5."""
    m, q = np.asarray(mults[1:]), np.asarray(q[1:])
    if q[0] < 0.5:
        return f'<{m[0]:g}', np.nan
    below = np.flatnonzero(q < 0.5)
    if not below.size:
        return f'>{m[-1]:g}', np.nan
    i = below[0]
    lx = np.interp(0.5, [q[i], q[i - 1]], np.log([m[i], m[i - 1]]))
    return f'{np.exp(lx):.2g}', float(np.exp(lx))


def cluster_boot(values_by_trial, rng):
    """(mean, lo, hi) over all checkpoints, resampling trials."""
    groups = [np.asarray(v) for v in values_by_trial.values()]
    point = np.concatenate(groups).mean(axis=0)
    boots = []
    for _ in range(N_BOOT):
        pick = rng.integers(len(groups), size=len(groups))
        boots.append(np.concatenate([groups[i] for i in pick]).mean(axis=0))
    lo, hi = np.percentile(np.asarray(boots), [2.5, 97.5], axis=0)
    return point, lo, hi


def cmd_table(args):
    rows = []
    for path in sorted(OUT.joinpath('raw').glob('*.json')):
        rows += json.loads(path.read_text())
    order = {p[0]: i for i, p in enumerate(PANELS)}
    groups = {}
    for r in rows:
        groups.setdefault((r['panel'], r['arm'], r['base']), []).append(r)
    grid = sorted({0.0} | {m for r in rows for m in r['multiples']})
    rng = np.random.default_rng(0)
    summary = []
    for (panel, arm, base), sel in sorted(groups.items(),
                                          key=lambda kv: (order[kv[0][0]], kv[0][1], kv[0][2])):
        grids = {tuple(r['multiples']) for r in sel}
        assert len(grids) == 1, f'{panel} {arm} {base}: mixed width grids {grids}; re-run it'
        mults = [0.0] + list(grids.pop())
        entry = dict(panel=panel, title=sel[0]['title'], arm=arm, base=base,
                     label=_label(arm, base), methods=sorted({r['method'] for r in sel}),
                     span=sel[0]['span'], floor=sel[0]['floor'], best=sel[0]['best'],
                     episodes=sel[0]['episodes'], children=sel[0]['children'],
                     trials=len({r['trial'] for r in sel}), checkpoints=len(sel),
                     multiples=mults,
                     sigma_median=float(np.median([r['sigma'] for r in sel])),
                     sigma_range=[float(min(r['sigma'] for r in sel)),
                                  float(max(r['sigma'] for r in sel))],
                     sigma_source=sel[0]['sigma_source'],
                     parent_rescored=float(np.mean([per_checkpoint(r, 0)[2] for r in sel])),
                     parent_rescored_scaled=float(np.mean(
                         [(per_checkpoint(r, 0)[2] - r['floor']) / r['span'] for r in sel])))
        stored = [(r['stored'], per_checkpoint(r, 0)[2]) for r in sel if r['stored'] is not None]
        losses = {}
        for r in sel:
            losses.setdefault(r['trial'], []).append(best_child_loss(r))
        b, blo, bhi = cluster_boot(losses, rng)
        entry['bcl'] = dict(mean=b.tolist(), lo=blo.tolist(), hi=bhi.tolist(),
                            median=np.median(np.concatenate(
                                [np.asarray(v) for v in losses.values()]), axis=0).tolist())
        if stored:
            entry['stored_minus_rescored'] = float(np.mean([a - b for a, b in stored]))
        for tol in args.tolerances:
            qs, ups = {}, {}
            for r in sel:
                q, up, _ = per_checkpoint(r, tol)
                qs.setdefault(r['trial'], []).append(q)
                ups.setdefault(r['trial'], []).append(up)
            q, qlo, qhi = cluster_boot(qs, rng)
            u, ulo, uhi = cluster_boot(ups, rng)
            w_text, w = width50(mults, q)
            entry[f'tol{tol:g}'] = dict(q=q.tolist(), q_lo=qlo.tolist(), q_hi=qhi.tolist(),
                                        up=u.tolist(), up_lo=ulo.tolist(), up_hi=uhi.tolist(),
                                        width50=w_text, width50_value=w)
        summary.append(align(entry, mults, grid))
    OUT.joinpath('child_survival.json').write_text(json.dumps(summary, indent=1) + '\n')
    OUT.joinpath('child_survival.md').write_text(markdown(summary, args.tolerances))
    print(f'wrote {OUT}/child_survival.{{json,md}} ({len(summary)} rows)')


def align(entry, mults, grid):
    """Every per-multiple list on the union `grid`, NaN where not probed."""
    pos = [grid.index(m) for m in mults]

    def fill(v):
        out = [float('nan')] * len(grid)
        for i, x in zip(pos, v):
            out[i] = x
        return out
    for k, v in list(entry.items()):
        if isinstance(v, dict):
            entry[k] = {kk: fill(vv) if isinstance(vv, list) else vv for kk, vv in v.items()}
    entry['probed_multiples'] = mults
    entry['multiples'] = grid
    return entry


def _f(x, fmt):
    return '—' if x != x else format(x, fmt)


def _label(arm, base):
    if arm == 'es':
        return 'ES'
    return 'GA' if base == 'run' else 'GA (explorer width)'


def markdown(summary, tolerances):
    mults = summary[0]['multiples']
    heads = ' / '.join(f'{m:g}x' for m in mults)
    lines = ['# Child survival: the toy q on the continual_main tasks', '',
             'Written by `scripts/analysis/child_survival.py table` (see its docstring).', '',
             '**q** = share of N gaussian children of the saved agent (GA incumbent, ES centroid) '
             'whose fresh-episode score is >= the re-scored parent - tol, at width multiple x '
             "the run's own sigma at that checkpoint. **0x** = N copies of the parent: what "
             'evaluation noise alone gives. **up** = share strictly above the re-scored parent. '
             '**w50** = width multiple where mean q falls to 0.5. tol = fraction x span, '
             'span = best mean LA of any reported arm - untrained return (stability_plasticity). '
             'Brackets: 95% bootstrap over seeds (checkpoints of a seed resampled together). '
             'n = seeds / checkpoints.', '']
    for tol in tolerances:
        lines += [f'## tol = {tol:g} x span', '',
                  f'| Task | Arm | sigma (median) | n | q: {heads} | w50 | up: {heads} |',
                  '|' + '---|' * 7]
        for e in summary:
            s = e[f'tol{tol:g}']
            q = ' / '.join(_f(v, '.2f') for v in s['q'])
            up = ' / '.join(_f(v, '.2f') for v in s['up'])
            lines.append(f"| {e['title']} | {e['label']} | {e['sigma_median']:.2g} | "
                         f"{e['trials']}/{e['checkpoints']} | {q} | {s['width50']} | {up} |")
        lines.append('')
    lines += ['## Best-child loss / span (mean over checkpoints; median in brackets)', '',
              '(re-scored parent - best of the N children) / span. 0x = best of N noisy parent '
              'copies: the noise bias of a max. Positive = no child reaches the parent.', '',
              f'| Task | Arm | sigma (median) | n | {heads} | 1x [95% CI] |', '|' + '---|' * 6]
    i1 = mults.index(1.0)
    for e in summary:
        b = e['bcl']
        vals = ' / '.join('—' if m != m else f'{m:+.3f} ({md:+.3f})'
                          for m, md in zip(b['mean'], b['median']))
        lines.append(f"| {e['title']} | {e['label']} | {e['sigma_median']:.2g} | "
                     f"{e['trials']}/{e['checkpoints']} | {vals} | "
                     f"{b['mean'][i1]:+.3f} [{b['lo'][i1]:+.3f}, {b['hi'][i1]:+.3f}] |")
    lines.append('')
    lines += ['## q at 1x with its interval, and the span', '',
              '| Task | Arm | span (floor, best) | episodes | parent re-score (scaled) | '
              'q 0x | q 1x [95% CI] | up 1x [95% CI] | sigma source |',
              '|' + '---|' * 9]
    tol = tolerances[0]
    for e in summary:
        s = e[f'tol{tol:g}']
        lines.append(
            f"| {e['title']} | {e['label']} | {e['span']:.4g} ({e['floor']:g}, {e['best']:.4g}) | "
            f"{e['episodes']} | {e['parent_rescored']:.4g} ({e['parent_rescored_scaled']:.2f}) | "
            f"{s['q'][0]:.2f} | {s['q'][i1]:.2f} [{s['q_lo'][i1]:.2f}, {s['q_hi'][i1]:.2f}] | "
            f"{s['up'][i1]:.2f} [{s['up_lo'][i1]:.2f}, {s['up_hi'][i1]:.2f}] | "
            f"{e['sigma_source']}; methods {', '.join(e['methods'])} |")
    return '\n'.join(lines) + '\n'


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='cmd', required=True)
    s = sub.add_parser('probe')
    s.add_argument('--panels', nargs='+', default=[q[0] for q in PANELS[:6]],
                   choices=list(PANEL))
    s.add_argument('--arms', nargs='+', default=['ga', 'es'], choices=['ga', 'es'])
    s.add_argument('--children', type=int, default=64)
    s.add_argument('--episodes', type=int, default=None,
                   help='per point; default the reported curve\'s (10, MiniGrid 16, Kinetix 1)')
    s.add_argument('--phases', default=':', help='python slice over saved phases, e.g. 1::2')
    s.add_argument('--trials', type=int, default=None, help='first k trials only')
    s.add_argument('--seed', type=int, default=0)
    s.add_argument('--multiples', nargs='+', type=float, default=list(MULTIPLES),
                   help='width multiples; 0 (parent copies) is always probed')
    s.add_argument('--gpus', default=None)
    s.add_argument('--shard', default=None,
                   help="'i/n': every n-th trial from the i-th, into its own raw file")
    t = sub.add_parser('table')
    t.add_argument('--tolerances', nargs='+', type=float, default=[TOLERANCE, 0.05])
    c = sub.add_parser('curvature', help='Liang et al. (2026) perturbation probe; '
                       'see the curvature section')
    c.add_argument('--panels', nargs='+', default=['kinetix'], choices=list(PANEL))
    c.add_argument('--arms', nargs='+', default=['es', 'ga', 'ppo'],
                   help='arm directories under continual/ (es -> nes/es as in probe)')
    c.add_argument('--sigmas', nargs='+', type=float, default=list(CURV_SIGMAS))
    c.add_argument('--pairs', type=int, default=120, help='antithetic pairs: 240 draws')
    c.add_argument('--copies', type=int, default=4, help='parent re-scores (noise control)')
    c.add_argument('--episodes', type=int, default=None)
    c.add_argument('--phases', default='15:', help='python slice over saved phases')
    c.add_argument('--trials', type=int, default=None)
    c.add_argument('--seed', type=int, default=0)
    c.add_argument('--gpus', default=None)
    c.add_argument('--tree', default=None, help='run tree under projects/iclr_2027 '
                   'instead of the panel\'s paper tree (one arm)')
    c.add_argument('--label', default=None, help='arm name to save under with --tree')
    sub.add_parser('curvature-table')
    def overlap_options(p, phases, copies, out):
        # Shared by `overlap` and `direction`. Not argparse `parents`: a parent
        # shares its option objects with the child, so set_defaults on the
        # child rewrote the parent's defaults (2026-09-25).
        p.add_argument('--panels', nargs='+', default=[q[0] for q in OVERLAP_PANELS[:7]],
                       choices=list(OVERLAP_PANEL))
        p.add_argument('--arms', nargs='+', default=['es', 'ga', 'ppo', 'pbt'])
        p.add_argument('--directions', type=int, default=64)
        p.add_argument('--copies', type=int, default=copies)
        p.add_argument('--episodes', type=int, default=None)
        p.add_argument('--phases', default=phases, help='python slice over checkpoints 0..T-2')
        p.add_argument('--trials', type=int, default=None)
        p.add_argument('--seed', type=int, default=0)
        p.add_argument('--gpus', default=None)
        p.add_argument('--source', default=None, help='checkpoint key (default final/centroid)')
        p.add_argument('--stage', default='continual', help="'noncontinual' for the *_stationary panels")
        p.add_argument('--label', default=None, help='arm name to save under')
        p.add_argument('--radii', type=float, nargs='+', default=None,
                       help='fixed radii instead of RELATIVE_RADII / ABSOLUTE_RADII (the width ladder of '
                            'plot_basin_width_return: 0.001 0.003 0.01 0.03 0.1 0.3 1 3)')
        p.add_argument('--out', default=out, help='results/<out>/raw')
    o = sub.add_parser('overlap', help='both tasks around every checkpoint; see the overlap section')
    overlap_options(o, ':', 4, 'overlap_probe')
    od = sub.add_parser('direction', help='the basin along the actual step, both signs, against random '
                                          'directions of the same size; see the direction comment')
    overlap_options(od, '-5:', 16, 'direction_width')
    ot = sub.add_parser('overlap-table')
    ot.add_argument('--good', type=float, default=0.8, help='rescaled score counted as solved')
    ot.add_argument('--tol', type=float, default=0.05, help='keep: within tol x span of the parent')
    ov = sub.add_parser('overlap-vs-tradeoff')
    ov.add_argument('--key', default='matched:1', help="point set, e.g. matched:1 or fixed:0.1")
    ov.add_argument('--good', type=float, default=0.8)
    args = p.parse_args()
    {'probe': cmd_probe, 'table': cmd_table, 'curvature': cmd_curvature,
     'curvature-table': cmd_curvature_table, 'overlap': cmd_overlap,
     'overlap-table': cmd_overlap_table, 'direction': cmd_overlap,
     'overlap-vs-tradeoff': cmd_overlap_vs_tradeoff}[args.cmd](args)


if __name__ == '__main__':
    main()
