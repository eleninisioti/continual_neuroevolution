"""Return-landscape slices around a saved two-sub-task agent, for every panel of
`plot_generalist_outcomes.py` and every method, with ONE checkpoint rule.

    .venv/bin/python scripts/analysis/landscape_slices.py --gpus 0 \\
        --panels gymnax/noise/2task:CartPole_v1_sigma0.5 --arms ga es ppo redo
    .venv/bin/python scripts/analysis/landscape_slices.py --list

    -> paper/<dir>/results/centroid/landscape/<cell>.json      one row a method
       paper/<dir>/results/centroid/landscape/<cell>_<arm>.npz  the two grids

The paper-directory version of
`projects/iclr_2027/runs_mechanism/analysis/scripts/landscape_all.py`, which
drew NES against PPO only and chose checkpoints to fit the hypothesis (NES a
generalist it kept, PPO one it lost). Here the run tree and the ES/NES arm are
the paper directory's (`es_arm.json`; `pbt` is whichever of PBT N=8 / N=2
`plot_metrics_overview.keep_one_arm` keeps), the agent is the centroid one
(`generalist_checkpoints.SOURCES`), and the threshold is the one the outcome
figure classifies with (registry for gymnax, 0.8 MiniGrid, 2000 HalfCheetah).

The plane passes through checkpoint theta_t (the agent saved at the end of
phase t) and is spanned by its actual movement over the next phase
(theta_{t+1} - theta_t) and a random orthogonal direction of the same length,
so both axes are in units of that run's own drift. Every grid point is a real
policy scored with common random numbers on sub-task t (at risk during the
next phase) and on sub-task t+1 (trained next).

Checkpoint rule, the same for every method: each post-switch checkpoint t
(1 <= t <= T-2) gets a TYPE from its scores on sub-task t (shown) and on the
previous one, as `generalist_checkpoints.py` classifies it, with generalists
split by whether checkpoint t+1 still is one:

    kept       generalist at t and at t+1
    lost       generalist at t, not at t+1
    switching  only sub-task t solved
    stuck      only the previous sub-task solved
    neither    neither

The slice shows a generalist whenever the method has one (kept or lost,
whichever is more common), otherwise the method's MOST COMMON type (ties in
that order), at the
checkpoint of that type whose relative drift is closest to the type's median
(ties to the latest phase). MiniGrid uses odd t only (end of a 16x16 phase),
so kept / lost is the outcome figure's generalist / switching at t+1, the
direction `plot_generalist_outcomes.ONE_DIRECTION` keeps.

`within_unit_drift` is the fraction of the unit disc (one drift around theta_t)
where the worse sub-task clears the threshold, for that one checkpoint.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent.parent
PAPER = REPO / 'projects/iclr_2027/paper'
ARMS = ['ga', 'es', 'ppo', 'redo', 'cchain', 'trac', 'pbt']
# generalist_checkpoints.SOURCES['centroid']: the population mean where one is saved
# (NE and PBT), else the single policy.
SOURCE = {'ga': 'centroid', 'es': 'centroid', 'nes': 'centroid', 'pbt': 'centroid',
          'pbt2': 'centroid'}
GYM = {'CartPole': (475.0, 0.0), 'Acrobot': (-80.0, -160.0), 'MountainCar': (-150.0, -250.0)}
# (threshold, colour floor, grid points a side, episodes, chunk) per body.
SPEC = {'minigrid': (0.8, 0.0, 31, 4, 256), 'cheetah': (2000.0, 0.0, 21, 2, 512)}
WINDOW = ((-1.0, 2.0), (-1.5, 1.5))
REL_BETA, REL_WINDOW = 0.1, (-2.5, 2.5)   # slice_plane(rel_beta=): y unit 0.1|theta|
ODD_ONLY = {'MiniGrid_8x8_16x16'}
TYPES = ['kept', 'lost', 'generalist', 'switching', 'stuck', 'neither']


def panels():
    """[(paper dir, cell)] in `plot_generalist_outcomes.GRID` order."""
    sys.path.insert(0, str(REPO / 'scripts'))
    sys.path.insert(0, str(REPO / 'scripts/analysis'))
    import plot_generalist_outcomes as pgo
    return [(p[0], p[1]) for _, row in pgo.GRID for p in row if p]


def spec(sub, cell):
    if cell.startswith('MiniGrid'):
        return SPEC['minigrid']
    if cell.startswith('cheetah'):
        return SPEC['cheetah']
    thr, lo = GYM[cell.split('_')[0]]
    return thr, lo, 41, 32, 512


def choose(np, cell_dir, src, thr, odd_only):
    rows = []
    for d in sorted(cell_dir.glob('trial_*')):
        ev = d / 'evaluation.json'
        if not ev.exists():
            continue
        th = np.load(d / 'checkpoints.npz')
        src_ = src if src in th.files else 'final'
        th = th[src_]
        per = {e['task_idx']: e for e in json.loads(ev.read_text())['per_task']
               if e['source'] == src_}
        own = {t: float(np.mean(e['returns'])) for t, e in per.items()}
        prev = {t: float(np.mean(e['prev_returns'])) for t, e in per.items() if e.get('prev_returns')}
        for t in range(1, th.shape[0] - 1):
            if t not in own or t not in prev or (odd_only and t % 2 == 0):
                continue
            rel = float(np.linalg.norm(th[t + 1] - th[t]) / np.linalg.norm(th[t]))
            if not rel > 1e-6:
                continue          # a search that did not move defines no drift axis
            shown, other = own[t] >= thr, prev[t] >= thr
            if shown and other:
                nxt = (t + 1) in own and (t + 1) in prev
                kind = ('kept' if own[t + 1] >= thr and prev[t + 1] >= thr else 'lost') \
                    if nxt else 'generalist'
            else:
                kind = 'switching' if shown else 'stuck' if other else 'neither'
            rows.append(dict(d=d, src=src_, trial=int(d.name.rsplit('_', 1)[1]), t=t, rel=rel,
                             kind=kind))
    if not rows:
        return None, None
    counts = {k: sum(r['kind'] == k for r in rows) for k in TYPES}
    # A method that ever holds a generalist is drawn at one (the more common of
    # kept / lost); otherwise at its most common other type.
    pool = ['kept', 'lost'] if counts['kept'] + counts['lost'] else TYPES
    kind = max(pool, key=lambda k: (counts[k], -TYPES.index(k)))
    of = [r for r in rows if r['kind'] == kind]
    med = float(np.median([r['rel'] for r in of]))
    r = min(of, key=lambda r: (abs(r['rel'] - med), -r['t']))
    gen = counts['kept'] + counts['lost']
    return r, dict(desc=kind, checkpoints=len(rows), counts=counts, generalists=gen,
                   kept_rate=counts['kept'] / gen if gen else None)


def make_scorer(results_path, ckpt, episodes):
    import jax
    import jax.numpy as jnp
    import numpy as np
    from source.studies import evaluate_continual as ec
    from source.envs.run_context import RunContext, run_config, is_gymnax_run
    cfg = run_config(json.loads(pathlib.Path(results_path).read_text()))
    if is_gymnax_run(cfg):
        cache = {}
        ec._gymnax_rollout(cfg, ckpt, episodes, cache)
        rollout, stock = next(iter(cache.values()))
        offsets, bodies, _ = ec.saved_task_rows(cfg, ckpt, stock, cfg['env'])
        batched = jax.jit(jax.vmap(rollout, in_axes=(0, None, None, None)))

        def score(points, task, key):
            return np.asarray(batched(points, key, jnp.asarray(offsets[task]), bodies[task])).mean(-1)
        return score
    ctx = RunContext(cfg, episodes)
    nv = jnp.asarray(ckpt['noise_vectors'])

    def score(points, task, key):
        tasks = jnp.repeat(nv[task][None], points.shape[0], axis=0)
        return np.asarray(ctx.returns_own_tasks(points, key, tasks)).mean(-1)
    return score


def slice_plane(d, src, t, trial, n, episodes, chunk, axis=None, rel_beta=None, zoom=1.0):
    """The plane through checkpoint t of run `d` along its movement to t+1 (or
    from checkpoint axis[0] to axis[1]) and a random orthogonal direction of the
    same length (seeded by trial and t), scored on sub-tasks t and t+1 with
    common random numbers. With `rel_beta` the random direction instead has
    length rel_beta * |theta_t| (the betas are in that unit, window REL_WINDOW),
    so its extent compares across methods as fig:basin's relative noise does.
    at['next'] is theta_{t+1}'s (alpha, beta), which is (1, 0) unless `axis` is
    some other pair (for a run that did not move, or a plane toward an earlier
    checkpoint). `zoom` widens both axes of the window about the plane's
    centre, keeping the units.

    -> (alphas, betas, {'at_risk', 'trained_next': (n, n) mean returns},
        {same keys: (return at theta_t, return at theta_{t+1})}, |drift|)"""
    import numpy as np
    import jax.numpy as jnp
    from jax import random
    ck = np.load(d / 'checkpoints.npz')
    th = np.asarray(ck[src], dtype=np.float64)
    score = make_scorer(d / 'results.json', ck, episodes)
    key = random.key(12345)                                  # common random numbers
    a, b = axis or (t, t + 1)
    e1 = th[b] - th[a]
    e2 = np.random.default_rng(1000 * trial + t).standard_normal(e1.shape)
    e2 -= e2 @ e1 / (e1 @ e1) * e1
    if rel_beta is None:
        e2 *= np.linalg.norm(e1) / np.linalg.norm(e2)
        window = WINDOW
    else:
        e2 *= rel_beta * np.linalg.norm(th[t]) / np.linalg.norm(e2)
        window = (WINDOW[0], REL_WINDOW)
    window = tuple((c - zoom * (hi - lo) / 2, c + zoom * (hi - lo) / 2)
                   for lo, hi in window for c in [(lo + hi) / 2])
    al, be = np.linspace(*window[0], n), np.linspace(*window[1], n)
    A, B = np.meshgrid(al, be, indexing='xy')
    pts = th[t][None] + A.reshape(-1, 1) * e1[None] + B.reshape(-1, 1) * e2[None]
    pts = np.concatenate([pts, th[t][None], th[t + 1][None]]).astype(np.float32)
    grids, at = {}, {}
    for name, task in (('at_risk', t), ('trained_next', t + 1)):
        vals = np.concatenate([score(jnp.asarray(pts[i:i + chunk]), task, key)
                               for i in range(0, len(pts), chunk)])
        grids[name] = vals[:-2].reshape(A.shape)
        at[name] = (float(vals[-2]), float(vals[-1]))
    dv = th[t + 1] - th[t]
    at['next'] = (float(dv @ e1 / (e1 @ e1)), float(dv @ e2 / (e2 @ e2)))
    return al, be, grids, at, float(np.linalg.norm(e1))


def compute(sub, cell, arms, force):
    import numpy as np
    es = json.loads((PAPER / sub / 'es_arm.json').read_text())
    sys.path.insert(0, str(REPO / 'scripts'))
    sys.path.insert(0, str(REPO / 'scripts/analysis'))
    import plot_metrics_overview as pmo
    pbt_arm = pmo.keep_one_arm(sub, ['pbt', 'pbt2'])[1][1]
    root = REPO / es['root'] / 'continual'
    thr, lo, n, episodes, chunk = spec(sub, cell)
    out = PAPER / sub / 'results/centroid/landscape'
    out.mkdir(parents=True, exist_ok=True)
    jpath = out / f'{cell}.json'
    blob = json.loads(jpath.read_text()) if jpath.exists() else {}
    blob.update(paper_dir=sub, cell=cell, root=str(root.relative_to(REPO)), threshold=thr,
                floor=lo, grid=n, episodes=episodes, window=WINDOW, es_arm=es['kept'])
    blob.setdefault('rows', {})
    for arm in arms:
        run_arm = {'es': es['kept'], 'pbt': pbt_arm}.get(arm, arm)
        if arm in blob['rows'] and not force and (out / f'{cell}_{arm}.npz').exists():
            print(f'{sub} {cell} {arm}: done, skipped', flush=True)
            continue
        r, info = choose(np, root / run_arm / cell, SOURCE.get(run_arm, 'final'), thr, cell in ODD_ONLY)
        if r is None:
            print(f'{sub} {cell} {arm}: no evaluated run under {root / run_arm / cell}', flush=True)
            continue
        al, be, grids, at, drift = slice_plane(r['d'], r['src'], r['t'], r['trial'],
                                               n, episodes, chunk)
        np.savez(out / f'{cell}_{arm}.npz', alphas=al, betas=be, **grids)
        A, B = np.meshgrid(al, be, indexing='xy')
        worse = np.minimum(grids['at_risk'], grids['trained_next'])
        within = float((worse[A ** 2 + B ** 2 <= 1.0] >= thr).mean())
        blob['rows'][arm] = dict(arm=run_arm, trial=r['trial'], t=r['t'], source=r['src'],
                                 rel_drift=r['rel'], abs_drift=drift,
                                 at_risk_at_t=at['at_risk'][0], at_risk_at_t1=at['at_risk'][1],
                                 trained_next_at_t=at['trained_next'][0],
                                 trained_next_at_t1=at['trained_next'][1],
                                 within_unit_drift=within, **info)
        jpath.write_text(json.dumps(blob, indent=1))
        print(f'{sub} {cell} {arm} ({run_arm}): trial {r["trial"]} phase {r["t"]} ({info["desc"]}; '
              f'{info["generalists"]}/{info["checkpoints"]} generalists, kept rate '
              f'{info["kept_rate"]}), drift {100 * r["rel"]:.0f}% of |θ|, '
              f'within 1 drift {within:.2f}', flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--gpus', default='0')
    ap.add_argument('--panels', nargs='+', help='<paper dir>:<cell>; default every panel')
    ap.add_argument('--arms', nargs='+', default=ARMS)
    ap.add_argument('--force', action='store_true')
    ap.add_argument('--list', action='store_true')
    args = ap.parse_args()
    todo = [tuple(p.split(':')) for p in args.panels] if args.panels else panels()
    if args.list:
        print('\n'.join(f'{s}:{c}' for s, c in todo))
        return 0
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
    sys.path.insert(0, str(REPO))
    os.chdir(REPO)
    for sub, cell in todo:
        compute(sub, cell, args.arms, args.force)
    return 0


if __name__ == '__main__':
    sys.exit(main())
