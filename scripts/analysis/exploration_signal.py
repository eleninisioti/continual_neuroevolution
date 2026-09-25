"""Return signal at the switch under weight noise and under action noise.

    JAX_PLATFORMS=cpu .venv/bin/python scripts/analysis/exploration_signal.py \\
        --panels gymnax/noise/2task:MountainCar_v0_sigma0.05 --arms ga es ppo trac
    .venv/bin/python scripts/analysis/exploration_signal.py --list

    -> paper/<dir>/results/centroid/exploration_signal/<cell>.npz
       one array per arm, returns[trial, t, condition, sample]

Why. fig:basin measures how much a policy's actions change under weight noise,
which is how NE explores. PPO explores by sampling actions, and a gradient
method needs no basin to cross to the next solution, only a non-zero return
signal: sampled episodes whose returns differ. This pass asks, for every
method and every switch, whether each kind of noise produces that signal.

Every checkpoint theta_t (the centroid saved at the end of sub-task t, as
landscape_slices.py scores it) is rolled out on sub-task t+1, the task it is
about to train on, `--samples` times under each condition:

    greedy       argmax actions; the only noise is the environment's reset
    softmax      actions sampled from softmax(logits), PPO's own exploration.
                 NE never fixes its logit scale (the GA's softmax is one-hot),
                 so this row is meaningful for the RL arms only
    eps_<e>      epsilon-greedy: each step's action is uniform with prob e
    sticky_<e>   the episode is cut into STICKY-step segments and each segment,
                 with prob e, plays ONE uniform action throughout: the same
                 amount of action noise as eps_<e>, correlated in time
    weight_<e>   one draw of fig:basin's filter-normalised weight noise per
                 sample (each tensor gets gaussian noise of norm e * its own
                 norm, curvature_width.py's `width_<e>`), then greedy actions

The spread of the samples' returns is the signal both families learn from: ES
and the GA rank perturbed copies by it, PPO's advantages are built from it.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'scripts'))
sys.path.insert(0, str(REPO / 'scripts/analysis'))

import landscape_slices as ls  # noqa: E402

PAPER = ls.PAPER
EPS_ACT = (0.1, 0.3)
EPS_W = (0.03, 0.1, 0.3)
STICKY = 20
CONDITIONS = (['greedy', 'softmax'] + [f'eps_{e}' for e in EPS_ACT]
              + [f'sticky_{e}' for e in EPS_ACT] + [f'weight_{e}' for e in EPS_W])


def gymnax_panels():
    return [(sub, cell) for sub, cell in ls.panels() if sub.startswith('gymnax/')]


def make_runner(cfg, ckpt, samples, cache):
    """`run(thetas (T, P), key) -> (T, len(CONDITIONS), samples)` returns of
    theta_t on sub-task t+1, and the per-sub-task rows."""
    import jax
    import jax.numpy as jnp
    from jax import random
    from source.envs.gymnax_classic import (make_gymnax_env, wrap_actions, build_policy,
                                            gymnax_task_type, saved_task_rows,
                                            unflatten_params)
    task_type = gymnax_task_type(cfg)
    ck = ('gymnax', cfg['env'], tuple(cfg['hidden_dims']), int(cfg['episode_length']),
          samples, task_type)
    if ck not in cache:
        env, env_params = make_gymnax_env(cfg['env'])
        env_params = env_params.replace(max_steps_in_episode=int(cfg['episode_length']))
        obs, _ = env.reset(random.key(0), env_params)
        num_actions = env.action_space(env_params).n
        if task_type == 'actions':
            env = wrap_actions(env)
        policy, template, _ = build_policy(random.key(0), obs.shape[-1], num_actions,
                                           tuple(cfg['hidden_dims']))
        length = int(cfg['episode_length'])
        nseg = length // STICKY + 1

        def episode(flat, key, offset, body, mode, eps):
            params = unflatten_params(flat, template)
            key, rk, fk, ak = random.split(key, 4)
            obs, state = env.reset(rk, body)
            seg_on = random.uniform(fk, (nseg,)) < eps
            seg_act = random.randint(ak, (nseg,), 0, num_actions)

            def step(carry, i):
                obs, state, total, done, key = carry
                logits = policy.apply(params, obs + offset)
                key, k1, k2, sk = random.split(key, 4)
                greedy = jnp.argmax(logits)
                if mode == 'softmax':
                    action = random.categorical(k1, logits)
                elif mode == 'eps':
                    action = jnp.where(random.uniform(k2) < eps,
                                       random.randint(k1, (), 0, num_actions), greedy)
                elif mode == 'sticky':
                    s = i // STICKY
                    action = jnp.where(seg_on[s], seg_act[s], greedy)
                else:
                    action = greedy
                obs, state, reward, d, _ = env.step(sk, state, action, body)
                total = total + reward * (1.0 - done)
                return (obs, state, total, jnp.maximum(done, d.astype(jnp.float32)), key), None

            (_, _, total, _, _), _ = jax.lax.scan(
                step, (obs, state, 0.0, 0.0, key), jnp.arange(length))
            return total

        def perturb(flat, key, eps):
            params = unflatten_params(flat, template)
            leaves, treedef = jax.tree_util.tree_flatten(params)
            keys = random.split(key, len(leaves))
            noisy = [p + eps * e * jnp.linalg.norm(p) / (jnp.linalg.norm(e) + 1e-12)
                     for p, e in zip(leaves, [random.normal(k, jnp.shape(p))
                                              for k, p in zip(keys, leaves)])]
            return jax.flatten_util.ravel_pytree(jax.tree_util.tree_unflatten(treedef, noisy))[0]

        def one_condition(mode, eps, weight):
            # theta (P,), key, offset, body -> (samples,)
            def f(flat, key, offset, body):
                keys = random.split(key, samples)
                if weight:
                    flats = jax.vmap(lambda k: perturb(flat, random.fold_in(k, 1), eps))(keys)
                    return jax.vmap(lambda p, k: episode(p, k, offset, body, 'greedy', 0.0))(
                        flats, keys)
                return jax.vmap(lambda k: episode(flat, k, offset, body, mode, eps))(keys)
            # over checkpoints: theta (T, P), offsets (T, O), bodies stacked (T, ...)
            return jax.jit(jax.vmap(f, in_axes=(0, None, 0, 0)))

        fns = []
        for c in CONDITIONS:
            kind, _, e = c.partition('_')
            fns.append(one_condition(kind, float(e or 0.0), kind == 'weight'))
        cache[ck] = (fns, env_params)
    fns, stock = cache[ck]
    offsets, bodies, _ = saved_task_rows(cfg, ckpt, stock, cfg['env'])

    def run(thetas, key):
        import numpy as np
        T = thetas.shape[0]
        nxt = list(range(1, T + 1))
        off = jnp.asarray(np.asarray(offsets)[nxt])
        body = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *[bodies[t] for t in nxt])
        th = jnp.asarray(thetas, jnp.float32)
        # common random numbers across conditions and methods
        return np.stack([np.asarray(fn(th, key, off, body)) for fn in fns], axis=1)
    return run


def compute(sub, cell, arms, samples, force, root=None):
    import numpy as np
    import jax
    from jax import random
    from source.envs.run_context import run_config
    import plot_metrics_overview as pmo
    es = json.loads((PAPER / sub / 'es_arm.json').read_text())
    pbt_arm = pmo.keep_one_arm(sub, ['pbt', 'pbt2'])[1][1]
    root = REPO / (root or es['root']) / 'continual'
    out = PAPER / sub / 'results/centroid/exploration_signal'
    out.mkdir(parents=True, exist_ok=True)
    path = out / f'{cell}.npz'
    have = dict(np.load(path, allow_pickle=True)) if path.exists() else {}
    meta = json.loads(str(have.pop('meta'))) if 'meta' in have else {}
    meta.update(paper_dir=sub, cell=cell, conditions=CONDITIONS, sticky=STICKY,
                samples=samples, threshold=ls.spec(sub, cell)[0], floor=ls.spec(sub, cell)[1])
    meta.pop('root', None)                  # per arm: one cell can read two trees
    meta.setdefault('arms', {})
    cache = {}
    for arm in arms:
        run_arm = {'es': es['kept'], 'pbt': pbt_arm}.get(arm, arm)
        if arm in have and not force:
            print(f'{sub} {cell} {arm}: done, skipped', flush=True)
            continue
        cell_dir = root / run_arm / cell
        trials = sorted((d for d in cell_dir.glob('trial_*') if (d / 'checkpoints.npz').exists()),
                        key=lambda d: int(d.name.rsplit('_', 1)[1]))
        if not trials:
            print(f'{sub} {cell} {arm}: no runs under {cell_dir}', flush=True)
            continue
        rows, ids, srcs = [], [], []
        for d in trials:
            ckpt = np.load(d / 'checkpoints.npz')
            src = ls.SOURCE.get(run_arm, 'final')
            src = src if src in ckpt.files else 'final'
            cfg = run_config(json.loads((d / 'results.json').read_text()))
            run = make_runner(cfg, ckpt, samples, cache)
            trial = int(d.name.rsplit('_', 1)[1])
            th = np.asarray(ckpt[src])[:-1]
            rows.append(run(th, random.key(trial)))
            ids.append(trial)
            srcs.append(src)
            g = rows[-1]
            print(f'  {arm} trial {trial}: greedy {g[:, 0].mean():.1f}  softmax {g[:, 1].mean():.1f}'
                  f'  std greedy/softmax/eps0.3/weight0.1 '
                  f'{g[:, 0].std(-1).mean():.1f}/{g[:, 1].std(-1).mean():.1f}/'
                  f'{g[:, 3].std(-1).mean():.1f}/{g[:, 7].std(-1).mean():.1f}', flush=True)
        have[arm] = np.stack(rows).astype(np.float32)
        meta['arms'][arm] = dict(run_arm=run_arm, root=str(root.relative_to(REPO)),
                                trials=ids, source=sorted(set(srcs)))
        np.savez(path, meta=json.dumps(meta), **have)
        print(f'{sub} {cell} {arm} ({run_arm}): {len(ids)} trials -> {path}', flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--panels', nargs='+', help='<paper dir>:<cell>; default every gymnax panel')
    ap.add_argument('--arms', nargs='+', default=ls.ARMS)
    ap.add_argument('--samples', type=int, default=32)
    ap.add_argument('--root', help="run tree (default: es_arm.json's); e.g. the paper's "
                    'gymnax/data/physics_2task, where the physics PBT runs are linked')
    ap.add_argument('--force', action='store_true')
    ap.add_argument('--list', action='store_true')
    args = ap.parse_args()
    panels = ([tuple(p.split(':')) for p in args.panels] if args.panels else gymnax_panels())
    if args.list:
        for p in panels:
            print(*p)
        return 0
    import jax.flatten_util  # noqa: F401
    for sub, cell in panels:
        compute(sub, cell, args.arms, args.samples, args.force, args.root)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
