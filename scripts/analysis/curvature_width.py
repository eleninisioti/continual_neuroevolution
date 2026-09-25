"""Scale-free curvature and width of the saved continual agents.

    JAX_PLATFORMS=cpu .venv/bin/python scripts/analysis/curvature_width.py \\
        projects/iclr_2027/runs_noise_2task/gymnax \\
        --out projects/iclr_2027/paper/gymnax/noise/2task/results/centroid \\
        --cells CartPole_v1_sigma0.5 --methods nes ppo

    .venv/bin/python scripts/analysis/curvature_width.py --merge \\
        <out>/shards/*/curvature_width.json --out <out>
        folds per-arm shards into <out>/curvature_width.json, keeping the
        cells and arms already there that no shard replaces.

Writes `<out>/curvature_width.json`, one record per trial, one value per
checkpoint (the agent saved at the end of each sub-task, `--agent`).

Why a second pass. `plasticity_checkpoints.fisher_rank` differentiates
log pi(argmax|s) of the RAW logits. An NE policy is scored on the argmax alone,
so nothing fixes its logit scale: the GA centroid's weights reach an RMS of 30
on gymnax, its softmax saturates to exactly one-hot in float32, and 382 of the
600 GA checkpoints (220 of 600 for ES) on the two-sub-task noise family return
a Fisher of exactly zero. That column then reads "NE loses curvature", which is
a statement about |w|. Every quantity here is invariant to the logit scale.

    fisher_cal      effective rank of the empirical Fisher (Lewandowski et al.,
                    2024, the Hessian-rank proxy) of log pi_T(argmax|s), with
                    each network's temperature T CALIBRATED so its mean policy
                    entropy on the probe is `--entropy-frac` * log|A| (the
                    convention `calibrated_policy_kl` uses). On sub-task t's
                    states at checkpoint t.
    fisher_cal_next the same network on sub-task t+1's states: the curvature
                    the agent meets at the START of the next sub-task, which is
                    where Lewandowski et al. measure it.
    ntk_logit       effective rank of the empirical NTK of the CENTRED logits
                    (logits minus their mean over actions, which a softmax and
                    an argmax both ignore), Gram over (state, action) pairs.
                    A global rescaling of the output scales every entry equally
                    and leaves the rank unchanged.
    width_abs       the same, with independent gaussian noise of a FIXED s.d.
                    on every weight (`ABS_SIGMA`), identical for every
                    method; `--width-abs-only`. Where methods' weight scales
                    differ by 10x (Kinetix: ES 1.0, PPO 0.12), the relative
                    and absolute rankings can disagree.
    width           filter-normalised perturbation robustness (Li et al., 2018):
                    each parameter tensor gets gaussian noise rescaled to
                    `eps` times that tensor's own norm, and the value is the
                    fraction of probe states whose greedy action changes, mean
                    over `--directions` draws. Low = a wide basin in policy
                    space at that relative radius; ReLU layers are
                    scale-equivariant, so this does not move with |w| either.

    --fold-offset   every checkpoint measured with its sub-task's observation
                    offset FOLDED INTO the first layer's bias, on a probe
                    drawn WITHOUT the offset, in the whitened coordinates the
                    NE runs train in. The policy is unchanged to rounding; the
                    NTK is not, because it depends on the parametrisation. On
                    HalfCheetah under noise the offset is up to 24 per-dim
                    standard deviations of the whitened input, so with it in
                    the input every probe state's first-layer gradient shares
                    one direction: the NE rank read ~35 on every noisy
                    sub-task and ~125 on the clean one (sub-tasks 1 and 11),
                    while the RL `final` checkpoints, whose folded normaliser
                    (`actors.fold_normalizer`) already carries the offset in
                    the bias, read ~120 throughout (2026-09-20). Without the
                    fold the two families were measured in different
                    parametrisations. Under the fold an RL checkpoint is
                    re-expressed in the NE whitening (W' = std * W,
                    b' = b + (mean + offset) @ W) and an NE checkpoint gets
                    b' = b + (offset / std) @ W; a gymnax checkpoint (no
                    whitening on either family) gets b' = b + offset @ W.
    --ntk-only      compute `ntk_logit` alone. `--merge` then updates that
                    key INSIDE the existing records (the width columns are
                    the unfolded parametrisation's and stay), keeping the
                    previous value as `ntk_logit_offset_in_input`.

A continuous head (the mjx bodies, whose policy returns the action in [-1, 1])
gets `width` as the normalised action distance ||a - a'|| / (2 sqrt(d)) that
`plasticity_checkpoints.argmax_shift` uses, and `ntk_logit` on the UNCENTRED
action output. It has no `fisher_cal`: the empirical Fisher of a gaussian
policy taken at its own mean is identically zero. Kinetix's multi-discrete
head (`groups`, one categorical per action dimension over contiguous slices of
the logit vector) gets `ntk_logit` with each slice centred on its own mean,
and `width` as the fraction of (state, dimension) greedy choices that change;
no `fisher_cal`, which is defined here for a single categorical.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'scripts' / 'analysis'))

import jax                                                   # noqa: E402
import jax.numpy as jnp                                      # noqa: E402
import gymnax                                                # noqa: E402

import plasticity_checkpoints as pc                          # noqa: E402
from source.envs.gymnax_classic import (                     # noqa: E402
    make_gymnax_env, wrap_actions,
    ENV_CONFIGS, build_policy, unflatten_params)
from source.envs.run_context import RunContext, run_config as merged_config  # noqa: E402
from source.metrics.ntk import rank_stats                    # noqa: E402

EPS = (0.03, 0.1, 0.3)
# `width_abs`: the same test with an ABSOLUTE radius -- independent gaussian
# noise of this s.d. on every weight, the same for every method whatever its
# weight scale. The relative `width` asks how sharp a network is at its own
# scale; this asks how far, in weight space, one can move before the policy
# changes, which is the scale a search with a fixed sigma (ES) works at.
# 0.02 is Kinetix ES's own sigma; it is last so the noise draws of
# the other radii, and so their stored values, do not change.
ABS_SIGMA = (0.01, 0.03, 0.1, 0.3, 0.02)
ARMS = ['ga', 'es', 'nes', 'ppo', 'trac', 'redo', 'cchain', 'pbt', 'pbt2']


def calibrate(logits, target):
    """log T whose mean softmax entropy over the batch is `target`, or None."""
    L = logits - logits.max(axis=-1, keepdims=True)

    def mean_entropy(log_t):
        p = np.exp(L / np.exp(log_t))
        p /= p.sum(axis=-1, keepdims=True)
        return float(np.mean(-(p * np.log(np.clip(p, 1e-300, None))).sum(-1)))

    lo, hi = -40.0, 60.0
    if mean_entropy(hi) < target or mean_entropy(lo) > target:
        return None
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if mean_entropy(mid) < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


class Probe:
    """The jitted measurements for one policy shape."""

    def __init__(self, policy, template, continuous=False, groups=None):
        ends = np.cumsum(groups) if groups else None
        slices = ([slice(int(e - g), int(e)) for g, e in zip(groups, ends)]
                  if groups else None)

        def centre(z):
            if continuous:
                return z
            if slices:
                return jnp.concatenate(
                    [z[..., s] - z[..., s].mean(axis=-1, keepdims=True)
                     for s in slices], axis=-1)
            return z - z.mean(axis=-1, keepdims=True)

        def logits(flat, x):
            return policy.apply(unflatten_params(flat, template), x)

        def logp_greedy(flat, x, inv_t):
            z = policy.apply(unflatten_params(flat, template), x[None])[0] * inv_t
            return jax.nn.log_softmax(z)[jnp.argmax(jax.lax.stop_gradient(z))]

        def fisher_grads(flat, obs, inv_t):
            return jax.vmap(lambda x: jax.grad(logp_greedy)(flat, x, inv_t))(obs)

        def centred_jacobian(flat, obs):
            def out(f):
                z = logits(f, obs)
                # A softmax and an argmax ignore a shift shared by every
                # logit (of one categorical); an action output does not, so it
                # stays uncentred.
                return centre(z)
            return jax.jacrev(out)(flat)          # (N, A, P)

        def act(params, obs):
            out = policy.apply(params, obs)
            if continuous:
                return out
            if slices:
                return jnp.stack([jnp.argmax(out[..., s], axis=-1)
                                  for s in slices], axis=-1)
            return jnp.argmax(out, axis=-1)

        def change(a, b):
            if continuous:
                return (jnp.linalg.norm(a - b, axis=-1).mean()
                        / (2.0 * jnp.sqrt(a.shape[-1])))
            return jnp.mean(a != b)

        def flips(flat, obs, noise, eps):
            params = unflatten_params(flat, template)
            base = act(params, obs)

            def one(n):
                pert = jax.tree_util.tree_map(
                    lambda p, e: p + eps * e * jnp.linalg.norm(p)
                    / (jnp.linalg.norm(e) + 1e-12), params, n)
                return change(act(pert, obs), base)
            return jax.vmap(one)(noise).mean()

        def flips_abs(flat, obs, noise, sigma):
            params = unflatten_params(flat, template)
            base = act(params, obs)

            def one(n):
                pert = jax.tree_util.tree_map(lambda p, e: p + sigma * e, params, n)
                return change(act(pert, obs), base)
            return jax.vmap(one)(noise).mean()

        self.flips_abs = jax.jit(flips_abs)
        self.logits = jax.jit(logits)
        self.fisher_grads = jax.jit(fisher_grads)
        self.centred_jacobian = jax.jit(centred_jacobian)
        self.flips = jax.jit(flips)
        self.template = template
        self.continuous = continuous
        self.groups = groups


def erank(mat):
    """Effective rank of the Gram of the rows of `mat`, in float64."""
    g = np.asarray(mat, dtype=np.float64)
    if not np.isfinite(g).all():
        return None
    return rank_stats(g @ g.T, prefix='x')['x_effective_rank']


def analyse(flat, probe, meas, num_actions, args, seed):
    T = flat.shape[0]
    target = (None if meas.continuous or meas.groups
              else args.entropy_frac * float(np.log(num_actions)))
    key = jax.random.key(seed)
    leaves, treedef = jax.tree_util.tree_flatten(meas.template)
    if args.width_abs_only:
        out = {f'width_abs_{s}': [] for s in ABS_SIGMA}
        for t in range(T):
            if not np.isfinite(flat[t]).all():
                for k in out:
                    out[k].append(None)
                continue
            f = jnp.asarray(flat[t])
            for s in ABS_SIGMA:
                key, sub = jax.random.split(key)
                subs = jax.random.split(sub, len(leaves))
                noise = jax.tree_util.tree_unflatten(treedef, [
                    jax.random.normal(k, (args.directions,) + jnp.shape(l))
                    for k, l in zip(subs, leaves)])
                out[f'width_abs_{s}'].append(
                    float(meas.flips_abs(f, probe[t], noise, jnp.float32(s))))
        return out
    if args.ntk_only:
        out = {'ntk_logit': []}
        for t in range(T):
            if not np.isfinite(flat[t]).all():
                out['ntk_logit'].append(None)
                continue
            J = np.asarray(meas.centred_jacobian(jnp.asarray(flat[t]),
                                                 probe[t][:args.num_fisher]))
            out['ntk_logit'].append(erank(J.reshape(-1, J.shape[-1])))
        return out
    out = {'fisher_cal': [], 'fisher_cal_next': [], 'ntk_logit': [],
           'log_temperature': [], **{f'width_{e}': [] for e in EPS}}
    for t in range(T):
        f = jnp.asarray(flat[t])
        if not np.isfinite(flat[t]).all():
            for k in out:
                if not (k == 'fisher_cal_next' and t == T - 1):
                    out[k].append(None)
            continue
        own = probe[t]
        small = own[:args.num_fisher]
        log_t = (None if meas.continuous or meas.groups else
                 calibrate(np.asarray(meas.logits(f, small), np.float64), target))
        out['log_temperature'].append(log_t)
        if log_t is None:
            out['fisher_cal'].append(None)
            if t < T - 1:
                out['fisher_cal_next'].append(None)
        else:
            inv_t = jnp.float32(np.exp(-log_t))
            out['fisher_cal'].append(erank(meas.fisher_grads(f, small, inv_t)))
            if t < T - 1:
                nxt = probe[t + 1][:args.num_fisher]
                out['fisher_cal_next'].append(
                    erank(meas.fisher_grads(f, nxt, inv_t)))
        J = np.asarray(meas.centred_jacobian(f, small))
        out['ntk_logit'].append(erank(J.reshape(-1, J.shape[-1])))
        for e in EPS:
            key, sub = jax.random.split(key)
            subs = jax.random.split(sub, len(leaves))
            noise = jax.tree_util.tree_unflatten(treedef, [
                jax.random.normal(k, (args.directions,) + jnp.shape(l))
                for k, l in zip(subs, leaves)])
            out[f'width_{e}'].append(float(meas.flips(f, own, noise, jnp.float32(e))))
    return out


def fold_first_layer(flat, template, scale, shift, first_layer='Dense_0'):
    """The same policy on the input `u`, where the old input was `scale * u + shift`.

    Exact to rounding: `(scale * u + shift) @ W + b == u @ (scale[:, None] * W)
    + (b + shift @ W)`. `flat` is `(T, P)`; a non-finite checkpoint is left as
    it is, `analyse` skips it.
    """
    out = []
    for t in range(flat.shape[0]):
        if not np.isfinite(flat[t]).all():
            out.append(flat[t])
            continue
        params = unflatten_params(jnp.asarray(flat[t]), template)
        p = dict(params['params'])
        layer = dict(p[first_layer])
        W = np.asarray(layer['kernel'], np.float64)
        b = np.asarray(layer['bias'], np.float64)
        sc = np.broadcast_to(np.asarray(scale[t], np.float64), W.shape[:1])
        sh = np.broadcast_to(np.asarray(shift[t], np.float64), W.shape[:1])
        p[first_layer] = {**layer,
                          'kernel': jnp.asarray(sc[:, None] * W, jnp.float32),
                          'bias': jnp.asarray(b + sh @ W, jnp.float32)}
        from source.algorithms.networks import get_flat_params
        out.append(np.asarray(get_flat_params({**params, 'params': p}), np.float32))
    return np.stack(out)


def reference_whitening(root, cell, ctx, cache):
    """`(mean, std)` the cell's NE runs whiten with, for re-expressing an RL
    checkpoint in the same coordinates under `--fold-offset`.

    A whitened run's own spec first; otherwise the statistics an NE run of the
    cell recorded (`obs_mean`/`obs_std` in its config, since 2026-09-13); as a
    last resort the fixed-seed measurement `source/envs/mjx._measure_obs_stats`
    (which does not reproduce bit for bit on MJX; see there).
    """
    spec = ctx.env_params
    if getattr(spec, 'obs_mean', None) is not None:
        return np.asarray(spec.obs_mean, np.float64), np.asarray(spec.obs_std, np.float64)
    if cell in cache:
        return cache[cell]
    for arm in ('nes', 'es', 'ga', 'dns'):
        for res in sorted((root / arm / cell).glob('trial_*/results.json')):
            cfg = merged_config(json.loads(res.read_text()))
            if cfg.get('obs_mean') and cfg.get('obs_std'):
                cache[cell] = (np.asarray(cfg['obs_mean'], np.float64),
                               np.asarray(cfg['obs_std'], np.float64))
                print(f'  reference whitening for {cell}: {res}', flush=True)
                return cache[cell]
    from source.envs.mjx import _measure_obs_stats
    mean, std = _measure_obs_stats(ctx.env)
    cache[cell] = (np.asarray(mean, np.float64), np.asarray(std, np.float64))
    print(f'  reference whitening for {cell}: re-measured', flush=True)
    return cache[cell]


def fold_mjx(flat, rows, ctx, template, ref, num_probe, seed):
    """`--fold-offset` on an mjx run: `(flat, probe)` in the NE whitening,
    offset in the bias, probe without it (the matched probe otherwise)."""
    spec = ctx.env_params
    T = flat.shape[0]
    obs_dim = ctx.obs_dim
    offsets = np.stack([np.broadcast_to(np.asarray(spec.obs_offset(rows[t]), np.float64),
                                        (obs_dim,)) for t in range(T)])
    whitened = getattr(spec, 'obs_mean', None) is not None
    ref_mean, ref_std = ref
    if whitened:
        std = np.asarray(spec.obs_std, np.float64)
        flat = fold_first_layer(flat, template, np.ones((T, obs_dim)), offsets / std)
    else:
        flat = fold_first_layer(flat, template, np.broadcast_to(ref_std, (T, obs_dim)),
                                ref_mean[None, :] + offsets)
    probe = []
    for t in range(T):
        row = rows[t] * 0.0 if spec.task_mod == 'obs_noise' else rows[t]
        u = np.asarray(ctx.probe(row, num_probe, seed), np.float64)
        if not whitened:
            u = (u - ref_mean) / ref_std
        probe.append(u.astype(np.float32))
    return flat, np.stack(probe)


def merge(shards, out_dir):
    target = out_dir / 'curvature_width.json'
    result = json.loads(target.read_text()) if target.exists() else None
    for path in shards:
        blob = json.loads(pathlib.Path(path).read_text())
        if result is None:
            result = {'meta': blob['meta'], 'cells': {}}
        elif blob['meta'].get('root') != result['meta'].get('root'):
            raise SystemExit(f"{path}: root {blob['meta'].get('root')} is not "
                             f"{result['meta'].get('root')}")
        partial = bool(blob['meta'].get('ntk_only'))
        for cell, by_m in blob['cells'].items():
            if not partial:
                result['cells'].setdefault(cell, {}).update(by_m)
                continue
            # An ntk-only shard replaces `ntk_logit` inside the records it
            # names and leaves every other column and trial alone.
            for m, rec in by_m.items():
                dst = result['cells'].setdefault(cell, {}).setdefault(m, {'trials': {}})
                for trial, vals in rec['trials'].items():
                    old = dst['trials'].setdefault(trial, {})
                    if 'ntk_logit' in old and 'ntk_logit_offset_in_input' not in old:
                        old['ntk_logit_offset_in_input'] = old['ntk_logit']
                    old['ntk_logit'] = vals['ntk_logit']
            if blob['meta'].get('fold_offset'):
                folded = result['meta'].setdefault('fold_offset_cells', [])
                if cell not in folded:
                    folded.append(cell)
    out_dir.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result))
    print('wrote', target, {c: sorted(m) for c, m in result['cells'].items()})
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('root', nargs='?',
                    help='<tree>/<suite>, the directory holding continual/')
    ap.add_argument('--merge', nargs='+', default=None,
                    help='shard curvature_width.json files to fold into --out')
    ap.add_argument('--out', required=True)
    ap.add_argument('--cells', nargs='+', default=None)
    ap.add_argument('--methods', nargs='+', default=ARMS)
    ap.add_argument('--agent', default='centroid', choices=list(pc.AGENT_SOURCES))
    ap.add_argument('--num-probe', type=int, default=512)
    ap.add_argument('--num-fisher', type=int, default=128)
    ap.add_argument('--directions', type=int, default=8)
    ap.add_argument('--entropy-frac', type=float, default=0.5)
    ap.add_argument('--probe-seed', type=int, default=0)
    ap.add_argument('--trials', nargs='+', default=None)
    ap.add_argument('--width-abs-only', action='store_true',
                    help='compute only `width_abs_<sigma>` (fixed absolute '
                         'noise per weight); write it to its own --out, since '
                         'a merge replaces whole arms')
    ap.add_argument('--fold-offset', action='store_true',
                    help='sub-task offset folded into the first-layer bias, '
                         'probe without it, NE whitening for every arm (see above)')
    ap.add_argument('--ntk-only', action='store_true',
                    help='only `ntk_logit`; --merge updates it inside existing records')
    args = ap.parse_args()
    if args.merge:
        return merge(args.merge, pathlib.Path(args.out))
    if args.root is None:
        ap.error('root is required unless --merge is given')

    root = pathlib.Path(args.root) / 'continual'
    out_dir = pathlib.Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache, shapes, contexts, refs = {}, {}, {}, {}
    result = {'meta': {'root': str(args.root), 'agent': args.agent,
                       'entropy_frac': args.entropy_frac, 'eps': list(EPS),
                       'directions': args.directions, 'num_probe': args.num_probe,
                       'num_fisher': args.num_fisher, 'probe': 'matched',
                       'abs_sigma': list(ABS_SIGMA),
                       'width_abs_only': args.width_abs_only,
                       'fold_offset': args.fold_offset, 'ntk_only': args.ntk_only},
              'cells': {}}
    for method in args.methods:
        mdir = root / method
        if not mdir.is_dir():
            continue
        for cell_dir in sorted(p for p in mdir.iterdir() if p.is_dir()):
            if args.cells and cell_dir.name not in args.cells:
                continue
            name = pc.env_key(cell_dir.name)
            gym = name in ENV_CONFIGS
            trials = {}
            for trial_dir in sorted(cell_dir.glob('trial_*')):
                if args.trials and trial_dir.name not in args.trials:
                    continue
                ckpt = trial_dir / 'checkpoints.npz'
                if not ckpt.exists():
                    continue
                blob = np.load(ckpt, allow_pickle=True)
                source = next((s for s in pc.AGENT_SOURCES[args.agent]
                               if s in blob.files), None)
                if source is None:
                    continue
                flat = np.asarray(blob[source], dtype=np.float32)
                T = flat.shape[0]
                t0 = time.time()
                if gym:
                    if name not in shapes:
                        env, env_params = make_gymnax_env(name)
                        obs_dim = int(pc._base_rollout(name, args.num_probe,
                                                       args.probe_seed, cache).shape[-1])
                        n_act = int(env.action_space(env_params).n)
                        policy, template, _ = build_policy(
                            jax.random.key(0), obs_dim, n_act,
                            ENV_CONFIGS[name]['hidden_dims'])
                        shapes[name] = (Probe(policy, template), n_act, obs_dim)
                    meas, n_act, obs_dim = shapes[name]
                    cfg = pc.run_config(trial_dir) or {}
                    offsets, mults = pc.subtask_sequence(name, blob, cfg, T, obs_dim)
                    if offsets is None:
                        continue
                    if args.fold_offset:
                        offsets = np.asarray(offsets, np.float64).reshape(T, -1)
                        flat = fold_first_layer(flat, meas.template,
                                                np.ones_like(offsets), offsets)
                        offsets = np.zeros_like(offsets)
                    probe = np.asarray(pc.subtask_probes(
                        name, offsets, mults, args.num_probe, args.probe_seed,
                        'matched', cache))
                else:
                    cfg = merged_config(json.loads((trial_dir / 'results.json').read_text()))
                    ck = RunContext.cache_key(cfg, 1)
                    if ck not in contexts:
                        ctx = RunContext(cfg, episodes=1)
                        groups = ctx.suite.action_dims(ctx.env)
                        contexts[ck] = (ctx, Probe(
                            ctx.policy, ctx.template, ctx.continuous,
                            tuple(int(g) for g in groups) if groups else None))
                    ctx, meas = contexts[ck]
                    n_act = ctx.num_actions
                    rows = np.asarray(blob['noise_vectors'])[:T]
                    if args.fold_offset and hasattr(ctx.env_params, 'obs_offset'):
                        ref = reference_whitening(root, cell_dir.name, ctx, refs)
                        flat, probe = fold_mjx(flat, rows, ctx, meas.template, ref,
                                               args.num_probe, args.probe_seed)
                    else:
                        probe = np.stack([np.asarray(ctx.probe(rows[t], args.num_probe,
                                                               args.probe_seed))
                                          for t in range(T)])
                trials[trial_dir.name] = dict(
                    agent_source=source,
                    **analyse(flat, probe, meas, n_act, args,
                              seed=int(trial_dir.name.split('_')[-1])))
                print(f'{method:8s} {cell_dir.name:28s} {trial_dir.name:9s} '
                      f'{time.time() - t0:5.1f}s', flush=True)
            if trials:
                result['cells'].setdefault(cell_dir.name, {})[method] = {'trials': trials}
            # Written after every cell, so a long pass can be read while it runs.
            (out_dir / 'curvature_width.json').write_text(json.dumps(result))
    print('wrote', out_dir / 'curvature_width.json')
    return 0


if __name__ == '__main__':
    sys.exit(main())
