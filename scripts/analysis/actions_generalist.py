"""Found / Held / Retention for the action-reversal family at sub-task resolution.

Usage (from the repo root; CPU is enough, 1600 agents x 2 orders x 20 episodes):
    PYTHONPATH=. JAX_PLATFORMS=cpu .venv/bin/python scripts/analysis/actions_generalist.py [out_stem] [centroid|elite]
The second argument picks the saved agent: `centroid` (default; the mean of
the population's weights, `final` on the single-policy RL arms) or `elite`
(`finalgen`, the best member of the sub-task's final generation).
Writes <out_stem>.md (the table) and <out_stem>.json (per-trial returns under
each order, 20 checkpoints). The sub-task-resolution analogue of the
generalists report's R1 table (docs/generalists/generalist_report_short.md),
whose per-generation version needs both regimes scored at every record,
which the ICLR trainers do not log.

Every saved end-of-sub-task agent is rolled under BOTH action orders; it is a
generalist when its worse regime clears the env's solved threshold. Found =
any of the 20 checkpoints is a generalist; Held = the last one is; Retention =
fraction of checkpoints after the first discovery that still are.

The second table is the generalists report's phase-outcome categorisation
(its R5), one class per checkpoint from which regimes the agent solves:
    generalist   both orders solved
    switching    only the order it was just trained on -- the specialist
                 that relearns each regime and drops the other
    stuck        only the OTHER order -- it did not learn what it was shown
                 and still holds the previous regime
    neither      neither order
reported as the fraction of the 19 post-switch checkpoints (checkpoint 0 has
no previous regime and is left out), plus `learned shown` = generalist +
switching, the fraction of switches after which the shown regime was solved.
"""
import json, sys, numpy as np, jax, jax.numpy as jnp, gymnax
from jax import random
from source.envs.gymnax_classic import FlipEnv, FlippedParams, build_policy, make_episode_fn, make_gymnax_env, wrap_actions
THR = {'CartPole-v1': 475.0, 'Acrobot-v1': -80.0, 'MountainCar-v0': -150.0}
ARMS = ['nes', 'es', 'ga', 'dns_gaussian', 'ppo', 'trac', 'redo', 'cchain']
CELLS = {'CartPole-v1': 'CartPole_v1_sigma1.0', 'Acrobot-v1': 'Acrobot_v1_sigma1.0', 'MountainCar-v0': 'MountainCar_v0_sigma1.0'}
EPISODES = 20
AGENT = sys.argv[2] if len(sys.argv) > 2 else 'centroid'
SOURCE = {'centroid': 'centroid', 'elite': 'finalgen'}[AGENT]
rows, detail = [], {}
for env_name, cell in CELLS.items():
    cache = {}
    for m in ARMS:
        found = held = 0; ret = []; n = 0; gen_curve = []; outcomes = []
        for t in range(1, 11):
            d = f'projects/iclr_2027/runs_actions/gymnax/continual/{m}/{cell}/trial_{t}'
            try:
                ck = np.load(d + '/checkpoints.npz'); cfg = json.load(open(d + '/results.json'))
            except FileNotFoundError:
                continue
            key = (tuple(cfg['hidden_dims']), int(cfg['episode_length']))
            if key not in cache:
                env, ep = make_gymnax_env(env_name); ep = ep.replace(max_steps_in_episode=key[1])
                obs, _ = env.reset(random.key(0), ep); na = int(env.action_space(ep).n)
                fenv = wrap_actions(env)
                policy, tmpl, _ = build_policy(random.key(0), obs.shape[-1], na, key[0])
                episode = make_episode_fn(fenv, policy, tmpl, key[1])
                roll = jax.jit(lambda A, k, p: jax.vmap(jax.vmap(episode, in_axes=(None, 0, None, None)), in_axes=(0, None, None, None))(A, random.split(k, EPISODES), jnp.zeros(obs.shape[-1]), p))
                cache[key] = (roll, ep)
            roll, ep = cache[key]
            src = SOURCE if SOURCE in ck.files else 'final'
            A = jnp.asarray(ck[src])
            r = np.stack([np.asarray(roll(A, random.key(11), FlippedParams(ep, jnp.float32(f)))).mean(1) for f in (0, 1)])  # (2, T)
            g = r.min(0) >= THR[env_name]
            n += 1; gen_curve.append(g.astype(float))
            flips = np.asarray(ck['action_flips'])[:r.shape[1]]
            shown = r[flips, np.arange(r.shape[1])] >= THR[env_name]      # the order trained on
            other = r[1 - flips, np.arange(r.shape[1])] >= THR[env_name]  # the previous order
            cls = np.where(shown & other, 'generalist', np.where(shown, 'switching', np.where(other, 'stuck', 'neither')))
            outcomes.append(cls[1:])
            if g.any():
                found += 1; first = int(np.argmax(g)); ret.append(float(g[first:].mean())); held += int(g[-1])
            detail[f'{env_name}/{m}/trial_{t}'] = {'stock': r[0].round(1).tolist(), 'reversed': r[1].round(1).tolist()}
        if n:
            allc = np.concatenate(outcomes)
            frac = {c: float(np.mean(allc == c)) for c in ('generalist', 'switching', 'stuck', 'neither')}
            rows.append((env_name, m, found, held, n, float(np.mean(ret)) if ret else 0.0, np.mean(gen_curve, 0), frac))
            print(f'{env_name:15s} {m:13s} found {found:2d}/{n} held {held:2d}/{n} retention {rows[-1][5]:.2f}', flush=True)
out = sys.argv[1] if len(sys.argv) > 1 else \
    f"projects/iclr_2027/paper/gymnax/actions/2task/generalist_checkpoints_{AGENT}"
with open(out + '.md', 'w') as f:
    f.write('| Env | Method | Found | Held | Retention |\n|---|---|---|---|---|\n')
    for e, m, fo, he, n, re, _, _ in rows:
        f.write(f'| {e} | {m} | {fo}/{n} | {he}/{n} | {re:.2f} |\n')
    f.write('\nPhase outcomes, fraction of the 19 post-switch checkpoints pooled over trials '
            '(`learned shown` = generalist + switching):\n\n')
    f.write('| Env | Method | Generalist | Switching | Stuck | Neither | Learned shown |\n|---|---|---|---|---|---|---|\n')
    for e, m, *_, fr in rows:
        f.write(f"| {e} | {m} | {fr['generalist']:.2f} | {fr['switching']:.2f} | {fr['stuck']:.2f} | "
                f"{fr['neither']:.2f} | {fr['generalist'] + fr['switching']:.2f} |\n")
    f.write('\nFraction of trials whose checkpoint t is a generalist (t = 0..19):\n\n')
    for e, m, *_, curve, _ in rows:
        f.write(f'- {e} {m}: ' + ' '.join(f'{c:.1f}' for c in curve) + '\n')
json.dump(detail, open(out + '.json', 'w'))
