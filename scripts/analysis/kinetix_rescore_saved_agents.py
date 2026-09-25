"""Re-score the LAST saved reported agent of every Kinetix stationary run.

The same script on two machines, so any difference between them is the
hardware and nothing else. Per run it records:
  train   the training record's own score of that agent at its last generation
          (incumbent_task0 for NE, centroid_task0 for RL)
  score   this machine's re-score of the saved vector (1 rollout: a level is
          deterministic), through suite.make_scoring_fn
  table   the post-hoc evaluation.json number the paper table reads (where
          the file exists)

Why it exists (2026-09-13): the Kinetix levels are deterministic and chaotic,
and the saved NE elites re-scored exactly on GH200 (100/100 runs per arm)
but flipped on the home server's x86 GPUs (GA 70/100). Run it on each machine
and compare:

    python scripts/analysis/kinetix_rescore_saved_agents.py <repo> <tree>/kinetix/noncontinual gh200.json
    python scripts/analysis/kinetix_rescore_saved_agents.py --compare gh200.json x86.json
"""
import os, sys, json, glob, time
import numpy as np


def compare(path_a, path_b, threshold=1.0):
    """Per arm: solved counts by training record, machine A, machine B, table,
    and how often A matches training and B matches A."""
    A, B = json.load(open(path_a)), json.load(open(path_b))
    key = lambda r: (r["arm"], r["cell"], r["trial"])
    a = {key(r): r for r in A["runs"]}; b = {key(r): r for r in B["runs"]}
    print(f"A = {A['host']} {A['device']}    B = {B['host']} {B['device']}")
    print(f"{'arm':13s} {'train':>6s} {'A':>6s} {'B':>6s} {'table':>6s} | A==train  B==A")
    for arm in sorted({k[0] for k in a}):
        ks = [k for k in a if k[0] == arm and k in b]
        solved = lambda f: sum(1 for k in ks if f(k) is not None and f(k) >= threshold)
        print(f"{arm:13s} {solved(lambda k: a[k]['train']):6d} {solved(lambda k: a[k]['score']):6d} "
              f"{solved(lambda k: b[k]['score']):6d} {solved(lambda k: b[k]['table']):6d} | "
              f"{sum(abs(a[k]['score'] - a[k]['train']) < 1e-3 for k in ks):4d}/{len(ks)} "
              f"{sum(abs(a[k]['score'] - b[k]['score']) < 1e-3 for k in ks):4d}/{len(ks)}")


if len(sys.argv) == 4 and sys.argv[1] == "--compare":
    compare(sys.argv[2], sys.argv[3])
    sys.exit(0)

REPO, ROOT, OUT = sys.argv[1], os.path.abspath(sys.argv[2]), sys.argv[3]
sys.path.insert(0, REPO); os.chdir(REPO)
import jax, jax.numpy as jnp
from source.envs.registry import make_env_for_run, get_suite, suite_for
from source.envs.run_context import run_config

ARMS = ('ga', 'dns', 'dns_gaussian', 'es', 'nes', 'ppo', 'trac', 'redo', 'cchain')
cells = sorted(os.path.basename(p) for p in glob.glob(f'{ROOT}/ga/*'))
out, t0 = [], time.time()
for cell in cells:
    runs = []
    for arm in ARMS:
        for tdir in sorted(glob.glob(f'{ROOT}/{arm}/{cell}/trial_*')):
            if not os.path.exists(f'{tdir}/training_metrics.json'):
                continue
            z = np.load(f'{tdir}/checkpoints.npz')
            src = 'incumbent' if 'incumbent' in z.files else 'final'
            last = json.load(open(f'{tdir}/training_metrics.json'))[-1]
            train = last.get('incumbent_task0', last.get('centroid_task0'))
            table = None
            ev = f'{tdir}/evaluation.json'
            if os.path.exists(ev):
                per = json.load(open(ev)).get('per_task', [])
                idx = max((p['task_idx'] for p in per if p['source'] == src), default=None)
                hit = [p for p in per if p['source'] == src and p['task_idx'] == idx]
                if hit:
                    table = float(np.mean(hit[0]['returns']))
            runs.append(dict(arm=arm, cell=cell, trial=os.path.basename(tdir), source=src,
                             vec=np.asarray(z[src][-1], np.float32),
                             task=np.asarray(z['noise_vectors'][-1]), train=train, table=table,
                             cfg_path=f'{tdir}/results.json'))
    if not runs:
        continue
    cfg = run_config(json.load(open(runs[0]['cfg_path'])))
    env, env_params, obs_dim, action_dim = make_env_for_run(cfg)
    suite = get_suite(suite_for(cfg['env']))
    policy, template, _ = suite.build_policy(jax.random.key(0), obs_dim, action_dim,
                                             cfg['hidden_dims'])
    score = suite.make_scoring_fn(env, env_params, policy, template,
                                  int(cfg['episode_length']), 1)
    # The Kinetix scorer runs the population in chunks of 32 and refuses any
    # other size; pad with copies of the first vector and drop the padding.
    vecs = np.stack([r['vec'] for r in runs])
    pad = (-len(vecs)) % 32
    if pad:
        vecs = np.concatenate([vecs, np.repeat(vecs[:1], pad, axis=0)])
    vals = np.asarray(score(jnp.asarray(vecs), jax.random.key(0),
                            jnp.asarray(runs[0]['task'])))[:len(runs)]
    for r, v in zip(runs, vals):
        out.append({k: r[k] for k in ('arm', 'cell', 'trial', 'source', 'train', 'table')}
                   | {'score': float(v)})
    print(f'{cell}: {len(runs)} runs, {time.time() - t0:.0f}s', flush=True)
json.dump({'host': os.uname().nodename, 'jax': jax.__version__,
           'device': str(jax.devices()[0]), 'runs': out}, open(OUT, 'w'), indent=1)
print('wrote', OUT, len(out))
