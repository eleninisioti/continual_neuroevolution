"""Does a target-speed cell have GRADED conflict? And what is its floor?

Scores two policies at every target the cell's sub-tasks carry, 10 episodes
each: a do-nothing network (all weights zero: zero torque), whose row IS the
cell's floor per sub-task, and optionally a TRAINED policy (a saved centroid).
A trained policy scoring well at its own target and worse at the other is the
conflict the family exists to create; the same score on both means the
sub-tasks do not conflict.

Measured with it: ant_speed (2 vs 8 m/s) floor 1514 / 1512, an ES centroid
trained at 2 m/s 4289 / 1764 (2026-09-12).

    python scripts/analysis/speed_conflict_probe.py --cell ant_speed24 \
        [--trained <run>/checkpoints.npz] [--episodes 10]
"""
import argparse, os
os.environ["NE_OBS_NORM"] = "1"          # before build_env: NE policies read whitened inputs
import sys
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO)

import jax
import jax.numpy as jnp
import numpy as np

from source.envs import mjx
from source.studies.mjx import settings as S


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cell", default="ant_speed24")
    ap.add_argument("--trained", default=None, help="checkpoints.npz whose last `centroid` row is scored")
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--episode_length", type=int, default=1000)
    ap.add_argument("--hidden", type=int, nargs="+", default=[128, 128])
    args = ap.parse_args()

    body = S.CELLS[args.cell][0]
    opts = S.task_options(args.cell)
    env, spec = mjx.build_env(body, args.episode_length, opts)
    obs_dim, action_dim = mjx.env_dims(env, jax.random.key(0))
    policy, template, num_params = mjx.build_policy(jax.random.key(0), obs_dim, action_dim, tuple(args.hidden))
    flat, _ = jax.flatten_util.ravel_pytree(template)
    score = mjx.make_scoring_fn(env, spec, policy, template, args.episode_length, args.episodes, whiten=True)
    n_targets = len(str(opts.get("speed_targets", "")).split(",")) or 2
    targets = [float(t) for t in np.asarray(mjx.task_vectors(spec, 1, n_targets, obs_dim, 0.0)).ravel()]
    print(f"cell {args.cell}: body {body}, targets {targets}, {num_params} parameters")

    rows = [("do-nothing (floor)", jnp.zeros_like(flat))]
    if args.trained:
        z = np.load(args.trained, allow_pickle=True)
        a = np.asarray(z["centroid"])
        rows.append((f"trained ({os.path.basename(os.path.dirname(args.trained))})", jnp.asarray(a[-1] if a.ndim > 1 else a)))
    print(f"{'policy':28s}" + "".join(f"{'target ' + str(t):>13s}" for t in targets))
    for name, g in rows:
        vals = [float(score(g[None, :], jax.random.key(0), jnp.asarray([t]))[0]) for t in targets]
        print(f"{name:28s}" + "".join(f"{v:13.1f}" for v in vals))


if __name__ == "__main__":
    main()
