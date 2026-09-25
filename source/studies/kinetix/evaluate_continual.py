"""Post-hoc scoring of the Kinetix continual runs.

The Kinetix counterpart of `source/studies/evaluate_continual.py`, and it exists
for the same reason: training should not decide the reported numbers. Training
saves, per sub-task, the agent it ended with; scoring happens afterwards, so the
episode count, the agent-selection rule and the choice of metric can all change
without re-running a generation.

What it produces
----------------
For each (variant, trial) it rolls every saved agent out on **every** sub-task
and writes the full square matrix

    M[i, j] = performance of the agent saved after sub-task i, evaluated on
              sub-task j

from which the three continual quantities fall out directly:

    diagonal        M[i, i]           how well each sub-task was learned
    backward (retention/forgetting)   M[i, j] for j < i
    zero-shot transfer                M[i, j] for j > i, and M[i, i+1] in particular

This is the shape `scripts/neurips_2026_rebuttal/` already expects, and it is
why the training run does not need to evaluate on all 20 levels itself: the
Kinetix continual runner starts a fresh process per level and each one saves its
agent, so the whole matrix is recoverable afterwards from what is already on
disk. Nothing here requires re-running training.

Usage
-----
    python experiments/evaluate_continual.py \
        --root ../../projects/kinetix --episodes 20 --gpu 0

Run it from `third_party/kinetix/` (the trainers' working directory), so that
`import kinetix` resolves to this tree.
"""

import argparse
import glob
import json
import os
import pickle
import sys
import time

import numpy as np


# The sub-task sequence the continual runner uses, in order. Kept here rather
# than inferred from directory listing because the order is the experiment --
# sorting the directory names alphabetically would put h10 after h1.
ENVIRONMENTS = [
    "h0_unicycle",
    "h1_car_left",
    "h2_car_right",
    "h3_car_thrust",
    "h4_thrust_the_needle",
    "h5_angry_birds",
    "h6_thrust_over",
    "h7_car_flip",
    "h8_weird_vehicle",
    "h9_spin_the_right_way",
    "h10_thrust_right_easy",
    "h11_thrust_left_easy",
    "h12_thrustfall_left",
    "h13_thrustfall_right",
    "h14_thrustblock",
    "h15_thrustshoot",
    "h16_thrustcontrol_right",
    "h17_thrustcontrol_left",
    "h18_thrust_right_very_easy",
    "h19_thrust_left_very_easy",
]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default="../../projects/kinetix",
                   help="Project tree holding continual/<variant>/<env>/trial_<n>/")
    p.add_argument("--episodes", type=int, default=20,
                   help="Rollouts per (agent, sub-task) cell")
    p.add_argument("--gpu", default=None, help="CUDA_VISIBLE_DEVICES value")
    p.add_argument("--variants", nargs="*", default=None,
                   help="Restrict to these variants (default: every one found)")
    p.add_argument("--trials", nargs="*", type=int, default=None)
    p.add_argument("--overwrite", action="store_true",
                   help="Re-score runs that already have evaluation.json")
    return p.parse_args()


def find_runs(root, variants, trials):
    """Every (variant, trial) that has at least one saved sub-task agent."""
    runs = {}
    pattern = os.path.join(root, "continual", "*", "*", "trial_*", "*_best.pkl")
    for ckpt in sorted(glob.glob(pattern)):
        trial_dir = os.path.dirname(ckpt)
        env_name = os.path.basename(os.path.dirname(trial_dir))
        variant = os.path.basename(os.path.dirname(os.path.dirname(trial_dir)))
        try:
            trial = int(os.path.basename(trial_dir).split("_")[1])
        except (IndexError, ValueError):
            continue
        if variants and variant not in variants:
            continue
        if trials and trial not in trials:
            continue
        runs.setdefault((variant, trial), {})[env_name] = ckpt
    return runs


def main():
    args = parse_args()
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    # Imported after CUDA_VISIBLE_DEVICES is set, or JAX grabs every device.
    import jax
    import jax.numpy as jnp
    from hydra import compose, initialize
    from omegaconf import OmegaConf
    from flax.training.train_state import TrainState
    import optax

    from kinetix.environment.env import make_kinetix_env
    from kinetix.environment import make_reset_fn_from_config
    from kinetix.models import make_network_from_config
    from kinetix.util import (
        general_eval,
        generate_params_from_config,
        load_evaluation_levels,
        normalise_config,
    )

    root = os.path.abspath(args.root)
    runs = find_runs(root, args.variants, args.trials)
    if not runs:
        print(f"No continual runs with saved agents under {root}")
        return 1
    print(f"Found {len(runs)} (variant, trial) run(s) under {root}")

    # Rebuild exactly the config the runs trained under, so the network the
    # checkpoints were written from is the network we instantiate here.
    with initialize(version_base=None, config_path="../configs"):
        cfg = compose(config_name="ppo", overrides=[
            "env=pixels",
            "env_size=m",
            "train_levels=m",
            f'train_levels.train_levels_list=["m/{ENVIRONMENTS[0]}.json"]',
        ])
    config = normalise_config(OmegaConf.to_container(cfg), "PPO-eval")
    env_params, static_env_params = generate_params_from_config(config)

    # All 20 sub-tasks as one batched level set: general_eval rolls the policy
    # out on every level at once, so a whole matrix row is one call.
    level_paths = [f"m/{e}" for e in ENVIRONMENTS]
    levels, _ = load_evaluation_levels(
        level_paths, static_env_params_override=static_env_params)

    eval_env = make_kinetix_env(
        observation_type=config["observation_type"],
        action_type=config["action_type"],
        reset_fn=make_reset_fn_from_config(config, env_params, static_env_params),
        static_env_params=static_env_params,
    )
    network = make_network_from_config(eval_env, env_params, config)

    n_tasks = len(ENVIRONMENTS)

    def eval_agent(params, rng):
        """One agent against all sub-tasks: (returns, solve_rates), each (n_tasks,)."""
        dummy = TrainState.create(
            apply_fn=network.apply, params=params,
            tx=optax.chain(optax.clip_by_global_norm(1.0), optax.adam(1e-4)))

        def one_attempt(key):
            (_, returns, _, ep_lengths, infos), (dones, _) = general_eval(
                key, eval_env, env_params, dummy, levels,
                env_params.max_timesteps, n_tasks,
                keep_states=False, return_trajectories=True,
            )
            mask = jnp.arange(env_params.max_timesteps)[..., None] < ep_lengths[None, :]
            solved = (infos["returned_episode_solved"] * dones * mask).sum(axis=0) / jnp.maximum(
                1, (dones * mask).sum(axis=0))
            return returns, solved

        rets, solves = jax.vmap(one_attempt)(jax.random.split(rng, args.episodes))
        return rets.mean(axis=0), solves.mean(axis=0)

    eval_agent = jax.jit(eval_agent)

    for (variant, trial), by_env in sorted(runs.items()):
        out_dir = os.path.join(root, "continual", variant, f"trial_{trial}_eval")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "evaluation.json")
        if os.path.exists(out_path) and not args.overwrite:
            print(f"  {variant} trial {trial}: evaluation.json exists, skipping")
            continue

        # Row i is the agent saved after sub-task i. Sub-tasks whose agent is
        # missing (a level that failed) stay NaN rather than being dropped, so
        # the matrix indices keep meaning sub-task indices.
        returns = np.full((n_tasks, n_tasks), np.nan, dtype=np.float64)
        solves = np.full((n_tasks, n_tasks), np.nan, dtype=np.float64)
        present = []

        print(f"\n=== {variant} trial {trial}: {len(by_env)}/{n_tasks} agents ===")
        t0 = time.time()
        for i, env_name in enumerate(ENVIRONMENTS):
            ckpt = by_env.get(env_name)
            if ckpt is None:
                print(f"  [{i:2d}] {env_name:30s} MISSING")
                continue
            with open(ckpt, "rb") as f:
                params = pickle.load(f)["params"]
            r, s = eval_agent(params, jax.random.PRNGKey(1000 + i))
            returns[i], solves[i] = np.asarray(r), np.asarray(s)
            present.append(i)
            print(f"  [{i:2d}] {env_name:30s} own={returns[i, i]:+.3f} "
                  f"solve={solves[i, i]:.3f}")

        # Summaries, computed only over sub-tasks whose agent exists.
        def _summarise(mat):
            diag = [mat[i, i] for i in present]
            backward = [mat[i, j] - mat[j, j]
                        for i in present for j in present if j < i]
            zero_shot = [mat[i, i + 1] for i in present
                         if i + 1 < n_tasks and (i + 1) in present]
            final = present[-1] if present else None
            retention = ([mat[final, j] - mat[j, j] for j in present if j < final]
                         if final is not None else [])
            return {
                "own_task_mean": float(np.mean(diag)) if diag else None,
                # Negative = the agent got worse on an earlier task than it was
                # when it left it. This is the forgetting number.
                "backward_transfer_mean": float(np.mean(backward)) if backward else None,
                "zero_shot_next_mean": float(np.mean(zero_shot)) if zero_shot else None,
                "final_agent_retention_mean": float(np.mean(retention)) if retention else None,
            }

        result = {
            "variant": variant,
            "trial": trial,
            "episodes_per_cell": args.episodes,
            "tasks": ENVIRONMENTS,
            "agents_present": present,
            "return_matrix": returns.tolist(),
            "solve_matrix": solves.tolist(),
            "summary_return": _summarise(returns),
            "summary_solve": _summarise(solves),
            "elapsed_seconds": time.time() - t0,
        }
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"  wrote {out_path}")
        s = result["summary_solve"]
        print(f"  solve: own={s['own_task_mean']}, "
              f"backward={s['backward_transfer_mean']}, "
              f"zero-shot={s['zero_shot_next_mean']}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
