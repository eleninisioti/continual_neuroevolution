"""
Compute KL divergence between consecutive task checkpoints for gymnax continual runs.

Works with both PBT and GA checkpoints saved by:
  - source/gymnax/train_PBT_gymnax_continual.py
  - source/gymnax/train_GA_gymnax_continual.py

Each checkpoint (task_0.pkl, task_1.pkl, ...) is saved at the end of each task
and contains the best agent's policy parameters plus the task's noise vector.

For each pair of consecutive tasks (i, i+1), we:
  1. Load both policies
  2. Collect states by rolling out with policy_i under task_i's noise
  3. Compute action logits for both policies on those states
  4. Convert to probabilities via softmax
  5. Compute KL(pi_i || pi_{i+1}) averaged over states

Usage:
    python scripts/postprocess/compute_kl_gymnax.py \\
        --project_dir projects/gymnax/pbt_full_CartPole_v1_continual/trial_1 \\
        --method pbt --env CartPole-v1

    python scripts/postprocess/compute_kl_gymnax.py \\
        --project_dir projects/gymnax/ga_CartPole-v1_continual/trial_1 \\
        --method ga --env CartPole-v1
"""

import argparse
import os
import sys
import pickle
import json

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import jax
import jax.numpy as jnp
from jax import random
import numpy as np
import flax.linen as nn
from jax import flatten_util
import gymnax


# ============================================================================
# Network definitions (must match training scripts)
# ============================================================================

class PolicyNetwork(nn.Module):
    """Discrete policy network (used by PBT/PPO). Same as train_RL_gymnax_continual.py."""
    hidden_dims: tuple = (16, 16)
    action_dim: int = 2

    @nn.compact
    def __call__(self, x):
        for dim in self.hidden_dims:
            x = nn.Dense(dim)(x)
            x = nn.relu(x)
        logits = nn.Dense(self.action_dim)(x)
        return logits


class MLPPolicy(nn.Module):
    """MLP policy (used by GA). Same as train_GA_gymnax_continual.py."""
    hidden_dims: tuple = (16, 16)
    action_dim: int = 2

    @nn.compact
    def __call__(self, x):
        for dim in self.hidden_dims:
            x = nn.Dense(dim)(x)
            x = nn.relu(x)
        x = nn.Dense(self.action_dim)(x)
        return x


# ============================================================================
# Environment + method configs
# ============================================================================

ENV_CONFIGS = {
    "CartPole-v1": {"episode_length": 500, "hidden_dims": (16, 16)},
    "Acrobot-v1":  {"episode_length": 500, "hidden_dims": (16, 16)},
    "MountainCar-v0": {"episode_length": 500, "hidden_dims": (16, 16)},
}


# ============================================================================
# KL divergence utilities
# ============================================================================

def logits_to_probs(logits, eps=1e-8):
    """Convert logits to a valid probability distribution via softmax."""
    logits = logits - jnp.max(logits)
    probs = jax.nn.softmax(logits)
    probs = jnp.clip(probs, eps, 1.0 - eps)
    probs = probs / jnp.sum(probs)
    return probs


def kl_divergence(p, q, eps=1e-8):
    """KL(P || Q) = sum P(x) * log(P(x) / Q(x))."""
    p = jnp.clip(p, eps, 1.0 - eps)
    q = jnp.clip(q, eps, 1.0 - eps)
    p = p / jnp.sum(p)
    q = q / jnp.sum(q)
    return jnp.sum(p * jnp.log(p / q))


# ============================================================================
# State collection
# ============================================================================

def collect_states(env, env_params, apply_fn, noise_vector, rng, num_episodes=10, episode_length=500):
    """
    Collect states by rolling out a policy in the environment.

    Returns a list of observation arrays.
    """
    all_obs = []
    for ep in range(num_episodes):
        rng, reset_key, step_key = random.split(rng, 3)
        obs, env_state = env.reset(reset_key, env_params)
        all_obs.append(np.array(obs))

        for _ in range(episode_length):
            noisy_obs = obs + noise_vector
            logits = apply_fn(noisy_obs)
            action = jnp.argmax(logits)
            step_key, sub_key = random.split(step_key)
            obs, env_state, reward, done, _ = env.step(sub_key, env_state, action, env_params)
            all_obs.append(np.array(obs))
            if bool(done):
                break

    return all_obs


# ============================================================================
# Checkpoint loading helpers
# ============================================================================

def load_pbt_checkpoint(ckpt_path):
    """Load PBT per-task checkpoint."""
    with open(ckpt_path, 'rb') as f:
        data = pickle.load(f)
    return data['policy_params'], data['noise_vector'], data


def load_ga_checkpoint(ckpt_path):
    """Load GA per-task checkpoint."""
    with open(ckpt_path, 'rb') as f:
        data = pickle.load(f)
    return data['flat_params'], data['noise_vector'], data


def make_pbt_apply(policy_network, policy_params):
    """Create an apply function for PBT policy params."""
    @jax.jit
    def apply_fn(obs):
        return policy_network.apply(policy_params, obs)
    return apply_fn


def make_ga_apply(policy, flat_params, param_template):
    """Create an apply function for GA flat params."""
    _, unravel_fn = flatten_util.ravel_pytree(param_template)
    params = unravel_fn(jnp.array(flat_params))

    @jax.jit
    def apply_fn(obs):
        return policy.apply(params, obs)
    return apply_fn


# ============================================================================
# Main KL computation
# ============================================================================

def compute_kl_between_tasks(
    apply_fn_1, apply_fn_2, states, noise_vector_1, noise_vector_2
):
    """
    Compute KL(pi_1 || pi_2) over a set of raw observations.

    We evaluate both policies with the *first* task's noise (the task that pi_1
    was trained on) so we measure how much pi_2 has drifted on that task's inputs.
    Also compute with the second task's noise for completeness.
    """
    results = {}
    for label, nv in [("task_i_noise", noise_vector_1), ("task_i+1_noise", noise_vector_2)]:
        kl_vals = []
        for obs in states:
            noisy_obs = jnp.array(obs) + jnp.array(nv)
            logits_1 = apply_fn_1(noisy_obs)
            logits_2 = apply_fn_2(noisy_obs)
            p = logits_to_probs(logits_1)
            q = logits_to_probs(logits_2)
            kl_vals.append(float(kl_divergence(p, q)))

        kl_arr = np.array(kl_vals)
        results[label] = {
            "mean": float(np.mean(kl_arr)),
            "std": float(np.std(kl_arr)),
            "min": float(np.min(kl_arr)),
            "max": float(np.max(kl_arr)),
            "median": float(np.median(kl_arr)),
            "num_states": len(kl_arr),
        }
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Compute KL divergence between consecutive task checkpoints (gymnax)"
    )
    parser.add_argument("--project_dir", type=str, required=True,
                        help="Path to trial directory containing checkpoints/ folder")
    parser.add_argument("--method", type=str, required=True, choices=["pbt", "ga", "dns"],
                        help="Method that produced the checkpoints")
    parser.add_argument("--env", type=str, required=True,
                        choices=["CartPole-v1", "Acrobot-v1", "MountainCar-v0"])
    parser.add_argument("--num_episodes", type=int, default=10,
                        help="Number of rollout episodes for state collection")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed for state collection rollouts")
    args = parser.parse_args()

    project_dir = args.project_dir
    env_name = args.env
    method = args.method
    ecfg = ENV_CONFIGS[env_name]

    # Create environment
    env, env_params = gymnax.make(env_name)
    env_params = env_params.replace(max_steps_in_episode=ecfg["episode_length"])
    obs_dim = env.observation_space(env_params).shape[0]
    action_dim = env.action_space(env_params).n

    # Discover task checkpoints
    ckpt_dir = os.path.join(project_dir, "checkpoints")
    if not os.path.isdir(ckpt_dir):
        print(f"Error: No checkpoints directory found at {ckpt_dir}")
        print("Re-run training with the updated script that saves per-task checkpoints.")
        sys.exit(1)

    ckpt_files = sorted(
        [f for f in os.listdir(ckpt_dir) if f.startswith("task_") and f.endswith(".pkl")],
        key=lambda f: int(f.replace("task_", "").replace(".pkl", ""))
    )

    if len(ckpt_files) < 2:
        print(f"Error: Need at least 2 task checkpoints, found {len(ckpt_files)} in {ckpt_dir}")
        sys.exit(1)

    print(f"Found {len(ckpt_files)} task checkpoints in {ckpt_dir}")
    print(f"Method: {method} | Env: {env_name} | obs_dim={obs_dim} action_dim={action_dim}")

    # Set up policy template (needed for GA param unflattening)
    hidden_dims = ecfg["hidden_dims"]
    if method in ("ga", "dns"):
        policy = MLPPolicy(hidden_dims=hidden_dims, action_dim=action_dim)
        dummy_obs = jnp.zeros((obs_dim,))
        param_template = policy.init(random.key(0), dummy_obs)
    else:
        policy_network = PolicyNetwork(hidden_dims=hidden_dims, action_dim=action_dim)

    # Load all checkpoints
    all_ckpts = []
    for fname in ckpt_files:
        path = os.path.join(ckpt_dir, fname)
        if method == "pbt":
            params, noise_vec, meta = load_pbt_checkpoint(path)
        else:  # ga or dns
            params, noise_vec, meta = load_ga_checkpoint(path)
        all_ckpts.append({
            "params": params, "noise_vector": noise_vec, "meta": meta, "file": fname
        })
        task_idx = meta.get("task_idx", "?")
        gen = meta.get("generation", "?")
        print(f"  Loaded {fname}: task={task_idx}, gen={gen}")

    # Compute KL divergence between consecutive tasks
    rng = random.key(args.seed)
    results = {}

    for i in range(len(all_ckpts) - 1):
        ckpt_i = all_ckpts[i]
        ckpt_j = all_ckpts[i + 1]
        task_i = ckpt_i["meta"].get("task_idx", i)
        task_j = ckpt_j["meta"].get("task_idx", i + 1)

        print(f"\n{'='*60}")
        print(f"Comparing task {task_i} vs task {task_j}")
        print(f"{'='*60}")

        # Build apply functions
        if method == "pbt":
            apply_i = make_pbt_apply(policy_network, ckpt_i["params"])
            apply_j = make_pbt_apply(policy_network, ckpt_j["params"])
        else:  # ga or dns
            apply_i = make_ga_apply(policy, ckpt_i["params"], param_template)
            apply_j = make_ga_apply(policy, ckpt_j["params"], param_template)

        # Collect states using policy_i under task_i's noise
        rng, collect_rng = random.split(rng)
        states = collect_states(
            env, env_params, apply_i, jnp.array(ckpt_i["noise_vector"]),
            collect_rng, num_episodes=args.num_episodes,
            episode_length=ecfg["episode_length"]
        )
        print(f"  Collected {len(states)} states from {args.num_episodes} episodes")

        # Compute KL
        kl_results = compute_kl_between_tasks(
            apply_i, apply_j, states,
            ckpt_i["noise_vector"], ckpt_j["noise_vector"]
        )

        pair_key = f"task_{task_i}_vs_{task_j}"
        results[pair_key] = {
            "task_i": int(task_i),
            "task_j": int(task_j),
            "gen_i": int(ckpt_i["meta"].get("generation", 0)),
            "gen_j": int(ckpt_j["meta"].get("generation", 0)),
            **kl_results,
        }

        # Print summary (using task_i noise as the primary metric)
        kl_main = kl_results["task_i_noise"]
        print(f"  KL(pi_{task_i} || pi_{task_j}) on task {task_i} noise:")
        print(f"    mean={kl_main['mean']:.6f}  std={kl_main['std']:.6f}")
        print(f"    min={kl_main['min']:.6f}  max={kl_main['max']:.6f}")

    # Save results
    out_path = os.path.join(project_dir, "kl_divergence.json")
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    # Print summary table
    print(f"\n{'='*70}")
    print(f"{'Task Pair':<20} {'Mean KL':>12} {'Std':>12} {'Num States':>12}")
    print(f"{'-'*70}")
    for key, data in results.items():
        kl = data["task_i_noise"]
        print(f"{key:<20} {kl['mean']:>12.6f} {kl['std']:>12.6f} {kl['num_states']:>12}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
