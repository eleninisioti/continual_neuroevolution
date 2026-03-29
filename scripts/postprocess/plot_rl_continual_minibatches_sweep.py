"""
Plot continual RL performance vs minibatches from the minibatches sweep.

Reads per-task checkpoints from:
  projects/gymnax/rl_minibatches_sweep/{env}/minibatches_{N}/trial_{T}/checkpoints/task_{K}.pkl

Creates two plots:
  1. Average reward across all 10 tasks vs minibatches (barplot)
  2. Success rate (fraction of tasks solved) vs minibatches (barplot)

Solved thresholds (standard gymnax):
  - CartPole-v1:    reward >= 475  (max 500)
  - Acrobot-v1:     reward >= -100 (higher is better towards 0)
  - MountainCar-v0: reward >= -110 (higher is better towards 0)

Usage:
    python scripts/postprocess/plot_rl_continual_minibatches_sweep.py
    python scripts/postprocess/plot_rl_continual_minibatches_sweep.py --method ppo
    python scripts/postprocess/plot_rl_continual_minibatches_sweep.py --method ppo trac redo
"""

import argparse
import os
import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


ENVS = ["CartPole-v1", "Acrobot-v1", "MountainCar-v0"]
MINIBATCH_COUNTS = [1, 2, 4, 8, 16, 32, 64]
NUM_TASKS = 10

# Standard "solved" thresholds
SOLVED_THRESHOLDS = {
    "CartPole-v1": 475.0,
    "Acrobot-v1": -70.0,
    "MountainCar-v0": -110.0,
}


def load_results(sweep_dir, methods):
    """
    Load per-task eval_mean from all checkpoints.

    Returns:
        results: {method: {env: {minibatches: [ [task_rewards for trial_1], ... ]}}}
    """
    results = {}

    for method in methods:
        results[method] = {}
        for env in ENVS:
            results[method][env] = {}
            for mb in MINIBATCH_COUNTS:
                trial_task_rewards = []
                for trial in range(1, 100):
                    ckpts_dir = os.path.join(
                        sweep_dir, env, f"minibatches_{mb}", f"trial_{trial}", "checkpoints"
                    )
                    if not os.path.isdir(ckpts_dir):
                        if trial > 10:
                            break
                        continue

                    task_rewards = []
                    for task_idx in range(NUM_TASKS):
                        ckpt_path = os.path.join(ckpts_dir, f"task_{task_idx}.pkl")
                        if not os.path.exists(ckpt_path):
                            continue
                        try:
                            with open(ckpt_path, 'rb') as f:
                                data = pickle.load(f)
                            task_rewards.append(data['eval_mean'])
                        except Exception as e:
                            print(f"Warning: Failed to load {ckpt_path}: {e}")

                    if task_rewards:
                        trial_task_rewards.append(task_rewards)

                if trial_task_rewards:
                    results[method][env][mb] = trial_task_rewards

    return results


def compute_metrics(results):
    """
    Compute per-minibatch metrics:
      - avg_reward: mean reward across all tasks and trials
      - success_rate: fraction of tasks solved (averaged across trials)
    """
    metrics = {}
    for method in results:
        metrics[method] = {}
        for env in ENVS:
            metrics[method][env] = {}
            threshold = SOLVED_THRESHOLDS[env]
            for mb in sorted(results[method].get(env, {}).keys()):
                trial_data = results[method][env][mb]

                all_rewards = [r for trial in trial_data for r in trial]
                avg_reward_mean = np.mean(all_rewards)
                trial_avgs = [np.mean(trial) for trial in trial_data]
                avg_reward_std = np.std(trial_avgs)

                trial_success_rates = []
                for trial in trial_data:
                    solved = sum(1 for r in trial if r >= threshold)
                    trial_success_rates.append(solved / len(trial))
                success_mean = np.mean(trial_success_rates)
                success_std = np.std(trial_success_rates)

                metrics[method][env][mb] = {
                    'avg_reward_mean': avg_reward_mean,
                    'avg_reward_std': avg_reward_std,
                    'success_mean': success_mean,
                    'success_std': success_std,
                    'num_trials': len(trial_data),
                }
    return metrics


# Colors for methods
METHOD_COLORS = {
    'ppo': 'steelblue',
    'trac': 'seagreen',
    'redo': 'darkorange',
}

METHOD_COLORS_SUCCESS = {
    'ppo': 'coral',
    'trac': 'mediumseagreen',
    'redo': 'goldenrod',
}


def plot_results(metrics, methods, output_dir):
    """Create barplots for average reward and success rate."""

    num_methods = len(methods)
    bar_width = 0.8 / num_methods

    # --- Plot 1: Average reward across tasks ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, env in zip(axes, ENVS):
        # Collect all minibatch counts that have data across any method
        all_mbs = sorted(set(
            mb for m in methods for mb in metrics.get(m, {}).get(env, {}).keys()
        ))
        if not all_mbs:
            ax.set_title(env)
            ax.text(0.5, 0.5, "No data", ha='center', va='center', transform=ax.transAxes)
            continue

        x = np.arange(len(all_mbs))

        for mi, method in enumerate(methods):
            means = []
            stds = []
            for mb in all_mbs:
                if mb in metrics.get(method, {}).get(env, {}):
                    m = metrics[method][env][mb]
                    means.append(m['avg_reward_mean'])
                    stds.append(m['avg_reward_std'])
                else:
                    means.append(0)
                    stds.append(0)

            offset = (mi - (num_methods - 1) / 2) * bar_width
            color = METHOD_COLORS.get(method, 'steelblue')
            ax.bar(x + offset, means, bar_width, yerr=stds, capsize=3,
                   color=color, edgecolor='black', alpha=0.85,
                   label=method.upper())

            for i, mb in enumerate(all_mbs):
                if mb in metrics.get(method, {}).get(env, {}):
                    n = metrics[method][env][mb]['num_trials']
                    ax.text(i + offset, means[i] + stds[i] + 0.02 * max(abs(v) for v in means if v != 0),
                            f"n={n}", ha='center', va='bottom', fontsize=7, color='gray')

        ax.set_xticks(x)
        ax.set_xticklabels([str(mb) for mb in all_mbs], rotation=45)
        ax.set_xlabel("Number of Minibatches")
        ax.set_ylabel("Avg Reward (across 10 tasks)")
        ax.set_title(env)
        ax.grid(axis='y', alpha=0.3)
        if num_methods > 1:
            ax.legend(fontsize=8)

    fig.suptitle("Continual RL: Average Reward vs Number of Minibatches", fontsize=14, y=1.02)
    plt.tight_layout()
    path1 = os.path.join(output_dir, "continual_minibatches_sweep_reward.png")
    plt.savefig(path1, dpi=150, bbox_inches='tight')
    print(f"Saved: {path1}")
    plt.close()

    # --- Plot 2: Success rate ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, env in zip(axes, ENVS):
        all_mbs = sorted(set(
            mb for m in methods for mb in metrics.get(m, {}).get(env, {}).keys()
        ))
        if not all_mbs:
            ax.set_title(env)
            ax.text(0.5, 0.5, "No data", ha='center', va='center', transform=ax.transAxes)
            continue

        x = np.arange(len(all_mbs))

        for mi, method in enumerate(methods):
            means = []
            stds = []
            for mb in all_mbs:
                if mb in metrics.get(method, {}).get(env, {}):
                    m = metrics[method][env][mb]
                    means.append(m['success_mean'])
                    stds.append(m['success_std'])
                else:
                    means.append(0)
                    stds.append(0)

            offset = (mi - (num_methods - 1) / 2) * bar_width
            color = METHOD_COLORS_SUCCESS.get(method, 'coral')
            ax.bar(x + offset, means, bar_width, yerr=stds, capsize=3,
                   color=color, edgecolor='black', alpha=0.85,
                   label=method.upper())

            for i, mb in enumerate(all_mbs):
                if mb in metrics.get(method, {}).get(env, {}):
                    n = metrics[method][env][mb]['num_trials']
                    ax.text(i + offset, means[i] + stds[i] + 0.02,
                            f"n={n}", ha='center', va='bottom', fontsize=7, color='gray')

        ax.set_xticks(x)
        ax.set_xticklabels([str(mb) for mb in all_mbs], rotation=45)
        ax.set_xlabel("Number of Minibatches")
        ax.set_ylabel("Success Rate (frac. of tasks solved)")
        ax.set_title(f"{env} (threshold={SOLVED_THRESHOLDS[env]})")
        ax.set_ylim(0, 1.05)
        ax.grid(axis='y', alpha=0.3)
        if num_methods > 1:
            ax.legend(fontsize=8)

    fig.suptitle("Continual RL: Task Success Rate vs Number of Minibatches", fontsize=14, y=1.02)
    plt.tight_layout()
    path2 = os.path.join(output_dir, "continual_minibatches_sweep_success.png")
    plt.savefig(path2, dpi=150, bbox_inches='tight')
    print(f"Saved: {path2}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Plot continual RL minibatches sweep results")
    parser.add_argument("--sweep_dir", type=str,
                        default="projects/gymnax/rl_minibatches_sweep",
                        help="Root directory of the sweep results")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory for plots (default: same as sweep_dir)")
    parser.add_argument("--method", type=str, nargs='+', default=["ppo"],
                        help="Method(s) to plot (default: ppo)")
    args = parser.parse_args()

    output_dir = args.output_dir or args.sweep_dir
    os.makedirs(output_dir, exist_ok=True)

    methods = args.method

    results = load_results(args.sweep_dir, methods)

    metrics = compute_metrics(results)

    # Print summary table
    print(f"\n{'Method':<8} {'Env':<16} {'Minibatch':>10} {'AvgReward':>12} {'Std':>10} {'Success':>10} {'N':>4}")
    print("-" * 74)
    for method in methods:
        for env in ENVS:
            for mb in sorted(metrics.get(method, {}).get(env, {}).keys()):
                m = metrics[method][env][mb]
                print(f"{method:<8} {env:<16} {mb:>10} {m['avg_reward_mean']:>12.2f} "
                      f"{m['avg_reward_std']:>10.2f} {m['success_mean']:>10.2%} {m['num_trials']:>4}")

    plot_results(metrics, methods, output_dir)


if __name__ == "__main__":
    main()
