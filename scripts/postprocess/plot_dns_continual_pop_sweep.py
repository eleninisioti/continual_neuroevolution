"""
Plot continual DNS performance vs population size from the pop-size sweep.

Reads per-task checkpoints from:
  projects/gymnax/dns_continual_pop_sweep/{env}/pop_{N}/trial_{T}/checkpoints/task_{K}.pkl

Creates two plots:
  1. Average reward across all 10 tasks vs population size (barplot)
  2. Success rate (fraction of tasks solved) vs population size (barplot)

Solved thresholds (standard gymnax):
  - CartPole-v1:    reward >= 475  (max 500)
  - Acrobot-v1:     reward >= -100 (higher is better towards 0)
  - MountainCar-v0: reward >= -110 (higher is better towards 0)

Usage:
    python scripts/postprocess/plot_dns_continual_pop_sweep.py
"""

import argparse
import os
import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


ENVS = ["CartPole-v1", "Acrobot-v1", "MountainCar-v0"]
POP_SIZES = [1, 2, 4, 8, 16, 32, 128, 256, 512]
NUM_TASKS = 10

# Standard "solved" thresholds
SOLVED_THRESHOLDS = {
    "CartPole-v1": 475.0,
    "Acrobot-v1": -70.0,
    "MountainCar-v0": -110.0,
}


def load_results(sweep_dir):
    """
    Load per-task best_fitness from all checkpoints.

    Returns:
        results: {env: {pop_size: [ [task_rewards for trial_1], [task_rewards for trial_2], ... ]}}
    """
    results = {}

    for env in ENVS:
        results[env] = {}
        for pop_size in POP_SIZES:
            trial_task_rewards = []
            for trial in range(1, 100):
                ckpts_dir = os.path.join(
                    sweep_dir, env, f"pop_{pop_size}", f"trial_{trial}", "checkpoints"
                )
                if not os.path.isdir(ckpts_dir):
                    if trial > 5:
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
                        task_rewards.append(data['best_fitness'])
                    except Exception as e:
                        print(f"Warning: Failed to load {ckpt_path}: {e}")

                if task_rewards:
                    trial_task_rewards.append(task_rewards)

            if trial_task_rewards:
                results[env][pop_size] = trial_task_rewards

    return results


def compute_metrics(results):
    """
    Compute per-pop-size metrics:
      - avg_reward: mean reward across all tasks and trials
      - success_rate: fraction of tasks solved (averaged across trials)
    """
    metrics = {}
    for env in ENVS:
        metrics[env] = {}
        threshold = SOLVED_THRESHOLDS[env]
        for pop_size in sorted(results.get(env, {}).keys()):
            trial_data = results[env][pop_size]  # list of lists

            # Average reward: mean across all tasks in all trials
            all_rewards = [r for trial in trial_data for r in trial]
            avg_reward_mean = np.mean(all_rewards)
            # Std across trial-level averages
            trial_avgs = [np.mean(trial) for trial in trial_data]
            avg_reward_std = np.std(trial_avgs)

            # Success rate per trial: fraction of tasks with reward >= threshold
            trial_success_rates = []
            for trial in trial_data:
                solved = sum(1 for r in trial if r >= threshold)
                trial_success_rates.append(solved / len(trial))
            success_mean = np.mean(trial_success_rates)
            success_std = np.std(trial_success_rates)

            metrics[env][pop_size] = {
                'avg_reward_mean': avg_reward_mean,
                'avg_reward_std': avg_reward_std,
                'success_mean': success_mean,
                'success_std': success_std,
                'num_trials': len(trial_data),
            }
    return metrics


def plot_results(metrics, output_dir):
    """Create barplots for average reward and success rate."""

    # --- Plot 1: Average reward across tasks ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, env in zip(axes, ENVS):
        if env not in metrics or not metrics[env]:
            ax.set_title(env)
            ax.text(0.5, 0.5, "No data", ha='center', va='center', transform=ax.transAxes)
            continue

        pop_sizes = sorted(metrics[env].keys())
        means = [metrics[env][p]['avg_reward_mean'] for p in pop_sizes]
        stds = [metrics[env][p]['avg_reward_std'] for p in pop_sizes]
        x = np.arange(len(pop_sizes))

        ax.bar(x, means, yerr=stds, capsize=4, color='mediumseagreen',
               edgecolor='black', alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels([str(p) for p in pop_sizes], rotation=45)
        ax.set_xlabel("Population Size")
        ax.set_ylabel("Avg Reward (across 10 tasks)")
        ax.set_title(env)
        ax.grid(axis='y', alpha=0.3)

        for i, p in enumerate(pop_sizes):
            n = metrics[env][p]['num_trials']
            ax.text(i, means[i] + stds[i] + 0.02 * max(abs(m) for m in means),
                    f"n={n}", ha='center', va='bottom', fontsize=8, color='gray')

    fig.suptitle("Continual DNS: Average Reward vs Population Size", fontsize=14, y=1.02)
    plt.tight_layout()
    path1 = os.path.join(output_dir, "dns_continual_pop_sweep_reward.png")
    plt.savefig(path1, dpi=150, bbox_inches='tight')
    print(f"Saved: {path1}")
    plt.close()

    # --- Plot 2: Success rate ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, env in zip(axes, ENVS):
        if env not in metrics or not metrics[env]:
            ax.set_title(env)
            ax.text(0.5, 0.5, "No data", ha='center', va='center', transform=ax.transAxes)
            continue

        pop_sizes = sorted(metrics[env].keys())
        means = [metrics[env][p]['success_mean'] for p in pop_sizes]
        stds = [metrics[env][p]['success_std'] for p in pop_sizes]
        x = np.arange(len(pop_sizes))

        ax.bar(x, means, yerr=stds, capsize=4, color='darkorange',
               edgecolor='black', alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels([str(p) for p in pop_sizes], rotation=45)
        ax.set_xlabel("Population Size")
        ax.set_ylabel("Success Rate (frac. of tasks solved)")
        ax.set_title(f"{env} (threshold={SOLVED_THRESHOLDS[env]})")
        ax.set_ylim(0, 1.05)
        ax.grid(axis='y', alpha=0.3)

        for i, p in enumerate(pop_sizes):
            n = metrics[env][p]['num_trials']
            ax.text(i, means[i] + stds[i] + 0.02,
                    f"n={n}", ha='center', va='bottom', fontsize=8, color='gray')

    fig.suptitle("Continual DNS: Task Success Rate vs Population Size", fontsize=14, y=1.02)
    plt.tight_layout()
    path2 = os.path.join(output_dir, "dns_continual_pop_sweep_success.png")
    plt.savefig(path2, dpi=150, bbox_inches='tight')
    print(f"Saved: {path2}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Plot continual DNS pop-size sweep results")
    parser.add_argument("--sweep_dir", type=str,
                        default="projects/gymnax/dns_continual_pop_sweep",
                        help="Root directory of the sweep results")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory for plots (default: same as sweep_dir)")
    args = parser.parse_args()

    output_dir = args.output_dir or args.sweep_dir
    os.makedirs(output_dir, exist_ok=True)

    results = load_results(args.sweep_dir)

    metrics = compute_metrics(results)

    # Print summary table
    print(f"\n{'Env':<16} {'Pop':>6} {'AvgReward':>12} {'Std':>10} {'Success':>10} {'N':>4}")
    print("-" * 60)
    for env in ENVS:
        for pop_size in sorted(metrics.get(env, {}).keys()):
            m = metrics[env][pop_size]
            print(f"{env:<16} {pop_size:>6} {m['avg_reward_mean']:>12.2f} "
                  f"{m['avg_reward_std']:>10.2f} {m['success_mean']:>10.2%} {m['num_trials']:>4}")

    plot_results(metrics, output_dir)


if __name__ == "__main__":
    main()
