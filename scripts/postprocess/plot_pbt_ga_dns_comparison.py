"""
Compare PBT (weights_only) vs GA vs DNS for CartPole and MountainCar
across multiple population sizes.

Loads results from:
  - projects/gymnax/pbt_weights_only_{env}_continual/pop_{N}/
  - projects/gymnax/ga_continual_pop_sweep/{env}/pop_{N}/
  - projects/gymnax/dns_continual_pop_sweep/{env}/pop_{N}/

Creates grouped barplots (grouped by pop_size) showing:
  1. Average reward across all tasks
  2. Success rate (fraction of tasks solved)

Usage:
    python scripts/postprocess/plot_pbt_ga_dns_comparison.py
"""

import argparse
import os
import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


NUM_TASKS = 10
POP_SIZES = [2, 4, 8, 16]

# Standard "solved" thresholds
SOLVED_THRESHOLDS = {
    "CartPole-v1": 475.0,
    "MountainCar-v0": -200.0,
    "Acrobot-v1": -70.0,
}

# PBT directory naming patterns
PBT_DIR_PATTERNS = {
    "CartPole-v1": "pbt_weights_only_CartPole_v1_continual",
    "MountainCar-v0": "pbt_weights_only_MountainCar_v0_continual",
    "Acrobot-v1": "pbt_weights_only_Acrobot_v1_continual",
}

# Colors for each method
COLORS = {
    'PBT': 'orchid',
    'GA': 'steelblue',
    'DNS': 'mediumseagreen',
}

METHODS_ORDER = ['PBT', 'GA', 'DNS']


def load_trials_from_dir(base_dir, reward_keys=('best_fitness',)):
    """
    Load per-task rewards from trial subdirectories under base_dir.

    Returns:
        list of lists: [[task_rewards for trial_1], ...]
    """
    trial_task_rewards = []

    for trial in range(1, 100):
        trial_dir = os.path.join(base_dir, f"trial_{trial}")
        if not os.path.isdir(trial_dir):
            if trial > 10:
                break
            continue

        ckpts_dir = os.path.join(trial_dir, "checkpoints")
        if not os.path.isdir(ckpts_dir):
            continue

        task_rewards = []
        for task_idx in range(NUM_TASKS):
            ckpt_path = os.path.join(ckpts_dir, f"task_{task_idx}.pkl")
            if not os.path.exists(ckpt_path):
                continue
            try:
                with open(ckpt_path, 'rb') as f:
                    data = pickle.load(f)
                for key in reward_keys:
                    if key in data:
                        task_rewards.append(data[key])
                        break
            except Exception as e:
                print(f"Warning: Failed to load {ckpt_path}: {e}")

        if task_rewards:
            trial_task_rewards.append(task_rewards)

    return trial_task_rewards if trial_task_rewards else None


def load_results(data_dir, env, pop_size, method):
    """
    Load per-task rewards for a given method, env, and pop_size.
    """
    if method == 'PBT':
        base = os.path.join(data_dir, PBT_DIR_PATTERNS[env], f"pop_{pop_size}")
        reward_keys = ('best_reward', 'best_fitness')
    elif method == 'GA':
        base = os.path.join(data_dir, "ga_continual_pop_sweep", env, f"pop_{pop_size}")
        reward_keys = ('best_fitness',)
    else:  # DNS
        base = os.path.join(data_dir, "dns_continual_pop_sweep", env, f"pop_{pop_size}")
        reward_keys = ('best_fitness',)

    if not os.path.isdir(base):
        print(f"  Warning: {method} pop_{pop_size} not found: {base}")
        return None

    return load_trials_from_dir(base, reward_keys)


def compute_metrics(task_rewards_list, threshold):
    """
    Compute metrics from list of trial task rewards.

    Returns:
        dict with avg_reward_mean, avg_reward_std, success_mean, success_std, num_trials
    """
    if not task_rewards_list:
        return None

    # Average reward: mean across all tasks in all trials
    all_rewards = [r for trial in task_rewards_list for r in trial]
    avg_reward_mean = np.mean(all_rewards)
    # Std across trial-level averages
    trial_avgs = [np.mean(trial) for trial in task_rewards_list]
    avg_reward_std = np.std(trial_avgs)

    # Success rate per trial: fraction of tasks with reward >= threshold
    trial_success_rates = []
    for trial in task_rewards_list:
        solved = sum(1 for r in trial if r >= threshold)
        trial_success_rates.append(solved / len(trial))
    success_mean = np.mean(trial_success_rates)
    success_std = np.std(trial_success_rates)

    return {
        'avg_reward_mean': avg_reward_mean,
        'avg_reward_std': avg_reward_std,
        'success_mean': success_mean,
        'success_std': success_std,
        'num_trials': len(task_rewards_list),
    }


def main():
    parser = argparse.ArgumentParser(description="Plot PBT vs GA vs DNS comparison")
    parser.add_argument("--data_dir", type=str,
                        default="projects/gymnax",
                        help="Directory containing experiment data")
    parser.add_argument("--output_dir", type=str,
                        default=None,
                        help="Output directory for plots")
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = args.data_dir

    envs = ["CartPole-v1", "MountainCar-v0", "Acrobot-v1"]

    # Collect all data: all_data[env][pop_size][method] = metrics
    all_data = {}
    for env in envs:
        print(f"\nLoading data for {env}...")
        all_data[env] = {}
        threshold = SOLVED_THRESHOLDS[env]

        for pop_size in POP_SIZES:
            all_data[env][pop_size] = {}
            for method in METHODS_ORDER:
                results = load_results(args.data_dir, env, pop_size, method)
                if results:
                    metrics = compute_metrics(results, threshold)
                    if metrics:
                        all_data[env][pop_size][method] = metrics
                        print(f"  {method} pop_{pop_size}: {metrics['num_trials']} trials")

    # Create plots: one column per env, two rows (avg reward, success rate)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    n_methods = len(METHODS_ORDER)
    n_groups = len(POP_SIZES)
    bar_width = 0.25
    group_width = n_methods * bar_width

    for col, env in enumerate(envs):
        # Check if any data exists
        has_data = any(all_data[env][p] for p in POP_SIZES)
        if not has_data:
            for row in range(2):
                axes[row, col].set_title(env)
                axes[row, col].text(0.5, 0.5, "No data", ha='center', va='center',
                                     transform=axes[row, col].transAxes)
            continue

        x = np.arange(n_groups)  # group positions

        # --- Row 0: Average reward ---
        ax = axes[0, col]
        for i, method in enumerate(METHODS_ORDER):
            means = []
            stds = []
            for pop_size in POP_SIZES:
                m = all_data[env][pop_size].get(method)
                if m:
                    means.append(m['avg_reward_mean'])
                    stds.append(m['avg_reward_std'])
                else:
                    means.append(np.nan)
                    stds.append(np.nan)

            offset = (i - (n_methods - 1) / 2) * bar_width
            bars = ax.bar(x + offset, means, bar_width, yerr=stds, capsize=3,
                          color=COLORS[method], edgecolor='black', label=method)

        ax.set_xlabel("Population Size", fontsize=11)
        ax.set_ylabel("Average Reward", fontsize=11)
        ax.set_title(f"{env}", fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels([str(p) for p in POP_SIZES], fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(axis='y', alpha=0.3)

        # --- Row 1: Success rate ---
        ax = axes[1, col]
        for i, method in enumerate(METHODS_ORDER):
            means = []
            stds = []
            has_data_mask = []
            for pop_size in POP_SIZES:
                m = all_data[env][pop_size].get(method)
                if m:
                    means.append(m['success_mean'] * 100)
                    stds.append(m['success_std'] * 100)
                    has_data_mask.append(True)
                else:
                    means.append(np.nan)
                    stds.append(np.nan)
                    has_data_mask.append(False)

            offset = (i - (n_methods - 1) / 2) * bar_width
            bars = ax.bar(x + offset, means, bar_width, yerr=stds, capsize=3,
                          color=COLORS[method], edgecolor='black', label=method)

            # Draw a visible line at y=0 for methods that have data but zero success
            for j, (val, has) in enumerate(zip(means, has_data_mask)):
                if has and (val == 0 or np.isclose(val, 0)):
                    xpos = x[j] + offset
                    ax.plot([xpos - bar_width/2, xpos + bar_width/2], [0, 0],
                            color=COLORS[method], linewidth=3, solid_capstyle='round')

        ax.set_xlabel("Population Size", fontsize=11)
        ax.set_ylabel("Success Rate (%)", fontsize=11)
        ax.set_title(f"{env}", fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels([str(p) for p in POP_SIZES], fontsize=10)
        ax.set_ylim(0, 105)
        ax.legend(fontsize=9)
        ax.grid(axis='y', alpha=0.3)

    fig.suptitle("PBT vs GA vs DNS: Continual Learning Comparison",
                 fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()

    output_path = os.path.join(args.output_dir, "pbt_ga_dns_comparison.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved: {output_path}")
    plt.close()


if __name__ == "__main__":
    main()
