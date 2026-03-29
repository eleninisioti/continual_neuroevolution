"""
Plot continual GA and DNS performance vs population size (grouped comparison).

Reads per-task checkpoints from:
  - projects/gymnax/ga_continual_pop_sweep/{env}/pop_{N}/trial_{T}/checkpoints/task_{K}.pkl
  - projects/gymnax/dns_continual_pop_sweep/{env}/pop_{N}/trial_{T}/checkpoints/task_{K}.pkl
  - projects/mujoco/ga_CheetahRun_continual_pop_sweep/pop_{N}/trial_{T}/checkpoints/task_{K}.pkl
  - projects/mujoco/dns_CheetahRun_continual_pop_sweep/pop_{N}/trial_{T}/checkpoints/task_{K}.pkl

Creates two grouped barplots comparing GA and DNS:
  1. Average reward across all tasks vs population size
  2. Success rate (fraction of tasks solved) vs population size

Solved thresholds:
  - CartPole-v1:    reward >= 475
  - Acrobot-v1:     reward >= -70
  - MountainCar-v0: reward >= -200
  - CheetahRun:     reward >= 3000

Usage:
    python scripts/postprocess/plot_ga_dns_continual_pop_sweep.py
"""

import argparse
import glob
import os
import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


ENVS = ["CartPole-v1", "Acrobot-v1", "MountainCar-v0", "CheetahRun"]
POP_SIZES = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
NUM_TASKS = 10
NUM_TASKS_MUJOCO = 30

# Standard "solved" thresholds
SOLVED_THRESHOLDS = {
    "CartPole-v1": 475.0,
    "Acrobot-v1": -70.0,
    "MountainCar-v0": -200.0,
    "CheetahRun": 3000.0,
}

# Colors for each method
COLORS = {
    'GA': 'steelblue',
    'DNS': 'mediumseagreen'
}


def load_results(sweep_dir, method):
    """
    Load per-task best_fitness from gymnax checkpoints.

    Returns:
        results: {env: {pop_size: [ [task_rewards for trial_1], [task_rewards for trial_2], ... ]}}
    """
    results = {}
    gymnax_envs = ["CartPole-v1", "Acrobot-v1", "MountainCar-v0"]

    for env in gymnax_envs:
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


def load_mujoco_results(sweep_dir, method):
    """
    Load per-task best_fitness from mujoco CheetahRun checkpoints (flat layout).

    Returns:
        results: {"CheetahRun": {pop_size: [ [task_rewards for trial_1], ... ]}}
    """
    results = {}
    env = "CheetahRun"
    results[env] = {}

    if not os.path.isdir(sweep_dir):
        return results

    subdirs = os.listdir(sweep_dir)
    
    for pop_dir in subdirs:
        if not pop_dir.startswith("pop_"):
            continue
        try:
            pop_size = int(pop_dir.split("_")[1])
        except ValueError:
            continue
        
        trial_task_rewards = []
        pop_path = os.path.join(sweep_dir, pop_dir)
        
        for trial in range(1, 100):
            trial_dir = os.path.join(pop_path, f"trial_{trial}")
            ckpts_dir = os.path.join(trial_dir, "checkpoints")
            
            if not os.path.isdir(ckpts_dir):
                if trial > 5:
                    break
                continue

            task_rewards = []
            for task_idx in range(NUM_TASKS_MUJOCO):
                # Try both naming patterns: task_0.pkl and task_00_friction_*.pkl
                ckpt_path = os.path.join(ckpts_dir, f"task_{task_idx}.pkl")
                if not os.path.exists(ckpt_path):
                    # Try mujoco naming pattern: task_00_friction_*.pkl
                    pattern = os.path.join(ckpts_dir, f"task_{task_idx:02d}_*.pkl")
                    matches = glob.glob(pattern)
                    if matches:
                        ckpt_path = matches[0]
                    else:
                        continue
                try:
                    with open(ckpt_path, 'rb') as f:
                        data = pickle.load(f)
                    # Try both keys: gymnax uses 'best_fitness', mujoco uses 'fitness'
                    if 'best_fitness' in data:
                        task_rewards.append(data['best_fitness'])
                    elif 'fitness' in data:
                        task_rewards.append(data['fitness'])
                    else:
                        print(f"Warning: No fitness key in {ckpt_path}")
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
        threshold = SOLVED_THRESHOLDS.get(env, 0)
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


def plot_grouped_results(ga_metrics, dns_metrics, output_dir):
    """Create grouped barplots comparing GA and DNS for reward and success rate."""
    bar_width = 0.35

    # --- Plot 1: Average reward across tasks ---
    fig, axes = plt.subplots(1, 4, figsize=(24, 5))
    for ax, env in zip(axes, ENVS):
        # Get union of pop sizes from both methods
        ga_pops = set(ga_metrics.get(env, {}).keys())
        dns_pops = set(dns_metrics.get(env, {}).keys())
        all_pops = sorted(ga_pops | dns_pops)
        
        if not all_pops:
            ax.set_title(env)
            ax.text(0.5, 0.5, "No data", ha='center', va='center', transform=ax.transAxes)
            continue

        x = np.arange(len(all_pops))
        x_labels = [str(p) for p in all_pops]

        # GA data
        ga_means = []
        ga_stds = []
        ga_counts = []
        for p in all_pops:
            if p in ga_metrics.get(env, {}):
                ga_means.append(ga_metrics[env][p]['avg_reward_mean'])
                ga_stds.append(ga_metrics[env][p]['avg_reward_std'])
                ga_counts.append(ga_metrics[env][p]['num_trials'])
            else:
                ga_means.append(0)
                ga_stds.append(0)
                ga_counts.append(0)

        # DNS data
        dns_means = []
        dns_stds = []
        dns_counts = []
        for p in all_pops:
            if p in dns_metrics.get(env, {}):
                dns_means.append(dns_metrics[env][p]['avg_reward_mean'])
                dns_stds.append(dns_metrics[env][p]['avg_reward_std'])
                dns_counts.append(dns_metrics[env][p]['num_trials'])
            else:
                dns_means.append(0)
                dns_stds.append(0)
                dns_counts.append(0)

        # Plot bars
        ax.bar(x - bar_width/2, ga_means, bar_width, yerr=ga_stds,
               label='GA', color=COLORS['GA'], edgecolor='black', 
               alpha=0.85, capsize=3)
        ax.bar(x + bar_width/2, dns_means, bar_width, yerr=dns_stds,
               label='DNS', color=COLORS['DNS'], edgecolor='black',
               alpha=0.85, capsize=3)

        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, rotation=45)
        ax.set_xlabel("Population Size")
        ax.set_ylabel("Avg Reward (across tasks)")
        ax.set_title(env)
        ax.grid(axis='y', alpha=0.3)
        ax.legend(loc='best')

        # Annotate number of trials
        max_val = max(max(ga_means) if ga_means else 0, max(dns_means) if dns_means else 0)
        for i, p in enumerate(all_pops):
            if ga_counts[i] > 0:
                y_pos = ga_means[i] + ga_stds[i] + 0.02 * abs(max_val)
                ax.text(x[i] - bar_width/2, y_pos, f"n={ga_counts[i]}",
                        ha='center', va='bottom', fontsize=7, color='gray')
            if dns_counts[i] > 0:
                y_pos = dns_means[i] + dns_stds[i] + 0.02 * abs(max_val)
                ax.text(x[i] + bar_width/2, y_pos, f"n={dns_counts[i]}",
                        ha='center', va='bottom', fontsize=7, color='gray')

    fig.suptitle("Continual GA vs DNS: Average Reward vs Population Size", fontsize=14, y=1.02)
    plt.tight_layout()
    path1 = os.path.join(output_dir, "ga_dns_continual_pop_sweep_reward.png")
    plt.savefig(path1, dpi=150, bbox_inches='tight')
    print(f"Saved: {path1}")
    plt.close()

    # --- Plot 2: Success rate ---
    fig, axes = plt.subplots(1, 4, figsize=(24, 5))
    for ax, env in zip(axes, ENVS):
        ga_pops = set(ga_metrics.get(env, {}).keys())
        dns_pops = set(dns_metrics.get(env, {}).keys())
        all_pops = sorted(ga_pops | dns_pops)
        
        if not all_pops:
            ax.set_title(env)
            ax.text(0.5, 0.5, "No data", ha='center', va='center', transform=ax.transAxes)
            continue

        x = np.arange(len(all_pops))
        x_labels = [str(p) for p in all_pops]

        # GA data
        ga_means = []
        ga_stds = []
        ga_counts = []
        for p in all_pops:
            if p in ga_metrics.get(env, {}):
                ga_means.append(ga_metrics[env][p]['success_mean'])
                ga_stds.append(ga_metrics[env][p]['success_std'])
                ga_counts.append(ga_metrics[env][p]['num_trials'])
            else:
                ga_means.append(0)
                ga_stds.append(0)
                ga_counts.append(0)

        # DNS data
        dns_means = []
        dns_stds = []
        dns_counts = []
        for p in all_pops:
            if p in dns_metrics.get(env, {}):
                dns_means.append(dns_metrics[env][p]['success_mean'])
                dns_stds.append(dns_metrics[env][p]['success_std'])
                dns_counts.append(dns_metrics[env][p]['num_trials'])
            else:
                dns_means.append(0)
                dns_stds.append(0)
                dns_counts.append(0)

        # Plot bars
        ax.bar(x - bar_width/2, ga_means, bar_width, yerr=ga_stds,
               label='GA', color=COLORS['GA'], edgecolor='black', 
               alpha=0.85, capsize=3)
        ax.bar(x + bar_width/2, dns_means, bar_width, yerr=dns_stds,
               label='DNS', color=COLORS['DNS'], edgecolor='black',
               alpha=0.85, capsize=3)

        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, rotation=45)
        ax.set_xlabel("Population Size")
        ax.set_ylabel("Success Rate (frac. of tasks solved)")
        threshold = SOLVED_THRESHOLDS.get(env, 0)
        ax.set_title(f"{env} (threshold={threshold})")
        ax.set_ylim(0, 1.05)
        ax.grid(axis='y', alpha=0.3)
        ax.legend(loc='best')

        for i, p in enumerate(all_pops):
            if ga_counts[i] > 0:
                ax.text(x[i] - bar_width/2, ga_means[i] + ga_stds[i] + 0.02,
                        f"n={ga_counts[i]}", ha='center', va='bottom', fontsize=7, color='gray')
            if dns_counts[i] > 0:
                ax.text(x[i] + bar_width/2, dns_means[i] + dns_stds[i] + 0.02,
                        f"n={dns_counts[i]}", ha='center', va='bottom', fontsize=7, color='gray')

    fig.suptitle("Continual GA vs DNS: Task Success Rate vs Population Size", fontsize=14, y=1.02)
    plt.tight_layout()
    path2 = os.path.join(output_dir, "ga_dns_continual_pop_sweep_success.png")
    plt.savefig(path2, dpi=150, bbox_inches='tight')
    print(f"Saved: {path2}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Plot continual GA vs DNS pop-size sweep comparison")
    parser.add_argument("--ga_dir", type=str,
                        default="projects/gymnax/ga_continual_pop_sweep",
                        help="Root directory of GA sweep results (gymnax)")
    parser.add_argument("--dns_dir", type=str,
                        default="projects/gymnax/dns_continual_pop_sweep",
                        help="Root directory of DNS sweep results (gymnax)")
    parser.add_argument("--ga_mujoco_dir", type=str,
                        default="projects/mujoco/ga_CheetahRun_continual_pop_sweep",
                        help="Root directory of GA sweep results (mujoco/CheetahRun)")
    parser.add_argument("--dns_mujoco_dir", type=str,
                        default="projects/mujoco/dns_CheetahRun_continual_pop_sweep",
                        help="Root directory of DNS sweep results (mujoco/CheetahRun)")
    parser.add_argument("--output_dir", type=str,
                        default="projects/gymnax",
                        help="Output directory for plots")
    args = parser.parse_args()

    print("Loading GA results (gymnax)...")
    ga_results = load_results(args.ga_dir, 'ga')

    print("Loading DNS results (gymnax)...")
    dns_results = load_results(args.dns_dir, 'dns')

    # Load mujoco results for CheetahRun
    print("Loading GA results (mujoco/CheetahRun)...")
    ga_mujoco = load_mujoco_results(args.ga_mujoco_dir, 'ga')
    ga_results.update(ga_mujoco)

    print("Loading DNS results (mujoco/CheetahRun)...")
    dns_mujoco = load_mujoco_results(args.dns_mujoco_dir, 'dns')
    dns_results.update(dns_mujoco)

    # Compute metrics
    print("Computing metrics...")
    ga_metrics = compute_metrics(ga_results)
    dns_metrics = compute_metrics(dns_results)

    # Print summary table
    print(f"\n{'Method':<8} {'Env':<16} {'Pop':>6} {'AvgRwd':>10} {'Std':>8} {'Success':>8} {'N':>4}")
    print("-" * 70)
    for method, metrics in [('GA', ga_metrics), ('DNS', dns_metrics)]:
        for env in ENVS:
            for pop_size in sorted(metrics.get(env, {}).keys()):
                m = metrics[env][pop_size]
                print(f"{method:<8} {env:<16} {pop_size:>6} {m['avg_reward_mean']:>10.2f} "
                      f"{m['avg_reward_std']:>8.2f} {m['success_mean']:>8.2%} {m['num_trials']:>4}")

    os.makedirs(args.output_dir, exist_ok=True)
    plot_grouped_results(ga_metrics, dns_metrics, args.output_dir)


if __name__ == "__main__":
    main()
