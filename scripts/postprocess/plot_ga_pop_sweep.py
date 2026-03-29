"""
Plot GA performance vs population size from the pop-size sweep.

Reads checkpoints from projects/gymnax/ga_pop_sweep/{env}/pop_{N}/trial_{T}/ga_{env}_best.pkl
and creates a grouped barplot showing final_eval_mean for each env and pop size.

Usage:
    python scripts/postprocess/plot_ga_pop_sweep.py [--sweep_dir DIR] [--output FILE]
"""

import argparse
import os
import sys
import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


ENVS = ["CartPole-v1", "Acrobot-v1", "MountainCar-v0"]
POP_SIZES = [1, 2, 4, 8, 16, 32,64, 128, 256, 512]


def load_results(sweep_dir):
    """Load final_eval_mean from all checkpoints in the sweep directory.
    
    Supports two layouts:
      1. {sweep_dir}/{env}/pop_{N}/trial_{T}/ga_{env}_best.pkl  (gymnax)
      2. {sweep_dir}/pop_{N}/trial_{T}/*.pkl                    (mujoco, flat)
    """
    results = {}  # {env: {pop_size: [mean_rewards across trials]}}

    # Detect layout: check if sweep_dir contains pop_* directly
    subdirs = os.listdir(sweep_dir) if os.path.isdir(sweep_dir) else []
    has_pop_dirs = any(d.startswith("pop_") for d in subdirs)

    if has_pop_dirs:
        # Flat layout: no env subdirectory
        env_label = os.path.basename(sweep_dir)
        results[env_label] = {}
        for pop_dir in subdirs:
            if not pop_dir.startswith("pop_"):
                continue
            try:
                pop_size = int(pop_dir.split("_")[1])
            except ValueError:
                continue
            trial_means = []
            pop_path = os.path.join(sweep_dir, pop_dir)
            for trial in range(1, 100):
                trial_dir = os.path.join(pop_path, f"trial_{trial}")
                if not os.path.isdir(trial_dir):
                    if trial > 5:
                        break
                    continue
                # Find any .pkl file in trial dir
                pkl_files = [f for f in os.listdir(trial_dir) if f.endswith('.pkl')]
                if not pkl_files:
                    continue
                ckpt_path = os.path.join(trial_dir, pkl_files[0])
                try:
                    with open(ckpt_path, 'rb') as f:
                        data = pickle.load(f)
                    if 'final_eval_mean' in data:
                        trial_means.append(data['final_eval_mean'])
                    elif 'best_fitness' in data:
                        trial_means.append(float(data['best_fitness']))
                except Exception as e:
                    print(f"Warning: Failed to load {ckpt_path}: {e}")
            if trial_means:
                results[env_label][pop_size] = trial_means
    else:
        # Nested layout: env subdirectories
        for env in ENVS:
            results[env] = {}
            for pop_size in POP_SIZES:
                trial_means = []
                for trial in range(1, 100):
                    trial_dir = os.path.join(sweep_dir, env, f"pop_{pop_size}", f"trial_{trial}")
                    ckpt_path = os.path.join(trial_dir, f"ga_{env}_best.pkl")
                    if not os.path.exists(ckpt_path):
                        if trial > 5:
                            break
                        continue
                    try:
                        with open(ckpt_path, 'rb') as f:
                            data = pickle.load(f)
                        if 'final_eval_mean' in data:
                            trial_means.append(data['final_eval_mean'])
                        elif 'best_fitness' in data:
                            trial_means.append(float(data['best_fitness']))
                    except Exception as e:
                        print(f"Warning: Failed to load {ckpt_path}: {e}")
                if trial_means:
                    results[env][pop_size] = trial_means

    return results


def plot_barplot(results, output_path):
    """Create a grouped barplot: x=pop_size, bars=envs, y=final_eval_mean."""
    envs_with_data = [e for e in results if results[e]]
    n_envs = max(len(envs_with_data), 1)
    fig, axes = plt.subplots(1, n_envs, figsize=(6 * n_envs, 5), squeeze=False)
    axes = axes[0]

    for ax, env in zip(axes, envs_with_data):

        pop_sizes_found = sorted(results[env].keys())
        means = [np.mean(results[env][p]) for p in pop_sizes_found]
        stds = [np.std(results[env][p]) for p in pop_sizes_found]
        x_labels = [str(p) for p in pop_sizes_found]

        x = np.arange(len(pop_sizes_found))
        bars = ax.bar(x, means, yerr=stds, capsize=4, color='steelblue',
                      edgecolor='black', alpha=0.85)

        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, rotation=45)
        ax.set_xlabel("Population Size")
        ax.set_ylabel("Final Eval Reward (mean ± std)")
        ax.set_title(env)
        ax.grid(axis='y', alpha=0.3)

        # Annotate number of trials
        for i, p in enumerate(pop_sizes_found):
            n = len(results[env][p])
            ax.text(i, means[i] + stds[i] + 0.02 * max(means), f"n={n}",
                    ha='center', va='bottom', fontsize=8, color='gray')

    fig.suptitle("GA Performance vs Population Size (Non-Continual)", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved plot: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Plot GA pop-size sweep results")
    parser.add_argument("--sweep_dir", type=str,
                        default="projects/gymnax/ga_pop_sweep",
                        help="Root directory of the sweep results")
    parser.add_argument("--output", type=str,
                        default="projects/gymnax/ga_pop_sweep/pop_sweep_barplot.png",
                        help="Output plot file path")
    args = parser.parse_args()

    results = load_results(args.sweep_dir)

    # Print summary table
    print(f"\n{'Env':<16} {'Pop':>6} {'Mean':>10} {'Std':>10} {'N':>4}")
    print("-" * 50)
    for env in ENVS:
        for pop_size in sorted(results.get(env, {}).keys()):
            vals = results[env][pop_size]
            print(f"{env:<16} {pop_size:>6} {np.mean(vals):>10.2f} {np.std(vals):>10.2f} {len(vals):>4}")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    plot_barplot(results, args.output)


if __name__ == "__main__":
    main()
