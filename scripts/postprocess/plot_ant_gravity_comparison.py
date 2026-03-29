"""
Plot fitness and dormant neurons for PPO/ReDo/TRAC on Ant with random gravity.

Reads CSV exports from wandb and creates two side-by-side line plots.

Usage:
    python scripts/postprocess/plot_ant_gravity_comparison.py
"""

import argparse
import os
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description="Plot Ant random gravity comparison")
    parser.add_argument("--data_dir", type=str,
                        default="projects/brax/easy_ant",
                        help="Directory containing CSV files")
    parser.add_argument("--output", type=str,
                        default=None,
                        help="Output plot file path")
    args = parser.parse_args()

    # Load all CSV files
    csv_files = sorted([f for f in os.listdir(args.data_dir) if f.endswith('.csv')])
    dfs = [pd.read_csv(os.path.join(args.data_dir, f)) for f in csv_files]

    # Separate into fitness and dormant dataframes
    fitness_dfs = []
    dormant_dfs = []
    for df in dfs:
        data_col = [c for c in df.columns if c != 'Step'][0]
        if 'fitness' in data_col.lower():
            fitness_dfs.append(df)
        else:
            dormant_dfs.append(df)

    methods = ['TRAC', 'ReDo', 'PPO']
    prefixes = {'TRAC': 'trac_', 'ReDo': 'redo_', 'PPO': 'ppo_'}
    colors = {'TRAC': 'tab:blue', 'ReDo': 'tab:orange', 'PPO': 'tab:green'}

    def get_trial_columns(dfs_list, method):
        """Get all trial value columns for a method across all dataframes."""
        prefix = prefixes[method]
        cols_with_df = []
        for df in dfs_list:
            for c in df.columns:
                if c.lower().startswith(prefix) and 'MIN' not in c and 'MAX' not in c and c != 'Step':
                    cols_with_df.append((df, c))
        return cols_with_df

    def compute_stats(dfs_list, method):
        """Compute mean and 95% CI across trials for a method, aligning on Step."""
        trials = get_trial_columns(dfs_list, method)
        if not trials:
            return None, None, None, None

        # Collect (step, values) from each trial
        all_series = []
        for df, col in trials:
            s = df[['Step', col]].dropna().set_index('Step')[col]
            all_series.append(s)

        # Align on common steps
        combined = pd.concat(all_series, axis=1)
        steps = combined.index.values
        values = combined.values  # shape: (n_steps, n_trials)

        mean = np.nanmean(values, axis=1)
        n = np.sum(~np.isnan(values), axis=1)
        std = np.nanstd(values, axis=1, ddof=0)
        se = std / np.sqrt(np.maximum(n, 1))
        ci_low = mean - 1.96 * se
        ci_high = mean + 1.96 * se

        return steps, mean, ci_low, ci_high

    # Create figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Fitness
    ax1 = axes[0]
    for method in methods:
        steps, mean, ci_low, ci_high = compute_stats(fitness_dfs, method)
        if steps is not None:
            ax1.plot(steps, mean, label=method, color=colors[method], linewidth=2)
            ax1.fill_between(steps, ci_low, ci_high, color=colors[method], alpha=0.2)

    ax1.set_xlabel("Step", fontsize=12)
    ax1.set_ylabel("Best Fitness", fontsize=12)
    ax1.set_title("Fitness vs Training Step", fontsize=14)
    ax1.legend(loc='best')
    ax1.grid(alpha=0.3)

    # Plot 2: Dormant neurons
    ax2 = axes[1]
    for method in methods:
        steps, mean, ci_low, ci_high = compute_stats(dormant_dfs, method)
        if steps is not None:
            ax2.plot(steps, mean, label=method, color=colors[method], linewidth=2)
            ax2.fill_between(steps, ci_low, ci_high, color=colors[method], alpha=0.2)

    ax2.set_xlabel("Step", fontsize=12)
    ax2.set_ylabel("Dormant Neurons (%)", fontsize=12)
    ax2.set_title("Dormant Neuron Percentage vs Training Step", fontsize=14)
    ax2.legend(loc='best')
    ax2.grid(alpha=0.3)

    fig.suptitle("Ant Random Gravity: PPO vs ReDo vs TRAC", fontsize=16, y=1.02)
    plt.tight_layout()

    output_path = args.output or os.path.join(args.data_dir, "ant_gravity_comparison.png")
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


if __name__ == "__main__":
    main()
