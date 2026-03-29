"""
Plot DNS performance vs population size from the pop-size sweep.

Reads checkpoints from {sweep_dir}/{env}/pop_{N}/trial_{T}/dns_{env}_best.pkl
and creates a grouped barplot showing final_eval_mean for each env and pop size.

Usage:
    python scripts/postprocess/plot_dns_pop_sweep.py [--sweep_dir DIR] [--output FILE]
"""

import argparse
import os
import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


ENVS = ["CartPole-v1", "Acrobot-v1", "MountainCar-v0"]
POP_SIZES = [1, 2, 4, 8, 16, 32,64, 128, 256, 512]


def load_results(sweep_dir):
    """Load final_eval_mean from all checkpoints in the sweep directory."""
    results = {}  # {env: {pop_size: [mean_rewards across trials]}}

    for env in ENVS:
        results[env] = {}
        for pop_size in POP_SIZES:
            trial_means = []
            for trial in range(1, 100):  # scan for available trials
                trial_dir = os.path.join(sweep_dir, env, f"pop_{pop_size}", f"trial_{trial}")
                ckpt_path = os.path.join(trial_dir, f"dns_{env}_best.pkl")
                if not os.path.exists(ckpt_path):
                    if trial > 5:
                        break
                    continue
                try:
                    with open(ckpt_path, 'rb') as f:
                        data = pickle.load(f)
                    trial_means.append(data['final_eval_mean'])
                except Exception as e:
                    print(f"Warning: Failed to load {ckpt_path}: {e}")

            if trial_means:
                results[env][pop_size] = trial_means

    return results


def plot_barplot(results, output_path):
    """Create a grouped barplot: x=pop_size, bars=envs, y=final_eval_mean."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for ax, env in zip(axes, ENVS):
        if env not in results or not results[env]:
            ax.set_title(env)
            ax.text(0.5, 0.5, "No data", ha='center', va='center', transform=ax.transAxes)
            continue

        pop_sizes_found = sorted(results[env].keys())
        means = [np.mean(results[env][p]) for p in pop_sizes_found]
        stds = [np.std(results[env][p]) for p in pop_sizes_found]
        x_labels = [str(p) for p in pop_sizes_found]

        x = np.arange(len(pop_sizes_found))
        ax.bar(x, means, yerr=stds, capsize=4, color='mediumseagreen',
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
            ax.text(i, means[i] + stds[i] + 0.02 * max(abs(m) for m in means), f"n={n}",
                    ha='center', va='bottom', fontsize=8, color='gray')

    fig.suptitle("DNS Performance vs Population Size (Non-Continual)", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved plot: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Plot DNS pop-size sweep results")
    parser.add_argument("--sweep_dir", type=str,
                        default="projects/gymnax/dns_pop_sweep",
                        help="Root directory of the sweep results")
    parser.add_argument("--output", type=str,
                        default=None,
                        help="Output plot file path (default: {sweep_dir}/dns_pop_sweep_barplot.png)")
    args = parser.parse_args()

    output_path = args.output or os.path.join(args.sweep_dir, "dns_pop_sweep_barplot.png")

    results = load_results(args.sweep_dir)

    # Print summary table
    print(f"\n{'Env':<16} {'Pop':>6} {'Mean':>10} {'Std':>10} {'N':>4}")
    print("-" * 50)
    for env in ENVS:
        for pop_size in sorted(results.get(env, {}).keys()):
            vals = results[env][pop_size]
            print(f"{env:<16} {pop_size:>6} {np.mean(vals):>10.2f} {np.std(vals):>10.2f} {len(vals):>4}")

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    plot_barplot(results, output_path)


if __name__ == "__main__":
    main()
