"""
Plot GA and DNS performance vs population size (grouped comparison).

Reads checkpoints from:
  - projects/gymnax/ga_pop_sweep/{env}/pop_{N}/trial_{T}/ga_{env}_best.pkl
  - projects/gymnax/dns_pop_sweep/{env}/pop_{N}/trial_{T}/dns_{env}_best.pkl

Creates a grouped barplot showing final_eval_mean for GA and DNS side by side.

Usage:
    python scripts/postprocess/plot_ga_dns_pop_sweep.py [--ga_dir DIR] [--dns_dir DIR] [--output FILE]
"""

import argparse
import os
import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


ENVS = ["CartPole-v1", "Acrobot-v1", "MountainCar-v0", "CheetahRun"]
POP_SIZES = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]

# Colors for each method
COLORS = {
    'GA': 'steelblue',
    'DNS': 'mediumseagreen'
}


def load_results(sweep_dir, method):
    """Load final_eval_mean from all checkpoints in the sweep directory.
    
    Args:
        sweep_dir: Root directory of the sweep
        method: 'ga' or 'dns' (used for checkpoint naming)
    """
    results = {}  # {env: {pop_size: [mean_rewards across trials]}}

    for env in ENVS:
        results[env] = {}
        for pop_size in POP_SIZES:
            trial_means = []
            for trial in range(1, 100):  # scan for available trials
                trial_dir = os.path.join(sweep_dir, env, f"pop_{pop_size}", f"trial_{trial}")
                ckpt_path = os.path.join(trial_dir, f"{method}_{env}_best.pkl")
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


def load_mujoco_results(sweep_dir, method):
    """Load results from mujoco sweep directory (flat layout with pop_N/trial_T/*.pkl).
    
    Args:
        sweep_dir: Root directory like projects/mujoco/ga_pop_sweep
        method: 'ga' or 'dns' (used for checkpoint naming)
    """
    results = {}  # {env: {pop_size: [mean_rewards across trials]}}
    
    if not os.path.isdir(sweep_dir):
        return results
    
    # Use CheetahRun as the env label
    env = "CheetahRun"
    results[env] = {}
    
    subdirs = os.listdir(sweep_dir) if os.path.isdir(sweep_dir) else []
    
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
            results[env][pop_size] = trial_means
    
    return results


def plot_grouped_barplot(ga_results, dns_results, output_path):
    """Create a grouped barplot comparing GA and DNS: x=pop_size, grouped bars=method."""
    fig, axes = plt.subplots(1, 4, figsize=(24, 5))
    bar_width = 0.35

    for ax, env in zip(axes, ENVS):
        # Get union of pop sizes from both methods
        ga_pops = set(ga_results.get(env, {}).keys())
        dns_pops = set(dns_results.get(env, {}).keys())
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
            if p in ga_results.get(env, {}):
                vals = ga_results[env][p]
                ga_means.append(np.mean(vals))
                ga_stds.append(np.std(vals))
                ga_counts.append(len(vals))
            else:
                ga_means.append(0)
                ga_stds.append(0)
                ga_counts.append(0)

        # DNS data
        dns_means = []
        dns_stds = []
        dns_counts = []
        for p in all_pops:
            if p in dns_results.get(env, {}):
                vals = dns_results[env][p]
                dns_means.append(np.mean(vals))
                dns_stds.append(np.std(vals))
                dns_counts.append(len(vals))
            else:
                dns_means.append(0)
                dns_stds.append(0)
                dns_counts.append(0)

        # Plot bars
        ga_bars = ax.bar(x - bar_width/2, ga_means, bar_width, yerr=ga_stds,
                         label='GA', color=COLORS['GA'], edgecolor='black', 
                         alpha=0.85, capsize=3)
        dns_bars = ax.bar(x + bar_width/2, dns_means, bar_width, yerr=dns_stds,
                          label='DNS', color=COLORS['DNS'], edgecolor='black',
                          alpha=0.85, capsize=3)

        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, rotation=45)
        ax.set_xlabel("Population Size")
        ax.set_ylabel("Final Eval Reward (mean ± std)")
        ax.set_title(env)
        ax.grid(axis='y', alpha=0.3)
        ax.legend(loc='best')

        # Annotate number of trials
        max_val = max(max(ga_means) if ga_means else 0, max(dns_means) if dns_means else 0)
        for i, p in enumerate(all_pops):
            # GA annotation
            if ga_counts[i] > 0:
                y_pos = ga_means[i] + ga_stds[i] + 0.02 * abs(max_val)
                ax.text(x[i] - bar_width/2, y_pos, f"n={ga_counts[i]}",
                        ha='center', va='bottom', fontsize=7, color='gray')
            # DNS annotation
            if dns_counts[i] > 0:
                y_pos = dns_means[i] + dns_stds[i] + 0.02 * abs(max_val)
                ax.text(x[i] + bar_width/2, y_pos, f"n={dns_counts[i]}",
                        ha='center', va='bottom', fontsize=7, color='gray')

    fig.suptitle("GA vs DNS Performance by Population Size (Non-Continual)", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved plot: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Plot GA vs DNS pop-size sweep comparison")
    parser.add_argument("--ga_dir", type=str,
                        default="projects/gymnax/ga_pop_sweep",
                        help="Root directory of GA sweep results (gymnax)")
    parser.add_argument("--dns_dir", type=str,
                        default="projects/gymnax/dns_pop_sweep",
                        help="Root directory of DNS sweep results (gymnax)")
    parser.add_argument("--ga_mujoco_dir", type=str,
                        default="projects/mujoco/ga_CheetahRun_pop_sweep",
                        help="Root directory of GA sweep results (mujoco/CheetahRun)")
    parser.add_argument("--dns_mujoco_dir", type=str,
                        default="projects/mujoco/dns_CheetahRun_pop_sweep",
                        help="Root directory of DNS sweep results (mujoco/CheetahRun)")
    parser.add_argument("--output", type=str,
                        default="projects/gymnax/ga_dns_pop_sweep_comparison.png",
                        help="Output plot file path")
    args = parser.parse_args()

    print("Loading GA results (gymnax)...")
    ga_results = load_results(args.ga_dir, 'ga')

    print("Loading DNS results (gymnax)...")
    dns_results = load_results(args.dns_dir, 'dns')

    # Load mujoco results for HalfCheetah
    print("Loading GA results (mujoco)...")
    ga_mujoco = load_mujoco_results(args.ga_mujoco_dir, 'ga')
    ga_results.update(ga_mujoco)

    print("Loading DNS results (mujoco)...")
    dns_mujoco = load_mujoco_results(args.dns_mujoco_dir, 'dns')
    dns_results.update(dns_mujoco)

    # Print summary table
    print(f"\n{'Method':<8} {'Env':<16} {'Pop':>6} {'Mean':>10} {'Std':>10} {'N':>4}")
    print("-" * 60)
    for method, results in [('GA', ga_results), ('DNS', dns_results)]:
        for env in ENVS:
            for pop_size in sorted(results.get(env, {}).keys()):
                vals = results[env][pop_size]
                print(f"{method:<8} {env:<16} {pop_size:>6} {np.mean(vals):>10.2f} {np.std(vals):>10.2f} {len(vals):>4}")

    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    plot_grouped_barplot(ga_results, dns_results, args.output)


if __name__ == "__main__":
    main()
