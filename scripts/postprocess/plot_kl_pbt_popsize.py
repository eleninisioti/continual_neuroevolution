"""
Barplot of KL divergence vs population size for PBT, GA, DNS continual experiments.

Reads kl_divergence.json files produced by compute_kl_gymnax.py from:
  projects/gymnax/pbt_weights_only_{env}_continual/pop_{N}/trial_{t}/kl_divergence.json
  projects/gymnax/ga_continual_pop_sweep/{env}/pop_{N}/trial_{t}/kl_divergence.json
  projects/gymnax/dns_continual_pop_sweep/{env}/pop_{N}/trial_{t}/kl_divergence.json

Grouped barplots: x-axis = pop size, one bar per method (PBT, GA, DNS).

Usage:
    python scripts/postprocess/plot_kl_pbt_popsize.py --env all
    python scripts/postprocess/plot_kl_pbt_popsize.py --env Acrobot-v1
"""

import argparse
import os
import json
import glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


PBT_DIR_PATTERNS = {
    "CartPole-v1": "pbt_weights_only_CartPole_v1_continual",
    "MountainCar-v0": "pbt_weights_only_MountainCar_v0_continual",
    "Acrobot-v1": "pbt_weights_only_Acrobot_v1_continual",
}

ALL_ENVS = ["CartPole-v1", "MountainCar-v0", "Acrobot-v1"]
METHODS = ["PBT", "GA", "DNS"]
COLORS = {"PBT": "orchid", "GA": "steelblue", "DNS": "mediumseagreen"}


def get_method_env_dir(data_dir, method, env):
    """Return the base directory for a method/env combination."""
    if method == "PBT":
        return os.path.join(data_dir, PBT_DIR_PATTERNS[env])
    elif method == "GA":
        return os.path.join(data_dir, "ga_continual_pop_sweep", env)
    else:  # DNS
        return os.path.join(data_dir, "dns_continual_pop_sweep", env)


def collect_kl_from_dir(base_dir):
    """
    Collect KL data from a method/env directory.
    Returns dict: {pop_size: {"mean": float, "std": float, "n": int}}
    """
    if not os.path.isdir(base_dir):
        return {}

    pop_dirs = sorted(
        [d for d in os.listdir(base_dir) if d.startswith("pop_")],
        key=lambda d: int(d.replace("pop_", ""))
    )

    result = {}
    for pop_dir_name in pop_dirs:
        pop_size = int(pop_dir_name.replace("pop_", ""))
        pop_path = os.path.join(base_dir, pop_dir_name)

        trial_avg_kls = []
        kl_files = glob.glob(os.path.join(pop_path, "trial_*", "kl_divergence.json"))

        for kl_file in kl_files:
            try:
                with open(kl_file, 'r') as f:
                    data = json.load(f)
            except Exception:
                continue

            pair_kls = []
            for pair_data in data.values():
                kl_info = pair_data.get("task_i_noise", {})
                if "mean" in kl_info:
                    pair_kls.append(kl_info["mean"])

            if pair_kls:
                trial_avg_kls.append(np.mean(pair_kls))

        if trial_avg_kls:
            result[pop_size] = {
                "mean": np.mean(trial_avg_kls),
                "std": np.std(trial_avg_kls),
                "n": len(trial_avg_kls),
            }

    return result


def plot_env(ax, data_dir, env):
    """Plot grouped bars for one env on a given axis."""
    # Collect data for all methods
    method_data = {}
    all_pop_sizes = set()
    for method in METHODS:
        base_dir = get_method_env_dir(data_dir, method, env)
        kl_data = collect_kl_from_dir(base_dir)
        method_data[method] = kl_data
        all_pop_sizes.update(kl_data.keys())
        for pop, d in sorted(kl_data.items()):
            print(f"  {method} {env} pop_{pop}: n={d['n']}, KL={d['mean']:.6f}±{d['std']:.6f}")

    if not all_pop_sizes:
        ax.set_title(env)
        ax.text(0.5, 0.5, "No data", ha='center', va='center', transform=ax.transAxes)
        return

    pop_sizes = sorted(all_pop_sizes)
    n_methods = len(METHODS)
    bar_width = 0.25
    x = np.arange(len(pop_sizes))

    for i, method in enumerate(METHODS):
        means = []
        stds = []
        for pop in pop_sizes:
            d = method_data[method].get(pop)
            if d:
                means.append(d["mean"])
                stds.append(d["std"])
            else:
                means.append(np.nan)
                stds.append(np.nan)

        offset = (i - (n_methods - 1) / 2) * bar_width
        ax.bar(x + offset, means, bar_width, yerr=stds, capsize=3,
               color=COLORS[method], edgecolor='black', label=method, alpha=0.85)

    ax.set_xlabel("Population Size", fontsize=11)
    ax.set_ylabel("Mean KL Divergence", fontsize=11)
    ax.set_title(env, fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels([str(p) for p in pop_sizes], fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)


def plot_all_envs(data_dir, output_dir):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for col, env in enumerate(ALL_ENVS):
        print(f"\n{env}:")
        plot_env(axes[col], data_dir, env)

    fig.suptitle("KL Divergence vs Population Size — PBT / GA / DNS",
                 fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()

    output_path = os.path.join(output_dir, "kl_all_methods_popsize.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved: {output_path}")
    plt.close()


def plot_single_env(data_dir, env, output_dir):
    fig, ax = plt.subplots(figsize=(8, 5))
    print(f"\n{env}:")
    plot_env(ax, data_dir, env)
    plt.tight_layout()

    output_path = os.path.join(output_dir, f"kl_all_methods_{env.replace('-', '_')}.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Plot KL divergence vs pop size (PBT/GA/DNS)")
    parser.add_argument("--env", type=str, required=True,
                        choices=["CartPole-v1", "MountainCar-v0", "Acrobot-v1", "all"])
    parser.add_argument("--data_dir", type=str, default="projects/gymnax")
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = args.data_dir

    if args.env == "all":
        plot_all_envs(args.data_dir, args.output_dir)
    else:
        plot_single_env(args.data_dir, args.env, args.output_dir)


if __name__ == "__main__":
    main()
