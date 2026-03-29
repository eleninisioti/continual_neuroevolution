"""
Barplot of KL divergence vs population size for PBT continual experiments.

Reads kl_divergence.json files produced by compute_kl_gymnax.py from:
  projects/gymnax/pbt_weights_only_{env}_continual/pop_{N}/trial_{t}/kl_divergence.json

For each pop size, averages KL across all task pairs and trials.

Usage:
    python scripts/postprocess/plot_kl_pbt_popsize.py --env Acrobot-v1
    python scripts/postprocess/plot_kl_pbt_popsize.py --env all
"""

import argparse
import os
import json
import glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


ENV_DIR_PATTERNS = {
    "CartPole-v1": "pbt_weights_only_CartPole_v1_continual",
    "MountainCar-v0": "pbt_weights_only_MountainCar_v0_continual",
    "Acrobot-v1": "pbt_weights_only_Acrobot_v1_continual",
}

ALL_ENVS = ["CartPole-v1", "MountainCar-v0", "Acrobot-v1"]


def collect_kl_for_env(data_dir, env):
    """Collect KL data for one env. Returns (pop_sizes, kl_means, kl_stds, trial_counts)."""
    env_dir = os.path.join(data_dir, ENV_DIR_PATTERNS[env])
    if not os.path.isdir(env_dir):
        print(f"  Skipping {env}: {env_dir} not found")
        return [], [], [], []

    pop_dirs = sorted(
        [d for d in os.listdir(env_dir) if d.startswith("pop_")],
        key=lambda d: int(d.replace("pop_", ""))
    )

    pop_sizes, kl_means, kl_stds, trial_counts = [], [], [], []

    for pop_dir_name in pop_dirs:
        pop_size = int(pop_dir_name.replace("pop_", ""))
        pop_path = os.path.join(env_dir, pop_dir_name)

        trial_avg_kls = []
        kl_files = glob.glob(os.path.join(pop_path, "trial_*", "kl_divergence.json"))

        for kl_file in kl_files:
            try:
                with open(kl_file, 'r') as f:
                    data = json.load(f)
            except Exception as e:
                print(f"  Warning: failed to load {kl_file}: {e}")
                continue

            pair_kls = []
            for pair_key, pair_data in data.items():
                kl_info = pair_data.get("task_i_noise", {})
                if "mean" in kl_info:
                    pair_kls.append(kl_info["mean"])

            if pair_kls:
                trial_avg_kls.append(np.mean(pair_kls))

        if trial_avg_kls:
            pop_sizes.append(pop_size)
            kl_means.append(np.mean(trial_avg_kls))
            kl_stds.append(np.std(trial_avg_kls))
            trial_counts.append(len(trial_avg_kls))
            print(f"  {env} pop_{pop_size}: {len(trial_avg_kls)} trials, "
                  f"mean KL = {np.mean(trial_avg_kls):.6f} ± {np.std(trial_avg_kls):.6f}")
        else:
            print(f"  {env} pop_{pop_size}: no kl_divergence.json files found")

    return pop_sizes, kl_means, kl_stds, trial_counts


def plot_single_env(data_dir, env, output_dir):
    pop_sizes, kl_means, kl_stds, trial_counts = collect_kl_for_env(data_dir, env)
    if not pop_sizes:
        print(f"No KL data for {env}.")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(pop_sizes))
    bars = ax.bar(x, kl_means, yerr=kl_stds, capsize=5,
                  color='orchid', edgecolor='black', alpha=0.85)

    for i, (bar, n) in enumerate(zip(bars, trial_counts)):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + kl_stds[i] + 0.002,
                f'n={n}', ha='center', va='bottom', fontsize=9)

    ax.set_xlabel("Population Size", fontsize=12)
    ax.set_ylabel("Mean KL Divergence", fontsize=12)
    ax.set_title(f"PBT KL Divergence vs Population Size — {env}", fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels([str(p) for p in pop_sizes], fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()

    output_path = os.path.join(output_dir, f"kl_pbt_popsize_{env.replace('-', '_')}.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_all_envs(data_dir, output_dir):
    env_data = {}
    for env in ALL_ENVS:
        pop_sizes, kl_means, kl_stds, trial_counts = collect_kl_for_env(data_dir, env)
        if pop_sizes:
            env_data[env] = (pop_sizes, kl_means, kl_stds, trial_counts)

    if not env_data:
        print("No KL data found for any environment.")
        return

    n_envs = len(env_data)
    fig, axes = plt.subplots(1, n_envs, figsize=(7 * n_envs, 5), squeeze=False)

    for col, (env, (pop_sizes, kl_means, kl_stds, trial_counts)) in enumerate(env_data.items()):
        ax = axes[0, col]
        x = np.arange(len(pop_sizes))
        bars = ax.bar(x, kl_means, yerr=kl_stds, capsize=5,
                      color='orchid', edgecolor='black', alpha=0.85)

        for i, (bar, n) in enumerate(zip(bars, trial_counts)):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + kl_stds[i] + 0.002,
                    f'n={n}', ha='center', va='bottom', fontsize=9)

        ax.set_xlabel("Population Size", fontsize=12)
        if col == 0:
            ax.set_ylabel("Mean KL Divergence", fontsize=12)
        ax.set_title(env, fontsize=13)
        ax.set_xticks(x)
        ax.set_xticklabels([str(p) for p in pop_sizes], fontsize=11)
        ax.grid(axis='y', alpha=0.3)

    fig.suptitle("PBT KL Divergence vs Population Size", fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()

    output_path = os.path.join(output_dir, "kl_pbt_popsize_all_envs.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Plot KL divergence vs pop size for PBT")
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
