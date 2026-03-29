#!/usr/bin/env python3
"""
Compute and visualize pairwise distances between best solutions across tasks
for PBT continual runs.

Loads per-task checkpoints (task_0.pkl, task_1.pkl, ...), flattens the policy
params to vectors, and produces:
  1. A pairwise Euclidean distance heatmap between task solutions
  2. A cosine similarity heatmap
  3. Distance-from-initial bar chart (drift from task 0)

Usage:
    python scripts/postprocess/pairwise_distance_heatmap.py \
        --project_dir projects/gymnax/pbt_full_CartPole_v1_continual/trial_1

    # Process all trials for an env:
    python scripts/postprocess/pairwise_distance_heatmap.py \
        --project_dir projects/gymnax/pbt_full_CartPole_v1_continual \
        --all_trials
"""

import argparse
import os
import sys
import glob
import pickle

import numpy as np
import jax
import jax.numpy as jnp
import jax.flatten_util
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import cosine


def flatten_params(params):
    """Flatten a Flax param tree to a 1D numpy array."""
    flat, _ = jax.flatten_util.ravel_pytree(params)
    return np.array(flat)


def load_task_checkpoints(project_dir):
    """Load all task_*.pkl checkpoints from a project directory."""
    ckpt_dir = os.path.join(project_dir, "checkpoints")
    if not os.path.isdir(ckpt_dir):
        raise FileNotFoundError(f"No checkpoints/ directory in {project_dir}")

    files = sorted(
        [f for f in os.listdir(ckpt_dir) if f.startswith("task_") and f.endswith(".pkl")],
        key=lambda f: int(f.replace("task_", "").replace(".pkl", ""))
    )
    if not files:
        raise FileNotFoundError(f"No task_*.pkl files in {ckpt_dir}")

    checkpoints = []
    for fname in files:
        path = os.path.join(ckpt_dir, fname)
        with open(path, 'rb') as f:
            ckpt = pickle.load(f)
        task_idx = ckpt.get('task_idx', int(fname.replace("task_", "").replace(".pkl", "")))
        checkpoints.append({
            'task_idx': task_idx,
            'params': ckpt['policy_params'],
            'best_reward': ckpt.get('best_reward', None),
            'generation': ckpt.get('generation', None),
            'noise_vector': ckpt.get('noise_vector', None),
        })
        print(f"  Loaded {fname}: task={task_idx}, reward={ckpt.get('best_reward', '?')}")

    return checkpoints


def compute_distances(flat_vectors):
    """Compute pairwise Euclidean distance and cosine similarity matrices."""
    euclidean = pairwise_distances(flat_vectors, metric='euclidean')
    n = len(flat_vectors)
    cosine_sim = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            cosine_sim[i, j] = 1.0 - cosine(flat_vectors[i], flat_vectors[j])
    return euclidean, cosine_sim


def plot_heatmaps(euclidean, cosine_sim, task_indices, rewards, save_dir, title_suffix=""):
    """Create heatmap figures and save them."""
    n = len(task_indices)
    tick_labels = [f"Task {t}" for t in task_indices]

    fig, axes = plt.subplots(1, 3, figsize=(22, 6))

    # 1. Euclidean distance heatmap
    ax1 = axes[0]
    im1 = ax1.imshow(euclidean, cmap='YlOrRd', interpolation='nearest')
    ax1.set_title('Pairwise Euclidean Distance', fontsize=14)
    ax1.set_xticks(range(n))
    ax1.set_yticks(range(n))
    ax1.set_xticklabels(tick_labels, rotation=45, ha='right')
    ax1.set_yticklabels(tick_labels)
    # Annotate cells
    for i in range(n):
        for j in range(n):
            ax1.text(j, i, f"{euclidean[i, j]:.2f}", ha='center', va='center',
                     fontsize=8, color='white' if euclidean[i, j] > euclidean.max() * 0.6 else 'black')
    plt.colorbar(im1, ax=ax1, shrink=0.8)

    # 2. Cosine similarity heatmap
    ax2 = axes[1]
    im2 = ax2.imshow(cosine_sim, cmap='RdYlBu', interpolation='nearest', vmin=-1, vmax=1)
    ax2.set_title('Pairwise Cosine Similarity', fontsize=14)
    ax2.set_xticks(range(n))
    ax2.set_yticks(range(n))
    ax2.set_xticklabels(tick_labels, rotation=45, ha='right')
    ax2.set_yticklabels(tick_labels)
    for i in range(n):
        for j in range(n):
            ax2.text(j, i, f"{cosine_sim[i, j]:.3f}", ha='center', va='center',
                     fontsize=8, color='white' if cosine_sim[i, j] < 0.3 else 'black')
    plt.colorbar(im2, ax=ax2, shrink=0.8)

    # 3. Drift from task 0
    ax3 = axes[2]
    drift = euclidean[0, :]  # distance from task 0 to each other task
    colors = plt.cm.YlOrRd(drift / (drift.max() + 1e-8))
    bars = ax3.bar(range(n), drift, color=colors, edgecolor='black', linewidth=0.5)
    ax3.set_title('Drift from Initial Solution (Task 0)', fontsize=14)
    ax3.set_xlabel('Task', fontsize=12)
    ax3.set_ylabel('Euclidean Distance', fontsize=12)
    ax3.set_xticks(range(n))
    ax3.set_xticklabels(tick_labels, rotation=45, ha='right')
    ax3.grid(True, alpha=0.3, axis='y')

    # Add reward annotations on drift bars if available
    if rewards and any(r is not None for r in rewards):
        for i, (r, d) in enumerate(zip(rewards, drift)):
            if r is not None:
                ax3.text(i, d + drift.max() * 0.02, f"r={r:.0f}", ha='center',
                         va='bottom', fontsize=7, color='blue')

    plt.suptitle(f'Pairwise Solution Distances Across Tasks{title_suffix}', fontsize=16, y=1.02)
    plt.tight_layout()
    out_path = os.path.join(save_dir, f'pairwise_distance_heatmap{title_suffix.replace(" ", "_")}.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")

    # 4. Consecutive-task distance plot
    if n > 1:
        fig2, ax = plt.subplots(figsize=(8, 5))
        consecutive_dists = [euclidean[i, i + 1] for i in range(n - 1)]
        x = range(len(consecutive_dists))
        labels = [f"T{task_indices[i]}→T{task_indices[i+1]}" for i in range(n - 1)]
        ax.bar(x, consecutive_dists, color='steelblue', edgecolor='black', linewidth=0.5)
        ax.set_title(f'Consecutive Task Distance{title_suffix}', fontsize=14)
        ax.set_xlabel('Task Transition', fontsize=12)
        ax.set_ylabel('Euclidean Distance', fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        out_path2 = os.path.join(save_dir, f'consecutive_distance{title_suffix.replace(" ", "_")}.png')
        plt.savefig(out_path2, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {out_path2}")


def process_single_trial(project_dir, title_suffix=""):
    """Process a single trial directory."""
    print(f"\nProcessing: {project_dir}")
    checkpoints = load_task_checkpoints(project_dir)
    if len(checkpoints) < 2:
        print(f"  Need at least 2 task checkpoints, found {len(checkpoints)}. Skipping.")
        return

    task_indices = [c['task_idx'] for c in checkpoints]
    rewards = [c['best_reward'] for c in checkpoints]
    flat_vectors = np.array([flatten_params(c['params']) for c in checkpoints])

    print(f"  {len(checkpoints)} tasks, param vector length: {flat_vectors.shape[1]}")

    euclidean, cosine_sim = compute_distances(flat_vectors)

    # Print summary
    print(f"\n  Euclidean distance matrix:")
    for i, ti in enumerate(task_indices):
        row = "    "
        for j, tj in enumerate(task_indices):
            row += f"{euclidean[i, j]:8.3f}"
        print(row)

    print(f"\n  Cosine similarity matrix:")
    for i, ti in enumerate(task_indices):
        row = "    "
        for j, tj in enumerate(task_indices):
            row += f"{cosine_sim[i, j]:8.4f}"
        print(row)

    plot_heatmaps(euclidean, cosine_sim, task_indices, rewards, project_dir, title_suffix)


def main():
    parser = argparse.ArgumentParser(description="Pairwise distance heatmaps for PBT continual task solutions")
    parser.add_argument('--project_dir', type=str, required=True,
                        help='Path to a trial dir (with checkpoints/) or parent dir (with --all_trials)')
    parser.add_argument('--all_trials', action='store_true',
                        help='Process all trial_* subdirectories under project_dir')
    args = parser.parse_args()

    if args.all_trials:
        trial_dirs = sorted(glob.glob(os.path.join(args.project_dir, "trial_*")))
        if not trial_dirs:
            print(f"No trial_* directories found in {args.project_dir}")
            sys.exit(1)
        for td in trial_dirs:
            trial_name = os.path.basename(td)
            try:
                process_single_trial(td, title_suffix=f" ({trial_name})")
            except Exception as e:
                print(f"  Error processing {td}: {e}")
    else:
        process_single_trial(args.project_dir)


if __name__ == "__main__":
    main()
