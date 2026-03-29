"""
Plot reward and diversity over generations for PBT Acrobot,
one line per pop size with confidence intervals across trials.

Reads wandb CSV exports from projects/gymnax/pbt/acrobot/.

Usage:
    python scripts/postprocess/plot_pbt_acrobot_reward_diversity.py
"""

import os
import re
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DATA_DIR = "projects/gymnax/pbt/acrobot"

# First CSV = reward, second = diversity
REWARD_CSV = "wandb_export_2026-03-29T12_24_51.966+02_00.csv"
DIVERSITY_CSV = "wandb_export_2026-03-29T12_25_06.583+02_00.csv"

COLORS = {4: 'steelblue', 8: 'mediumseagreen', 16: 'orchid', 2: 'coral',
           32: 'goldenrod', 64: 'slategray'}


def parse_wandb_csv(csv_path, metric_keyword):
    """
    Parse a wandb export CSV. Returns dict:
      {pop_size: {trial: (generations, values)}}
    
    metric_keyword: substring to identify the metric column (e.g. 'pop_best_reward', 'diversity')
    """
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = list(reader)

    # Find columns: generation + metric columns (not __MIN/__MAX/_step)
    gen_col = header.index("generation")

    # Find metric columns: match keyword, exclude __MIN, __MAX, _step
    metric_cols = {}
    for i, h in enumerate(header):
        if metric_keyword in h and '__MIN' not in h and '__MAX' not in h and '_step' not in h:
            # Extract pop size and trial from column name
            match = re.search(r'pop(\d+)_trial(\d+)', h)
            if match:
                pop_size = int(match.group(1))
                trial = int(match.group(2))
                metric_cols[i] = (pop_size, trial)

    # Parse data
    data = {}  # {pop_size: {trial: (gens, vals)}}
    for pop_size, trial in metric_cols.values():
        if pop_size not in data:
            data[pop_size] = {}

    for col_idx, (pop_size, trial) in metric_cols.items():
        gens = []
        vals = []
        for row in rows:
            gen_val = row[gen_col].strip()
            metric_val = row[col_idx].strip()
            if gen_val and metric_val:
                gens.append(float(gen_val))
                vals.append(float(metric_val))
        if gens:
            data[pop_size][trial] = (np.array(gens), np.array(vals))

    return data


def interpolate_to_common_grid(data):
    """
    Interpolate all trials to a common generation grid per pop size.
    Returns {pop_size: (common_gens, trial_values_2d)}
    """
    result = {}
    for pop_size, trials in data.items():
        if not trials:
            continue
        # Common grid = union of all generation points, clipped to shared range
        all_gens = np.unique(np.concatenate([g for g, v in trials.values()]))
        min_gen = max(g.min() for g, v in trials.values())
        max_gen = min(g.max() for g, v in trials.values())
        common_gens = all_gens[(all_gens >= min_gen) & (all_gens <= max_gen)]

        trial_vals = []
        for trial, (gens, vals) in sorted(trials.items()):
            interp_vals = np.interp(common_gens, gens, vals)
            trial_vals.append(interp_vals)

        result[pop_size] = (common_gens, np.array(trial_vals))
    return result


def plot_metric(data, title, ylabel, output_path):
    """Plot lines with confidence intervals."""
    interp_data = interpolate_to_common_grid(data)

    fig, ax = plt.subplots(figsize=(10, 5))

    for pop_size in sorted(interp_data.keys()):
        gens, trial_vals = interp_data[pop_size]
        mean = np.mean(trial_vals, axis=0)
        std = np.std(trial_vals, axis=0)
        n = trial_vals.shape[0]
        ci = 1.96 * std / np.sqrt(n)  # 95% CI

        color = COLORS.get(pop_size, 'gray')
        ax.plot(gens, mean, label=f'pop={pop_size} (n={n})', color=color, linewidth=1.5)
        ax.fill_between(gens, mean - ci, mean + ci, alpha=0.2, color=color)

    ax.set_xlabel("Generation", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def main():
    reward_data = parse_wandb_csv(
        os.path.join(DATA_DIR, REWARD_CSV), "pop_best_reward"
    )
    diversity_data = parse_wandb_csv(
        os.path.join(DATA_DIR, DIVERSITY_CSV), "diversity"
    )

    print("Reward data:")
    for pop, trials in sorted(reward_data.items()):
        print(f"  pop={pop}: {len(trials)} trials, "
              f"gens range: {[int(t[0].min()) for t in trials.values()]} - {[int(t[0].max()) for t in trials.values()]}")

    print("Diversity data:")
    for pop, trials in sorted(diversity_data.items()):
        print(f"  pop={pop}: {len(trials)} trials")

    plot_metric(
        reward_data,
        "PBT Acrobot — Best Reward over Generations",
        "Best Reward",
        os.path.join(DATA_DIR, "pbt_acrobot_reward.png")
    )

    plot_metric(
        diversity_data,
        "PBT Acrobot — Diversity over Generations",
        "Diversity",
        os.path.join(DATA_DIR, "pbt_acrobot_diversity.png")
    )


if __name__ == "__main__":
    main()
