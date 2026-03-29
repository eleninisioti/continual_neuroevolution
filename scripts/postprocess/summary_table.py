"""
Generate summary table of continual learning metrics across methods.

Reads metrics.yaml files (or per-task checkpoints) from:
  - GA:   projects/gymnax/{sweep_dir}/{env}/pop_{N}/trial_{T}/
  - DNS:  projects/gymnax/{sweep_dir}/{env}/pop_{N}/trial_{T}/
  - RL:   projects/gymnax/{sweep_dir}/{env}/{method}/trial_{T}/

Computes per-method:
  - Success Rate (S): fraction of tasks solved (eval_mean >= threshold)
  - Zero-Shot Transfer (ZT): mean first-episode reward on new task
  - Speed-Up (SU): (steps_scratch - steps_warmstarted) / steps_scratch

Reports mean ± std across trials with Welch's t-test for pairwise significance.

Usage:
    python scripts/postprocess/summary_table.py --env CartPole-v1
    python scripts/postprocess/summary_table.py --env CartPole-v1 --task_type param
"""

import argparse
import os
import sys
import pickle
import yaml
import numpy as np
from scipy import stats

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))

SOLVED_THRESHOLDS = {
    'CartPole-v1': 475,
    'Acrobot-v1': -70,
    'MountainCar-v0': -110,
}

NUM_TASKS = 10


def load_metrics_from_yaml(yaml_path):
    """Load metrics from a metrics.yaml file."""
    with open(yaml_path, 'r') as f:
        return yaml.safe_load(f)


def load_metrics_from_checkpoints(ckpt_dir, env_name, steps_key='gens_to_threshold'):
    """Fallback: reconstruct metrics from per-task checkpoint .pkl files."""
    threshold = SOLVED_THRESHOLDS.get(env_name)
    per_task = []
    for task_idx in range(NUM_TASKS):
        ckpt_path = os.path.join(ckpt_dir, f"task_{task_idx}.pkl")
        if not os.path.exists(ckpt_path):
            continue
        with open(ckpt_path, 'rb') as f:
            ckpt = pickle.load(f)
        task_metrics = {
            'task_idx': task_idx,
            'eval_mean': ckpt.get('eval_mean', float('nan')),
            'eval_std': ckpt.get('eval_std', float('nan')),
            'zero_shot_eval_mean': ckpt.get('zero_shot_eval_mean', None),
            'zero_shot_eval_std': ckpt.get('zero_shot_eval_std', None),
            steps_key: ckpt.get(steps_key, None),
        }
        per_task.append(task_metrics)

    if not per_task:
        return None

    num_tasks = len(per_task)
    num_solved = sum(1 for m in per_task if threshold is not None and m['eval_mean'] >= threshold)

    return {
        'num_tasks': num_tasks,
        'success_rate': num_solved / num_tasks,
        'num_solved': num_solved,
        'per_task': per_task,
    }


def collect_trial_metrics(base_dir, env_name, method_name, dir_structure, steps_key='gens_to_threshold'):
    """
    Collect metrics across all trials for a given method.

    dir_structure: 'ga' | 'dns' | 'rl'
    Returns list of per-trial metric dicts.
    """
    trial_metrics = []

    for trial in range(1, 100):
        if dir_structure in ('ga', 'dns'):
            trial_dir = os.path.join(base_dir, f"trial_{trial}")
        elif dir_structure == 'rl':
            trial_dir = os.path.join(base_dir, method_name, f"trial_{trial}")
        else:
            trial_dir = os.path.join(base_dir, f"trial_{trial}")

        if not os.path.isdir(trial_dir):
            if trial > 5:
                break
            continue

        # Try metrics.yaml first
        yaml_path = os.path.join(trial_dir, "metrics.yaml")
        if os.path.exists(yaml_path):
            m = load_metrics_from_yaml(yaml_path)
            trial_metrics.append(m)
        else:
            # Fallback to checkpoints
            ckpt_dir = os.path.join(trial_dir, "checkpoints")
            if os.path.isdir(ckpt_dir):
                m = load_metrics_from_checkpoints(ckpt_dir, env_name, steps_key)
                if m is not None:
                    trial_metrics.append(m)

    return trial_metrics


def aggregate_metrics(trial_list, steps_key='gens_to_threshold'):
    """
    Aggregate per-trial metrics into mean ± std.

    Returns dict with:
        success_rates: array of per-trial success rates
        zt_means: array of per-trial mean ZT values
        steps_to_threshold: array of per-trial mean steps-to-threshold (NaN if not reached)
    """
    success_rates = []
    zt_means = []
    steps_list = []

    for trial in trial_list:
        success_rates.append(trial.get('success_rate', 0.0))
        per_task = trial.get('per_task', [])
        zt_vals = [t.get('zero_shot_eval_mean') for t in per_task if t.get('zero_shot_eval_mean') is not None]
        zt_means.append(np.mean(zt_vals) if zt_vals else float('nan'))

        step_vals = [t.get(steps_key) for t in per_task if t.get(steps_key) is not None]
        steps_list.append(np.mean(step_vals) if step_vals else float('nan'))

    return {
        'success_rates': np.array(success_rates),
        'zt_means': np.array(zt_means),
        'steps_to_threshold': np.array(steps_list),
        'n_trials': len(trial_list),
    }


def format_mean_std(values):
    """Format as 'mean ± std' ignoring NaNs."""
    valid = values[~np.isnan(values)]
    if len(valid) == 0:
        return "N/A"
    return f"{np.mean(valid):.2f} ± {np.std(valid):.2f}"


def welch_t_test(a, b):
    """Welch's t-test between two arrays, ignoring NaNs."""
    a = a[~np.isnan(a)]
    b = b[~np.isnan(b)]
    if len(a) < 2 or len(b) < 2:
        return float('nan')
    _, p = stats.ttest_ind(a, b, equal_var=False)
    return p


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, required=True, choices=['CartPole-v1', 'Acrobot-v1', 'MountainCar-v0'])
    parser.add_argument('--task_type', type=str, default='param', choices=['noise', 'param'])
    parser.add_argument('--projects_dir', type=str, default=os.path.join(REPO_ROOT, 'projects', 'gymnax'))
    parser.add_argument('--ga_dir', type=str, default=None, help='Override GA sweep dir name')
    parser.add_argument('--dns_dir', type=str, default=None, help='Override DNS sweep dir name')
    parser.add_argument('--rl_dir', type=str, default=None, help='Override RL sweep dir name')
    parser.add_argument('--ga_pop', type=int, default=512, help='GA pop size to report')
    parser.add_argument('--dns_pop', type=int, default=512, help='DNS pop size to report')
    parser.add_argument('--output', type=str, default=None, help='Output file (markdown)')
    args = parser.parse_args()

    env = args.env
    threshold = SOLVED_THRESHOLDS.get(env)
    projects_dir = args.projects_dir

    # Default directory names
    ga_sweep_dir = args.ga_dir or 'ga_continual_pop_sweep'
    dns_sweep_dir = args.dns_dir or 'dns_continual_pop_sweep'
    rl_sweep_dir = args.rl_dir or 'rl_continual'

    # Method configs: (label, base_dir, dir_structure, steps_key)
    methods = []

    # GA
    ga_base = os.path.join(projects_dir, ga_sweep_dir, env, f"pop_{args.ga_pop}")
    if os.path.isdir(ga_base):
        methods.append(('GA', ga_base, 'ga', 'gens_to_threshold'))

    # DNS
    dns_base = os.path.join(projects_dir, dns_sweep_dir, env, f"pop_{args.dns_pop}")
    if os.path.isdir(dns_base):
        methods.append(('DNS', dns_base, 'dns', 'gens_to_threshold'))

    # RL methods
    rl_base = os.path.join(projects_dir, rl_sweep_dir, env)
    for rl_method in ['ppo', 'trac', 'redo']:
        rl_method_dir = os.path.join(rl_base, rl_method)
        if os.path.isdir(rl_method_dir):
            methods.append((rl_method.upper(), rl_base, 'rl', 'updates_to_threshold'))

    if not methods:
        print(f"No results found for {env} in {projects_dir}")
        sys.exit(1)

    # Collect and aggregate
    results = {}
    for label, base_dir, dir_struct, steps_key in methods:
        method_name = label.lower()
        trials = collect_trial_metrics(base_dir, env, method_name, dir_struct, steps_key)
        if trials:
            results[label] = aggregate_metrics(trials, steps_key)
        else:
            print(f"  Warning: No trials found for {label}")

    # Build table
    header = f"## Continual Learning Metrics: {env} ({args.task_type})\n"
    header += f"Solved threshold: {threshold}\n\n"
    header += f"| Method | Trials | Success Rate (S) | Zero-Shot Transfer (ZT) | Steps to Threshold |\n"
    header += f"|--------|--------|-----------------|------------------------|--------------------|\n"

    rows = []
    for label in [m[0] for m in methods]:
        if label not in results:
            continue
        r = results[label]
        s_str = format_mean_std(r['success_rates'])
        zt_str = format_mean_std(r['zt_means'])
        su_str = format_mean_std(r['steps_to_threshold'])
        rows.append(f"| {label:6s} | {r['n_trials']:6d} | {s_str:15s} | {zt_str:22s} | {su_str:18s} |")

    # Significance tests (pairwise)
    method_labels = [m[0] for m in methods if m[0] in results]
    sig_lines = []
    if len(method_labels) >= 2:
        sig_lines.append("\n### Pairwise Significance (Welch's t-test, p-values)\n")
        sig_lines.append("| Comparison | Success (p) | ZT (p) | Steps (p) |")
        sig_lines.append("|------------|-------------|--------|-----------|")
        for i in range(len(method_labels)):
            for j in range(i + 1, len(method_labels)):
                a_label, b_label = method_labels[i], method_labels[j]
                a, b = results[a_label], results[b_label]
                p_s = welch_t_test(a['success_rates'], b['success_rates'])
                p_zt = welch_t_test(a['zt_means'], b['zt_means'])
                p_su = welch_t_test(a['steps_to_threshold'], b['steps_to_threshold'])

                def fmt_p(p):
                    if np.isnan(p):
                        return "N/A"
                    if p < 0.001:
                        return f"{p:.1e} ***"
                    if p < 0.01:
                        return f"{p:.3f} **"
                    if p < 0.05:
                        return f"{p:.3f} *"
                    return f"{p:.3f}"

                sig_lines.append(f"| {a_label} vs {b_label} | {fmt_p(p_s)} | {fmt_p(p_zt)} | {fmt_p(p_su)} |")

    output = header + "\n".join(rows)
    if sig_lines:
        output += "\n" + "\n".join(sig_lines)
    output += "\n"

    print(output)

    if args.output:
        with open(args.output, 'w') as f:
            f.write(output)
        print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
