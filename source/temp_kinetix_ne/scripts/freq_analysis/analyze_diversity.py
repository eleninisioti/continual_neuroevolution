import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os
from scipy import stats

def load_and_analyze_frequency(freq_value, data_dir):
    """Load diversity CSV file for a given frequency and analyze average diversity per task
    
    Args:
        freq_value: Frequency (how often task changes)
        data_dir: Directory containing CSV files
    """
    csv_path = data_dir / f"{freq_value}_diversity.csv"
    
    if not csv_path.exists():
        print(f"File not found: {csv_path}")
        return None
    
    print(f"Loading {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Get the Step column
    steps = df['Step'].values
    
    # Find columns for diversity (not MIN or MAX)
    diversity_cols = [col for col in df.columns 
                     if 'diversity' in col.lower() 
                     and '__' not in col]
    
    if len(diversity_cols) == 0:
        print(f"  No diversity columns found in {csv_path}")
        return None
    
    all_task_data = []
    
    # For each task window
    task_num = 0
    start_gen = 0
    
    while start_gen < len(steps) and steps[start_gen] < 20000:
        end_gen = start_gen
        
        # Find the end of this task window (when step exceeds task duration)
        while end_gen < len(steps) and steps[end_gen] < (task_num + 1) * freq_value:
            end_gen += 1
        
        task_window_steps = steps[start_gen:end_gen]
        
        if len(task_window_steps) == 0:
            break
        
        # For each trial, calculate average diversity in this task window
        for trial_col in diversity_cols:
            trial_data = df[trial_col].values[start_gen:end_gen]
            
            # Calculate average diversity during this task
            if len(trial_data) > 0:
                avg_diversity = np.mean(trial_data)
                all_task_data.append({
                    'frequency': freq_value,
                    'task': task_num,
                    'avg_diversity': avg_diversity,
                    'trial': trial_col
                })
        
        task_num += 1
        start_gen = end_gen
        
        # Stop if we've exceeded reasonable bounds
        if task_num > 1000:
            break
    
    return all_task_data


def calculate_confidence_interval_continuous(values, confidence=0.95):
    """Calculate confidence interval for continuous data using t-distribution"""
    if len(values) == 0:
        return np.nan, np.nan, np.nan
    
    mean_val = np.mean(values)
    std_err = stats.sem(values)  # Standard error of the mean
    
    # Use t-distribution for confidence interval
    df = len(values) - 1  # degrees of freedom
    t_critical = stats.t.ppf((1 + confidence) / 2, df)
    
    margin = t_critical * std_err
    lower = mean_val - margin
    upper = mean_val + margin
    
    return lower, mean_val, upper


def analyze_task(task_name, data_dir, frequencies, title):
    """Analyze diversity for a task with given parameters"""
    print(f"\n{'='*60}")
    print(f"Analyzing {title}")
    print(f"{'='*60}\n")
    
    # Collect all data
    all_results = []
    diversity_summary = []
    
    for freq in frequencies:
        result = load_and_analyze_frequency(freq, data_dir)
        if result is not None:
            all_results.extend(result)
            
            # Calculate statistics for this frequency
            freq_data = [r['avg_diversity'] for r in result]
            if len(freq_data) > 0:
                mean_diversity = np.mean(freq_data)
                std_diversity = np.std(freq_data)
                n_tasks = len(freq_data)
                
                # Calculate confidence interval
                ci_lower, mean_ci, ci_upper = calculate_confidence_interval_continuous(freq_data)
                
                diversity_summary.append({
                    'frequency': freq,
                    'avg_diversity': mean_diversity,
                    'std_diversity': std_diversity,
                    'n_tasks': n_tasks,
                    'ci_lower': ci_lower,
                    'ci_upper': ci_upper
                })
                
                print(f"  Frequency {freq}: Mean diversity = {mean_diversity:.4f} "
                      f"(std={std_diversity:.4f}, n={n_tasks} tasks)")
    
    # Convert to DataFrame
    results_df = pd.DataFrame(all_results)
    summary_df = pd.DataFrame(diversity_summary)
    
    if len(results_df) == 0:
        print("\nNo diversity data found!")
        return
    
    print(f"\nTotal task observations: {len(results_df)}")
    
    print("\nAverage diversity per frequency:")
    print(summary_df[['frequency', 'avg_diversity', 'std_diversity', 'n_tasks']])
    
    print("\nAverage diversity with 95% confidence intervals:")
    for _, row in summary_df.iterrows():
        print(f"  Frequency {row['frequency']}: {row['avg_diversity']:.4f} "
              f"[{row['ci_lower']:.4f}, {row['ci_upper']:.4f}]")
    
    # Create bar plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    freqs = summary_df['frequency'].astype(str)
    x_pos = np.arange(len(freqs))
    
    bars = ax.bar(x_pos, summary_df['avg_diversity'], 
            color='steelblue', alpha=0.7)
    
    # Compute error bar values (confidence interval margin)
    yerr_lower = np.maximum(0, summary_df['avg_diversity'] - summary_df['ci_lower'])
    yerr_upper = np.maximum(0, summary_df['ci_upper'] - summary_df['avg_diversity'])
    
    ax.errorbar(x_pos, summary_df['avg_diversity'],
                 yerr=[yerr_lower, yerr_upper],
                 fmt='none', color='black', capsize=5, elinewidth=1.5)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(freqs)
    ax.set_xlabel('Task Change Frequency', fontsize=12)
    ax.set_ylabel('Average Diversity', fontsize=12)
    ax.set_title(f'{title}: Average Diversity During Tasks', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    # Save the plot
    base_dir = Path('/home/eleni/workspace/continual_control_through_neuroevolution/scripts/freq_analysis')
    output_path = base_dir / f'{task_name}_diversity_analysis.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: {output_path}")
    
    # Also show the plot
    plt.show()


def main():
    # Set up paths
    base_dir = Path('/home/eleni/workspace/continual_control_through_neuroevolution/scripts/freq_analysis')
    
    # Look for diversity files - check both data and data_old directories
    # First check the main data directory structure
    cartpole_dir = base_dir / 'freq_anal' / 'cartpole' / 'ga'
    if cartpole_dir.exists():
        cartpole_frequencies = [5, 20, 50, 100, 200, 500]
        # Filter to only frequencies that have files
        existing_freqs = []
        for freq in cartpole_frequencies:
            if (cartpole_dir / f"{freq}_diversity.csv").exists():
                existing_freqs.append(freq)
        if existing_freqs:
            analyze_task('cartpole', cartpole_dir, existing_freqs, 'Cartpole GA')
    
    # Also check data_old directory
    data_old_cartpole_dir = base_dir / 'data_old' / 'cartpole' / 'ga'
    if data_old_cartpole_dir.exists():
        cartpole_frequencies = [5, 20, 50, 100, 200, 500]
        existing_freqs = []
        for freq in cartpole_frequencies:
            if (data_old_cartpole_dir / f"{freq}_diversity.csv").exists():
                existing_freqs.append(freq)
        if existing_freqs:
            analyze_task('cartpole_ga_old', data_old_cartpole_dir, existing_freqs, 'Cartpole GA (Old Data)')
    
    # Check for other environments similarly
    acrobot_dir = base_dir / 'freq_anal' / 'acrobot' / 'ga'
    if acrobot_dir.exists():
        acrobot_frequencies = [5, 10, 20, 50, 100, 200]
        existing_freqs = []
        for freq in acrobot_frequencies:
            if (acrobot_dir / f"{freq}_diversity.csv").exists():
                existing_freqs.append(freq)
        if existing_freqs:
            analyze_task('acrobot', acrobot_dir, existing_freqs, 'Acrobot GA')
    
    mountaincar_dir = base_dir / 'freq_anal' / 'mountaincar' / 'ga'
    if mountaincar_dir.exists():
        mountaincar_frequencies = [5, 10, 50, 100, 200]
        existing_freqs = []
        for freq in mountaincar_frequencies:
            if (mountaincar_dir / f"{freq}_diversity.csv").exists():
                existing_freqs.append(freq)
        if existing_freqs:
            analyze_task('mountaincar', mountaincar_dir, existing_freqs, 'Mountaincar GA')


if __name__ == "__main__":
    main()

