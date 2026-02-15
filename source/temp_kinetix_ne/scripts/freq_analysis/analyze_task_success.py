import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os
from scipy import stats

def load_and_analyze_frequency(freq_value, data_dir, success_threshold=-80, use_greater_than_or_equal=False):
    """Load CSV file for a given frequency and analyze task success
    
    Args:
        freq_value: Frequency (how often task changes)
        data_dir: Directory containing CSV files
        success_threshold: Threshold for success
        use_greater_than_or_equal: If True, use >= for comparison; if False, use >
    """
    csv_path = data_dir / f"{freq_value}.csv"
    
    if not csv_path.exists():
        print(f"File not found: {csv_path}")
        return None
    
    print(f"Loading {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Get the Step column
    steps = df['Step'].values
    
    # Find columns for current_best_fitness (not MIN or MAX)
    fitness_cols = [col for col in df.columns 
                    if 'current_best_fitness' in col 
                    and '__' not in col]
    
    all_task_data = []
    success_count = 0
    total_attempts = 0
    
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
        
        # For each trial, check if it succeeds in this task
        for trial_col in fitness_cols:
            trial_data = df[trial_col].values[start_gen:end_gen]
            
            # Check if this trial ever reached fitness above threshold in this task window
            if use_greater_than_or_equal:
                succeeded = np.any(trial_data >= success_threshold)
                success_indices = np.where(trial_data >= success_threshold)[0]
            else:
                succeeded = np.any(trial_data > success_threshold)
                success_indices = np.where(trial_data > success_threshold)[0]
            total_attempts += 1
            
            if succeeded:
                success_count += 1
                # Find the generation within this task when it succeeded
                if len(success_indices) > 0:
                    generations_in_task = success_indices[0]
                    all_task_data.append({
                        'frequency': freq_value,
                        'task': task_num,
                        'generations_to_success': generations_in_task,
                        'trial': trial_col
                    })
        
        task_num += 1
        start_gen = end_gen
        
        # Stop if we've exceeded reasonable bounds
        if task_num > 1000:
            break
    
    return all_task_data, success_count, total_attempts


def wilson_confidence_interval(success_count, total_attempts, confidence=0.95):
    """Calculate Wilson score confidence interval for binomial proportion"""
    if total_attempts == 0:
        return 0.0, 0.0, 0.0
    
    # Wilson score interval
    p = success_count / total_attempts
    z = stats.norm.ppf((1 + confidence) / 2)
    
    denominator = 1 + z**2 / total_attempts
    center = (p + z**2 / (2 * total_attempts)) / denominator
    margin = z * np.sqrt((p * (1 - p) + z**2 / (4 * total_attempts)) / total_attempts) / denominator
    
    lower = max(0, center - margin)
    upper = min(1, center + margin)
    
    return lower, center, upper


def analyze_task(task_name, data_dir, frequencies, success_threshold, title, use_greater_than_or_equal=False):
    """Analyze a task with given parameters"""
    print(f"\n{'='*60}")
    print(f"Analyzing {title}")
    print(f"{'='*60}\n")
    
    # Collect all data
    all_results = []
    success_rate_data = []
    
    for freq in frequencies:
        result = load_and_analyze_frequency(freq, data_dir, success_threshold, use_greater_than_or_equal)
        if result is not None:
            task_data, success_count, total_attempts = result
            all_results.extend(task_data)
            success_rate = success_count / total_attempts if total_attempts > 0 else 0
            success_rate_data.append({
                'frequency': freq,
                'success_rate': success_rate,
                'success_count': success_count,
                'total_attempts': total_attempts
            })
            print(f"  Processed {len(task_data)} successful task completions out of {total_attempts} attempts ({success_rate*100:.1f}%)")
    
    # Convert to DataFrame
    results_df = pd.DataFrame(all_results)
    
    print(f"\nTotal successful task completions: {len(results_df)}")
    
    # Create success rate DataFrame
    success_rate_df = pd.DataFrame(success_rate_data)
    
    print("\nSuccess rate per frequency:")
    print(success_rate_df[['frequency', 'success_rate', 'success_count', 'total_attempts']])
    
    # Always calculate average generations (even if empty)
    if len(results_df) > 0:
        avg_gens_per_freq = results_df.groupby('frequency')['generations_to_success'].agg(['mean', 'std', 'count']).reset_index()
        avg_gens_per_freq.columns = ['frequency', 'avg_generations_to_success', 'std_generations_to_success', 'n_successes']
        
        # Calculate standard error and confidence intervals
        avg_gens_per_freq['se_generations_to_success'] = avg_gens_per_freq['std_generations_to_success'] / np.sqrt(avg_gens_per_freq['n_successes'])
        # Use t-distribution for confidence intervals
        confidence = 0.95
        # Handle case where n_successes = 1 (df = 0, use normal distribution instead)
        df = np.maximum(1, avg_gens_per_freq['n_successes'] - 1)
        t_critical = stats.t.ppf((1 + confidence) / 2, df)
        avg_gens_per_freq['ci_lower_gens'] = avg_gens_per_freq['avg_generations_to_success'] - t_critical * avg_gens_per_freq['se_generations_to_success']
        avg_gens_per_freq['ci_upper_gens'] = avg_gens_per_freq['avg_generations_to_success'] + t_critical * avg_gens_per_freq['se_generations_to_success']
        
        # Merge with success_rate_df to ensure all frequencies are represented
        # Set NaN for frequencies with no successes
        success_rate_df = success_rate_df.merge(avg_gens_per_freq[['frequency', 'avg_generations_to_success', 'ci_lower_gens', 'ci_upper_gens']], on='frequency', how='left')
        success_rate_df['avg_generations_to_success'] = success_rate_df['avg_generations_to_success'].fillna(np.nan)
        success_rate_df['ci_lower_gens'] = success_rate_df['ci_lower_gens'].fillna(np.nan)
        success_rate_df['ci_upper_gens'] = success_rate_df['ci_upper_gens'].fillna(np.nan)
        
        print("\nAverage generations to success per frequency:")
        print(success_rate_df[['frequency', 'avg_generations_to_success']])
    else:
        # No successes at all - set all to NaN
        success_rate_df['avg_generations_to_success'] = np.nan
        success_rate_df['ci_lower_gens'] = np.nan
        success_rate_df['ci_upper_gens'] = np.nan
        print("\nNo successful task completions - cannot calculate average generations to success.")
    
    # Calculate confidence intervals for success rates (always do this)
    if len(success_rate_df) > 0:
        success_rate_df['ci_lower'] = success_rate_df.apply(
            lambda row: wilson_confidence_interval(row['success_count'], row['total_attempts'])[0], 
            axis=1
        )
        success_rate_df['ci_upper'] = success_rate_df.apply(
            lambda row: wilson_confidence_interval(row['success_count'], row['total_attempts'])[2], 
            axis=1
        )
        
        print("\nSuccess rate with 95% confidence intervals:")
        for _, row in success_rate_df.iterrows():
            print(f"  Frequency {row['frequency']}: {row['success_rate']:.4f} "
                  f"[{row['ci_lower']:.4f}, {row['ci_upper']:.4f}]")
        
        # Create two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Plot 1: Success rate with error bars
        freqs = success_rate_df['frequency'].astype(str)
        x_pos = np.arange(len(freqs))
        
        bars = ax1.bar(x_pos, success_rate_df['success_rate'], 
                color='steelblue', alpha=0.7)
        
        # Compute error bar values (ensure non-negative)
        yerr_lower = np.maximum(0, success_rate_df['success_rate'] - success_rate_df['ci_lower'])
        yerr_upper = np.maximum(0, success_rate_df['ci_upper'] - success_rate_df['success_rate'])
        
        ax1.errorbar(x_pos, success_rate_df['success_rate'],
                     yerr=[yerr_lower, yerr_upper],
                     fmt='none', color='black', capsize=5, elinewidth=1.5)
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(freqs)
        ax1.set_xlabel('Task Change Frequency', fontsize=12)
        ax1.set_ylabel('Success Rate (%)', fontsize=12)
        ax1.set_title(f'Probability of Success\n(reward > {success_threshold})', fontsize=14, fontweight='bold')
        ax1.set_ylim(0, 1)
        ax1.grid(axis='y', alpha=0.3)
        
        # Plot 2: Average generations to success
        # Handle NaN values by creating a boolean mask
        valid_gens = ~np.isnan(success_rate_df['avg_generations_to_success'])
        x_pos_valid = np.where(valid_gens)[0]
        
        if len(x_pos_valid) > 0:
            bars = ax2.bar(x_pos_valid, 
                          success_rate_df.loc[valid_gens, 'avg_generations_to_success'],
                          color='coral', alpha=0.7)
            
            # Add error bars for valid generations
            yerr_lower = np.maximum(0, 
                success_rate_df.loc[valid_gens, 'avg_generations_to_success'] - 
                success_rate_df.loc[valid_gens, 'ci_lower_gens']
            )
            yerr_upper = np.maximum(0,
                success_rate_df.loc[valid_gens, 'ci_upper_gens'] - 
                success_rate_df.loc[valid_gens, 'avg_generations_to_success']
            )
            
            ax2.errorbar(x_pos_valid, 
                        success_rate_df.loc[valid_gens, 'avg_generations_to_success'],
                        yerr=[yerr_lower, yerr_upper],
                        fmt='none', color='black', capsize=5, elinewidth=1.5)
        
        # Add gray bars for NaN values
        invalid_gens = np.isnan(success_rate_df['avg_generations_to_success'])
        if invalid_gens.any():
            ax2.bar(np.where(invalid_gens)[0], 
                   [0.1] * invalid_gens.sum(),
                   color='lightgray', alpha=0.5, label='No successes')
        ax2.set_xticks(range(len(success_rate_df)))
        ax2.set_xticklabels(success_rate_df['frequency'].astype(str))
        ax2.set_xlabel('Task Change Frequency', fontsize=12)
        ax2.set_ylabel('Average Generations to Success', fontsize=12)
        ax2.set_title(f'Time to Success\n(reward > {success_threshold})', fontsize=14, fontweight='bold')
        ax2.grid(axis='y', alpha=0.3)
        
        plt.suptitle(f'{title}: Task Success Analysis', fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        # Save the plot
        base_dir = Path('/home/eleni/workspace/continual_control_through_neuroevolution/scripts/freq_analysis')
        output_path = base_dir / f'{task_name}_ga_success_analysis.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\nPlot saved to: {output_path}")
        
        # Also show the plot
        plt.show()
    


def main():
    # Set up paths
    base_dir = Path('/home/eleni/workspace/continual_control_through_neuroevolution/scripts/freq_analysis')
    data_dir = "freq_anal"
    
    # Analyze Cartpole PPO
    cartpole_ppo_dir = base_dir / data_dir / 'cartpole' / 'ppo'
    cartpole_ppo_frequencies = [5, 20, 50, 100, 200]
    analyze_task('cartpole_ppo', cartpole_ppo_dir, cartpole_ppo_frequencies, 198, 'Cartpole PPO', use_greater_than_or_equal=False)
    
  
    
      # Analyze Acrobot PPO
    acrobot_ppo_dir = base_dir / data_dir / 'acrobot' / 'ppo'
    acrobot_ppo_frequencies = [5, 10, 20, 50, 100, 200]
    analyze_task('acrobot_ppo', acrobot_ppo_dir, acrobot_ppo_frequencies, -80, 'Acrobot PPO', use_greater_than_or_equal=False)
    
  
        # Analyze Acrobot PPO
    mountaincar_ppo_dir = base_dir / data_dir / 'mountaincar' / 'ppo'
    mountaincar_ppo_frequencies = [5, 10, 20, 50, 100, 200]
    analyze_task('mountaincar_ppo', mountaincar_ppo_dir, mountaincar_ppo_frequencies, -139, 'Mountaincar PPO', use_greater_than_or_equal=False)
    
  
    
    
      # Analyze Cartpole GA (use >= since goal is exactly 199)
    cartpole_ga_dir = base_dir / data_dir / 'cartpole' / 'ga'
    cartpole_frequencies = [5, 20, 50, 100, 200, 500]
    analyze_task('cartpole_ga', cartpole_ga_dir, cartpole_frequencies, 199, 'Cartpole GA', use_greater_than_or_equal=True)
    
    
      # Analyze Mountaincar GA
    mountaincar_ga_dir = base_dir / data_dir / 'mountaincar' / 'ga'
    mountaincar_frequencies = [5, 10, 50, 100, 200, 500, 1000]
    analyze_task('mountaincar_ga', mountaincar_ga_dir, mountaincar_frequencies, -130, 'Mountaincar GA', use_greater_than_or_equal=False)

    # Analyze Acrobot GA
    data_dir = "data_old"
    acrobot_ga_dir = base_dir / data_dir / 'acrobot' / 'ga'
    acrobot_frequencies = [5, 10, 20, 50, 100, 200]
    analyze_task('acrobot_ga', acrobot_ga_dir, acrobot_frequencies, -80, 'Acrobot GA', use_greater_than_or_equal=False)
    
 
  
    quit()
     # Analyze Acrobot PPO
    acrobot_ppo_dir = base_dir / data_dir / 'acrobot' / 'ppo'
    acrobot_ppo_frequencies = [10, 20]
    analyze_task('acrobot_ppo', acrobot_ppo_dir, acrobot_ppo_frequencies, -80, 'Acrobot PPO', use_greater_than_or_equal=False)
    
  

if __name__ == "__main__":
    main()

