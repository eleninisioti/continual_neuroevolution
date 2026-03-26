#!/usr/bin/env python3
"""
Script to create line plots for episode_solved, episode_length, and episode_return.
Creates one plot for each task found in the data directory.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import re

def extract_task_id(filename):
    """Extract task identifier (h0, h1, etc.) from filename."""
    match = re.search(r'_h(\d+)_', filename)
    if match:
        return f"h{match.group(1)}"
    return None

def process_csv_file(csv_file, script_dir):
    """Process a single CSV file and create plots for it."""
    task_id = extract_task_id(csv_file.name)
    if task_id is None:
        print(f"Warning: Could not extract task ID from {csv_file.name}, skipping...")
        return False
    
    print(f"\n{'='*80}")
    print(f"Processing: {csv_file.name}")
    print(f"Task ID: {task_id}")
    print(f"{'='*80}")
    
    try:
        df = pd.read_csv(csv_file)
        print(f"Loaded {len(df)} rows")
        
        # Check if general episode columns exist
        if 'episode_solved' not in df.columns:
            print(f"Warning: 'episode_solved' column not found in {csv_file.name}, skipping...")
            return False
        if 'episode_length' not in df.columns:
            print(f"Warning: 'episode_length' column not found in {csv_file.name}, skipping...")
            return False
        if 'episode_return' not in df.columns:
            print(f"Warning: 'episode_return' column not found in {csv_file.name}, skipping...")
            return False
        
        # Get step column for x-axis (use timing/num_env_steps if available, otherwise _step, otherwise index)
        if 'timing/num_env_steps' in df.columns:
            x_data = df['timing/num_env_steps'].values
            x_label = 'Number of Environment Steps'
        elif '_step' in df.columns:
            x_data = df['_step'].values
            x_label = 'Step'
        else:
            x_data = np.arange(len(df))
            x_label = 'Index'
        
        # Extract the three metrics, removing NaN values
        episode_solved = df['episode_solved'].values
        episode_length = df['episode_length'].values
        episode_return = df['episode_return'].values
        
        # Create mask for valid data (not NaN)
        valid_mask = ~(np.isnan(episode_solved) | np.isnan(episode_length) | np.isnan(episode_return))
        
        if not np.any(valid_mask):
            print(f"Warning: No valid data points found (all NaN) in {csv_file.name}, skipping...")
            return False
        
        # Filter data
        x_valid = x_data[valid_mask]
        solved_valid = episode_solved[valid_mask]
        length_valid = episode_length[valid_mask]
        return_valid = episode_return[valid_mask]
        
        print(f"Valid data points: {len(x_valid)}")
        print(f"X-axis range: {x_valid.min():.0f} to {x_valid.max():.0f}")
        print(f"X-axis label: {x_label}")
        
        # Create the combined plot
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
        fig.suptitle(f'Episode Metrics for Task {task_id}', fontsize=14, fontweight='bold')
        
        # Plot episode_solved
        axes[0].plot(x_valid, solved_valid, linewidth=2, color='blue', alpha=0.8, marker='o', markersize=4)
        axes[0].set_title('Episode Solved', fontsize=12, fontweight='bold')
        axes[0].set_ylabel('Episode Solved', fontsize=11)
        axes[0].grid(True, alpha=0.3)
        
        # Plot episode_length
        axes[1].plot(x_valid, length_valid, linewidth=2, color='green', alpha=0.8, marker='o', markersize=4)
        axes[1].set_title('Episode Length', fontsize=12, fontweight='bold')
        axes[1].set_ylabel('Episode Length', fontsize=11)
        axes[1].grid(True, alpha=0.3)
        
        # Plot episode_return
        axes[2].plot(x_valid, return_valid, linewidth=2, color='purple', alpha=0.8, marker='o', markersize=4)
        axes[2].set_title('Episode Return', fontsize=12, fontweight='bold')
        axes[2].set_xlabel(x_label, fontsize=11)
        axes[2].set_ylabel('Episode Return', fontsize=11)
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save the combined plot
        output_path = script_dir / f"episode_metrics_{task_id}_plot.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Combined plot saved to: {output_path}")
        plt.close()
        
        # Create individual plots
        for metric_name, data, color in [('episode_solved', solved_valid, 'blue'),
                                         ('episode_length', length_valid, 'green'),
                                         ('episode_return', return_valid, 'purple')]:
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(x_valid, data, linewidth=2, color=color, alpha=0.8, marker='o', markersize=4)
            ax.set_title(f'{metric_name.replace("_", " ").title()} - Task {task_id}', fontsize=14, fontweight='bold')
            ax.set_xlabel(x_label, fontsize=12)
            ax.set_ylabel(metric_name.replace("_", " ").title(), fontsize=12)
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            output_path_individual = script_dir / f"{metric_name}_{task_id}_plot.png"
            plt.savefig(output_path_individual, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Individual plot saved to: {output_path_individual}")
        
        return True
        
    except Exception as e:
        print(f"Error processing {csv_file.name}: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    # Get the script directory and construct path to data directory
    script_dir = Path(__file__).parent
    data_dir = script_dir / "data"
    
    # Check if data directory exists
    if not data_dir.exists():
        print(f"Error: Data directory not found at {data_dir}")
        return
    
    # Find all history CSV files in the data directory
    history_files = list(data_dir.glob("*_history.csv"))
    
    if not history_files:
        print(f"No history CSV files found in {data_dir}")
        return
    
    print(f"Found {len(history_files)} history CSV file(s)")
    
    # Sort files by task ID for consistent processing
    history_files.sort(key=lambda x: extract_task_id(x.name) or "")
    
    # Process each history file
    successful = 0
    failed = 0
    
    for csv_file in history_files:
        if process_csv_file(csv_file, script_dir):
            successful += 1
        else:
            failed += 1
    
    print(f"\n{'='*80}")
    print(f"Summary: {successful} files processed successfully, {failed} failed")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()
