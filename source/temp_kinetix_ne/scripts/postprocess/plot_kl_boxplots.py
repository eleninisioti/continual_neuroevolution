"""Script to create boxplots for KL divergences comparing GA and OpenES."""
import os
import yaml
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12


def load_kl_data(base_dir: str) -> Dict:
    """
    Load all KL divergence data from YAML files.
    
    Returns:
        Dictionary with structure:
        {
            'continual': {
                'acrobot': {'ga': [kl_values...], 'openes': [kl_values...]},
                'cartpole': {...},
                'mountaincar': {...}
            },
            'vanilla': {...}
        }
    """
    data = {
        'continual': defaultdict(lambda: defaultdict(list)),
        'vanilla': defaultdict(lambda: defaultdict(list))
    }
    
    base_path = Path(base_dir)
    
    # Methods and variants to process
    methods = ['ga', 'openes']
    variants = ['continual', 'vanilla']
    tasks = ['acrobot', 'cartpole', 'mountaincar']
    
    for method in methods:
        for variant in variants:
            for task in tasks:
                task_dir = base_path / method / variant / task
                
                if not task_dir.exists():
                    print(f"Warning: Directory not found: {task_dir}")
                    continue
                
                # Find all trial directories
                trial_dirs = sorted([d for d in task_dir.iterdir() if d.is_dir() and d.name.startswith('trial_')])
                
                if not trial_dirs:
                    print(f"Warning: No trial directories found in {task_dir}")
                    continue
                
                # Load KL data from each trial
                for trial_dir in trial_dirs:
                    kl_file = trial_dir / 'data' / 'eval' / 'kl_divergence_comparison.yaml'
                    
                    if not kl_file.exists():
                        print(f"Warning: KL file not found: {kl_file}")
                        continue
                    
                    try:
                        with open(kl_file, 'r') as f:
                            kl_data = yaml.safe_load(f)
                        
                        if kl_data is None:
                            continue
                        
                        # Extract mean_kl values from all checkpoint pairs
                        for key, value in kl_data.items():
                            if isinstance(value, dict) and 'mean_kl' in value:
                                mean_kl = value['mean_kl']
                                # Only add non-zero values (or all values if you want to include zeros)
                                if mean_kl is not None:
                                    data[variant][task][method].append(mean_kl)
                    
                    except Exception as e:
                        print(f"Error loading {kl_file}: {e}")
                        continue
    
    return data


def create_boxplots(data: Dict, output_dir: str):
    """
    Create boxplots for KL divergences.
    
    One boxplot per task, showing GA vs OpenES, separately for continual and vanilla.
    """
    tasks = ['acrobot', 'cartpole', 'mountaincar']
    variants = ['continual', 'vanilla']
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create separate plots for continual and vanilla
    for variant in variants:
        fig, axes = plt.subplots(1, len(tasks), figsize=(15, 5))
        
        if len(tasks) == 1:
            axes = [axes]
        
        for idx, task in enumerate(tasks):
            ax = axes[idx]
            
            # Prepare data for boxplot
            plot_data = []
            labels = []
            
            for method in ['ga', 'openes']:
                kl_values = data[variant][task][method]
                
                if len(kl_values) == 0:
                    print(f"Warning: No data for {variant}/{task}/{method}")
                    continue
                
                plot_data.append(kl_values)
                labels.append(method.upper())
            
            if len(plot_data) == 0:
                ax.text(0.5, 0.5, 'No data available', 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'{task.capitalize()}')
                continue
            
            # Create boxplot
            bp = ax.boxplot(plot_data, patch_artist=True)
            ax.set_xticklabels(labels)
            
            # Color the boxes
            colors = ['#3498db', '#e74c3c']  # Blue for GA, Red for OpenES
            for patch, color in zip(bp['boxes'], colors[:len(bp['boxes'])]):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            
            # Customize plot
            ax.set_title(f'{task.capitalize()}', fontsize=14, fontweight='bold')
            ax.set_ylabel('KL Divergence', fontsize=12)
            ax.grid(True, alpha=0.3)
            
            # Add statistics text
            if len(plot_data) > 0:
                stats_text = []
                for method_label, values in zip(labels, plot_data):
                    if len(values) > 0:
                        mean_val = np.mean(values)
                        median_val = np.median(values)
                        stats_text.append(f'{method_label}: μ={mean_val:.4f}, med={median_val:.4f}')
                
                if stats_text:
                    ax.text(0.02, 0.98, '\n'.join(stats_text),
                           transform=ax.transAxes, fontsize=9,
                           verticalalignment='top',
                           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Set overall title
        variant_title = variant.capitalize()
        fig.suptitle(f'KL Divergence Comparison: GA vs OpenES ({variant_title})', 
                     fontsize=16, fontweight='bold', y=1.02)
        
        plt.tight_layout()
        
        # Save figure
        output_file = output_path / f'kl_boxplots_{variant}.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved boxplot to: {output_file}")
        
        # Also save as PDF
        output_file_pdf = output_path / f'kl_boxplots_{variant}.pdf'
        plt.savefig(output_file_pdf, bbox_inches='tight')
        print(f"Saved boxplot to: {output_file_pdf}")
        
        plt.close()


def print_summary_statistics(data: Dict):
    """Print summary statistics for the loaded data."""
    print("\n" + "="*80)
    print("KL Divergence Summary Statistics")
    print("="*80)
    
    for variant in ['continual', 'vanilla']:
        print(f"\n{variant.upper()}:")
        print("-" * 80)
        
        for task in ['acrobot', 'cartpole', 'mountaincar']:
            print(f"\n  {task.capitalize()}:")
            
            for method in ['ga', 'openes']:
                values = data[variant][task][method]
                
                if len(values) == 0:
                    print(f"    {method.upper()}: No data")
                    continue
                
                values_arr = np.array(values)
                print(f"    {method.upper()}:")
                print(f"      Count: {len(values)}")
                print(f"      Mean:  {np.mean(values_arr):.6f}")
                print(f"      Median: {np.median(values_arr):.6f}")
                print(f"      Std:   {np.std(values_arr):.6f}")
                print(f"      Min:   {np.min(values_arr):.6f}")
                print(f"      Max:   {np.max(values_arr):.6f}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Create boxplots for KL divergences comparing GA and OpenES"
    )
    parser.add_argument(
        "--base_dir",
        type=str,
        default="projects/reports/iclr_rebuttal/kl_divergence",
        help="Base directory containing KL divergence data (default: projects/reports/iclr_rebuttal/kl_divergence)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="projects/reports/iclr_rebuttal/kl_divergence/plots",
        help="Output directory for plots (default: projects/reports/iclr_rebuttal/kl_divergence/plots)"
    )
    
    args = parser.parse_args()
    
    print("Loading KL divergence data...")
    data = load_kl_data(args.base_dir)
    
    print("\nPrinting summary statistics...")
    print_summary_statistics(data)
    
    print("\nCreating boxplots...")
    create_boxplots(data, args.output_dir)
    
    print("\nDone!")


if __name__ == "__main__":
    main()

