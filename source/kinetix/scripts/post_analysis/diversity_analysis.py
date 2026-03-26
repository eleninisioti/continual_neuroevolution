#!/usr/bin/env python3
"""
Script to analyze diversity evolution across generations, averaging across trials.
Focuses only on diversity metrics and saves plots in the project directory.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances
import argparse
import os
import glob
from pathlib import Path
from scipy import stats

def load_archive_data(file_path):
    """Load archive data from file."""
    if file_path.endswith('.npy'):
        return np.load(file_path)
    elif file_path.endswith('.csv'):
        return np.loadtxt(file_path, delimiter=',')
    else:
        raise ValueError("Unsupported file format. Use .npy or .csv")

def calculate_diversity_direct(archive_history):
    """
    Calculate diversity metrics directly on original weight vectors.
    
    Args:
        archive_history: Shape (gens, population_size, features)
    
    Returns:
        diversity_metrics: Dictionary with various diversity measures
    """
    gens, pop_size, features = archive_history.shape
    
    print(f"Calculating diversity directly on {features}-dimensional weight vectors")
    
    # Initialize metrics
    mean_pairwise_distance = []
    std_pairwise_distance = []
    max_pairwise_distance = []
    centroid_spread = []
    feature_variance = []
    
    for gen in range(gens):
        generation_data = archive_history[gen, :, :]  # Shape: (pop_size, features)
        
        # 1. Mean pairwise distance
        pairwise_dists = pairwise_distances(generation_data)
        # Remove diagonal (self-distances)
        upper_triangle = np.triu_indices(pop_size, k=1)
        distances = pairwise_dists[upper_triangle]
        
        mean_pairwise_distance.append(np.mean(distances))
        std_pairwise_distance.append(np.std(distances))
        max_pairwise_distance.append(np.max(distances))
        
        # 2. Centroid spread (distance from centroid)
        centroid = np.mean(generation_data, axis=0)
        distances_from_centroid = np.linalg.norm(generation_data - centroid, axis=1)
        centroid_spread.append(np.mean(distances_from_centroid))
        
        # 3. Feature variance (sum of variances across all features)
        feature_variances = np.var(generation_data, axis=0)
        feature_variance.append(np.sum(feature_variances))
    
    return {
        'generations': np.arange(gens),
        'mean_pairwise_distance': np.array(mean_pairwise_distance),
        'std_pairwise_distance': np.array(std_pairwise_distance),
        'max_pairwise_distance': np.array(max_pairwise_distance),
        'centroid_spread': np.array(centroid_spread),
        'feature_variance': np.array(feature_variance)
    }


def find_trial_archive_files(project_dir):
    """
    Find all archive_history.npy files in trial directories.
    
    Args:
        project_dir: Path to project directory
    
    Returns:
        List of paths to archive_history.npy files
    """
    pattern = os.path.join(project_dir, "trial_*/data/train/archive_history.npy")
    archive_files = glob.glob(pattern)
    archive_files.sort()  # Sort for consistent ordering
    
    print(f"Found {len(archive_files)} archive files:")
    for file in archive_files:
        print(f"  {file}")
    
    return archive_files

def find_comparison_projects(base_project_dir):
    """
    Find both cont_False and cont_True versions of a project.
    
    Args:
        base_project_dir: Base project directory path
    
    Returns:
        Dictionary with 'normal' and 'lifelong' project directories
    """
    # Extract the base path without the continual part
    if 'cont_False' in base_project_dir:
        base_path = base_project_dir.replace('cont_False', 'cont_True')
    elif 'cont_True' in base_project_dir:
        base_path = base_project_dir.replace('cont_True', 'cont_False')
    elif 'continual_False' in base_project_dir:
        base_path = base_project_dir.replace('continual_False', 'continual_True')
    elif 'continual_True' in base_project_dir:
        base_path = base_project_dir.replace('continual_True', 'continual_False')
    else:
        # If no continual part found, assume we need to add it
        if not base_project_dir.endswith('/'):
            base_project_dir += '/'
        base_path = base_project_dir + 'cont_True'
        base_project_dir = base_project_dir + 'cont_False'
    
    projects = {}
    
    # Check for cont_False (normal)
    if os.path.exists(base_project_dir):
        projects['normal'] = base_project_dir
        print(f"Found normal project: {base_project_dir}")
    else:
        print(f"Normal project not found: {base_project_dir}")
    
    # Check for cont_True (lifelong)
    if os.path.exists(base_path):
        projects['lifelong'] = base_path
        print(f"Found lifelong project: {base_path}")
    else:
        print(f"Lifelong project not found: {base_path}")
    
    return projects

def load_and_process_trials(archive_files):
    """
    Load archive data from all trials and calculate diversity metrics.
    
    Args:
        archive_files: List of paths to archive_history.npy files
    
    Returns:
        List of diversity metrics dictionaries, one per trial
    """
    all_diversity_metrics = []
    
    for i, archive_file in enumerate(archive_files):
        print(f"\nProcessing trial {i+1}/{len(archive_files)}: {archive_file}")
        
        try:
            # Load archive data
            archive_history = load_archive_data(archive_file)
            print(f"  Archive data shape: {archive_history.shape}")
            
            # Calculate diversity metrics directly
            diversity_metrics = calculate_diversity_direct(archive_history)
            all_diversity_metrics.append(diversity_metrics)
            
            print(f"  Processed {len(diversity_metrics['generations'])} generations")
            
        except Exception as e:
            print(f"  Error processing {archive_file}: {e}")
            continue
    
    return all_diversity_metrics

def average_diversity_across_trials(all_diversity_metrics):
    """
    Average diversity metrics across all trials and calculate confidence intervals.
    
    Args:
        all_diversity_metrics: List of diversity metrics dictionaries
    
    Returns:
        Averaged diversity metrics dictionary with confidence intervals
    """
    if not all_diversity_metrics:
        raise ValueError("No diversity metrics to average")
    
    # Get the number of generations (should be the same for all trials)
    num_gens = len(all_diversity_metrics[0]['generations'])
    num_trials = len(all_diversity_metrics)
    
    # Initialize arrays for storing all trial data
    all_mean_pairwise_distance = np.zeros((num_trials, num_gens))
    all_std_pairwise_distance = np.zeros((num_trials, num_gens))
    all_max_pairwise_distance = np.zeros((num_trials, num_gens))
    all_centroid_spread = np.zeros((num_trials, num_gens))
    all_feature_variance = np.zeros((num_trials, num_gens))
    
    # Collect data from all trials
    for i, metrics in enumerate(all_diversity_metrics):
        all_mean_pairwise_distance[i] = metrics['mean_pairwise_distance']
        all_std_pairwise_distance[i] = metrics['std_pairwise_distance']
        all_max_pairwise_distance[i] = metrics['max_pairwise_distance']
        all_centroid_spread[i] = metrics['centroid_spread']
        all_feature_variance[i] = metrics['feature_variance']
    
    # Calculate means
    mean_pairwise_distance = np.mean(all_mean_pairwise_distance, axis=0)
    std_pairwise_distance = np.mean(all_std_pairwise_distance, axis=0)
    max_pairwise_distance = np.mean(all_max_pairwise_distance, axis=0)
    centroid_spread = np.mean(all_centroid_spread, axis=0)
    feature_variance = np.mean(all_feature_variance, axis=0)
    
    # Calculate confidence intervals (95% CI)
    confidence_level = 0.95
    alpha = 1 - confidence_level
    df = num_trials - 1  # degrees of freedom
    
    # Calculate standard error of the mean
    sem_mean_pairwise = np.std(all_mean_pairwise_distance, axis=0) / np.sqrt(num_trials)
    sem_std_pairwise = np.std(all_std_pairwise_distance, axis=0) / np.sqrt(num_trials)
    sem_max_pairwise = np.std(all_max_pairwise_distance, axis=0) / np.sqrt(num_trials)
    sem_centroid_spread = np.std(all_centroid_spread, axis=0) / np.sqrt(num_trials)
    sem_feature_variance = np.std(all_feature_variance, axis=0) / np.sqrt(num_trials)
    
    # Calculate t-value for confidence interval
    t_value = stats.t.ppf(1 - alpha/2, df)
    
    # Calculate confidence intervals
    ci_mean_pairwise = t_value * sem_mean_pairwise
    ci_std_pairwise = t_value * sem_std_pairwise
    ci_max_pairwise = t_value * sem_max_pairwise
    ci_centroid_spread = t_value * sem_centroid_spread
    ci_feature_variance = t_value * sem_feature_variance
    
    return {
        'generations': np.arange(num_gens),
        'mean_pairwise_distance': mean_pairwise_distance,
        'mean_pairwise_distance_ci': ci_mean_pairwise,
        'std_pairwise_distance': std_pairwise_distance,
        'std_pairwise_distance_ci': ci_std_pairwise,
        'max_pairwise_distance': max_pairwise_distance,
        'max_pairwise_distance_ci': ci_max_pairwise,
        'centroid_spread': centroid_spread,
        'centroid_spread_ci': ci_centroid_spread,
        'feature_variance': feature_variance,
        'feature_variance_ci': ci_feature_variance,
        'num_trials': num_trials
    }

def plot_diversity_comparison(project_metrics, save_dir):
    """
    Plot diversity evolution comparison between normal and lifelong learning.
    
    Args:
        project_metrics: Dictionary with 'normal' and 'lifelong' averaged metrics
        save_dir: Directory to save plot
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    # Define colors for normal vs lifelong
    colors = {'normal': 'blue', 'lifelong': 'red'}
    labels = {'normal': 'Normal', 'lifelong': 'Lifelong'}
    
    # Plot 1: Mean Pairwise Distance with Confidence Intervals
    ax = axes[0]
    for project_type, metrics in project_metrics.items():
        # Plot mean line
        ax.plot(metrics['generations'], metrics['mean_pairwise_distance'], 
                color=colors[project_type], linewidth=2, marker='o', markersize=4,
                label=labels[project_type])
        
        # Plot confidence intervals
        upper_bound = metrics['mean_pairwise_distance'] + metrics['mean_pairwise_distance_ci']
        lower_bound = metrics['mean_pairwise_distance'] - metrics['mean_pairwise_distance_ci']
        ax.fill_between(metrics['generations'], lower_bound, upper_bound, 
                       color=colors[project_type], alpha=0.2)
    
    ax.set_title('Mean Pairwise Distance (95% CI)', fontsize=12)
    ax.set_xlabel('Generation')
    ax.set_ylabel('Distance')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Plot 2: Standard Deviation of Pairwise Distance
    ax = axes[1]
    for project_type, metrics in project_metrics.items():
        ax.plot(metrics['generations'], metrics['std_pairwise_distance'], 
                color=colors[project_type], linewidth=2, marker='s', markersize=4,
                label=labels[project_type])
    ax.set_title('Std Dev of Pairwise Distance', fontsize=12)
    ax.set_xlabel('Generation')
    ax.set_ylabel('Standard Deviation')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Plot 3: Maximum Pairwise Distance
    ax = axes[2]
    for project_type, metrics in project_metrics.items():
        ax.plot(metrics['generations'], metrics['max_pairwise_distance'], 
                color=colors[project_type], linewidth=2, marker='^', markersize=4,
                label=labels[project_type])
    ax.set_title('Maximum Pairwise Distance', fontsize=12)
    ax.set_xlabel('Generation')
    ax.set_ylabel('Distance')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Plot 4: Centroid Spread
    ax = axes[3]
    for project_type, metrics in project_metrics.items():
        ax.plot(metrics['generations'], metrics['centroid_spread'], 
                color=colors[project_type], linewidth=2, marker='d', markersize=4,
                label=labels[project_type])
    ax.set_title('Centroid Spread', fontsize=12)
    ax.set_xlabel('Generation')
    ax.set_ylabel('Mean Distance from Centroid')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Plot 5: Feature Variance
    ax = axes[4]
    for project_type, metrics in project_metrics.items():
        ax.plot(metrics['generations'], metrics['feature_variance'], 
                color=colors[project_type], linewidth=2, marker='v', markersize=4,
                label=labels[project_type])
    ax.set_title('Feature Variance', fontsize=12)
    ax.set_xlabel('Generation')
    ax.set_ylabel('Sum of Feature Variances')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Plot 6: Combined Diversity Index (normalized)
    ax = axes[5]
    
    # Calculate global maxima across both projects for proper normalization
    all_mean_distances = np.concatenate([metrics['mean_pairwise_distance'] for metrics in project_metrics.values()])
    all_centroid_spreads = np.concatenate([metrics['centroid_spread'] for metrics in project_metrics.values()])
    all_feature_variances = np.concatenate([metrics['feature_variance'] for metrics in project_metrics.values()])
    
    global_max_mean_dist = np.max(all_mean_distances)
    global_max_centroid_spread = np.max(all_centroid_spreads)
    global_max_feature_var = np.max(all_feature_variances)
    
    for project_type, metrics in project_metrics.items():
        # Normalize metrics using global maxima
        normalized_mean_dist = metrics['mean_pairwise_distance'] / global_max_mean_dist
        normalized_centroid_spread = metrics['centroid_spread'] / global_max_centroid_spread
        normalized_feature_var = metrics['feature_variance'] / global_max_feature_var
        
        combined_diversity = (normalized_mean_dist + normalized_centroid_spread + normalized_feature_var) / 3
        
        ax.plot(metrics['generations'], combined_diversity, 
                color=colors[project_type], linewidth=3, marker='o', markersize=5,
                label=labels[project_type])
    ax.set_title('Combined Diversity Index', fontsize=12)
    ax.set_xlabel('Generation')
    ax.set_ylabel('Normalized Diversity')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Add overall title
    num_trials_normal = project_metrics['normal']['num_trials'] if 'normal' in project_metrics else 0
    num_trials_lifelong = project_metrics['lifelong']['num_trials'] if 'lifelong' in project_metrics else 0
    fig.suptitle(f'Diversity Evolution: Normal vs Lifelong Learning\n(Normal: {num_trials_normal} trials, Lifelong: {num_trials_lifelong} trials)', 
                 fontsize=16, y=0.98)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)
    plt.savefig(os.path.join(save_dir, 'diversity_evolution_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Diversity comparison plot saved to {save_dir}/diversity_evolution_comparison.png")
    
    # Also create a single comprehensive comparison plot
    plt.figure(figsize=(12, 8))
    
    # Calculate global maxima across both projects for proper normalization
    all_mean_distances = np.concatenate([metrics['mean_pairwise_distance'] for metrics in project_metrics.values()])
    all_centroid_spreads = np.concatenate([metrics['centroid_spread'] for metrics in project_metrics.values()])
    all_feature_variances = np.concatenate([metrics['feature_variance'] for metrics in project_metrics.values()])
    
    global_max_mean_dist = np.max(all_mean_distances)
    global_max_centroid_spread = np.max(all_centroid_spreads)
    global_max_feature_var = np.max(all_feature_variances)
    
    for project_type, metrics in project_metrics.items():
        # Normalize metrics using global maxima
        normalized_mean_dist = metrics['mean_pairwise_distance'] / global_max_mean_dist
        normalized_centroid_spread = metrics['centroid_spread'] / global_max_centroid_spread
        normalized_feature_var = metrics['feature_variance'] / global_max_feature_var
        
        combined_diversity = (normalized_mean_dist + normalized_centroid_spread + normalized_feature_var) / 3
        
        plt.plot(metrics['generations'], combined_diversity, 
                color=colors[project_type], linewidth=2, marker='o', markersize=4,
                label=f'{labels[project_type]} (Combined Diversity)')
    
    plt.xlabel('Generation', fontsize=12)
    plt.ylabel('Normalized Diversity', fontsize=12)
    plt.title(f'Diversity Evolution Comparison: Normal vs Lifelong Learning\n(Normal: {num_trials_normal} trials, Lifelong: {num_trials_lifelong} trials)', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'diversity_evolution_comparison_combined.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Combined diversity comparison plot saved to {save_dir}/diversity_evolution_comparison_combined.png")
    
    # Create standalone pairwise mean distance plot as PDF
    create_pairwise_distance_pdf(project_metrics, save_dir)

def create_pairwise_distance_pdf(project_metrics, save_dir):
    """
    Create a standalone PDF plot showing pairwise mean distance comparison.
    
    Args:
        project_metrics: Dictionary with 'normal' and 'lifelong' averaged metrics
        save_dir: Directory to save plot
    """
    # Define colors for normal vs lifelong
    colors = {'normal': 'blue', 'lifelong': 'red'}
    labels = {'normal': 'Normal', 'lifelong': 'Lifelong'}
    
    # Create figure with publication-quality settings
    plt.figure(figsize=(10, 6))
    
    # Plot pairwise mean distance for both project types with confidence intervals
    for project_type, metrics in project_metrics.items():
        # Plot mean line
        plt.plot(metrics['generations'], metrics['mean_pairwise_distance'], 
                color=colors[project_type], linewidth=3, marker='o', markersize=6,
                label=labels[project_type], alpha=0.8)
        
        # Plot confidence intervals
        upper_bound = metrics['mean_pairwise_distance'] + metrics['mean_pairwise_distance_ci']
        lower_bound = metrics['mean_pairwise_distance'] - metrics['mean_pairwise_distance_ci']
        plt.fill_between(metrics['generations'], lower_bound, upper_bound, 
                        color=colors[project_type], alpha=0.2)
    
    # Customize the plot
    plt.xlabel('Generation', fontsize=16, fontweight='bold')
    plt.ylabel('Mean Pairwise Distance', fontsize=16, fontweight='bold')
    
    # Add title with trial information
    num_trials_normal = project_metrics['normal']['num_trials'] if 'normal' in project_metrics else 0
    num_trials_lifelong = project_metrics['lifelong']['num_trials'] if 'lifelong' in project_metrics else 0
    plt.title(f'Mean Pairwise Distance Evolution (95% CI)\nNormal vs Lifelong Learning\n(Normal: {num_trials_normal} trials, Lifelong: {num_trials_lifelong} trials)', 
              fontsize=18, fontweight='bold', pad=20)
    
    # Customize legend
    plt.legend(fontsize=14, loc='best', frameon=True, fancybox=True, shadow=True)
    
    # Customize grid
    plt.grid(True, alpha=0.3, linestyle='--')
    
    # Customize axes
    plt.tick_params(axis='both', which='major', labelsize=14)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save as PDF
    pdf_path = os.path.join(save_dir, 'pairwise_mean_distance_comparison.pdf')
    plt.savefig(pdf_path, format='pdf', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Standalone pairwise mean distance plot saved to {pdf_path}")

def plot_diversity_evolution(averaged_metrics, save_dir):
    """
    Plot diversity evolution across generations (averaged across trials).
    
    Args:
        averaged_metrics: Averaged diversity metrics dictionary
        save_dir: Directory to save plot
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    # Plot 1: Mean Pairwise Distance
    axes[0].plot(averaged_metrics['generations'], averaged_metrics['mean_pairwise_distance'], 
                 'b-', linewidth=2, marker='o', markersize=4)
    axes[0].set_title('Mean Pairwise Distance', fontsize=12)
    axes[0].set_xlabel('Generation')
    axes[0].set_ylabel('Distance')
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Standard Deviation of Pairwise Distance
    axes[1].plot(averaged_metrics['generations'], averaged_metrics['std_pairwise_distance'], 
                 'g-', linewidth=2, marker='s', markersize=4)
    axes[1].set_title('Std Dev of Pairwise Distance', fontsize=12)
    axes[1].set_xlabel('Generation')
    axes[1].set_ylabel('Standard Deviation')
    axes[1].grid(True, alpha=0.3)
    
    # Plot 3: Maximum Pairwise Distance
    axes[2].plot(averaged_metrics['generations'], averaged_metrics['max_pairwise_distance'], 
                 'r-', linewidth=2, marker='^', markersize=4)
    axes[2].set_title('Maximum Pairwise Distance', fontsize=12)
    axes[2].set_xlabel('Generation')
    axes[2].set_ylabel('Distance')
    axes[2].grid(True, alpha=0.3)
    
    # Plot 4: Centroid Spread
    axes[3].plot(averaged_metrics['generations'], averaged_metrics['centroid_spread'], 
                 'm-', linewidth=2, marker='d', markersize=4)
    axes[3].set_title('Centroid Spread', fontsize=12)
    axes[3].set_xlabel('Generation')
    axes[3].set_ylabel('Mean Distance from Centroid')
    axes[3].grid(True, alpha=0.3)
    
    # Plot 5: Feature Variance
    axes[4].plot(averaged_metrics['generations'], averaged_metrics['feature_variance'], 
                 'c-', linewidth=2, marker='v', markersize=4)
    axes[4].set_title('Feature Variance', fontsize=12)
    axes[4].set_xlabel('Generation')
    axes[4].set_ylabel('Sum of Feature Variances')
    axes[4].grid(True, alpha=0.3)
    
    # Plot 6: Combined Diversity Index (normalized)
    normalized_mean_dist = averaged_metrics['mean_pairwise_distance'] / np.max(averaged_metrics['mean_pairwise_distance'])
    normalized_centroid_spread = averaged_metrics['centroid_spread'] / np.max(averaged_metrics['centroid_spread'])
    normalized_feature_var = averaged_metrics['feature_variance'] / np.max(averaged_metrics['feature_variance'])
    
    combined_diversity = (normalized_mean_dist + normalized_centroid_spread + normalized_feature_var) / 3
    
    axes[5].plot(averaged_metrics['generations'], combined_diversity, 
                 'k-', linewidth=3, marker='o', markersize=5)
    axes[5].set_title('Combined Diversity Index', fontsize=12)
    axes[5].set_xlabel('Generation')
    axes[5].set_ylabel('Normalized Diversity')
    axes[5].grid(True, alpha=0.3)
    
    # Add overall title
    fig.suptitle(f'Diversity Evolution Across Generations\n(Averaged across {averaged_metrics["num_trials"]} trials)', 
                 fontsize=16, y=0.98)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)
    plt.savefig(os.path.join(save_dir, 'diversity_evolution_averaged.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Diversity evolution plot saved to {save_dir}/diversity_evolution_averaged.png")
    
    # Also create a single comprehensive plot
    plt.figure(figsize=(12, 8))
    
    # Normalize all metrics to [0, 1] for comparison
    metrics_to_plot = {
        'Mean Pairwise Distance': averaged_metrics['mean_pairwise_distance'],
        'Centroid Spread': averaged_metrics['centroid_spread'],
        'Feature Variance': averaged_metrics['feature_variance'],
        'Combined Diversity': combined_diversity
    }
    
    colors = ['b', 'g', 'r', 'k']
    linestyles = ['-', '--', '-.', ':']
    
    for i, (name, values) in enumerate(metrics_to_plot.items()):
        # Normalize to [0, 1] except for combined diversity (already normalized)
        if name != 'Combined Diversity':
            normalized_values = values / np.max(values)
        else:
            normalized_values = values
            
        plt.plot(averaged_metrics['generations'], normalized_values, 
                color=colors[i], linestyle=linestyles[i], linewidth=2, 
                marker='o', markersize=4, label=name)
    
    plt.xlabel('Generation', fontsize=12)
    plt.ylabel('Normalized Diversity', fontsize=12)
    plt.title(f'Diversity Evolution Across Generations\n(Averaged across {averaged_metrics["num_trials"]} trials)', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'diversity_evolution_combined_averaged.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Combined diversity evolution plot saved to {save_dir}/diversity_evolution_combined_averaged.png")
    
    return averaged_metrics

def main():
    parser = argparse.ArgumentParser(description='Analyze diversity evolution across generations, comparing normal vs lifelong learning')
    parser.add_argument('--project_dir', type=str, default="projects/benchmarking/2025_09_20/gymnax_Acrobot-v1_max_steps_in_episode_200_noise_range_2.0/evosax_SimpleGA/generations_2000_strategy_SimpleGA_popsize_512_es_kws_{'sigma_init': 0.5, 'elite_ratio': 0.5}/MLP_num_layers_2_num_hidden_16_activation_relu_final_activation_linear_cont_False",
                       help='Path to project directory containing trial subdirectories')
    
    args = parser.parse_args()
    
    # Validate project directory
    if not os.path.exists(args.project_dir):
        raise ValueError(f"Project directory does not exist: {args.project_dir}")
    
    print(f"Analyzing diversity evolution comparison for project: {args.project_dir}")
    
    # Find both normal and lifelong projects
    projects = find_comparison_projects(args.project_dir)
    
    if not projects:
        raise ValueError("No projects found for comparison")
    
    project_metrics = {}
    
    # Process each project type
    for project_type, project_dir in projects.items():
        print(f"\n{'='*60}")
        print(f"Processing {project_type.upper()} project: {project_dir}")
        print(f"{'='*60}")
        
        # Find all trial archive files
        archive_files = find_trial_archive_files(project_dir)
        
        if not archive_files:
            print(f"No archive_history.npy files found in {project_dir}")
            continue
        
        # Load and process all trials
        all_diversity_metrics = load_and_process_trials(archive_files)
        
        if not all_diversity_metrics:
            print(f"No trials were successfully processed for {project_type}")
            continue
        
        # Average across trials
        print(f"\nAveraging diversity metrics across {len(all_diversity_metrics)} trials for {project_type}...")
        averaged_metrics = average_diversity_across_trials(all_diversity_metrics)
        project_metrics[project_type] = averaged_metrics
        
        print(f"DEBUG: Successfully processed {project_type} project with {len(all_diversity_metrics)} trials")
        
        # Save individual project metrics
        np.save(os.path.join(project_dir, 'averaged_diversity_metrics.npy'), averaged_metrics)
        print(f"Averaged diversity metrics saved to {project_dir}/averaged_diversity_metrics.npy")
    
    # Create comparison plots
    if len(project_metrics) > 1:
        print(f"\n{'='*60}")
        print("Creating diversity comparison plots...")
        print(f"{'='*60}")
        
        # Use the first project directory for saving comparison plots
        save_dir = list(projects.values())[0]
        plot_diversity_comparison(project_metrics, save_dir)
        
        print(f"\nComparison analysis completed!")
        print(f"Generated comparison plots saved in: {save_dir}")
        print(f"  - diversity_evolution_comparison.png")
        print(f"  - diversity_evolution_comparison_combined.png")
    else:
        print(f"\nOnly one project type found. Creating individual plots...")
        for project_type, metrics in project_metrics.items():
            plot_diversity_evolution(metrics, projects[project_type])

if __name__ == "__main__":
    main()
