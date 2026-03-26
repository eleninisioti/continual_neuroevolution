#!/usr/bin/env python3
"""
Script to visualize fitness history as a heatmap.
Shows evolution of fitness across generations and population.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os
import glob
from matplotlib.colors import LinearSegmentedColormap
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import pdist, squareform
from scipy import stats
import pandas as pd

def load_fitness_data(file_path):
    """Load fitness data from file."""
    if file_path.endswith('.npy'):
        return np.load(file_path)
    elif file_path.endswith('.csv'):
        return np.loadtxt(file_path, delimiter=',')
    else:
        raise ValueError("Unsupported file format. Use .npy or .csv")

def find_trial_fitness_files(project_dir):
    """
    Find all fitnesses_history.npy files in trial directories.
    
    Args:
        project_dir: Path to project directory
    
    Returns:
        List of paths to fitnesses_history.npy files
    """
    pattern = os.path.join(project_dir, "trial_*/data/train/fitnesses_history.npy")
    fitness_files = glob.glob(pattern)
    fitness_files.sort()  # Sort for consistent ordering
    
    print(f"Found {len(fitness_files)} fitness files:")
    for file in fitness_files:
        print(f"  {file}")
    
    return fitness_files

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

def process_all_trials(project_dir, title_suffix=""):
    """
    Process all trials in a project and create visualizations.
    
    Args:
        project_dir: Path to project directory
        title_suffix: Additional text for plot titles
    """
    print(f"Processing all trials in project: {project_dir}")
    
    # Find fitness files
    fitness_files = find_trial_fitness_files(project_dir)
    
    if not fitness_files:
        print(f"No fitnesses_history.npy files found in {project_dir}")
        return
    
    # Find archive files
    archive_files = find_trial_archive_files(project_dir)
    
    # Process each trial
    for i, fitness_file in enumerate(fitness_files):
        print(f"\n{'='*60}")
        print(f"Processing trial {i+1}/{len(fitness_files)}: {fitness_file}")
        print(f"{'='*60}")
        
        # Load fitness data
        fitness_history = load_fitness_data(fitness_file)
        print(f"Fitness data shape: {fitness_history.shape}")
        
        # Create trial-specific output directory
        trial_dir = os.path.dirname(fitness_file)
        trial_output_dir = os.path.join(trial_dir, "visuals", "train", "fitness")
        os.makedirs(trial_output_dir, exist_ok=True)
        
        # Create visualizations for this trial
        print("Creating basic fitness heatmap...")
        create_fitness_heatmap(fitness_history, trial_output_dir, f"{title_suffix}_trial_{i}")
        
        print("Creating detailed fitness heatmap...")
        create_detailed_fitness_heatmap(fitness_history, trial_output_dir, f"{title_suffix}_trial_{i}")
        
        print("Creating fitness distribution snapshots...")
        create_generation_snapshots(fitness_history, trial_output_dir, f"{title_suffix}_trial_{i}")
        
        print("Creating best individual highlights...")
        create_best_individual_highlights(fitness_history, trial_output_dir, f"{title_suffix}_trial_{i}")
        
        # Try to find corresponding archive file
        trial_archive_file = fitness_file.replace('fitnesses_history.npy', 'archive_history.npy')
        if os.path.exists(trial_archive_file):
            print(f"Loading archive data from {trial_archive_file}")
            archive_history = load_fitness_data(trial_archive_file)
            print(f"Archive data shape: {archive_history.shape}")
            
            print("Analyzing archive similarity...")
            analyze_archive_similarity(fitness_history, archive_history, trial_output_dir, f"{title_suffix}_trial_{i}")
        else:
            print(f"Archive file not found: {trial_archive_file}")
            print("Skipping archive similarity analysis...")
        
        # Save processed data
        np.save(os.path.join(trial_output_dir, 'fitness_history.npy'), fitness_history)
        print(f"Fitness data saved to {trial_output_dir}/fitness_history.npy")
    
    print(f"\nAll trials processed! Check individual trial directories for visualizations.")

def create_fitness_heatmap(fitness_history, save_dir, title_suffix=""):
    """
    Create a heatmap of fitness history.
    
    Args:
        fitness_history: Shape (num_gens, population_size)
        save_dir: Directory to save plots
        title_suffix: Additional text for plot title
    """
    print(f"Fitness history shape: {fitness_history.shape}")
    print(f"Fitness history dimensions: {fitness_history.ndim}")
    
    # Handle different possible shapes
    if fitness_history.ndim == 2:
        num_gens, pop_size = fitness_history.shape
    elif fitness_history.ndim == 3:
        # If 3D, take the first 2 dimensions
        num_gens, pop_size = fitness_history.shape[:2]
        fitness_history = fitness_history.reshape(num_gens, pop_size)
    else:
        raise ValueError(f"Unexpected fitness history shape: {fitness_history.shape}")
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(20, 12))
    
    # 1. Basic heatmap
    ax1 = axes[0, 0]
    im1 = ax1.imshow(fitness_history.T, aspect='auto', cmap='viridis', interpolation='nearest')
    ax1.set_title(f'Fitness Evolution Heatmap{title_suffix}\n(Darker = Higher Fitness)', fontsize=14)
    ax1.set_xlabel('Generation', fontsize=12)
    ax1.set_ylabel('Individual Index', fontsize=12)
    ax1.set_xticks(np.linspace(0, num_gens-1, min(10, num_gens), dtype=int))
    ax1.set_yticks(np.linspace(0, pop_size-1, min(10, pop_size), dtype=int))
    
    # Add colorbar
    cbar1 = plt.colorbar(im1, ax=ax1)
    cbar1.set_label('Fitness Value', fontsize=12)
    
    # 2. Heatmap with better colormap (red for high fitness)
    ax2 = axes[0, 1]
    im2 = ax2.imshow(fitness_history.T, aspect='auto', cmap='RdYlBu_r', interpolation='nearest')
    ax2.set_title(f'Fitness Evolution Heatmap{title_suffix}\n(Red = High, Blue = Low)', fontsize=14)
    ax2.set_xlabel('Generation', fontsize=12)
    ax2.set_ylabel('Individual Index', fontsize=12)
    ax2.set_xticks(np.linspace(0, num_gens-1, min(10, num_gens), dtype=int))
    ax2.set_yticks(np.linspace(0, pop_size-1, min(10, pop_size), dtype=int))
    
    # Add colorbar
    cbar2 = plt.colorbar(im2, ax=ax2)
    cbar2.set_label('Fitness Value', fontsize=12)
    
    # 3. Heatmap with seaborn styling
    ax3 = axes[1, 0]
    
    # Sample data for seaborn (every 40th generation and every 10th individual for performance)
    sample_gens = np.linspace(0, num_gens-1, min(50, num_gens), dtype=int)
    sample_inds = np.linspace(0, pop_size-1, min(50, pop_size), dtype=int)
    
    fitness_sample = fitness_history[np.ix_(sample_gens, sample_inds)]
    
    sns.heatmap(fitness_sample.T, ax=ax3, cmap='plasma', cbar_kws={'label': 'Fitness Value'})
    ax3.set_title(f'Fitness Evolution Heatmap{title_suffix}\n(Seaborn Style - Sampled)', fontsize=14)
    ax3.set_xlabel('Generation (Sampled)', fontsize=12)
    ax3.set_ylabel('Individual Index (Sampled)', fontsize=12)
    
    # Set ticks to show actual generation numbers
    ax3.set_xticks(np.linspace(0, len(sample_gens)-1, min(10, len(sample_gens)), dtype=int))
    ax3.set_yticks(np.linspace(0, len(sample_inds)-1, min(10, len(sample_inds)), dtype=int))
    
    # Set tick labels to show actual values
    ax3.set_xticklabels([str(sample_gens[i]) for i in np.linspace(0, len(sample_gens)-1, min(10, len(sample_gens)), dtype=int)])
    ax3.set_yticklabels([str(sample_inds[i]) for i in np.linspace(0, len(sample_inds)-1, min(10, len(sample_inds)), dtype=int)])
    
    # 4. Fitness statistics over time
    ax4 = axes[1, 1]
    
    # Calculate statistics
    mean_fitness = np.mean(fitness_history, axis=1)
    std_fitness = np.std(fitness_history, axis=1)
    max_fitness = np.max(fitness_history, axis=1)
    min_fitness = np.min(fitness_history, axis=1)
    
    generations = np.arange(num_gens)
    
    ax4.plot(generations, mean_fitness, 'b-', linewidth=2, label='Mean Fitness')
    ax4.fill_between(generations, mean_fitness - std_fitness, mean_fitness + std_fitness, 
                     alpha=0.3, color='blue', label='±1 Std Dev')
    ax4.plot(generations, max_fitness, 'r-', linewidth=2, label='Max Fitness')
    ax4.plot(generations, min_fitness, 'g-', linewidth=2, label='Min Fitness')
    
    ax4.set_title('Fitness Statistics Over Time', fontsize=14)
    ax4.set_xlabel('Generation', fontsize=12)
    ax4.set_ylabel('Fitness Value', fontsize=12)
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'fitness_heatmap{title_suffix}.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Fitness heatmap saved to {save_dir}/fitness_heatmap{title_suffix}.png")

def create_detailed_fitness_heatmap(fitness_history, save_dir, title_suffix=""):
    """
    Create a more detailed heatmap with additional analysis.
    
    Args:
        fitness_history: Shape (num_gens, population_size)
        save_dir: Directory to save plots
        title_suffix: Additional text for plot title
    """
    # Handle different possible shapes
    if fitness_history.ndim == 2:
        num_gens, pop_size = fitness_history.shape
    elif fitness_history.ndim == 3:
        # If 3D, take the first 2 dimensions
        num_gens, pop_size = fitness_history.shape[:2]
        fitness_history = fitness_history.reshape(num_gens, pop_size)
    else:
        raise ValueError(f"Unexpected fitness history shape: {fitness_history.shape}")
    
    # Create a large figure for detailed analysis
    fig = plt.figure(figsize=(24, 16))
    
    # 1. Main heatmap (large)
    ax1 = plt.subplot(2, 3, (1, 2))
    im1 = ax1.imshow(fitness_history.T, aspect='auto', cmap='RdYlBu_r', interpolation='nearest')
    ax1.set_title(f'Detailed Fitness Evolution Heatmap{title_suffix}\n(Red = High Fitness, Blue = Low Fitness)', 
                  fontsize=16)
    ax1.set_xlabel('Generation', fontsize=14)
    ax1.set_ylabel('Individual Index', fontsize=14)
    
    # Set ticks
    ax1.set_xticks(np.linspace(0, num_gens-1, min(20, num_gens), dtype=int))
    ax1.set_yticks(np.linspace(0, pop_size-1, min(20, pop_size), dtype=int))
    
    # Add colorbar
    cbar1 = plt.colorbar(im1, ax=ax1, shrink=0.8)
    cbar1.set_label('Fitness Value', fontsize=14)
    
    # 2. Fitness distribution histogram
    ax2 = plt.subplot(2, 3, 3)
    all_fitness = fitness_history.flatten()
    ax2.hist(all_fitness, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    ax2.set_title('Overall Fitness Distribution', fontsize=14)
    ax2.set_xlabel('Fitness Value', fontsize=12)
    ax2.set_ylabel('Frequency', fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    # Add statistics text
    mean_fit = np.mean(all_fitness)
    std_fit = np.std(all_fitness)
    ax2.axvline(mean_fit, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_fit:.3f}')
    ax2.axvline(mean_fit + std_fit, color='orange', linestyle='--', linewidth=2, label=f'+1σ: {mean_fit + std_fit:.3f}')
    ax2.axvline(mean_fit - std_fit, color='orange', linestyle='--', linewidth=2, label=f'-1σ: {mean_fit - std_fit:.3f}')
    ax2.legend(fontsize=10)
    
    # 3. Best individual trajectory
    ax3 = plt.subplot(2, 3, 4)
    best_individual_idx = np.argmax(np.max(fitness_history, axis=0))
    best_trajectory = fitness_history[:, best_individual_idx]
    
    ax3.plot(range(num_gens), best_trajectory, 'b-', linewidth=2, marker='o', markersize=4)
    ax3.set_title(f'Best Individual Trajectory\n(Individual {best_individual_idx})', fontsize=14)
    ax3.set_xlabel('Generation', fontsize=12)
    ax3.set_ylabel('Fitness Value', fontsize=12)
    ax3.grid(True, alpha=0.3)
    
    # Highlight best generation for this individual
    best_gen = np.argmax(best_trajectory)
    ax3.scatter(best_gen, best_trajectory[best_gen], color='red', s=100, zorder=5)
    ax3.annotate(f'Best: Gen {best_gen}', 
                xy=(best_gen, best_trajectory[best_gen]), 
                xytext=(10, 10), textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                fontsize=10)
    
    # 4. Population diversity over time
    ax4 = plt.subplot(2, 3, 5)
    
    # Calculate diversity as standard deviation
    diversity = np.std(fitness_history, axis=1)
    ax4.plot(range(num_gens), diversity, 'g-', linewidth=2, marker='s', markersize=4)
    ax4.set_title('Population Diversity Over Time\n(Std Dev of Fitness)', fontsize=14)
    ax4.set_xlabel('Generation', fontsize=12)
    ax4.set_ylabel('Fitness Std Dev', fontsize=12)
    ax4.grid(True, alpha=0.3)
    
    # 5. Fitness improvement over time
    ax5 = plt.subplot(2, 3, 6)
    
    # Calculate improvement metrics
    mean_fitness = np.mean(fitness_history, axis=1)
    max_fitness = np.max(fitness_history, axis=1)
    
    # Cumulative improvement
    cumulative_improvement = np.cumsum(np.diff(mean_fitness, prepend=mean_fitness[0]))
    
    ax5.plot(range(num_gens), mean_fitness, 'b-', linewidth=2, label='Mean Fitness')
    ax5.plot(range(num_gens), max_fitness, 'r-', linewidth=2, label='Max Fitness')
    ax5.plot(range(num_gens), cumulative_improvement, 'g--', linewidth=2, label='Cumulative Improvement')
    
    ax5.set_title('Fitness Improvement Over Time', fontsize=14)
    ax5.set_xlabel('Generation', fontsize=12)
    ax5.set_ylabel('Fitness Value', fontsize=12)
    ax5.legend(fontsize=10)
    ax5.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'detailed_fitness_heatmap{title_suffix}.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Detailed fitness heatmap saved to {save_dir}/detailed_fitness_heatmap{title_suffix}.png")

def create_best_individual_highlights(fitness_history, save_dir, title_suffix=""):
    """
    Create a heatmap highlighting the best performing individual at specific generations.
    
    Args:
        fitness_history: Shape (num_gens, population_size)
        save_dir: Directory to save plots
        title_suffix: Additional text for plot title
    """
    # Handle different possible shapes
    if fitness_history.ndim == 2:
        num_gens, pop_size = fitness_history.shape
    elif fitness_history.ndim == 3:
        # If 3D, take the first 2 dimensions
        num_gens, pop_size = fitness_history.shape[:2]
        fitness_history = fitness_history.reshape(num_gens, pop_size)
    else:
        raise ValueError(f"Unexpected fitness history shape: {fitness_history.shape}")
    
    # Select generations: 199, 399, 599, etc. (every 200 generations starting from 199)
    highlight_gens = []
    for i in range(199, num_gens, 200):
        if i < num_gens:
            highlight_gens.append(i)
    
    # Limit to reasonable number for visualization
    highlight_gens = highlight_gens[:10]  # Show max 10 highlighted generations
    
    fig, ax = plt.subplots(figsize=(16, 10))
    
    # Create the main heatmap
    im = ax.imshow(fitness_history.T, aspect='auto', cmap='viridis', interpolation='nearest')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Fitness Value', fontsize=14)
    
    # Define colors for bounding boxes
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
    
    # Highlight best individuals at specific generations
    for i, gen in enumerate(highlight_gens):
        # Find best individual at this generation
        best_individual = np.argmax(fitness_history[gen, :])
        
        # Create bounding box around the best individual's row
        color = colors[i % len(colors)]
        
        # Draw rectangle around the entire row for this generation
        rect = plt.Rectangle((gen-0.5, best_individual-0.5), 1, 1, 
                           linewidth=3, edgecolor=color, facecolor='none', alpha=0.8)
        ax.add_patch(rect)
        
        # Add text annotation
        ax.annotate(f'Gen {gen}\nBest: {best_individual}', 
                   xy=(gen, best_individual), 
                   xytext=(10, 10), textcoords='offset points',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.7),
                   fontsize=8, color='white', weight='bold')
    
    ax.set_title(f'Best Individual Highlights{title_suffix}\n(Colored boxes show best individual at each highlighted generation)', 
                fontsize=16)
    ax.set_xlabel('Generation', fontsize=14)
    ax.set_ylabel('Individual Index', fontsize=14)
    
    # Set ticks
    ax.set_xticks(np.linspace(0, num_gens-1, min(20, num_gens), dtype=int))
    ax.set_yticks(np.linspace(0, pop_size-1, min(20, pop_size), dtype=int))
    
    # Add legend for colors
    legend_elements = []
    for i, gen in enumerate(highlight_gens):
        color = colors[i % len(colors)]
        legend_elements.append(plt.Rectangle((0,0),1,1, facecolor=color, alpha=0.7, 
                                           label=f'Gen {gen}'))
    
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'best_individual_highlights{title_suffix}.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Best individual highlights plot saved to {save_dir}/best_individual_highlights{title_suffix}.png")
    
    # Also create a summary table
    print(f"\nBest Individuals Summary:")
    print(f"{'Generation':<12} {'Best Individual':<15} {'Fitness Value':<15}")
    print("-" * 45)
    for i, gen in enumerate(highlight_gens):
        best_individual = np.argmax(fitness_history[gen, :])
        best_fitness = fitness_history[gen, best_individual]
        print(f"{gen:<12} {best_individual:<15} {best_fitness:<15.3f}")
    
    # Create a separate heatmap showing only the best individuals' rows
    create_best_individuals_rows_heatmap(fitness_history, highlight_gens, save_dir, title_suffix)

def create_best_individuals_rows_heatmap(fitness_history, highlight_gens, save_dir, title_suffix=""):
    """
    Create a smaller heatmap showing only the rows of the best individuals.
    
    Args:
        fitness_history: Shape (num_gens, population_size)
        highlight_gens: List of generations to highlight
        save_dir: Directory to save plots
        title_suffix: Additional text for plot title
    """
    # Handle different possible shapes
    if fitness_history.ndim == 2:
        num_gens, pop_size = fitness_history.shape
    elif fitness_history.ndim == 3:
        # If 3D, take the first 2 dimensions
        num_gens, pop_size = fitness_history.shape[:2]
        fitness_history = fitness_history.reshape(num_gens, pop_size)
    else:
        raise ValueError(f"Unexpected fitness history shape: {fitness_history.shape}")
    
    # Get the best individuals for each highlighted generation
    best_individuals = []
    best_fitness_values = []
    
    for gen in highlight_gens:
        best_individual = np.argmax(fitness_history[gen, :])
        best_fitness = fitness_history[gen, best_individual]
        best_individuals.append(best_individual)
        best_fitness_values.append(best_fitness)
    
    # Extract rows for best individuals
    best_rows_data = fitness_history[:, best_individuals]  # Shape: (num_gens, num_best_individuals)
    
    # Create the heatmap
    fig, ax = plt.subplots(figsize=(16, 8))
    
    # Create the heatmap
    im = ax.imshow(best_rows_data.T, aspect='auto', cmap='viridis', interpolation='nearest')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Fitness Value', fontsize=14)
    
    # Set labels and title
    ax.set_title(f'Best Individuals Rows Heatmap{title_suffix}\n(Each row shows fitness evolution of best individual at highlighted generation)', 
                fontsize=16)
    ax.set_xlabel('Generation', fontsize=14)
    ax.set_ylabel('Best Individual Index', fontsize=14)
    
    # Set y-axis labels to show the actual individual indices
    ax.set_yticks(range(len(best_individuals)))
    ax.set_yticklabels([f'Individual {idx}' for idx in best_individuals])
    
    # Set x-axis ticks
    ax.set_xticks(np.linspace(0, num_gens-1, min(20, num_gens), dtype=int))
    
    # Add vertical lines at highlighted generations
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
    for i, gen in enumerate(highlight_gens):
        color = colors[i % len(colors)]
        ax.axvline(x=gen, color=color, linestyle='--', linewidth=2, alpha=0.7)
        
        # Add text annotation
        ax.text(gen, len(best_individuals)-0.5, f'Gen {gen}', 
               rotation=90, ha='right', va='top', fontsize=10, 
               bbox=dict(boxstyle='round,pad=0.2', facecolor=color, alpha=0.7))
    
    # Add legend for the vertical lines
    legend_elements = []
    for i, gen in enumerate(highlight_gens):
        color = colors[i % len(colors)]
        legend_elements.append(plt.Line2D([0], [0], color=color, linewidth=2, 
                                       label=f'Gen {gen} (Individual {best_individuals[i]})'))
    
    ax.legend(handles=legend_elements, loc='upper left', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'best_individuals_rows{title_suffix}.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Best individuals rows heatmap saved to {save_dir}/best_individuals_rows{title_suffix}.png")
    
    # Also create a summary of the best individuals' performance over time
    print(f"\nBest Individuals Performance Summary:")
    print(f"{'Individual':<12} {'Peak Gen':<10} {'Peak Fitness':<15} {'Final Fitness':<15}")
    print("-" * 60)
    for i, individual in enumerate(best_individuals):
        individual_fitness = fitness_history[:, individual]
        peak_gen = np.argmax(individual_fitness)
        peak_fitness = individual_fitness[peak_gen]
        final_fitness = individual_fitness[-1]
        print(f"{individual:<12} {peak_gen:<10} {peak_fitness:<15.3f} {final_fitness:<15.3f}")
    
    # Create a grouped bar plot comparing performance at highlighted generations
    create_best_individuals_comparison_plot(fitness_history, highlight_gens, save_dir, title_suffix)

def create_best_individuals_comparison_plot(fitness_history, highlight_gens, save_dir, title_suffix=""):
    """
    Create a grouped bar plot comparing how best individuals perform at highlighted generations.
    
    Args:
        fitness_history: Shape (num_gens, population_size)
        highlight_gens: List of generations to highlight
        save_dir: Directory to save plots
        title_suffix: Additional text for plot title
    """
    # Handle different possible shapes
    if fitness_history.ndim == 2:
        num_gens, pop_size = fitness_history.shape
    elif fitness_history.ndim == 3:
        # If 3D, take the first 2 dimensions
        num_gens, pop_size = fitness_history.shape[:2]
        fitness_history = fitness_history.reshape(num_gens, pop_size)
    else:
        raise ValueError(f"Unexpected fitness history shape: {fitness_history.shape}")
    
    # Get the best individuals for each highlighted generation
    best_individuals = []
    best_fitness_values = []
    
    for gen in highlight_gens:
        best_individual = np.argmax(fitness_history[gen, :])
        best_fitness = fitness_history[gen, best_individual]
        best_individuals.append(best_individual)
        best_fitness_values.append(best_fitness)
    
    # Create the comparison plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12))
    
    # Plot 1: Bar plot showing fitness at highlighted generations
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
    
    x_pos = np.arange(len(highlight_gens))
    bars = ax1.bar(x_pos, best_fitness_values, color=colors[:len(highlight_gens)], alpha=0.7, edgecolor='black')
    
    # Add value labels on bars
    for i, (bar, value) in enumerate(zip(bars, best_fitness_values)):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{value:.1f}', ha='center', va='bottom', fontweight='bold')
    
    ax1.set_xlabel('Highlighted Generation', fontsize=14)
    ax1.set_ylabel('Fitness Value', fontsize=14)
    ax1.set_title(f'Best Individual Performance at Highlighted Generations{title_suffix}', fontsize=16)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels([f'Gen {gen}\n(Individual {best_individuals[i]})' for i, gen in enumerate(highlight_gens)])
    ax1.grid(True, alpha=0.3)
    
    # Add legend
    legend_elements = [plt.Rectangle((0,0),1,1, facecolor=colors[i], alpha=0.7, 
                                   label=f'Gen {gen} (Individual {best_individuals[i]})') 
                      for i, gen in enumerate(highlight_gens)]
    ax1.legend(handles=legend_elements, loc='upper right', fontsize=10)
    
    # Plot 2: Grouped bar plot showing all best individuals' performance at each generation
    # Create a matrix where each row is a best individual and each column is a highlighted generation
    comparison_matrix = np.zeros((len(best_individuals), len(highlight_gens)))
    
    for i, individual in enumerate(best_individuals):
        for j, gen in enumerate(highlight_gens):
            comparison_matrix[i, j] = fitness_history[gen, individual]
    
    # Create grouped bar plot
    x_pos = np.arange(len(highlight_gens))
    width = 0.8 / len(best_individuals)  # Width of bars
    
    for i, individual in enumerate(best_individuals):
        offset = (i - len(best_individuals)/2 + 0.5) * width
        bars = ax2.bar(x_pos + offset, comparison_matrix[i, :], width, 
                      label=f'Individual {individual}', alpha=0.7, edgecolor='black')
        
        # Add value labels on bars
        for bar, value in zip(bars, comparison_matrix[i, :]):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{value:.1f}', ha='center', va='bottom', fontsize=8)
    
    ax2.set_xlabel('Highlighted Generation', fontsize=14)
    ax2.set_ylabel('Fitness Value', fontsize=14)
    ax2.set_title(f'All Best Individuals Performance Comparison{title_suffix}\n(Each individual\'s performance at each highlighted generation)', fontsize=16)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([f'Gen {gen}' for gen in highlight_gens])
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'best_individuals_comparison{title_suffix}.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Best individuals comparison plot saved to {save_dir}/best_individuals_comparison{title_suffix}.png")
    
    # Create a summary table
    print(f"\nBest Individuals Comparison Table:")
    print(f"{'Generation':<12}", end='')
    for individual in best_individuals:
        print(f"{'Individual ' + str(individual):<15}", end='')
    print()
    print("-" * (12 + 15 * len(best_individuals)))
    
    for gen in highlight_gens:
        print(f"{gen:<12}", end='')
        for individual in best_individuals:
            fitness = fitness_history[gen, individual]
            print(f"{fitness:<15.3f}", end='')
        print()

def analyze_archive_similarity(fitness_history, archive_history, save_dir, title_suffix=""):
    """
    Analyze whether best individuals are more similar to each other compared to non-best individuals.
    
    Args:
        fitness_history: Shape (num_gens, population_size)
        archive_history: Shape (num_gens, population_size, phenotype_size)
        save_dir: Directory to save plots
        title_suffix: Additional text for plot title
    """
    print(f"Archive history shape: {archive_history.shape}")
    print(f"Archive history dimensions: {archive_history.ndim}")
    
    # Handle different possible shapes for archive data
    if archive_history.ndim == 3:
        num_gens, pop_size, phenotype_size = archive_history.shape
    elif archive_history.ndim == 4:
        # If 4D, take the first 3 dimensions
        num_gens, pop_size, phenotype_size = archive_history.shape[:3]
        archive_history = archive_history.reshape(num_gens, pop_size, phenotype_size)
    else:
        raise ValueError(f"Unexpected archive history shape: {archive_history.shape}")
    
    # Select generations: 199, 399, 599, etc. (every 200 generations starting from 199)
    highlight_gens = []
    for i in range(199, num_gens, 200):
        if i < num_gens:
            highlight_gens.append(i)
    
    # Limit to reasonable number for analysis
    highlight_gens = highlight_gens[:10]  # Analyze first 10 generations
    
    print(f"Analyzing similarity for generations: {highlight_gens}")
    
    # Get best individuals for each generation
    best_individuals = []
    for gen in highlight_gens:
        best_individual = np.argmax(fitness_history[gen, :])
        best_individuals.append(best_individual)
    
    # Remove duplicates while preserving order
    unique_best_individuals = list(dict.fromkeys(best_individuals))
    print(f"Unique best individuals: {unique_best_individuals}")
    
    # Get non-best individuals (all others)
    all_individuals = set(range(pop_size))
    non_best_individuals = list(all_individuals - set(unique_best_individuals))
    
    print(f"Number of best individuals: {len(unique_best_individuals)}")
    print(f"Number of non-best individuals: {len(non_best_individuals)}")
    
    # Perform analysis for each generation
    results = []
    
    for gen in highlight_gens:
        print(f"\nAnalyzing generation {gen}...")
        
        # Get archive data for this generation
        gen_archive = archive_history[gen, :, :]  # Shape: (pop_size, phenotype_size)
        
        # Calculate pairwise distances
        distances = pairwise_distances(gen_archive, metric='euclidean')
        
        # Get distances within best individuals
        best_distances = []
        for i in range(len(unique_best_individuals)):
            for j in range(i+1, len(unique_best_individuals)):
                idx1, idx2 = unique_best_individuals[i], unique_best_individuals[j]
                best_distances.append(distances[idx1, idx2])
        
        # Get distances within non-best individuals (sample to avoid memory issues)
        non_best_sample = np.random.choice(non_best_individuals, 
                                          min(50, len(non_best_individuals)), 
                                          replace=False)
        non_best_distances = []
        for i in range(len(non_best_sample)):
            for j in range(i+1, len(non_best_sample)):
                idx1, idx2 = non_best_sample[i], non_best_sample[j]
                non_best_distances.append(distances[idx1, idx2])
        
        # Statistical test
        if len(best_distances) > 1 and len(non_best_distances) > 1:
            t_stat, p_value = stats.ttest_ind(best_distances, non_best_distances)
            
            results.append({
                'generation': gen,
                'best_mean_dist': np.mean(best_distances),
                'best_std_dist': np.std(best_distances),
                'non_best_mean_dist': np.mean(non_best_distances),
                'non_best_std_dist': np.std(non_best_distances),
                't_statistic': t_stat,
                'p_value': p_value,
                'best_distances': best_distances,
                'non_best_distances': non_best_distances
            })
            
            print(f"  Best individuals mean distance: {np.mean(best_distances):.3f} ± {np.std(best_distances):.3f}")
            print(f"  Non-best individuals mean distance: {np.mean(non_best_distances):.3f} ± {np.std(non_best_distances):.3f}")
            print(f"  t-statistic: {t_stat:.3f}, p-value: {p_value:.3f}")
            
            if p_value < 0.05:
                if np.mean(best_distances) < np.mean(non_best_distances):
                    print(f"  → Best individuals are MORE similar (p < 0.05)")
                else:
                    print(f"  → Best individuals are LESS similar (p < 0.05)")
            else:
                print(f"  → No significant difference in similarity (p ≥ 0.05)")
    
    # Create visualizations (even if no statistical results)
    create_similarity_visualizations(results, archive_history, fitness_history, unique_best_individuals, 
                                   highlight_gens, save_dir, title_suffix)
    
    # Summary statistics
    print(f"\n=== SUMMARY ===")
    significant_cases = sum(1 for r in results if r['p_value'] < 0.05)
    more_similar_cases = sum(1 for r in results if r['p_value'] < 0.05 and r['best_mean_dist'] < r['non_best_mean_dist'])
    
    print(f"Generations analyzed: {len(results)}")
    print(f"Significant differences found: {significant_cases}/{len(results)}")
    print(f"Cases where best individuals are more similar: {more_similar_cases}/{significant_cases}")
    
    if significant_cases > 0:
        avg_effect = np.mean([r['best_mean_dist'] - r['non_best_mean_dist'] for r in results if r['p_value'] < 0.05])
        print(f"Average difference in similarity: {avg_effect:.3f}")
        if avg_effect < 0:
            print("→ Overall trend: Best individuals tend to be MORE similar")
        else:
            print("→ Overall trend: Best individuals tend to be LESS similar")

def create_similarity_visualizations(results, archive_history, fitness_history, best_individuals, highlight_gens, save_dir, title_suffix=""):
    """
    Create visualizations for the similarity analysis.
    """
    # 1. Distance comparison plot
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Box plot comparison
    ax1 = axes[0, 0]
    generations = [r['generation'] for r in results]
    best_means = [r['best_mean_dist'] for r in results]
    non_best_means = [r['non_best_mean_dist'] for r in results]
    
    x_pos = np.arange(len(generations))
    width = 0.35
    
    bars1 = ax1.bar(x_pos - width/2, best_means, width, label='Best Individuals', alpha=0.7, color='red')
    bars2 = ax1.bar(x_pos + width/2, non_best_means, width, label='Non-Best Individuals', alpha=0.7, color='blue')
    
    ax1.set_xlabel('Generation', fontsize=12)
    ax1.set_ylabel('Mean Pairwise Distance', fontsize=12)
    ax1.set_title('Mean Pairwise Distance Comparison', fontsize=14)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels([f'Gen {g}' for g in generations])
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add significance markers
    for i, r in enumerate(results):
        if r['p_value'] < 0.05:
            ax1.text(i, max(r['best_mean_dist'], r['non_best_mean_dist']) + 0.5, '*', 
                    ha='center', va='bottom', fontsize=16, color='red')
    
    # Plot 2: P-values
    ax2 = axes[0, 1]
    p_values = [r['p_value'] for r in results]
    bars = ax2.bar(x_pos, p_values, alpha=0.7, color='green')
    ax2.axhline(y=0.05, color='red', linestyle='--', label='p = 0.05')
    ax2.set_xlabel('Generation', fontsize=12)
    ax2.set_ylabel('P-value', fontsize=12)
    ax2.set_title('Statistical Significance', fontsize=14)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([f'Gen {g}' for g in generations])
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: PCA visualization showing evolution over time
    ax3 = axes[1, 0]
    if len(results) > 0:
        # Use all highlighted generations for PCA
        all_gen_archives = []
        all_gen_labels = []
        
        for gen in highlight_gens:
            gen_archive = archive_history[gen, :, :]
            all_gen_archives.append(gen_archive)
            all_gen_labels.extend([gen] * gen_archive.shape[0])
        
        # Combine all generations for PCA
        combined_archive = np.vstack(all_gen_archives)  # Shape: (num_gens * pop_size, phenotype_size)
        
        # Perform PCA on combined data
        pca = PCA(n_components=2)
        pca_data = pca.fit_transform(combined_archive)
        
        # Plot all individuals with generation-based color gradient
        scatter = ax3.scatter(pca_data[:, 0], pca_data[:, 1], c=all_gen_labels, alpha=0.4, s=20, 
                            cmap='viridis')
        
        # Highlight best individuals across all generations with individual numbers
        # First, collect all positions and calculate smart offsets
        positions = []
        for i, gen in enumerate(highlight_gens):
            gen_start_idx = highlight_gens.index(gen) * archive_history.shape[1]
            best_individual = np.argmax(fitness_history[gen, :])
            best_idx = gen_start_idx + best_individual
            positions.append((pca_data[best_idx, 0], pca_data[best_idx, 1], gen))
        
        # Calculate smart offsets to avoid overlaps
        offsets = []
        for i, (x, y, gen) in enumerate(positions):
            # Start with a base offset
            base_distance = 30
            angle = (i * 36) % 360  # 36 degrees = 360/10 for 10 labels
            
            # Check for nearby positions and adjust angle if needed
            for j, (other_x, other_y, other_gen) in enumerate(positions):
                if i != j:
                    distance = np.sqrt((x - other_x)**2 + (y - other_y)**2)
                    if distance < 8:  # If points are close, spread labels more
                        angle_offset = (i - j) * 40  # Add extra angle separation
                        angle = (angle + angle_offset) % 360
            
            # Calculate final offset
            offset_distance = base_distance + i * 2
            offset_x = offset_distance * np.cos(np.radians(angle))
            offset_y = offset_distance * np.sin(np.radians(angle))
            offsets.append((offset_x, offset_y))
        
        # Now plot all points and labels
        for i, ((x, y, gen), (offset_x, offset_y)) in enumerate(zip(positions, offsets)):
            # Plot each best individual as a black dot
            ax3.scatter(x, y, alpha=0.9, color='black', s=60, 
                       edgecolors='white', linewidth=1)
            
            # Add generation number with arrow pointing to the dot
            ax3.annotate(f'Gen {gen}', 
                        xy=(x, y),
                        xytext=(offset_x, offset_y), textcoords='offset points',
                        fontsize=7, fontweight='bold', color='black',
                        bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.9, edgecolor='black'),
                        arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.1', 
                                      color='black', lw=1))
        
        ax3.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})', fontsize=12)
        ax3.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})', fontsize=12)
        ax3.set_title(f'PCA Visualization - Evolution Over Time\n(Color = Generation Index)', fontsize=14)
        ax3.grid(True, alpha=0.3)
        
        # Add colorbar for generation
        cbar = plt.colorbar(scatter, ax=ax3)
        cbar.set_label('Generation Index', fontsize=10)
        
        # Create standalone PCA visualization PDF
        create_pca_visualization_pdf(pca_data, all_gen_labels, fitness_history, archive_history, 
                                   highlight_gens, save_dir, title_suffix)
    
    # Plot 4: Distance distributions
    ax4 = axes[1, 1]
    if len(results) > 0:
        r = results[0]  # Use first generation
        ax4.hist(r['best_distances'], alpha=0.7, label='Best Individuals', bins=20, color='red')
        ax4.hist(r['non_best_distances'], alpha=0.7, label='Non-Best Individuals', bins=20, color='blue')
        ax4.set_xlabel('Pairwise Distance', fontsize=12)
        ax4.set_ylabel('Frequency', fontsize=12)
        ax4.set_title(f'Distance Distributions - Generation {r["generation"]}', fontsize=14)
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'archive_similarity_analysis{title_suffix}.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Archive similarity analysis plot saved to {save_dir}/archive_similarity_analysis{title_suffix}.png")

def create_pca_visualization_pdf(pca_data, all_gen_labels, fitness_history, archive_history, 
                               highlight_gens, save_dir, title_suffix=""):
    """
    Create a standalone PDF plot showing PCA visualization with best individuals highlighted.
    
    Args:
        pca_data: PCA transformed data
        all_gen_labels: Generation labels for all data points
        fitness_history: Fitness data for finding best individuals
        archive_history: Archive data for PCA
        highlight_gens: List of generations to highlight
        save_dir: Directory to save plot
        title_suffix: Additional text for plot title
    """
    # Create figure with publication-quality settings (one third of A4)
    plt.figure(figsize=(5.5, 3.5))  # One third of A4 (8.27" x 11.69")
    
    # Plot all individuals with generation-based color gradient
    scatter = plt.scatter(pca_data[:, 0], pca_data[:, 1], c=all_gen_labels, alpha=0.4, s=40, 
                        cmap='viridis')
    
    # Highlight best individuals across all generations with individual numbers
    # First, collect all positions and calculate smart offsets
    positions = []
    for i, gen in enumerate(highlight_gens):
        gen_start_idx = highlight_gens.index(gen) * archive_history.shape[1]
        best_individual = np.argmax(fitness_history[gen, :])
        best_idx = gen_start_idx + best_individual
        positions.append((pca_data[best_idx, 0], pca_data[best_idx, 1], gen))
    
    # Calculate smart offsets to avoid overlaps
    offsets = []
    for i, (x, y, gen) in enumerate(positions):
        # Start with a base offset
        base_distance = 25
        angle = (i * 40) % 360  # 40 degrees separation for better spacing
        
        # Check for nearby positions and adjust angle if needed
        for j, (other_x, other_y, other_gen) in enumerate(positions):
            if i != j:
                distance = np.sqrt((x - other_x)**2 + (y - other_y)**2)
                if distance < 15:  # If points are close, spread labels more
                    angle_offset = (i - j) * 60  # Add extra angle separation
                    angle = (angle + angle_offset) % 360
        
        # Calculate final offset with better spacing
        offset_distance = base_distance + i * 2
        offset_x = offset_distance * np.cos(np.radians(angle))
        offset_y = offset_distance * np.sin(np.radians(angle))
        
        # Check if label would go out of bounds on the right side
        if offset_x > 0:  # If label is on the right side
            # Position label to the left of the point instead
            offset_x = -abs(offset_x)  # Make it negative to go left
        
        offsets.append((offset_x, offset_y))
    
    # Now plot all points and labels
    for i, ((x, y, gen), (offset_x, offset_y)) in enumerate(zip(positions, offsets)):
        # Plot each best individual with their original color from the colormap
        # Get the color from the viridis colormap based on generation
        color = plt.cm.viridis(gen / max(all_gen_labels))
        plt.scatter(x, y, alpha=0.9, color=color, s=120, 
                   edgecolors='white', linewidth=2)
        
        # Add generation number with arrow pointing to the dot
        plt.annotate(f'Gen {gen}', 
                    xy=(x, y),
                    xytext=(offset_x, offset_y), textcoords='offset points',
                    fontsize=9, fontweight='bold', color='black',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9, edgecolor='black'),
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.1', 
                                  color='black', lw=1.5))
    
    # Customize the plot - remove axis labels
    # plt.xlabel('PC1', fontsize=11, fontweight='bold')  # Removed
    # plt.ylabel('PC2', fontsize=11, fontweight='bold')  # Removed
    
    # Customize grid
    plt.grid(True, alpha=0.3, linestyle='--')
    
    # Customize axes - show tick labels with smaller font
    plt.tick_params(axis='both', which='major', labelsize=8)
    
    # Set tick locations to show range
    x_min, x_max = pca_data[:, 0].min(), pca_data[:, 0].max()
    y_min, y_max = pca_data[:, 1].min(), pca_data[:, 1].max()
    
    # Set 5 ticks across the range
    plt.xticks(np.linspace(x_min, x_max, 5))
    plt.yticks(np.linspace(y_min, y_max, 5))
    
    # Add colorbar for generation (without label)
    cbar = plt.colorbar(scatter)
    # cbar.set_label('Generation Index', fontsize=9, fontweight='bold')  # Removed label
    cbar.ax.tick_params(labelsize=9)
    
    # Apply tight layout
    plt.tight_layout()
    
    # Save original PDF with text labels
    pdf_path = os.path.join(save_dir, f'pca_visualization_evolution{title_suffix}.pdf')
    plt.savefig(pdf_path, format='pdf', dpi=300, bbox_inches='tight')
    print(f"Standalone PCA visualization plot saved to {pdf_path}")
    
    # Save version without text labels
    pdf_path_no_labels = os.path.join(save_dir, f'pca_visualization_evolution_no_labels{title_suffix}.pdf')
    # Remove all text annotations
    for artist in plt.gca().get_children():
        if hasattr(artist, 'get_text') and artist.get_text():
            artist.remove()
    plt.savefig(pdf_path_no_labels, format='pdf', dpi=300, bbox_inches='tight')
    print(f"PCA visualization without labels saved to {pdf_path_no_labels}")
    
    # Save individual text labels as separate PDFs with transparent background
    for i, ((x, y, gen), (offset_x, offset_y)) in enumerate(zip(positions, offsets)):
        # Create a new figure for each label
        fig, ax = plt.subplots(figsize=(2, 1))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        
        # Add the text label
        ax.annotate(f'Gen {gen}', 
                    xy=(0.5, 0.5),
                    xytext=(0, 0), textcoords='offset points',
                    fontsize=9, fontweight='bold', color='black',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9, edgecolor='black'),
                    ha='center', va='center')
        
        # Save with transparent background
        label_pdf_path = os.path.join(save_dir, f'pca_label_gen_{gen}{title_suffix}.pdf')
        plt.savefig(label_pdf_path, format='pdf', dpi=300, bbox_inches='tight', 
                   facecolor='none', edgecolor='none', transparent=True)
        plt.close()
        print(f"Individual label for Gen {gen} saved to {label_pdf_path}")
    
    plt.close()

def create_generation_snapshots(fitness_history, save_dir, title_suffix="", num_snapshots=6):
    """
    Create snapshots of fitness distribution at different generations.
    
    Args:
        fitness_history: Shape (num_gens, population_size)
        save_dir: Directory to save plots
        title_suffix: Additional text for plot title
        num_snapshots: Number of generation snapshots to create
    """
    # Handle different possible shapes
    if fitness_history.ndim == 2:
        num_gens, pop_size = fitness_history.shape
    elif fitness_history.ndim == 3:
        # If 3D, take the first 2 dimensions
        num_gens, pop_size = fitness_history.shape[:2]
        fitness_history = fitness_history.reshape(num_gens, pop_size)
    else:
        raise ValueError(f"Unexpected fitness history shape: {fitness_history.shape}")
    
    # Select equally spaced generations
    snapshot_gens = np.linspace(0, num_gens-1, num_snapshots, dtype=int)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for i, gen in enumerate(snapshot_gens):
        ax = axes[i]
        
        # Get fitness distribution for this generation
        gen_fitness = fitness_history[gen, :]
        
        # Create histogram
        ax.hist(gen_fitness, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        ax.set_title(f'Generation {gen}\nMean: {np.mean(gen_fitness):.3f}, Std: {np.std(gen_fitness):.3f}', 
                    fontsize=12)
        ax.set_xlabel('Fitness Value', fontsize=10)
        ax.set_ylabel('Frequency', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Add vertical lines for statistics
        mean_fit = np.mean(gen_fitness)
        ax.axvline(mean_fit, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_fit:.3f}')
        ax.axvline(np.max(gen_fitness), color='green', linestyle='--', linewidth=2, label=f'Max: {np.max(gen_fitness):.3f}')
        ax.legend(fontsize=8)
    
    plt.suptitle(f'Fitness Distribution Snapshots{title_suffix}', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'fitness_snapshots{title_suffix}.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Fitness snapshots saved to {save_dir}/fitness_snapshots{title_suffix}.png")

def main():
    parser = argparse.ArgumentParser(description='Visualize fitness history as heatmap for all trials in a project')
    parser.add_argument('--project_dir', type=str, 
                       default="projects/reports/iclr_final/classic_control/diversity_analysis/mountaincar/256",
                       help='Path to project directory containing trial subdirectories')
    parser.add_argument('--title_suffix', type=str, default='',
                       help='Additional text for plot titles')
    
    args = parser.parse_args()
    
    # Validate project directory
    if not os.path.exists(args.project_dir):
        raise ValueError(f"Project directory does not exist: {args.project_dir}")
    
    print(f"Processing fitness visualizations for project: {args.project_dir}")
    
    # Process all trials in the project
    process_all_trials(args.project_dir, args.title_suffix)
    
    print("All fitness visualizations completed!")

if __name__ == "__main__":
    main()
