"""Script to compare consecutive checkpoints using KL-divergence between policies."""
import sys
import os
sys.path.append(".")
sys.path.append("methods/evosax_wrapper/")

import argparse
import pickle
import yaml
import jax
import jax.numpy as jnp
import numpy as np
import equinox as eqx
from functools import partial
import evosax
from typing import Dict, List, Tuple

from scripts.train.evosax.train_utils import EvosaxExperiment
from methods.evosax_wrapper.direct_encodings.model import make_model


def load_config(project_dir):
    """Load config from yaml file."""
    config_path = os.path.join(project_dir, "config.yaml")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, "r") as f:
        # Use UnsafeLoader to handle Python objects (e.g., gymnax EnvParams)
        config = yaml.load(f, Loader=yaml.UnsafeLoader)
    
    return config


def setup_experiment(config, trial_dir):
    """Set up experiment with environment and model."""
    # Create experiment instance
    exp = EvosaxExperiment(
        env_config=config["env_config"],
        model_config=config["model_config"],
        exp_config=config["exp_config"],
        optimizer_config=config["optimizer_config"]
    )
    
    # Set trial directory
    exp.config["exp_config"]["trial_dir"] = trial_dir
    
    # Set up trial keys first (needed for model initialization)
    exp.setup_trial_keys()
    
    # Set up environment (this also creates the task)
    exp.setup_env()
    
    # Initialize model (needs environment to be set up)
    exp.init_model()
    
    return exp


def load_checkpoint(checkpoint_path, exp):
    """Load checkpoint from .eqx file."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
    
    # Handle .eqx files (Equinox serialized checkpoints)
    if checkpoint_path.endswith(".eqx"):
        # Partition model to get template structure
        params, statics = eqx.partition(exp.model, eqx.is_array)
        params_shaper = evosax.ParameterReshaper(params)
        # Load checkpoint using template
        best_member = eqx.tree_deserialise_leaves(
            checkpoint_path, 
            jnp.zeros((params_shaper.total_params,))
        )
        # Extract generation number from filename (e.g., ckpt_100.eqx -> 100)
        filename = os.path.basename(checkpoint_path)
        if filename.startswith("ckpt_") and filename.endswith(".eqx"):
            try:
                generation = int(filename.replace("ckpt_", "").replace(".eqx", ""))
            except ValueError:
                generation = 0
        else:
            generation = 0
        return generation, best_member
    else:
        raise ValueError(f"Unsupported checkpoint format: {checkpoint_path}")


def create_policy_from_checkpoint(exp, best_member):
    """Create policy from checkpoint weights."""
    # Partition model into params and statics
    params, statics = eqx.partition(exp.model, eqx.is_array)
    
    # Create parameter reshaper
    params_shaper = evosax.ParameterReshaper(params)
    
    # Reshape best_member to match model structure
    policy_params = params_shaper.reshape_single(best_member)
    
    # Combine with statics to create full policy
    policy = eqx.combine(policy_params, statics)
    
    return policy, params_shaper


def get_action_distribution(policy, obs, policy_state, obs_size, action_size, key):
    """Get action probability distribution from policy."""
    # Get action logits from policy
    action_output = policy(obs, policy_state, key, obs_size=obs_size, action_size=action_size)
    
    # Handle different policy output formats
    if isinstance(action_output, tuple):
        action_logits, new_policy_state = action_output
    else:
        action_logits = action_output
        new_policy_state = policy_state
    
    # Convert logits to probabilities using softmax
    # Add small epsilon to avoid numerical issues
    epsilon = 1e-8
    action_probs = jax.nn.softmax(action_logits)
    action_probs = jnp.clip(action_probs, epsilon, 1.0 - epsilon)
    # Renormalize to ensure valid probability distribution
    action_probs = action_probs / jnp.sum(action_probs)
    
    return action_probs, new_policy_state


def logits_to_probs(logits, epsilon=1e-8):
    """
    Convert logits to probabilities using softmax.
    
    Args:
        logits: Array of logits
        epsilon: Small value for numerical stability
    
    Returns:
        Probability distribution (normalized to sum to 1)
    """
    # Convert to numpy if needed
    logits = np.array(logits)
    
    # Subtract max for numerical stability
    logits = logits - np.max(logits)
    
    # Compute softmax
    exp_logits = np.exp(logits)
    probs = exp_logits / (np.sum(exp_logits) + epsilon)
    
    # Clip to avoid numerical issues
    probs = np.clip(probs, epsilon, 1.0 - epsilon)
    
    # Renormalize
    probs = probs / np.sum(probs)
    
    return probs


def calculate_kl_from_logits(
    trajectories1: List[Dict],
    trajectories2: List[Dict],
    states: List[np.ndarray],
    action_size: int
):
    """
    Calculate KL-divergence between two policies using logits from trajectories.
    
    For each state, we:
    1. Find the logits for that state in both trajectories
    2. Convert logits to probabilities using softmax
    3. Calculate KL(P1 || P2) = Σ P1(a|s) * log(P1(a|s) / P2(a|s))
    4. Average across all states
    
    Args:
        trajectories1: List of trajectory dictionaries for policy 1, each containing "observations" and "logits"
        trajectories2: List of trajectory dictionaries for policy 2, each containing "observations" and "logits"
        states: List of states (numpy arrays) to evaluate on
        action_size: Number of possible actions
    
    Returns:
        Tuple of (mean_kl, std_kl, total_kl, kl_divergences)
        - mean_kl: Mean KL-divergence per state (standard metric)
        - std_kl: Standard deviation of KL-divergences
        - total_kl: Sum of KL-divergences across all states
        - kl_divergences: List of per-state KL-divergences
    """
    kl_divergences = []
    
    # Build lookup dictionaries: state_tuple -> logits
    logits_dict1 = {}
    logits_dict2 = {}
    
    # Extract logits from trajectories for policy 1
    for traj in trajectories1:
        if "observations" in traj and "logits" in traj:
            for obs, logit in zip(traj["observations"], traj["logits"]):
                state_tuple = tuple(obs)
                if state_tuple not in logits_dict1:
                    logits_dict1[state_tuple] = logit
    
    # Extract logits from trajectories for policy 2
    for traj in trajectories2:
        if "observations" in traj and "logits" in traj:
            for obs, logit in zip(traj["observations"], traj["logits"]):
                state_tuple = tuple(obs)
                if state_tuple not in logits_dict2:
                    logits_dict2[state_tuple] = logit
    
    for state in states:
        # Convert state to tuple for dictionary lookup
        state_tuple = tuple(state)
        
        # Get logits for this state
        if state_tuple not in logits_dict1:
            print(f"Warning: Logits not found for state in policy 1, skipping...")
            continue
        if state_tuple not in logits_dict2:
            print(f"Warning: Logits not found for state in policy 2, skipping...")
            continue
        
        logits1 = logits_dict1[state_tuple]
        logits2 = logits_dict2[state_tuple]
        
        # Convert logits to probabilities using softmax
        probs1 = logits_to_probs(logits1)
        probs2 = logits_to_probs(logits2)
        
        # Calculate KL-divergence for this state
        kl = kl_divergence(probs1, probs2)
        kl_divergences.append(kl)
        
        # Debug: print first few states to check if policies agree
        if len(kl_divergences) <= 3:
            action1 = np.argmax(probs1)
            action2 = np.argmax(probs2)
            print(f"  Debug state {len(kl_divergences)-1}: Policy1 action={action1} (prob={probs1[action1]:.3f}), Policy2 action={action2} (prob={probs2[action2]:.3f}), KL={kl:.6f}")
    
    if len(kl_divergences) == 0:
        raise ValueError("No valid state-logit pairs found for KL calculation")
    
    mean_kl = np.mean(kl_divergences)
    std_kl = np.std(kl_divergences)
    total_kl = np.sum(kl_divergences)
    
    return mean_kl, std_kl, total_kl, kl_divergences


def kl_divergence(p, q, epsilon=1e-8):
    """
    Calculate KL-divergence: KL(P || Q) = Σ P(x) * log(P(x) / Q(x))
    
    Args:
        p: Probability distribution P (numpy array)
        q: Probability distribution Q (numpy array)
        epsilon: Small value to avoid log(0)
    
    Returns:
        KL-divergence value
    """
    # Ensure probabilities sum to 1
    p = np.clip(p, epsilon, 1.0 - epsilon)
    q = np.clip(q, epsilon, 1.0 - epsilon)
    p = p / np.sum(p)
    q = q / np.sum(q)
    
    # Calculate KL(P || Q) = Σ P(x) * log(P(x) / Q(x))
    kl = np.sum(p * np.log(p / q))
    
    return kl


def load_trajectories(trajectories_path):
    """Load trajectories with logits from pickle file."""
    if not os.path.exists(trajectories_path):
        raise FileNotFoundError(f"Trajectories file not found: {trajectories_path}")
    
    with open(trajectories_path, "rb") as f:
        trajectories_data = pickle.load(f)
    
    return trajectories_data


def load_collected_states(states_path):
    """Load collected states from pickle file."""
    if not os.path.exists(states_path):
        raise FileNotFoundError(f"Collected states file not found: {states_path}")
    
    with open(states_path, "rb") as f:
        states_data = pickle.load(f)
    
    return states_data


def compare_consecutive_checkpoints(
    project_dir: str,
    trial: int,
    min_gen: int,
    max_gen: int
):
    """
    Compare consecutive checkpoints using KL-divergence.
    
    Uses pre-computed action probabilities from eval_checkpoints.py.
    For each state, computes KL divergence between policies (handling deterministic policies),
    then averages across all states.
    
    Args:
        project_dir: Project directory path
        trial: Trial number
        min_gen: Minimum generation
        max_gen: Maximum generation
    """
    # Load config
    config = load_config(project_dir)
    
    # Set up trial directory
    trial_dir = os.path.join(project_dir, f"trial_{trial}")
    if not os.path.exists(trial_dir):
        raise FileNotFoundError(f"Trial directory not found: {trial_dir}")
    
    config["exp_config"]["trial_dir"] = trial_dir
    config["exp_config"]["trial_seed"] = trial
    
    # Set up experiment
    print("Setting up experiment...")
    exp = setup_experiment(config, trial_dir)
    
    # Find checkpoint directories
    checkpoint_dirs = []
    for root, dirs, files in os.walk(project_dir):
        if "best_model" in dirs:
            best_model_dir = os.path.join(root, "best_model")
            if os.path.exists(best_model_dir):
                checkpoint_dirs.append(best_model_dir)
                break  # Only need one
    
    if not checkpoint_dirs:
        raise FileNotFoundError(f"No checkpoint directories found under {project_dir}")
    
    checkpoint_dir = checkpoint_dirs[0]
    
    # Find all checkpoints in range
    checkpoints = []
    for filename in os.listdir(checkpoint_dir):
        if filename.startswith("ckpt_") and filename.endswith(".eqx"):
            try:
                gen = int(filename.replace("ckpt_", "").replace(".eqx", ""))
                if min_gen <= gen <= max_gen:
                    checkpoints.append((gen, os.path.join(checkpoint_dir, filename)))
            except ValueError:
                continue
    
    checkpoints = sorted(checkpoints, key=lambda x: x[0])
    
    if len(checkpoints) < 2:
        raise ValueError(f"Need at least 2 checkpoints for comparison, found {len(checkpoints)}")
    
    print(f"Found {len(checkpoints)} checkpoints to compare")
    
    # Load collected states
    eval_dir_base = os.path.join(trial_dir, "data/eval/checkpoint_eval")
    states_file = os.path.join(eval_dir_base, "collected_states.pkl")
    
    print(f"\nLoading collected states from: {states_file}")
    states_data = load_collected_states(states_file)
    all_states = states_data["all_states"]
    print(f"Loaded {len(all_states)} unique states")
    
    # Compare consecutive checkpoints
    results = {}
    action_size = exp.config["env_config"]["action_size"]
    
    for i in range(len(checkpoints) - 1):
        gen1, ckpt_path1 = checkpoints[i]
        gen2, ckpt_path2 = checkpoints[i + 1]
        
        print(f"\n{'='*60}")
        print(f"Comparing generation {gen1} vs {gen2}")
        print(f"{'='*60}")
        
        # Load trajectories with logits for both checkpoints
        trajectories_file1 = os.path.join(eval_dir_base, f"gen_{gen1}_trajectories.pkl")
        trajectories_file2 = os.path.join(eval_dir_base, f"gen_{gen2}_trajectories.pkl")
        
        if not os.path.exists(trajectories_file1):
            print(f"Warning: Trajectories file not found for generation {gen1}: {trajectories_file1}, skipping...")
            continue
        if not os.path.exists(trajectories_file2):
            print(f"Warning: Trajectories file not found for generation {gen2}: {trajectories_file2}, skipping...")
            continue
        
        print(f"Loading trajectories for gen {gen1} from: {trajectories_file1}")
        trajectories1 = load_trajectories(trajectories_file1)
        print(f"Loading trajectories for gen {gen2} from: {trajectories_file2}")
        trajectories2 = load_trajectories(trajectories_file2)
        
        print(f"Trajectories for gen {gen1}: {len(trajectories1)} episodes")
        print(f"Trajectories for gen {gen2}: {len(trajectories2)} episodes")
        
        # Calculate KL-divergence using logits from trajectories
        print("Calculating KL-divergence from logits...")
        mean_kl, std_kl, total_kl, kl_values = calculate_kl_from_logits(
            trajectories1,
            trajectories2,
            all_states,
            action_size
        )
        
        results[f"{gen1}_vs_{gen2}"] = {
            "gen1": gen1,
            "gen2": gen2,
            "mean_kl": float(mean_kl),
            "std_kl": float(std_kl),
            "total_kl": float(total_kl),
            "min_kl": float(np.min(kl_values)),
            "max_kl": float(np.max(kl_values)),
            "num_states": len(kl_values)
        }
        
        print(f"KL-divergence (mean): {mean_kl:.6f} ± {std_kl:.6f}")
        print(f"KL-divergence (total): {total_kl:.6f}")
        print(f"Range: [{np.min(kl_values):.6f}, {np.max(kl_values):.6f}]")
    
    # Save results
    results_path = os.path.join(trial_dir, "data/eval/kl_divergence_comparison.yaml")
    with open(results_path, "w") as f:
        yaml.dump(results, f)
    
    print(f"\n{'='*60}")
    print(f"Results saved to: {results_path}")
    print(f"{'='*60}")
    
    # Print summary
    print("\nSummary:")
    print("Generation Pair | Mean KL-divergence | Std | Num States")
    print("-" * 70)
    for key, data in results.items():
        print(f"{data['gen1']} vs {data['gen2']:4d} | {data['mean_kl']:>18.6f} | {data['std_kl']:>10.6f} | {data['num_states']:>10}")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Compare consecutive checkpoints using KL-divergence"
    )
    parser.add_argument(
        "--project_dir",
        type=str,
        default="projects/reports/iclr_rebuttal/kl_divergence/ga/continual/acrobot",
        help="Path to project directory containing config.yaml (default: projects/reports/iclr_rebuttal/kl_divergence/ga/continual/acrobot)"
    )
    parser.add_argument(
        "--trial",
        type=int,
        default=0,
        help="Trial number (default: 0)"
    )
    parser.add_argument(
        "--min_gen",
        type=int,
        default=100,
        help="Minimum generation to compare (default: 100)"
    )
    parser.add_argument(
        "--max_gen",
        type=int,
        default=2000,
        help="Maximum generation to compare (default: 2000)"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=None,
        help="Number of observations to sample for KL calculation (deprecated: now uses all collected states)"
    )
    
    args = parser.parse_args()
    
    compare_consecutive_checkpoints(
        args.project_dir,
        args.trial,
        args.min_gen,
        args.max_gen
    )


if __name__ == "__main__":
    main()

