"""Test script to verify checkpoints and trajectories are different."""
import sys
import os
sys.path.append(".")
sys.path.append("methods/evosax_wrapper/")

import pickle
import yaml
import jax
import jax.numpy as jnp
import numpy as np
import equinox as eqx
import evosax

from scripts.train.evosax.train_utils import EvosaxExperiment
from scripts.postprocess.compare_checkpoints_kl import (
    load_config, setup_experiment, load_checkpoint, create_policy_from_checkpoint
)


def compare_checkpoints(project_dir, trial, gen1, gen2):
    """Compare two checkpoints to see if they're different."""
    print(f"\n{'='*60}")
    print(f"Comparing checkpoints: gen {gen1} vs gen {gen2}")
    print(f"{'='*60}")
    
    # Load config
    config = load_config(project_dir)
    
    # Set up trial directory
    trial_dir = os.path.join(project_dir, f"trial_{trial}")
    config["exp_config"]["trial_dir"] = trial_dir
    config["exp_config"]["trial_seed"] = trial
    
    # Set up experiment
    print("Setting up experiment...")
    exp = setup_experiment(config, trial_dir)
    
    # Find checkpoint directory
    checkpoint_dir = os.path.join(trial_dir, "data/train/best_model")
    
    # Load checkpoints
    ckpt_path1 = os.path.join(checkpoint_dir, f"ckpt_{gen1}.eqx")
    ckpt_path2 = os.path.join(checkpoint_dir, f"ckpt_{gen2}.eqx")
    
    print(f"Loading checkpoint {gen1}...")
    _, best_member1 = load_checkpoint(ckpt_path1, exp)
    
    print(f"Loading checkpoint {gen2}...")
    _, best_member2 = load_checkpoint(ckpt_path2, exp)
    
    # Compare parameters
    print("\nComparing parameters...")
    diff = jnp.abs(best_member1 - best_member2)
    max_diff = float(jnp.max(diff))
    mean_diff = float(jnp.mean(diff))
    min_diff = float(jnp.min(diff))
    l2_diff = float(jnp.linalg.norm(best_member1 - best_member2))
    
    print(f"Max difference: {max_diff:.10f}")
    print(f"Mean difference: {mean_diff:.10f}")
    print(f"Min difference: {min_diff:.10f}")
    print(f"L2 norm of difference: {l2_diff:.10f}")
    print(f"Are they identical? {jnp.allclose(best_member1, best_member2, atol=1e-10)}")
    
    # Also check if they're exactly equal
    are_equal = jnp.array_equal(best_member1, best_member2)
    print(f"Are they exactly equal? {are_equal}")
    
    return {
        "max_diff": max_diff,
        "mean_diff": mean_diff,
        "min_diff": min_diff,
        "l2_diff": l2_diff,
        "are_identical": bool(jnp.allclose(best_member1, best_member2, atol=1e-10)),
        "are_equal": bool(are_equal)
    }


def compare_trajectories(project_dir, trial, gen1, gen2):
    """Compare trajectories from two consecutive generations."""
    print(f"\n{'='*60}")
    print(f"Comparing trajectories: gen {gen1} vs gen {gen2}")
    print(f"{'='*60}")
    
    trial_dir = os.path.join(project_dir, f"trial_{trial}")
    eval_dir_base = os.path.join(trial_dir, "data/eval/checkpoint_eval")
    
    traj_path1 = os.path.join(eval_dir_base, f"gen_{gen1}", "trajectories.pkl")
    traj_path2 = os.path.join(eval_dir_base, f"gen_{gen2}", "trajectories.pkl")
    
    # Check if files exist
    if not os.path.exists(traj_path1):
        print(f"Warning: Trajectories not found for gen {gen1}: {traj_path1}")
        return None
    
    if not os.path.exists(traj_path2):
        print(f"Warning: Trajectories not found for gen {gen2}: {traj_path2}")
        return None
    
    # Load trajectories
    print(f"Loading trajectories from gen {gen1}...")
    with open(traj_path1, "rb") as f:
        traj_data1 = pickle.load(f)
    
    print(f"Loading trajectories from gen {gen2}...")
    with open(traj_path2, "rb") as f:
        traj_data2 = pickle.load(f)
    
    # Compare structure
    print("\nComparing trajectory structures...")
    print(f"Gen {gen1} keys: {list(traj_data1.keys())}")
    print(f"Gen {gen2} keys: {list(traj_data2.keys())}")
    
    # Compare observations if available
    all_obs1 = []
    all_obs2 = []
    
    for task_key in traj_data1.keys():
        if task_key in traj_data2:
            if "trajectories" in traj_data1[task_key]:
                for trial_data in traj_data1[task_key]["trajectories"]:
                    if "observations" in trial_data:
                        all_obs1.extend(trial_data["observations"])
            
            if "trajectories" in traj_data2[task_key]:
                for trial_data in traj_data2[task_key]["trajectories"]:
                    if "observations" in trial_data:
                        all_obs2.extend(trial_data["observations"])
    
    print(f"\nGen {gen1}: {len(all_obs1)} observations")
    print(f"Gen {gen2}: {len(all_obs2)} observations")
    
    if len(all_obs1) > 0 and len(all_obs2) > 0:
        # Compare first few observations
        min_len = min(len(all_obs1), len(all_obs2), 10)
        print(f"\nComparing first {min_len} observations...")
        
        differences = []
        for i in range(min_len):
            obs1 = np.array(all_obs1[i])
            obs2 = np.array(all_obs2[i])
            diff = np.abs(obs1 - obs2)
            max_diff = float(np.max(diff))
            differences.append(max_diff)
            if i < 5:  # Print first 5
                print(f"  Obs {i}: max_diff = {max_diff:.10f}")
        
        mean_diff = np.mean(differences)
        print(f"\nMean difference (first {min_len} obs): {mean_diff:.10f}")
        print(f"Are observations identical? {np.allclose(all_obs1[0], all_obs2[0], atol=1e-10) if len(all_obs1) > 0 and len(all_obs2) > 0 else 'N/A'}")
    
    # Compare actions if available
    all_actions1 = []
    all_actions2 = []
    
    for task_key in traj_data1.keys():
        if task_key in traj_data2:
            if "trajectories" in traj_data1[task_key]:
                for trial_data in traj_data1[task_key]["trajectories"]:
                    if "actions" in trial_data:
                        all_actions1.extend(trial_data["actions"])
            
            if "trajectories" in traj_data2[task_key]:
                for trial_data in traj_data2[task_key]["trajectories"]:
                    if "actions" in trial_data:
                        all_actions2.extend(trial_data["actions"])
    
    if len(all_actions1) > 0 and len(all_actions2) > 0:
        print(f"\nGen {gen1}: {len(all_actions1)} actions")
        print(f"Gen {gen2}: {len(all_actions2)} actions")
        
        min_len = min(len(all_actions1), len(all_actions2), 10)
        print(f"\nComparing first {min_len} actions...")
        
        action_diffs = []
        for i in range(min_len):
            if all_actions1[i] != all_actions2[i]:
                action_diffs.append(i)
        
        if action_diffs:
            print(f"  Found {len(action_diffs)} different actions in first {min_len}")
            print(f"  First few different indices: {action_diffs[:5]}")
        else:
            print(f"  All first {min_len} actions are identical")
    
    return {
        "num_obs1": len(all_obs1),
        "num_obs2": len(all_obs2),
        "num_actions1": len(all_actions1),
        "num_actions2": len(all_actions2)
    }


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Test checkpoints and trajectories")
    parser.add_argument("--project_dir", type=str, required=True)
    parser.add_argument("--trial", type=int, default=0)
    parser.add_argument("--gen1", type=int, default=199)
    parser.add_argument("--gen2", type=int, default=399)
    
    args = parser.parse_args()
    
    # Test checkpoints
    ckpt_results = compare_checkpoints(args.project_dir, args.trial, args.gen1, args.gen2)
    
    # Test trajectories
    traj_results = compare_trajectories(args.project_dir, args.trial, args.gen1, args.gen2)
    
    print(f"\n{'='*60}")
    print("Summary:")
    print(f"{'='*60}")
    print(f"Checkpoints are identical: {ckpt_results['are_identical']}")
    print(f"Checkpoints are exactly equal: {ckpt_results['are_equal']}")
    print(f"Max parameter difference: {ckpt_results['max_diff']:.10f}")
    if traj_results:
        print(f"Trajectory observations: gen {args.gen1} has {traj_results['num_obs1']}, gen {args.gen2} has {traj_results['num_obs2']}")


if __name__ == "__main__":
    main()

