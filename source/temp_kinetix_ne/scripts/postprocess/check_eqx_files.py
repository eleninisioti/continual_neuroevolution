"""Script to check if .eqx checkpoint files are identical."""
import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "5"
sys.path.append(".")
sys.path.append("methods/evosax_wrapper/")

import yaml
import jax.numpy as jnp
import jax
import numpy as onp
import equinox as eqx
import evosax
from scripts.train.evosax.train_utils import EvosaxExperiment

def load_config(trial_dir):
    """Load config from yaml file."""
    # Try multiple possible locations
    possible_paths = [
        os.path.join(trial_dir, "config.yaml"),
        os.path.join(os.path.dirname(trial_dir), "config.yaml"),
        os.path.join(os.path.dirname(os.path.dirname(trial_dir)), "config.yaml"),
    ]
    
    config_path = None
    for path in possible_paths:
        if os.path.exists(path):
            config_path = path
            break
    
    if config_path is None:
        # If config doesn't exist, we can try to infer from the directory structure
        # or use eval_checkpoints.py's load_config
        raise FileNotFoundError(f"Config file not found. Tried: {possible_paths}")
    
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config

def get_param_size_from_file(file_path):
    """Try to determine parameter size from a checkpoint file."""
    # Try binary search approach - read file size first
    file_size = os.path.getsize(file_path)
    # Each float32 is 4 bytes, so approximate size
    approx_size = file_size // 4
    
    # Try sizes around the approximation
    sizes_to_try = [approx_size - 100, approx_size, approx_size + 100, approx_size * 2]
    # Also try common sizes
    sizes_to_try.extend([1000, 2000, 5000, 10000, 20000, 50000, 100000])
    sizes_to_try = sorted(set(sizes_to_try))
    
    for size in sizes_to_try:
        if size <= 0:
            continue
        try:
            data = eqx.tree_deserialise_leaves(file_path, jnp.zeros((size,)))
            actual_size = len(data) if isinstance(data, jnp.ndarray) else len(jnp.concatenate([x.flatten() for x in jax.tree_util.tree_leaves(data)]))
            if actual_size == size:
                return size
        except:
            continue
    
    # Last resort: try to read as raw and infer
    return approx_size

def compare_eqx_files(file1_path, file2_path, param_size):
    """Compare two .eqx files to see if they're identical."""
    try:
        # Load both files as flat arrays
        data1 = eqx.tree_deserialise_leaves(
            file1_path, 
            jnp.zeros((param_size,))
        )
        data2 = eqx.tree_deserialise_leaves(
            file2_path, 
            jnp.zeros((param_size,))
        )
        
        # Convert to numpy for comparison
        arr1 = onp.array(data1)
        arr2 = onp.array(data2)
        
        # Check if arrays are identical
        are_identical = onp.array_equal(arr1, arr2)
        max_diff = float(onp.max(onp.abs(arr1 - arr2))) if not are_identical else 0.0
        mean_diff = float(onp.mean(onp.abs(arr1 - arr2))) if not are_identical else 0.0
        
        return are_identical, max_diff, mean_diff, arr1.shape
    except Exception as e:
        print(f"Error comparing {file1_path} and {file2_path}: {e}")
        return None, None, None, None

def find_eqx_files(directory):
    """Find all .eqx files in best_model directory."""
    eqx_files = []
    # Try different possible paths
    possible_paths = [
        os.path.join(directory, "best_model"),
        os.path.join(directory, "data", "train", "best_model"),
        os.path.join(directory, "data", "best_model"),
    ]
    
    for best_model_dir in possible_paths:
        if os.path.exists(best_model_dir):
            for file in os.listdir(best_model_dir):
                if file.endswith(".eqx"):
                    eqx_files.append(os.path.join(best_model_dir, file))
            break
    
    return sorted(eqx_files)

def main(trial_dir):
    """Check all .eqx files in a trial directory."""
    eqx_files = find_eqx_files(trial_dir)
    
    if len(eqx_files) == 0:
        print(f"No .eqx files found in {trial_dir}/best_model")
        return
    
    print(f"Found {len(eqx_files)} .eqx files in {trial_dir}/best_model")
    
    # Determine parameter size from first file
    print("Determining parameter size from first checkpoint...")
    param_size = get_param_size_from_file(eqx_files[0])
    if param_size is None:
        print("ERROR: Could not determine parameter size. Trying to load with config...")
        # Fallback: try to use config
        try:
            config = load_config(trial_dir)
            config["exp_config"]["trial_dir"] = trial_dir
            exp = EvosaxExperiment(
                env_config=config["env_config"],
                model_config=config["model_config"],
                exp_config=config["exp_config"],
                optimizer_config=config["optimizer_config"]
            )
            exp.setup()
            exp.init_run()
            params, statics = eqx.partition(exp.model, eqx.is_array)
            params_shaper = evosax.ParameterReshaper(params)
            param_size = params_shaper.total_params
        except Exception as e:
            print(f"ERROR: Could not load config either: {e}")
            return
    else:
        print(f"Parameter size: {param_size}")
    
    print("\nComparing files...")
    print("=" * 80)
    
    identical_pairs = []
    different_pairs = []
    
    for i, file1 in enumerate(eqx_files):
        for file2 in eqx_files[i+1:]:
            result = compare_eqx_files(file1, file2, param_size)
            if result[0] is None:
                continue
                
            are_identical, max_diff, mean_diff, shape = result
            
            file1_name = os.path.basename(file1)
            file2_name = os.path.basename(file2)
            
            if are_identical:
                print(f"✗ {file1_name} == {file2_name} (IDENTICAL - PROBLEM!)")
                identical_pairs.append((file1_name, file2_name))
            else:
                print(f"✓ {file1_name} != {file2_name} (different)")
                print(f"  Max diff: {max_diff:.6e}, Mean diff: {mean_diff:.6e}, Shape: {shape}")
                different_pairs.append((file1_name, file2_name))
    
    print("\n" + "=" * 80)
    if identical_pairs:
        print(f"\n✗ PROBLEM: Found {len(identical_pairs)} identical file pairs:")
        for f1, f2 in identical_pairs:
            print(f"  - {f1} == {f2}")
    else:
        print(f"\n✓ All {len(eqx_files)} files are different (good!)")
        print(f"  Compared {len(different_pairs)} pairs, all were different")

if __name__ == "__main__":
    # Default trial_dir for easy debugging
    default_trial_dir = "projects/reports/iclr_rebuttal/kl_divergence/acrobot/trial_0"
    
    if len(sys.argv) < 2:
        print(f"Usage: python check_eqx_files.py <trial_dir>")
        print(f"Example: python check_eqx_files.py {default_trial_dir}")
        print(f"\nUsing default trial_dir: {default_trial_dir}")
        trial_dir = default_trial_dir
    else:
        trial_dir = sys.argv[1]
    
    main(trial_dir)

