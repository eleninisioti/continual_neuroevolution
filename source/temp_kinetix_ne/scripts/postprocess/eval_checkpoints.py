"""Script to load evosax checkpoints and collect states, then evaluate policies on collected states for KL divergence."""
import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "5"
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
from PIL import Image
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Set
from collections import defaultdict

from scripts.train.evosax.train_utils import EvosaxExperiment
from scripts.train.base.task import Task
from methods.evosax_wrapper.direct_encodings.model import make_model
from methods.evosax_wrapper.base.tasks.rl import State, GymnaxState
import jax.random as jr


def load_config(project_dir):
    """Load config from yaml file."""
    config_path = os.path.join(project_dir, "config.yaml")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, "r") as f:
        # Use UnsafeLoader to handle Python objects (e.g., gymnax EnvParams)
        config = yaml.load(f, Loader=yaml.UnsafeLoader)
    
    return config


def load_checkpoint(checkpoint_path, exp=None):
    """Load checkpoint from pickle file or .eqx file."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
    
    # Handle .eqx files (Equinox serialized checkpoints)
    if checkpoint_path.endswith(".eqx"):
        if exp is None:
            raise ValueError("exp must be provided to load .eqx files")
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
    
    # Handle .pkl files
    with open(checkpoint_path, "rb") as f:
        data = pickle.load(f)
    
    # Checkpoint format: (generation, best_member)
    if isinstance(data, tuple) and len(data) == 2:
        generation, best_member = data
        return generation, best_member
    else:
        # Try to handle different formats
        if isinstance(data, dict):
            return data.get("generation", 0), data.get("params", data)
        return 0, data


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
    exp.config["exp_config"]["trial_seed"] = int(trial_dir.split("trial_")[-1]) if "trial_" in trial_dir else 0
    
    # Set up trial keys first (needed for model initialization)
    exp.setup_trial_keys()
    
    # Set up environment (this also creates the task)
    exp.setup_env()
    
    # Initialize model (needs environment to be set up)
    exp.init_model()
    
    return exp


def create_policy_from_checkpoint(exp, best_member):
    """Create policy from checkpoint weights."""
    # Partition model into params and statics
    params, statics = eqx.partition(exp.model, eqx.is_array)
    exp.statics = statics
    
    # Create parameter reshaper
    params_shaper = evosax.ParameterReshaper(params)
    
    # Reshape best_member to match model structure
    policy_params = params_shaper.reshape_single(best_member)
    
    # Combine with statics to create full policy
    policy = eqx.combine(policy_params, statics)
    
    return policy, params_shaper


def find_checkpoint_dirs(project_dir):
    """Find all best_model directories under the project directory."""
    checkpoint_dirs = []
    for root, dirs, files in os.walk(project_dir):
        if "best_model" in dirs:
            best_model_dir = os.path.join(root, "best_model")
            if os.path.exists(best_model_dir):
                checkpoint_dirs.append(best_model_dir)
    return checkpoint_dirs


def find_checkpoints(checkpoint_dir):
    """Find all checkpoint files in best_model directory."""
    checkpoints = []
    if not os.path.exists(checkpoint_dir):
        return checkpoints
    
    for file in os.listdir(checkpoint_dir):
        if file.endswith(".eqx"):
            # Extract generation number from filename
            try:
                gen = int(file.replace("ckpt_", "").replace(".eqx", ""))
                checkpoints.append((gen, os.path.join(checkpoint_dir, file)))
            except ValueError:
                continue
    
    return sorted(checkpoints, key=lambda x: x[0])


def collect_states_from_evaluation(exp, policy, num_episodes, task_id=None, for_eval=None):
    """
    Run evaluation episodes and collect all states visited, following the rollout pattern from rl.py.
    Returns a list of states (observations) as numpy arrays, and trajectories with logits.
    """
    
    states_collected = []
    trajectories = []  # Store full trajectories with logits
    
    # Determine tasks to evaluate
    if task_id is not None:
        tasks = [task_id]
    else:
        tasks = range(exp.config["env_config"]["num_tasks"])
    
    # Run evaluation and collect states
    env = exp.task.env
    obs_size = exp.config["env_config"]["observation_size"]
    action_size = exp.config["env_config"]["action_size"]
    max_steps = exp.config["env_config"]["episode_length"]
    
    for task in tasks:
        for eval_trial in range(num_episodes):
            # Initialize keys
            rng = jax.random.PRNGKey(seed=eval_trial)
            init_env_key, init_policy_key, rollout_key = jr.split(rng, 3)
            
            # Initialize policy state
            policy_state, _ = policy.initialize(init_policy_key)
            
            # Reset environment (gymnax only)
            obs, gymnax_state = env.reset(init_env_key, jnp.array([task]))
            
            # Collect initial state
            obs_np = np.array(obs).flatten()
            states_collected.append(obs_np)
            
            # Create initial state following rl.py pattern (line 298)
            # Note: initial state doesn't have logit, it's added in env_step
            dummy_logit = jnp.zeros(action_size)
            env_state_wrapper = GymnaxState(
                env_state=gymnax_state,
                obs=obs,
                reward=0.0,
                done=False,
                logit=dummy_logit  # Dummy logit for initial state
            )
            init_state = State(env_state=env_state_wrapper, policy_state=policy_state)
            
            # Define env_step function following rl.py pattern (lines 319-338)
            def env_step(carry, x):
                state, key = carry
                key, _key = jr.split(key)
                
                # Get logits from policy (following rl.py line 322)
                logit, policy_state = policy(
                    state.env_state.obs,
                    state.policy_state,
                    _key,
                    obs_size=obs_size,
                    action_size=action_size
                )
                action = jnp.argmax(logit)
                
                # Step environment (following rl.py line 326)
                obs, gymnax_state, reward, done, _ = env.step(
                    key,
                    state.env_state.env_state,
                    action,
                    exp.config["env_config"]["gymnax_env_params"]
                )
                
                # Create new state with logit (following rl.py line 336)
                new_env_state = GymnaxState(
                    env_state=gymnax_state,
                    obs=obs,
                    reward=reward,
                    done=done,
                    logit=logit
                )
                new_state = State(env_state=new_env_state, policy_state=policy_state)
                
                # Return following rl.py pattern (line 338) - return (state, action)
                return [new_state, key], (state, action)
            
            # Run rollout using jax.lax.scan (following rl.py line 340)
            [final_state, _], (states, actions) = jax.lax.scan(
                env_step,
                [init_state, rollout_key],
                None,
                max_steps
            )
            
            # Collect states and build trajectory
            trajectory = {
                "observations": [],
                "actions": [],
                "logits": [],
                "rewards": [],
                "dones": []
            }
            
            # Process initial state (no logit for initial state in rl.py pattern)
            trajectory["observations"].append(np.array(init_state.env_state.obs).flatten())
            trajectory["logits"].append(np.array(init_state.env_state.logit))  # Dummy logit
            
            # Process scanned states (following rl.py line 341 pattern)
            # After scan, states is a tree structure where each field is an array
            num_steps = len(states.env_state.obs)
            for i in range(num_steps):
                obs = np.array(states.env_state.obs[i]).flatten()
                logit = np.array(states.env_state.logit[i])  # Logit is stored in GymnaxState
                reward = float(states.env_state.reward[i])
                done = bool(states.env_state.done[i])
                
                states_collected.append(obs)
                trajectory["observations"].append(obs)
                trajectory["actions"].append(int(actions[i]) if i < len(actions) else 0)
                trajectory["logits"].append(logit)
                trajectory["rewards"].append(reward)
                trajectory["dones"].append(done)
                
                # Stop if done
                if done:
                    break
            
            trajectories.append(trajectory)
    
    return states_collected, trajectories


def get_deterministic_action_probs(policy, obs, policy_state, obs_size, action_size, key):
    """
    Get deterministic action probabilities (0 or 1) for a given state.
    Returns a numpy array of probabilities where the selected action has prob 1, others have 0.
    Only handles gymnax environments.
    """
    # Get action logits from policy (gymnax uses positional arguments)
    action_output = policy(obs, policy_state, key, obs_size=obs_size, action_size=action_size)
    
    # Handle different policy output formats
    if isinstance(action_output, tuple):
        action_logits, new_policy_state = action_output
    else:
        action_logits = action_output
        new_policy_state = policy_state
    
    # Get deterministic action (argmax)
    action = jnp.argmax(action_logits)
    
    # Create probability distribution: 1 for selected action, 0 for others
    action_probs = jnp.zeros_like(action_logits)
    action_probs = action_probs.at[action].set(1.0)
    
    return np.array(action_probs), new_policy_state


def evaluate_policy_on_states(exp, policy, states, for_eval=None):
    """
    Evaluate a policy on a set of states and return action probabilities.
    Returns a dictionary mapping (state_tuple, action) to probability (0 or 1).
    For deterministic policies, we reset policy state for each state to ensure consistency.
    Only handles gymnax environments.
    """
    # Initialize policy state (will be reset for each state)
    init_policy_state, _ = policy.initialize(jax.random.PRNGKey(0))
    
    obs_size = exp.config["env_config"]["observation_size"]
    action_size = exp.config["env_config"]["action_size"]
    key = jax.random.PRNGKey(42)
    
    action_probs_dict = {}
    
    for state in states:
        # Reset policy state for each state to ensure deterministic evaluation
        policy_state = init_policy_state
        
        # Convert state to JAX array
        state_jax = jnp.array(state)
        
        # Get deterministic action probabilities
        key, subkey = jax.random.split(key)
        probs, _ = get_deterministic_action_probs(
            policy, state_jax, policy_state, obs_size, action_size, subkey
        )
        
        # Store probabilities for each action
        # Use state as tuple for hashing
        state_tuple = tuple(state)
        for action_idx in range(action_size):
            action_probs_dict[(state_tuple, action_idx)] = float(probs[action_idx])
    
    return action_probs_dict


def main():
    parser = argparse.ArgumentParser(
        description="Collect states from checkpoints and evaluate policies on collected states for KL divergence"
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
        "--task_id",
        type=int,
        default=None,
        help="Specific task ID to evaluate (default: evaluate all tasks)"
    )
    parser.add_argument(
        "--num_episodes",
        type=int,
        default=1,
        help="Number of evaluation episodes for state collection (default: 1)"
    )
    parser.add_argument(
        "--min_gen",
        type=int,
        default=100,
        help="Minimum generation to evaluate (default: 100)"
    )
    parser.add_argument(
        "--max_gen",
        type=int,
        default=2000,
        help="Maximum generation to evaluate (default: 2000)"
    )
    parser.add_argument(
        "--skip_state_collection",
        action="store_true",
        help="Skip state collection phase and use existing states file"
    )
    parser.add_argument(
        "--force_rerun",
        action="store_true",
        help="Force rerun even if data already exists (will overwrite existing files)"
    )
    
    args = parser.parse_args()
    
    # Load config
    print(f"Loading config from: {args.project_dir}")
    config = load_config(args.project_dir)
    
    # Set up trial directory
    trial_dir = os.path.join(args.project_dir, f"trial_{args.trial}")
    if not os.path.exists(trial_dir):
        raise FileNotFoundError(f"Trial directory not found: {trial_dir}")
    
    config["exp_config"]["trial_dir"] = trial_dir
    config["exp_config"]["trial_seed"] = args.trial
    
    # Set up experiment
    print("Setting up experiment...")
    exp = setup_experiment(config, trial_dir)
    
    # Override num_eval_trials if specified
    if args.num_episodes is not None:
        exp.task.num_eval_trials = args.num_episodes
        print(f"Running {args.num_episodes} evaluation episodes per checkpoint")
    
    # Get for_eval if needed (for kinetix)
    for_eval = getattr(exp, "for_eval", None)
    
    # Find all best_model directories
    checkpoint_dirs = find_checkpoint_dirs(args.project_dir)
    if not checkpoint_dirs:
        raise FileNotFoundError(f"No best_model directories found under {args.project_dir}")
    
    print(f"Found {len(checkpoint_dirs)} best_model directory(ies)")
    
    # Find all checkpoints
    all_checkpoints = []
    for ckpt_dir in checkpoint_dirs:
        checkpoints = find_checkpoints(ckpt_dir)
        all_checkpoints.extend(checkpoints)
    
    # Filter by generation range
    filtered_checkpoints = [
        (gen, path) for gen, path in all_checkpoints
        if args.min_gen <= gen <= args.max_gen
    ]
    
    if not filtered_checkpoints:
        print(f"No checkpoints found in generation range [{args.min_gen}, {args.max_gen}]")
        return
    
    print(f"Found {len(filtered_checkpoints)} checkpoints to evaluate")
    print(f"Generation range: {filtered_checkpoints[0][0]} to {filtered_checkpoints[-1][0]}")
    
    # Create evaluation directory structure
    eval_dir_base = os.path.join(trial_dir, "data/eval/checkpoint_eval")
    os.makedirs(eval_dir_base, exist_ok=True)
    
    states_file = os.path.join(eval_dir_base, "collected_states.pkl")
    
    # ============================================================================
    # PHASE 1: Collect all states from all checkpoints and trials
    # ============================================================================
    if not args.skip_state_collection:
        print(f"\n{'='*60}")
        print("PHASE 1: Collecting states from all checkpoints")
        print(f"{'='*60}")
        
        all_states = []
        states_by_checkpoint = {}
        
        # Try to load existing states if file exists (for resuming)
        # Skip loading if force_rerun is set
        if not args.force_rerun and os.path.exists(states_file):
            print(f"Found existing states file, loading to resume...")
            try:
                with open(states_file, "rb") as f:
                    existing_data = pickle.load(f)
                all_states = existing_data.get("all_states", [])
                states_by_checkpoint = existing_data.get("states_by_checkpoint", {})
                print(f"  Loaded {len(all_states)} existing states from {len(states_by_checkpoint)} checkpoints")
            except Exception as e:
                print(f"  Warning: Could not load existing states file: {e}")
                print(f"  Starting fresh collection...")
                all_states = []
                states_by_checkpoint = {}
        elif args.force_rerun:
            print(f"Force rerun enabled: starting fresh collection (will overwrite existing data)")
            all_states = []
            states_by_checkpoint = {}
        
        for gen, ckpt_path in filtered_checkpoints:
            # Skip if we already have states for this checkpoint (unless force_rerun)
            if not args.force_rerun and gen in states_by_checkpoint:
                print(f"\nSkipping checkpoint: generation {gen} (already collected)")
                continue
            
            print(f"\nCollecting states from checkpoint: generation {gen}")
            
            try:
                # Load checkpoint
                _, best_member = load_checkpoint(ckpt_path, exp)
                
                # Create policy from checkpoint
                policy, _ = create_policy_from_checkpoint(exp, best_member)
                
                # Collect states and trajectories from this checkpoint
                states, trajectories = collect_states_from_evaluation(
                    exp, policy, args.num_episodes, args.task_id, for_eval
                )
                
                all_states.extend(states)
                states_by_checkpoint[gen] = states
                
                # Save trajectories with logits for this checkpoint
                gen_trajectories_file = os.path.join(eval_dir_base, f"gen_{gen}_trajectories.pkl")
                with open(gen_trajectories_file, "wb") as f:
                    pickle.dump(trajectories, f)
                print(f"  Saved {len(trajectories)} trajectories with logits to {gen_trajectories_file}")
                
                print(f"  Collected {len(states)} states from generation {gen}")
                
                # Save incrementally after each checkpoint to avoid losing progress
                # Remove duplicates while preserving order
                seen = set()
                unique_states = []
                for state in all_states:
                    state_tuple = tuple(state)
                    if state_tuple not in seen:
                        seen.add(state_tuple)
                        unique_states.append(state)
                
                states_data = {
                    "all_states": unique_states,
                    "states_by_checkpoint": states_by_checkpoint,
                    "num_episodes": args.num_episodes,
                    "task_id": args.task_id
                }
                
                with open(states_file, "wb") as f:
                    pickle.dump(states_data, f)
                
                print(f"  Saved {len(unique_states)} unique states (incremental save)")
                
            except Exception as e:
                print(f"  ✗ Error collecting states from generation {gen}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # Final deduplication and save
        seen = set()
        unique_states = []
        for state in all_states:
            state_tuple = tuple(state)
            if state_tuple not in seen:
                seen.add(state_tuple)
                unique_states.append(state)
        
        print(f"\nTotal states collected: {len(all_states)}")
        print(f"Unique states: {len(unique_states)}")
        
        # Final save
        states_data = {
            "all_states": unique_states,
            "states_by_checkpoint": states_by_checkpoint,
            "num_episodes": args.num_episodes,
            "task_id": args.task_id
        }
        
        with open(states_file, "wb") as f:
            pickle.dump(states_data, f)
        
        print(f"States saved to: {states_file}")
    else:
        # Load existing states
        print(f"\nLoading existing states from: {states_file}")
        if not os.path.exists(states_file):
            raise FileNotFoundError(f"States file not found: {states_file}")
        
        with open(states_file, "rb") as f:
            states_data = pickle.load(f)
        
        unique_states = states_data["all_states"]
        print(f"Loaded {len(unique_states)} unique states")
    
    # ============================================================================
    # PHASE 2: Evaluate all checkpoints on collected states
    # ============================================================================
    print(f"\n{'='*60}")
    print("PHASE 2: Evaluating all checkpoints on collected states")
    print(f"{'='*60}")
    
    action_probs_file = os.path.join(eval_dir_base, "action_probabilities.pkl")
    action_probs_results = {}
    
    # Load existing action probabilities if they exist and not forcing rerun
    if not args.force_rerun and os.path.exists(action_probs_file):
        print(f"Loading existing action probabilities from: {action_probs_file}")
        try:
            with open(action_probs_file, "rb") as f:
                action_probs_results = pickle.load(f)
            print(f"  Loaded action probabilities for {len(action_probs_results)} checkpoints")
        except Exception as e:
            print(f"  Warning: Could not load existing action probabilities: {e}")
            action_probs_results = {}
    elif args.force_rerun:
        print(f"Force rerun enabled: will recompute all action probabilities")
        action_probs_results = {}
    
    for gen, ckpt_path in filtered_checkpoints:
        # Skip if we already have action probabilities for this checkpoint (unless force_rerun)
        if not args.force_rerun and gen in action_probs_results:
            print(f"\nSkipping checkpoint: generation {gen} (action probabilities already computed)")
            continue
        print(f"\nEvaluating checkpoint: generation {gen}")
        
        try:
            # Load checkpoint
            _, best_member = load_checkpoint(ckpt_path, exp)
            
            # Create policy from checkpoint
            policy, _ = create_policy_from_checkpoint(exp, best_member)
            
            # Evaluate policy on all collected states
            action_probs = evaluate_policy_on_states(exp, policy, unique_states, for_eval)
            
            action_probs_results[gen] = action_probs
            
            print(f"  ✓ Evaluated generation {gen} on {len(unique_states)} states")
            print(f"    Action probability dictionary size: {len(action_probs)}")
            
        except Exception as e:
            print(f"  ✗ Error evaluating generation {gen}: {e}")
            import traceback
            traceback.print_exc()
            continue
        else:
            # Save incrementally after each checkpoint to avoid losing progress
            with open(action_probs_file, "wb") as f:
                pickle.dump(action_probs_results, f)
            print(f"  Saved action probabilities incrementally (total: {len(action_probs_results)} checkpoints)")
    
    # Final save of action probabilities
    with open(action_probs_file, "wb") as f:
        pickle.dump(action_probs_results, f)
    
    print(f"\n{'='*60}")
    print(f"Evaluation complete!")
    print(f"Evaluated {len(action_probs_results)} checkpoints")
    print(f"States file: {states_file}")
    print(f"Action probabilities file: {action_probs_file}")
    print(f"{'='*60}")
    
    # Print summary
    print("\nSummary:")
    print(f"  Unique states: {len(unique_states)}")
    print(f"  Checkpoints evaluated: {len(action_probs_results)}")
    for gen in sorted(action_probs_results.keys()):
        num_entries = len(action_probs_results[gen])
        print(f"    Generation {gen}: {num_entries} state-action pairs")


if __name__ == "__main__":
    main()
