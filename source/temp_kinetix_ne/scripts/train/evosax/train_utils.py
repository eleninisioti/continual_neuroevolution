
import functools
import os
import pickle
import yaml
from scripts.train.rl.ppo.hyperparams import hyperparams
from scripts.train.base.experiment import Experiment
from functools import partial
import numpy as onp
import jax.numpy as jnp
import jax
from scripts.train.base.visuals import viz_histogram, viz_heatmap
from scripts.train.rl.ppo.hyperparams import hyperparams
from scipy import stats
from methods.evosax_wrapper.base.tasks.rl import EcorobotTask
from methods.evosax_wrapper.direct_encodings.model import make_model
from methods.evosax_wrapper.base.training.evolution import EvosaxTrainer
from methods.evosax_wrapper.base.training.logging  import Logger
import equinox as eqx
from brax import envs as brax_envs

import evosax
from methods.evosax_wrapper.base.tasks.rl import GatesTask
from methods.evosax_wrapper.base.tasks.rl import GymnaxTask, GymnaxTaskWithPerturbation, MinatarMultiTask, CraftaxTask, KinetixTask, BraxTask
from craftax.craftax.envs.craftax_symbolic_env import CraftaxSymbolicEnvNoAutoReset
from methods.evosax_wrapper.base.tasks.rl import CraftaxState
import wandb
import gymnax
from kinetix.environment.env import make_kinetix_env
from kinetix.util import generate_params_from_config
from kinetix.environment.ued.ued import make_reset_fn_from_config
from kinetix.environment.utils import ActionType, ObservationType
from kinetix.environment.env_state import EnvParams, StaticEnvParams
from kinetix.util.config import normalise_config
from flax.serialization import to_state_dict
from ecorobot import envs as ecorobot_envs
from methods.Kinetix.kinetix.util.saving import load_from_json_file


def _unpmap(v):
  return jax.tree_util.tree_map(lambda x: x[0], v)

class EvosaxExperiment(Experiment):

    def __init__(self, env_config, model_config, exp_config, optimizer_config):
        super().__init__(env_config, model_config, exp_config, optimizer_config)
        # Store weights history for PCA visualization
        self.weights_history = []  # List of (generation, weights, fitnesses, best_indiv)
        # Track first optimal solution for drift analysis
        self.first_optimal_weights = None
        self.first_optimal_gen = None
        self.first_optimal_fitness = None
        

    def setup_trial_keys(self):
        key = jax.random.PRNGKey(self.config["exp_config"]["trial_seed"])

        self.model_key, self.train_key = jax.random.split(key, 2)
        
    def init_model(self):
      self.model = make_model(self.config, self.model_key, env=self.env,
                              env_params=self.config["env_config"]["gymnax_env_params"],
                              kinetix_config=self.config["env_config"]["kinetix_config"])

    def cleanup(self):
        pass
    
    
    
    def setup_stepping_gates_env(self):
        
        self.env = stepping_gates_envs.get_environment(env_name=self.config["env_config"]["env_name"],
                                                      **self.config["env_config"]["env_params"])
        
        self.config["env_config"]["action_size"] = self.env.action_size
        self.config["env_config"]["observation_size"] = self.env.observation_size
        self.config["env_config"]["episode_length"] = self.env.episode_length
        self.config["env_config"]["num_tasks"] = self.env.num_tasks
        
    def setup_ecorobot_env(self):
        self.env = ecorobot_envs.get_environment(env_name=self.config["env_config"]["env_name"],
                                                      **self.config["env_config"]["env_params"])
        
        self.config["env_config"]["action_size"] = self.env.action_size
        self.config["env_config"]["observation_size"] = self.env.observation_size
        self.config["env_config"]["episode_length"] = 1000
        self.config["env_config"]["num_tasks"] = self.env.num_tasks
        self.config["env_config"]["gymnax_env_params"] = None
        self.config["env_config"]["kinetix_config"] = None
        self.for_eval = None
        
        
    def setup_brax_env(self):
        self.env = brax_envs.get_environment(env_name=self.config["env_config"]["env_name"],backend="mjx",
                                                      **self.config["env_config"]["env_params"])
        
        self.config["env_config"]["action_size"] = self.env.action_size
        self.config["env_config"]["observation_size"] = self.env.observation_size
        self.config["env_config"]["episode_length"] = 100
        self.config["env_config"]["num_tasks"] = 1
        self.config["env_config"]["gymnax_env_params"] = None
        self.config["env_config"]["kinetix_config"] = None
        self.for_eval = None

        
        
        
    def setup_kinetix_env(self):
        
        with open("scripts/train/evosax/kinetix_config_pixels.yaml", "r") as f:
            config = yaml.load(f, Loader=yaml.SafeLoader)
            
        config = normalise_config(config, name="PPO")

       
        env_state, static_env_params, env_params= load_from_json_file(self.config["env_config"]["env_name"])
        #self.env_params, static_env_params = generate_params_from_config(config)
        config["env_params"] = to_state_dict(env_params)
        config["static_env_params"] = to_state_dict(static_env_params)

        reset_fn = make_reset_fn_from_config(config, env_params, static_env_params)
        self.env = make_kinetix_env(
                                                observation_type=config["observation_type"],
                                                action_type=config["action_type"],
                                                reset_fn=reset_fn,
                                                env_params=env_params,
                                                static_env_params=static_env_params)
        
        
     
        self.config["env_config"]["kinetix_config"] = config

        config["env_params"] = to_state_dict(env_params)
        config["static_env_params"] = to_state_dict(static_env_params)
        
        self.for_eval = {"env_params": env_params, "static_env_params": static_env_params, "env_state": env_state}

     
        #env_params["noise"] = 2.0

        #if self.config["env_config"]["env_params"]:
        #    env_params = env_params.replace(**self.config["env_config"]["env_params"])
        self.config["env_config"]["gymnax_env_params"] = env_params

        self.config["env_config"]["action_size"] = 20
        obs_size = 20 # random value
        self.config["env_config"]["observation_size"] = obs_size

        self.config["env_config"]["num_tasks"] = 1
        self.config["env_config"]["episode_length"] = 1000
        
        
        
        
        
    def setup_craftax_env(self):
        self.env = CraftaxSymbolicEnvNoAutoReset()
        env_params = self.env.default_params
        self.config["env_config"]["gymnax_env_params"] = env_params

        self.config["env_config"]["observation_size"] = self.env.observation_space(env_params).shape[0]
        print(self.config["env_config"]["observation_size"])
        self.config["env_config"]["action_size"] = self.env.num_actions
        self.config["env_config"]["episode_length"] = 1000
        self.config["env_config"]["num_tasks"] = 1
        
        
    def setup_minatar_multienv(self):
        self.config["env_config"]["gymnax_env_params"] = []
        action_size = []
        obs_size = []
        
        if self.config["env_config"]["env_name"] == "asterix_and_breakout":
            #env_names = ["Breakout-MinAtar", "Asterix-MinAtar", "SpaceInvaders-MinAtar", "Freeway-MinAtar"]
           # env_names = [ "SpaceInvaders-MinAtar",   "Breakout-MinAtar"]
            env_names = [ "Breakout-MinAtar", "Asterix-MinAtar", "SpaceInvaders-MinAtar"]
            env_names = ["Breakout-MinAtar"]



        obs_sizes = []
        action_sizes = []
        self.env = []
        for env_name in env_names:
            env, env_params = gymnax.make(env_id=env_name)
            #if self.config["env_config"]["env_params"]:
            #    env_params = env_params.replace(**self.config["env_config"]["env_params"])
            self.config["env_config"]["gymnax_env_params"].append(env_params)
            obs_size =  env.obs_shape
            obs_sizes.append(obs_size[-1])
            action_sizes.append(env.num_actions)
            print(obs_size)
            print(env.num_actions)
            self.env.append(env_name)
            
        action_size = max(action_sizes)
        obs_size = max(obs_sizes)
        

        
            
        self.config["env_config"]["action_size"] = action_size
        self.config["env_config"]["observation_size"] = obs_size
        self.config["env_config"]["num_tasks"] = 1
        self.config["env_config"]["episode_length"] = 1000
        self.config["env_config"]["gymnax_env_params"] = None
        self.config["env_config"]["kinetix_config"] = None

        
        
        
        
        
        
        
    def setup_gymnax_env(self):
        self.env, env_params = gymnax.make(env_id=self.config["env_config"]["env_name"])
        
        self.for_eval = {}
        #env_params["noise"] = 2.0

        #if self.config["env_config"]["env_params"]:
        #    env_params = env_params.replace(**self.config["env_config"]["env_params"])
        self.config["env_config"]["gymnax_env_params"] = env_params

        self.config["env_config"]["action_size"] = self.env.num_actions
        if  "MountainCar" in self.config["env_config"]["env_name"]:
            obs_size = 2
        elif "MinAtar" in self.config["env_config"]["env_name"]:
            obs_size =  self.env.obs_shape[-1]
        else:
            obs_size = self.env.obs_shape[0]
        self.config["env_config"]["observation_size"] = obs_size

        self.config["env_config"]["num_tasks"] = 1
        self.config["env_config"]["episode_length"] = self.config["env_config"]["env_params"]["max_steps_in_episode"]
        self.config["env_config"]["kinetix_config"] = None
        
        
        
        






    def metrics_fn(self, log_info,  data,behaviroral_diversity, episode_length, task_params, num_nodes, num_edges, noise, current_gravity, n_dormant, diversity, 
				   flat_mean, flat_min, flat_max, flat_var,
				   mean_individual_mean, mean_individual_min, mean_individual_max, mean_individual_var,
				   var_individual_mean, var_individual_min, var_individual_max, var_individual_var,
				   mean_skewness, mean_kurtosis, mean_uniformity_ratio,
				   mean_upper_tail_ratio, mean_lower_tail_ratio,
				   individual_skewness, individual_kurtosis, uniformity_ratio,
				   flat_params_for_testing):
        
        # Create a file path for saving weights history (captured in closure)
        try:
            weights_history_file = self.config["exp_config"]["trial_dir"] + "/data/train/weights_history.pkl"
        except:
            weights_history_file = None

        def callback(log_info, behaviroral_diversity, episode_length, task_params, data, num_nodes, num_edge, noise, current_gravity, n_dormant, diversity,
					flat_mean, flat_min, flat_max, flat_var,
					mean_individual_mean, mean_individual_min, mean_individual_max, mean_individual_var,
					var_individual_mean, var_individual_min, var_individual_max, var_individual_var,
					mean_skewness, mean_kurtosis, mean_uniformity_ratio,
					mean_upper_tail_ratio, mean_lower_tail_ratio,
					individual_skewness, individual_kurtosis, uniformity_ratio,
					flat_params_for_testing):
            fitnesses_np = onp.array(data["fitness"])
            best_indiv = int(onp.argmax(fitnesses_np))
            
            # Extract generation counter before reassigning log_info
            # log_info is a TrainState object that gets serialized, so we need to access it properly
            if hasattr(log_info, 'gen_counter'):
                generation = log_info.gen_counter
            elif isinstance(log_info, dict) and 'gen_counter' in log_info:
                generation = log_info['gen_counter']
            else:
                # Fallback: try to get it from the state if available
                generation = 0  # Default fallback

            # Compute fitness statistics
            fitness_mean = onp.mean(fitnesses_np)
            fitness_max = onp.max(fitnesses_np)
            fitness_min = onp.min(fitnesses_np)
            fitness_var = onp.var(fitnesses_np)
            fitness_std = onp.std(fitnesses_np)

            log_info = {
                "current_best_fitness": fitness_max,
                "mean_fitness": fitness_mean,
                "min_fitness": fitness_min,
                "fitness_var": fitness_var,
                "fitness_std": fitness_std,
                "best_individual_index": best_indiv,
                "behaviroral_diversity": behaviroral_diversity,

                "mean episode_length": onp.mean(onp.mean(onp.array(episode_length), axis=1)),
                "max episode_length": onp.max(onp.mean(onp.array(episode_length), axis=1)),
                "min episode_length": onp.min(onp.mean(onp.array(episode_length), axis=1)),
                "best indiv episode length": onp.mean(onp.array(episode_length)[best_indiv]),


                "generation": generation,
                "current_task": task_params,
                "current_gravity": current_gravity,
                "diversity": diversity,
                "flat_params/mean": float(onp.array(flat_mean)) if flat_mean is not None else None,
                "flat_params/min": float(onp.array(flat_min)) if flat_min is not None else None,
                "flat_params/max": float(onp.array(flat_max)) if flat_max is not None else None,
                "flat_params/var": float(onp.array(flat_var)) if flat_var is not None else None,
                # Per-individual statistics (mean across population)
                "individual/mean_of_means": float(onp.array(mean_individual_mean)) if mean_individual_mean is not None else None,
                "individual/mean_of_mins": float(onp.array(mean_individual_min)) if mean_individual_min is not None else None,
                "individual/mean_of_maxs": float(onp.array(mean_individual_max)) if mean_individual_max is not None else None,
                "individual/mean_of_vars": float(onp.array(mean_individual_var)) if mean_individual_var is not None else None,
                # Variance across population (how much individuals differ)
                "individual/var_of_means": float(onp.array(var_individual_mean)) if var_individual_mean is not None else None,
                "individual/var_of_mins": float(onp.array(var_individual_min)) if var_individual_min is not None else None,
                "individual/var_of_maxs": float(onp.array(var_individual_max)) if var_individual_max is not None else None,
                "individual/var_of_vars": float(onp.array(var_individual_var)) if var_individual_var is not None else None,
                # Distribution shape metrics
                "dist/skewness": float(onp.array(mean_skewness)) if mean_skewness is not None else None,
                "dist/kurtosis": float(onp.array(mean_kurtosis)) if mean_kurtosis is not None else None,
                "dist/uniformity_ratio": float(onp.array(mean_uniformity_ratio)) if mean_uniformity_ratio is not None else None,
                "dist/upper_tail_ratio": float(onp.array(mean_upper_tail_ratio)) if mean_upper_tail_ratio is not None else None,
                "dist/lower_tail_ratio": float(onp.array(mean_lower_tail_ratio)) if mean_lower_tail_ratio is not None else None,
                #"navigability": log_info.navig,
                #"navigability_online": log_info.navig_online,
                #"robustness_fitness": log_info.robustness_fitness,
                #"robustness": log_info.robustness,
                "num_nodes": num_nodes,
                "num_edges": 0,
                "noise": onp.array(noise[0]) if len(onp.array(noise).shape) > 0 else onp.array(noise),
                "mean_n_dormant": onp.mean(onp.array(n_dormant)),
                "max_n_dormant": onp.max(onp.array(n_dormant)),
                "min_n_dormant": onp.min(onp.array(n_dormant)),
                "best_indiv_n_dormant": onp.array(n_dormant)[best_indiv]

            }
            
            max_level = 0
            mean_level = 0 
            
            if "info" in data["data"]:
                for key, value in data["data"]["info"].items():
                    log_info["mean_" + key] = jnp.mean(value)
                    log_info["max_" + key] = jnp.max(value)
                    
                    if "enter_dungeon" in key:
                        best_value = int(jnp.max(value))
                        if best_value > 0:
                            max_level = 1
                        
                    if "enter_gnomish_mines" in key:
                        best_value = int(jnp.max(value))
                        if best_value > 0:
                            max_level = 2
                            
                    if "enter_sewers" in key:
                        best_value = int(jnp.max(value))
                        if best_value > 0:
                            max_level = 3
                            
                    if "enter_vault" in key:
                        best_value = int(jnp.max(value))

                            
                    if "enter_troll_mines" in key:
                        best_value = int(jnp.max(value))
                        if best_value > 0:
                            max_level = 4                    
                
                        
                    
            log_info["deepest_level"] = max_level
            
            # Save weights history for PCA visualization (generation already extracted above)
            """
            if flat_params_for_testing is not None and weights_history_file is not None:
                flat_params_np = onp.array(flat_params_for_testing)
                fitnesses_np = onp.array(data["fitness"])
                best_indiv = int(onp.argmax(fitnesses_np))
                current_best_weights = flat_params_np[best_indiv, :]
                
                # Compute distance from previous best individual
                prev_best_file = weights_history_file.replace("weights_history.pkl", "previous_best_weights.pkl")
                distance_metrics = {}
                
                try:
                    # Compute weight statistics needed for normalization
                    avg_weight_magnitude = onp.mean(onp.abs(current_best_weights))
                    norm_current = onp.linalg.norm(current_best_weights)
                    
                    if os.path.exists(prev_best_file):
                        with open(prev_best_file, "rb") as f:
                            prev_best_data = pickle.load(f)
                            prev_best_weights = prev_best_data['weights']
                            prev_gen = prev_best_data['generation']
                            
                            # Compute various distance metrics
                            # L2 distance (Euclidean)
                            l2_distance = onp.linalg.norm(current_best_weights - prev_best_weights)
                            distance_metrics['best_diff/l2_distance'] = float(l2_distance)
                            
                            # Normalized L2 distance (relative to weight magnitude)
                            normalized_l2 = l2_distance / (avg_weight_magnitude * len(current_best_weights) + 1e-8)
                            distance_metrics['best_diff/normalized_l2'] = float(normalized_l2)
                            
                            # Cosine similarity (1 = identical, 0 = orthogonal, -1 = opposite)
                            dot_product = onp.dot(current_best_weights, prev_best_weights)
                            norm_prev = onp.linalg.norm(prev_best_weights)
                            cosine_sim = dot_product / (norm_current * norm_prev + 1e-8)
                            distance_metrics['best_diff/cosine_similarity'] = float(cosine_sim)
                            distance_metrics['best_diff/cosine_distance'] = float(1.0 - cosine_sim)  # Distance = 1 - similarity
                            
                            # Mean absolute difference
                            mean_abs_diff = onp.mean(onp.abs(current_best_weights - prev_best_weights))
                            distance_metrics['best_diff/mean_abs_diff'] = float(mean_abs_diff)
                            
                            # Max absolute difference
                            max_abs_diff = onp.max(onp.abs(current_best_weights - prev_best_weights))
                            distance_metrics['best_diff/max_abs_diff'] = float(max_abs_diff)
                            
                            # Generation gap
                            distance_metrics['best_diff/generations_since_update'] = generation - prev_gen
                            
                            log_info.update(distance_metrics)
                    else:
                        # First generation - no previous best to compare
                        distance_metrics['best_diff/l2_distance'] = 0.0
                        distance_metrics['best_diff/normalized_l2'] = 0.0
                        distance_metrics['best_diff/cosine_similarity'] = 1.0
                        distance_metrics['best_diff/cosine_distance'] = 0.0
                        distance_metrics['best_diff/mean_abs_diff'] = 0.0
                        distance_metrics['best_diff/max_abs_diff'] = 0.0
                        distance_metrics['best_diff/generations_since_update'] = 0
                        log_info.update(distance_metrics)
                    
                    # Save current best as previous best for next generation
                    with open(prev_best_file, "wb") as f:
                        pickle.dump({
                            'generation': generation,
                            'weights': current_best_weights.copy(),
                            'fitness': float(fitnesses_np[best_indiv])
                        }, f)
                    
                    # Track drift from first optimal solution
                    current_best_fitness = float(fitnesses_np[best_indiv])
                    
                    # Check if we've reached a new maximum fitness (first optimal solution)
                    if (self.first_optimal_weights is None or 
                        current_best_fitness > self.first_optimal_fitness):
                        # This is the first time we've reached this fitness level
                        self.first_optimal_weights = current_best_weights.copy()
                        self.first_optimal_gen = generation
                        self.first_optimal_fitness = current_best_fitness
                    
                    # Compute distance from current best to first optimal solution
                    if self.first_optimal_weights is not None:
                        drift_l2 = onp.linalg.norm(current_best_weights - self.first_optimal_weights)
                        drift_normalized = drift_l2 / (avg_weight_magnitude * len(current_best_weights) + 1e-8)
                        
                        dot_product_drift = onp.dot(current_best_weights, self.first_optimal_weights)
                        norm_first = onp.linalg.norm(self.first_optimal_weights)
                        cosine_sim_drift = dot_product_drift / (norm_current * norm_first + 1e-8)
                        
                        drift_mean_abs = onp.mean(onp.abs(current_best_weights - self.first_optimal_weights))
                        
                        log_info["drift_from_first_optimal/l2_distance"] = float(drift_l2)
                        log_info["drift_from_first_optimal/normalized_l2"] = float(drift_normalized)
                        log_info["drift_from_first_optimal/cosine_similarity"] = float(cosine_sim_drift)
                        log_info["drift_from_first_optimal/cosine_distance"] = float(1.0 - cosine_sim_drift)
                        log_info["drift_from_first_optimal/mean_abs_diff"] = float(drift_mean_abs)
                        log_info["drift_from_first_optimal/generations_since_optimal"] = generation - self.first_optimal_gen
                    
                except Exception as e:
                    print(f"Warning: Could not compute/save best individual difference: {e}")
                
                
                # Save to pickle file (append mode - load, update, save)
                try:
                    if os.path.exists(weights_history_file):
                        with open(weights_history_file, "rb") as f:
                            weights_history = pickle.load(f)
                    else:
                        weights_history = []
                    
                    weights_history.append({
                        'generation': generation,
                        'weights': flat_params_np.copy(),  # (pop_size, num_params)
                        'fitnesses': fitnesses_np.copy(),
                        'best_indiv': best_indiv
                    })
                    
                    with open(weights_history_file, "wb") as f:
                        pickle.dump(weights_history, f)
                except Exception as e:
                    print(f"Warning: Could not save weights history: {e}")
            
            # Classify individuals by distribution type and count
            # Statistical tests for uniform, normal, and power-law distributions
            if (flat_params_for_testing is not None and 
                individual_skewness is not None and 
                individual_kurtosis is not None and 
                uniformity_ratio is not None):
                
                flat_params_np = onp.array(flat_params_for_testing)
                pop_size = flat_params_np.shape[0]
                skewness_np = onp.array(individual_skewness)
                kurtosis_np = onp.array(individual_kurtosis)
                uniformity_np = onp.array(uniformity_ratio)
                
                # Use statistical tests to determine distribution type
                # Test each individual's weights against uniform, normal, and power-law distributions
                pvalue_threshold = 0.05  # Significance level for rejecting null hypothesis
                
                count_normal = 0
                count_uniform = 0
                count_powerlaw = 0
                count_other = 0
                
                # Store p-values for debugging
                pvalues_uniform = []
                pvalues_normal = []
                
                for idx in range(pop_size):
                    individual_weights = flat_params_np[idx, :]
                    
                    # Test 1: Kolmogorov-Smirnov test against uniform distribution
                    # Normalize weights to [0, 1] using observed min/max
                    # This tests if the distribution is uniform within its current range
                    min_w = individual_weights.min()
                    max_w = individual_weights.max()
                    uniform_pvalue = None
                    if max_w > min_w and len(individual_weights) > 1:
                        normalized_uniform = (individual_weights - min_w) / (max_w - min_w)
                        try:
                            _, uniform_pvalue = stats.kstest(normalized_uniform, 'uniform')
                            pvalues_uniform.append(uniform_pvalue)
                        except:
                            uniform_pvalue = None
                    
                    # Test 2: Test against normal distribution
                    # Normalize to standard normal (mean=0, std=1)
                    mean_w = individual_weights.mean()
                    std_w = individual_weights.std()
                    normal_pvalue = None
                    if std_w > 1e-8 and len(individual_weights) > 1:
                        normalized_normal = (individual_weights - mean_w) / std_w
                        try:
                            # Use Shapiro-Wilk for small samples, KS for larger
                            if len(normalized_normal) <= 5000:
                                _, normal_pvalue = stats.shapiro(normalized_normal)
                            else:
                                # For large samples, use KS test
                                _, normal_pvalue = stats.kstest(normalized_normal, 'norm')
                            pvalues_normal.append(normal_pvalue)
                        except:
                            normal_pvalue = None
                    
                    # Test 3: Test for power-law using tail analysis
                    # Power-law has heavy tails - check if upper tail follows power-law
                    # Sort absolute values and check tail behavior
                    abs_weights_sorted = onp.sort(onp.abs(individual_weights))[::-1]  # Descending
                    n_tail = max(10, len(abs_weights_sorted) // 20)  # Use top 5% for tail
                    tail_weights = abs_weights_sorted[:n_tail]
                    
                    powerlaw_pvalue = None
                    if len(tail_weights) > 3 and tail_weights[0] > 0:
                        # Test if log(tail) vs log(rank) is linear (power-law signature)
                        ranks = onp.arange(1, len(tail_weights) + 1, dtype=float)
                        log_ranks = onp.log(ranks)
                        log_tail = onp.log(tail_weights + 1e-10)  # Add small epsilon to avoid log(0)
                        
                        # Fit linear regression: log(tail) = a + b*log(rank)
                        # Power-law would have b ≈ -α (negative slope)
                        try:
                            from scipy.stats import linregress
                            slope, intercept, r_value, p_value, std_err = linregress(log_ranks, log_tail)
                            # For power-law, we expect negative slope and good fit (high |r|)
                            # Use p-value from regression as indicator
                            # If slope is significantly negative and R² is high, likely power-law
                            powerlaw_pvalue = 1.0 - abs(r_value)  # Lower when correlation is high
                        except:
                            powerlaw_pvalue = None
                    
                    # Classify based on p-values: accept distribution if p-value > threshold
                    # Choose the distribution with highest p-value (least rejected)
                    classifications = {}
                    if uniform_pvalue is not None:
                        classifications['uniform'] = uniform_pvalue
                    if normal_pvalue is not None:
                        classifications['normal'] = normal_pvalue
                    if powerlaw_pvalue is not None:
                        classifications['powerlaw'] = powerlaw_pvalue
                    
                    # If we have test results, choose the best fit
                    if classifications:
                        best_dist = max(classifications, key=classifications.get)
                        best_pvalue = classifications[best_dist]
                        
                        # Only classify if p-value suggests we can't reject the hypothesis
                        if best_pvalue > pvalue_threshold:
                            if best_dist == 'uniform':
                                count_uniform += 1
                            elif best_dist == 'normal':
                                count_normal += 1
                            elif best_dist == 'powerlaw':
                                count_powerlaw += 1
                        else:
                            # All tests rejected - doesn't fit any standard distribution
                            count_other += 1
                    else:
                        count_other += 1
                
                # Log summary statistics of p-values
                if pvalues_uniform:
                    log_info["dist_test/mean_uniform_pvalue"] = float(onp.mean(pvalues_uniform))
                    log_info["dist_test/min_uniform_pvalue"] = float(onp.min(pvalues_uniform))
                if pvalues_normal:
                    log_info["dist_test/mean_normal_pvalue"] = float(onp.mean(pvalues_normal))
                    log_info["dist_test/min_normal_pvalue"] = float(onp.min(pvalues_normal))
                
                log_info["dist_count/normal"] = count_normal
                log_info["dist_count/uniform"] = count_uniform
                log_info["dist_count/powerlaw"] = count_powerlaw
                log_info["dist_count/other"] = count_other
                log_info["dist_count/total"] = pop_size
            """

            wandb.log(log_info)
            
            for key, value in log_info.items():
                if "mean_" not in key and "max_" not in key:
                    print(key, value)
                else:
                    if value > 0.0:
                        print(key, value)
            
        jax.debug.callback(callback, log_info, behaviroral_diversity, episode_length, task_params, data, num_nodes, num_edges, noise, current_gravity, n_dormant, diversity,
						   flat_mean, flat_min, flat_max, flat_var,
						   mean_individual_mean, mean_individual_min, mean_individual_max, mean_individual_var,
						   var_individual_mean, var_individual_min, var_individual_max, var_individual_var,
						   mean_skewness, mean_kurtosis, mean_uniformity_ratio,
						   mean_upper_tail_ratio, mean_lower_tail_ratio,
						   individual_skewness, individual_kurtosis, uniformity_ratio,
						   flat_params_for_testing)

    def eval_task(self, best_member, tasks, gens, final_policy=False):

        policy_params = self.params_shaper.reshape_single(best_member)

        policy = eqx.combine(policy_params, self.statics)


        init_policy_state, _ = policy.initialize(jax.random.PRNGKey(0))

        if self.config["env_config"]["env_type"] == "kinetix":
            act_fn = partial(policy,
                             key=self.model_key,
                             state=init_policy_state,
                             obs_size=self.config["env_config"]["observation_size"],
                             action_size=self.config["env_config"]["action_size"])

            
        else:

            act_fn = partial(policy, key=self.model_key, state=init_policy_state, obs_size=self.config["env_config"]["observation_size"], action_size=self.config["env_config"]["action_size"])


        super().run_eval(act_fn, tasks, final_policy=final_policy, gens=gens, for_eval=self.for_eval)

   
    def get_final_policy(self):
        policy_params = self.params_shaper.reshape_single(self.final_state["params"])

        policy = eqx.combine(policy_params, self.statics)

        init_policy_state, dev_states = policy.initialize(jax.random.PRNGKey(0))
        dev_steps = self.config["model_config"]["model_params"]["max_dev_steps"] + 2
        data = jax.tree_map(lambda x: x[dev_steps, ...], dev_states)
        return data.weights

    def train_trial(self):

        def data_fn(data: dict):
            return {}

        logger = Logger(True,
                        metrics_fn=self.metrics_fn,
                        ckpt_freq=200,
                        aim_freq=1,
                        ckpt_dir=self.config["exp_config"]["trial_dir"] + "/data/train")

        fitness_shaper = evosax.FitnessShaper(maximize=True,
                                              centered_rank=False)

        #phenotype_size = (self.model.max_nodes, self.model.max_nodes)
        params, statics = eqx.partition(self.model, eqx.is_array)
        self.statics = statics
        self.params_shaper = evosax.ParameterReshaper(params)

    
        """
        self.env = GatesTask(statics,

                        env=self.config["env_config"]["env_name"],
                        max_steps= self.config["env_config"]["episode_length"],
                        data_fn=data_fn, env_kwargs={**self.config["env_config"]["env_params"]})
        """
        
        if self.config["env_config"]["env_type"] == "ecorobot":
        
            self.env = EcorobotTask(statics=self.statics,
                                env=self.config["env_config"]["env_name"],
                                max_steps=500,
                                data_fn=data_fn,
                                env_kwargs={**self.config["env_config"]["env_params"]})
            
        elif self.config["env_config"]["env_type"] == "brax":
                self.env = BraxTask(statics=self.statics,
                                    env=self.config["env_config"]["env_name"],
                                    max_steps=1000,
                                    data_fn=data_fn,
                                    env_kwargs={**self.config["env_config"]["env_params"]})
        elif self.config["env_config"]["env_type"] == "gymnax":
            self.env = GymnaxTaskWithPerturbation(statics=self.statics,
                                env=self.config["env_config"]["env_name"],
                                max_steps=self.config["env_config"]["episode_length"],
                                obs_size=self.config["env_config"]["observation_size"],
                                action_size=self.config["env_config"]["action_size"],
                                data_fn=data_fn,
                                env_kwargs={**self.config["env_config"]["env_params"]})
        elif self.config["env_config"]["env_type"] == "minatar_multi":
            self.env = MinatarMultiTask(statics=self.statics,
                                env=self.env,
                                max_steps=1000,
                                obs_size=self.config["env_config"]["observation_size"],
                                action_size=self.config["env_config"]["action_size"],
                                data_fn=data_fn,
                                env_kwargs={**self.config["env_config"]["env_params"]})
            
        elif self.config["env_config"]["env_type"] == "craftax":
            self.env = CraftaxTask(statics=self.statics,
                                env=self.config["env_config"]["env_name"],
                                max_steps=1000,
                                obs_size=self.config["env_config"]["observation_size"],
                                action_size=self.config["env_config"]["action_size"],
                                data_fn=data_fn,
                                env_kwargs={**self.config["env_config"]["env_params"]})
            
        elif self.config["env_config"]["env_type"] == "kinetix":
            # For multi-task, pass a list of env names (must match env_names in train_)
            env_names = ["m/h0_unicycle", "l/lever_puzzle"]
            self.env = KinetixTask(statics=self.statics,
                                env=env_names,
                                max_steps=256,
                                data_fn=data_fn,
                                env_kwargs={**self.config["env_config"]["env_params"]})
        else:
            self.env = GatesTask(statics=self.statics,
                                env=self.config["env_config"]["env_name"],
                                max_steps=1000,
                                data_fn=data_fn,
                                env_kwargs={**self.config["env_config"]["env_params"]})
       



        # Extract init_min and init_max from es_kws if present (they need to be applied to es_params, not es_kws)
        es_kws = {**self.config["optimizer_config"]["optimizer_params"]["es_kws"]}
        #init_min = es_kws.pop("init_min", None)
        
        
        if self.config["env_config"]["env_type"] == "kinetix":
            continual_config = {"perturbe_every_n_gens": 200, "noise_range": 1.0}
        else:
            continual_config = {"perturbe_every_n_gens": self.config["env_config"]["env_params"]["perturbe_every_n_gens"],
                                "noise_range": self.config["env_config"]["env_params"]["noise_range"]}
        
        
        trainer = EvosaxTrainer(train_steps=self.config["optimizer_config"]["optimizer_params"]["generations"],
                                task=self.env,
                                save_params_fn=self.save_params,
                                continual_config= continual_config,
                                strategy=self.config["optimizer_config"]["optimizer_params"]["strategy"],
                                params_shaper=self.params_shaper,
                                popsize=self.config["optimizer_config"]["optimizer_params"]["popsize"],
                                fitness_shaper=fitness_shaper,
                                num_tasks = self.env.num_tasks,
                                reward_for_solved=self.env.reward_for_solved,
                                # sigma_init = 0.01,
                                es_kws=es_kws,
                                logger=logger,
                                progress_bar=False,
                                n_devices=1,
                                eval_reps=2)

        popsize = self.config["optimizer_config"]["optimizer_params"]["popsize"]
        initial_info = {
			'Achievements/wake_up': 0.0,
			'Achievements/make_iron_armour': 0.0,
			'Achievements/eat_bat': 0.0,
			'Achievements/make_stone_pickaxe': 0.0,
			'Achievements/defeat_orc_solider': 0.0,
			'Achievements/enter_sewers': 0.0,
			'Achievements/cast_iceball': 0.0,
			'Achievements/defeat_skeleton': 0.0,
			'Achievements/defeat_ice_elemental': 0.0,
			'Achievements/collect_stone': 0.0,
			'Achievements/make_wood_sword': 0.0,
			'Achievements/cast_fireball': 0.0,
			'Achievements/place_table': 0.0,
			'Achievements/eat_cow': 0.0,
			'Achievements/defeat_orc_mage': 0.0,
			'Achievements/defeat_deep_thing': 0.0,
			'Achievements/eat_plant': 0.0,
			'Achievements/learn_fireball': 0.0,
			'Achievements/collect_drink': 0.0,
			'Achievements/make_torch': 0.0,
			'Achievements/collect_diamond': 0.0,
			'Achievements/defeat_troll': 0.0,
			'Achievements/find_bow': 0.0,
			'Achievements/make_diamond_pickaxe': 0.0,
			'Achievements/open_chest': 0.0,
			'Achievements/defeat_frost_troll': 0.0,
			'Achievements/defeat_knight': 0.0,
			'Achievements/enchant_sword': 0.0,
			'Achievements/make_diamond_sword': 0.0,
			'Achievements/collect_iron': 0.0,
			'Achievements/enter_fire_realm': 0.0,
			'Achievements/defeat_archer': 0.0,
			'Achievements/learn_iceball': 0.0,
			'Achievements/eat_snail': 0.0,
			'Achievements/defeat_fire_elemental': 0.0,
			'Achievements/make_diamond_armour': 0.0,
			'Achievements/collect_sapling': 0.0,
			'Achievements/drink_potion': 0.0,
			'Achievements/enter_gnomish_mines': 0.0,
			'Achievements/place_torch': 0.0,
			'Achievements/enter_dungeon': 0.0,
			'Achievements/collect_sapphire': 0.0,
			'Achievements/make_iron_sword': 0.0,
			'Achievements/defeat_lizard': 0.0,
			'Achievements/enter_ice_realm': 0.0,
			'Achievements/defeat_gnome_warrior': 0.0,
			'Achievements/place_furnace': 0.0,
			'Achievements/defeat_kobold': 0.0,
			'Achievements/damage_necromancer': 0.0,
			'Achievements/collect_ruby': 0.0,
			'Achievements/enter_vault': 0.0,
			'Achievements/make_wood_pickaxe': 0.0,
			'Achievements/defeat_gnome_archer': 0.0,
			'Achievements/defeat_necromancer': 0.0,
			'Achievements/defeat_pigman': 0.0,
			'Achievements/enchant_armour': 0.0,
			'Achievements/enter_graveyard': 0.0,
			'Achievements/collect_wood': 0.0,
			'Achievements/enter_troll_mines': 0.0,
			'Achievements/defeat_zombie': 0.0,
			'Achievements/fire_bow': 0.0,
			'Achievements/make_iron_pickaxe': 0.0,
			'discount': 0.0,
			'Achievements/place_stone': 0.0,
			'Achievements/make_arrow': 0.0,
			'Achievements/collect_coal': 0.0,
			'Achievements/make_stone_sword': 0.0,
			'Achievements/place_plant': 0.0
		}
        #obs, init_env_state = self.env.initialize(jax.random.PRNGKey(0), current_task=0)
        #obs, init_env_state = self.env.initialize(jax.random.PRNGKey(0), current_task=0)

        #init_env_state = CraftaxState(env_state=init_env_state, obs=obs, reward=0.0, done=False, info=initial_info)
        #init_env_state = jax.tree_map(lambda x: jnp.repeat(jnp.expand_dims(x, axis=0), popsize, axis=0), init_env_state)


        #final_info = trainer.init_and_train_(self.train_key, init_env_state=init_env_state)
        final_info, total_noise, archive_history, fitnesses_history = trainer.init_and_train_(self.train_key)
        
        
        
        #total_noise_np = onp.array(total_noise)
        #onp.savetxt(self.config["exp_config"]["trial_dir"] + "/data/train/total_noise.csv", 
        #        total_noise_np, delimiter=',', fmt='%.6f')
        
        # Save archive history for PCA visualization
        # archive_history_np = onp.array(archive_history)
        #onp.save(self.config["exp_config"]["trial_dir"] + "/data/train/archive_history.npy", archive_history_np)
        #print(f"Archive history saved with shape: {archive_history_np.shape}")

        #fitnesses_history_np = onp.array(fitnesses_history)
        ##onp.save(self.config["exp_config"]["trial_dir"] + "/data/train/fitnesses_history.npy", fitnesses_history_np)
        #print(f"Fitnesses history saved with shape: {fitnesses_history_np.shape}")

        self.final_state = {"params": final_info.best_member}
        
        # Create PCA visualization after training
        self.create_pca_visualization()
        
        
    def create_pca_visualization(self):
        """Create 2D PCA visualization of weight evolution across generations."""
        try:
            weights_history_file = self.config["exp_config"]["trial_dir"] + "/data/train/weights_history.pkl"
            
            if not os.path.exists(weights_history_file):
                print("Warning: weights_history.pkl not found, skipping PCA visualization")
                return
            
            # Load weights history
            with open(weights_history_file, "rb") as f:
                weights_history = pickle.load(f)
            
            if not weights_history:
                print("Warning: weights_history is empty, skipping PCA visualization")
                return
            
            print(f"Creating PCA visualization from {len(weights_history)} generations...")
            
            # Collect all weights across all generations
            all_weights = []
            generation_labels = []
            individual_indices = []
            is_best = []
            
            for gen_data in weights_history:
                gen = gen_data['generation']
                weights = gen_data['weights']  # (pop_size, num_params)
                best_idx = gen_data['best_indiv']
                
                pop_size = weights.shape[0]
                for idx in range(pop_size):
                    all_weights.append(weights[idx, :])
                    generation_labels.append(gen)
                    individual_indices.append(idx)
                    is_best.append(idx == best_idx)
            
            # Convert to numpy array
            all_weights = onp.array(all_weights)  # (total_individuals, num_params)
            
            # Perform PCA
            from sklearn.decomposition import PCA
            pca = PCA(n_components=2)
            weights_2d = pca.fit_transform(all_weights)
            
            print(f"PCA explained variance ratio: {pca.explained_variance_ratio_}")
            print(f"Total explained variance: {sum(pca.explained_variance_ratio_):.4f}")
            
            # Create visualization
            import matplotlib.pyplot as plt
            import matplotlib.colors as mcolors
            
            fig, ax = plt.subplots(figsize=(12, 10))
            
            # Get unique generations and create color map
            unique_gens = sorted(set(generation_labels))
            colors = plt.cm.viridis(onp.linspace(0, 1, len(unique_gens)))
            gen_to_color = {gen: colors[i] for i, gen in enumerate(unique_gens)}
            
            # Plot each generation separately
            for gen in unique_gens:
                gen_mask = onp.array(generation_labels) == gen
                gen_weights_2d = weights_2d[gen_mask]
                gen_is_best = onp.array(is_best)[gen_mask]
                
                # Plot regular individuals
                regular_mask = ~gen_is_best
                if regular_mask.any():
                    ax.scatter(weights_2d[gen_mask][regular_mask, 0], 
                             weights_2d[gen_mask][regular_mask, 1],
                             c=[gen_to_color[gen]], alpha=0.4, s=20,
                             label=f'Gen {gen}' if gen == unique_gens[0] or gen % max(1, len(unique_gens)//10) == 0 else '',
                             edgecolors='none')
                
                # Highlight best individual with a star
                best_mask = gen_is_best
                if best_mask.any():
                    best_weights_2d = gen_weights_2d[best_mask]
                    ax.scatter(best_weights_2d[:, 0], best_weights_2d[:, 1],
                             c='red', marker='*', s=300, alpha=0.9,
                             edgecolors='black', linewidths=1.5,
                             zorder=10)
            
            ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.2f}% variance)', fontsize=12)
            ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.2f}% variance)', fontsize=12)
            ax.set_title('Weight Evolution: PCA Visualization\n(Red stars = best individuals per generation)', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8, ncol=2)
            
            plt.tight_layout()
            
            # Save figure
            pca_plot_path = self.config["exp_config"]["trial_dir"] + "/data/train/pca_evolution.png"
            plt.savefig(pca_plot_path, dpi=300, bbox_inches='tight')
            print(f"PCA visualization saved to: {pca_plot_path}")
            
            plt.close()
            
        except ImportError:
            print("Warning: sklearn or matplotlib not available, skipping PCA visualization")
            print("Install with: pip install scikit-learn matplotlib")
        except Exception as e:
            print(f"Warning: Could not create PCA visualization: {e}")
            import traceback
            traceback.print_exc()
    
    def save_params(self, training_state):

        def callback(info):
            current_gen, current_task, state, interm_policies, best_indiv = info
            last_dev_step = 1
            best_member = jax.tree_map(lambda x: x[best_indiv, ...], state)

            file_path = self.config["exp_config"]["trial_dir"] + "/data/train/checkpoints/params_task_" + str(current_task-1) + ".pkl"
            if not os.path.exists(file_path):

                with open(file_path, "wb") as f:
                    pickle.dump( (current_gen,best_member), f)


            interm_policies = jax.tree_map(lambda x: x[best_indiv,0, ...], interm_policies)


            file_path = self.config["exp_config"]["trial_dir"] + "/data/train/checkpoints/policy_states_task_" + str(
                current_task - 1) + ".pkl"
            if not os.path.exists(file_path):
                with open(file_path, "wb") as f:
                    pickle.dump(interm_policies, f)

        jax.debug.callback(callback, training_state)

    def save_training_info(self):


        #TODO: here we need to load from latest generation

        #last_gen = self.config["optimizer_config"]["optimizer_params"]["generations"]
        #with open(self.config["exp_config"]["trial_dir"] + "/data/train/all_info/gen_" + str(last_gen) + "/dev_" +str(self.config["model_config"]["model_params"]["max_dev_steps"]+2) +".pkl", "rb") as f:
        #    policy_state = pickle.load(f)
        policy_state = self.final_state["params"]

        checkpoint_policy_states = []
        for task in range(self.config["env_config"]["num_tasks"]):

            try:

                with open(self.config["exp_config"]["trial_dir"] + "/data/train/checkpoints/params_task_" + str(
                        task) + ".pkl","rb") as f:
                    data = pickle.load(f)
                    checkpoint_policy_states.append(data)
            except FileNotFoundError:
                continue

        # save final policy matrix
        self.training_info = {"policy_network": {"final": policy_state,
                                                 "checkpoints": checkpoint_policy_states}}

        super().save_training_info()