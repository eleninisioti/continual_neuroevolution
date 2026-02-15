

from .utils import progress_bar_scan, progress_bar_fori
import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
from typing import Callable, Optional, Tuple, Any, TypeAlias, Union
import jax.experimental.host_callback as hcb
import yaml
from jaxtyping import PyTree
from kinetix.environment.env_state import EnvParams, StaticEnvParams
from kinetix.util.config import normalise_config
import yaml
from methods.Kinetix.kinetix.util.saving import load_from_json_file
from flax.serialization import to_state_dict
from kinetix.environment.env import make_kinetix_env
from kinetix.util import generate_params_from_config
from kinetix.environment.ued.ued import make_reset_fn_from_config
from kinetix.environment.utils import ActionType, ObservationType
Data: TypeAlias = PyTree[...]
TaskParams: TypeAlias = PyTree[...]
TrainState: TypeAlias = PyTree[...]


from .logging import Logger

class BaseTrainer(eqx.Module):
	
	"""
	"""
	#-------------------------------------------------------------------
	train_steps: int
	logger: Optional[Logger]
	progress_bar: Optional[bool]
	#-------------------------------------------------------------------

	def __init__(self, 
				 train_steps: int, 
				 logger: Optional[Logger]=None,
				 progress_bar: Optional[bool]=False):
		
		self.train_steps = train_steps
		self.progress_bar = progress_bar
		self.logger = logger

	#-------------------------------------------------------------------

	def __call__(self, key: jr.PRNGKey):

		return self.init_and_train(key)

	#-------------------------------------------------------------------

	def train(self, state: TrainState, key: jax.Array, data: Optional[Data]=None)->Tuple[TrainState, Data]:

		def _step(c, x):
			s, k = c
			k, k_ = jr.split(k)
			s, data = self.train_step(s, k_)
			
			if self.logger is not None:
				self.logger.log(s, data)

			return [s, k], {"states": s, "metrics": data}

		if self.progress_bar:
			_step = progress_bar_scan(self.train_steps)(_step) #type: ignore

		[state, key], data = jax.lax.scan(_step, [state, key], jnp.arange(self.train_steps))

		return state, data

	#-------------------------------------------------------------------

	def train_old_(self, state: TrainState, key: jax.Array, data: Optional[Data]=None, init_env_state: Optional=None)->TrainState:
     
     
		original_env = self.task_keep.env

		def _step(i, c):
			s, k, task_params, env_state, noise= c
			k, k_ = jr.split(k)
			#if self.logger is not None:
			#	dummy_data = {"fitness": jnp.array([0]), "interm_policies": [], "best_indiv": []}
			s, data, task_params, env_state = self.train_step(s, k_, task_params, i, env_state, noise)
   


			def save_params(data):
				for dev_step in range(self.logger.dev_steps + 1):
					current_dev = jax.tree_map(lambda x: x[data["best_indiv"], 0, dev_step, ...],
											   data["interm_policies"])
					self.logger.save_chkpt(current_dev, task_params, jnp.array(dev_step))
     
     

			jax.lax.cond(state.best_fitness >= self.reward_for_solved,
											   lambda x: save_params(x), lambda x: None, data)

			# Check early termination condition
			#should_stop = (i >= 1000) & (s.best_fitness < 6.0)
			should_stop = False


   
			
			# Print message when condition is met
			
			self.logger.log(s, data, task_params, noise, current_gravity=0.0)

			return [s, k, task_params, should_stop, env_state]




		if self.progress_bar:
			_step = progress_bar_fori(self.train_steps)(_step) #type: ignore

		task_params_init = 0
		#keys = jr.split(key, 10)
		num_tasks = int(self.train_steps / self.perturbe_every_n_gens) + 50
		print("train_steps", self.train_steps)
		print("perturbe_every_n_gens", self.perturbe_every_n_gens)
		print("num_tasks", num_tasks)
		quit()
		total_noise = jax.random.normal(key, (num_tasks,self.obs_size))*self.noise_range


		
		# Use scan with early termination support
		
		def _step_with_early_stop(carry, x):
			state, key, task_params, should_stop, env_state = carry
			generation = x
			current_task = generation // self.perturbe_every_n_gens
			noise = total_noise[current_task, :]
   
			#noise = jax.numpy.where(generation % self.perturbe_every_n_gens == 0, new_noise, noise)
			

			# If we should stop, return current state without training
			"""
			new_carry = jax.lax.cond(
				should_stop,
				lambda: [state, key, task_params, should_stop, env_state],  # Keep stopped state
				lambda: _step(generation, (state, key, task_params, env_state))  # Continue training
			)
			"""
			new_carry = _step(generation, (state, key, task_params, env_state, noise))
			
			# Extract the should_stop flag from the training step result
			_, _, _, new_should_stop, env_state = new_carry
			
			# Return (carry, output) pair as required by scan
			return new_carry, None
		
	

		# Run scan with early termination
		(state, key, task_params, _, _), _ = jax.lax.scan(
			_step_with_early_stop, 
			[state, key, task_params_init, False, init_env_state],  # Use list to match return type
			jnp.arange(self.train_steps)
		)
		return (state, total_noise)


	def train_(self, state: TrainState, key: jax.Array, data: Optional[Data]=None, init_env_state: Optional=None)->TrainState:
     
     
		# Get environment information from task
		# Different task types have different attributes:
		# - Kinetix tasks: env, env_params, static_env_params, env_state
		# - Gymnax tasks: env, gymnax_env_params (not env_params)
		# - Other tasks may vary
		current_env = self.task_keep.env
		
		# Try to get env_params, fallback to gymnax_env_params for Gymnax tasks
		if hasattr(self.task_keep, 'env_params'):
			current_env_params = self.task_keep.env_params
		elif hasattr(self.task_keep, 'gymnax_env_params'):
			current_env_params = self.task_keep.gymnax_env_params
		else:
			current_env_params = None
		
		# static_env_params only exists for Kinetix tasks
		current_static_env_params = getattr(self.task_keep, 'static_env_params', None)
		
		# env_state only exists for Kinetix tasks
		current_init_env_state = getattr(self.task_keep, 'env_state', None)

		def _step(i, c):
			s, k, task_params, env_state, noise= c
			k, k_ = jr.split(k)
			#if self.logger is not None:
			#	dummy_data = {"fitness": jnp.array([0]), "interm_policies": [], "best_indiv": []}
			s, data, task_params, env_state = self.train_step(
				s, k_, task_params, i, env_state, noise,
				env=current_env,
				env_params=current_env_params,
				static_env_params=current_static_env_params,
				init_env_state=current_init_env_state
			)
   


			def save_params(data):
				for dev_step in range(self.logger.dev_steps + 1):
					current_dev = jax.tree_map(lambda x: x[data["best_indiv"], 0, dev_step, ...],
											   data["interm_policies"])
					self.logger.save_chkpt(current_dev, task_params, jnp.array(dev_step))
     
     

			jax.lax.cond(state.best_fitness >= self.reward_for_solved,
											   lambda x: save_params(x), lambda x: None, data)

			# Check early termination condition
			#should_stop = (i >= 1000) & (s.best_fitness < 6.0)
			should_stop = False


   
			
			# Print message when condition is met
			# Compute pairwise distances in JAX (flatten per individual)
			params = data["parameters"]

   
   
			pop_size = params.shape[0]
			flat_params = jnp.reshape(params, (pop_size, -1))
   
   
	   
			# Compute per-individual statistics (for each individual: mean, min, max, var of their weights)
			individual_means = jnp.mean(flat_params, axis=1)  # (pop_size,) - mean weight per individual
			individual_mins = jnp.min(flat_params, axis=1)     # (pop_size,) - min weight per individual
			individual_maxs = jnp.max(flat_params, axis=1)     # (pop_size,) - max weight per individual
			individual_vars = jnp.var(flat_params, axis=1)     # (pop_size,) - var of weights per individual
			
			# Compute statistics across population for each per-individual metric
			# Mean across population
			mean_individual_mean = jnp.mean(individual_means)
			mean_individual_min = jnp.mean(individual_mins)
			mean_individual_max = jnp.mean(individual_maxs)
			mean_individual_var = jnp.mean(individual_vars)
			
			# Variance across population (measures how much individuals differ in these metrics)
			var_individual_mean = jnp.var(individual_means)
			var_individual_min = jnp.var(individual_mins)
			var_individual_max = jnp.var(individual_maxs)
			var_individual_var = jnp.var(individual_vars)
			
			# Distribution shape statistics for each individual
			# Normalized data for each individual (centered and scaled)
			individual_stds = jnp.std(flat_params, axis=1, keepdims=True) + 1e-8  # (pop_size, 1)
			normalized_params = (flat_params - individual_means[:, None]) / individual_stds  # (pop_size, num_params)
			
			# Skewness (3rd moment) - measures asymmetry, ~0 for normal, high for heavy-tailed
			# skewness = E[(X-μ)³] / σ³
			individual_skewness = jnp.mean(normalized_params ** 3, axis=1)  # (pop_size,)
			
			# Kurtosis (4th moment) - measures tail heaviness, ~0 for normal, >0 for heavy tails
			# kurtosis = E[(X-μ)⁴] / σ⁴ - 3 (excess kurtosis)
			individual_kurtosis = jnp.mean(normalized_params ** 4, axis=1) - 3.0  # (pop_size,)
			
			# Uniformity metrics
			# For uniform distribution: min and max should be at bounds, variance should be (max-min)²/12
			# Check how close variance is to expected uniform variance
			individual_ranges = individual_maxs - individual_mins  # (pop_size,)
			expected_uniform_var = (individual_ranges ** 2) / 12.0  # (pop_size,)
			uniformity_ratio = individual_vars / (expected_uniform_var + 1e-8)  # close to 1.0 for uniform
			
			# Tail heaviness indicators (for power law detection)
			# Check quantile ratios: heavy tails have large ratios between extreme quantiles
			# Use quantile with [0,1] range (0.99 = 99th percentile)
			q99_individual = jnp.quantile(flat_params, 0.99, axis=1)  # 99th percentile per individual
			q95_individual = jnp.quantile(flat_params, 0.95, axis=1)  # 95th percentile per individual
			q5_individual = jnp.quantile(flat_params, 0.05, axis=1)   # 5th percentile per individual
			q1_individual = jnp.quantile(flat_params, 0.01, axis=1)    # 1st percentile per individual
			upper_tail_ratio = (q99_individual - q95_individual) / (individual_stds[:, 0] + 1e-8)  # upper tail
			lower_tail_ratio = (q5_individual - q1_individual) / (individual_stds[:, 0] + 1e-8)     # lower tail
			
			# Population-level statistics for distribution metrics
			mean_skewness = jnp.mean(individual_skewness)
			mean_kurtosis = jnp.mean(individual_kurtosis)
			mean_uniformity_ratio = jnp.mean(uniformity_ratio)
			mean_upper_tail_ratio = jnp.mean(upper_tail_ratio)
			mean_lower_tail_ratio = jnp.mean(lower_tail_ratio)
			
			# Keep old global stats for backwards compatibility
			flat_mean = jnp.mean(flat_params)
			flat_min = jnp.min(flat_params)
			flat_max = jnp.max(flat_params)
			flat_var = jnp.var(flat_params)
			temp_min = jnp.min(flat_params, axis=1, keepdims=True)
			temp_max = jnp.max(flat_params, axis=1, keepdims=True)
			flat_params = (flat_params - temp_min) / (temp_max - temp_min)

			diffs = flat_params[:, None, :] - flat_params[None, :, :]
			pairwise_dists = jnp.linalg.norm(diffs, axis=-1)
			iu0, iu1 = jnp.triu_indices(pop_size, k=1)
			distances = pairwise_dists[iu0, iu1]
			diversity = jnp.mean(distances)
			self.logger.log(s, data, task_params, noise, diversity=diversity, current_gravity=0.0, 
						 flat_mean=flat_mean, flat_min=flat_min, flat_max=flat_max, flat_var=flat_var,
						 mean_individual_mean=mean_individual_mean, mean_individual_min=mean_individual_min,
						 mean_individual_max=mean_individual_max, mean_individual_var=mean_individual_var,
						 var_individual_mean=var_individual_mean, var_individual_min=var_individual_min,
						 var_individual_max=var_individual_max, var_individual_var=var_individual_var,
						 mean_skewness=mean_skewness, mean_kurtosis=mean_kurtosis,
						 mean_uniformity_ratio=mean_uniformity_ratio,
       behaviroral_diversity=data["behaviroral_diversity"],
						 mean_upper_tail_ratio=mean_upper_tail_ratio, mean_lower_tail_ratio=mean_lower_tail_ratio,
						 individual_skewness=individual_skewness, individual_kurtosis=individual_kurtosis,
						 uniformity_ratio=uniformity_ratio, flat_params_for_testing=flat_params)

			return [s, k, task_params, should_stop, env_state, data["parameters"], data["fitness"]]




		if self.progress_bar:
			_step = progress_bar_fori(self.train_steps)(_step) #type: ignore

		task_params_init = 0
		#keys = jr.split(key, 10)
		num_tasks = int(self.train_steps / self.continual_config["perturbe_every_n_gens"]) + 50
		total_noise = jax.random.normal(key, (num_tasks,self.obs_size))*self.continual_config["noise_range"]

		# Use scan with early termination support
		
		def _step_with_early_stop(carry, x):
			state, key, task_params, should_stop, env_state = carry
			generation = x
			current_task = generation // self.continual_config["perturbe_every_n_gens"]
			#current_task = 0 
			noise = total_noise[current_task, :]

			#jax.debug.print("noise: {}", noise)
			#noise = jax.numpy.where(generation % self.perturbe_every_n_gens == 0, new_noise, noise)

			# If we should stop, return current state without training
			"""
			new_carry = jax.lax.cond(
				should_stop,
				lambda: [state, key, task_params, should_stop, env_state],  # Keep stopped state
				lambda: _step(generation, (state, key, task_params, env_state))  # Continue training
			)
			"""
			new_state, new_key, new_task_params, new_should_stop, new_env_state, parameters, fitnesses  = _step(generation, (state, key, task_params, env_state, noise))
			
			# Extract the should_stop flag from the training step result
			#_, _, _, new_should_stop, env_state, parameters, fitnesses = new_carry

			new_carry = [new_state, new_key, new_task_params, new_should_stop, new_env_state]
			
			# Return (carry, output) pair as required by scan
			return new_carry, (None, None)

		# Run scan with early termination
		# Use task's env_state if init_env_state is not provided, otherwise use the provided one
		initial_env_state = current_init_env_state if init_env_state is None else init_env_state
		(state, key, task_params, _, _), (archive_history, fitnesses_history) = jax.lax.scan(
			_step_with_early_stop, 
			[state, key, task_params_init, False, initial_env_state],  # Use list to match return type
			jnp.arange(self.train_steps)
		)
		return (state, total_noise, None, fitnesses_history)


	def train_brax_(self, state: TrainState, key: jax.Array, data: Optional[Data]=None, init_env_state: Optional=None)->TrainState:
		"""
		Modified training function that uses 10 phases instead of one big loop.
		Each phase samples a new gravity value, updates the XML file, and creates a new environment.
		"""
		from methods.brax_wrapper.continual_utils import recreate_environment_with_gravity
		
		# Define phase parameters
		num_phases = 1
		steps_per_phase = self.train_steps // num_phases
		print("steps_per_phase", steps_per_phase)
		gravity_range = (0.2, 4.0)  # Gravity multiplier range
		
		# Initialize phase variables
		task_params_init = 0
		noise = jax.random.normal(key, (self.obs_size,)) * self.noise_range
		#noise = 0*-1.021
        
		current_gravity = 1.0  # Start with normal gravity
		
		# Store the original environment for reference
		# Access environment through task if available
		original_env = self.task_keep.env
  
		print("steps_per_phase", steps_per_phase)

		

		

		
		# Run training in 10 phases
		for phase in range(num_phases):
      
			def _step(i, c):
				s, k, task_params, env_state, noise = c
				k, k_ = jr.split(k)
				s, data, task_params, env_state = self.train_step(s, k_, task_params, i, env_state, noise)
				
				def save_params(data):
					for dev_step in range(self.logger.dev_steps + 1):
						current_dev = jax.tree_map(lambda x: x[data["best_indiv"], 0, dev_step, ...],
												data["interm_policies"])
						self.logger.save_chkpt(current_dev, task_params, jnp.array(dev_step))
     
     

				jax.lax.cond(s.best_fitness >= self.reward_for_solved,
												lambda x: save_params(x), lambda x: None, data)

				# Check early termination condition
				#should_stop = (i >= 1000) & (s.best_fitness < 6.0)
				should_stop = False


	
				
				# Print message when condition is met
				# Compute pairwise distances in JAX (flatten per individual)
				params = data["parameters"]

	
	
				pop_size = params.shape[0]
				flat_params = jnp.reshape(params, (pop_size, -1))
	
	
		
				# Compute per-individual statistics (for each individual: mean, min, max, var of their weights)
				individual_means = jnp.mean(flat_params, axis=1)  # (pop_size,) - mean weight per individual
				individual_mins = jnp.min(flat_params, axis=1)     # (pop_size,) - min weight per individual
				individual_maxs = jnp.max(flat_params, axis=1)     # (pop_size,) - max weight per individual
				individual_vars = jnp.var(flat_params, axis=1)     # (pop_size,) - var of weights per individual
				
				# Compute statistics across population for each per-individual metric
				# Mean across population
				mean_individual_mean = jnp.mean(individual_means)
				mean_individual_min = jnp.mean(individual_mins)
				mean_individual_max = jnp.mean(individual_maxs)
				mean_individual_var = jnp.mean(individual_vars)
				
				# Variance across population (measures how much individuals differ in these metrics)
				var_individual_mean = jnp.var(individual_means)
				var_individual_min = jnp.var(individual_mins)
				var_individual_max = jnp.var(individual_maxs)
				var_individual_var = jnp.var(individual_vars)
				
				# Distribution shape statistics for each individual
				# Normalized data for each individual (centered and scaled)
				individual_stds = jnp.std(flat_params, axis=1, keepdims=True) + 1e-8  # (pop_size, 1)
				normalized_params = (flat_params - individual_means[:, None]) / individual_stds  # (pop_size, num_params)
				
				# Skewness (3rd moment) - measures asymmetry, ~0 for normal, high for heavy-tailed
				# skewness = E[(X-μ)³] / σ³
				individual_skewness = jnp.mean(normalized_params ** 3, axis=1)  # (pop_size,)
				
				# Kurtosis (4th moment) - measures tail heaviness, ~0 for normal, >0 for heavy tails
				# kurtosis = E[(X-μ)⁴] / σ⁴ - 3 (excess kurtosis)
				individual_kurtosis = jnp.mean(normalized_params ** 4, axis=1) - 3.0  # (pop_size,)
				
				# Uniformity metrics
				# For uniform distribution: min and max should be at bounds, variance should be (max-min)²/12
				# Check how close variance is to expected uniform variance
				individual_ranges = individual_maxs - individual_mins  # (pop_size,)
				expected_uniform_var = (individual_ranges ** 2) / 12.0  # (pop_size,)
				uniformity_ratio = individual_vars / (expected_uniform_var + 1e-8)  # close to 1.0 for uniform
				
				# Tail heaviness indicators (for power law detection)
				# Check quantile ratios: heavy tails have large ratios between extreme quantiles
				# Use quantile with [0,1] range (0.99 = 99th percentile)
				q99_individual = jnp.quantile(flat_params, 0.99, axis=1)  # 99th percentile per individual
				q95_individual = jnp.quantile(flat_params, 0.95, axis=1)  # 95th percentile per individual
				q5_individual = jnp.quantile(flat_params, 0.05, axis=1)   # 5th percentile per individual
				q1_individual = jnp.quantile(flat_params, 0.01, axis=1)    # 1st percentile per individual
				upper_tail_ratio = (q99_individual - q95_individual) / (individual_stds[:, 0] + 1e-8)  # upper tail
				lower_tail_ratio = (q5_individual - q1_individual) / (individual_stds[:, 0] + 1e-8)     # lower tail
				
				# Population-level statistics for distribution metrics
				mean_skewness = jnp.mean(individual_skewness)
				mean_kurtosis = jnp.mean(individual_kurtosis)
				mean_uniformity_ratio = jnp.mean(uniformity_ratio)
				mean_upper_tail_ratio = jnp.mean(upper_tail_ratio)
				mean_lower_tail_ratio = jnp.mean(lower_tail_ratio)
				
				# Keep old global stats for backwards compatibility
				flat_mean = jnp.mean(flat_params)
				flat_min = jnp.min(flat_params)
				flat_max = jnp.max(flat_params)
				flat_var = jnp.var(flat_params)

				diffs = flat_params[:, None, :] - flat_params[None, :, :]
				pairwise_dists = jnp.linalg.norm(diffs, axis=-1)
				iu0, iu1 = jnp.triu_indices(pop_size, k=1)
				distances = pairwise_dists[iu0, iu1]
				diversity = jnp.mean(distances)
				self.logger.log(s, data, task_params, noise, diversity=diversity, current_gravity=current_gravity, 
							flat_mean=flat_mean, flat_min=flat_min, flat_max=flat_max, flat_var=flat_var,
							mean_individual_mean=mean_individual_mean, mean_individual_min=mean_individual_min,
							mean_individual_max=mean_individual_max, mean_individual_var=mean_individual_var,
							var_individual_mean=var_individual_mean, var_individual_min=var_individual_min,
							var_individual_max=var_individual_max, var_individual_var=var_individual_var,
							mean_skewness=mean_skewness, mean_kurtosis=mean_kurtosis,
							mean_uniformity_ratio=mean_uniformity_ratio,
							mean_upper_tail_ratio=mean_upper_tail_ratio, mean_lower_tail_ratio=mean_lower_tail_ratio,
							individual_skewness=individual_skewness, individual_kurtosis=individual_kurtosis,
							uniformity_ratio=uniformity_ratio, flat_params_for_testing=flat_params)

				return [s, k, task_params, should_stop, env_state, data["parameters"], data["fitness"]]
		# Sample new gravity value for this phase
		phase_key, key = jr.split(key)
		"""
		current_gravity = jax.random.uniform(
			phase_key, (), 
			minval=gravity_range[0], 
			maxval=gravity_range[1]
		)
			#current_gravity = 1.0
			
			# Recreate environment with new gravity (non-JIT operation)
			# This is the non-JIT part - environment recreation
			env, modified_xml_path = recreate_environment_with_gravity(
				env_name=self.task_keep.env_name,
				params=self.task_keep.env_params,
				gravity_multiplier=current_gravity, 
				xml_file=original_env.robot.xml_file,
				save_file=True
			)
			"""
		# Update the environment in the task
		#self.task.env = env

		
		# Log the gravity change
		#print(f"Phase {phase + 1}/{num_phases}: Changed gravity to {current_gravity:.2f}x normal gravity")
		#rint(f"Modified XML saved to: {modified_xml_path}")
		
		# Calculate steps for this phase
		phase_start = phase * steps_per_phase
		phase_end = (phase + 1) * steps_per_phase if phase < num_phases - 1 else self.train_steps
		phase_steps = phase_end - phase_start
		
		# Run training for this phase
		def _step_with_early_stop(carry, x):
			state, key, task_params, should_stop, env_state = carry
			generation = x

			#noise = jax.numpy.where(generation % self.perturbe_every_n_gens == 0, new_noise, noise)
			
			new_state, new_key, new_task_params, new_should_stop, new_env_state, parameters, fitnesses  = _step(generation, (state, key, task_params, env_state, 0))
		
			# Extract the should_stop flag from the training step result
			#_, _, _, new_should_stop, env_state, parameters, fitnesses = new_carry

			new_carry = [new_state, new_key, new_task_params, new_should_stop, new_env_state]
			
			# Return (carry, output) pair as required by scan
			return new_carry, (None, fitnesses)
			
		
		# Run scan with early termination
		(state, key, task_params, _, _), (archive_history, fitnesses_history) = jax.lax.scan(
			_step_with_early_stop, 
			[state, key, task_params_init, False, init_env_state],  # Use list to match return type
			jnp.arange(steps_per_phase)
		)
		return (state, jnp.zeros((num_phases, self.obs_size)), None, fitnesses_history)




	def train_kinetixcont(self, state: TrainState, key: jax.Array, data: Optional[Data]=None, init_env_state: Optional=None)->TrainState:
		"""
		Modified training function that uses multiple phases.
		Each phase uses a different environment selected via task_params.
		"""
		# Define phase parameters
		env_names = [
            # Medium (m) environments - more challenging tasks
            "m/h0_unicycle",
            "m/h1_car_left",
            "m/h2_car_right", 
            "m/h3_car_thrust",
            "m/h4_thrust_the_needle",
            "m/h5_angry_birds",
            "m/h6_thrust_over",
            "m/h7_car_flip",
            "m/h8_weird_vehicle",
            "m/h9_spin_the_right_way",
            "m/h10_thrust_right_easy",
            "m/h11_thrust_left_easy",
            "m/h12_thrustfall_left",
            "m/h13_thrustfall_right",
            "m/h14_thrustblock",
            "m/h15_thrustshoot",
            "m/h16_thrustcontrol_right",
            "m/h17_thrustcontrol_left",
            "m/h18_thrust_right_very_easy",
            "m/h19_thrust_left_very_easy",
        ]
		#env_names = ["m/h0_unicycle", "l/lever_puzzle"]
		num_phases = len(env_names)
		steps_per_phase = self.train_steps // num_phases
		
		# Create all environments before the loop
		envs = []
		env_params_list = []
		static_env_params_list = []
		init_env_states_list = []
		
		with open("scripts/train/evosax/kinetix_config_pixels.yaml", "r") as f:
			config = yaml.load(f, Loader=yaml.SafeLoader)
		config = normalise_config(config, name="PPO")
		
		for env_name in env_names:
			print(f"Loading Kinetix env: {env_name}")
			env_state, static_env_params, env_params = load_from_json_file(env_name)
			config["env_params"] = to_state_dict(env_params)
			config["static_env_params"] = to_state_dict(static_env_params)
			
			reset_fn = make_reset_fn_from_config(config, env_params, static_env_params)
			env_obj = make_kinetix_env(
				observation_type=config["observation_type"],
				action_type=config["action_type"],
				reset_fn=reset_fn,
				env_params=env_params,
				static_env_params=static_env_params)
			
			envs.append(env_obj)
			env_params_list.append(env_params)
			static_env_params_list.append(static_env_params)
			init_env_states_list.append(env_state)
		
		noise = jax.random.normal(key, (self.obs_size,)) * 0.0

		for phase in range(num_phases):
			# Set task_params to the current phase index
			task_params_init = phase
			# Get environment for this phase
			current_env = envs[phase]
			current_env_params = env_params_list[phase]
			current_static_env_params = static_env_params_list[phase]
			current_init_env_state = init_env_states_list[phase]
			
			def _step(i, c):
				s, k, task_params, env_state= c
				k, k_ = jr.split(k)
				#if self.logger is not None:
				#	dummy_data = {"fitness": jnp.array([0]), "interm_policies": [], "best_indiv": []}
				s, data, task_params, env_state = self.train_step(
					s, k_, task_params, i, env_state, noise,
					env=current_env,
					env_params=current_env_params,
					static_env_params=current_static_env_params,
					init_env_state=current_init_env_state
				)
	


				def save_params(data):
					for dev_step in range(self.logger.dev_steps + 1):
						current_dev = jax.tree_map(lambda x: x[data["best_indiv"], 0, dev_step, ...],
												data["interm_policies"])
						self.logger.save_chkpt(current_dev, task_params, jnp.array(dev_step))
		
		

				jax.lax.cond(state.best_fitness >= self.reward_for_solved,
												lambda x: save_params(x), lambda x: None, data)

				# Check early termination condition
				#should_stop = (i >= 1000) & (s.best_fitness < 6.0)
				should_stop = False


	
				
				# Print message when condition is met
				# Compute pairwise distances in JAX (flatten per individual)
				params = data["parameters"]

	
	
				pop_size = params.shape[0]
				flat_params = jnp.reshape(params, (pop_size, -1))
	
	
		
				# Compute per-individual statistics (for each individual: mean, min, max, var of their weights)
				individual_means = jnp.mean(flat_params, axis=1)  # (pop_size,) - mean weight per individual
				individual_mins = jnp.min(flat_params, axis=1)     # (pop_size,) - min weight per individual
				individual_maxs = jnp.max(flat_params, axis=1)     # (pop_size,) - max weight per individual
				individual_vars = jnp.var(flat_params, axis=1)     # (pop_size,) - var of weights per individual
				
				# Compute statistics across population for each per-individual metric
				# Mean across population
				mean_individual_mean = jnp.mean(individual_means)
				mean_individual_min = jnp.mean(individual_mins)
				mean_individual_max = jnp.mean(individual_maxs)
				mean_individual_var = jnp.mean(individual_vars)
				
				# Variance across population (measures how much individuals differ in these metrics)
				var_individual_mean = jnp.var(individual_means)
				var_individual_min = jnp.var(individual_mins)
				var_individual_max = jnp.var(individual_maxs)
				var_individual_var = jnp.var(individual_vars)
				
				# Distribution shape statistics for each individual
				# Normalized data for each individual (centered and scaled)
				individual_stds = jnp.std(flat_params, axis=1, keepdims=True) + 1e-8  # (pop_size, 1)
				normalized_params = (flat_params - individual_means[:, None]) / individual_stds  # (pop_size, num_params)
				
				# Skewness (3rd moment) - measures asymmetry, ~0 for normal, high for heavy-tailed
				# skewness = E[(X-μ)³] / σ³
				individual_skewness = jnp.mean(normalized_params ** 3, axis=1)  # (pop_size,)
				
				# Kurtosis (4th moment) - measures tail heaviness, ~0 for normal, >0 for heavy tails
				# kurtosis = E[(X-μ)⁴] / σ⁴ - 3 (excess kurtosis)
				individual_kurtosis = jnp.mean(normalized_params ** 4, axis=1) - 3.0  # (pop_size,)
				
				# Uniformity metrics
				# For uniform distribution: min and max should be at bounds, variance should be (max-min)²/12
				# Check how close variance is to expected uniform variance
				individual_ranges = individual_maxs - individual_mins  # (pop_size,)
				expected_uniform_var = (individual_ranges ** 2) / 12.0  # (pop_size,)
				uniformity_ratio = individual_vars / (expected_uniform_var + 1e-8)  # close to 1.0 for uniform
				
				# Tail heaviness indicators (for power law detection)
				# Check quantile ratios: heavy tails have large ratios between extreme quantiles
				# Use quantile with [0,1] range (0.99 = 99th percentile)
				q99_individual = jnp.quantile(flat_params, 0.99, axis=1)  # 99th percentile per individual
				q95_individual = jnp.quantile(flat_params, 0.95, axis=1)  # 95th percentile per individual
				q5_individual = jnp.quantile(flat_params, 0.05, axis=1)   # 5th percentile per individual
				q1_individual = jnp.quantile(flat_params, 0.01, axis=1)    # 1st percentile per individual
				upper_tail_ratio = (q99_individual - q95_individual) / (individual_stds[:, 0] + 1e-8)  # upper tail
				lower_tail_ratio = (q5_individual - q1_individual) / (individual_stds[:, 0] + 1e-8)     # lower tail
				
				# Population-level statistics for distribution metrics
				mean_skewness = jnp.mean(individual_skewness)
				mean_kurtosis = jnp.mean(individual_kurtosis)
				mean_uniformity_ratio = jnp.mean(uniformity_ratio)
				mean_upper_tail_ratio = jnp.mean(upper_tail_ratio)
				mean_lower_tail_ratio = jnp.mean(lower_tail_ratio)
				
				# Keep old global stats for backwards compatibility
				flat_mean = jnp.mean(flat_params)
				flat_min = jnp.min(flat_params)
				flat_max = jnp.max(flat_params)
				flat_var = jnp.var(flat_params)
				temp_min = jnp.min(flat_params, axis=1, keepdims=True)
				temp_max = jnp.max(flat_params, axis=1, keepdims=True)
				flat_params = (flat_params - temp_min) / (temp_max - temp_min)

				diffs = flat_params[:, None, :] - flat_params[None, :, :]
				pairwise_dists = jnp.linalg.norm(diffs, axis=-1)
				iu0, iu1 = jnp.triu_indices(pop_size, k=1)
				distances = pairwise_dists[iu0, iu1]
				diversity = jnp.mean(distances)
				self.logger.log(s, data, task_params, noise, diversity=diversity, current_gravity=0.0, 
							flat_mean=flat_mean, flat_min=flat_min, flat_max=flat_max, flat_var=flat_var,
							mean_individual_mean=mean_individual_mean, mean_individual_min=mean_individual_min,
							mean_individual_max=mean_individual_max, mean_individual_var=mean_individual_var,
							var_individual_mean=var_individual_mean, var_individual_min=var_individual_min,
							var_individual_max=var_individual_max, var_individual_var=var_individual_var,
							mean_skewness=mean_skewness, mean_kurtosis=mean_kurtosis,
							mean_uniformity_ratio=mean_uniformity_ratio,
		behaviroral_diversity=data["behaviroral_diversity"],
							mean_upper_tail_ratio=mean_upper_tail_ratio, mean_lower_tail_ratio=mean_lower_tail_ratio,
							individual_skewness=individual_skewness, individual_kurtosis=individual_kurtosis,
							uniformity_ratio=uniformity_ratio, flat_params_for_testing=flat_params)

				return [s, k, task_params, should_stop, env_state, data["parameters"], data["fitness"]]


      
			


			# Calculate steps for this phase
			phase_start = phase * steps_per_phase
			phase_end = (phase + 1) * steps_per_phase if phase < num_phases - 1 else self.train_steps
			phase_steps = phase_end - phase_start
   
   
			def _step_with_early_stop(carry, x):
				state, key, task_params, should_stop, env_state = carry
				generation = x
				# Keep task_params as the phase index (it's already set to phase)
				noise = jnp.array([0.0])

				new_state, new_key, new_task_params, new_should_stop, new_env_state, parameters, fitnesses  = _step(generation, (state, key, task_params, env_state))
				
				# Extract the should_stop flag from the training step result
				#_, _, _, new_should_stop, env_state, parameters, fitnesses = new_carry

				new_carry = [new_state, new_key, new_task_params, new_should_stop, new_env_state]
				
				# Return (carry, output) pair as required by scan
				return new_carry, (None, None)

			# Run scan with early termination
			# Use phase-specific init_env_state for each phase
			# Ensure we always have a valid env_state
			phase_init_env_state = current_init_env_state  # Use current phase's init_env_state
			(state, key, task_params, _, _), (archive_history, fitnesses_history) = jax.lax.scan(
				_step_with_early_stop, 
				[state, key, task_params_init, False, phase_init_env_state],  # Use list to match return type
				jnp.arange(steps_per_phase)
			)
		total_noise = jnp.zeros((num_phases, self.obs_size))  # Dummy noise for compatibility
		return (state, total_noise, None, fitnesses_history)

	#-------------------------------------------------------------------

	def log(self, data):
		hcb.id_tap(
			lambda d, *_: wandb.log(d), data
		)

	#-------------------------------------------------------------------

	def init_and_train(self, key: jr.PRNGKey, data: Optional[Data]=None, init_env_state: Optional=None)->Tuple[TrainState, Data]:
		init_key, train_key = jr.split(key)
		state = self.initialize(init_key)
		return self.train(state, train_key, data, init_env_state)

	#-------------------------------------------------------------------

	def init_and_train_(self, key: jr.PRNGKey, data: Optional[Data]=None, init_env_state: Optional=None)->TrainState:
		init_key, train_key = jr.split(key)
		state = self.initialize(init_key)
		return self.train_(state, train_key, data, init_env_state)

	#-------------------------------------------------------------------

	def train_step(self, state: TrainState, key: jr.PRNGKey, data: Optional[Data]=None)->Tuple[TrainState, Any]:
		raise NotImplementedError

	#-------------------------------------------------------------------

	def initialize(self, key: jr.PRNGKey)->TrainState:
		raise NotImplementedError

	#-------------------------------------------------------------------

	def train_from_model_ckpt(self, ckpt_path: str, key: Optional[jax.Array]=None)->Tuple[TrainState, Data]:
		raise NotImplementedError

	#-------------------------------------------------------------------

	def train_from_model_ckpt_(self, ckpt_path: str, key: Optional[jax.Array]=None)->Tuple[TrainState, Data]:
		raise NotImplementedError

	#-------------------------------------------------------------------

	def train_from_training_ckpt(self, ckpt_path: str, key: Optional[jax.Array]=None)->Tuple[TrainState, Data]:
		raise NotImplementedError

	#-------------------------------------------------------------------

	def train_from_training_ckpt_(self, ckpt_path: str, key: Optional[jax.Array]=None)->Tuple[TrainState, Data]:
		raise NotImplementedError




class BaseMultiTrainer(eqx.Module):
	
	"""
	"""
	#-------------------------------------------------------------------
	trainers: list[BaseTrainer]
	transform_fns: Union[list[Callable[[TrainState, TrainState], TrainState]], 
						 Callable[[TrainState, TrainState], TrainState]]
	#-------------------------------------------------------------------

	def __init__(self, 
				 trainers: list[BaseTrainer], 
				 transform_fns: Union[list[Callable[[TrainState, TrainState], TrainState]], 
				 					  Callable[[TrainState, TrainState], TrainState]]):
		
		self.trainers = trainers
		self.transform_fns = transform_fns

	#-------------------------------------------------------------------

	def init_and_train_(self, key: jax.Array):
		
		for i, trainer in enumerate(self.trainers):
			key, key_init, key_train = jr.split(key, 3)
			new_train_state = trainer.initialize(key_init)
			if i:
				tf = self.transform_fns[i-1] if isinstance(self.transform_fns, list) else self.transform_fns 
				train_state = tf(train_state, new_train_state)
			else:
				train_state = new_train_state
			train_state = trainer.train_(train_state, key_train)

		return train_state

