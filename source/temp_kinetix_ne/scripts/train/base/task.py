import jax
import os
from scripts.train.base.utils import max_rewards
import numpy as onp
import numpy as np
import jax.numpy as jnp
from kinetix.render import make_render_pixels
import matplotlib.pyplot as plt
from PIL import Image
import io



class Task:

    def __init__(self, env, config):
        self.env = env
        self.config = config

        if config["env_config"]["curriculum"]:
            self.num_tasks = env.num_tasks
            self.current_task = env.current_task
        else:
            self.num_tasks = 1
            self.current_task = 0

        if config["env_config"]["env_type"] == "stepping_gates":
            if config["optimizer_config"]["optimizer_type"] == "tensorneat":
                self.num_eval_trials = 1
            elif config["optimizer_config"]["optimizer_type"] == "evosax":
                self.num_eval_trials = 1
            elif config["optimizer_config"]["optimizer_type"] == "brax":
                self.num_eval_trials = 2

        elif config["env_config"]["env_type"] == "brax":
            self.num_eval_trials = 2

        elif config["env_config"]["env_type"] == "gymnax":
            self.num_eval_trials = 10

        elif config["env_config"]["env_type"] == "pattern_match":
            self.num_eval_trials = 1
            
        elif config["env_config"]["env_type"] == "kinetix":
            self.num_eval_trials = 100
            
        self.env_type = config["env_config"]["env_type"]

        self.eval_info = {}
        
        
    def get_input_ouput(self, task):
        obs_size = self.config["env_config"]["observation_size"]
        action_size = self.config["env_config"]["action_size"]

            
        return obs_size, action_size
    
    def create_gif_from_frames(self, frames, task, eval_trial):
        """Create a GIF from a list of frames and save it."""
        if not frames:
            return
            
        # Convert frames to PIL Images
        pil_frames = []
        for frame in frames:
            # Convert JAX array to numpy array, then to PIL Image
            frame_np = np.array(frame)
            pil_image = Image.fromarray(frame_np)
            pil_frames.append(pil_image)
        
        # Create GIF filename
        gif_filename = self.config["exp_config"]["trial_dir"] + f"/task_{task}_trial_{eval_trial}.gif"
        
        # Save as GIF with duration (in milliseconds)
        pil_frames[0].save(
            gif_filename,
            save_all=True,
            append_images=pil_frames[1:],
            duration=100,  # 100ms per frame (10 FPS)
            loop=0  # Infinite loop
        )
        
        print(f"GIF saved as: {gif_filename}")
    
        
    def run_eval_trial_kinetix(self, env, task, eval_trial, act_fn, obs_size, action_size, for_eval):
        trial_rewards = []
        trial_success = []
        jit_env_reset = jax.jit(env.reset )
        jit_env_step = jax.jit(env.step)
        
       
        renderer = make_render_pixels(for_eval["env_params"], for_eval["static_env_params"])
        
        # List to store frames for GIF creation
        frames = []
        
        rng = jax.random.PRNGKey(seed=eval_trial)
        obs, state = env.reset(rng, env_params=for_eval["env_params"], override_reset_state=for_eval["env_state"])
        cum_reward = 0
        infos = []
        actions = []
        states = [] 
        success = 0
        episode_length = 0
        rewards = []
        done = jnp.zeros(1, dtype=jnp.bool_)

        policy_state = None

        for step in range(self.config["env_config"]["episode_length"]):
            # Render current frame and add to frames list
            pixels = renderer(state)
            frame = pixels.astype(jnp.uint8).transpose(1, 0, 2)[::-1]
            frames.append(frame)

            prev_obs = obs
            
            rng, _key = jax.random.split(rng)

            act_rng, rng = jax.random.split(rng)
            
            if step:
                action, policy_state = act_fn(obs=obs, state=policy_state, done=done, key=_key
                                              ) 
            else:
                action, policy_state = act_fn(obs=obs,  done=done, key=_key)
                

            


            obs, state, reward, done, _ = jit_env_step(act_rng, state, action, env_params=for_eval["env_params"])
            done = jnp.expand_dims(done, axis=0)


            cum_reward += float(reward)
            rewards.append(reward)
            actions.append(action)

            episode_length += 1

            if done:
                break
            
        print(rewards)
        print(episode_length)
        
        # Create GIF from collected frames  
        if frames:
            self.create_gif_from_frames(frames, task, eval_trial)
        
        trial_success = int(cum_reward > 1)
        trial_rewards = float(cum_reward)
        
        return trial_rewards, trial_success, episode_length
    
    def run_eval_trial_gymnax(self, env, task, eval_trial, act_fn, obs_size, action_size):
        trial_rewards = []
        trial_success = []
        jit_env_reset = jax.jit(env.reset)
        jit_env_step = jax.jit(env.step)
        
        rng = jax.random.PRNGKey(seed=eval_trial)
        obs, state = jit_env_reset(rng, jax.numpy.array([task]))
        cum_reward = 0
        infos = []
        actions = []
        states = []
        success = 0
        episode_length = 0

        for step in range(self.config["env_config"]["episode_length"]):

            prev_obs = obs

            act_rng, rng = jax.random.split(rng)
            
            
            if self.config["optimizer_config"]["optimizer_name"] == "ppo":
                act = act_fn(prev_obs)
            else:
                act = act_fn(prev_obs, action_size=action_size, obs_size=obs_size)


            #act = act_fn(prev_obs, action_size=action_size, obs_size=obs_size)
            if isinstance(act, tuple):
                act, info = act
                infos.append(info)
            act = jnp.argmax(act)

            obs, state, reward, done, _ = jit_env_step(act_rng, state, act, self.config["env_config"]["gymnax_env_params"])

            cum_reward += float(reward)
            actions.append(act)
            if reward == max_rewards[self.config["env_config"]["env_name"]]:
                success += 1
            episode_length += 1

            if done:
                break
        trial_success = int(success > 0)
        trial_rewards = float(cum_reward)
        
        return trial_rewards, trial_success, episode_length    
    
        
    def run_eval_trial_brax(self, env,task, eval_trial, act_fn, obs_size, action_size):
        trial_rewards = []
        trial_success = []
        jit_env_reset = jax.jit(env.reset)
        jit_env_step = jax.jit(env.step)
        
        jit_env_reset = jax.jit(env.reset)
        jit_env_step = jax.jit(env.step)
        rng = jax.random.PRNGKey(seed=eval_trial)
        saving_dir = self.config["exp_config"]["trial_dir"] + "/visuals/eval/trajs"

        for eval_trial in range(self.num_eval_trials):

            state = jit_env_reset(rng)
            cum_reward = 0
            infos = []
            actions = []
            states = []
            success = 0
            episode_length = 0

            for step in range(self.config["env_config"]["episode_length"]):

                prev_obs = state.obs

                act_rng, rng = jax.random.split(rng)

                if self.config["optimizer_config"]["optimizer_name"] == "ppo":
                    act = act_fn(prev_obs)

                else:
                    act = act_fn(prev_obs, action_size=action_size, obs_size=obs_size)

                if isinstance(act, tuple):
                    act, info = act
                    infos.append(info)

                state = jit_env_step(state, act)
                reward = state.reward
                done = state.done

                cum_reward += float(reward)
                actions.append(act)
                if reward == max_rewards[self.config["env_config"]["env_name"]]:
                    success += 1
                episode_length += 1

                states.append(state.pipeline_state)
                if done:
                    break
            trial_success.append(float(success / episode_length))
            trial_rewards.append(float(cum_reward))

            gif_path = self.env.show_rollout(states, save_dir=saving_dir + "/task_" + str(task),
                                                    filename="eval_trial_" + str(eval_trial)  + "_rew_" + str(cum_reward))

        task_alias = "task_" + str(task)
        if final_policy:
            task_alias += "_final_policy"

        self.eval_info[task_alias] = {"rewards": [float(el) for el in trial_rewards],
                                      "success": [float(el) for el in trial_success]}
        if not final_policy:
            self.eval_info[task_alias]["gens"] = gens
            
        return trial_rewards, trial_success  
    
    def run_eval_trial(self, env, task, eval_trial, act_fn, obs_size, action_size, for_eval):
        if self.env_type == "gymnax":
            return self.run_eval_trial_gymnax(env, task, eval_trial, act_fn, obs_size, action_size)
        elif self.env_type == "kinetix":
            return self.run_eval_trial_kinetix(env, task, eval_trial, act_fn, obs_size, action_size, for_eval)
        #else:
        #    return self.run_eval_trial_brax(task=task, env=env, eval_trial=eval_trial, act_fn=act_fn, obs_size=obs_size, action_size=action_size)
        
        
        
        
        
        

    def run_eval(self, act_fn, saving_dir, tasks, gens=None, final_policy=False, for_eval=None):
        env = self.env

        for task in tasks:
            


            if not os.path.exists(saving_dir + "/task_" + str(task)):
                os.makedirs(saving_dir + "/task_" + str(task))

            if final_policy:
                obs_size, action_size = self.get_input_ouput(self.num_tasks)


            else:
                obs_size, action_size = self.get_input_ouput(task)
                
            
            total_trial_rewards = []
            total_trial_success = []
            total_episode_lengths = []
            for eval_trial in range(self.num_eval_trials):
                
                trial_rewards, trial_success, episode_lengths= self.run_eval_trial(env, task, eval_trial, act_fn, obs_size, action_size, for_eval)

                total_trial_rewards.append(trial_rewards)
                total_trial_success.append(trial_success)
                total_episode_lengths.append(episode_lengths)

            task_alias = "task_" + str(task)
            if final_policy:
                task_alias += "_final_policy"

                self.eval_info[task_alias] = {"rewards": [float(el) for el in total_trial_rewards],
                                        "success": [float(el) for el in total_trial_success],
                                        "episode_lengths": [float(el) for el in total_episode_lengths]}
            else:
                self.eval_info[task_alias] = {"rewards": [float(el) for el in total_trial_rewards],
                                        "success": [float(el) for el in total_trial_success],
                                        "episode_lengths": [float(el) for el in total_episode_lengths],
                                        "gens": gens}
