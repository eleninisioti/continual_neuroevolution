""" Script for training Proximal Policy Optimisation"""
import sys
import os
sys.path.append(".")
import envs

from scripts.train.rl.ppo.train_utils import PPOExperiment as Experiment
import os
from scripts.train.base.utils import default_env_params
from scripts.train.rl.ppo.hyperparams import train_timesteps, arch
import argparse



def train_gymnax(num_trials, env_name, noise_range, continual=True, perturbe_every_n_gens=None):

    # configure experiment
    exp_config = {"seed": 0,
                  "num_trials": num_trials}
    
    # configure environment
    env_params = default_env_params[env_name]
    env_params["noise_range"] = noise_range
    env_params["perturbe_every_n_gens"] = perturbe_every_n_gens
    env_config = {"env_type": "gymnax",
                  "env_name": env_name,
                  "curriculum": False,
                  "env_params": env_params,
                  "continual": continual}
    
    
    # configure method
    num_timesteps = train_timesteps[env_name]
    optimizer_config = {"optimizer_name": "ppo",
                        "optimizer_type": "brax",
                        "optimizer_params": {"num_timesteps": num_timesteps}}
    
    model_config = {"network_type": "MLP",
                    "model_params": arch[(env_name)]}


    exp = Experiment(env_config=env_config,
                     optimizer_config=optimizer_config,
                     model_config = model_config,
                     exp_config=exp_config)
    exp.run()

def train_classic_control_all(num_trials, optimizer):
       
    #train_gymnax(num_trials=num_trials, env_name="CartPole-v1",noise_range=0.0, continual=True)
    train_gymnax(num_trials=num_trials, env_name="MountainCar-v0",noise_range=0.0, continual=True)
    #train_gymnax(num_trials=num_trials, env_name="Acrobot-v1",noise_range=0.0, continual=True)
    
    
    #train_gymnax(num_trials=num_trials, env_name="CartPole-v1",noise_range=1.0, continual=True)
    #train_gymnax(num_trials=num_trials, env_name="MountainCar-v0",noise_range=1.0, continual=True)
    #train_gymnax(num_trials=num_trials, env_name="Acrobot-v1",noise_range=1.0, continual=True)      
    
    
def train_gymnax_all(num_trials):
    # train_gymnax(num_trials=num_trials, env_name="Acrobot-v1", continual=True)
    train_gymnax(num_trials=num_trials, env_name="CartPole-v1",noise_range=0.0, continual=True)
    train_gymnax(num_trials=num_trials, env_name="MountainCar-v0",noise_range=0.0, continual=True)
    train_gymnax(num_trials=num_trials, env_name="Acrobot-v1",noise_range=0.0, continual=True)
    
    
    train_gymnax(num_trials=num_trials, env_name="CartPole-v1",noise_range=1.0, continual=True)
    train_gymnax(num_trials=num_trials, env_name="MountainCar-v0",noise_range=1.0, continual=True)
    train_gymnax(num_trials=num_trials, env_name="Acrobot-v1",noise_range=1.0, continual=True)

    #train_gymnax(num_trials=num_trials, env_name="MountainCar-v0",continual=False)
    #train_gymnax(num_trials=num_trials, env_name="CartPole-v1",continual=False)

    #train_gymnax(num_trials=num_trials, env_name="MountainCar-v0")
    #train_gymnax(num_trials=num_trials, env_name="CartPole-v1")
    #train_gymnax(num_trials=num_trials, env_name="Breakout-MinAtar")


def train_classic_control_parameteric(num_trials, optimizer):
    for perturbe_every_n_gens in [5,10, 20, 50, 100, 200, 500, 1000]:
        train_gymnax(num_trials=num_trials, env_name="CartPole-v1",   noise_range=1.0,  perturbe_every_n_gens=perturbe_every_n_gens)
        train_gymnax(num_trials=num_trials, env_name="Acrobot-v1", noise_range=1.0,perturbe_every_n_gens=perturbe_every_n_gens)
        train_gymnax(num_trials=num_trials, env_name="MountainCar-v0",  noise_range=1.0, perturbe_every_n_gens=perturbe_every_n_gens)
        
      
      
def train_brax(num_trials, env_name):

    # configure experiment
    exp_config = {"seed": 0,
                  "num_trials": num_trials}
    
    # configure environment
    env_params = default_env_params[env_name]
    env_config = {"env_type": "brax",
                  "env_name": env_name,
                  "curriculum": False,
                  "env_params": env_params}
    
    
    # configure method
    num_timesteps = train_timesteps[env_name]
    optimizer_config = {"optimizer_name": "ppo",
                        "optimizer_type": "brax",
                        "optimizer_params": {"num_timesteps": num_timesteps}}
    
    model_config = {"network_type": "MLP",
                    "model_params": arch[env_name]}


    exp = Experiment(env_config=env_config,
                     optimizer_config=optimizer_config,
                     model_config = model_config,
                     exp_config=exp_config)
    exp.run()
    
def train_ecorobot(num_trials, env_name, robot_type):

    # configure experiment
    exp_config = {"seed": 0,
                  "num_trials": num_trials}
    
    # configure environment
    env_params = default_env_params[env_name]
    env_config = {"env_type": "ecorobot",
                  "env_name": env_name,
                  "curriculum": False,
                  "env_params": {"robot_type": robot_type}}
    
    
    # configure method
    num_timesteps = train_timesteps[env_name]
    optimizer_config = {"optimizer_name": "ppo",
                        "optimizer_type": "brax",
                        "optimizer_params": {"num_timesteps": num_timesteps}}
    
    model_config = {"network_type": "MLP",
                    "model_params": arch[(env_name, robot_type)]}


    exp = Experiment(env_config=env_config,
                     optimizer_config=optimizer_config,
                     model_config = model_config,
                     exp_config=exp_config)
    exp.run()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="This script trains Proximal Policy Optimisation on the stepping gates and ecorobot benchmarks")
    parser.add_argument("--num_trials", type=int, help="Number of trials", default=10)
    parser.add_argument("--optimizer", type=str, help="Choose between SimpleGA and OpenES", default="SompleGA")
    args = parser.parse_args()
    
    #train_brax(num_trials=args.num_trials, env_name="ant")
    train_ecorobot(num_trials=args.num_trials, env_name="locomotion", robot_type="ant")
    #train_classic_control_parameteric(num_trials=args.num_trials, optimizer=args.optimizer)

    
    
    # will train for the lifelong variations of Acrobot, Cartpole, MountainCar 
    #train_classic_control_all(num_trials=args.num_trials, optimizer=args.optimizer)

    # will train Minatar (Breakout, Asterix, SpaceInvaders)
    #train_minatar(num_trials=args.num_trials, optimizer=args.optimizer)

    # will train Kineitx (medium difficuly tasks))
    #train_kinetix_all(num_trials=args.num_trials, optimizer=args.optimizer)

