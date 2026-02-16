# Continual Reinforcement Learning through Neuroevolution

Benchmarking continual learning algorithms on MuJoCo Playground, Kinetix, and Gymnax environments using JAX.

This repository compares evolutionary strategies (SimpleGA, OpenES) against reinforcement learning methods (PPO and variants) on continual learning tasks where the environment changes over time.

## Methods

### Reinforcement Learning
- **PPO**: Proximal Policy Optimization baseline
- **ReDo-PPO**: PPO with Reactivating Dormant Neurons - periodically reinitializes inactive neurons to combat plasticity loss
- **TRAC-PPO**: PPO with Trust Region-Aware Continual learning optimizer - adaptive learning rates for continual learning

We have implemented RL algorithms by modifying the PPO implementation of brax to support TRAC-PPO and ReDO-PPO (our implementation is under folder my_brax).

### Evolutionary Algorithms
- **SimpleGA**: Simple Genetic Algorithm with elitism
- **OpenES**: OpenAI Evolution Strategy with adaptive noise
- **DNS**: Dominated Novelty Search - combines fitness optimization with novelty search using Pareto-based selection

## Tasks

### CheetahRun - Friction Continual Learning
- **Environment**: DeepMind Control Suite CheetahRun
- **Task variation**: Friction coefficient cycles through 3 values (low=0.2x, default=1.0x, high=5.0x)
- **30 tasks** per trial, each with different friction
- **Continual setting**: Network weights preserved across tasks

### Go1 Quadruped - Leg Damage Continual Learning
- **Environment**: Unitree Go1 robot locomotion (MuJoCo Playground)
- **Task variation**: Different leg is damaged (locked in bent position) each task
- **20 tasks** per trial (first task = healthy baseline)
- **Continual setting**: Network weights preserved, random leg selection avoiding consecutive same leg

### Kinetix - Sequential 2D Physics Tasks
- **Environment**: Kinetix (JAX-based 2D physics engine with pixel observations)
- **20 medium-difficulty tasks**: h0_unicycle through h19_thrust_left_very_easy, covering locomotion (unicycle, car), jumping, balancing, pushing, and thrust control
- **Pixel-based observations** with recurrent network (ActorOnlyPixelsRNN with ScannedRNN)
- **Task variation**: Agent trains sequentially on each of the 20 tasks
- **Continual setting**: Population/weights carry over between tasks (no reset), agent must retain skills while learning new ones
- **Non-continual setting**: Each task trained independently

### Gymnax Classic Control - Observation Noise Continual Learning
- **Environments**: CartPole-v1, Acrobot-v1, MountainCar-v0 (JAX-based gymnax)
- **3 environments** with discrete action spaces
- **Task variation**: Observation noise vector sampled from N(0, noise_range) and added to all observations
- **Task switching**: Every `task_interval` updates (default=200), a new noise vector is sampled
- **First task (Task 0)**: Zero noise (baseline), subsequent tasks have random observation perturbations
- **Continual setting**: Network weights preserved across tasks, agent must adapt to changing observation distributions

## Installation

```bash
# Clone the repository
git clone https://github.com/eleninisioti/continual_neuroevolution.git
cd continual_neuroevolution

# Create virtual environment with uv
uv venv .venv
source .venv/bin/activate

# Install dependencies
uv pip install -r requirements.txt

# For GPU support (CUDA 12)
uv pip install -U "jax[cuda12]"
```


## Running Experiments

All experiment scripts are located in `scripts/mujoco/`. Each script runs 10 trials sequentially on a specified GPU.

### CheetahRun Continual Friction

```bash
# PPO, ReDo-PPO, TRAC-PPO (30 trials total)
./scripts/mujoco/run_RL_cheetah_continual.sh [GPU_ID]

# SimpleGA (10 trials)
./scripts/mujoco/run_GA_cheetah_continual.sh [GPU_ID]

# OpenES (10 trials)
./scripts/mujoco/run_ES_cheetah_continual.sh [GPU_ID]

# DNS (10 trials)
./scripts/mujoco/run_DNS_cheetah_continual.sh [GPU_ID]
```

### Go1 Quadruped Continual Leg Damage

```bash
# PPO, ReDo-PPO, TRAC-PPO (30 trials total)
./scripts/mujoco/run_RL_quadruped_continual.sh [GPU_ID]

# SimpleGA (10 trials)
./scripts/mujoco/run_GA_quadruped_continual.sh [GPU_ID]

# OpenES (10 trials)
./scripts/mujoco/run_ES_quadruped_continual.sh [GPU_ID]
```

### Kinetix 2D Physics Tasks

```bash
# PPO, ReDo-PPO, TRAC-PPO continual
./scripts/kinetix/run_kinetix_continual.sh

# PPO, ReDo-PPO, TRAC-PPO non-continual
./scripts/kinetix/run_kinetix_noncontinual.sh

# SimpleGA continual / non-continual
./scripts/kinetix/run_GA_kinetix_continual.sh
./scripts/kinetix/run_GA_kinetix_noncontinual.sh

# OpenES continual / non-continual
./scripts/kinetix/run_ES_kinetix_continual.sh
./scripts/kinetix/run_ES_kinetix_noncontinual.sh

# DNS continual / non-continual
./scripts/kinetix/run_DNS_kinetix_continual.sh
./scripts/kinetix/run_DNS_kinetix_noncontinual.sh
```

### Gymnax Classic Control - Observation Noise

```bash
# SimpleGA continual (10 trials per environment)
./scripts/gymnax/run_GA_gymnax_continual.sh

# OpenES continual (10 trials per environment)
./scripts/gymnax/run_ES_gymnax_continual.sh

# DNS continual (10 trials per environment)
./scripts/gymnax/run_DNS_gymnax_continual.sh

# PPO continual (10 trials per environment)
./scripts/gymnax/run_RL_gymnax_continual.sh
```


## Output Structure

Results are saved under `projects/mujoco/`, `projects/gymnax/`, and `projects/kinetix/`:

```
projects/mujoco/
├── ppo_CheetahRun_continual_friction/
│   ├── trial_1/
│   │   ├── train.log
│   │   ├── config.json
│   │   └── checkpoints/
│   ├── trial_2/
│   └── ...

projects/gymnax/
├── ppo_CartPole_v1_continual/
│   ├── trial_1/
│   │   ├── training_metrics.json
│   │   ├── gifs/
│   │   └── *.pkl
│   ├── trial_2/
│   └── ...

projects/kinetix/
├── dns_continual/
│   ├── trial_0/
│   │   ├── gifs/
│   │   └── *.pkl
│   └── ...
├── ga_continual/
├── es_continual/
└── ppo_continual/
```

## Monitoring

All experiments log to Weights & Biases (wandb). Monitor progress:

```bash
# Check log files
tail -f projects/mujoco/ppo_CheetahRun_continual_friction/trial_1/train.log

# View wandb dashboard
# Project: continual_neuroevolution_ICML_2026_ppo (or _redo, _trac, _ga, _es)
```

## Hyperparameters

### PPO / ReDo-PPO / TRAC-PPO
- Timesteps per task: 51.2M (CheetahRun), 25.6M (Quadruped)
- Network: (512, 256, 128) hidden layers
- Learning rate: 3e-4
- Batch size: 2048
- ReDo frequency: every epoch

### SimpleGA
- Population size: 512
- Generations per task: 500 (CheetahRun), 50 (Quadruped)
- Mutation std: 1.0

### OpenES
- Population size: 512
- Generations per task: 100 (CheetahRun), 50 (Quadruped)
- Sigma: 0.04
- Learning rate: 0.01

### Kinetix (2D Physics)
- 20 sequential medium-difficulty tasks
- Pixel observations with recurrent network (ActorOnlyPixelsRNN)
- Population size (GA/ES/DNS): 256
- Generations per task: 50 (continual), 200 (non-continual)
- DNS: iso_sigma=0.05, line_sigma=0.5, k=3

### Gymnax (Classic Control)
- Task interval: 200 updates (new noise vector sampled)
- Noise range: 1.0 (std of observation noise)
- Network: (64, 64) hidden layers
- Population size (GA/ES/DNS): 128
- Total updates: 2000


## License

MIT License
