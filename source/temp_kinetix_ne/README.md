# Continual control: evolution versus reinforcement learning


This is the code that accompanies our submission to ICLR 2026 "Lifelong control through neuroevolution"


## Installing dependencies

We provide the library dependenices in the file [requirements.txt](requirements.txt).
You can create a virtual environment and install them using [uv](https://docs.astral.sh/uv/) with the following commands:

```
uv venv --python 3.12.9
uv pip install -r requirements.txt
```


## Repository overview
This repo contains the following directories:
* [methods](methods) contains the implementation of the methods we have benchmarked (evolution and RL)
  * [brax](methods/RL) contains the implementation of PPO  (extending [Brax](https://github.com/google/brax/tree/main/brax))
  * [neuroevolution](methods/evosax_wrapper) contains the implementation of SimpleGA and OpenES (extending [evosax](https://github.com/RobertTLange/evosax))
  * [kinetix](methods/kinetix) contains the implementation of PPO Transformer that the original work employed [kinetix ](https://kinetix-env.github.io/))


* [scripts](scripts) contains:
  * [train](scripts/train) scripts for rerunning traning. For each method we provide code for training in all tasks described in the paper, with the hyperparameters provided in a separate file 

## Training

To train methods on the control tasks you can run the script for neuroevolution [scripts/train/evosax/train.py](scripts/train/evosax/train.py) and [scripts/train/rl/ppo/train.py](scripts/train/rl/ppo/train.py) for PPO.

Running as is will launch training sequentially in all tasks described in the paper. 
Each family of tasks is ran with a separate function call so you can easily choose a subset of tasks to run.

We have ran all training on a single NVIDIA RTX 6000 GPU.

## Hyperparameters
The hyperparameters for each optimizer are within the training scripts. We provide them here for neuro-evolution for an easier reference:

### SimpleGA Hyperparameters

| Parameter | Classic Control | MinAtar | Kinetix |
|-----------|----------------|---------|---------|
| sigma_init | 0.5 | 0.5 | 0.01 |
| population_ssize | 512 | 1025 | 2024 |


### OpenES Hyperparameters

| Parameter | Classic Control | MinAtar | Kinetix |
|-----------|----------------|---------|---------|
| sigma_init | 0.3 | 0.3 | 0.01 |




## Pre-trained data

We provide the data that our experiments produced on an [online zip file](https://drive.google.com/drive/folders/1h4GLJ0iEtcyi4oj-h_hewTblWHTxV3LJ?usp=sharing).
These include the evaluation info and the post-hoc diversity analysis.

## Computing diversity and PCA plots
The data produced during training for computing the diversity metrics require a considerable amount of space, so we instead provide the script we used for computing them.

Once you've finished a  training run, data will be saved in a specific directory under projects.
To compute diversity and PCA plots for this projects, you can run

```
python scripts/post_analysis/diversity_analysis.py --project dir your_project_directory
```
