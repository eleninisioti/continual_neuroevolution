# Continual Neuroevolution

This codebase contains code for benchmarking neuro-evolution in continual reinforcement learning tasks.

We consider two neuroevolution algorithms: [Evolution Strategies](https://arxiv.org/abs/1703.03864) and [Genetic Algorithms](https://arxiv.org/abs/1712.06567). For Genetic Algorithms, we also consider a variant that jointly maximises fitness and novelty ([Dominated Novelty Search](https://arxiv.org/abs/2502.00593)).


We consider a wide diversity of environments (gymnax, MiniGrid, mujoco, Kinetix) and types of environmental changes (observation noise and different types of change in transition dynamics, action reversal).


## Set up

Install all package dependencies with [uv](https://docs.astral.sh/uv/):

```bash
uv sync && source .venv/bin/activate
```

All experiments were run on a single NVIDIA RTX 6000 GPU.

## How to run experiments

To run all experiments:

```bash
bash scripts/train/run.sh
```

To choose a subset of environments, methods or settings:

```bash
bash scripts/train/run.sh [--envs ENVS] [--methods METHODS] [--settings continual,noncontinual]
```

For example:

```bash
# quick check: GA on stationary CartPole, one trial
bash scripts/train/run.sh --envs CartPole-v1 --methods ga --settings noncontinual --trials 1

# GA and PPO on all continual gymnax environments
bash scripts/train/run.sh --benchmarks gymnax --methods ga,ppo --settings continual
```

Any option left out defaults to everything in the paper. Jobs are spread over all visible GPUs; use `--gpus N` to use only the first N.

| Option | Values |
|---|---|
| `--benchmarks` | `gymnax` `minigrid` `kinetix` `mjx` |
| `--envs` | gymnax: `CartPole-v1` `Acrobot-v1` `MountainCar-v0`; minigrid: `MiniGrid_8x8` `MiniGrid_16x16` `MiniGrid_8x8_16x16`; kinetix: `Kinetix-<level>` `Kinetix20`; mjx: `cheetah` `cheetah_noise` `cheetah_friction` |
| `--methods` | `ga` `dns` `dns_gaussian` `es` `pbt` `pbt2` `ppo` `trac` `redo` `cchain` |
| `--settings` | `noncontinual` `continual` |
| `--trials` | trials per method and environment (default 10) |
| `--gpus` | number of GPUs to use (default: all visible) |
| `--dry-run` | print the job list and exit |

Runs go to `runs/<benchmark>/<setting>/<method>/<env>/trial_<n>/`. Finished trials are skipped on rerun. See `--help` for more options.

The toy-landscape experiments run on CPU with their own script:

```bash
bash scripts/train/run_toy.sh                       # every stage
ONLY="wells_d400 wells_pop" bash scripts/train/run_toy.sh   # a subset
```

## Notebooks

The [notebooks](notebooks/) analyse saved runs and redraw figures of the paper from the data in `notebooks/data/`.

- [basin_width.ipynb](notebooks/basin_width.ipynb): how the basin width and the shared basin are measured, step by step on an ES and a PPO run, ending with the paper's basin figure.
- [compare_methods.ipynb](notebooks/compare_methods.ipynb): every method on CartPole with observation offsets: training curves, the stability-plasticity trade-off (LA, F, Cum., ZT), dormant units and weight growth, and population diversity.
