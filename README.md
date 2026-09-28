# Continual Reinforcement Learning with Neuroevolution

Code for the ICLR 2027 submission. It benchmarks neuroevolution (ES, GA, GA + Novelty,
PBT) and deep RL (PPO, TRAC-PPO, ReDo-PPO, C-CHAIN-PPO) on non-stationary tasks in
gymnax, Brax/MJX, MiniGrid and Kinetix.

## Layout

| Path | Contents |
|---|---|
| `source/algorithms/` | one implementation per method (`ne/` neuroevolution, `rl/` PPO and its variants) |
| `source/envs/` | environment wrappers and task schedules (gymnax, Brax/MJX, MiniGrid, Kinetix) |
| `source/configs/` | one YAML config per benchmark: cells, methods, hyperparameters and budgets |
| `source/runners/` | the two training loops every benchmark shares: `train_nes.py` (GA, ES, DNS) and `train_ppo.py` (PPO and its variants, PBT) |
| `source/run.py` | the one entry point that reads a config and calls a runner |
| `source/metrics/`, `source/utils/` | plasticity metrics and shared utilities |
| `scripts/train/` | `run.sh`, which runs the experiments in the paper |
| `scripts/analysis/`, `scripts/make_*.py` | post-hoc evaluation, tables and figures |
| `third_party/kinetix/` | a modified copy of Kinetix, with dormant-neuron instrumentation and the level set we use |

## Installation

Requirements: Linux, Python 3.11–3.13, an NVIDIA GPU with CUDA 12 drivers (the code also runs
on CPU, but slowly), and [uv](https://docs.astral.sh/uv/).

```bash
# 1. Install uv (skip this if you already have it)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Create the environment from the lock file. This also installs third_party/kinetix in editable mode.
cd <repo>
uv sync
source .venv/bin/activate

# 3. Check the install
python -c "import jax; print(jax.devices())"
```

`uv sync` installs the exact versions in `uv.lock` (JAX 0.5.3 with CUDA 12, Brax 0.14,
gymnax 0.0.9, xminigrid 0.9.3, evosax 0.2). On a machine without a GPU, JAX falls back to
the CPU.

Without uv, a plain virtual environment also works:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e third_party/kinetix
pip install -e .
```

## Running experiments

Run every command from the repository root. One script runs everything:

```bash
bash scripts/train/run.sh [--gpus N] SETTING[:METHODS] [SETTING[:METHODS] ...]
```

It queues every trial of every setting you name, in that order, and runs them one per
GPU on the first `N` GPUs (default: all visible). Each benchmark has a stationary and a
continual setting:

| Benchmark | Stationary setting | Continual setting | Cells (stationary / continual) |
|---|---|---|---|
| gymnax | `gymnax_noncontinual` | `gymnax_continual` | CartPole-v1, Acrobot-v1, MountainCar-v0 / each at noise σ 0.02, 1.0, 2.0 |
| MiniGrid | `minigrid_noncontinual` | `minigrid_continual` | MiniGrid_8x8, MiniGrid_16x16 / MiniGrid_8x8_16x16 |
| Kinetix | `kinetix_noncontinual` | `kinetix_continual` | the 20 `Kinetix-h*` levels / Kinetix20 |
| MJX HalfCheetah | `cheetah_noncontinual` | `cheetah_continual` | cheetah / cheetah_noise, cheetah_friction |

`all` stands for all eight, and `gymnax_popsize` is the population-size sweep.

Methods:

| Method | Name(s) |
|---|---|
| GA | `ga` |
| GA + Novelty (DNS) | `dns` (gymnax, Kinetix), `dns_gaussian` (MiniGrid, Kinetix, MJX) |
| ES | `es` (z-scored fitness + SGD on gymnax, MiniGrid and MJX; centred ranks + Adam on Kinetix) |
| PBT | `pbt` (population of 8), `pbt2` (population of 2) |
| PPO, TRAC-PPO, ReDo-PPO, C-CHAIN-PPO | `ppo`, `trac`, `redo`, `cchain` |

Without `:METHODS` a setting runs the methods the paper reports for that benchmark
(`reported_arms` in `source/configs/<benchmark>.yaml`). Examples:

```bash
# every experiment in the paper, on 8 GPUs
bash scripts/train/run.sh --gpus 8 all

# all methods on the gymnax continual setting, on 4 GPUs
bash scripts/train/run.sh --gpus 4 gymnax_continual

# only GA and PPO, on two settings one after the other
bash scripts/train/run.sh --gpus 2 gymnax_noncontinual:ga,ppo gymnax_continual:ga,ppo

# one quick trial, to check the setup
ENVS=CartPole-v1 bash scripts/train/run.sh --gpus 1 --trials 1 --root runs gymnax_noncontinual:ga
```

Options: `--gpu-ids 0,2` picks exact GPUs, `--trials N` sets trials per method and cell
(default 10), `--root DIR` sets where runs go, and `--dry-run` prints the job list
without running it. `bash scripts/train/run.sh --help` lists the rest.

Each run is written to `<root>/<benchmark>/<setting>/<method>/<cell>/trial_<n>/`, and
its log to `logs/run/`. A trial that already has a `training_metrics.json` is skipped,
so running the same command again runs only what is missing or failed.

### How a run is put together

| Path | Role |
|---|---|
| `source/run.py` | the one Python entry point: `--suite <benchmark> --env <cell> --method <m>` |
| `source/configs/<benchmark>.yaml` | what an experiment is: cells, methods, hyperparameters, compute-matched budgets |
| `source/utils/config.py` | loads a benchmark's YAML and derives what `run.py` needs from it (cells, budgets, budget checks) |
| `source/runners/` | the two training loops every benchmark shares: `train_nes.py` (GA, ES, DNS) and `train_ppo.py` (PPO and its variants, PBT) |
| `source/algorithms/` | the methods themselves (`ne/`, `rl/`) |

## Evaluation and figures

```bash
# Zero-shot transfer: writes evaluation.json next to each run
python scripts/analysis/evaluate_continual.py --root <runs>/gymnax/continual --episodes 100

# Forgetting: evaluates every checkpoint on every task
python scripts/analysis/behavioural_divergence.py --runs_root <runs> ...

# Training curves and the metric table
python scripts/plotting/make_lineplot.py --help
```

## License

`third_party/kinetix/` keeps its original license (see the `LICENSE` file in that directory).
