# Continual Reinforcement Learning with Neuroevolution

Code for the ICLR 2027 submission. It benchmarks neuroevolution (ES, GA, GA + Novelty,
PBT) and deep RL (PPO, TRAC-PPO, ReDo-PPO, C-CHAIN-PPO) on non-stationary tasks in
gymnax, Brax/MJX, MiniGrid and Kinetix.

## Layout

| Path | Contents |
|---|---|
| `source/algorithms/` | one implementation per method (`ne/` neuroevolution, `rl/` PPO and its variants) |
| `source/envs/` | environment wrappers and task schedules (gymnax, Brax/MJX, MiniGrid, Kinetix) |
| `source/studies/` | per-benchmark settings and command-line entry points (`cli.py`) |
| `source/metrics/`, `source/utils/` | plasticity metrics and shared utilities |
| `scripts/train/` | launch scripts for the experiments in the paper |
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
python scripts/check_imports.py
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

Run every command from the repository root. Each benchmark has one entry point, and
`--help` lists its options:

```bash
python source/studies/minigrid/cli.py --help
python source/studies/kinetix/cli.py --help
python source/studies/mjx/cli.py --help      # Brax / MJX: HalfCheetah and Ant
```

For example, a quick check that the MiniGrid configuration resolves:

```bash
python source/studies/minigrid/cli.py --env MiniGrid_8x8_16x16 --method ppo \
    --output_dir /tmp/minigrid_check --dry_run
```

The full experiment grids are defined in `scripts/train/run_experiments.sh` and
started with `scripts/train/launch.sh`, which spreads the jobs over the available GPUs:

```bash
bash scripts/train/launch.sh gymnax_noncontinual nes ga dns ppo trac redo cchain
bash scripts/train/launch.sh gymnax_continual    nes ga dns ppo trac redo cchain
```

The `scripts/train/queue_iclr_*.sh` scripts run the configurations reported in the
paper, one per benchmark (for example, `queue_iclr_kinetix_continual.sh` and
`queue_iclr_minigrid_continual.sh`).

## Evaluation and figures

```bash
# Zero-shot transfer: writes evaluation.json next to each run
python -m source.studies.evaluate_continual --root <runs>/gymnax/continual --episodes 100

# Forgetting: evaluates every checkpoint on every task
python scripts/analysis/behavioural_divergence.py --runs_root <runs> ...

# Training curves and the metric table
python scripts/make_lineplot.py --help
```

## License

`third_party/kinetix/` keeps its original license (see the `LICENSE` file in that directory).
