# Continual Neuroevolution

Neuroevolution vs. deep RL on non-stationary tasks, in JAX.

**Methods:** GA · GA + Novelty · ES · PBT · PPO · TRAC-PPO · ReDo-PPO · C-CHAIN-PPO

**Benchmarks:** gymnax · MiniGrid · Kinetix · MJX HalfCheetah

## Install

Needs Linux, Python 3.11–3.13, CUDA 12 (optional, since it also runs on CPU) and [uv](https://docs.astral.sh/uv/).

```bash
uv sync && source .venv/bin/activate
python -c "import jax; print(jax.devices())"
```

## Run

```bash
bash scripts/train/run.sh [--gpus N] SETTING[:METHODS] ...
```

```bash
# quick check
ENVS=CartPole-v1 bash scripts/train/run.sh --gpus 1 --trials 1 --root runs gymnax_noncontinual:ga

# GA and PPO on continual gymnax
bash scripts/train/run.sh --gpus 2 gymnax_continual:ga,ppo

# everything
bash scripts/train/run.sh --gpus 8 all
```

| Settings | Methods |
|---|---|
| `{gymnax,minigrid,kinetix,cheetah}_{continual,noncontinual}`, `all` | `ga` `dns` `dns_gaussian` `es` `pbt` `pbt2` `ppo` `trac` `redo` `cchain` |

Runs go to `<root>/<benchmark>/<setting>/<method>/<cell>/trial_<n>/`. Finished trials are skipped on rerun. See `--help` for more options, and [scripts/README.md](scripts/README.md) for evaluation and figures.

---

`third_party/kinetix/` is a modified copy of [Kinetix](https://github.com/FLAIROx/Kinetix) and keeps its original license.
