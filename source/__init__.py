"""The study's code: methods, environments, measurements and training loops.

The study is an 8-method x 3-suite grid.

Sub-packages, split by what the code is FOR rather than by suite:

  algorithms  the methods. Changing anything here changes what a run does:
              `ne/` (GA, DNS, ES/NES and the searchers that wrap them), `rl/`
              (PPO, ReDo, C-CHAIN, PBT) and `networks` (the policy and value
              nets, flat<->pytree conversion).

  runners     the two training loops every benchmark shares, `train_nes` and
              `train_ppo`.

  configs     one YAML file per benchmark: its cells, methods, hyperparameters
              and compute-matched budgets. `utils/config.py` loads one and
              `run.py` calls a runner with it.

  metrics     what a run MEASURES -- both as it goes and afterwards. Every
              during-run module is a pure observer: it draws on a private RNG
              stream and never feeds back into the search, so a run with
              diagnostics on and one with them off are the same run. That is
              the property that makes this directory safe, and it is worth
              re-checking before adding to it. `evaluation_metrics` is the
              post-hoc exception, read against finished runs on disk; it is
              the definition of the continual metrics `scripts/compare.py`
              reports. Plasticity (dormancy, churn), weight statistics, the
              brax -> gymnax metric-name mapping, and the behavioural
              descriptors and diversity trackers.

  utils       process setup with no opinion about the science: GPU selection
              before JAX is imported, the stdout tee, `write_run_config`.

  envs        how an environment is built and what a sub-task does to it, one
              module per task family. Nothing here imports a trainer or an
              argument parser -- see `source/envs/__init__.py`.

The dependency arrows point one way: metrics may use algorithms (dormancy is
scored with ReDo's own criterion, deliberately), the runners may
use anything, envs depend on none of them, and nothing in algorithms imports
from metrics.
"""
