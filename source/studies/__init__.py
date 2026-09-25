"""The experiments, one package per study.

A *study* is a set of runs meant to be read together: a grid of methods over a
task family, its trainers, and the flags that drive them. Studies are the only
place a trainer lives.

    generalists   two trainers -- `train_nes` and `train_ppo` -- over every
                  task family in `source/envs/`, selected by `--env` alone.
    gymnax        the paper's continual classic-control block.
    brax          the paper's continual ant block. The cheetah is a brax body
                  too since 2026-09-08 and is driven from `generalists`.
    kinetix       NE and RL over Kinetix's hand-designed levels.

A study imports `source/algorithms/`, `source/envs/`, `source/metrics/` and
`source/utils/`. It must not import another study: when two studies needed the
same body, the answer was to lift the environment into `source/envs/`, which is
why `envs/brax_ant.py` and `envs/brax_common.py` exist.

The vendored Kinetix checkout the kinetix study runs on is NOT a study and is
not here. It lives in `third_party/kinetix/`, which pyproject installs editable;
our experiments lived inside that checkout until 2026-09-08, which made them
look like part of a third-party package.

A scheduling study (job-shop and load-balancing) and a `mujoco` study (the
cheetah on mujoco_playground) sat here until 2026-09-08. Scheduling was deleted
with the MinAtar suite -- neither separated the methods, and
carrying either was a cost with no return.
"""
