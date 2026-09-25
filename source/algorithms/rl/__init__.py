"""The gradient-based search algorithms: PPO and everything built on top of it.

PPO itself (`ppo.py`), ReDo's dormant-neuron reset (`redo.py`) and
Population-Based Training's exploit/explore rule (`pbt.py`). PBT-PPO lives here
rather than under `ne/` because its population members are PPO learners, not
genomes -- it is population-based, not neuro-evolutionary.

TRAC and C-CHAIN are not here: both are per-suite wrappers over these
(`source/studies/gymnax/cchain.py`, `source/studies/brax/my_brax/cchain.py`,
`third_party/kinetix/kinetix/models/cchain.py`).
"""
