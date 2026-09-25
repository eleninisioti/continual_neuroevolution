"""The search algorithms, independent of any task.

Split by arm: `ne/` is the neuro-evolution methods (GA, DNS, NES), `rl/` is PPO
and everything built on top of it (ReDo, PBT). `networks.py` sits above both,
because the same policy definition is used by every method in the study.

Changing anything here changes what a run does -- this is the one shared
directory that is not pure observation. See `source/__init__.py`.
"""
