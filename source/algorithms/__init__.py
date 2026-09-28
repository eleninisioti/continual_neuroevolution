"""The methods, independent of any task.

`ne/` is neuroevolution (GA, DNS, ES; `searchers.py` puts them behind one
ask/tell interface), `rl/` is PPO and everything built on it (TRAC, ReDo,
C-CHAIN, PBT). `networks.py` sits above both, because every method searches
the same policy network.

Changing anything here changes what a run does. See `source/__init__.py`.
"""
