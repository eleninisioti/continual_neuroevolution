"""The two training loops every benchmark shares.

    train_nes.py  `run_nes`: neuroevolution -- GA, ES and DNS, one loop
                  over the searchers in `source/algorithms/ne/searchers.py`.
    train_ppo.py  `run_ppo`: PPO and its continual-RL variants (TRAC, ReDo,
                  C-CHAIN), and PBT populations of PPO.
    common.py     the task schedule and the run artifacts both loops write.

A loop owns the run: environment, sub-task schedule, evaluation, checkpoints
and output files. What a method IS lives in `source/algorithms/`, and what an
experiment is -- cells, methods, hyperparameters, budgets -- in
`source/configs/<suite>.yaml`. `source/run.py` reads a config and calls a loop.
"""
