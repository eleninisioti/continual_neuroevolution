"""The gradient-based methods: PPO and everything built on top of it.

    ppo.py          PPO's loss, GAE, one gradient step, data collection and
                    the epochs of minibatch updates
    action_heads.py the distributions PPO samples actions from (categorical,
                    multi-discrete, Gaussian) and its observation normaliser
    redo.py         ReDo's dormant-neuron recycling and dormancy criterion
    cchain.py       C-CHAIN's churn regulariser and coefficient controller
    pbt.py          Population-Based Training's exploit/explore rule, and
                    what a PBT method name (`pbt2_weights`, ...) stands for

TRAC is the `trac_optimizer` package, wrapped around the optimiser in
`source/runners/train_ppo.py`. PBT lives here rather than under `ne/` because
its population members are PPO learners, not genomes.
"""
