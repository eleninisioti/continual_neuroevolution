"""One naming for the RL plasticity diagnostics, shared by every trainer.

brax's PPO emits its metrics under brax's names (`policy/dormant_fraction`,
`v_loss`, `eval/episode_reward`); the gymnax trainers emit the same quantities
under gymnax's (`policy_dormant_frac`, `vf_loss`, `mean_reward`). Every reader
downstream -- `scripts/compare.py`, the churn and dormancy figures -- was
written against the gymnax names, so the brax trees have to speak them too or
each reader needs a per-suite special case.

It lives here rather than in a trainer so every brax trainer gets it without
a second copy that can drift -- the drift being exactly what
`docs/unify_implementations.md` is a record of.

The left-hand name is what lands in `training_metrics.json`; the tuple is the
candidate keys a given brax version may have used, tried in order.
"""


from source.metrics import weight_stats

# Keys common to every PPO-family run, on any environment.
CORE_KEYS = {
    'policy_dormant_frac': ('policy/dormant_fraction',),
    # 'dormant_frac' is the policy net under its original gymnax name, kept so
    # one reader serves both trees.
    'dormant_frac': ('policy/dormant_fraction',),
    'value_dormant_frac': ('value/dormant_fraction',),
    'policy_dormant_neurons': ('policy/dormant/total_count',),
    'policy_total_neurons': ('policy/dormant/total_neurons',),
    'value_dormant_neurons': ('value/dormant/total_count',),
    'value_total_neurons': ('value/dormant/total_neurons',),
    'dormant_neurons': ('policy/dormant/total_count',),
    'policy_dormant_age_mean': ('policy/dormant/age_mean',),
    'value_dormant_age_mean': ('value/dormant/age_mean',),
    # The `training/` variants are not optional. `ppo_train` wraps everything it
    # collected during the epoch as `training/<name>` before handing it to
    # progress_fn, so the bare names only ever match the continual core. Without
    # these the churn columns would be silently None on a C-CHAIN run that had
    # in fact computed them.
    #
    # Policy churn, present for EVERY method rather than C-CHAIN alone.
    # Squared difference of action means against the policy one gradient step
    # back -- the published estimator (Tang et al.). C-CHAIN reads its reference off
    # chain_state; the others get it from `prev_params` threaded out of the
    # minibatch scan. Same delta, same estimator, one column.
    'policy_churn': ('policy_churn', 'training/policy_churn'),
    'value_churn': ('value_churn', 'training/value_churn'),
    'chain_p_reg_loss': ('chain_p_reg_loss', 'chain_reg_loss',
                         'training/chain_reg_loss'),
    'chain_p_reg_coef': ('chain_coef', 'training/chain_coef'),
    # Losses, under whichever name this brax version emitted them.
    'entropy': ('entropy_loss', 'training/entropy_loss', 'entropy'),
    'pg_loss': ('policy_loss', 'training/policy_loss'),
    'vf_loss': ('v_loss', 'value_loss', 'training/v_loss'),
    # The spread across evaluation episodes. gymnax records std_reward next to
    # mean_reward and brax reports the same quantity as an eval metric; without
    # it a single-number reward curve cannot carry error bars from within a run.
    'std_reward': ('eval/episode_reward_std',),
}

# Weight statistics. Unlike everything above these are not renames of a brax
# metric: `ppo_continual_train` computes them with the same
# `source/metrics/weight_stats.py` the NE trainers use and writes them
# into `metrics` under their final names, because progress_fn is handed metrics
# and not parameters. They are listed here anyway so they travel with the rest
# of the schema -- a record missing them then reads as None, like any other
# absent diagnostic, rather than as a missing column.
WEIGHT_KEYS = {
    f'{prefix}_{suffix}': (f'{prefix}_{suffix}',)
    for prefix in ('policy_weight', 'value_weight')
    for suffix in weight_stats.STAT_SUFFIXES
}

# NTK rank, written into `metrics` under its final names by the same route the
# weight statistics take. C-CHAIN's own plasticity indicator; see ntk.py.
NTK_KEYS = {
    f'policy_ntk_{suffix}': (f'policy_ntk_{suffix}',)
    for suffix in ('effective_rank', 'srank', 'trace', 'rank_ratio', 'num_probe')
}

DIAGNOSTIC_KEYS = {**CORE_KEYS, **WEIGHT_KEYS, **NTK_KEYS}

