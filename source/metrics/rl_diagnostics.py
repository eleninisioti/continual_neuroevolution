"""One naming for the RL plasticity diagnostics, shared by every trainer.

brax's PPO emits its metrics under brax's names (`policy/dormant_fraction`,
`v_loss`, `eval/episode_reward`); the gymnax trainers emit the same quantities
under gymnax's (`policy_dormant_frac`, `vf_loss`, `mean_reward`). Every reader
downstream -- `scripts/compare.py`, the churn and dormancy figures -- was
written against the gymnax names, so the brax trees have to speak them too or
each reader needs a per-suite special case.

This mapping was written and debugged inside
`source/studies/brax/train_RL_ant_continual_legs.py`. It is here rather than there so
the stationary ant trainer, and later the cheetah ones, get it without a second
copy that can drift -- the drift being exactly what
`docs/unify_implementations.md` is a record of.

The left-hand name is what lands in `training_metrics.json`; the tuple is the
candidate keys a given brax version may have used, tried in order.
"""

import numpy as np

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
    # these the churn columns were silently None on a C-CHAIN run that had in
    # fact computed them.
    #
    # Policy churn, now present for EVERY method rather than C-CHAIN alone.
    # Squared difference of action means against the policy one gradient step
    # back -- the published estimator, which the reference logs for vanilla PPO
    # (crl_run_ppo_dmc.py:318) exactly as it does for C-CHAIN
    # (crl_run_ppo_c_chain_dmc.py:339). C-CHAIN reads its reference off
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

# Ant-specific. What the robot is actually DOING, not just what it scored: on
# the target-speed task a reward near 1000 is equally consistent with standing
# still while paying control cost, running far past the target for no credit,
# and falling over halfway. Three separate misreadings of that reward were only
# settled by measuring velocity, so it is logged rather than inferred.
ANT_KEYS = {
    'x_velocity': ('eval/episode_x_velocity',),
    'x_velocity_std': ('eval/episode_x_velocity_std',),
    'y_velocity': ('eval/episode_y_velocity',),
    'episode_length': ('eval/avg_episode_length',),
    'reward_forward': ('eval/episode_reward_forward',),
    'reward_survive': ('eval/episode_reward_survive',),
    'reward_ctrl': ('eval/episode_reward_ctrl',),
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

DIAGNOSTIC_KEYS = {**CORE_KEYS, **ANT_KEYS, **WEIGHT_KEYS, **NTK_KEYS}


def extract(metrics, keys=DIAGNOSTIC_KEYS):
    """The gymnax-named diagnostics present in `metrics`, else None.

    None rather than omitted: `training_metrics.json` then has one schema for
    every method and every record, which is what lets a reader tell "this method
    does not have churn" from "this record predates churn logging".
    """
    out = {}
    for name, candidates in keys.items():
        value = None
        for key in candidates:
            raw = metrics.get(key)
            if raw is None:
                continue
            # np.asarray().mean() rather than float(): the loss metrics come
            # straight off training_epoch_pmap and are still per-device arrays,
            # and float() on those raises -- which an earlier version swallowed
            # into None, so every loss column was silently empty while the
            # dormant ones (already host floats) worked.
            try:
                value = float(np.asarray(raw, dtype=float).mean())
            except (TypeError, ValueError):
                value = None
            break
        out[name] = value
    return out


def make_missing_reporter(keys=DIAGNOSTIC_KEYS, min_records=4, printer=print):
    """A callable that reports, once, which tracked keys never showed up.

    The mapping above is against key names a brax version bump can rename, and a
    diagnostic that silently logs None for a whole sweep is the failure this
    guards against.

    It deliberately does not fire on the first record: each sub-task's *initial*
    evaluation runs with `training_metrics={}`, so it carries no losses and no
    dormant stats by construction. Reporting there listed every key as missing
    on a run where they all in fact arrive one record later -- and a warning
    that fires on every healthy run is one nobody reads. So: wait for a record
    that has at least one diagnostic and report what is still absent there; if
    several records pass with nothing at all, say that instead.
    """
    state = {'done': False, 'seen': 0}

    def report(metrics):
        if state['done']:
            return
        got = extract(metrics, keys)
        present = {k for k, v in got.items() if v is not None}
        state['seen'] += 1
        if not present:
            # C-CHAIN's churn keys are absent for every other method, so "no
            # diagnostics at all" is only alarming once a few records have gone
            # by with nothing.
            if state['seen'] >= min_records:
                state['done'] = True
                printer("  [diagnostics] WARNING: no tracked key appeared in "
                        f"the first {state['seen']} records.")
                printer(f"  [diagnostics] metrics keys were: {sorted(metrics)}")
            return
        state['done'] = True
        missing = sorted(k for k, v in got.items() if v is None)
        if missing:
            printer(f"  [diagnostics] present: {len(present)}; absent: {missing}")
        else:
            printer("  [diagnostics] all tracked keys present")

    return report
