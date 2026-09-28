"""Population-Based Training: the one exploit/explore rule, and its diagnostics.

PBT-PPO is the population-based member of the RL arm: `pop_size` independent PPO
learners, periodically copying parameters from the top of the population to the
bottom (Jaderberg et al., 2017).
"""

import numpy as np
from jax import random


# -- The settings that must agree across suites -----------------------------

DEFAULT_EXPLOIT_FRACTION = 0.2
DEFAULT_PERTURB_FACTOR = 0.2

#: Explored hyperparameters and their clip bounds, applied only in mode 'full'.
#: Twenty compounding multiplicative perturbations drift a rate by orders of
#: magnitude in either direction, and an agent at lr 1e-9 is a dead slot rather
#: than a population member.
HYPERPARAM_BOUNDS = {
    'learning_rate': (1e-6, 1e-2),
    'ent_coef': (1e-5, 1e-1),
}

#: What happens to a loser's optimizer state when it copies a winner's weights:
#: a fresh one, not the loser's stale moments. Named rather than inlined so
#: the choice is visible; `run_ppo`'s PBT block implements it.
OPTIMIZER_ON_EXPLOIT = 'reset'

PBT_MODES = ('full', 'weights_only', 'hp_only')


def mode_flags(mode):
    """(copy_weights, perturb_hyperparams) for a `--pbt_mode`.

    Modes follow the ablation in Jaderberg et al. Sect. 4.1.2, Fig. 5c:
      full          exploit (copy weights) AND explore (perturb hyperparameters)
      weights_only  exploit only -- hyperparameters stay as initialised
      hp_only       explore only -- each agent keeps its own weights
    """
    if mode not in PBT_MODES:
        raise ValueError(f"pbt_mode must be one of {PBT_MODES}, got {mode!r}")
    return mode in ('full', 'weights_only'), mode in ('full', 'hp_only')


def pbt_moves(order, key, exploit_fraction=DEFAULT_EXPLOIT_FRACTION):
    """Who copies from whom. Returns `[(loser, winner), ...]`.

    `order` ranks the population worst-first -- `np.argsort(rewards)` where
    return is the metric, `np.lexsort` where something else is. The bottom
    `exploit_fraction` each draw a winner uniformly from the top
    `exploit_fraction`.

    An agent never exploits itself. The two index sets overlap once
    `exploit_fraction >= 0.5` (and always at `pop_size == 1`), and a self-copy
    is a no-op that would still reset the agent's optimizer and be logged as an
    exploit. Such a pair is dropped, so the returned list can be shorter than
    the bottom fraction -- callers that log an exploit count should count what
    comes back rather than assume `num_replace`.
    """
    order = np.asarray(order)
    pop_size = len(order)
    if pop_size < 2:
        # A population of one has nobody to exploit; PBT degenerates to PPO,
        # which is exactly what the pop=1 reference point should be.
        return []

    num_replace = max(1, int(pop_size * exploit_fraction))
    bottom, top = order[:num_replace], order[-num_replace:]

    moves = []
    for loser in bottom:
        key, sel_key = random.split(key)
        winner = int(top[int(random.randint(sel_key, (), 0, len(top)))])
        if winner == int(loser):
            continue
        moves.append((int(loser), winner))
    return moves


def perturb_hyperparams(winner_hypers, key, perturb_factor=DEFAULT_PERTURB_FACTOR,
                        bounds=None):
    """PBT explore: the winner's hyperparameters, each scaled by U(1-f, 1+f).

    Only keys present in `bounds` (default `HYPERPARAM_BOUNDS`) are perturbed;
    anything else the caller carries in its hyperparameter dict comes across
    unchanged. Returns a new dict -- the input is not mutated.
    """
    bounds = HYPERPARAM_BOUNDS if bounds is None else bounds
    out = dict(winner_hypers)
    for name, (lo, hi) in bounds.items():
        if name not in out:
            continue
        key, h_key = random.split(key)
        factor = 1.0 + perturb_factor * (2.0 * float(random.uniform(h_key)) - 1.0)
        out[name] = float(np.clip(out[name] * factor, lo, hi))
    return out


# ---------------------------------------------------------------------------
# Method names. An arm name such as `pbt2_weights` stands for a population size
# and a mode; this is the one place that decodes it.
# ---------------------------------------------------------------------------

#: Arm-name suffix for PBT without explore: the loser copies the winner's
#: weights and every member trains at the config's one set of
#: hyperparameters, no initial spread (`--pbt_mode weights_only`, Jaderberg
#: et al. Sect. 4.1.2). `pbt_weights` / `pbt2_weights` are the ablation of
#: `pbt` / `pbt2`. Mode `full` is `run_ppo`'s default.
PBT_WEIGHTS_SUFFIX = '_weights'
PBT_HP_SUFFIX = '_hp'            # mode hp_only: explore without exploit


def pbt_arm(name):
    """``(runner method, pbt_pop_size, pbt_mode)`` for an ARM name.

    `pbt` is the N = 8 population and `pbt<N>` the same method at N members
    (`pbt2`): two compute-matched population sizes, one method,
    one run directory per size so the two conditions are never averaged
    together. A `_weights` suffix (PBT_WEIGHTS_SUFFIX) is the same population
    in mode `weights_only`, a `_hp` suffix (PBT_HP_SUFFIX) the same population
    in mode `hp_only`; without either the mode is `full`. Any other arm is
    its own method and has no population (None, None).
    """
    mode = 'full'
    if name.endswith(PBT_WEIGHTS_SUFFIX):
        name, mode = name[:-len(PBT_WEIGHTS_SUFFIX)], 'weights_only'
    elif name.endswith(PBT_HP_SUFFIX):
        name, mode = name[:-len(PBT_HP_SUFFIX)], 'hp_only'
    if name == 'pbt':
        return 'pbt', 8, mode
    if name.startswith('pbt') and name[3:].isdigit():
        return 'pbt', int(name[3:]), mode
    return name, None, None


def pbt_kwargs(name):
    """The `run_ppo` keyword arguments an ARM name stands for: `method`, plus
    `pbt_pop_size` and `pbt_mode` when the arm is a PBT population. The suite
    CLIs splat this, so an arm name is decoded in one place."""
    method, pop_size, mode = pbt_arm(name)
    if pop_size is None:
        return {'method': method}
    return {'method': method, 'pbt_pop_size': pop_size, 'pbt_mode': mode}
