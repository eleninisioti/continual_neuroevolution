"""Population-Based Training: the one exploit/explore rule, and its diagnostics.

PBT-PPO is the population-based member of the RL arm: `pop_size` independent PPO
learners, periodically copying parameters from the top of the population to the
bottom (Jaderberg et al., 2017).

## What this replaced (2026-08-06)

Three trainers held three different rules:

  gymnax    `pbt_exploit_and_explore` in train_PBT_gymnax.py (also used by the
            continual trainer)
  mujoco +  this module's `pbt_exploit_explore`, on Brax TrainingStates
  brax
  kinetix   a third copy in experiments/pbt_continual.py, on raw pytrees

They disagreed on four things, and the disagreements were invisible in the
figures because every panel is labelled "PBT-PPO":

  1. **The optimizer state of a loser.** gymnax rebuilt the TrainState, wiping
     Adam's moments. brax kept the *loser's* moments and paired them with the
     *winner's* weights. Canonical PBT copies the winner's whole state, but
     kinetix structurally cannot -- `make_train` builds a fresh optimiser every
     sub-task -- so the only policy all three can share is **reset**, which is
     also the one that is defensible on its own terms: Adam's moments describe
     the loser's trajectory and mean nothing for the weights that just replaced
     them. That is now `OPTIMIZER_ON_EXPLOIT = 'reset'`, and all three do it.
  2. **Whether explore did anything.** It was a no-op here, implemented in the
     other two. Now implemented once, in `perturb_hyperparams`.
  3. **The bounds.** gymnax clipped ent_coef to (1e-4, 1.0), kinetix to
     (1e-5, 1e-1). An ent_coef of 1.0 swamps the policy loss outright, so the
     tighter pair wins. lr bounds already agreed.
  4. **The defaults.** gymnax's stationary trainer defaulted to `full` and its
     *continual* trainer to `weights_only`, so those two panels of the paper
     were not the same method. One default now: `DEFAULT_PBT_MODE`.

The ranking is the caller's, because "worst" is not the same quantity
everywhere: gymnax and brax rank on mean return, kinetix on solve rate with
return as the tiebreak. That is a per-benchmark metric choice, not a difference
in PBT, so callers pass an `order` and everything downstream is shared.

`TrainingState` is the caller's, not imported here: the cheetah and ant
trainers build it from their own Brax variants, so it is passed in.
"""

import jax
import jax.numpy as jnp
import numpy as np
from jax import random


# -- The settings that must agree across suites -----------------------------

DEFAULT_EXPLOIT_FRACTION = 0.2
DEFAULT_PERTURB_FACTOR = 0.2
DEFAULT_PBT_MODE = 'weights_only'

#: Explored hyperparameters and their clip bounds, applied only in mode 'full'.
#: Twenty compounding multiplicative perturbations drift a rate by orders of
#: magnitude in either direction, and an agent at lr 1e-9 is a dead slot rather
#: than a population member.
HYPERPARAM_BOUNDS = {
    'learning_rate': (1e-6, 1e-2),
    'ent_coef': (1e-5, 1e-1),
}

#: What happens to a loser's optimizer state when it copies a winner's weights.
#: 'reset' is the only policy all three suites can implement -- see the module
#: docstring. Named rather than inlined so that changing it is one edit and is
#: visibly a decision.
OPTIMIZER_ON_EXPLOIT = 'reset'

PBT_MODES = ('full', 'weights_only', 'hp_only')


def _unpmap(v):
    return jax.tree_util.tree_map(lambda x: x[0], v)


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


def pbt_exploit_explore(population_states, population_rewards, key, pop_size,
                        TrainingState, *, optimizer_init,
                        exploit_fraction=DEFAULT_EXPLOIT_FRACTION,
                        perturb_factor=DEFAULT_PERTURB_FACTOR,
                        mode=DEFAULT_PBT_MODE):
    """PBT exploit/explore on Brax TrainingStates (cheetah and ant).

    `optimizer_init(params) -> optimizer_state` builds the fresh optimizer state
    required by `OPTIMIZER_ON_EXPLOIT == 'reset'`; it takes the winner's
    parameters in whatever sharding the caller keeps them and must return a
    matching optimizer state. Required, with no default, because the answer used
    to be "silently keep the loser's" and that is exactly the divergence this
    module exists to remove. Passing `None` explicitly restores that old
    behaviour, for reproducing a pre-2026-08-06 run on purpose.

    Explore is not implemented for Brax: its optimizer state carries a fixed
    learning rate baked in at construction, so there is nothing to perturb
    without rebuilding the optimizer. `mode='full'` therefore raises here
    rather than silently behaving as `weights_only`, which is what it used to
    do -- that silence is how the cheetah and ant PBT panels came to be a
    different method from the gymnax one while sharing its label.
    """
    copy_weights, perturb_hp = mode_flags(mode)
    if perturb_hp:
        raise NotImplementedError(
            "PBT explore is not available on the Brax trainers: the optimizer "
            "state carries a fixed learning rate. Use --pbt_mode weights_only "
            "(the default), or teach my_brax to build its optimizer with "
            "optax.inject_hyperparams as the gymnax trainer does."
        )
    if pop_size <= 1 or not copy_weights:
        return population_states

    order = np.argsort(np.asarray(population_rewards))
    for bot_idx, top_idx in pbt_moves(order, key, exploit_fraction):
        top_state = population_states[top_idx]
        bot_state = population_states[bot_idx]
        # The copy is essential, not defensive. `training_epoch_pmap` is
        # built with donate_argnums=(0, 1), so training an agent invalidates
        # its parameter buffer. Assigning `top_state.params` by reference
        # would leave two agents sharing one buffer; training the first
        # donates it and the second then reads freed memory, which aborts
        # XLA with "Check failed: on_device_shape().has_layout()".
        new_params = jax.tree_util.tree_map(lambda x: x.copy(), top_state.params)
        population_states[bot_idx] = TrainingState(
            optimizer_state=(bot_state.optimizer_state if optimizer_init is None
                             else optimizer_init(new_params)),
            params=new_params,
            normalizer_params=jax.tree_util.tree_map(
                lambda x: x.copy(), top_state.normalizer_params),
            env_steps=bot_state.env_steps,
        )

    return population_states


def compute_param_diversity(population_states, pop_size):
    """Mean pairwise L2 distance between the population's policy parameters.

    Weight-space, not behavioural -- the behavioural measures live in
    `source/metrics/behaviour_descriptors.py`. Reported beside them because a
    population that has collapsed in weight space cannot be diverse in any
    other, so this is the cheap necessary condition.
    """
    if pop_size <= 1:
        return 0.0
    flat_params = []
    for i in range(pop_size):
        params = _unpmap(population_states[i].params.policy)
        flat = jnp.concatenate([p.flatten() for p in jax.tree_util.tree_leaves(params)])
        flat_params.append(flat)
    flat_params = jnp.stack(flat_params)
    dists = []
    for i in range(pop_size):
        for j in range(i + 1, pop_size):
            dists.append(float(jnp.linalg.norm(flat_params[i] - flat_params[j])))
    return float(np.mean(dists)) if dists else 0.0
