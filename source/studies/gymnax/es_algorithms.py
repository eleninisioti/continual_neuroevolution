"""Which distribution-based NE method a gymnax ES run searches with.

`train_ES_gymnax.py` and `train_ES_gymnax_continual.py` both take `--algo
{openes,nes}` and are otherwise one runner, so the choice of strategy -- and the
settings that go with it -- is factored out here rather than living in one
trainer and being imported from the other. Importing a trainer has side effects
(it reads `--gpus` and sets CUDA_VISIBLE_DEVICES at module level); this has
none.

    openes   centered ranks + Adam. The arm every OpenES number in this study
             came from.
    nes      standardized fitness + SGD, the textbook (1, lambda) search
             gradient.

Both are `source/algorithms/ne/es.py` with two arguments changed -- one
implementation, shared with the generalists study, so that an `openes` vs `nes`
row in a table is the difference between those two settings and nothing else.
`openes` used to construct `evosax.algorithms.Open_ES` instead; the class here
reproduces it to the bit (`scripts/check_es.py`, check 2), so the numbers
already on disk are the numbers this produces.
"""

import jax.numpy as jnp
import optax

from source.algorithms.ne.es import EvosaxES


# NES is not OpenES with a different label, and it cannot inherit OpenES's
# sigma/lr: those are Adam step sizes read against centered ranks, while NES
# takes a plain SGD step against standardized fitness, where the same numbers
# mean a step of roughly `lr / sigma`. These are the values the generalists
# study swept on these three environments (single sub-task, no observation
# noise). They are NOT from this study's own sweep, which never ran NES, and
# that is why they sit in their own table instead of being written into the
# trainers' ENV_CONFIGS as though they had been. Everything else -- generations,
# population, network, episode length, evaluations -- still comes from
# ENV_CONFIGS, so the two arms stay compute-matched.
NES_ENV_CONFIGS = {
    "CartPole-v1": {"sigma": 0.1, "learning_rate": 0.05},
    "Acrobot-v1": {"sigma": 0.1, "learning_rate": 0.05},
    "MountainCar-v0": {"sigma": 0.1, "learning_rate": 0.05},
}
# DeepSea<N> (source/envs/gymnax_classic.py): CartPole's setting, untuned.
for _n in (8, 10, 12, 14, 16, 20):
    NES_ENV_CONFIGS[f"DeepSea{_n}-bsuite"] = dict(NES_ENV_CONFIGS["CartPole-v1"])


def resolve_algo_config(algo, env_name, cfg):
    """Per-env sigma/learning_rate for `algo`, as a dict.

    `cfg` is the trainer's own ENV_CONFIGS entry, which holds the OpenES values.
    """
    if algo == 'nes':
        return dict(NES_ENV_CONFIGS[env_name])
    return {"sigma": cfg["sigma"], "learning_rate": cfg["learning_rate"]}


# The two settings that define each method. See `es.py` for what and why.
ALGO_SETTINGS = {
    'nes': dict(shaping='zscore', optimizer=optax.sgd),
    'openes': dict(shaping='centered_rank', optimizer=optax.adam),
}


def build_strategy(algo, pop_size, num_params, sigma, learning_rate, std_lr=0.0):
    """One class, two settings. `std_lr > 0` adapts the width and is NES-only."""
    settings = ALGO_SETTINGS[algo]
    return EvosaxES(
        population_size=pop_size,
        solution=jnp.zeros(num_params),
        std_init=sigma,
        std_lr=std_lr,
        shaping=settings['shaping'],
        optimizer=settings['optimizer'](learning_rate=learning_rate),
        use_antithetic_sampling=True,
    )
