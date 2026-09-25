"""Kinetix's level names and the cells built from them. NO JAX.

Split out of `source/envs/kinetix.py` for one reason: `source/envs/registry.py`
names every cell in `ENV_SUITE`, `scripts/train` reads that table to build a
parser, and importing jax before `CUDA_VISIBLE_DEVICES` is set freezes the
device mask (see the note above `ENV_SUITE`). The suite module imports jax,
jax2d and the Kinetix renderer; this one imports nothing.

So this file is the level list, and `source/envs/kinetix.py` is the body.
Nothing here is duplicated there -- it imports these names.
"""

from __future__ import annotations

# The twenty hand-designed MEDIUM levels, in the order the previous codebase's
# continual chain visited them (`source/studies/kinetix/ga.py:ENVIRONMENTS`).
# The order matters for the continual cell and nowhere else.
LEVELS = (
    'h0_unicycle',
    'h1_car_left',
    'h2_car_right',
    'h3_car_thrust',
    'h4_thrust_the_needle',
    'h5_angry_birds',
    'h6_thrust_over',
    'h7_car_flip',
    'h8_weird_vehicle',
    'h9_spin_the_right_way',
    'h10_thrust_right_easy',
    'h11_thrust_left_easy',
    'h12_thrustfall_left',
    'h13_thrustfall_right',
    'h14_thrustblock',
    'h15_thrustshoot',
    'h16_thrustcontrol_right',
    'h17_thrustcontrol_left',
    'h18_thrust_right_very_easy',
    'h19_thrust_left_very_easy',
)

# A cell is what `--env` names and what a directory in the run tree is named
# after. `Kinetix20` is the continual chain over all twenty levels;
# `Kinetix-<level>` is the stationary run on one of them -- twenty of those,
# and they are what the non-continual block runs.
CELL_ALL = 'Kinetix20'

CELLS = {CELL_ALL: LEVELS}
CELLS.update({f'Kinetix-{level}': (level,) for level in LEVELS})


def cell_for(level):
    """The stationary cell that runs `level` alone."""
    if level not in LEVELS:
        raise KeyError(f'unknown Kinetix level {level!r}')
    return f'Kinetix-{level}'
