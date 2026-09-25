"""The environments this repo studies, one module per task family.

Environment construction used to live inside whichever trainer directory first
needed it -- the ant's factories in ``source/studies/brax/``, the cheetah's in
``source/studies/mujoco/`` (since deleted), the toy landscapes' fitness
functions inside the analysis
scripts that plotted them. Two studies then had to reach across that boundary to
share a body: ``source/envs/mjx.py`` got the ant by importing
``source.studies.brax.train_GA_ant``, a *trainer*, which at import time scans
``sys.argv`` for ``--gpus`` and assigns ``CUDA_VISIBLE_DEVICES``. That only ever
failed to fire because the generalists trainer spells the flag ``--gpu``.

So an environment lives here, and nothing here imports a trainer, a searcher or
an argument parser. A study depends on this package; this package depends on no
study. CLI flags, task labels and GIF rendering are not environment construction
and stay with the trainers that parse them (``source/studies/brax/cli.py``).
"""
