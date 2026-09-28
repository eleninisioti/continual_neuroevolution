"""The environments this repo studies, one module per task family.

Environment construction used to live inside whichever trainer directory first
needed it -- the cheetah's in the per-method mujoco trainers (since deleted), the
toy landscapes' fitness functions inside the analysis scripts that plotted
them. Two studies then had to reach across that boundary to share a body, and
importing a *trainer* to build an environment also ran its import-time side
effects (``CUDA_VISIBLE_DEVICES`` assignment).

So an environment lives here, and nothing here imports a trainer, a searcher or
an argument parser. A study depends on this package; this package depends on no
study.
"""
