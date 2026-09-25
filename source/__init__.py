"""The study's code: methods, environments, measurements, and the studies themselves.

The study is an 8-method x 3-suite grid, and before this package each cell was a
standalone trainer: `train_GA_gymnax.py` and `train_ES_gymnax.py` differed in 46
of their first 410 lines and agreed on the rest, down to the byte. What lives
here is the part that was agreeing -- each symbol was verified identical across
the trainers it was lifted from, so importing it cannot change a run.

Sub-packages, split by what the code is FOR rather than by which suite it
grew up in:

  algorithms  the search itself. Changing anything here changes what a run
              does: `ppo` (the core every RL method builds on -- PPO, ReDo,
              TRAC, C-CHAIN, and the per-member update inside PBT), `redo`
              (recycling, and the dormancy criterion it scores on), `networks`
              (the policy and value nets, flat<->pytree conversion) and
              `ne.{ga,dns,pbt}`.

  metrics     what a run MEASURES -- both as it goes and afterwards. Every
              during-run module is a pure observer: it draws on a private RNG
              stream and never feeds back into the search, so a run with
              diagnostics on and one with them off are the same run.
              `evaluation_metrics` is the post-hoc exception, read against
              finished runs on disk; it is the definition of the continual
              metrics `scripts/compare.py` reports. It lived in its own
              `source/analysis/` package until 2026-09-08, which was one
              directory for one file. That is the property that makes
              this directory safe, and it is worth re-checking before adding to
              it. Plasticity (dormancy, churn), weight statistics, the brax ->
              gymnax metric-name mapping, and the behavioural descriptors and
              diversity trackers.

  utils       process setup with no opinion about the science: GPU selection
              before JAX is imported, the stdout tee, `write_run_config`.

  envs        how an environment is built and what a sub-task does to it, one
              module per task family. Nothing here imports a trainer or an
              argument parser -- see `source/envs/__init__.py`.

The dependency arrows point one way: metrics may use algorithms (dormancy is
scored with ReDo's own criterion, deliberately), the studies may use anything,
envs depend on none of them, and nothing in algorithms imports from metrics.

Was `source/` until 2026-09-08. It stopped being 'the shared part' when
the per-suite trainer directories stopped being the main event: these ARE the
methods and the measurements, so they sit at the top rather than inside a
package named for the fact that more than one caller uses them.
"""
