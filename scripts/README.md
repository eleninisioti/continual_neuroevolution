# scripts

Run everything from the repository root.

| Path | Contents |
|---|---|
| `train/run.sh` | runs one or more experiment settings over the GPUs you give it |
| `train/queue_toy.sh`, `train/toy_sweep.py` | the toy-landscape runs: the queue of every stage, and the sweep it calls |
| `analysis/` | post-hoc passes over saved runs, and the paper's figure scripts |
| `plotting/` | shared plotting and metric-table code the figure scripts import (`make_lineplot.py`, `make_metrics_figure.py`, `make_plasticity_figure.py`), and `make_toy_figures.py`, which `train/queue_toy.sh` runs on each toy stage |
| `verify_runs.py` | checks a run tree is complete and that every method saw the same steps and task boundaries |

## Post-hoc passes

Some figure columns are not functions of a training curve and need a pass over
the saved checkpoints first. All are CPU-only and skip work already done.

| Script | Produces |
|---|---|
| `analysis/evaluate_continual.py` | zero-shot transfer (`evaluation.json` beside each run) |
| `analysis/behavioural_divergence.py` | forgetting and behavioural divergence (the T x T rollout sweep) |
| `analysis/plasticity_checkpoints.py`, `analysis/merge_plasticity_checkpoints.py` | dormancy, weight and NTK-rank diagnostics from `checkpoints.npz` |
| `analysis/curvature_width.py` | basin width around each saved agent |
| `analysis/child_survival.py` | how many of a saved agent's Gaussian children stay as good (basin-width return tables) |
| `analysis/generalist_checkpoints.py` | Found / Held / Retention per phase, from `evaluation.json` |
| `analysis/build_freq_matched.py` | the budget-matched trees for the switch-frequency appendix |

## Figures

Most figure scripts work in two steps: `--extract` reads the runs and writes
the numbers the figure needs, and the same command without `--extract` redraws
from those numbers alone. `--help` lists each script's options.

| Paper figure | Script |
|---|---|
| continual curves and trade-off (`continual_tradeoff`, `continual_curves`) | `analysis/plot_continual_combined.py` (uses `plot_continual_lineplots.py`, `plot_stability_plasticity.py`) |
| stationary tasks (`noncontinual`) | `analysis/plot_noncontinual_solve.py` |
| continual metrics (`metrics_main`, `metrics_continual`) | `analysis/plot_metrics_appendix.py` |
| plasticity (`plasticity_loss`, `plasticity_all`), `tab:dormant_init` | `analysis/plot_plasticity_overview.py`, `analysis/dormancy_at_init.py` |
| novelty search (`novelty_lineplots`, `novelty_lineplots_all`) | `analysis/plot_novelty_lineplots.py` |
| population diversity (`population_diversity_continual`) | `analysis/plot_population_diversity.py` |
| generalist scores (`generalist_scores_centroid_grid`) | `analysis/plot_generalist_scores.py` |
| basin width (`basin_width_return`, `basin_width_table`, `basin_width_return_table`) | `analysis/plot_basin_width_methods.py`, `analysis/plot_basin_width_return.py` |
| basin width, appendix (`basin_width_curves`, `basin_width_direction`, `basin_width_evolution`) | `analysis/plot_basin_width_{levels,direction,evolution}.py` |
| shared basin (`shared_basin_main`, `shared_basin`) | `analysis/plot_shared_basin.py` |
| return landscapes (`landscape_examples`) | `analysis/plot_landscape_examples.py` (uses `landscape_slices.py`) |
| toy landscapes (`landscapes`, `toy_local_optima`, `toy_local_optima_dims`) | `analysis/plot_toy_sigma_basin.py`, `analysis/plot_toy_local_optima.py` |
| switch frequency (`frequency`) | `analysis/plot_frequency.py` |
| hyperparameters (`hparam_sigma`, `hparam_sigma_curves`, `hparam_minibatches`, `hparam_minibatches_plane`) | `analysis/plot_hparam_sigma.py`, `analysis/plot_hparam_updates.py` |
