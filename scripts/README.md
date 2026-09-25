# scripts

Four scripts, and an archive.

| | |
|---|---|
| `verify_runs.py` | is a run tree fit to put in a figure? |
| `make_lineplot.py` | the paper's training lineplot **and its metric table** |
| `make_plasticity_figure.py` | the paper's plasticity figure, the one that goes after the lineplot and its table |
| `check_imports.py` | do this repo's cross-module references still resolve? |
| `train/launch.sh` | spread one experiment block over every GPU |
| `train/run_experiments.sh` | the block definitions `launch.sh` drives |
| `analysis/behavioural_divergence.py` | the post-hoc sweep that produces F and BD |
| `analysis/plasticity_checkpoints.py` | the plasticity columns that need a saved AGENT rather than a curve |
| `outdated/` | everything else that was here before 2026-09-08 |

Anything moved out of `outdated/` is something we actually run. Their repo-root
search walks up to `pyproject.toml` rather than counting `dirname`s, because
counting broke every time a file moved — once silently (a `runs_root` pointing
inside `scripts/`) and once loudly (60 jobs exiting 127 before anything
trained).

## Completing the metric table

Three of the six columns are not functions of a training curve, so they need
two passes over the saved checkpoints before `make_lineplot.py` can report
them:

```bash
# ZT -- writes evaluation.json beside each run
.venv/bin/python -m source.studies.evaluate_continual \
    --root projects/iclr_2027/runs/gymnax/continual --episodes 100

# F and BD -- the T x T rollout sweep
.venv/bin/python scripts/analysis/behavioural_divergence.py \
    --runs_root projects/iclr_2027/runs \
    --methods nes es ga ga_refresh dns ppo trac redo cchain \
    --envs CartPole_v1_sigma1.0 Acrobot_v1_sigma1.0 MountainCar_v0_sigma1.0 \
    --num_tasks 20 --episodes 20 --num_states 2000 \
    --results_dir projects/iclr_2027/results/gymnax_continual/sigma1.0
```

Then pass `--results-dir projects/iclr_2027/results/gymnax_continual/sigma1.0`
to `make_lineplot.py`. ZT needs no flag — it is read from each run's own
`evaluation.json`. Both passes are CPU-only and skip work already done, so they
can run beside GPU training and be re-run safely.

`outdated/` is the previous study's tooling, moved wholesale rather than
deleted because the analysis in it is the only description of how several
published numbers were produced. It still runs — module paths inside it were
rewritten, so `scripts.generalists.x` is now `scripts.outdated.generalists.x` —
but nothing new should be added to it. **Training still runs from there** until
a clean launcher replaces it:

```bash
.venv/bin/python -m scripts.outdated.generalists.train.train_all --help
```

## The training lineplot, gymnax

The figure is: every method's learning curve, one panel per task, a dashed rule
at each task boundary. Two commands, from the repo root.

**Continual, all nine methods, 20 sub-tasks, at observation-offset sigma 1.0:**

```bash
.venv/bin/python scripts/make_lineplot.py projects/iclr_2027/runs/gymnax \
    --phase continual --sigma 1.0 \
    --out projects/iclr_2027/figures/continual_gymnax_sigma1.0
```

**The stationary control:**

```bash
.venv/bin/python scripts/make_lineplot.py projects/iclr_2027/runs/gymnax \
    --phase noncontinual --assume-episode-length 500 \
    --out projects/iclr_2027/figures/noncontinual_gymnax
```

Each writes from the `--out` stem: `.png`, `.pdf`, a separate
`_legend.png`/`.pdf`, and `_table.md` — **the paper's metric table**, not just
the figure's numbers:

| column | source |
|---|---|
| Cum. max | area under the plotted curve, reward x generations /1000 |
| Cum. mean | the same integral of the population-average curve; `--` for the single-policy RL methods |
| FT | mean sub-task window mean minus that method's own stationary run |

with one-sided Mann-Whitney U marks (NE vs RL, Holm-Bonferroni corrected,
`*` `**` `***`). The definitions live in
[`source/metrics/continual_metrics.py`](../source/metrics/continual_metrics.py),
extracted from `compare.py` so the figure and the table cannot define a column
differently.

FT needs each method's stationary run: it uses the `noncontinual` phase of the
same tree, or of `--ref-root <tree>`. `ga_refresh`'s reference is plain `ga` —
there is no boundary to re-score at in a stationary run.

**F, BD and ZT are not in it**, and that is not an omission to paper over: they
are not functions of a training curve. They come from re-rolling every
sub-task's saved agent against every sub-task, which
`source.studies.evaluate_continual.py` writes into `evaluation.json`,
and that pass has not been run over these trees. The table says so on every
build.

The other sigmas are `--sigma 0.02` and `--sigma 2.0`; the directory names in
the run tree are the authority on what exists.

### Useful flags

| flag | why |
|---|---|
| `--width 6.9` | figure width in **inches**, default 17.5 cm = A4 text width at 2 cm margins |
| `--panel-height 1.9` | height of the panel row, inches |
| `--font-size 7` | base point size **at 1:1** |
| `--legend separate` | default: the legend is its own `<stem>_legend.png/.pdf`. `inline` puts it above the panels |
| `--legend-ncol` | legend columns; default is one row |
| `--metric` | which column: `current` (default), `popmean_score`, `best_so_far`, `centroid`, `popmean` |
| `--smooth 1` | the raw series; default is a rolling median over 1% of the run |
| `--band sd` | mean ± 1 s.d. instead of median with an IQR band |
| `--x steps` | environment steps on the axis instead of generations |
| `--include-superseded` | also plot arms the script drops by default (see below) |

### Size and the legend

The defaults produce a **17.5 × 4.8 cm** figure with **7 pt** type, to be
included at **1:1**. Drawing a figure wide and scaling it down in LaTeX scales
its fonts down too, which is how 10 pt becomes unreadable 5 pt on the page.

The legend is written separately by default, at exactly `--width` and with no
tight bounding box, so that including the panels and the legend at the same
width keeps their type at the same size. One legend can then caption several
stacked figures.

### Arms the script will not plot by default

`ga` is dropped: its elite archive was never re-scored after a task switch, so
selection compared fresh offspring against fitness measured on the *previous*
sub-task. `ga_refresh` is the corrected arm and is the one labelled **GA**.
`--include-superseded` puts the old one back, labelled as superseded.

`cchain` is plotted but warned about on every run: it ran at
`chain_target_rel_scale` 0.1 against the reference implementation's 10000, so
those rows do not test C-CHAIN as published. There is no corrected arm to
substitute, which is why it is a warning and not an exclusion.

### Two things the script refuses to do quietly

**It will not mix a per-record column with a best-so-far one.** Both families
log both. The NE trainers write `best_fitness` (this generation) *and*
`best_overall` (running maximum); the RL trainers write `mean_reward` (this
evaluation) *and* `best_reward` (running maximum). Plotting `best_fitness`
against `best_reward` gives a figure where NE dips at every boundary and RL is
a flat ceiling — which reads as a result about forgetting and is a units error.
`--metric current` resolves to `best_fitness` / `mean_reward`, and any series
that comes out monotone in a tree whose task *does* change is refused.

**It will not drop a method silently.** A method with no column for the chosen
metric is named in a warning, because a panel missing one line still looks like
a complete comparison.

## The plasticity figure

Goes after the lineplot and its table. Those say which methods keep learning
across a task change; this says which of the mechanisms the plasticity
literature blames for not keeping it are actually present, for both families
under one definition each.

Two commands. The first is a post-hoc pass and only has to be run once per
tree; it is CPU-only, takes a few minutes over the whole gymnax tree, and is
safe to run beside GPU training.

```bash
# per-unit dormancy masks, parameter distances and the empirical-Fisher rank,
# from checkpoints.npz -- writes plasticity_checkpoints.json
.venv/bin/python scripts/analysis/plasticity_checkpoints.py \
    --runs_root projects/iclr_2027/runs/gymnax --phase continual --sigma 1.0 \
    --out projects/iclr_2027/results/gymnax_continual/sigma1.0

# the figure
.venv/bin/python scripts/make_plasticity_figure.py projects/iclr_2027/runs/gymnax \
    --phase continual --sigma 1.0 \
    --checkpoints projects/iclr_2027/results/gymnax_continual/sigma1.0 \
    --out projects/iclr_2027/figures/plasticity_gymnax_sigma1.0
```

It writes from the `--out` stem, like the lineplot: `.png`, `.pdf`, a separate
`_legend`, a `_table.md`, and **a second figure, `_persistence`**.

| row | quantity | whose claim |
|---|---|---|
| Dormant units | fraction of the 32 hidden units at or below ReDo's tau on the frozen probe | Sokar et al. 2023 |
| Policy churn | fraction of probe states whose greedy action changed since the last record | Tang et al., C-CHAIN's symptom |
| Weight RMS | RMS of the parameter vector | Nikishin et al. 2022; Juliani & Ash 2024's strongest single predictor, and what TRAC holds down |
| NTK effective rank | effective rank of the empirical NTK Gram | Tang et al.'s *cause*: rank collapse -> correlated gradients -> churn |
| Step norm | distance the saved agent moved over one sub-task | Juliani & Ash's "weight difference" |

`--rows` picks; four more are defined and not drawn by default
(`pop_dormancy`, `churn_ce`, `weight_max`, `ntk_trace`, `drift`).

**Why these are comparable across the NE/RL divide.** The four dense rows are
read from `training_metrics.json`, where the trainers wrote them under one
definition for both families -- same ReDo tau and criterion, same frozen
512-state probe batch, same argmax-disagreement churn, and, because the gymnax
NE and RL trainers share `MLPPolicy`, literally the same 32 hidden units to be
dormant. Three caveats survive that and are printed on every run: the churn
clocks differ by 33% (one NE generation is 768,000 steps against one PPO
rollout's 1,024,000), GA's and DNS's saved agent is an argmax over a
population and can change lineage between records, and DNS's genotypes diverge
on this tree, which is why the weight rows are logarithmic and why a third of
its NTK records are null.

### The second figure: dead, or just sparse right now?

The dormant *fraction* cannot tell a network whose dead units are dead for
good from one that is merely sparse at any instant, using a different handful
of units each time -- and only the first is plasticity loss. `_persistence`
separates them from the per-unit masks, as

    (P(dormant at t+k | dormant at t) - p) / (1 - p)

with `p` the method's own instant dormant fraction, which is exactly the
chance level. **0 is functional sparsity, 1 is capacity that has gone.** Its
resolution is one checkpoint per sub-task, so it is a lower bound on
transience -- read `analysis/plasticity_checkpoints.py`'s docstring before
quoting a number from it.

## Checking a run tree first

```bash
.venv/bin/python scripts/verify_runs.py projects/iclr_2027/runs/gymnax --phase continual
```

Per (method, task) cell it checks that every trial is present and parses, that
all methods saw the **same number of environment steps**, and that their task
boundaries fall at the **same step** — CLAUDE.md rule (c). It derives both from
each run's own recorded config and never from a table of environment defaults.

What it cannot check is whether a method is the method it claims to be: a
mis-set coefficient produces a perfectly well-formed run. Those caveats live in
`projects/iclr_2027/README.md`.
