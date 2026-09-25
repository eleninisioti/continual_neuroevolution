"""The paper's training lineplot: every method's learning curve, one panel per task.

Follows the convention already set by
`projects/neurips_2026_rebuttal/results_repro2/*/continual_gymnax.png`, because
a paper's figures have to look like each other: the method palette and labels
of `scripts/outdated/compare.py`, one horizontal legend above the panels, bold
short task titles, "Episodic reward" against "Generations", dashed rules at the
task boundaries, and a `.png` beside a `.pdf` beside a `_table.md`.

    .venv/bin/python scripts/make_lineplot.py projects/iclr_2027/runs/gymnax \\
        --phase continual --sigma 1.0 \\
        --out projects/iclr_2027/figures/gymnax_continual_sigma1.0

`--out` is a stem; the three files are written beside it.

## Which column is plotted, and the trap in it

Two generations of trainer wrote these trees, they do not share a column name,
and -- the part that matters -- each records BOTH a per-record value and a
best-so-far value. Plotting one family's per-record value against the other's
best-so-far produces a figure where one family dips at every task boundary and
the other is a flat ceiling. That is not a result, it is a units error, and it
is what the first version of this script did.

So `--metric` is a SEMANTIC name, resolved per run:

    current  what the agent scores NOW, which is the learning curve.
             NE: `best_fitness`, the best individual of this generation on the
             active sub-task. RL: `mean_reward`, the agent's evaluation at this
             record. (`best_reward` is NOT the RL analogue of `best_fitness` --
             it is best-so-far. See `--allow-monotone`.)
    elite_eval  the same genome as `current`, re-scored on FRESH keys over
             --report_episodes episodes: `elite_eval_fitness` / `mean_reward`.
             This is the unbiased reading of "what the best individual
             scores"; `current` is not, because it is a max over the
             population of the search's own noisy draw.
    popmean_score  the population's mean SCORE: `mean_fitness` / `mean_reward`.
             Not the score of the mean weights -- that is `popmean` below.
    best_so_far  the running maximum: `best_overall` / `best_reward`. Monotone
             by construction, so it cannot show forgetting; useful only to say
             what a run ever reached.
    centroid the score of the coordinate-wise mean of the population's
             WEIGHTS. `centroid_fitness` in the per-method NE trainers,
             `popmean_generalist` in the generalists ones. Equal to the
             incumbent for a distribution-based search -- for ES and NES the
             distribution mean IS the point handed back -- and different for
             GA and DNS, where the gap measures whether the archive has
             collapsed onto one solution.
    incumbent what the search HANDS BACK, on a HELD-OUT protocol across every
             sub-task (`centroid_generalist`). Generalists trainers only.

The axis label and the caption name the column actually read, any method
lacking it is reported loudly rather than dropped quietly, and a series that
comes out monotone under a non-`best_so_far` metric is refused -- that is the
signature of having picked a best-so-far column by mistake.

A low `centroid` on GA or DNS is evidence of a SPREAD archive, not of a bad
search: averaging the weights of two networks that differ by a permutation of
their hidden units gives a network worse than both. DNS keeps its repertoire
diverse on purpose, so expect the widest gap there.

## The band

The line is the MEAN over seeds and the band its 95% percentile-bootstrap
confidence interval (`--band ci`, the default; `bootstrap_ci` in
`source/metrics/continual_metrics.py`, the same helper the metrics figure
uses). Not the median: the per-seed outcomes on these tasks are bimodal -- a
PPO seed on CartPole scores 500 on a sub-task or 9 -- and a median over 10
seeds flips to the ceiling as soon as 6 solve, with the failed seeds hidden
under its interquartile band. Between 2026-09-08 and 2026-09-10 this script
drew that median, and the RL arms looked a full ceiling better on CartPole
than the table beside them, which averages over trials, said they were. The
mean is the only summary that still encodes how many seeds failed, and it is
what every column of the table reports, so the figure and the table now
aggregate the same way. `--band sd` and `--band iqr` remain for the appendix.

## The x axis

"Generations" is the label; the quantity underneath is ENVIRONMENT STEPS,
rescaled by one NE generation's worth of them. An NE generation costs
`pop x evals x episode_length` steps and a PPO update costs
`num_envs x num_steps`, so the families only line up once both are in steps --
and they line up here only because every method in these cells spends the same
total (CLAUDE.md rule (c), which `scripts/verify_runs.py` checks). `--x steps`
shows the raw axis.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import textwrap
from collections import defaultdict

import numpy as np

# The metric definitions live in source/, not here, so the figure script and
# anything else that reports these columns cannot drift apart.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from source.metrics.continual_metrics import (            # noqa: E402
    bootstrap_ci, cumulative_reward, forward_transfer_trials,
    mann_whitney_marks)

# The paper's palette and labels, from scripts/outdated/compare.py METHOD_STYLE.
# Deliberately kept in step with it: a method must be the same colour in every
# figure or the reader re-learns the legend on each one.
METHOD_STYLE = {
    # `ga_refresh` is THE GA and is labelled as such: it re-evaluates its elite
    # archive every generation out of the same budget, so selection compares
    # like with like. `ga` is the same trainer without that, which left elites
    # carrying fitness scored on the PREVIOUS sub-task -- at the first boundary
    # a stored 500 against an actual 30 -- so its selection is against stale
    # numbers. It is superseded, not a variant, and is excluded by default.
    'ga':         {'label': 'GA',                            'color': '#4CBB47'},
    'ga_refresh': {'label': 'GA (refresh, historical name)',  'color': '#9ED89B'},
    'ga_reeval':  {'label': 'GA (re-scored at boundaries)',   'color': '#7FC97F'},
    'dns':        {'label': 'GA + Novelty (Iso+LineDD)',     'color': '#3B8FD4'},
    # The operator ablation. `dns` breeds with Iso+LineDD and `ga_refresh` with
    # gaussian mutation, so a DNS-over-GA gap is a claim about novelty
    # selection AND about recombination. These two cross the pair: same
    # selection rule as their name's method, the other one's operator.
    # Reading the figure: dns vs ga_isoline isolates the selection rule with
    # the operator held at Iso+LineDD, ga_refresh vs dns_gaussian isolates it
    # with the operator held at gaussian mutation, and the two comparisons
    # agreeing is what makes "diversity is what does it" a supported claim.
    'ga_isoline': {'label': 'GA (Iso+LineDD)',               'color': '#7FD4C0'},
    'dns_gaussian': {'label': 'GA + Novelty',                'color': '#8FB8E8'},
    'es':         {'label': 'ES',                            'color': '#F0C33C'},
    'nes':        {'label': 'NES',                           'color': '#C99A18'},
    'ppo':        {'label': 'PPO',                           'color': '#E8504F'},
    'trac':       {'label': 'TRAC-PPO',                      'color': '#F08C4B'},
    'redo':       {'label': 'ReDo-PPO',                      'color': '#8B4A2B'},
    'cchain':     {'label': 'C-CHAIN',                       'color': '#7B5EA7'},
    # PBT is a PPO variant, so it sits in the RL arms' warm family (it was teal,
    # which read as a GA shade); raspberry and its light tint, chosen for
    # separation from PPO/ReDo/C-CHAIN under protan and deutan simulation.
    'pbt':        {'label': 'PBT-PPO (N=8)',                 'color': '#B5306B'},
    'pbt2':       {'label': 'PBT-PPO (N=2)',                 'color': '#F4A6C6'},
    # The same populations without explore (--pbt_mode weights_only).
    'pbt_weights':  {'label': 'PBT-PPO (N=8, weights only)', 'color': '#7A1F47'},
    'pbt2_weights': {'label': 'PBT-PPO (N=2, weights only)', 'color': '#D96A9E'},
}
METHOD_ORDER = ['ga', 'ga_refresh', 'ga_reeval', 'ga_isoline', 'dns',
                'dns_gaussian', 'es', 'nes',
                'ppo', 'trac', 'redo', 'cchain', 'pbt', 'pbt2',
                'pbt_weights', 'pbt2_weights']

# Arms whose data does not measure what their name says.
#
# KEYED ON WHAT THE RUN RECORDED, not on the method name, because the same
# name now covers both a broken and a fixed run: `ga` froze its archive until
# 2026-09-08 and re-scores it unconditionally after, and `cchain` ran at a
# pinned coefficient in the reported tree and at the reference's value since.
# A name-keyed list would drop the fixed runs along with the broken ones, and
# it would do it silently. Each entry is (predicate over the run's config,
# reason); a cell is affected only if EVERY trial in it satisfies the
# predicate, and mixed cells are reported as mixed rather than resolved
# either way.
def _ga_archive_frozen(cfg):
    # The fixed trainer records `refresh_archive: True`. The broken runs
    # predate the field entirely, so absence IS the defect -- but only under a
    # SWITCHING schedule: the defect is an elite keeping fitness measured on
    # the previous sub-task, and a stationary run has no previous sub-task.
    # The stationary trainer is a plain evosax SimpleGA and records nothing
    # here, so without the phase gate at the call site every `noncontinual`
    # GA would be dropped as superseded for a bug it cannot have.
    #
    # The shared runners (MiniGrid, mjx, kinetix) record the same fact under
    # `searcher_resolved.refresh` -- their GA has re-scored its archive since
    # 2026-08-27 and `ga_stale` is the frozen one -- so a run carrying that
    # block is judged by it. Reading `refresh_archive` alone dropped every
    # shared-runner GA from the continual figures (found on cheetah,
    # 2026-09-11; MiniGrid had been drawn without its GA too).
    resolved = cfg.get('searcher_resolved') or {}
    if 'refresh' in resolved:
        return not bool(resolved['refresh'])
    return not cfg.get('refresh_archive', False)


def _cchain_coef_pinned(cfg):
    # The categorical controller floors the coefficient at 1, so a target
    # scale below 1 pins it there. The brax (continuous-action) controller has
    # no floor (`chain_floor` 0) and 0.05 is the reference's DMC value.
    scale = cfg.get('chain_target_rel_scale')
    return (scale is not None and float(scale) < 1.0
            and float(cfg.get('chain_floor', 1.0)) > 0)


# Excluded unless asked for, because a figure that shows them is making a
# claim about a method it did not run. `--include-superseded` puts them back.
# `phases` is which settings the defect can occur in; a run in any other
# phase is never flagged.
SUPERSEDED = {
    'ga': (_ga_archive_frozen, ('continual',),
           'archive was never re-scored after a task switch, so selection '
           'compared against fitness from the previous sub-task (no '
           '`refresh_archive` in the run config)'),
}
# Arms that ran, but not at the published method's settings. Kept -- there is
# no corrected arm to replace them with -- and warned about every time.
MISLABELLED = {
    'cchain': (_cchain_coef_pinned, ('continual', 'noncontinual'),
               'chain_target_rel_scale was below 1 against the reference '
               "implementation's 10000, so these rows do not test C-CHAIN as "
               'published'),
}


def defect_state(root, phase, method, cell_filter, predicate):
    """'all' | 'none' | 'mixed' -- how many of a method's trials are defective.

    Reads the same `results.json`/`config.json` the budget does, so a run that
    records nothing about the setting counts as defective when absence is what
    the predicate tests for.
    """
    hits = total = 0
    method_dir = root / phase / method
    if not method_dir.is_dir():
        return 'none'
    for cell_dir in sorted(p for p in method_dir.iterdir() if p.is_dir()):
        if not cell_filter(cell_dir.name):
            continue
        for trial_dir in sorted(cell_dir.glob('trial_*')):
            cfg, _res = load_config(trial_dir)
            if cfg is None:
                continue
            total += 1
            hits += bool(predicate(cfg))
    if not total:
        return 'none'
    return 'all' if hits == total else ('none' if hits == 0 else 'mixed')

ENV_TITLES = {
    **{f'DeepSea{n}_bsuite': f'DeepSea {n}' for n in (8, 10, 12, 14, 16, 20)},
    'CartPole_v1': 'CartPole',
    'Acrobot_v1': 'Acrobot',
    'MountainCar_v0': 'MountainCar',
    # MiniGrid: the cell IS the room pair (no sigma in the name), so the title
    # says which rooms alternate. Stationary cells are one room each.
    'MiniGrid_8x8_16x16': 'MiniGrid 8x8 / 16x16',
    'MiniGrid_8x8': 'MiniGrid 8x8',
    'MiniGrid_16x16': 'MiniGrid 16x16',
    # The mjx bodies (source/studies/mjx/settings.py): two continual families
    # per body, `<body>_noise` (observation offset, sigma 2.0) and
    # `<body>_friction` (ground-friction multiplier), and ONE stationary cell
    # `<body>` that serves both, sub-task 0 being the stock body in either.
    'ant': 'Ant',
    'ant_noise': 'Ant, observation offset',
    'ant_friction': 'Ant, friction',
    'ant_speed': 'Ant, target speed (2 vs 8 m/s)',
    'ant_speed24': 'Ant, target speed (2 vs 4 m/s)',
    'cheetah': 'Cheetah',
    'cheetah_noise': 'Cheetah, observation offset',
    'cheetah_friction': 'Cheetah, friction',
    'ant_action': 'Ant, action reversal',
    'cheetah_action': 'Cheetah, action reversal',
}
# The stationary cell a continual cell's FT column subtracts, where the two
# are not named alike. gymnax's are (`CartPole_v1_sigma1.0` -> `CartPole_v1`
# by stripping the sigma); the mjx families share one stationary body.
REF_ENV = {
    'ant_noise': 'ant', 'ant_friction': 'ant',
    # Sub-task 0 of the target-speed family is the stock body at 2 m/s, the
    # same cell `ant` is.
    'ant_speed': 'ant',
    'ant_speed24': 'ant',
    'cheetah_noise': 'cheetah', 'cheetah_friction': 'cheetah',
    'ant_action': 'ant', 'cheetah_action': 'cheetah',
}
# Kinetix: the continual cell is the chain over all twenty medium levels and
# the stationary cells are one level each, `Kinetix_<level>`. Listed from the
# jax-free level module, in the chain's order, so twenty panels come out as
# h0 .. h19 rather than the string sort's h0, h1, h10, h11, ...
from source.envs.kinetix_levels import LEVELS as _KINETIX_LEVELS  # noqa: E402
ENV_TITLES['Kinetix20'] = 'Kinetix, 20 levels'
ENV_TITLES.update({f'Kinetix_{lvl}': lvl.replace('_', ' ')
                   for lvl in _KINETIX_LEVELS})

# Semantic metric -> candidate columns, most specific first. See the docstring
# on why `current` must not fall back to `best_reward`.
METRIC_COLUMNS = {
    'current':       ['best_fitness', 'mean_reward'],
    'popmean_score': ['mean_fitness', 'mean_reward'],
    'best_so_far':   ['best_overall', 'best_reward'],
    # The CENTROID: the score of the coordinate-wise mean of the population's
    # WEIGHTS. Two spellings because two trainers record it under two
    # evaluation protocols -- `centroid_fitness` is the per-method gymnax
    # trainers' active-sub-task score, `popmean_{agg}` the generalists
    # trainers' held-out one -- but it is the same genome either way, and the
    # RL arms have no population so they fall back to their own curve.
    'centroid':      ['centroid_fitness', 'popmean_{agg}', 'mean_reward'],
    # The ELITE, re-scored OUT OF SAMPLE. Same genome `current` names, but
    # evaluated on fresh keys over --report_episodes episodes instead of being
    # read off the search's own draw, so it does not carry the winner's curse
    # that `best_fitness` does -- a max over pop_size noisy means, inflated by
    # +5 for the GA and +53 for DNS on the sigma=1.0 tree, i.e. by an amount
    # that differs per method and does not cancel in a comparison. The RL arms
    # have no population, and `mean_reward` is ALREADY this protocol at the
    # same episode count, so they fall back to it and the two families are one
    # estimator. Runs made before 2026-09-08 do not have the column; they fall
    # back to `best_fitness` and the figure says so.
    'elite_eval':    ['elite_eval_fitness', 'mean_reward'],
    # What the search HANDS BACK, held out across every sub-task. Generalists
    # trainers only, and NOT the same thing as `centroid` on GA or DNS.
    'incumbent':     ['centroid_{agg}'],
}
MONOTONE_OK = {'best_so_far'}


# The table's headline column is an integral of whatever curve `--metric`
# selected, so its NAME has to follow that metric. It used to be the literal
# string "Cum. max" for every metric, which under `--metric centroid` labelled
# the score of the population's mean WEIGHTS as a maximum -- of nothing. The
# two reported directories are meant to be read side by side, so a reader who
# takes both tables' first column for the same quantity is exactly the error
# the elite/centroid split exists to prevent.
METRIC_TABLE_LABEL = {
    'current':       'Cum. max',        # a max over the generation: correct
    'best_so_far':   'Cum. best-so-far',
    'elite_eval':    'Cum. elite',
    'centroid':      'Cum. centroid',
    'popmean_score': 'Cum. pop. score',
    'incumbent':     'Cum. incumbent',
}


# The reported gymnax grid uses a DIFFERENT observation noise per environment
# (CartPole and Acrobot at sigma 1.0, MountainCar at sigma 0.1 -- sigma 1.0 is
# 7-15x MountainCar's velocity range, so that column was noise rather than a
# perturbation). `--sigma` cannot express that: it is one suffix for the whole
# figure, so it splits the three panels across two files that cannot be read
# as one result. `--cells` names the directories instead, one per environment,
# and every downstream pass takes the same `cell_filter`, so the boundary
# schedule, the table, the significance test and the post-hoc columns all see
# exactly the cells the panels do.
def cell_selector(cells, sigma):
    """`(filter, label)` for the cells a figure is drawn over."""
    if cells:
        wanted = set(cells)
        parts = []
        for c in cells:
            env, _, sig = c.partition('_sigma')
            parts.append(f"{ENV_TITLES.get(env, env)} sigma {sig or '--'}")
        return (lambda c: c in wanted), ', '.join(parts)
    if sigma:
        return (lambda c: c.endswith(f'_sigma{sigma}')), f'sigma={sigma}'
    return (lambda c: '_sigma' not in c), ''


def resolve_column(metric: str, aggregate: str, available,
                   columns=METRIC_COLUMNS) -> str | None:
    """The first of `columns[metric]`'s candidates this run actually has.

    `columns` is a parameter because the plasticity figure resolves a DIFFERENT
    table of semantic names (`scripts/make_plasticity_figure.py`) over the same
    trees, with the same per-family fallback and the same "say so when a method
    has no column" behaviour. One loader, two tables, rather than two loaders.
    """
    for candidate in columns[metric]:
        name = candidate.format(agg=aggregate)
        if name in available:
            return name
    return None


def smooth(curves: np.ndarray, window: int) -> np.ndarray:
    """Rolling median along the record axis, edges included.

    Median rather than mean: a continual curve collapses towards random at
    every boundary and recovers, so the series is full of genuine one-record
    steps. A mean smears them and makes each recovery look slower than it was.
    """
    if window <= 1:
        return curves
    pad = window // 2
    padded = np.pad(curves, ((0, 0), (pad, pad)), mode='edge')
    out = np.empty_like(curves)
    for i in range(curves.shape[1]):
        out[:, i] = np.median(padded[:, i:i + window], axis=1)
    return out


# ---------------------------------------------------------------------------
# The post-hoc columns. These are NOT functions of a training curve: they come
# from re-rolling each sub-task's saved agent, which two passes write.
#   evaluate_continual.py        -> <run>/evaluation.json         (ZT)
#   scripts/analysis/behavioural_divergence.py -> <results>/behavioural_divergence.json
#                                                                 (F, BD)
# ---------------------------------------------------------------------------

# Which saved agent the columns are read off. `finalgen` is the best member of
# a sub-task's final generation; the single-policy methods write `final`.
# Which saved agent the ZT column scores, per reported metric. `final` is the
# single-policy RL arms' only agent and stands in for both: it is at once the
# best individual and the centroid of a population of one.
ZT_SOURCES = {'elite_eval': ('finalgen', 'final'),
              'centroid': ('centroid', 'final')}


def agent_sources_for(cell_dir, metric):
    """The saved agents ZT and Final are read for, matching the CURVE.

    Under `--metric elite_eval` the gymnax trainers' column is the best member
    of the generation re-scored, which is the `finalgen` checkpoint. On a
    shared-runner tree (marker `column_naming: gymnax_aliases_v1`, see
    `migrate_shared_runner_columns.py`) the same column is the INCUMBENT
    re-scored -- the best archive member on GA/DNS, the distribution mean on
    ES/NES -- so the post-hoc columns must read `incumbent` there, or an ES
    whose mean never solved a level reports its best sampled member as
    having done so. `centroid` is the same network on both.
    """
    sources = ZT_SOURCES.get(metric, ZT_SOURCES['elite_eval'])
    if metric == 'elite_eval':
        for trial_dir in sorted(cell_dir.glob('trial_*')):
            cfg, _ = load_config(trial_dir)
            if cfg:
                # The elite is the best-performing agent (2026-09-13). On a
                # shared-runner tree that is the incumbent for GA/DNS (the
                # best archive member) and PBT, but on NES/OpenES the
                # incumbent is the distribution mean, so their elite is the
                # saved best member of the phase's last generation (`finalgen`,
                # the default here), matching the gymnax trainers.
                if (cfg.get('column_naming') == 'gymnax_aliases_v1'
                        and cfg.get('method') not in ('openes', 'nes')):
                    sources = ('incumbent',) + tuple(sources)
                break
    return sources


def load_zero_shot(cell_dir, sources=ZT_SOURCES['elite_eval']):
    """`{trial_name: ZT}` for one (method, env) cell, or `{}`.

    ZT is the return a sub-task's own agent gets on the NEXT sub-task, before
    any search has been done on it, meaned over sub-tasks. `sources` is the
    preference order of saved agents to read it for; an NE run evaluated
    before `centroid` was scored has no entry under it and is left out, not
    silently read at the elite.
    """
    return load_per_task_mean(cell_dir, sources, 'zero_shot_next_returns')


def load_end_of_subtask(cell_dir, sources=ZT_SOURCES['elite_eval']):
    """`{trial_name: final performance}` for one (method, env) cell, or `{}`.

    Each sub-task's own agent, saved at the END of that sub-task, on that same
    sub-task, meaned over sub-tasks: how well every task was finally learnt,
    with no credit for speed and no charge for what is forgotten later. It is
    the diagonal of the forgetting pass's reward matrix, read here from the
    `evaluate` pass instead -- more episodes, and present on trees the
    forgetting pass has not reached. Same sources as ZT, so it follows the
    reported curve.
    """
    return load_per_task_mean(cell_dir, sources, 'returns')


def load_per_task_mean(cell_dir, sources, key):
    """`{trial_name: mean over sub-tasks of mean(entry[key])}` from each
    trial's `evaluation.json`, for the first of `sources` with any entry."""
    out = {}
    for trial_dir in sorted(cell_dir.glob('trial_*')):
        path = trial_dir / 'evaluation.json'
        if not path.exists():
            continue
        try:
            blob = json.loads(path.read_text())
        except Exception:                                # noqa: BLE001
            continue
        per_source = defaultdict(list)
        for entry in blob.get('per_task') or []:
            returns = entry.get(key)
            if returns:
                per_source[entry.get('source')].append(float(np.mean(returns)))
        for src in sources:
            if per_source.get(src):
                out[trial_dir.name] = float(np.mean(per_source[src]))
                break
    return out


def load_final_eval(cell_dir, sources=ZT_SOURCES['elite_eval']):
    """`{trial_name: mean return}` of the LAST phase's saved agent on its own
    sub-task, from `evaluation.json` -- the end-of-training evaluation at the
    pass's episode count and fresh keys. Same source preference as ZT."""
    out = {}
    for trial_dir in sorted(cell_dir.glob('trial_*')):
        path = trial_dir / 'evaluation.json'
        if not path.exists():
            continue
        try:
            blob = json.loads(path.read_text())
        except Exception:                                # noqa: BLE001
            continue
        last = {}
        for entry in blob.get('per_task') or []:
            src = entry.get('source')
            if entry.get('returns') and (src not in last or
                                         entry['task_idx'] > last[src][0]):
                last[src] = (entry['task_idx'], float(np.mean(entry['returns'])))
        for src in sources:
            if src in last:
                out[trial_dir.name] = last[src][1]
                break
    return out


def solved_thresholds(root, phase, cell_filter, override=None):
    """`{env: threshold or None}` for the `Solved` column: `--threshold` if
    given, else the run's own `solved_threshold` (Kinetix writes 1.0), else
    the registry's (the gymnax environments), else None -- MiniGrid carries
    none, so its threshold has to be passed."""
    out = {}
    base = root / phase
    if not base.is_dir():
        return out
    for method_dir in sorted(p for p in base.iterdir() if p.is_dir()):
        for cell_dir in sorted(p for p in method_dir.iterdir() if p.is_dir()):
            env = cell_dir.name.split('_sigma')[0]
            if not cell_filter(cell_dir.name) or env in out:
                continue
            if override is not None:
                out[env] = float(override)
                continue
            thr = None
            for trial_dir in sorted(cell_dir.glob('trial_*')):
                cfg, _ = load_config(trial_dir)
                if not cfg:
                    continue
                thr = cfg.get('solved_threshold')
                if thr is None:
                    try:
                        from source.envs.registry import threshold_for
                        thr = threshold_for(cfg['env'])
                    except Exception:                    # noqa: BLE001
                        thr = None
                break
            out[env] = None if thr is None else float(thr)
    return out


# The arrays a checkpoint saves that together identify a phase's sub-task:
# the observation offset (every suite; the level or room index on MiniGrid
# and Kinetix), the action-reversal flag and the physics multiplier.
TASK_SEQUENCE_ARRAYS = ('noise_vectors', 'action_flips', 'param_mults')


def distinct_subtasks(run_dir, num_phases=None):
    """How many DIFFERENT sub-tasks the run's first `num_phases` phases visit.

    None when the checkpoint is missing or saves no sequence, so the caller
    falls back to the many-sub-task definition rather than guessing.
    """
    path = pathlib.Path(run_dir) / 'checkpoints.npz'
    if not path.exists():
        return None
    try:
        with np.load(path) as z:
            cols = [np.asarray(z[k], dtype=float).reshape(len(z[k]), -1)
                    for k in TASK_SEQUENCE_ARRAYS if k in z.files]
    except Exception:                                    # noqa: BLE001
        return None
    if not cols:
        return None
    seq = np.concatenate(cols, axis=1)[:num_phases]
    return len(np.unique(seq, axis=0))


def load_divergence(results_dir):
    """`{run_dir: {'F': ..., 'BD': ...}}` from the divergence sweep's payload.

    Keyed on `run_dir` rather than on (method, env, trial), which is what the
    records themselves are selected on: the directory is the only identifier
    that cannot be ambiguous across sigmas and agent sources.
    """
    out = {}
    if results_dir is None:
        return out
    path = pathlib.Path(results_dir) / 'behavioural_divergence.json'
    if not path.exists():
        return out
    try:
        blob = json.loads(path.read_text())
    except Exception:                                    # noqa: BLE001
        return out
    for rec in blob.get('runs') or []:
        summary = rec.get('summary') or {}
        run_dir = rec.get('run_dir')
        if not run_dir:
            continue
        out[str(pathlib.Path(run_dir))] = {
            # F depends on how many DIFFERENT sub-tasks the run alternates:
            #  * more than two -- the literature's forgetting (Lopez-Paz &
            #    Ranzato 2017 backward transfer, Continual World's F): the
            #    END-of-sequence agent on every earlier sub-task, against the
            #    agent that had just been trained there.
            #  * two -- the switch forgetting: each sub-task's own agent
            #    against the agent after the NEXT switch, meaned over every
            #    switch. The final agent was always trained on the same one of
            #    the two, so the standard F would score forgetting of the other
            #    alone: on MiniGrid that is the 8x8 room after the 16x16, ~0
            #    for every arm, while the 16x16 after the 8x8 loses up to 0.73.
            #    With two sub-tasks this IS the standard definition, taken at
            #    every switch and in both directions. Since 2026-09-11.
            'F': summary.get(
                'switch_forgetting'
                if distinct_subtasks(run_dir, rec.get('num_tasks')) == 2
                else 'final_forgetting'),
            'BD': summary.get('consecutive_disagreement'),
        }
    # Continual World's average performance: the END-of-sequence agent on
    # every sub-task, meaned. Only the .npz carries the reward matrix
    # (`reward[i, j]` is agent j on sub-task i, so the last column is the
    # final agent), and it is keyed by the same run_dir as the JSON.
    npz = path.with_suffix('.npz')
    if npz.exists():
        try:
            archive = np.load(npz)
            for i, raw in enumerate(archive['index']):
                run_dir = json.loads(str(raw)).get('run_dir')
                key = f'run{i}_reward'
                if run_dir and key in archive.files:
                    rec = out.setdefault(str(pathlib.Path(run_dir)), {})
                    rec['FinalAll'] = float(np.mean(archive[key][:, -1]))
        except Exception:                                # noqa: BLE001
            pass
    return out


def load_config(trial_dir: pathlib.Path):
    """The run's settings: `results.json` nests them, `config.json` IS them."""
    for name in ('results.json', 'config.json'):
        path = trial_dir / name
        if path.exists():
            blob = json.loads(path.read_text())
            return blob.get('config') or blob, blob
    return None, None


def budget(cfg: dict, res: dict, assume_ep):
    """`(total_steps, steps_per_generation)`; the second is None for RL runs."""
    if 'num_timesteps' in cfg:
        return int(cfg['num_timesteps']), None
    pop = cfg.get('pop_size') or res.get('pop_size')
    evals = cfg.get('num_evals') or res.get('num_evals')
    ep = cfg.get('episode_length') or res.get('episode_length') or assume_ep
    gens = cfg.get('num_generations') or res.get('num_generations')
    if None in (pop, evals, ep, gens):
        return None, None
    per_gen = int(pop) * int(evals) * int(ep)
    return per_gen * int(gens), per_gen


_COLUMNS = {}


def load_columns(mpath):
    """`training_metrics.json` as `{column: float array}`, parsed once a process.

    A figure script reads the same tree several times (the curve, the
    population mean, the stationary reference, es_arm's picks), and the
    per-update RL records of the stationary MiniGrid tree alone are ~1 GB of
    JSON: kept as Python dicts, five passes held >100 GB and took most of an
    hour. Columns are the ones present in the first record; a `null` reads as
    NaN and its column is listed under `_null`. `{}` for an empty or unreadable
    file, which callers skip."""
    key = str(mpath)
    if key not in _COLUMNS:
        try:
            records = json.loads(pathlib.Path(mpath).read_text())
        except Exception:                                # noqa: BLE001
            records = None
        table = {}
        if records:
            nulls = set()
            for col, first in records[0].items():
                if first is not None and not isinstance(first, (int, float)):
                    continue
                values = [r.get(col) for r in records]
                if any(v is None for v in values):
                    nulls.add(col)
                try:
                    table[col] = np.array([np.nan if v is None else float(v)
                                           for v in values])
                except (TypeError, ValueError):
                    continue
            table['_null'] = nulls
        _COLUMNS[key] = table
    return _COLUMNS[key]


def collect(root, phase, cell_filter, metric, aggregate, assume_ep,
            columns=METRIC_COLUMNS, monotone_ok=MONOTONE_OK,
            allow_missing=False, methods=None):
    """`{env: {method: (x_steps, curves)}}`, plus diagnostics.

    `methods` reads only those arm directories (None = every one): a caller
    that needs two arms need not parse the RL arms' per-update records.

    `columns` and `monotone_ok` are what the plasticity figure varies: its
    metrics live in another table, and a weight norm that only ever grows is
    the RESULT there rather than the units error it would be here.

    `allow_missing` turns a `null` record into a NaN instead of a crash, and
    is OFF for the reward curves on purpose: an absent reward is a broken run
    and should stop the figure, whereas a diagnostic that declined to measure
    itself is a normal event -- DNS's NTK Gram overflows on about a third of
    its records and `source/metrics/ntk.py` returns None rather than taking the
    run down with it. Callers that pass True must aggregate with the nan-aware
    reducers; the count of NaNs comes back so it can be reported instead of
    quietly thinning a median.
    """
    out = defaultdict(dict)
    base = root / phase
    if not base.is_dir():
        sys.exit(f'no such directory: {base}')
    missing, used_columns, per_gen_seen = set(), set(), set()
    monotone, flat, unbudgeted = set(), [], set()
    # Per (method, column): how many trials were monotone, out of how many.
    mono_hits, mono_seen = {}, {}
    absent = defaultdict(int)
    for method_dir in sorted(p for p in base.iterdir() if p.is_dir()):
        method = method_dir.name
        if methods is not None and method not in methods:
            continue
        for cell_dir in sorted(p for p in method_dir.iterdir() if p.is_dir()):
            if not cell_filter(cell_dir.name):
                continue
            env = cell_dir.name.split('_sigma')[0]
            curves, x = [], None
            for trial_dir in sorted(cell_dir.glob('trial_*')):
                mpath = trial_dir / 'training_metrics.json'
                if not mpath.exists():
                    continue
                table = load_columns(mpath)
                if not table:
                    continue
                key = resolve_column(metric, aggregate, table, columns)
                if key is None:
                    missing.add((method, '/'.join(
                        c.format(agg=aggregate) for c in columns[metric])))
                    break
                cfg, res = load_config(trial_dir)
                if cfg is None:
                    unbudgeted.add((method, 'no results.json or config.json'))
                    continue
                total, per_gen = budget(cfg, res, assume_ep)
                if total is None:
                    # A trial whose budget cannot be computed used to be
                    # dropped HERE, silently, and a method all of whose trials
                    # hit this vanished from the figure with nothing said. It
                    # is how the stationary NE arms went missing: the
                    # noncontinual NE trainers recorded no `episode_length`,
                    # so `generations x pop x evals x episode_length` had a
                    # None in it. Reported now.
                    missing_field = next(
                        (k for k in ('pop_size', 'num_evals', 'episode_length',
                                     'num_generations')
                         if not (cfg.get(k) or (res or {}).get(k))), 'unknown')
                    unbudgeted.add((method, f'no {missing_field} recorded'))
                    continue
                if allow_missing:
                    series = table[key].copy()
                    n_absent = int(np.isnan(series).sum())
                    if n_absent:
                        absent[(method, key)] += n_absent
                    if n_absent == series.size:
                        continue
                else:
                    series = table[key].copy()
                    if key in table.get('_null', ()):
                        raise TypeError(f'{mpath}: null {key} in a reward column')
                # A CONSTANT series is a run that never moved -- a real result
                # about that trial, and it still belongs in the median. A
                # series that rises and never falls is something else: the
                # signature of a best-so-far column. Distinguish them, or the
                # second check fires on the first phenomenon.
                #
                # Counted per trial and judged per (method, column) below: a
                # best-so-far column is monotone in EVERY trial by
                # construction, whereas a learning curve that climbs to the
                # ceiling and is never knocked off it -- one ES trial on the
                # physics CartPole cell, whose sub-tasks are easy -- is
                # monotone in ONE trial and is a result, not a units error.
                if np.nanmax(series) == np.nanmin(series):
                    flat.append((method, cell_dir.name, trial_dir.name,
                                 float(np.nanmin(series))))
                elif metric not in monotone_ok:
                    mono_seen[(method, key)] = mono_seen.get((method, key), 0) + 1
                    if np.all(np.diff(series[~np.isnan(series)]) >= -1e-9):
                        mono_hits[(method, key)] = mono_hits.get((method, key), 0) + 1
                if per_gen:
                    per_gen_seen.add(per_gen)
                used_columns.add(key)
                curves.append(series)
                # Records are evenly spaced across the budget whatever the
                # trainer's own logging stride was.
                x = (np.arange(1, len(series) + 1) / len(series)) * total
            if not curves:
                continue
            n = min(len(c) for c in curves)
            out[env][method] = (x[:n], np.array([c[:n] for c in curves]))
    per_gen = sorted(per_gen_seen)[0] if per_gen_seen else None
    if absent:
        print('note: records whose diagnostic was not measured, counted as '
              'NaN and left out of the median rather than dropping the trial:')
        for (method, key), n in sorted(absent.items()):
            print(f'  {method}: {n} null {key}')
    if unbudgeted:
        print('WARNING: trials dropped because their environment-step budget '
              'could not be computed -- the x axis is in steps, so a run that '
              'cannot be priced cannot be plotted:')
        for method, why in sorted(unbudgeted):
            print(f'  {method}: {why} '
                  f'(--assume-episode-length fills a missing episode_length)')
    monotone = {k for k, n in mono_seen.items() if mono_hits.get(k, 0) == n}
    return out, missing, used_columns, per_gen, monotone, flat


def phase_edges(cfg, res, every, per, total):
    """`[0, ..., total]` in steps: where each phase starts, and where the last ends.

    Every phase is `every` long except, under `task_warmup`, the first, which
    holds sub-task 0 for that many generations (updates, for RL) before the
    switches start -- ten ordinary phases' worth on the ant warm-up trees. So
    the switches are the multiples of `every` only when there is no warm-up,
    and the boundary lines, the FT windows and the checkpoint positions read
    these edges instead of assuming it.
    """
    warmup = int(cfg.get('task_warmup') or res.get('task_warmup') or 0) * per
    if not warmup:
        return np.arange(int(round(total / every)) + 1) * float(every)
    n = int(round((total - warmup) / every))
    return np.concatenate([[0.0], warmup + every * np.arange(n + 1, dtype=float)])


def boundaries(root, phase, cell_filter, assume_ep):
    """`(every_n_steps, total_steps, phase_edges)` for the task switches."""
    base = root / phase
    for method_dir in sorted(p for p in base.iterdir() if p.is_dir()):
        for cell_dir in sorted(p for p in method_dir.iterdir() if p.is_dir()):
            if not cell_filter(cell_dir.name):
                continue
            for trial_dir in sorted(cell_dir.glob('trial_*')):
                cfg, res = load_config(trial_dir)
                if not cfg:
                    continue
                interval = cfg.get('task_interval') or res.get('task_interval')
                total, per_gen = budget(cfg, res, assume_ep)
                if not interval or total is None:
                    return None, None, None
                per = (int(cfg['num_envs']) * int(cfg['num_steps'])
                       if 'num_timesteps' in cfg else per_gen)
                every = int(interval) * per
                return every, total, phase_edges(cfg, res, every, per, total)
    return None, None, None


# Which family a method belongs to, for the significance test: each method is
# compared against every member of the OTHER family.
FAMILY = {'ga': 'ne', 'ga_refresh': 'ne', 'ga_reeval': 'ne', 'dns': 'ne',
          'ga_isoline': 'ne', 'dns_gaussian': 'ne',
          'es': 'ne', 'nes': 'ne',
          'ppo': 'rl', 'trac': 'rl', 'redo': 'rl', 'cchain': 'rl', 'pbt': 'rl', 'pbt2': 'rl',
          'pbt_weights': 'rl', 'pbt2_weights': 'rl'}

# FT needs each method's own STATIONARY run, and a few continual arms have no
# directory of that name in the noncontinual tree because they are a SETTING of
# a method rather than a method: `ga_refresh` is the GA with its archive
# re-scored every generation, and there is no boundary to re-score at in a
# stationary run, so its reference is plain `ga`. Mapping it is right; leaving
# FT blank for the one arm the paper calls "GA" would not be.
REF_METHOD = {'ga_refresh': 'ga', 'ga_reeval': 'ga'}
# `ga_isoline` is NOT mapped: it is a different search from `ga` (DNS's
# Iso+LineDD instead of gaussian mutation), so it gets its own stationary run
# and pointing it at the `ga` reference would charge the operator swap to FT.


# Columns that only a post-hoc evaluation pass can fill. Named here so the
# table says what is missing and why, rather than omitting them silently.
POSTHOC_COLUMNS = {
    'F': 'forgetting',
    'BD': 'behavioural divergence at the switch',
    'ZT': 'zero-shot transfer',
}


def posthoc_table(root, phase, cell_filter, results_dir, metric='elite_eval'):
    """`{env: {method: {column: [per-trial values]}}}` for the post-hoc columns.

    F, BD and FinalAll (the end agent on every sub-task) are the forgetting
    pass's; ZT, Final (the last agent on its own sub-task) and End (every
    sub-task's own agent at the end of it, meaned) the evaluation pass's.

    F and BD come from the divergence sweep in `results_dir`, which was run
    for ONE saved agent (`--agent_source`); ZT is read for the agent that
    matches `metric`. Pass the matching results directory, or the table mixes
    the elite's forgetting with the centroid's curve.
    """
    diverg = load_divergence(results_dir)
    out = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    base = root / phase
    if not base.is_dir():
        return out
    for method_dir in sorted(p for p in base.iterdir() if p.is_dir()):
        for cell_dir in sorted(p for p in method_dir.iterdir() if p.is_dir()):
            if not cell_filter(cell_dir.name):
                continue
            env = cell_dir.name.split('_sigma')[0]
            zt_sources = agent_sources_for(cell_dir, metric)
            zt = load_zero_shot(cell_dir, zt_sources)
            final = load_final_eval(cell_dir, zt_sources)
            end = load_end_of_subtask(cell_dir, zt_sources)
            for trial_dir in sorted(cell_dir.glob('trial_*')):
                rec = diverg.get(str(pathlib.Path(*trial_dir.parts)))
                if rec:
                    for key in ('F', 'BD', 'FinalAll'):
                        if rec.get(key) is not None:
                            out[env][method_dir.name][key].append(float(rec[key]))
                if trial_dir.name in zt:
                    out[env][method_dir.name]['ZT'].append(zt[trial_dir.name])
                if trial_dir.name in final:
                    out[env][method_dir.name]['Final'].append(final[trial_dir.name])
                if trial_dir.name in end:
                    out[env][method_dir.name]['End'].append(end[trial_dir.name])
    return out


def metric_table(data, pop_data, ref_data, per_gen, edges, args):
    """Cum. max, Cum. mean and FT per (method, env), with significance marks.

    All three are integrals or window means of a training curve, so they come
    from the same load the figure used. F, BD and ZT do not -- see
    POSTHOC_COLUMNS.
    """
    envs = [e for e in ENV_TITLES if e in data] + \
           [e for e in sorted(data) if e not in ENV_TITLES]
    rows = {}
    for env in envs:
        gens = None
        cum_max, cum_mean, ft = {}, {}, {}
        for method, (x, curves) in data[env].items():
            xg = x / per_gen if per_gen else x
            gens = float(xg[-1]) if gens is None else gens
            cum_max[method] = [cumulative_reward(xg, c, xg[-1]) / 1e3
                               for c in curves]
            pop = pop_data.get(env, {}).get(method)
            if pop is not None:
                px, pcurves = pop
                pxg = px / per_gen if per_gen else px
                cum_mean[method] = [cumulative_reward(pxg, c, pxg[-1]) / 1e3
                                    for c in pcurves]
            ref_env = REF_ENV.get(env, env)
            ref = ref_data.get(ref_env, {}).get(method)
            if ref is None:
                ref = ref_data.get(ref_env, {}).get(REF_METHOD.get(method))
            if ref is not None and edges is not None:
                rx, rcurves = ref
                v = forward_transfer_trials(
                    ref=[(rx / per_gen if per_gen else rx, c) for c in rcurves],
                    cont=[(xg, c) for c in curves],
                    edges=edges / per_gen if per_gen else edges)
                if v.size:
                    ft[method] = list(v)
        rows[env] = {'cum_max': cum_max, 'cum_mean': cum_mean, 'ft': ft,
                     'gens': gens}
    return rows, envs


# Above this many environments the metric table is written one column at a
# time with the environments down the rows (see `write_table`).
WIDE_TABLE_ENVS = 4


def _fmt(values, marks, method, digits=1):
    if not values:
        return '--'
    v = [x for x in values if np.isfinite(x)]
    if not v:
        return '--'
    return f"{np.mean(v):.{digits}f}{marks.get(method, '')}"


def _digits(column):
    """Decimals for a table column: 1 where the reward scale is tens or
    hundreds (gymnax), 3 where it is [0, 1] (MiniGrid), decided by the
    column's largest mean so every method in it is written alike."""
    means = [abs(np.mean([x for x in v if np.isfinite(x)]))
             for v in column.values() if v and np.isfinite(v).any()]
    return 1 if (means and max(means) >= 10) else 3


def write_table(path, data, pop_data, ref_data, posthoc, used_columns, per_gen,
                edges, args, thresholds=None):
    """The paper's metric table, beside the figure it belongs to."""
    rows, envs = metric_table(data, pop_data, ref_data, per_gen, edges,
                              args)
    col = ', '.join(sorted(used_columns))
    headline = METRIC_TABLE_LABEL[args.metric]
    missing_ref = not any(ref_data.get(REF_ENV.get(e, e)) for e in envs)

    out = [f'# {path.stem}', '',
           f'`{col}` &middot; {args.phase}'
           + (f' &middot; {cell_selector(args.cells, args.sigma)[1]}'
              if (args.cells or args.sigma) else ''), '',
           '| Column | What it is |', '|---|---|',
           f'| **{headline}** | Area under the training curve over the whole '
           'sequence (reward x generations, /1000). Integrated, not summed: NE '
           'logs per generation and RL per update, so the same budget gives '
           'one family more points. It prices in every drop at a switch and '
           f'every recovery after one. The curve is `{col}`. |',
           '| **Cum. mean** | The same integral of the population-average '
           'curve -- the mean of the population\'s FITNESSES (`mean_fitness`), '
           'which is NOT the score of the mean of their weights. That is the '
           '`centroid` metric, and it is a different number. The single-policy '
           'RL methods have no population, so it is `--` for them -- '
           f'undefined, not equal to their {headline}. |',
           '| **FT** | Forward transfer: the mean over sub-tasks of the '
           "continual curve's window mean, minus that method's OWN stationary "
           'run over an equal window, in the environment\'s reward units. 0 '
           'means a sub-task was learnt as well as the stationary task. '
           '**Read it beside an absolute column**: it subtracts each method\'s '
           'own reference, so a method that learns the stationary task badly '
           'has little left to lose. |']
    # The legend rows for the post-hoc columns are appended to this table
    # below; the paragraph and the notes come AFTER it, so the legend is one
    # markdown table rather than two halves around a paragraph.
    significance = [
        '', 'Mean over trials. Significance: one-sided Mann-Whitney U of each '
        "method against every member of the other family, Holm-Bonferroni "
        "corrected within that method's comparisons; a mark means its "
        'WEAKEST comparison survived. \\* p<0.05, \\*\\* p<0.01, '
        '\\*\\*\\* p<0.001.', '']
    notes = []
    # ES/NES on a shared-runner tree logged no best-member elite before
    # 2026-09-13: the curve under `elite_eval` there is the distribution MEAN
    # (the incumbent), while Final / ZT read the saved best member. Said in
    # the table, per method, rather than left to be discovered.
    if args.metric == 'elite_eval':
        mean_curve = set()
        for e in envs:
            for m in data[e]:
                if m not in ('es', 'nes'):
                    continue
                for cell_dir in sorted((pathlib.Path(args.root) / args.phase / m).glob('*')):
                    trial = next(iter(sorted(cell_dir.glob('trial_*'))), None)
                    cfg = load_config(trial)[0] if trial is not None else None
                    if (cfg and cfg.get('column_naming') == 'gymnax_aliases_v1'
                            and not cfg.get('elite_convention')):
                        mean_curve.add(METHOD_STYLE.get(m, {}).get('label', m))
                    break
        if mean_curve:
            notes += [f"> **{', '.join(sorted(mean_curve))}: the elite CURVE is the "
                      'distribution mean.** These runs predate the best-member elite '
                      '(2026-09-13), so for them the curve equals the centroid; the '
                      'post-hoc columns (Final, ZT) read the saved best member of each '
                      "phase's last generation.", '']
    if missing_ref:
        notes += ['> **FT is `--` throughout**: it needs each method\'s '
                  'stationary run, and no `noncontinual` tree was given. Pass '
                  '`--ref-root <tree>` (its `noncontinual` phase) to fill it.', '']
    have_posthoc = any(posthoc.get(e, {}).get(m, {}).get(k)
                       for e in envs for m in data[e]
                       for k in ('F', 'BD', 'ZT', 'Final'))
    stationary = args.phase == 'noncontinual'
    thresholds = thresholds or {}
    if not have_posthoc:
        notes += ['> **F, BD and ZT are absent, not zero.** They are not '
                  'functions of a training curve: they come from re-rolling every '
                  "sub-task's saved agent against every sub-task. Run "
                  '`source/studies/evaluate_continual.py` for ZT and '
                  '`scripts/analysis/behavioural_divergence.py` for F and BD, '
                  'then pass `--results-dir` here.', '']
    elif not stationary:
        out += [
            '| **F** | Forgetting (Lopez-Paz & Ranzato 2017; Wolczyk et al. '
            '2021): what each sub-task\'s own agent scored on it minus what '
            'the agent at the END of the sequence scores on it, meaned over '
            'every earlier sub-task, in the same reward units as FT. With '
            'TWO alternating sub-tasks the final agent was always trained on '
            'the same one, so F is instead taken at every switch: the own '
            'agent minus the agent after the next switch, meaned over all '
            'switches and both directions. Negative is backward transfer. '
            '**Read it against FT** -- a method that never acquired '
            'a sub-task has nothing left to lose on it. |',
            '| **BD** | Behavioural divergence at the switch: the fraction of '
            "states visited by a sub-task's own agent on which the NEXT "
            'sub-task\'s agent takes a different action. In [0, 1] and '
            'temperature-free, so a deterministic evolved policy and a greedily '
            'evaluated RL one are directly comparable. |',
            '| **ZT** | Zero-shot transfer: the return that agent gets on the '
            'next sub-task, before any search has been done on it. |']

    if stationary and have_posthoc:
        known = {e: thresholds[e] for e in envs if thresholds.get(e) is not None}
        if not known:
            thr_note = 'none known'
        elif len(set(known.values())) == 1:
            thr_note = f'{next(iter(known.values())):g} for every environment'
        else:
            thr_note = ', '.join(f'{ENV_TITLES.get(e, e)} {v:g}'
                                 for e, v in known.items())
        out += [
            '| **ZT** | On a STATIONARY tree the "next" sub-task is the same '
            'one, so this is the saved agent re-scored on its own '
            "environment at the evaluation pass's episode count and fresh "
            'keys, meaned over the checkpoint phases -- the post-hoc estimate '
            'of the training curve, not a transfer. F and BD are undefined '
            'here and left out. |',
            '| **Final** | The end-of-training evaluation: the LAST saved '
            'agent (the reported network) re-scored on its own environment, '
            'mean return over the evaluation episodes. This is the number to '
            'read for "did it learn the task". |',
            '| **Solved** | Trials whose Final clears the environment\'s '
            f'solved threshold, as k/n. Thresholds: {thr_note}. `--` where '
            'no threshold is known (pass `--threshold`). |']
    out += significance + notes
    if stationary:
        cols = [headline, 'Cum. mean', 'FT'] + (
            ['ZT', 'Final', 'Solved'] if have_posthoc else [])
    else:
        cols = [headline, 'Cum. mean', 'FT'] + (
            ['F', 'BD', 'ZT'] if have_posthoc else [])

    methods = [m for m in METHOD_ORDER if any(m in data[e] for e in envs)]
    methods += [m for e in envs for m in sorted(data[e]) if m not in METHOD_ORDER]
    methods = list(dict.fromkeys(methods))
    # F and BD are LOWER-is-better; the others higher.
    marks = {}
    for env in envs:
        ph = posthoc.get(env, {})
        marks[env] = {
            'cum_max': mann_whitney_marks(rows[env]['cum_max'], FAMILY, True),
            'cum_mean': mann_whitney_marks(rows[env]['cum_mean'], FAMILY, True),
            'ft': mann_whitney_marks(rows[env]['ft'], FAMILY, True),
            'F': mann_whitney_marks({m: v.get('F') for m, v in ph.items()},
                                    FAMILY, False),
            'BD': mann_whitney_marks({m: v.get('BD') for m, v in ph.items()},
                                     FAMILY, False),
            'ZT': mann_whitney_marks({m: v.get('ZT') for m, v in ph.items()},
                                     FAMILY, True),
            'Final': mann_whitney_marks({m: v.get('Final') for m, v in ph.items()},
                                        FAMILY, True),
        }
    def cell(env, method, col):
        r, ph = rows[env], posthoc.get(env, {})
        mine = ph.get(method, {})
        if col == headline:
            return _fmt(r['cum_max'].get(method), marks[env]['cum_max'], method,
                        _digits(r['cum_max']))
        if col == 'Cum. mean':
            return _fmt(r['cum_mean'].get(method), marks[env]['cum_mean'],
                        method, _digits(r['cum_mean']))
        if col == 'FT':
            return _fmt(r['ft'].get(method), marks[env]['ft'], method, 3)
        if col == 'BD':
            return _fmt(mine.get('BD'), marks[env]['BD'], method, 3)
        if col == 'Solved':
            thr, finals = thresholds.get(env), mine.get('Final')
            if thr is None or not finals:
                return '--'
            return f'{sum(f >= thr for f in finals)}/{len(finals)}'
        return _fmt(mine.get(col), marks[env][col], method,
                    _digits({m: v.get(col) for m, v in ph.items()}))

    def trials(env, method):
        return data[env][method][1].shape[0] if method in data[env] else 0

    if len(envs) <= WIDE_TABLE_ENVS:
        # Methods down, (environment x column) across: the paper's table.
        header = '| Method | ' + ' | '.join(
            f'{ENV_TITLES.get(e, e)} {c}' for e in envs for c in cols) + ' | n |'
        out += [header, '|' + '---|' * (len(cols) * len(envs) + 2)]
        for method in methods:
            n = max(trials(e, method) for e in envs)
            out.append(f"| {METHOD_STYLE.get(method, {}).get('label', method)} | "
                       + ' | '.join(cell(e, method, c) for e in envs for c in cols)
                       + f' | {n} |')
    else:
        # Twenty Kinetix levels x six columns do not fit across a page: one
        # table per column, environments down and methods across. Same
        # numbers, same marks, same `cell`.
        labels = [METHOD_STYLE.get(m, {}).get('label', m) for m in methods]
        for col in cols:
            out += [f'### {col}', '',
                    '| Environment | ' + ' | '.join(labels) + ' | n |',
                    '|' + '---|' * (len(methods) + 2)]
            for env in envs:
                ns = sorted({trials(env, m) for m in methods if trials(env, m)})
                out.append(f'| {ENV_TITLES.get(env, env)} | '
                           + ' | '.join(cell(env, m, col) for m in methods)
                           + ' | ' + '/'.join(str(n) for n in ns) + ' |')
            out.append('')
    path.write_text('\n'.join(out) + '\n')


def build_parser():
    """The CLI. `make_metrics_figure.py` extends it rather than copying it."""
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('root')
    ap.add_argument('--phase', default='continual')
    ap.add_argument('--sigma', default=None)
    ap.add_argument('--cells', nargs='*', default=None,
                    help='explicit cell directory names, one per '
                         'environment, e.g. CartPole_v1_sigma1.0 '
                         'Acrobot_v1_sigma1.0 MountainCar_v0_sigma0.1. '
                         'Use this when the reported noise differs by '
                         'environment; --sigma cannot express that. '
                         'Mutually exclusive with --sigma.')
    ap.add_argument('--metric', default='current', choices=list(METRIC_COLUMNS))
    ap.add_argument('--aggregate', default='generalist',
                    choices=['generalist', 'mean_over_tasks'])
    ap.add_argument('--methods', nargs='*', default=None)
    ap.add_argument('--x', default='generations', choices=['generations', 'steps'])
    ap.add_argument('--width', type=float, default=6.9,
                    help='figure width in INCHES. Default 6.9 = 17.5 cm, an A4 '
                         'text width at 2 cm margins, so the figure is included '
                         'at 1:1 and the font sizes below are the sizes on the '
                         'printed page.')
    ap.add_argument('--threshold', type=float, default=None,
                    help='solved threshold for the stationary table\'s Solved '
                         'column, for every cell drawn. Default: the run\'s '
                         'own `solved_threshold`, else the registry\'s '
                         '(gymnax). MiniGrid has neither: pass 0.8.')
    ap.add_argument('--ncols', type=int, default=None,
                    help='wrap the panels into rows of this many; the default '
                         'is one row (the three gymnax cells, the two MiniGrid '
                         'rooms). The twenty Kinetix stationary cells use 5. '
                         'Each row is --panel-height tall.')
    ap.add_argument('--panel-height', type=float, default=1.9,
                    help='height of the panel row in inches (default 1.9)')
    ap.add_argument('--font-size', type=float, default=7.0,
                    help='base font size in points, at 1:1 (default 7)')
    ap.add_argument('--legend', default='separate',
                    choices=['separate', 'inline', 'none'],
                    help="separate (default): write the legend as its own "
                         "<stem>_legend.png/.pdf, so the panels keep their full "
                         "height and one legend can serve several figures. "
                         "inline: above the panels.")
    ap.add_argument('--legend-ncol', type=int, default=None,
                    help='columns in the legend; default is one row')
    ap.add_argument('--results-dir', default=None,
                    help='directory holding behavioural_divergence.json, for '
                         'the F and BD columns. ZT is read from each run\'s '
                         'own evaluation.json and needs no flag.')
    ap.add_argument('--ref-root', default=None,
                    help="run tree holding the STATIONARY reference for FT; "
                         "its `noncontinual` phase is used. Defaults to `root` "
                         "when that tree has one. FT is `--` without it.")
    ap.add_argument('--include-superseded', action='store_true',
                    help='also plot arms listed in SUPERSEDED')
    ap.add_argument('--band', default='ci', choices=['ci', 'sd', 'iqr', 'none'],
                    help='ci (default): mean over seeds with a 95%% percentile-'
                         'bootstrap band. sd: mean +- one s.d. iqr: median with '
                         'the interquartile band -- NOT for the paper, see the '
                         'docstring. The table always reports the mean.')
    ap.add_argument('--smooth', type=int, default=None,
                    help='rolling-median window in records; default 1%% of the run')
    ap.add_argument('--assume-episode-length', type=int, default=None)
    ap.add_argument('--allow-monotone', action='store_true',
                    help='plot anyway when a series is monotone under a metric '
                         'that should not be; normally that means a best-so-far '
                         'column was resolved by mistake')
    ap.add_argument('--out', required=True, help='output stem, no extension')
    return ap


def parse_args(argv=None):
    return build_parser().parse_args(argv)


class Report:
    """Everything one figure or table is drawn from, loaded ONCE.

    `make_metrics_figure.py` draws the table's columns as a figure and reads
    them through this, so the lineplot, the table and the metrics figure
    cannot disagree about which runs, arms and cells the paper reports.
    """
    root: pathlib.Path
    cell_filter: object
    cell_label: str
    data: dict            # {env: {method: (x_steps, curves)}} at --metric
    pop_data: dict        # the same at popmean_score, population arms only
    ref_data: dict        # the stationary reference for FT
    used_columns: set
    per_gen: int | None
    every: int | None     # steps per sub-task
    total: int | None     # steps in the whole sequence
    edges: object        # phase_edges: [0, ..., total] in steps
    missing: list
    flat: list


def load_report(args) -> Report:
    """Load, filter and check the runs `args` names. Exits on a broken tree."""
    root = pathlib.Path(args.root)
    if args.cells and args.sigma:
        sys.exit('ERROR: pass --cells or --sigma, not both. --cells already '
                 'names the sigma of each environment.')
    cell_filter, cell_label = cell_selector(args.cells, args.sigma)

    data, missing, used_columns, per_gen, monotone, flat = collect(
        root, args.phase, cell_filter, args.metric, args.aggregate,
        args.assume_episode_length)
    if not data:
        sys.exit(f'no runs matched under {root}/{args.phase}'
                 + (f' with {cell_label}' if cell_label else ''))
    present = {m for env in data for m in data[env]}
    # `--methods` is a restriction on the DATA, not on the drawing. Applied at
    # plot time only, it left the table, the Mann-Whitney comparison set and
    # the figure disagreeing about which arms the paper reports -- the table
    # would still score a method the figure does not show, and the
    # significance marks would still be corrected over it. Dropping it here,
    # once, is the same discipline the superseded pass below follows.
    if args.methods:
        unknown = sorted(set(args.methods) - present)
        if unknown:
            sys.exit(f'ERROR: --methods names {unknown}, which have no runs '
                     f'under {root}/{args.phase}'
                     + (f' at {cell_label}' if cell_label else '')
                     + f'. Present: {sorted(present)}')
        for env in data:
            for m in list(data[env]):
                if m not in args.methods:
                    del data[env][m]
        print('note: --methods restricts the figure AND the table to '
              + ' '.join(args.methods))
        present &= set(args.methods)
        # Ordering stays METHOD_ORDER's, so the same method keeps the same
        # position and colour whether or not the flag was passed.
        args.methods = None
    for method in sorted(present & set(SUPERSEDED)):
        predicate, phases, reason = SUPERSEDED[method]
        state = ('none' if args.phase not in phases else
                 defect_state(root, args.phase, method, cell_filter, predicate))
        if state == 'none':
            continue
        if state == 'mixed':
            sys.exit(f'ERROR: some but not all {method!r} trials are '
                     f'defective ({reason}). One directory holds two '
                     f'algorithms; separate them before plotting.')
        if args.include_superseded:
            print(f'WARNING: plotting superseded arm {method!r} -- {reason}')
            continue
        for env in data:
            data[env].pop(method, None)
        print(f'note: dropping superseded arm {method!r} -- {reason}'
              f'\n      (--include-superseded plots it anyway)')
    for method in sorted(present & set(MISLABELLED)):
        predicate, phases, reason = MISLABELLED[method]
        state = ('none' if args.phase not in phases else
                 defect_state(root, args.phase, method, cell_filter, predicate))
        if state == 'none':
            continue
        if state == 'mixed':
            sys.exit(f'ERROR: some but not all {method!r} trials ran at the '
                     f'published setting ({reason}). Separate them before '
                     f'plotting.')
        print(f'WARNING: {method!r} is plotted under its own name but '
              f'{reason}')

    if missing:
        print(f'WARNING: {len(missing)} method(s) have no column for '
              f'--metric {args.metric} and are ABSENT from the figure:')
        for method, key in sorted(missing):
            print(f'  {method}: none of [{key}]')
    if flat:
        # Worth saying out loud: a flat trial drags the mean and widens the
        # band, and a reader of the figure cannot see that it exists.
        print(f'note: {len(flat)} trial(s) never moved from their initial '
              f'value and are included as-is:')
        for method, cell, trial, value in sorted(flat):
            print(f'  {method}/{cell}/{trial}: constant {value:g}')
    # The population-average curve, for Cum. mean, and the stationary run, for
    # FT. Both go through the same loader as the plotted curve, so the table
    # and the figure cannot disagree about which runs they read.
    pop_data, _, _, _, _, _ = collect(root, args.phase, cell_filter,
                                      'popmean_score', args.aggregate,
                                      args.assume_episode_length)
    # NE arms only. PBT has a population too, but its `mean_fitness` is the
    # members' mean TRAINING return -- PPO's per-update GAE-return mean over a
    # rollout window, not an episode score -- so it is not the NE arms' mean
    # population fitness and does not belong in the same column; the RL arms,
    # PBT included, show `--` there.
    pop_data = {e: {m: v for m, v in d.items() if FAMILY.get(m) == 'ne'}
                for e, d in pop_data.items()}
    ref_root = pathlib.Path(args.ref_root) if args.ref_root else root
    ref_data = {}
    if (ref_root / 'noncontinual').is_dir():
        ref_data, _, _, _, _, _ = collect(
            ref_root, 'noncontinual', lambda c: '_sigma' not in c,
            args.metric, args.aggregate, args.assume_episode_length or 500)

    every, total, edges = boundaries(root, args.phase, cell_filter,
                                     args.assume_episode_length)
    # The monotone check only means something where the task CHANGES. In a
    # stationary run a curve that rises and never falls is just learning, so
    # applying the check there reports every well-behaved run as broken.
    # A STATIONARY phase is cut into checkpoint phases too (ten on Kinetix and
    # MiniGrid), so `every < total` alone would call it switching and reject a
    # GA that sits at the ceiling from its first record. The phase name is
    # the convention every tree follows: nothing changes under noncontinual.
    switches = bool(every and total and every < total
                    and args.phase != 'noncontinual')
    if monotone and switches:
        msg = (f'{len(monotone)} (method, column) pair(s) are monotone '
               f'non-decreasing under --metric {args.metric}, in a tree whose '
               f'task DOES change:\n'
               + '\n'.join(f'  {m}: {k}' for m, k in sorted(monotone))
               + '\n  A best-so-far column cannot show forgetting, and plotting '
                 'it against another family\'s per-record column compares two '
                 'different quantities.')
        if not args.allow_monotone:
            sys.exit('ERROR: ' + msg + '\n  Use --metric best_so_far to plot it '
                     'deliberately, or --allow-monotone to override.')
        print('WARNING: ' + msg)
    rep = Report()
    rep.root, rep.cell_filter, rep.cell_label = root, cell_filter, cell_label
    rep.data, rep.pop_data, rep.ref_data = data, pop_data, ref_data
    rep.used_columns, rep.per_gen = used_columns, per_gen
    rep.every, rep.total, rep.missing, rep.flat = every, total, missing, flat
    rep.edges = edges
    return rep


def main() -> int:
    args = parse_args()

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    rep = load_report(args)
    root, cell_filter, cell_label = rep.root, rep.cell_filter, rep.cell_label
    data, pop_data, ref_data = rep.data, rep.pop_data, rep.ref_data
    used_columns, per_gen = rep.used_columns, rep.per_gen
    edges = rep.edges

    scale = per_gen if (args.x == 'generations' and per_gen) else 1.0
    xlabel = 'Generations' if scale != 1.0 else 'Environment steps'

    envs = [e for e in ENV_TITLES if e in data] + \
           [e for e in sorted(data) if e not in ENV_TITLES]
    # Sized for inclusion at 1:1 on A4: a figure that is drawn wide and then
    # scaled down in LaTeX has its fonts scaled down with it, which is how
    # 10 pt becomes unreadable 5 pt on the page.
    fs = args.font_size
    plt.rcParams.update({
        'font.size': fs, 'axes.labelsize': fs, 'axes.titlesize': fs + 1,
        'xtick.labelsize': fs - 0.5, 'ytick.labelsize': fs - 0.5,
        'legend.fontsize': fs, 'axes.linewidth': 0.6,
        'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
        'xtick.major.size': 2.5, 'ytick.major.size': 2.5,
    })
    ncols = min(args.ncols or len(envs), len(envs))
    nrows = -(-len(envs) // ncols)
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(args.width, args.panel_height * nrows),
                             squeeze=False)
    flat_axes = [ax for row in axes for ax in row]
    for ax in flat_axes[len(envs):]:
        ax.set_visible(False)
    drawn: dict[str, object] = {}
    for ax, env in zip(flat_axes, envs):
        methods = args.methods or (
            [m for m in METHOD_ORDER if m in data[env]]
            + [m for m in sorted(data[env]) if m not in METHOD_ORDER])
        if edges is not None:
            for b in edges[1:-1]:
                ax.axvline(b / scale, color='0.6', lw=0.45, ls='--', zorder=0)
        for method in methods:
            if method not in data[env]:
                continue
            x, curves = data[env][method]
            win = (args.smooth if args.smooth is not None
                   else max((int(curves.shape[1] * 0.01) | 1), 1))
            curves = smooth(curves, win)
            colour = METHOD_STYLE.get(method, {}).get('color')
            if args.band == 'ci':
                mid, lo, hi = bootstrap_ci(curves)
            elif args.band == 'sd':
                mid = curves.mean(axis=0)
                lo, hi = mid - curves.std(axis=0), mid + curves.std(axis=0)
            elif args.band == 'iqr':
                mid = np.median(curves, axis=0)
                lo = np.percentile(curves, 25, axis=0)
                hi = np.percentile(curves, 75, axis=0)
            else:
                mid = curves.mean(axis=0)
                lo = hi = mid
            line, = ax.plot(x / scale, mid, color=colour, lw=1.0, zorder=3)
            drawn.setdefault(method, line)
            if args.band != 'none':
                ax.fill_between(x / scale, lo, hi, color=colour, alpha=0.18,
                                lw=0, zorder=2)
        title = ENV_TITLES.get(env, env)
        if ncols > 3:
            # Five Kinetix panels across 17.5 cm are 3.3 cm each; "h18 thrust
            # right very easy" does not fit on one line at 8 pt.
            title = '\n'.join(textwrap.wrap(title, 16))
        ax.set_title(title, fontweight='bold')
        ax.set_xlabel(xlabel)
        ax.margins(x=0)
        ax.spines[['top', 'right']].set_visible(False)

    if args.metric in ('incumbent', 'popmean'):
        ylab = ('Episodic reward, worst sub-task'
                if args.aggregate == 'generalist'
                else 'Episodic reward, mean over sub-tasks')
    else:
        ylab = 'Episodic reward'
    for row in axes:
        row[0].set_ylabel(ylab)

    order = [m for m in METHOD_ORDER if m in drawn] + \
            [m for m in drawn if m not in METHOD_ORDER]
    handles = [drawn[m] for m in order]
    labels = [METHOD_STYLE.get(m, {}).get('label', m) for m in order]
    if args.legend == 'inline':
        fig.legend(handles, labels, frameon=False,
                   ncol=min(len(order), 7), loc='upper center',
                   bbox_to_anchor=(0.5, 1.02 + 0.06 * ((len(order) - 1) // 7)))
    fig.tight_layout()

    stem = pathlib.Path(args.out)
    stem.parent.mkdir(parents=True, exist_ok=True)
    written = []
    for ext in ('png', 'pdf'):
        # Not `with_suffix`: a stem like `..._sigma1.0` looks to pathlib like a
        # name with suffix `.0`, and with_suffix would REPLACE it, writing
        # `..._sigma1.png` and silently losing which sigma the figure is of.
        out = stem.parent / f'{stem.name}.{ext}'
        fig.savefig(out, dpi=400, bbox_inches='tight')
        written.append(out.name)

    if args.legend == 'separate':
        # Its own figure. One legend can then caption a column of panels, and a
        # legend needing two rows does not eat a third of a 1.9 in figure.
        #
        # Saved at EXACTLY `--width` with no tight bounding box, unlike the
        # panels: a legend cropped to its content is narrower than the figure,
        # so including both at the same width in LaTeX scales the legend up and
        # its labels stop matching the axis labels. Same width in, same width
        # out, same point size on the page.
        import matplotlib.pyplot as _plt
        ncol = args.legend_ncol or len(order)
        rows = 1 + (len(order) - 1) // ncol
        lfig = _plt.figure(figsize=(args.width, 0.16 * rows + 0.06))
        lfig.legend(handles, labels, frameon=False, ncol=ncol, loc='center',
                    handlelength=1.6, columnspacing=1.2, handletextpad=0.5,
                    borderpad=0)
        for ext in ('png', 'pdf'):
            out = stem.parent / f'{stem.name}_legend.{ext}'
            lfig.savefig(out, dpi=400)
            written.append(out.name)
        _plt.close(lfig)
    posthoc = posthoc_table(root, args.phase, cell_filter, args.results_dir,
                            args.metric)
    write_table(stem.parent / f'{stem.name}_table.md', data, pop_data,
                ref_data, posthoc, used_columns, per_gen, edges, args,
                thresholds=solved_thresholds(root, args.phase, cell_filter,
                                             args.threshold))
    written.append(f'{stem.name}_table.md')
    print(f'wrote into {stem.parent}/: ' + ', '.join(written))
    height = args.panel_height * nrows
    print(f'  figure {args.width} x {height:g} in '
          f'({args.width * 2.54:.1f} x {height * 2.54:.1f} cm), '
          f'{args.font_size} pt base -- include at 1:1')
    print(f'  column(s): {", ".join(sorted(used_columns))}; x in '
          f'{xlabel.lower()}'
          + (f' (1 generation = {per_gen:,} steps)' if scale != 1.0 else ''))
    for env in envs:
        print(f'  {env}: ' + ', '.join(
            f'{m}(n={data[env][m][1].shape[0]})' for m in sorted(data[env])))
    return 0


if __name__ == '__main__':
    sys.exit(main())
