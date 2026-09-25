"""The paper's plasticity figure: every symptom the RL literature reports, both families.

    .venv/bin/python scripts/make_plasticity_figure.py projects/iclr_2027/runs/gymnax \\
        --phase continual --sigma 1.0 \\
        --checkpoints projects/iclr_2027/results/gymnax_continual/sigma1.0 \\
        --out projects/iclr_2027/figures/plasticity_gymnax_sigma1.0

Reads the same trees, draws in the same house style and writes the same four
file types as `scripts/make_lineplot.py` -- whose palette, labels, loader,
smoothing, task-boundary rules and superseded-arm handling this imports rather
than restates, so a method is the same colour and the same set of trials in
both figures.

It goes AFTER the lineplot and its table: those say which methods keep
learning across task changes, and this says which of the mechanisms the
plasticity literature blames for not keeping it are actually present.

## The rows, and whose claim each one is

| row | quantity | claimed by |
|---|---|---|
| `dormancy` | fraction of hidden units whose layer-normalised mean activation is <= tau on a frozen probe batch | Sokar et al. 2023; ReDo recycles exactly these |
| `action_shift` | fraction of probe states whose greedy action changed since the last record | the symptom C-CHAIN's regulariser targets, in the one estimator both families can report. NOT C-CHAIN's own churn -- see `churn_ce` -- and not a parameter-space distance either; those are `step` and `drift` |
| `churn_ce` | C-CHAIN's published churn, H(pi_before, pi_after) on the same probe batch (`--rows`; RL only) | Tang et al. verbatim |
| `weight`   | RMS of the SAVED AGENT's parameter vector, one point per sub-task | Nikishin et al. 2022; Juliani & Ash 2024 find it the strongest single predictor, and it is what TRAC's rescaling holds down |
| `weight_pop` | the same RMS over the whole POPULATION pooled, per generation (`--rows`; NE only meaningfully) | not a plasticity claim -- population scale, drawn beside `weight` when the question is whether the search has spread or collapsed |
| `weight_mean` | the SIGNED mean of the parameters (`--rows`; not in the default five) | the same claim read as a drift off the initialisation's centre rather than as a magnitude |
| `dormant_age` | mean number of consecutive sub-tasks the currently dormant units have been dormant, from the per-sub-task checkpoints | the same claim as `dormancy` read as a DURATION: dead units age along the ceiling `t + 1`, recycled or replaced ones stay near 1. `persistence_lag` is its chance-corrected control and stays selectable |
| `curvature`| effective rank of the empirical NTK Gram | Tang et al.'s own cause: rank collapse -> correlated gradients -> churn |
| `step`     | \\|theta_t - theta_{t-1}\\| between sub-task checkpoints | Juliani & Ash's "weight difference"; Abbas et al. found it predictive off-policy and they did not on-policy |

`--rows` selects; the default draws all five. A second figure,
`<stem>_persistence`, answers the question the dormant FRACTION cannot -- see
below. `<stem>_table.md` carries the summary numbers for all of it.

## Why these columns are comparable across the NE/RL divide, and where they are not

The four dense rows are read from `training_metrics.json`, where the trainers
wrote them under one definition for both families (`source/metrics/plasticity.py`,
`source/metrics/ntk.py`): the same ReDo tau and criterion, the same frozen
512-state probe batch, the same argmax-disagreement churn, and -- because the
gymnax NE and RL trainers share `MLPPolicy` -- literally the same 32 hidden
units to be dormant. Three caveats survive that and are printed on every run:

  **the churn clock.** NE compares the elite against the elite of the previous
  GENERATION (768,000 environment steps here); RL compares the policy against
  its parameters at the last ROLLOUT (1,024,000 steps, not one gradient step
  -- `prev_policy_params` is taken before the epoch loop). Within 33% of each
  other, which is why they are drawn together, and not identical, which is why
  a small gap is not a result.

  **the elite is not one network over time.** For NES, OpenES and the RL arms
  the tracked policy is a point that moves; for GA and DNS it is an argmax
  over a population and can change lineage between records. Action shift and
  persistence read high and low respectively for that reason alone there.
  `--agent centroid` is the answer to all three: the centroid is one vector
  moving over time in every arm, so its columns carry none of these caveats.

  **DNS diverges on this tree.** Its genotypes reach |w| ~ 1e14, which is why
  the weight row is logarithmic and why its NTK rank is missing for about a
  third of its records: the Gram overflowed and the observer refused to raise.

  **the weight row is not one of them.** It is a CHECKPOINT row, read off the
  saved agent under `--agent`, so it names the same individual the lineplot
  scores in both families. The per-generation NE weight columns describe the
  POPULATION and never the elite or the centroid, which is why they are
  `weight_pop` and not the default.

## The dormant fraction cannot tell dead from sparse

Two networks with the same dormant fraction, and only one has lost capacity:

    dead     a fixed subset of units is dormant and stays dormant.
    sparse   a similar-sized subset is dormant at every checkpoint, but a
             DIFFERENT subset each time -- few units at a time, all of them
             over time. That is functional sparsity and not a pathology.

`<stem>_persistence` separates them, from
`scripts/analysis/plasticity_checkpoints.py`'s per-unit masks. Its y axis is

    persistence index = (P(dormant at t+k | dormant at t) - p) / (1 - p)

with `p` the method's own instant dormant fraction, which is exactly the
chance level: a dormant set redrawn independently at each checkpoint would
give survival `p`. So 0 is sparsity, 1 is death, and the index can be read
across methods whose dormant fractions differ -- which raw survival cannot,
since a method that is dormant everywhere survives trivially.

Its resolution is one checkpoint per sub-task, so it is a lower bound on
transience: see that script's docstring before quoting a number from it.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import warnings
from collections import defaultdict

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from make_lineplot import (                                  # noqa: E402
    ENV_TITLES, FAMILY, METHOD_ORDER, METHOD_STYLE, MISLABELLED, SUPERSEDED,
    boundaries, budget, cell_selector, collect, defect_state, load_config,
    smooth)

# Semantic row -> [NE column, RL column], most specific first, resolved per run
# by `make_lineplot.resolve_column`. A method with no column for a row is named
# in a warning and left out of that row rather than dropped silently.
PLASTICITY_COLUMNS = {
    # The elite / the deployed policy, on the frozen probe batch. `pop_dormancy`
    # is the population mean, which only the NE arms can answer -- it is here
    # because a population can hold a dormant elite while its members are not,
    # and the pair says which.
    'dormancy':     ['ne_elite_dormant_fraction', 'policy_dormant_fraction_probe'],
    'pop_dormancy': ['pop_dormant_fraction', 'policy_dormant_fraction_probe'],
    # NOT C-CHAIN's churn, which is the cross-entropy row below. This is a
    # BEHAVIOURAL distance -- the fraction of probe states whose greedy action
    # changed -- so it says the policy now acts differently, not that its
    # parameters moved (that is `step` and `drift`, which are parameter-space
    # norms). Named `action_shift` because calling it churn invited exactly
    # that confusion with the published quantity.
    'action_shift': ['ne_elite_churn_action', 'policy_churn_action'],
    # The published cross-entropy churn. NOT the default: H(pi_b, pi_a) =
    # H(pi_b) + KL(pi_b || pi_a), so it has an additive floor at the old
    # policy's own entropy and a run whose entropy collapses reports a smaller
    # number for that reason alone. The NE policies are deterministic argmax
    # policies whose softmax temperature is arbitrary, so it is not defined for
    # them at all and they are absent from this row.
    'churn_ce':     ['policy_churn'],
    # POPULATION weight statistics, and NOT the `weight` row. The NE column
    # here is `population_weight_stats(population)` -- every weight of every
    # member pooled -- so it is not any one network and `--agent` cannot make
    # it one: no NE trainer logs per-generation weight statistics of the elite
    # or of the centroid. The RL column IS the deployed policy, so the row
    # compares a population against a single network across the family divide.
    # Kept because "how large are this population's weights" is a real
    # question -- for NES/OpenES it sits a sigma above the centroid, and for a
    # converged GA it approaches the elite -- but the paper's weight row is
    # the checkpoint one below, which is the SAME individual the lineplot
    # scores. Selectable with `--rows weight_pop`.
    'weight_pop':     ['weight_rms', 'policy_weight_rms'],
    'weight_max_pop': ['weight_max_abs', 'policy_weight_max_abs'],
    'curvature':    ['ne_elite_ntk_effective_rank', 'policy_ntk_effective_rank'],
    'ntk_trace':    ['ne_elite_ntk_trace', 'policy_ntk_trace'],
    # POPULATION WIDTH, NE only: there is no RL column because a single policy
    # has no spread. Mean pairwise L2 between genomes, and the s.d. of the
    # population's fitnesses, per generation. Not a plasticity claim -- these
    # are what the action-reversal family turns on: ES/NES halt on a reversed
    # sub-task with a fitness s.d. of exactly 0 (no gradient to estimate), GA
    # and DNS cross on population width alone. Drawn as their own figure
    # (`--rows genomic_diversity fitness_std`), not in the default rows.
    'genomic_diversity': ['bd_genomic_diversity'],
    'fitness_std':       ['bd_fitness_std'],
}

# Rows that come from the checkpoint pass instead of the training curve, and
# the key each reads out of its per-trial record.
CHECKPOINT_ROWS = {
    'step':  ('step_norm', 1),      # (key, first sub-task index it exists for)
    # C-CHAIN's churn, made cross-family: KL(pi_t-1 || pi_t) on the probe, with
    # both temperatures calibrated so an argmax policy has a defined softmax at
    # all and the cross-entropy's entropy floor drops out. See
    # `plasticity_checkpoints.calibrated_policy_kl`. A CHECKPOINT row, so the
    # gap is one sub-task -- the policy-space counterpart of `step`, and NOT
    # the per-update `churn_ce`.
    'policy_kl': ('policy_kl', 1),
    # The same sub-task gap read through the ARGMAX instead: the fraction of
    # probe states whose greedy action changed, both networks scored on the
    # same batch. Both families deploy an argmax policy, so this needs no
    # temperature convention; it is quantised at 1/|probe| and blind to a
    # change that does not cross a decision boundary. Distinct from the curve
    # row `action_shift`, which is the same test at the per-record gap.
    'action_shift_ckpt': ('action_shift', 1),
    'drift': ('drift', 0),
    # The dormant fraction from the checkpoint pass rather than from the
    # training curve. Three reasons it is the default and `dormancy` is not:
    # it is scored on the sub-task the agent is actually in (`--probe matched`)
    # where the curve column uses a batch frozen on sub-task 0, which in this
    # benchmark is a DIFFERENT observation distribution and not a drifted one;
    # it exists for every arm, where the resampled curve column exists only for
    # the RL ones; and it shares its probe, its agent and its resolution with
    # the persistence row below, so the pair can be read as one statement.
    'dormancy_ckpt': ('dormant_fraction', 0),
    # Chance-corrected one-step persistence of the dormant set. The row that
    # says whether the fraction above is capacity leaving (1) or a different
    # handful of units going quiet each time (0), which is functional sparsity
    # and not a pathology. See `plasticity_checkpoints.persistence_index`.
    'persistence': ('persistence_index', 1),
    # Mean age, in sub-tasks, of the units dormant at each checkpoint --
    # `plasticity_checkpoints.dormant_age`. The default second row: it answers
    # the same question as the persistence index ("is it the same units each
    # time?") in a unit the reader already has, sub-tasks, and on the figure's
    # own time axis. NaN where nothing is dormant. Its ceiling is `t + 1`, the
    # record so far, which the panel draws; a curve on the ceiling is a set of
    # units dead since the start. Not chance-corrected -- see the docstring of
    # `dormant_age` -- which is why `persistence_lag` remains selectable.
    'dormant_age': ('dormant_age', 0),
    # The SIGNED weight mean. It is a checkpoint row and not a curve row
    # because the curve column does not exist for everyone: ES and DNS never
    # logged `weight_mean`, and the RL trainers log `policy_weight_mean_abs`,
    # a magnitude that is blind to exactly the drift off centre this is meant
    # to show. Recomputed from the saved agent, every arm has it.
    'weight_mean': ('weight_mean', 0),
    'weight_var': ('weight_var', 0),
    # RMS of the parameter vector. A CHECKPOINT row for the same reason
    # `weight_mean` is: the curve column exists only as a POPULATION pooled
    # statistic on the NE side (`weight_pop` above), so the figure's headline
    # weight row would have been a population for the NE arms and a single
    # policy for the RL ones, and `--agent` would not have moved it. Read off
    # the saved agent instead, which is `finalgen` under `--agent elite` and
    # `centroid` under `--agent centroid` -- the individual the lineplot
    # beside it scores, for every arm and both reported networks. The cost is
    # resolution: one point per sub-task instead of one per generation.
    'weight': ('weight_rms', 0),
}

ROW_LABELS = {
    'dormancy':     'Dormant units\n(frozen probe)',
    'dormancy_ckpt': 'Dormant units',
    'persistence':  'Dormancy persistence\n(vs training time)',
    'persistence_lag': 'Dormancy\npersistence',
    'dormant_age':  'Dormancy age\n(sub-tasks)',
    'pop_dormancy': 'Dormant units\n(population)',
    'action_shift': 'Action shift',
    'churn_ce':     'Policy churn\n(C-CHAIN, nats)',
    'weight':       'Weight RMS',
    'weight_pop':     'Weight RMS\n(population)',
    'weight_max_pop': 'Max |weight|\n(population)',
    'curvature':    'NTK effective rank',
    'ntk_trace':    'NTK trace',
    'step':         r'$\|\Delta\theta\|$ per sub-task',
    'policy_kl':    'Policy change\n(KL, calibrated)',
    'action_shift_ckpt': 'Policy change\n(action shift)',
    'drift':        r'$\|\theta_t-\theta_0\|$',
    'weight_mean':  'Weight mean\n(signed)',
    'weight_var':   'Weight variance',
    'genomic_diversity': 'Genomic diversity\n(mean pairwise L2)',
    'fitness_std':  'Fitness s.d.\n(population)',
}
# Rows whose spread across methods is orders of magnitude. DNS reaches 1e14 on
# this tree and PPO stays near 1; on a linear axis every other arm is one flat
# line at the bottom.
LOG_ROWS = {'weight', 'weight_pop', 'weight_max_pop', 'ntk_trace',
            'drift', 'weight_var', 'genomic_diversity'}
# Rows spanning orders of magnitude that also take the value ZERO legitimately.
# A frozen policy churns exactly 0, and that is the observation -- a plain log
# axis would drop it, and a linear one shared across environments puts every
# arm on the floor next to NES's opening transient at 1.0. symlog is linear
# below `linthresh` and logarithmic above, so both are visible and the zero is
# still drawn. The value is where the interesting range starts, not a tuning
# knob: 1e-3 is one probe state in 512 having changed its action.
# `action_shift_ckpt` is deliberately NOT here: it is a fraction of the probe
# batch, bounded in [0, 1] and quantised at 1/512, so a log-like axis magnifies
# differences finer than the estimator can resolve. Linear, like the dormant
# fraction it sits under.
SYMLOG_ROWS = {'action_shift': 1e-3, 'churn_ce': 1e-3,
               # A policy that did not move gives a KL of 0 up to float noise,
               # and a plain log axis answers that with 1e-16 and a panel whose
               # whole range is numerical dust. `1e-4` is where a change starts
               # to mean anything at 512 probe states.
               'policy_kl': 1e-4,
               # The signed weight mean is the other case symlog exists for:
               # it takes both signs, so no log axis can hold it, and it spans
               # 1e-3 (every arm but one) to 1e11 (DNS, whose genotypes
               # diverge), so no linear one can either -- on a shared linear
               # axis every arm but DNS is a flat line on zero.
               'weight_mean': 1e-2,
               # A population whose members all score the same has a fitness
               # s.d. of exactly 0, and that IS the observation on reversed
               # Acrobot for ES/NES; a log axis would drop it.
               'fitness_std': 1e-2,
               # A saved agent that did not move between two sub-task
               # checkpoints has a step of 0 up to float noise (1e-30 on the
               # MountainCar ReDo trials). On a plain log axis that one trial
               # drags the panel's floor to 1e-36 and every other arm is a
               # line at the top; symlog draws the zero and keeps the
               # 1e-2 .. 1e3 range the rest of the figure lives in.
               'step': 1e-2}
# Of those, the ones that cannot go below zero, so the axis should not either.
NONNEGATIVE_ROWS = {'action_shift', 'action_shift_ckpt', 'churn_ce',
                    'policy_kl', 'fitness_std', 'step'}
# Rows whose x axis is LAG IN SUB-TASKS rather than training time. They are
# excused from the column's shared x axis, from the sub-task boundary rules,
# and from the figure's x label, and they carry their own.
LAG_ROWS = {'persistence_lag'}

# `dormancy_ckpt` and `dormant_age` lead, and in that order: the fraction says
# how much of the network is quiet and the age says for how long it has been
# the same part, which is the only pairing from which "lost plasticity" can be
# read. `dormant_age` replaced `persistence_lag` as the default on 2026-09-10:
# both separate the families the same way (PPO/C-CHAIN long-lived, GA/DNS/ReDo
# a sub-task or two) and the age is in sub-tasks on the figure's own time axis
# where the index is a lag curve on its own. `persistence_lag` is still
# selectable with `--rows` and still drawn as the standalone `_persistence`
# figure, since it is the chance-corrected version. `dormancy` -- the
# frozen-probe curve column -- is still selectable as the control.
# `action_shift_ckpt` and not `policy_kl`: both are behavioural and both are
# the same sub-task gap, but every method here ACTS BY ARGMAX, so the argmax
# disagreement compares the policies as they are deployed and assumes nothing.
# A KL needs a softmax, and the NE policies have no temperature -- fitness sees
# only the argmax, so their logit SCALE is arbitrary and a temperature has to
# be invented before a KL exists at all. Both networks of a pair are scored on
# the same batch, so neither version can mistake a change of probe for a change
# of policy. `policy_kl` stays selectable with `--rows`, and the two agree on
# the ordering of the arms (Spearman 0.90 over the 10 centroid cells) -- the
# argmax version is the one that needs no knob. NOT `action_shift`, which is
# the same test at the per-record gap and quantises to 0 there.
#
# `curvature` stays beside it rather than standing in for it. C-CHAIN's
# argument is that rank collapse CAUSES churn, so the two rows would be
# redundant if that held here -- and it does not: across the 21 (env, arm)
# cells of this tree the rank correlation between NTK effective rank and the
# behavioural row is +0.20 (p=0.38), and it flips sign per environment (-0.96
# CartPole, -0.37 MountainCar, +0.50 Acrobot). Cause and symptom come apart,
# which is a result, and it is only visible with both rows drawn.
DEFAULT_ROWS = ['dormancy_ckpt', 'dormant_age', 'action_shift_ckpt',
                'weight', 'curvature', 'step']

# Rows that legitimately only ever grow. `collect` refuses a monotone series by
# default because in the LINEPLOT that is the signature of a best-so-far column
# picked by mistake; here a weight norm that never falls is the result.
MONOTONE_OK = set(PLASTICITY_COLUMNS)


def load_checkpoint_rows(results_dir, cell_filter, edges):
    """`{row: {env: {method: (x_steps, curves)}}}` from the checkpoint pass.

    `x` is put in ENVIRONMENT STEPS -- checkpoint `t` is the agent saved at the
    END of sub-task `t`, so it sits at `edges[t + 1]` -- and not in sub-task
    index, so these rows share an axis with the dense ones and the boundary
    rules line up across the whole figure.
    """
    out = {row: defaultdict(dict) for row in CHECKPOINT_ROWS}
    if not results_dir:
        return out, None
    path = pathlib.Path(results_dir) / 'plasticity_checkpoints.json'
    if not path.exists():
        return out, f'no {path}'
    blob = json.loads(path.read_text())
    for cell, methods in blob['cells'].items():
        if not cell_filter(cell):
            continue
        env = cell.split('_sigma')[0]
        for method, entry in methods.items():
            for row, (key, first) in CHECKPOINT_ROWS.items():
                curves = [np.asarray(t[key], dtype=float)
                          for t in entry['trials'].values() if key in t]
                if not curves:
                    continue
                n = min(len(c) for c in curves)
                if edges is None:
                    x = np.arange(first, first + n) + 1
                    out[row][env][method] = (x, np.array([c[:n] for c in curves]))
                    continue
                # Right-aligned on the run's last checkpoint, which every series
                # ends at, rather than counted from `first`: the pairwise rows
                # do not agree on how they store the undefined first gap --
                # `step_norm` omits it (one entry fewer than checkpoints), while
                # `action_shift` and `policy_kl` keep it as a leading null -- so
                # counting from `first` drew those two one sub-task late, with
                # their last point past the end of the run.
                n = min(n, len(edges) - 1)
                out[row][env][method] = (edges[-n:],
                                         np.array([c[-n:] for c in curves]))
    return out, None


def saturated_arms(results_dir, cell_filter):
    """`(method, cell, saturated_trials, trials)` for arms that ran away.

    See `plasticity_checkpoints.analyse_trial` for what saturation is and why
    a finiteness check does not catch it.
    """
    if not results_dir:
        return []
    path = pathlib.Path(results_dir) / 'plasticity_checkpoints.json'
    if not path.exists():
        return []
    out = []
    for cell, methods in json.loads(path.read_text())['cells'].items():
        if not cell_filter(cell):
            continue
        for method, entry in methods.items():
            trials = entry['trials'].values()
            n = sum(1 for t in trials
                    if t.get('num_saturated_checkpoints', 0) > 0)
            if n:
                out.append((method, cell, n, len(entry['trials'])))
    return sorted(out)


def persistence_curves(results_dir, cell_filter, mask='dormant'):
    """`{env: {method: (lags, index)}}`: the persistence index per trial.

    index = (survival(k) - p) / (1 - p), p the trial's own instant fraction.
    See the module docstring for why the normalisation is the readable form.
    A trial with no dormant unit at all has no survival to speak of and is
    dropped from this row, counted in the returned `dropped`.
    """
    out, dropped = defaultdict(dict), []
    if not results_dir:
        return out, dropped
    path = pathlib.Path(results_dir) / 'plasticity_checkpoints.json'
    if not path.exists():
        return out, dropped
    blob = json.loads(path.read_text())
    for cell, methods in blob['cells'].items():
        if not cell_filter(cell):
            continue
        env = cell.split('_sigma')[0]
        for method, entry in methods.items():
            rows = []
            for name, trial in entry['trials'].items():
                stats = trial[mask]
                p = stats['instant']
                if p <= 0 or p >= 1 or stats['num_episodes'] == 0:
                    dropped.append((method, cell, name, p))
                    continue
                surv = np.array([np.nan if s is None else s
                                 for s in stats['survival']], dtype=float)
                rows.append((surv - p) / (1.0 - p))
            if rows:
                n = min(len(r) for r in rows)
                out[env][method] = (np.arange(1, n + 1),
                                    np.array([r[:n] for r in rows]))
    return out, dropped


def quiet_nan(fn, *a, **kw):
    """`fn` over data that may be all-NaN at some x, without the warning.

    A column of NaNs here is EXPECTED and is already reported by name -- DNS's
    NTK Gram overflows and the observer declines to measure rather than taking
    the run down. numpy warns per call, which at 15 panels x 8 methods buries
    the warnings that do mean something.
    """
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        return fn(*a, **kw)


def draw(ax, x, curves, colour, band, window, log):
    """One method's median-and-band on one panel. The lineplot's own drawing."""
    curves = smooth(curves, window)
    if band == 'sd':
        mid = quiet_nan(np.nanmean, curves, axis=0)
        sd = quiet_nan(np.nanstd, curves, axis=0)
        lo, hi = mid - sd, mid + sd
    else:
        mid = quiet_nan(np.nanmedian, curves, axis=0)
        lo = quiet_nan(np.nanpercentile, curves, 25, axis=0)
        hi = quiet_nan(np.nanpercentile, curves, 75, axis=0)
    if log:
        # A log axis cannot show a non-positive value, and clipping the BAND
        # while leaving the line is what silently turns a zero into the panel's
        # floor. Mask instead: a gap in the line is what "not representable
        # here" looks like.
        mid = np.where(mid > 0, mid, np.nan)
        lo = np.where(lo > 0, lo, np.nan)
    line, = ax.plot(x, mid, color=colour, lw=1.0, zorder=3)
    if band != 'none':
        ax.fill_between(x, lo, hi, color=colour, alpha=0.18, lw=0, zorder=2)
    return line


def summary_table(path, dense, checkpoints, persist, results_dir, rows, args,
                  every, num_tasks):
    """The numbers behind the figure, one row per (env, method)."""
    lines = [
        f'# Plasticity diagnostics -- {args.phase}'
        + (f', {cell_selector(args.cells, args.sigma)[1]}'
           if (args.cells or args.sigma) else ''),
        '',
        'Every column is the median over trials, and over the LAST sub-task of',
        'the run unless said otherwise -- the end state, which is what a',
        'plasticity claim is about. Read it beside the lineplot table: this one',
        'says what the network looks like, that one says whether it still learns.',
        '',
        '| Method | Env | Dormant | Persist. | Age | Action shift | Weight RMS '
        '| NTK rank | Step |',
        '|---|---|---|---|---|---|---|---|---|',
    ]

    def last_window(entry):
        if entry is None:
            return None
        x, curves = entry
        keep = x >= (x.max() - (every or x.max()))
        v = float(quiet_nan(np.nanmedian,
                            quiet_nan(np.nanmedian, curves[:, keep], axis=1)))
        # A column every one of whose records was null is MISSING, and the
        # table says so. Printing the NaN that comes back from a median over
        # nothing reads as a measurement that came out strange.
        return None if np.isnan(v) else v

    ck = {}
    if results_dir:
        p = pathlib.Path(results_dir) / 'plasticity_checkpoints.json'
        if p.exists():
            ck = json.loads(p.read_text())['cells']

    envs = [e for e in ENV_TITLES
            if e in (dense.get('dormancy_ckpt') or dense.get('dormancy', {}))] or \
           sorted({e for r in dense.values() for e in r})
    for env in envs:
        present = sorted({m for r in dense.values() for m in r.get(env, {})})
        for method in [m for m in METHOD_ORDER if m in present] + \
                      [m for m in present if m not in METHOD_ORDER]:
            def cell(row):
                v = last_window(dense.get(row, {}).get(env, {}).get(method))
                return '--' if v is None else (f'{v:.3g}')
            age, always = '--', '--'
            for name, methods in ck.items():
                if not name.startswith(env) or method not in methods:
                    continue
                trials = methods[method]['trials']
                v = last_window(dense.get('dormant_age', {}).get(env, {}).get(method))
                age = (f'{v:.1f}' if v is not None else
                       f"{np.median([t['dormant']['age_mean'] for t in trials.values()]):.1f}")
                always = f"{np.median([t['dormant']['always'] for t in trials.values()]):.3g}"
            idx = persist.get(env, {}).get(method)
            pidx = ('--' if idx is None else
                    f'{quiet_nan(np.nanmedian, idx[1][:, 0]):.2f}')
            step = last_window(checkpoints.get('step', {}).get(env, {}).get(method))
            # The behavioural column follows whichever row was drawn: the
            # argmax test by default, the calibrated KL when that row was
            # selected instead.
            churn_cell = (cell('action_shift_ckpt')
                          if 'action_shift_ckpt' in rows else cell('policy_kl'))
            lines.append(
                f"| {METHOD_STYLE.get(method, {}).get('label', method)} "
                f"| {ENV_TITLES.get(env, env)} """
                f"| {cell('dormancy_ckpt') if 'dormancy_ckpt' in dense else cell('dormancy')} "
                f"| {pidx} "
                f"| {age} | {churn_cell} | {cell('weight')} "
                f"| {cell('curvature')} | "
                + ('--' if step is None else f'{step:.3g}') + ' |')

    # The unit count and the meaning of the behavioural column are read off
    # the checkpoint pass's records: 32 hidden units on gymnax, 256 on the
    # mjx bodies; an argmax fraction on a logits head, a normalised action
    # distance on the continuous one (`plasticity_checkpoints.argmax_shift`).
    all_trials = [t for methods in ck.values() for e in methods.values()
                  for t in e['trials'].values()]
    units = sorted({int(t['num_units']) for t in all_trials if 'num_units' in t})
    units_text = '/'.join(str(u) for u in units) if units else '?'
    kinds = {t.get('action_shift_kind', 'argmax') for t in all_trials}
    if kinds == {'distance'}:
        shift_text = [
            '**Action shift** is how far the policy moved over one sub-task: the',
            'mean normalised L2 distance between the ACTIONS the two checkpoints',
            'take on the same probe states, ||a - a\'|| / (sqrt(d) * 2) for',
            'actions in [-1, 1]^d, so 0 is identical behaviour and 1 is every',
            'actuator flipped end to end. The continuous-action counterpart of',
            'the argmax fraction the logits suites report, and equally free of',
            'any softmax temperature; there is no calibrated-KL row on this',
            'body. BOTH checkpoints are scored on the same batch, so it is a',
            'difference between policies and not between batches. It is NOT',
            'the per-update churn C-CHAIN penalises (the action-mean MSE across',
            'one update, online, on the batch at hand); that column is',
            '`--rows churn_ce` and is RL-only. **Step** is the same gap in',
            'parameter space.']
    else:
        shift_text = [
            '**Action shift** is how far the policy moved over one sub-task: the',
            'fraction of probe states whose greedy action differs between the two',
            'checkpoints, BOTH scored on the same batch, so it is a difference',
            'between policies and not between batches. Every method here acts by',
            'argmax, so this needs no softmax temperature -- unlike a KL, which is',
            'undefined for the NE arms until one is chosen. It is NOT the',
            'per-update churn C-CHAIN penalises (a cross-entropy across one',
            'update, online, on the batch at hand); that column is `--rows',
            'churn_ce` and is RL-only. **Step** is the same gap in parameter',
            'space. The calibrated-KL version of this column is `--rows',
            'policy_kl`.']
        if 'distance' in kinds:
            shift_text.insert(0, '(Mixed heads in this table: on the continuous-'
                              'action cells the column is a normalised action '
                              'distance, see `action_shift_kind`.)')
    lines += [
        '',
        f'**Dormant** is the fraction of the {units_text} hidden units at or below ReDo\'s',
        'tau = 0.025, over the last sub-task, on the probe named in the',
        'checkpoint pass\'s meta (`matched` by default: sub-task t\'s own',
        'observation distribution at the checkpoint saved at the end of it).',
        '**Persist.** is the persistence index at a lag of one sub-task:',
        '0 means a dormant set redrawn every checkpoint (functional sparsity),',
        '1 means the same units dormant throughout (capacity gone). **Age** is',
        'the figure\'s dormancy-age row at the last sub-task: the mean number',
        'of consecutive sub-tasks the units dormant then have been dormant.',
        'It is capped by the record (20 sub-tasks), so a value near the cap',
        'is a set of units dead since the start and a value near 1 is a set',
        'replaced every sub-task. Not chance-corrected, unlike Persist.',
        *shift_text,
        '',
        'Persistence, age, step and weight RMS come from `checkpoints.npz` at',
        'one point per sub-task, off the agent named in its meta; every other',
        'column is per record.',
    ]
    # The lineage caveat is TRUE OF THE ELITE ONLY. The elite is an argmax over
    # a population and can hop between lineages from one record to the next, so
    # part of its churn and its low persistence is that hop rather than any
    # plasticity loss. The centroid is a mean over the whole population: it
    # cannot hop, which is why its persistence reads higher and its step lower
    # in the table beside this one. Printing the caveat unconditionally, as
    # this did until 2026-09-09, told the reader of the CENTROID table that its
    # numbers carry an artefact they do not.
    if args.agent == 'centroid':
        lines.append(
            'The NE columns describe the coordinate-wise MEAN of the '
            'population\'s weights, which cannot change lineage between '
            'records -- so unlike the elite table, no part of the churn or '
            'the persistence here is lineage hopping.')
    else:
        lines.append(
            'GA and DNS save an argmax over a population, which can change '
            'lineage between records, so their churn is high and their '
            'persistence low partly for that reason.')
    path.write_text('\n'.join(lines) + '\n')


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('root')
    ap.add_argument('--phase', default='continual')
    ap.add_argument('--sigma', default=None)
    ap.add_argument('--cells', nargs='*', default=None,
                    help='explicit cell directory names, one per '
                         'environment, e.g. CartPole_v1_sigma1.0 '
                         'Acrobot_v1_sigma1.0 MountainCar_v0_sigma0.1 -- '
                         'for a reported grid whose noise differs by '
                         'environment. Mutually exclusive with --sigma. '
                         'Pass the SAME list to '
                         'plasticity_checkpoints.py, or the checkpoint '
                         'rows and the curve rows describe different '
                         'cells.')
    ap.add_argument('--rows', nargs='*', default=DEFAULT_ROWS,
                    choices=list(PLASTICITY_COLUMNS) + list(CHECKPOINT_ROWS)
                    + sorted(LAG_ROWS))
    ap.add_argument('--methods', nargs='*', default=None)
    ap.add_argument('--envs', nargs='*', default=None)
    ap.add_argument('--agent', default='elite', choices=['elite', 'centroid'],
                    help="which NE network the CURVE rows describe. `elite` "
                         "(default) is the best member of the generation -- a "
                         "sampled offspring for ES/NES and a lineage-hopping "
                         "argmax for GA/DNS. `centroid` reads the "
                         "`ne_centroid_*` columns instead: the coordinate-wise "
                         "mean of the population's weights, which is the "
                         "network `centroid_fitness` scores, so the figure "
                         "describes the individual the centroid lineplot "
                         "draws. Pass it together with a `--checkpoints` "
                         "directory built with the same `--agent`. The RL arms "
                         "have one policy and are unaffected.")
    ap.add_argument('--checkpoints', default=None,
                    help='directory holding plasticity_checkpoints.json, from '
                         'scripts/analysis/plasticity_checkpoints.py. Without '
                         'it the `step`/`drift` rows and the persistence '
                         'figure are skipped and the run says so.')
    ap.add_argument('--x', default='generations', choices=['generations', 'steps'])
    ap.add_argument('--width', type=float, default=6.9,
                    help='figure width in INCHES; 6.9 = 17.5 cm, A4 text width')
    ap.add_argument('--panel-height', type=float, default=1.25,
                    help='height of ONE row, inches (default 1.25)')
    ap.add_argument('--font-size', type=float, default=7.0)
    ap.add_argument('--legend', default='separate',
                    choices=['separate', 'inline', 'none'])
    ap.add_argument('--legend-ncol', type=int, default=None)
    ap.add_argument('--band', default='iqr', choices=['iqr', 'sd', 'none'])
    ap.add_argument('--sharey', default='row', choices=['row', 'none'],
                    help="share the y axis along each row (default): unlike "
                         "the reward lineplot, whose units differ per "
                         "environment, every panel of one row here is the SAME "
                         "quantity, and a dormant fraction that means 0.4 in "
                         "one panel and 0.6 in the next is read wrong.")
    ap.add_argument('--smooth', type=int, default=None,
                    help='rolling-median window in records; default 1%% of the run')
    ap.add_argument('--include-superseded', action='store_true')
    ap.add_argument('--assume-episode-length', type=int, default=None)
    ap.add_argument('--out', required=True, help='output stem, no extension')
    args = ap.parse_args()

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    root = pathlib.Path(args.root)
    if args.cells and args.sigma:
        sys.exit('ERROR: pass --cells or --sigma, not both. --cells already '
                 'names the sigma of each environment.')
    sigma_ok, cell_label = cell_selector(args.cells, args.sigma)

    def cell_filter(c):
        return sigma_ok(c) and (not args.envs
                                or any(c.startswith(e) for e in args.envs))

    # Under `--agent centroid` every NE column becomes its `ne_centroid_*`
    # twin, with the elite name kept AFTER it as the fallback: a run written
    # before the trainers logged the centroid has only the elite columns, and
    # dropping it from the figure entirely would be a worse answer than drawing
    # it and saying so. `resolve_column` takes the first name a run actually
    # has, so the fallback is per run and is reported below.
    columns = PLASTICITY_COLUMNS
    if args.agent == 'centroid':
        columns = {row: ([c.replace('ne_elite_', 'ne_centroid_') for c in cols
                          if c.startswith('ne_elite_')] + list(cols))
                   for row, cols in PLASTICITY_COLUMNS.items()}

    dense, missing, per_gen = {}, set(), None
    used_columns = set()
    for row in [r for r in args.rows if r in PLASTICITY_COLUMNS]:
        data, miss, _used, row_per_gen, _monotone, _flat = collect(
            root, args.phase, cell_filter, row, 'generalist',
            args.assume_episode_length, columns=columns,
            monotone_ok=MONOTONE_OK, allow_missing=True)
        used_columns |= {(row, c) for c in _used}
        dense[row] = data
        missing |= {(row, m, c) for m, c in miss}
        # Every row reads the same runs, so their generation costs agree; the
        # first row that finds one settles the x axis.
        per_gen = per_gen or row_per_gen
    if not dense and not args.checkpoints:
        sys.exit('nothing to draw: no dense rows selected and no --checkpoints')
    if per_gen is None:
        # No dense row settled the clock -- a checkpoint-only figure, which
        # is every MiniGrid one -- so read it off any NE run in the cells, as
        # the lineplot's `budget` does, and the x axis stays in generations
        # like the lineplot beside it.
        for trial_dir in sorted((root / args.phase).glob('*/*/trial_*')):
            if not cell_filter(trial_dir.parent.name):
                continue
            cfg, res = load_config(trial_dir)
            if cfg:
                _total, pg = budget(cfg, res, args.assume_episode_length)
                if pg:
                    per_gen = pg
                    break

    every, total, edges = boundaries(root, args.phase, cell_filter,
                                     args.assume_episode_length)
    num_tasks = len(edges) - 1 if edges is not None else 0

    # Superseded and mislabelled arms, decided from what the runs RECORDED --
    # the lineplot's own predicates, so the two figures never show different
    # sets of methods for the same tree.
    # The checkpoint rows are loaded HERE, before the arm set is settled: a
    # figure drawn from checkpoint rows alone (MiniGrid, whose shared runners
    # log no per-record plasticity for the NE arms) has no dense row, and an
    # arm set read off the dense rows alone was empty, so `--methods` refused
    # every arm the checkpoint pass had just measured.
    checkpoints, ck_problem = load_checkpoint_rows(
        args.checkpoints, cell_filter, edges)
    if ck_problem:
        print(f'WARNING: {ck_problem} -- the checkpoint rows are empty')
    present = {m for row in dense.values() for env in row for m in row[env]}
    present |= {m for row in checkpoints.values() for env in row for m in row[env]}
    dropped_methods: set[str] = set()
    # `--methods` restricts the DATA, for the same reason the lineplot does it
    # there: applied at plot time only it filtered the panel grid but not the
    # table and not the persistence figure, so one figure reported arms the
    # other did not show. Folding it into `dropped_methods` reuses the single
    # drop below.
    if args.methods:
        selectable = present
        unknown = sorted(set(args.methods) - selectable)
        if unknown:
            sys.exit(f'ERROR: --methods names {unknown}, which have no runs '
                     f'under {root}/{args.phase}'
                     + (f' at {cell_label}' if cell_label else '')
                     + f'. Present: {sorted(selectable)}')
        dropped_methods |= selectable - set(args.methods)
        print('note: --methods restricts the panels, the table AND the '
              'persistence figure to ' + ' '.join(args.methods))
        args.methods = None

    for method in sorted((present - dropped_methods) & set(SUPERSEDED)):
        predicate, phases, reason = SUPERSEDED[method]
        state = ('none' if args.phase not in phases else
                 defect_state(root, args.phase, method, cell_filter, predicate))
        if state == 'none':
            continue
        if state == 'mixed':
            sys.exit(f'ERROR: some but not all {method!r} trials are defective '
                     f'({reason}). One directory holds two algorithms.')
        if args.include_superseded:
            print(f'WARNING: plotting superseded arm {method!r} -- {reason}')
            continue
        dropped_methods.add(method)
        print(f'note: dropping superseded arm {method!r} -- {reason}')
    for method in sorted((present - dropped_methods) & set(MISLABELLED)):
        predicate, phases, reason = MISLABELLED[method]
        state = ('none' if args.phase not in phases else
                 defect_state(root, args.phase, method, cell_filter, predicate))
        if state != 'none':
            print(f'WARNING: {method!r} is plotted under its own name but {reason}')

    for method, cell, n_sat, n_trials in saturated_arms(args.checkpoints,
                                                        cell_filter):
        if method in dropped_methods:
            continue
        print(f'WARNING: {method}/{cell}: {n_sat} of {n_trials} trials have '
              'parameters at float32\'s maximum. Those genotypes have run '
              'away and stopped at the representable limit, so their weight '
              'row is a ceiling and their step norm is 0 because consecutive '
              'checkpoints are the same saturated vector -- not a search that '
              'settled.')
    for row in CHECKPOINT_ROWS:
        if row in args.rows:
            dense[row] = checkpoints[row]

    # The lag curves feed BOTH the standalone `_persistence` figure and, when
    # selected, the main grid's `persistence_lag` row -- one computation, so
    # the two can never disagree about a number they both show.
    persist, dropped = persistence_curves(args.checkpoints, cell_filter)
    if 'persistence_lag' in args.rows and persist:
        dense['persistence_lag'] = persist

    # One drop, applied to every source of rows there is. Doing it before the
    # checkpoint pass was merged left `ga` out of the four dense rows and in
    # the step row and the persistence figure -- one figure showing two
    # different sets of methods, which is worse than showing either.
    for method in dropped_methods:
        for row in list(dense.values()) + list(checkpoints.values()):
            for env in row:
                row[env].pop(method, None)

    rows = [r for r in args.rows if dense.get(r)]
    dropped_rows = [r for r in args.rows if r not in rows]
    if dropped_rows:
        print(f'WARNING: no data for row(s) {", ".join(dropped_rows)}; '
              'they are absent from the figure')
    if not rows:
        sys.exit('no row has any data')

    envs = [e for e in ENV_TITLES if any(e in dense[r] for r in rows)] + \
           sorted({e for r in rows for e in dense[r]} - set(ENV_TITLES))
    scale = per_gen if (args.x == 'generations' and per_gen) else 1.0
    xlabel = 'Generations' if scale != 1.0 else 'Environment steps'

    fs = args.font_size
    plt.rcParams.update({
        'font.size': fs, 'axes.labelsize': fs, 'axes.titlesize': fs + 1,
        'xtick.labelsize': fs - 0.5, 'ytick.labelsize': fs - 0.5,
        'legend.fontsize': fs, 'axes.linewidth': 0.6,
        'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
        'xtick.major.size': 2.5, 'ytick.major.size': 2.5,
    })
    fig, axes = plt.subplots(len(rows), len(envs),
                             figsize=(args.width,
                                      args.panel_height * len(rows)),
                             sharey=args.sharey, squeeze=False)
    # x sharing by hand rather than `sharex='col'`: a LAG row's x axis is a
    # lag in sub-tasks and the rest are training time, so one shared axis per
    # column would squash both onto whichever range is wider.
    time_rows = [r for r, row in enumerate(rows) if row not in LAG_ROWS]
    for c in range(len(envs)):
        for r in time_rows[1:]:
            axes[r][c].sharex(axes[time_rows[0]][c])
    for r in range(len(rows)):
        for c in range(1, len(envs)):
            if rows[r] in LAG_ROWS:
                axes[r][c].sharex(axes[r][0])
    # `sharex='col'` used to hide the inner x tick labels for us.
    last_time = time_rows[-1] if time_rows else None
    drawn: dict[str, object] = {}
    for r, row in enumerate(rows):
        for c, env in enumerate(envs):
            ax = axes[r][c]
            lag_row = row in LAG_ROWS
            if edges is not None and not lag_row:
                for b in edges[1:-1]:
                    ax.axvline(b / scale, color='0.6', lw=0.45, ls='--', zorder=0)
            if row in ('persistence', 'persistence_lag'):
                # The two readings of the row, drawn so the panel can be read
                # without the caption: 0 is the chance level (a dormant set
                # redrawn independently each sub-task -- functional sparsity),
                # 1 is the same units dormant throughout.
                for y in (0.0, 1.0):
                    ax.axhline(y, color='0.35', lw=0.5, ls=':', zorder=1)
            cell = dense[row].get(env, {})
            methods = args.methods or (
                [m for m in METHOD_ORDER if m in cell]
                + [m for m in sorted(cell) if m not in METHOD_ORDER])
            for method in methods:
                if method not in cell:
                    continue
                x, curves = cell[method]
                # Columns logged every k-th record (the population-width
                # diagnostics, every 10 generations) are NaN everywhere else.
                # Compress them to the records that were measured, or the
                # rolling median is NaN over every window and a log axis has
                # nothing to scale.
                measured = ~np.all(np.isnan(curves), axis=0)
                if measured.any() and not measured.all():
                    x, curves = x[measured], curves[:, measured]
                if lag_row:
                    # Median only, with a marker per lag. The band is left off
                    # deliberately: nineteen lags x nine methods of overlapping
                    # IQR is unreadable, and the standalone figure this row is
                    # lifted from made the same call.
                    line, = ax.plot(
                        x, quiet_nan(np.nanmedian, curves, axis=0),
                        color=METHOD_STYLE.get(method, {}).get('color'),
                        lw=1.0, marker='o', ms=1.5, zorder=3)
                    drawn.setdefault(method, line)
                    continue
                win = (1 if row in CHECKPOINT_ROWS else
                       args.smooth if args.smooth is not None
                       else max((int(curves.shape[1] * 0.01) | 1), 1))
                line = draw(ax, x / scale, curves,
                            METHOD_STYLE.get(method, {}).get('color'),
                            args.band, win, row in LOG_ROWS)
                drawn.setdefault(method, line)
            if row in LOG_ROWS:
                ax.set_yscale('log')
            elif row in SYMLOG_ROWS:
                ax.set_yscale('symlog', linthresh=SYMLOG_ROWS[row],
                              linscale=0.4)
                if row in NONNEGATIVE_ROWS:
                    ax.set_ylim(bottom=0)
            if r == 0:
                ax.set_title(ENV_TITLES.get(env, env), fontweight='bold')
            if lag_row:
                # Its own x label under its own panels: this row's axis is not
                # the figure's, and an unlabelled one would be read as time.
                ax.set_xlabel('Lag (sub-tasks)')
                ax.tick_params(labelbottom=True)
            else:
                if r == last_time:
                    ax.set_xlabel(xlabel)
                ax.tick_params(labelbottom=(r == last_time))
            if c == 0:
                ax.set_ylabel(ROW_LABELS.get(row, row))
            if c > 0 and args.sharey == 'row':
                ax.tick_params(labelleft=False)
            ax.margins(x=0 if not lag_row else 0.02)
            ax.spines[['top', 'right']].set_visible(False)

    if 'dormant_age' in rows and num_tasks:
        # The censoring ceiling on the age row: checkpoint t is the end of
        # sub-task t, so no unit can have been dormant for more than t + 1 of
        # them. A curve on this line is a set of units dead since the first
        # checkpoint; one flat near 1 is a set replaced every sub-task. Drawn
        # AFTER every curve is in, with `scaley=False`, so the data and not
        # the ceiling set the range -- the ceiling reaches the record length
        # and would squash arms whose ages stay in the single digits onto the
        # floor; where it exceeds the range it leaves the panel. Setting the
        # bottom limit turns y autoscaling off, which is why this cannot run
        # inside the drawing loop before the row's other panels are drawn.
        for ax in axes[rows.index('dormant_age')]:
            ax.plot(edges[1:] / scale,
                    np.arange(1, num_tasks + 1), color='0.35', lw=0.5,
                    ls=':', zorder=1, scaley=False)
            ax.set_ylim(bottom=0)

    order = [m for m in METHOD_ORDER if m in drawn] + \
            [m for m in drawn if m not in METHOD_ORDER]
    handles = [drawn[m] for m in order]
    labels = [METHOD_STYLE.get(m, {}).get('label', m) for m in order]
    if args.legend == 'inline':
        fig.legend(handles, labels, frameon=False, ncol=min(len(order), 7),
                   loc='upper center', bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout()

    stem = pathlib.Path(args.out)
    stem.parent.mkdir(parents=True, exist_ok=True)
    written = []
    for ext in ('png', 'pdf'):
        # Not `with_suffix`: a stem ending `_sigma1.0` looks to pathlib like a
        # `.0` suffix and would be truncated to `_sigma1`.
        out = stem.parent / f'{stem.name}.{ext}'
        fig.savefig(out, dpi=400, bbox_inches='tight')
        written.append(out.name)
    if args.legend == 'separate':
        ncol = args.legend_ncol or len(order)
        nrow = 1 + (len(order) - 1) // ncol
        lfig = plt.figure(figsize=(args.width, 0.16 * nrow + 0.06))
        lfig.legend(handles, labels, frameon=False, ncol=ncol, loc='center',
                    handlelength=1.6, columnspacing=1.2, handletextpad=0.5,
                    borderpad=0)
        for ext in ('png', 'pdf'):
            out = stem.parent / f'{stem.name}_legend.{ext}'
            lfig.savefig(out, dpi=400)
            written.append(out.name)
        plt.close(lfig)

    # ---- the persistence figure: dead, or just sparse right now ----
    for env in persist:
        for method in dropped_methods:
            persist[env].pop(method, None)
    if persist:
        pfig, paxes = plt.subplots(1, len(envs),
                                   figsize=(args.width, args.panel_height + 0.35),
                                   squeeze=False)
        for ax, env in zip(paxes[0], envs):
            cell = persist.get(env, {})
            ax.axhline(0.0, color='0.6', lw=0.5, ls='--', zorder=1)
            for method in ([m for m in METHOD_ORDER if m in cell]
                           + [m for m in sorted(cell) if m not in METHOD_ORDER]):
                lags, idx = cell[method]
                ax.plot(lags, quiet_nan(np.nanmedian, idx, axis=0),
                        color=METHOD_STYLE.get(method, {}).get('color'),
                        lw=1.0, marker='o', ms=1.6, zorder=3)
            ax.set_title(ENV_TITLES.get(env, env), fontweight='bold')
            ax.set_xlabel('Lag (sub-tasks)')
            ax.set_ylim(-0.05, 1.05)
            ax.spines[['top', 'right']].set_visible(False)
        paxes[0][0].set_ylabel('Dormancy\npersistence index')
        pfig.tight_layout()
        for ext in ('png', 'pdf'):
            out = stem.parent / f'{stem.name}_persistence.{ext}'
            pfig.savefig(out, dpi=400, bbox_inches='tight')
            written.append(out.name)
        plt.close(pfig)
        if dropped:
            print(f'note: {len(dropped)} trial(s) left out of the persistence '
                  'figure -- no unit was ever dormant in them, so there is no '
                  'survival to normalise:')
            for method, cell_name, trial, p in dropped[:6]:
                print(f'  {method}/{cell_name}/{trial} (instant={p:.3g})')
    elif args.checkpoints:
        print('WARNING: no persistence curves -- is plasticity_checkpoints.json '
              'covering these cells?')

    table = stem.parent / f'{stem.name}_table.md'
    summary_table(table, dense, checkpoints, persist, args.checkpoints, rows,
                  args, every, num_tasks)
    written.append(table.name)

    if missing:
        print(f'WARNING: {len(missing)} (row, method) pair(s) have no column '
              'and are ABSENT from that row:')
        for row, method, cols in sorted(missing):
            print(f'  {row}: {method} has none of {cols}')
    print(f'wrote into {stem.parent}/: ' + ', '.join(written))
    print(f'  {len(rows)} x {len(envs)} panels, {args.width:.2f} x '
          f'{args.panel_height * len(rows):.2f} in, {args.font_size} pt -- '
          'include at 1:1')
    if per_gen:
        print(f'  action-shift clock: 1 NE generation = {per_gen:,} steps '
              'against one PPO rollout; within ~33% and not equal.')
    # Under `--agent centroid` a run that predates the `ne_centroid_*` columns
    # is drawn from the ELITE ones instead. Silently mixing the two would put
    # two different networks in one panel, so name it.
    if args.agent == 'centroid':
        stale = sorted({(row, c) for row, c in used_columns
                        if c.startswith('ne_elite_')})
        if stale:
            print('WARNING: --agent centroid, but these rows fell back to the '
                  'ELITE column on runs that predate ne_centroid_* -- re-run '
                  'those trials or read the row as the elite:')
            for row, c in stale:
                print(f'  {row}: {c}')
    for env in envs:
        n = {m: dense[rows[0]][env][m][1].shape[0]
             for m in sorted(dense[rows[0]].get(env, {}))}
        print(f'  {env} ({rows[0]}): ' + ', '.join(f'{m}(n={v})'
                                                   for m, v in n.items()))
    return 0


if __name__ == '__main__':
    sys.exit(main())
