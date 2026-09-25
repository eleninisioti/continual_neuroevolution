"""Evaluation metrics for continual learning runs.

A single place to define what the reported numbers *mean*, so that every
training script, post-processing script and report computes them the same way.
Currently implements success rate, zero-shot transfer and speed-up; the registry
below is the extension point for the metrics that follow. docs/metrics.md states
the same definitions in prose.

Design decisions (see docstrings for rationale):

  1. **Outcome metrics consume evaluation returns, never training fitness.**
     The fitness a population sees during search is a biased quantity: it is
     the maximum over N noisy per-episode scores, so it drifts upward with
     population size even when nothing has actually improved. A success rate
     computed from it would confound "larger populations are better" with
     "larger populations take more draws from the same noise". Success rate and
     zero-shot transfer therefore take returns from dedicated evaluation
     episodes run *after* the sub-task finished.

     Speed-up is the exception and cannot be otherwise: "how long until the
     agent reached the threshold" is a statement about the search trajectory,
     and the only record of that is the training fitness trace. It is read from
     a separate field, ``TaskEvaluation.training_trace``, so that no metric
     touches training fitness by accident, and it carries the bias above --
     see ``speed_up``.

  2. **Raw per-episode returns are the interface, not summaries.** A metric
     that only ever sees a mean cannot express dispersion, quantiles, or
     per-episode solve probability, and adding one later would mean re-running
     the experiments. Storing the individual episode returns costs almost
     nothing and keeps future metrics a post-processing change.

  3. **A sub-task is solved when a location statistic of its evaluation
     returns reaches the threshold**, with the statistic explicit and
     swappable (`reducer`). The default is the mean, which is what the
     gymnax training scripts and the PBT report already use. The alternative
     readings -- median, or "a fraction f of episodes individually clear the
     threshold" -- are available rather than hard-coded, because on bimodal
     environments such as Acrobot they differ materially and the choice should
     be visible in the call site.

  4. **Thresholds are data, and the repo currently disagrees with itself about
     them.** Both sets live here, named, so that a report states which one it
     used instead of silently baking in constants. See THRESHOLD_SETS.

  5. **One agent per sub-task is scored, never the population.** Every metric
     here reads the agent that the run's selection rule picked at the end of a
     sub-task -- by default the best member of its final generation. Scoring
     the whole generation instead would make each number a maximum over N
     draws and so a function of population size (decision 1 again), and it
     would cost a rollout per member. The selection rule is recorded in
     ``TaskEvaluation.source``.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np

__all__ = [
    "THRESHOLD_SETS",
    "get_threshold",
    "TaskEvaluation",
    "RunEvaluation",
    "register_metric",
    "METRICS",
    "success_rate",
    "zero_shot_transfer",
    "generations_to_threshold",
    "speed_up",
    "compute_metrics",
    "aggregate",
]


# ---------------------------------------------------------------------------
# Solved thresholds
# ---------------------------------------------------------------------------

# Three named sets, because the repo contains all of them and they are not
# interchangeable:
#
#   "repo"     -- the constants in source/studies/gymnax/train_*_gymnax_continual.py.
#   "pbt_doc"  -- the values docs/pbt_rebuttal_hypotheses.md reports PBT
#                 against. That document states explicitly that any GA/ES
#                 numbers compared with its table must be recomputed at these
#                 values, so a comparison using "repo" thresholds is invalid.
#   "rebuttal" -- the set the NeurIPS 2026 rebuttal figures report against;
#                 scripts/neurips_2026_rebuttal/make_figures.py reads it
#                 directly, so this is the one definition behind every table
#                 in that figure set.
#
# The choice matters. On Acrobot the repo's -70 sits past what the gymnax
# neuroevolution runs reach at any population size, so every cell floors at
# 0.000 and a population-size trend that is clearly present at -90 becomes
# invisible. Report the set you used.
#
# "rebuttal" is looser still, for two reasons specific to that suite. Its
# episodes are capped at 500 steps rather than gymnax's defaults, so the
# standard thresholds describe a different task; and the two method families
# do not report the same statistic -- NE logs `best_fitness`, a max over its
# 512-member population, while RL logs `mean_reward` for a single policy, so
# NE sits ~8 (Acrobot) to ~45 (MountainCar) reward above RL by construction.
# A threshold inside that gap reads as "RL never solves the task" when it is
# really measuring the max-over-population advantage. These values sit below
# the gap, so the column reports whether a method reached competence.
#
# CartPole is 400 rather than the usual 475 for the same reason: with a 500-step
# cap the last 75 reward separate a policy that balances the whole episode from
# one that balances all but a handful of steps, which is not the distinction
# these tables are about.
THRESHOLD_SETS: Mapping[str, Mapping[str, float]] = {
    "repo": {
        "CartPole-v1": 475.0,
        "Acrobot-v1": -70.0,
        "MountainCar-v0": -110.0,
    },
    "pbt_doc": {
        "CartPole-v1": 475.0,
        "Acrobot-v1": -90.0,
        "MountainCar-v0": -120.0,
    },
    "rebuttal": {
        "CartPole-v1": 400.0,
        "Acrobot-v1": -120.0,
        "MountainCar-v0": -200.0,
    },
}
# DeepSea: reaching the treasure pays 1.0 and anything else 0.0.
for _set in THRESHOLD_SETS.values():
    _set.update({f"DeepSea{_n}-bsuite": 0.5 for _n in (8, 10, 12, 14, 16, 20)})


def get_threshold(env: str, threshold_set: str = "repo") -> float:
    """Solved threshold for ``env`` under a named threshold set."""
    try:
        table = THRESHOLD_SETS[threshold_set]
    except KeyError:
        raise KeyError(
            f"Unknown threshold set {threshold_set!r}; "
            f"expected one of {sorted(THRESHOLD_SETS)}"
        ) from None
    try:
        return table[env]
    except KeyError:
        raise KeyError(
            f"No {threshold_set!r} threshold for environment {env!r}; "
            f"known: {sorted(table)}"
        ) from None


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------

@dataclass
class TaskEvaluation:
    """Evaluation of one agent on one sub-task, after training on it finished.

    Attributes:
        task_idx: Index of the sub-task within the continual sequence.
        returns: Undiscounted return of each evaluation episode. These are the
            raw episode outcomes, not a summary -- see design decision 2.
        source: How the evaluated agent was chosen, e.g. "final_generation_best"
            or "incumbent". Recorded because the selection rule is part of the
            metric's meaning: picking the best of a population is not the same
            claim as reporting what the optimizer would hand you.
        zero_shot_returns: Undiscounted return of each episode of *this*
            sub-task's agent -- the same agent ``returns`` scores, picked by the
            same ``source`` rule -- evaluated on sub-task ``task_idx + 1``
            before any search on it. None when the run has no successor
            sub-task, or when the evaluation did not measure zero-shot
            transfer.
        training_trace: Training fitness of the best agent in each generation
            spent on this sub-task, in generation order. The one piece of
            training-side data any metric reads, and only ``speed_up`` reads it,
            because time-to-threshold exists nowhere else. None when the run's
            trace was not loaded.
    """

    task_idx: int
    returns: np.ndarray
    source: str = "final_generation_best"
    zero_shot_returns: np.ndarray | None = None
    training_trace: np.ndarray | None = None

    def __post_init__(self):
        self.returns = np.asarray(self.returns, dtype=float).ravel()
        if self.returns.size == 0:
            raise ValueError(f"task {self.task_idx} has no evaluation episodes")
        if self.zero_shot_returns is not None:
            self.zero_shot_returns = np.asarray(
                self.zero_shot_returns, dtype=float).ravel()
            if self.zero_shot_returns.size == 0:
                raise ValueError(
                    f"task {self.task_idx}: zero_shot_returns is empty; pass None "
                    "when the sub-task has no zero-shot evaluation"
                )
        if self.training_trace is not None:
            self.training_trace = np.asarray(self.training_trace, dtype=float).ravel()
            if self.training_trace.size == 0:
                raise ValueError(
                    f"task {self.task_idx}: training_trace is empty; pass None "
                    "when the trace is unavailable"
                )

    @property
    def mean_return(self) -> float:
        return float(self.returns.mean())


@dataclass
class RunEvaluation:
    """All sub-task evaluations from a single training run."""

    env: str
    tasks: Sequence[TaskEvaluation]
    method: str = ""
    pop_size: int | None = None
    trial: int | None = None
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        if not self.tasks:
            raise ValueError("RunEvaluation needs at least one TaskEvaluation")
        self.tasks = sorted(self.tasks, key=lambda t: t.task_idx)

    @property
    def num_tasks(self) -> int:
        return len(self.tasks)

    def first(self, n: int) -> "RunEvaluation":
        """A view over the first ``n`` sub-tasks.

        Useful when runs of different length must be compared: a deterministic
        per-trial task sequence means the first n sub-tasks of a long run are
        identical to those of a short one.

        The last sub-task in the view loses its zero-shot returns: they are
        scored on sub-task n, which this view does not contain. Keeping them
        would make an n-sub-task view of a long run differ from an actual
        n-sub-task run, which is the one thing this method exists to avoid.
        """
        kept = []
        for i, t in enumerate(self.tasks[:n]):
            last = i == min(n, len(self.tasks)) - 1
            kept.append(TaskEvaluation(
                task_idx=t.task_idx, returns=t.returns, source=t.source,
                zero_shot_returns=None if last else t.zero_shot_returns,
                training_trace=t.training_trace,
            ))
        return RunEvaluation(env=self.env, tasks=kept,
                             method=self.method, pop_size=self.pop_size,
                             trial=self.trial, metadata=dict(self.metadata))


# ---------------------------------------------------------------------------
# Metric registry
# ---------------------------------------------------------------------------

METRICS: dict[str, Callable[..., float]] = {}


def register_metric(name: str) -> Callable:
    """Register a metric under ``name``.

    A metric takes a RunEvaluation as its first positional argument, accepts
    keyword arguments only, and returns a single float per run. Aggregation
    across seeds is the caller's job -- see ``aggregate``.

        @register_metric("my_metric")
        def my_metric(run: RunEvaluation, *, threshold: float) -> float:
            ...
    """

    def decorator(fn):
        if name in METRICS:
            raise ValueError(f"metric {name!r} is already registered")
        METRICS[name] = fn
        return fn

    return decorator


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

@register_metric("success_rate")
def success_rate(
    run: RunEvaluation,
    *,
    threshold: float | None = None,
    threshold_set: str = "repo",
    reducer: Callable[[np.ndarray], float] = np.mean,
) -> float:
    """Fraction of sub-tasks solved.

    A sub-task counts as solved when ``reducer`` applied to its evaluation
    returns reaches ``threshold``. With the default reducer this reads: run the
    evaluation episodes at the end of the sub-task, take the mean return, and
    compare it with the threshold.

    Nothing observed during training enters this number. The one place the
    search still shows through is *which* agent was evaluated -- some rule has
    to pick one, and for a population method that rule necessarily uses
    training fitness. That is recorded per task in ``TaskEvaluation.source``
    rather than hidden.

    Args:
        run: The run to score.
        threshold: Explicit solved threshold. If omitted it is looked up from
            ``threshold_set`` and ``run.env``.
        threshold_set: Named set to look the threshold up in when ``threshold``
            is not given. See THRESHOLD_SETS.
        reducer: Statistic mapping a task's evaluation returns to the value
            compared against the threshold. ``np.mean`` (default) or
            ``np.median``; pass a partial for a fraction-of-episodes rule.

    Returns:
        A value in [0, 1].
    """
    if threshold is None:
        threshold = get_threshold(run.env, threshold_set)
    solved = [float(reducer(t.returns)) >= threshold for t in run.tasks]
    return float(np.mean(solved))


@register_metric("zero_shot_transfer")
def zero_shot_transfer(
    run: RunEvaluation,
    *,
    reducer: Callable[[np.ndarray], float] = np.mean,
) -> float:
    """Mean return on a sub-task before any search has been done on it.

    The paper's definition is the return on task tau immediately after training
    on task tau - 1 and before training on tau begins, averaged over the
    sequence:

        ZT_tau = R_tau^(0),    ZT = (1/T) sum_tau ZT_tau.

    Two things that definition leaves open are settled the same way the rest of
    this module settles them:

    *Which agent.* R^(0) is written for a single agent, while a population
    method carries a whole generation across the task boundary. The agent
    scored here is the one the run's selection rule already picked at the end of
    sub-task tau - 1 -- by default the best member of its final generation --
    which is the same agent success rate scores, so the two metrics describe one
    agent's trajectory through the sequence rather than two different objects.
    Taking a maximum over the generation instead would make ZT climb with
    population size on its own and cost a rollout per member.

    *One episode or many.* The paper says first episode, i.e. a single noisy
    draw. Evaluation stores several fresh episodes instead and ``reducer``
    (default the mean) collapses them, which estimates the same quantity with
    less variance. Pass a reducer that takes element 0 for the literal
    single-episode definition.

    Averaging runs over the sub-tasks that *have* a successor, tau = 1..T-1:
    task 0 has no preceding task to transfer from, so it contributes nothing,
    and the T in the paper's denominator is that many terms.

    Args:
        run: The run to score. Its tasks must carry ``zero_shot_returns``,
            which ``source/studies/evaluate_continual.py`` produces.
        reducer: Statistic mapping a sub-task's zero-shot episode returns to
            ZT_tau.

    Returns:
        Mean ZT over sub-tasks, in the environment's own return units. Not
        normalised, so it is only comparable within an environment.
    """
    per_task = [float(reducer(t.zero_shot_returns))
                for t in run.tasks if t.zero_shot_returns is not None]
    if not per_task:
        raise ValueError(
            f"run {run.method}/{run.env}/pop={run.pop_size}/trial={run.trial} has "
            "no zero-shot evaluations; re-run source.studies.gymnax.evaluate_continual "
            "so that each sub-task's agent is also scored on the next sub-task"
        )
    return float(np.mean(per_task))


@register_metric("eval_performance")
def eval_performance(
    run: RunEvaluation,
    *,
    reducer: Callable[[np.ndarray], float] = np.mean,
) -> float:
    """Mean evaluation return of each sub-task's own agent, averaged over sub-tasks.

        P = (1/T) sum_tau reducer(R_tau)

    where R_tau are the fresh evaluation episodes of the agent sub-task tau
    ended with. It is `success_rate` without the threshold: the same agents,
    the same episodes, reported on the environment's own return scale instead
    of being compared against `SOLVED_THRESHOLDS`. That is the point of adding
    it -- CheetahRun has no threshold to compare against, and on gymnax the
    threshold choice moves the success rate, while this number does not move
    with anything but the data.

    What it is *not* is a memory measure. Each sub-task is scored by the agent
    that finished it, so nothing here is affected by what later sub-tasks do to
    that agent; a method that forgets everything scores exactly as well as one
    that retains everything. Read it against `F`, which is the same agents
    measured again at the end of the sequence, and the pair separates "learnt
    it" from "still has it".

    Not normalised, so like `ZT` it is comparable down a column and meaningless
    across environments.

    Args:
        run: The run to score.
        reducer: Statistic mapping a sub-task's evaluation episodes to its
            value. ``np.mean`` (default) or ``np.median``.

    Returns:
        Mean over sub-tasks, in the environment's own return units.
    """
    return float(np.mean([float(reducer(t.returns)) for t in run.tasks]))


def generations_to_threshold(
    trace: np.ndarray, threshold: float, unreached: str = "budget",
) -> float:
    """Generations spent on a sub-task before its threshold was first reached.

    Counted from 1, so a sub-task solved by its opening generation costs 1.

    Args:
        trace: Best-agent training fitness per generation, in generation order.
        threshold: The sub-task's solved threshold.
        unreached: What a sub-task whose threshold is never reached costs.
            "budget" charges the generations actually spent, which is a lower
            bound on the true cost -- the run might have needed twice as many.
            "nan" refuses to guess and propagates.

    Returns:
        Generation count, or nan.
    """
    hit = np.flatnonzero(np.asarray(trace, dtype=float) >= threshold)
    if hit.size:
        return float(hit[0] + 1)
    if unreached == "budget":
        return float(len(trace))
    if unreached == "nan":
        return float("nan")
    raise ValueError(f"unknown unreached policy {unreached!r}; "
                     "expected 'budget' or 'nan'")


@register_metric("speed_up")
def speed_up(
    run: RunEvaluation,
    *,
    threshold: float | None = None,
    threshold_set: str = "repo",
    scratch_generations: float | Mapping[int, float] | None = None,
    unreached: str = "budget",
) -> float:
    """How much faster a warm-started sub-task reaches its threshold.

    Following the paper, with E_tau the time to reach R_tau >= kappa_tau when
    warm-started and E_tau^scratch the same starting from nothing:

        SU_tau = (E_tau^scratch - E_tau) / E_tau^scratch,
        SU     = (1/T) sum_tau SU_tau.

    Positive means the warm start helped, 0 means it made no difference,
    negative means it hurt -- prior weights that have to be undone before the
    new sub-task can be learned. The scale is unbounded below: a warm start
    twice as slow as scratch scores -1.

    Three decisions the paper's formula leaves to the implementation:

    *Time in generations, not episodes.* A generation costs pop_size x num_evals
    episodes, and both terms of the ratio come from runs at the same population
    size, so that factor cancels and SU is unchanged. Supplying
    ``scratch_generations`` from a run at a *different* population size breaks
    that cancellation -- convert to a common unit first.

    *Whose fitness.* The best agent in each generation, matching the agent that
    every other metric in this module scores. Note what this inherits: the best
    of N is a maximum over N noisy draws, so a larger population crosses the
    threshold earlier partly for statistical reasons. Both E terms are affected
    in the same direction, which damps but does not remove the effect. SU across
    a population-size sweep is therefore a weaker claim than SU between methods
    at one population size.

    *What "from scratch" is.* By default sub-task 0 of this same run: it starts
    from a random initialisation, so it is literally the scratch condition, at
    the same population size and with the same budget. It is a different
    sub-task, though, so any difficulty gap between sub-task 0 and sub-task tau
    lands in SU. Pass ``scratch_generations`` -- a scalar, or a mapping from
    task index to its own scratch cost -- to use dedicated from-scratch runs
    instead, which is the cleaner comparison when they exist.

    Averaged over tau = 1..T-1; sub-task 0 is the reference and is not scored
    against itself.

    Args:
        run: The run to score. Its tasks must carry ``training_trace``.
        threshold: Explicit solved threshold; looked up from ``threshold_set``
            and ``run.env`` when omitted.
        threshold_set: Named set for that lookup. See THRESHOLD_SETS.
        scratch_generations: Scratch cost E^scratch. None uses sub-task 0 of
            this run; a scalar applies to every sub-task; a mapping gives each
            sub-task its own.
        unreached: Passed to ``generations_to_threshold`` for sub-tasks whose
            threshold is never reached.

    Returns:
        Mean SU over sub-tasks. Unbounded below, at most 1.
    """
    if threshold is None:
        threshold = get_threshold(run.env, threshold_set)
    missing = [t.task_idx for t in run.tasks if t.training_trace is None]
    if missing:
        raise ValueError(
            f"run {run.method}/{run.env}/pop={run.pop_size}/trial={run.trial} has "
            f"no training trace for sub-task(s) {missing}; speed-up is a "
            "statement about the search trajectory and cannot be recovered from "
            "the evaluation returns"
        )

    gens = {t.task_idx: generations_to_threshold(t.training_trace, threshold,
                                                 unreached)
            for t in run.tasks}

    if scratch_generations is None:
        reference = run.tasks[0].task_idx
        scratch = {t.task_idx: gens[reference] for t in run.tasks}
        scored = [t.task_idx for t in run.tasks[1:]]
    elif isinstance(scratch_generations, Mapping):
        scratch = {int(k): float(v) for k, v in scratch_generations.items()}
        scored = [t.task_idx for t in run.tasks if t.task_idx in scratch]
    else:
        scratch = {t.task_idx: float(scratch_generations) for t in run.tasks}
        scored = [t.task_idx for t in run.tasks]

    per_task = [(scratch[i] - gens[i]) / scratch[i]
                for i in scored if scratch.get(i)]
    if not per_task:
        raise ValueError(
            "speed-up has no sub-task to score; a run needs at least one "
            "sub-task with a scratch reference, and that reference must be "
            "non-zero"
        )
    # No nanmean: under unreached="nan" a censored sub-task makes the whole
    # run's speed-up undefined rather than quietly shrinking the average's
    # denominator. Dropping those sub-tasks is a decision for the caller.
    return float(np.mean(per_task))


# ---------------------------------------------------------------------------
# Driving the registry
# ---------------------------------------------------------------------------

def compute_metrics(run: RunEvaluation, names: Iterable[str] | None = None,
                    **kwargs) -> dict[str, float]:
    """Compute several registered metrics over one run.

    Each keyword argument goes to the metrics that accept it and is skipped for
    those that do not, since the metrics no longer share a parameter list
    (``threshold`` means nothing to zero-shot transfer). A keyword no requested
    metric accepts is an error rather than a no-op, so a typo still surfaces.
    """
    names = list(METRICS) if names is None else list(names)
    unknown = [n for n in names if n not in METRICS]
    if unknown:
        raise KeyError(f"unknown metric(s) {unknown}; known: {sorted(METRICS)}")

    accepted = {n: set(inspect.signature(METRICS[n]).parameters) for n in names}
    unused = [k for k in kwargs if not any(k in a for a in accepted.values())]
    if unused:
        raise TypeError(
            f"keyword(s) {sorted(unused)} accepted by none of the metrics {names}"
        )
    return {n: METRICS[n](run, **{k: v for k, v in kwargs.items() if k in accepted[n]})
            for n in names}


def aggregate(runs: Sequence[RunEvaluation], metric: str = "success_rate",
              **kwargs) -> dict[str, float]:
    """Aggregate one metric across runs (typically the seeds of one cell).

    Returns mean, std (population, matching what the reports quote), and n.
    Std over a handful of seeds is a rough dispersion estimate, not a
    confidence interval; treat it as such.
    """
    if metric not in METRICS:
        raise KeyError(f"unknown metric {metric!r}; known: {sorted(METRICS)}")
    if not runs:
        return {"mean": float("nan"), "std": float("nan"), "n": 0}
    values = [METRICS[metric](r, **kwargs) for r in runs]
    # `values` is carried out with the summary so callers can run a test
    # across seeds. Recomputing it outside would mean re-deriving the metric
    # from the runs, and the two copies could then disagree.
    return {"mean": float(np.mean(values)), "std": float(np.std(values)),
            "n": len(values), "values": [float(v) for v in values]}
