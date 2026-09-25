"""Zero-shot transfer across a sub-task switch, for the population methods.

The problem this exists to fix
------------------------------
Every continual NE trainer in this repo logs its first record of a sub-task
AFTER that sub-task's first generation: ask -> evaluate the offspring on the NEW
env -> tell/select, then log. And the value it logs is `max` over the whole
population. Read as "what the learner had when the task changed" -- which is how
it was read, in scripts/neurips_2026_rebuttal and in the ant write-ups -- that
number is inflated two ways at once:

  * it is a best-of-N order statistic over 512 individuals, while the RL
    trainers log a single policy's evaluation return, and
  * it is measured after a full generation of search on the new task: 512
    offspring rolled out on the new env, i.e. pop_size x num_evals x
    episode_length env steps (1.5M at the ant settings, 3.1M for DNS, which also
    re-measures the carried population at the boundary). The RL trainers
    evaluate BEFORE the sub-task's first training epoch, at 0 steps.

On the damage-only ant block that made GA+Novelty look like it retained 0.53 of
the new sub-task's performance across a leg switch against PPO's 0.06. Measured
properly -- the carried best individual, unchanged, on the new leg -- it retains
0.15 against PPO's 0.04, and the effect it was being credited with is one-
generation recovery, not zero-shot transfer.

What the trainers log now
-------------------------
At every sub-task boundary after the first, the single policy the run carries
across the switch is evaluated on the new sub-task before any variation,
selection or gradient step, and the result is attached to the FIRST record of
that sub-task under `zero_shot_carried_best`. What "the carried policy" means is
per-method and is the method's own answer to "if the run had to hand over one
network at the switch, which one":

  * GA, GA+Novelty (DNS): the best individual of the previous sub-task's last
    generation -- the genome the checkpoint stores.
  * ES: the distribution mean, which is the ES solution and what its checkpoints
    store; the population is noise around it and is not carried as such.

Cost is one individual x num_evals episodes per switch, so it does not move the
env-step budget the NE and RL curves are matched on -- which is also why it
evaluates the carried best rather than the whole carried population.

Reading it
----------
`zero_shot_carried_best` is absent on sub-task 0 (there is nothing to transfer
from) and on runs from before 2026-07-30. It is directly comparable to the RL
trainers' first evaluation of a sub-task, which is the same quantity for the
same reason, and is tagged with the same name there.

It is NOT comparable to `best_fitness` on the same record: that one is still the
post-first-generation max over the population, unchanged, because the training
curves are built from it.
"""

ZERO_SHOT_KEY = "zero_shot_carried_best"
ZERO_SHOT_SOURCE_KEY = "zero_shot_source_task"


def zero_shot_fitness(score_batch, flat_params):
    """Score one carried genome under `score_batch`, a batched scoring function.

    `score_batch` is the trainer's own scoring function with everything but the
    genome batch already bound, so the episode count, the env and the masking
    rules are exactly the ones the sub-task is about to be trained under -- the
    measurement cannot drift from the training protocol.

    Returns None for a missing genome (sub-task 0, or a run that has not
    produced one yet), so a call site can attach the result unconditionally.
    """
    if flat_params is None:
        return None
    import jax.numpy as jnp

    scores = score_batch(jnp.asarray(flat_params)[None, :])
    # Scoring functions in this repo return either a fitness array or a tuple
    # whose first element is one (DNS asks for descriptors, the tracked variants
    # for behaviours); a genome batch of one makes the rest uninteresting.
    if isinstance(scores, tuple):
        scores = scores[0]
    return float(scores[0])


def attach(record, value, source_task):
    """Attach a zero-shot measurement to `record`, in place, if there is one."""
    if value is not None:
        record[ZERO_SHOT_KEY] = value
        record[ZERO_SHOT_SOURCE_KEY] = source_task
    return record
