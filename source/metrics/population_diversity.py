"""Population diversity: three numbers, one definition, every runner.

Every population method in this repo -- the NE arms and PBT -- logs the same
three columns per record, so a diversity figure reads one vocabulary across
bodies and across families:

    bd_genomic_diversity      mean pairwise L2 distance between the members'
                              flat parameter vectors (a subsample of at most
                              `max_n`, evenly spaced, the convention
                              train_nes has used since the generalists study)
    bd_fitness_std            standard deviation of the members' training
                              fitness this generation / update window
    bd_behavioural_diversity  mean pairwise distance between the members'
                              ACTIONS on one frozen batch of probe states:
                              the argmax disagreement rate for a categorical
                              or multi-discrete policy (in [0, 1]), the mean
                              per-actuator |difference| of the mean action
                              for a continuous one (in [0, 2] for tanh
                              actions). Same statistic the churn column
                              uses between two versions of ONE policy
                              (`source/metrics/plasticity.py`), here between
                              every pair of members.

The column names are the ones `scripts/make_plasticity_figure.py` already
resolves for its `genomic_diversity` and `fitness_std` rows (the gymnax
trainers' behaviour tracker wrote them), so the diversity figure needs no
new vocabulary.

Host-side numpy for the first two (a population is at most a few hundred
short vectors here); the behavioural statistic is one jitted call per record.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

BEHAVIOUR_KINDS = ('categorical', 'multi_discrete', 'gaussian')


def _evenly(n, max_n):
    """Indices of an evenly spaced subsample; all of them when n <= max_n."""
    if n <= max_n:
        return np.arange(n)
    return np.linspace(0, n - 1, max_n).astype(int)


def genomic_diversity(members, max_n=32):
    """Mean pairwise L2 distance over (a subsample of) the flat genomes."""
    members = np.asarray(members, dtype=np.float64)
    members = members.reshape(len(members), -1)
    if len(members) < 2:
        return 0.0
    sub = members[_evenly(len(members), max_n)]
    # Through the Gram matrix, never the (n, n, P) difference tensor: on the
    # Kinetix pixel policy (P = 1.13M) that tensor is 9 GB of float64 at
    # n = 32, every generation. |a - b|^2 = |a|^2 + |b|^2 - 2 a.b.
    sq = np.einsum('ij,ij->i', sub, sub)
    d2 = np.maximum(sq[:, None] + sq[None, :] - 2.0 * (sub @ sub.T), 0.0)
    iu = np.triu_indices(len(sub), k=1)
    return float(np.sqrt(d2[iu]).mean())


def fitness_std(fitnesses):
    """Standard deviation of the finite entries; 0 with fewer than two."""
    f = np.asarray(fitnesses, dtype=np.float64).ravel()
    f = f[np.isfinite(f)]
    return float(f.std()) if f.size > 1 else 0.0


def make_pairwise_behaviour_fn(kind, action_dims=None):
    """``distance(outputs) -> scalar`` for a stack of policy outputs.

    ``outputs`` is ``(n_members, n_probe, out_dim)``: categorical or padded
    multi-discrete logits, or the mean action of a continuous policy (the
    caller strips a Gaussian head's log-std half before calling). ``kind`` is
    the action head's name (`actors.head_for(...).name`); ``action_dims`` is
    the per-dimension choice count a multi-discrete head is built from
    (`suite.action_dims(env)`), e.g. ``(3, 3, 3, 3, 2, 2)`` on Kinetix.
    """
    if kind not in BEHAVIOUR_KINDS:
        raise ValueError(f'kind must be one of {BEHAVIOUR_KINDS}, got {kind!r}')
    if kind == 'multi_discrete':
        if not action_dims:
            raise ValueError('multi_discrete behaviour needs action_dims')
        dims = tuple(int(d) for d in action_dims)
        offsets = tuple(int(o) for o in np.cumsum((0,) + dims[:-1]))

    def actions_of(out):
        if kind == 'categorical':
            return jnp.argmax(out, axis=-1)[..., None]              # (n, B, 1)
        if kind == 'multi_discrete':
            return jnp.stack([jnp.argmax(out[..., o:o + d], axis=-1)
                              for o, d in zip(offsets, dims)], axis=-1)   # (n, B, D)
        return out                                                    # (n, B, nu)

    @jax.jit
    def distance(outputs):
        acts = actions_of(outputs)
        n = acts.shape[0]
        if kind == 'gaussian':
            diff = jnp.abs(acts[:, None] - acts[None, :])              # (n, n, B, nu)
        else:
            diff = (acts[:, None] != acts[None, :]).astype(jnp.float32)
        per_pair = diff.mean(axis=(-1, -2))                             # (n, n)
        mask = jnp.triu(jnp.ones((n, n)), k=1)
        return jnp.sum(per_pair * mask) / jnp.maximum(jnp.sum(mask), 1.0)

    return distance


def diversity_columns(flat_members, fitnesses, outputs=None, distance=None,
                      max_n=32):
    """The three columns. ``outputs`` (see `make_pairwise_behaviour_fn`) and
    ``distance`` may be None, in which case the behavioural column is left
    out rather than written as a placeholder -- a body whose probe states
    cannot be rebuilt (Kinetix's NE trace keeps step features, not frames)
    has no such number, and the figure should say so."""
    out = {'bd_genomic_diversity': genomic_diversity(flat_members, max_n),
           'bd_fitness_std': fitness_std(fitnesses)}
    if outputs is not None and distance is not None and outputs.shape[0] >= 2:
        out['bd_behavioural_diversity'] = float(distance(jnp.asarray(outputs)))
    return out


def probe_subsample(members, max_n=32):
    """The evenly spaced subsample used for the behavioural statistic, so a
    caller applies its policy to `max_n` genomes rather than the whole
    population."""
    members = np.asarray(members)
    return members[_evenly(len(members), max_n)]
