"""Empirical NTK rank -- C-CHAIN's own plasticity indicator, for every method.

Tang et al. build their whole argument on this quantity, not on dormant neurons:

    the loss of plasticity is accompanied by the exacerbation of churn due to
    the gradual RANK DECREASE of the Neural Tangent Kernel matrix

(abstract; the mechanism is §4.1). The chain they propose is

    non-stationarity -> NTK rank collapses -> gradients become correlated
                     -> churn is exacerbated -> learning destabilises

so churn is the SYMPTOM and rank collapse is the CAUSE. Logging churn without
rank measures the end of that chain and not the thing it is attributed to,
which is why this module exists alongside `plasticity.py`.

## What is computed

For a network f_theta and probe states x_1..x_n, the paper's Eq. 2 is

    N_theta(i, j) = grad_theta f_theta(x_i)^T grad_theta f_theta(x_j)
    N_theta       = G_theta^T G_theta

`G_theta` is the matrix of per-sample parameter gradients. Networks here have
vector outputs (action logits, or a mean-action vector) while the paper writes
f_theta(x) in R, so the scalar summary is the MEAN over output dimensions --
stated rather than assumed, because summing instead would weight a 16-action
policy differently from an 8-actuator one and the columns must be comparable
across the suites' different action spaces.

## Which rank

Three numbers, because "the rank" of a numerically low-rank matrix is a choice
and the plasticity literature has not settled on one:

  ntk_effective_rank  exp(entropy of the normalised eigenvalue spectrum). Smooth,
                      needs no threshold, and moves before a hard rank does --
                      the one to plot.
  ntk_srank           smallest k whose top-k eigenvalues carry >= 99% of the
                      total (Kumar et al.'s srank_delta, delta = 0.01). The
                      thresholded version, reported because it is what several
                      plasticity papers tabulate.
  ntk_trace           sum of eigenvalues. NOT a rank -- it is the total NTK
                      energy, and it is here because rank and trace move
                      independently: a spectrum can collapse in rank while
                      growing in trace. Reading a rank drop without it invites
                      the conclusion that the network stopped responding, when
                      it may have concentrated its response instead.

`ntk_rank_ratio` is effective_rank / n, so a run can be read without knowing how
many probe states were used.

## Cost, and why it is capped

`G_theta` is (n, |theta|) and the Gram matrix is (n, n). At n = 128 and the
ant's ~21k parameters that is ~11 MB and a 128x128 eigendecomposition -- cheap,
but it is a full jacobian, so it is NOT free per generation the way a weight
norm is. `max_samples` caps n; the default 128 places the effective rank to
well inside the between-generation variation.

ONE NETWORK PER RECORD. The NE side measures the elite (the policy you would
deploy), not the population -- a jacobian per individual across 512 genomes
would dominate the run. That asymmetry with `population_dormancy`, which does
sample the population, is deliberate and is about cost, not about meaning.

## Observer, like everything else here

No RNG is consumed: the probe batch is the frozen one `plasticity.py` already
holds, and the subsample is a deterministic head slice of it. A run with these
columns and one without are the same run.
"""

import jax
import jax.numpy as jnp
import numpy as np

from source.metrics.probe import probe_expand, probe_size, probe_take


def _scalar_summary(apply_fn, params, obs):
    """Mean over output dimensions -- the paper's scalar f_theta(x)."""
    return jnp.mean(apply_fn(params, obs))


def empirical_ntk_gram(apply_fn, params, probe_obs, max_samples=128):
    """The (n, n) Gram matrix N = G G^T of per-sample parameter gradients.

    `apply_fn(params, obs_batch) -> outputs`, the same callable the churn
    measures use, so the two diagnostics describe the same network.
    """
    n = int(min(max_samples, probe_size(probe_obs)))
    obs = probe_take(probe_obs, n)

    def per_sample_grad(x):
        # x is one state; keep the batch axis the network expects.
        g = jax.grad(lambda p: _scalar_summary(apply_fn, p, probe_expand(x)))(params)
        leaves = jax.tree_util.tree_leaves(g)
        return jnp.concatenate([jnp.ravel(l) for l in leaves])

    grads = jax.vmap(per_sample_grad)(obs)          # (n, |theta|)
    return grads @ grads.T                          # (n, n)


def _empty(prefix, n):
    """All-None stats: the shape of a measurement that could not be taken."""
    return {f'{prefix}_effective_rank': None, f'{prefix}_srank': None,
            f'{prefix}_trace': None, f'{prefix}_rank_ratio': None,
            f'{prefix}_num_probe': n}


def rank_stats(gram, prefix='ntk'):
    """Effective rank, srank and trace of a PSD Gram matrix.

    NOTHING HERE MAY RAISE. An observer that kills the run it is observing is
    worse than no observer, and this one did exactly that: `eigvalsh` threw
    `LinAlgError: Eigenvalues did not converge` and took down ALL 30 gymnax
    continual DNS trials and 8 TRAC trials before it was caught -- runs that
    were otherwise healthy, lost to a diagnostic. DNS is the one method whose
    genotypes can diverge (see the line_sigma note in ne/dns.py), so its NTK
    carries the huge and non-finite entries LAPACK will not factor.

    A failed measurement is therefore reported as None -- the same "no reading"
    the churn columns use at generation 0 -- and never as an exception.
    """
    g = np.asarray(gram, dtype=np.float64)
    n = g.shape[0]
    # Symmetrise before eigvalsh: G G^T is symmetric in exact arithmetic and
    # slightly asymmetric in float32, which can produce small negative or
    # complex-looking eigenvalues.
    g = 0.5 * (g + g.T)
    if not np.all(np.isfinite(g)):
        # A diverged genome gives inf/nan gradients. Report no reading rather
        # than feeding LAPACK something it will refuse.
        return _empty(prefix, n)
    try:
        eig = np.linalg.eigvalsh(g)
    except np.linalg.LinAlgError:
        return _empty(prefix, n)
    eig = np.clip(eig, 0.0, None)[::-1]             # descending, non-negative
    total = float(eig.sum())
    if total <= 0.0:
        # A dead network: every gradient is zero, so the NTK is the zero matrix.
        # Rank 0 is the honest answer, not a division by zero.
        return {f'{prefix}_effective_rank': 0.0, f'{prefix}_srank': 0.0,
                f'{prefix}_trace': 0.0, f'{prefix}_rank_ratio': 0.0,
                f'{prefix}_num_probe': n}

    p = eig / total
    nz = p[p > 0]
    effective_rank = float(np.exp(-np.sum(nz * np.log(nz))))
    srank = int(np.searchsorted(np.cumsum(p), 0.99) + 1)
    return {
        f'{prefix}_effective_rank': effective_rank,
        f'{prefix}_srank': float(srank),
        f'{prefix}_trace': total,
        f'{prefix}_rank_ratio': effective_rank / max(n, 1),
        f'{prefix}_num_probe': n,
    }


def ntk_rank_stats(apply_fn, params, probe_obs, max_samples=128, prefix='ntk'):
    """`rank_stats` of the empirical NTK of one network. The entry point.

    Total containment: any failure in the jacobian, the Gram product or the
    eigendecomposition yields a None reading, never an exception. See
    `rank_stats` for what this cost before it was contained.
    """
    if probe_obs is None:
        return {}
    n = int(min(max_samples, probe_size(probe_obs)))
    try:
        gram = empirical_ntk_gram(apply_fn, params, probe_obs,
                                  max_samples=max_samples)
    except Exception:
        # A diverged genome can make the forward pass itself fail, and the
        # search must not care. Deliberately broad: there is no failure here
        # worth propagating into a training run.
        return _empty(prefix, n)
    return rank_stats(gram, prefix=prefix)
