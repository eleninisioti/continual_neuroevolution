"""Behavioral divergence between consecutive-task policies.

Implements the metric reported in the paper:

    BD_tau = (1 / |D_tau|) * sum_{s in D_tau} D(s)

where D_tau is the *state visitation distribution of the previous policy on the
previous task*, estimated by Monte Carlo rollouts, and D(s) is a per-state
distance between pi_tau and pi_{tau-1}.

Design decisions (see docstrings for rationale):

  1. D_tau is anchored on (pi_{tau-1}, task tau-1). Divergence is meant to be
     forgetting-relevant: it asks whether the new policy still behaves the same
     on the situations the old policy actually encountered.

     This is a choice, not a derivation -- the literature anchors three
     different ways (old-task visitation in Li & Hoiem 2017 / Kirkpatrick 2017,
     new-task visitation in RL's Razor, a state-space-covering grid in
     Anand & Precup 2023). The compute scripts therefore expose all feasible
     anchors so that the choice can be shown not to matter; see
     scripts/postprocess/compare_bd_anchors.py.

  2. D_tau is subsampled to a fixed size so that methods whose episodes
     terminate early do not contribute fewer (or differently distributed)
     states than methods that survive the full episode.

  3. States are stored as *raw, unnormalized* observations. Each policy applies
     its own preprocessing (e.g. a PPO run's saved observation normalizer).
     Sharing normalized states across policies would feed each policy inputs
     that were normalized by a different policy's statistics.

  4. The primary metric is scale-free and defined identically for
     deterministic (neuroevolution) and stochastic (RL) policies:
       - discrete actions  -> action disagreement rate
       - continuous actions -> normalized action distance
     Distributional metrics (KL / JS) are reported as secondary values and only
     where the policy is genuinely stochastic. Softmaxing the logits of a
     deterministic evolved network at temperature 1 makes the resulting KL scale
     with the magnitude of the evolved weights, which is unconstrained and
     differs systematically between evolved and gradient-trained networks, so it
     is not comparable across methods.
"""

import numpy as np


# ============================================================================
# State collection
# ============================================================================

def collect_states(
    reset_fn,
    step_fn,
    act_fn,
    num_episodes,
    max_steps,
    seed=0,
    max_states=None,
):
    """Roll out a policy and return every raw observation it visits.

    The environment interface is passed in as closures so that this works
    unchanged for gymnax, Kinetix and MuJoCo playground:

        reset_fn(episode_index) -> obs
        step_fn(obs_and_env_state, action) -> (obs, env_state, done)
        act_fn(obs) -> action

    Rather than a generic env wrapper, callers build these three closures for
    their own stack; see the per-benchmark scripts in scripts/postprocess/.

    Args:
        reset_fn: callable(episode_index) -> (obs, env_state)
        step_fn: callable(env_state, action, step_index) -> (obs, env_state, done)
        act_fn: callable(obs) -> action, the deterministic executed action
        num_episodes: number of rollout episodes
        max_steps: step cap per episode
        seed: unused here, kept so callers can document their seeding
        max_states: hard cap on collected states (None = no cap)

    Returns:
        np.ndarray of shape (N, obs_dim), raw unnormalized observations.
    """
    states = []

    for episode in range(num_episodes):
        obs, env_state = reset_fn(episode)
        states.append(np.asarray(obs).flatten())

        for step in range(max_steps):
            action = act_fn(obs)
            obs, env_state, done = step_fn(env_state, action, step)
            states.append(np.asarray(obs).flatten())

            if bool(done):
                break
            if max_states is not None and len(states) >= max_states:
                break

        if max_states is not None and len(states) >= max_states:
            break

    return np.array(states)


def subsample_states(states, num_states, seed=0):
    """Uniformly subsample a state set to a fixed size, without replacement.

    Two reasons this matters. First, episodes are strongly autocorrelated in
    time, so 1000 consecutive MuJoCo states carry far less information than
    their count suggests. Second, and more importantly for cross-method
    comparison, a policy that falls over after 40 steps otherwise contributes
    40 states while a policy that survives 1000 contributes 1000 -- the two
    BD estimates would then be averages over differently sized and differently
    distributed sets.

    If there are fewer states than requested, all of them are returned (and the
    caller should report the shortfall).
    """
    if len(states) <= num_states:
        return states

    rng = np.random.default_rng(seed)
    idx = rng.choice(len(states), size=num_states, replace=False)
    idx.sort()  # keep temporal order, purely for readability when inspecting
    return states[idx]


def sample_grid_states(low, high, num_states, seed=0, mode="auto"):
    """Sample a state-space-*covering* set, as opposed to a visitation-weighted one.

    This is the convention used by prediction-focused continual RL work, e.g.
    Anand & Precup (2023) evaluate on "225 evenly spaced points in the grid".
    It is only feasible in low dimension -- an evenly spaced mesh needs
    points_per_dim^d states -- so we use it as a robustness check on the
    low-dimensional classic control tasks and fall back to uniform sampling
    above three dimensions.

    Args:
        low, high: (obs_dim,) box bounds. Must be finite; callers that only have
            an unbounded observation space should pass empirical bounds.
        num_states: target number of states. A mesh may return slightly more or
            fewer, since points_per_dim is rounded.
        seed: seed for the uniform mode
        mode: "mesh" (evenly spaced), "uniform" (random), or "auto" (mesh when
            obs_dim <= 3, uniform otherwise)

    Returns:
        np.ndarray of shape (N, obs_dim)
    """
    low = np.asarray(low, dtype=np.float64)
    high = np.asarray(high, dtype=np.float64)

    if not (np.all(np.isfinite(low)) and np.all(np.isfinite(high))):
        raise ValueError("sample_grid_states requires finite bounds")

    dim = len(low)
    if mode == "auto":
        mode = "mesh" if dim <= 3 else "uniform"

    if mode == "mesh":
        points_per_dim = max(2, int(round(num_states ** (1.0 / dim))))
        axes = [np.linspace(low[i], high[i], points_per_dim) for i in range(dim)]
        mesh = np.meshgrid(*axes, indexing="ij")
        return np.stack([m.ravel() for m in mesh], axis=-1)

    rng = np.random.default_rng(seed)
    return rng.uniform(low, high, size=(num_states, dim))


def empirical_bounds(states, margin=0.0):
    """Per-dimension min/max of a state set, optionally widened by a margin.

    Used when the environment declares an unbounded observation space (gymnax
    reports infinite bounds for several velocity dimensions), so that
    sample_grid_states still has a finite box to cover. The box is then the
    region the policy actually operates in, which is the honest support for a
    covering set anyway.
    """
    states = np.asarray(states, dtype=np.float64)
    low = states.min(axis=0)
    high = states.max(axis=0)

    if margin:
        span = high - low
        low = low - margin * span
        high = high + margin * span

    return low, high


def make_pairs(num_checkpoints, reference="consecutive"):
    """Which (reference, current) checkpoint pairs to compare.

    "consecutive" gives (tau-1, tau) -- how much behavior changed at each task
    switch. "first" gives (1, tau) -- cumulative drift away from the policy at
    the end of the first task. The two answer different questions: per-step
    divergence can be small at every switch while total drift grows large, and
    it is the cumulative quantity that the LLM fine-tuning literature relates to
    forgetting (both RL's Razor and ES-at-Scale measure against a fixed base
    model rather than against the previous checkpoint).

    Returns:
        list of (reference_index, current_index) into the checkpoint list.
    """
    if reference == "consecutive":
        return [(i, i + 1) for i in range(num_checkpoints - 1)]
    if reference == "first":
        return [(0, j) for j in range(1, num_checkpoints)]
    raise ValueError(f"Unknown reference: {reference}")


def pool_states(state_sets, num_states, seed=0):
    """Concatenate per-method state sets and subsample to a shared set.

    Used to build a single D_tau shared by every method under comparison. If
    each method's BD is measured on its own visited states, a lower BD can mean
    either "this policy changed less" or "this policy visits a narrower region
    of state space" -- the metric confounds behavioral change with state
    coverage. Evaluating all methods on one pooled set removes that confound.
    """
    if not state_sets:
        raise ValueError("pool_states requires at least one state set")

    pooled = np.concatenate([s for s in state_sets if len(s) > 0], axis=0)
    return subsample_states(pooled, num_states, seed=seed)


# ============================================================================
# Per-state distances: discrete actions
# ============================================================================

def disagreement_rate(logits_prev, logits_curr):
    """Fraction of states where the greedy action differs.

    This is the primary discrete-action metric. It is bounded in [0, 1],
    free of any temperature choice, and identical in meaning for a
    deterministic evolved policy and for an RL policy evaluated greedily, which
    is what makes NE and RL numbers comparable.

    Args:
        logits_prev: (N, num_actions) logits of pi_{tau-1}
        logits_curr: (N, num_actions) logits of pi_tau

    Returns:
        float in [0, 1]
    """
    a_prev = np.argmax(np.asarray(logits_prev), axis=-1)
    a_curr = np.argmax(np.asarray(logits_curr), axis=-1)
    return float(np.mean(a_prev != a_curr))


def _softmax(logits):
    logits = np.asarray(logits, dtype=np.float64)
    logits = logits - np.max(logits, axis=-1, keepdims=True)
    exp = np.exp(logits)
    return exp / np.sum(exp, axis=-1, keepdims=True)


def mean_kl_categorical(logits_prev, logits_curr, eps=1e-8):
    """Mean KL(pi_{tau-1} || pi_tau) over states, for genuinely stochastic policies.

    Only meaningful when the policy really is a categorical distribution (e.g.
    a PPO actor). Reported as a secondary metric; see the module docstring for
    why this should not be used to compare evolved against gradient-trained
    networks.
    """
    p = np.clip(_softmax(logits_prev), eps, 1.0)
    q = np.clip(_softmax(logits_curr), eps, 1.0)
    p = p / p.sum(axis=-1, keepdims=True)
    q = q / q.sum(axis=-1, keepdims=True)
    kl = np.sum(p * np.log(p / q), axis=-1)
    return float(np.mean(kl))


def mean_js_categorical(logits_prev, logits_curr, eps=1e-8):
    """Mean Jensen-Shannon divergence over states, in nats.

    Preferred over KL whenever a distributional number is wanted: symmetric,
    bounded by log 2, and finite when the supports differ. Unbounded KL lets a
    single pathological state dominate BD_tau, and a single pathological task
    dominate the mean over tau.
    """
    p = np.clip(_softmax(logits_prev), eps, 1.0)
    q = np.clip(_softmax(logits_curr), eps, 1.0)
    p = p / p.sum(axis=-1, keepdims=True)
    q = q / q.sum(axis=-1, keepdims=True)
    m = 0.5 * (p + q)

    kl_pm = np.sum(p * np.log(p / m), axis=-1)
    kl_qm = np.sum(q * np.log(q / m), axis=-1)
    return float(np.mean(0.5 * (kl_pm + kl_qm)))


# ============================================================================
# Per-state distances: continuous actions
# ============================================================================

def normalized_action_distance(actions_prev, actions_curr, action_range=2.0):
    """Mean L2 distance between executed actions, normalized to be scale-free.

        (1/|D|) * sum_s ||a_tau(s) - a_{tau-1}(s)||_2 / (sqrt(d) * action_range)

    The sqrt(d) removes the dependence on action dimensionality and the
    action_range removes the dependence on the action units, so a HalfCheetah
    number and a Quadruped number mean the same thing. With the default range of
    2.0 (actions in [-1, 1]) the result is bounded in [0, 1]: 0 means identical
    behavior, 1 means every actuator flipped from one extreme to the other.

    Actions must be the *executed* actions, i.e. post-tanh for a Brax policy.

    Args:
        actions_prev: (N, action_dim) actions of pi_{tau-1}
        actions_curr: (N, action_dim) actions of pi_tau
        action_range: width of the action interval (2.0 for [-1, 1])

    Returns:
        float
    """
    a_prev = np.asarray(actions_prev, dtype=np.float64)
    a_curr = np.asarray(actions_curr, dtype=np.float64)

    dist = np.linalg.norm(a_curr - a_prev, axis=-1)
    dim = a_prev.shape[-1]
    return float(np.mean(dist) / (np.sqrt(dim) * action_range))


def mean_kl_diag_gaussian(mean_prev, log_std_prev, mean_curr, log_std_curr):
    """Mean KL(pi_{tau-1} || pi_tau) for diagonal Gaussian policies.

    Closed form, summed over action dimensions and averaged over states:

        KL = sum_i [ log(s2_i/s1_i) + (s1_i^2 + (m1_i - m2_i)^2) / (2 s2_i^2) - 1/2 ]

    Applies unchanged to Brax's NormalTanhDistribution. tanh is a fixed
    diffeomorphism applied identically by both policies, so the change-of-
    variables Jacobians cancel and the KL between the two squashed distributions
    equals the KL between the underlying Normals. The pre-tanh (mean, log_std)
    straight out of the network is therefore the correct input here -- no
    correction term is needed.

    RL-only: evolved policies are deterministic and have no log_std.
    """
    m1 = np.asarray(mean_prev, dtype=np.float64)
    m2 = np.asarray(mean_curr, dtype=np.float64)
    ls1 = np.asarray(log_std_prev, dtype=np.float64)
    ls2 = np.asarray(log_std_curr, dtype=np.float64)

    var1 = np.exp(2.0 * ls1)
    var2 = np.exp(2.0 * ls2)

    per_dim = (ls2 - ls1) + (var1 + (m1 - m2) ** 2) / (2.0 * var2) - 0.5
    kl = np.sum(per_dim, axis=-1)
    return float(np.mean(kl))


# ============================================================================
# State-visitation shift (a separate quantity from BD, reported separately)
# ============================================================================
#
# BD asks how differently two policies *act* on a fixed set of states. It says
# nothing about whether the new policy still *goes* where the old one went, and
# a policy can keep BD low on D_tau while abandoning that region entirely. The
# two are complementary, so the visitation quantity is defined here but never
# folded into `bd`.
#
# The distance between two visitation distributions is estimated by the MMD
# under an RBF kernel, following AutoQD (Hedayatian & Nikolaidis,
# arXiv:2506.05634). For k(s,s') = exp(-||s-s'||^2 / 2 sigma^2), Rahimi & Recht
# (2007) give
#
#     z(s) = sqrt(2/D) cos(W^T s + b),  W ~ N(0, sigma^-2 I),  b ~ U[0, 2pi]
#
# with z(s)^T z(s') ~= k(s,s'), so embedding a visitation distribution by its
# mean feature makes ||phi_a - phi_b|| ~= MMD(d_a, d_b). Nothing is trained, and
# the feature map is drawn once per task from a fixed seed and reused unchanged,
# which is what makes the numbers comparable across methods and runs.
#
# KL is deliberately not used here: the states are continuous, so a KL estimate
# would need a density model per policy, and it is unbounded and infinite the
# moment one policy visits a region the other never does -- which is exactly the
# case this metric exists to measure. MMD is bounded, needs no density estimate
# and stays finite under disjoint support.


def rff_feature_space(states, seed=0, rff_dim=512, max_median_samples=4000):
    """Fit the random-Fourier feature map used to embed visitation distributions.

    `states` should pool every policy the comparison will cover, so that the
    standardisation and the kernel bandwidth are not fitted to one of them.

    Returns a dict of plain numpy arrays; `embed_visitation` consumes it, and
    callers running under jax can lift `W` and `b` onto the device themselves.
    """
    rng = np.random.default_rng(seed)
    states = np.asarray(states, dtype=np.float64).reshape(-1, np.shape(states)[-1])

    mean = states.mean(axis=0)
    std = states.std(axis=0)
    std[std < 1e-8] = 1.0
    z = (states - mean) / std

    # Median heuristic for sigma, on a subsample to bound the pairwise cost.
    idx = rng.choice(z.shape[0], size=min(max_median_samples, z.shape[0]),
                     replace=False)
    sub = z[idx]
    d2 = np.sum((sub[:, None, :] - sub[None, :, :]) ** 2, axis=-1)
    median_sq = np.median(d2[np.triu_indices(sub.shape[0], k=1)])
    sigma = float(np.sqrt(median_sq / 2.0)) if median_sq > 0 else 1.0
    if sigma < 1e-8:
        sigma = 1.0

    dim = states.shape[1]
    return {
        "mean": mean,
        "std": std,
        "sigma": sigma,
        "W": rng.normal(0.0, 1.0 / sigma, size=(dim, rff_dim)),
        "b": rng.uniform(0.0, 2.0 * np.pi, size=rff_dim),
        "rff_dim": rff_dim,
    }


def rff_features(states, space):
    """Per-state features z(s); the mean over a rollout embeds its visitation."""
    z = (np.asarray(states, dtype=np.float64) - space["mean"]) / space["std"]
    return np.sqrt(2.0 / space["rff_dim"]) * np.cos(z @ space["W"] + space["b"])


def embed_visitation(states, space):
    """Embed one policy's visited states as a single vector (its mean feature)."""
    return rff_features(states, space).mean(axis=0)


def visitation_shift(embedding_a, embedding_b):
    """MMD between two visitation distributions, from their mean features."""
    return float(np.linalg.norm(np.asarray(embedding_a) - np.asarray(embedding_b)))


# ============================================================================
# Top-level driver
# ============================================================================

def behavioral_divergence_discrete(
    logits_prev,
    logits_curr,
    stochastic=False,
):
    """BD_tau for a discrete action space, evaluated on a shared state set.

    Args:
        logits_prev: (N, num_actions) logits of pi_{tau-1} on D_tau
        logits_curr: (N, num_actions) logits of pi_tau on D_tau
        stochastic: whether the policies are genuinely categorical (RL). If
            False, the distributional metrics are omitted rather than computed
            from a temperature-1 softmax of deterministic logits.

    Returns:
        dict with 'bd' (the primary metric) plus secondary values.
    """
    results = {
        "bd": disagreement_rate(logits_prev, logits_curr),
        "metric": "action_disagreement_rate",
        "num_states": int(len(logits_prev)),
    }

    if stochastic:
        results["kl"] = mean_kl_categorical(logits_prev, logits_curr)
        results["js"] = mean_js_categorical(logits_prev, logits_curr)

    return results


def behavioral_divergence_continuous(
    actions_prev,
    actions_curr,
    action_range=2.0,
    gaussian_prev=None,
    gaussian_curr=None,
):
    """BD_tau for a continuous action space, evaluated on a shared state set.

    Args:
        actions_prev: (N, action_dim) executed (post-tanh) actions of pi_{tau-1}
        actions_curr: (N, action_dim) executed actions of pi_tau
        action_range: width of the action interval (2.0 for [-1, 1])
        gaussian_prev: optional (mean, log_std) arrays for pi_{tau-1}, pre-tanh
        gaussian_curr: optional (mean, log_std) arrays for pi_tau, pre-tanh

    Returns:
        dict with 'bd' (the primary metric) plus secondary values.
    """
    results = {
        "bd": normalized_action_distance(actions_prev, actions_curr, action_range),
        "metric": "normalized_action_distance",
        "num_states": int(len(actions_prev)),
    }

    if gaussian_prev is not None and gaussian_curr is not None:
        mean_prev, log_std_prev = gaussian_prev
        mean_curr, log_std_curr = gaussian_curr
        results["kl"] = mean_kl_diag_gaussian(
            mean_prev, log_std_prev, mean_curr, log_std_curr
        )

    return results


def aggregate_over_tasks(per_task_bd):
    """Mean BD over tasks tau = 2..T.

    Starts at tau = 2 because there is no previous policy for the first task;
    the caller passes the already-paired per-task values, so this is just the
    mean, kept as a named function so the paper's aggregation lives in one place.
    """
    values = np.asarray([v for v in per_task_bd if v is not None], dtype=np.float64)
    if len(values) == 0:
        return float("nan")
    return float(np.mean(values))
