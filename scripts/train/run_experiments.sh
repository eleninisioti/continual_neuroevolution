#!/bin/bash
# ============================================================================
# NeurIPS 2026 rebuttal: single entry point for every paper experiment.
#
# All runs write under projects/neurips_2026_rebuttal/ using one fixed layout
# that make_figures.py knows how to read:
#
#   projects/neurips_2026_rebuttal/<suite>/<setting>/<method>/<env>/trial_<T>/
#       training_metrics.json   <- per-generation (NE) / per-update (RL) history
#       *_best.pkl              <- final checkpoint
#
# Usage:
#   ./run_experiments.sh gymnax_noncontinual ga             # one method
#   ./run_experiments.sh gymnax_noncontinual ga dns es ppo  # several
#   ./run_experiments.sh gymnax_noncontinual all            # everything in the block
#   ./run_experiments.sh gymnax_continual all               # the continual block
#
# To fill a whole machine, use the launcher instead of calling this directly:
#   ./run_gymnax_continual_6gpu.sh
#
# Environment overrides:
#   GPU=0            GPU index passed to the trainers
#   NUM_TRIALS=10    independent training trials per condition (paper uses 10)
#   TRIALS="1 2 3"   run only these trials (for sharding one condition
#                    across GPUs); defaults to 1..NUM_TRIALS
#   BASE_SEED=42     seed of trial 1; trial T uses BASE_SEED + T - 1
#   ENVS="..."       whitespace-separated env list
#   DRY_RUN=1        print the commands instead of running them
# ============================================================================

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Walk up to the directory holding pyproject.toml rather than counting
# `dirname`s: every time these scripts moved between `scripts/` and
# `scripts/outdated/` a hardcoded count went stale, and the failure was 60 jobs
# exiting 127 before anything trained.
REPO_ROOT="$SCRIPT_DIR"
while [ ! -f "$REPO_ROOT/pyproject.toml" ] && [ "$REPO_ROOT" != "/" ]; do
    REPO_ROOT="$(dirname "$REPO_ROOT")"
done
cd "$REPO_ROOT"

# Line-buffer the trainers' stdout. Every launcher redirects a condition's
# output to a file, and Python block-buffers stdout at 8 KB when it is not a
# tty -- so a trainer that prints one line per 10 generations shows nothing in
# its log for hours and only flushes when the process exits. A run that is
# compiling, a run that is training and a run that has hung all look identical
# until then, which is exactly when you want to be able to tell them apart.
export PYTHONUNBUFFERED=1

# Where runs land. Overridable so a smoke test can write somewhere else;
# make_figures.py reads the same tree via --runs_root.
PROJECT_ROOT="${PROJECT_ROOT:-projects/neurips_2026_rebuttal/runs}"

GPU="${GPU:-0}"
NUM_TRIALS="${NUM_TRIALS:-10}"
BASE_SEED="${BASE_SEED:-42}"
WANDB_PROJECT="${WANDB_PROJECT:-neurips_2026_rebuttal}"
DRY_RUN="${DRY_RUN:-0}"

BLOCK="${1:-}"
shift || true
METHODS="$*"

if [ -z "$BLOCK" ] || [ -z "$METHODS" ]; then
    echo "Usage: $0 <block> <method> [method ...]"
    echo "  blocks : gymnax_noncontinual | gymnax_continual | gymnax_popsize"
    echo "           | mujoco_noncontinual | mujoco_continual"
    echo "           | mujoco_continual | brax_noncontinual | brax_continual"
    echo "  methods: ga | ga_isoline | dns | dns_gaussian | es | nes | ppo | trac | redo | cchain | pbt | pbt2 | all"
    echo "           (ga_isoline is the GA bred with DNS's Iso+LineDD operator"
    echo "            and nothing else changed, so dns-vs-ga_isoline isolates"
    echo "            the selection rule; ga_refresh/ga_reeval are gone -- the"
    echo "            GA always re-scores its archive now)"
    echo "           (nes is gymnax only: the same ES trainers at --algo nes)"
    echo "           (gymnax_popsize is continual NE only: ga | dns | es | nes | all)"
    echo "           (the mujoco blocks run cchain at continuous-control defaults)"
    echo "           (pbt is a population of the shared PPO over PPO's own budget:"
    echo "            gymnax blocks here; the other suites through their cli.py)"
    echo "           (brax_noncontinual is the same block on brax ant)"
    echo "           (brax_continual damages one ant leg per sub-task, cycling 1-2-3-4)"
    echo "           | ant_noncontinual | ant_continual | cheetah_noncontinual | cheetah_continual"
    echo "           | minigrid_noncontinual | minigrid_continual"
    echo "           (the ant_*/cheetah_* blocks are the ICLR mjx study --"
    echo "            source/studies/mjx/, the shared runners, centroid saved,"
    echo "            gaussian NE pair. They are NOT brax_*/mujoco_*, the old"
    echo "            per-method trainers; block_mujoco_* is dead -- its"
    echo "            trainers were deleted on 2026-09-08)"
    exit 1
fi

# Prefer the repo virtualenv so the script works without activating it first.
#
# Every block -- gymnax, brax and mujoco alike -- runs out of the single .venv
# built by scripts/setup_venv.sh. (It used to be two: the mujoco trainers need
# an mjx older than the one a plain `uv sync` installs, so they lived in a
# separate .venv-mujoco. That pin is now applied to .venv itself, and the older
# mjx runs the gymnax and brax suites unchanged.) PYTHON=... still overrides.
VENV_FOR_BLOCK="$REPO_ROOT/.venv"

if [ -z "${PYTHON:-}" ]; then
    if [ -x "$VENV_FOR_BLOCK/bin/python" ]; then
        PYTHON="$VENV_FOR_BLOCK/bin/python"
    elif command -v python3 >/dev/null 2>&1; then
        PYTHON="python3"
    else
        PYTHON="python"
    fi
fi

# Checked up front, because a missing interpreter otherwise fails once per
# trial and the sweep still reports itself complete after leaving a tree of
# empty output directories behind.
if ! command -v "$PYTHON" >/dev/null 2>&1; then
    echo "ERROR: interpreter '$PYTHON' not found." >&2
    echo "       Expected $VENV_FOR_BLOCK/bin/python, or set PYTHON=..." >&2
    exit 1
fi

# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------

# Turn "CartPole-v1" into "CartPole_v1" for use as a directory name.
#
# ENV_DIR_SUFFIX is appended after that, so the same env can be run at more than
# one setting without the two overwriting each other. The continual sweeps use
# it to tag the observation-noise sigma -- MountainCar is run at both 0.02 and
# 1.0, which are different experiments answering different questions and so
# cannot share a directory. Empty by default, which reproduces the old layout.
sanitize() { echo "${1//-/_}${ENV_DIR_SUFFIX:-}"; }

# Standard deviation of the per-sub-task observation-noise vector, per env.
#
# This is NOT one number across environments, because the observation
# components are not on one scale. MountainCar's velocity spans only +-0.07,
# so a sigma that is reasonable for CartPole swamps it: at sigma=1 the offset
# is ~15x the entire range of the dimension the task is about, and every
# method -- NE and RL alike -- floors at -500 on every sub-task after the
# unperturbed one.
#
# The values are Tang et al. (2025) appendix A.1, who hit the same problem:
# sigma=2.0 for CartPole and Acrobot (inherited from TRAC, Muppidi et al.
# 2024), and sigma=0.02 for MountainCar, which they added and for which they
# report 2.0 "is too large for this environment to be learnable".
#
# Applied identically to every method, so the sub-task sequence stays shared
# and the comparison stays fair -- this is a property of the benchmark, not of
# any algorithm.
#
# NOISE_RANGE overrides every env; the per-env vars override a single one.
noise_range_for() {
    case "$1" in
        MountainCar-v0) echo "${NOISE_RANGE:-${NOISE_RANGE_MOUNTAINCAR:-0.02}}" ;;
        CartPole-v1)    echo "${NOISE_RANGE:-${NOISE_RANGE_CARTPOLE:-2.0}}" ;;
        Acrobot-v1)     echo "${NOISE_RANGE:-${NOISE_RANGE_ACROBOT:-2.0}}" ;;
        *)              echo "${NOISE_RANGE:-2.0}" ;;
    esac
}

# Trials that exited non-zero. Reported at the end and turned into the script's
# exit status, so an unattended sweep cannot look like it succeeded.
FAILED_TRIALS=()

run() {
    echo "+ $*"
    if [ "$DRY_RUN" = "0" ]; then
        "$@"
    fi
}

# run_condition <suite> <setting> <method> <train_script> <env> [extra args...]
run_condition() {
    local suite="$1" setting="$2" method="$3" train_script="$4" env="$5"
    shift 5
    local env_dir
    env_dir="$(sanitize "$env")"

    # TRIALS lets one condition be split across GPUs: two invocations with
    # TRIALS="1 2 3 4 5" and TRIALS="6 7 8 9 10" run disjoint halves of the same
    # condition into the same output tree. Unset, it is every trial, which is
    # what every existing caller gets.
    for trial in ${TRIALS:-$(seq 1 "$NUM_TRIALS")}; do
        local seed=$((BASE_SEED + trial - 1))
        local out_dir="$PROJECT_ROOT/$suite/$setting/$method/$env_dir/trial_$trial"

        if [ -f "$out_dir/training_metrics.json" ]; then
            echo ">>> SKIP (already done) | $method | $env | trial $trial"
            continue
        fi

        echo ""
        echo ">>> $method | $env | trial $trial | seed $seed -> $out_dir"
        [ "$DRY_RUN" = "0" ] && mkdir -p "$out_dir"
        run "$PYTHON" "$train_script" \
            --env "$env" \
            --gpus "$GPU" \
            --trial "$trial" \
            --seed "$seed" \
            --output_dir "$out_dir" \
            --wandb_project "$WANDB_PROJECT" \
            "$@"
        local status=$?

        # Keep going on failure -- one bad trial should not cost the rest of
        # the sweep -- but remember it.
        if [ "$DRY_RUN" = "0" ] && [ "$status" -ne 0 ]; then
            echo ">>> FAILED (exit $status) | $method | $env | trial $trial"
            FAILED_TRIALS+=("$method/$env/trial_$trial (exit $status)")
        elif [ "$DRY_RUN" = "0" ] && [ ! -f "$out_dir/training_metrics.json" ]; then
            # Exited cleanly without writing the file every downstream script
            # keys off. Treated as a failure so it is not silently skipped as
            # "already done" on the next run either.
            echo ">>> FAILED (no training_metrics.json) | $method | $env | trial $trial"
            FAILED_TRIALS+=("$method/$env/trial_$trial (no training_metrics.json)")
        fi
    done
}

wants() {
    case " $METHODS " in
        *" all "*) return 0 ;;
        *" $1 "*)  return 0 ;;
        *)         return 1 ;;
    esac
}

# The PBT arms: `pbt` (N=8) and `pbt2` (N=2) are the paper's, mode `full`
# (exploit AND explore); `pbt_weights` / `pbt2_weights` are the same
# populations in mode `weights_only` (no hyperparameter perturbation), the
# ablation added 2026-09-19; `pbt_hp` / `pbt2_hp` are mode `hp_only` (explore
# without exploit, 2026-09-25). The `_weights` / `_hp` arms never run under `all`: they
# have to be named, so an existing `all` launch keeps producing exactly the
# arms it did. The name is decoded ONCE, in the runner's `pbt_arm`; the
# gymnax blocks below build their own command line, so they ask it.
PBT_ARMS="pbt pbt2 pbt_weights pbt2_weights pbt_hp pbt2_hp"
wants_pbt() {
    case "$1" in
        *_weights|*_hp) case " $METHODS " in *" $1 "*) return 0 ;; *) return 1 ;; esac ;;
        *)         wants "$1" ;;
    esac
}
pbt_settings() {   # sets pbt_n and pbt_mode for an arm name
    # On the CPU: importing train_ppo initialises JAX, and on a card that is
    # already full the import itself died with CUDA_ERROR_OUT_OF_MEMORY and
    # left pbt_n empty (DeepSea trial 6, 2026-09-20). A lookup table needs no
    # device; fail loudly rather than pass '' to --pbt_pop_size.
    read -r pbt_n pbt_mode <<< "$(JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= "$PYTHON" -c "
from source.studies.generalists.train_ppo import pbt_arm
_, n, mode = pbt_arm('$1'); print(n, mode)")"
    [ -n "$pbt_n" ] && [ -n "$pbt_mode" ] \
        || { echo "FATAL: pbt_settings could not resolve arm '$1'" >&2; return 1; }
}

# ----------------------------------------------------------------------------
# Block: gymnax_noncontinual  (non-continual control tasks)
# ----------------------------------------------------------------------------

block_gymnax_noncontinual() {
    local suite="gymnax" setting="noncontinual"
    local envs="${ENVS:-CartPole-v1 Acrobot-v1 MountainCar-v0}"

    # Every method gets the same budget on every env: GENERATIONS generations of
    # a population of NE_POP_SIZE evaluated for NE_EPISODE_LENGTH steps. PPO gets
    # the equivalent number of environment steps, which is how the paper matches
    # sample complexity between NE and RL.
    local generations="${GENERATIONS:-600}"
    local ne_pop_size="${NE_POP_SIZE:-512}"
    local ne_episode_length="${NE_EPISODE_LENGTH:-500}"

    # Rollouts per individual. Selection uses only the first rollout here, but
    # all NE_NUM_EVALS of them are executed (see make_scoring_fn: it vmaps over
    # pop_size x num_evals keys), so they are all paid for in environment steps.
    # The trainers default to 10 for GA but 1 for ES; pinning both makes the
    # plotted y-values comparable instead of a 10-rollout mean against a single
    # noisy rollout.
    # 3 IS THE PAPER VALUE, verified against every reported gymnax run's
    # train.log (GA, ES, DNS; stationary, continual and every population
    # size) and against the RL budgets matched to it: 512 x 3 x 500 x 600
    # = 4.608e8 env steps = the 4500 PPO updates those logs show. It used to
    # default to 10, which is a different experiment at 3.3x the cost whose
    # RL arm is no longer compute-matched to its NE arm.
    local ne_num_evals="${NE_NUM_EVALS:-3}"

    # PPO's matched budget. num_evals belongs in this product: a generation
    # costs pop x evals x episode_length environment steps, not pop x
    # episode_length. Leaving it out is what gave NE NE_NUM_EVALS times the
    # samples PPO got while the figures claimed matched sample complexity.
    local rl_timesteps=$((ne_pop_size * ne_num_evals * ne_episode_length * generations))

    # Pinned, not left to the trainer's default: the RL trainer's per-env
    # ENV_CONFIGS are not guaranteed to match the continual trainer's, and the
    # CartPole entry did not (20 against 50). Passing it explicitly means the
    # stationary reference and the continual runs share a batch size and GAE
    # horizon by construction rather than by two files agreeing.
    local rl_num_steps="${RL_NUM_STEPS:-50}"

    # Pinned for the same reason the continual block pins it: the NE trainers'
    # per-env defaults are not equal across methods. DNS and ES default to 32x32
    # on Acrobot and MountainCar while GA uses 16x16, which gave them ~3x the
    # parameters of GA and PPO on two of the three tasks -- a network-size
    # advantage read as a method difference. Same value as the continual block,
    # so the two settings are also comparable with each other.
    local ne_hidden_dims="${NE_HIDDEN_DIMS:-16 16}"

    # THE REPORTED number, which is not the number either family selects on.
    # NE: `centroid_fitness` and `elite_eval_fitness` are scored on fresh keys
    # over this many episodes (--report_episodes), because the search's own
    # columns are a max over the population of a `num_evals` mean whose FIRST
    # draw did the selecting -- optimistically biased, and by a different
    # amount per method (+5 for the GA, +53 for DNS on the sigma=1.0 tree), so
    # the bias does not cancel in a comparison.
    # RL: the same number, passed explicitly rather than left to the trainer's
    # default, so "10 episodes" is recorded in every run's config instead of
    # being a property of whichever trainer version ran.
    # NOT part of the budget product below: 2 genomes x 10 episodes a
    # generation against pop x evals = 1536 rollouts, and it never feeds
    # selection -- exactly as the RL evaluation has always sat outside
    # --num_timesteps.
    local report_episodes="${REPORT_EPISODES:-10}"

    # Behavioural diversity of the population, logged every DIVERSITY_INTERVAL
    # generations by every population-based method, plus BEHAVIOUR_SNAPSHOTS
    # saved populations per run for the shared-encoder analysis. Tracking is an
    # observer: it never feeds back into selection, and it draws from its own
    # random stream, so a run is identical with it on or off.
    local diversity_args=""
    if [ "${TRACK_DIVERSITY:-1}" = "1" ]; then
        diversity_args="--track_diversity 1 \
            --diversity_interval ${DIVERSITY_INTERVAL:-10} \
            --occupancy_bins ${OCCUPANCY_BINS:-12} \
            --behaviour_snapshots ${BEHAVIOUR_SNAPSHOTS:-8}"
    else
        diversity_args="--track_diversity 0"
    fi

    echo "=========================================="
    echo "Block: gymnax_noncontinual"
    echo "  envs       : $envs"
    echo "  methods    : $METHODS"
    echo "  trials     : $NUM_TRIALS (base seed $BASE_SEED)"
    echo "  generations: $generations (pop $ne_pop_size x $ne_num_evals evals x $ne_episode_length steps)"
    echo "  PPO budget : $rl_timesteps env steps (matched, num_evals included)"
    echo "  NE network : $ne_hidden_dims (pinned across methods)"
    echo "  diversity  : ${TRACK_DIVERSITY:-1} (every ${DIVERSITY_INTERVAL:-10} gens, NE methods only)"
    echo "  GPU        : $GPU"
    echo "=========================================="

    for env in $envs; do
        if wants ga; then
            run_condition "$suite" "$setting" ga \
                source/studies/gymnax/train_GA_gymnax.py "$env" \
                --num_generations "$generations" --pop_size "$ne_pop_size" \
                --num_evals "$ne_num_evals" --report_episodes "$report_episodes" \
                --hidden_dims $ne_hidden_dims \
                $diversity_args
        fi
        # The operator-matched GA's OWN stationary run. FT subtracts each
        # method's stationary reference from its continual curve, and
        # `ga_isoline` is a different search from `ga` -- reusing the `ga`
        # reference would charge the operator swap to forward transfer.
        # Same widths as the continual arm, for the same reason.
        if wants ga_isoline; then
            run_condition "$suite" "$setting" ga_isoline \
                source/studies/gymnax/train_GA_gymnax.py "$env" \
                --variation isoline \
                --iso_sigma "${GA_ISO_SIGMA:-0.05}" --line_sigma "${GA_LINE_SIGMA:-0.5}" \
                --num_generations "$generations" --pop_size "$ne_pop_size" \
                --num_evals "$ne_num_evals" --report_episodes "$report_episodes" \
                --hidden_dims $ne_hidden_dims \
                $diversity_args
        fi
        if wants dns; then
            # DNS uses AURORA (unsupervised, learned online) descriptors by
            # default -- the paper setting for tasks with no established
            # behaviour descriptor. Set DNS_DESCRIPTOR=handcrafted to override.
            run_condition "$suite" "$setting" dns \
                source/studies/gymnax/train_DNS_gymnax.py "$env" \
                --num_generations "$generations" --pop_size "$ne_pop_size" \
                --num_evals "$ne_num_evals" --report_episodes "$report_episodes" \
                --hidden_dims $ne_hidden_dims \
                --descriptor "${DNS_DESCRIPTOR:-aurora}" $diversity_args
        fi
        # DNS bred with the GA's GAUSSIAN operator instead of Iso+LineDD, and
        # nothing else changed. Iso+LineDD displaces a child along the line
        # between its parents, so its step scales with the population's own
        # spread and compounds: measured on the sigma=1.0 continual tree, `dns`
        # reaches |w| ~1e13 and `ga_isoline` saturates float32 at 3.4e38, while
        # every gaussian arm stays at ~1e2. Neither divergence buys anything --
        # across DNS trials |w| correlates with neither performance nor
        # diversity (|rho| <= 0.49, p >= 0.15, n=10 per task).
        #
        # With this arm the 2x2 closes: {gaussian, isoline} x {fitness
        # truncation, dominated novelty} is `ga` / `ga_isoline` / `dns_gaussian`
        # / `dns`, so the novelty-selection claim can be read off a pair whose
        # operator is not diverging. `--mutation_std` defaults to the GA's own
        # 0.5, which is what makes it operator-matched.
        #
        # Its OWN stationary reference, for the same reason ga_isoline has one:
        # FT subtracts a method's own noncontinual run, and this is a different
        # search from `dns`.
        if wants dns_gaussian; then
            run_condition "$suite" "$setting" dns_gaussian \
                source/studies/gymnax/train_DNS_gymnax.py "$env" \
                --variation gaussian --mutation_std "${DNS_MUTATION_STD:-0.5}" \
                --num_generations "$generations" --pop_size "$ne_pop_size" \
                --num_evals "$ne_num_evals" --report_episodes "$report_episodes" \
                --hidden_dims $ne_hidden_dims \
                --descriptor "${DNS_DESCRIPTOR:-aurora}" $diversity_args
        fi
        if wants es; then
            run_condition "$suite" "$setting" es \
                source/studies/gymnax/train_ES_gymnax.py "$env" \
                --num_generations "$generations" --pop_size "$ne_pop_size" \
                --num_evals "$ne_num_evals" --report_episodes "$report_episodes" \
                --hidden_dims $ne_hidden_dims \
                $diversity_args
        fi
        # NES is the same trainer at --algo nes: standardized fitness and a
        # plain SGD step instead of centered ranks and Adam. Same budget, same
        # network, same evals -- so `es` and `nes` differ by those two choices
        # and nothing else. Its sigma/learning rate are NOT the es arm's and
        # are not passed here; see NES_ENV_CONFIGS in
        # source/studies/gymnax/es_algorithms.py for which values it uses and why.
        if wants nes; then
            run_condition "$suite" "$setting" nes \
                source/studies/gymnax/train_ES_gymnax.py "$env" --algo nes \
                --num_generations "$generations" --pop_size "$ne_pop_size" \
                --num_evals "$ne_num_evals" --report_episodes "$report_episodes" \
                --hidden_dims $ne_hidden_dims \
                $diversity_args
        fi

        # HYPERPARAMETER ARMS (stationary), as in block_gymnax_continual: the
        # value is in the arm name and everything else is the reported `nes`
        # command line above, so a `nes_sigma0.2_lr0.02` directory is NES at
        # those two settings and nothing else changed.
        #
        #     nes_sigma<S>_lr<L>   NES at --sigma S --learning_rate L
        #
        # Added 2026-09-24 for the MountainCar tuning sweep
        # (scripts/train/queue_iclr_nes_mcar_tune.sh): the reported NES
        # setting (sigma 0.1, lr 0.05, NES_ENV_CONFIGS) finds MountainCar and
        # loses it again inside a phase, so width and step are re-tuned on
        # the stationary task, where every other arm's settings were chosen.
        # Matched on the literal name, not through `wants`: `all` does not
        # queue a sweep.
        #     es_sigma<S>_lr<L>    OpenES (centered ranks + Adam) at --sigma S
        #                          --learning_rate L, the `es` command line
        local hp_arm hp_sigma hp_lr
        for hp_arm in $METHODS; do
            case "$hp_arm" in
                es_sigma*_lr*)
                    hp_sigma="${hp_arm#es_sigma}"; hp_sigma="${hp_sigma%%_lr*}"
                    hp_lr="${hp_arm##*_lr}"
                    run_condition "$suite" "$setting" "$hp_arm" \
                        source/studies/gymnax/train_ES_gymnax.py "$env" \
                        --sigma "$hp_sigma" --learning_rate "$hp_lr" \
                        --num_generations "$generations" --pop_size "$ne_pop_size" \
                        --num_evals "$ne_num_evals" --report_episodes "$report_episodes" \
                        --hidden_dims $ne_hidden_dims \
                        $diversity_args ;;
                nes_sigma*_lr*)
                    hp_sigma="${hp_arm#nes_sigma}"; hp_sigma="${hp_sigma%%_lr*}"
                    hp_lr="${hp_arm##*_lr}"
                    run_condition "$suite" "$setting" "$hp_arm" \
                        source/studies/gymnax/train_ES_gymnax.py "$env" --algo nes \
                        --sigma "$hp_sigma" --learning_rate "$hp_lr" \
                        --num_generations "$generations" --pop_size "$ne_pop_size" \
                        --num_evals "$ne_num_evals" --report_episodes "$report_episodes" \
                        --hidden_dims $ne_hidden_dims \
                        $diversity_args ;;
            esac
        done
        # PPO and its continual-learning variants all share the RL trainer and
        # the same env-step budget; they differ only in --method.
        for rl_method in ppo trac redo; do
            if wants "$rl_method"; then
                run_condition "$suite" "$setting" "$rl_method" \
                    source/studies/gymnax/train_RL_gymnax.py "$env" --method "$rl_method" \
                    --num_timesteps "$rl_timesteps" --episode_length "$ne_episode_length" \
                    --num_eval_episodes "$report_episodes" \
                    --num_steps "$rl_num_steps"
            fi
        done

        # C-CHAIN takes extra controller args; values match run_CCHAIN_gymnax.sh.
        if wants cchain; then
            run_condition "$suite" "$setting" cchain \
                source/studies/gymnax/train_RL_gymnax.py "$env" --method cchain \
                --num_timesteps "$rl_timesteps" --episode_length "$ne_episode_length" \
                --num_eval_episodes "$report_episodes" \
                --num_steps "$rl_num_steps" \
                --chain_target_rel_scale "${CHAIN_TARGET_REL_SCALE:-10000}" \
                --chain_warmup_updates "${CHAIN_WARMUP_UPDATES:-10}" \
                --chain_coef_window "${CHAIN_COEF_WINDOW:-50}"
        fi

        # PBT: a population of PPO learners over PPO's OWN budget, on the
        # suite-generic runner (source/studies/generalists/train_ppo.py) -- the
        # one PPO this repo keeps (CLAUDE.md (a)). Same steps as the RL arms
        # above: $rl_timesteps at 2048 envs x $rl_num_steps steps an update,
        # ten checkpoint phases as every stationary cell has.
        for pbt_arm in $PBT_ARMS; do
        if wants_pbt "$pbt_arm"; then
            pbt_settings "$pbt_arm"
            local pbt_updates=$(( rl_timesteps / (2048 * rl_num_steps) ))
            run_condition "$suite" "$setting" "$pbt_arm" \
                source/studies/generalists/train_ppo.py "$env" --method pbt \
                --schedule task0 --num_tasks 1 \
                --num_updates "$pbt_updates" --task_interval $(( pbt_updates / 10 )) \
                --num_steps "$rl_num_steps" --eval_episodes "$report_episodes" \
                --pbt_pop_size "$pbt_n" --pbt_mode "$pbt_mode" \
                --pbt_interval "${PBT_INTERVAL:-10}"
        fi
        done
    done
}

# ----------------------------------------------------------------------------
# Block: gymnax_continual  (the same tasks, presented as a sequence of sub-tasks)
# ----------------------------------------------------------------------------
#
# A run is NUM_TASKS sub-tasks of TASK_INTERVAL generations each, seen one after
# the other by a single learner that is never told a switch happened. Sub-task 0
# is the unperturbed environment -- identical to the noncontinual block -- and
# every later sub-task adds a fixed observation-noise vector.
#
# The sub-task sequence is seeded from the trial index alone
# (source/studies/gymnax/continual_common.py), not from the training RNG, so at a given
# trial every method faces exactly the same sub-tasks in the same order and the
# runs are comparable one for one.

block_gymnax_continual() {
    local suite="gymnax" setting="continual"
    local envs="${ENVS:-CartPole-v1 Acrobot-v1 MountainCar-v0}"

    local num_tasks="${NUM_TASKS:-10}"
    local task_interval="${TASK_INTERVAL:-200}"      # generations per sub-task
    local generations=$((num_tasks * task_interval))
    local ne_pop_size="${NE_POP_SIZE:-512}"
    local ne_episode_length="${NE_EPISODE_LENGTH:-500}"

    # Pinned rather than left to each trainer's env defaults: ES defaults to 1
    # eval and GA/DNS to 10, and DNS/ES default to a 32x32 net on Acrobot and
    # MountainCar where GA and PPO use 16x16. Comparing methods requires the
    # same network and the same fitness estimator, so both are set here.
    # Declared before the budgets below because they are priced per rollout.
    # 3 IS THE PAPER VALUE, verified against every reported gymnax run's
    # train.log (GA, ES, DNS; stationary, continual and every population
    # size) and against the RL budgets matched to it: 512 x 3 x 500 x 600
    # = 4.608e8 env steps = the 4500 PPO updates those logs show. It used to
    # default to 10, which is a different experiment at 3.3x the cost whose
    # RL arm is no longer compute-matched to its NE arm.
    local ne_num_evals="${NE_NUM_EVALS:-3}"
    local ne_hidden_dims="${NE_HIDDEN_DIMS:-16 16}"

    # THE REPORTED number, which is not the number either family selects on.
    # NE: `centroid_fitness` and `elite_eval_fitness` are scored on fresh keys
    # over this many episodes (--report_episodes), because the search's own
    # columns are a max over the population of a `num_evals` mean whose FIRST
    # draw did the selecting -- optimistically biased, and by a different
    # amount per method (+5 for the GA, +53 for DNS on the sigma=1.0 tree), so
    # the bias does not cancel in a comparison.
    # RL: the same number, passed explicitly rather than left to the trainer's
    # default, so "10 episodes" is recorded in every run's config instead of
    # being a property of whichever trainer version ran.
    # NOT part of the budget product below: 2 genomes x 10 episodes a
    # generation against pop x evals = 1536 rollouts, and it never feeds
    # selection -- exactly as the RL evaluation has always sat outside
    # --num_timesteps.
    local report_episodes="${REPORT_EPISODES:-10}"

    # num_evals is part of the product: every one of the NE_NUM_EVALS rollouts
    # per individual is actually run, so a generation costs pop x evals x
    # episode_length environment steps. See block_gymnax_noncontinual.
    local rl_timesteps=$((ne_pop_size * ne_num_evals * ne_episode_length * generations))

    # TASK_PERIOD makes the sequence revisit sub-tasks it has already learned,
    # which is the only way forgetting can be read off a learner that carries
    # one policy and gets no task label -- see cycle_task_sequence(). Off by
    # default, so this block keeps producing the runs already on disk. The
    # revisit sweep is NUM_TASKS=20 TASK_PERIOD=10: each sub-task seen twice,
    # every one revisited after exactly 10 others.
    local task_period="${TASK_PERIOD:-0}"
    local task_period_args=""
    [ "$task_period" -gt 0 ] && task_period_args="--task_period $task_period"

    # The RL trainer counts sub-tasks in PPO updates, not generations, so the
    # switch points have to be converted or PPO would see a different task
    # sequence than the NE methods on the same wall of environment steps.
    # steps_per_update = num_envs x num_steps, which is what the RL trainer's
    # ENV_CONFIGS use for all three envs.
    local rl_steps_per_update="${RL_STEPS_PER_UPDATE:-102400}"
    local rl_task_interval=$((ne_pop_size * ne_num_evals * ne_episode_length * task_interval / rl_steps_per_update))
    if [ "$rl_task_interval" -lt 1 ]; then
        # Only reachable at test-sized budgets; the trainer takes a modulo by
        # this, so 0 would abort the run.
        echo "!!! WARNING: a sub-task is shorter than one PPO update"
        echo "!!! ($((ne_pop_size * ne_num_evals * ne_episode_length * task_interval)) steps vs $rl_steps_per_update per update);"
        echo "!!! clamping the RL switch interval to 1 update. The RL and NE"
        echo "!!! task sequences no longer line up on env steps."
        rl_task_interval=1
    fi

    local task_type="${TASK_TYPE:-noise}"

    # THE PHYSICS SUB-TASK FAMILY (TASK_TYPE=param). A sub-task rescales a
    # named group of the body's physics -- CartPole's pole length, Acrobot's
    # link masses, MountainCar's gravity -- and leaves the observation alone.
    # PARAM_NAME and PARAM_RANGE override the per-env defaults in
    # source/utils/task_sequence.py; PARAM_RANGE is a MULTIPLIER range, and
    # sub-task 0 is always 1.0x, the stock body.
    #
    # BOTH ARE EMPTY UNDER `noise`, AND SO IS the ES/NES --task_type flag.
    # That is not tidiness: the ES trainer only grew a --task_type flag on
    # 2026-09-08, and passing it unconditionally would make every ES and NES
    # job launched by an already-running noise sweep exit 2 on an unrecognised
    # argument. Under `noise` the command lines below are byte-identical to
    # what they were.
    local task_type_args=""
    local param_args=""
    if [ "$task_type" != "noise" ]; then
        task_type_args="--task_type $task_type"
        [ -n "${PARAM_NAME:-}" ] && param_args="--param_name $PARAM_NAME"
        [ -n "${PARAM_RANGE:-}" ] && param_args="$param_args --param_range $PARAM_RANGE"
    fi

    # THE THREE REPORTED SIGMAS. The continual gymnax experiment is run at
    # three observation-noise levels, and all three are reported
    # (results/gymnax_continual/sigma{0.02,1.0,2.0}/), so the sigma is a LOOP
    # here rather than a single setting.
    #
    # The directory tag is derived from the sigma inside that loop, not passed
    # separately. It used to take two variables that had to agree -- NOISE_RANGE
    # for the physics and ENV_DIR_SUFFIX for the name -- and nothing checked
    # them, so a run could train at one sigma and be filed under another.
    #
    # NOISE_RANGE still forces a single value, for a one-off at some other
    # sigma; it then names the directory itself, consistently.
    local sigmas="${SIGMAS:-0.02 1.0 2.0}"
    [ -n "${NOISE_RANGE:-}" ] && sigmas="$NOISE_RANGE"
    local noise_range

    # GIFs cost ~20 extra rollouts plus matplotlib at every sub-task boundary
    # and feed no metric; make_gifs.py renders them from the checkpoints
    # afterwards. Set CONTINUAL_GIFS=1 to render them inline anyway.
    local gif_args="--no_gifs"
    [ "${CONTINUAL_GIFS:-0}" = "1" ] && gif_args=""

    # Behavioural-diversity tracking for the three population-based methods.
    # Off by default here: the continual runs already on disk were made without
    # it, and turning it on silently would mean a condition whose trials are
    # half tracked and half not. Tracking is an observer -- it draws from its
    # own random stream and never feeds selection, verified by comparing
    # fitness traces with it on and off -- so a tracked trial and an untracked
    # one at the same seed are the same run; only the extra bd_* columns and
    # behaviour_snapshots.npz differ.
    #
    # BEHAVIOUR_SNAPSHOTS counts snapshots *per sub-task* in the continual
    # trainers, not over the whole run: diversity at a switch is the quantity
    # of interest, so the schedule is pinned to each sub-task.
    local ne_diversity_args=""
    if [ "${TRACK_DIVERSITY:-0}" = "1" ]; then
        ne_diversity_args="--track_diversity 1 \
            --diversity_interval ${DIVERSITY_INTERVAL:-10} \
            --occupancy_bins ${OCCUPANCY_BINS:-12} \
            --behaviour_snapshots ${BEHAVIOUR_SNAPSHOTS:-4}"
    fi

    echo "=========================================="
    echo "Block: gymnax_continual"
    echo "  envs       : $envs"
    echo "  methods    : $METHODS"
    echo "  trials     : $NUM_TRIALS (base seed $BASE_SEED)"
    echo "  tasks      : $num_tasks x $task_interval gens = $generations generations"
    echo "  NE budget  : pop $ne_pop_size x $ne_num_evals evals x $ne_episode_length steps"
    echo "  NE network : $ne_hidden_dims (pinned across methods)"
    echo "  RL budget  : $rl_timesteps env steps, switch every $rl_task_interval updates"
    echo "  task type  : $task_type"
    echo "  diversity  : ${TRACK_DIVERSITY:-0} (ga, dns, es; the RL methods have"
    echo "               a single policy, so there is no population to measure)"
    echo "  sigmas     : $sigmas (one run tree each, tagged _sigma<S>)"
    echo "  GPU        : $GPU"
    echo "=========================================="

    for env in $envs; do
      for sigma in $sigmas; do
        noise_range="$sigma"
        # Tag the run directory with the sigma it was actually trained at. One
        # assignment, so the two cannot drift apart.
        ENV_DIR_SUFFIX="_sigma${sigma}"

        if wants ga; then
            run_condition "$suite" "$setting" ga \
                source/studies/gymnax/train_GA_gymnax_continual.py "$env" \
                --num_generations "$generations" --pop_size "$ne_pop_size" \
                --num_evals "$ne_num_evals" --report_episodes "$report_episodes" \
                --hidden_dims $ne_hidden_dims \
                --task_interval "$task_interval" --task_type "$task_type" $param_args \
                --noise_range "$noise_range" $task_period_args $gif_args $ne_diversity_args
        fi
        # THE OPERATOR-MATCHED GA. Same fitness-truncation selection as `ga`,
        # bred with DNS's Iso+LineDD instead of gaussian mutation, so that
        # `dns` vs `ga_isoline` differs in the SELECTION RULE alone. Without
        # it, "DNS beats the GA" is a claim about novelty selection AND
        # recombination at once, and no run in the tree separates them.
        #
        # The widths are passed, not defaulted: this trainer's isoline
        # defaults are the DNS paper's corrected 0.005 / 0.05, while the
        # gymnax `dns` arm runs at the earlier study's 0.05 / 0.5. Leaving
        # them unset would put the two arms at operator scales a factor of ten
        # apart and reintroduce exactly the confound this arm exists to remove.
        if wants ga_isoline; then
            run_condition "$suite" "$setting" ga_isoline \
                source/studies/gymnax/train_GA_gymnax_continual.py "$env" \
                --variation isoline \
                --iso_sigma "${GA_ISO_SIGMA:-0.05}" --line_sigma "${GA_LINE_SIGMA:-0.5}" \
                --num_generations "$generations" --pop_size "$ne_pop_size" \
                --num_evals "$ne_num_evals" --report_episodes "$report_episodes" \
                --hidden_dims $ne_hidden_dims \
                --task_interval "$task_interval" --task_type "$task_type" $param_args \
                --noise_range "$noise_range" $task_period_args $gif_args $ne_diversity_args
        fi
        # `ga_refresh` and `ga_reeval` ARE GONE, and are not renamed arms.
        # The GA now re-scores its whole archive every generation
        # unconditionally (train_GA_gymnax_continual.py: "Unconditional since
        # 2026-09-08"), so `ga` IS what `ga_refresh` used to be and the two
        # flags those arms passed no longer exist -- queueing them exits 2.
        # `ga_reeval`, which re-scored only AT a boundary, is additionally a
        # method that branches on a task switch, which CLAUDE.md rule (d)
        # forbids. The operator ablation `ga_isoline` above replaces them as
        # the GA's second arm.
        if wants dns; then
            # Same AURORA descriptors as the noncontinual block. The encoder
            # is NOT refit at a sub-task switch: that would hand DNS alone the
            # signal that a switch happened, which no other method here gets.
            # Pass --aurora_retrain_on_task_switch 1 to restore the old
            # behaviour; DNS_RETRAIN_ON_SWITCH below is the block's knob.
            run_condition "$suite" "$setting" dns \
                source/studies/gymnax/train_DNS_gymnax_continual.py "$env" \
                --num_generations "$generations" --pop_size "$ne_pop_size" \
                --num_evals "$ne_num_evals" --report_episodes "$report_episodes" \
                --hidden_dims $ne_hidden_dims \
                --task_interval "$task_interval" --task_type "$task_type" $param_args \
                --noise_range "$noise_range" $task_period_args \
                --descriptor "${DNS_DESCRIPTOR:-aurora}" \
                --aurora_retrain_on_task_switch "${DNS_RETRAIN_ON_SWITCH:-0}" \
                $gif_args $ne_diversity_args
        fi
        # The gaussian-operator DNS; see the noncontinual block for why.
        if wants dns_gaussian; then
            run_condition "$suite" "$setting" dns_gaussian \
                source/studies/gymnax/train_DNS_gymnax_continual.py "$env" \
                --variation gaussian --mutation_std "${DNS_MUTATION_STD:-0.5}" \
                --num_generations "$generations" --pop_size "$ne_pop_size" \
                --num_evals "$ne_num_evals" --report_episodes "$report_episodes" \
                --hidden_dims $ne_hidden_dims \
                --task_interval "$task_interval" --task_type "$task_type" $param_args \
                --noise_range "$noise_range" $task_period_args \
                --descriptor "${DNS_DESCRIPTOR:-aurora}" \
                --aurora_retrain_on_task_switch "${DNS_RETRAIN_ON_SWITCH:-0}" \
                ${DNS_REPERTOIRE_RATIO:+--repertoire_ratio $DNS_REPERTOIRE_RATIO} \
                $gif_args $ne_diversity_args
        fi
        if wants es; then
            # $task_type_args is empty under `noise`, which is every run this
            # block made before 2026-09-08. Until that date this trainer had no
            # param branch and this call passed no --task_type at all, so an
            # `es` arm queued into a TASK_TYPE=param sweep would silently have
            # run the OBSERVATION-NOISE experiment into the param tree while
            # every other arm ran the physics one -- eight methods in one
            # directory, two of them on a different task sequence.
            run_condition "$suite" "$setting" es \
                source/studies/gymnax/train_ES_gymnax_continual.py "$env" \
                --num_generations "$generations" --pop_size "$ne_pop_size" \
                --num_evals "$ne_num_evals" --report_episodes "$report_episodes" \
                --hidden_dims $ne_hidden_dims \
                --task_interval "$task_interval" $task_type_args $param_args \
                --noise_range "$noise_range" $task_period_args $gif_args $ne_diversity_args ${ES_EXTRA_ARGS:-}
        fi
        # The continual block is where the two are expected to differ: ranks
        # discard how much better a perturbation was, and Adam carries a
        # second-moment estimate across a sub-task switch. See the module
        # docstring of source/algorithms/ne/es.py.
        if wants nes; then
            run_condition "$suite" "$setting" nes \
                source/studies/gymnax/train_ES_gymnax_continual.py "$env" --algo nes \
                --num_generations "$generations" --pop_size "$ne_pop_size" \
                --num_evals "$ne_num_evals" --report_episodes "$report_episodes" \
                --hidden_dims $ne_hidden_dims \
                --task_interval "$task_interval" $task_type_args $param_args \
                --noise_range "$noise_range" $task_period_args $gif_args $ne_diversity_args ${ES_EXTRA_ARGS:-}
        fi

        # PPO and its continual-learning variants share the RL trainer and the
        # same env-step budget; they differ only in --method.
        #
        # RL_EXTRA_ARGS / ES_EXTRA_ARGS (below) are appended verbatim, for the
        # hyperparameter appendix (scripts/train/queue_iclr_hparam.sh): a
        # sweep over PPO's epochs, rollout length or learning rate and NES's
        # sigma / learning rate / population size, each into its own
        # PROJECT_ROOT. Empty by default, so every other caller's command line
        # is unchanged. A rollout-length change must also set
        # RL_STEPS_PER_UPDATE, or the switches drift off the NE ones.
        for rl_method in ppo trac redo; do
            if wants "$rl_method"; then
                run_condition "$suite" "$setting" "$rl_method" \
                    source/studies/gymnax/train_RL_gymnax_continual.py "$env" --method "$rl_method" \
                    --num_timesteps "$rl_timesteps" --episode_length "$ne_episode_length" \
                    --num_eval_episodes "$report_episodes" \
                    --task_interval "$rl_task_interval" --task_type "$task_type" $param_args \
                    --noise_range "$noise_range" $task_period_args $gif_args ${RL_EXTRA_ARGS:-}
            fi
        done

        # C-CHAIN takes extra controller args; values match
        # run_CCHAIN_gymnax_continual.sh. Its coefficient is reset at every
        # sub-task switch, which is the point of the baseline here.
        if wants cchain; then
            run_condition "$suite" "$setting" cchain \
                source/studies/gymnax/train_RL_gymnax_continual.py "$env" --method cchain \
                --num_timesteps "$rl_timesteps" --episode_length "$ne_episode_length" \
                --num_eval_episodes "$report_episodes" \
                --task_interval "$rl_task_interval" --task_type "$task_type" $param_args \
                --noise_range "$noise_range" $task_period_args $gif_args \
                --chain_target_rel_scale "${CHAIN_TARGET_REL_SCALE:-10000}" \
                --chain_warmup_updates "${CHAIN_WARMUP_UPDATES:-10}" \
                --chain_coef_window "${CHAIN_COEF_WINDOW:-50}" ${RL_EXTRA_ARGS:-}
        fi

        # PBT: a population of PPO learners over PPO's OWN budget, on the
        # suite-generic runner (source/studies/generalists/train_ppo.py) -- the
        # one PPO this repo keeps (CLAUDE.md (a)). The SAME sub-tasks as the
        # arms above: both trainers draw the offsets with
        # `gymnax_classic.task_noise_vectors` at this trial, and the shared
        # runner's `switch` schedule over $task_period sub-tasks is exactly
        # --task_period's cycle; the same phase grid, $rl_task_interval updates
        # a phase at $rl_steps_per_update steps an update. The physics family
        # (task_type=param) is the shared suite's `physics_mult_range`: the
        # same `physics_mult_sequence(trial, ...)` draw the trainers above
        # make, so the multipliers match sub-task for sub-task.
        for pbt_arm in $PBT_ARMS; do
        if wants_pbt "$pbt_arm"; then
            pbt_settings "$pbt_arm"
            local pbt_tasks="$num_tasks" pbt_task_opts=""
            [ "$task_period" -gt 0 ] && pbt_tasks="$task_period"
            [ "$task_type" = actions ] && pbt_task_opts="--task_options task_mod=actions"
            if [ "$task_type" = param ]; then
                local pbt_range="${PARAM_RANGE:-default}"; pbt_range="${pbt_range// /,}"
                pbt_task_opts="--task_options task_mod=physics physics_mult_range=$pbt_range${PARAM_NAME:+ physics_param=$PARAM_NAME}"
            fi
            run_condition "$suite" "$setting" "$pbt_arm" \
                source/studies/generalists/train_ppo.py "$env" --method pbt \
                --schedule switch --num_tasks "$pbt_tasks" \
                --num_updates $(( rl_timesteps / rl_steps_per_update )) \
                --task_interval "$rl_task_interval" \
                --num_steps $(( rl_steps_per_update / 2048 )) \
                --noise_range "$noise_range" --eval_episodes "$report_episodes" \
                --pbt_pop_size "$pbt_n" --pbt_mode "$pbt_mode" \
                --pbt_interval "${PBT_INTERVAL:-10}" \
                $pbt_task_opts
        fi
        done

        # HYPERPARAMETER ARMS (appendix: effect of hyperparameters). The value
        # is part of the arm name, as N is for `ga_pop8` in gymnax_popsize, so
        # the directory says what the run is and the launcher's one job list
        # carries the whole sweep:
        #
        #     ga_sigma0.15    the GA at --mutation_std 0.15
        #     nes_sigma0.03   NES at --sigma 0.03 (its learning rate untouched)
        #     ppo_lr0.001     PPO at --learning_rate 0.001
        #     ppo_ent0.03     PPO at --ent_coef 0.03
        #
        # Everything else is the reported arm's command line above, so the
        # reported value IS the reported run (runs_centroid) and is not re-run
        # here. Matched on the literal name, not through `wants`: `all` does
        # not queue a sweep. scripts/train/queue_iclr_hparam.sh is the caller.
        local hp_arm hp_val hp_sigma hp_lr
        for hp_arm in $METHODS; do
            case "$hp_arm" in
                ga_sigma?*)
                    hp_val="${hp_arm#ga_sigma}"
                    run_condition "$suite" "$setting" "$hp_arm" \
                        source/studies/gymnax/train_GA_gymnax_continual.py "$env" \
                        --mutation_std "$hp_val" \
                        --num_generations "$generations" --pop_size "$ne_pop_size" \
                        --num_evals "$ne_num_evals" --report_episodes "$report_episodes" \
                        --hidden_dims $ne_hidden_dims \
                        --task_interval "$task_interval" --task_type "$task_type" $param_args \
                        --noise_range "$noise_range" $task_period_args $gif_args $ne_diversity_args ;;
                nes_sigma*_lr*)
                    # Width AND step together (the stationary block has the
                    # same arm): `nes_sigma0.2_lr0.02`. Listed before the
                    # width-only case, which would otherwise swallow it.
                    hp_sigma="${hp_arm#nes_sigma}"; hp_sigma="${hp_sigma%%_lr*}"
                    hp_lr="${hp_arm##*_lr}"
                    run_condition "$suite" "$setting" "$hp_arm" \
                        source/studies/gymnax/train_ES_gymnax_continual.py "$env" --algo nes \
                        --sigma "$hp_sigma" --learning_rate "$hp_lr" \
                        --num_generations "$generations" --pop_size "$ne_pop_size" \
                        --num_evals "$ne_num_evals" --report_episodes "$report_episodes" \
                        --hidden_dims $ne_hidden_dims \
                        --task_interval "$task_interval" $task_type_args $param_args \
                        --noise_range "$noise_range" $task_period_args $gif_args $ne_diversity_args ${ES_EXTRA_ARGS:-} ;;
                nes_sigma?*)
                    hp_val="${hp_arm#nes_sigma}"
                    run_condition "$suite" "$setting" "$hp_arm" \
                        source/studies/gymnax/train_ES_gymnax_continual.py "$env" --algo nes \
                        --sigma "$hp_val" \
                        --num_generations "$generations" --pop_size "$ne_pop_size" \
                        --num_evals "$ne_num_evals" --report_episodes "$report_episodes" \
                        --hidden_dims $ne_hidden_dims \
                        --task_interval "$task_interval" $task_type_args $param_args \
                        --noise_range "$noise_range" $task_period_args $gif_args $ne_diversity_args ;;
                ppo_lr?*|ppo_ent?*)
                    case "$hp_arm" in
                        ppo_lr*)  hp_val="--learning_rate ${hp_arm#ppo_lr}" ;;
                        *)        hp_val="--ent_coef ${hp_arm#ppo_ent}" ;;
                    esac
                    run_condition "$suite" "$setting" "$hp_arm" \
                        source/studies/gymnax/train_RL_gymnax_continual.py "$env" --method ppo \
                        $hp_val \
                        --num_timesteps "$rl_timesteps" --episode_length "$ne_episode_length" \
                        --num_eval_episodes "$report_episodes" \
                        --task_interval "$rl_task_interval" --task_type "$task_type" $param_args \
                        --noise_range "$noise_range" $task_period_args $gif_args ;;
            esac
        done
      done
    done
}

# ----------------------------------------------------------------------------
# Block: gymnax_popsize  (effect of population size on continual NE)
# ----------------------------------------------------------------------------
#
# The NE half of the population-size question, and the counterpart to
# gymnax_popsize_pbt (deleted 2026-09-13): both were CONTINUAL, both write to
# popsize_effect/pop_<N>/<method>/, so the NE and PBT curves are read off the
# same x-axis and the same sub-task sequence.
#
# Sub-task structure is inherited from block_gymnax_continual unchanged --
# NUM_TASKS sub-tasks of TASK_INTERVAL generations, per-env sigma from
# noise_range_for -- so a popsize run at N=512 is the same experiment as that
# block's run, and the continual block doubles as the N=512 point.

block_gymnax_popsize() {
    local suite="gymnax"
    local envs="${ENVS:-CartPole-v1 Acrobot-v1 MountainCar-v0}"
    # Stops at 128 on purpose. N=512 IS block_gymnax_continual -- same 10
    # sub-tasks x 200 generations, same evals, same 16x16 net, same sigma -- so
    # those runs are the top of this axis and `compare.py` reads them from
    # `continual/<method>` as the anchor. Adding 512 here would train the same
    # experiment a second time under `<method>_pop512` and put it in the figure
    # twice.
    local pop_sizes="${POP_SIZES:-2 8 32 128}"

    local num_tasks="${NUM_TASKS:-10}"
    local task_interval="${TASK_INTERVAL:-200}"      # generations per sub-task
    local generations=$((num_tasks * task_interval))
    local ne_episode_length="${NE_EPISODE_LENGTH:-500}"

    # Generations are held FIXED across N, so the evaluation budget grows with
    # N. This measures what a larger population buys per generation; it is
    # deliberately not the matched-compute comparison, in which small N would
    # instead run proportionally more generations. Same convention as
    # the deleted gymnax_popsize_pbt, which held updates fixed across N.
    # 3 IS THE PAPER VALUE, verified against every reported gymnax run's
    # train.log (GA, ES, DNS; stationary, continual and every population
    # size) and against the RL budgets matched to it: 512 x 3 x 500 x 600
    # = 4.608e8 env steps = the 4500 PPO updates those logs show. It used to
    # default to 10, which is a different experiment at 3.3x the cost whose
    # RL arm is no longer compute-matched to its NE arm.
    local ne_num_evals="${NE_NUM_EVALS:-3}"
    local ne_hidden_dims="${NE_HIDDEN_DIMS:-16 16}"

    local task_type="${TASK_TYPE:-noise}"
    # One noise level for the whole sweep -- see the note at the env loop.
    local sigma="${SIGMA:-1.0}"
    local noise_range

    # Sub-task revisits, exactly as block_gymnax_continual defines them. This
    # block used to ignore TASK_PERIOD even though all three trainers it calls
    # accept --task_period, so a population sweep could not be run at the
    # revisit structure the continual block was measuring forgetting with --
    # the two would not have been the same experiment at N=512.
    local task_period="${TASK_PERIOD:-0}"
    local task_period_args=""
    [ "$task_period" -gt 0 ] && task_period_args="--task_period $task_period"

    # Same reasoning as block_gymnax_continual: GIFs cost rollouts at every
    # sub-task boundary and feed no metric.
    local gif_args="--no_gifs"
    [ "${CONTINUAL_GIFS:-0}" = "1" ] && gif_args=""

    # Behavioural diversity of the population, as in gymnax_noncontinual. All
    # three continual NE trainers take these flags.
    #
    # OFF by default here, and that is the one place in this file where it is:
    # the sweep on disk was run without it, and switching it on silently would
    # produce a tree whose cells are half tracked and half not, which is worse
    # than one where none are. Turn it on deliberately, on runs being made
    # from scratch, and archive the untracked runs of that condition first so
    # the two are not silently mixed.
    local ne_diversity_args=""
    if [ "${TRACK_DIVERSITY:-0}" = "1" ]; then
        ne_diversity_args="--track_diversity 1 \
            --diversity_interval ${DIVERSITY_INTERVAL:-10} \
            --occupancy_bins ${OCCUPANCY_BINS:-12} \
            --behaviour_snapshots ${BEHAVIOUR_SNAPSHOTS:-8}"
    fi

    echo "=========================================="
    echo "Block: gymnax_popsize (continual)"
    echo "  envs       : $envs"
    echo "  methods    : $METHODS"
    echo "  pop sizes  : $pop_sizes"
    echo "  trials     : $NUM_TRIALS (base seed $BASE_SEED)"
    echo "  tasks      : $num_tasks x $task_interval gens = $generations generations"
    echo "  NE network : $ne_hidden_dims (pinned across methods)"
    echo "  diversity  : ${TRACK_DIVERSITY:-0} (every ${DIVERSITY_INTERVAL:-10} gens, all NE methods)"
    echo "  task type  : $task_type"
    for e in $envs; do
        echo "     sigma($e) = ${SIGMA:-1.0}"
    done
    echo "  GPU        : $GPU"
    echo "=========================================="

    for pop in $pop_sizes; do
        # POPULATION SIZE IS PART OF THE METHOD NAME, not a hierarchy above it:
        # `continual/ga_pop8/` rather than `popsize_effect/pop_8/ga/`. That is what
        # `compare.py` reads (popsize_method()) and what the run tree holds, and it
        # is what lets the sweep's largest N BE the continual block's own run
        # instead of a second copy of it filed under another path.
        local setting="continual"

        for env in $envs; do
            # THE POPULATION-SIZE SWEEP IS SIGMA 1.0 ONLY. The three-sigma sweep
            # is the method comparison (block_gymnax_continual); this block asks
            # a different question -- how the same method scales with N -- and
            # answers it at one noise level, which is what is on disk
            # (every pop_<N> arm has _sigma1.0 and nothing else).
            noise_range="$sigma"
            ENV_DIR_SUFFIX="_sigma${sigma}"

            if wants ga; then
                run_condition "$suite" "$setting" ga_pop${pop} \
                    source/studies/gymnax/train_GA_gymnax_continual.py "$env" \
                    --num_generations "$generations" --pop_size "$pop" \
                    --num_evals "$ne_num_evals" --hidden_dims $ne_hidden_dims \
                    --task_interval "$task_interval" --task_type "$task_type" $param_args \
                    --noise_range "$noise_range" $task_period_args \
                    $gif_args $ne_diversity_args
            fi
            if wants dns; then
                run_condition "$suite" "$setting" dns_pop${pop} \
                    source/studies/gymnax/train_DNS_gymnax_continual.py "$env" \
                    --num_generations "$generations" --pop_size "$pop" \
                    --num_evals "$ne_num_evals" --hidden_dims $ne_hidden_dims \
                    --task_interval "$task_interval" --task_type "$task_type" $param_args \
                    --noise_range "$noise_range" $task_period_args \
                    --descriptor "${DNS_DESCRIPTOR:-aurora}" $gif_args \
                    $ne_diversity_args
            fi
            if wants es; then
                # No --task_type: the ES trainer only implements the noise variant.
                run_condition "$suite" "$setting" es_pop${pop} \
                    source/studies/gymnax/train_ES_gymnax_continual.py "$env" \
                    --num_generations "$generations" --pop_size "$pop" \
                    --num_evals "$ne_num_evals" --hidden_dims $ne_hidden_dims \
                    --task_interval "$task_interval" \
                    --noise_range "$noise_range" $task_period_args \
                    $gif_args $ne_diversity_args
            fi
            # NES's population-size axis. A (1, lambda) search gradient is
            # estimated from the population, so N is a gradient-noise knob for
            # it in a way it is not for a GA -- worth having on the same axis
            # as the other three rather than at N=512 alone.
            if wants nes; then
                run_condition "$suite" "$setting" nes_pop${pop} \
                    source/studies/gymnax/train_ES_gymnax_continual.py "$env" --algo nes \
                    --num_generations "$generations" --pop_size "$pop" \
                    --num_evals "$ne_num_evals" --hidden_dims $ne_hidden_dims \
                    --task_interval "$task_interval" \
                    --noise_range "$noise_range" $task_period_args \
                    $gif_args $ne_diversity_args
            fi
        done
    done
}

# ----------------------------------------------------------------------------
# Block: mujoco_noncontinual  (CheetahRun, the continuous-control counterpart)
# ----------------------------------------------------------------------------
#
# The same question as gymnax_noncontinual -- how does each method do when the
# task never changes -- on a task where the policy is 128x128 over a 1000-step
# MJX episode instead of a 16x16 net over a 500-step gymnax episode. It is the
# control for the mujoco continual sweep in scripts/mujoco/, so it runs at that
# sweep's *default* friction (FRICTION_DEFAULT=1.0 there), which is its
# unperturbed sub-task.
#
# Differences from the gymnax blocks, all of them properties of the mujoco
# trainers rather than choices made here:
#
#   * cchain runs at the reference implementation's *continuous-control*
#     hyperparameters, not the gymnax ones. CheetahRun's policy is a Normal, so
#     the churn term is the MSE between action means (crl_dmc in the C-CHAIN
#     repo) rather than a cross-entropy over logits, and the two sit on
#     completely different scales: target_rel_scale 0.05 here against 10000 for
#     gymnax. Passing the gymnax value would swamp the PPO objective.
#   * No --hidden_dims. All three NE trainers hardcode (128, 128), so the
#     networks are already matched and there is nothing to pin.
#   * Diversity is tracked, but not with the same descriptor families as gymnax.
#     CheetahRun's hand-designed BD is the per-foot duty factor; it has no
#     occupancy family, and its continuous action space replaces
#     bd_action_freq_diversity/bd_probe_js with bd_action_usage_diversity/
#     bd_probe_action_dist. See the diversity_args block below and
#     source/common/behaviour_descriptors.py.

block_mujoco_noncontinual() {
    local suite="mujoco" setting="noncontinual"
    local envs="${ENVS:-CheetahRun}"

    # Friction multipliers to run. 1.0 is the unperturbed environment and the
    # default; the continual sweep's other two values (0.2, 5.0) are available
    # as FRICTIONS="0.2 1.0 5.0" if the non-continual ceiling is wanted at each.
    local frictions="${FRICTIONS:-1.0}"

    # Same budget convention as gymnax_noncontinual: every method gets
    # GENERATIONS generations of NE_POP_SIZE individuals rolled out for
    # NE_EPISODE_LENGTH steps, and PPO gets the equivalent environment steps.
    local generations="${GENERATIONS:-500}"
    local ne_pop_size="${NE_POP_SIZE:-512}"
    local ne_episode_length="${NE_EPISODE_LENGTH:-1000}"

    # Pinned across methods for the reason the gymnax blocks pin it: the
    # trainers' defaults are not equal. GA and DNS default to 3 rollouts per
    # individual and ES to 1, which would plot a 3-rollout mean against a single
    # noisy rollout and read as a method difference.
    local ne_num_evals="${NE_NUM_EVALS:-3}"

    # num_evals IS in this product. It used to be left out on the grounds that
    # the gymnax block matched on the same num_evals-free product, but that made
    # both suites wrong the same way rather than comparable: the mujoco scoring
    # function runs all num_evals rollouts and selects on their mean, so they
    # are real environment steps. Both suites now price them.
    local rl_timesteps="${RL_TIMESTEPS:-$((ne_pop_size * ne_num_evals * ne_episode_length * generations))}"

    # Behavioural diversity, same observer contract as the gymnax blocks: it
    # never feeds selection and draws from its own random stream, so a run is
    # identical with it on or off. Two things differ here because the task does:
    #
    #   * The hand-designed descriptor is the duty factor of each foot -- how
    #     much of the episode it spends on the ground -- which is the standard
    #     QD descriptor for a legged robot and, unlike anything built on the
    #     root velocity, is not CheetahRun's reward under another name.
    #   * There is no occupancy family and no `bd_probe_js`. CheetahRun declares
    #     no occupancy coordinates, and its actions are 6 continuous torques
    #     rather than a distribution to take a divergence between, so the probe
    #     row is `bd_probe_action_dist` and the action row
    #     `bd_action_usage_diversity`. OCCUPANCY_BINS is still passed and still
    #     ignored, so one exported setting works for both suites.
    local diversity_args=""
    if [ "${TRACK_DIVERSITY:-1}" = "1" ]; then
        diversity_args="--track_diversity 1 \
            --diversity_interval ${DIVERSITY_INTERVAL:-10} \
            --occupancy_bins ${OCCUPANCY_BINS:-12} \
            --behaviour_snapshots ${BEHAVIOUR_SNAPSHOTS:-8}"
    else
        diversity_args="--track_diversity 0"
    fi

    # DNS knobs, shared verbatim with block_mujoco_continual.
    #
    # DNS_DESCRIPTOR picks the space selection measures novelty in. `aurora`
    # (unbounded LSTM latent) is what the paper's abstract advertises, but the
    # DNS release never actually runs it -- main_aurora.py is a one-line stub --
    # and every task it *was* validated on has a bounded descriptor. On
    # CheetahRun the unbounded space lets diverging genotypes buy survival with
    # sheer descriptor magnitude; `handcrafted` (final torso height and pitch)
    # is bounded and is the closer analogue of the reference's `_uni`/`_omni`
    # descriptors. See docs/dns_cheetah_diagnosis.md.
    #
    # DNS_REEVAL_SURVIVORS=0 matches the reference, whose repertoire `add()`
    # keeps each individual's fitness from when it was created. Re-evaluating
    # costs a second full population of rollouts per generation -- so DNS spends
    # 1.5x GA's budget -- and re-rolls the one elitism guarantee DNS has (an
    # individual with no fitter neighbour) under evaluation noise every
    # generation, which is why DNS was the only method whose best fitness fell.
    #
    # DNS_DIR is the directory the condition lands in, so a handcrafted sweep
    # does not overwrite an aurora one; make_figures.py reads whichever is named.
    local dns_dir="${DNS_DIR:-dns}"
    # --batch_size defaults to the full population, not the trainer's pop_size/2.
    # GA and ES roll out all pop_size individuals every generation; DNS rolls out
    # only its offspring, so at pop_size/2 it spends exactly half the env steps
    # per generation and the shared generation axis stops meaning the same thing
    # for all three. This costs the reference's 1:4 offspring:population ratio
    # (ours becomes 1:1) and buys budget parity, which the figures need more.
    local dns_args="--descriptor ${DNS_DESCRIPTOR:-aurora} \
        --batch_size ${DNS_BATCH_SIZE:-$ne_pop_size} \
        --iso_sigma ${DNS_ISO_SIGMA:-0.005} \
        --line_sigma ${DNS_LINE_SIGMA:-0.05} \
        --reeval_survivors ${DNS_REEVAL_SURVIVORS:-0} \
        --normalize_descriptors ${DNS_NORMALIZE_DESCRIPTORS:-0} \
        --aurora_bounded_descriptor ${DNS_BOUNDED_DESCRIPTOR:-0}"

    echo "=========================================="
    echo "Block: mujoco_noncontinual"
    echo "  envs       : $envs"
    echo "  frictions  : $frictions"
    echo "  DNS        : $dns_args -> $dns_dir"
    echo "  methods    : $METHODS"
    echo "  trials     : $NUM_TRIALS (base seed $BASE_SEED)"
    echo "  generations: $generations (pop $ne_pop_size x $ne_num_evals evals x $ne_episode_length steps)"
    echo "  PPO budget : $rl_timesteps env steps (matched, num_evals included)"
    echo "  NE network : (128, 128), hardcoded in the trainers"
    echo "  diversity  : ${TRACK_DIVERSITY:-1} (every ${DIVERSITY_INTERVAL:-10} gens, foot-contact BD, NE methods only)"
    echo "  GPU        : $GPU"
    echo "=========================================="

    for env in $envs; do
        for friction in $frictions; do
            # projects/.../mujoco/noncontinual/<method>/CheetahRun_friction1_0/trial_N.
            # The friction is in the directory name because a run at 0.2 and a
            # run at 1.0 are different experiments that must not overwrite each
            # other -- the same job ENV_DIR_SUFFIX does for sigma in the gymnax
            # continual block.
            ENV_DIR_SUFFIX="_friction${friction//./_}"

            if wants ga; then
                run_condition "$suite" "$setting" ga \
                    source/studies/mujoco/train_GA_cheetah.py "$env" \
                    --friction "$friction" \
                    --num_generations "$generations" --pop_size "$ne_pop_size" \
                    --num_evals "$ne_num_evals" \
                    --episode_length "$ne_episode_length" \
                    $diversity_args
            fi
            if wants dns; then
                run_condition "$suite" "$setting" "$dns_dir" \
                    source/studies/mujoco/train_DNS_cheetah.py "$env" \
                    --friction "$friction" \
                    --num_generations "$generations" --pop_size "$ne_pop_size" \
                    --num_evals "$ne_num_evals" \
                    --episode_length "$ne_episode_length" \
                    $dns_args \
                    $diversity_args
            fi
            if wants es; then
                run_condition "$suite" "$setting" es \
                    source/studies/mujoco/train_ES_cheetah.py "$env" \
                    --friction "$friction" \
                    --num_generations "$generations" --pop_size "$ne_pop_size" \
                    --num_evals "$ne_num_evals" \
                    --episode_length "$ne_episode_length" \
                    $diversity_args
            fi

            # PPO and its two continual-learning variants share the RL trainer
            # and the same env-step budget; they differ only in the flags below.
            if wants ppo; then
                run_condition "$suite" "$setting" ppo \
                    source/studies/mujoco/train_RL_cheetah.py "$env" \
                    --friction "$friction" \
                    --num_timesteps "$rl_timesteps" \
                    --episode_length "$ne_episode_length"
            fi
            if wants trac; then
                run_condition "$suite" "$setting" trac \
                    source/studies/mujoco/train_RL_cheetah.py "$env" \
                    --friction "$friction" \
                    --num_timesteps "$rl_timesteps" \
                    --episode_length "$ne_episode_length" \
                    --use_trac
            fi
            if wants redo; then
                # redo_frequency 1 matches scripts/mujoco/run_ReDo_PPO_cheetah.sh.
                run_condition "$suite" "$setting" redo \
                    source/studies/mujoco/train_RL_cheetah.py "$env" \
                    --friction "$friction" \
                    --num_timesteps "$rl_timesteps" \
                    --episode_length "$ne_episode_length" \
                    --use_redo --redo_frequency "${REDO_FREQUENCY:-1}"
            fi
            if wants cchain; then
                # Continuous-control defaults; see the block header. These are
                # deliberately NOT the CHAIN_* values the gymnax blocks pass.
                run_condition "$suite" "$setting" cchain \
                    source/studies/mujoco/train_RL_cheetah.py "$env" \
                    --friction "$friction" \
                    --num_timesteps "$rl_timesteps" \
                    --episode_length "$ne_episode_length" \
                    --use_cchain \
                    --chain_target_rel_scale "${MUJOCO_CHAIN_TARGET_REL_SCALE:-0.05}" \
                    --chain_warmup_iterations "${MUJOCO_CHAIN_WARMUP_ITERATIONS:-50}" \
                    --chain_coef_window "${MUJOCO_CHAIN_COEF_WINDOW:-100}"
            fi

            unset ENV_DIR_SUFFIX
        done
    done
}

# ----------------------------------------------------------------------------
# Block: mujoco_continual  (CheetahRun as a sequence of friction sub-tasks)
# ----------------------------------------------------------------------------
#
# The continuous-control counterpart of gymnax_continual, and the continual
# counterpart of mujoco_noncontinual. A run is NUM_TASKS sub-tasks of
# TASK_INTERVAL generations each, seen one after the other by a single learner
# that is never told a switch happened. Sub-task 0 is friction 1.0 -- the
# unperturbed environment, identical to the noncontinual block -- and the
# multiplier then cycles 1.0 -> 0.2 -> 5.0 (source/envs/mjx_cheetah.py).
#
# The sequence is a deterministic cycle rather than a sample, so at a given
# trial every method faces exactly the same physics at the same point in its
# budget. Nothing about it depends on the training RNG.
#
# Differences from gymnax_continual, all of them properties of the task:
#
#   * The sub-task variable is a physics multiplier, not an observation-noise
#     vector, so there is no per-env sigma and no --task_type. NOISE_RANGE and
#     friends do not apply here.
#   * NUM_TASKS defaults to 12 rather than 10: the cycle has period 3, and 12 is
#     the nearest count that gives each multiplier the same number of sub-tasks
#     (four each). At 10 the default friction would appear four times and the
#     other two three times, which biases every average over sub-tasks.
#   * No RL step-to-update conversion. The RL trainer takes --timesteps_per_task
#     directly rather than a switch interval counted in PPO updates, so nothing
#     has to be divided by a steps-per-update constant that the trainer's own
#     --num_envs/--batch_size could silently invalidate. The figure still puts
#     RL on the generation axis the same way the gymnax one does, with
#     timestep / (pop_size * episode_length).
#   * The NE trainers hardcode (128, 128); the RL trainer does not, so its
#     hidden sizes are pinned here to match.

block_mujoco_continual() {
    local suite="mujoco" setting="continual"
    local envs="${ENVS:-CheetahRun}"
    local task_mod="${TASK_MOD:-friction}"

    # 30 IS THE PAPER VALUE. The reported tree is CheetahRun_friction_t30:
    # 3000 generations of 100, cycling friction 1.0 / 0.2 / 5.0, and PPO's saved
    # config.json agrees (timesteps_per_task 153,600,000 = 512 x 3 x 1000 x 100,
    # total 4.608e9 over 30). This defaulted to 12, which is a shorter sequence
    # that `compare.py` does not have a figure for.
    local num_tasks="${NUM_TASKS:-30}"
    local task_interval="${TASK_INTERVAL:-100}"     # generations per sub-task
    # The `_t<N>` half of the directory tag is derived from num_tasks where the
    # tag is composed, below. A 30-task run filed as `CheetahRun_friction` would
    # be read by `mujoco_continual_config` as a 12-task one -- it keys on that
    # suffix to know how many switch verticals to draw and what budget to
    # integrate over.
    local generations=$((num_tasks * task_interval))
    local ne_pop_size="${NE_POP_SIZE:-512}"
    local ne_episode_length="${NE_EPISODE_LENGTH:-1000}"
    local ne_num_evals="${NE_NUM_EVALS:-3}"

    # PPO gets the NE budget per sub-task, in environment steps: the same
    # pop x evals x episode x generations product the noncontinual block matches
    # on, num_evals included because every rollout of every eval is executed.
    local rl_timesteps_per_task="${RL_TIMESTEPS_PER_TASK:-$((ne_pop_size * ne_num_evals * ne_episode_length * task_interval))}"
    # Evaluations (and logged points) per sub-task. The trainer spreads the
    # sub-task's training steps over exactly this many epochs, so the budget is
    # hit on the nose only when this divides the number of training steps in a
    # sub-task -- otherwise it rounds and warns.
    #
    # At the defaults a training step is 256 x 10 x 32 = 81,920 env steps, so a
    # 51.2M-step sub-task is 625 of them; 125 divides it (5 training steps per
    # evaluation) and 100 does not. This is NOT required to be task_interval:
    # make_figures.py puts the RL curve on the generation axis with
    # timestep / (pop_size * episode_length), not with the eval index.
    local rl_evals_per_task="${RL_EVALS_PER_TASK:-125}"
    local rl_hidden="${RL_HIDDEN_SIZES:-128,128}"

    # Physics multipliers of the cycle. Sub-task 0 uses the first.
    local mult_default="${FRICTION_DEFAULT_MULT:-1.0}"
    local mult_low="${FRICTION_LOW_MULT:-0.2}"
    local mult_high="${FRICTION_HIGH_MULT:-5.0}"

    # Same observer contract and same descriptor families as
    # block_mujoco_noncontinual: foot duty factor, AURORA, action usage, probe
    # action distance. Never feeds selection, own random stream.
    local diversity_args=""
    if [ "${TRACK_DIVERSITY:-1}" = "1" ]; then
        diversity_args="--track_diversity 1 \
            --diversity_interval ${DIVERSITY_INTERVAL:-10} \
            --occupancy_bins ${OCCUPANCY_BINS:-12} \
            --behaviour_snapshots ${BEHAVIOUR_SNAPSHOTS:-8}"
    else
        diversity_args="--track_diversity 0"
    fi

    # The observation-offset axis, and it MUST be passed rather than left to the
    # trainers' defaults. Both the RL and the NE cheetah trainers already accept
    # `--obs_noise_range` and `--task_period` -- they come from the shared
    # `add_continual_args` in source/envs/mjx_cheetah.py -- but this block
    # never passed either, so `TASK_MOD=obs_noise` alone ran the whole arm at
    # SIGMA 0: an unperturbed cheetah filed under an obs-noise name, with
    # nothing in the run's own output to say so. That is the same silent zero
    # that made runs_repro2's ant "obsnoise" tree a friction experiment.
    #
    # Off by default (sigma 0, no revisit), so every existing mujoco tree is
    # unchanged: under TASK_MOD=friction the offset wrapper is not constructed
    # at all and these two are inert.
    #
    # The reference arm is `runs/mujoco/continual_cheetah_obsnoise_g16`
    # (CheetahRun_obsnoise_g16_2p0): sigma 2.0, period 0, 12 sub-tasks x 16
    # generations, `sampling_mode cycle`.
    local obs_noise_range="${OBS_NOISE_RANGE:-0.0}"
    local obs_task_period="${TASK_PERIOD:-0}"
    local obs_noise_args="--obs_noise_range $obs_noise_range \
        --task_period $obs_task_period"

    # PPO's rollout width and step size. Both are knobs rather than the
    # trainer's defaults because the reported obs-noise arm does not use those
    # defaults: its config.json records num_envs 4096 against the trainer's
    # 2048, and learning_rate 1e-4 against 3e-4. Left implicit, a rerun of that
    # arm would be a different experiment while looking like a reproduction.
    # The defaults here ARE the trainer's, so the friction trees are unchanged.
    local rl_num_envs="${MUJOCO_RL_NUM_ENVS:-2048}"
    local rl_learning_rate="${MUJOCO_RL_LEARNING_RATE:-3e-4}"
    local rl_ppo_args="--num_envs $rl_num_envs \
        --learning_rate $rl_learning_rate"

    local task_args="--num_tasks $num_tasks --task_mod $task_mod \
        --friction_default_mult $mult_default \
        --friction_low_mult $mult_low --friction_high_mult $mult_high \
        $obs_noise_args"

    # Same DNS knobs as block_mujoco_noncontinual, which documents them.
    local dns_dir="${DNS_DIR:-dns}"

    # Footage of the sub-task best genome, rendered at each boundary into
    # gifs/task_NN_<label>/ -- the same layout the RL trainers already write, so
    # an NE and an RL run of the same sequence can be flipped through together.
    # Drawn from its own RNG stream, so a run is identical with this at 0 or 3.
    local ne_gif_args="--gifs_per_task ${GIFS_PER_TASK:-3}"
    # --batch_size defaults to the full population, not the trainer's pop_size/2.
    # GA and ES roll out all pop_size individuals every generation; DNS rolls out
    # only its offspring, so at pop_size/2 it spends exactly half the env steps
    # per generation and the shared generation axis stops meaning the same thing
    # for all three. This costs the reference's 1:4 offspring:population ratio
    # (ours becomes 1:1) and buys budget parity, which the figures need more.
    local dns_args="--descriptor ${DNS_DESCRIPTOR:-aurora} \
        --batch_size ${DNS_BATCH_SIZE:-$ne_pop_size} \
        --iso_sigma ${DNS_ISO_SIGMA:-0.005} \
        --line_sigma ${DNS_LINE_SIGMA:-0.05} \
        --reeval_survivors ${DNS_REEVAL_SURVIVORS:-0} \
        --normalize_descriptors ${DNS_NORMALIZE_DESCRIPTORS:-0} \
        --aurora_bounded_descriptor ${DNS_BOUNDED_DESCRIPTOR:-0}"

    echo "=========================================="
    echo "Block: mujoco_continual"
    echo "  envs       : $envs"
    echo "  methods    : $METHODS"
    echo "  DNS        : $dns_args -> $dns_dir"
    echo "  trials     : $NUM_TRIALS (base seed $BASE_SEED)"
    echo "  tasks      : $num_tasks x $task_interval gens = $generations generations"
    echo "  task var   : $task_mod, cycling $mult_default -> $mult_low -> $mult_high"
    echo "  NE budget  : pop $ne_pop_size x $ne_num_evals evals x $ne_episode_length steps"
    echo "  RL budget  : $rl_timesteps_per_task env steps per sub-task, "
    echo "               $rl_evals_per_task evals per sub-task, hidden $rl_hidden"
    echo "  PBT        : pops ${PBT_POP_SIZES:-2 8}, mode ${PBT_MODE:-weights_only}, "
    echo "               interval ${PBT_INTERVAL:-10}, same budget PER MEMBER as PPO"
    echo "  NE network : (128, 128), hardcoded in the trainers"
    echo "  diversity  : ${TRACK_DIVERSITY:-1} (every ${DIVERSITY_INTERVAL:-10} gens, foot-contact BD, NE methods only)"
    echo "  GPU        : $GPU"
    echo "=========================================="

    for env in $envs; do
        # The task variable is in the directory name for the reason the
        # noncontinual block puts friction there: a friction sweep and a gravity
        # sweep are different experiments that must not share a directory.
        # ENV_DIR_EXTRA tags a sub-task *count* into the directory on top of the
        # task variable, for the same reason the variable is there at all: a
        # 12-sub-task sequence and a 30-sub-task one are different experiments
        # and must not pool. The RL-only long sequence uses `_t30`; unset, the
        # layout is unchanged.
        ENV_DIR_SUFFIX="_${task_mod}${ENV_DIR_EXTRA:-_t${num_tasks}}"

        if wants ga; then
            run_condition "$suite" "$setting" ga \
                source/studies/mujoco/train_GA_cheetah_continual.py "$env" \
                $task_args \
                --gens_per_task "$task_interval" --pop_size "$ne_pop_size" \
                --num_evals "$ne_num_evals" \
                --episode_length "$ne_episode_length" \
                $diversity_args \
                $ne_gif_args
        fi
        if wants dns; then
            # Under --descriptor aurora the encoder is retrained at every
            # sub-task switch (the switch is what changes the observation
            # distribution the space is learned from); disable with
            # --no_aurora_retrain_on_task_switch. Under --descriptor handcrafted
            # there is no encoder and the flag is inert.
            run_condition "$suite" "$setting" "$dns_dir" \
                source/studies/mujoco/train_DNS_cheetah_continual.py "$env" \
                $task_args \
                --gens_per_task "$task_interval" --pop_size "$ne_pop_size" \
                --num_evals "$ne_num_evals" \
                --episode_length "$ne_episode_length" \
                $dns_args \
                $diversity_args \
                $ne_gif_args
        fi
        if wants es; then
            run_condition "$suite" "$setting" es \
                source/studies/mujoco/train_ES_cheetah_continual.py "$env" \
                $task_args \
                --gens_per_task "$task_interval" --pop_size "$ne_pop_size" \
                --num_evals "$ne_num_evals" \
                --episode_length "$ne_episode_length" \
                $diversity_args \
                $ne_gif_args
        fi

        # PPO and its three continual-learning variants share the RL trainer and
        # the same per-sub-task step budget; they differ only in the flags below.
        local rl_args="$task_args \
            --timesteps_per_task $rl_timesteps_per_task \
            --num_evals_per_task $rl_evals_per_task \
            --episode_length $ne_episode_length \
            --policy_hidden_sizes $rl_hidden --value_hidden_sizes $rl_hidden \
            $rl_ppo_args \
            --gifs_per_task ${GIFS_PER_TASK:-3}"

        if wants ppo; then
            run_condition "$suite" "$setting" ppo \
                source/studies/mujoco/train_RL_cheetah_continual.py "$env" $rl_args
        fi
        if wants trac; then
            run_condition "$suite" "$setting" trac \
                source/studies/mujoco/train_RL_cheetah_continual.py "$env" $rl_args \
                --use_trac
        fi
        if wants redo; then
            # redo_frequency 1 matches the noncontinual block.
            run_condition "$suite" "$setting" redo \
                source/studies/mujoco/train_RL_cheetah_continual.py "$env" $rl_args \
                --use_redo --redo_frequency "${REDO_FREQUENCY:-1}"
        fi
        if wants cchain; then
            # Continuous-control C-CHAIN values, as in block_mujoco_noncontinual
            # -- deliberately NOT the CHAIN_* the gymnax blocks pass. The
            # coefficient controller is reset at every sub-task switch, which is
            # the continual half of the method.
            run_condition "$suite" "$setting" cchain \
                source/studies/mujoco/train_RL_cheetah_continual.py "$env" $rl_args \
                --use_cchain \
                --chain_target_rel_scale "${MUJOCO_CHAIN_TARGET_REL_SCALE:-0.05}" \
                --chain_warmup_iterations "${MUJOCO_CHAIN_WARMUP_ITERATIONS:-50}" \
                --chain_coef_window "${MUJOCO_CHAIN_COEF_WINDOW:-100}"
        fi

        # PBT-PPO, the population-based RL baseline, at one directory per
        # population size: pbt_pop2, pbt_pop8. A population size is a different
        # experiment, not a different seed of one, so pooling them under a
        # single `pbt/` would silently average two conditions -- the same reason
        # the gymnax sweep files each N under popsize_effect/pop_<N>/.
        #
        # Every member gets the SAME per-member budget as the single-policy PPO
        # run above, so N=8 spends 8x the environment steps. That is the
        # convention the gymnax_popsize_pbt block used (deleted 2026-09-13) (updates fixed across
        # N), and it is what makes the runs comparable to that sweep; the
        # trainer's `pop_env_steps` column is the number to divide by when the
        # question is what a population bought per unit of compute.
        #
        # weights_only, not the trainer's `full` default: explore perturbs
        # hyperparameters only, and brax fixes the learning rate inside the
        # optimizer, so `full` differs from `weights_only` by a no-op here. It
        # is also the mode the gymnax sweep runs, and the like-for-like one
        # against GA, which does not adapt hyperparameters either (docs/pbt.md).

        unset ENV_DIR_SUFFIX
    done
}

# ----------------------------------------------------------------------------
# Block: brax_noncontinual  (ant, the second continuous-control task)
# ----------------------------------------------------------------------------
#
# The same question as mujoco_noncontinual on a harder body: eight actuators and
# four legs instead of six actuators and two, and an episode that can terminate
# early when the ant flips, which CheetahRun never does.
#
# Ant is brax's `ant` rather than a mujoco_playground task. It is a separate
# block because the two suites share no trainer, and the ant trainers take flags
# (--sigma_schedule, --sigma_final) the cheetah ones do not have.
#
# Hyperparameters are NOT the cheetah ones. Ant is far more sensitive to
# parameter-space noise -- the GA sweep on this env settled on sigma 0.005
# against the cheetah's 0.1, and OpenES on sigma 0.02 with lr 0.01 -- so every
# NE default below comes from a sweep on ant and is overridable per method.
#
# Diversity tracking is on, as it is for gymnax: the ant descriptor is the duty
# factor of each of the four feet (source/common/behaviour_descriptors.py), the
# four-legged version of the cheetah's. It is an observer only.

block_brax_noncontinual() {
    local suite="brax" setting="noncontinual"
    local envs="${ENVS:-ant}"

    # The target-speed reward, so this block runs the SAME objective as
    # block_brax_continual. Off by default, which reproduces every existing
    # noncontinual ant run; set ANT_SPEED_TARGET to the continual block's
    # target and the two become comparable. Without it this block can only run
    # brax's unbounded forward-velocity reward while the continual block tracks
    # a speed -- different reward functions, so their returns are not on one
    # scale and calling this the control is wrong.
    local speed_target="${ANT_SPEED_TARGET:-}"
    local speed_weight="${ANT_SPEED_WEIGHT:-5.0}"
    local speed_args=""
    [ -n "$speed_target" ] && speed_args="--speed_target $speed_target --speed_weight $speed_weight"
    # Must match block_brax_continual's: this block is its control, so a
    # backend difference between them would read as a setting effect.
    local backend="${BACKEND:-mjx}"

    # Same budget convention as the other non-continual blocks: GENERATIONS
    # generations of NE_POP_SIZE individuals rolled out for NE_EPISODE_LENGTH
    # steps, and PPO gets the equivalent environment steps.
    local generations="${GENERATIONS:-500}"
    local ne_pop_size="${NE_POP_SIZE:-512}"
    local ne_episode_length="${NE_EPISODE_LENGTH:-1000}"

    # Pinned across methods: the trainers default to 1 (GA, ES) and 3 (DNS),
    # which would plot a 3-rollout mean against a single noisy rollout. 3 also
    # matters more on ant than on cheetah -- a single episode of a genome that
    # happens not to flip is badly optimistic.
    local ne_num_evals="${NE_NUM_EVALS:-3}"

    # num_evals is priced here, and the ant trainers already charge themselves
    # for it: steps_per_gen = pop_size * num_evals * episode_length in
    # train_{GA,ES,DNS}_ant.py. Without it in this product, PPO was given
    # 256M steps against the 768M the NE curves' own x-axis reported.
    local rl_timesteps="${RL_TIMESTEPS:-$((ne_pop_size * ne_num_evals * ne_episode_length * generations))}"

    # PPO's rollout shape, and it MUST be block_brax_continual's -- this block
    # is that one's control, so a rollout-shape difference between them reads
    # as a setting effect. See the long note there for why ant runs the
    # small-batch configuration rather than brax's tuned 4096/2048/4. The
    # variables are the same ANT_RL_* ones, so overriding one moves both blocks.
    local rl_ppo_args="--num_envs ${ANT_RL_NUM_ENVS:-512} \
                       --batch_size ${ANT_RL_BATCH_SIZE:-16} \
                       --num_minibatches ${ANT_RL_NUM_MINIBATCHES:-32} \
                       --num_updates_per_batch ${ANT_RL_UPDATES_PER_BATCH:-10} \
                       --unroll_length ${ANT_RL_UNROLL_LENGTH:-5}"

    # Per-method search hyperparameters, from the ant tuning sweep. Overridable
    # so a re-tune does not need this file edited.
    # 0.01, from the ant tuning sweep (its launcher has since been deleted;
    # the sweep's output is under projects/neurips_2026_rebuttal/tuning/):
    # over 60 generations at this population it reached 1253 against 199 for the
    # 0.005 inherited from the older pop-256 ant runs, and 0.02 collapses to 250.
    local ga_sigma="${ANT_GA_SIGMA:-0.01}"
    local ga_sigma_final="${ANT_GA_SIGMA_FINAL:-0.002}"
    local ga_elite_ratio="${ANT_GA_ELITE_RATIO:-0.1}"
    local es_sigma="${ANT_ES_SIGMA:-0.02}"
    local es_lr="${ANT_ES_LR:-0.01}"
    local dns_iso_sigma="${ANT_DNS_ISO_SIGMA:-0.005}"
    local dns_line_sigma="${ANT_DNS_LINE_SIGMA:-0.05}"
    local dns_k="${ANT_DNS_K:-3}"

    # DNS's offspring batch, and it MUST be passed rather than left to the
    # trainer, which defaults it to pop_size/2.
    #
    # DNS only rolls out the `batch_size` offspring each generation -- surviving
    # parents keep their stored fitness -- so
    # steps_per_gen = batch_size x num_evals x episode_length. At the trainer's
    # default that is 256 x 3 x 1000, half of GA's and ES's 512 x 3 x 1000, and
    # the block would put DNS on 384M env steps against 768M for every other
    # method in the same figure. Compute-matching is the one thing this block
    # exists to guarantee.
    #
    # The reported run is at 512 (its checkpoint records batch_size 512 and its
    # metrics end at 768,000,000), so this restores that run as well as matching
    # GA and ES. block_brax_continual has always passed it for the same reason;
    # only this block was left reading the default.
    local dns_batch_size="${ANT_DNS_BATCH_SIZE:-$ne_pop_size}"

    local diversity_args=""
    if [ "${TRACK_DIVERSITY:-1}" = "1" ]; then
        diversity_args="--track_diversity 1 \
            --diversity_interval ${DIVERSITY_INTERVAL:-10} \
            --occupancy_bins ${OCCUPANCY_BINS:-12} \
            --behaviour_snapshots ${BEHAVIOUR_SNAPSHOTS:-8}"
    else
        diversity_args="--track_diversity 0"
    fi

    echo "=========================================="
    echo "Block: brax_noncontinual"
    echo "  envs       : $envs"
    echo "  methods    : $METHODS"
    echo "  trials     : $NUM_TRIALS (base seed $BASE_SEED)"
    echo "  generations: $generations (pop $ne_pop_size x $ne_num_evals evals x $ne_episode_length steps)"
    echo "  PPO budget : $rl_timesteps env steps (matched, num_evals included)"
    echo "  NE network : (128, 128), hardcoded in the trainers"
    echo "  GA         : sigma $ga_sigma -> $ga_sigma_final, elite_ratio $ga_elite_ratio"
    echo "  ES         : sigma $es_sigma, lr $es_lr"
    echo "  DNS        : iso $dns_iso_sigma, line $dns_line_sigma, k $dns_k, batch $dns_batch_size"
    echo "  diversity  : ${TRACK_DIVERSITY:-1} (every ${DIVERSITY_INTERVAL:-10} gens, NE methods only)"
    echo "  backend    : $backend"
    echo "  GPU        : $GPU"
    echo "=========================================="

    for env in $envs; do
        if wants ga; then
            run_condition "$suite" "$setting" ga \
                source/studies/brax/train_GA_ant.py "$env" \
                --backend "$backend" \
                --num_generations "$generations" --pop_size "$ne_pop_size" \
                --num_evals "$ne_num_evals" \
                --episode_length "$ne_episode_length" \
                --sigma "$ga_sigma" --sigma_final "$ga_sigma_final" \
                --elite_ratio "$ga_elite_ratio" \
                $speed_args \
                $diversity_args
        fi
        if wants dns; then
            run_condition "$suite" "$setting" dns \
                source/studies/brax/train_DNS_ant.py "$env" \
                --backend "$backend" \
                --num_generations "$generations" --pop_size "$ne_pop_size" \
                --num_evals "$ne_num_evals" \
                --episode_length "$ne_episode_length" \
                --iso_sigma "$dns_iso_sigma" --line_sigma "$dns_line_sigma" \
                --k "$dns_k" --batch_size "$dns_batch_size" \
                $speed_args \
                $diversity_args
        fi
        if wants es; then
            run_condition "$suite" "$setting" es \
                source/studies/brax/train_ES_ant.py "$env" \
                --backend "$backend" \
                --num_generations "$generations" --pop_size "$ne_pop_size" \
                --num_evals "$ne_num_evals" \
                --episode_length "$ne_episode_length" \
                --sigma "$es_sigma" --learning_rate "$es_lr" \
                $speed_args \
                $diversity_args
        fi

        # PPO and its three continual-learning variants share the RL trainer and
        # the same env-step budget; they differ only in the flags below.
        if wants ppo; then
            run_condition "$suite" "$setting" ppo \
                source/studies/brax/train_RL_ant.py "$env" \
                --backend "$backend" \
                --num_timesteps "$rl_timesteps" \
                $speed_args \
                $rl_ppo_args \
                --episode_length "$ne_episode_length"
        fi
        if wants trac; then
            run_condition "$suite" "$setting" trac \
                source/studies/brax/train_RL_ant.py "$env" \
                --backend "$backend" \
                --num_timesteps "$rl_timesteps" \
                --episode_length "$ne_episode_length" \
                $speed_args \
                $rl_ppo_args \
                --use_trac
        fi
        if wants redo; then
            run_condition "$suite" "$setting" redo \
                source/studies/brax/train_RL_ant.py "$env" \
                --backend "$backend" \
                --num_timesteps "$rl_timesteps" \
                --episode_length "$ne_episode_length" \
                $speed_args \
                $rl_ppo_args \
                --use_redo --redo_frequency "${REDO_FREQUENCY:-1}"
        fi
        if wants cchain; then
            # Continuous-control defaults, the same ones the cheetah block
            # passes: ant's policy is a Normal, so the churn term is an MSE
            # between action means rather than a cross-entropy over logits.
            run_condition "$suite" "$setting" cchain \
                source/studies/brax/train_RL_ant.py "$env" \
                --backend "$backend" \
                --num_timesteps "$rl_timesteps" \
                --episode_length "$ne_episode_length" \
                --use_cchain \
                --chain_target_rel_scale "${MUJOCO_CHAIN_TARGET_REL_SCALE:-0.05}" \
                --chain_warmup_iterations "${MUJOCO_CHAIN_WARMUP_ITERATIONS:-50}" \
                $speed_args \
                $rl_ppo_args \
                --chain_coef_window "${MUJOCO_CHAIN_COEF_WINDOW:-100}"
        fi
    done
}


# ----------------------------------------------------------------------------
# Block: brax_continual  (ant, one damaged leg per sub-task)
# ----------------------------------------------------------------------------
#
# The ant counterpart of block_mujoco_continual. The non-stationarity is a
# different kind: cheetah's sub-tasks change a global friction multiplier, while
# ant's damage one leg -- its actuators are zeroed and its joints locked in
# place. It is a *structural* change to the robot rather than a change to the
# ground.
#
# The leg cycles 1 -> 2 -> 3 -> 4 -> 1 ..., deterministically and identically for
# every method and trial, so a GA run and a PPO run at the same trial face the
# same damage at the same point in their budget -- the guarantee the cheetah
# block gets from cycling its friction multipliers. NUM_TASKS defaults to 12,
# which is three whole cycles, so no leg is over-represented. The sequence, the
# damage wrapper lives in source/envs/brax_ant.py and the flags in source/studies/brax/cli.py,
# shared by all seven trainers; LEG_ORDER=random restores the sampled sequence
# the RL trainer used before 2026-07-29.
#
# Unlike the cheetah sequence there is no healthy sub-task: 12 sub-tasks over 4
# legs divides evenly only if every one of them damages a leg. The healthy ant
# is block_brax_noncontinual, which is the control this block is read against.
#
# Budget: sub-tasks are counted in generations so the numbers line up with the
# other blocks, and RL's per-sub-task step budget is the NE equivalent, exactly
# as block_mujoco_continual converts them.
#
# NE hyperparameters are block_brax_noncontinual's, for the same reason that
# block's comment gives: it is this block's control, so any drift between them
# would show up in the figures as a setting effect. The one deliberate
# difference is ANT_DNS_BATCH_SIZE (see below).

block_brax_continual() {
    local suite="brax"
    # Overridable so the two ant continual variants -- the friction sequence
    # and the target-speed sequence -- write to separate trees and can run at
    # the same time. run_condition builds $PROJECT_ROOT/$suite/$setting/..., so
    # sharing a setting name would have the second variant's trial_1 land on
    # top of the first's and, worse, be SKIPPED as already done.
    local setting="${BRAX_CONTINUAL_SETTING:-continual}"
    local envs="${ENVS:-ant}"

    # 24 x 16 IS THE PAPER VALUE, from the reported ant runs: 384 generations
    # over 24 sub-tasks, and PPO's config.json says num_tasks 24. This
    # defaulted to 12 x 100, which is neither the sequence nor the budget.
    local num_tasks="${NUM_TASKS:-24}"
    local task_interval="${TASK_INTERVAL:-16}"       # generations per sub-task
    local ne_pop_size="${NE_POP_SIZE:-512}"
    local ne_episode_length="${NE_EPISODE_LENGTH:-1000}"
    local leg_order="${LEG_ORDER:-cycle}"

    # Ground friction cycles default -> low -> high alongside the leg cycle, so a
    # sub-task perturbs the robot AND the ground. Leg damage on its own was not a
    # meaningful shift for ant: in the 2026-07-29 sweep PPO reached 4045 on the
    # damaged sequence against 4437 on the healthy noncontinual ant, and recovered
    # from each switch inside ~13% of the sub-task budget. FRICTION_ORDER=none
    # restores the damage-only sequence. The high multiplier is 3.0, not the
    # cheetah block's 5.0: brax's generalized solver goes non-finite on ant at
    # 5.0 -- see source/envs/brax_ant.py.
    local friction_order="${FRICTION_ORDER:-cycle}"
    local friction_default="${ANT_FRICTION_DEFAULT_MULT:-1.0}"
    local friction_low="${ANT_FRICTION_LOW_MULT:-0.2}"
    # 5.0 is the cheetah block's sticky value, reachable now that the ant runs
    # on mjx as well. It goes non-finite under BACKEND=generalized -- use 3.0
    # there, which is what the pre-2026-07-30 trees ran.
    local friction_high="${ANT_FRICTION_HIGH_MULT:-5.0}"
    local friction_args="--friction_order $friction_order \
                         --friction_default_mult $friction_default \
                         --friction_low_mult $friction_low \
                         --friction_high_mult $friction_high"

    # The observation-offset axis, the gymnax continual protocol on the ant.
    # Off by default (sigma 0). Sub-task 0 is always unperturbed; every later
    # sub-task adds a fixed offset vector to the observation, seeded off the
    # trial. Sigma needs per-body calibration -- see the gymnax README's
    # MountainCar story -- so probe with a bracket before committing a grid.
    local obs_noise_range="${OBS_NOISE_RANGE:-0.0}"
    # Sub-task revisits for the observation offset. Off by default, so every
    # existing ant tree is reproduced. The offset is the one axis here that is
    # drawn fresh per sub-task rather than cycled through a handful of values,
    # so it is the only one that never repeats on its own -- and therefore the
    # only one on which forgetting cannot be measured without this.
    local obs_task_period="${TASK_PERIOD:-0}"
    local obs_noise_args="--obs_noise_range $obs_noise_range"
    [ "$obs_task_period" -gt 0 ] && obs_noise_args="$obs_noise_args --task_period $obs_task_period"

    # The motor-flip axis. Off by default, so every existing sequence is
    # unchanged. FLIP_ORDER=cycle inverts one leg's motor polarity on every odd
    # sub-task (none, leg1, none, leg2, ...) -- the anti-optimality mechanism
    # for a radially symmetric body, where direction reversal is a no-op. Run
    # it on the healthy ant (LEG_ORDER=none FRICTION_ORDER=none) so the flip is
    # the only thing that changes.
    local flip_order="${FLIP_ORDER:-none}"
    local flip_args="--flip_order $flip_order"

    # The target-speed axis. Off by default, so the friction runs are unchanged.
    # SPEED_ORDER=cycle with FRICTION_ORDER=none gives the speed sequence: the
    # reward becomes -|v_x - target| instead of unbounded forward velocity, the
    # perturbation family C-CHAIN's Continual DMC benchmark uses (walker
    # stand/walk/run, quadruped walk/run/walk) and which MAML's Ant-Vel and
    # HalfCheetah-Vel use in exactly this form. Unlike friction and single-leg
    # damage, it is not something four legs can re-route around.
    local speed_order="${SPEED_ORDER:-none}"
    local speed_targets="${ANT_SPEED_TARGETS:-0.5,2.0}"
    local speed_weight="${ANT_SPEED_WEIGHT:-5.0}"
    local speed_args="--speed_order $speed_order --speed_targets $speed_targets \
                      --speed_weight $speed_weight"

    # The gravity axis. Off by default. Gravity rescales every contact force and
    # the body's weight at once, which four legs cannot re-route around the way
    # they do a tangential friction limit. It cycles with period 3, the SAME
    # period as friction, so running both would lock them in phase (every
    # slippery sub-task also heavy) and make their effects inseparable -- run
    # GRAVITY_ORDER=cycle with FRICTION_ORDER=none. Multipliers below 1.0 float
    # the torso into ant's healthy_z_range bound and end episodes early, so the
    # cycle is the heavy side only; see source/envs/brax_ant.py.
    local gravity_order="${GRAVITY_ORDER:-none}"
    local gravity_default="${ANT_GRAVITY_DEFAULT_MULT:-1.0}"
    local gravity_mid="${ANT_GRAVITY_MID_MULT:-2.0}"
    local gravity_high="${ANT_GRAVITY_HIGH_MULT:-4.0}"
    local gravity_args="--gravity_order $gravity_order \
                        --gravity_default_mult $gravity_default \
                        --gravity_mid_mult $gravity_mid \
                        --gravity_high_mult $gravity_high"

    # ---- PPO's rollout shape, and why ant runs the small-batch one ---------
    #
    # THE ANT'S PPO IS 512 ENVS / BATCH 16 / 10 UPDATES, NOT BRAX'S 4096 / 2048
    # / 4. Until this existed the block passed no rollout flags at all and took
    # the trainer's defaults, which are brax's tuned ant configuration -- and
    # that configuration solves this sequence rather than being challenged by
    # it: on the friction arm it ends sub-tasks at 4282 +/- 283 and finishes at
    # 4780, above the 4380 the *stationary* control reaches. A continual arm
    # that beats its own control is not measuring plasticity loss.
    #
    # The small-batch configuration is what every reported ant friction run
    # used (`runs/brax/continual_ant_friction{,_t24}/ppo_entropy`), and it ends
    # sub-tasks at 2336-2551 on the same friction band. It is not a handicap
    # invented here: at batch 16 with 10 updates per batch each sub-task's
    # gradient signal is small and repeatedly reused, which is the regime the
    # plasticity literature's loss-of-plasticity results are measured in.
    #
    # 512 = batch_size x num_minibatches is forced: ppo_continual_train.py
    # asserts `batch_size * num_minibatches % num_envs == 0`. Change one of the
    # three and the other two must follow.
    #
    # Gymnax is deliberately untouched -- its RL arms are tuned separately and
    # this is an ant-only change.
    local rl_num_envs="${ANT_RL_NUM_ENVS:-512}"
    local rl_batch_size="${ANT_RL_BATCH_SIZE:-16}"
    local rl_num_minibatches="${ANT_RL_NUM_MINIBATCHES:-32}"
    local rl_updates_per_batch="${ANT_RL_UPDATES_PER_BATCH:-10}"
    local rl_unroll_length="${ANT_RL_UNROLL_LENGTH:-5}"
    local rl_ppo_args="--num_envs $rl_num_envs \
                       --batch_size $rl_batch_size \
                       --num_minibatches $rl_num_minibatches \
                       --num_updates_per_batch $rl_updates_per_batch \
                       --unroll_length $rl_unroll_length"

    # Pinned across methods for the reason block_brax_noncontinual pins it: the
    # trainers would otherwise plot a 3-rollout mean against a single noisy one.
    # RL's per-task budget carries the same factor, so the two stay matched.
    local ne_num_evals="${NE_NUM_EVALS:-3}"
    local rl_timesteps_per_task="${RL_TIMESTEPS_PER_TASK:-$((ne_pop_size * ne_num_evals * ne_episode_length * task_interval))}"
    # Kept before the rounding below overwrites it, so the equality check
    # afterwards has the NE-equivalent target to compare against.
    local ne_equivalent_per_task=$((ne_pop_size * ne_num_evals * ne_episode_length * task_interval))

    # Round the per-task budget to something the trainer can actually hit.
    #
    # train_continual runs whole PPO training steps and whole evaluation epochs,
    # so a per-task budget that is not (training steps) x (steps per training
    # step), with the training steps divisible by num_evals_per_task, is rounded
    # *up*: the requested 51,200,000 became 65,536,000 -- 28% more environment
    # steps than the NE-equivalent budget this block exists to match, which
    # would quietly hand RL a bigger budget than the figure claims.
    #
    # ANT_RL_STEPS_PER_TRAIN_STEP is DERIVED, not a constant: ppo_continual_train.py
    # spends `batch_size x unroll_length x num_minibatches x action_repeat` env
    # steps per training step, so it follows the rollout shape above and cannot
    # go stale when that changes. It was hardcoded to 327,680 -- correct only
    # for brax's 2048/5/32 -- which under the small-batch shape's 2,560 would
    # have divided the budget by 128 and silently handed RL 128x less compute
    # than the NE methods it is plotted against. Override only to reproduce a
    # tree built before this was derived.
    local steps_per_train_step="${ANT_RL_STEPS_PER_TRAIN_STEP:-$((rl_batch_size * rl_unroll_length * rl_num_minibatches))}"
    # MUST DIVIDE the available train steps, or the round-down below throws away
    # most of the budget rather than trimming it.
    #
    # The NE-equivalent per-sub-task budget is 512 x 3 x 1000 x 16 =
    # 24,576,000, which is exactly 75 training steps of 327,680. This defaulted
    # to 52: 75 mod 52 = 23, so the round-down dropped 23 of the 75 steps and
    # handed RL 17,039,360 -- 69.3% of what GA, ES and DNS spend on the same
    # sub-task, or 408,944,640 against 589,824,000 over the 24-sub-task
    # sequence. The block's whole purpose is that those two numbers are equal,
    # and the comment above correctly rejects rounding UP by 28% while the code
    # rounded DOWN by 31%.
    #
    # 75 = 3 x 5 x 5, so the exact choices are 1, 3, 5, 15, 25 and 75. 25 is
    # taken: it divides exactly (ratio 1.000) and leaves 25 evaluation points
    # per sub-task, 600 over the sequence, which is finer than the NE curve's
    # 16 generations per sub-task and so does not limit any shared axis. Use 75
    # for one evaluation per training step, or 15 for cheaper logging; anything
    # that does not divide 75 silently costs budget again.
    #
    # That arithmetic is for the 327,680-step shape. Under the small-batch
    # default the step is 16 x 5 x 32 = 2,560 and the same budget is 9,600
    # training steps, which 25 also divides exactly (384 steps between
    # evaluations) -- so the ratio stays 1.000 and the evaluation grid stays at
    # 25 points per sub-task. The check below is what verifies this rather than
    # this comment; read its output, not this paragraph, after changing shape.
    local evals_per_task="${ANT_RL_EVALS_PER_TASK:-25}"
    local train_steps=$((rl_timesteps_per_task / steps_per_train_step))
    train_steps=$((train_steps - train_steps % evals_per_task))
    if [ "$train_steps" -lt "$evals_per_task" ]; then
        echo "ERROR: per-task budget $rl_timesteps_per_task is too small for" >&2
        echo "       $evals_per_task evaluations; lower ANT_RL_EVALS_PER_TASK." >&2
        exit 1
    fi
    rl_timesteps_per_task=$((train_steps * steps_per_train_step))
    # Loud, because this is the one place RL's budget can silently stop matching
    # NE's. Anything below 100% is RL being handed less compute than the methods
    # it is plotted against.
    if [ "$rl_timesteps_per_task" -ne "$ne_equivalent_per_task" ]; then
        echo "WARNING: RL per-sub-task budget $rl_timesteps_per_task != NE-equivalent" >&2
        echo "         $ne_equivalent_per_task ($(( 100 * rl_timesteps_per_task / ne_equivalent_per_task ))%)." >&2
        echo "         Pick an ANT_RL_EVALS_PER_TASK that divides" >&2
        echo "         $((ne_equivalent_per_task / steps_per_train_step)) exactly." >&2
    fi

    # The non-continual ant RL runs build their networks at (256, 256), the brax
    # default this trainer also defaults to. Pinned explicitly so the continual
    # and non-continual ant results stay comparable if either default moves.
    # Matched to the NE policy, which is (128, 128) hardcoded in the NE
    # trainers. At 256,256 the PPO policy carried 77,072 parameters against
    # the NE policy's 21,128 -- 3.65x -- plus a 73,217-parameter value net;
    # at 128,128 it is 22,160, a 1.05x match. Capacity is not a neutral knob
    # in this block: loss of plasticity is weaker at high capacity, so the
    # unmatched setting both favoured PPO and worked against the effect this
    # block exists to measure. The value net has no NE counterpart and is an
    # inherent cost of the method, so it is sized with the policy, not
    # against it.
    local rl_hidden_sizes="${ANT_RL_HIDDEN_SIZES:-128,128}"

    local trainer=source/studies/brax/train_RL_ant_continual_legs.py
    # Plasticity diagnostics. The trainer has carried --track_dormant since it
    # was written and this runner never passed it, so every ant continual run
    # before 2026-07-30 measured no plasticity at all. On by default now: the
    # dormant fraction is the mechanism this block claims to be about, and it
    # costs one extra forward pass per evaluation.
    local rl_diagnostic_args=""
    if [ "${TRACK_DORMANT:-1}" = "1" ]; then
        rl_diagnostic_args="--track_dormant --dormant_tau ${DORMANT_TAU:-0.025}"
    fi

    # Which simulator. 'mjx' is the trainers' own default and is what the
    # cheetah block runs on; BACKEND=generalized reproduces every ant tree dated
    # on or before 2026-07-30. Passed explicitly rather than left to the default
    # so the choice is visible in the launcher's echo and in nohup logs -- it
    # changes what the numbers mean, and a silent default would not show up in
    # either. See source/envs/brax_ant.py.
    local backend="${BACKEND:-mjx}"

    # Observation normalization, RL only -- the NE trainers have no normalizer
    # at all. 'true' is the trainer's own default and reproduces every ant tree
    # on disk; it is passed explicitly for the reason $backend is, because it
    # changes what the numbers mean and a silent default shows up in neither
    # the launcher echo nor the nohup log.
    #
    # It is a knob rather than a constant because on the obs-offset task the
    # normalizer is a confound, not a neutral preprocessing step. brax's
    # running statistics are part of the preserved training state, so the
    # sample count accumulates across sub-tasks (24.6M -> 295M over 12) and
    # updates decay as 1/count. Each sub-task's fixed offset is therefore
    # pooled into one estimate rather than tracked: measured over 27 dims, the
    # mean stays put (the offsets are zero-mean across sub-tasks and cancel,
    # |mean| 0.23 -> 0.48) while the variance takes the full hit -- 2.06 ->
    # 5.60, against a law-of-total-variance prediction of +sigma^2 x 11/12 =
    # +3.67. The network is then handed the current offset uncorrected AND a
    # signal divided by the pooled spread. NE runs with no such handicap, so
    # any RL-vs-NE plasticity gap on that task is partly this.
    #
    # RL_NORMALIZE_OBS=false is the control. Do NOT "fix" this by resetting the
    # normalizer per sub-task instead: the perturbation is a constant additive
    # offset, which is exactly what mean-subtraction removes, so a freshly
    # fitted normalizer would cancel the manipulation and report a retention
    # number that means nothing.
    local normalize_obs="${RL_NORMALIZE_OBS:-true}"

    local task_args="--num_tasks $num_tasks \
                     --backend $backend \
                     --leg_order $leg_order \
                     $friction_args \
                     $flip_args \
                     $obs_noise_args \
                     $speed_args \
                     $gravity_args \
                     --normalize_observations $normalize_obs \
                     --timesteps_per_task $rl_timesteps_per_task \
                     --episode_length $ne_episode_length \
                     --num_evals_per_task $evals_per_task \
                     --policy_hidden_sizes $rl_hidden_sizes \
                     --value_hidden_sizes $rl_hidden_sizes \
                     $rl_ppo_args \
                     $rl_diagnostic_args"

    # ---- NE side -----------------------------------------------------------
    # Per-method search hyperparameters, the same ANT_* variables
    # block_brax_noncontinual reads, so overriding one moves both blocks.
    local ga_sigma="${ANT_GA_SIGMA:-0.01}"
    local ga_sigma_final="${ANT_GA_SIGMA_FINAL:-0.002}"
    local ga_elite_ratio="${ANT_GA_ELITE_RATIO:-0.1}"
    # Which window sigma decays over. The noncontinual block has no sub-tasks and
    # has no such flag; here it is the difference between every sub-task exploring
    # less than the last and each getting the same budget. Selected by
    # scripts/neurips_2026_rebuttal/tune_ne_ant_continual.sh.
    # 'sequence' after tune_ne_ant_continual.sh: all four arms measured, and the
    # ORIGINAL schedule won on retention (last/1st 0.57) against per_task 0.49,
    # const_0.01 0.04 and per_task at sigma 0.02 0.05. The two large-sigma arms
    # collapse to ~50 reward by sub-task 3 and never recover. The monotone decay
    # is what keeps the GA alive: a small sigma preserves the converged solution
    # instead of shredding it. The hypothesis that the decay caused GA's fall
    # across sub-tasks was wrong, and the flag is kept only because it is what
    # let that be measured.
    local ga_sigma_schedule="${ANT_GA_SIGMA_SCHEDULE:-sequence}"
    local es_sigma="${ANT_ES_SIGMA:-0.02}"
    # From tune_ne_ant_continual.sh. lr is the whole story: both lr 0.01 arms
    # collapse (retention 0.01 and 0.00, the distribution mean reaching NEGATIVE
    # reward), both lr 0.005 arms survive. Between those two, per_task sigma
    # decay posts a higher mean (2826 vs 2435) but degrades across the sequence
    # (retention 0.55 vs 1.09); constant sigma is promoted because retention is
    # the quantity this block is about. sigma_final == sigma makes the schedule
    # constant, which is what the trainer does when the two are equal.
    local es_sigma_final="${ANT_ES_SIGMA_FINAL:-$es_sigma}"
    local es_sigma_schedule="${ANT_ES_SIGMA_SCHEDULE:-sequence}"
    local es_lr="${ANT_ES_LR:-0.005}"
    local dns_iso_sigma="${ANT_DNS_ISO_SIGMA:-0.005}"
    local dns_line_sigma="${ANT_DNS_LINE_SIGMA:-0.05}"
    local dns_k="${ANT_DNS_K:-3}"

    # The one deliberate departure from block_brax_noncontinual, and the same
    # choice the cheetah blocks make: DNS's offspring batch is the full
    # population rather than the trainer's pop_size/2, so GA, ES and DNS all
    # spend the same env steps per generation and share one honest generation
    # axis. block_brax_noncontinual leaves it at pop_size/2, so ant DNS
    # noncontinual costs half as much per generation as ant DNS continual --
    # re-run it with ANT_DNS_BATCH_SIZE=$ne_pop_size if the two need to sit on
    # one env-step axis.
    local dns_batch_size="${ANT_DNS_BATCH_SIZE:-$ne_pop_size}"

    local ne_task_args="--num_tasks $num_tasks \
                        --backend $backend \
                        --leg_order $leg_order \
                        $friction_args \
                        $flip_args \
                        $obs_noise_args \
                        $speed_args \
                        $gravity_args \
                        --gens_per_task $task_interval \
                        --pop_size $ne_pop_size \
                        --num_evals $ne_num_evals \
                        --episode_length $ne_episode_length"

    local ne_diversity_args=""
    if [ "${TRACK_DIVERSITY:-1}" = "1" ]; then
        ne_diversity_args="--track_diversity 1 \
            --diversity_interval ${DIVERSITY_INTERVAL:-10} \
            --occupancy_bins ${OCCUPANCY_BINS:-12} \
            --behaviour_snapshots ${BEHAVIOUR_SNAPSHOTS:-8}"
    else
        ne_diversity_args="--track_diversity 0"
    fi

    # GIFs of each sub-task's best genome, as the cheetah continual block does.
    # Passed to the RL conditions as well as the NE ones: since 2026-07-30
    # train_RL_ant_continual_legs.py renders the policy at each sub-task
    # boundary into the same gifs/task_NN_<label>/ layout the NE trainers
    # use, so an RL run and a GA run can be flipped through side by side.
    local gif_args="--gifs_per_task ${GIFS_PER_TASK:-3}"

    echo "=========================================="
    echo "Block: brax_continual"
    echo "  envs       : $envs"
    echo "  methods    : $METHODS"
    echo "  trials     : $NUM_TRIALS (base seed $BASE_SEED)"
    echo "  tasks      : $num_tasks sub-tasks x $task_interval generations,"
    echo "               one damaged leg each, order '$leg_order'"
    echo "  backend    : $backend"
    echo "  friction   : $friction_order ($friction_default / $friction_low / $friction_high)"
    echo "  motor flip : $flip_order"
    echo "  obs offset : sigma $obs_noise_range, revisit period $obs_task_period"
    echo "  RL obs norm: $normalize_obs (NE has no normalizer)"
    echo "  speed      : $speed_order ($speed_targets)"
    echo "  gravity    : $gravity_order ($gravity_default / $gravity_mid / $gravity_high)"
    echo "  output tree: $suite/$setting"
    echo "  NE budget  : pop $ne_pop_size x $ne_num_evals evals x $ne_episode_length steps"
    echo "  NE network : (128, 128), hardcoded in the trainers"
    echo "  GA         : sigma $ga_sigma -> $ga_sigma_final ($ga_sigma_schedule), elite_ratio $ga_elite_ratio"
    echo "  ES         : sigma $es_sigma -> $es_sigma_final ($es_sigma_schedule), lr $es_lr"
    echo "  DNS        : iso $dns_iso_sigma, line $dns_line_sigma, k $dns_k, batch $dns_batch_size"
    echo "  RL budget  : $rl_timesteps_per_task env steps per sub-task"
    echo "               ($train_steps training steps x $evals_per_task evals)"
    echo "  RL network : $rl_hidden_sizes (matched to the NE policy)"
    echo "  dormant    : ${TRACK_DORMANT:-1} (tau ${DORMANT_TAU:-0.025})"
    echo "  diversity  : ${TRACK_DIVERSITY:-1} (every ${DIVERSITY_INTERVAL:-10} gens, NE methods only)"
    echo "  GPU        : $GPU"
    echo "=========================================="

    for env in $envs; do
        if wants ga; then
            run_condition "$suite" "$setting" ga \
                source/studies/brax/train_GA_ant_continual_legs.py "$env" \
                $ne_task_args \
                --sigma "$ga_sigma" --sigma_final "$ga_sigma_final" \
                --sigma_schedule "$ga_sigma_schedule" \
                --elite_ratio "$ga_elite_ratio" \
                $gif_args $ne_diversity_args
        fi
        if wants dns; then
            run_condition "$suite" "$setting" dns \
                source/studies/brax/train_DNS_ant_continual_legs.py "$env" \
                $ne_task_args \
                --iso_sigma "$dns_iso_sigma" --line_sigma "$dns_line_sigma" \
                --k "$dns_k" --batch_size "$dns_batch_size" \
                $gif_args $ne_diversity_args
        fi
        if wants es; then
            run_condition "$suite" "$setting" es \
                source/studies/brax/train_ES_ant_continual_legs.py "$env" \
                $ne_task_args \
                --sigma "$es_sigma" --sigma_final "$es_sigma_final" \
                --sigma_schedule "$es_sigma_schedule" \
                --learning_rate "$es_lr" \
                $gif_args $ne_diversity_args
        fi

        if wants ppo; then
            run_condition "$suite" "$setting" ppo "$trainer" "$env" $task_args $gif_args
        fi
        if wants trac; then
            run_condition "$suite" "$setting" trac "$trainer" "$env" $task_args $gif_args --use_trac
        fi
        if wants redo; then
            # redo_frequency 1 matches the other continuous-control blocks.
            run_condition "$suite" "$setting" redo "$trainer" "$env" $task_args $gif_args \
                --use_redo --redo_frequency "${REDO_FREQUENCY:-1}"
        fi
        if wants cchain; then
            # Continuous-control defaults, deliberately NOT the gymnax values.
            # The coefficient is reset at every leg change, which is the point
            # of the baseline here.
            run_condition "$suite" "$setting" cchain "$trainer" "$env" $task_args $gif_args \
                --use_cchain \
                --chain_target_rel_scale "${MUJOCO_CHAIN_TARGET_REL_SCALE:-0.05}" \
                --chain_warmup_iterations "${MUJOCO_CHAIN_WARMUP_ITERATIONS:-50}" \
                --chain_coef_window "${MUJOCO_CHAIN_COEF_WINDOW:-100}"
        fi

        # PBT-PPO on ant, one directory per population size, exactly as
        # block_mujoco_continual files pbt_pop2 / pbt_pop8: a population size is
        # a different experiment, not a different seed of one.
        #
        # source/studies/brax/train_PBT_ant_continual.py landed in 8849429 but nothing
        # called it, so the ant population arm could not be run from this file
        # at all. It takes a DIFFERENT argument set from the other six methods
        # here -- `--target_speed` (one fixed float) and `--obs_noise_range`,
        # with no --leg_order/--friction_order/--speed_order -- because it is
        # purpose-built for the obs-noise sequence at a constant target speed,
        # which is the only ant task the population arm is defined on. So it
        # cannot take $task_args and is spelled out instead.
        #
        # ANT_PBT_TARGET_SPEED is the constant the sequence runs at; it
        # defaults to the FIRST entry of ANT_SPEED_TARGETS so it matches the
        # NE/RL runs it is compared against by construction. Set it explicitly
        # if those runs cycle more than one target.
    done
}


block_cheetah_popsize() {
    # The cheetah counterpart of block_gymnax_popsize: the same 30-sub-task
    # continual sequence the cheetah continual block runs, at a range of
    # population sizes, so N can be varied on a continuous-control task and not
    # only on the gymnax ones.
    #
    # A separate block rather than a POP_SIZES loop inside
    # block_mujoco_continual, for the reason gymnax has two: the continual block
    # writes `continual/` and this one writes `popsize_effect/pop_<N>/`, and the
    # continual block also drives the RL methods and PBT, none of which belong on
    # this axis.
    #
    # **N=512 is not run here.** block_mujoco_continual runs exactly this
    # experiment at pop 512 -- same 30 sub-tasks of 100 generations, same evals,
    # same episode length, same friction cycle -- so those runs ARE the top of
    # this axis, which is why POP_SIZES stops at 128. Same substitution as the
    # gymnax NE sweep's, and nothing is substituted but the directory.
    local suite="mujoco"
    local envs="${ENVS:-CheetahRun}"
    local task_mod="${TASK_MOD:-friction}"

    # The t30 sequence, matching the continual block's `_t30` runs that anchor
    # this axis. 30 x 100 = 3000 generations.
    local num_tasks="${NUM_TASKS:-30}"
    local task_interval="${TASK_INTERVAL:-100}"
    local generations=$((num_tasks * task_interval))
    local pop_sizes="${POP_SIZES:-2 8 32 128}"
    local ne_episode_length="${NE_EPISODE_LENGTH:-1000}"
    local ne_num_evals="${NE_NUM_EVALS:-3}"

    local mult_default="${FRICTION_DEFAULT_MULT:-1.0}"
    local mult_low="${FRICTION_LOW_MULT:-0.2}"
    local mult_high="${FRICTION_HIGH_MULT:-5.0}"

    # Behaviour tracking ON by default, unlike block_gymnax_popsize.
    #
    # That block defaults it off because its runs were made without it and
    # switching it on silently would have produced a half-tracked grid. This
    # sweep has no runs yet, so there is nothing to be inconsistent with -- and
    # the half-tracked gymnax grid cost a 182-trial re-run
    # (rerun_gymnax_popsize_tracking.sh) that this default is what avoids.
    # Population diversity cannot be recovered post hoc: checkpoints hold one
    # agent per sub-task, not the population it came from.
    local diversity_args=""
    if [ "${TRACK_DIVERSITY:-1}" = "1" ]; then
        diversity_args="--track_diversity 1 \
            --diversity_interval ${DIVERSITY_INTERVAL:-10} \
            --occupancy_bins ${OCCUPANCY_BINS:-12} \
            --behaviour_snapshots ${BEHAVIOUR_SNAPSHOTS:-4}"
    else
        diversity_args="--track_diversity 0"
    fi

    local task_args="--num_tasks $num_tasks --task_mod $task_mod \
        --friction_default_mult $mult_default \
        --friction_low_mult $mult_low --friction_high_mult $mult_high"

    local dns_dir="${DNS_DIR:-dns}"

    # No GIFs. They cost rollouts at every one of 30 sub-task boundaries and feed
    # no metric on this axis; block_gymnax_popsize drops them for the same reason.
    local ne_gif_args="--gifs_per_task ${GIFS_PER_TASK:-0}"

    echo "=========================================="
    echo "Block: cheetah_popsize (continual, 30 sub-tasks)"
    echo "  envs       : $envs"
    echo "  methods    : $METHODS"
    echo "  pop sizes  : $pop_sizes  (512 = the continual block's own runs)"
    echo "  trials     : $NUM_TRIALS (base seed $BASE_SEED)"
    echo "  tasks      : $num_tasks x $task_interval gens = $generations generations"
    echo "  task var   : $task_mod, cycling $mult_default -> $mult_low -> $mult_high"
    echo "  NE budget  : pop <N> x $ne_num_evals evals x $ne_episode_length steps"
    echo "  diversity  : ${TRACK_DIVERSITY:-1} (every ${DIVERSITY_INTERVAL:-10} gens)"
    echo "  GPU        : $GPU"
    echo "=========================================="

    for pop in $pop_sizes; do
        # Same layout as the gymnax sweep -- population size in the method name:
        # `continual/ga_pop8/` rather than `popsize_effect/pop_8/ga/`. That is what
        # `compare.py` reads (popsize_method()) and what the run tree holds, and it
        # is what lets the sweep's largest N BE the continual block's own run
        # instead of a second copy of it filed under another path.
        local setting="continual"

        for env in $envs; do
            # `_t30` for the reason block_mujoco_continual tags it: a
            # 12-sub-task sequence and a 30-sub-task one are different
            # experiments and must not pool. It also has to match the anchor's
            # directory name, which carries it.
            ENV_DIR_SUFFIX="_${task_mod}${ENV_DIR_EXTRA:-_t30}"

            if wants ga; then
                run_condition "$suite" "$setting" ga_pop${pop} \
                    source/studies/mujoco/train_GA_cheetah_continual.py "$env" \
                    $task_args \
                    --gens_per_task "$task_interval" --pop_size "$pop" \
                    --num_evals "$ne_num_evals" \
                    --episode_length "$ne_episode_length" \
                    $diversity_args $ne_gif_args
            fi
            if wants dns; then
                run_condition "$suite" "$setting" "${dns_dir}_pop${pop}" \
                    source/studies/mujoco/train_DNS_cheetah_continual.py "$env" \
                    $task_args \
                    --gens_per_task "$task_interval" --pop_size "$pop" \
                    --num_evals "$ne_num_evals" \
                    --episode_length "$ne_episode_length" \
                    --descriptor ${DNS_DESCRIPTOR:-aurora} \
                    --batch_size ${DNS_BATCH_SIZE:-$pop} \
                    --iso_sigma ${DNS_ISO_SIGMA:-0.005} \
                    --line_sigma ${DNS_LINE_SIGMA:-0.05} \
                    --reeval_survivors ${DNS_REEVAL_SURVIVORS:-0} \
                    --normalize_descriptors ${DNS_NORMALIZE_DESCRIPTORS:-0} \
                    --aurora_bounded_descriptor ${DNS_BOUNDED_DESCRIPTOR:-0} \
                    $diversity_args $ne_gif_args
            fi
            if wants es; then
                run_condition "$suite" "$setting" es_pop${pop} \
                    source/studies/mujoco/train_ES_cheetah_continual.py "$env" \
                    $task_args \
                    --gens_per_task "$task_interval" --pop_size "$pop" \
                    --num_evals "$ne_num_evals" \
                    --episode_length "$ne_episode_length" \
                    $diversity_args $ne_gif_args
            fi
        done
    done
}


# ----------------------------------------------------------------------------
# Dispatch
# ----------------------------------------------------------------------------


# ----------------------------------------------------------------------------
# Block: minigrid_continual / minigrid_noncontinual
# ----------------------------------------------------------------------------
#
# The MiniGrid body (`source/envs/minigrid.py`), as its own suite in the run
# tree. A sub-task here is WHICH ROOM -- EmptyRandom-8x8 against
# EmptyRandom-16x16 -- rather than an observation offset, so there is no
# NOISE_RANGE and no SIGMAS: `ENV_DIR_SUFFIX` has nothing to tag and the cell
# names carry the pair instead.
#
# WHY THESE BLOCKS ARE FIVE LINES WHEN THE GYMNAX ONES ARE TWO HUNDRED.
# Everything the gymnax blocks spend their length on -- pinning each method's
# population, evals, network and budget so the arms are compute-matched -- is
# in `source/studies/minigrid/settings.py` instead, next to the arm
# definitions it has to agree with, where a mismatch is a failed assertion
# (`settings.check()`) rather than a number that drifted between two shell
# variables. There is one entry point for every arm because there is one
# implementation of every method; see that file's header.
#
# The cells:
#     minigrid_continual     MiniGrid_8x8_16x16   the rooms alternate every
#                            phase, twenty phases, so each is revisited ten
#                            times and a revisit measures retention.
#     minigrid_noncontinual  MiniGrid_8x8, MiniGrid_16x16   the stationary
#                            controls, same budget, same twenty checkpoints,
#                            nothing changing at them.
#
# Env: ENVS (the cells), TRIALS/NUM_TRIALS, PROJECT_ROOT, GPU, and
# MINIGRID_EXTRA for anything to pass straight through to the CLI.
# ----------------------------------------------------------------------------
# Blocks: ant_* / cheetah_*   (the ICLR mjx study, both continuous bodies)
# ----------------------------------------------------------------------------
#
# The MiniGrid arrangement on the two MJX bodies: `source/studies/mjx/cli.py`
# is argument plumbing over the two suite-generic runners under
# `source/studies/generalists/`, and `settings.py` is what an arm is. There is
# no ant trainer and no cheetah trainer, which is the point -- CLAUDE.md (a).
# The four blocks below differ only in which cells they default to.
#
# These are NOT block_brax_{non,}continual or block_mujoco_{non,}continual.
# Those drive the per-method trainers under `source/studies/brax/` and
# `source/studies/mujoco/`, which produced `runs_repro2`, cannot save a
# centroid and have no gaussian DNS arm. The brax ones still run; THE MUJOCO
# ONES DO NOT -- `source/studies/mujoco/` and `source/envs/mjx_cheetah.py`
# were deleted on 2026-09-08, so block_mujoco_* points at files that are not
# on this checkout and will fail on its first job. Nothing new should use
# either.
#
# A cell is a body plus a KIND of sub-task (`<body>_noise`, `<body>_friction`)
# or that body's stationary control (`<body>`), and every budget number lives
# in settings.py, which asserts the NE and RL halves are matched at the start
# of every trial. The BODY comes from the cell, so nothing here names one.

block_mjx() {
    local setting="$1" envs="$2"
    local script="source/studies/mjx/cli.py"

    # The eight arms the ICLR grid settled on. `ga` and `dns_gaussian` are the
    # GAUSSIAN pair -- same mutation, different selection rule -- and the
    # Iso+LineDD arms (`ga_isoline`, `dns`) are the operator ablation, which is
    # not in the paper and is not run here.
    for method in ${METHOD_ORDER:-ga es nes dns_gaussian ppo trac redo cchain pbt pbt2 pbt_weights pbt2_weights}; do
        wants "$method" || continue
        for env in $envs; do
            run_condition "mjx" "$setting" "$method" "$script" "$env" \
                --method "$method" ${MJX_EXTRA:-${ANT_EXTRA:-}}
        done
    done
}

block_ant_continual() {
    block_mjx continual "${ENVS:-ant_noise ant_friction}"
}

block_ant_noncontinual() {
    # One stationary cell serves both families: sub-task 0 is the unperturbed
    # body under either task_mod, so the two would be the same run twice. See
    # source/studies/mjx/settings.py.
    block_mjx noncontinual "${ENVS:-ant}"
}

block_cheetah_continual() {
    block_mjx continual "${ENVS:-cheetah_noise cheetah_friction}"
}

block_cheetah_noncontinual() {
    block_mjx noncontinual "${ENVS:-cheetah}"
}


block_minigrid() {
    local setting="$1" envs="$2"
    local script="source/studies/minigrid/cli.py"

    # The eight arms the ICLR grid settled on. `ga` and `dns_gaussian` are the
    # GAUSSIAN pair -- same mutation, different selection rule -- and the
    # Iso+LineDD arms (`ga_isoline`, `dns`) are the operator ablation, which is
    # not in the paper and is not run here.
    for method in ${METHOD_ORDER:-ga es nes dns_gaussian ppo trac redo cchain pbt pbt2 pbt_weights pbt2_weights}; do
        wants "$method" || continue
        for env in $envs; do
            run_condition "minigrid" "$setting" "$method" "$script" "$env" \
                --method "$method" ${MINIGRID_EXTRA:-}
        done
    done

    # HYPERPARAMETER ARMS, as in block_gymnax_continual: `ga_sigma0.03` is
    # the `ga` arm with settings.NE_ARMS['ga']['sigma'] replaced by 0.03
    # (--ne_override), `nes_sigma0.3` the same for `nes`. The directory is
    # the arm name, so the value is what the run is; matched on the literal
    # name, never through `all`.
    local hp_arm hp_base
    for hp_arm in $METHODS; do
        case "$hp_arm" in
            ga_sigma?*|nes_sigma?*)
                hp_base="${hp_arm%%_sigma*}"
                for env in $envs; do
                    run_condition "minigrid" "$setting" "$hp_arm" "$script" "$env" \
                        --method "$hp_base" --ne_override "sigma=${hp_arm#*_sigma}" \
                        ${MINIGRID_EXTRA:-}
                done ;;
        esac
    done
}

block_minigrid_continual() {
    block_minigrid continual "${ENVS:-MiniGrid_8x8_16x16}"
}

block_minigrid_noncontinual() {
    block_minigrid noncontinual "${ENVS:-MiniGrid_8x8 MiniGrid_16x16}"
}

# ----------------------------------------------------------------------------
# Block: kinetix  (the twenty hand-designed levels; a sub-task is WHICH LEVEL)
# ----------------------------------------------------------------------------
#
# `source/studies/kinetix/cli.py` is the whole of the difference between a
# stationary cell and the continual chain: both go through the same two
# suite-generic runners, and the cell name decides the schedule. The eighteen
# old trainers under source/studies/kinetix/ are the PREVIOUS codebase and are
# not run from here.

block_kinetix() {
    local setting="$1" envs="$2"
    local script="source/studies/kinetix/cli.py"

    for method in ${METHOD_ORDER:-ga es nes dns dns_gaussian ppo trac redo cchain pbt pbt2 pbt_weights pbt2_weights}; do
        wants "$method" || continue
        for env in $envs; do
            run_condition "kinetix" "$setting" "$method" "$script" "$env" \
                --method "$method" ${KINETIX_EXTRA:-}
        done
    done
}

# Every cell name comes from the suite's own table -- twenty `Kinetix-<level>`
# and one `Kinetix20` -- so a level renamed there cannot leave a stale name here.
kinetix_cells() {
    "${PYTHON:-python}" -c \
        "from source.envs.kinetix_levels import LEVELS, cell_for; print(' '.join(cell_for(l) for l in LEVELS))"
}

block_kinetix_noncontinual() {
    block_kinetix noncontinual "${ENVS:-$(kinetix_cells)}"
}

block_kinetix_continual() {
    block_kinetix continual "${ENVS:-Kinetix20}"
}

case "$BLOCK" in
    ant_noncontinual)    block_ant_noncontinual ;;
    ant_continual)       block_ant_continual ;;
    cheetah_noncontinual) block_cheetah_noncontinual ;;
    cheetah_continual)   block_cheetah_continual ;;
    gymnax_noncontinual) block_gymnax_noncontinual ;;
    gymnax_continual)    block_gymnax_continual ;;
    gymnax_popsize)      block_gymnax_popsize ;;
    mujoco_noncontinual) block_mujoco_noncontinual ;;
    mujoco_continual)    block_mujoco_continual ;;
    cheetah_popsize)     block_cheetah_popsize ;;
    brax_noncontinual)   block_brax_noncontinual ;;
    brax_continual)      block_brax_continual ;;
    minigrid_continual)    block_minigrid_continual ;;
    minigrid_noncontinual) block_minigrid_noncontinual ;;
    kinetix_noncontinual)  block_kinetix_noncontinual ;;
    kinetix_continual)     block_kinetix_continual ;;
    *)
        echo "Unknown block: $BLOCK"
        exit 1
        ;;
esac

echo ""
echo "=========================================="
if [ "${#FAILED_TRIALS[@]}" -eq 0 ]; then
    echo "Block '$BLOCK' complete."
    echo "=========================================="
else
    echo "Block '$BLOCK' finished with ${#FAILED_TRIALS[@]} failed trial(s):"
    for t in "${FAILED_TRIALS[@]}"; do
        echo "  - $t"
    done
    echo ""
    echo "Re-running the block retries exactly these: a trial is skipped only"
    echo "once its training_metrics.json exists."
    echo "=========================================="
    exit 1
fi
