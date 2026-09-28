#!/bin/bash
# ============================================================================
# Run experiments: one or more settings, over as many GPUs as you give it.
#
#   bash scripts/train/run.sh [options] SETTING[:METHODS] [SETTING[:METHODS] ...]
#
# Examples
#   bash scripts/train/run.sh --gpus 4 gymnax_continual
#   bash scripts/train/run.sh --gpus 2 gymnax_noncontinual gymnax_continual:ga,ppo
#   bash scripts/train/run.sh --gpus 8 all
#   bash scripts/train/run.sh --dry-run --trials 1 minigrid_continual:es
#
# SETTINGS
#   gymnax_noncontinual   gymnax_continual   gymnax_popsize
#   minigrid_noncontinual minigrid_continual
#   kinetix_noncontinual  kinetix_continual
#   cheetah_noncontinual  cheetah_continual
#   all                   every setting above except gymnax_popsize, in that
#                         order
#
# METHODS (after the colon, comma-separated)
#   Default: the methods the paper reports for that benchmark, read from
#   reported_arms in source/configs/<benchmark>.yaml. `all` is every method the
#   setting defines. Anything a config knows is accepted, including the
#   hyperparameter arms (`es_sigma0.3`, `ppo_lr0.001`, ...; see the gymnax
#   blocks below) and `<method>_pop<N>` under gymnax_popsize.
#
# OPTIONS
#   --gpus N          use N GPUs (the first N visible). Default: all visible.
#   --gpu-ids 0,2,5   use exactly these GPUs.
#   --jobs-per-gpu K  run K jobs on each GPU at once (default 1; see below).
#   --trials N        trials per (method, cell), seeds BASE_SEED..+N-1 (default 10)
#   --root DIR        where runs are written (default $PROJECT_ROOT)
#   --dry-run         print the job list and exit
#
# Runs land in <root>/<benchmark>/<setting>/<method>/<cell>/trial_<n>/, and
# each job's log in $LOG_DIR (default logs/run). A trial whose
# training_metrics.json exists is skipped, so re-running the same command
# runs exactly what is missing or failed. Every setting's jobs are queued in
# the order the settings are given; one job runs per GPU at a time.
#
# More knobs, as environment variables: BASE_SEED (42), TRIALS ("3 4": only
# these trials), ENVS (a setting's cells), PYTHON, and each block's own
# (documented at the block).
#
# ============================================================================

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$SCRIPT_DIR"
while [ ! -f "$REPO_ROOT/pyproject.toml" ] && [ "$REPO_ROOT" != "/" ]; do
    REPO_ROOT="$(dirname "$REPO_ROOT")"
done
cd "$REPO_ROOT"

# Line-buffer the trainers' stdout, so a job's log shows progress as it goes
# rather than in 8 KB blocks.
export PYTHONUNBUFFERED=1

PROJECT_ROOT="${PROJECT_ROOT:-runs}"
NUM_TRIALS="${NUM_TRIALS:-10}"
BASE_SEED="${BASE_SEED:-42}"
LOG_DIR="${LOG_DIR:-logs/run}"
DRY_RUN="${DRY_RUN:-0}"
NUM_GPUS=""
GPU_IDS=""
JOBS_PER_GPU=1

usage() { sed -n '2,50p' "$0" | sed 's/^# \{0,1\}//'; exit "${1:-0}"; }

SPECS=()
while [ $# -gt 0 ]; do
    case "$1" in
        --gpus)          NUM_GPUS="$2"; shift 2 ;;
        --gpu-ids)       GPU_IDS="${2//,/ }"; shift 2 ;;
        --jobs-per-gpu)  JOBS_PER_GPU="$2"; shift 2 ;;
        --trials)        NUM_TRIALS="$2"; shift 2 ;;
        --root)          PROJECT_ROOT="$2"; shift 2 ;;
        --dry-run)       DRY_RUN=1; shift ;;
        -h|--help)       usage 0 ;;
        -*)              echo "unknown option $1" >&2; usage 1 ;;
        *)               SPECS+=("$1"); shift ;;
    esac
done
[ "${#SPECS[@]}" -gt 0 ] || usage 1

# The repo virtualenv if there is one, so the script works without activating it.
if [ -z "${PYTHON:-}" ]; then
    if [ -x "$REPO_ROOT/.venv/bin/python" ]; then
        PYTHON="$REPO_ROOT/.venv/bin/python"
    else
        PYTHON="python3"
    fi
fi
# Checked up front: a missing interpreter otherwise fails once per trial.
if ! command -v "$PYTHON" >/dev/null 2>&1; then
    echo "ERROR: interpreter '$PYTHON' not found; set PYTHON=..." >&2
    exit 1
fi

# ----------------------------------------------------------------------------
# Collecting the jobs
# ----------------------------------------------------------------------------
#
# The blocks below do not run anything: `run_condition` appends one line per
# missing trial to $JOBFILE -- a tag, the output directory and the command,
# with @GPU@ where the leased GPU goes. The last section runs the file.

mkdir -p "$LOG_DIR"
# One job file per invocation: two launches sharing a log directory must not
# rewrite each other's queue. Kept afterwards as the record of what ran.
JOBFILE="$LOG_DIR/jobs.$$.tsv"
: > "$JOBFILE"
SKIPPED=0

# "CartPole-v1" -> "CartPole_v1", for a directory name. ENV_DIR_SUFFIX, if
# set, is appended.
sanitize() { echo "${1//-/_}${ENV_DIR_SUFFIX:-}"; }

# run_condition <suite> <setting> <method> <script> <cell> [args...]
run_condition() {
    local suite="$1" setting="$2" method="$3" script="$4" env="$5"
    shift 5
    local env_dir trial seed out_dir cmd
    env_dir="$(sanitize "$env")"
    for trial in ${TRIALS:-$(seq 1 "$NUM_TRIALS")}; do
        seed=$((BASE_SEED + trial - 1))
        out_dir="$PROJECT_ROOT/$suite/$setting/$method/$env_dir/trial_$trial"
        if [ -f "$out_dir/training_metrics.json" ]; then
            SKIPPED=$((SKIPPED + 1))
            continue
        fi
        cmd=$(printf '%q ' "$PYTHON" "$script" --env "$env" --gpus @GPU@ \
              --trial "$trial" --seed "$seed" --output_dir "$out_dir" "$@")
        printf '%s\t%s\t%s\n' "$suite.$setting.$method.$env_dir.trial$trial" \
            "$out_dir" "$cmd" >> "$JOBFILE"
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
# populations in mode `weights_only`, `pbt_hp` / `pbt2_hp` in mode `hp_only`.
# The `_weights` / `_hp` arms never run under `all`: they have to be named.
# The name is decoded in `source/algorithms/rl/pbt.py:pbt_arm`.
PBT_ARMS="pbt pbt2 pbt_weights pbt2_weights pbt_hp pbt2_hp"
wants_pbt() {
    case "$1" in
        *_weights|*_hp) case " $METHODS " in *" $1 "*) return 0 ;; *) return 1 ;; esac ;;
        *)         wants "$1" ;;
    esac
}

# ----------------------------------------------------------------------------
# Blocks: gymnax_noncontinual, gymnax_continual, gymnax_popsize
# ----------------------------------------------------------------------------
#
# CartPole, Acrobot and MountainCar through `source/run.py --suite gymnax`, the
# shared runners every benchmark uses. What a method, a cell and a budget ARE
# lives in `source/configs/gymnax.yaml`; nothing here sets a hyperparameter. The
# per-method gymnax trainers these blocks used to call were retired on
# 2026-09-28 -- see that config's docstring for what changed.
#
# A cell is an environment (`CartPole-v1`, stationary) or an environment at an
# observation-noise width (`CartPole-v1_sigma1.0`, continual), and names the
# run directory, so the tree is laid out as before:
#
#     <root>/gymnax/<noncontinual|continual>/<method>/CartPole_v1_sigma1.0/trial_<n>
#
# Environment knobs:
#   ENVS          environments (default CartPole-v1 Acrobot-v1 MountainCar-v0)
#   SIGMAS        continual noise widths (default 0.02 1.0 2.0; NOISE_RANGE
#                 forces one). Each must be in the config's SIGMAS.
#   NUM_TASKS     continual phases (default 10, 200 generations each)
#   TASK_PERIOD   distinct sub-tasks the phases cycle through; NUM_TASKS=20
#                 TASK_PERIOD=10 visits each twice
#   TASK_TYPE     noise | param | actions, with PARAM_NAME / PARAM_RANGE="lo hi"
#   DNS_REPERTOIRE_RATIO   dns_gaussian's repertoire share of the population
#   POP_SIZES / SIGMA      gymnax_popsize only (default "2 8 32 128" / 1.0)
#   GYMNAX_EXTRA  anything else, passed straight to source/run.py
#
# HYPERPARAMETER ARMS carry their value in the name and are matched on the
# literal name, never through `all`:
#
#     ga_sigma0.15          ga at sigma 0.15
#     es_sigma0.03          es at sigma 0.03
#     es_sigma0.2_lr0.02    es at sigma 0.2, learning rate 0.02
#     ppo_lr0.001           ppo at learning rate 0.001
#     ppo_ent0.03           ppo at entropy coefficient 0.03

GYMNAX_ENVS_DEFAULT="CartPole-v1 Acrobot-v1 MountainCar-v0"

# The command-line arguments a continual gymnax run takes from the knobs above.
gymnax_continual_args() {
    local args="--num_phases ${NUM_TASKS:-10}"
    [ "${TASK_PERIOD:-0}" -gt 0 ] && args="$args --num_tasks $TASK_PERIOD"
    if [ "${TASK_TYPE:-noise}" != noise ]; then
        args="$args --task_type $TASK_TYPE"
        [ -n "${PARAM_NAME:-}" ] && args="$args --param_name $PARAM_NAME"
        [ -n "${PARAM_RANGE:-}" ] && args="$args --param_range $PARAM_RANGE"
    fi
    echo "$args"
}

# block_gymnax <setting> <cells> [args for every run...]
block_gymnax() {
    local setting="$1" cells="$2"
    shift 2
    local script="source/run.py"
    local method cell method_args

    echo "=========================================="
    echo "Block: gymnax_$setting"
    echo "  cells      : $cells"
    echo "  methods    : $METHODS"
    echo "  trials     : $NUM_TRIALS (base seed $BASE_SEED)"
    echo "  run args   : $* ${GYMNAX_EXTRA:-}"
    echo "=========================================="

    for method in ${METHOD_ORDER:-ga dns dns_gaussian es ppo trac redo cchain $PBT_ARMS}; do
        case "$method" in
            pbt*) wants_pbt "$method" || continue ;;
            *)    wants "$method" || continue ;;
        esac
        method_args=""
        [ "$method" = dns_gaussian ] && [ -n "${DNS_REPERTOIRE_RATIO:-}" ] \
            && method_args="--searcher_override repertoire_ratio=$DNS_REPERTOIRE_RATIO"
        for cell in $cells; do
            run_condition gymnax "$setting" "$method" "$script" "$cell" \
                --suite gymnax --method "$method" $method_args "$@" ${GYMNAX_EXTRA:-}
        done
    done

    local hp_arm hp_base hp_sigma hp_lr
    for hp_arm in $METHODS; do
        case "$hp_arm" in
            es_sigma*_lr*)
                hp_base="${hp_arm%%_sigma*}"
                hp_sigma="${hp_arm#*_sigma}"; hp_sigma="${hp_sigma%%_lr*}"
                hp_lr="${hp_arm##*_lr}"
                method_args="--ne_override sigma=$hp_sigma learning_rate=$hp_lr" ;;
            ga_sigma?*|es_sigma?*)
                hp_base="${hp_arm%%_sigma*}"
                method_args="--ne_override sigma=${hp_arm#*_sigma}" ;;
            ppo_lr?*)
                hp_base=ppo; method_args="--ppo_override learning_rate=${hp_arm#ppo_lr}" ;;
            ppo_ent?*)
                hp_base=ppo; method_args="--ppo_override ent_coef=${hp_arm#ppo_ent}" ;;
            *) continue ;;
        esac
        for cell in $cells; do
            run_condition gymnax "$setting" "$hp_arm" "$script" "$cell" \
                --suite gymnax --method "$hp_base" $method_args "$@" ${GYMNAX_EXTRA:-}
        done
    done
}

block_gymnax_noncontinual() {
    block_gymnax noncontinual "${ENVS:-$GYMNAX_ENVS_DEFAULT}"
}

block_gymnax_continual() {
    local sigmas="${NOISE_RANGE:-${SIGMAS:-0.02 1.0 2.0}}"
    local cells="" env sigma
    for env in ${ENVS:-$GYMNAX_ENVS_DEFAULT}; do
        for sigma in $sigmas; do cells="$cells ${env}_sigma${sigma}"; done
    done
    block_gymnax continual "$cells" $(gymnax_continual_args)
}

# The effect of population size on continual NE, at one noise width. N=512 is
# gymnax_continual itself, so the sweep stops below it. The population size is
# part of the method name (`ga_pop8`), which is what the figure scripts read.
block_gymnax_popsize() {
    local sigma="${SIGMA:-1.0}"
    local cells="" env pop method cell
    for env in ${ENVS:-$GYMNAX_ENVS_DEFAULT}; do cells="$cells ${env}_sigma${sigma}"; done
    echo "=========================================="
    echo "Block: gymnax_popsize (continual)"
    echo "  cells      : $cells"
    echo "  pop sizes  : ${POP_SIZES:-2 8 32 128}"
    echo "  methods    : $METHODS"
    echo "=========================================="
    for pop in ${POP_SIZES:-2 8 32 128}; do
        for method in ga dns es; do
            wants "$method" || continue
            for cell in $cells; do
                run_condition gymnax continual "${method}_pop${pop}" source/run.py "$cell" \
                    --suite gymnax --method "$method" --pop_size "$pop" \
                    $(gymnax_continual_args) ${GYMNAX_EXTRA:-}
            done
        done
    done
}

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
# in `source/configs/minigrid.yaml` instead, next to the arm
# definitions it has to agree with, where a mismatch is a failed assertion
# (`check_against_ppo` in `source/utils/config.py`) rather than a number that drifted between two shell
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
# Blocks: cheetah_*   (the ICLR mjx study)
# ----------------------------------------------------------------------------
#
# The MiniGrid arrangement on the MJX body: `source/run.py --suite mjx`
# is argument plumbing over the two suite-generic runners under
# `source/runners/`, and `source/configs/mjx.yaml` is what an arm is. There
# is no cheetah trainer, which is the point -- one shared code path.
# The two blocks below differ only in which cells they default to.
#
# A cell is a body plus a KIND of sub-task (`<body>_noise`, `<body>_friction`)
# or that body's stationary control (`<body>`), and every budget number lives
# in `source/configs/mjx.yaml`; `source/utils/config.py` asserts the NE and RL halves are matched at the start
# of every trial. The BODY comes from the cell, so nothing here names one.

block_mjx() {
    local setting="$1" envs="$2"
    local script="source/run.py"

    # The eight arms the ICLR grid settled on. `ga` and `dns_gaussian` are the
    # GAUSSIAN pair -- same mutation, different selection rule. Iso+LineDD
    # `dns` is not in this grid.
    for method in ${METHOD_ORDER:-ga es dns_gaussian ppo trac redo cchain pbt pbt2 pbt_weights pbt2_weights}; do
        wants "$method" || continue
        for env in $envs; do
            run_condition "mjx" "$setting" "$method" "$script" "$env" \
                --suite mjx --method "$method" ${MJX_EXTRA:-}
        done
    done
}

block_cheetah_continual() {
    block_mjx continual "${ENVS:-cheetah_noise cheetah_friction}"
}

block_cheetah_noncontinual() {
    # One stationary cell serves both families: sub-task 0 is the unperturbed
    # body under either task_mod, so the two would be the same run twice. See
    # source/configs/mjx.yaml.
    block_mjx noncontinual "${ENVS:-cheetah}"
}


block_minigrid() {
    local setting="$1" envs="$2"
    local script="source/run.py"

    # The eight arms the ICLR grid settled on. `ga` and `dns_gaussian` are the
    # GAUSSIAN pair -- same mutation, different selection rule. Iso+LineDD
    # `dns` is not in this grid.
    for method in ${METHOD_ORDER:-ga es dns_gaussian ppo trac redo cchain pbt pbt2 pbt_weights pbt2_weights}; do
        wants "$method" || continue
        for env in $envs; do
            run_condition "minigrid" "$setting" "$method" "$script" "$env" \
                --suite minigrid --method "$method" ${MINIGRID_EXTRA:-}
        done
    done

    # HYPERPARAMETER ARMS, as in block_gymnax_continual: `ga_sigma0.03` is
    # the `ga` arm with settings.NE_ARMS['ga']['sigma'] replaced by 0.03
    # (--ne_override), `es_sigma0.3` the same for `es`. The directory is
    # the arm name, so the value is what the run is; matched on the literal
    # name, never through `all`.
    local hp_arm hp_base
    for hp_arm in $METHODS; do
        case "$hp_arm" in
            ga_sigma?*|es_sigma?*)
                hp_base="${hp_arm%%_sigma*}"
                for env in $envs; do
                    run_condition "minigrid" "$setting" "$hp_arm" "$script" "$env" \
                        --suite minigrid --method "$hp_base" --ne_override "sigma=${hp_arm#*_sigma}" \
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
# `source/run.py --suite kinetix` is the whole of the difference between a
# stationary cell and the continual chain: both go through the same two
# suite-generic runners, and the cell name decides the schedule.

block_kinetix() {
    local setting="$1" envs="$2"
    local script="source/run.py"

    for method in ${METHOD_ORDER:-ga es dns dns_gaussian ppo trac redo cchain pbt pbt2 pbt_weights pbt2_weights}; do
        wants "$method" || continue
        for env in $envs; do
            run_condition "kinetix" "$setting" "$method" "$script" "$env" \
                --suite kinetix --method "$method" ${KINETIX_EXTRA:-}
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

# ----------------------------------------------------------------------------
# Which settings, which methods
# ----------------------------------------------------------------------------

ALL_SETTINGS="gymnax_noncontinual gymnax_continual minigrid_noncontinual
minigrid_continual kinetix_noncontinual kinetix_continual
cheetah_noncontinual cheetah_continual"

suite_of() {
    case "$1" in
        gymnax_*)            echo gymnax ;;
        minigrid_*)          echo minigrid ;;
        kinetix_*)           echo kinetix ;;
        cheetah_*)           echo mjx ;;
        *)                   return 1 ;;
    esac
}

# The methods the paper reports on a benchmark: its config's reported_arms.
reported_methods() {
    "$PYTHON" -c "import sys; sys.path.insert(0, '.')
from source.utils.config import load; print(' '.join(load('$1').reported_arms))"
}

EXPANDED=()
for spec in "${SPECS[@]}"; do
    if [ "${spec%%:*}" = all ]; then
        for s in $ALL_SETTINGS; do EXPANDED+=("$s${spec#all}"); done
    else
        EXPANDED+=("$spec")
    fi
done

echo "=========================================="
for spec in "${EXPANDED[@]}"; do
    setting="${spec%%:*}"
    suite="$(suite_of "$setting")" && declare -F "block_$setting" >/dev/null \
        || { echo "unknown setting '$setting'; see --help" >&2; exit 1; }
    if [ "$spec" != "$setting" ]; then
        METHODS="${spec#*:}"; METHODS="${METHODS//,/ }"
    else
        METHODS="$(reported_methods "$suite")" \
            || { echo "could not read REPORTED_ARMS for $suite" >&2; exit 1; }
    fi
    before=$(wc -l < "$JOBFILE")
    "block_$setting" >> "$LOG_DIR/collect.$$.log" 2>&1 \
        || { echo "setting '$setting' failed to build its jobs; see $LOG_DIR/collect.$$.log" >&2; exit 1; }
    printf '%-22s %4d jobs  (%s)\n' "$setting" $(( $(wc -l < "$JOBFILE") - before )) "$METHODS"
done
NJOBS=$(wc -l < "$JOBFILE")

# ----------------------------------------------------------------------------
# Running them
# ----------------------------------------------------------------------------

if [ -z "$GPU_IDS" ]; then
    GPU_IDS=$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | tr '\n' ' ')
    GPU_IDS=${GPU_IDS:-0}
    if [ -n "$NUM_GPUS" ]; then
        GPU_IDS=$(echo $GPU_IDS | tr ' ' '\n' | head -n "$NUM_GPUS" | tr '\n' ' ')
    fi
fi
SLOTS=()
for g in $GPU_IDS; do
    for _ in $(seq 1 "$JOBS_PER_GPU"); do SLOTS+=("$g"); done
done

echo "------------------------------------------"
echo "jobs       : $NJOBS to run, $SKIPPED already done"
echo "gpus       : $(echo $GPU_IDS) (${JOBS_PER_GPU} job(s) each)"
echo "runs ->    : $PROJECT_ROOT"
echo "logs ->    : $LOG_DIR   (job list $JOBFILE)"
echo "=========================================="

if [ "$DRY_RUN" = "1" ]; then
    cut -f3 "$JOBFILE" | sed "s/@GPU@/<gpu>/"
    exit 0
fi
[ "$NJOBS" -gt 0 ] || { echo "nothing to run"; exit 0; }

STATUS="$LOG_DIR/status.$$.tsv"
: > "$STATUS"

# One token per slot in a FIFO; a job takes one and puts it back when it exits.
POOL_DIR="$(mktemp -d)"
mkfifo "$POOL_DIR/pool"
exec 9<>"$POOL_DIR/pool"          # read+write, so opening never blocks
rm -rf "$POOL_DIR"
for g in "${SLOTS[@]}"; do echo "$g" >&9; done
trap 'exec 9>&- 2>/dev/null || true' EXIT

pids=()
while IFS=$'\t' read -r tag out_dir cmd; do
    read -r gpu <&9                            # waits for a free GPU
    (
        log="$LOG_DIR/$tag.log"
        mkdir -p "$out_dir"
        eval "${cmd//@GPU@/$gpu}" > "$log" 2>&1
        code=$?
        # A run that exits cleanly without its metrics file is a failure too:
        # it would otherwise be retried forever or, worse, look done.
        [ "$code" = 0 ] && [ ! -f "$out_dir/training_metrics.json" ] && code=99
        printf '%s\texit=%d\tgpu=%s\t%s\n' "$tag" "$code" "$gpu" "$log" \
            | tee -a "$LOG_DIR/status.tsv" >> "$STATUS"
        [ "$code" = 0 ] && echo "   done $tag" || echo "!! FAILED $tag (exit $code, see $log)"
        echo "$gpu" >&9                        # hand it back, success or not
    ) &
    pids+=($!)
done < "$JOBFILE"
for p in "${pids[@]}"; do wait "$p"; done

failed=$(grep -vc 'exit=0' "$STATUS")
echo "=========================================="
if [ "$failed" -eq 0 ]; then
    echo "All $NJOBS jobs finished."
else
    echo "$failed of $NJOBS jobs failed:"
    grep -v 'exit=0' "$STATUS" | cut -f1,2,4 | sed 's/^/  /'
    echo "Re-running the same command retries exactly these."
    exit 1
fi
