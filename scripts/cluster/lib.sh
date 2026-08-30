#!/bin/bash
# Helpers shared by submit_search.sh, submit_train.sh and submit_eval.sh.
#
# The block parser below is the one piece that must stay in step with
# scripts/optuna/make_train_catalogues.py: it defines what a "command" in a
# run*_{search,train,eval}.sh file is. Copy-pasting it into each submitter
# would mean a change to the generated layout has to be made in three places
# or one submitter silently stops finding commands.
#
# Source it as:  . "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

# Clusters the submitters know how to prepend a launcher for; a value of
# --cluster outside this list is a typo, not a new site.
KNOWN_CLUSTERS="euler clariden"

# A block opens at the commented 'python3 src/train.py' or 'python3 src/eval.py'
# line and runs to the first line without a trailing continuation. Emits one
# flat command per block, starting with the entry point ('src/train.py ...'), so
# that a caller need not know which of the two a catalogue holds.
extract_cmds() {
    awk '
        /^# python3 src\/(train|eval)\.py/ {
            line = $0
            sub(/^# python3[ \t]+/, "", line)
            gsub(/^[ \t]+|[ \t]+$/, "", line)
            ends = (line ~ /\\$/)
            sub(/[ \t]*\\$/, "", line)
            cmd = line
            collecting = 1
            if (!ends) { collecting = 0; print cmd }
            next
        }
        collecting {
            line = $0
            sub(/^# ?/, "", line)
            gsub(/^[ \t]+|[ \t]+$/, "", line)
            ends = (line ~ /\\$/)
            sub(/[ \t]*\\$/, "", line)
            if (line != "") cmd = cmd " " line
            if (!ends) { collecting = 0; print cmd }
        }
    ' "$1"
}

# The value of an override, e.g. field "$cmd" 'data\.method' or
# field "$cmd" 'hydra\.run\.dir'. Empty when the command has none.
field() { printf "%s" "$1" | grep -oE "$2=[^ ]+" | head -1 | cut -d= -f2- || true; }

# The block's grid position, '<model>_nconst<N>_<variant>', reconstructed from
# the overrides. It is what --only matches on and what names the driver logs.
# The variant names match the log directories the catalogues write to, so
# '--only symm_so2' and '--only nconst150' both select the obvious thing.
block_label() {
    local exp model nconst method scheme variant
    exp=$(field "$1" 'experiment')
    exp=${exp##*/}
    model=${exp%_*}
    nconst=${exp##*_}
    method=$(field "$1" 'data\.method')
    scheme=$(field "$1" 'data\.augmentation_scheme')
    case "$method" in
        symm_loss) variant="symm_$scheme" ;;
        *) if [ "$scheme" = "none" ]; then variant="baseline"; else variant="aug_$scheme"; fi ;;
    esac
    printf "%s_nconst%s_%s" "$model" "$nconst" "$variant"
}

# $1 = 1-based index, $2 = name. Matches against the caller's $only, a
# comma-separated list of name substrings or indices; an empty $only selects
# everything. Written with explicit ifs rather than 'test && action': under
# 'set -e' a bare failing && list aborts the script.
selected() {
    if [ -z "$only" ]; then return 0; fi
    local idx="$1" name="$2" s
    local IFS=','
    for s in $only; do
        if [ -n "$s" ]; then
            case "$s" in
                # An all-digit selector is an index only. Matching it as a
                # substring too would make '--only 1' also pick nconst16, ...
                *[!0-9]*) case "$name" in *"$s"*) return 0 ;; esac ;;
                *) if [ "$s" = "$idx" ]; then return 0; fi ;;
            esac
        fi
    done
    return 1
}

# Every command is relative to the repository root, and the log directories are
# created there too, so refuse to run from anywhere else.
require_repo_root() {
    if [ ! -f .project-root ]; then
        echo "run this from the repository root" >&2
        exit 1
    fi
}

# Export what scripts/setup.sh wrote, so that a nohup'd driver inherits
# PROJECT_ROOT (configs/paths/default.yaml will not compose without it) and
# HF_HUB_CACHE (huggingface_hub reads it at import). Absent .env is not an
# error here: the caller may have exported both itself.
load_env() {
    if [ -f .env ]; then
        set -a
        # shellcheck disable=SC1091  # .env is generated, not checked in
        . ./.env
        set +a
    fi
}

# A run without HF_HUB_CACHE downloads the 3.1 GB record snapshot again into
# whatever default cache the driver's home directory offers, which on a shared
# machine means one copy per user and a very slow first epoch.
require_hf_cache() {
    if [ -z "${HF_HUB_CACHE:-}" ]; then
        echo "error: HF_HUB_CACHE is not set - the record snapshot would be re-downloaded." >&2
        echo "       run 'bash scripts/setup.sh' and edit .env, or pass --hf-cache <path>." >&2
        exit 1
    fi
}

# $1 = command, $2 = GPU index. The catalogues all say trainer.devices=[0]; this
# is what makes several queues on one multi-GPU host independent of each other.
pin_gpu() {
    printf "%s" "$1" | sed -E "s/trainer\.devices=\[[0-9]+\]/trainer.devices=[$2]/"
}

# Turn a single-run command into a multirun one. Hydra writes a multirun job's
# output to <sweep.dir>/<job number> rather than to <run.dir>, so the eval
# blocks' manifest path has to gain the same '/0/' that the train job's
# checkpoints did.
to_sweep_dir() {
    printf "%s" "$1" |
        sed -E 's#hydra\.run\.dir=#hydra.sweep.dir=#' |
        sed -E 's#checkpoint_manifest_path=([^ ]*)/checkpoint_manifest\.json#checkpoint_manifest_path=\1/0/checkpoint_manifest.json#'
}

# $1 = cluster, $2 = slurm time limit in minutes. The launcher's own default
# (60) is shorter than any of these trainings. The '-m' that turns these into
# multiruns is left to the caller: a search block already carries one, a train
# or eval block does not.
cluster_prefix() {
    case " $KNOWN_CLUSTERS " in
        *" $1 "*) ;;
        *) echo "unknown cluster: $1 (known: $KNOWN_CLUSTERS)" >&2; exit 2 ;;
    esac
    printf -- "hydra/launcher=submitit_slurm_%s hydra.launcher.timeout_min=%s" "$1" "$2"
}

# Preflight: the drivers are nohup'd, so an unimportable dependency shows up
# only as a tiny log per driver and nothing ever gets trained. Forgetting to
# activate the env once costs a whole batch, so it is checked up front.
# $1 = python, $2 = modules, rest = hint lines.
require_imports() {
    local py="$1" mods="$2" line
    shift 2
    "$py" -c "import $mods" >/dev/null 2>&1 && return 0
    echo "error: '$py' cannot import $mods - the training env is not active." >&2
    for line in "$@"; do echo "       $line" >&2; done
    exit 1
}
