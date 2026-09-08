#!/bin/bash
# Helpers shared by submit_search.sh, submit_train.sh and submit_eval.sh.
#
# The block parser below is the single definition of what a "command" in a
# run*_{search,train,eval}.sh file is, and must stay in step with the generator,
# scripts/optuna/make_train_catalogues.py. Copied into each submitter, it would
# need the same edit in three places whenever the generated layout changed.

# Clusters with a submitit launcher config; anything else is a typo, not a site.
KNOWN_CLUSTERS="euler clariden"

# A block opens at a commented 'python3 src/{train,eval}.py' line and runs to
# the first line without a trailing continuation. Emits one flat command per
# block, entry point first, so the caller need not branch on which of the two it holds.
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

# The value of an override, e.g. field "$cmd" 'data\.method'. Empty when absent.
# The key is a regex, and it has to sit at the start of the command or behind a
# space: unanchored, 'checkpoint_manifest_path' would also match the tail of
# 'baseline_checkpoint_manifest_path' and return the wrong manifest.
field() {
    printf "%s" "$1" |
        grep -oE "(^|[[:space:]])$2=[^ ]+" |
        head -1 |
        sed -E 's/^[[:space:]]+//' |
        cut -d= -f2- || true
}

# The block's grid position, '<model>_nconst<N>_<variant>', rebuilt from the
# overrides: what --only matches on and what names the driver logs. The variant
# names are those of the log directories the catalogues write to.
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

# $1 = 1-based index, $2 = name, matched against the caller's $only (comma-
# separated name substrings or indices; empty selects everything). Explicit ifs
# rather than 'test && action': under 'set -e' a failing && list aborts.
selected() {
    if [ -z "$only" ]; then return 0; fi
    local idx="$1" name="$2" s
    local IFS=','
    for s in $only; do
        if [ -n "$s" ]; then
            case "$s" in
                # All-digit selector is an index only: '--only 1' must not also
                # match nconst16 as a substring.
                *[!0-9]*) case "$name" in *"$s"*) return 0 ;; esac ;;
                *) if [ "$s" = "$idx" ]; then return 0; fi ;;
            esac
        fi
    done
    return 1
}

# Every command and every log directory is relative to the repository root.
require_repo_root() {
    if [ ! -f .project-root ]; then
        echo "run this from the repository root" >&2
        exit 1
    fi
}

# Export what scripts/setup.sh wrote, so a nohup'd driver inherits PROJECT_ROOT
# (configs/paths/default.yaml will not compose without it) and HF_HUB_CACHE.
# Absent .env is not an error: the caller may have exported both itself.
load_env() {
    if [ -f .env ]; then
        set -a
        # shellcheck disable=SC1091  # .env is generated, not checked in
        . ./.env
        set +a
    fi
    # The loader runs inside the driver and its CPU work is small, so torch's default of one
    # thread per core would only make the drivers sharing a machine contend; .env or the caller
    # may still choose the count.
    export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
}

# Without HF_HUB_CACHE the 3.1 GB record snapshot is downloaded again into the
# driver's home directory, one copy per user on a shared machine.
require_hf_cache() {
    if [ -z "${HF_HUB_CACHE:-}" ]; then
        echo "error: HF_HUB_CACHE is not set - the record snapshot would be re-downloaded." >&2
        echo "       run 'bash scripts/setup.sh' and edit .env, or pass --hf-cache <path>." >&2
        exit 1
    fi
}

# $1 = command, $2 = GPU index. The catalogues all say trainer.devices=[0]; this
# is what makes several queues on one multi-GPU host independent.
pin_gpu() {
    printf "%s" "$1" | sed -E "s/trainer\.devices=\[[0-9]+\]/trainer.devices=[$2]/"
}

# Turn a single-run command into a multirun one. Hydra writes a multirun job to
# <sweep.dir>/<job number>, so both manifest paths of an eval block have to gain
# the same '/0/' that the train job's checkpoints did. The alternation is
# anchored and the substitution global: without the 'g', sed would rewrite
# whichever of checkpoint_manifest_path and baseline_checkpoint_manifest_path
# came first and leave the other pointing one directory too high.
to_sweep_dir() {
    printf "%s" "$1" |
        sed -E 's#hydra\.run\.dir=#hydra.sweep.dir=#' |
        sed -E 's#(^|[[:space:]])(baseline_)?checkpoint_manifest_path=([^ ]*)/checkpoint_manifest\.json#\1\2checkpoint_manifest_path=\3/0/checkpoint_manifest.json#g'
}

# $1 = cluster, $2 = slurm time limit in minutes (the launcher's own default of
# 60 is shorter than any of these trainings). The '-m' is left to the caller: a
# search block already carries one, a train or eval block does not.
cluster_prefix() {
    case " $KNOWN_CLUSTERS " in
        *" $1 "*) ;;
        *) echo "unknown cluster: $1 (known: $KNOWN_CLUSTERS)" >&2; exit 2 ;;
    esac
    printf -- "hydra/launcher=submitit_slurm_%s hydra.launcher.timeout_min=%s" "$1" "$2"
}

# $1 = python, $2 = modules, rest = hint lines. The drivers are nohup'd, so an
# inactive env shows up only as a tiny log per driver and a whole batch is lost.
require_imports() {
    local py="$1" mods="$2" line
    shift 2
    "$py" -c "import $mods" >/dev/null 2>&1 && return 0
    echo "error: '$py' cannot import $mods - the training env is not active." >&2
    for line in "$@"; do echo "       $line" >&2; done
    exit 1
}
