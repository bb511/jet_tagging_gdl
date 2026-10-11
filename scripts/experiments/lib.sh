#!/bin/bash
# Helpers shared by submit_search.sh, submit_train.sh and submit_eval.sh.
#
# The block parser below is the single definition of what a "command" in a
# run*_{search,train,eval}.sh file is, and must stay in step with the generator,
# scripts/experiments/make_train_catalogues.py. Copied into each submitter, it
# would need the same edit in three places whenever the generated layout changed.
#
# State shared with the callers: read_catalogue sets cmds (one command per line)
# and n_total, and zeroes n_sub, the count of selected blocks, which
# dispatch_block raises and finish_queue and require_selection read.

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
# space, so that a key is never read as the tail of a longer one.
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

# The name of catalogue $1's log directory, <its directory>_<its stem>. A
# feature-set catalogue of the replica, under
# scripts/experiments/top_tagging/<label>/, carries the data set in front, since
# <label>_<stem> alone also names the hls4ml catalogue of the same label.
catalogue_stem() {
    local dir prefix=""
    dir=$(dirname "$1")
    if [ "$(basename "$(dirname "$dir")")" = top_tagging ]; then
        prefix=top_tagging_
    fi
    printf "%s%s_%s" "$prefix" "$(basename "$dir")" "$(basename "$1" .sh)"
}

# Every command and every log directory is relative to the repository root.
require_repo_root() {
    if [ ! -f .project-root ]; then
        echo "run this from the repository root" >&2
        exit 1
    fi
}

# $1 = catalogue, $2 = the noun of its commands in the error message. Sets cmds,
# n_total and n_sub (see the header); a catalogue without a block is an error.
read_catalogue() {
    require_repo_root
    if [ ! -f "$1" ]; then
        echo "no such file: $1" >&2
        exit 1
    fi
    cmds=$(extract_cmds "$1")
    n_total=$(printf "%s\n" "$cmds" | grep -c . || true)
    if [ "$n_total" -eq 0 ]; then
        echo "No $2 commands found in $1" >&2
        exit 1
    fi
    n_sub=0
}

# $1 = 1 under --dry-run. Exports what scripts/setup.sh wrote, so a nohup'd
# driver inherits PROJECT_ROOT (configs/paths/default.yaml will not compose
# without it) and HF_HUB_CACHE, which a real launch requires. Absent .env is not
# an error: the caller may have exported both itself.
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
    if [ "$1" -eq 0 ]; then require_hf_cache; fi
}

# Without HF_HUB_CACHE the 3.1 GB record snapshot is downloaded again into the
# driver's home directory, one copy per user on a shared machine.
require_hf_cache() {
    if [ -z "${HF_HUB_CACHE:-}" ]; then
        echo "error: HF_HUB_CACHE is not set - the record snapshot would be re-downloaded." >&2
        echo "       run 'bash scripts/setup.sh' and edit .env, or export HF_HUB_CACHE." >&2
        exit 1
    fi
}

# $1 = python, $2 = modules. The drivers are nohup'd, so an inactive env shows
# up only as a tiny log per driver and a whole batch is lost.
require_imports() {
    "$1" -c "import $2" >/dev/null 2>&1 && return 0
    echo "error: '$1' cannot import $2 - the training env is not active." >&2
    echo "       conda activate jet_tagging_gdl" >&2
    echo "       (or re-run with PYTHON=/path/to/env/bin/python3)" >&2
    exit 1
}

# $1 = command, $2 = GPU index. The catalogues all say trainer.devices=[0]; this
# is what makes several queues on one multi-GPU host independent.
pin_gpu() {
    printf "%s" "$1" | sed -E "s/trainer\.devices=\[[0-9]+\]/trainer.devices=[$2]/"
}

# $1 = command, $2 = the GPU of --gpu, empty to keep the catalogue's.
place_block() {
    if [ -n "$2" ]; then pin_gpu "$1" "$2"; else printf "%s" "$1"; fi
}

# $1 = queue, $2 = the submitter's name, $3 = timestamp, $4 = catalogue.
# Starts the script that runs the selected blocks one after another.
open_queue() {
    printf '#!/bin/bash\n# Generated by %s on %s from %s.\n' "$2" "$3" "$4" > "$1"
    # No 'set -e': one block that dies must not take the rest of the queue with it.
    printf 'cd %s || exit 1\n' "$PWD" >> "$1"
}

# $1 = 1 under --dry-run, $2 = queue, $3 = block index, $4 = label, $5 = the full
# command. Counts the block in n_sub, then prints it or appends it to the queue.
dispatch_block() {
    n_sub=$((n_sub + 1))
    if [ "$1" -eq 1 ]; then
        echo "[$3/$n_total] $5"
        return 0
    fi
    printf 'echo "[%d/%d] %s"\n%s || echo "FAILED: %s" >&2\n' \
        "$n_sub" "$n_total" "$4" "$5" "$4" >> "$2"
}

# $1 = 1 under --dry-run, $2 = queue, $3 = the GPU of --gpu. Launches the queue
# detached, or under --dry-run only counts what it would hold.
finish_queue() {
    local log
    if [ "$1" -eq 1 ]; then
        echo
        echo "$n_sub of $n_total command(s) would run."
        return 0
    fi
    log="${2%.sh}.log"
    nohup bash "$2" > "$log" 2>&1 &
    echo
    echo "$n_sub of $n_total command(s) queued on GPU ${3:-0}, running detached."
    echo "Queue:  $2"
    echo "Log:    $log"
}

# $1 = the --only selection, $2 = catalogue, named in the message when given.
require_selection() {
    if [ "$n_sub" -eq 0 ]; then
        echo "nothing matched --only '$1'${2:+ in $2}" >&2
        exit 1
    fi
}
