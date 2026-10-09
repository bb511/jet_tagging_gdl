#!/bin/bash
# Run the evaluation blocks of a run*_eval.sh catalogue.
#
# Each block names its training's run directory, from which src/eval.py merges
# the main run's manifest with every fold_<i>/ one; it tests every checkpoint
# they name and appends one row to the results table. The selected blocks are
# queued behind a single nohup, one GPU at a time.
#
# Usage (from the repository root):
#   bash scripts/cluster/submit_eval.sh --list    scripts/hls4ml/rundeepsets_eval.sh
#   bash scripts/cluster/submit_eval.sh --dry-run scripts/hls4ml/rundeepsets_eval.sh
#   bash scripts/cluster/submit_eval.sh --only nconst16 --gpu 0 scripts/hls4ml/rundeepsets_eval.sh
#
#   --list             list the blocks in the catalogue and exit
#   --dry-run          print the full commands without running anything; it
#                      needs neither the env nor the data
#   --only <sel>       comma-separated label substrings ('symm_so2', 'nconst150')
#                      or 1-based indices; default is every block in the file
#   --gpu <i>          rewrite trainer.devices to this GPU
#
# Extra hydra overrides after the file name are appended to every command; hydra
# takes the last of a duplicated override, so they win over the file's value.
set -eu

# shellcheck source=scripts/cluster/lib.sh
. "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

only=""
dry=0
list=0
gpu=""

while [ $# -gt 0 ]; do
    case "$1" in
        --dry-run) dry=1; shift ;;
        --list) list=1; shift ;;
        --only) only="${2:?--only needs a value}"; shift 2 ;;
        --gpu) gpu="${2:?--gpu needs an index}"; shift 2 ;;
        --) shift; break ;;
        -*) echo "unknown option: $1" >&2; exit 2 ;;
        *) break ;;
    esac
done

# A block names either a run directory or one manifest file, which src/eval.py
# then tests alone.
has_manifest() {
    [ -f "$1" ] || [ -f "$1/checkpoint_manifest.json" ]
}

usage="usage: submit_eval.sh [--list|--dry-run] [--only sel] [--gpu i] <run*_eval.sh> [overrides...]"
file=${1:?$usage}
shift || true
extra="$*"

read_catalogue "$file" evaluation

if [ "$list" -eq 1 ]; then
    printf "%-3s %-34s %-8s %s\n" "#" "block" "manifest" "path"
    i=0
    while IFS= read -r cmd; do
        i=$((i + 1))
        manifest=$(field "$cmd" 'checkpoint_manifest_path')
        printf "%-3s %-34s %-8s %s\n" "$i" "$(block_label "$cmd")" \
            "$(has_manifest "$manifest" && echo yes || echo MISSING)" "$manifest"
    done <<EOF
$cmds
EOF
    exit 0
fi

load_env "$dry"

# PYTHON=/path/to/env/bin/python3 skips activation; --dry-run needs no env.
PYTHON=${PYTHON:-python3}
[ "$dry" -eq 1 ] || require_imports "$PYTHON" "hydra, lightning"

stem="$(basename "$(dirname "$file")")_$(basename "$file" .sh)"
logdir="logs/submit/$stem"
run_ts=$(date +%Y-%m-%d_%H-%M-%S)
[ "$dry" -eq 1 ] || mkdir -p "$logdir"

queue="$logdir/queue_${run_ts}${gpu:+_gpu$gpu}.sh"
[ "$dry" -eq 1 ] || open_queue "$queue" submit_eval.sh "$run_ts" "$file"

i=0
n_missing=0
while IFS= read -r cmd; do
    i=$((i + 1))
    label=$(block_label "$cmd")
    selected "$i" "$label" || continue

    cmd=$(place_block "$cmd" "$gpu")
    # A missing manifest means the training has not finished, the normal state
    # of most of a catalogue: report it and evaluate the ready blocks anyway.
    manifest=$(field "$cmd" 'checkpoint_manifest_path')
    if [ "$dry" -eq 0 ] && ! has_manifest "$manifest"; then
        echo "warning: no manifest yet for $label ($manifest)" >&2
        n_missing=$((n_missing + 1))
    fi
    dispatch_block "$dry" "$queue" "$i" "$label" "$PYTHON $cmd${extra:+ $extra}"
done <<EOF
$cmds
EOF

require_selection "$only" "$file"
[ "$n_missing" -eq 0 ] || echo "$n_missing of $n_sub selected block(s) have no manifest yet." >&2
finish_queue "$dry" "$queue" "$gpu"
