#!/bin/bash
# Run the training blocks of a run*_train.sh catalogue.
#
# Local mode (no --cluster) queues the selected blocks behind a single nohup, so
# a GPU runs one training at a time; four invocations with --gpu 0..3 and
# disjoint --only selections fill a four-GPU host. Cluster mode gives each block
# its own submitit driver, with hydra.run.dir rewritten to hydra.sweep.dir, so
# the checkpoints and the manifest land in <sweep.dir>/0 (submit_eval.sh rewrites
# the eval catalogue's manifest paths the same way). --folds is the third shape:
# each selected block gets a detached runner that fits the main run first and
# then one process per fold, spread over the devices of --gpus, so a five-fold
# block takes two trainings' time on a multi-GPU host or an NGT pod; the folds
# nest under the main run in MLflow, through the run id its manifest records.
#
# Usage (from the repository root):
#   bash scripts/cluster/submit_train.sh --list    scripts/hls4ml/rundeepsets_train.sh
#   bash scripts/cluster/submit_train.sh --dry-run scripts/hls4ml/rundeepsets_train.sh
#   bash scripts/cluster/submit_train.sh --only nconst16 --gpu 0 scripts/hls4ml/rundeepsets_train.sh
#   bash scripts/cluster/submit_train.sh --only 1 --folds 5 --gpus 0,1,2 scripts/hls4ml/rundeepsets_train.sh
#   bash scripts/cluster/submit_train.sh --cluster clariden scripts/hls4ml/rundeepsets_train.sh
#
#   --list             list the blocks in the catalogue and exit
#   --dry-run          print the full commands without running anything; works
#                      off the cluster, so it needs neither the env nor the data
#   --only <sel>       comma-separated label substrings ('symm_so2', 'nconst150')
#                      or 1-based indices; default is every block in the file
#   --gpu <i>          rewrite trainer.devices to this GPU (local mode)
#   --folds <k>        local mode: one detached runner per block instead of one
#                      queue entry: the main run with data.n_folds=1, then, once
#                      its manifest exists, fold i with fold=i data.n_folds=k
#                      parent_run_id=<the main run's MLflow run> under
#                      <run dir>/fold_<i>, which is what src/eval.py merges when
#                      its checkpoint_manifest_path names <run dir>. Needs --gpus,
#                      refuses --cluster
#   --gpus <list>      comma-separated devices the --folds processes take in
#                      turn, the count running on across blocks: an index goes
#                      into trainer.devices, a MIG-... or GPU-... UUID into
#                      CUDA_VISIBLE_DEVICES with trainer.devices=[0]
#   --cluster <c>      euler | clariden: one submitit job per block
#   --timeout-min <n>  slurm time limit per job (default 1440)
#   --hf-cache <p>     value for HF_HUB_CACHE; default is the one .env exports
#
# Extra hydra overrides after the file name are appended to every command; hydra
# takes the last of a duplicated override, so they win over the file's value.
# Under --folds the launcher's own fold, data.n_folds, hydra.run.dir and
# parent_run_id come last of all, so the layout the merge depends on cannot be
# overridden away.
set -eu

# shellcheck source=scripts/cluster/lib.sh
. "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

# Between the processes of --folds, so k+1 hydra start-ups do not race on one run dir tree.
STAGGER_SECONDS=3

only=""
dry=0
list=0
gpu=""
folds=""
gpus=""
cluster=""
timeout_min=1440
hf_cache=""

while [ $# -gt 0 ]; do
    case "$1" in
        --dry-run) dry=1; shift ;;
        --list) list=1; shift ;;
        --only) only="${2:?--only needs a value}"; shift 2 ;;
        --gpu) gpu="${2:?--gpu needs an index}"; shift 2 ;;
        --folds) folds="${2:?--folds needs a number}"; shift 2 ;;
        --gpus) gpus="${2:?--gpus needs a list}"; shift 2 ;;
        --cluster) cluster="${2:?--cluster needs a name}"; shift 2 ;;
        --timeout-min) timeout_min="${2:?--timeout-min needs a number}"; shift 2 ;;
        --hf-cache) hf_cache="${2:?--hf-cache needs a path}"; shift 2 ;;
        --) shift; break ;;
        -*) echo "unknown option: $1" >&2; exit 2 ;;
        *) break ;;
    esac
done

usage="usage: submit_train.sh [--list|--dry-run] [--only sel] [--gpu i] [--folds k --gpus list]"
usage="$usage [--cluster c] [--timeout-min n] [--hf-cache p] <run*_train.sh> [overrides...]"
file=${1:?$usage}
shift || true
extra="$*"

require_repo_root
[ -f "$file" ] || { echo "no such file: $file" >&2; exit 1; }

if [ -n "$folds" ]; then
    case "$folds" in
        '' | *[!0-9]* | 0 | 1) echo "--folds takes an integer of at least 2" >&2; exit 2 ;;
    esac
    [ -z "$cluster" ] || { echo "--folds is local-only; under --cluster a block is already one job" >&2; exit 2; }
    [ -n "$gpus" ] || { echo "--folds needs --gpus: $((folds + 1)) processes per block want devices" >&2; exit 2; }
    IFS=',' read -r -a devices <<< "$gpus"
    for d in "${devices[@]}"; do
        case "$d" in
            MIG-?* | GPU-?*) ;;
            '' | *[!0-9]*) echo "bad --gpus token '$d': an index, or a MIG-/GPU- UUID" >&2; exit 2 ;;
        esac
    done
fi

cmds=$(extract_cmds "$file")
n_total=$(printf "%s\n" "$cmds" | grep -c . || true)
if [ "$n_total" -eq 0 ]; then
    echo "No training commands found in $file" >&2
    exit 1
fi

if [ "$list" -eq 1 ]; then
    printf "%-3s %-34s %-8s %s\n" "#" "block" "n_folds" "run dir"
    i=0
    while IFS= read -r cmd; do
        i=$((i + 1))
        printf "%-3s %-34s %-8s %s\n" "$i" \
            "$(block_label "$cmd")" \
            "$(field "$cmd" 'data\.n_folds')" \
            "$(field "$cmd" 'hydra\.run\.dir')"
    done <<EOF
$cmds
EOF
    exit 0
fi

load_env
[ -z "$hf_cache" ] || export HF_HUB_CACHE="$hf_cache"
[ "$dry" -eq 1 ] || require_hf_cache

launcher=""
if [ -n "$cluster" ]; then
    launcher=" -m $(cluster_prefix "$cluster" "$timeout_min")"
    command -v sbatch >/dev/null 2>&1 || echo "warning: sbatch not on PATH - are you on $cluster?" >&2
fi

# PYTHON=/path/to/env/bin/python3 skips activation; --dry-run needs no env.
PYTHON=${PYTHON:-python3}
[ "$dry" -eq 1 ] || require_imports "$PYTHON" "hydra, lightning" \
    "conda activate jet_tagging_gdl" \
    "(or re-run with PYTHON=/path/to/env/bin/python3)"

stem="$(basename "$(dirname "$file")")_$(basename "$file" .sh)"
logdir="logs/submit/$stem"
run_ts=$(date +%Y-%m-%d_%H-%M-%S)
[ "$dry" -eq 1 ] || mkdir -p "$logdir"

# Local mode collects the selected commands here for one background queue.
queue="$logdir/queue_${run_ts}${gpu:+_gpu$gpu}.sh"
if [ "$dry" -eq 0 ] && [ -z "$cluster" ] && [ -z "$folds" ]; then
    printf '#!/bin/bash\n# Generated by submit_train.sh on %s from %s.\n' "$run_ts" "$file" > "$queue"
    # No 'set -e': one block that dies must not take the rest of the queue with it.
    printf 'cd %s || exit 1\n' "$PWD" >> "$queue"
fi

# Prints the MLflow run id the main run logged into, empty under logger=none.
MAIN_RUN_ID_PY='import json, sys; runs = json.load(open(sys.argv[1]))["runs"]; print(next(run.get("mlflow_run_id") or "" for run in runs if run["fold_label"] == "main"))'

# $1 = the process count since the launcher started. The devices of --gpus are
# taken in turn, the count running on across blocks, so two blocks over two
# devices load them evenly.
device_of() {
    printf '%s' "${devices[$(($1 % ${#devices[@]}))]}"
}

# $1 = device token, $2 = command, $3 = the run's own overrides, appended
# behind the extras (see the header). Prints the full command of one process.
run_command() {
    local device="$1" cmd="$2" suffix="$3" prefix=""
    case "$device" in
        MIG-* | GPU-*) prefix="CUDA_VISIBLE_DEVICES=$device "; cmd=$(pin_gpu "$cmd" 0) ;;
        *) cmd=$(pin_gpu "$cmd" "$device") ;;
    esac
    printf '%s' "$prefix$PYTHON $cmd${extra:+ $extra} $suffix"
}

# $1 = script path, $2 = block index, $3 = label, $4 = the block's run dir,
# $5 = the main run's command, the rest the fold commands. The runner fits the
# main run, reads its MLflow run id off its manifest into $parent, and fans the
# folds out under it. No manifest, no folds: the evaluation needs it as well.
write_fold_runner() {
    local script="$1" idx="$2" label="$3" base="$4" main="$5" j=0
    shift 5
    {
        printf '#!/bin/bash\n# Generated by submit_train.sh on %s: block %s (%s) of %s.\n' \
            "$run_ts" "$idx" "$label" "$file"
        printf 'cd %s || exit 1\n' "$PWD"
        printf '%s > %s 2>&1\n' "$main" "$logdir/${idx}_${label}_main.log"
        printf 'manifest=%s/checkpoint_manifest.json\n' "$base"
        # shellcheck disable=SC2016 # the runner expands $manifest, not this script
        printf '[ -f "$manifest" ] || { echo "FAILED: %s: the main run left no manifest, so the folds were not launched" >&2; exit 1; }\n' "$label"
        printf "parent=\$(%s -c '%s' \"\$manifest\")\n" "$PYTHON" "$MAIN_RUN_ID_PY"
        for fold in "$@"; do
            j=$((j + 1))
            printf '%s > %s 2>&1 &\nsleep %s\n' "$fold" "$logdir/${idx}_${label}_fold_$j.log" "$STAGGER_SECONDS"
        done
        printf 'wait\n'
    } > "$script"
}

# $1 = block index, $2 = label, $3 = command. One detached runner per block: the
# main run first, then the k folds in parallel, each on the next device of --gpus.
launch_folds() {
    local idx="$1" label="$2" cmd="$3" base main fold j script
    local -a fold_cmds=()
    base=$(field "$cmd" 'hydra\.run\.dir')
    [ -n "$base" ] || { echo "block $idx ($label) has no hydra.run.dir to put fold_<i> under" >&2; exit 1; }
    main=$(run_command "$(device_of "$n_launched")" "$cmd" "data.n_folds=1")
    n_launched=$((n_launched + 1))
    echo "[$idx/$n_total] main: $main"
    for j in $(seq 1 "$folds"); do
        # The runner expands $parent, from the main run's manifest.
        fold=$(run_command "$(device_of "$n_launched")" "$cmd" \
            "fold=$j data.n_folds=$folds hydra.run.dir=$base/fold_$j parent_run_id=\$parent")
        n_launched=$((n_launched + 1))
        fold_cmds+=("$fold")
        echo "[$idx/$n_total] fold_$j: $fold"
    done
    if [ "$dry" -eq 1 ]; then return 0; fi
    script="$logdir/folds_${run_ts}_${idx}_${label}.sh"
    write_fold_runner "$script" "$idx" "$label" "$base" "$main" "${fold_cmds[@]}"
    # setsid: a process tree started under 'kubectl exec' dies with the session.
    setsid nohup bash "$script" > "$logdir/${idx}_${label}_${run_ts}.log" 2>&1 &
}

i=0
n_sub=0
n_launched=0
while IFS= read -r cmd; do
    i=$((i + 1))
    label=$(block_label "$cmd")
    selected "$i" "$label" || continue

    if [ -n "$folds" ]; then
        launch_folds "$i" "$label" "$cmd"
        n_sub=$((n_sub + 1))
        continue
    fi
    if [ -n "$cluster" ]; then
        # One GPU per slurm job, and the run dir becomes a sweep dir.
        cmd=$(to_sweep_dir "$(pin_gpu "$cmd" 0)")
    elif [ -n "$gpu" ]; then
        cmd=$(pin_gpu "$cmd" "$gpu")
    fi
    # The launcher sits behind the entry point, where hydra documents '-m'.
    full="$PYTHON ${cmd%% *}$launcher ${cmd#* }${extra:+ $extra}"

    n_sub=$((n_sub + 1))
    if [ "$dry" -eq 1 ]; then
        echo "[$i/$n_total] $full"
        continue
    fi
    if [ -z "$cluster" ]; then
        printf 'echo "[%d/%d] %s"\n%s || echo "FAILED: %s" >&2\n' \
            "$n_sub" "$n_total" "$label" "$full" "$label" >> "$queue"
        continue
    fi
    echo "[$i/$n_total] submitting: $label"
    nohup bash -c "$full" > "$logdir/${i}_${label}_${run_ts}.log" 2>&1 &
    sleep 1
done <<EOF
$cmds
EOF

if [ "$n_sub" -eq 0 ]; then
    echo "nothing matched --only '$only' in $file" >&2
    exit 1
fi

if [ -n "$folds" ]; then
    echo
    if [ "$dry" -eq 1 ]; then
        echo "$n_sub of $n_total block(s) would run as $n_launched process(es), each block its main run first; nothing was launched."
        echo "\$parent is the main run's MLflow run id, which the block's runner reads off its manifest."
    else
        echo "$n_sub of $n_total block(s) running as $n_sub detached runner(s) of $((folds + 1)) processes each (logs: $logdir)."
        echo "Runners are setsid nohup'd, safe to log out; each fits the main run, then the folds nested under it in MLflow."
        echo "Evaluate a block with checkpoint_manifest_path=<its hydra.run.dir>, which merges the manifests."
    fi
elif [ "$dry" -eq 1 ]; then
    echo
    echo "$n_sub of $n_total command(s) would run."
elif [ -n "$cluster" ]; then
    echo
    echo "$n_sub of $n_total job(s) submitted (logs: $logdir)."
    echo "Jobs run under slurm; drivers are nohup'd, safe to log out."
    echo "Check the queue with: squeue --me"
else
    log="${queue%.sh}.log"
    nohup bash "$queue" > "$log" 2>&1 &
    echo
    echo "$n_sub of $n_total command(s) queued on GPU ${gpu:-0}, running detached."
    echo "Queue:  $queue"
    echo "Log:    $log"
fi
