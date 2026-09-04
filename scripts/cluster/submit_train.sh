#!/bin/bash
# Run the training blocks of a run*_train.sh catalogue.
#
# Local mode (no --cluster) queues the selected blocks behind a single nohup, so
# a GPU runs one training at a time; four invocations with --gpu 0..3 and
# disjoint --only selections fill a four-GPU host. Cluster mode gives each block
# its own submitit driver, with hydra.run.dir rewritten to hydra.sweep.dir, so
# the checkpoints and the manifest land in <sweep.dir>/0 (submit_eval.sh rewrites
# the eval catalogue's manifest paths the same way). --folds is the third shape:
# each selected block fans out into its main run and one process per fold,
# spread over the devices of --gpus, so a five-fold block takes one training's
# time on a multi-GPU host or an NGT pod.
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
#   --folds <k>        local mode: k+1 detached processes per block instead of
#                      one queue entry, the main run with data.n_folds=1 and
#                      fold i with fold=i data.n_folds=k under <run dir>/fold_<i>,
#                      which is what src/eval.py merges when its
#                      checkpoint_manifest_path names <run dir>. Needs --gpus,
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
# Under --folds the launcher's own fold, data.n_folds and hydra.run.dir come
# last of all, so the layout the merge depends on cannot be overridden away.
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

# $1 = block index, $2 = label, $3 = run name, $4 = device token, $5 = command,
# $6 = the run's own overrides, appended behind the extras (see the header).
launch_one() {
    local idx="$1" label="$2" name="$3" device="$4" cmd="$5" suffix="$6" prefix="" full
    case "$device" in
        MIG-* | GPU-*) prefix="CUDA_VISIBLE_DEVICES=$device "; cmd=$(pin_gpu "$cmd" 0) ;;
        *) cmd=$(pin_gpu "$cmd" "$device") ;;
    esac
    full="$prefix$PYTHON $cmd${extra:+ $extra} $suffix"
    echo "[$idx/$n_total] $name: $full"
    if [ "$dry" -eq 1 ]; then return 0; fi
    # setsid: a process tree started under 'kubectl exec' dies with the session.
    setsid nohup bash -c "$full" > "$logdir/${idx}_${label}_${name}.log" 2>&1 &
    sleep "$STAGGER_SECONDS"
}

# $1 = block index, $2 = label, $3 = command. The main run and the k folds of
# one block, each on the next device of --gpus; the counter runs on across
# blocks, so two blocks over two devices load them evenly.
launch_folds() {
    local idx="$1" label="$2" cmd="$3" base j device
    base=$(field "$cmd" 'hydra\.run\.dir')
    [ -n "$base" ] || { echo "block $idx ($label) has no hydra.run.dir to put fold_<i> under" >&2; exit 1; }
    for j in $(seq 0 "$folds"); do
        device=${devices[$((n_launched % ${#devices[@]}))]}
        n_launched=$((n_launched + 1))
        if [ "$j" -eq 0 ]; then
            launch_one "$idx" "$label" main "$device" "$cmd" "data.n_folds=1"
        else
            launch_one "$idx" "$label" "fold_$j" "$device" "$cmd" \
                "fold=$j data.n_folds=$folds hydra.run.dir=$base/fold_$j"
        fi
    done
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
        echo "$n_sub of $n_total block(s) would run as $n_launched process(es); nothing was launched."
    else
        echo "$n_sub of $n_total block(s) running as $n_launched detached process(es) (logs: $logdir)."
        echo "Processes are setsid nohup'd, safe to log out."
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
