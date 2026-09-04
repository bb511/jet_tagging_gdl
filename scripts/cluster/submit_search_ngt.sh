#!/bin/bash
# Sweep one Optuna study with several concurrent workers on an NGT pod.
#
# The optuna sweeper opens its study with load_if_exists=True, so drivers that
# agree on the study name and the storage share one study: their trials
# accumulate and n_startup_trials is read from the study's own count, which is
# what makes a pool of short drivers equivalent to one long one. Two things must
# differ per worker, and supplying them is all this script does: the sampler
# seed, without which every worker draws the same random start-up points, and a
# tag, because logs/ on the pods is one NFS volume shared by all of them.
#
# The pod is the machine; there is no scheduler. NVIDIA_VISIBLE_DEVICES lists one
# UUID per MIG slice, or a single 'GPU-...' on a full-card pod, and CUDA
# enumerates only one slice per parent card, so trainer.devices=[i] cannot reach
# the others. A worker is pinned with CUDA_VISIBLE_DEVICES=<uuid> instead and
# keeps the catalogue's trainer.devices=[0].
#
# Usage (from the repository root, inside an NGT session):
#
#   # MIG pod: one worker per slice, 100 trials shared out between them
#   bash scripts/cluster/submit_search_ngt.sh --only nconst150 --workers 4 \
#       --trials 100 --seed 1234 --tag s1 scripts/hls4ml/runlorentznet_search.sh
#
#   # full-card pod: two workers time-sharing the one card
#   bash scripts/cluster/submit_search_ngt.sh --only nconst150 --workers 2 \
#       --trials 50 --seed 1238 --tag s6 scripts/hls4ml/runlorentznet_search.sh
#
#   --only <sel>    which block of the catalogue to sweep, passed through to
#                   submit_search.sh; required, since a pool sweeps one study
#   --gpus <list>   comma-separated indices into NVIDIA_VISIBLE_DEVICES; default
#                   is every device it lists, in order
#   --workers <k>   drivers to start (default: one per chosen device). More
#                   workers than devices take them round-robin, which is how a
#                   full card is time-shared by two
#   --trials <T>    trials in total, split into k near-equal parts; required,
#                   because hydra.sweeper.n_trials is per driver and accumulates
#   --seed <s>      sampler seed of worker 0; worker j gets s+j. Required: it is
#                   the one setting the workers must not share
#   --tag <t>       names the logs and the sweep directories (default: hostname)
#   --dry-run       print each worker's command, launch nothing
#
# Anything after the catalogue file is appended to every worker as an override.
#
# A process tree started under 'kubectl exec' dies with the exec session, so each
# worker is launched with setsid nohup and outlives the disconnection.
set -eu

# shellcheck source=scripts/cluster/lib.sh
. "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

# Workers of one study race to create their study row in the same sqlite file.
STAGGER_SECONDS=3

only=""
gpus=""
workers=""
trials=""
seed=""
tag=""
dry=0

while [ $# -gt 0 ]; do
    case "$1" in
        --only) only="${2:?--only needs a value}"; shift 2 ;;
        --gpus) gpus="${2:?--gpus needs a list}"; shift 2 ;;
        --workers) workers="${2:?--workers needs a number}"; shift 2 ;;
        --trials) trials="${2:?--trials needs a number}"; shift 2 ;;
        --seed) seed="${2:?--seed needs a number}"; shift 2 ;;
        --tag) tag="${2:?--tag needs a value}"; shift 2 ;;
        --dry-run) dry=1; shift ;;
        --) shift; break ;;
        -*) echo "unknown option: $1" >&2; exit 2 ;;
        *) break ;;
    esac
done

usage="usage: submit_search_ngt.sh --only <sel> --trials <T> --seed <s> [--gpus i,j]"
usage="$usage [--workers k] [--tag t] [--dry-run] <run*_search.sh> [overrides...]"
file=${1:?$usage}
shift || true

require_repo_root
[ -f "$file" ] || { echo "no such file: $file" >&2; exit 1; }
[ -n "$only" ] || { echo "--only is required: a worker pool sweeps one study" >&2; exit 2; }
[ -n "$trials" ] || { echo "--trials is required: it is split between the workers" >&2; exit 2; }
[ -n "$seed" ] || { echo "--seed is required: one seed for all is one search" >&2; exit 2; }
case "$trials$seed${workers:-0}" in
    *[!0-9]*) echo "--trials, --seed and --workers take non-negative integers" >&2; exit 2 ;;
esac
[ -n "$tag" ] || tag=$(hostname)

devices=${NVIDIA_VISIBLE_DEVICES:-}
if [ -z "$devices" ]; then
    if [ "$dry" -eq 0 ]; then
        echo "error: NVIDIA_VISIBLE_DEVICES is empty - this is not an NGT pod." >&2
        exit 1
    fi
    # A dry run off the pod still has to show the shape of the launch.
    n=${workers:-1}
    if [ "$n" -lt 1 ]; then n=1; fi
    for i in $(seq 0 $((n - 1))); do devices="${devices:+$devices,}<gpu$i>"; done
fi

IFS=',' read -r -a all_devices <<< "$devices"
declare -a chosen
if [ -n "$gpus" ]; then
    IFS=',' read -r -a wanted <<< "$gpus"
    for i in "${wanted[@]}"; do
        case "$i" in '' | *[!0-9]*) echo "bad --gpus index '$i'" >&2; exit 2 ;; esac
        if [ "$i" -ge "${#all_devices[@]}" ]; then
            echo "--gpus index $i is past NVIDIA_VISIBLE_DEVICES, which lists" \
                "${#all_devices[@]} device(s)" >&2
            exit 2
        fi
        chosen+=("${all_devices[$i]}")
    done
else
    chosen=("${all_devices[@]}")
fi

n_devices=${#chosen[@]}
[ -n "$workers" ] || workers=$n_devices
if [ "$workers" -lt 1 ]; then echo "--workers must be at least 1" >&2; exit 2; fi
if [ "$workers" -gt "$trials" ]; then
    echo "--workers $workers exceeds --trials $trials: a worker without a trial exits at once" >&2
    exit 2
fi

# The first T mod k workers carry the remainder, so no worker is left idle.
per_worker=$((trials / workers))
remainder=$((trials % workers))
trials_of() {
    if [ "$1" -lt "$remainder" ]; then echo $((per_worker + 1)); else echo "$per_worker"; fi
}

split=""
for j in $(seq 0 $((workers - 1))); do split="${split:+$split/}$(trials_of "$j")"; done

logdir="logs/searches/ngt"
[ "$dry" -eq 1 ] || mkdir -p "$logdir"

echo "tag=$tag  workers=$workers  devices=[$(IFS=,; printf '%s' "${chosen[*]}")]" \
    "trials=$trials ($split)  seeds=$seed..$((seed + workers - 1))"
echo

for j in $(seq 0 $((workers - 1))); do
    device=${chosen[$((j % n_devices))]}
    worker_tag="${tag}_w${j}"
    log="$logdir/${worker_tag}.log"
    worker=(scripts/cluster/submit_search.sh --only "$only" --tag "$worker_tag" "$file" "$@"
        "hydra.sweeper.n_trials=$(trials_of "$j")" "hydra.sweeper.sampler.seed=$((seed + j))")

    printf 'CUDA_VISIBLE_DEVICES=%s setsid nohup bash %s > %s 2>&1 &\n' \
        "$device" "${worker[*]}" "$log"
    if [ "$dry" -eq 1 ]; then
        CUDA_VISIBLE_DEVICES="$device" bash "${worker[0]}" --dry-run "${worker[@]:1}"
        continue
    fi
    CUDA_VISIBLE_DEVICES="$device" setsid nohup bash "${worker[@]}" > "$log" 2>&1 &
    sleep "$STAGGER_SECONDS"
done

echo
if [ "$dry" -eq 1 ]; then
    echo "$workers worker(s) would start; nothing was launched."
    exit 0
fi
echo "$workers worker(s) running detached (launch logs: $logdir)."
echo "Workers are setsid nohup'd, safe to leave the pod."
echo "Progress: python3 scripts/cluster/sweep_status.py"
