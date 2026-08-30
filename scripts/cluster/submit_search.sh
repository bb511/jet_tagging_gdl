#!/bin/bash
# Launch the Optuna drivers of a run*_search.sh catalogue.
#
# A search block already carries '-m' and its sweeper settings, so this adds
# only what the catalogue cannot know: which GPU to pin to, a private sweep
# directory, and, with --cluster, the submitit launcher. Each block is one
# long-lived driver that keeps n_jobs trials in flight until its study reaches
# n_trials, so the drivers are nohup'd and this command returns once they are up.
#
# Without --cluster the drivers run here, all 21 of them at once by default,
# which is not what a four-GPU host wants: select a few blocks with --only and
# pin them with --gpu, one invocation per GPU.
#
# Usage (from the repository root):
#   bash scripts/cluster/submit_search.sh --list    scripts/hls4ml/rundeepsets_search.sh
#   bash scripts/cluster/submit_search.sh --dry-run scripts/hls4ml/rundeepsets_search.sh
#   bash scripts/cluster/submit_search.sh --only nconst16 --gpu 0 scripts/hls4ml/rundeepsets_search.sh
#   bash scripts/cluster/submit_search.sh --cluster clariden --n-jobs 6 scripts/hls4ml/rundeepsets_search.sh
#
#   --list           list the blocks in the catalogue and exit
#   --dry-run        print the full commands without running anything; works off
#                    the cluster, so it needs neither the env nor the data
#   --only <sel>     comma-separated label substrings ('symm_so2', 'nconst150')
#                    or 1-based indices; default is every block in the file
#   --gpu <i>        rewrite trainer.devices to this GPU (local runs only)
#   --cluster <c>    euler | clariden: prepend the submitit launcher
#   --n-jobs <n>     trials in flight per driver under --cluster (default 6)
#   --hf-cache <p>   value for HF_HUB_CACHE; default is the one .env exports
#
# Any extra hydra overrides after the file name are appended to every driver.
# Hydra takes the last of a duplicated override, so appending wins over the
# value baked into the file.
set -eu

# shellcheck source=scripts/cluster/lib.sh
. "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

# Slurm time limit per trial. A 100-epoch trial at nconst=150 is the long case;
# 720 min is also the `normal` partition cap on clariden, so a trial that needs
# longer is killed rather than requeued.
TIMEOUT_MIN=720

only=""
dry=0
list=0
gpu=""
cluster=""
n_jobs=6
hf_cache=""

while [ $# -gt 0 ]; do
    case "$1" in
        --dry-run) dry=1; shift ;;
        --list) list=1; shift ;;
        --only) only="${2:?--only needs a value}"; shift 2 ;;
        --gpu) gpu="${2:?--gpu needs an index}"; shift 2 ;;
        --cluster) cluster="${2:?--cluster needs a name}"; shift 2 ;;
        --n-jobs) n_jobs="${2:?--n-jobs needs a number}"; shift 2 ;;
        --hf-cache) hf_cache="${2:?--hf-cache needs a path}"; shift 2 ;;
        --) shift; break ;;
        -*) echo "unknown option: $1" >&2; exit 2 ;;
        *) break ;;
    esac
done

usage="usage: submit_search.sh [--list|--dry-run] [--only sel] [--gpu i] [--cluster c]"
usage="$usage [--n-jobs n] [--hf-cache p] <run*_search.sh> [overrides...]"
file=${1:?$usage}
shift || true
extra="$*"

require_repo_root
[ -f "$file" ] || { echo "no such file: $file" >&2; exit 1; }

# Parsed once; every loop below reads from this list.
cmds=$(extract_cmds "$file")
n_total=$(printf "%s\n" "$cmds" | grep -c . || true)
if [ "$n_total" -eq 0 ]; then
    echo "No search commands found in $file" >&2
    exit 1
fi

if [ "$list" -eq 1 ]; then
    printf "%-3s %-34s %-30s %-9s %s\n" "#" "block" "hparams_search" "n_trials" "max_epochs"
    i=0
    while IFS= read -r cmd; do
        i=$((i + 1))
        printf "%-3s %-34s %-30s %-9s %s\n" "$i" \
            "$(block_label "$cmd")" \
            "$(field "$cmd" 'hparams_search')" \
            "$(field "$cmd" 'hydra\.sweeper\.n_trials')" \
            "$(field "$cmd" 'trainer\.max_epochs')"
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
    # The block supplies the '-m'; cluster_prefix supplies only the overrides.
    launcher=" $(cluster_prefix "$cluster" "$TIMEOUT_MIN") hydra.sweeper.n_jobs=$n_jobs"
    command -v sbatch >/dev/null 2>&1 || echo "warning: sbatch not on PATH - are you on $cluster?" >&2
fi

# Override with PYTHON=/path/to/env/bin/python3 to skip activation entirely.
# Skipped for --dry-run, which is meant to be usable off the cluster.
PYTHON=${PYTHON:-python3}
[ "$dry" -eq 1 ] || require_imports "$PYTHON" "hydra, optuna" \
    "conda activate jet_tagging_gdl" \
    "(or re-run with PYTHON=/path/to/env/bin/python3)"

stem="$(basename "$(dirname "$file")")_$(basename "$file" .sh)"
logdir="logs/searches/$stem"
sweep_ts=$(date +%Y-%m-%d_%H-%M-%S)
[ "$dry" -eq 1 ] || mkdir -p "$logdir"

# The storage URL is not on the CLI: it lives in the hparams_search config the
# block selects, and for the _scl_ variants in the base config it inherits.
storage_in_file() {
    grep -oE "^[[:space:]]*storage:[[:space:]]*\S+" "$1" | head -1 | awk '{print $2}' || true
}

storage_of() {
    local hp f s d base
    hp=$(field "$1" 'hparams_search')
    f="configs/hparams_search/${hp}.yaml"
    if [ -z "$hp" ] || [ ! -f "$f" ]; then return 0; fi
    s=$(storage_in_file "$f")
    if [ -z "$s" ]; then
        # A defaults entry is written either group-relative ('deepsets_optuna')
        # or from the group root ('hls4ml/deepsets_optuna'); try both.
        while IFS= read -r d; do
            for base in "$(dirname "$f")" "configs/hparams_search"; do
                if [ -f "$base/$d.yaml" ]; then s=$(storage_in_file "$base/$d.yaml"); fi
                if [ -n "$s" ]; then break 2; fi
            done
        done <<EOF
$(grep -oE '^[[:space:]]*-[[:space:]]*[A-Za-z0-9_/]+$' "$f" | awk '{print $2}')
EOF
    fi
    printf "%s" "$s"
}

# Create each sqlite study database, serially, before any driver starts.
# Drivers launched together against a not-yet-existing file all run optuna's
# alembic bootstrap at once and lose the race, observed as 'UNIQUE constraint
# failed: alembic_version.version_num' and 'database is locked', which kills
# the driver seconds in. Idempotent: on an existing database this only opens it.
if [ "$dry" -eq 0 ]; then
    printf "%s\n" "$cmds" | while IFS= read -r c; do storage_of "$c"; echo; done |
        sort -u | grep -v '^$' | while IFS= read -r url; do
        case "$url" in
            sqlite:///*)
                path=${url#sqlite:///}
                # The URLs carry a '?timeout=600' busy timeout (sqlite raises
                # 'database is locked' immediately without one). SQLAlchemy
                # strips the query itself; this check needs the bare path or it
                # would never find an existing database.
                path=${path%%\?*}
                if [ ! -f "$path" ]; then
                    mkdir -p "$(dirname "$path")"
                    echo "initialising study database: $path"
                    "$PYTHON" -c 'import optuna,sys; optuna.storages.RDBStorage(sys.argv[1])' "$url"
                fi ;;
        esac
    done
fi

i=0
n_sub=0
while IFS= read -r cmd; do
    i=$((i + 1))
    label=$(block_label "$cmd")
    selected "$i" "$label" || continue

    if [ -n "$cluster" ]; then
        # One GPU per slurm job, whatever the catalogue says.
        cmd=$(pin_gpu "$cmd" 0)
    elif [ -n "$gpu" ]; then
        cmd=$(pin_gpu "$cmd" "$gpu")
    fi

    # Private sweep dir per driver: hydra's default second-resolution timestamp
    # collides when several drivers start together, merging their outputs (and
    # the slurm launchers nest submitit_folder under hydra.sweep.dir).
    sweep=" hydra.sweep.dir=logs/searches/multiruns/${sweep_ts}_${label}"
    full="$PYTHON $cmd$launcher$sweep${extra:+ $extra}"

    n_sub=$((n_sub + 1))
    if [ "$dry" -eq 1 ]; then
        echo "[$i/$n_total] $full"
        echo
        continue
    fi
    echo "[$i/$n_total] starting driver: $label"
    # Keep the previous log: relaunching a driver that died is the main reason
    # to re-run this script, and truncating would destroy the evidence of why.
    log="$logdir/${i}_${label}.log"
    if [ -f "$log" ]; then mv "$log" "$log.$(date +%Y%m%d-%H%M%S)"; fi
    nohup bash -c "$full" > "$log" 2>&1 &
    # Drivers sharing one sqlite file still contend while creating their study
    # rows; stagger them rather than starting several in the same second.
    sleep 5
done <<EOF
$cmds
EOF

if [ "$n_sub" -eq 0 ]; then
    echo "nothing matched --only '$only'" >&2
    exit 1
fi

if [ "$dry" -eq 0 ]; then
    echo
    echo "$n_sub driver(s) running detached (logs: $logdir)."
    echo "Drivers are nohup'd, safe to log out."
    echo "Progress: python3 scripts/cluster/sweep_status.py"
    echo "Drivers:  pgrep -af 'src/train.py' | grep hls4ml"
fi
