#!/bin/bash
# Launch the Optuna drivers of a run*_search.sh catalogue.
#
# A search block already carries '-m' and its sweeper settings; this adds only
# what the catalogue does not carry: the GPU, a private sweep directory and, with
# --cluster, the submitit launcher. Each block becomes one long-lived nohup'd
# driver, so without --cluster every block in the file starts on this host at
# once: select a few with --only and pin them with --gpu, one invocation per GPU.
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
#   --tag <t>        suffix for this driver's log name and sweep directory. One
#                    study may be swept by several drivers at once (the optuna
#                    sweeper opens its study with load_if_exists), and on the NGT
#                    pods logs/ is one shared volume, so without a tag per driver
#                    the second worker rotates the first one's live log, and two
#                    starting in the same second share a sweep directory as well.
#                    See submit_search_ngt.sh.
#
# Extra hydra overrides after the file name are appended to every driver; hydra
# takes the last of a duplicated override, so they win over the file's value.
#
# A block that selects a feature set with data.features must name its study
# database with hydra.sweeper.storage too, or the whole file is refused before
# anything is created or launched: see check_features_storage below.
set -eu

# shellcheck source=scripts/cluster/lib.sh
. "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

# Slurm time limit per trial: covers a 100-epoch trial at nconst=150, and is
# also clariden's `normal` partition cap, so a longer trial is killed.
TIMEOUT_MIN=720

only=""
dry=0
list=0
gpu=""
cluster=""
n_jobs=6
hf_cache=""
tag=""

while [ $# -gt 0 ]; do
    case "$1" in
        --dry-run) dry=1; shift ;;
        --list) list=1; shift ;;
        --only) only="${2:?--only needs a value}"; shift 2 ;;
        --gpu) gpu="${2:?--gpu needs an index}"; shift 2 ;;
        --cluster) cluster="${2:?--cluster needs a name}"; shift 2 ;;
        --n-jobs) n_jobs="${2:?--n-jobs needs a number}"; shift 2 ;;
        --hf-cache) hf_cache="${2:?--hf-cache needs a path}"; shift 2 ;;
        --tag) tag="${2:?--tag needs a value}"; shift 2 ;;
        --) shift; break ;;
        -*) echo "unknown option: $1" >&2; exit 2 ;;
        *) break ;;
    esac
done

usage="usage: submit_search.sh [--list|--dry-run] [--only sel] [--gpu i] [--cluster c]"
usage="$usage [--n-jobs n] [--hf-cache p] [--tag t] <run*_search.sh> [overrides...]"
file=${1:?$usage}
shift || true
extra="$*"

require_repo_root
[ -f "$file" ] || { echo "no such file: $file" >&2; exit 1; }

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

# PYTHON=/path/to/env/bin/python3 skips activation; --dry-run needs no env.
PYTHON=${PYTHON:-python3}
[ "$dry" -eq 1 ] || require_imports "$PYTHON" "hydra, optuna" \
    "conda activate jet_tagging_gdl" \
    "(or re-run with PYTHON=/path/to/env/bin/python3)"

stem="$(basename "$(dirname "$file")")_$(basename "$file" .sh)"
logdir="logs/searches/$stem"
sweep_ts=$(date +%Y-%m-%d_%H-%M-%S)
[ "$dry" -eq 1 ] || mkdir -p "$logdir"

# The storage URL of one config: the hparams_search config a block selects, or the
# base config that one inherits.
storage_in_file() {
    grep -oE "^[[:space:]]*storage:[[:space:]]*\S+" "$1" | head -1 | awk '{print $2}' || true
}

# A block that names its own storage wins over its config, as it does for every
# duplicated hydra override: a feature-set catalogue names its database on the
# command line, one study name serving every feature set.
storage_of() {
    local hp f s d base
    # Quoted twice in the catalogues: hydra's grammar wants the '=' of '?timeout=600'
    # inside quotes of its own, and the driver is launched through bash -c, which
    # strips the outer pair and hands the inner pair through.
    s=$(field "$1" 'hydra\.sweeper\.storage')
    s=${s#\"}; s=${s%\"}; s=${s#\'}; s=${s%\'}
    if [ -n "$s" ]; then printf "%s" "$s"; return 0; fi
    hp=$(field "$1" 'hparams_search')
    f="configs/hparams_search/${hp}.yaml"
    if [ -z "$hp" ] || [ ! -f "$f" ]; then return 0; fi
    s=$(storage_in_file "$f")
    if [ -z "$s" ]; then
        # A defaults entry is group-relative or from the group root; try both.
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

# A block that chooses a feature set but leaves the storage to the config would
# append its trials to the four-momentum study of the same name: the study name is
# built from the regime alone and knows nothing of data.features.
check_features_storage() {
    if [ -z "$(field "$1" 'data\.features')" ]; then return 0; fi
    if [ -n "$(field "$1" 'hydra\.sweeper\.storage')" ]; then return 0; fi
    echo "error: $(block_label "$1") sets data.features but no hydra.sweeper.storage," >&2
    echo "       so its trials would silently join the px_py_pz_e study of the same" >&2
    echo "       name. Give the block (or the command line) a storage URL of its own," >&2
    echo "       e.g. hydra.sweeper.storage='sqlite:///logs/optuna/hls4ml/<label>/<model>.db?timeout=600'." >&2
    exit 2
}

# Checked over every selected block first, so an unpaired one refuses the file
# rather than leaving half its drivers running against the wrong database.
i=0
while IFS= read -r cmd; do
    i=$((i + 1))
    selected "$i" "$(block_label "$cmd")" || continue
    check_features_storage "$cmd${extra:+ $extra}"
done <<EOF
$cmds
EOF

# Create each sqlite study database serially before any driver starts: drivers
# racing optuna's alembic bootstrap on a missing file die seconds in with
# 'UNIQUE constraint failed: alembic_version.version_num'. Idempotent.
if [ "$dry" -eq 0 ]; then
    printf "%s\n" "$cmds" | while IFS= read -r c; do storage_of "$c"; echo; done |
        sort -u | grep -v '^$' | while IFS= read -r url; do
        case "$url" in
            sqlite:///*)
                path=${url#sqlite:///}
                # SQLAlchemy strips the URLs' '?timeout=600' busy timeout
                # itself; this check needs the bare path to find the file.
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

    # Private sweep dir per driver: hydra's second-resolution timestamp collides
    # when several start together, merging their outputs.
    sweep=" hydra.sweep.dir=logs/searches/multiruns/${sweep_ts}_${label}${tag:+_$tag}"
    full="$PYTHON $cmd$launcher$sweep${extra:+ $extra}"

    n_sub=$((n_sub + 1))
    if [ "$dry" -eq 1 ]; then
        echo "[$i/$n_total] $full"
        echo
        continue
    fi
    echo "[$i/$n_total] starting driver: $label"
    # Keep the previous log: it is the evidence of why the driver died.
    log="$logdir/${i}_${label}${tag:+_$tag}.log"
    if [ -f "$log" ]; then mv "$log" "$log.$(date +%Y%m%d-%H%M%S)"; fi
    nohup bash -c "$full" > "$log" 2>&1 &
    # Drivers sharing one sqlite file contend while creating their study rows.
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
