#!/bin/bash
# One-time project bootstrap: create the working directories and write .env.
#
# configs/paths/default.yaml reads ${oc.env:PROJECT_ROOT} from .env, so no
# config in the repository can be composed until this has run. Run it from
# anywhere; it always sets the repository root up, not the current directory.
#
#   bash scripts/setup.sh
#
# To keep the heavy directories on another filesystem, edit RES_DIR in .env
# afterwards and run scripts/symbolink.sh, which replaces them with symlinks.
set -euo pipefail

cd "$(dirname "$0")/.."

# data/ holds the record's per-nconstituents caches, results/ the tables the
# ResultsTableCallback writes; logs/ and outputs/ are the roots hydra is given.
# There is no checkpoints/: checkpoints live under the run's own output dir.
for folder in data logs outputs results; do
    if [ -d "$folder" ]; then
        echo "Folder already exists: $folder"
    else
        mkdir -p "$folder"
        echo "Created folder: $folder"
    fi
done

# Never silently replace an existing .env: it may already point RES_DIR at a
# scratch filesystem, and losing that quietly means the next run fills the home
# directory with checkpoints.
env_file=".env"
if [ -e "$env_file" ]; then
    echo
    echo "$PWD/$env_file already exists - left untouched."
    echo "Delete it first if you want a fresh one."
    exit 0
fi

# MLFLOW_ALLOW_FILE_STORE: mlflow 3 refuses the file: tracking store that
# configs/logger/mlflow.yaml points at unless this is set, and every experiment
# config selects that logger.
# HF_HUB_CACHE is read by huggingface_hub at import, and both entry points load
# .env through rootutils before importing it. It holds the 3.1 GB record
# snapshot; the loader's restricted, normalised caches are separate and land in
# data/hf/nconst_<N>/. On a machine that already has a hub cache, point it there
# rather than downloading a second copy.
cat > "$env_file" << EOL
PROJECT_ROOT="."
RES_DIR="." # set to the desired location, then run scripts/symbolink.sh
DATA_DIR="\${RES_DIR}/data"
LOG_DIR="\${RES_DIR}/logs"
OUTPUT_DIR="\${RES_DIR}/outputs"
RESULTS_DIR="\${RES_DIR}/results"
HF_HUB_CACHE="\${RES_DIR}/data/hf_hub"
MLFLOW_ALLOW_FILE_STORE="true"
EOL

echo
echo ".env file created successfully at $PWD/$env_file"
