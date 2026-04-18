#!/bin/bash
#SBATCH --job-name=model-sweep
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=200G
#SBATCH --time=60:00:00
#SBATCH --array=0-0
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

# ---------------------------------------------------------------
# Job array: runs all 12 experiments for a given model
#
# Usage:
#   sbatch scripts/run_euler_array.sh deepsets     # uses model from experiment config
#   sbatch scripts/run_euler_array.sh transformer  # overrides model to transformer
# ---------------------------------------------------------------

set -euo pipefail

# Model from command-line argument (required)
MODEL="${1:?Usage: sbatch scripts/run_euler_array.sh <model_name>}"

# Build model override flag
# "deepsets" -> no override (experiment configs already pick deepsets_8/16/32)
# "mlp" -> resolve to mlp_8/mlp_16/mlp_32 based on experiment name
# anything else -> override the model on the command line as-is
resolve_model_flag() {
    local model="$1"
    local experiment="$2"

    if [[ "$model" == "deepsets" ]]; then
        echo ""
        return
    fi

    # For models with per-nconst configs (mlp, etc.), extract nconst from experiment name
    if [[ "$model" == "mlp" ]]; then
        local nconst
        nconst=$(echo "$experiment" | grep -oP 'nconst\K[0-9]+')
        if [[ -n "$nconst" ]]; then
            echo "model=${model}_${nconst}"
            return
        fi
    fi

    echo "model=$model"
}

# --- Experiments (nconst x augmentation) ---
EXPERIMENTS=(
    # hls4ml_nconst8_aug_none
    hls4ml_nconst8_aug_so2
    # hls4ml_nconst8_aug_so3
    # hls4ml_nconst8_aug_lorentz
    # hls4ml_nconst16_aug_none
    # hls4ml_nconst16_aug_so2
    # hls4ml_nconst16_aug_so3
    # hls4ml_nconst16_aug_lorentz
    # hls4ml_nconst32_aug_none
    # hls4ml_nconst32_aug_so2
    # hls4ml_nconst32_aug_so3
    # hls4ml_nconst32_aug_lorentz
)

EXPERIMENT="${EXPERIMENTS[$SLURM_ARRAY_TASK_ID]}"
MODEL_FLAG=$(resolve_model_flag "$MODEL" "$EXPERIMENT")

# Project root
# export PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# cd "$PROJECT_ROOT"

# Create slurm log directory per job
mkdir -p "logs/slurm/${SLURM_ARRAY_JOB_ID}"

# Descriptive log file name: <model>_<experiment>_<taskid>
LOG_BASE="logs/slurm/${SLURM_ARRAY_JOB_ID}/${MODEL}_${EXPERIMENT}_${SLURM_ARRAY_TASK_ID}"
exec > "${LOG_BASE}.out" 2>&1

echo "============================================"
echo "Job ID:       $SLURM_ARRAY_JOB_ID"
echo "Task ID:      $SLURM_ARRAY_TASK_ID"
echo "Node:         $(hostname)"
echo "GPU:          $CUDA_VISIBLE_DEVICES"

echo "Model:        $MODEL"
echo "Experiment:   $EXPERIMENT"
# echo "Project root: $PROJECT_ROOT"
echo "============================================"

METHOD="equivariant_model"

srun python src/train.py $MODEL_FLAG $METHOD experiment="$EXPERIMENT" experiment_name="$EXPERIMENT" trainer=gpu n_folds=5
