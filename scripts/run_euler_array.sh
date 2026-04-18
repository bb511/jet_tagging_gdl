#!/bin/bash
#SBATCH --job-name=model-sweep
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=200G
#SBATCH --time=60:00:00
#SBATCH --array=0-11
#SBATCH --output=logs/slurm/null
#SBATCH --error=logs/slurm/null

# ---------------------------------------------------------------
# Job array: runs method-specific experiments for a given model.
#
# Usage:
#   sbatch scripts/run_euler_array.sh deepsets_32
#   sbatch scripts/run_euler_array.sh transformer
#   sbatch scripts/run_euler_array.sh deepsets augmentation
#   sbatch scripts/run_euler_array.sh deepsets symm_loss
#   sbatch scripts/run_euler_array.sh mlp symm_loss
# ---------------------------------------------------------------

set -euo pipefail

# Model from command-line argument (required).
MODEL="${1:?Usage: sbatch scripts/run_euler_array.sh <model_name> <augmentation|symm_loss>}"
# Method from command-line argument (required).
METHOD_RAW="${2:?Usage: sbatch scripts/run_euler_array.sh <model_name> <augmentation|symm_loss>}"
METHOD="$(echo "$METHOD_RAW" | tr '[:upper:]' '[:lower:]')"

<<<<<<< Updated upstream
# --- Experiments (nconst x augmentation) ---
EXPERIMENTS=(
    hls4ml_nconst8_aug_none
    hls4ml_nconst8_aug_so2
    hls4ml_nconst8_aug_so3
    hls4ml_nconst8_aug_lorentz
    hls4ml_nconst16_aug_none
    hls4ml_nconst16_aug_so2
    hls4ml_nconst16_aug_so3
    hls4ml_nconst16_aug_lorentz
    hls4ml_nconst32_aug_none
    hls4ml_nconst32_aug_so2
    hls4ml_nconst32_aug_so3
    hls4ml_nconst32_aug_lorentz
=======
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

# --- Experiment sets ---
AUGMENTATION_EXPERIMENTS=(
    augmentation/hls4ml_nconst8_aug_none
    augmentation/hls4ml_nconst8_aug_so2
    augmentation/hls4ml_nconst8_aug_so3
    augmentation/hls4ml_nconst8_aug_lorentz
    augmentation/hls4ml_nconst16_aug_none
    augmentation/hls4ml_nconst16_aug_so2
    augmentation/hls4ml_nconst16_aug_so3
    augmentation/hls4ml_nconst16_aug_lorentz
    augmentation/hls4ml_nconst32_aug_none
    augmentation/hls4ml_nconst32_aug_so2
    augmentation/hls4ml_nconst32_aug_so3
    augmentation/hls4ml_nconst32_aug_lorentz
)

SYMM_LOSS_EXPERIMENTS=(
    symm_loss/hls4ml_symm_loss_nconst8_so2
    symm_loss/hls4ml_symm_loss_nconst8_so3
    symm_loss/hls4ml_symm_loss_nconst8_lorentz
    symm_loss/hls4ml_symm_loss_nconst16_so2
    symm_loss/hls4ml_symm_loss_nconst16_so3
    symm_loss/hls4ml_symm_loss_nconst16_lorentz
    symm_loss/hls4ml_symm_loss_nconst32_so2
    symm_loss/hls4ml_symm_loss_nconst32_so3
    symm_loss/hls4ml_symm_loss_nconst32_lorentz
)

if [[ "$METHOD" == "augmentation" ]]; then
    EXPERIMENTS=("${AUGMENTATION_EXPERIMENTS[@]}")
elif [[ "$METHOD" == "symm_loss" ]]; then
    EXPERIMENTS=("${SYMM_LOSS_EXPERIMENTS[@]}")
else
    echo "Unsupported method '$METHOD_RAW'. Use one of: augmentation, symm_loss"
    exit 1
fi

# If the job array is larger than the chosen method's experiment set, skip extras.
if (( SLURM_ARRAY_TASK_ID >= ${#EXPERIMENTS[@]} )); then
    echo "Task ${SLURM_ARRAY_TASK_ID} has no experiment for method '${METHOD}'. Skipping."
    exit 0
fi

EXPERIMENT="${EXPERIMENTS[$SLURM_ARRAY_TASK_ID]}"

# Project root
export PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
=======
MODEL_FLAG=$(resolve_model_flag "$MODEL" "$EXPERIMENT")
RUN_NAME="${EXPERIMENT}_${METHOD}"

# Create slurm log directory
mkdir -p logs/slurm

# Activate your environment
source "$PROJECT_ROOT/HYDRA/bin/activate"

# Descriptive log file name: <model>_<experiment_basename>_<taskid>
EXPERIMENT_BASENAME=$(basename "$EXPERIMENT")
LOG_BASE="logs/slurm/${SLURM_ARRAY_JOB_ID}/${MODEL}_${METHOD}_${EXPERIMENT_BASENAME}_${SLURM_ARRAY_TASK_ID}"
exec > "${LOG_BASE}.out" 2> "${LOG_BASE}.err"
# SLURM will handle stdout and stderr redirection

echo "============================================"
echo "Job ID:       $SLURM_ARRAY_JOB_ID"
echo "Task ID:      $SLURM_ARRAY_TASK_ID"
echo "Node:         $(hostname)"
echo "GPU:          $CUDA_VISIBLE_DEVICES"
echo "Model:        $MODEL"
echo "Method:       $METHOD"
echo "Experiment:   $EXPERIMENT"
echo "Project root: $PROJECT_ROOT"
echo "============================================"

python src/train.py model="$MODEL" experiment="$EXPERIMENT" trainer=gpu logger=mlflow
echo "============================================"

srun python src/train.py $MODEL_FLAG experiment="$EXPERIMENT" experiment_name="$RUN_NAME" trainer=gpu logger=mlflow n_folds=5 "${DATA_OVERRIDES[@]}"