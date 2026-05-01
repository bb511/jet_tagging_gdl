#!/bin/bash
#SBATCH --job-name=model-sweep
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=20G
#SBATCH --time=60:00:00
#SBATCH --array=0-5
#SBATCH --output=logs/slurm/null
#SBATCH --error=logs/slurm/null

# ---------------------------------------------------------------
# Job array: runs method-specific experiments for a given model.
#
# Usage:
#   sbatch scripts/run_euler_array.sh deepsets augmentation
#   sbatch scripts/run_euler_array.sh irc_deepsets augmentation
#   sbatch scripts/run_euler_array.sh transformer augmentation
#   sbatch scripts/run_euler_array.sh deepsets symm_loss
#   sbatch scripts/run_euler_array.sh irc_deepsets symm_loss
#   sbatch scripts/run_euler_array.sh mlp symm_loss
#   sbatch scripts/run_euler_array.sh deepsets equivariance       # permutation test
#   sbatch scripts/run_euler_array.sh irc_deepsets equivariance   # no-augmentation baseline for invariant models
#   sbatch scripts/run_euler_array.sh nano_pelican equivariance   # Lorentz+perm invariant baseline
#
# Task-id mapping:
#   augmentation:        task 0-3  -> nconst=8, task 4-7  -> nconst=16, task 8-11 -> nconst=32
#   symm_loss:           task 0-2  -> nconst=8, task 3-5  -> nconst=16, task 6-8  -> nconst=32
#   equivariance (deepsets):         task 0 -> nconst=8, task 1 -> nconst=16, task 2 -> nconst=32
#   equivariance (invariant models): task 0 -> nconst=8, task 1 -> nconst=16, task 2 -> nconst=32
#   (nano_pelican uses a single config; nconst is set by the experiment YAML)
# ---------------------------------------------------------------

set -euo pipefail

# Model from command-line argument (required).
MODEL="${1:?Usage: sbatch scripts/run_euler_array.sh <model_name> <augmentation|symm_loss>}"
# Method from command-line argument (required).
METHOD_RAW="${2:?Usage: sbatch scripts/run_euler_array.sh <model_name> <augmentation|symm_loss>}"
METHOD="$(echo "$METHOD_RAW" | tr '[:upper:]' '[:lower:]')"
# Any extra Hydra overrides passed after model/method, e.g. trainer.max_epochs=1
EXTRA_ARGS=("${@:3}")

resolve_model_flag() {
    local model="$1"
    local experiment="$2"

    if [[ "$model" == "deepsets" ]]; then
        echo ""
        return
    fi

    # For models with per-nconst configs (mlp, irc_deepsets, etc.), extract nconst from experiment name
    if [[ "$model" == "mlp" || "$model" == "irc_deepsets" ]]; then
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
    # augmentation/hls4ml_nconst8_aug_none
    # augmentation/hls4ml_nconst8_aug_so2
    # augmentation/hls4ml_nconst8_aug_so3
    # augmentation/hls4ml_nconst8_aug_lorentz
    # augmentation/hls4ml_nconst16_aug_none
    # augmentation/hls4ml_nconst16_aug_so2
    # augmentation/hls4ml_nconst16_aug_so3
    # augmentation/hls4ml_nconst16_aug_lorentz
    # augmentation/hls4ml_nconst32_aug_none
    # augmentation/hls4ml_nconst32_aug_so2
    # augmentation/hls4ml_nconst32_aug_so3
    # augmentation/hls4ml_nconst32_aug_lorentz
)

# SYMM_LOSS_EXPERIMENTS=(
#     symm_loss/hls4ml_symm_loss_nconst8_so2
#     symm_loss/hls4ml_symm_loss_nconst8_so3
#     symm_loss/hls4ml_symm_loss_nconst8_lorentz
#     symm_loss/hls4ml_symm_loss_nconst16_so2
#     symm_loss/hls4ml_symm_loss_nconst16_so3
#     symm_loss/hls4ml_symm_loss_nconst16_lorentz
#     symm_loss/hls4ml_symm_loss_nconst32_so2
#     symm_loss/hls4ml_symm_loss_nconst32_so3
#     symm_loss/hls4ml_symm_loss_nconst32_lorentz
# )

EQUIVARIANCE_EXPERIMENTS=(
    equivariance/hls4ml_nconst8_permutation
    equivariance/hls4ml_nconst16_permutation
    equivariance/hls4ml_nconst32_permutation
)

# Baseline for inherently invariant/equivariant models: no augmentation, no symm_loss.
# Model is always provided via CLI (resolve_model_flag), not hardcoded in the experiment YAML.
EQUIVARIANCE_NONE_EXPERIMENTS=(
    equivariance/hls4ml_nconst8_none
    equivariance/hls4ml_nconst16_none
    equivariance/hls4ml_nconst32_none
)

if [[ "$METHOD" == "augmentation" ]]; then
    EXPERIMENTS=("${AUGMENTATION_EXPERIMENTS[@]}")
elif [[ "$METHOD" == "symm_loss" ]]; then
    EXPERIMENTS=("${SYMM_LOSS_EXPERIMENTS[@]}")
elif [[ "$METHOD" == "equivariance" ]]; then
    if [[ "$MODEL" == "deepsets" ]]; then
        EXPERIMENTS=("${EQUIVARIANCE_EXPERIMENTS[@]}")
    fi
else
    echo "Unsupported method '$METHOD_RAW'. Use one of: augmentation, symm_loss, equivariance"
    exit 1
fi

# If the job array is larger than the chosen method's experiment set, skip extras.
if (( SLURM_ARRAY_TASK_ID >= ${#EXPERIMENTS[@]} )); then
    echo "Task ${SLURM_ARRAY_TASK_ID} has no experiment for method '${METHOD}'. Skipping."
    echo "Hint: augmentation has 12 tasks (0-11), symm_loss has 9 (0-8), equivariance has 3 (0-2)."
    exit 0
fi


EXPERIMENT="${EXPERIMENTS[$SLURM_ARRAY_TASK_ID]}"

MODEL_FLAG=$(resolve_model_flag "$MODEL" "$EXPERIMENT")
RUN_NAME="${EXPERIMENT}_${METHOD}"

# Extract nconst from experiment string if present
NCONST=""
if [[ "$EXPERIMENT" =~ nconst([0-9]+) ]]; then
    NCONST="${BASH_REMATCH[1]}"
fi

# Create slurm log directory
mkdir -p logs/slurm/${SLURM_ARRAY_JOB_ID}

# Descriptive log file name: <model>_<experiment_basename>_<taskid>
EXPERIMENT_BASENAME=$(basename "$EXPERIMENT")
LOG_BASE="logs/slurm/${SLURM_ARRAY_JOB_ID}/${MODEL}_${EXPERIMENT_BASENAME}_${SLURM_ARRAY_TASK_ID}"
exec > "${LOG_BASE}.out" 2> "${LOG_BASE}.err"
# SLURM will handle stdout and stderr redirection

# Force a unique Hydra run dir per SLURM array task to avoid checkpoint collisions
# between concurrent tasks with different model shapes.
HYDRA_RUN_DIR="logs/train/runs/${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}_${MODEL}_${METHOD}_${EXPERIMENT_BASENAME}"

echo "============================================"
echo "Job ID:       $SLURM_ARRAY_JOB_ID"
echo "Task ID:      $SLURM_ARRAY_TASK_ID"
echo "Node:         $(hostname)"
echo "GPU:          $CUDA_VISIBLE_DEVICES"
echo "Model:        $MODEL"
echo "Method:       $METHOD"
echo "Experiment:   $EXPERIMENT"
echo "Nconst:       $NCONST"
echo "============================================"

# For transformer, override data.nconstituents if NCONST is set
if [[ "$MODEL" == "transformer" && -n "$NCONST" ]]; then
    srun python src/train.py $MODEL_FLAG experiment="$EXPERIMENT" +experiment_name="$RUN_NAME" trainer=gpu +data.nconstituents="$NCONST" +data.n_folds=5 hydra.run.dir="$HYDRA_RUN_DIR" "${EXTRA_ARGS[@]}"
else
    srun python src/train.py $MODEL_FLAG experiment="$EXPERIMENT" +experiment_name="$RUN_NAME" trainer=gpu data.n_folds=5 hydra.run.dir="$HYDRA_RUN_DIR" "${EXTRA_ARGS[@]}"
fi
