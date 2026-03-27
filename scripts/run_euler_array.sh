#!/bin/bash
#SBATCH --job-name=model-sweep
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=60G
#SBATCH --time=08:00:00
#SBATCH --array=0-11
#SBATCH --output=logs/slurm/%A_%a.out
#SBATCH --error=logs/slurm/%A_%a.err

# ---------------------------------------------------------------
# Job array: runs all 12 experiments for a given model
#
# Usage:
#   sbatch scripts/run_euler_array.sh deepsets_32
#   sbatch scripts/run_euler_array.sh transformer
# ---------------------------------------------------------------

set -euo pipefail

# Model from command-line argument (required)
MODEL="${1:?Usage: sbatch scripts/run_euler_array.sh <model_name>}"

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
)

EXPERIMENT="${EXPERIMENTS[$SLURM_ARRAY_TASK_ID]}"

# Project root
export PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# Create slurm log directory
mkdir -p logs/slurm

# Activate your environment
source "$PROJECT_ROOT/HYDRA/bin/activate"

echo "============================================"
echo "Job ID:       $SLURM_ARRAY_JOB_ID"
echo "Task ID:      $SLURM_ARRAY_TASK_ID"
echo "Node:         $(hostname)"
echo "GPU:          $CUDA_VISIBLE_DEVICES"
echo "Model:        $MODEL"
echo "Experiment:   $EXPERIMENT"
echo "Project root: $PROJECT_ROOT"
echo "============================================"

python src/train.py model="$MODEL" experiment="$EXPERIMENT" trainer=gpu logger=mlflow
