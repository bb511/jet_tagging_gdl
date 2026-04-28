#!/bin/bash
#SBATCH --job-name=deepsets
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G
#SBATCH --time=08:00:00
#SBATCH --output=logs/slurm/%j.out
#SBATCH --error=logs/slurm/%j.err

# ---------------------------------------------------------------
# Single experiment on ETH Euler

# Usage:
#   sbatch scripts/run_euler.sh transformer augmentation/hls4ml_nconst16_aug_so2
#   sbatch scripts/run_euler.sh irc_deepsets_32 augmentation/hls4ml_nconst32_aug_so2
#   sbatch scripts/run_euler.sh nano_pelican equivariance/hls4ml_nconst8_none
#   sbatch scripts/run_euler.sh nano_pelican equivariance/hls4ml_nconst32_none
#   sbatch scripts/run_euler.sh                          # uses defaults
# ---------------------------------------------------------------

set -euo pipefail

# Model name (first argument, or default)
MODEL="${1:-deepsets_32}"
# Experiment name (second argument, or default)
EXPERIMENT="${2:-augmentation/hls4ml_nconst32_aug_none}"

# Project root (where this repo lives on Euler)
export PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# Create slurm log directory
mkdir -p logs/slurm

# Activate your environment
source "$PROJECT_ROOT/HYDRA/bin/activate"

echo "============================================"
echo "Job ID:       $SLURM_JOB_ID"
echo "Node:         $(hostname)"
echo "GPU:          $CUDA_VISIBLE_DEVICES"
echo "Model:        $MODEL"
echo "Experiment:   $EXPERIMENT"
echo "Project root: $PROJECT_ROOT"
echo "============================================"

python src/train.py model="$MODEL" experiment="$EXPERIMENT" trainer=gpu
