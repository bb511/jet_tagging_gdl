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
#
# Usage:
#   sbatch scripts/run_euler.sh hls4ml_nconst16_aug_so2
#   sbatch scripts/run_euler.sh                          # uses default
# ---------------------------------------------------------------

set -euo pipefail

# Experiment name (first argument, or default)
EXPERIMENT="${1:-hls4ml_nconst32_aug_none}"

# Project root (where this repo lives on Euler)
export PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# Create slurm log directory
mkdir -p logs/slurm

# Activate your environment — uncomment and edit the line that matches your setup:
# source /cluster/home/$USER/miniconda3/bin/activate deepsets
# source "$PROJECT_ROOT/bin/activate"

echo "============================================"
echo "Job ID:       $SLURM_JOB_ID"
echo "Node:         $(hostname)"
echo "GPU:          $CUDA_VISIBLE_DEVICES"
echo "Experiment:   $EXPERIMENT"
echo "Project root: $PROJECT_ROOT"
echo "============================================"

python src/train.py experiment="$EXPERIMENT" trainer=gpu
