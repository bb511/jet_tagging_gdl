#!/bin/bash
# Evaluation counterpart of run_euler_array.sh.
#
# For each training command, uncomment the matching eval block below.
# The manifest path is derived from hydra.sweep.dir used during training:
#   <sweep_dir>/0/checkpoint_manifest.json
#
# eval.py in manifest mode evaluates main + all k-fold checkpoints in
# sequence and writes an aggregated row to results/results_table.csv.

export HLS4ML_DATA_DIR=/cluster/scratch/agogia/datasets/

# Absolute project root so the manifest path survives the submitit CWD change.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# ==================================================
# BASELINE EVAL FOR ALL MODELS
# ==================================================

# python src/eval.py \
#     -m \
#     hydra/launcher=submitit_slurm_euler \
#     experiment=baseline/hls4ml_nconst8_none \
#     +experiment_name=hls4ml_nconst8_aug_none \
#     model=deepsets_8 \
#     data.nconstituents=8 \
#     data.n_folds=5 \
#     trainer=gpu \
#     data.data_dir="$HLS4ML_DATA_DIR" \
#     checkpoint_manifest_path="${REPO_ROOT}/logs/slurm/baseline_deepsets_nconst8/0/checkpoint_manifest.json" \
#     callbacks.results_table.invariance_metric=flip \
#     hydra.sweep.dir="logs/slurm/eval_baseline_deepsets_nconst8" \
#     hydra.job.name="eval_baseline_deepsets_nconst8"

# python src/eval.py \
#     -m \
#     hydra/launcher=submitit_slurm_euler \
#     experiment=baseline/hls4ml_nconst16_none \
#     +experiment_name=hls4ml_nconst16_aug_none \
#     model=deepsets_16 \
#     data.nconstituents=16 \
#     data.n_folds=5 \
#     trainer=gpu \
#     data.data_dir="$HLS4ML_DATA_DIR" \
#     checkpoint_manifest_path="${REPO_ROOT}/logs/slurm/baseline_deepsets_nconst16/0/checkpoint_manifest.json" \
#     callbacks.results_table.invariance_metric=flip \
#     hydra.sweep.dir="logs/slurm/eval_baseline_deepsets_nconst16" \
#     hydra.job.name="eval_baseline_deepsets_nconst16"

# python src/eval.py \
#     -m \
#     hydra/launcher=submitit_slurm_euler \
#     experiment=baseline/hls4ml_nconst32_none \
#     +experiment_name=hls4ml_nconst32_aug_none \
#     model=deepsets_32 \
#     data.nconstituents=32 \
#     data.n_folds=5 \
#     trainer=gpu \
#     data.data_dir="$HLS4ML_DATA_DIR" \
#     checkpoint_manifest_path="${REPO_ROOT}/logs/slurm/baseline_deepsets_nconst32/0/checkpoint_manifest.json" \
#     callbacks.results_table.invariance_metric=flip \
#     hydra.sweep.dir="logs/slurm/eval_baseline_deepsets_nconst32" \
#     hydra.job.name="eval_baseline_deepsets_nconst32"


# ==================================================
# AUGMENTATION EVAL FOR ALL MODELS
# ==================================================

# python src/eval.py \
#     -m \
#     hydra/launcher=submitit_slurm_euler \
#     experiment=augmentation/hls4ml_nconst8_aug_so2 \
#     +experiment_name=hls4ml_nconst8_aug_so2 \
#     model=mlp_8 \
#     data.nconstituents=8 \
#     data.n_folds=5 \
#     trainer=gpu \
#     data.data_dir="$HLS4ML_DATA_DIR" \
#     checkpoint_manifest_path="${REPO_ROOT}/logs/slurm/aug_so2_mlp_nconst8/0/checkpoint_manifest.json" \
#     callbacks.results_table.invariance_metric=flip \
#     hydra.sweep.dir="logs/slurm/eval_aug_so2_mlp_nconst8" \
#     hydra.job.name="eval_aug_so2_mlp_nconst8"

# python src/eval.py \
#     -m \
#     hydra/launcher=submitit_slurm_euler \
#     experiment=augmentation/hls4ml_nconst16_aug_so2 \
#     +experiment_name=hls4ml_nconst16_aug_so2 \
#     model=mlp_16 \
#     data.nconstituents=16 \
#     data.n_folds=5 \
#     trainer=gpu \
#     data.data_dir="$HLS4ML_DATA_DIR" \
#     checkpoint_manifest_path="${REPO_ROOT}/logs/slurm/aug_so2_mlp_nconst16/0/checkpoint_manifest.json" \
#     callbacks.results_table.invariance_metric=flip \
#     hydra.sweep.dir="logs/slurm/eval_aug_so2_mlp_nconst16" \
#     hydra.job.name="eval_aug_so2_mlp_nconst16"

# python src/eval.py \
#     -m \
#     hydra/launcher=submitit_slurm_euler \
#     experiment=augmentation/hls4ml_nconst32_aug_so2 \
#     +experiment_name=hls4ml_nconst32_aug_so2 \
#     model=mlp_32 \
#     data.nconstituents=32 \
#     data.n_folds=5 \
#     trainer=gpu \
#     data.data_dir="$HLS4ML_DATA_DIR" \
#     checkpoint_manifest_path="${REPO_ROOT}/logs/slurm/aug_so2_mlp_nconst32/0/checkpoint_manifest.json" \
#     callbacks.results_table.invariance_metric=flip \
#     hydra.sweep.dir="logs/slurm/eval_aug_so2_mlp_nconst32" \
#     hydra.job.name="eval_aug_so2_mlp_nconst32"


# ==================================================
# SYMM_LOSS EVAL FOR ALL MODELS
# ==================================================

# python src/eval.py \
#     -m \
#     hydra/launcher=submitit_slurm_euler \
#     experiment=symm_loss/hls4ml_symm_loss_nconst8_so2 \
#     +experiment_name=hls4ml_symm_loss_nconst8_so2 \
#     model=mlp_8 \
#     data.nconstituents=8 \
#     data.n_folds=5 \
#     trainer=gpu \
#     data.data_dir="$HLS4ML_DATA_DIR" \
#     checkpoint_manifest_path="${REPO_ROOT}/logs/slurm/symm_loss_mlp_nconst8/0/checkpoint_manifest.json" \
#     callbacks.results_table.invariance_metric=flip \
#     hydra.sweep.dir="logs/slurm/eval_symm_loss_mlp_nconst8" \
#     hydra.job.name="eval_symm_loss_mlp_nconst8"

# python src/eval.py \
#     -m \
#     hydra/launcher=submitit_slurm_euler \
#     experiment=symm_loss/hls4ml_symm_loss_nconst16_so2 \
#     +experiment_name=hls4ml_symm_loss_nconst16_so2 \
#     model=mlp_16 \
#     data.nconstituents=16 \
#     data.n_folds=5 \
#     trainer=gpu \
#     data.data_dir="$HLS4ML_DATA_DIR" \
#     checkpoint_manifest_path="${REPO_ROOT}/logs/slurm/symm_loss_mlp_nconst16/0/checkpoint_manifest.json" \
#     callbacks.results_table.invariance_metric=flip \
#     hydra.sweep.dir="logs/slurm/eval_symm_loss_mlp_nconst16" \
#     hydra.job.name="eval_symm_loss_mlp_nconst16"

# python src/eval.py \
#     -m \
#     hydra/launcher=submitit_slurm_euler \
#     experiment=symm_loss/hls4ml_symm_loss_nconst32_so2 \
#     +experiment_name=hls4ml_symm_loss_nconst32_so2 \
#     model=mlp_32 \
#     data.nconstituents=32 \
#     data.n_folds=5 \
#     trainer=gpu \
#     data.data_dir="$HLS4ML_DATA_DIR" \
#     checkpoint_manifest_path="${REPO_ROOT}/logs/slurm/symm_loss_mlp_nconst32/0/checkpoint_manifest.json" \
#     callbacks.results_table.invariance_metric=flip \
#     hydra.sweep.dir="logs/slurm/eval_symm_loss_mlp_nconst32" \
#     hydra.job.name="eval_symm_loss_mlp_nconst32"


# ==================================================
# PERMUTATION EVAL FOR DEEPSETS
# ==================================================

# python src/eval.py \
#     -m \
#     hydra/launcher=submitit_slurm_euler \
#     experiment=permutation/hls4ml_nconst8_permutation \
#     +experiment_name=hls4ml_nconst8_permutation \
#     model=deepsets_8 \
#     data.nconstituents=8 \
#     data.n_folds=5 \
#     trainer=gpu \
#     data.data_dir="$HLS4ML_DATA_DIR" \
#     checkpoint_manifest_path="${REPO_ROOT}/logs/slurm/permutation_deepsets_nconst8/0/checkpoint_manifest.json" \
#     callbacks.results_table.invariance_metric=flip \
#     hydra.sweep.dir="logs/slurm/eval_permutation_deepsets_nconst8" \
#     hydra.job.name="eval_permutation_deepsets_nconst8"

# python src/eval.py \
#     -m \
#     hydra/launcher=submitit_slurm_euler \
#     experiment=permutation/hls4ml_nconst16_permutation \
#     +experiment_name=hls4ml_nconst16_permutation \
#     model=deepsets_16 \
#     data.nconstituents=16 \
#     data.n_folds=5 \
#     trainer=gpu \
#     data.data_dir="$HLS4ML_DATA_DIR" \
#     checkpoint_manifest_path="${REPO_ROOT}/logs/slurm/permutation_deepsets_nconst16/0/checkpoint_manifest.json" \
#     callbacks.results_table.invariance_metric=flip \
#     hydra.sweep.dir="logs/slurm/eval_permutation_deepsets_nconst16" \
#     hydra.job.name="eval_permutation_deepsets_nconst16"

# python src/eval.py \
#     -m \
#     hydra/launcher=submitit_slurm_euler \
#     experiment=permutation/hls4ml_nconst32_permutation \
#     +experiment_name=hls4ml_nconst32_permutation \
#     model=deepsets_32 \
#     data.nconstituents=32 \
#     data.n_folds=5 \
#     trainer=gpu \
#     data.data_dir="$HLS4ML_DATA_DIR" \
#     checkpoint_manifest_path="${REPO_ROOT}/logs/slurm/permutation_deepsets_nconst32/0/checkpoint_manifest.json" \
#     callbacks.results_table.invariance_metric=flip \
#     hydra.sweep.dir="logs/slurm/eval_permutation_deepsets_nconst32" \
#     hydra.job.name="eval_permutation_deepsets_nconst32"
