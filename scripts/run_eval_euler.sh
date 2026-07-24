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
export MLFLOW_ALLOW_FILE_STORE=true

# ==================================================
# BASELINE EVAL FOR ALL MODELS
# ==================================================

# python src/eval.py \
#     -m \
#     hydra/launcher=submitit_slurm_euler \
#     experiment=baseline/hls4ml_nconst8_none \
#     +experiment_name=hls4ml_nconst8_aug_none \
#     model=mlp_8 \
#     model.nbr_feats=4 \
#     data.features="[px,py,pz,e]" \
#     data.keep_full_features_at_test=true \
#     data.nconstituents=8 \
#     data.n_folds=5 \
#     trainer=gpu \
#     data.data_dir="$HLS4ML_DATA_DIR" \
#     checkpoint_manifest_path="${REPO_ROOT}/logs/slurm/px_py_pz_e/mlp/baseline_mlp_nconst8/0/checkpoint_manifest.json" \
#     callbacks.results_table.invariance_metric=wasserstein \
#     hydra.sweep.dir="logs/slurm/px_py_pz_e/mlp/eval_baseline_mlp_nconst8" \
#     hydra.job.name="eval_baseline_mlp_nconst8"

# python src/eval.py \
#     -m \
#     hydra/launcher=submitit_slurm_euler \
#     experiment=baseline/hls4ml_nconst16_none \
#     +experiment_name=hls4ml_nconst16_aug_none \
#     model=mlp_16 \
#     model.nbr_feats=4 \
#     data.features="[px,py,pz,e]" \
#     data.keep_full_features_at_test=true \
#     data.nconstituents=16 \
#     data.n_folds=5 \
#     trainer=gpu \
#     data.data_dir="$HLS4ML_DATA_DIR" \
#     checkpoint_manifest_path="${REPO_ROOT}/logs/slurm/px_py_pz_e/mlp/baseline_mlp_nconst16/0/checkpoint_manifest.json" \
#     callbacks.results_table.invariance_metric=wasserstein \
#     hydra.sweep.dir="logs/slurm/px_py_pz_e/mlp/eval_baseline_mlp_nconst16" \
#     hydra.job.name="eval_baseline_mlp_nconst16"

# python src/eval.py \
#     -m \
#     hydra/launcher=submitit_slurm_euler \
#     experiment=baseline/hls4ml_nconst32_none \
#     +experiment_name=hls4ml_nconst32_aug_none \
#     model=mlp_32 \
#     model.nbr_feats=4 \
#     data.features="[px,py,pz,e]" \
#     data.keep_full_features_at_test=true \
#     data.nconstituents=32 \
#     data.n_folds=5 \
#     trainer=gpu \
#     data.data_dir="$HLS4ML_DATA_DIR" \
#     checkpoint_manifest_path="${REPO_ROOT}/logs/slurm/px_py_pz_e/mlp/baseline_mlp_nconst32/0/checkpoint_manifest.json" \
#     callbacks.results_table.invariance_metric=wasserstein \
#     hydra.sweep.dir="logs/slurm/px_py_pz_e/mlp/eval_baseline_mlp_nconst32" \
#     hydra.job.name="eval_baseline_mlp_nconst32"






# python src/eval.py \
#     -m \
#     hydra/launcher=submitit_slurm_euler \
#     experiment=baseline/hls4ml_nconst8_none \
#     +experiment_name=hls4ml_nconst8_aug_none \
#     model=deepsets_8 \
#     model.nbr_feats=4 \
#     data.features="[px,py,pz,e]" \
#     data.keep_full_features_at_test=true \
#     data.nconstituents=8 \
#     data.n_folds=5 \
#     trainer=gpu \
#     data.data_dir="$HLS4ML_DATA_DIR" \
#     checkpoint_manifest_path="${REPO_ROOT}/logs/slurm/px_py_pz_e/deepsets/baseline_deepsets_nconst8/0/checkpoint_manifest.json" \
#     callbacks.results_table.invariance_metric=wasserstein \
#     hydra.sweep.dir="logs/slurm/px_py_pz_e/deepsets/eval_baseline_deepsets_nconst8" \
#     hydra.job.name="eval_baseline_deepsets_nconst8"

# python src/eval.py \
#     -m \
#     hydra/launcher=submitit_slurm_euler \
#     experiment=baseline/hls4ml_nconst16_none \
#     +experiment_name=hls4ml_nconst16_aug_none \
#     model=deepsets_16 \
#     model.nbr_feats=4 \
#     data.features="[px,py,pz,e]" \
#     data.keep_full_features_at_test=true \
#     data.nconstituents=16 \
#     data.n_folds=5 \
#     trainer=gpu \
#     data.data_dir="$HLS4ML_DATA_DIR" \
#     checkpoint_manifest_path="${REPO_ROOT}/logs/slurm/px_py_pz_e/deepsets/baseline_deepsets_nconst16/0/checkpoint_manifest.json" \
#     callbacks.results_table.invariance_metric=wasserstein \
#     hydra.sweep.dir="logs/slurm/px_py_pz_e/deepsets/eval_baseline_deepsets_nconst16" \
#     hydra.job.name="eval_baseline_deepsets_nconst16"

# python src/eval.py \
#     -m \
#     hydra/launcher=submitit_slurm_euler \
#     experiment=baseline/hls4ml_nconst32_none \
#     +experiment_name=hls4ml_nconst32_aug_none \
#     model=deepsets_32 \
#     model.nbr_feats=4 \
#     data.features="[px,py,pz,e]" \
#     data.keep_full_features_at_test=true \
#     data.nconstituents=32 \
#     data.n_folds=5 \
#     trainer=gpu \
#     data.data_dir="$HLS4ML_DATA_DIR" \
#     checkpoint_manifest_path="${REPO_ROOT}/logs/slurm/px_py_pz_e/deepsets/baseline_deepsets_nconst32/0/checkpoint_manifest.json" \
#     callbacks.results_table.invariance_metric=wasserstein \
#     hydra.sweep.dir="logs/slurm/px_py_pz_e/deepsets/eval_baseline_deepsets_nconst32" \
#     hydra.job.name="eval_baseline_deepsets_nconst32"





# python src/eval.py \
#     -m \
#     hydra/launcher=submitit_slurm_euler \
#     experiment=baseline/hls4ml_nconst8_none \
#     +experiment_name=hls4ml_nconst8_aug_none \
#     model=jetformer \
#     model.nbr_feats=4 \
#     model.normalization=layer \
#     data.features="[px,py,pz,e]" \
#     data.keep_full_features_at_test=true \
#     data.nconstituents=8 \
#     data.n_folds=5 \
#     trainer=gpu \
#     data.data_dir="$HLS4ML_DATA_DIR" \
#     checkpoint_manifest_path="${REPO_ROOT}/logs/slurm/px_py_pz_e/jetformer/baseline_jetformer_nconst8/0/checkpoint_manifest.json" \
#     callbacks.results_table.invariance_metric=wasserstein \
#     hydra.sweep.dir="logs/slurm/px_py_pz_e/jetformer/eval_baseline_jetformer_nconst8" \
#     hydra.job.name="eval_baseline_jetformer_nconst8"

# python src/eval.py \
#     -m \
#     hydra/launcher=submitit_slurm_euler \
#     experiment=baseline/hls4ml_nconst16_none \
#     +experiment_name=hls4ml_nconst16_aug_none \
#     model=jetformer \
#     model.nbr_feats=4 \
#     model.normalization=layer \
#     data.features="[px,py,pz,e]" \
#     data.keep_full_features_at_test=true \
#     data.nconstituents=16 \
#     data.n_folds=5 \
#     trainer=gpu \
#     data.data_dir="$HLS4ML_DATA_DIR" \
#     checkpoint_manifest_path="${REPO_ROOT}/logs/slurm/px_py_pz_e/jetformer/baseline_jetformer_nconst16/0/checkpoint_manifest.json" \
#     callbacks.results_table.invariance_metric=wasserstein \
#     hydra.sweep.dir="logs/slurm/px_py_pz_e/jetformer/eval_baseline_jetformer_nconst16" \
#     hydra.job.name="eval_baseline_jetformer_nconst16"

# python src/eval.py \
#     -m \
#     hydra/launcher=submitit_slurm_euler \
#     experiment=baseline/hls4ml_nconst32_none \
#     +experiment_name=hls4ml_nconst32_aug_none \
#     model=jetformer \
#     model.nbr_feats=4 \
#     model.normalization=layer \
#     data.features="[px,py,pz,e]" \
#     data.keep_full_features_at_test=true \
#     data.nconstituents=32 \
#     data.n_folds=5 \
#     trainer=gpu \
#     data.data_dir="$HLS4ML_DATA_DIR" \
#     checkpoint_manifest_path="${REPO_ROOT}/logs/slurm/px_py_pz_e/jetformer/baseline_jetformer_nconst32/0/checkpoint_manifest.json" \
#     callbacks.results_table.invariance_metric=wasserstein \
#     hydra.sweep.dir="logs/slurm/px_py_pz_e/jetformer/eval_baseline_jetformer_nconst32" \
#     hydra.job.name="eval_baseline_jetformer_nconst32"




# ==================================================
# AUGMENTATION EVAL FOR ALL MODELS
# ==================================================

# python src/eval.py \
#     -m \
#     hydra/launcher=submitit_slurm_euler \
#     experiment=augmentation/hls4ml_nconst8_aug_lorentz \
#     +experiment_name=hls4ml_nconst8_aug_lorentz \
#     model=mlp_8 \
#     model.nbr_feats=4 \
#     data.features="[px,py,pz,e]" \
#     data.keep_full_features_at_test=true \
#     data.nconstituents=8 \
#     data.n_folds=5 \
#     trainer=gpu \
#     data.data_dir="$HLS4ML_DATA_DIR" \
#     checkpoint_manifest_path="${REPO_ROOT}/logs/slurm/px_py_pz_e/mlp/aug_lorentz_mlp_nconst8/0/checkpoint_manifest.json" \
#     callbacks.results_table.invariance_metric=wasserstein \
#     hydra.sweep.dir="logs/slurm/px_py_pz_e/mlp/eval_aug_lorentz_mlp_nconst8" \
#     hydra.job.name="eval_aug_lorentz_mlp_nconst8" & 

# python src/eval.py \
#     -m \
#     hydra/launcher=submitit_slurm_euler \
#     experiment=augmentation/hls4ml_nconst16_aug_lorentz \
#     +experiment_name=hls4ml_nconst16_aug_lorentz \
#     model=mlp_16 \
#     model.nbr_feats=4 \
#     data.features="[px,py,pz,e]" \
#     data.keep_full_features_at_test=true \
#     data.nconstituents=16 \
#     data.n_folds=5 \
#     trainer=gpu \
#     data.data_dir="$HLS4ML_DATA_DIR" \
#     checkpoint_manifest_path="${REPO_ROOT}/logs/slurm/px_py_pz_e/mlp/aug_lorentz_mlp_nconst16/0/checkpoint_manifest.json" \
#     callbacks.results_table.invariance_metric=wasserstein \
#     hydra.sweep.dir="logs/slurm/px_py_pz_e/mlp/eval_aug_lorentz_mlp_nconst16" \
#     hydra.job.name="eval_aug_lorentz_mlp_nconst16" &

# python src/eval.py \
#     -m \
#     hydra/launcher=submitit_slurm_euler \
#     experiment=augmentation/hls4ml_nconst32_aug_lorentz \
#     +experiment_name=hls4ml_nconst32_aug_lorentz \
#     model=mlp_32 \
#     model.nbr_feats=4 \
#     data.features="[px,py,pz,e]" \
#     data.keep_full_features_at_test=true \
#     data.nconstituents=32 \
#     data.n_folds=5 \
#     trainer=gpu \
#     data.data_dir="$HLS4ML_DATA_DIR" \
#     checkpoint_manifest_path="${REPO_ROOT}/logs/slurm/px_py_pz_e/mlp/aug_lorentz_mlp_nconst32/0/checkpoint_manifest.json" \
#     callbacks.results_table.invariance_metric=wasserstein \
#     hydra.sweep.dir="logs/slurm/px_py_pz_e/mlp/eval_aug_lorentz_mlp_nconst32" \
#     hydra.job.name="eval_aug_lorentz_mlp_nconst32" &





# python src/eval.py \
#     -m \
#     hydra/launcher=submitit_slurm_euler \
#     experiment=augmentation/hls4ml_nconst8_aug_lorentz \
#     +experiment_name=hls4ml_nconst8_aug_lorentz \
#     model=deepsets_8 \
#     model.nbr_feats=4 \
#     data.features="[px,py,pz,e]" \
#     data.keep_full_features_at_test=true \
#     data.nconstituents=8 \
#     data.n_folds=5 \
#     trainer=gpu \
#     data.data_dir="$HLS4ML_DATA_DIR" \
#     checkpoint_manifest_path="${REPO_ROOT}/logs/slurm/px_py_pz_e/deepsets/aug_lorentz_deepsets_nconst8/0/checkpoint_manifest.json" \
#     callbacks.results_table.invariance_metric=wasserstein \
#     hydra.sweep.dir="logs/slurm/px_py_pz_e/deepsets/eval_aug_lorentz_deepsets_nconst8" \
#     hydra.job.name="eval_aug_lorentz_deepsets_nconst8" &

# python src/eval.py \
#     -m \
#     hydra/launcher=submitit_slurm_euler \
#     experiment=augmentation/hls4ml_nconst16_aug_lorentz \
#     +experiment_name=hls4ml_nconst16_aug_lorentz \
#     model=deepsets_16 \
#     model.nbr_feats=4 \
#     data.features="[px,py,pz,e]" \
#     data.keep_full_features_at_test=true \
#     data.nconstituents=16 \
#     data.n_folds=5 \
#     trainer=gpu \
#     data.data_dir="$HLS4ML_DATA_DIR" \
#     checkpoint_manifest_path="${REPO_ROOT}/logs/slurm/px_py_pz_e/deepsets/aug_lorentz_deepsets_nconst16/0/checkpoint_manifest.json" \
#     callbacks.results_table.invariance_metric=wasserstein \
#     hydra.sweep.dir="logs/slurm/px_py_pz_e/deepsets/eval_aug_lorentz_deepsets_nconst16" \
#     hydra.job.name="eval_aug_lorentz_deepsets_nconst16" &

# python src/eval.py \
#     -m \
#     hydra/launcher=submitit_slurm_euler \
#     experiment=augmentation/hls4ml_nconst32_aug_lorentz \
#     +experiment_name=hls4ml_nconst32_aug_lorentz \
#     model=deepsets_32 \
#     model.nbr_feats=4 \
#     data.features="[px,py,pz,e]" \
#     data.keep_full_features_at_test=true \
#     data.nconstituents=32 \
#     data.n_folds=5 \
#     trainer=gpu \
#     data.data_dir="$HLS4ML_DATA_DIR" \
#     checkpoint_manifest_path="${REPO_ROOT}/logs/slurm/px_py_pz_e/deepsets/aug_lorentz_deepsets_nconst32/0/checkpoint_manifest.json" \
#     callbacks.results_table.invariance_metric=wasserstein \
#     hydra.sweep.dir="logs/slurm/px_py_pz_e/deepsets/eval_aug_lorentz_deepsets_nconst32" \
#     hydra.job.name="eval_aug_lorentz_deepsets_nconst32" &




# python src/eval.py \
#     -m \
#     hydra/launcher=submitit_slurm_euler \
#     experiment=augmentation/hls4ml_nconst8_aug_lorentz \
#     +experiment_name=hls4ml_nconst8_aug_lorentz \
#     model=jetformer \
#     model.nbr_feats=4 \
#     model.normalization=layer \
#     data.features="[px,py,pz,e]" \
#     data.keep_full_features_at_test=true \
#     data.nconstituents=8 \
#     data.n_folds=5 \
#     trainer=gpu \
#     data.data_dir="$HLS4ML_DATA_DIR" \
#     checkpoint_manifest_path="${REPO_ROOT}/logs/slurm/px_py_pz_e/jetformer/aug_lorentz_jetformer_nconst8/0/checkpoint_manifest.json" \
#     callbacks.results_table.invariance_metric=wasserstein \
#     hydra.sweep.dir="logs/slurm/px_py_pz_e/jetformer/eval_aug_lorentz_jetformer_nconst8" \
#     hydra.job.name="eval_aug_lorentz_jetformer_nconst8" &

# python src/eval.py \
#     -m \
#     hydra/launcher=submitit_slurm_euler \
#     experiment=augmentation/hls4ml_nconst16_aug_lorentz \
#     +experiment_name=hls4ml_nconst16_aug_lorentz \
#     model=jetformer \
#     model.nbr_feats=4 \
#     model.normalization=layer \
#     data.features="[px,py,pz,e]" \
#     data.keep_full_features_at_test=true \
#     data.nconstituents=16 \
#     data.n_folds=5 \
#     trainer=gpu \
#     data.data_dir="$HLS4ML_DATA_DIR" \
#     checkpoint_manifest_path="${REPO_ROOT}/logs/slurm/px_py_pz_e/jetformer/aug_lorentz_jetformer_nconst16/0/checkpoint_manifest.json" \
#     callbacks.results_table.invariance_metric=wasserstein \
#     hydra.sweep.dir="logs/slurm/px_py_pz_e/jetformer/eval_aug_lorentz_jetformer_nconst16" \
#     hydra.job.name="eval_aug_lorentz_jetformer_nconst16" &

# python src/eval.py \
#     -m \
#     hydra/launcher=submitit_slurm_euler \
#     experiment=augmentation/hls4ml_nconst32_aug_lorentz \
#     +experiment_name=hls4ml_nconst32_aug_lorentz \
#     model=jetformer \
#     model.nbr_feats=4 \
#     model.normalization=layer \
#     data.features="[px,py,pz,e]" \
#     data.keep_full_features_at_test=true \
#     data.nconstituents=32 \
#     data.n_folds=5 \
#     trainer=gpu \
#     data.data_dir="$HLS4ML_DATA_DIR" \
#     checkpoint_manifest_path="${REPO_ROOT}/logs/slurm/px_py_pz_e/jetformer/aug_lorentz_jetformer_nconst32/0/checkpoint_manifest.json" \
#     callbacks.results_table.invariance_metric=wasserstein \
#     hydra.sweep.dir="logs/slurm/px_py_pz_e/jetformer/eval_aug_lorentz_jetformer_nconst32" \
#     hydra.job.name="eval_aug_lorentz_jetformer_nconst32" &

# # ==================================================
# # SYMM_LOSS EVAL FOR ALL MODELS
# # ==================================================

python src/eval.py \
    -m \
    hydra/launcher=submitit_slurm_euler \
    experiment=symm_loss/hls4ml_symm_loss_nconst8_lorentz \
    +experiment_name=hls4ml_symm_loss_nconst8_lorentz \
    model=mlp_8 \
    model.nbr_feats=4 \
    data.features="[px,py,pz,e]" \
    data.keep_full_features_at_test=true \
    data.nconstituents=8 \
    data.n_folds=5 \
    trainer=gpu \
    data.data_dir="$HLS4ML_DATA_DIR" \
    checkpoint_manifest_path="${REPO_ROOT}/logs/slurm/px_py_pz_e/mlp/symm_loss_lorentz_mlp_nconst8/0/checkpoint_manifest.json" \
    callbacks.results_table.invariance_metric=wasserstein \
    hydra.sweep.dir="logs/slurm/px_py_pz_e/mlp/eval_symm_loss_lorentz_mlp_nconst8" \
    hydra.job.name="eval_symm_loss_lorentz_mlp_nconst8" &

python src/eval.py \
    -m \
    hydra/launcher=submitit_slurm_euler \
    experiment=symm_loss/hls4ml_symm_loss_nconst16_lorentz \
    +experiment_name=hls4ml_symm_loss_nconst16_lorentz \
    model=mlp_16 \
    model.nbr_feats=4 \
    data.features="[px,py,pz,e]" \
    data.keep_full_features_at_test=true \
    data.nconstituents=16 \
    data.n_folds=5 \
    trainer=gpu \
    data.data_dir="$HLS4ML_DATA_DIR" \
    checkpoint_manifest_path="${REPO_ROOT}/logs/slurm/px_py_pz_e/mlp/symm_loss_lorentz_mlp_nconst16/0/checkpoint_manifest.json" \
    callbacks.results_table.invariance_metric=wasserstein \
    hydra.sweep.dir="logs/slurm/px_py_pz_e/mlp/eval_symm_loss_lorentz_mlp_nconst16" \
    hydra.job.name="eval_symm_loss_lorentz_mlp_nconst16" &

python src/eval.py \
    -m \
    hydra/launcher=submitit_slurm_euler \
    experiment=symm_loss/hls4ml_symm_loss_nconst32_lorentz \
    +experiment_name=hls4ml_symm_loss_nconst32_lorentz \
    model=mlp_32 \
    model.nbr_feats=4 \
    data.features="[px,py,pz,e]" \
    data.keep_full_features_at_test=true \
    data.nconstituents=32 \
    data.n_folds=5 \
    trainer=gpu \
    data.data_dir="$HLS4ML_DATA_DIR" \
    checkpoint_manifest_path="${REPO_ROOT}/logs/slurm/px_py_pz_e/mlp/symm_loss_lorentz_mlp_nconst32/0/checkpoint_manifest.json" \
    callbacks.results_table.invariance_metric=wasserstein \
    hydra.sweep.dir="logs/slurm/px_py_pz_e/mlp/eval_symm_loss_lorentz_mlp_nconst32" \
    hydra.job.name="eval_symm_loss_lorentz_mlp_nconst32" &




python src/eval.py \
    -m \
    hydra/launcher=submitit_slurm_euler \
    experiment=symm_loss/hls4ml_symm_loss_nconst8_lorentz \
    +experiment_name=hls4ml_symm_loss_nconst8_lorentz \
    model=deepsets_8 \
    model.nbr_feats=4 \
    data.features="[px,py,pz,e]" \
    data.keep_full_features_at_test=true \
    data.nconstituents=8 \
    data.n_folds=5 \
    trainer=gpu \
    data.data_dir="$HLS4ML_DATA_DIR" \
    checkpoint_manifest_path="${REPO_ROOT}/logs/slurm/px_py_pz_e/deepsets/symm_loss_lorentz_deepsets_nconst8/0/checkpoint_manifest.json" \
    callbacks.results_table.invariance_metric=wasserstein \
    hydra.sweep.dir="logs/slurm/px_py_pz_e/deepsets/eval_symm_loss_lorentz_deepsets_nconst8" \
    hydra.job.name="eval_symm_loss_lorentz_deepsets_nconst8" &

python src/eval.py \
    -m \
    hydra/launcher=submitit_slurm_euler \
    experiment=symm_loss/hls4ml_symm_loss_nconst16_lorentz \
    +experiment_name=hls4ml_symm_loss_nconst16_lorentz \
    model=deepsets_16 \
    model.nbr_feats=4 \
    data.features="[px,py,pz,e]" \
    data.keep_full_features_at_test=true \
    data.nconstituents=16 \
    data.n_folds=5 \
    trainer=gpu \
    data.data_dir="$HLS4ML_DATA_DIR" \
    checkpoint_manifest_path="${REPO_ROOT}/logs/slurm/px_py_pz_e/deepsets/symm_loss_lorentz_deepsets_nconst16/0/checkpoint_manifest.json" \
    callbacks.results_table.invariance_metric=wasserstein \
    hydra.sweep.dir="logs/slurm/px_py_pz_e/deepsets/eval_symm_loss_lorentz_deepsets_nconst16" \
    hydra.job.name="eval_symm_loss_lorentz_deepsets_nconst16" &

python src/eval.py \
    -m \
    hydra/launcher=submitit_slurm_euler \
    experiment=symm_loss/hls4ml_symm_loss_nconst32_lorentz \
    +experiment_name=hls4ml_symm_loss_nconst32_lorentz \
    model=deepsets_32 \
    model.nbr_feats=4 \
    data.features="[px,py,pz,e]" \
    data.keep_full_features_at_test=true \
    data.nconstituents=32 \
    data.n_folds=5 \
    trainer=gpu \
    data.data_dir="$HLS4ML_DATA_DIR" \
    checkpoint_manifest_path="${REPO_ROOT}/logs/slurm/px_py_pz_e/deepsets/symm_loss_lorentz_deepsets_nconst32/0/checkpoint_manifest.json" \
    callbacks.results_table.invariance_metric=wasserstein \
    hydra.sweep.dir="logs/slurm/px_py_pz_e/deepsets/eval_symm_loss_lorentz_deepsets_nconst32" \
    hydra.job.name="eval_symm_loss_lorentz_deepsets_nconst32" &




python src/eval.py \
    -m \
    hydra/launcher=submitit_slurm_euler \
    experiment=symm_loss/hls4ml_symm_loss_nconst8_lorentz \
    +experiment_name=hls4ml_symm_loss_nconst8_lorentz \
    model=jetformer \
    model.nbr_feats=4 \
    model.normalization=layer \
    data.features="[px,py,pz,e]" \
    data.keep_full_features_at_test=true \
    data.nconstituents=8 \
    data.n_folds=5 \
    trainer=gpu \
    data.data_dir="$HLS4ML_DATA_DIR" \
    checkpoint_manifest_path="${REPO_ROOT}/logs/slurm/px_py_pz_e/jetformer/symm_loss_lorentz_jetformer_nconst8/0/checkpoint_manifest.json" \
    callbacks.results_table.invariance_metric=wasserstein \
    hydra.sweep.dir="logs/slurm/px_py_pz_e/jetformer/eval_symm_loss_lorentz_jetformer_nconst8" \
    hydra.job.name="eval_symm_loss_lorentz_jetformer_nconst8" &

python src/eval.py \
    -m \
    hydra/launcher=submitit_slurm_euler \
    experiment=symm_loss/hls4ml_symm_loss_nconst16_lorentz \
    +experiment_name=hls4ml_symm_loss_nconst16_lorentz \
    model=jetformer \
    model.nbr_feats=4 \
    model.normalization=layer \
    data.features="[px,py,pz,e]" \
    data.keep_full_features_at_test=true \
    data.nconstituents=16 \
    data.n_folds=5 \
    trainer=gpu \
    data.data_dir="$HLS4ML_DATA_DIR" \
    checkpoint_manifest_path="${REPO_ROOT}/logs/slurm/px_py_pz_e/jetformer/symm_loss_lorentz_jetformer_nconst16/0/checkpoint_manifest.json" \
    callbacks.results_table.invariance_metric=wasserstein \
    hydra.sweep.dir="logs/slurm/px_py_pz_e/jetformer/eval_symm_loss_lorentz_jetformer_nconst16" \
    hydra.job.name="eval_symm_loss_lorentz_jetformer_nconst16" &

python src/eval.py \
    -m \
    hydra/launcher=submitit_slurm_euler \
    experiment=symm_loss/hls4ml_symm_loss_nconst32_lorentz \
    +experiment_name=hls4ml_symm_loss_nconst32_lorentz \
    model=jetformer \
    model.nbr_feats=4 \
    model.normalization=layer \
    data.features="[px,py,pz,e]" \
    data.keep_full_features_at_test=true \
    data.nconstituents=32 \
    data.n_folds=5 \
    trainer=gpu \
    data.data_dir="$HLS4ML_DATA_DIR" \
    checkpoint_manifest_path="${REPO_ROOT}/logs/slurm/px_py_pz_e/jetformer/symm_loss_lorentz_jetformer_nconst32/0/checkpoint_manifest.json" \
    callbacks.results_table.invariance_metric=wasserstein \
    hydra.sweep.dir="logs/slurm/px_py_pz_e/jetformer/eval_symm_loss_lorentz_jetformer_nconst32" \
    hydra.job.name="eval_symm_loss_lorentz_jetformer_nconst32" &



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
#     callbacks.results_table.invariance_metric=wasserstein \
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
#     callbacks.results_table.invariance_metric=wasserstein \
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
#     callbacks.results_table.invariance_metric=wasserstein \
#     hydra.sweep.dir="logs/slurm/eval_permutation_deepsets_nconst32" \
#     hydra.job.name="eval_permutation_deepsets_nconst32"
