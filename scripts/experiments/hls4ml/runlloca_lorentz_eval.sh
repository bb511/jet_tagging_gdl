#!/usr/bin/env bash
# ========================================================================
# LLOCA_LORENTZ EVALUATION COMMANDS
# ========================================================================
# One block per (constituent count, regime): 3 x 1 = 3 eval runs. Generated
# by scripts/experiments/make_train_catalogues.py; regenerate rather than edit by hand.
# Baseline only: this architecture embeds the symmetry rather than acquiring it,
# so the augmentation and SCL regimes are not run on it.
# Each block carries the best of the first 200 completed trials, by trial
# number, of its own study under logs/optuna/hls4ml/;
# a block whose study is missing carries none and says so (transferred ablation).
#
# Every command is commented out: copy one to run it by hand, or feed the whole
# file to scripts/experiments/submit_eval.sh, which runs them under nohup.

# ========================================================================
# lloca_lorentz  nconst=16
# ========================================================================

# ------------------------------------------------------------------------
# lloca_lorentz  nconst=16  baseline  (method=augmentation, group=none)
# best trial 127 of lloca_lorentz_nconst16_augmentation_none, val/acc_best=0.646952
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=hls4ml/lloca_lorentz_16 \
#     data.method=augmentation \
#     data.augmentation_scheme=none \
#     data.n_folds=5 \
#     algorithm.attn_scalars=8 \
#     algorithm.attn_vectors=2 \
#     algorithm.dropout=0.1 \
#     algorithm.framesnet_hidden=32 \
#     algorithm.mlp_factor=2 \
#     algorithm.num_blocks=8 \
#     algorithm.num_heads=2 \
#     algorithm.optimizer.lr=0.004980493117765767 \
#     algorithm.optimizer.weight_decay=0.05 \
#     checkpoint_manifest_path=logs/train/hls4ml/lloca_lorentz/nconst16/baseline \
#     results_table.invariance_metric=tv \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/hls4ml/lloca_lorentz/nconst16/baseline

# ========================================================================
# lloca_lorentz  nconst=50
# ========================================================================

# ------------------------------------------------------------------------
# lloca_lorentz  nconst=50  baseline  (method=augmentation, group=none)
# best trial 170 of lloca_lorentz_nconst50_augmentation_none, val/acc_best=0.776259
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=hls4ml/lloca_lorentz_50 \
#     data.method=augmentation \
#     data.augmentation_scheme=none \
#     data.n_folds=5 \
#     algorithm.attn_scalars=8 \
#     algorithm.attn_vectors=2 \
#     algorithm.dropout=0.1 \
#     algorithm.framesnet_hidden=16 \
#     algorithm.mlp_factor=2 \
#     algorithm.num_blocks=12 \
#     algorithm.num_heads=2 \
#     algorithm.optimizer.lr=0.004304692250707242 \
#     algorithm.optimizer.weight_decay=0.0 \
#     checkpoint_manifest_path=logs/train/hls4ml/lloca_lorentz/nconst50/baseline \
#     results_table.invariance_metric=tv \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/hls4ml/lloca_lorentz/nconst50/baseline

# ========================================================================
# lloca_lorentz  nconst=150
# ========================================================================

# ------------------------------------------------------------------------
# lloca_lorentz  nconst=150  baseline  (method=augmentation, group=none)
# best trial 187 of lloca_lorentz_nconst150_augmentation_none, val/acc_best=0.787709
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=hls4ml/lloca_lorentz_150 \
#     data.method=augmentation \
#     data.augmentation_scheme=none \
#     data.n_folds=5 \
#     algorithm.attn_scalars=8 \
#     algorithm.attn_vectors=1 \
#     algorithm.dropout=0.0 \
#     algorithm.framesnet_hidden=32 \
#     algorithm.mlp_factor=4 \
#     algorithm.num_blocks=12 \
#     algorithm.num_heads=4 \
#     algorithm.optimizer.lr=0.002221091419475758 \
#     algorithm.optimizer.weight_decay=0.05 \
#     checkpoint_manifest_path=logs/train/hls4ml/lloca_lorentz/nconst150/baseline \
#     results_table.invariance_metric=tv \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/hls4ml/lloca_lorentz/nconst150/baseline
