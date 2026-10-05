#!/usr/bin/env bash
# ========================================================================
# LLOCA_Z EVALUATION COMMANDS
# ========================================================================
# One block per (constituent count, regime): 3 x 1 = 3 eval runs. Generated
# by scripts/optuna/make_train_catalogues.py; regenerate rather than edit by hand.
# Baseline only: this architecture embeds the symmetry rather than acquiring it,
# so the augmentation and SCL regimes are not run on it.
# Each block carries the best of the first 200 completed trials, by trial
# number, of its own study under logs/optuna/hls4ml/;
# a block whose study is missing carries none and says so (transferred ablation).
#
# Every command is commented out: copy one to run it by hand, or feed the whole
# file to scripts/cluster/submit_eval.sh, which queues them on a GPU or slurm.

# ========================================================================
# lloca_z  nconst=16
# ========================================================================

# ------------------------------------------------------------------------
# lloca_z  nconst=16  baseline  (method=augmentation, group=none)
# best trial 114 of lloca_z_nconst16_augmentation_none, val/acc_best=0.724675
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=hls4ml/lloca_z_16 \
#     data.method=augmentation \
#     data.augmentation_scheme=none \
#     data.n_folds=5 \
#     algorithm.attn_scalars=8 \
#     algorithm.attn_vectors=2 \
#     algorithm.dropout=0.0 \
#     algorithm.framesnet_hidden=32 \
#     algorithm.mlp_factor=4 \
#     algorithm.num_blocks=12 \
#     algorithm.num_heads=2 \
#     algorithm.optimizer.lr=0.0036498055261613297 \
#     algorithm.optimizer.weight_decay=0.05 \
#     checkpoint_manifest_path=logs/train/hls4ml/lloca_z/nconst16/baseline \
#     results_table.invariance_metric=tv \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/hls4ml/lloca_z/nconst16/baseline

# ========================================================================
# lloca_z  nconst=50
# ========================================================================

# ------------------------------------------------------------------------
# lloca_z  nconst=50  baseline  (method=augmentation, group=none)
# best trial 198 of lloca_z_nconst50_augmentation_none, val/acc_best=0.825583
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=hls4ml/lloca_z_50 \
#     data.method=augmentation \
#     data.augmentation_scheme=none \
#     data.n_folds=5 \
#     algorithm.attn_scalars=4 \
#     algorithm.attn_vectors=2 \
#     algorithm.dropout=0.0 \
#     algorithm.framesnet_hidden=16 \
#     algorithm.mlp_factor=4 \
#     algorithm.num_blocks=12 \
#     algorithm.num_heads=4 \
#     algorithm.optimizer.lr=0.003928057899947726 \
#     algorithm.optimizer.weight_decay=0.05 \
#     checkpoint_manifest_path=logs/train/hls4ml/lloca_z/nconst50/baseline \
#     results_table.invariance_metric=tv \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/hls4ml/lloca_z/nconst50/baseline

# ========================================================================
# lloca_z  nconst=150
# ========================================================================

# ------------------------------------------------------------------------
# lloca_z  nconst=150  baseline  (method=augmentation, group=none)
# best trial 176 of lloca_z_nconst150_augmentation_none, val/acc_best=0.831754
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=hls4ml/lloca_z_150 \
#     data.method=augmentation \
#     data.augmentation_scheme=none \
#     data.n_folds=5 \
#     algorithm.attn_scalars=8 \
#     algorithm.attn_vectors=1 \
#     algorithm.dropout=0.0 \
#     algorithm.framesnet_hidden=32 \
#     algorithm.mlp_factor=2 \
#     algorithm.num_blocks=8 \
#     algorithm.num_heads=8 \
#     algorithm.optimizer.lr=0.0008936726114451381 \
#     algorithm.optimizer.weight_decay=0.05 \
#     checkpoint_manifest_path=logs/train/hls4ml/lloca_z/nconst150/baseline \
#     results_table.invariance_metric=tv \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/hls4ml/lloca_z/nconst150/baseline
