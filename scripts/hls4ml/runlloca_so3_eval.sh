#!/usr/bin/env bash
# ========================================================================
# LLOCA_SO3 EVALUATION COMMANDS
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
# lloca_so3  nconst=16
# ========================================================================

# ------------------------------------------------------------------------
# lloca_so3  nconst=16  baseline  (method=augmentation, group=none)
# best trial 196 of lloca_so3_nconst16_augmentation_none, val/acc_best=0.681454
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=hls4ml/lloca_so3_16 \
#     data.method=augmentation \
#     data.augmentation_scheme=none \
#     data.n_folds=5 \
#     algorithm.attn_scalars=8 \
#     algorithm.attn_vectors=2 \
#     algorithm.dropout=0.0 \
#     algorithm.framesnet_hidden=32 \
#     algorithm.mlp_factor=2 \
#     algorithm.num_blocks=6 \
#     algorithm.num_heads=4 \
#     algorithm.optimizer.lr=0.0019035833384979878 \
#     algorithm.optimizer.weight_decay=0.05 \
#     checkpoint_manifest_path=logs/train/hls4ml/lloca_so3/nconst16/baseline \
#     results_table.invariance_metric=wasserstein \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/hls4ml/lloca_so3/nconst16/baseline

# ========================================================================
# lloca_so3  nconst=50
# ========================================================================

# ------------------------------------------------------------------------
# lloca_so3  nconst=50  baseline  (method=augmentation, group=none)
# best trial 138 of lloca_so3_nconst50_augmentation_none, val/acc_best=0.809258
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=hls4ml/lloca_so3_50 \
#     data.method=augmentation \
#     data.augmentation_scheme=none \
#     data.n_folds=5 \
#     algorithm.attn_scalars=8 \
#     algorithm.attn_vectors=1 \
#     algorithm.dropout=0.0 \
#     algorithm.framesnet_hidden=64 \
#     algorithm.mlp_factor=4 \
#     algorithm.num_blocks=8 \
#     algorithm.num_heads=8 \
#     algorithm.optimizer.lr=0.0006710355529113164 \
#     algorithm.optimizer.weight_decay=0.05 \
#     checkpoint_manifest_path=logs/train/hls4ml/lloca_so3/nconst50/baseline \
#     results_table.invariance_metric=wasserstein \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/hls4ml/lloca_so3/nconst50/baseline

# ========================================================================
# lloca_so3  nconst=150
# ========================================================================

# ------------------------------------------------------------------------
# lloca_so3  nconst=150  baseline  (method=augmentation, group=none)
# best trial 158 of lloca_so3_nconst150_augmentation_none, val/acc_best=0.820394
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=hls4ml/lloca_so3_150 \
#     data.method=augmentation \
#     data.augmentation_scheme=none \
#     data.n_folds=5 \
#     algorithm.attn_scalars=8 \
#     algorithm.attn_vectors=2 \
#     algorithm.dropout=0.0 \
#     algorithm.framesnet_hidden=64 \
#     algorithm.mlp_factor=2 \
#     algorithm.num_blocks=12 \
#     algorithm.num_heads=4 \
#     algorithm.optimizer.lr=0.0021632637761112876 \
#     algorithm.optimizer.weight_decay=0.05 \
#     checkpoint_manifest_path=logs/train/hls4ml/lloca_so3/nconst150/baseline \
#     results_table.invariance_metric=wasserstein \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/hls4ml/lloca_so3/nconst150/baseline
