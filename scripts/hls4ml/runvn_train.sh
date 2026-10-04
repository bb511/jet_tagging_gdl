#!/usr/bin/env bash
# ========================================================================
# VN TRAINING COMMANDS
# ========================================================================
# One block per (constituent count, regime): 3 x 1 = 3 train runs. Generated
# by scripts/optuna/make_train_catalogues.py; regenerate rather than edit by hand.
# Baseline only: this architecture embeds the symmetry rather than acquiring it,
# so the augmentation and SCL regimes are not run on it.
# Each block carries the best of the first 200 completed trials, by trial
# number, of its own study under logs/optuna/hls4ml/;
# a block whose study is missing carries none and says so (transferred ablation).
#
# Every command is commented out: copy one to run it by hand, or feed the whole
# file to scripts/cluster/submit_train.sh, which queues them on a GPU or slurm.

# ========================================================================
# vn  nconst=16
# ========================================================================

# ------------------------------------------------------------------------
# vn  nconst=16  baseline  (method=augmentation, group=none)
# best trial 155 of vn_nconst16_augmentation_none, val/acc_best=0.721749
# ------------------------------------------------------------------------
# python3 src/train.py \
#     experiment=hls4ml/vn_16 \
#     data.method=augmentation \
#     data.augmentation_scheme=none \
#     data.n_folds=5 \
#     algorithm.dropout=0.0 \
#     algorithm.head_layers='[64,32]' \
#     algorithm.invariants=8 \
#     algorithm.num_blocks=2 \
#     algorithm.optimizer.lr=0.004986660327956767 \
#     algorithm.optimizer.weight_decay=0.05 \
#     algorithm.scalar_channels=32 \
#     algorithm.stream_channels=8 \
#     algorithm.tensor_invariants=4 \
#     algorithm.vector_channels=16 \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/train/hls4ml/vn/nconst16/baseline

# ========================================================================
# vn  nconst=50
# ========================================================================

# ------------------------------------------------------------------------
# vn  nconst=50  baseline  (method=augmentation, group=none)
# best trial 157 of vn_nconst50_augmentation_none, val/acc_best=0.822866
# ------------------------------------------------------------------------
# python3 src/train.py \
#     experiment=hls4ml/vn_50 \
#     data.method=augmentation \
#     data.augmentation_scheme=none \
#     data.n_folds=5 \
#     algorithm.dropout=0.0 \
#     algorithm.head_layers='[64,32]' \
#     algorithm.invariants=8 \
#     algorithm.num_blocks=3 \
#     algorithm.optimizer.lr=0.004688432069452879 \
#     algorithm.optimizer.weight_decay=0.05 \
#     algorithm.scalar_channels=32 \
#     algorithm.stream_channels=16 \
#     algorithm.tensor_invariants=4 \
#     algorithm.vector_channels=32 \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/train/hls4ml/vn/nconst50/baseline

# ========================================================================
# vn  nconst=150
# ========================================================================

# ------------------------------------------------------------------------
# vn  nconst=150  baseline  (method=augmentation, group=none)
# best trial 195 of vn_nconst150_augmentation_none, val/acc_best=0.826858
# ------------------------------------------------------------------------
# python3 src/train.py \
#     experiment=hls4ml/vn_150 \
#     data.method=augmentation \
#     data.augmentation_scheme=none \
#     data.n_folds=5 \
#     algorithm.dropout=0.0 \
#     algorithm.head_layers='[32]' \
#     algorithm.invariants=32 \
#     algorithm.num_blocks=3 \
#     algorithm.optimizer.lr=0.0049250827523425545 \
#     algorithm.optimizer.weight_decay=0.05 \
#     algorithm.scalar_channels=32 \
#     algorithm.stream_channels=32 \
#     algorithm.tensor_invariants=2 \
#     algorithm.vector_channels=32 \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/train/hls4ml/vn/nconst150/baseline
