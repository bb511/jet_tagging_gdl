#!/usr/bin/env bash
# ========================================================================
# LGATR_SLIM_LORENTZ TRAINING COMMANDS
# ========================================================================
# One block per (constituent count, regime): 3 x 1 = 3 train runs. Generated
# by scripts/experiments/make_train_catalogues.py; regenerate rather than edit by hand.
# Baseline only: this architecture embeds the symmetry rather than acquiring it,
# so the augmentation and SCL regimes are not run on it.
# Each block carries the best of the first 200 completed trials, by trial
# number, of its own study under logs/optuna/hls4ml/;
# a block whose study is missing carries none and says so (transferred ablation).
#
# Every command is commented out: copy one to run it by hand, or feed the whole
# file to scripts/experiments/submit_train.sh, which queues them on a GPU or slurm.

# ========================================================================
# lgatr_slim_lorentz  nconst=16
# ========================================================================

# ------------------------------------------------------------------------
# lgatr_slim_lorentz  nconst=16  baseline  (method=augmentation, group=none)
# best trial 168 of lgatr_slim_lorentz_nconst16_augmentation_none, val/acc_best=0.687154
# ------------------------------------------------------------------------
# python3 src/train.py \
#     experiment=hls4ml/lgatr_slim_lorentz_16 \
#     data.method=augmentation \
#     data.augmentation_scheme=none \
#     data.n_folds=5 \
#     algorithm.dropout=0.1 \
#     algorithm.hidden_s_channels=32 \
#     algorithm.hidden_v_channels=32 \
#     algorithm.mlp_ratio=2 \
#     algorithm.num_blocks=12 \
#     algorithm.num_heads=8 \
#     algorithm.optimizer.lr=0.004397679258182307 \
#     algorithm.optimizer.weight_decay=0.0 \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/train/hls4ml/lgatr_slim_lorentz/nconst16/baseline

# ========================================================================
# lgatr_slim_lorentz  nconst=50
# ========================================================================

# ------------------------------------------------------------------------
# lgatr_slim_lorentz  nconst=50  baseline  (method=augmentation, group=none)
# best trial 167 of lgatr_slim_lorentz_nconst50_augmentation_none, val/acc_best=0.814658
# ------------------------------------------------------------------------
# python3 src/train.py \
#     experiment=hls4ml/lgatr_slim_lorentz_50 \
#     data.method=augmentation \
#     data.augmentation_scheme=none \
#     data.n_folds=5 \
#     algorithm.dropout=0.0 \
#     algorithm.hidden_s_channels=96 \
#     algorithm.hidden_v_channels=32 \
#     algorithm.mlp_ratio=2 \
#     algorithm.num_blocks=12 \
#     algorithm.num_heads=8 \
#     algorithm.optimizer.lr=0.0013607618089358372 \
#     algorithm.optimizer.weight_decay=0.05 \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/train/hls4ml/lgatr_slim_lorentz/nconst50/baseline

# ========================================================================
# lgatr_slim_lorentz  nconst=150
# ========================================================================

# ------------------------------------------------------------------------
# lgatr_slim_lorentz  nconst=150  baseline  (method=augmentation, group=none)
# best trial 127 of lgatr_slim_lorentz_nconst150_augmentation_none, val/acc_best=0.826551
# ------------------------------------------------------------------------
# python3 src/train.py \
#     experiment=hls4ml/lgatr_slim_lorentz_150 \
#     data.method=augmentation \
#     data.augmentation_scheme=none \
#     data.n_folds=5 \
#     algorithm.dropout=0.0 \
#     algorithm.hidden_s_channels=64 \
#     algorithm.hidden_v_channels=32 \
#     algorithm.mlp_ratio=4 \
#     algorithm.num_blocks=12 \
#     algorithm.num_heads=4 \
#     algorithm.optimizer.lr=0.0002599157572021059 \
#     algorithm.optimizer.weight_decay=0.01 \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/train/hls4ml/lgatr_slim_lorentz/nconst150/baseline
