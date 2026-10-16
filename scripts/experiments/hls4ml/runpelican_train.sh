#!/usr/bin/env bash
# ========================================================================
# PELICAN TRAINING COMMANDS
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
# file to scripts/experiments/submit_train.sh, which runs them under nohup.

# ========================================================================
# pelican  nconst=16
# ========================================================================

# ------------------------------------------------------------------------
# pelican  nconst=16  baseline  (method=augmentation, group=none)
# best trial 96 of pelican_nconst16_augmentation_none, val/acc_best=0.732812
# ------------------------------------------------------------------------
# python3 src/train.py \
#     experiment=hls4ml/pelican_16 \
#     data.method=augmentation \
#     data.augmentation_scheme=none \
#     data.n_folds=5 \
#     algorithm.dropout=0.025 \
#     algorithm.n_blocks=5 \
#     algorithm.n_hidden=35 \
#     algorithm.n_message=35 \
#     algorithm.optimizer.lr=0.0021494989913238053 \
#     algorithm.optimizer.weight_decay=0.001 \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/train/hls4ml/pelican/nconst16/baseline

# ========================================================================
# pelican  nconst=50
# ========================================================================

# ------------------------------------------------------------------------
# pelican  nconst=50  baseline  (method=augmentation, group=none)
# best trial 150 of pelican_nconst50_augmentation_none, val/acc_best=0.838264
# ------------------------------------------------------------------------
# python3 src/train.py \
#     experiment=hls4ml/pelican_50 \
#     data.method=augmentation \
#     data.augmentation_scheme=none \
#     data.n_folds=5 \
#     algorithm.dropout=0.1 \
#     algorithm.n_blocks=5 \
#     algorithm.n_hidden=35 \
#     algorithm.n_message=35 \
#     algorithm.optimizer.lr=0.0016749912106198838 \
#     algorithm.optimizer.weight_decay=0.005 \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/train/hls4ml/pelican/nconst50/baseline

# ========================================================================
# pelican  nconst=150
# ========================================================================

# ------------------------------------------------------------------------
# pelican  nconst=150  baseline  (method=augmentation, group=none)
# best trial 131 of pelican_nconst150_augmentation_none, val/acc_best=0.844693
# ------------------------------------------------------------------------
# python3 src/train.py \
#     experiment=hls4ml/pelican_150 \
#     data.method=augmentation \
#     data.augmentation_scheme=none \
#     data.n_folds=5 \
#     algorithm.dropout=0.0 \
#     algorithm.n_blocks=5 \
#     algorithm.n_hidden=35 \
#     algorithm.n_message=35 \
#     algorithm.optimizer.lr=0.0004672189427263489 \
#     algorithm.optimizer.weight_decay=1e-05 \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/train/hls4ml/pelican/nconst150/baseline
