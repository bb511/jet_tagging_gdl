#!/usr/bin/env bash
# ========================================================================
# DEEPSETS_MOMENTS TRAINING COMMANDS
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
# deepsets_moments  nconst=16
# ========================================================================

# ------------------------------------------------------------------------
# deepsets_moments  nconst=16  baseline  (method=augmentation, group=none)
# best trial 95 of deepsets_moments_nconst16_augmentation_none, val/acc_best=0.720922
# ------------------------------------------------------------------------
# python3 src/train.py \
#     experiment=hls4ml/deepsets_moments_16 \
#     data.method=augmentation \
#     data.augmentation_scheme=none \
#     data.n_folds=5 \
#     algorithm.aggregation=max \
#     algorithm.moments=rankcut \
#     algorithm.optimizer.lr=0.00462491923056384 \
#     algorithm.optimizer.weight_decay=0.0 \
#     algorithm.phi_layers='[64,64,64]' \
#     algorithm.rho_layers='[64,32]' \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/train/hls4ml/deepsets_moments/nconst16/baseline

# ========================================================================
# deepsets_moments  nconst=50
# ========================================================================

# ------------------------------------------------------------------------
# deepsets_moments  nconst=50  baseline  (method=augmentation, group=none)
# best trial 173 of deepsets_moments_nconst50_augmentation_none, val/acc_best=0.817057
# ------------------------------------------------------------------------
# python3 src/train.py \
#     experiment=hls4ml/deepsets_moments_50 \
#     data.method=augmentation \
#     data.augmentation_scheme=none \
#     data.n_folds=5 \
#     algorithm.aggregation=max \
#     algorithm.moments=rankcut \
#     algorithm.optimizer.lr=0.004942126473585737 \
#     algorithm.optimizer.weight_decay=0.0 \
#     algorithm.phi_layers='[64,64,64]' \
#     algorithm.rho_layers='[64,32]' \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/train/hls4ml/deepsets_moments/nconst50/baseline

# ========================================================================
# deepsets_moments  nconst=150
# ========================================================================

# ------------------------------------------------------------------------
# deepsets_moments  nconst=150  baseline  (method=augmentation, group=none)
# best trial 194 of deepsets_moments_nconst150_augmentation_none, val/acc_best=0.822694
# ------------------------------------------------------------------------
# python3 src/train.py \
#     experiment=hls4ml/deepsets_moments_150 \
#     data.method=augmentation \
#     data.augmentation_scheme=none \
#     data.n_folds=5 \
#     algorithm.aggregation=sum \
#     algorithm.moments=rankcut \
#     algorithm.optimizer.lr=0.0027931828696334776 \
#     algorithm.optimizer.weight_decay=0.0 \
#     algorithm.phi_layers='[64,64,64]' \
#     algorithm.rho_layers='[64,32]' \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/train/hls4ml/deepsets_moments/nconst150/baseline
