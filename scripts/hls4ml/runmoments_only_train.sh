#!/usr/bin/env bash
# ========================================================================
# MOMENTS_ONLY TRAINING COMMANDS
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
# moments_only  nconst=16
# ========================================================================

# ------------------------------------------------------------------------
# moments_only  nconst=16  baseline  (method=augmentation, group=none)
# best trial 197 of moments_only_nconst16_augmentation_none, val/acc_best=0.702764
# ------------------------------------------------------------------------
# python3 src/train.py \
#     experiment=hls4ml/moments_only_16 \
#     data.method=augmentation \
#     data.augmentation_scheme=none \
#     data.n_folds=5 \
#     algorithm.moments=rankcut \
#     algorithm.optimizer.lr=0.007152396369766525 \
#     algorithm.optimizer.weight_decay=1e-07 \
#     algorithm.rho_layers='[64,32]' \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/train/hls4ml/moments_only/nconst16/baseline

# ========================================================================
# moments_only  nconst=50
# ========================================================================

# ------------------------------------------------------------------------
# moments_only  nconst=50  baseline  (method=augmentation, group=none)
# best trial 19 of moments_only_nconst50_augmentation_none, val/acc_best=0.804513
# ------------------------------------------------------------------------
# python3 src/train.py \
#     experiment=hls4ml/moments_only_50 \
#     data.method=augmentation \
#     data.augmentation_scheme=none \
#     data.n_folds=5 \
#     algorithm.moments=rankcut \
#     algorithm.optimizer.lr=0.008611550799237118 \
#     algorithm.optimizer.weight_decay=0.0 \
#     algorithm.rho_layers='[64,32]' \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/train/hls4ml/moments_only/nconst50/baseline

# ========================================================================
# moments_only  nconst=150
# ========================================================================

# ------------------------------------------------------------------------
# moments_only  nconst=150  baseline  (method=augmentation, group=none)
# best trial 105 of moments_only_nconst150_augmentation_none, val/acc_best=0.817685
# ------------------------------------------------------------------------
# python3 src/train.py \
#     experiment=hls4ml/moments_only_150 \
#     data.method=augmentation \
#     data.augmentation_scheme=none \
#     data.n_folds=5 \
#     algorithm.moments=rankcut \
#     algorithm.optimizer.lr=0.005462008936973458 \
#     algorithm.optimizer.weight_decay=1e-06 \
#     algorithm.rho_layers='[64,32]' \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/train/hls4ml/moments_only/nconst150/baseline
