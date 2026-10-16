#!/usr/bin/env bash
# ========================================================================
# MOMENTS_ONLY TRAINING COMMANDS
# ========================================================================
# One block per (constituent count, regime): 1 x 1 = 1 train runs. Generated
# by scripts/experiments/make_train_catalogues.py; regenerate rather than edit by hand.
# Baseline only: this architecture embeds the symmetry rather than acquiring it,
# so the augmentation and SCL regimes are not run on it.
# Each block carries the best of the first 100 completed trials, by trial
# number, of its own study under logs/optuna/top_tagging/;
# a block whose study is missing carries none and says so (transferred ablation).
#
# Every command is commented out: copy one to run it by hand, or feed the whole
# file to scripts/experiments/submit_train.sh, which runs them under nohup.

# ========================================================================
# moments_only  nconst=100
# ========================================================================

# ------------------------------------------------------------------------
# moments_only  nconst=100  baseline  (method=augmentation, group=none)
# best trial 84 of top_tagging_moments_only_nconst100_augmentation_none, val/acc_best=0.927021
# ------------------------------------------------------------------------
# python3 src/train.py \
#     experiment=top_tagging/moments_only_100 \
#     data.method=augmentation \
#     data.augmentation_scheme=none \
#     data.n_folds=5 \
#     algorithm.moments=rankcut \
#     algorithm.optimizer.lr=0.0018797090792310467 \
#     algorithm.optimizer.weight_decay=0.0 \
#     algorithm.rho_layers='[128,64]' \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/train/top_tagging/moments_only/nconst100/baseline
