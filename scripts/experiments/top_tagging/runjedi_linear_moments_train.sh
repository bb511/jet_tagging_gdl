#!/usr/bin/env bash
# ========================================================================
# JEDI_LINEAR_MOMENTS TRAINING COMMANDS
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
# jedi_linear_moments  nconst=100
# ========================================================================

# ------------------------------------------------------------------------
# jedi_linear_moments  nconst=100  baseline  (method=augmentation, group=none)
# best trial 55 of top_tagging_jedi_linear_moments_nconst100_augmentation_none, val/acc_best=0.929381
# ------------------------------------------------------------------------
# python3 src/train.py \
#     experiment=top_tagging/jedi_linear_moments_100 \
#     data.method=augmentation \
#     data.augmentation_scheme=none \
#     data.n_folds=5 \
#     algorithm.aggregation=sum_over_n \
#     algorithm.head_layers='[128,64,32]' \
#     algorithm.hidden=256 \
#     algorithm.moments=uniform \
#     algorithm.optimizer.lr=0.002667789745251796 \
#     algorithm.optimizer.weight_decay=1e-07 \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/train/top_tagging/jedi_linear_moments/nconst100/baseline
