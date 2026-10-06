#!/usr/bin/env bash
# ========================================================================
# JEDI_LINEAR_MOMENTS TRAINING COMMANDS
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
# jedi_linear_moments  nconst=16
# ========================================================================

# ------------------------------------------------------------------------
# jedi_linear_moments  nconst=16  baseline  (method=augmentation, group=none)
# transferred from experiment config (no study yet)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     experiment=hls4ml/jedi_linear_moments_16 \
#     data.method=augmentation \
#     data.augmentation_scheme=none \
#     data.n_folds=5 \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/train/hls4ml/jedi_linear_moments/nconst16/baseline

# ========================================================================
# jedi_linear_moments  nconst=50
# ========================================================================

# ------------------------------------------------------------------------
# jedi_linear_moments  nconst=50  baseline  (method=augmentation, group=none)
# transferred from experiment config (no study yet)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     experiment=hls4ml/jedi_linear_moments_50 \
#     data.method=augmentation \
#     data.augmentation_scheme=none \
#     data.n_folds=5 \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/train/hls4ml/jedi_linear_moments/nconst50/baseline

# ========================================================================
# jedi_linear_moments  nconst=150
# ========================================================================

# ------------------------------------------------------------------------
# jedi_linear_moments  nconst=150  baseline  (method=augmentation, group=none)
# transferred from experiment config (no study yet)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     experiment=hls4ml/jedi_linear_moments_150 \
#     data.method=augmentation \
#     data.augmentation_scheme=none \
#     data.n_folds=5 \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/train/hls4ml/jedi_linear_moments/nconst150/baseline
