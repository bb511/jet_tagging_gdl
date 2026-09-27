#!/usr/bin/env bash
# ========================================================================
# PELICAN_LORENTZ TRAINING COMMANDS
# ========================================================================
# One block per (constituent count, regime): 3 x 1 = 3 train runs. Generated
# by scripts/optuna/make_train_catalogues.py; regenerate rather than edit by hand.
# Baseline only: this architecture embeds the symmetry rather than acquiring it,
# so the augmentation and SCL regimes are not run on it.
# Each block carries the best trial of its own study under logs/optuna/hls4ml/;
# a block whose study is missing carries none and says so (transferred ablation).
#
# Every command is commented out: copy one to run it by hand, or feed the whole
# file to scripts/cluster/submit_train.sh, which queues them on a GPU or slurm.

# ========================================================================
# pelican_lorentz  nconst=16
# ========================================================================

# ------------------------------------------------------------------------
# pelican_lorentz  nconst=16  baseline  (method=augmentation, group=none)
# transferred from experiment config (no study yet)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     experiment=hls4ml/pelican_lorentz_16 \
#     data.method=augmentation \
#     data.augmentation_scheme=none \
#     data.n_folds=5 \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/train/hls4ml/pelican_lorentz/nconst16/baseline

# ========================================================================
# pelican_lorentz  nconst=50
# ========================================================================

# ------------------------------------------------------------------------
# pelican_lorentz  nconst=50  baseline  (method=augmentation, group=none)
# transferred from experiment config (no study yet)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     experiment=hls4ml/pelican_lorentz_50 \
#     data.method=augmentation \
#     data.augmentation_scheme=none \
#     data.n_folds=5 \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/train/hls4ml/pelican_lorentz/nconst50/baseline

# ========================================================================
# pelican_lorentz  nconst=150
# ========================================================================

# ------------------------------------------------------------------------
# pelican_lorentz  nconst=150  baseline  (method=augmentation, group=none)
# transferred from experiment config (no study yet)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     experiment=hls4ml/pelican_lorentz_150 \
#     data.method=augmentation \
#     data.augmentation_scheme=none \
#     data.n_folds=5 \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/train/hls4ml/pelican_lorentz/nconst150/baseline
