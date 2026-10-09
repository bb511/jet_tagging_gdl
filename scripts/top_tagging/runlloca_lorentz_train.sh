#!/usr/bin/env bash
# ========================================================================
# LLOCA_LORENTZ TRAINING COMMANDS
# ========================================================================
# One block per (constituent count, regime): 1 x 1 = 1 train runs. Generated
# by scripts/optuna/make_train_catalogues.py; regenerate rather than edit by hand.
# Baseline only: the top-tagging replica (Zenodo 2603256) runs every row in the
# unaugmented regime, the embedded and the grid rows alike.
# The block follows the published recipe its experiment config holds; no study
# tunes it.
#
# Every command is commented out: copy one to run it by hand, or feed the whole
# file to scripts/cluster/submit_train.sh, which queues them on a GPU or slurm.

# ========================================================================
# lloca_lorentz  nconst=100
# ========================================================================

# ------------------------------------------------------------------------
# lloca_lorentz  nconst=100  baseline  (method=augmentation, group=none)
# published top-tagging recipe, from the experiment config
# ------------------------------------------------------------------------
# python3 src/train.py \
#     experiment=top_tagging/lloca_lorentz_100 \
#     data.method=augmentation \
#     data.augmentation_scheme=none \
#     data.n_folds=5 \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/train/top_tagging/lloca_lorentz/nconst100/baseline
