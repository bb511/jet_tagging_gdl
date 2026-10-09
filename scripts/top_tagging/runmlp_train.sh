#!/usr/bin/env bash
# ========================================================================
# MLP TRAINING COMMANDS
# ========================================================================
# One block per (constituent count, regime): 1 x 1 = 1 train runs. Generated
# by scripts/optuna/make_train_catalogues.py; regenerate rather than edit by hand.
# Baseline only: the top-tagging replica (Zenodo 2603256) runs every row in the
# unaugmented regime, the embedded and the grid rows alike.
# Each block carries the best of the first 100 completed trials, by trial
# number, of its own study under logs/optuna/top_tagging/;
# a block whose study is missing carries none and says so (transferred ablation).
#
# Every command is commented out: copy one to run it by hand, or feed the whole
# file to scripts/cluster/submit_train.sh, which queues them on a GPU or slurm.

# ========================================================================
# mlp  nconst=100
# ========================================================================

# ------------------------------------------------------------------------
# mlp  nconst=100  baseline  (method=augmentation, group=none)
# transferred from experiment config (no study yet)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     experiment=top_tagging/mlp_100 \
#     data.method=augmentation \
#     data.augmentation_scheme=none \
#     data.n_folds=5 \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/train/top_tagging/mlp/nconst100/baseline
