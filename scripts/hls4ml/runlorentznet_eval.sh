#!/usr/bin/env bash
# ========================================================================
# LORENTZNET EVALUATION COMMANDS
# ========================================================================
# One block per (constituent count, regime): 3 x 1 = 3 eval runs. Generated
# by scripts/optuna/make_train_catalogues.py; regenerate rather than edit by hand.
# Baseline only: this architecture embeds the symmetry rather than acquiring it,
# so the augmentation and SCL regimes are not run on it.
# Each block carries the best of the first 200 completed trials, by trial
# number, of its own study under logs/optuna/hls4ml/;
# a block whose study is missing carries none and says so (transferred ablation).
#
# Every command is commented out: copy one to run it by hand, or feed the whole
# file to scripts/cluster/submit_eval.sh, which queues them on a GPU or slurm.

# ========================================================================
# lorentznet  nconst=16
# ========================================================================

# ------------------------------------------------------------------------
# lorentznet  nconst=16  baseline  (method=augmentation, group=none)
# best trial 170 of lorentznet_nconst16_augmentation_none, val/acc_best=0.660080
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=hls4ml/lorentznet_16 \
#     data.method=augmentation \
#     data.augmentation_scheme=none \
#     data.n_folds=5 \
#     algorithm.c_weight=0.3125694315437945 \
#     algorithm.dropout=0.0 \
#     algorithm.n_hidden=72 \
#     algorithm.n_layers=6 \
#     algorithm.optimizer.lr=0.0018759351863429335 \
#     algorithm.optimizer.weight_decay=1e-07 \
#     checkpoint_manifest_path=logs/train/hls4ml/lorentznet/nconst16/baseline \
#     results_table.invariance_metric=wasserstein \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/hls4ml/lorentznet/nconst16/baseline

# ========================================================================
# lorentznet  nconst=50
# ========================================================================

# ------------------------------------------------------------------------
# lorentznet  nconst=50  baseline  (method=augmentation, group=none)
# best trial 68 of lorentznet_nconst50_augmentation_none, val/acc_best=0.791322
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=hls4ml/lorentznet_50 \
#     data.method=augmentation \
#     data.augmentation_scheme=none \
#     data.n_folds=5 \
#     algorithm.c_weight=0.0027709064824151504 \
#     algorithm.dropout=0.1 \
#     algorithm.n_hidden=72 \
#     algorithm.n_layers=6 \
#     algorithm.optimizer.lr=0.0016106992417331738 \
#     algorithm.optimizer.weight_decay=0.0 \
#     checkpoint_manifest_path=logs/train/hls4ml/lorentznet/nconst50/baseline \
#     results_table.invariance_metric=wasserstein \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/hls4ml/lorentznet/nconst50/baseline

# ========================================================================
# lorentznet  nconst=150
# ========================================================================

# ------------------------------------------------------------------------
# lorentznet  nconst=150  baseline  (method=augmentation, group=none)
# best trial 39 of lorentznet_nconst150_augmentation_none, val/acc_best=0.799047
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=hls4ml/lorentznet_150 \
#     data.method=augmentation \
#     data.augmentation_scheme=none \
#     data.n_folds=5 \
#     algorithm.c_weight=0.0001011765309617452 \
#     algorithm.dropout=0.1 \
#     algorithm.n_hidden=72 \
#     algorithm.n_layers=6 \
#     algorithm.optimizer.lr=0.0012538415205578719 \
#     algorithm.optimizer.weight_decay=0.0 \
#     checkpoint_manifest_path=logs/train/hls4ml/lorentznet/nconst150/baseline \
#     results_table.invariance_metric=wasserstein \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/hls4ml/lorentznet/nconst150/baseline
