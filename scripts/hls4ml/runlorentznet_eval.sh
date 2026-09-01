#!/usr/bin/env bash
# ========================================================================
# LORENTZNET EVALUATION COMMANDS
# ========================================================================
# One block per (constituent count, regime): 3 x 1 = 3 eval runs. Generated
# by scripts/optuna/make_train_catalogues.py; regenerate rather than edit by hand.
# Baseline only: this architecture embeds the symmetry rather than acquiring it,
# so the augmentation and SCL regimes are not run on it.
# Each block carries the best trial of its own study under logs/optuna/hls4ml/;
# a block whose study is missing carries none and says so (transferred ablation).
#
# Every command is commented out: copy one to run it by hand, or feed the whole
# file to scripts/cluster/submit_eval.sh, which queues them on a GPU or slurm.

# ========================================================================
# lorentznet  nconst=16
# ========================================================================

# ------------------------------------------------------------------------
# lorentznet  nconst=16  baseline  (method=augmentation, group=none)
# transferred from experiment config (no study yet)
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=hls4ml/lorentznet_16 \
#     data.method=augmentation \
#     data.augmentation_scheme=none \
#     data.n_folds=5 \
#     checkpoint_manifest_path=logs/train/hls4ml/lorentznet/nconst16/baseline/checkpoint_manifest.json \
#     callbacks.results_table.invariance_metric=wasserstein \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/hls4ml/lorentznet/nconst16/baseline

# ========================================================================
# lorentznet  nconst=50
# ========================================================================

# ------------------------------------------------------------------------
# lorentznet  nconst=50  baseline  (method=augmentation, group=none)
# transferred from experiment config (no study yet)
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=hls4ml/lorentznet_50 \
#     data.method=augmentation \
#     data.augmentation_scheme=none \
#     data.n_folds=5 \
#     checkpoint_manifest_path=logs/train/hls4ml/lorentznet/nconst50/baseline/checkpoint_manifest.json \
#     callbacks.results_table.invariance_metric=wasserstein \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/hls4ml/lorentznet/nconst50/baseline

# ========================================================================
# lorentznet  nconst=150
# ========================================================================

# ------------------------------------------------------------------------
# lorentznet  nconst=150  baseline  (method=augmentation, group=none)
# transferred from experiment config (no study yet)
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=hls4ml/lorentznet_150 \
#     data.method=augmentation \
#     data.augmentation_scheme=none \
#     data.n_folds=5 \
#     checkpoint_manifest_path=logs/train/hls4ml/lorentznet/nconst150/baseline/checkpoint_manifest.json \
#     callbacks.results_table.invariance_metric=wasserstein \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/hls4ml/lorentznet/nconst150/baseline
