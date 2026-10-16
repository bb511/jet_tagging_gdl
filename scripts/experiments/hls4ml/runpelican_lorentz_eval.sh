#!/usr/bin/env bash
# ========================================================================
# PELICAN_LORENTZ EVALUATION COMMANDS
# ========================================================================
# One block per (constituent count, regime): 3 x 1 = 3 eval runs. Generated
# by scripts/experiments/make_train_catalogues.py; regenerate rather than edit by hand.
# Baseline only: this architecture embeds the symmetry rather than acquiring it,
# so the augmentation and SCL regimes are not run on it.
# Each block carries the best of the first 200 completed trials, by trial
# number, of its own study under logs/optuna/hls4ml/;
# a block whose study is missing carries none and says so (transferred ablation).
#
# Every command is commented out: copy one to run it by hand, or feed the whole
# file to scripts/experiments/submit_eval.sh, which runs them under nohup.

# ========================================================================
# pelican_lorentz  nconst=16
# ========================================================================

# ------------------------------------------------------------------------
# pelican_lorentz  nconst=16  baseline  (method=augmentation, group=none)
# best trial 111 of pelican_lorentz_nconst16_augmentation_none, val/acc_best=0.660002
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=hls4ml/pelican_lorentz_16 \
#     data.method=augmentation \
#     data.augmentation_scheme=none \
#     data.n_folds=5 \
#     algorithm.dropout=0.0 \
#     algorithm.n_blocks=5 \
#     algorithm.n_hidden=25 \
#     algorithm.n_message=35 \
#     algorithm.optimizer.lr=0.0035038960298257225 \
#     algorithm.optimizer.weight_decay=0.005 \
#     checkpoint_manifest_path=logs/train/hls4ml/pelican_lorentz/nconst16/baseline \
#     results_table.invariance_metric=tv \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/hls4ml/pelican_lorentz/nconst16/baseline

# ========================================================================
# pelican_lorentz  nconst=50
# ========================================================================

# ------------------------------------------------------------------------
# pelican_lorentz  nconst=50  baseline  (method=augmentation, group=none)
# best trial 163 of pelican_lorentz_nconst50_augmentation_none, val/acc_best=0.793824
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=hls4ml/pelican_lorentz_50 \
#     data.method=augmentation \
#     data.augmentation_scheme=none \
#     data.n_folds=5 \
#     algorithm.dropout=0.025 \
#     algorithm.n_blocks=5 \
#     algorithm.n_hidden=35 \
#     algorithm.n_message=35 \
#     algorithm.optimizer.lr=0.0025294648648246003 \
#     algorithm.optimizer.weight_decay=0.005 \
#     checkpoint_manifest_path=logs/train/hls4ml/pelican_lorentz/nconst50/baseline \
#     results_table.invariance_metric=tv \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/hls4ml/pelican_lorentz/nconst50/baseline

# ========================================================================
# pelican_lorentz  nconst=150
# ========================================================================

# ------------------------------------------------------------------------
# pelican_lorentz  nconst=150  baseline  (method=augmentation, group=none)
# transferred from experiment config (no study yet)
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=hls4ml/pelican_lorentz_150 \
#     data.method=augmentation \
#     data.augmentation_scheme=none \
#     data.n_folds=5 \
#     checkpoint_manifest_path=logs/train/hls4ml/pelican_lorentz/nconst150/baseline \
#     results_table.invariance_metric=tv \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/hls4ml/pelican_lorentz/nconst150/baseline
