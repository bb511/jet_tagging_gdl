#!/usr/bin/env bash
# ========================================================================
# JET_LEVEL_ALL53 EVALUATION COMMANDS
# ========================================================================
# One block per (constituent count, regime): 1 x 1 = 1 eval runs. Generated
# by scripts/experiments/make_train_catalogues.py; regenerate rather than edit by hand.
# Baseline only: the jet-level observables are computed upstream and never
# transformed, so the augmentation and SCL regimes have nothing to act on.
# Each block carries the best of the first 200 completed trials, by trial
# number, of its own study under logs/optuna/hls4ml/;
# a block whose study is missing carries none and says so (transferred ablation).
#
# Every command is commented out: copy one to run it by hand, or feed the whole
# file to scripts/experiments/submit_eval.sh, which queues them on a GPU or slurm.

# ========================================================================
# jet_level_all53  nconst=150
# ========================================================================

# ------------------------------------------------------------------------
# jet_level_all53  nconst=150  baseline  (method=augmentation, group=none)
# best trial 104 of jet_level_all53_nconst150_augmentation_none, val/acc_best=0.831480
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=hls4ml/jet_level_all53_150 \
#     data.method=augmentation \
#     data.augmentation_scheme=none \
#     data.n_folds=5 \
#     algorithm.batchnorm=True \
#     algorithm.hidden_layers='[128,64,32]' \
#     algorithm.optimizer.lr=0.003549886541946912 \
#     algorithm.optimizer.weight_decay=1e-07 \
#     checkpoint_manifest_path=logs/train/hls4ml/jet_level_all53/nconst150/baseline \
#     results_table.invariance_metric=tv \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/hls4ml/jet_level_all53/nconst150/baseline
