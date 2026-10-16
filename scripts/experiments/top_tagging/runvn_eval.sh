#!/usr/bin/env bash
# ========================================================================
# VN EVALUATION COMMANDS
# ========================================================================
# One block per (constituent count, regime): 1 x 1 = 1 eval runs. Generated
# by scripts/experiments/make_train_catalogues.py; regenerate rather than edit by hand.
# Baseline only: this architecture embeds the symmetry rather than acquiring it,
# so the augmentation and SCL regimes are not run on it.
# Each block carries the best of the first 100 completed trials, by trial
# number, of its own study under logs/optuna/top_tagging/;
# a block whose study is missing carries none and says so (transferred ablation).
#
# Every command is commented out: copy one to run it by hand, or feed the whole
# file to scripts/experiments/submit_eval.sh, which runs them under nohup.

# ========================================================================
# vn  nconst=100
# ========================================================================

# ------------------------------------------------------------------------
# vn  nconst=100  baseline  (method=augmentation, group=none)
# best trial 30 of top_tagging_vn_nconst100_augmentation_none, val/acc_best=0.930119
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=top_tagging/vn_100 \
#     data.method=augmentation \
#     data.augmentation_scheme=none \
#     data.n_folds=5 \
#     algorithm.dropout=0.0 \
#     algorithm.head_layers='[32]' \
#     algorithm.invariants=32 \
#     algorithm.num_blocks=4 \
#     algorithm.optimizer.lr=0.00608161269652372 \
#     algorithm.optimizer.weight_decay=0.05 \
#     algorithm.scalar_channels=32 \
#     algorithm.stream_channels=32 \
#     algorithm.tensor_invariants=4 \
#     algorithm.vector_channels=64 \
#     checkpoint_manifest_path=logs/train/top_tagging/vn/nconst100/baseline \
#     results_table.invariance_metric=tv \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/top_tagging/vn/nconst100/baseline
