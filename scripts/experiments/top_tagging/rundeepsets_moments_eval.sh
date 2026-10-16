#!/usr/bin/env bash
# ========================================================================
# DEEPSETS_MOMENTS EVALUATION COMMANDS
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
# deepsets_moments  nconst=100
# ========================================================================

# ------------------------------------------------------------------------
# deepsets_moments  nconst=100  baseline  (method=augmentation, group=none)
# best trial 88 of top_tagging_deepsets_moments_nconst100_augmentation_none, val/acc_best=0.928727
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=top_tagging/deepsets_moments_100 \
#     data.method=augmentation \
#     data.augmentation_scheme=none \
#     data.n_folds=5 \
#     algorithm.aggregation=sum \
#     algorithm.moments=rankcut \
#     algorithm.optimizer.lr=0.0016133952441116876 \
#     algorithm.optimizer.weight_decay=0.0 \
#     algorithm.phi_layers='[64,64,64]' \
#     algorithm.rho_layers='[64,32]' \
#     checkpoint_manifest_path=logs/train/top_tagging/deepsets_moments/nconst100/baseline \
#     results_table.invariance_metric=tv \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/top_tagging/deepsets_moments/nconst100/baseline
