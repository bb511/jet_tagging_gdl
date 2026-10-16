#!/usr/bin/env bash
# ========================================================================
# LLOCA_LORENTZ EVALUATION COMMANDS
# ========================================================================
# One block per (constituent count, regime): 1 x 1 = 1 eval runs. Generated
# by scripts/experiments/make_train_catalogues.py; regenerate rather than edit by hand.
# Baseline only: this architecture embeds the symmetry rather than acquiring it,
# so the augmentation and SCL regimes are not run on it.
# The block follows the published recipe its experiment config holds; no study
# tunes it.
#
# Every command is commented out: copy one to run it by hand, or feed the whole
# file to scripts/experiments/submit_eval.sh, which runs them under nohup.

# ========================================================================
# lloca_lorentz  nconst=100
# ========================================================================

# ------------------------------------------------------------------------
# lloca_lorentz  nconst=100  baseline  (method=augmentation, group=none)
# published top-tagging recipe, from the experiment config
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=top_tagging/lloca_lorentz_100 \
#     data.method=augmentation \
#     data.augmentation_scheme=none \
#     data.n_folds=5 \
#     checkpoint_manifest_path=logs/train/top_tagging/lloca_lorentz/nconst100/baseline \
#     results_table.invariance_metric=tv \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/top_tagging/lloca_lorentz/nconst100/baseline
