#!/usr/bin/env bash
# ========================================================================
# PELICAN_LORENTZ EVALUATION COMMANDS
# ========================================================================
# One block per (constituent count, regime): 1 x 1 = 1 eval runs. Generated
# by scripts/experiments/make_train_catalogues.py; regenerate rather than edit by hand.
# Baseline only: the top-tagging replica (Zenodo 2603256) runs every row in the
# unaugmented regime, the embedded and the grid rows alike.
# The block follows the published recipe its experiment config holds; no study
# tunes it.
#
# Every command is commented out: copy one to run it by hand, or feed the whole
# file to scripts/experiments/submit_eval.sh, which queues them on a GPU or slurm.

# ========================================================================
# pelican_lorentz  nconst=100
# ========================================================================

# ------------------------------------------------------------------------
# pelican_lorentz  nconst=100  baseline  (method=augmentation, group=none)
# published top-tagging recipe, from the experiment config
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=top_tagging/pelican_lorentz_100 \
#     data.method=augmentation \
#     data.augmentation_scheme=none \
#     data.n_folds=5 \
#     checkpoint_manifest_path=logs/train/top_tagging/pelican_lorentz/nconst100/baseline \
#     results_table.invariance_metric=tv \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/top_tagging/pelican_lorentz/nconst100/baseline
