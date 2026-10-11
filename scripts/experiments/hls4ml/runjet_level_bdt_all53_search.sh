#!/usr/bin/env bash
# ========================================================================
# JET-LEVEL OBSERVABLES (53), BOOSTED TREES: SEARCH COMMANDS
# ========================================================================
# Gradient-boosted trees on the record's 53 jet-level observables,
# standardised with the moments of its training split: the second model
# family of the ceiling the networks are read against.
# The observables are computed upstream from the whole jet and never
# transformed, so the row runs the baseline alone, labelled with the one
# constituent count that never truncates the jet: 1 study of 200 trials.
#
# No block sets data.features: the datamodule of configs/data/jet_level.yaml
# reads the observables off the record's shards. Trials run at
# data.n_folds=1 with callbacks=search, so they draw no figure and never
# write to results/. The study name is not on the command line:
# configs/hparams_search/hls4ml/jet_level_bdt_all53_optuna.yaml derives it
# from the regime as jet_level_bdt_all53_nconst150_augmentation_none.
# Each epoch adds ten boosting iterations, so a trial of 50 epochs grows
# up to 500. scikit-learn fits the trees on the CPU: the block selects
# trainer=cpu and no device, so --gpu leaves it unchanged.
#
# Hand-maintained, unlike the train and eval catalogues. Every command is
# commented out; feed the whole file to scripts/experiments/submit_search.sh.

# ========================================================================
# jet_level_bdt_all53  nconst=150
# ========================================================================

# ------------------------------------------------------------------------
# jet_level_bdt_all53  nconst=150  baseline  (method=augmentation, group=none)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     experiment=hls4ml/jet_level_bdt_all53_150 \
#     hparams_search=hls4ml/jet_level_bdt_all53_optuna \
#     hydra.sweeper.n_trials=200 \
#     data.method=augmentation \
#     data.augmentation_scheme=none \
#     data.n_folds=1 \
#     callbacks=search \
#     logger=none \
#     trainer=cpu \
#     trainer.max_epochs=50
