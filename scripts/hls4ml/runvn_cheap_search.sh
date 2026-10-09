#!/usr/bin/env bash
# ========================================================================
# VECTOR NEURONS, CHEAP FAMILY: HYPERPARAMETER SEARCH COMMANDS
# ========================================================================
# The one-block vector neurons without gate or stream at 16 constituents,
# searched twice in the baseline regime: block 1 as they are, block 2 with
# the leader's eta and phi from the full jet's axis as two more scalars,
# read off the record's etarel and phirel (data.full_axis). The pair
# measures what that axis is worth; neither is a row of the grid.
#
# Each search config sets its switches, its features and its study name
# (vn_cheap_nconst16_augmentation_none, vn_cheap_axis_nconst16_...), both
# in logs/optuna/hls4ml/vn_cheap.db. Both blocks share the experiment and
# the regime, hence one label: select them with --only 1 and --only 2,
# never by name, and give their pools different --tag values.
#
# Hand-maintained, unlike the train and eval catalogues. Every command is
# commented out; feed the whole file to scripts/cluster/submit_search.sh
# or scripts/cluster/submit_search_ngt.sh.

# ========================================================================
# vn_cheap  nconst=16
# ========================================================================

# ------------------------------------------------------------------------
# vn_cheap  nconst=16  baseline  (method=augmentation, group=none)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     experiment=hls4ml/vn_16 \
#     hparams_search=hls4ml/vn_cheap_optuna \
#     hydra.sweeper.n_trials=200 \
#     data.method=augmentation \
#     data.augmentation_scheme=none \
#     data.n_folds=1 \
#     callbacks=search \
#     logger=none \
#     trainer=gpu \
#     trainer.devices=[0] \
#     trainer.max_epochs=50

# ========================================================================
# vn_cheap_axis  nconst=16
# ========================================================================

# ------------------------------------------------------------------------
# vn_cheap_axis  nconst=16  baseline  (method=augmentation, group=none)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     experiment=hls4ml/vn_16 \
#     hparams_search=hls4ml/vn_cheap_axis_optuna \
#     hydra.sweeper.n_trials=200 \
#     data.method=augmentation \
#     data.augmentation_scheme=none \
#     data.n_folds=1 \
#     callbacks=search \
#     logger=none \
#     trainer=gpu \
#     trainer.devices=[0] \
#     trainer.max_epochs=50
