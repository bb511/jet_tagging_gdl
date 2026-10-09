#!/usr/bin/env bash
# ========================================================================
# VECTOR NEURONS, CHEAP FAMILY: HYPERPARAMETER SEARCH COMMANDS
# ========================================================================
# The one-block vector neurons without gate or stream, searched twice per
# constituent count in the baseline regime: as they are (blocks 1, 3, 5 at
# 16, 50, 150 constituents) and with the leader's eta and phi from the full
# jet's axis as two more scalars, read off the record's etarel and phirel
# (data.full_axis; blocks 2, 4, 6). Each pair measures what that axis is
# worth at its count; none is a row of the grid.
#
# Each search config sets its switches, its features and its study name
# (vn_cheap_nconst<N>_augmentation_none, vn_cheap_axis_nconst<N>_...), all
# in logs/optuna/hls4ml/vn_cheap.db. The two blocks of a count share the
# experiment and the regime, hence one label: select them by index with
# --only, never by name, and give their pools different --tag values.
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

# ========================================================================
# vn_cheap  nconst=50
# ========================================================================

# ------------------------------------------------------------------------
# vn_cheap  nconst=50  baseline  (method=augmentation, group=none)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     experiment=hls4ml/vn_50 \
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
# vn_cheap_axis  nconst=50
# ========================================================================

# ------------------------------------------------------------------------
# vn_cheap_axis  nconst=50  baseline  (method=augmentation, group=none)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     experiment=hls4ml/vn_50 \
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

# ========================================================================
# vn_cheap  nconst=150
# ========================================================================

# ------------------------------------------------------------------------
# vn_cheap  nconst=150  baseline  (method=augmentation, group=none)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     experiment=hls4ml/vn_150 \
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
# vn_cheap_axis  nconst=150
# ========================================================================

# ------------------------------------------------------------------------
# vn_cheap_axis  nconst=150  baseline  (method=augmentation, group=none)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     experiment=hls4ml/vn_150 \
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
