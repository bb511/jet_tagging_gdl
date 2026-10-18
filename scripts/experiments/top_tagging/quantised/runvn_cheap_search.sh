#!/usr/bin/env bash
# ========================================================================
# VECTOR NEURONS, CHEAP FAMILY: TOP-TAGGING HYPERPARAMETER SEARCH COMMANDS
# ========================================================================
# The one-block vector neurons without gate or stream on the leading 100
# constituents of the top-tagging reference set (Zenodo 2603256), searched
# twice in the baseline regime, as hls4ml/runvn_cheap_search.sh searches
# them at 16, 50 and 150: as they are (block 1) and with the leader's eta
# and phi from the jet's axis as two more scalars (block 2). The axis is
# that of the kept constituents, the whole jet for all but 0.5 % of the
# jets, since the cache holds no other. Each study runs 100 trials of 50
# epochs, the replica's budget, in the hls4ml space, the latency class of
# the FPGA designs; their trials with head [32] and 32 scalar channels give
# the float designs that runvn_hgq_train.sh starts from.
#
# Each search config sets its switches, its features and its study name
# (top_tagging_vn_cheap_nconst100_augmentation_none,
# top_tagging_vn_cheap_axis_nconst100_...), both in
# logs/optuna/top_tagging/vn_cheap.db. The two blocks share the experiment
# and the regime, hence one label: select them by index with --only, never
# by name, and give their pools different --tag values.
#
# Kept out of the data set's root, where scripts/experiments/sweep_status.py
# would read the catalogue as the vn row's. Hand-maintained, unlike the
# train and eval catalogues. Every command is commented out; feed the whole
# file to scripts/experiments/submit_search.sh or
# scripts/experiments/submit_search_ngt.sh.

# ========================================================================
# vn_cheap  nconst=100
# ========================================================================

# ------------------------------------------------------------------------
# vn_cheap  nconst=100  baseline  (method=augmentation, group=none)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     experiment=top_tagging/vn_100 \
#     hparams_search=top_tagging/vn_cheap_optuna \
#     hydra.sweeper.n_trials=100 \
#     data.method=augmentation \
#     data.augmentation_scheme=none \
#     data.n_folds=1 \
#     callbacks=search \
#     logger=none \
#     trainer=gpu \
#     trainer.devices=[0] \
#     trainer.max_epochs=50

# ========================================================================
# vn_cheap_axis  nconst=100
# ========================================================================

# ------------------------------------------------------------------------
# vn_cheap_axis  nconst=100  baseline  (method=augmentation, group=none)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     experiment=top_tagging/vn_100 \
#     hparams_search=top_tagging/vn_cheap_axis_optuna \
#     hydra.sweeper.n_trials=100 \
#     data.method=augmentation \
#     data.augmentation_scheme=none \
#     data.n_folds=1 \
#     callbacks=search \
#     logger=none \
#     trainer=gpu \
#     trainer.devices=[0] \
#     trainer.max_epochs=50
