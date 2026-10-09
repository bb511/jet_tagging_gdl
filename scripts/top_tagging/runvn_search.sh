#!/usr/bin/env bash
# ========================================================================
# VECTOR NEURONS: TOP-TAGGING HYPERPARAMETER SEARCH COMMANDS
# ========================================================================
# Vector neurons on the lifts of [pt, eta, phi] of the leading 100
# constituents of the top-tagging reference set (Zenodo 2603256), in the
# baseline regime alone. One Optuna driver: one study of 100 trials of 50
# epochs each, in a space narrowed around the optimum of the row's hls4ml
# N = 150 study on the same features, which the campaign enqueues as the
# first trial before any driver starts. The objective is the main run's
# clean val/acc_best.
#
# No block sets data.features: the experiment config sets [pt, eta, phi],
# the network's own columns. Trials run at data.n_folds=1 with
# callbacks=search, so they draw no figure and never write to results/.
# The study name is not on the command line:
# configs/hparams_search/top_tagging/vn_optuna.yaml derives it from the
# regime as top_tagging_vn_nconst100_augmentation_none and keeps it in
# logs/optuna/top_tagging/vn.db.
#
# Hand-maintained, unlike the train and eval catalogues. Every command is
# commented out; feed the whole file to scripts/cluster/submit_search.sh.

# ========================================================================
# vn  nconst=100
# ========================================================================

# ------------------------------------------------------------------------
# vn  nconst=100  baseline  (method=augmentation, group=none)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     experiment=top_tagging/vn_100 \
#     hparams_search=top_tagging/vn_optuna \
#     hydra.sweeper.n_trials=100 \
#     data.method=augmentation \
#     data.augmentation_scheme=none \
#     data.n_folds=1 \
#     callbacks=search \
#     logger=none \
#     trainer=gpu \
#     trainer.devices=[0] \
#     trainer.max_epochs=50
