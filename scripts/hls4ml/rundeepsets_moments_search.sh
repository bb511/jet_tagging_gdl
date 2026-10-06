#!/usr/bin/env bash
# ========================================================================
# DEEPSETS WITH JET MOMENTS: HYPERPARAMETER SEARCH COMMANDS
# ========================================================================
# DeepSets on [pt, etarel, phirel] whose rho also reads the jet moments,
# the shift-invariant angular moments the vector-neuron row multiplies.
# A rotation about the beam leaves the three columns unchanged, so the
# probe's so2 group is exact; lorentz, so3 and lorentz_iso are measured.
# Pinned to its columns and filed with the embedded rows, so baseline
# only, never under augmentation or the SCL. One Optuna driver per
# constituent count: 3 studies of 200 trials.
#
# No block sets data.features: the experiment configs set
# [pt, etarel, phirel], the network's own columns. Trials run at
# data.n_folds=1 with callbacks=search, so they draw no figure and never
# write to results/. The study name is not on the command line:
# configs/hparams_search/hls4ml/deepsets_moments_optuna.yaml derives it
# from the regime as deepsets_moments_nconst<N>_augmentation_none. The
# space is that of DeepSets plus the moment set {uniform, rankcut}.
#
# Hand-maintained, unlike the train and eval catalogues. Every command is
# commented out; feed the whole file to scripts/cluster/submit_search.sh.

# ========================================================================
# deepsets_moments  nconst=16
# ========================================================================

# ------------------------------------------------------------------------
# deepsets_moments  nconst=16  baseline  (method=augmentation, group=none)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     experiment=hls4ml/deepsets_moments_16 \
#     hparams_search=hls4ml/deepsets_moments_optuna \
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
# deepsets_moments  nconst=50
# ========================================================================

# ------------------------------------------------------------------------
# deepsets_moments  nconst=50  baseline  (method=augmentation, group=none)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     experiment=hls4ml/deepsets_moments_50 \
#     hparams_search=hls4ml/deepsets_moments_optuna \
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
# deepsets_moments  nconst=150
# ========================================================================

# ------------------------------------------------------------------------
# deepsets_moments  nconst=150  baseline  (method=augmentation, group=none)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     experiment=hls4ml/deepsets_moments_150 \
#     hparams_search=hls4ml/deepsets_moments_optuna \
#     hydra.sweeper.n_trials=200 \
#     data.method=augmentation \
#     data.augmentation_scheme=none \
#     data.n_folds=1 \
#     callbacks=search \
#     logger=none \
#     trainer=gpu \
#     trainer.devices=[0] \
#     trainer.max_epochs=50
