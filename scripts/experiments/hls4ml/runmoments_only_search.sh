#!/usr/bin/env bash
# ========================================================================
# MOMENTS ONLY: HYPERPARAMETER SEARCH COMMANDS
# ========================================================================
# The control of the moment rows: an MLP on the jet moments of
# [pt, etarel, phirel] and the multiplicity alone, with no
# per-constituent network. The probe's so2 group leaves the columns
# unchanged, and a boost along the beam shifts every etarel of a massless
# constituent by one amount, which the moments do not see, so so2 and
# lorentz are exact; so3 and lorentz_iso are measured. Pinned to its
# columns and filed with the embedded rows, so baseline only, never under
# augmentation or the SCL. One Optuna driver per constituent count:
# 3 studies of 200 trials.
#
# No block sets data.features: the experiment configs set
# [pt, etarel, phirel], the network's own columns. Trials run at
# data.n_folds=1 with callbacks=search, so they draw no figure and never
# write to results/. The study name is not on the command line:
# configs/hparams_search/hls4ml/moments_only_optuna.yaml derives it from
# the regime as moments_only_nconst<N>_augmentation_none. The space is
# that of DeepSets without phi and the pooling, plus the moment set
# {uniform, rankcut}.
#
# Hand-maintained, unlike the train and eval catalogues. Every command is
# commented out; feed the whole file to scripts/experiments/submit_search.sh.

# ========================================================================
# moments_only  nconst=16
# ========================================================================

# ------------------------------------------------------------------------
# moments_only  nconst=16  baseline  (method=augmentation, group=none)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     experiment=hls4ml/moments_only_16 \
#     hparams_search=hls4ml/moments_only_optuna \
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
# moments_only  nconst=50
# ========================================================================

# ------------------------------------------------------------------------
# moments_only  nconst=50  baseline  (method=augmentation, group=none)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     experiment=hls4ml/moments_only_50 \
#     hparams_search=hls4ml/moments_only_optuna \
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
# moments_only  nconst=150
# ========================================================================

# ------------------------------------------------------------------------
# moments_only  nconst=150  baseline  (method=augmentation, group=none)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     experiment=hls4ml/moments_only_150 \
#     hparams_search=hls4ml/moments_only_optuna \
#     hydra.sweeper.n_trials=200 \
#     data.method=augmentation \
#     data.augmentation_scheme=none \
#     data.n_folds=1 \
#     callbacks=search \
#     logger=none \
#     trainer=gpu \
#     trainer.devices=[0] \
#     trainer.max_epochs=50
