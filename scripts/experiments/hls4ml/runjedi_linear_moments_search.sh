#!/usr/bin/env bash
# ========================================================================
# JEDI-LINEAR WITH JET MOMENTS: HYPERPARAMETER SEARCH COMMANDS
# ========================================================================
# JEDI-linear on [pt, etarel, phirel] whose head also reads the jet
# moments, the shift-invariant angular moments the vector-neuron row
# multiplies. A rotation about the beam leaves the three columns
# unchanged, so the probe's so2 group is exact; lorentz, so3 and
# lorentz_iso are measured. Pinned to its columns and filed with the
# embedded rows, so baseline only, never under augmentation or the SCL.
# One Optuna driver per constituent count: 3 studies of 200 trials.
#
# No block sets data.features: the experiment configs set
# [pt, etarel, phirel], the network's own columns. Trials run at
# data.n_folds=1 with callbacks=search, so they draw no figure and never
# write to results/. The study name is not on the command line:
# configs/hparams_search/hls4ml/jedi_linear_moments_optuna.yaml derives it
# from the regime as jedi_linear_moments_nconst<N>_augmentation_none. The
# space is that of JEDI-linear plus the moment set {uniform, rankcut}.
#
# Hand-maintained, unlike the train and eval catalogues. Every command is
# commented out; feed the whole file to scripts/experiments/submit_search.sh.

# ========================================================================
# jedi_linear_moments  nconst=16
# ========================================================================

# ------------------------------------------------------------------------
# jedi_linear_moments  nconst=16  baseline  (method=augmentation, group=none)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     experiment=hls4ml/jedi_linear_moments_16 \
#     hparams_search=hls4ml/jedi_linear_moments_optuna \
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
# jedi_linear_moments  nconst=50
# ========================================================================

# ------------------------------------------------------------------------
# jedi_linear_moments  nconst=50  baseline  (method=augmentation, group=none)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     experiment=hls4ml/jedi_linear_moments_50 \
#     hparams_search=hls4ml/jedi_linear_moments_optuna \
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
# jedi_linear_moments  nconst=150
# ========================================================================

# ------------------------------------------------------------------------
# jedi_linear_moments  nconst=150  baseline  (method=augmentation, group=none)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     experiment=hls4ml/jedi_linear_moments_150 \
#     hparams_search=hls4ml/jedi_linear_moments_optuna \
#     hydra.sweeper.n_trials=200 \
#     data.method=augmentation \
#     data.augmentation_scheme=none \
#     data.n_folds=1 \
#     callbacks=search \
#     logger=none \
#     trainer=gpu \
#     trainer.devices=[0] \
#     trainer.max_epochs=50
