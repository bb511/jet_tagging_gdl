#!/usr/bin/env bash
# ========================================================================
# PELICAN WITHOUT BEAMS: HYPERPARAMETER SEARCH COMMANDS
# ========================================================================
# PELICAN with add_beams=false, on the same search space and the same
# 200-trial budget as the beams-on studies in runpelican_search.sh, so the
# two rows differ in the embedded group alone: without the two beam
# four-vectors the network reads the constituents' Minkowski products and
# nothing else, and its logits are invariant under the full Lorentz group
# rather than SO(2) about the beam. Embedded, so baseline only, never under
# augmentation or the SCL; every probe group is exact for it, so its flip
# rates measure the fp32 floor of the probe.
#
# Two studies, not three: the 150-constituent row is not searched (decided
# 2026-09-27, to spare the H100 week) and runs at the beams-on optimum,
# which configs/experiment/hls4ml/pelican_lorentz_150.yaml carries.
#
# Trials run at data.n_folds=1 with callbacks=search, so they draw no figure and
# never write to results/. The study name is not on the command line:
# configs/hparams_search/hls4ml/pelican_lorentz_optuna.yaml derives it from
# the regime as pelican_lorentz_nconst<N>_augmentation_none. lorentz_eta_max
# defines the probe and is never tuned.
#
# Hand-maintained, unlike the train and eval catalogues. Every command is
# commented out; feed the whole file to scripts/experiments/submit_search.sh.

# ========================================================================
# pelican_lorentz  nconst=16
# ========================================================================

# ------------------------------------------------------------------------
# pelican_lorentz  nconst=16  baseline  (method=augmentation, group=none)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     experiment=hls4ml/pelican_lorentz_16 \
#     hparams_search=hls4ml/pelican_lorentz_optuna \
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
# pelican_lorentz  nconst=50
# ========================================================================

# ------------------------------------------------------------------------
# pelican_lorentz  nconst=50  baseline  (method=augmentation, group=none)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     experiment=hls4ml/pelican_lorentz_50 \
#     hparams_search=hls4ml/pelican_lorentz_optuna \
#     hydra.sweeper.n_trials=200 \
#     data.method=augmentation \
#     data.augmentation_scheme=none \
#     data.n_folds=1 \
#     callbacks=search \
#     logger=none \
#     trainer=gpu \
#     trainer.devices=[0] \
#     trainer.max_epochs=50
