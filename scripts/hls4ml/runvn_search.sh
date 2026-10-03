#!/usr/bin/env bash
# ========================================================================
# VECTOR NEURONS: HYPERPARAMETER SEARCH COMMANDS
# ========================================================================
# Vector neurons on the lifts p (1, eta, phi, eta^2 + phi^2) of the
# constituents, with the angles measured from the leading one. The logits
# are exact under shifts of eta and phi, which are the probe's so2 group
# and, for massless constituents, its lorentz group, and under rotations
# of the (eta, phi) plane; so3 and lorentz_iso are measured. Embedded, so
# baseline only, never under augmentation or the SCL. One Optuna driver
# per constituent count: 3 studies of 200 trials.
#
# No block sets data.features: the experiment configs set [pt, eta, phi],
# the network's own columns. Trials run at data.n_folds=1 with
# callbacks=search, so they draw no figure and never write to results/.
# The study name is not on the command line:
# configs/hparams_search/hls4ml/vn_optuna.yaml derives it from the regime
# as vn_nconst<N>_augmentation_none. units is a numerical setting and
# lorentz_eta_max defines the probe; neither is tuned.
#
# Hand-maintained, unlike the train and eval catalogues. Every command is
# commented out; feed the whole file to scripts/cluster/submit_search.sh.

# ========================================================================
# vn  nconst=16
# ========================================================================

# ------------------------------------------------------------------------
# vn  nconst=16  baseline  (method=augmentation, group=none)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     experiment=hls4ml/vn_16 \
#     hparams_search=hls4ml/vn_optuna \
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
# vn  nconst=50
# ========================================================================

# ------------------------------------------------------------------------
# vn  nconst=50  baseline  (method=augmentation, group=none)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     experiment=hls4ml/vn_50 \
#     hparams_search=hls4ml/vn_optuna \
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
# vn  nconst=150
# ========================================================================

# ------------------------------------------------------------------------
# vn  nconst=150  baseline  (method=augmentation, group=none)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     experiment=hls4ml/vn_150 \
#     hparams_search=hls4ml/vn_optuna \
#     hydra.sweeper.n_trials=200 \
#     data.method=augmentation \
#     data.augmentation_scheme=none \
#     data.n_folds=1 \
#     callbacks=search \
#     logger=none \
#     trainer=gpu \
#     trainer.devices=[0] \
#     trainer.max_epochs=50
