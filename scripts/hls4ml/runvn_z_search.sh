#!/usr/bin/env bash
# ========================================================================
# VECTOR NEURONS, SO(2)xSO(1,1): HYPERPARAMETER SEARCH COMMANDS
# ========================================================================
# Vector neurons with symmetry=z: the four-vector is split into the
# transverse pair (px, py) and the longitudinal pair (E, pz), each with
# weights of its own, so the logits are invariant under SO(2)xSO(1,1),
# the probe's so2 and lorentz groups are exact, and so3 and lorentz_iso
# are measured. Embedded, so baseline only, never under augmentation or
# the SCL. One Optuna driver per constituent count: 3 studies, on the
# search space and the 200-trial budget that the two runvn_*_search.sh
# share, so the two rows differ in the partition alone.
#
# Trials run at data.n_folds=1 with callbacks=search, so they draw no
# figure and never write to results/. The study name is not on the
# command line: configs/hparams_search/hls4ml/vn_z_optuna.yaml derives
# it from the regime as vn_z_nconst<N>_augmentation_none. The search
# space is provisional until the campaign discussion. units is a
# numerical setting and lorentz_eta_max defines the probe; neither is
# tuned.
#
# Hand-maintained, unlike the train and eval catalogues. Every command is
# commented out; feed the whole file to scripts/cluster/submit_search.sh.

# ========================================================================
# vn_z  nconst=16
# ========================================================================

# ------------------------------------------------------------------------
# vn_z  nconst=16  baseline  (method=augmentation, group=none)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     experiment=hls4ml/vn_z_16 \
#     hparams_search=hls4ml/vn_z_optuna \
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
# vn_z  nconst=50
# ========================================================================

# ------------------------------------------------------------------------
# vn_z  nconst=50  baseline  (method=augmentation, group=none)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     experiment=hls4ml/vn_z_50 \
#     hparams_search=hls4ml/vn_z_optuna \
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
# vn_z  nconst=150
# ========================================================================

# ------------------------------------------------------------------------
# vn_z  nconst=150  baseline  (method=augmentation, group=none)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     experiment=hls4ml/vn_z_150 \
#     hparams_search=hls4ml/vn_z_optuna \
#     hydra.sweeper.n_trials=200 \
#     data.method=augmentation \
#     data.augmentation_scheme=none \
#     data.n_folds=1 \
#     callbacks=search \
#     logger=none \
#     trainer=gpu \
#     trainer.devices=[0] \
#     trainer.max_epochs=50
