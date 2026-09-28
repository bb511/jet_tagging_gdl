#!/usr/bin/env bash
# ========================================================================
# L-GATR-SLIM HYPERPARAMETER SEARCH COMMANDS
# ========================================================================
# L-GATr-slim with the spurion tokens: the time vector (1, 0, 0, 0) and
# the two lightlike beams (1, 0, 0, +-1) enter as tokens beside the
# constituents, so the logits are invariant under SO(2) about the beam and
# under p_z -> -p_z, and the probe measures every other group. Embedded, so
# baseline only, never under augmentation or the SCL. One Optuna driver per
# constituent count: 3 studies, on the search space and the 200-trial
# budget that runlgatr_slim_lorentz_search.sh shares, so the two rows
# differ in the embedded group alone.
#
# Trials run at data.n_folds=1 with callbacks=search, so they draw no figure and
# never write to results/. The study name is not on the command line:
# configs/hparams_search/hls4ml/lgatr_slim_optuna.yaml derives it from the
# regime as lgatr_slim_nconst<N>_augmentation_none. units is the references'
# momentum scale and lorentz_eta_max defines the probe; neither is tuned.
#
# Hand-maintained, unlike the train and eval catalogues. Every command is
# commented out; feed the whole file to scripts/cluster/submit_search.sh.

# ========================================================================
# lgatr_slim  nconst=16
# ========================================================================

# ------------------------------------------------------------------------
# lgatr_slim  nconst=16  baseline  (method=augmentation, group=none)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     experiment=hls4ml/lgatr_slim_16 \
#     hparams_search=hls4ml/lgatr_slim_optuna \
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
# lgatr_slim  nconst=50
# ========================================================================

# ------------------------------------------------------------------------
# lgatr_slim  nconst=50  baseline  (method=augmentation, group=none)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     experiment=hls4ml/lgatr_slim_50 \
#     hparams_search=hls4ml/lgatr_slim_optuna \
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
# lgatr_slim  nconst=150
# ========================================================================

# ------------------------------------------------------------------------
# lgatr_slim  nconst=150  baseline  (method=augmentation, group=none)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     experiment=hls4ml/lgatr_slim_150 \
#     hparams_search=hls4ml/lgatr_slim_optuna \
#     hydra.sweeper.n_trials=200 \
#     data.method=augmentation \
#     data.augmentation_scheme=none \
#     data.n_folds=1 \
#     callbacks=search \
#     logger=none \
#     trainer=gpu \
#     trainer.devices=[0] \
#     trainer.max_epochs=50
