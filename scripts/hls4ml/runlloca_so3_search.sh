#!/usr/bin/env bash
# ========================================================================
# LLOCA WITH SO(3) FRAMES: HYPERPARAMETER SEARCH COMMANDS
# ========================================================================
# LLoCa with frames=so3 (LearnedSO3Frames): the local frames are
# rotations, so the logits are invariant under SO(3), the probe's so2 and
# so3 groups are exact, and lorentz and lorentz_iso are measured.
# Embedded, so baseline only, never under augmentation or the SCL. One
# Optuna driver per constituent count: 3 studies, on the search space and
# the 200-trial budget that the four runlloca_*_search.sh share, so the
# four rows differ in the frames class alone.
#
# Trials run at data.n_folds=1 with callbacks=search, so they draw no
# figure and never write to results/. The study name is not on the
# command line: configs/hparams_search/hls4ml/lloca_so3_optuna.yaml
# derives it from the regime as lloca_so3_nconst<N>_augmentation_none.
# The search space is provisional until the campaign discussion. units
# and mass_reg are numerical settings and lorentz_eta_max defines the
# probe; none of the three is tuned.
#
# Hand-maintained, unlike the train and eval catalogues. Every command is
# commented out; feed the whole file to scripts/cluster/submit_search.sh.

# ========================================================================
# lloca_so3  nconst=16
# ========================================================================

# ------------------------------------------------------------------------
# lloca_so3  nconst=16  baseline  (method=augmentation, group=none)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     experiment=hls4ml/lloca_so3_16 \
#     hparams_search=hls4ml/lloca_so3_optuna \
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
# lloca_so3  nconst=50
# ========================================================================

# ------------------------------------------------------------------------
# lloca_so3  nconst=50  baseline  (method=augmentation, group=none)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     experiment=hls4ml/lloca_so3_50 \
#     hparams_search=hls4ml/lloca_so3_optuna \
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
# lloca_so3  nconst=150
# ========================================================================

# ------------------------------------------------------------------------
# lloca_so3  nconst=150  baseline  (method=augmentation, group=none)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     experiment=hls4ml/lloca_so3_150 \
#     hparams_search=hls4ml/lloca_so3_optuna \
#     hydra.sweeper.n_trials=200 \
#     data.method=augmentation \
#     data.augmentation_scheme=none \
#     data.n_folds=1 \
#     callbacks=search \
#     logger=none \
#     trainer=gpu \
#     trainer.devices=[0] \
#     trainer.max_epochs=50
