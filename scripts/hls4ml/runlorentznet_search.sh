#!/usr/bin/env bash
# ========================================================================
# LORENTZNET HYPERPARAMETER SEARCH COMMANDS
# ========================================================================
# One Optuna driver per constituent count: 3 studies, on the same search space
# and the same 100-trial budget the other architectures get, so the baselines
# stay comparable. LorentzNet embeds the symmetry rather than acquiring it, so
# it is searched at baseline only, never under augmentation or the SCL. The
# objective is the main run's clean val/acc_best; the invariance metrics are
# never tuned on, since that would select for the probe rather than for the
# symmetry.
#
# Trials run at data.n_folds=1 with callbacks=search, so they draw no figure and
# never write to results/. The study name is not on the command line:
# configs/hparams_search/hls4ml/lorentznet_optuna.yaml derives it from the regime
# as lorentznet_nconst<N>_augmentation_none. lorentz_eta_max defines the probe
# and is never tuned.
#
# Hand-maintained, unlike the train and eval catalogues. Every command is
# commented out; feed the whole file to scripts/cluster/submit_search.sh.

# ========================================================================
# lorentznet  nconst=16
# ========================================================================

# ------------------------------------------------------------------------
# lorentznet  nconst=16  baseline  (method=augmentation, group=none)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     experiment=hls4ml/lorentznet_16 \
#     hparams_search=hls4ml/lorentznet_optuna \
#     hydra.sweeper.n_trials=100 \
#     data.method=augmentation \
#     data.augmentation_scheme=none \
#     data.n_folds=1 \
#     callbacks=search \
#     logger=none \
#     trainer=gpu \
#     trainer.devices=[0] \
#     trainer.max_epochs=100

# ========================================================================
# lorentznet  nconst=50
# ========================================================================

# ------------------------------------------------------------------------
# lorentznet  nconst=50  baseline  (method=augmentation, group=none)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     experiment=hls4ml/lorentznet_50 \
#     hparams_search=hls4ml/lorentznet_optuna \
#     hydra.sweeper.n_trials=100 \
#     data.method=augmentation \
#     data.augmentation_scheme=none \
#     data.n_folds=1 \
#     callbacks=search \
#     logger=none \
#     trainer=gpu \
#     trainer.devices=[0] \
#     trainer.max_epochs=100

# ========================================================================
# lorentznet  nconst=150
# ========================================================================

# ------------------------------------------------------------------------
# lorentznet  nconst=150  baseline  (method=augmentation, group=none)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     experiment=hls4ml/lorentznet_150 \
#     hparams_search=hls4ml/lorentznet_optuna \
#     hydra.sweeper.n_trials=100 \
#     data.method=augmentation \
#     data.augmentation_scheme=none \
#     data.n_folds=1 \
#     callbacks=search \
#     logger=none \
#     trainer=gpu \
#     trainer.devices=[0] \
#     trainer.max_epochs=100
