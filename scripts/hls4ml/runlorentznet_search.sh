#!/usr/bin/env bash
# ========================================================================
# LORENTZNET HYPERPARAMETER SEARCH COMMANDS
# ========================================================================
# One Optuna driver per (constituent count, regime): 3 x 7 = 21 studies, with
# the same search space and the same 100-trial budget for every one of them,
# so that the regimes stay comparable.
#
# The objective is the main run's clean val/acc_best, which src/train.py
# returns. The invariance metrics are reported by the callbacks and never
# tuned on: a search that maximised them would select for the probe rather
# than for the symmetry.
#
# Trials run at data.n_folds=1 with callbacks=search, so none of them draws a
# figure or writes to results/. The study name is not on the command line:
# configs/hparams_search/hls4ml/lorentznet_optuna.yaml derives it from the regime
# as <model>_nconst<N>_<method>_<scheme>. The three SCL regimes select the
# _scl_optuna variant instead, which adds algorithm.symm_loss_weight to the
# space; lorentz_eta_max is never tuned, since it defines the probe.
#
# Hand-maintained, unlike the train and eval catalogues: no tool regenerates
# this file. Every command is commented out; feed the whole file to
# scripts/cluster/submit_search.sh, which parses these blocks and launches one
# nohup'd driver each.

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

# ------------------------------------------------------------------------
# lorentznet  nconst=16  aug_so2  (method=augmentation, group=so2)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     experiment=hls4ml/lorentznet_16 \
#     hparams_search=hls4ml/lorentznet_optuna \
#     hydra.sweeper.n_trials=100 \
#     data.method=augmentation \
#     data.augmentation_scheme=so2 \
#     data.n_folds=1 \
#     callbacks=search \
#     logger=none \
#     trainer=gpu \
#     trainer.devices=[0] \
#     trainer.max_epochs=100

# ------------------------------------------------------------------------
# lorentznet  nconst=16  aug_so3  (method=augmentation, group=so3)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     experiment=hls4ml/lorentznet_16 \
#     hparams_search=hls4ml/lorentznet_optuna \
#     hydra.sweeper.n_trials=100 \
#     data.method=augmentation \
#     data.augmentation_scheme=so3 \
#     data.n_folds=1 \
#     callbacks=search \
#     logger=none \
#     trainer=gpu \
#     trainer.devices=[0] \
#     trainer.max_epochs=100

# ------------------------------------------------------------------------
# lorentznet  nconst=16  aug_lorentz  (method=augmentation, group=lorentz)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     experiment=hls4ml/lorentznet_16 \
#     hparams_search=hls4ml/lorentznet_optuna \
#     hydra.sweeper.n_trials=100 \
#     data.method=augmentation \
#     data.augmentation_scheme=lorentz \
#     data.n_folds=1 \
#     callbacks=search \
#     logger=none \
#     trainer=gpu \
#     trainer.devices=[0] \
#     trainer.max_epochs=100

# ------------------------------------------------------------------------
# lorentznet  nconst=16  symm_so2  (method=symm_loss, group=so2)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     experiment=hls4ml/lorentznet_16 \
#     hparams_search=hls4ml/lorentznet_scl_optuna \
#     hydra.sweeper.n_trials=100 \
#     data.method=symm_loss \
#     data.augmentation_scheme=so2 \
#     data.n_folds=1 \
#     callbacks=search \
#     logger=none \
#     trainer=gpu \
#     trainer.devices=[0] \
#     trainer.max_epochs=100

# ------------------------------------------------------------------------
# lorentznet  nconst=16  symm_so3  (method=symm_loss, group=so3)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     experiment=hls4ml/lorentznet_16 \
#     hparams_search=hls4ml/lorentznet_scl_optuna \
#     hydra.sweeper.n_trials=100 \
#     data.method=symm_loss \
#     data.augmentation_scheme=so3 \
#     data.n_folds=1 \
#     callbacks=search \
#     logger=none \
#     trainer=gpu \
#     trainer.devices=[0] \
#     trainer.max_epochs=100

# ------------------------------------------------------------------------
# lorentznet  nconst=16  symm_lorentz  (method=symm_loss, group=lorentz)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     experiment=hls4ml/lorentznet_16 \
#     hparams_search=hls4ml/lorentznet_scl_optuna \
#     hydra.sweeper.n_trials=100 \
#     data.method=symm_loss \
#     data.augmentation_scheme=lorentz \
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

# ------------------------------------------------------------------------
# lorentznet  nconst=50  aug_so2  (method=augmentation, group=so2)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     experiment=hls4ml/lorentznet_50 \
#     hparams_search=hls4ml/lorentznet_optuna \
#     hydra.sweeper.n_trials=100 \
#     data.method=augmentation \
#     data.augmentation_scheme=so2 \
#     data.n_folds=1 \
#     callbacks=search \
#     logger=none \
#     trainer=gpu \
#     trainer.devices=[0] \
#     trainer.max_epochs=100

# ------------------------------------------------------------------------
# lorentznet  nconst=50  aug_so3  (method=augmentation, group=so3)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     experiment=hls4ml/lorentznet_50 \
#     hparams_search=hls4ml/lorentznet_optuna \
#     hydra.sweeper.n_trials=100 \
#     data.method=augmentation \
#     data.augmentation_scheme=so3 \
#     data.n_folds=1 \
#     callbacks=search \
#     logger=none \
#     trainer=gpu \
#     trainer.devices=[0] \
#     trainer.max_epochs=100

# ------------------------------------------------------------------------
# lorentznet  nconst=50  aug_lorentz  (method=augmentation, group=lorentz)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     experiment=hls4ml/lorentznet_50 \
#     hparams_search=hls4ml/lorentznet_optuna \
#     hydra.sweeper.n_trials=100 \
#     data.method=augmentation \
#     data.augmentation_scheme=lorentz \
#     data.n_folds=1 \
#     callbacks=search \
#     logger=none \
#     trainer=gpu \
#     trainer.devices=[0] \
#     trainer.max_epochs=100

# ------------------------------------------------------------------------
# lorentznet  nconst=50  symm_so2  (method=symm_loss, group=so2)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     experiment=hls4ml/lorentznet_50 \
#     hparams_search=hls4ml/lorentznet_scl_optuna \
#     hydra.sweeper.n_trials=100 \
#     data.method=symm_loss \
#     data.augmentation_scheme=so2 \
#     data.n_folds=1 \
#     callbacks=search \
#     logger=none \
#     trainer=gpu \
#     trainer.devices=[0] \
#     trainer.max_epochs=100

# ------------------------------------------------------------------------
# lorentznet  nconst=50  symm_so3  (method=symm_loss, group=so3)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     experiment=hls4ml/lorentznet_50 \
#     hparams_search=hls4ml/lorentznet_scl_optuna \
#     hydra.sweeper.n_trials=100 \
#     data.method=symm_loss \
#     data.augmentation_scheme=so3 \
#     data.n_folds=1 \
#     callbacks=search \
#     logger=none \
#     trainer=gpu \
#     trainer.devices=[0] \
#     trainer.max_epochs=100

# ------------------------------------------------------------------------
# lorentznet  nconst=50  symm_lorentz  (method=symm_loss, group=lorentz)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     experiment=hls4ml/lorentznet_50 \
#     hparams_search=hls4ml/lorentznet_scl_optuna \
#     hydra.sweeper.n_trials=100 \
#     data.method=symm_loss \
#     data.augmentation_scheme=lorentz \
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

# ------------------------------------------------------------------------
# lorentznet  nconst=150  aug_so2  (method=augmentation, group=so2)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     experiment=hls4ml/lorentznet_150 \
#     hparams_search=hls4ml/lorentznet_optuna \
#     hydra.sweeper.n_trials=100 \
#     data.method=augmentation \
#     data.augmentation_scheme=so2 \
#     data.n_folds=1 \
#     callbacks=search \
#     logger=none \
#     trainer=gpu \
#     trainer.devices=[0] \
#     trainer.max_epochs=100

# ------------------------------------------------------------------------
# lorentznet  nconst=150  aug_so3  (method=augmentation, group=so3)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     experiment=hls4ml/lorentznet_150 \
#     hparams_search=hls4ml/lorentznet_optuna \
#     hydra.sweeper.n_trials=100 \
#     data.method=augmentation \
#     data.augmentation_scheme=so3 \
#     data.n_folds=1 \
#     callbacks=search \
#     logger=none \
#     trainer=gpu \
#     trainer.devices=[0] \
#     trainer.max_epochs=100

# ------------------------------------------------------------------------
# lorentznet  nconst=150  aug_lorentz  (method=augmentation, group=lorentz)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     experiment=hls4ml/lorentznet_150 \
#     hparams_search=hls4ml/lorentznet_optuna \
#     hydra.sweeper.n_trials=100 \
#     data.method=augmentation \
#     data.augmentation_scheme=lorentz \
#     data.n_folds=1 \
#     callbacks=search \
#     logger=none \
#     trainer=gpu \
#     trainer.devices=[0] \
#     trainer.max_epochs=100

# ------------------------------------------------------------------------
# lorentznet  nconst=150  symm_so2  (method=symm_loss, group=so2)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     experiment=hls4ml/lorentznet_150 \
#     hparams_search=hls4ml/lorentznet_scl_optuna \
#     hydra.sweeper.n_trials=100 \
#     data.method=symm_loss \
#     data.augmentation_scheme=so2 \
#     data.n_folds=1 \
#     callbacks=search \
#     logger=none \
#     trainer=gpu \
#     trainer.devices=[0] \
#     trainer.max_epochs=100

# ------------------------------------------------------------------------
# lorentznet  nconst=150  symm_so3  (method=symm_loss, group=so3)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     experiment=hls4ml/lorentznet_150 \
#     hparams_search=hls4ml/lorentznet_scl_optuna \
#     hydra.sweeper.n_trials=100 \
#     data.method=symm_loss \
#     data.augmentation_scheme=so3 \
#     data.n_folds=1 \
#     callbacks=search \
#     logger=none \
#     trainer=gpu \
#     trainer.devices=[0] \
#     trainer.max_epochs=100

# ------------------------------------------------------------------------
# lorentznet  nconst=150  symm_lorentz  (method=symm_loss, group=lorentz)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     experiment=hls4ml/lorentznet_150 \
#     hparams_search=hls4ml/lorentznet_scl_optuna \
#     hydra.sweeper.n_trials=100 \
#     data.method=symm_loss \
#     data.augmentation_scheme=lorentz \
#     data.n_folds=1 \
#     callbacks=search \
#     logger=none \
#     trainer=gpu \
#     trainer.devices=[0] \
#     trainer.max_epochs=100
