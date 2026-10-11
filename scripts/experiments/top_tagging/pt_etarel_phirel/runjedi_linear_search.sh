#!/usr/bin/env bash
# ========================================================================
# JEDI-LINEAR: TOP-TAGGING HYPERPARAMETER SEARCH COMMANDS
# ========================================================================
# JEDI-linear on [pt, etarel, phirel] of the leading 100 constituents
# of the top-tagging reference set (Zenodo 2603256). One block per regime,
# each one study of 100 trials of 50 epochs. Before any driver starts, the
# campaign enqueues as the first trial of each study the optimum of the
# row's hls4ml N = 150 study of the same regime on the same features. The
# objective is the main run's clean val/acc_best; the invariance metrics
# are never tuned on.
#
# No block sets data.features: the experiment config sets
# [pt, etarel, phirel]. Trials run at data.n_folds=1 with
# callbacks=search, so they draw no figure and never write to results/.
# The study name is not on the command line:
# configs/hparams_search/top_tagging/jedi_linear_optuna.yaml derives it from
# the regime as top_tagging_jedi_linear_nconst100_<method>_<scheme> and keeps
# it in logs/optuna/top_tagging/jedi_linear.db; the SCL regimes select
# _scl_optuna instead, which adds algorithm.symm_loss_weight.
# lorentz_eta_max defines the probe and is never tuned.
#
# Hand-maintained, unlike the train and eval catalogues. Every command is
# commented out; feed the whole file to scripts/experiments/submit_search.sh.

# ========================================================================
# jedi_linear  nconst=100
# ========================================================================

# ------------------------------------------------------------------------
# jedi_linear  nconst=100  baseline  (method=augmentation, group=none)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     experiment=top_tagging/jedi_linear_100 \
#     hparams_search=top_tagging/jedi_linear_optuna \
#     hydra.sweeper.n_trials=100 \
#     data.method=augmentation \
#     data.augmentation_scheme=none \
#     data.n_folds=1 \
#     callbacks=search \
#     logger=none \
#     trainer=gpu \
#     trainer.devices=[0] \
#     trainer.max_epochs=50

# ------------------------------------------------------------------------
# jedi_linear  nconst=100  aug_so2  (method=augmentation, group=so2)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     experiment=top_tagging/jedi_linear_100 \
#     hparams_search=top_tagging/jedi_linear_optuna \
#     hydra.sweeper.n_trials=100 \
#     data.method=augmentation \
#     data.augmentation_scheme=so2 \
#     data.n_folds=1 \
#     callbacks=search \
#     logger=none \
#     trainer=gpu \
#     trainer.devices=[0] \
#     trainer.max_epochs=50

# ------------------------------------------------------------------------
# jedi_linear  nconst=100  aug_so3  (method=augmentation, group=so3)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     experiment=top_tagging/jedi_linear_100 \
#     hparams_search=top_tagging/jedi_linear_optuna \
#     hydra.sweeper.n_trials=100 \
#     data.method=augmentation \
#     data.augmentation_scheme=so3 \
#     data.n_folds=1 \
#     callbacks=search \
#     logger=none \
#     trainer=gpu \
#     trainer.devices=[0] \
#     trainer.max_epochs=50

# ------------------------------------------------------------------------
# jedi_linear  nconst=100  aug_lorentz  (method=augmentation, group=lorentz)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     experiment=top_tagging/jedi_linear_100 \
#     hparams_search=top_tagging/jedi_linear_optuna \
#     hydra.sweeper.n_trials=100 \
#     data.method=augmentation \
#     data.augmentation_scheme=lorentz \
#     data.n_folds=1 \
#     callbacks=search \
#     logger=none \
#     trainer=gpu \
#     trainer.devices=[0] \
#     trainer.max_epochs=50

# ------------------------------------------------------------------------
# jedi_linear  nconst=100  symm_so2  (method=symm_loss, group=so2)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     experiment=top_tagging/jedi_linear_100 \
#     hparams_search=top_tagging/jedi_linear_scl_optuna \
#     hydra.sweeper.n_trials=100 \
#     data.method=symm_loss \
#     data.augmentation_scheme=so2 \
#     data.n_folds=1 \
#     callbacks=search \
#     logger=none \
#     trainer=gpu \
#     trainer.devices=[0] \
#     trainer.max_epochs=50

# ------------------------------------------------------------------------
# jedi_linear  nconst=100  symm_so3  (method=symm_loss, group=so3)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     experiment=top_tagging/jedi_linear_100 \
#     hparams_search=top_tagging/jedi_linear_scl_optuna \
#     hydra.sweeper.n_trials=100 \
#     data.method=symm_loss \
#     data.augmentation_scheme=so3 \
#     data.n_folds=1 \
#     callbacks=search \
#     logger=none \
#     trainer=gpu \
#     trainer.devices=[0] \
#     trainer.max_epochs=50

# ------------------------------------------------------------------------
# jedi_linear  nconst=100  symm_lorentz  (method=symm_loss, group=lorentz)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     experiment=top_tagging/jedi_linear_100 \
#     hparams_search=top_tagging/jedi_linear_scl_optuna \
#     hydra.sweeper.n_trials=100 \
#     data.method=symm_loss \
#     data.augmentation_scheme=lorentz \
#     data.n_folds=1 \
#     callbacks=search \
#     logger=none \
#     trainer=gpu \
#     trainer.devices=[0] \
#     trainer.max_epochs=50
