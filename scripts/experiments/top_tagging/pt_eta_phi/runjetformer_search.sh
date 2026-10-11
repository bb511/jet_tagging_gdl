#!/usr/bin/env bash
# ========================================================================
# JETFORMER: TOP-TAGGING HYPERPARAMETER SEARCH COMMANDS, FEATURE SET
# [pt, eta, phi]
# ========================================================================
# The 7 blocks of ../pt_etarel_phirel/runjetformer_search.sh with two overrides added to each:
# data.features, which the model reads in place of [pt, etarel, phirel],
# and hydra.sweeper.storage, a study database of this feature set's own
# under logs/optuna/top_tagging/pt_eta_phi/. The study name derives from the
# regime alone (top_tagging_<model>_nconst100_<method>_<scheme>), so
# without the second override the trials would join the [pt, etarel,
# phirel] study of the same name; submit_search.sh refuses a block that
# carries the first without the second. The first trial of each study,
# enqueued by the campaign before any driver starts, is the optimum of the
# row's hls4ml N = 150 study of the same regime on [pt, eta, phi]. The
# evaluation of the trained models lands in results/top_tagging/pt_eta_phi/
# by the same feature list.
#
# Trials run at data.n_folds=1 with callbacks=search, so they draw no
# figure and never write to results/. The SCL regimes select _scl_optuna
# instead, which adds algorithm.symm_loss_weight. lorentz_eta_max defines
# the probe and is never tuned.
#
# Hand-maintained, unlike the train and eval catalogues. Every command is
# commented out; feed the whole file to scripts/experiments/submit_search.sh.

# ========================================================================
# jetformer  nconst=100
# ========================================================================

# ------------------------------------------------------------------------
# jetformer  nconst=100  baseline  (method=augmentation, group=none)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     experiment=top_tagging/jetformer_100 \
#     hparams_search=top_tagging/jetformer_optuna \
#     data.features='[pt,eta,phi]' \
#     hydra.sweeper.storage="'sqlite:///logs/optuna/top_tagging/pt_eta_phi/jetformer.db?timeout=600'" \
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
# jetformer  nconst=100  aug_so2  (method=augmentation, group=so2)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     experiment=top_tagging/jetformer_100 \
#     hparams_search=top_tagging/jetformer_optuna \
#     data.features='[pt,eta,phi]' \
#     hydra.sweeper.storage="'sqlite:///logs/optuna/top_tagging/pt_eta_phi/jetformer.db?timeout=600'" \
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
# jetformer  nconst=100  aug_so3  (method=augmentation, group=so3)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     experiment=top_tagging/jetformer_100 \
#     hparams_search=top_tagging/jetformer_optuna \
#     data.features='[pt,eta,phi]' \
#     hydra.sweeper.storage="'sqlite:///logs/optuna/top_tagging/pt_eta_phi/jetformer.db?timeout=600'" \
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
# jetformer  nconst=100  aug_lorentz  (method=augmentation, group=lorentz)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     experiment=top_tagging/jetformer_100 \
#     hparams_search=top_tagging/jetformer_optuna \
#     data.features='[pt,eta,phi]' \
#     hydra.sweeper.storage="'sqlite:///logs/optuna/top_tagging/pt_eta_phi/jetformer.db?timeout=600'" \
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
# jetformer  nconst=100  symm_so2  (method=symm_loss, group=so2)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     experiment=top_tagging/jetformer_100 \
#     hparams_search=top_tagging/jetformer_scl_optuna \
#     data.features='[pt,eta,phi]' \
#     hydra.sweeper.storage="'sqlite:///logs/optuna/top_tagging/pt_eta_phi/jetformer.db?timeout=600'" \
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
# jetformer  nconst=100  symm_so3  (method=symm_loss, group=so3)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     experiment=top_tagging/jetformer_100 \
#     hparams_search=top_tagging/jetformer_scl_optuna \
#     data.features='[pt,eta,phi]' \
#     hydra.sweeper.storage="'sqlite:///logs/optuna/top_tagging/pt_eta_phi/jetformer.db?timeout=600'" \
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
# jetformer  nconst=100  symm_lorentz  (method=symm_loss, group=lorentz)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     experiment=top_tagging/jetformer_100 \
#     hparams_search=top_tagging/jetformer_scl_optuna \
#     data.features='[pt,eta,phi]' \
#     hydra.sweeper.storage="'sqlite:///logs/optuna/top_tagging/pt_eta_phi/jetformer.db?timeout=600'" \
#     hydra.sweeper.n_trials=100 \
#     data.method=symm_loss \
#     data.augmentation_scheme=lorentz \
#     data.n_folds=1 \
#     callbacks=search \
#     logger=none \
#     trainer=gpu \
#     trainer.devices=[0] \
#     trainer.max_epochs=50
