#!/usr/bin/env bash
# ========================================================================
# JEDI-LINEAR HYPERPARAMETER SEARCH COMMANDS, FEATURE SET [pt, eta, phi]
# ========================================================================
# The 3 blocks of ../runjedi_linear_search.sh with two overrides added to each:
# data.features, which the model reads in place of the four-momentum, and
# hydra.sweeper.storage, a study database of this feature set's own under
# logs/optuna/hls4ml/pt_eta_phi/. The study name derives from the regime alone
# (<model>_nconst<N>_<method>_<scheme>), so without the second override the
# trials would join the px_py_pz_e study of the same name; submit_search.sh
# refuses a block that carries the first without the second. The evaluation of
# the trained models lands in results/pt_eta_phi/ by the same feature list.
#
# Trials run at data.n_folds=1 with callbacks=search, so they draw no figure and
# never write to results/. The row runs the baseline regime alone, so there is
# no _scl_optuna counterpart. lorentz_eta_max defines the probe and is never
# tuned.
#
# Hand-maintained, unlike the train and eval catalogues. Every command is
# commented out; feed the whole file to scripts/cluster/submit_search.sh.

# ========================================================================
# jedi_linear  nconst=16
# ========================================================================

# ------------------------------------------------------------------------
# jedi_linear  nconst=16  baseline  (method=augmentation, group=none)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     experiment=hls4ml/jedi_linear_16 \
#     hparams_search=hls4ml/jedi_linear_optuna \
#     data.features='[pt,eta,phi]' \
#     hydra.sweeper.storage="'sqlite:///logs/optuna/hls4ml/pt_eta_phi/jedi_linear.db?timeout=600'" \
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
# jedi_linear  nconst=50
# ========================================================================

# ------------------------------------------------------------------------
# jedi_linear  nconst=50  baseline  (method=augmentation, group=none)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     experiment=hls4ml/jedi_linear_50 \
#     hparams_search=hls4ml/jedi_linear_optuna \
#     data.features='[pt,eta,phi]' \
#     hydra.sweeper.storage="'sqlite:///logs/optuna/hls4ml/pt_eta_phi/jedi_linear.db?timeout=600'" \
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
# jedi_linear  nconst=150
# ========================================================================

# ------------------------------------------------------------------------
# jedi_linear  nconst=150  baseline  (method=augmentation, group=none)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     experiment=hls4ml/jedi_linear_150 \
#     hparams_search=hls4ml/jedi_linear_optuna \
#     data.features='[pt,eta,phi]' \
#     hydra.sweeper.storage="'sqlite:///logs/optuna/hls4ml/pt_eta_phi/jedi_linear.db?timeout=600'" \
#     hydra.sweeper.n_trials=200 \
#     data.method=augmentation \
#     data.augmentation_scheme=none \
#     data.n_folds=1 \
#     callbacks=search \
#     logger=none \
#     trainer=gpu \
#     trainer.devices=[0] \
#     trainer.max_epochs=50
