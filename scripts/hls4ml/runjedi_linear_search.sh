#!/usr/bin/env bash
# ========================================================================
# JEDI-LINEAR: HYPERPARAMETER SEARCH COMMANDS
# ========================================================================
# JEDI-linear (arXiv:2508.15468) is, in float, DeepSets with one
# permutation-equivariant context layer. It is a comparator from the
# literature and not a subject of the augmentation-against-SCL grid, so it
# runs the baseline regime alone. One Optuna driver per constituent count: 3
# studies, on the search space and the 200-trial budget of
# configs/hparams_search/hls4ml/jedi_linear_optuna.yaml. The objective is the
# main run's clean val/acc_best. Unlike the embedded rows, the model reads
# data.features, so the feature-set catalogues under pt_eta_phi/ and
# pt_etarel_phirel/ carry the same three blocks with the two overrides of
# their set.
#
# Trials run at data.n_folds=1 with callbacks=search, so they draw no figure
# and never write to results/. The study name is not on the command line:
# the search config derives it from the regime as
# jedi_linear_nconst<N>_augmentation_none. lorentz_eta_max defines the probe
# and is never tuned.
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
#     hydra.sweeper.n_trials=200 \
#     data.method=augmentation \
#     data.augmentation_scheme=none \
#     data.n_folds=1 \
#     callbacks=search \
#     logger=none \
#     trainer=gpu \
#     trainer.devices=[0] \
#     trainer.max_epochs=50
