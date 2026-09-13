#!/usr/bin/env bash
# ========================================================================
# PELICAN HYPERPARAMETER SEARCH COMMANDS
# ========================================================================
# One Optuna driver per constituent count: 3 studies, on the same search space
# and the same 200-trial budget the other architectures get, so the baselines
# stay comparable. PELICAN builds its inputs out of the pairwise Minkowski
# products, and with the beams appended its exact symmetry is SO(2) about the
# beam axis, so it embeds the symmetry rather than acquiring it and is searched
# at baseline only, never under augmentation or the SCL. The beam boosts, SO(3)
# and the isotropic boosts are measured by the probe, as for every other model.
# The objective is the main run's clean val/acc_best; the invariance metrics are
# never tuned on, since that would select for the probe rather than for the
# symmetry.
#
# Trials run at data.n_folds=1 with callbacks=search, so they draw no figure and
# never write to results/. The study name is not on the command line:
# configs/hparams_search/hls4ml/pelican_optuna.yaml derives it from the regime
# as pelican_nconst<N>_augmentation_none. lorentz_eta_max defines the probe
# and is never tuned.
#
# Hand-maintained, unlike the train and eval catalogues. Every command is
# commented out; feed the whole file to scripts/cluster/submit_search.sh.

# ========================================================================
# pelican  nconst=16
# ========================================================================

# ------------------------------------------------------------------------
# pelican  nconst=16  baseline  (method=augmentation, group=none)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     experiment=hls4ml/pelican_16 \
#     hparams_search=hls4ml/pelican_optuna \
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
# pelican  nconst=50
# ========================================================================

# ------------------------------------------------------------------------
# pelican  nconst=50  baseline  (method=augmentation, group=none)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     experiment=hls4ml/pelican_50 \
#     hparams_search=hls4ml/pelican_optuna \
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
# pelican  nconst=150
# ========================================================================

# ------------------------------------------------------------------------
# pelican  nconst=150  baseline  (method=augmentation, group=none)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     experiment=hls4ml/pelican_150 \
#     hparams_search=hls4ml/pelican_optuna \
#     hydra.sweeper.n_trials=200 \
#     data.method=augmentation \
#     data.augmentation_scheme=none \
#     data.n_folds=1 \
#     callbacks=search \
#     logger=none \
#     trainer=gpu \
#     trainer.devices=[0] \
#     trainer.max_epochs=50
