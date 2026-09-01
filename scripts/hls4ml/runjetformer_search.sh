#!/usr/bin/env bash
# ========================================================================
# JETFORMER HYPERPARAMETER SEARCH COMMANDS
# ========================================================================
# One Optuna driver per (constituent count, regime): 3 x 7 = 21 studies, same
# search space and same 100-trial budget throughout so the regimes stay
# comparable. The objective is the main run's clean val/acc_best; the invariance
# metrics are never tuned on, since that would select for the probe rather than
# for the symmetry.
#
# Trials run at data.n_folds=1 with callbacks=search, so they draw no figure and
# never write to results/. The study name is not on the command line:
# configs/hparams_search/hls4ml/jetformer_optuna.yaml derives it from the regime as
# <model>_nconst<N>_<method>_<scheme>; the SCL regimes select _scl_optuna instead,
# which adds algorithm.symm_loss_weight. lorentz_eta_max defines the probe and is
# never tuned.
#
# Hand-maintained, unlike the train and eval catalogues. Every command is
# commented out; feed the whole file to scripts/cluster/submit_search.sh.

# ========================================================================
# jetformer  nconst=16
# ========================================================================

# ------------------------------------------------------------------------
# jetformer  nconst=16  baseline  (method=augmentation, group=none)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     experiment=hls4ml/jetformer_16 \
#     hparams_search=hls4ml/jetformer_optuna \
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
# jetformer  nconst=16  aug_so2  (method=augmentation, group=so2)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     experiment=hls4ml/jetformer_16 \
#     hparams_search=hls4ml/jetformer_optuna \
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
# jetformer  nconst=16  aug_so3  (method=augmentation, group=so3)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     experiment=hls4ml/jetformer_16 \
#     hparams_search=hls4ml/jetformer_optuna \
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
# jetformer  nconst=16  aug_lorentz  (method=augmentation, group=lorentz)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     experiment=hls4ml/jetformer_16 \
#     hparams_search=hls4ml/jetformer_optuna \
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
# jetformer  nconst=16  symm_so2  (method=symm_loss, group=so2)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     experiment=hls4ml/jetformer_16 \
#     hparams_search=hls4ml/jetformer_scl_optuna \
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
# jetformer  nconst=16  symm_so3  (method=symm_loss, group=so3)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     experiment=hls4ml/jetformer_16 \
#     hparams_search=hls4ml/jetformer_scl_optuna \
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
# jetformer  nconst=16  symm_lorentz  (method=symm_loss, group=lorentz)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     experiment=hls4ml/jetformer_16 \
#     hparams_search=hls4ml/jetformer_scl_optuna \
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
# jetformer  nconst=50
# ========================================================================

# ------------------------------------------------------------------------
# jetformer  nconst=50  baseline  (method=augmentation, group=none)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     experiment=hls4ml/jetformer_50 \
#     hparams_search=hls4ml/jetformer_optuna \
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
# jetformer  nconst=50  aug_so2  (method=augmentation, group=so2)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     experiment=hls4ml/jetformer_50 \
#     hparams_search=hls4ml/jetformer_optuna \
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
# jetformer  nconst=50  aug_so3  (method=augmentation, group=so3)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     experiment=hls4ml/jetformer_50 \
#     hparams_search=hls4ml/jetformer_optuna \
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
# jetformer  nconst=50  aug_lorentz  (method=augmentation, group=lorentz)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     experiment=hls4ml/jetformer_50 \
#     hparams_search=hls4ml/jetformer_optuna \
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
# jetformer  nconst=50  symm_so2  (method=symm_loss, group=so2)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     experiment=hls4ml/jetformer_50 \
#     hparams_search=hls4ml/jetformer_scl_optuna \
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
# jetformer  nconst=50  symm_so3  (method=symm_loss, group=so3)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     experiment=hls4ml/jetformer_50 \
#     hparams_search=hls4ml/jetformer_scl_optuna \
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
# jetformer  nconst=50  symm_lorentz  (method=symm_loss, group=lorentz)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     experiment=hls4ml/jetformer_50 \
#     hparams_search=hls4ml/jetformer_scl_optuna \
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
# jetformer  nconst=150
# ========================================================================

# ------------------------------------------------------------------------
# jetformer  nconst=150  baseline  (method=augmentation, group=none)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     experiment=hls4ml/jetformer_150 \
#     hparams_search=hls4ml/jetformer_optuna \
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
# jetformer  nconst=150  aug_so2  (method=augmentation, group=so2)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     experiment=hls4ml/jetformer_150 \
#     hparams_search=hls4ml/jetformer_optuna \
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
# jetformer  nconst=150  aug_so3  (method=augmentation, group=so3)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     experiment=hls4ml/jetformer_150 \
#     hparams_search=hls4ml/jetformer_optuna \
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
# jetformer  nconst=150  aug_lorentz  (method=augmentation, group=lorentz)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     experiment=hls4ml/jetformer_150 \
#     hparams_search=hls4ml/jetformer_optuna \
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
# jetformer  nconst=150  symm_so2  (method=symm_loss, group=so2)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     experiment=hls4ml/jetformer_150 \
#     hparams_search=hls4ml/jetformer_scl_optuna \
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
# jetformer  nconst=150  symm_so3  (method=symm_loss, group=so3)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     experiment=hls4ml/jetformer_150 \
#     hparams_search=hls4ml/jetformer_scl_optuna \
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
# jetformer  nconst=150  symm_lorentz  (method=symm_loss, group=lorentz)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     experiment=hls4ml/jetformer_150 \
#     hparams_search=hls4ml/jetformer_scl_optuna \
#     hydra.sweeper.n_trials=100 \
#     data.method=symm_loss \
#     data.augmentation_scheme=lorentz \
#     data.n_folds=1 \
#     callbacks=search \
#     logger=none \
#     trainer=gpu \
#     trainer.devices=[0] \
#     trainer.max_epochs=100
