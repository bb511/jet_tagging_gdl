#!/usr/bin/env bash
# ========================================================================
# JETFORMER HYPERPARAMETER SEARCH COMMANDS, FEATURE SET [pt, eta, phi]
# ========================================================================
# The 21 blocks of ../runjetformer_search.sh with two overrides added to each:
# data.features, which the model reads in place of the four-momentum, and
# hydra.sweeper.storage, a study database of this feature set's own under
# logs/optuna/hls4ml/pt_eta_phi/. The study name derives from the regime alone
# (<model>_nconst<N>_<method>_<scheme>), so without the second override the
# trials would join the px_py_pz_e study of the same name; submit_search.sh
# refuses a block that carries the first without the second. The evaluation of
# the trained models lands in results/pt_eta_phi/ by the same feature list.
#
# Trials run at data.n_folds=1 with callbacks=search, so they draw no figure and
# never write to results/. The SCL regimes select _scl_optuna instead, which
# adds algorithm.symm_loss_weight. lorentz_eta_max defines the probe and is
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
#     data.features='[pt,eta,phi]' \
#     hydra.sweeper.storage="'sqlite:///logs/optuna/hls4ml/pt_eta_phi/jetformer.db?timeout=600'" \
#     hydra.sweeper.n_trials=200 \
#     data.method=augmentation \
#     data.augmentation_scheme=none \
#     data.n_folds=1 \
#     callbacks=search \
#     logger=none \
#     trainer=gpu \
#     trainer.devices=[0] \
#     trainer.max_epochs=50

# ------------------------------------------------------------------------
# jetformer  nconst=16  aug_so2  (method=augmentation, group=so2)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     experiment=hls4ml/jetformer_16 \
#     hparams_search=hls4ml/jetformer_optuna \
#     data.features='[pt,eta,phi]' \
#     hydra.sweeper.storage="'sqlite:///logs/optuna/hls4ml/pt_eta_phi/jetformer.db?timeout=600'" \
#     hydra.sweeper.n_trials=200 \
#     data.method=augmentation \
#     data.augmentation_scheme=so2 \
#     data.n_folds=1 \
#     callbacks=search \
#     logger=none \
#     trainer=gpu \
#     trainer.devices=[0] \
#     trainer.max_epochs=50

# ------------------------------------------------------------------------
# jetformer  nconst=16  aug_so3  (method=augmentation, group=so3)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     experiment=hls4ml/jetformer_16 \
#     hparams_search=hls4ml/jetformer_optuna \
#     data.features='[pt,eta,phi]' \
#     hydra.sweeper.storage="'sqlite:///logs/optuna/hls4ml/pt_eta_phi/jetformer.db?timeout=600'" \
#     hydra.sweeper.n_trials=200 \
#     data.method=augmentation \
#     data.augmentation_scheme=so3 \
#     data.n_folds=1 \
#     callbacks=search \
#     logger=none \
#     trainer=gpu \
#     trainer.devices=[0] \
#     trainer.max_epochs=50

# ------------------------------------------------------------------------
# jetformer  nconst=16  aug_lorentz  (method=augmentation, group=lorentz)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     experiment=hls4ml/jetformer_16 \
#     hparams_search=hls4ml/jetformer_optuna \
#     data.features='[pt,eta,phi]' \
#     hydra.sweeper.storage="'sqlite:///logs/optuna/hls4ml/pt_eta_phi/jetformer.db?timeout=600'" \
#     hydra.sweeper.n_trials=200 \
#     data.method=augmentation \
#     data.augmentation_scheme=lorentz \
#     data.n_folds=1 \
#     callbacks=search \
#     logger=none \
#     trainer=gpu \
#     trainer.devices=[0] \
#     trainer.max_epochs=50

# ------------------------------------------------------------------------
# jetformer  nconst=16  symm_so2  (method=symm_loss, group=so2)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     experiment=hls4ml/jetformer_16 \
#     hparams_search=hls4ml/jetformer_scl_optuna \
#     data.features='[pt,eta,phi]' \
#     hydra.sweeper.storage="'sqlite:///logs/optuna/hls4ml/pt_eta_phi/jetformer.db?timeout=600'" \
#     hydra.sweeper.n_trials=200 \
#     data.method=symm_loss \
#     data.augmentation_scheme=so2 \
#     data.n_folds=1 \
#     callbacks=search \
#     logger=none \
#     trainer=gpu \
#     trainer.devices=[0] \
#     trainer.max_epochs=50

# ------------------------------------------------------------------------
# jetformer  nconst=16  symm_so3  (method=symm_loss, group=so3)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     experiment=hls4ml/jetformer_16 \
#     hparams_search=hls4ml/jetformer_scl_optuna \
#     data.features='[pt,eta,phi]' \
#     hydra.sweeper.storage="'sqlite:///logs/optuna/hls4ml/pt_eta_phi/jetformer.db?timeout=600'" \
#     hydra.sweeper.n_trials=200 \
#     data.method=symm_loss \
#     data.augmentation_scheme=so3 \
#     data.n_folds=1 \
#     callbacks=search \
#     logger=none \
#     trainer=gpu \
#     trainer.devices=[0] \
#     trainer.max_epochs=50

# ------------------------------------------------------------------------
# jetformer  nconst=16  symm_lorentz  (method=symm_loss, group=lorentz)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     experiment=hls4ml/jetformer_16 \
#     hparams_search=hls4ml/jetformer_scl_optuna \
#     data.features='[pt,eta,phi]' \
#     hydra.sweeper.storage="'sqlite:///logs/optuna/hls4ml/pt_eta_phi/jetformer.db?timeout=600'" \
#     hydra.sweeper.n_trials=200 \
#     data.method=symm_loss \
#     data.augmentation_scheme=lorentz \
#     data.n_folds=1 \
#     callbacks=search \
#     logger=none \
#     trainer=gpu \
#     trainer.devices=[0] \
#     trainer.max_epochs=50

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
#     data.features='[pt,eta,phi]' \
#     hydra.sweeper.storage="'sqlite:///logs/optuna/hls4ml/pt_eta_phi/jetformer.db?timeout=600'" \
#     hydra.sweeper.n_trials=200 \
#     data.method=augmentation \
#     data.augmentation_scheme=none \
#     data.n_folds=1 \
#     callbacks=search \
#     logger=none \
#     trainer=gpu \
#     trainer.devices=[0] \
#     trainer.max_epochs=50

# ------------------------------------------------------------------------
# jetformer  nconst=50  aug_so2  (method=augmentation, group=so2)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     experiment=hls4ml/jetformer_50 \
#     hparams_search=hls4ml/jetformer_optuna \
#     data.features='[pt,eta,phi]' \
#     hydra.sweeper.storage="'sqlite:///logs/optuna/hls4ml/pt_eta_phi/jetformer.db?timeout=600'" \
#     hydra.sweeper.n_trials=200 \
#     data.method=augmentation \
#     data.augmentation_scheme=so2 \
#     data.n_folds=1 \
#     callbacks=search \
#     logger=none \
#     trainer=gpu \
#     trainer.devices=[0] \
#     trainer.max_epochs=50

# ------------------------------------------------------------------------
# jetformer  nconst=50  aug_so3  (method=augmentation, group=so3)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     experiment=hls4ml/jetformer_50 \
#     hparams_search=hls4ml/jetformer_optuna \
#     data.features='[pt,eta,phi]' \
#     hydra.sweeper.storage="'sqlite:///logs/optuna/hls4ml/pt_eta_phi/jetformer.db?timeout=600'" \
#     hydra.sweeper.n_trials=200 \
#     data.method=augmentation \
#     data.augmentation_scheme=so3 \
#     data.n_folds=1 \
#     callbacks=search \
#     logger=none \
#     trainer=gpu \
#     trainer.devices=[0] \
#     trainer.max_epochs=50

# ------------------------------------------------------------------------
# jetformer  nconst=50  aug_lorentz  (method=augmentation, group=lorentz)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     experiment=hls4ml/jetformer_50 \
#     hparams_search=hls4ml/jetformer_optuna \
#     data.features='[pt,eta,phi]' \
#     hydra.sweeper.storage="'sqlite:///logs/optuna/hls4ml/pt_eta_phi/jetformer.db?timeout=600'" \
#     hydra.sweeper.n_trials=200 \
#     data.method=augmentation \
#     data.augmentation_scheme=lorentz \
#     data.n_folds=1 \
#     callbacks=search \
#     logger=none \
#     trainer=gpu \
#     trainer.devices=[0] \
#     trainer.max_epochs=50

# ------------------------------------------------------------------------
# jetformer  nconst=50  symm_so2  (method=symm_loss, group=so2)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     experiment=hls4ml/jetformer_50 \
#     hparams_search=hls4ml/jetformer_scl_optuna \
#     data.features='[pt,eta,phi]' \
#     hydra.sweeper.storage="'sqlite:///logs/optuna/hls4ml/pt_eta_phi/jetformer.db?timeout=600'" \
#     hydra.sweeper.n_trials=200 \
#     data.method=symm_loss \
#     data.augmentation_scheme=so2 \
#     data.n_folds=1 \
#     callbacks=search \
#     logger=none \
#     trainer=gpu \
#     trainer.devices=[0] \
#     trainer.max_epochs=50

# ------------------------------------------------------------------------
# jetformer  nconst=50  symm_so3  (method=symm_loss, group=so3)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     experiment=hls4ml/jetformer_50 \
#     hparams_search=hls4ml/jetformer_scl_optuna \
#     data.features='[pt,eta,phi]' \
#     hydra.sweeper.storage="'sqlite:///logs/optuna/hls4ml/pt_eta_phi/jetformer.db?timeout=600'" \
#     hydra.sweeper.n_trials=200 \
#     data.method=symm_loss \
#     data.augmentation_scheme=so3 \
#     data.n_folds=1 \
#     callbacks=search \
#     logger=none \
#     trainer=gpu \
#     trainer.devices=[0] \
#     trainer.max_epochs=50

# ------------------------------------------------------------------------
# jetformer  nconst=50  symm_lorentz  (method=symm_loss, group=lorentz)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     experiment=hls4ml/jetformer_50 \
#     hparams_search=hls4ml/jetformer_scl_optuna \
#     data.features='[pt,eta,phi]' \
#     hydra.sweeper.storage="'sqlite:///logs/optuna/hls4ml/pt_eta_phi/jetformer.db?timeout=600'" \
#     hydra.sweeper.n_trials=200 \
#     data.method=symm_loss \
#     data.augmentation_scheme=lorentz \
#     data.n_folds=1 \
#     callbacks=search \
#     logger=none \
#     trainer=gpu \
#     trainer.devices=[0] \
#     trainer.max_epochs=50

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
#     data.features='[pt,eta,phi]' \
#     hydra.sweeper.storage="'sqlite:///logs/optuna/hls4ml/pt_eta_phi/jetformer.db?timeout=600'" \
#     hydra.sweeper.n_trials=200 \
#     data.method=augmentation \
#     data.augmentation_scheme=none \
#     data.n_folds=1 \
#     callbacks=search \
#     logger=none \
#     trainer=gpu \
#     trainer.devices=[0] \
#     trainer.max_epochs=50

# ------------------------------------------------------------------------
# jetformer  nconst=150  aug_so2  (method=augmentation, group=so2)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     experiment=hls4ml/jetformer_150 \
#     hparams_search=hls4ml/jetformer_optuna \
#     data.features='[pt,eta,phi]' \
#     hydra.sweeper.storage="'sqlite:///logs/optuna/hls4ml/pt_eta_phi/jetformer.db?timeout=600'" \
#     hydra.sweeper.n_trials=200 \
#     data.method=augmentation \
#     data.augmentation_scheme=so2 \
#     data.n_folds=1 \
#     callbacks=search \
#     logger=none \
#     trainer=gpu \
#     trainer.devices=[0] \
#     trainer.max_epochs=50

# ------------------------------------------------------------------------
# jetformer  nconst=150  aug_so3  (method=augmentation, group=so3)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     experiment=hls4ml/jetformer_150 \
#     hparams_search=hls4ml/jetformer_optuna \
#     data.features='[pt,eta,phi]' \
#     hydra.sweeper.storage="'sqlite:///logs/optuna/hls4ml/pt_eta_phi/jetformer.db?timeout=600'" \
#     hydra.sweeper.n_trials=200 \
#     data.method=augmentation \
#     data.augmentation_scheme=so3 \
#     data.n_folds=1 \
#     callbacks=search \
#     logger=none \
#     trainer=gpu \
#     trainer.devices=[0] \
#     trainer.max_epochs=50

# ------------------------------------------------------------------------
# jetformer  nconst=150  aug_lorentz  (method=augmentation, group=lorentz)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     experiment=hls4ml/jetformer_150 \
#     hparams_search=hls4ml/jetformer_optuna \
#     data.features='[pt,eta,phi]' \
#     hydra.sweeper.storage="'sqlite:///logs/optuna/hls4ml/pt_eta_phi/jetformer.db?timeout=600'" \
#     hydra.sweeper.n_trials=200 \
#     data.method=augmentation \
#     data.augmentation_scheme=lorentz \
#     data.n_folds=1 \
#     callbacks=search \
#     logger=none \
#     trainer=gpu \
#     trainer.devices=[0] \
#     trainer.max_epochs=50

# ------------------------------------------------------------------------
# jetformer  nconst=150  symm_so2  (method=symm_loss, group=so2)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     experiment=hls4ml/jetformer_150 \
#     hparams_search=hls4ml/jetformer_scl_optuna \
#     data.features='[pt,eta,phi]' \
#     hydra.sweeper.storage="'sqlite:///logs/optuna/hls4ml/pt_eta_phi/jetformer.db?timeout=600'" \
#     hydra.sweeper.n_trials=200 \
#     data.method=symm_loss \
#     data.augmentation_scheme=so2 \
#     data.n_folds=1 \
#     callbacks=search \
#     logger=none \
#     trainer=gpu \
#     trainer.devices=[0] \
#     trainer.max_epochs=50

# ------------------------------------------------------------------------
# jetformer  nconst=150  symm_so3  (method=symm_loss, group=so3)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     experiment=hls4ml/jetformer_150 \
#     hparams_search=hls4ml/jetformer_scl_optuna \
#     data.features='[pt,eta,phi]' \
#     hydra.sweeper.storage="'sqlite:///logs/optuna/hls4ml/pt_eta_phi/jetformer.db?timeout=600'" \
#     hydra.sweeper.n_trials=200 \
#     data.method=symm_loss \
#     data.augmentation_scheme=so3 \
#     data.n_folds=1 \
#     callbacks=search \
#     logger=none \
#     trainer=gpu \
#     trainer.devices=[0] \
#     trainer.max_epochs=50

# ------------------------------------------------------------------------
# jetformer  nconst=150  symm_lorentz  (method=symm_loss, group=lorentz)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     experiment=hls4ml/jetformer_150 \
#     hparams_search=hls4ml/jetformer_scl_optuna \
#     data.features='[pt,eta,phi]' \
#     hydra.sweeper.storage="'sqlite:///logs/optuna/hls4ml/pt_eta_phi/jetformer.db?timeout=600'" \
#     hydra.sweeper.n_trials=200 \
#     data.method=symm_loss \
#     data.augmentation_scheme=lorentz \
#     data.n_folds=1 \
#     callbacks=search \
#     logger=none \
#     trainer=gpu \
#     trainer.devices=[0] \
#     trainer.max_epochs=50
