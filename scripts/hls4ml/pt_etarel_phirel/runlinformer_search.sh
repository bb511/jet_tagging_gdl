#!/usr/bin/env bash
# ========================================================================
# LINFORMER HYPERPARAMETER SEARCH COMMANDS, FEATURE SET [pt, etarel, phirel]
# ========================================================================
# The 21 blocks of ../runlinformer_search.sh with two overrides added to each:
# data.features, which the model reads in place of the four-momentum, and
# hydra.sweeper.storage, a study database of this feature set's own under
# logs/optuna/hls4ml/pt_etarel_phirel/. The study name derives from the regime alone
# (<model>_nconst<N>_<method>_<scheme>), so without the second override the
# trials would join the px_py_pz_e study of the same name; submit_search.sh
# refuses a block that carries the first without the second. The evaluation of
# the trained models lands in results/pt_etarel_phirel/ by the same feature list.
#
# phirel is a difference of azimuths and etarel of pseudorapidities, so the
# SO(2) regimes leave every input unchanged: the augmentation is a no-op and
# the SCL term is zero there, by construction of the feature set. The beam
# boost moves etarel only through the masses involved: a massless
# constituent's pseudorapidity shifts rigidly, the jet's does not. Both are
# what the ablation measures, not a fault.
#
# Trials run at data.n_folds=1 with callbacks=search, so they draw no figure and
# never write to results/. The SCL regimes select _scl_optuna instead, which
# adds algorithm.symm_loss_weight. lorentz_eta_max defines the probe and is
# never tuned.
#
# Hand-maintained, unlike the train and eval catalogues. Every command is
# commented out; feed the whole file to scripts/cluster/submit_search.sh.

# ========================================================================
# linformer  nconst=16
# ========================================================================

# ------------------------------------------------------------------------
# linformer  nconst=16  baseline  (method=augmentation, group=none)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     experiment=hls4ml/linformer_16 \
#     hparams_search=hls4ml/linformer_optuna \
#     data.features='[pt,etarel,phirel]' \
#     hydra.sweeper.storage="'sqlite:///logs/optuna/hls4ml/pt_etarel_phirel/linformer.db?timeout=600'" \
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
# linformer  nconst=16  aug_so2  (method=augmentation, group=so2)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     experiment=hls4ml/linformer_16 \
#     hparams_search=hls4ml/linformer_optuna \
#     data.features='[pt,etarel,phirel]' \
#     hydra.sweeper.storage="'sqlite:///logs/optuna/hls4ml/pt_etarel_phirel/linformer.db?timeout=600'" \
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
# linformer  nconst=16  aug_so3  (method=augmentation, group=so3)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     experiment=hls4ml/linformer_16 \
#     hparams_search=hls4ml/linformer_optuna \
#     data.features='[pt,etarel,phirel]' \
#     hydra.sweeper.storage="'sqlite:///logs/optuna/hls4ml/pt_etarel_phirel/linformer.db?timeout=600'" \
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
# linformer  nconst=16  aug_lorentz  (method=augmentation, group=lorentz)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     experiment=hls4ml/linformer_16 \
#     hparams_search=hls4ml/linformer_optuna \
#     data.features='[pt,etarel,phirel]' \
#     hydra.sweeper.storage="'sqlite:///logs/optuna/hls4ml/pt_etarel_phirel/linformer.db?timeout=600'" \
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
# linformer  nconst=16  symm_so2  (method=symm_loss, group=so2)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     experiment=hls4ml/linformer_16 \
#     hparams_search=hls4ml/linformer_scl_optuna \
#     data.features='[pt,etarel,phirel]' \
#     hydra.sweeper.storage="'sqlite:///logs/optuna/hls4ml/pt_etarel_phirel/linformer.db?timeout=600'" \
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
# linformer  nconst=16  symm_so3  (method=symm_loss, group=so3)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     experiment=hls4ml/linformer_16 \
#     hparams_search=hls4ml/linformer_scl_optuna \
#     data.features='[pt,etarel,phirel]' \
#     hydra.sweeper.storage="'sqlite:///logs/optuna/hls4ml/pt_etarel_phirel/linformer.db?timeout=600'" \
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
# linformer  nconst=16  symm_lorentz  (method=symm_loss, group=lorentz)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     experiment=hls4ml/linformer_16 \
#     hparams_search=hls4ml/linformer_scl_optuna \
#     data.features='[pt,etarel,phirel]' \
#     hydra.sweeper.storage="'sqlite:///logs/optuna/hls4ml/pt_etarel_phirel/linformer.db?timeout=600'" \
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
# linformer  nconst=50
# ========================================================================

# ------------------------------------------------------------------------
# linformer  nconst=50  baseline  (method=augmentation, group=none)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     experiment=hls4ml/linformer_50 \
#     hparams_search=hls4ml/linformer_optuna \
#     data.features='[pt,etarel,phirel]' \
#     hydra.sweeper.storage="'sqlite:///logs/optuna/hls4ml/pt_etarel_phirel/linformer.db?timeout=600'" \
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
# linformer  nconst=50  aug_so2  (method=augmentation, group=so2)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     experiment=hls4ml/linformer_50 \
#     hparams_search=hls4ml/linformer_optuna \
#     data.features='[pt,etarel,phirel]' \
#     hydra.sweeper.storage="'sqlite:///logs/optuna/hls4ml/pt_etarel_phirel/linformer.db?timeout=600'" \
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
# linformer  nconst=50  aug_so3  (method=augmentation, group=so3)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     experiment=hls4ml/linformer_50 \
#     hparams_search=hls4ml/linformer_optuna \
#     data.features='[pt,etarel,phirel]' \
#     hydra.sweeper.storage="'sqlite:///logs/optuna/hls4ml/pt_etarel_phirel/linformer.db?timeout=600'" \
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
# linformer  nconst=50  aug_lorentz  (method=augmentation, group=lorentz)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     experiment=hls4ml/linformer_50 \
#     hparams_search=hls4ml/linformer_optuna \
#     data.features='[pt,etarel,phirel]' \
#     hydra.sweeper.storage="'sqlite:///logs/optuna/hls4ml/pt_etarel_phirel/linformer.db?timeout=600'" \
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
# linformer  nconst=50  symm_so2  (method=symm_loss, group=so2)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     experiment=hls4ml/linformer_50 \
#     hparams_search=hls4ml/linformer_scl_optuna \
#     data.features='[pt,etarel,phirel]' \
#     hydra.sweeper.storage="'sqlite:///logs/optuna/hls4ml/pt_etarel_phirel/linformer.db?timeout=600'" \
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
# linformer  nconst=50  symm_so3  (method=symm_loss, group=so3)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     experiment=hls4ml/linformer_50 \
#     hparams_search=hls4ml/linformer_scl_optuna \
#     data.features='[pt,etarel,phirel]' \
#     hydra.sweeper.storage="'sqlite:///logs/optuna/hls4ml/pt_etarel_phirel/linformer.db?timeout=600'" \
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
# linformer  nconst=50  symm_lorentz  (method=symm_loss, group=lorentz)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     experiment=hls4ml/linformer_50 \
#     hparams_search=hls4ml/linformer_scl_optuna \
#     data.features='[pt,etarel,phirel]' \
#     hydra.sweeper.storage="'sqlite:///logs/optuna/hls4ml/pt_etarel_phirel/linformer.db?timeout=600'" \
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
# linformer  nconst=150
# ========================================================================

# ------------------------------------------------------------------------
# linformer  nconst=150  baseline  (method=augmentation, group=none)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     experiment=hls4ml/linformer_150 \
#     hparams_search=hls4ml/linformer_optuna \
#     data.features='[pt,etarel,phirel]' \
#     hydra.sweeper.storage="'sqlite:///logs/optuna/hls4ml/pt_etarel_phirel/linformer.db?timeout=600'" \
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
# linformer  nconst=150  aug_so2  (method=augmentation, group=so2)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     experiment=hls4ml/linformer_150 \
#     hparams_search=hls4ml/linformer_optuna \
#     data.features='[pt,etarel,phirel]' \
#     hydra.sweeper.storage="'sqlite:///logs/optuna/hls4ml/pt_etarel_phirel/linformer.db?timeout=600'" \
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
# linformer  nconst=150  aug_so3  (method=augmentation, group=so3)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     experiment=hls4ml/linformer_150 \
#     hparams_search=hls4ml/linformer_optuna \
#     data.features='[pt,etarel,phirel]' \
#     hydra.sweeper.storage="'sqlite:///logs/optuna/hls4ml/pt_etarel_phirel/linformer.db?timeout=600'" \
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
# linformer  nconst=150  aug_lorentz  (method=augmentation, group=lorentz)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     experiment=hls4ml/linformer_150 \
#     hparams_search=hls4ml/linformer_optuna \
#     data.features='[pt,etarel,phirel]' \
#     hydra.sweeper.storage="'sqlite:///logs/optuna/hls4ml/pt_etarel_phirel/linformer.db?timeout=600'" \
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
# linformer  nconst=150  symm_so2  (method=symm_loss, group=so2)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     experiment=hls4ml/linformer_150 \
#     hparams_search=hls4ml/linformer_scl_optuna \
#     data.features='[pt,etarel,phirel]' \
#     hydra.sweeper.storage="'sqlite:///logs/optuna/hls4ml/pt_etarel_phirel/linformer.db?timeout=600'" \
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
# linformer  nconst=150  symm_so3  (method=symm_loss, group=so3)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     experiment=hls4ml/linformer_150 \
#     hparams_search=hls4ml/linformer_scl_optuna \
#     data.features='[pt,etarel,phirel]' \
#     hydra.sweeper.storage="'sqlite:///logs/optuna/hls4ml/pt_etarel_phirel/linformer.db?timeout=600'" \
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
# linformer  nconst=150  symm_lorentz  (method=symm_loss, group=lorentz)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     experiment=hls4ml/linformer_150 \
#     hparams_search=hls4ml/linformer_scl_optuna \
#     data.features='[pt,etarel,phirel]' \
#     hydra.sweeper.storage="'sqlite:///logs/optuna/hls4ml/pt_etarel_phirel/linformer.db?timeout=600'" \
#     hydra.sweeper.n_trials=200 \
#     data.method=symm_loss \
#     data.augmentation_scheme=lorentz \
#     data.n_folds=1 \
#     callbacks=search \
#     logger=none \
#     trainer=gpu \
#     trainer.devices=[0] \
#     trainer.max_epochs=50
