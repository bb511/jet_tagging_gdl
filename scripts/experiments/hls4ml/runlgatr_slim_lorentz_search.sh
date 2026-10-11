#!/usr/bin/env bash
# ========================================================================
# L-GATR-SLIM WITHOUT SPURIONS: HYPERPARAMETER SEARCH COMMANDS
# ========================================================================
# L-GATr-slim with add_beams=false and add_time=false, on the same search
# space and the same 200-trial budget as runlgatr_slim_search.sh, so the
# two rows differ in the embedded group alone: without the spurion tokens
# the network reads the constituents' four-vectors and no fixed direction,
# and its logits are invariant under the full Lorentz group rather than
# SO(2) about the beam. Embedded, so baseline only, never under
# augmentation or the SCL; every probe group is exact for it, so its flip
# rates measure the fp32 floor of the probe. One Optuna driver per
# constituent count: 3 studies.
#
# Trials run at data.n_folds=1 with callbacks=search, so they draw no figure and
# never write to results/. The study name is not on the command line:
# configs/hparams_search/hls4ml/lgatr_slim_lorentz_optuna.yaml derives it
# from the regime as lgatr_slim_lorentz_nconst<N>_augmentation_none. units
# is the references' momentum scale and lorentz_eta_max defines the probe;
# neither is tuned.
#
# Hand-maintained, unlike the train and eval catalogues. Every command is
# commented out; feed the whole file to scripts/experiments/submit_search.sh.

# ========================================================================
# lgatr_slim_lorentz  nconst=16
# ========================================================================

# ------------------------------------------------------------------------
# lgatr_slim_lorentz  nconst=16  baseline  (method=augmentation, group=none)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     experiment=hls4ml/lgatr_slim_lorentz_16 \
#     hparams_search=hls4ml/lgatr_slim_lorentz_optuna \
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
# lgatr_slim_lorentz  nconst=50
# ========================================================================

# ------------------------------------------------------------------------
# lgatr_slim_lorentz  nconst=50  baseline  (method=augmentation, group=none)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     experiment=hls4ml/lgatr_slim_lorentz_50 \
#     hparams_search=hls4ml/lgatr_slim_lorentz_optuna \
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
# lgatr_slim_lorentz  nconst=150
# ========================================================================

# ------------------------------------------------------------------------
# lgatr_slim_lorentz  nconst=150  baseline  (method=augmentation, group=none)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     -m \
#     experiment=hls4ml/lgatr_slim_lorentz_150 \
#     hparams_search=hls4ml/lgatr_slim_lorentz_optuna \
#     hydra.sweeper.n_trials=200 \
#     data.method=augmentation \
#     data.augmentation_scheme=none \
#     data.n_folds=1 \
#     callbacks=search \
#     logger=none \
#     trainer=gpu \
#     trainer.devices=[0] \
#     trainer.max_epochs=50
