#!/usr/bin/env bash
# ========================================================================
# LORENTZNET TRAINING COMMANDS
# ========================================================================
# One block per (constituent count, regime): 3 x 1 = 3 train runs. Generated
# by scripts/optuna/make_train_catalogues.py; regenerate rather than edit by hand.
# Baseline only: this architecture embeds the symmetry rather than acquiring it,
# so the augmentation and SCL regimes are not run on it.
# Each block carries the best trial of its own study under logs/optuna/hls4ml/;
# a block whose study is missing carries none and says so (transferred ablation).
#
# Every command is commented out: copy one to run it by hand, or feed the whole
# file to scripts/cluster/submit_train.sh, which queues them on a GPU or slurm.

# ========================================================================
# lorentznet  nconst=16
# ========================================================================

# ------------------------------------------------------------------------
# lorentznet  nconst=16  baseline  (method=augmentation, group=none)
# best trial 236 of lorentznet_nconst16_augmentation_none, val/acc_best=0.681064
# ------------------------------------------------------------------------
# python3 src/train.py \
#     experiment=hls4ml/lorentznet_16 \
#     data.method=augmentation \
#     data.augmentation_scheme=none \
#     data.n_folds=5 \
#     algorithm.c_weight=0.00925637363766269 \
#     algorithm.dropout=0.2 \
#     algorithm.n_hidden=48 \
#     algorithm.n_layers=6 \
#     algorithm.optimizer.lr=0.004229335353885937 \
#     algorithm.optimizer.weight_decay=0.0 \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/train/hls4ml/lorentznet/nconst16/baseline

# ========================================================================
# lorentznet  nconst=50
# ========================================================================

# ------------------------------------------------------------------------
# lorentznet  nconst=50  baseline  (method=augmentation, group=none)
# best trial 219 of lorentznet_nconst50_augmentation_none, val/acc_best=0.806258
# ------------------------------------------------------------------------
# python3 src/train.py \
#     experiment=hls4ml/lorentznet_50 \
#     data.method=augmentation \
#     data.augmentation_scheme=none \
#     data.n_folds=5 \
#     algorithm.c_weight=0.08470704719307252 \
#     algorithm.dropout=0.2 \
#     algorithm.n_hidden=32 \
#     algorithm.n_layers=6 \
#     algorithm.optimizer.lr=0.0016318812488392122 \
#     algorithm.optimizer.weight_decay=0.0 \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/train/hls4ml/lorentznet/nconst50/baseline

# ========================================================================
# lorentznet  nconst=150
# ========================================================================

# ------------------------------------------------------------------------
# lorentznet  nconst=150  baseline  (method=augmentation, group=none)
# best trial 208 of lorentznet_nconst150_augmentation_none, val/acc_best=0.816503
# ------------------------------------------------------------------------
# python3 src/train.py \
#     experiment=hls4ml/lorentznet_150 \
#     data.method=augmentation \
#     data.augmentation_scheme=none \
#     data.n_folds=5 \
#     algorithm.c_weight=0.06664154445895842 \
#     algorithm.dropout=0.2 \
#     algorithm.n_hidden=32 \
#     algorithm.n_layers=6 \
#     algorithm.optimizer.lr=0.0019270787396216814 \
#     algorithm.optimizer.weight_decay=0.0 \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/train/hls4ml/lorentznet/nconst150/baseline
