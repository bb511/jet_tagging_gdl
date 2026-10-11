#!/usr/bin/env bash
# ========================================================================
# LGATR_SLIM TRAINING COMMANDS
# ========================================================================
# One block per (constituent count, regime): 3 x 1 = 3 train runs. Generated
# by scripts/experiments/make_train_catalogues.py; regenerate rather than edit by hand.
# Baseline only: this architecture embeds the symmetry rather than acquiring it,
# so the augmentation and SCL regimes are not run on it.
# Each block carries the best of the first 200 completed trials, by trial
# number, of its own study under logs/optuna/hls4ml/;
# a block whose study is missing carries none and says so (transferred ablation).
#
# Every command is commented out: copy one to run it by hand, or feed the whole
# file to scripts/experiments/submit_train.sh, which queues them on a GPU or slurm.

# ========================================================================
# lgatr_slim  nconst=16
# ========================================================================

# ------------------------------------------------------------------------
# lgatr_slim  nconst=16  baseline  (method=augmentation, group=none)
# best trial 196 of lgatr_slim_nconst16_augmentation_none, val/acc_best=0.730916
# ------------------------------------------------------------------------
# python3 src/train.py \
#     experiment=hls4ml/lgatr_slim_16 \
#     data.method=augmentation \
#     data.augmentation_scheme=none \
#     data.n_folds=5 \
#     algorithm.dropout=0.0 \
#     algorithm.hidden_s_channels=64 \
#     algorithm.hidden_v_channels=32 \
#     algorithm.mlp_ratio=2 \
#     algorithm.num_blocks=12 \
#     algorithm.num_heads=8 \
#     algorithm.optimizer.lr=0.0005319961902662282 \
#     algorithm.optimizer.weight_decay=0.01 \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/train/hls4ml/lgatr_slim/nconst16/baseline

# ========================================================================
# lgatr_slim  nconst=50
# ========================================================================

# ------------------------------------------------------------------------
# lgatr_slim  nconst=50  baseline  (method=augmentation, group=none)
# best trial 74 of lgatr_slim_nconst50_augmentation_none, val/acc_best=0.837432
# ------------------------------------------------------------------------
# python3 src/train.py \
#     experiment=hls4ml/lgatr_slim_50 \
#     data.method=augmentation \
#     data.augmentation_scheme=none \
#     data.n_folds=5 \
#     algorithm.dropout=0.1 \
#     algorithm.hidden_s_channels=96 \
#     algorithm.hidden_v_channels=32 \
#     algorithm.mlp_ratio=4 \
#     algorithm.num_blocks=12 \
#     algorithm.num_heads=4 \
#     algorithm.optimizer.lr=0.0013838018595506154 \
#     algorithm.optimizer.weight_decay=0.01 \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/train/hls4ml/lgatr_slim/nconst50/baseline

# ========================================================================
# lgatr_slim  nconst=150
# ========================================================================

# ------------------------------------------------------------------------
# lgatr_slim  nconst=150  baseline  (method=augmentation, group=none)
# best trial 182 of lgatr_slim_nconst150_augmentation_none, val/acc_best=0.843879
# ------------------------------------------------------------------------
# python3 src/train.py \
#     experiment=hls4ml/lgatr_slim_150 \
#     data.method=augmentation \
#     data.augmentation_scheme=none \
#     data.n_folds=5 \
#     algorithm.dropout=0.1 \
#     algorithm.hidden_s_channels=64 \
#     algorithm.hidden_v_channels=16 \
#     algorithm.mlp_ratio=2 \
#     algorithm.num_blocks=12 \
#     algorithm.num_heads=8 \
#     algorithm.optimizer.lr=0.0021133664084203494 \
#     algorithm.optimizer.weight_decay=0.01 \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/train/hls4ml/lgatr_slim/nconst150/baseline
