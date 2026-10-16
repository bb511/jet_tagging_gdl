#!/usr/bin/env bash
# ========================================================================
# LINFORMER TRAINING COMMANDS
# ========================================================================
# One block per (constituent count, regime): 1 x 7 = 7 train runs. Generated
# by scripts/experiments/make_train_catalogues.py; regenerate rather than edit by hand.
# Feature set [px, py, pz, e]: studies from logs/optuna/top_tagging/px_py_pz_e/, runs under logs/{train,eval}/top_tagging/px_py_pz_e/.
# Each block carries the best of the first 100 completed trials, by trial
# number, of its own study under logs/optuna/top_tagging/px_py_pz_e/;
# a block whose study is missing carries none and says so (transferred ablation).
#
# Every command is commented out: copy one to run it by hand, or feed the whole
# file to scripts/experiments/submit_train.sh, which runs them under nohup.

# ========================================================================
# linformer  nconst=100
# ========================================================================

# ------------------------------------------------------------------------
# linformer  nconst=100  baseline  (method=augmentation, group=none)
# best trial 83 of top_tagging_linformer_nconst100_augmentation_none, val/acc_best=0.915862
# ------------------------------------------------------------------------
# python3 src/train.py \
#     experiment=top_tagging/linformer_100 \
#     data.method=augmentation \
#     data.augmentation_scheme=none \
#     data.n_folds=5 \
#     data.features='[px,py,pz,e]' \
#     algorithm.aggregation=sum \
#     algorithm.batchnorm=True \
#     algorithm.embed_dim=96 \
#     algorithm.ff_dim=32 \
#     algorithm.head_layers='[32,32,16]' \
#     algorithm.kv_proj_dim=8 \
#     algorithm.optimizer.lr=0.00033655156509915365 \
#     algorithm.optimizer.weight_decay=1e-07 \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/train/top_tagging/px_py_pz_e/linformer/nconst100/baseline

# ------------------------------------------------------------------------
# linformer  nconst=100  aug_so2  (method=augmentation, group=so2)
# best trial 87 of top_tagging_linformer_nconst100_augmentation_so2, val/acc_best=0.916414
# ------------------------------------------------------------------------
# python3 src/train.py \
#     experiment=top_tagging/linformer_100 \
#     data.method=augmentation \
#     data.augmentation_scheme=so2 \
#     data.n_folds=5 \
#     data.features='[px,py,pz,e]' \
#     algorithm.aggregation=max \
#     algorithm.batchnorm=False \
#     algorithm.embed_dim=64 \
#     algorithm.ff_dim=256 \
#     algorithm.head_layers='[64,64,32]' \
#     algorithm.kv_proj_dim=8 \
#     algorithm.optimizer.lr=0.003778225977867745 \
#     algorithm.optimizer.weight_decay=1e-06 \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/train/top_tagging/px_py_pz_e/linformer/nconst100/aug_so2

# ------------------------------------------------------------------------
# linformer  nconst=100  aug_so3  (method=augmentation, group=so3)
# best trial 84 of top_tagging_linformer_nconst100_augmentation_so3, val/acc_best=0.915201
# ------------------------------------------------------------------------
# python3 src/train.py \
#     experiment=top_tagging/linformer_100 \
#     data.method=augmentation \
#     data.augmentation_scheme=so3 \
#     data.n_folds=5 \
#     data.features='[px,py,pz,e]' \
#     algorithm.aggregation=max \
#     algorithm.batchnorm=True \
#     algorithm.embed_dim=96 \
#     algorithm.ff_dim=256 \
#     algorithm.head_layers='[32,32,32]' \
#     algorithm.kv_proj_dim=2 \
#     algorithm.optimizer.lr=0.0005121607900163559 \
#     algorithm.optimizer.weight_decay=1e-05 \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/train/top_tagging/px_py_pz_e/linformer/nconst100/aug_so3

# ------------------------------------------------------------------------
# linformer  nconst=100  aug_lorentz  (method=augmentation, group=lorentz)
# best trial 87 of top_tagging_linformer_nconst100_augmentation_lorentz, val/acc_best=0.918611
# ------------------------------------------------------------------------
# python3 src/train.py \
#     experiment=top_tagging/linformer_100 \
#     data.method=augmentation \
#     data.augmentation_scheme=lorentz \
#     data.n_folds=5 \
#     data.features='[px,py,pz,e]' \
#     algorithm.aggregation=max \
#     algorithm.batchnorm=True \
#     algorithm.embed_dim=96 \
#     algorithm.ff_dim=256 \
#     algorithm.head_layers='[64,64,32]' \
#     algorithm.kv_proj_dim=8 \
#     algorithm.optimizer.lr=0.00033923418013073357 \
#     algorithm.optimizer.weight_decay=0.0001 \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/train/top_tagging/px_py_pz_e/linformer/nconst100/aug_lorentz

# ------------------------------------------------------------------------
# linformer  nconst=100  symm_so2  (method=symm_loss, group=so2)
# best trial 97 of top_tagging_linformer_nconst100_symm_loss_so2, val/acc_best=0.921089
# ------------------------------------------------------------------------
# python3 src/train.py \
#     experiment=top_tagging/linformer_100 \
#     data.method=symm_loss \
#     data.augmentation_scheme=so2 \
#     data.n_folds=5 \
#     data.features='[px,py,pz,e]' \
#     algorithm.aggregation=sum \
#     algorithm.batchnorm=True \
#     algorithm.embed_dim=64 \
#     algorithm.ff_dim=128 \
#     algorithm.head_layers='[32,32,32]' \
#     algorithm.kv_proj_dim=4 \
#     algorithm.optimizer.lr=0.0005986716654719367 \
#     algorithm.optimizer.weight_decay=1e-06 \
#     algorithm.symm_loss_weight=0.07768297060634832 \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/train/top_tagging/px_py_pz_e/linformer/nconst100/symm_so2

# ------------------------------------------------------------------------
# linformer  nconst=100  symm_so3  (method=symm_loss, group=so3)
# best trial 93 of top_tagging_linformer_nconst100_symm_loss_so3, val/acc_best=0.917256
# ------------------------------------------------------------------------
# python3 src/train.py \
#     experiment=top_tagging/linformer_100 \
#     data.method=symm_loss \
#     data.augmentation_scheme=so3 \
#     data.n_folds=5 \
#     data.features='[px,py,pz,e]' \
#     algorithm.aggregation=max \
#     algorithm.batchnorm=False \
#     algorithm.embed_dim=96 \
#     algorithm.ff_dim=256 \
#     algorithm.head_layers='[64,64,32]' \
#     algorithm.kv_proj_dim=4 \
#     algorithm.optimizer.lr=0.0016311343743321755 \
#     algorithm.optimizer.weight_decay=0.0 \
#     algorithm.symm_loss_weight=0.00733627410943253 \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/train/top_tagging/px_py_pz_e/linformer/nconst100/symm_so3

# ------------------------------------------------------------------------
# linformer  nconst=100  symm_lorentz  (method=symm_loss, group=lorentz)
# best trial 56 of top_tagging_linformer_nconst100_symm_loss_lorentz, val/acc_best=0.922042
# ------------------------------------------------------------------------
# python3 src/train.py \
#     experiment=top_tagging/linformer_100 \
#     data.method=symm_loss \
#     data.augmentation_scheme=lorentz \
#     data.n_folds=5 \
#     data.features='[px,py,pz,e]' \
#     algorithm.aggregation=sum_over_n \
#     algorithm.batchnorm=True \
#     algorithm.embed_dim=64 \
#     algorithm.ff_dim=256 \
#     algorithm.head_layers='[64,32]' \
#     algorithm.kv_proj_dim=4 \
#     algorithm.optimizer.lr=0.0009061768751020968 \
#     algorithm.optimizer.weight_decay=1e-06 \
#     algorithm.symm_loss_weight=0.15388088110237294 \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/train/top_tagging/px_py_pz_e/linformer/nconst100/symm_lorentz
