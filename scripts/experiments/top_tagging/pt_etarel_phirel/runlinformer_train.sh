#!/usr/bin/env bash
# ========================================================================
# LINFORMER TRAINING COMMANDS
# ========================================================================
# One block per (constituent count, regime): 1 x 7 = 7 train runs. Generated
# by scripts/experiments/make_train_catalogues.py; regenerate rather than edit by hand.
# Each block carries the best of the first 100 completed trials, by trial
# number, of its own study under logs/optuna/top_tagging/;
# a block whose study is missing carries none and says so (transferred ablation).
#
# Every command is commented out: copy one to run it by hand, or feed the whole
# file to scripts/experiments/submit_train.sh, which runs them under nohup.

# ========================================================================
# linformer  nconst=100
# ========================================================================

# ------------------------------------------------------------------------
# linformer  nconst=100  baseline  (method=augmentation, group=none)
# best trial 93 of top_tagging_linformer_nconst100_augmentation_none, val/acc_best=0.930448
# ------------------------------------------------------------------------
# python3 src/train.py \
#     experiment=top_tagging/linformer_100 \
#     data.method=augmentation \
#     data.augmentation_scheme=none \
#     data.n_folds=5 \
#     algorithm.aggregation=max \
#     algorithm.batchnorm=True \
#     algorithm.embed_dim=64 \
#     algorithm.ff_dim=256 \
#     algorithm.head_layers='[16]' \
#     algorithm.kv_proj_dim=2 \
#     algorithm.optimizer.lr=0.0008246964844513512 \
#     algorithm.optimizer.weight_decay=0.0 \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/train/top_tagging/linformer/nconst100/baseline

# ------------------------------------------------------------------------
# linformer  nconst=100  aug_so2  (method=augmentation, group=so2)
# best trial 86 of top_tagging_linformer_nconst100_augmentation_so2, val/acc_best=0.930257
# ------------------------------------------------------------------------
# python3 src/train.py \
#     experiment=top_tagging/linformer_100 \
#     data.method=augmentation \
#     data.augmentation_scheme=so2 \
#     data.n_folds=5 \
#     algorithm.aggregation=max \
#     algorithm.batchnorm=True \
#     algorithm.embed_dim=96 \
#     algorithm.ff_dim=256 \
#     algorithm.head_layers='[32]' \
#     algorithm.kv_proj_dim=2 \
#     algorithm.optimizer.lr=0.001839882372304528 \
#     algorithm.optimizer.weight_decay=0.0 \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/train/top_tagging/linformer/nconst100/aug_so2

# ------------------------------------------------------------------------
# linformer  nconst=100  aug_so3  (method=augmentation, group=so3)
# best trial 65 of top_tagging_linformer_nconst100_augmentation_so3, val/acc_best=0.929597
# ------------------------------------------------------------------------
# python3 src/train.py \
#     experiment=top_tagging/linformer_100 \
#     data.method=augmentation \
#     data.augmentation_scheme=so3 \
#     data.n_folds=5 \
#     algorithm.aggregation=max \
#     algorithm.batchnorm=True \
#     algorithm.embed_dim=48 \
#     algorithm.ff_dim=256 \
#     algorithm.head_layers='[64,64,32]' \
#     algorithm.kv_proj_dim=16 \
#     algorithm.optimizer.lr=0.0008330019032697148 \
#     algorithm.optimizer.weight_decay=1e-05 \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/train/top_tagging/linformer/nconst100/aug_so3

# ------------------------------------------------------------------------
# linformer  nconst=100  aug_lorentz  (method=augmentation, group=lorentz)
# best trial 96 of top_tagging_linformer_nconst100_augmentation_lorentz, val/acc_best=0.930327
# ------------------------------------------------------------------------
# python3 src/train.py \
#     experiment=top_tagging/linformer_100 \
#     data.method=augmentation \
#     data.augmentation_scheme=lorentz \
#     data.n_folds=5 \
#     algorithm.aggregation=max \
#     algorithm.batchnorm=True \
#     algorithm.embed_dim=96 \
#     algorithm.ff_dim=256 \
#     algorithm.head_layers='[16]' \
#     algorithm.kv_proj_dim=8 \
#     algorithm.optimizer.lr=0.000790984272297916 \
#     algorithm.optimizer.weight_decay=1e-05 \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/train/top_tagging/linformer/nconst100/aug_lorentz

# ------------------------------------------------------------------------
# linformer  nconst=100  symm_so2  (method=symm_loss, group=so2)
# best trial 85 of top_tagging_linformer_nconst100_symm_loss_so2, val/acc_best=0.930478
# ------------------------------------------------------------------------
# python3 src/train.py \
#     experiment=top_tagging/linformer_100 \
#     data.method=symm_loss \
#     data.augmentation_scheme=so2 \
#     data.n_folds=5 \
#     algorithm.aggregation=max \
#     algorithm.batchnorm=True \
#     algorithm.embed_dim=48 \
#     algorithm.ff_dim=256 \
#     algorithm.head_layers='[32,16]' \
#     algorithm.kv_proj_dim=4 \
#     algorithm.optimizer.lr=0.0016860090721078696 \
#     algorithm.optimizer.weight_decay=1e-06 \
#     algorithm.symm_loss_weight=0.008359484642248071 \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/train/top_tagging/linformer/nconst100/symm_so2

# ------------------------------------------------------------------------
# linformer  nconst=100  symm_so3  (method=symm_loss, group=so3)
# best trial 98 of top_tagging_linformer_nconst100_symm_loss_so3, val/acc_best=0.930522
# ------------------------------------------------------------------------
# python3 src/train.py \
#     experiment=top_tagging/linformer_100 \
#     data.method=symm_loss \
#     data.augmentation_scheme=so3 \
#     data.n_folds=5 \
#     algorithm.aggregation=max \
#     algorithm.batchnorm=False \
#     algorithm.embed_dim=64 \
#     algorithm.ff_dim=128 \
#     algorithm.head_layers='[64,32]' \
#     algorithm.kv_proj_dim=2 \
#     algorithm.optimizer.lr=0.001194951006990365 \
#     algorithm.optimizer.weight_decay=1e-07 \
#     algorithm.symm_loss_weight=0.04903811514358603 \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/train/top_tagging/linformer/nconst100/symm_so3

# ------------------------------------------------------------------------
# linformer  nconst=100  symm_lorentz  (method=symm_loss, group=lorentz)
# best trial 33 of top_tagging_linformer_nconst100_symm_loss_lorentz, val/acc_best=0.930706
# ------------------------------------------------------------------------
# python3 src/train.py \
#     experiment=top_tagging/linformer_100 \
#     data.method=symm_loss \
#     data.augmentation_scheme=lorentz \
#     data.n_folds=5 \
#     algorithm.aggregation=max \
#     algorithm.batchnorm=True \
#     algorithm.embed_dim=96 \
#     algorithm.ff_dim=128 \
#     algorithm.head_layers='[32,32,32]' \
#     algorithm.kv_proj_dim=8 \
#     algorithm.optimizer.lr=0.0008850890838473411 \
#     algorithm.optimizer.weight_decay=0.0 \
#     algorithm.symm_loss_weight=0.016336671919723187 \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/train/top_tagging/linformer/nconst100/symm_lorentz
