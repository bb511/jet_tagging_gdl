#!/usr/bin/env bash
# ========================================================================
# JETFORMER TRAINING COMMANDS
# ========================================================================
# One block per (constituent count, regime): 1 x 7 = 7 train runs. Generated
# by scripts/experiments/make_train_catalogues.py; regenerate rather than edit by hand.
# Feature set [pt, eta, phi]: studies from logs/optuna/top_tagging/pt_eta_phi/, runs under logs/{train,eval}/top_tagging/pt_eta_phi/.
# Each block carries the best of the first 100 completed trials, by trial
# number, of its own study under logs/optuna/top_tagging/pt_eta_phi/;
# a block whose study is missing carries none and says so (transferred ablation).
#
# Every command is commented out: copy one to run it by hand, or feed the whole
# file to scripts/experiments/submit_train.sh, which runs them under nohup.

# ========================================================================
# jetformer  nconst=100
# ========================================================================

# ------------------------------------------------------------------------
# jetformer  nconst=100  baseline  (method=augmentation, group=none)
# best trial 81 of top_tagging_jetformer_nconst100_augmentation_none, val/acc_best=0.934701
# ------------------------------------------------------------------------
# python3 src/train.py \
#     experiment=top_tagging/jetformer_100 \
#     data.method=augmentation \
#     data.augmentation_scheme=none \
#     data.n_folds=5 \
#     data.features='[pt,eta,phi]' \
#     algorithm.dropout=0.0 \
#     algorithm.embed_dim=32 \
#     algorithm.num_heads=8 \
#     algorithm.num_transformers=5 \
#     algorithm.optimizer.lr=0.009135148517570918 \
#     algorithm.optimizer.weight_decay=0.01 \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/train/top_tagging/pt_eta_phi/jetformer/nconst100/baseline

# ------------------------------------------------------------------------
# jetformer  nconst=100  aug_so2  (method=augmentation, group=so2)
# best trial 25 of top_tagging_jetformer_nconst100_augmentation_so2, val/acc_best=0.936936
# ------------------------------------------------------------------------
# python3 src/train.py \
#     experiment=top_tagging/jetformer_100 \
#     data.method=augmentation \
#     data.augmentation_scheme=so2 \
#     data.n_folds=5 \
#     data.features='[pt,eta,phi]' \
#     algorithm.dropout=0.0 \
#     algorithm.embed_dim=96 \
#     algorithm.num_heads=4 \
#     algorithm.num_transformers=5 \
#     algorithm.optimizer.lr=0.0042909965463948496 \
#     algorithm.optimizer.weight_decay=0.01 \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/train/top_tagging/pt_eta_phi/jetformer/nconst100/aug_so2

# ------------------------------------------------------------------------
# jetformer  nconst=100  aug_so3  (method=augmentation, group=so3)
# best trial 89 of top_tagging_jetformer_nconst100_augmentation_so3, val/acc_best=0.935519
# ------------------------------------------------------------------------
# python3 src/train.py \
#     experiment=top_tagging/jetformer_100 \
#     data.method=augmentation \
#     data.augmentation_scheme=so3 \
#     data.n_folds=5 \
#     data.features='[pt,eta,phi]' \
#     algorithm.dropout=0.0 \
#     algorithm.embed_dim=48 \
#     algorithm.num_heads=8 \
#     algorithm.num_transformers=5 \
#     algorithm.optimizer.lr=0.012645638085517958 \
#     algorithm.optimizer.weight_decay=0.0001 \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/train/top_tagging/pt_eta_phi/jetformer/nconst100/aug_so3

# ------------------------------------------------------------------------
# jetformer  nconst=100  aug_lorentz  (method=augmentation, group=lorentz)
# best trial 36 of top_tagging_jetformer_nconst100_augmentation_lorentz, val/acc_best=0.934669
# ------------------------------------------------------------------------
# python3 src/train.py \
#     experiment=top_tagging/jetformer_100 \
#     data.method=augmentation \
#     data.augmentation_scheme=lorentz \
#     data.n_folds=5 \
#     data.features='[pt,eta,phi]' \
#     algorithm.dropout=0.0 \
#     algorithm.embed_dim=128 \
#     algorithm.num_heads=8 \
#     algorithm.num_transformers=4 \
#     algorithm.optimizer.lr=0.005507582737090801 \
#     algorithm.optimizer.weight_decay=0.05 \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/train/top_tagging/pt_eta_phi/jetformer/nconst100/aug_lorentz

# ------------------------------------------------------------------------
# jetformer  nconst=100  symm_so2  (method=symm_loss, group=so2)
# best trial 95 of top_tagging_jetformer_nconst100_symm_loss_so2, val/acc_best=0.935259
# ------------------------------------------------------------------------
# python3 src/train.py \
#     experiment=top_tagging/jetformer_100 \
#     data.method=symm_loss \
#     data.augmentation_scheme=so2 \
#     data.n_folds=5 \
#     data.features='[pt,eta,phi]' \
#     algorithm.dropout=0.0 \
#     algorithm.embed_dim=64 \
#     algorithm.num_heads=8 \
#     algorithm.num_transformers=5 \
#     algorithm.optimizer.lr=0.001134324956156878 \
#     algorithm.optimizer.weight_decay=0.0001 \
#     algorithm.symm_loss_weight=0.0467480048606003 \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/train/top_tagging/pt_eta_phi/jetformer/nconst100/symm_so2

# ------------------------------------------------------------------------
# jetformer  nconst=100  symm_so3  (method=symm_loss, group=so3)
# best trial 70 of top_tagging_jetformer_nconst100_symm_loss_so3, val/acc_best=0.936346
# ------------------------------------------------------------------------
# python3 src/train.py \
#     experiment=top_tagging/jetformer_100 \
#     data.method=symm_loss \
#     data.augmentation_scheme=so3 \
#     data.n_folds=5 \
#     data.features='[pt,eta,phi]' \
#     algorithm.dropout=0.0 \
#     algorithm.embed_dim=64 \
#     algorithm.num_heads=8 \
#     algorithm.num_transformers=5 \
#     algorithm.optimizer.lr=0.0028057168446089507 \
#     algorithm.optimizer.weight_decay=0.05 \
#     algorithm.symm_loss_weight=0.13665646281926455 \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/train/top_tagging/pt_eta_phi/jetformer/nconst100/symm_so3

# ------------------------------------------------------------------------
# jetformer  nconst=100  symm_lorentz  (method=symm_loss, group=lorentz)
# best trial 95 of top_tagging_jetformer_nconst100_symm_loss_lorentz, val/acc_best=0.934395
# ------------------------------------------------------------------------
# python3 src/train.py \
#     experiment=top_tagging/jetformer_100 \
#     data.method=symm_loss \
#     data.augmentation_scheme=lorentz \
#     data.n_folds=5 \
#     data.features='[pt,eta,phi]' \
#     algorithm.dropout=0.0 \
#     algorithm.embed_dim=96 \
#     algorithm.num_heads=8 \
#     algorithm.num_transformers=5 \
#     algorithm.optimizer.lr=0.0006052761085297995 \
#     algorithm.optimizer.weight_decay=0.05 \
#     algorithm.symm_loss_weight=0.021152885972037887 \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/train/top_tagging/pt_eta_phi/jetformer/nconst100/symm_lorentz
