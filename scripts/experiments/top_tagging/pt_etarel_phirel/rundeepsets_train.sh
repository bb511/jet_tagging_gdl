#!/usr/bin/env bash
# ========================================================================
# DEEPSETS TRAINING COMMANDS
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
# deepsets  nconst=100
# ========================================================================

# ------------------------------------------------------------------------
# deepsets  nconst=100  baseline  (method=augmentation, group=none)
# best trial 38 of top_tagging_deepsets_nconst100_augmentation_none, val/acc_best=0.929558
# ------------------------------------------------------------------------
# python3 src/train.py \
#     experiment=top_tagging/deepsets_100 \
#     data.method=augmentation \
#     data.augmentation_scheme=none \
#     data.n_folds=5 \
#     algorithm.aggregation=sum \
#     algorithm.optimizer.lr=0.0023558863048995108 \
#     algorithm.optimizer.weight_decay=0.0 \
#     algorithm.phi_layers='[64,64,64]' \
#     algorithm.rho_layers='[128,64]' \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/train/top_tagging/deepsets/nconst100/baseline

# ------------------------------------------------------------------------
# deepsets  nconst=100  aug_so2  (method=augmentation, group=so2)
# best trial 77 of top_tagging_deepsets_nconst100_augmentation_so2, val/acc_best=0.928837
# ------------------------------------------------------------------------
# python3 src/train.py \
#     experiment=top_tagging/deepsets_100 \
#     data.method=augmentation \
#     data.augmentation_scheme=so2 \
#     data.n_folds=5 \
#     algorithm.aggregation=sum \
#     algorithm.optimizer.lr=0.001572878651540804 \
#     algorithm.optimizer.weight_decay=1e-06 \
#     algorithm.phi_layers='[64,64,64]' \
#     algorithm.rho_layers='[64,32]' \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/train/top_tagging/deepsets/nconst100/aug_so2

# ------------------------------------------------------------------------
# deepsets  nconst=100  aug_so3  (method=augmentation, group=so3)
# best trial 29 of top_tagging_deepsets_nconst100_augmentation_so3, val/acc_best=0.927468
# ------------------------------------------------------------------------
# python3 src/train.py \
#     experiment=top_tagging/deepsets_100 \
#     data.method=augmentation \
#     data.augmentation_scheme=so3 \
#     data.n_folds=5 \
#     algorithm.aggregation=sum_over_n \
#     algorithm.optimizer.lr=0.005475622019399547 \
#     algorithm.optimizer.weight_decay=0.0 \
#     algorithm.phi_layers='[64,64,64]' \
#     algorithm.rho_layers='[128,64]' \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/train/top_tagging/deepsets/nconst100/aug_so3

# ------------------------------------------------------------------------
# deepsets  nconst=100  aug_lorentz  (method=augmentation, group=lorentz)
# best trial 93 of top_tagging_deepsets_nconst100_augmentation_lorentz, val/acc_best=0.929977
# ------------------------------------------------------------------------
# python3 src/train.py \
#     experiment=top_tagging/deepsets_100 \
#     data.method=augmentation \
#     data.augmentation_scheme=lorentz \
#     data.n_folds=5 \
#     algorithm.aggregation=sum \
#     algorithm.optimizer.lr=0.001880090109852315 \
#     algorithm.optimizer.weight_decay=0.0 \
#     algorithm.phi_layers='[64,64,64]' \
#     algorithm.rho_layers='[128,64]' \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/train/top_tagging/deepsets/nconst100/aug_lorentz

# ------------------------------------------------------------------------
# deepsets  nconst=100  symm_so2  (method=symm_loss, group=so2)
# best trial 45 of top_tagging_deepsets_nconst100_symm_loss_so2, val/acc_best=0.929383
# ------------------------------------------------------------------------
# python3 src/train.py \
#     experiment=top_tagging/deepsets_100 \
#     data.method=symm_loss \
#     data.augmentation_scheme=so2 \
#     data.n_folds=5 \
#     algorithm.aggregation=sum \
#     algorithm.optimizer.lr=0.00300628047498411 \
#     algorithm.optimizer.weight_decay=0.0 \
#     algorithm.phi_layers='[64,64,64]' \
#     algorithm.rho_layers='[128,64]' \
#     algorithm.symm_loss_weight=0.057307331652758864 \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/train/top_tagging/deepsets/nconst100/symm_so2

# ------------------------------------------------------------------------
# deepsets  nconst=100  symm_so3  (method=symm_loss, group=so3)
# best trial 87 of top_tagging_deepsets_nconst100_symm_loss_so3, val/acc_best=0.929883
# ------------------------------------------------------------------------
# python3 src/train.py \
#     experiment=top_tagging/deepsets_100 \
#     data.method=symm_loss \
#     data.augmentation_scheme=so3 \
#     data.n_folds=5 \
#     algorithm.aggregation=sum \
#     algorithm.optimizer.lr=0.0025192542004751997 \
#     algorithm.optimizer.weight_decay=1e-05 \
#     algorithm.phi_layers='[64,64,64]' \
#     algorithm.rho_layers='[128,64]' \
#     algorithm.symm_loss_weight=0.03137383856405835 \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/train/top_tagging/deepsets/nconst100/symm_so3

# ------------------------------------------------------------------------
# deepsets  nconst=100  symm_lorentz  (method=symm_loss, group=lorentz)
# best trial 94 of top_tagging_deepsets_nconst100_symm_loss_lorentz, val/acc_best=0.929904
# ------------------------------------------------------------------------
# python3 src/train.py \
#     experiment=top_tagging/deepsets_100 \
#     data.method=symm_loss \
#     data.augmentation_scheme=lorentz \
#     data.n_folds=5 \
#     algorithm.aggregation=sum \
#     algorithm.optimizer.lr=0.0015156930799081504 \
#     algorithm.optimizer.weight_decay=0.0 \
#     algorithm.phi_layers='[128,128,64]' \
#     algorithm.rho_layers='[128,64]' \
#     algorithm.symm_loss_weight=0.0014481382770111999 \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/train/top_tagging/deepsets/nconst100/symm_lorentz
