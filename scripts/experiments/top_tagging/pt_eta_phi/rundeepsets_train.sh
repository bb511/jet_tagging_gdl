#!/usr/bin/env bash
# ========================================================================
# DEEPSETS TRAINING COMMANDS
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
# deepsets  nconst=100
# ========================================================================

# ------------------------------------------------------------------------
# deepsets  nconst=100  baseline  (method=augmentation, group=none)
# best trial 64 of top_tagging_deepsets_nconst100_augmentation_none, val/acc_best=0.894874
# ------------------------------------------------------------------------
# python3 src/train.py \
#     experiment=top_tagging/deepsets_100 \
#     data.method=augmentation \
#     data.augmentation_scheme=none \
#     data.n_folds=5 \
#     data.features='[pt,eta,phi]' \
#     algorithm.aggregation=max \
#     algorithm.optimizer.lr=0.004804650121513464 \
#     algorithm.optimizer.weight_decay=1e-06 \
#     algorithm.phi_layers='[64,64,64]' \
#     algorithm.rho_layers='[32,32,16]' \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/train/top_tagging/pt_eta_phi/deepsets/nconst100/baseline

# ------------------------------------------------------------------------
# deepsets  nconst=100  aug_so2  (method=augmentation, group=so2)
# best trial 77 of top_tagging_deepsets_nconst100_augmentation_so2, val/acc_best=0.894620
# ------------------------------------------------------------------------
# python3 src/train.py \
#     experiment=top_tagging/deepsets_100 \
#     data.method=augmentation \
#     data.augmentation_scheme=so2 \
#     data.n_folds=5 \
#     data.features='[pt,eta,phi]' \
#     algorithm.aggregation=max \
#     algorithm.optimizer.lr=0.003941864522374496 \
#     algorithm.optimizer.weight_decay=0.0 \
#     algorithm.phi_layers='[64,64,64]' \
#     algorithm.rho_layers='[64,32]' \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/train/top_tagging/pt_eta_phi/deepsets/nconst100/aug_so2

# ------------------------------------------------------------------------
# deepsets  nconst=100  aug_so3  (method=augmentation, group=so3)
# best trial 75 of top_tagging_deepsets_nconst100_augmentation_so3, val/acc_best=0.889201
# ------------------------------------------------------------------------
# python3 src/train.py \
#     experiment=top_tagging/deepsets_100 \
#     data.method=augmentation \
#     data.augmentation_scheme=so3 \
#     data.n_folds=5 \
#     data.features='[pt,eta,phi]' \
#     algorithm.aggregation=max \
#     algorithm.optimizer.lr=0.0023498686730325916 \
#     algorithm.optimizer.weight_decay=1e-05 \
#     algorithm.phi_layers='[128,128,64]' \
#     algorithm.rho_layers='[16]' \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/train/top_tagging/pt_eta_phi/deepsets/nconst100/aug_so3

# ------------------------------------------------------------------------
# deepsets  nconst=100  aug_lorentz  (method=augmentation, group=lorentz)
# best trial 99 of top_tagging_deepsets_nconst100_augmentation_lorentz, val/acc_best=0.895219
# ------------------------------------------------------------------------
# python3 src/train.py \
#     experiment=top_tagging/deepsets_100 \
#     data.method=augmentation \
#     data.augmentation_scheme=lorentz \
#     data.n_folds=5 \
#     data.features='[pt,eta,phi]' \
#     algorithm.aggregation=max \
#     algorithm.optimizer.lr=0.001593219185091393 \
#     algorithm.optimizer.weight_decay=0.0 \
#     algorithm.phi_layers='[128,128,64]' \
#     algorithm.rho_layers='[32,32,16]' \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/train/top_tagging/pt_eta_phi/deepsets/nconst100/aug_lorentz

# ------------------------------------------------------------------------
# deepsets  nconst=100  symm_so2  (method=symm_loss, group=so2)
# best trial 79 of top_tagging_deepsets_nconst100_symm_loss_so2, val/acc_best=0.895732
# ------------------------------------------------------------------------
# python3 src/train.py \
#     experiment=top_tagging/deepsets_100 \
#     data.method=symm_loss \
#     data.augmentation_scheme=so2 \
#     data.n_folds=5 \
#     data.features='[pt,eta,phi]' \
#     algorithm.aggregation=max \
#     algorithm.optimizer.lr=0.0008131133215273128 \
#     algorithm.optimizer.weight_decay=1e-06 \
#     algorithm.phi_layers='[64,64,64]' \
#     algorithm.rho_layers='[128,64]' \
#     algorithm.symm_loss_weight=0.016427933035815117 \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/train/top_tagging/pt_eta_phi/deepsets/nconst100/symm_so2

# ------------------------------------------------------------------------
# deepsets  nconst=100  symm_so3  (method=symm_loss, group=so3)
# best trial 25 of top_tagging_deepsets_nconst100_symm_loss_so3, val/acc_best=0.895603
# ------------------------------------------------------------------------
# python3 src/train.py \
#     experiment=top_tagging/deepsets_100 \
#     data.method=symm_loss \
#     data.augmentation_scheme=so3 \
#     data.n_folds=5 \
#     data.features='[pt,eta,phi]' \
#     algorithm.aggregation=max \
#     algorithm.optimizer.lr=0.002557288201967478 \
#     algorithm.optimizer.weight_decay=0.0 \
#     algorithm.phi_layers='[128,128,64]' \
#     algorithm.rho_layers='[32,32,16]' \
#     algorithm.symm_loss_weight=0.007099401857912865 \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/train/top_tagging/pt_eta_phi/deepsets/nconst100/symm_so3

# ------------------------------------------------------------------------
# deepsets  nconst=100  symm_lorentz  (method=symm_loss, group=lorentz)
# best trial 55 of top_tagging_deepsets_nconst100_symm_loss_lorentz, val/acc_best=0.895292
# ------------------------------------------------------------------------
# python3 src/train.py \
#     experiment=top_tagging/deepsets_100 \
#     data.method=symm_loss \
#     data.augmentation_scheme=lorentz \
#     data.n_folds=5 \
#     data.features='[pt,eta,phi]' \
#     algorithm.aggregation=max \
#     algorithm.optimizer.lr=0.003326816761509172 \
#     algorithm.optimizer.weight_decay=1e-06 \
#     algorithm.phi_layers='[64,64,64]' \
#     algorithm.rho_layers='[128,64]' \
#     algorithm.symm_loss_weight=0.003286480138015623 \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/train/top_tagging/pt_eta_phi/deepsets/nconst100/symm_lorentz
