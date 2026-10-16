#!/usr/bin/env bash
# ========================================================================
# DEEPSETS TRAINING COMMANDS
# ========================================================================
# One block per (constituent count, regime): 3 x 7 = 21 train runs. Generated
# by scripts/experiments/make_train_catalogues.py; regenerate rather than edit by hand.
# Feature set [pt, etarel, phirel]: studies from logs/optuna/hls4ml/pt_etarel_phirel/, runs under logs/{train,eval}/hls4ml/pt_etarel_phirel/.
# Each block carries the best of the first 200 completed trials, by trial
# number, of its own study under logs/optuna/hls4ml/pt_etarel_phirel/;
# a block whose study is missing carries none and says so (transferred ablation).
#
# Every command is commented out: copy one to run it by hand, or feed the whole
# file to scripts/experiments/submit_train.sh, which runs them under nohup.

# ========================================================================
# deepsets  nconst=16
# ========================================================================

# ------------------------------------------------------------------------
# deepsets  nconst=16  baseline  (method=augmentation, group=none)
# best trial 199 of deepsets_nconst16_augmentation_none, val/acc_best=0.705415
# ------------------------------------------------------------------------
# python3 src/train.py \
#     experiment=hls4ml/deepsets_16 \
#     data.method=augmentation \
#     data.augmentation_scheme=none \
#     data.n_folds=5 \
#     data.features='[pt,etarel,phirel]' \
#     algorithm.aggregation=sum \
#     algorithm.optimizer.lr=0.004300044834629764 \
#     algorithm.optimizer.weight_decay=0.0 \
#     algorithm.phi_layers='[64,64,64]' \
#     algorithm.rho_layers='[64,32]' \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/train/hls4ml/pt_etarel_phirel/deepsets/nconst16/baseline

# ------------------------------------------------------------------------
# deepsets  nconst=16  aug_so2  (method=augmentation, group=so2)
# best trial 194 of deepsets_nconst16_augmentation_so2, val/acc_best=0.706739
# ------------------------------------------------------------------------
# python3 src/train.py \
#     experiment=hls4ml/deepsets_16 \
#     data.method=augmentation \
#     data.augmentation_scheme=so2 \
#     data.n_folds=5 \
#     data.features='[pt,etarel,phirel]' \
#     algorithm.aggregation=sum \
#     algorithm.optimizer.lr=0.0050102028171930485 \
#     algorithm.optimizer.weight_decay=1e-07 \
#     algorithm.phi_layers='[64,64,32]' \
#     algorithm.rho_layers='[64,32]' \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/train/hls4ml/pt_etarel_phirel/deepsets/nconst16/aug_so2

# ------------------------------------------------------------------------
# deepsets  nconst=16  aug_so3  (method=augmentation, group=so3)
# best trial 75 of deepsets_nconst16_augmentation_so3, val/acc_best=0.654832
# ------------------------------------------------------------------------
# python3 src/train.py \
#     experiment=hls4ml/deepsets_16 \
#     data.method=augmentation \
#     data.augmentation_scheme=so3 \
#     data.n_folds=5 \
#     data.features='[pt,etarel,phirel]' \
#     algorithm.aggregation=sum \
#     algorithm.optimizer.lr=0.005343190284951742 \
#     algorithm.optimizer.weight_decay=0.0 \
#     algorithm.phi_layers='[64,64,32]' \
#     algorithm.rho_layers='[64,32]' \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/train/hls4ml/pt_etarel_phirel/deepsets/nconst16/aug_so3

# ------------------------------------------------------------------------
# deepsets  nconst=16  aug_lorentz  (method=augmentation, group=lorentz)
# best trial 134 of deepsets_nconst16_augmentation_lorentz, val/acc_best=0.706966
# ------------------------------------------------------------------------
# python3 src/train.py \
#     experiment=hls4ml/deepsets_16 \
#     data.method=augmentation \
#     data.augmentation_scheme=lorentz \
#     data.n_folds=5 \
#     data.features='[pt,etarel,phirel]' \
#     algorithm.aggregation=sum \
#     algorithm.optimizer.lr=0.004679157584642464 \
#     algorithm.optimizer.weight_decay=0.0 \
#     algorithm.phi_layers='[64,64,32]' \
#     algorithm.rho_layers='[64,32]' \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/train/hls4ml/pt_etarel_phirel/deepsets/nconst16/aug_lorentz

# ------------------------------------------------------------------------
# deepsets  nconst=16  symm_so2  (method=symm_loss, group=so2)
# best trial 176 of deepsets_nconst16_symm_loss_so2, val/acc_best=0.706369
# ------------------------------------------------------------------------
# python3 src/train.py \
#     experiment=hls4ml/deepsets_16 \
#     data.method=symm_loss \
#     data.augmentation_scheme=so2 \
#     data.n_folds=5 \
#     data.features='[pt,etarel,phirel]' \
#     algorithm.aggregation=sum \
#     algorithm.optimizer.lr=0.0036764179702193076 \
#     algorithm.optimizer.weight_decay=1e-06 \
#     algorithm.phi_layers='[128,64,32]' \
#     algorithm.rho_layers='[64,32]' \
#     algorithm.symm_loss_weight=0.06617415774335313 \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/train/hls4ml/pt_etarel_phirel/deepsets/nconst16/symm_so2

# ------------------------------------------------------------------------
# deepsets  nconst=16  symm_so3  (method=symm_loss, group=so3)
# best trial 125 of deepsets_nconst16_symm_loss_so3, val/acc_best=0.708284
# ------------------------------------------------------------------------
# python3 src/train.py \
#     experiment=hls4ml/deepsets_16 \
#     data.method=symm_loss \
#     data.augmentation_scheme=so3 \
#     data.n_folds=5 \
#     data.features='[pt,etarel,phirel]' \
#     algorithm.aggregation=sum \
#     algorithm.optimizer.lr=0.0043537240156101414 \
#     algorithm.optimizer.weight_decay=1e-06 \
#     algorithm.phi_layers='[64,64,32]' \
#     algorithm.rho_layers='[64,32]' \
#     algorithm.symm_loss_weight=0.002192477069547221 \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/train/hls4ml/pt_etarel_phirel/deepsets/nconst16/symm_so3

# ------------------------------------------------------------------------
# deepsets  nconst=16  symm_lorentz  (method=symm_loss, group=lorentz)
# best trial 184 of deepsets_nconst16_symm_loss_lorentz, val/acc_best=0.705378
# ------------------------------------------------------------------------
# python3 src/train.py \
#     experiment=hls4ml/deepsets_16 \
#     data.method=symm_loss \
#     data.augmentation_scheme=lorentz \
#     data.n_folds=5 \
#     data.features='[pt,etarel,phirel]' \
#     algorithm.aggregation=sum \
#     algorithm.optimizer.lr=0.003869930659661624 \
#     algorithm.optimizer.weight_decay=1e-07 \
#     algorithm.phi_layers='[128,64,32]' \
#     algorithm.rho_layers='[64,32]' \
#     algorithm.symm_loss_weight=0.010186334476836981 \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/train/hls4ml/pt_etarel_phirel/deepsets/nconst16/symm_lorentz

# ========================================================================
# deepsets  nconst=50
# ========================================================================

# ------------------------------------------------------------------------
# deepsets  nconst=50  baseline  (method=augmentation, group=none)
# best trial 175 of deepsets_nconst50_augmentation_none, val/acc_best=0.803156
# ------------------------------------------------------------------------
# python3 src/train.py \
#     experiment=hls4ml/deepsets_50 \
#     data.method=augmentation \
#     data.augmentation_scheme=none \
#     data.n_folds=5 \
#     data.features='[pt,etarel,phirel]' \
#     algorithm.aggregation=sum \
#     algorithm.optimizer.lr=0.0038020938205913656 \
#     algorithm.optimizer.weight_decay=1e-07 \
#     algorithm.phi_layers='[64,64,32]' \
#     algorithm.rho_layers='[64,32]' \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/train/hls4ml/pt_etarel_phirel/deepsets/nconst50/baseline

# ------------------------------------------------------------------------
# deepsets  nconst=50  aug_so2  (method=augmentation, group=so2)
# best trial 140 of deepsets_nconst50_augmentation_so2, val/acc_best=0.804063
# ------------------------------------------------------------------------
# python3 src/train.py \
#     experiment=hls4ml/deepsets_50 \
#     data.method=augmentation \
#     data.augmentation_scheme=so2 \
#     data.n_folds=5 \
#     data.features='[pt,etarel,phirel]' \
#     algorithm.aggregation=sum \
#     algorithm.optimizer.lr=0.004729345849112714 \
#     algorithm.optimizer.weight_decay=1e-06 \
#     algorithm.phi_layers='[128,64,32]' \
#     algorithm.rho_layers='[64,32]' \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/train/hls4ml/pt_etarel_phirel/deepsets/nconst50/aug_so2

# ------------------------------------------------------------------------
# deepsets  nconst=50  aug_so3  (method=augmentation, group=so3)
# best trial 185 of deepsets_nconst50_augmentation_so3, val/acc_best=0.772664
# ------------------------------------------------------------------------
# python3 src/train.py \
#     experiment=hls4ml/deepsets_50 \
#     data.method=augmentation \
#     data.augmentation_scheme=so3 \
#     data.n_folds=5 \
#     data.features='[pt,etarel,phirel]' \
#     algorithm.aggregation=sum \
#     algorithm.optimizer.lr=0.003451316423854901 \
#     algorithm.optimizer.weight_decay=1e-06 \
#     algorithm.phi_layers='[64,64,32]' \
#     algorithm.rho_layers='[64,32]' \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/train/hls4ml/pt_etarel_phirel/deepsets/nconst50/aug_so3

# ------------------------------------------------------------------------
# deepsets  nconst=50  aug_lorentz  (method=augmentation, group=lorentz)
# best trial 83 of deepsets_nconst50_augmentation_lorentz, val/acc_best=0.803799
# ------------------------------------------------------------------------
# python3 src/train.py \
#     experiment=hls4ml/deepsets_50 \
#     data.method=augmentation \
#     data.augmentation_scheme=lorentz \
#     data.n_folds=5 \
#     data.features='[pt,etarel,phirel]' \
#     algorithm.aggregation=sum \
#     algorithm.optimizer.lr=0.0034879918231822222 \
#     algorithm.optimizer.weight_decay=1e-07 \
#     algorithm.phi_layers='[128,64,32]' \
#     algorithm.rho_layers='[64,32]' \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/train/hls4ml/pt_etarel_phirel/deepsets/nconst50/aug_lorentz

# ------------------------------------------------------------------------
# deepsets  nconst=50  symm_so2  (method=symm_loss, group=so2)
# best trial 183 of deepsets_nconst50_symm_loss_so2, val/acc_best=0.805012
# ------------------------------------------------------------------------
# python3 src/train.py \
#     experiment=hls4ml/deepsets_50 \
#     data.method=symm_loss \
#     data.augmentation_scheme=so2 \
#     data.n_folds=5 \
#     data.features='[pt,etarel,phirel]' \
#     algorithm.aggregation=sum_over_n \
#     algorithm.optimizer.lr=0.00734100122907693 \
#     algorithm.optimizer.weight_decay=0.0 \
#     algorithm.phi_layers='[64,64,64]' \
#     algorithm.rho_layers='[64,32]' \
#     algorithm.symm_loss_weight=0.05138290400109676 \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/train/hls4ml/pt_etarel_phirel/deepsets/nconst50/symm_so2

# ------------------------------------------------------------------------
# deepsets  nconst=50  symm_so3  (method=symm_loss, group=so3)
# best trial 170 of deepsets_nconst50_symm_loss_so3, val/acc_best=0.807188
# ------------------------------------------------------------------------
# python3 src/train.py \
#     experiment=hls4ml/deepsets_50 \
#     data.method=symm_loss \
#     data.augmentation_scheme=so3 \
#     data.n_folds=5 \
#     data.features='[pt,etarel,phirel]' \
#     algorithm.aggregation=sum \
#     algorithm.optimizer.lr=0.0037028155350743246 \
#     algorithm.optimizer.weight_decay=0.0 \
#     algorithm.phi_layers='[64,64,32]' \
#     algorithm.rho_layers='[64,32]' \
#     algorithm.symm_loss_weight=0.0013238102183391243 \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/train/hls4ml/pt_etarel_phirel/deepsets/nconst50/symm_so3

# ------------------------------------------------------------------------
# deepsets  nconst=50  symm_lorentz  (method=symm_loss, group=lorentz)
# best trial 129 of deepsets_nconst50_symm_loss_lorentz, val/acc_best=0.804837
# ------------------------------------------------------------------------
# python3 src/train.py \
#     experiment=hls4ml/deepsets_50 \
#     data.method=symm_loss \
#     data.augmentation_scheme=lorentz \
#     data.n_folds=5 \
#     data.features='[pt,etarel,phirel]' \
#     algorithm.aggregation=sum \
#     algorithm.optimizer.lr=0.0038594316844185717 \
#     algorithm.optimizer.weight_decay=0.0 \
#     algorithm.phi_layers='[128,64,32]' \
#     algorithm.rho_layers='[64,32]' \
#     algorithm.symm_loss_weight=0.0019866484238524455 \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/train/hls4ml/pt_etarel_phirel/deepsets/nconst50/symm_lorentz

# ========================================================================
# deepsets  nconst=150
# ========================================================================

# ------------------------------------------------------------------------
# deepsets  nconst=150  baseline  (method=augmentation, group=none)
# best trial 55 of deepsets_nconst150_augmentation_none, val/acc_best=0.811866
# ------------------------------------------------------------------------
# python3 src/train.py \
#     experiment=hls4ml/deepsets_150 \
#     data.method=augmentation \
#     data.augmentation_scheme=none \
#     data.n_folds=5 \
#     data.features='[pt,etarel,phirel]' \
#     algorithm.aggregation=sum \
#     algorithm.optimizer.lr=0.004546765882927424 \
#     algorithm.optimizer.weight_decay=1e-07 \
#     algorithm.phi_layers='[64,64,32]' \
#     algorithm.rho_layers='[64,32]' \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/train/hls4ml/pt_etarel_phirel/deepsets/nconst150/baseline

# ------------------------------------------------------------------------
# deepsets  nconst=150  aug_so2  (method=augmentation, group=so2)
# best trial 139 of deepsets_nconst150_augmentation_so2, val/acc_best=0.812284
# ------------------------------------------------------------------------
# python3 src/train.py \
#     experiment=hls4ml/deepsets_150 \
#     data.method=augmentation \
#     data.augmentation_scheme=so2 \
#     data.n_folds=5 \
#     data.features='[pt,etarel,phirel]' \
#     algorithm.aggregation=sum \
#     algorithm.optimizer.lr=0.004804854638601682 \
#     algorithm.optimizer.weight_decay=0.0 \
#     algorithm.phi_layers='[128,64,32]' \
#     algorithm.rho_layers='[64,32]' \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/train/hls4ml/pt_etarel_phirel/deepsets/nconst150/aug_so2

# ------------------------------------------------------------------------
# deepsets  nconst=150  aug_so3  (method=augmentation, group=so3)
# best trial 161 of deepsets_nconst150_augmentation_so3, val/acc_best=0.788957
# ------------------------------------------------------------------------
# python3 src/train.py \
#     experiment=hls4ml/deepsets_150 \
#     data.method=augmentation \
#     data.augmentation_scheme=so3 \
#     data.n_folds=5 \
#     data.features='[pt,etarel,phirel]' \
#     algorithm.aggregation=sum \
#     algorithm.optimizer.lr=0.003404754196588863 \
#     algorithm.optimizer.weight_decay=0.0 \
#     algorithm.phi_layers='[64,64,64]' \
#     algorithm.rho_layers='[64,32]' \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/train/hls4ml/pt_etarel_phirel/deepsets/nconst150/aug_so3

# ------------------------------------------------------------------------
# deepsets  nconst=150  aug_lorentz  (method=augmentation, group=lorentz)
# best trial 137 of deepsets_nconst150_augmentation_lorentz, val/acc_best=0.813537
# ------------------------------------------------------------------------
# python3 src/train.py \
#     experiment=hls4ml/deepsets_150 \
#     data.method=augmentation \
#     data.augmentation_scheme=lorentz \
#     data.n_folds=5 \
#     data.features='[pt,etarel,phirel]' \
#     algorithm.aggregation=sum_over_n \
#     algorithm.optimizer.lr=0.009919808351833157 \
#     algorithm.optimizer.weight_decay=1e-07 \
#     algorithm.phi_layers='[64,64,64]' \
#     algorithm.rho_layers='[64,32]' \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/train/hls4ml/pt_etarel_phirel/deepsets/nconst150/aug_lorentz

# ------------------------------------------------------------------------
# deepsets  nconst=150  symm_so2  (method=symm_loss, group=so2)
# best trial 59 of deepsets_nconst150_symm_loss_so2, val/acc_best=0.811261
# ------------------------------------------------------------------------
# python3 src/train.py \
#     experiment=hls4ml/deepsets_150 \
#     data.method=symm_loss \
#     data.augmentation_scheme=so2 \
#     data.n_folds=5 \
#     data.features='[pt,etarel,phirel]' \
#     algorithm.aggregation=sum \
#     algorithm.optimizer.lr=0.0051609932570410005 \
#     algorithm.optimizer.weight_decay=1e-06 \
#     algorithm.phi_layers='[64,32,32]' \
#     algorithm.rho_layers='[64,32]' \
#     algorithm.symm_loss_weight=0.031001064129926585 \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/train/hls4ml/pt_etarel_phirel/deepsets/nconst150/symm_so2

# ------------------------------------------------------------------------
# deepsets  nconst=150  symm_so3  (method=symm_loss, group=so3)
# best trial 155 of deepsets_nconst150_symm_loss_so3, val/acc_best=0.814645
# ------------------------------------------------------------------------
# python3 src/train.py \
#     experiment=hls4ml/deepsets_150 \
#     data.method=symm_loss \
#     data.augmentation_scheme=so3 \
#     data.n_folds=5 \
#     data.features='[pt,etarel,phirel]' \
#     algorithm.aggregation=sum \
#     algorithm.optimizer.lr=0.00406072268318664 \
#     algorithm.optimizer.weight_decay=1e-07 \
#     algorithm.phi_layers='[64,64,32]' \
#     algorithm.rho_layers='[64,32]' \
#     algorithm.symm_loss_weight=0.0019557622919662686 \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/train/hls4ml/pt_etarel_phirel/deepsets/nconst150/symm_so3

# ------------------------------------------------------------------------
# deepsets  nconst=150  symm_lorentz  (method=symm_loss, group=lorentz)
# best trial 118 of deepsets_nconst150_symm_loss_lorentz, val/acc_best=0.812715
# ------------------------------------------------------------------------
# python3 src/train.py \
#     experiment=hls4ml/deepsets_150 \
#     data.method=symm_loss \
#     data.augmentation_scheme=lorentz \
#     data.n_folds=5 \
#     data.features='[pt,etarel,phirel]' \
#     algorithm.aggregation=sum \
#     algorithm.optimizer.lr=0.005206489381334099 \
#     algorithm.optimizer.weight_decay=1e-07 \
#     algorithm.phi_layers='[64,64,32]' \
#     algorithm.rho_layers='[64,32]' \
#     algorithm.symm_loss_weight=0.045525691080075294 \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/train/hls4ml/pt_etarel_phirel/deepsets/nconst150/symm_lorentz
