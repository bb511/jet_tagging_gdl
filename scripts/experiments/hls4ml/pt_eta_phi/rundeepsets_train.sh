#!/usr/bin/env bash
# ========================================================================
# DEEPSETS TRAINING COMMANDS
# ========================================================================
# One block per (constituent count, regime): 3 x 7 = 21 train runs. Generated
# by scripts/experiments/make_train_catalogues.py; regenerate rather than edit by hand.
# Feature set [pt, eta, phi]: studies from logs/optuna/hls4ml/pt_eta_phi/, runs under logs/{train,eval}/hls4ml/pt_eta_phi/.
# Each block carries the best of the first 200 completed trials, by trial
# number, of its own study under logs/optuna/hls4ml/pt_eta_phi/;
# a block whose study is missing carries none and says so (transferred ablation).
#
# Every command is commented out: copy one to run it by hand, or feed the whole
# file to scripts/experiments/submit_train.sh, which queues them on a GPU or slurm.

# ========================================================================
# deepsets  nconst=16
# ========================================================================

# ------------------------------------------------------------------------
# deepsets  nconst=16  baseline  (method=augmentation, group=none)
# best trial 108 of deepsets_nconst16_augmentation_none, val/acc_best=0.589368
# ------------------------------------------------------------------------
# python3 src/train.py \
#     experiment=hls4ml/deepsets_16 \
#     data.method=augmentation \
#     data.augmentation_scheme=none \
#     data.n_folds=5 \
#     data.features='[pt,eta,phi]' \
#     algorithm.aggregation=max \
#     algorithm.optimizer.lr=0.006579971043999283 \
#     algorithm.optimizer.weight_decay=0.0 \
#     algorithm.phi_layers='[64,64,64]' \
#     algorithm.rho_layers='[32,16]' \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/train/hls4ml/pt_eta_phi/deepsets/nconst16/baseline

# ------------------------------------------------------------------------
# deepsets  nconst=16  aug_so2  (method=augmentation, group=so2)
# best trial 69 of deepsets_nconst16_augmentation_so2, val/acc_best=0.590199
# ------------------------------------------------------------------------
# python3 src/train.py \
#     experiment=hls4ml/deepsets_16 \
#     data.method=augmentation \
#     data.augmentation_scheme=so2 \
#     data.n_folds=5 \
#     data.features='[pt,eta,phi]' \
#     algorithm.aggregation=max \
#     algorithm.optimizer.lr=0.0035636161687470917 \
#     algorithm.optimizer.weight_decay=1e-07 \
#     algorithm.phi_layers='[64,64,64]' \
#     algorithm.rho_layers='[32,16]' \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/train/hls4ml/pt_eta_phi/deepsets/nconst16/aug_so2

# ------------------------------------------------------------------------
# deepsets  nconst=16  aug_so3  (method=augmentation, group=so3)
# best trial 132 of deepsets_nconst16_augmentation_so3, val/acc_best=0.556904
# ------------------------------------------------------------------------
# python3 src/train.py \
#     experiment=hls4ml/deepsets_16 \
#     data.method=augmentation \
#     data.augmentation_scheme=so3 \
#     data.n_folds=5 \
#     data.features='[pt,eta,phi]' \
#     algorithm.aggregation=max \
#     algorithm.optimizer.lr=0.0012844339579265124 \
#     algorithm.optimizer.weight_decay=0.0 \
#     algorithm.phi_layers='[64,64,64]' \
#     algorithm.rho_layers='[32]' \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/train/hls4ml/pt_eta_phi/deepsets/nconst16/aug_so3

# ------------------------------------------------------------------------
# deepsets  nconst=16  aug_lorentz  (method=augmentation, group=lorentz)
# best trial 176 of deepsets_nconst16_augmentation_lorentz, val/acc_best=0.586348
# ------------------------------------------------------------------------
# python3 src/train.py \
#     experiment=hls4ml/deepsets_16 \
#     data.method=augmentation \
#     data.augmentation_scheme=lorentz \
#     data.n_folds=5 \
#     data.features='[pt,eta,phi]' \
#     algorithm.aggregation=max \
#     algorithm.optimizer.lr=0.0028797256951782445 \
#     algorithm.optimizer.weight_decay=1e-05 \
#     algorithm.phi_layers='[64,64,64]' \
#     algorithm.rho_layers='[16]' \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/train/hls4ml/pt_eta_phi/deepsets/nconst16/aug_lorentz

# ------------------------------------------------------------------------
# deepsets  nconst=16  symm_so2  (method=symm_loss, group=so2)
# best trial 155 of deepsets_nconst16_symm_loss_so2, val/acc_best=0.587213
# ------------------------------------------------------------------------
# python3 src/train.py \
#     experiment=hls4ml/deepsets_16 \
#     data.method=symm_loss \
#     data.augmentation_scheme=so2 \
#     data.n_folds=5 \
#     data.features='[pt,eta,phi]' \
#     algorithm.aggregation=max \
#     algorithm.optimizer.lr=0.0020972711301736457 \
#     algorithm.optimizer.weight_decay=1e-05 \
#     algorithm.phi_layers='[64,64,64]' \
#     algorithm.rho_layers='[16]' \
#     algorithm.symm_loss_weight=0.0010122471451378336 \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/train/hls4ml/pt_eta_phi/deepsets/nconst16/symm_so2

# ------------------------------------------------------------------------
# deepsets  nconst=16  symm_so3  (method=symm_loss, group=so3)
# best trial 191 of deepsets_nconst16_symm_loss_so3, val/acc_best=0.589660
# ------------------------------------------------------------------------
# python3 src/train.py \
#     experiment=hls4ml/deepsets_16 \
#     data.method=symm_loss \
#     data.augmentation_scheme=so3 \
#     data.n_folds=5 \
#     data.features='[pt,eta,phi]' \
#     algorithm.aggregation=max \
#     algorithm.optimizer.lr=0.0012766042953431238 \
#     algorithm.optimizer.weight_decay=1e-07 \
#     algorithm.phi_layers='[64,64,64]' \
#     algorithm.rho_layers='[16]' \
#     algorithm.symm_loss_weight=0.02980403136612197 \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/train/hls4ml/pt_eta_phi/deepsets/nconst16/symm_so3

# ------------------------------------------------------------------------
# deepsets  nconst=16  symm_lorentz  (method=symm_loss, group=lorentz)
# best trial 124 of deepsets_nconst16_symm_loss_lorentz, val/acc_best=0.592869
# ------------------------------------------------------------------------
# python3 src/train.py \
#     experiment=hls4ml/deepsets_16 \
#     data.method=symm_loss \
#     data.augmentation_scheme=lorentz \
#     data.n_folds=5 \
#     data.features='[pt,eta,phi]' \
#     algorithm.aggregation=max \
#     algorithm.optimizer.lr=0.001338495422327022 \
#     algorithm.optimizer.weight_decay=0.0 \
#     algorithm.phi_layers='[64,64,64]' \
#     algorithm.rho_layers='[32]' \
#     algorithm.symm_loss_weight=0.03028178063305558 \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/train/hls4ml/pt_eta_phi/deepsets/nconst16/symm_lorentz

# ========================================================================
# deepsets  nconst=50
# ========================================================================

# ------------------------------------------------------------------------
# deepsets  nconst=50  baseline  (method=augmentation, group=none)
# best trial 137 of deepsets_nconst50_augmentation_none, val/acc_best=0.627834
# ------------------------------------------------------------------------
# python3 src/train.py \
#     experiment=hls4ml/deepsets_50 \
#     data.method=augmentation \
#     data.augmentation_scheme=none \
#     data.n_folds=5 \
#     data.features='[pt,eta,phi]' \
#     algorithm.aggregation=max \
#     algorithm.optimizer.lr=0.003117112503426223 \
#     algorithm.optimizer.weight_decay=1e-05 \
#     algorithm.phi_layers='[64,64,64]' \
#     algorithm.rho_layers='[64,32]' \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/train/hls4ml/pt_eta_phi/deepsets/nconst50/baseline

# ------------------------------------------------------------------------
# deepsets  nconst=50  aug_so2  (method=augmentation, group=so2)
# best trial 195 of deepsets_nconst50_augmentation_so2, val/acc_best=0.626352
# ------------------------------------------------------------------------
# python3 src/train.py \
#     experiment=hls4ml/deepsets_50 \
#     data.method=augmentation \
#     data.augmentation_scheme=so2 \
#     data.n_folds=5 \
#     data.features='[pt,eta,phi]' \
#     algorithm.aggregation=max \
#     algorithm.optimizer.lr=0.004519421465452711 \
#     algorithm.optimizer.weight_decay=1e-06 \
#     algorithm.phi_layers='[64,64,64]' \
#     algorithm.rho_layers='[64,32]' \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/train/hls4ml/pt_eta_phi/deepsets/nconst50/aug_so2

# ------------------------------------------------------------------------
# deepsets  nconst=50  aug_so3  (method=augmentation, group=so3)
# best trial 160 of deepsets_nconst50_augmentation_so3, val/acc_best=0.595153
# ------------------------------------------------------------------------
# python3 src/train.py \
#     experiment=hls4ml/deepsets_50 \
#     data.method=augmentation \
#     data.augmentation_scheme=so3 \
#     data.n_folds=5 \
#     data.features='[pt,eta,phi]' \
#     algorithm.aggregation=max \
#     algorithm.optimizer.lr=0.0036959878548142385 \
#     algorithm.optimizer.weight_decay=0.0 \
#     algorithm.phi_layers='[64,64,64]' \
#     algorithm.rho_layers='[32,32,16]' \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/train/hls4ml/pt_eta_phi/deepsets/nconst50/aug_so3

# ------------------------------------------------------------------------
# deepsets  nconst=50  aug_lorentz  (method=augmentation, group=lorentz)
# best trial 165 of deepsets_nconst50_augmentation_lorentz, val/acc_best=0.627741
# ------------------------------------------------------------------------
# python3 src/train.py \
#     experiment=hls4ml/deepsets_50 \
#     data.method=augmentation \
#     data.augmentation_scheme=lorentz \
#     data.n_folds=5 \
#     data.features='[pt,eta,phi]' \
#     algorithm.aggregation=max \
#     algorithm.optimizer.lr=0.004583147431009476 \
#     algorithm.optimizer.weight_decay=1e-07 \
#     algorithm.phi_layers='[64,64,64]' \
#     algorithm.rho_layers='[64,32]' \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/train/hls4ml/pt_eta_phi/deepsets/nconst50/aug_lorentz

# ------------------------------------------------------------------------
# deepsets  nconst=50  symm_so2  (method=symm_loss, group=so2)
# best trial 145 of deepsets_nconst50_symm_loss_so2, val/acc_best=0.629318
# ------------------------------------------------------------------------
# python3 src/train.py \
#     experiment=hls4ml/deepsets_50 \
#     data.method=symm_loss \
#     data.augmentation_scheme=so2 \
#     data.n_folds=5 \
#     data.features='[pt,eta,phi]' \
#     algorithm.aggregation=max \
#     algorithm.optimizer.lr=0.003805824352762615 \
#     algorithm.optimizer.weight_decay=1e-06 \
#     algorithm.phi_layers='[64,64,64]' \
#     algorithm.rho_layers='[32,16]' \
#     algorithm.symm_loss_weight=0.0014015250621893603 \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/train/hls4ml/pt_eta_phi/deepsets/nconst50/symm_so2

# ------------------------------------------------------------------------
# deepsets  nconst=50  symm_so3  (method=symm_loss, group=so3)
# best trial 156 of deepsets_nconst50_symm_loss_so3, val/acc_best=0.629315
# ------------------------------------------------------------------------
# python3 src/train.py \
#     experiment=hls4ml/deepsets_50 \
#     data.method=symm_loss \
#     data.augmentation_scheme=so3 \
#     data.n_folds=5 \
#     data.features='[pt,eta,phi]' \
#     algorithm.aggregation=max \
#     algorithm.optimizer.lr=0.0037220884421394146 \
#     algorithm.optimizer.weight_decay=1e-07 \
#     algorithm.phi_layers='[64,64,64]' \
#     algorithm.rho_layers='[64,32]' \
#     algorithm.symm_loss_weight=0.019050384325485552 \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/train/hls4ml/pt_eta_phi/deepsets/nconst50/symm_so3

# ------------------------------------------------------------------------
# deepsets  nconst=50  symm_lorentz  (method=symm_loss, group=lorentz)
# best trial 56 of deepsets_nconst50_symm_loss_lorentz, val/acc_best=0.630531
# ------------------------------------------------------------------------
# python3 src/train.py \
#     experiment=hls4ml/deepsets_50 \
#     data.method=symm_loss \
#     data.augmentation_scheme=lorentz \
#     data.n_folds=5 \
#     data.features='[pt,eta,phi]' \
#     algorithm.aggregation=max \
#     algorithm.optimizer.lr=0.00307264982061151 \
#     algorithm.optimizer.weight_decay=1e-06 \
#     algorithm.phi_layers='[64,64,64]' \
#     algorithm.rho_layers='[32,32,16]' \
#     algorithm.symm_loss_weight=0.0034178308686909644 \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/train/hls4ml/pt_eta_phi/deepsets/nconst50/symm_lorentz

# ========================================================================
# deepsets  nconst=150
# ========================================================================

# ------------------------------------------------------------------------
# deepsets  nconst=150  baseline  (method=augmentation, group=none)
# best trial 163 of deepsets_nconst150_augmentation_none, val/acc_best=0.595177
# ------------------------------------------------------------------------
# python3 src/train.py \
#     experiment=hls4ml/deepsets_150 \
#     data.method=augmentation \
#     data.augmentation_scheme=none \
#     data.n_folds=5 \
#     data.features='[pt,eta,phi]' \
#     algorithm.aggregation=max \
#     algorithm.optimizer.lr=0.009731768402345803 \
#     algorithm.optimizer.weight_decay=0.0 \
#     algorithm.phi_layers='[64,64,64]' \
#     algorithm.rho_layers='[16]' \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/train/hls4ml/pt_eta_phi/deepsets/nconst150/baseline

# ------------------------------------------------------------------------
# deepsets  nconst=150  aug_so2  (method=augmentation, group=so2)
# best trial 165 of deepsets_nconst150_augmentation_so2, val/acc_best=0.593022
# ------------------------------------------------------------------------
# python3 src/train.py \
#     experiment=hls4ml/deepsets_150 \
#     data.method=augmentation \
#     data.augmentation_scheme=so2 \
#     data.n_folds=5 \
#     data.features='[pt,eta,phi]' \
#     algorithm.aggregation=max \
#     algorithm.optimizer.lr=0.003419126147423987 \
#     algorithm.optimizer.weight_decay=1e-07 \
#     algorithm.phi_layers='[64,64,64]' \
#     algorithm.rho_layers='[64,32]' \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/train/hls4ml/pt_eta_phi/deepsets/nconst150/aug_so2

# ------------------------------------------------------------------------
# deepsets  nconst=150  aug_so3  (method=augmentation, group=so3)
# best trial 93 of deepsets_nconst150_augmentation_so3, val/acc_best=0.558985
# ------------------------------------------------------------------------
# python3 src/train.py \
#     experiment=hls4ml/deepsets_150 \
#     data.method=augmentation \
#     data.augmentation_scheme=so3 \
#     data.n_folds=5 \
#     data.features='[pt,eta,phi]' \
#     algorithm.aggregation=max \
#     algorithm.optimizer.lr=0.0041058526788415 \
#     algorithm.optimizer.weight_decay=0.0 \
#     algorithm.phi_layers='[128,64,32]' \
#     algorithm.rho_layers='[64,32]' \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/train/hls4ml/pt_eta_phi/deepsets/nconst150/aug_so3

# ------------------------------------------------------------------------
# deepsets  nconst=150  aug_lorentz  (method=augmentation, group=lorentz)
# best trial 196 of deepsets_nconst150_augmentation_lorentz, val/acc_best=0.586998
# ------------------------------------------------------------------------
# python3 src/train.py \
#     experiment=hls4ml/deepsets_150 \
#     data.method=augmentation \
#     data.augmentation_scheme=lorentz \
#     data.n_folds=5 \
#     data.features='[pt,eta,phi]' \
#     algorithm.aggregation=max \
#     algorithm.optimizer.lr=0.004750127636883272 \
#     algorithm.optimizer.weight_decay=1e-07 \
#     algorithm.phi_layers='[128,64,32]' \
#     algorithm.rho_layers='[32,16]' \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/train/hls4ml/pt_eta_phi/deepsets/nconst150/aug_lorentz

# ------------------------------------------------------------------------
# deepsets  nconst=150  symm_so2  (method=symm_loss, group=so2)
# best trial 138 of deepsets_nconst150_symm_loss_so2, val/acc_best=0.584788
# ------------------------------------------------------------------------
# python3 src/train.py \
#     experiment=hls4ml/deepsets_150 \
#     data.method=symm_loss \
#     data.augmentation_scheme=so2 \
#     data.n_folds=5 \
#     data.features='[pt,eta,phi]' \
#     algorithm.aggregation=max \
#     algorithm.optimizer.lr=0.005260525751423092 \
#     algorithm.optimizer.weight_decay=1e-07 \
#     algorithm.phi_layers='[64,64,32]' \
#     algorithm.rho_layers='[32]' \
#     algorithm.symm_loss_weight=0.0060993745462393605 \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/train/hls4ml/pt_eta_phi/deepsets/nconst150/symm_so2

# ------------------------------------------------------------------------
# deepsets  nconst=150  symm_so3  (method=symm_loss, group=so3)
# best trial 154 of deepsets_nconst150_symm_loss_so3, val/acc_best=0.605777
# ------------------------------------------------------------------------
# python3 src/train.py \
#     experiment=hls4ml/deepsets_150 \
#     data.method=symm_loss \
#     data.augmentation_scheme=so3 \
#     data.n_folds=5 \
#     data.features='[pt,eta,phi]' \
#     algorithm.aggregation=max \
#     algorithm.optimizer.lr=0.006443991201804629 \
#     algorithm.optimizer.weight_decay=0.0 \
#     algorithm.phi_layers='[64,64,64]' \
#     algorithm.rho_layers='[32,16]' \
#     algorithm.symm_loss_weight=0.07157796008807137 \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/train/hls4ml/pt_eta_phi/deepsets/nconst150/symm_so3

# ------------------------------------------------------------------------
# deepsets  nconst=150  symm_lorentz  (method=symm_loss, group=lorentz)
# best trial 186 of deepsets_nconst150_symm_loss_lorentz, val/acc_best=0.589749
# ------------------------------------------------------------------------
# python3 src/train.py \
#     experiment=hls4ml/deepsets_150 \
#     data.method=symm_loss \
#     data.augmentation_scheme=lorentz \
#     data.n_folds=5 \
#     data.features='[pt,eta,phi]' \
#     algorithm.aggregation=max \
#     algorithm.optimizer.lr=0.0038468654828647305 \
#     algorithm.optimizer.weight_decay=1e-06 \
#     algorithm.phi_layers='[64,64,64]' \
#     algorithm.rho_layers='[64,32]' \
#     algorithm.symm_loss_weight=0.011887892290199783 \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/train/hls4ml/pt_eta_phi/deepsets/nconst150/symm_lorentz
