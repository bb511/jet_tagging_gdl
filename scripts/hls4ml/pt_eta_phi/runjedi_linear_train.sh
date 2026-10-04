#!/usr/bin/env bash
# ========================================================================
# JEDI_LINEAR TRAINING COMMANDS
# ========================================================================
# One block per (constituent count, regime): 3 x 7 = 21 train runs. Generated
# by scripts/optuna/make_train_catalogues.py; regenerate rather than edit by hand.
# Feature set [pt, eta, phi]: studies from logs/optuna/hls4ml/pt_eta_phi/, runs under logs/{train,eval}/hls4ml/pt_eta_phi/.
# Each block carries the best of the first 200 completed trials, by trial
# number, of its own study under logs/optuna/hls4ml/pt_eta_phi/;
# a block whose study is missing carries none and says so (transferred ablation).
#
# Every command is commented out: copy one to run it by hand, or feed the whole
# file to scripts/cluster/submit_train.sh, which queues them on a GPU or slurm.

# ========================================================================
# jedi_linear  nconst=16
# ========================================================================

# ------------------------------------------------------------------------
# jedi_linear  nconst=16  baseline  (method=augmentation, group=none)
# best trial 190 of jedi_linear_nconst16_augmentation_none, val/acc_best=0.611073
# ------------------------------------------------------------------------
# python3 src/train.py \
#     experiment=hls4ml/jedi_linear_16 \
#     data.method=augmentation \
#     data.augmentation_scheme=none \
#     data.n_folds=5 \
#     data.features='[pt,eta,phi]' \
#     algorithm.aggregation=max \
#     algorithm.head_layers='[64,32]' \
#     algorithm.hidden=128 \
#     algorithm.optimizer.lr=0.0029566314635514753 \
#     algorithm.optimizer.weight_decay=1e-06 \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/train/hls4ml/pt_eta_phi/jedi_linear/nconst16/baseline

# ------------------------------------------------------------------------
# jedi_linear  nconst=16  aug_so2  (method=augmentation, group=so2)
# best trial 145 of jedi_linear_nconst16_augmentation_so2, val/acc_best=0.613066
# ------------------------------------------------------------------------
# python3 src/train.py \
#     experiment=hls4ml/jedi_linear_16 \
#     data.method=augmentation \
#     data.augmentation_scheme=so2 \
#     data.n_folds=5 \
#     data.features='[pt,eta,phi]' \
#     algorithm.aggregation=max \
#     algorithm.head_layers='[32]' \
#     algorithm.hidden=128 \
#     algorithm.optimizer.lr=0.00460258224218509 \
#     algorithm.optimizer.weight_decay=0.0 \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/train/hls4ml/pt_eta_phi/jedi_linear/nconst16/aug_so2

# ------------------------------------------------------------------------
# jedi_linear  nconst=16  aug_so3  (method=augmentation, group=so3)
# best trial 133 of jedi_linear_nconst16_augmentation_so3, val/acc_best=0.581248
# ------------------------------------------------------------------------
# python3 src/train.py \
#     experiment=hls4ml/jedi_linear_16 \
#     data.method=augmentation \
#     data.augmentation_scheme=so3 \
#     data.n_folds=5 \
#     data.features='[pt,eta,phi]' \
#     algorithm.aggregation=max \
#     algorithm.head_layers='[64,32,16]' \
#     algorithm.hidden=64 \
#     algorithm.optimizer.lr=0.006816580984797155 \
#     algorithm.optimizer.weight_decay=1e-06 \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/train/hls4ml/pt_eta_phi/jedi_linear/nconst16/aug_so3

# ------------------------------------------------------------------------
# jedi_linear  nconst=16  aug_lorentz  (method=augmentation, group=lorentz)
# best trial 192 of jedi_linear_nconst16_augmentation_lorentz, val/acc_best=0.608258
# ------------------------------------------------------------------------
# python3 src/train.py \
#     experiment=hls4ml/jedi_linear_16 \
#     data.method=augmentation \
#     data.augmentation_scheme=lorentz \
#     data.n_folds=5 \
#     data.features='[pt,eta,phi]' \
#     algorithm.aggregation=max \
#     algorithm.head_layers='[64,32]' \
#     algorithm.hidden=128 \
#     algorithm.optimizer.lr=0.0063110165782613425 \
#     algorithm.optimizer.weight_decay=1e-07 \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/train/hls4ml/pt_eta_phi/jedi_linear/nconst16/aug_lorentz

# ------------------------------------------------------------------------
# jedi_linear  nconst=16  symm_so2  (method=symm_loss, group=so2)
# best trial 189 of jedi_linear_nconst16_symm_loss_so2, val/acc_best=0.613623
# ------------------------------------------------------------------------
# python3 src/train.py \
#     experiment=hls4ml/jedi_linear_16 \
#     data.method=symm_loss \
#     data.augmentation_scheme=so2 \
#     data.n_folds=5 \
#     data.features='[pt,eta,phi]' \
#     algorithm.aggregation=max \
#     algorithm.head_layers='[64,32]' \
#     algorithm.hidden=64 \
#     algorithm.optimizer.lr=0.006155494083481776 \
#     algorithm.optimizer.weight_decay=1e-05 \
#     algorithm.symm_loss_weight=0.02540550667489172 \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/train/hls4ml/pt_eta_phi/jedi_linear/nconst16/symm_so2

# ------------------------------------------------------------------------
# jedi_linear  nconst=16  symm_so3  (method=symm_loss, group=so3)
# best trial 166 of jedi_linear_nconst16_symm_loss_so3, val/acc_best=0.612435
# ------------------------------------------------------------------------
# python3 src/train.py \
#     experiment=hls4ml/jedi_linear_16 \
#     data.method=symm_loss \
#     data.augmentation_scheme=so3 \
#     data.n_folds=5 \
#     data.features='[pt,eta,phi]' \
#     algorithm.aggregation=max \
#     algorithm.head_layers='[32]' \
#     algorithm.hidden=128 \
#     algorithm.optimizer.lr=0.00498885362814374 \
#     algorithm.optimizer.weight_decay=0.0 \
#     algorithm.symm_loss_weight=0.016168870373836887 \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/train/hls4ml/pt_eta_phi/jedi_linear/nconst16/symm_so3

# ------------------------------------------------------------------------
# jedi_linear  nconst=16  symm_lorentz  (method=symm_loss, group=lorentz)
# best trial 108 of jedi_linear_nconst16_symm_loss_lorentz, val/acc_best=0.610123
# ------------------------------------------------------------------------
# python3 src/train.py \
#     experiment=hls4ml/jedi_linear_16 \
#     data.method=symm_loss \
#     data.augmentation_scheme=lorentz \
#     data.n_folds=5 \
#     data.features='[pt,eta,phi]' \
#     algorithm.aggregation=max \
#     algorithm.head_layers='[32]' \
#     algorithm.hidden=128 \
#     algorithm.optimizer.lr=0.004930675966609905 \
#     algorithm.optimizer.weight_decay=1e-06 \
#     algorithm.symm_loss_weight=0.001697213416215715 \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/train/hls4ml/pt_eta_phi/jedi_linear/nconst16/symm_lorentz

# ========================================================================
# jedi_linear  nconst=50
# ========================================================================

# ------------------------------------------------------------------------
# jedi_linear  nconst=50  baseline  (method=augmentation, group=none)
# best trial 192 of jedi_linear_nconst50_augmentation_none, val/acc_best=0.643541
# ------------------------------------------------------------------------
# python3 src/train.py \
#     experiment=hls4ml/jedi_linear_50 \
#     data.method=augmentation \
#     data.augmentation_scheme=none \
#     data.n_folds=5 \
#     data.features='[pt,eta,phi]' \
#     algorithm.aggregation=max \
#     algorithm.head_layers='[32]' \
#     algorithm.hidden=128 \
#     algorithm.optimizer.lr=0.003466103533722049 \
#     algorithm.optimizer.weight_decay=1e-07 \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/train/hls4ml/pt_eta_phi/jedi_linear/nconst50/baseline

# ------------------------------------------------------------------------
# jedi_linear  nconst=50  aug_so2  (method=augmentation, group=so2)
# best trial 146 of jedi_linear_nconst50_augmentation_so2, val/acc_best=0.643316
# ------------------------------------------------------------------------
# python3 src/train.py \
#     experiment=hls4ml/jedi_linear_50 \
#     data.method=augmentation \
#     data.augmentation_scheme=so2 \
#     data.n_folds=5 \
#     data.features='[pt,eta,phi]' \
#     algorithm.aggregation=max \
#     algorithm.head_layers='[32]' \
#     algorithm.hidden=128 \
#     algorithm.optimizer.lr=0.004699806669531636 \
#     algorithm.optimizer.weight_decay=1e-07 \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/train/hls4ml/pt_eta_phi/jedi_linear/nconst50/aug_so2

# ------------------------------------------------------------------------
# jedi_linear  nconst=50  aug_so3  (method=augmentation, group=so3)
# best trial 172 of jedi_linear_nconst50_augmentation_so3, val/acc_best=0.611292
# ------------------------------------------------------------------------
# python3 src/train.py \
#     experiment=hls4ml/jedi_linear_50 \
#     data.method=augmentation \
#     data.augmentation_scheme=so3 \
#     data.n_folds=5 \
#     data.features='[pt,eta,phi]' \
#     algorithm.aggregation=max \
#     algorithm.head_layers='[32,32,16]' \
#     algorithm.hidden=128 \
#     algorithm.optimizer.lr=0.005020735573952417 \
#     algorithm.optimizer.weight_decay=1e-06 \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/train/hls4ml/pt_eta_phi/jedi_linear/nconst50/aug_so3

# ------------------------------------------------------------------------
# jedi_linear  nconst=50  aug_lorentz  (method=augmentation, group=lorentz)
# best trial 194 of jedi_linear_nconst50_augmentation_lorentz, val/acc_best=0.644603
# ------------------------------------------------------------------------
# python3 src/train.py \
#     experiment=hls4ml/jedi_linear_50 \
#     data.method=augmentation \
#     data.augmentation_scheme=lorentz \
#     data.n_folds=5 \
#     data.features='[pt,eta,phi]' \
#     algorithm.aggregation=max \
#     algorithm.head_layers='[64,32]' \
#     algorithm.hidden=128 \
#     algorithm.optimizer.lr=0.004819756215793575 \
#     algorithm.optimizer.weight_decay=1e-06 \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/train/hls4ml/pt_eta_phi/jedi_linear/nconst50/aug_lorentz

# ------------------------------------------------------------------------
# jedi_linear  nconst=50  symm_so2  (method=symm_loss, group=so2)
# best trial 175 of jedi_linear_nconst50_symm_loss_so2, val/acc_best=0.644566
# ------------------------------------------------------------------------
# python3 src/train.py \
#     experiment=hls4ml/jedi_linear_50 \
#     data.method=symm_loss \
#     data.augmentation_scheme=so2 \
#     data.n_folds=5 \
#     data.features='[pt,eta,phi]' \
#     algorithm.aggregation=max \
#     algorithm.head_layers='[32]' \
#     algorithm.hidden=128 \
#     algorithm.optimizer.lr=0.006386184932552444 \
#     algorithm.optimizer.weight_decay=0.0 \
#     algorithm.symm_loss_weight=0.02161905090642747 \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/train/hls4ml/pt_eta_phi/jedi_linear/nconst50/symm_so2

# ------------------------------------------------------------------------
# jedi_linear  nconst=50  symm_so3  (method=symm_loss, group=so3)
# best trial 155 of jedi_linear_nconst50_symm_loss_so3, val/acc_best=0.643761
# ------------------------------------------------------------------------
# python3 src/train.py \
#     experiment=hls4ml/jedi_linear_50 \
#     data.method=symm_loss \
#     data.augmentation_scheme=so3 \
#     data.n_folds=5 \
#     data.features='[pt,eta,phi]' \
#     algorithm.aggregation=max \
#     algorithm.head_layers='[64,32,16]' \
#     algorithm.hidden=64 \
#     algorithm.optimizer.lr=0.0053160030651031465 \
#     algorithm.optimizer.weight_decay=0.0 \
#     algorithm.symm_loss_weight=0.021126746483144613 \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/train/hls4ml/pt_eta_phi/jedi_linear/nconst50/symm_so3

# ------------------------------------------------------------------------
# jedi_linear  nconst=50  symm_lorentz  (method=symm_loss, group=lorentz)
# best trial 192 of jedi_linear_nconst50_symm_loss_lorentz, val/acc_best=0.645061
# ------------------------------------------------------------------------
# python3 src/train.py \
#     experiment=hls4ml/jedi_linear_50 \
#     data.method=symm_loss \
#     data.augmentation_scheme=lorentz \
#     data.n_folds=5 \
#     data.features='[pt,eta,phi]' \
#     algorithm.aggregation=max \
#     algorithm.head_layers='[64,32]' \
#     algorithm.hidden=64 \
#     algorithm.optimizer.lr=0.006935913494785694 \
#     algorithm.optimizer.weight_decay=0.0 \
#     algorithm.symm_loss_weight=0.005607471407290283 \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/train/hls4ml/pt_eta_phi/jedi_linear/nconst50/symm_lorentz

# ========================================================================
# jedi_linear  nconst=150
# ========================================================================

# ------------------------------------------------------------------------
# jedi_linear  nconst=150  baseline  (method=augmentation, group=none)
# best trial 103 of jedi_linear_nconst150_augmentation_none, val/acc_best=0.622244
# ------------------------------------------------------------------------
# python3 src/train.py \
#     experiment=hls4ml/jedi_linear_150 \
#     data.method=augmentation \
#     data.augmentation_scheme=none \
#     data.n_folds=5 \
#     data.features='[pt,eta,phi]' \
#     algorithm.aggregation=max \
#     algorithm.head_layers='[64,32]' \
#     algorithm.hidden=128 \
#     algorithm.optimizer.lr=0.007424914463299956 \
#     algorithm.optimizer.weight_decay=1e-07 \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/train/hls4ml/pt_eta_phi/jedi_linear/nconst150/baseline

# ------------------------------------------------------------------------
# jedi_linear  nconst=150  aug_so2  (method=augmentation, group=so2)
# best trial 116 of jedi_linear_nconst150_augmentation_so2, val/acc_best=0.622681
# ------------------------------------------------------------------------
# python3 src/train.py \
#     experiment=hls4ml/jedi_linear_150 \
#     data.method=augmentation \
#     data.augmentation_scheme=so2 \
#     data.n_folds=5 \
#     data.features='[pt,eta,phi]' \
#     algorithm.aggregation=max \
#     algorithm.head_layers='[32]' \
#     algorithm.hidden=128 \
#     algorithm.optimizer.lr=0.006082101050788741 \
#     algorithm.optimizer.weight_decay=1e-07 \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/train/hls4ml/pt_eta_phi/jedi_linear/nconst150/aug_so2

# ------------------------------------------------------------------------
# jedi_linear  nconst=150  aug_so3  (method=augmentation, group=so3)
# best trial 180 of jedi_linear_nconst150_augmentation_so3, val/acc_best=0.590386
# ------------------------------------------------------------------------
# python3 src/train.py \
#     experiment=hls4ml/jedi_linear_150 \
#     data.method=augmentation \
#     data.augmentation_scheme=so3 \
#     data.n_folds=5 \
#     data.features='[pt,eta,phi]' \
#     algorithm.aggregation=max \
#     algorithm.head_layers='[64,32]' \
#     algorithm.hidden=128 \
#     algorithm.optimizer.lr=0.003789065657260455 \
#     algorithm.optimizer.weight_decay=1e-07 \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/train/hls4ml/pt_eta_phi/jedi_linear/nconst150/aug_so3

# ------------------------------------------------------------------------
# jedi_linear  nconst=150  aug_lorentz  (method=augmentation, group=lorentz)
# best trial 188 of jedi_linear_nconst150_augmentation_lorentz, val/acc_best=0.621742
# ------------------------------------------------------------------------
# python3 src/train.py \
#     experiment=hls4ml/jedi_linear_150 \
#     data.method=augmentation \
#     data.augmentation_scheme=lorentz \
#     data.n_folds=5 \
#     data.features='[pt,eta,phi]' \
#     algorithm.aggregation=max \
#     algorithm.head_layers='[64,32]' \
#     algorithm.hidden=128 \
#     algorithm.optimizer.lr=0.005130453270204083 \
#     algorithm.optimizer.weight_decay=0.0 \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/train/hls4ml/pt_eta_phi/jedi_linear/nconst150/aug_lorentz

# ------------------------------------------------------------------------
# jedi_linear  nconst=150  symm_so2  (method=symm_loss, group=so2)
# best trial 125 of jedi_linear_nconst150_symm_loss_so2, val/acc_best=0.621645
# ------------------------------------------------------------------------
# python3 src/train.py \
#     experiment=hls4ml/jedi_linear_150 \
#     data.method=symm_loss \
#     data.augmentation_scheme=so2 \
#     data.n_folds=5 \
#     data.features='[pt,eta,phi]' \
#     algorithm.aggregation=max \
#     algorithm.head_layers='[32]' \
#     algorithm.hidden=128 \
#     algorithm.optimizer.lr=0.0056595054069656665 \
#     algorithm.optimizer.weight_decay=1e-06 \
#     algorithm.symm_loss_weight=0.024308588844099886 \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/train/hls4ml/pt_eta_phi/jedi_linear/nconst150/symm_so2

# ------------------------------------------------------------------------
# jedi_linear  nconst=150  symm_so3  (method=symm_loss, group=so3)
# best trial 148 of jedi_linear_nconst150_symm_loss_so3, val/acc_best=0.629238
# ------------------------------------------------------------------------
# python3 src/train.py \
#     experiment=hls4ml/jedi_linear_150 \
#     data.method=symm_loss \
#     data.augmentation_scheme=so3 \
#     data.n_folds=5 \
#     data.features='[pt,eta,phi]' \
#     algorithm.aggregation=max \
#     algorithm.head_layers='[64,32]' \
#     algorithm.hidden=64 \
#     algorithm.optimizer.lr=0.006486248363502042 \
#     algorithm.optimizer.weight_decay=1e-06 \
#     algorithm.symm_loss_weight=0.012605039886592164 \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/train/hls4ml/pt_eta_phi/jedi_linear/nconst150/symm_so3

# ------------------------------------------------------------------------
# jedi_linear  nconst=150  symm_lorentz  (method=symm_loss, group=lorentz)
# best trial 122 of jedi_linear_nconst150_symm_loss_lorentz, val/acc_best=0.626333
# ------------------------------------------------------------------------
# python3 src/train.py \
#     experiment=hls4ml/jedi_linear_150 \
#     data.method=symm_loss \
#     data.augmentation_scheme=lorentz \
#     data.n_folds=5 \
#     data.features='[pt,eta,phi]' \
#     algorithm.aggregation=max \
#     algorithm.head_layers='[64,32]' \
#     algorithm.hidden=64 \
#     algorithm.optimizer.lr=0.007124021497254926 \
#     algorithm.optimizer.weight_decay=1e-06 \
#     algorithm.symm_loss_weight=0.0010648567530979836 \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/train/hls4ml/pt_eta_phi/jedi_linear/nconst150/symm_lorentz
