#!/usr/bin/env bash
# ========================================================================
# MLP TRAINING COMMANDS
# ========================================================================
# One block per (constituent count, regime): 3 x 7 = 21 train runs. Generated
# by scripts/experiments/make_train_catalogues.py; regenerate rather than edit by hand.
# Feature set [pt, etarel, phirel]: studies from logs/optuna/hls4ml/pt_etarel_phirel/, runs under logs/{train,eval}/hls4ml/pt_etarel_phirel/.
# Each block carries the best of the first 200 completed trials, by trial
# number, of its own study under logs/optuna/hls4ml/pt_etarel_phirel/;
# a block whose study is missing carries none and says so (transferred ablation).
#
# Every command is commented out: copy one to run it by hand, or feed the whole
# file to scripts/experiments/submit_train.sh, which queues them on a GPU or slurm.

# ========================================================================
# mlp  nconst=16
# ========================================================================

# ------------------------------------------------------------------------
# mlp  nconst=16  baseline  (method=augmentation, group=none)
# best trial 186 of mlp_nconst16_augmentation_none, val/acc_best=0.698553
# ------------------------------------------------------------------------
# python3 src/train.py \
#     experiment=hls4ml/mlp_16 \
#     data.method=augmentation \
#     data.augmentation_scheme=none \
#     data.n_folds=5 \
#     data.features='[pt,etarel,phirel]' \
#     algorithm.batchnorm=True \
#     algorithm.hidden_layers='[128,64,32]' \
#     algorithm.optimizer.lr=0.007682544985558676 \
#     algorithm.optimizer.weight_decay=1e-07 \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/train/hls4ml/pt_etarel_phirel/mlp/nconst16/baseline

# ------------------------------------------------------------------------
# mlp  nconst=16  aug_so2  (method=augmentation, group=so2)
# best trial 186 of mlp_nconst16_augmentation_so2, val/acc_best=0.698484
# ------------------------------------------------------------------------
# python3 src/train.py \
#     experiment=hls4ml/mlp_16 \
#     data.method=augmentation \
#     data.augmentation_scheme=so2 \
#     data.n_folds=5 \
#     data.features='[pt,etarel,phirel]' \
#     algorithm.batchnorm=True \
#     algorithm.hidden_layers='[128,64,32]' \
#     algorithm.optimizer.lr=0.00794589290588075 \
#     algorithm.optimizer.weight_decay=1e-07 \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/train/hls4ml/pt_etarel_phirel/mlp/nconst16/aug_so2

# ------------------------------------------------------------------------
# mlp  nconst=16  aug_so3  (method=augmentation, group=so3)
# best trial 126 of mlp_nconst16_augmentation_so3, val/acc_best=0.659421
# ------------------------------------------------------------------------
# python3 src/train.py \
#     experiment=hls4ml/mlp_16 \
#     data.method=augmentation \
#     data.augmentation_scheme=so3 \
#     data.n_folds=5 \
#     data.features='[pt,etarel,phirel]' \
#     algorithm.batchnorm=True \
#     algorithm.hidden_layers='[256,128,64]' \
#     algorithm.optimizer.lr=0.007913615021642608 \
#     algorithm.optimizer.weight_decay=0.0 \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/train/hls4ml/pt_etarel_phirel/mlp/nconst16/aug_so3

# ------------------------------------------------------------------------
# mlp  nconst=16  aug_lorentz  (method=augmentation, group=lorentz)
# best trial 125 of mlp_nconst16_augmentation_lorentz, val/acc_best=0.698649
# ------------------------------------------------------------------------
# python3 src/train.py \
#     experiment=hls4ml/mlp_16 \
#     data.method=augmentation \
#     data.augmentation_scheme=lorentz \
#     data.n_folds=5 \
#     data.features='[pt,etarel,phirel]' \
#     algorithm.batchnorm=True \
#     algorithm.hidden_layers='[128,64,32]' \
#     algorithm.optimizer.lr=0.006310028786947866 \
#     algorithm.optimizer.weight_decay=1e-06 \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/train/hls4ml/pt_etarel_phirel/mlp/nconst16/aug_lorentz

# ------------------------------------------------------------------------
# mlp  nconst=16  symm_so2  (method=symm_loss, group=so2)
# best trial 199 of mlp_nconst16_symm_loss_so2, val/acc_best=0.698641
# ------------------------------------------------------------------------
# python3 src/train.py \
#     experiment=hls4ml/mlp_16 \
#     data.method=symm_loss \
#     data.augmentation_scheme=so2 \
#     data.n_folds=5 \
#     data.features='[pt,etarel,phirel]' \
#     algorithm.batchnorm=True \
#     algorithm.hidden_layers='[128,64,32]' \
#     algorithm.optimizer.lr=0.007175150460645108 \
#     algorithm.optimizer.weight_decay=1e-07 \
#     algorithm.symm_loss_weight=0.03298367057305323 \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/train/hls4ml/pt_etarel_phirel/mlp/nconst16/symm_so2

# ------------------------------------------------------------------------
# mlp  nconst=16  symm_so3  (method=symm_loss, group=so3)
# best trial 117 of mlp_nconst16_symm_loss_so3, val/acc_best=0.697674
# ------------------------------------------------------------------------
# python3 src/train.py \
#     experiment=hls4ml/mlp_16 \
#     data.method=symm_loss \
#     data.augmentation_scheme=so3 \
#     data.n_folds=5 \
#     data.features='[pt,etarel,phirel]' \
#     algorithm.batchnorm=False \
#     algorithm.hidden_layers='[256,128,64]' \
#     algorithm.optimizer.lr=0.0015454604738159255 \
#     algorithm.optimizer.weight_decay=1e-05 \
#     algorithm.symm_loss_weight=0.018643183980897093 \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/train/hls4ml/pt_etarel_phirel/mlp/nconst16/symm_so3

# ------------------------------------------------------------------------
# mlp  nconst=16  symm_lorentz  (method=symm_loss, group=lorentz)
# best trial 162 of mlp_nconst16_symm_loss_lorentz, val/acc_best=0.699115
# ------------------------------------------------------------------------
# python3 src/train.py \
#     experiment=hls4ml/mlp_16 \
#     data.method=symm_loss \
#     data.augmentation_scheme=lorentz \
#     data.n_folds=5 \
#     data.features='[pt,etarel,phirel]' \
#     algorithm.batchnorm=True \
#     algorithm.hidden_layers='[128,64,32]' \
#     algorithm.optimizer.lr=0.007722417873633797 \
#     algorithm.optimizer.weight_decay=0.0 \
#     algorithm.symm_loss_weight=0.004824026825737415 \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/train/hls4ml/pt_etarel_phirel/mlp/nconst16/symm_lorentz

# ========================================================================
# mlp  nconst=50
# ========================================================================

# ------------------------------------------------------------------------
# mlp  nconst=50  baseline  (method=augmentation, group=none)
# best trial 134 of mlp_nconst50_augmentation_none, val/acc_best=0.779216
# ------------------------------------------------------------------------
# python3 src/train.py \
#     experiment=hls4ml/mlp_50 \
#     data.method=augmentation \
#     data.augmentation_scheme=none \
#     data.n_folds=5 \
#     data.features='[pt,etarel,phirel]' \
#     algorithm.batchnorm=True \
#     algorithm.hidden_layers='[128,64,32]' \
#     algorithm.optimizer.lr=0.006231966644198026 \
#     algorithm.optimizer.weight_decay=0.0 \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/train/hls4ml/pt_etarel_phirel/mlp/nconst50/baseline

# ------------------------------------------------------------------------
# mlp  nconst=50  aug_so2  (method=augmentation, group=so2)
# best trial 127 of mlp_nconst50_augmentation_so2, val/acc_best=0.779645
# ------------------------------------------------------------------------
# python3 src/train.py \
#     experiment=hls4ml/mlp_50 \
#     data.method=augmentation \
#     data.augmentation_scheme=so2 \
#     data.n_folds=5 \
#     data.features='[pt,etarel,phirel]' \
#     algorithm.batchnorm=True \
#     algorithm.hidden_layers='[128,64,32]' \
#     algorithm.optimizer.lr=0.009017252123651582 \
#     algorithm.optimizer.weight_decay=0.0 \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/train/hls4ml/pt_etarel_phirel/mlp/nconst50/aug_so2

# ------------------------------------------------------------------------
# mlp  nconst=50  aug_so3  (method=augmentation, group=so3)
# best trial 175 of mlp_nconst50_augmentation_so3, val/acc_best=0.768043
# ------------------------------------------------------------------------
# python3 src/train.py \
#     experiment=hls4ml/mlp_50 \
#     data.method=augmentation \
#     data.augmentation_scheme=so3 \
#     data.n_folds=5 \
#     data.features='[pt,etarel,phirel]' \
#     algorithm.batchnorm=True \
#     algorithm.hidden_layers='[256,128,64]' \
#     algorithm.optimizer.lr=0.006481548063142111 \
#     algorithm.optimizer.weight_decay=1e-07 \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/train/hls4ml/pt_etarel_phirel/mlp/nconst50/aug_so3

# ------------------------------------------------------------------------
# mlp  nconst=50  aug_lorentz  (method=augmentation, group=lorentz)
# best trial 189 of mlp_nconst50_augmentation_lorentz, val/acc_best=0.779557
# ------------------------------------------------------------------------
# python3 src/train.py \
#     experiment=hls4ml/mlp_50 \
#     data.method=augmentation \
#     data.augmentation_scheme=lorentz \
#     data.n_folds=5 \
#     data.features='[pt,etarel,phirel]' \
#     algorithm.batchnorm=True \
#     algorithm.hidden_layers='[128,64,32]' \
#     algorithm.optimizer.lr=0.004289022759823206 \
#     algorithm.optimizer.weight_decay=0.0 \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/train/hls4ml/pt_etarel_phirel/mlp/nconst50/aug_lorentz

# ------------------------------------------------------------------------
# mlp  nconst=50  symm_so2  (method=symm_loss, group=so2)
# best trial 193 of mlp_nconst50_symm_loss_so2, val/acc_best=0.780988
# ------------------------------------------------------------------------
# python3 src/train.py \
#     experiment=hls4ml/mlp_50 \
#     data.method=symm_loss \
#     data.augmentation_scheme=so2 \
#     data.n_folds=5 \
#     data.features='[pt,etarel,phirel]' \
#     algorithm.batchnorm=True \
#     algorithm.hidden_layers='[256,128,64]' \
#     algorithm.optimizer.lr=0.005662403116929158 \
#     algorithm.optimizer.weight_decay=1e-07 \
#     algorithm.symm_loss_weight=6.132053685787651 \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/train/hls4ml/pt_etarel_phirel/mlp/nconst50/symm_so2

# ------------------------------------------------------------------------
# mlp  nconst=50  symm_so3  (method=symm_loss, group=so3)
# best trial 194 of mlp_nconst50_symm_loss_so3, val/acc_best=0.780824
# ------------------------------------------------------------------------
# python3 src/train.py \
#     experiment=hls4ml/mlp_50 \
#     data.method=symm_loss \
#     data.augmentation_scheme=so3 \
#     data.n_folds=5 \
#     data.features='[pt,etarel,phirel]' \
#     algorithm.batchnorm=False \
#     algorithm.hidden_layers='[256,128,64]' \
#     algorithm.optimizer.lr=0.0023407740733241253 \
#     algorithm.optimizer.weight_decay=1e-05 \
#     algorithm.symm_loss_weight=0.08318507653571929 \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/train/hls4ml/pt_etarel_phirel/mlp/nconst50/symm_so3

# ------------------------------------------------------------------------
# mlp  nconst=50  symm_lorentz  (method=symm_loss, group=lorentz)
# best trial 149 of mlp_nconst50_symm_loss_lorentz, val/acc_best=0.782956
# ------------------------------------------------------------------------
# python3 src/train.py \
#     experiment=hls4ml/mlp_50 \
#     data.method=symm_loss \
#     data.augmentation_scheme=lorentz \
#     data.n_folds=5 \
#     data.features='[pt,etarel,phirel]' \
#     algorithm.batchnorm=True \
#     algorithm.hidden_layers='[256,128,64]' \
#     algorithm.optimizer.lr=0.00869430576886208 \
#     algorithm.optimizer.weight_decay=0.0 \
#     algorithm.symm_loss_weight=2.139471643104729 \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/train/hls4ml/pt_etarel_phirel/mlp/nconst50/symm_lorentz

# ========================================================================
# mlp  nconst=150
# ========================================================================

# ------------------------------------------------------------------------
# mlp  nconst=150  baseline  (method=augmentation, group=none)
# best trial 55 of mlp_nconst150_augmentation_none, val/acc_best=0.773190
# ------------------------------------------------------------------------
# python3 src/train.py \
#     experiment=hls4ml/mlp_150 \
#     data.method=augmentation \
#     data.augmentation_scheme=none \
#     data.n_folds=5 \
#     data.features='[pt,etarel,phirel]' \
#     algorithm.batchnorm=True \
#     algorithm.hidden_layers='[128,64,32]' \
#     algorithm.optimizer.lr=0.00137137525459169 \
#     algorithm.optimizer.weight_decay=0.0001 \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/train/hls4ml/pt_etarel_phirel/mlp/nconst150/baseline

# ------------------------------------------------------------------------
# mlp  nconst=150  aug_so2  (method=augmentation, group=so2)
# best trial 109 of mlp_nconst150_augmentation_so2, val/acc_best=0.773370
# ------------------------------------------------------------------------
# python3 src/train.py \
#     experiment=hls4ml/mlp_150 \
#     data.method=augmentation \
#     data.augmentation_scheme=so2 \
#     data.n_folds=5 \
#     data.features='[pt,etarel,phirel]' \
#     algorithm.batchnorm=True \
#     algorithm.hidden_layers='[128,64,32]' \
#     algorithm.optimizer.lr=0.0092175637234335 \
#     algorithm.optimizer.weight_decay=0.0 \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/train/hls4ml/pt_etarel_phirel/mlp/nconst150/aug_so2

# ------------------------------------------------------------------------
# mlp  nconst=150  aug_so3  (method=augmentation, group=so3)
# best trial 129 of mlp_nconst150_augmentation_so3, val/acc_best=0.770199
# ------------------------------------------------------------------------
# python3 src/train.py \
#     experiment=hls4ml/mlp_150 \
#     data.method=augmentation \
#     data.augmentation_scheme=so3 \
#     data.n_folds=5 \
#     data.features='[pt,etarel,phirel]' \
#     algorithm.batchnorm=True \
#     algorithm.hidden_layers='[256,128,64]' \
#     algorithm.optimizer.lr=0.007282876690143649 \
#     algorithm.optimizer.weight_decay=0.0 \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/train/hls4ml/pt_etarel_phirel/mlp/nconst150/aug_so3

# ------------------------------------------------------------------------
# mlp  nconst=150  aug_lorentz  (method=augmentation, group=lorentz)
# best trial 138 of mlp_nconst150_augmentation_lorentz, val/acc_best=0.775584
# ------------------------------------------------------------------------
# python3 src/train.py \
#     experiment=hls4ml/mlp_150 \
#     data.method=augmentation \
#     data.augmentation_scheme=lorentz \
#     data.n_folds=5 \
#     data.features='[pt,etarel,phirel]' \
#     algorithm.batchnorm=True \
#     algorithm.hidden_layers='[256,128,64]' \
#     algorithm.optimizer.lr=0.0047529725000286065 \
#     algorithm.optimizer.weight_decay=0.0001 \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/train/hls4ml/pt_etarel_phirel/mlp/nconst150/aug_lorentz

# ------------------------------------------------------------------------
# mlp  nconst=150  symm_so2  (method=symm_loss, group=so2)
# best trial 183 of mlp_nconst150_symm_loss_so2, val/acc_best=0.774269
# ------------------------------------------------------------------------
# python3 src/train.py \
#     experiment=hls4ml/mlp_150 \
#     data.method=symm_loss \
#     data.augmentation_scheme=so2 \
#     data.n_folds=5 \
#     data.features='[pt,etarel,phirel]' \
#     algorithm.batchnorm=True \
#     algorithm.hidden_layers='[128,64,32]' \
#     algorithm.optimizer.lr=0.008536844419300442 \
#     algorithm.optimizer.weight_decay=0.0 \
#     algorithm.symm_loss_weight=2.246072172559618 \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/train/hls4ml/pt_etarel_phirel/mlp/nconst150/symm_so2

# ------------------------------------------------------------------------
# mlp  nconst=150  symm_so3  (method=symm_loss, group=so3)
# best trial 194 of mlp_nconst150_symm_loss_so3, val/acc_best=0.777576
# ------------------------------------------------------------------------
# python3 src/train.py \
#     experiment=hls4ml/mlp_150 \
#     data.method=symm_loss \
#     data.augmentation_scheme=so3 \
#     data.n_folds=5 \
#     data.features='[pt,etarel,phirel]' \
#     algorithm.batchnorm=False \
#     algorithm.hidden_layers='[256,128,64]' \
#     algorithm.optimizer.lr=0.00206013740148708 \
#     algorithm.optimizer.weight_decay=0.0 \
#     algorithm.symm_loss_weight=0.12430073588125298 \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/train/hls4ml/pt_etarel_phirel/mlp/nconst150/symm_so3

# ------------------------------------------------------------------------
# mlp  nconst=150  symm_lorentz  (method=symm_loss, group=lorentz)
# best trial 166 of mlp_nconst150_symm_loss_lorentz, val/acc_best=0.779575
# ------------------------------------------------------------------------
# python3 src/train.py \
#     experiment=hls4ml/mlp_150 \
#     data.method=symm_loss \
#     data.augmentation_scheme=lorentz \
#     data.n_folds=5 \
#     data.features='[pt,etarel,phirel]' \
#     algorithm.batchnorm=True \
#     algorithm.hidden_layers='[256,128,64]' \
#     algorithm.optimizer.lr=0.007663833559947995 \
#     algorithm.optimizer.weight_decay=0.0 \
#     algorithm.symm_loss_weight=3.659734149884093 \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/train/hls4ml/pt_etarel_phirel/mlp/nconst150/symm_lorentz
