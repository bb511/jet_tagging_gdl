#!/usr/bin/env bash
# ========================================================================
# JETFORMER TRAINING COMMANDS
# ========================================================================
# One block per (constituent count, regime): 3 x 7 = 21 train runs. Generated
# by scripts/optuna/make_train_catalogues.py; regenerate rather than edit by hand.
# Feature set [pt, etarel, phirel]: studies from logs/optuna/hls4ml/pt_etarel_phirel/, runs under logs/{train,eval}/hls4ml/pt_etarel_phirel/.
# Each block carries the best of the first 200 completed trials, by trial
# number, of its own study under logs/optuna/hls4ml/pt_etarel_phirel/;
# a block whose study is missing carries none and says so (transferred ablation).
#
# Every command is commented out: copy one to run it by hand, or feed the whole
# file to scripts/cluster/submit_train.sh, which queues them on a GPU or slurm.

# ========================================================================
# jetformer  nconst=16
# ========================================================================

# ------------------------------------------------------------------------
# jetformer  nconst=16  baseline  (method=augmentation, group=none)
# best trial 85 of jetformer_nconst16_augmentation_none, val/acc_best=0.720364
# ------------------------------------------------------------------------
# python3 src/train.py \
#     experiment=hls4ml/jetformer_16 \
#     data.method=augmentation \
#     data.augmentation_scheme=none \
#     data.n_folds=5 \
#     data.features='[pt,etarel,phirel]' \
#     algorithm.dropout=0.0 \
#     algorithm.embed_dim=48 \
#     algorithm.num_heads=4 \
#     algorithm.num_transformers=3 \
#     algorithm.optimizer.lr=0.003110876421281397 \
#     algorithm.optimizer.weight_decay=0.001 \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/train/hls4ml/pt_etarel_phirel/jetformer/nconst16/baseline

# ------------------------------------------------------------------------
# jetformer  nconst=16  aug_so2  (method=augmentation, group=so2)
# best trial 192 of jetformer_nconst16_augmentation_so2, val/acc_best=0.722465
# ------------------------------------------------------------------------
# python3 src/train.py \
#     experiment=hls4ml/jetformer_16 \
#     data.method=augmentation \
#     data.augmentation_scheme=so2 \
#     data.n_folds=5 \
#     data.features='[pt,etarel,phirel]' \
#     algorithm.dropout=0.0 \
#     algorithm.embed_dim=96 \
#     algorithm.num_heads=4 \
#     algorithm.num_transformers=4 \
#     algorithm.optimizer.lr=0.0031599126348917997 \
#     algorithm.optimizer.weight_decay=0.05 \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/train/hls4ml/pt_etarel_phirel/jetformer/nconst16/aug_so2

# ------------------------------------------------------------------------
# jetformer  nconst=16  aug_so3  (method=augmentation, group=so3)
# best trial 91 of jetformer_nconst16_augmentation_so3, val/acc_best=0.672750
# ------------------------------------------------------------------------
# python3 src/train.py \
#     experiment=hls4ml/jetformer_16 \
#     data.method=augmentation \
#     data.augmentation_scheme=so3 \
#     data.n_folds=5 \
#     data.features='[pt,etarel,phirel]' \
#     algorithm.dropout=0.0 \
#     algorithm.embed_dim=96 \
#     algorithm.num_heads=4 \
#     algorithm.num_transformers=4 \
#     algorithm.optimizer.lr=0.0019241140904616473 \
#     algorithm.optimizer.weight_decay=0.0 \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/train/hls4ml/pt_etarel_phirel/jetformer/nconst16/aug_so3

# ------------------------------------------------------------------------
# jetformer  nconst=16  aug_lorentz  (method=augmentation, group=lorentz)
# best trial 152 of jetformer_nconst16_augmentation_lorentz, val/acc_best=0.722702
# ------------------------------------------------------------------------
# python3 src/train.py \
#     experiment=hls4ml/jetformer_16 \
#     data.method=augmentation \
#     data.augmentation_scheme=lorentz \
#     data.n_folds=5 \
#     data.features='[pt,etarel,phirel]' \
#     algorithm.dropout=0.0 \
#     algorithm.embed_dim=96 \
#     algorithm.num_heads=4 \
#     algorithm.num_transformers=4 \
#     algorithm.optimizer.lr=0.0032932836393839084 \
#     algorithm.optimizer.weight_decay=0.05 \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/train/hls4ml/pt_etarel_phirel/jetformer/nconst16/aug_lorentz

# ------------------------------------------------------------------------
# jetformer  nconst=16  symm_so2  (method=symm_loss, group=so2)
# best trial 193 of jetformer_nconst16_symm_loss_so2, val/acc_best=0.720974
# ------------------------------------------------------------------------
# python3 src/train.py \
#     experiment=hls4ml/jetformer_16 \
#     data.method=symm_loss \
#     data.augmentation_scheme=so2 \
#     data.n_folds=5 \
#     data.features='[pt,etarel,phirel]' \
#     algorithm.dropout=0.0 \
#     algorithm.embed_dim=64 \
#     algorithm.num_heads=4 \
#     algorithm.num_transformers=4 \
#     algorithm.optimizer.lr=0.001146181168770123 \
#     algorithm.optimizer.weight_decay=0.0 \
#     algorithm.symm_loss_weight=0.005753458194355721 \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/train/hls4ml/pt_etarel_phirel/jetformer/nconst16/symm_so2

# ------------------------------------------------------------------------
# jetformer  nconst=16  symm_so3  (method=symm_loss, group=so3)
# best trial 192 of jetformer_nconst16_symm_loss_so3, val/acc_best=0.722358
# ------------------------------------------------------------------------
# python3 src/train.py \
#     experiment=hls4ml/jetformer_16 \
#     data.method=symm_loss \
#     data.augmentation_scheme=so3 \
#     data.n_folds=5 \
#     data.features='[pt,etarel,phirel]' \
#     algorithm.dropout=0.0 \
#     algorithm.embed_dim=96 \
#     algorithm.num_heads=4 \
#     algorithm.num_transformers=4 \
#     algorithm.optimizer.lr=0.0033493337205097538 \
#     algorithm.optimizer.weight_decay=0.01 \
#     algorithm.symm_loss_weight=0.021695158243074063 \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/train/hls4ml/pt_etarel_phirel/jetformer/nconst16/symm_so3

# ------------------------------------------------------------------------
# jetformer  nconst=16  symm_lorentz  (method=symm_loss, group=lorentz)
# best trial 123 of jetformer_nconst16_symm_loss_lorentz, val/acc_best=0.721239
# ------------------------------------------------------------------------
# python3 src/train.py \
#     experiment=hls4ml/jetformer_16 \
#     data.method=symm_loss \
#     data.augmentation_scheme=lorentz \
#     data.n_folds=5 \
#     data.features='[pt,etarel,phirel]' \
#     algorithm.dropout=0.0 \
#     algorithm.embed_dim=64 \
#     algorithm.num_heads=4 \
#     algorithm.num_transformers=4 \
#     algorithm.optimizer.lr=0.0010248651882056295 \
#     algorithm.optimizer.weight_decay=0.05 \
#     algorithm.symm_loss_weight=0.0015652968286632564 \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/train/hls4ml/pt_etarel_phirel/jetformer/nconst16/symm_lorentz

# ========================================================================
# jetformer  nconst=50
# ========================================================================

# ------------------------------------------------------------------------
# jetformer  nconst=50  baseline  (method=augmentation, group=none)
# best trial 195 of jetformer_nconst50_augmentation_none, val/acc_best=0.823349
# ------------------------------------------------------------------------
# python3 src/train.py \
#     experiment=hls4ml/jetformer_50 \
#     data.method=augmentation \
#     data.augmentation_scheme=none \
#     data.n_folds=5 \
#     data.features='[pt,etarel,phirel]' \
#     algorithm.dropout=0.0 \
#     algorithm.embed_dim=96 \
#     algorithm.num_heads=4 \
#     algorithm.num_transformers=4 \
#     algorithm.optimizer.lr=0.002123939934809689 \
#     algorithm.optimizer.weight_decay=0.01 \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/train/hls4ml/pt_etarel_phirel/jetformer/nconst50/baseline

# ------------------------------------------------------------------------
# jetformer  nconst=50  aug_so2  (method=augmentation, group=so2)
# best trial 177 of jetformer_nconst50_augmentation_so2, val/acc_best=0.823133
# ------------------------------------------------------------------------
# python3 src/train.py \
#     experiment=hls4ml/jetformer_50 \
#     data.method=augmentation \
#     data.augmentation_scheme=so2 \
#     data.n_folds=5 \
#     data.features='[pt,etarel,phirel]' \
#     algorithm.dropout=0.0 \
#     algorithm.embed_dim=96 \
#     algorithm.num_heads=4 \
#     algorithm.num_transformers=4 \
#     algorithm.optimizer.lr=0.0017470594596271296 \
#     algorithm.optimizer.weight_decay=0.001 \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/train/hls4ml/pt_etarel_phirel/jetformer/nconst50/aug_so2

# ------------------------------------------------------------------------
# jetformer  nconst=50  aug_so3  (method=augmentation, group=so3)
# best trial 174 of jetformer_nconst50_augmentation_so3, val/acc_best=0.800189
# ------------------------------------------------------------------------
# python3 src/train.py \
#     experiment=hls4ml/jetformer_50 \
#     data.method=augmentation \
#     data.augmentation_scheme=so3 \
#     data.n_folds=5 \
#     data.features='[pt,etarel,phirel]' \
#     algorithm.dropout=0.0 \
#     algorithm.embed_dim=96 \
#     algorithm.num_heads=4 \
#     algorithm.num_transformers=4 \
#     algorithm.optimizer.lr=0.0043850010812732515 \
#     algorithm.optimizer.weight_decay=0.0001 \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/train/hls4ml/pt_etarel_phirel/jetformer/nconst50/aug_so3

# ------------------------------------------------------------------------
# jetformer  nconst=50  aug_lorentz  (method=augmentation, group=lorentz)
# best trial 155 of jetformer_nconst50_augmentation_lorentz, val/acc_best=0.823397
# ------------------------------------------------------------------------
# python3 src/train.py \
#     experiment=hls4ml/jetformer_50 \
#     data.method=augmentation \
#     data.augmentation_scheme=lorentz \
#     data.n_folds=5 \
#     data.features='[pt,etarel,phirel]' \
#     algorithm.dropout=0.0 \
#     algorithm.embed_dim=96 \
#     algorithm.num_heads=4 \
#     algorithm.num_transformers=4 \
#     algorithm.optimizer.lr=0.001947254765405048 \
#     algorithm.optimizer.weight_decay=0.0001 \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/train/hls4ml/pt_etarel_phirel/jetformer/nconst50/aug_lorentz

# ------------------------------------------------------------------------
# jetformer  nconst=50  symm_so2  (method=symm_loss, group=so2)
# best trial 170 of jetformer_nconst50_symm_loss_so2, val/acc_best=0.822527
# ------------------------------------------------------------------------
# python3 src/train.py \
#     experiment=hls4ml/jetformer_50 \
#     data.method=symm_loss \
#     data.augmentation_scheme=so2 \
#     data.n_folds=5 \
#     data.features='[pt,etarel,phirel]' \
#     algorithm.dropout=0.0 \
#     algorithm.embed_dim=96 \
#     algorithm.num_heads=4 \
#     algorithm.num_transformers=4 \
#     algorithm.optimizer.lr=0.0026976479782103934 \
#     algorithm.optimizer.weight_decay=0.01 \
#     algorithm.symm_loss_weight=0.06321752261189585 \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/train/hls4ml/pt_etarel_phirel/jetformer/nconst50/symm_so2

# ------------------------------------------------------------------------
# jetformer  nconst=50  symm_so3  (method=symm_loss, group=so3)
# best trial 177 of jetformer_nconst50_symm_loss_so3, val/acc_best=0.821875
# ------------------------------------------------------------------------
# python3 src/train.py \
#     experiment=hls4ml/jetformer_50 \
#     data.method=symm_loss \
#     data.augmentation_scheme=so3 \
#     data.n_folds=5 \
#     data.features='[pt,etarel,phirel]' \
#     algorithm.dropout=0.0 \
#     algorithm.embed_dim=96 \
#     algorithm.num_heads=4 \
#     algorithm.num_transformers=4 \
#     algorithm.optimizer.lr=0.004987908259983026 \
#     algorithm.optimizer.weight_decay=0.01 \
#     algorithm.symm_loss_weight=0.010160208138927146 \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/train/hls4ml/pt_etarel_phirel/jetformer/nconst50/symm_so3

# ------------------------------------------------------------------------
# jetformer  nconst=50  symm_lorentz  (method=symm_loss, group=lorentz)
# best trial 114 of jetformer_nconst50_symm_loss_lorentz, val/acc_best=0.823591
# ------------------------------------------------------------------------
# python3 src/train.py \
#     experiment=hls4ml/jetformer_50 \
#     data.method=symm_loss \
#     data.augmentation_scheme=lorentz \
#     data.n_folds=5 \
#     data.features='[pt,etarel,phirel]' \
#     algorithm.dropout=0.0 \
#     algorithm.embed_dim=96 \
#     algorithm.num_heads=4 \
#     algorithm.num_transformers=4 \
#     algorithm.optimizer.lr=0.0020276815384486988 \
#     algorithm.optimizer.weight_decay=0.0001 \
#     algorithm.symm_loss_weight=0.8440390189714261 \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/train/hls4ml/pt_etarel_phirel/jetformer/nconst50/symm_lorentz

# ========================================================================
# jetformer  nconst=150
# ========================================================================

# ------------------------------------------------------------------------
# jetformer  nconst=150  baseline  (method=augmentation, group=none)
# best trial 134 of jetformer_nconst150_augmentation_none, val/acc_best=0.831889
# ------------------------------------------------------------------------
# python3 src/train.py \
#     experiment=hls4ml/jetformer_150 \
#     data.method=augmentation \
#     data.augmentation_scheme=none \
#     data.n_folds=5 \
#     data.features='[pt,etarel,phirel]' \
#     algorithm.dropout=0.0 \
#     algorithm.embed_dim=96 \
#     algorithm.num_heads=4 \
#     algorithm.num_transformers=4 \
#     algorithm.optimizer.lr=0.004586890241115531 \
#     algorithm.optimizer.weight_decay=0.001 \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/train/hls4ml/pt_etarel_phirel/jetformer/nconst150/baseline

# ------------------------------------------------------------------------
# jetformer  nconst=150  aug_so2  (method=augmentation, group=so2)
# best trial 155 of jetformer_nconst150_augmentation_so2, val/acc_best=0.832033
# ------------------------------------------------------------------------
# python3 src/train.py \
#     experiment=hls4ml/jetformer_150 \
#     data.method=augmentation \
#     data.augmentation_scheme=so2 \
#     data.n_folds=5 \
#     data.features='[pt,etarel,phirel]' \
#     algorithm.dropout=0.0 \
#     algorithm.embed_dim=96 \
#     algorithm.num_heads=4 \
#     algorithm.num_transformers=4 \
#     algorithm.optimizer.lr=0.004563930492103511 \
#     algorithm.optimizer.weight_decay=0.001 \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/train/hls4ml/pt_etarel_phirel/jetformer/nconst150/aug_so2

# ------------------------------------------------------------------------
# jetformer  nconst=150  aug_so3  (method=augmentation, group=so3)
# best trial 130 of jetformer_nconst150_augmentation_so3, val/acc_best=0.813543
# ------------------------------------------------------------------------
# python3 src/train.py \
#     experiment=hls4ml/jetformer_150 \
#     data.method=augmentation \
#     data.augmentation_scheme=so3 \
#     data.n_folds=5 \
#     data.features='[pt,etarel,phirel]' \
#     algorithm.dropout=0.0 \
#     algorithm.embed_dim=96 \
#     algorithm.num_heads=4 \
#     algorithm.num_transformers=4 \
#     algorithm.optimizer.lr=0.004036982534673134 \
#     algorithm.optimizer.weight_decay=0.01 \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/train/hls4ml/pt_etarel_phirel/jetformer/nconst150/aug_so3

# ------------------------------------------------------------------------
# jetformer  nconst=150  aug_lorentz  (method=augmentation, group=lorentz)
# best trial 127 of jetformer_nconst150_augmentation_lorentz, val/acc_best=0.831519
# ------------------------------------------------------------------------
# python3 src/train.py \
#     experiment=hls4ml/jetformer_150 \
#     data.method=augmentation \
#     data.augmentation_scheme=lorentz \
#     data.n_folds=5 \
#     data.features='[pt,etarel,phirel]' \
#     algorithm.dropout=0.0 \
#     algorithm.embed_dim=96 \
#     algorithm.num_heads=4 \
#     algorithm.num_transformers=4 \
#     algorithm.optimizer.lr=0.003966981748659056 \
#     algorithm.optimizer.weight_decay=0.0001 \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/train/hls4ml/pt_etarel_phirel/jetformer/nconst150/aug_lorentz

# ------------------------------------------------------------------------
# jetformer  nconst=150  symm_so2  (method=symm_loss, group=so2)
# best trial 146 of jetformer_nconst150_symm_loss_so2, val/acc_best=0.831873
# ------------------------------------------------------------------------
# python3 src/train.py \
#     experiment=hls4ml/jetformer_150 \
#     data.method=symm_loss \
#     data.augmentation_scheme=so2 \
#     data.n_folds=5 \
#     data.features='[pt,etarel,phirel]' \
#     algorithm.dropout=0.0 \
#     algorithm.embed_dim=96 \
#     algorithm.num_heads=4 \
#     algorithm.num_transformers=4 \
#     algorithm.optimizer.lr=0.003959630149912562 \
#     algorithm.optimizer.weight_decay=0.001 \
#     algorithm.symm_loss_weight=0.37236369634577543 \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/train/hls4ml/pt_etarel_phirel/jetformer/nconst150/symm_so2

# ------------------------------------------------------------------------
# jetformer  nconst=150  symm_so3  (method=symm_loss, group=so3)
# best trial 150 of jetformer_nconst150_symm_loss_so3, val/acc_best=0.831839
# ------------------------------------------------------------------------
# python3 src/train.py \
#     experiment=hls4ml/jetformer_150 \
#     data.method=symm_loss \
#     data.augmentation_scheme=so3 \
#     data.n_folds=5 \
#     data.features='[pt,etarel,phirel]' \
#     algorithm.dropout=0.0 \
#     algorithm.embed_dim=96 \
#     algorithm.num_heads=4 \
#     algorithm.num_transformers=4 \
#     algorithm.optimizer.lr=0.0042278687005424245 \
#     algorithm.optimizer.weight_decay=0.0001 \
#     algorithm.symm_loss_weight=0.06472784057661676 \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/train/hls4ml/pt_etarel_phirel/jetformer/nconst150/symm_so3

# ------------------------------------------------------------------------
# jetformer  nconst=150  symm_lorentz  (method=symm_loss, group=lorentz)
# best trial 162 of jetformer_nconst150_symm_loss_lorentz, val/acc_best=0.832505
# ------------------------------------------------------------------------
# python3 src/train.py \
#     experiment=hls4ml/jetformer_150 \
#     data.method=symm_loss \
#     data.augmentation_scheme=lorentz \
#     data.n_folds=5 \
#     data.features='[pt,etarel,phirel]' \
#     algorithm.dropout=0.0 \
#     algorithm.embed_dim=96 \
#     algorithm.num_heads=4 \
#     algorithm.num_transformers=4 \
#     algorithm.optimizer.lr=0.003703381519089333 \
#     algorithm.optimizer.weight_decay=0.001 \
#     algorithm.symm_loss_weight=0.4410457925553868 \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/train/hls4ml/pt_etarel_phirel/jetformer/nconst150/symm_lorentz
