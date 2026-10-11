#!/usr/bin/env bash
# ========================================================================
# LINFORMER TRAINING COMMANDS
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
# linformer  nconst=16
# ========================================================================

# ------------------------------------------------------------------------
# linformer  nconst=16  baseline  (method=augmentation, group=none)
# best trial 192 of linformer_nconst16_augmentation_none, val/acc_best=0.717100
# ------------------------------------------------------------------------
# python3 src/train.py \
#     experiment=hls4ml/linformer_16 \
#     data.method=augmentation \
#     data.augmentation_scheme=none \
#     data.n_folds=5 \
#     data.features='[pt,etarel,phirel]' \
#     algorithm.aggregation=max \
#     algorithm.batchnorm=True \
#     algorithm.embed_dim=64 \
#     algorithm.ff_dim=128 \
#     algorithm.head_layers='[32,32,16]' \
#     algorithm.kv_proj_dim=8 \
#     algorithm.optimizer.lr=0.0006992307582606116 \
#     algorithm.optimizer.weight_decay=0.0 \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/train/hls4ml/pt_etarel_phirel/linformer/nconst16/baseline

# ------------------------------------------------------------------------
# linformer  nconst=16  aug_so2  (method=augmentation, group=so2)
# best trial 199 of linformer_nconst16_augmentation_so2, val/acc_best=0.717613
# ------------------------------------------------------------------------
# python3 src/train.py \
#     experiment=hls4ml/linformer_16 \
#     data.method=augmentation \
#     data.augmentation_scheme=so2 \
#     data.n_folds=5 \
#     data.features='[pt,etarel,phirel]' \
#     algorithm.aggregation=max \
#     algorithm.batchnorm=True \
#     algorithm.embed_dim=64 \
#     algorithm.ff_dim=128 \
#     algorithm.head_layers='[32,32,16]' \
#     algorithm.kv_proj_dim=8 \
#     algorithm.optimizer.lr=0.0004509649161383583 \
#     algorithm.optimizer.weight_decay=1e-07 \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/train/hls4ml/pt_etarel_phirel/linformer/nconst16/aug_so2

# ------------------------------------------------------------------------
# linformer  nconst=16  aug_so3  (method=augmentation, group=so3)
# best trial 117 of linformer_nconst16_augmentation_so3, val/acc_best=0.669984
# ------------------------------------------------------------------------
# python3 src/train.py \
#     experiment=hls4ml/linformer_16 \
#     data.method=augmentation \
#     data.augmentation_scheme=so3 \
#     data.n_folds=5 \
#     data.features='[pt,etarel,phirel]' \
#     algorithm.aggregation=max \
#     algorithm.batchnorm=True \
#     algorithm.embed_dim=64 \
#     algorithm.ff_dim=128 \
#     algorithm.head_layers='[64,32]' \
#     algorithm.kv_proj_dim=8 \
#     algorithm.optimizer.lr=0.0007437241579930207 \
#     algorithm.optimizer.weight_decay=1e-05 \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/train/hls4ml/pt_etarel_phirel/linformer/nconst16/aug_so3

# ------------------------------------------------------------------------
# linformer  nconst=16  aug_lorentz  (method=augmentation, group=lorentz)
# best trial 107 of linformer_nconst16_augmentation_lorentz, val/acc_best=0.717381
# ------------------------------------------------------------------------
# python3 src/train.py \
#     experiment=hls4ml/linformer_16 \
#     data.method=augmentation \
#     data.augmentation_scheme=lorentz \
#     data.n_folds=5 \
#     data.features='[pt,etarel,phirel]' \
#     algorithm.aggregation=max \
#     algorithm.batchnorm=True \
#     algorithm.embed_dim=64 \
#     algorithm.ff_dim=128 \
#     algorithm.head_layers='[32,32,32]' \
#     algorithm.kv_proj_dim=2 \
#     algorithm.optimizer.lr=0.0016343706331462713 \
#     algorithm.optimizer.weight_decay=1e-07 \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/train/hls4ml/pt_etarel_phirel/linformer/nconst16/aug_lorentz

# ------------------------------------------------------------------------
# linformer  nconst=16  symm_so2  (method=symm_loss, group=so2)
# best trial 183 of linformer_nconst16_symm_loss_so2, val/acc_best=0.718403
# ------------------------------------------------------------------------
# python3 src/train.py \
#     experiment=hls4ml/linformer_16 \
#     data.method=symm_loss \
#     data.augmentation_scheme=so2 \
#     data.n_folds=5 \
#     data.features='[pt,etarel,phirel]' \
#     algorithm.aggregation=max \
#     algorithm.batchnorm=True \
#     algorithm.embed_dim=64 \
#     algorithm.ff_dim=32 \
#     algorithm.head_layers='[32,32,32]' \
#     algorithm.kv_proj_dim=4 \
#     algorithm.optimizer.lr=0.0014317131316543973 \
#     algorithm.optimizer.weight_decay=1e-07 \
#     algorithm.symm_loss_weight=0.8905511521593555 \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/train/hls4ml/pt_etarel_phirel/linformer/nconst16/symm_so2

# ------------------------------------------------------------------------
# linformer  nconst=16  symm_so3  (method=symm_loss, group=so3)
# best trial 134 of linformer_nconst16_symm_loss_so3, val/acc_best=0.715072
# ------------------------------------------------------------------------
# python3 src/train.py \
#     experiment=hls4ml/linformer_16 \
#     data.method=symm_loss \
#     data.augmentation_scheme=so3 \
#     data.n_folds=5 \
#     data.features='[pt,etarel,phirel]' \
#     algorithm.aggregation=max \
#     algorithm.batchnorm=False \
#     algorithm.embed_dim=64 \
#     algorithm.ff_dim=64 \
#     algorithm.head_layers='[32,32,16]' \
#     algorithm.kv_proj_dim=2 \
#     algorithm.optimizer.lr=0.0028942038231157393 \
#     algorithm.optimizer.weight_decay=1e-06 \
#     algorithm.symm_loss_weight=0.010903268101968896 \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/train/hls4ml/pt_etarel_phirel/linformer/nconst16/symm_so3

# ------------------------------------------------------------------------
# linformer  nconst=16  symm_lorentz  (method=symm_loss, group=lorentz)
# best trial 75 of linformer_nconst16_symm_loss_lorentz, val/acc_best=0.718974
# ------------------------------------------------------------------------
# python3 src/train.py \
#     experiment=hls4ml/linformer_16 \
#     data.method=symm_loss \
#     data.augmentation_scheme=lorentz \
#     data.n_folds=5 \
#     data.features='[pt,etarel,phirel]' \
#     algorithm.aggregation=max \
#     algorithm.batchnorm=True \
#     algorithm.embed_dim=48 \
#     algorithm.ff_dim=128 \
#     algorithm.head_layers='[64,32]' \
#     algorithm.kv_proj_dim=8 \
#     algorithm.optimizer.lr=0.0016158384091991777 \
#     algorithm.optimizer.weight_decay=1e-07 \
#     algorithm.symm_loss_weight=0.10936253608184318 \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/train/hls4ml/pt_etarel_phirel/linformer/nconst16/symm_lorentz

# ========================================================================
# linformer  nconst=50
# ========================================================================

# ------------------------------------------------------------------------
# linformer  nconst=50  baseline  (method=augmentation, group=none)
# best trial 190 of linformer_nconst50_augmentation_none, val/acc_best=0.810229
# ------------------------------------------------------------------------
# python3 src/train.py \
#     experiment=hls4ml/linformer_50 \
#     data.method=augmentation \
#     data.augmentation_scheme=none \
#     data.n_folds=5 \
#     data.features='[pt,etarel,phirel]' \
#     algorithm.aggregation=max \
#     algorithm.batchnorm=True \
#     algorithm.embed_dim=48 \
#     algorithm.ff_dim=64 \
#     algorithm.head_layers='[64,32]' \
#     algorithm.kv_proj_dim=8 \
#     algorithm.optimizer.lr=0.0008451472605932015 \
#     algorithm.optimizer.weight_decay=0.0 \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/train/hls4ml/pt_etarel_phirel/linformer/nconst50/baseline

# ------------------------------------------------------------------------
# linformer  nconst=50  aug_so2  (method=augmentation, group=so2)
# best trial 136 of linformer_nconst50_augmentation_so2, val/acc_best=0.805627
# ------------------------------------------------------------------------
# python3 src/train.py \
#     experiment=hls4ml/linformer_50 \
#     data.method=augmentation \
#     data.augmentation_scheme=so2 \
#     data.n_folds=5 \
#     data.features='[pt,etarel,phirel]' \
#     algorithm.aggregation=sum \
#     algorithm.batchnorm=False \
#     algorithm.embed_dim=48 \
#     algorithm.ff_dim=64 \
#     algorithm.head_layers='[64,32]' \
#     algorithm.kv_proj_dim=2 \
#     algorithm.optimizer.lr=0.0031026493662926024 \
#     algorithm.optimizer.weight_decay=0.0 \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/train/hls4ml/pt_etarel_phirel/linformer/nconst50/aug_so2

# ------------------------------------------------------------------------
# linformer  nconst=50  aug_so3  (method=augmentation, group=so3)
# best trial 144 of linformer_nconst50_augmentation_so3, val/acc_best=0.777889
# ------------------------------------------------------------------------
# python3 src/train.py \
#     experiment=hls4ml/linformer_50 \
#     data.method=augmentation \
#     data.augmentation_scheme=so3 \
#     data.n_folds=5 \
#     data.features='[pt,etarel,phirel]' \
#     algorithm.aggregation=max \
#     algorithm.batchnorm=False \
#     algorithm.embed_dim=64 \
#     algorithm.ff_dim=64 \
#     algorithm.head_layers='[32,32,32]' \
#     algorithm.kv_proj_dim=2 \
#     algorithm.optimizer.lr=0.002982162004420879 \
#     algorithm.optimizer.weight_decay=0.0 \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/train/hls4ml/pt_etarel_phirel/linformer/nconst50/aug_so3

# ------------------------------------------------------------------------
# linformer  nconst=50  aug_lorentz  (method=augmentation, group=lorentz)
# best trial 192 of linformer_nconst50_augmentation_lorentz, val/acc_best=0.810540
# ------------------------------------------------------------------------
# python3 src/train.py \
#     experiment=hls4ml/linformer_50 \
#     data.method=augmentation \
#     data.augmentation_scheme=lorentz \
#     data.n_folds=5 \
#     data.features='[pt,etarel,phirel]' \
#     algorithm.aggregation=max \
#     algorithm.batchnorm=True \
#     algorithm.embed_dim=64 \
#     algorithm.ff_dim=128 \
#     algorithm.head_layers='[32,32,16]' \
#     algorithm.kv_proj_dim=4 \
#     algorithm.optimizer.lr=0.0004618592383378721 \
#     algorithm.optimizer.weight_decay=1e-07 \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/train/hls4ml/pt_etarel_phirel/linformer/nconst50/aug_lorentz

# ------------------------------------------------------------------------
# linformer  nconst=50  symm_so2  (method=symm_loss, group=so2)
# best trial 134 of linformer_nconst50_symm_loss_so2, val/acc_best=0.813262
# ------------------------------------------------------------------------
# python3 src/train.py \
#     experiment=hls4ml/linformer_50 \
#     data.method=symm_loss \
#     data.augmentation_scheme=so2 \
#     data.n_folds=5 \
#     data.features='[pt,etarel,phirel]' \
#     algorithm.aggregation=sum \
#     algorithm.batchnorm=True \
#     algorithm.embed_dim=64 \
#     algorithm.ff_dim=128 \
#     algorithm.head_layers='[32,32,16]' \
#     algorithm.kv_proj_dim=8 \
#     algorithm.optimizer.lr=0.00369895238582543 \
#     algorithm.optimizer.weight_decay=1e-06 \
#     algorithm.symm_loss_weight=0.2005471578916678 \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/train/hls4ml/pt_etarel_phirel/linformer/nconst50/symm_so2

# ------------------------------------------------------------------------
# linformer  nconst=50  symm_so3  (method=symm_loss, group=so3)
# best trial 119 of linformer_nconst50_symm_loss_so3, val/acc_best=0.806627
# ------------------------------------------------------------------------
# python3 src/train.py \
#     experiment=hls4ml/linformer_50 \
#     data.method=symm_loss \
#     data.augmentation_scheme=so3 \
#     data.n_folds=5 \
#     data.features='[pt,etarel,phirel]' \
#     algorithm.aggregation=sum \
#     algorithm.batchnorm=False \
#     algorithm.embed_dim=24 \
#     algorithm.ff_dim=128 \
#     algorithm.head_layers='[64,32]' \
#     algorithm.kv_proj_dim=4 \
#     algorithm.optimizer.lr=0.0048943725993541775 \
#     algorithm.optimizer.weight_decay=0.0 \
#     algorithm.symm_loss_weight=0.00917485067531035 \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/train/hls4ml/pt_etarel_phirel/linformer/nconst50/symm_so3

# ------------------------------------------------------------------------
# linformer  nconst=50  symm_lorentz  (method=symm_loss, group=lorentz)
# best trial 175 of linformer_nconst50_symm_loss_lorentz, val/acc_best=0.815421
# ------------------------------------------------------------------------
# python3 src/train.py \
#     experiment=hls4ml/linformer_50 \
#     data.method=symm_loss \
#     data.augmentation_scheme=lorentz \
#     data.n_folds=5 \
#     data.features='[pt,etarel,phirel]' \
#     algorithm.aggregation=max \
#     algorithm.batchnorm=True \
#     algorithm.embed_dim=64 \
#     algorithm.ff_dim=128 \
#     algorithm.head_layers='[32,32,16]' \
#     algorithm.kv_proj_dim=8 \
#     algorithm.optimizer.lr=0.0012156852255645826 \
#     algorithm.optimizer.weight_decay=1e-06 \
#     algorithm.symm_loss_weight=0.8856887579308064 \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/train/hls4ml/pt_etarel_phirel/linformer/nconst50/symm_lorentz

# ========================================================================
# linformer  nconst=150
# ========================================================================

# ------------------------------------------------------------------------
# linformer  nconst=150  baseline  (method=augmentation, group=none)
# best trial 142 of linformer_nconst150_augmentation_none, val/acc_best=0.819746
# ------------------------------------------------------------------------
# python3 src/train.py \
#     experiment=hls4ml/linformer_150 \
#     data.method=augmentation \
#     data.augmentation_scheme=none \
#     data.n_folds=5 \
#     data.features='[pt,etarel,phirel]' \
#     algorithm.aggregation=max \
#     algorithm.batchnorm=True \
#     algorithm.embed_dim=64 \
#     algorithm.ff_dim=128 \
#     algorithm.head_layers='[64,32]' \
#     algorithm.kv_proj_dim=8 \
#     algorithm.optimizer.lr=0.0007346058117514102 \
#     algorithm.optimizer.weight_decay=1e-07 \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/train/hls4ml/pt_etarel_phirel/linformer/nconst150/baseline

# ------------------------------------------------------------------------
# linformer  nconst=150  aug_so2  (method=augmentation, group=so2)
# best trial 145 of linformer_nconst150_augmentation_so2, val/acc_best=0.815199
# ------------------------------------------------------------------------
# python3 src/train.py \
#     experiment=hls4ml/linformer_150 \
#     data.method=augmentation \
#     data.augmentation_scheme=so2 \
#     data.n_folds=5 \
#     data.features='[pt,etarel,phirel]' \
#     algorithm.aggregation=max \
#     algorithm.batchnorm=True \
#     algorithm.embed_dim=64 \
#     algorithm.ff_dim=128 \
#     algorithm.head_layers='[16]' \
#     algorithm.kv_proj_dim=8 \
#     algorithm.optimizer.lr=0.000977207450334623 \
#     algorithm.optimizer.weight_decay=1e-06 \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/train/hls4ml/pt_etarel_phirel/linformer/nconst150/aug_so2

# ------------------------------------------------------------------------
# linformer  nconst=150  aug_so3  (method=augmentation, group=so3)
# best trial 149 of linformer_nconst150_augmentation_so3, val/acc_best=0.787449
# ------------------------------------------------------------------------
# python3 src/train.py \
#     experiment=hls4ml/linformer_150 \
#     data.method=augmentation \
#     data.augmentation_scheme=so3 \
#     data.n_folds=5 \
#     data.features='[pt,etarel,phirel]' \
#     algorithm.aggregation=max \
#     algorithm.batchnorm=False \
#     algorithm.embed_dim=48 \
#     algorithm.ff_dim=128 \
#     algorithm.head_layers='[32,32,32]' \
#     algorithm.kv_proj_dim=8 \
#     algorithm.optimizer.lr=0.004659511635945531 \
#     algorithm.optimizer.weight_decay=0.0 \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/train/hls4ml/pt_etarel_phirel/linformer/nconst150/aug_so3

# ------------------------------------------------------------------------
# linformer  nconst=150  aug_lorentz  (method=augmentation, group=lorentz)
# best trial 29 of linformer_nconst150_augmentation_lorentz, val/acc_best=0.817672
# ------------------------------------------------------------------------
# python3 src/train.py \
#     experiment=hls4ml/linformer_150 \
#     data.method=augmentation \
#     data.augmentation_scheme=lorentz \
#     data.n_folds=5 \
#     data.features='[pt,etarel,phirel]' \
#     algorithm.aggregation=max \
#     algorithm.batchnorm=True \
#     algorithm.embed_dim=64 \
#     algorithm.ff_dim=64 \
#     algorithm.head_layers='[32,32,32]' \
#     algorithm.kv_proj_dim=8 \
#     algorithm.optimizer.lr=0.0007900635935245033 \
#     algorithm.optimizer.weight_decay=1e-07 \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/train/hls4ml/pt_etarel_phirel/linformer/nconst150/aug_lorentz

# ------------------------------------------------------------------------
# linformer  nconst=150  symm_so2  (method=symm_loss, group=so2)
# best trial 164 of linformer_nconst150_symm_loss_so2, val/acc_best=0.820003
# ------------------------------------------------------------------------
# python3 src/train.py \
#     experiment=hls4ml/linformer_150 \
#     data.method=symm_loss \
#     data.augmentation_scheme=so2 \
#     data.n_folds=5 \
#     data.features='[pt,etarel,phirel]' \
#     algorithm.aggregation=sum \
#     algorithm.batchnorm=True \
#     algorithm.embed_dim=64 \
#     algorithm.ff_dim=128 \
#     algorithm.head_layers='[64,32]' \
#     algorithm.kv_proj_dim=4 \
#     algorithm.optimizer.lr=0.0027517156379299975 \
#     algorithm.optimizer.weight_decay=0.0 \
#     algorithm.symm_loss_weight=0.4933890066904961 \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/train/hls4ml/pt_etarel_phirel/linformer/nconst150/symm_so2

# ------------------------------------------------------------------------
# linformer  nconst=150  symm_so3  (method=symm_loss, group=so3)
# best trial 178 of linformer_nconst150_symm_loss_so3, val/acc_best=0.816153
# ------------------------------------------------------------------------
# python3 src/train.py \
#     experiment=hls4ml/linformer_150 \
#     data.method=symm_loss \
#     data.augmentation_scheme=so3 \
#     data.n_folds=5 \
#     data.features='[pt,etarel,phirel]' \
#     algorithm.aggregation=max \
#     algorithm.batchnorm=False \
#     algorithm.embed_dim=48 \
#     algorithm.ff_dim=128 \
#     algorithm.head_layers='[64,32]' \
#     algorithm.kv_proj_dim=4 \
#     algorithm.optimizer.lr=0.003025824071199341 \
#     algorithm.optimizer.weight_decay=1e-07 \
#     algorithm.symm_loss_weight=0.011585455483690275 \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/train/hls4ml/pt_etarel_phirel/linformer/nconst150/symm_so3

# ------------------------------------------------------------------------
# linformer  nconst=150  symm_lorentz  (method=symm_loss, group=lorentz)
# best trial 150 of linformer_nconst150_symm_loss_lorentz, val/acc_best=0.820064
# ------------------------------------------------------------------------
# python3 src/train.py \
#     experiment=hls4ml/linformer_150 \
#     data.method=symm_loss \
#     data.augmentation_scheme=lorentz \
#     data.n_folds=5 \
#     data.features='[pt,etarel,phirel]' \
#     algorithm.aggregation=max \
#     algorithm.batchnorm=True \
#     algorithm.embed_dim=64 \
#     algorithm.ff_dim=128 \
#     algorithm.head_layers='[32,32,32]' \
#     algorithm.kv_proj_dim=8 \
#     algorithm.optimizer.lr=0.0010037710088760068 \
#     algorithm.optimizer.weight_decay=0.0 \
#     algorithm.symm_loss_weight=0.00846124141463579 \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/train/hls4ml/pt_etarel_phirel/linformer/nconst150/symm_lorentz
