#!/usr/bin/env bash
# ========================================================================
# DEEPSETS TRAINING COMMANDS
# ========================================================================
# One block per (constituent count, regime): 3 x 7 = 21 train runs. Generated
# by scripts/experiments/make_train_catalogues.py; regenerate rather than edit by hand.
# Each block carries the best of the first 200 completed trials, by trial
# number, of its own study under logs/optuna/hls4ml/;
# a block whose study is missing carries none and says so (transferred ablation).
#
# Every command is commented out: copy one to run it by hand, or feed the whole
# file to scripts/experiments/submit_train.sh, which queues them on a GPU or slurm.

# ========================================================================
# deepsets  nconst=16
# ========================================================================

# ------------------------------------------------------------------------
# deepsets  nconst=16  baseline  (method=augmentation, group=none)
# best trial 151 of deepsets_nconst16_augmentation_none, val/acc_best=0.559371
# ------------------------------------------------------------------------
# python3 src/train.py \
#     experiment=hls4ml/deepsets_16 \
#     data.method=augmentation \
#     data.augmentation_scheme=none \
#     data.n_folds=5 \
#     algorithm.aggregation=max \
#     algorithm.optimizer.lr=0.004223160227177269 \
#     algorithm.optimizer.weight_decay=1e-06 \
#     algorithm.phi_layers='[64,64,64]' \
#     algorithm.rho_layers='[64,32]' \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/train/hls4ml/deepsets/nconst16/baseline

# ------------------------------------------------------------------------
# deepsets  nconst=16  aug_so2  (method=augmentation, group=so2)
# best trial 173 of deepsets_nconst16_augmentation_so2, val/acc_best=0.554006
# ------------------------------------------------------------------------
# python3 src/train.py \
#     experiment=hls4ml/deepsets_16 \
#     data.method=augmentation \
#     data.augmentation_scheme=so2 \
#     data.n_folds=5 \
#     algorithm.aggregation=max \
#     algorithm.optimizer.lr=0.004611855155744805 \
#     algorithm.optimizer.weight_decay=1e-06 \
#     algorithm.phi_layers='[64,64,64]' \
#     algorithm.rho_layers='[64,32]' \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/train/hls4ml/deepsets/nconst16/aug_so2

# ------------------------------------------------------------------------
# deepsets  nconst=16  aug_so3  (method=augmentation, group=so3)
# best trial 88 of deepsets_nconst16_augmentation_so3, val/acc_best=0.481904
# ------------------------------------------------------------------------
# python3 src/train.py \
#     experiment=hls4ml/deepsets_16 \
#     data.method=augmentation \
#     data.augmentation_scheme=so3 \
#     data.n_folds=5 \
#     algorithm.aggregation=max \
#     algorithm.optimizer.lr=0.0031471437518706697 \
#     algorithm.optimizer.weight_decay=1e-06 \
#     algorithm.phi_layers='[64,64,64]' \
#     algorithm.rho_layers='[32,32,16]' \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/train/hls4ml/deepsets/nconst16/aug_so3

# ------------------------------------------------------------------------
# deepsets  nconst=16  aug_lorentz  (method=augmentation, group=lorentz)
# best trial 153 of deepsets_nconst16_augmentation_lorentz, val/acc_best=0.554194
# ------------------------------------------------------------------------
# python3 src/train.py \
#     experiment=hls4ml/deepsets_16 \
#     data.method=augmentation \
#     data.augmentation_scheme=lorentz \
#     data.n_folds=5 \
#     algorithm.aggregation=max \
#     algorithm.optimizer.lr=0.005809266236802134 \
#     algorithm.optimizer.weight_decay=1e-06 \
#     algorithm.phi_layers='[64,64,32]' \
#     algorithm.rho_layers='[32,16]' \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/train/hls4ml/deepsets/nconst16/aug_lorentz

# ------------------------------------------------------------------------
# deepsets  nconst=16  symm_so2  (method=symm_loss, group=so2)
# best trial 129 of deepsets_nconst16_symm_loss_so2, val/acc_best=0.561920
# ------------------------------------------------------------------------
# python3 src/train.py \
#     experiment=hls4ml/deepsets_16 \
#     data.method=symm_loss \
#     data.augmentation_scheme=so2 \
#     data.n_folds=5 \
#     algorithm.aggregation=max \
#     algorithm.optimizer.lr=0.0035893074646699643 \
#     algorithm.optimizer.weight_decay=1e-06 \
#     algorithm.phi_layers='[64,64,64]' \
#     algorithm.rho_layers='[64,32]' \
#     algorithm.symm_loss_weight=0.0032765266241953387 \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/train/hls4ml/deepsets/nconst16/symm_so2

# ------------------------------------------------------------------------
# deepsets  nconst=16  symm_so3  (method=symm_loss, group=so3)
# best trial 194 of deepsets_nconst16_symm_loss_so3, val/acc_best=0.554636
# ------------------------------------------------------------------------
# python3 src/train.py \
#     experiment=hls4ml/deepsets_16 \
#     data.method=symm_loss \
#     data.augmentation_scheme=so3 \
#     data.n_folds=5 \
#     algorithm.aggregation=max \
#     algorithm.optimizer.lr=0.0031733147766268835 \
#     algorithm.optimizer.weight_decay=1e-06 \
#     algorithm.phi_layers='[128,64,32]' \
#     algorithm.rho_layers='[64,32]' \
#     algorithm.symm_loss_weight=0.00165058462790043 \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/train/hls4ml/deepsets/nconst16/symm_so3

# ------------------------------------------------------------------------
# deepsets  nconst=16  symm_lorentz  (method=symm_loss, group=lorentz)
# best trial 185 of deepsets_nconst16_symm_loss_lorentz, val/acc_best=0.560572
# ------------------------------------------------------------------------
# python3 src/train.py \
#     experiment=hls4ml/deepsets_16 \
#     data.method=symm_loss \
#     data.augmentation_scheme=lorentz \
#     data.n_folds=5 \
#     algorithm.aggregation=max \
#     algorithm.optimizer.lr=0.0049405863546596335 \
#     algorithm.optimizer.weight_decay=1e-07 \
#     algorithm.phi_layers='[64,64,64]' \
#     algorithm.rho_layers='[64,32]' \
#     algorithm.symm_loss_weight=0.010348764431477487 \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/train/hls4ml/deepsets/nconst16/symm_lorentz

# ========================================================================
# deepsets  nconst=50
# ========================================================================

# ------------------------------------------------------------------------
# deepsets  nconst=50  baseline  (method=augmentation, group=none)
# best trial 60 of deepsets_nconst50_augmentation_none, val/acc_best=0.535877
# ------------------------------------------------------------------------
# python3 src/train.py \
#     experiment=hls4ml/deepsets_50 \
#     data.method=augmentation \
#     data.augmentation_scheme=none \
#     data.n_folds=5 \
#     algorithm.aggregation=max \
#     algorithm.optimizer.lr=0.0021215262153298957 \
#     algorithm.optimizer.weight_decay=1e-07 \
#     algorithm.phi_layers='[64,64,64]' \
#     algorithm.rho_layers='[64,32]' \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/train/hls4ml/deepsets/nconst50/baseline

# ------------------------------------------------------------------------
# deepsets  nconst=50  aug_so2  (method=augmentation, group=so2)
# best trial 112 of deepsets_nconst50_augmentation_so2, val/acc_best=0.572505
# ------------------------------------------------------------------------
# python3 src/train.py \
#     experiment=hls4ml/deepsets_50 \
#     data.method=augmentation \
#     data.augmentation_scheme=so2 \
#     data.n_folds=5 \
#     algorithm.aggregation=max \
#     algorithm.optimizer.lr=0.003930187001225488 \
#     algorithm.optimizer.weight_decay=0.0 \
#     algorithm.phi_layers='[64,64,64]' \
#     algorithm.rho_layers='[64,32]' \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/train/hls4ml/deepsets/nconst50/aug_so2

# ------------------------------------------------------------------------
# deepsets  nconst=50  aug_so3  (method=augmentation, group=so3)
# best trial 114 of deepsets_nconst50_augmentation_so3, val/acc_best=0.482024
# ------------------------------------------------------------------------
# python3 src/train.py \
#     experiment=hls4ml/deepsets_50 \
#     data.method=augmentation \
#     data.augmentation_scheme=so3 \
#     data.n_folds=5 \
#     algorithm.aggregation=max \
#     algorithm.optimizer.lr=0.007141746536794088 \
#     algorithm.optimizer.weight_decay=1e-06 \
#     algorithm.phi_layers='[64,64,32]' \
#     algorithm.rho_layers='[16]' \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/train/hls4ml/deepsets/nconst50/aug_so3

# ------------------------------------------------------------------------
# deepsets  nconst=50  aug_lorentz  (method=augmentation, group=lorentz)
# best trial 183 of deepsets_nconst50_augmentation_lorentz, val/acc_best=0.574713
# ------------------------------------------------------------------------
# python3 src/train.py \
#     experiment=hls4ml/deepsets_50 \
#     data.method=augmentation \
#     data.augmentation_scheme=lorentz \
#     data.n_folds=5 \
#     algorithm.aggregation=max \
#     algorithm.optimizer.lr=0.0029407857219183693 \
#     algorithm.optimizer.weight_decay=0.0 \
#     algorithm.phi_layers='[64,64,64]' \
#     algorithm.rho_layers='[64,32]' \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/train/hls4ml/deepsets/nconst50/aug_lorentz

# ------------------------------------------------------------------------
# deepsets  nconst=50  symm_so2  (method=symm_loss, group=so2)
# best trial 139 of deepsets_nconst50_symm_loss_so2, val/acc_best=0.577413
# ------------------------------------------------------------------------
# python3 src/train.py \
#     experiment=hls4ml/deepsets_50 \
#     data.method=symm_loss \
#     data.augmentation_scheme=so2 \
#     data.n_folds=5 \
#     algorithm.aggregation=max \
#     algorithm.optimizer.lr=0.003706169219621946 \
#     algorithm.optimizer.weight_decay=1e-06 \
#     algorithm.phi_layers='[64,64,64]' \
#     algorithm.rho_layers='[64,32]' \
#     algorithm.symm_loss_weight=0.018344518799227945 \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/train/hls4ml/deepsets/nconst50/symm_so2

# ------------------------------------------------------------------------
# deepsets  nconst=50  symm_so3  (method=symm_loss, group=so3)
# best trial 155 of deepsets_nconst50_symm_loss_so3, val/acc_best=0.565292
# ------------------------------------------------------------------------
# python3 src/train.py \
#     experiment=hls4ml/deepsets_50 \
#     data.method=symm_loss \
#     data.augmentation_scheme=so3 \
#     data.n_folds=5 \
#     algorithm.aggregation=max \
#     algorithm.optimizer.lr=0.003072283204173756 \
#     algorithm.optimizer.weight_decay=0.0 \
#     algorithm.phi_layers='[128,64,32]' \
#     algorithm.rho_layers='[64,32]' \
#     algorithm.symm_loss_weight=0.010044327681113468 \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/train/hls4ml/deepsets/nconst50/symm_so3

# ------------------------------------------------------------------------
# deepsets  nconst=50  symm_lorentz  (method=symm_loss, group=lorentz)
# best trial 192 of deepsets_nconst50_symm_loss_lorentz, val/acc_best=0.583035
# ------------------------------------------------------------------------
# python3 src/train.py \
#     experiment=hls4ml/deepsets_50 \
#     data.method=symm_loss \
#     data.augmentation_scheme=lorentz \
#     data.n_folds=5 \
#     algorithm.aggregation=max \
#     algorithm.optimizer.lr=0.0035004194576327445 \
#     algorithm.optimizer.weight_decay=0.0 \
#     algorithm.phi_layers='[64,64,64]' \
#     algorithm.rho_layers='[64,32]' \
#     algorithm.symm_loss_weight=0.0013455864560427818 \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/train/hls4ml/deepsets/nconst50/symm_lorentz

# ========================================================================
# deepsets  nconst=150
# ========================================================================

# ------------------------------------------------------------------------
# deepsets  nconst=150  baseline  (method=augmentation, group=none)
# best trial 49 of deepsets_nconst150_augmentation_none, val/acc_best=0.529217
# ------------------------------------------------------------------------
# python3 src/train.py \
#     experiment=hls4ml/deepsets_150 \
#     data.method=augmentation \
#     data.augmentation_scheme=none \
#     data.n_folds=5 \
#     algorithm.aggregation=max \
#     algorithm.optimizer.lr=0.004026507126921539 \
#     algorithm.optimizer.weight_decay=1e-05 \
#     algorithm.phi_layers='[64,64,64]' \
#     algorithm.rho_layers='[64,32]' \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/train/hls4ml/deepsets/nconst150/baseline

# ------------------------------------------------------------------------
# deepsets  nconst=150  aug_so2  (method=augmentation, group=so2)
# best trial 134 of deepsets_nconst150_augmentation_so2, val/acc_best=0.546050
# ------------------------------------------------------------------------
# python3 src/train.py \
#     experiment=hls4ml/deepsets_150 \
#     data.method=augmentation \
#     data.augmentation_scheme=so2 \
#     data.n_folds=5 \
#     algorithm.aggregation=max \
#     algorithm.optimizer.lr=0.002803188641327001 \
#     algorithm.optimizer.weight_decay=1e-05 \
#     algorithm.phi_layers='[128,64,32]' \
#     algorithm.rho_layers='[64,32]' \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/train/hls4ml/deepsets/nconst150/aug_so2

# ------------------------------------------------------------------------
# deepsets  nconst=150  aug_so3  (method=augmentation, group=so3)
# best trial 55 of deepsets_nconst150_augmentation_so3, val/acc_best=0.459593
# ------------------------------------------------------------------------
# python3 src/train.py \
#     experiment=hls4ml/deepsets_150 \
#     data.method=augmentation \
#     data.augmentation_scheme=so3 \
#     data.n_folds=5 \
#     algorithm.aggregation=sum_over_n \
#     algorithm.optimizer.lr=0.007521422590737123 \
#     algorithm.optimizer.weight_decay=1e-07 \
#     algorithm.phi_layers='[128,64,32]' \
#     algorithm.rho_layers='[64,32]' \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/train/hls4ml/deepsets/nconst150/aug_so3

# ------------------------------------------------------------------------
# deepsets  nconst=150  aug_lorentz  (method=augmentation, group=lorentz)
# best trial 120 of deepsets_nconst150_augmentation_lorentz, val/acc_best=0.541182
# ------------------------------------------------------------------------
# python3 src/train.py \
#     experiment=hls4ml/deepsets_150 \
#     data.method=augmentation \
#     data.augmentation_scheme=lorentz \
#     data.n_folds=5 \
#     algorithm.aggregation=max \
#     algorithm.optimizer.lr=0.004031666782706369 \
#     algorithm.optimizer.weight_decay=1e-07 \
#     algorithm.phi_layers='[128,64,32]' \
#     algorithm.rho_layers='[64,32]' \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/train/hls4ml/deepsets/nconst150/aug_lorentz

# ------------------------------------------------------------------------
# deepsets  nconst=150  symm_so2  (method=symm_loss, group=so2)
# best trial 13 of deepsets_nconst150_symm_loss_so2, val/acc_best=0.536033
# ------------------------------------------------------------------------
# python3 src/train.py \
#     experiment=hls4ml/deepsets_150 \
#     data.method=symm_loss \
#     data.augmentation_scheme=so2 \
#     data.n_folds=5 \
#     algorithm.aggregation=max \
#     algorithm.optimizer.lr=0.0062287076299382405 \
#     algorithm.optimizer.weight_decay=1e-07 \
#     algorithm.phi_layers='[64,64,64]' \
#     algorithm.rho_layers='[64,32]' \
#     algorithm.symm_loss_weight=0.002008386137728743 \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/train/hls4ml/deepsets/nconst150/symm_so2

# ------------------------------------------------------------------------
# deepsets  nconst=150  symm_so3  (method=symm_loss, group=so3)
# best trial 151 of deepsets_nconst150_symm_loss_so3, val/acc_best=0.513595
# ------------------------------------------------------------------------
# python3 src/train.py \
#     experiment=hls4ml/deepsets_150 \
#     data.method=symm_loss \
#     data.augmentation_scheme=so3 \
#     data.n_folds=5 \
#     algorithm.aggregation=sum \
#     algorithm.optimizer.lr=0.0024826780085546916 \
#     algorithm.optimizer.weight_decay=0.0 \
#     algorithm.phi_layers='[64,32,32]' \
#     algorithm.rho_layers='[32,32,16]' \
#     algorithm.symm_loss_weight=0.003945235093920062 \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/train/hls4ml/deepsets/nconst150/symm_so3

# ------------------------------------------------------------------------
# deepsets  nconst=150  symm_lorentz  (method=symm_loss, group=lorentz)
# best trial 65 of deepsets_nconst150_symm_loss_lorentz, val/acc_best=0.569713
# ------------------------------------------------------------------------
# python3 src/train.py \
#     experiment=hls4ml/deepsets_150 \
#     data.method=symm_loss \
#     data.augmentation_scheme=lorentz \
#     data.n_folds=5 \
#     algorithm.aggregation=max \
#     algorithm.optimizer.lr=0.002105484202246209 \
#     algorithm.optimizer.weight_decay=0.0 \
#     algorithm.phi_layers='[128,64,32]' \
#     algorithm.rho_layers='[64,32]' \
#     algorithm.symm_loss_weight=0.04993198046892372 \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/train/hls4ml/deepsets/nconst150/symm_lorentz
