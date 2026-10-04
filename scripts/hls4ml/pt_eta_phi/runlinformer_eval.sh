#!/usr/bin/env bash
# ========================================================================
# LINFORMER EVALUATION COMMANDS
# ========================================================================
# One block per (constituent count, regime): 3 x 7 = 21 eval runs. Generated
# by scripts/optuna/make_train_catalogues.py; regenerate rather than edit by hand.
# Feature set [pt, eta, phi]: studies from logs/optuna/hls4ml/pt_eta_phi/, runs under logs/{train,eval}/hls4ml/pt_eta_phi/.
# Each block carries the best of the first 200 completed trials, by trial
# number, of its own study under logs/optuna/hls4ml/pt_eta_phi/;
# a block whose study is missing carries none and says so (transferred ablation).
#
# Every command is commented out: copy one to run it by hand, or feed the whole
# file to scripts/cluster/submit_eval.sh, which queues them on a GPU or slurm.

# ========================================================================
# linformer  nconst=16
# ========================================================================

# ------------------------------------------------------------------------
# linformer  nconst=16  baseline  (method=augmentation, group=none)
# best trial 87 of linformer_nconst16_augmentation_none, val/acc_best=0.690809
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=hls4ml/linformer_16 \
#     data.method=augmentation \
#     data.augmentation_scheme=none \
#     data.n_folds=5 \
#     data.features='[pt,eta,phi]' \
#     algorithm.aggregation=max \
#     algorithm.batchnorm=True \
#     algorithm.embed_dim=48 \
#     algorithm.ff_dim=128 \
#     algorithm.head_layers='[64,32]' \
#     algorithm.kv_proj_dim=4 \
#     algorithm.optimizer.lr=0.0022058978926696106 \
#     algorithm.optimizer.weight_decay=1e-06 \
#     checkpoint_manifest_path=logs/train/hls4ml/pt_eta_phi/linformer/nconst16/baseline \
#     results_table.invariance_metric=wasserstein \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/hls4ml/pt_eta_phi/linformer/nconst16/baseline

# ------------------------------------------------------------------------
# linformer  nconst=16  aug_so2  (method=augmentation, group=so2)
# best trial 173 of linformer_nconst16_augmentation_so2, val/acc_best=0.691344
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=hls4ml/linformer_16 \
#     data.method=augmentation \
#     data.augmentation_scheme=so2 \
#     data.n_folds=5 \
#     data.features='[pt,eta,phi]' \
#     algorithm.aggregation=max \
#     algorithm.batchnorm=True \
#     algorithm.embed_dim=64 \
#     algorithm.ff_dim=128 \
#     algorithm.head_layers='[64,32]' \
#     algorithm.kv_proj_dim=8 \
#     algorithm.optimizer.lr=0.0013455992130681322 \
#     algorithm.optimizer.weight_decay=0.0001 \
#     checkpoint_manifest_path=logs/train/hls4ml/pt_eta_phi/linformer/nconst16/aug_so2 \
#     results_table.invariance_metric=wasserstein \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/hls4ml/pt_eta_phi/linformer/nconst16/aug_so2

# ------------------------------------------------------------------------
# linformer  nconst=16  aug_so3  (method=augmentation, group=so3)
# best trial 170 of linformer_nconst16_augmentation_so3, val/acc_best=0.645031
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=hls4ml/linformer_16 \
#     data.method=augmentation \
#     data.augmentation_scheme=so3 \
#     data.n_folds=5 \
#     data.features='[pt,eta,phi]' \
#     algorithm.aggregation=max \
#     algorithm.batchnorm=True \
#     algorithm.embed_dim=32 \
#     algorithm.ff_dim=128 \
#     algorithm.head_layers='[64,32]' \
#     algorithm.kv_proj_dim=2 \
#     algorithm.optimizer.lr=0.0013455197693050998 \
#     algorithm.optimizer.weight_decay=0.0 \
#     checkpoint_manifest_path=logs/train/hls4ml/pt_eta_phi/linformer/nconst16/aug_so3 \
#     results_table.invariance_metric=wasserstein \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/hls4ml/pt_eta_phi/linformer/nconst16/aug_so3

# ------------------------------------------------------------------------
# linformer  nconst=16  aug_lorentz  (method=augmentation, group=lorentz)
# best trial 131 of linformer_nconst16_augmentation_lorentz, val/acc_best=0.689090
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=hls4ml/linformer_16 \
#     data.method=augmentation \
#     data.augmentation_scheme=lorentz \
#     data.n_folds=5 \
#     data.features='[pt,eta,phi]' \
#     algorithm.aggregation=sum_over_n \
#     algorithm.batchnorm=True \
#     algorithm.embed_dim=64 \
#     algorithm.ff_dim=128 \
#     algorithm.head_layers='[64,32]' \
#     algorithm.kv_proj_dim=8 \
#     algorithm.optimizer.lr=0.001872882431771738 \
#     algorithm.optimizer.weight_decay=1e-05 \
#     checkpoint_manifest_path=logs/train/hls4ml/pt_eta_phi/linformer/nconst16/aug_lorentz \
#     results_table.invariance_metric=wasserstein \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/hls4ml/pt_eta_phi/linformer/nconst16/aug_lorentz

# ------------------------------------------------------------------------
# linformer  nconst=16  symm_so2  (method=symm_loss, group=so2)
# best trial 187 of linformer_nconst16_symm_loss_so2, val/acc_best=0.696146
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=hls4ml/linformer_16 \
#     data.method=symm_loss \
#     data.augmentation_scheme=so2 \
#     data.n_folds=5 \
#     data.features='[pt,eta,phi]' \
#     algorithm.aggregation=max \
#     algorithm.batchnorm=True \
#     algorithm.embed_dim=48 \
#     algorithm.ff_dim=128 \
#     algorithm.head_layers='[64,32]' \
#     algorithm.kv_proj_dim=2 \
#     algorithm.optimizer.lr=0.0015493420833546415 \
#     algorithm.optimizer.weight_decay=0.0 \
#     algorithm.symm_loss_weight=0.061451795000935554 \
#     checkpoint_manifest_path=logs/train/hls4ml/pt_eta_phi/linformer/nconst16/symm_so2 \
#     results_table.invariance_metric=wasserstein \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/hls4ml/pt_eta_phi/linformer/nconst16/symm_so2

# ------------------------------------------------------------------------
# linformer  nconst=16  symm_so3  (method=symm_loss, group=so3)
# best trial 155 of linformer_nconst16_symm_loss_so3, val/acc_best=0.675627
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=hls4ml/linformer_16 \
#     data.method=symm_loss \
#     data.augmentation_scheme=so3 \
#     data.n_folds=5 \
#     data.features='[pt,eta,phi]' \
#     algorithm.aggregation=max \
#     algorithm.batchnorm=False \
#     algorithm.embed_dim=64 \
#     algorithm.ff_dim=128 \
#     algorithm.head_layers='[32,32,32]' \
#     algorithm.kv_proj_dim=2 \
#     algorithm.optimizer.lr=0.00326087235204059 \
#     algorithm.optimizer.weight_decay=1e-06 \
#     algorithm.symm_loss_weight=0.025369858871544162 \
#     checkpoint_manifest_path=logs/train/hls4ml/pt_eta_phi/linformer/nconst16/symm_so3 \
#     results_table.invariance_metric=wasserstein \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/hls4ml/pt_eta_phi/linformer/nconst16/symm_so3

# ------------------------------------------------------------------------
# linformer  nconst=16  symm_lorentz  (method=symm_loss, group=lorentz)
# best trial 199 of linformer_nconst16_symm_loss_lorentz, val/acc_best=0.688616
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=hls4ml/linformer_16 \
#     data.method=symm_loss \
#     data.augmentation_scheme=lorentz \
#     data.n_folds=5 \
#     data.features='[pt,eta,phi]' \
#     algorithm.aggregation=max \
#     algorithm.batchnorm=True \
#     algorithm.embed_dim=64 \
#     algorithm.ff_dim=128 \
#     algorithm.head_layers='[64,32]' \
#     algorithm.kv_proj_dim=2 \
#     algorithm.optimizer.lr=0.0017688144991983203 \
#     algorithm.optimizer.weight_decay=0.0 \
#     algorithm.symm_loss_weight=0.04647441526106747 \
#     checkpoint_manifest_path=logs/train/hls4ml/pt_eta_phi/linformer/nconst16/symm_lorentz \
#     results_table.invariance_metric=wasserstein \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/hls4ml/pt_eta_phi/linformer/nconst16/symm_lorentz

# ========================================================================
# linformer  nconst=50
# ========================================================================

# ------------------------------------------------------------------------
# linformer  nconst=50  baseline  (method=augmentation, group=none)
# best trial 194 of linformer_nconst50_augmentation_none, val/acc_best=0.765175
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=hls4ml/linformer_50 \
#     data.method=augmentation \
#     data.augmentation_scheme=none \
#     data.n_folds=5 \
#     data.features='[pt,eta,phi]' \
#     algorithm.aggregation=sum_over_n \
#     algorithm.batchnorm=True \
#     algorithm.embed_dim=32 \
#     algorithm.ff_dim=64 \
#     algorithm.head_layers='[64,32]' \
#     algorithm.kv_proj_dim=2 \
#     algorithm.optimizer.lr=0.0021768268006269625 \
#     algorithm.optimizer.weight_decay=1e-07 \
#     checkpoint_manifest_path=logs/train/hls4ml/pt_eta_phi/linformer/nconst50/baseline \
#     results_table.invariance_metric=wasserstein \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/hls4ml/pt_eta_phi/linformer/nconst50/baseline

# ------------------------------------------------------------------------
# linformer  nconst=50  aug_so2  (method=augmentation, group=so2)
# best trial 93 of linformer_nconst50_augmentation_so2, val/acc_best=0.770501
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=hls4ml/linformer_50 \
#     data.method=augmentation \
#     data.augmentation_scheme=so2 \
#     data.n_folds=5 \
#     data.features='[pt,eta,phi]' \
#     algorithm.aggregation=sum_over_n \
#     algorithm.batchnorm=True \
#     algorithm.embed_dim=64 \
#     algorithm.ff_dim=128 \
#     algorithm.head_layers='[32,32,16]' \
#     algorithm.kv_proj_dim=4 \
#     algorithm.optimizer.lr=0.0021684150755249976 \
#     algorithm.optimizer.weight_decay=1e-06 \
#     checkpoint_manifest_path=logs/train/hls4ml/pt_eta_phi/linformer/nconst50/aug_so2 \
#     results_table.invariance_metric=wasserstein \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/hls4ml/pt_eta_phi/linformer/nconst50/aug_so2

# ------------------------------------------------------------------------
# linformer  nconst=50  aug_so3  (method=augmentation, group=so3)
# best trial 84 of linformer_nconst50_augmentation_so3, val/acc_best=0.735217
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=hls4ml/linformer_50 \
#     data.method=augmentation \
#     data.augmentation_scheme=so3 \
#     data.n_folds=5 \
#     data.features='[pt,eta,phi]' \
#     algorithm.aggregation=sum_over_n \
#     algorithm.batchnorm=True \
#     algorithm.embed_dim=32 \
#     algorithm.ff_dim=128 \
#     algorithm.head_layers='[32,32,16]' \
#     algorithm.kv_proj_dim=8 \
#     algorithm.optimizer.lr=0.0020257293762235757 \
#     algorithm.optimizer.weight_decay=0.0 \
#     checkpoint_manifest_path=logs/train/hls4ml/pt_eta_phi/linformer/nconst50/aug_so3 \
#     results_table.invariance_metric=wasserstein \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/hls4ml/pt_eta_phi/linformer/nconst50/aug_so3

# ------------------------------------------------------------------------
# linformer  nconst=50  aug_lorentz  (method=augmentation, group=lorentz)
# best trial 198 of linformer_nconst50_augmentation_lorentz, val/acc_best=0.747260
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=hls4ml/linformer_50 \
#     data.method=augmentation \
#     data.augmentation_scheme=lorentz \
#     data.n_folds=5 \
#     data.features='[pt,eta,phi]' \
#     algorithm.aggregation=max \
#     algorithm.batchnorm=True \
#     algorithm.embed_dim=64 \
#     algorithm.ff_dim=64 \
#     algorithm.head_layers='[64,32]' \
#     algorithm.kv_proj_dim=4 \
#     algorithm.optimizer.lr=0.001884323499885085 \
#     algorithm.optimizer.weight_decay=1e-05 \
#     checkpoint_manifest_path=logs/train/hls4ml/pt_eta_phi/linformer/nconst50/aug_lorentz \
#     results_table.invariance_metric=wasserstein \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/hls4ml/pt_eta_phi/linformer/nconst50/aug_lorentz

# ------------------------------------------------------------------------
# linformer  nconst=50  symm_so2  (method=symm_loss, group=so2)
# best trial 130 of linformer_nconst50_symm_loss_so2, val/acc_best=0.749881
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=hls4ml/linformer_50 \
#     data.method=symm_loss \
#     data.augmentation_scheme=so2 \
#     data.n_folds=5 \
#     data.features='[pt,eta,phi]' \
#     algorithm.aggregation=max \
#     algorithm.batchnorm=True \
#     algorithm.embed_dim=48 \
#     algorithm.ff_dim=128 \
#     algorithm.head_layers='[64,32]' \
#     algorithm.kv_proj_dim=2 \
#     algorithm.optimizer.lr=0.0023350791094900103 \
#     algorithm.optimizer.weight_decay=1e-07 \
#     algorithm.symm_loss_weight=0.06569443865196674 \
#     checkpoint_manifest_path=logs/train/hls4ml/pt_eta_phi/linformer/nconst50/symm_so2 \
#     results_table.invariance_metric=wasserstein \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/hls4ml/pt_eta_phi/linformer/nconst50/symm_so2

# ------------------------------------------------------------------------
# linformer  nconst=50  symm_so3  (method=symm_loss, group=so3)
# best trial 56 of linformer_nconst50_symm_loss_so3, val/acc_best=0.735268
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=hls4ml/linformer_50 \
#     data.method=symm_loss \
#     data.augmentation_scheme=so3 \
#     data.n_folds=5 \
#     data.features='[pt,eta,phi]' \
#     algorithm.aggregation=max \
#     algorithm.batchnorm=True \
#     algorithm.embed_dim=64 \
#     algorithm.ff_dim=128 \
#     algorithm.head_layers='[32,32,16]' \
#     algorithm.kv_proj_dim=2 \
#     algorithm.optimizer.lr=0.002485449493153357 \
#     algorithm.optimizer.weight_decay=1e-05 \
#     algorithm.symm_loss_weight=0.006759578638171873 \
#     checkpoint_manifest_path=logs/train/hls4ml/pt_eta_phi/linformer/nconst50/symm_so3 \
#     results_table.invariance_metric=wasserstein \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/hls4ml/pt_eta_phi/linformer/nconst50/symm_so3

# ------------------------------------------------------------------------
# linformer  nconst=50  symm_lorentz  (method=symm_loss, group=lorentz)
# best trial 176 of linformer_nconst50_symm_loss_lorentz, val/acc_best=0.738284
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=hls4ml/linformer_50 \
#     data.method=symm_loss \
#     data.augmentation_scheme=lorentz \
#     data.n_folds=5 \
#     data.features='[pt,eta,phi]' \
#     algorithm.aggregation=max \
#     algorithm.batchnorm=True \
#     algorithm.embed_dim=48 \
#     algorithm.ff_dim=32 \
#     algorithm.head_layers='[16]' \
#     algorithm.kv_proj_dim=4 \
#     algorithm.optimizer.lr=0.0019146489892996619 \
#     algorithm.optimizer.weight_decay=1e-06 \
#     algorithm.symm_loss_weight=0.033202577022050886 \
#     checkpoint_manifest_path=logs/train/hls4ml/pt_eta_phi/linformer/nconst50/symm_lorentz \
#     results_table.invariance_metric=wasserstein \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/hls4ml/pt_eta_phi/linformer/nconst50/symm_lorentz

# ========================================================================
# linformer  nconst=150
# ========================================================================

# ------------------------------------------------------------------------
# linformer  nconst=150  baseline  (method=augmentation, group=none)
# best trial 36 of linformer_nconst150_augmentation_none, val/acc_best=0.737224
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=hls4ml/linformer_150 \
#     data.method=augmentation \
#     data.augmentation_scheme=none \
#     data.n_folds=5 \
#     data.features='[pt,eta,phi]' \
#     algorithm.aggregation=max \
#     algorithm.batchnorm=True \
#     algorithm.embed_dim=64 \
#     algorithm.ff_dim=64 \
#     algorithm.head_layers='[64,32]' \
#     algorithm.kv_proj_dim=8 \
#     algorithm.optimizer.lr=0.0012922073854315507 \
#     algorithm.optimizer.weight_decay=1e-06 \
#     checkpoint_manifest_path=logs/train/hls4ml/pt_eta_phi/linformer/nconst150/baseline \
#     results_table.invariance_metric=wasserstein \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/hls4ml/pt_eta_phi/linformer/nconst150/baseline

# ------------------------------------------------------------------------
# linformer  nconst=150  aug_so2  (method=augmentation, group=so2)
# best trial 196 of linformer_nconst150_augmentation_so2, val/acc_best=0.759506
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=hls4ml/linformer_150 \
#     data.method=augmentation \
#     data.augmentation_scheme=so2 \
#     data.n_folds=5 \
#     data.features='[pt,eta,phi]' \
#     algorithm.aggregation=sum_over_n \
#     algorithm.batchnorm=True \
#     algorithm.embed_dim=64 \
#     algorithm.ff_dim=128 \
#     algorithm.head_layers='[64,32]' \
#     algorithm.kv_proj_dim=8 \
#     algorithm.optimizer.lr=0.001885637117974637 \
#     algorithm.optimizer.weight_decay=1e-06 \
#     checkpoint_manifest_path=logs/train/hls4ml/pt_eta_phi/linformer/nconst150/aug_so2 \
#     results_table.invariance_metric=wasserstein \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/hls4ml/pt_eta_phi/linformer/nconst150/aug_so2

# ------------------------------------------------------------------------
# linformer  nconst=150  aug_so3  (method=augmentation, group=so3)
# best trial 143 of linformer_nconst150_augmentation_so3, val/acc_best=0.733677
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=hls4ml/linformer_150 \
#     data.method=augmentation \
#     data.augmentation_scheme=so3 \
#     data.n_folds=5 \
#     data.features='[pt,eta,phi]' \
#     algorithm.aggregation=sum_over_n \
#     algorithm.batchnorm=True \
#     algorithm.embed_dim=48 \
#     algorithm.ff_dim=128 \
#     algorithm.head_layers='[64,32]' \
#     algorithm.kv_proj_dim=4 \
#     algorithm.optimizer.lr=0.0018514151087396145 \
#     algorithm.optimizer.weight_decay=0.0 \
#     checkpoint_manifest_path=logs/train/hls4ml/pt_eta_phi/linformer/nconst150/aug_so3 \
#     results_table.invariance_metric=wasserstein \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/hls4ml/pt_eta_phi/linformer/nconst150/aug_so3

# ------------------------------------------------------------------------
# linformer  nconst=150  aug_lorentz  (method=augmentation, group=lorentz)
# best trial 122 of linformer_nconst150_augmentation_lorentz, val/acc_best=0.752362
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=hls4ml/linformer_150 \
#     data.method=augmentation \
#     data.augmentation_scheme=lorentz \
#     data.n_folds=5 \
#     data.features='[pt,eta,phi]' \
#     algorithm.aggregation=sum_over_n \
#     algorithm.batchnorm=True \
#     algorithm.embed_dim=32 \
#     algorithm.ff_dim=128 \
#     algorithm.head_layers='[64,32]' \
#     algorithm.kv_proj_dim=8 \
#     algorithm.optimizer.lr=0.0016665365457990155 \
#     algorithm.optimizer.weight_decay=1e-05 \
#     checkpoint_manifest_path=logs/train/hls4ml/pt_eta_phi/linformer/nconst150/aug_lorentz \
#     results_table.invariance_metric=wasserstein \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/hls4ml/pt_eta_phi/linformer/nconst150/aug_lorentz

# ------------------------------------------------------------------------
# linformer  nconst=150  symm_so2  (method=symm_loss, group=so2)
# best trial 98 of linformer_nconst150_symm_loss_so2, val/acc_best=0.743391
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=hls4ml/linformer_150 \
#     data.method=symm_loss \
#     data.augmentation_scheme=so2 \
#     data.n_folds=5 \
#     data.features='[pt,eta,phi]' \
#     algorithm.aggregation=max \
#     algorithm.batchnorm=True \
#     algorithm.embed_dim=48 \
#     algorithm.ff_dim=128 \
#     algorithm.head_layers='[32,32,32]' \
#     algorithm.kv_proj_dim=4 \
#     algorithm.optimizer.lr=0.004271948032943832 \
#     algorithm.optimizer.weight_decay=0.0 \
#     algorithm.symm_loss_weight=0.12279627569458536 \
#     checkpoint_manifest_path=logs/train/hls4ml/pt_eta_phi/linformer/nconst150/symm_so2 \
#     results_table.invariance_metric=wasserstein \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/hls4ml/pt_eta_phi/linformer/nconst150/symm_so2

# ------------------------------------------------------------------------
# linformer  nconst=150  symm_so3  (method=symm_loss, group=so3)
# best trial 67 of linformer_nconst150_symm_loss_so3, val/acc_best=0.727300
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=hls4ml/linformer_150 \
#     data.method=symm_loss \
#     data.augmentation_scheme=so3 \
#     data.n_folds=5 \
#     data.features='[pt,eta,phi]' \
#     algorithm.aggregation=max \
#     algorithm.batchnorm=True \
#     algorithm.embed_dim=64 \
#     algorithm.ff_dim=128 \
#     algorithm.head_layers='[64,32]' \
#     algorithm.kv_proj_dim=4 \
#     algorithm.optimizer.lr=0.002341636966660646 \
#     algorithm.optimizer.weight_decay=1e-05 \
#     algorithm.symm_loss_weight=0.017161688876792897 \
#     checkpoint_manifest_path=logs/train/hls4ml/pt_eta_phi/linformer/nconst150/symm_so3 \
#     results_table.invariance_metric=wasserstein \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/hls4ml/pt_eta_phi/linformer/nconst150/symm_so3

# ------------------------------------------------------------------------
# linformer  nconst=150  symm_lorentz  (method=symm_loss, group=lorentz)
# best trial 192 of linformer_nconst150_symm_loss_lorentz, val/acc_best=0.742609
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=hls4ml/linformer_150 \
#     data.method=symm_loss \
#     data.augmentation_scheme=lorentz \
#     data.n_folds=5 \
#     data.features='[pt,eta,phi]' \
#     algorithm.aggregation=max \
#     algorithm.batchnorm=True \
#     algorithm.embed_dim=48 \
#     algorithm.ff_dim=128 \
#     algorithm.head_layers='[32,32,32]' \
#     algorithm.kv_proj_dim=8 \
#     algorithm.optimizer.lr=0.0025325157822575087 \
#     algorithm.optimizer.weight_decay=1e-06 \
#     algorithm.symm_loss_weight=0.024975472653489785 \
#     checkpoint_manifest_path=logs/train/hls4ml/pt_eta_phi/linformer/nconst150/symm_lorentz \
#     results_table.invariance_metric=wasserstein \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/hls4ml/pt_eta_phi/linformer/nconst150/symm_lorentz
