#!/usr/bin/env bash
# ========================================================================
# LINFORMER EVALUATION COMMANDS
# ========================================================================
# One block per (constituent count, regime): 3 x 7 = 21 eval runs. Generated
# by scripts/optuna/make_train_catalogues.py; regenerate rather than edit by hand.
# Each block carries the best of the first 200 completed trials, by trial
# number, of its own study under logs/optuna/hls4ml/;
# a block whose study is missing carries none and says so (transferred ablation).
#
# Every command is commented out: copy one to run it by hand, or feed the whole
# file to scripts/cluster/submit_eval.sh, which queues them on a GPU or slurm.

# ========================================================================
# linformer  nconst=16
# ========================================================================

# ------------------------------------------------------------------------
# linformer  nconst=16  baseline  (method=augmentation, group=none)
# best trial 193 of linformer_nconst16_augmentation_none, val/acc_best=0.641951
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=hls4ml/linformer_16 \
#     data.method=augmentation \
#     data.augmentation_scheme=none \
#     data.n_folds=5 \
#     algorithm.aggregation=max \
#     algorithm.batchnorm=True \
#     algorithm.embed_dim=64 \
#     algorithm.ff_dim=128 \
#     algorithm.head_layers='[32,32,32]' \
#     algorithm.kv_proj_dim=8 \
#     algorithm.optimizer.lr=0.000509753641227993 \
#     algorithm.optimizer.weight_decay=0.0 \
#     checkpoint_manifest_path=logs/train/hls4ml/linformer/nconst16/baseline \
#     results_table.invariance_metric=tv \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/hls4ml/linformer/nconst16/baseline

# ------------------------------------------------------------------------
# linformer  nconst=16  aug_so2  (method=augmentation, group=so2)
# best trial 179 of linformer_nconst16_augmentation_so2, val/acc_best=0.654588
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=hls4ml/linformer_16 \
#     data.method=augmentation \
#     data.augmentation_scheme=so2 \
#     data.n_folds=5 \
#     algorithm.aggregation=max \
#     algorithm.batchnorm=True \
#     algorithm.embed_dim=64 \
#     algorithm.ff_dim=128 \
#     algorithm.head_layers='[64,32]' \
#     algorithm.kv_proj_dim=8 \
#     algorithm.optimizer.lr=0.0005646223394131361 \
#     algorithm.optimizer.weight_decay=1e-05 \
#     checkpoint_manifest_path=logs/train/hls4ml/linformer/nconst16/aug_so2 \
#     results_table.invariance_metric=tv \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/hls4ml/linformer/nconst16/aug_so2

# ------------------------------------------------------------------------
# linformer  nconst=16  aug_so3  (method=augmentation, group=so3)
# best trial 100 of linformer_nconst16_augmentation_so3, val/acc_best=0.612635
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=hls4ml/linformer_16 \
#     data.method=augmentation \
#     data.augmentation_scheme=so3 \
#     data.n_folds=5 \
#     algorithm.aggregation=max \
#     algorithm.batchnorm=True \
#     algorithm.embed_dim=48 \
#     algorithm.ff_dim=128 \
#     algorithm.head_layers='[32,32,16]' \
#     algorithm.kv_proj_dim=8 \
#     algorithm.optimizer.lr=0.0009857876776809822 \
#     algorithm.optimizer.weight_decay=1e-06 \
#     checkpoint_manifest_path=logs/train/hls4ml/linformer/nconst16/aug_so3 \
#     results_table.invariance_metric=tv \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/hls4ml/linformer/nconst16/aug_so3

# ------------------------------------------------------------------------
# linformer  nconst=16  aug_lorentz  (method=augmentation, group=lorentz)
# best trial 159 of linformer_nconst16_augmentation_lorentz, val/acc_best=0.644168
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=hls4ml/linformer_16 \
#     data.method=augmentation \
#     data.augmentation_scheme=lorentz \
#     data.n_folds=5 \
#     algorithm.aggregation=max \
#     algorithm.batchnorm=True \
#     algorithm.embed_dim=64 \
#     algorithm.ff_dim=128 \
#     algorithm.head_layers='[32,32,16]' \
#     algorithm.kv_proj_dim=8 \
#     algorithm.optimizer.lr=0.0007280340889097467 \
#     algorithm.optimizer.weight_decay=0.0001 \
#     checkpoint_manifest_path=logs/train/hls4ml/linformer/nconst16/aug_lorentz \
#     results_table.invariance_metric=tv \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/hls4ml/linformer/nconst16/aug_lorentz

# ------------------------------------------------------------------------
# linformer  nconst=16  symm_so2  (method=symm_loss, group=so2)
# best trial 197 of linformer_nconst16_symm_loss_so2, val/acc_best=0.657762
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=hls4ml/linformer_16 \
#     data.method=symm_loss \
#     data.augmentation_scheme=so2 \
#     data.n_folds=5 \
#     algorithm.aggregation=max \
#     algorithm.batchnorm=True \
#     algorithm.embed_dim=64 \
#     algorithm.ff_dim=128 \
#     algorithm.head_layers='[64,32]' \
#     algorithm.kv_proj_dim=8 \
#     algorithm.optimizer.lr=0.0008266835875519868 \
#     algorithm.optimizer.weight_decay=0.0001 \
#     algorithm.symm_loss_weight=0.050467494724832716 \
#     checkpoint_manifest_path=logs/train/hls4ml/linformer/nconst16/symm_so2 \
#     results_table.invariance_metric=tv \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/hls4ml/linformer/nconst16/symm_so2

# ------------------------------------------------------------------------
# linformer  nconst=16  symm_so3  (method=symm_loss, group=so3)
# best trial 176 of linformer_nconst16_symm_loss_so3, val/acc_best=0.634994
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=hls4ml/linformer_16 \
#     data.method=symm_loss \
#     data.augmentation_scheme=so3 \
#     data.n_folds=5 \
#     algorithm.aggregation=max \
#     algorithm.batchnorm=True \
#     algorithm.embed_dim=64 \
#     algorithm.ff_dim=128 \
#     algorithm.head_layers='[32,32,16]' \
#     algorithm.kv_proj_dim=8 \
#     algorithm.optimizer.lr=0.0007972999248101985 \
#     algorithm.optimizer.weight_decay=1e-07 \
#     algorithm.symm_loss_weight=0.004161255676567499 \
#     checkpoint_manifest_path=logs/train/hls4ml/linformer/nconst16/symm_so3 \
#     results_table.invariance_metric=tv \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/hls4ml/linformer/nconst16/symm_so3

# ------------------------------------------------------------------------
# linformer  nconst=16  symm_lorentz  (method=symm_loss, group=lorentz)
# best trial 197 of linformer_nconst16_symm_loss_lorentz, val/acc_best=0.642766
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=hls4ml/linformer_16 \
#     data.method=symm_loss \
#     data.augmentation_scheme=lorentz \
#     data.n_folds=5 \
#     algorithm.aggregation=max \
#     algorithm.batchnorm=True \
#     algorithm.embed_dim=64 \
#     algorithm.ff_dim=64 \
#     algorithm.head_layers='[64,32]' \
#     algorithm.kv_proj_dim=8 \
#     algorithm.optimizer.lr=0.0009815346928503243 \
#     algorithm.optimizer.weight_decay=0.0001 \
#     algorithm.symm_loss_weight=0.011883172902839743 \
#     checkpoint_manifest_path=logs/train/hls4ml/linformer/nconst16/symm_lorentz \
#     results_table.invariance_metric=tv \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/hls4ml/linformer/nconst16/symm_lorentz

# ========================================================================
# linformer  nconst=50
# ========================================================================

# ------------------------------------------------------------------------
# linformer  nconst=50  baseline  (method=augmentation, group=none)
# best trial 196 of linformer_nconst50_augmentation_none, val/acc_best=0.665072
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=hls4ml/linformer_50 \
#     data.method=augmentation \
#     data.augmentation_scheme=none \
#     data.n_folds=5 \
#     algorithm.aggregation=max \
#     algorithm.batchnorm=True \
#     algorithm.embed_dim=64 \
#     algorithm.ff_dim=64 \
#     algorithm.head_layers='[32,32,32]' \
#     algorithm.kv_proj_dim=4 \
#     algorithm.optimizer.lr=0.0005132862698877847 \
#     algorithm.optimizer.weight_decay=1e-05 \
#     checkpoint_manifest_path=logs/train/hls4ml/linformer/nconst50/baseline \
#     results_table.invariance_metric=tv \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/hls4ml/linformer/nconst50/baseline

# ------------------------------------------------------------------------
# linformer  nconst=50  aug_so2  (method=augmentation, group=so2)
# best trial 196 of linformer_nconst50_augmentation_so2, val/acc_best=0.680535
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=hls4ml/linformer_50 \
#     data.method=augmentation \
#     data.augmentation_scheme=so2 \
#     data.n_folds=5 \
#     algorithm.aggregation=max \
#     algorithm.batchnorm=True \
#     algorithm.embed_dim=48 \
#     algorithm.ff_dim=64 \
#     algorithm.head_layers='[32,32,16]' \
#     algorithm.kv_proj_dim=4 \
#     algorithm.optimizer.lr=0.0005726996143920488 \
#     algorithm.optimizer.weight_decay=1e-05 \
#     checkpoint_manifest_path=logs/train/hls4ml/linformer/nconst50/aug_so2 \
#     results_table.invariance_metric=tv \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/hls4ml/linformer/nconst50/aug_so2

# ------------------------------------------------------------------------
# linformer  nconst=50  aug_so3  (method=augmentation, group=so3)
# best trial 159 of linformer_nconst50_augmentation_so3, val/acc_best=0.665782
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=hls4ml/linformer_50 \
#     data.method=augmentation \
#     data.augmentation_scheme=so3 \
#     data.n_folds=5 \
#     algorithm.aggregation=max \
#     algorithm.batchnorm=True \
#     algorithm.embed_dim=64 \
#     algorithm.ff_dim=64 \
#     algorithm.head_layers='[32,32,32]' \
#     algorithm.kv_proj_dim=4 \
#     algorithm.optimizer.lr=0.0004802874419863691 \
#     algorithm.optimizer.weight_decay=1e-07 \
#     checkpoint_manifest_path=logs/train/hls4ml/linformer/nconst50/aug_so3 \
#     results_table.invariance_metric=tv \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/hls4ml/linformer/nconst50/aug_so3

# ------------------------------------------------------------------------
# linformer  nconst=50  aug_lorentz  (method=augmentation, group=lorentz)
# best trial 164 of linformer_nconst50_augmentation_lorentz, val/acc_best=0.691172
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=hls4ml/linformer_50 \
#     data.method=augmentation \
#     data.augmentation_scheme=lorentz \
#     data.n_folds=5 \
#     algorithm.aggregation=max \
#     algorithm.batchnorm=True \
#     algorithm.embed_dim=64 \
#     algorithm.ff_dim=128 \
#     algorithm.head_layers='[64,32]' \
#     algorithm.kv_proj_dim=4 \
#     algorithm.optimizer.lr=0.0006341264702899537 \
#     algorithm.optimizer.weight_decay=1e-07 \
#     checkpoint_manifest_path=logs/train/hls4ml/linformer/nconst50/aug_lorentz \
#     results_table.invariance_metric=tv \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/hls4ml/linformer/nconst50/aug_lorentz

# ------------------------------------------------------------------------
# linformer  nconst=50  symm_so2  (method=symm_loss, group=so2)
# best trial 164 of linformer_nconst50_symm_loss_so2, val/acc_best=0.688803
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=hls4ml/linformer_50 \
#     data.method=symm_loss \
#     data.augmentation_scheme=so2 \
#     data.n_folds=5 \
#     algorithm.aggregation=max \
#     algorithm.batchnorm=True \
#     algorithm.embed_dim=48 \
#     algorithm.ff_dim=64 \
#     algorithm.head_layers='[32,32,32]' \
#     algorithm.kv_proj_dim=4 \
#     algorithm.optimizer.lr=0.0009616815577954468 \
#     algorithm.optimizer.weight_decay=1e-06 \
#     algorithm.symm_loss_weight=0.001829229768421361 \
#     checkpoint_manifest_path=logs/train/hls4ml/linformer/nconst50/symm_so2 \
#     results_table.invariance_metric=tv \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/hls4ml/linformer/nconst50/symm_so2

# ------------------------------------------------------------------------
# linformer  nconst=50  symm_so3  (method=symm_loss, group=so3)
# best trial 199 of linformer_nconst50_symm_loss_so3, val/acc_best=0.675001
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=hls4ml/linformer_50 \
#     data.method=symm_loss \
#     data.augmentation_scheme=so3 \
#     data.n_folds=5 \
#     algorithm.aggregation=max \
#     algorithm.batchnorm=True \
#     algorithm.embed_dim=64 \
#     algorithm.ff_dim=128 \
#     algorithm.head_layers='[32,32,32]' \
#     algorithm.kv_proj_dim=4 \
#     algorithm.optimizer.lr=0.0009742317811267194 \
#     algorithm.optimizer.weight_decay=0.0001 \
#     algorithm.symm_loss_weight=0.1694961743617896 \
#     checkpoint_manifest_path=logs/train/hls4ml/linformer/nconst50/symm_so3 \
#     results_table.invariance_metric=tv \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/hls4ml/linformer/nconst50/symm_so3

# ------------------------------------------------------------------------
# linformer  nconst=50  symm_lorentz  (method=symm_loss, group=lorentz)
# best trial 124 of linformer_nconst50_symm_loss_lorentz, val/acc_best=0.681229
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=hls4ml/linformer_50 \
#     data.method=symm_loss \
#     data.augmentation_scheme=lorentz \
#     data.n_folds=5 \
#     algorithm.aggregation=max \
#     algorithm.batchnorm=True \
#     algorithm.embed_dim=48 \
#     algorithm.ff_dim=128 \
#     algorithm.head_layers='[32,32,16]' \
#     algorithm.kv_proj_dim=2 \
#     algorithm.optimizer.lr=0.00040824301911664054 \
#     algorithm.optimizer.weight_decay=1e-06 \
#     algorithm.symm_loss_weight=0.0107707732421278 \
#     checkpoint_manifest_path=logs/train/hls4ml/linformer/nconst50/symm_lorentz \
#     results_table.invariance_metric=tv \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/hls4ml/linformer/nconst50/symm_lorentz

# ========================================================================
# linformer  nconst=150
# ========================================================================

# ------------------------------------------------------------------------
# linformer  nconst=150  baseline  (method=augmentation, group=none)
# best trial 170 of linformer_nconst150_augmentation_none, val/acc_best=0.665199
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=hls4ml/linformer_150 \
#     data.method=augmentation \
#     data.augmentation_scheme=none \
#     data.n_folds=5 \
#     algorithm.aggregation=max \
#     algorithm.batchnorm=True \
#     algorithm.embed_dim=64 \
#     algorithm.ff_dim=64 \
#     algorithm.head_layers='[32,32,16]' \
#     algorithm.kv_proj_dim=8 \
#     algorithm.optimizer.lr=0.0003325317830354233 \
#     algorithm.optimizer.weight_decay=0.0001 \
#     checkpoint_manifest_path=logs/train/hls4ml/linformer/nconst150/baseline \
#     results_table.invariance_metric=tv \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/hls4ml/linformer/nconst150/baseline

# ------------------------------------------------------------------------
# linformer  nconst=150  aug_so2  (method=augmentation, group=so2)
# best trial 150 of linformer_nconst150_augmentation_so2, val/acc_best=0.667625
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=hls4ml/linformer_150 \
#     data.method=augmentation \
#     data.augmentation_scheme=so2 \
#     data.n_folds=5 \
#     algorithm.aggregation=max \
#     algorithm.batchnorm=True \
#     algorithm.embed_dim=64 \
#     algorithm.ff_dim=64 \
#     algorithm.head_layers='[32,32,16]' \
#     algorithm.kv_proj_dim=2 \
#     algorithm.optimizer.lr=0.00048669529955597826 \
#     algorithm.optimizer.weight_decay=1e-05 \
#     checkpoint_manifest_path=logs/train/hls4ml/linformer/nconst150/aug_so2 \
#     results_table.invariance_metric=tv \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/hls4ml/linformer/nconst150/aug_so2

# ------------------------------------------------------------------------
# linformer  nconst=150  aug_so3  (method=augmentation, group=so3)
# best trial 165 of linformer_nconst150_augmentation_so3, val/acc_best=0.646903
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=hls4ml/linformer_150 \
#     data.method=augmentation \
#     data.augmentation_scheme=so3 \
#     data.n_folds=5 \
#     algorithm.aggregation=max \
#     algorithm.batchnorm=True \
#     algorithm.embed_dim=64 \
#     algorithm.ff_dim=128 \
#     algorithm.head_layers='[32,16]' \
#     algorithm.kv_proj_dim=8 \
#     algorithm.optimizer.lr=0.00033701047643506306 \
#     algorithm.optimizer.weight_decay=0.0 \
#     checkpoint_manifest_path=logs/train/hls4ml/linformer/nconst150/aug_so3 \
#     results_table.invariance_metric=tv \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/hls4ml/linformer/nconst150/aug_so3

# ------------------------------------------------------------------------
# linformer  nconst=150  aug_lorentz  (method=augmentation, group=lorentz)
# best trial 176 of linformer_nconst150_augmentation_lorentz, val/acc_best=0.688194
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=hls4ml/linformer_150 \
#     data.method=augmentation \
#     data.augmentation_scheme=lorentz \
#     data.n_folds=5 \
#     algorithm.aggregation=max \
#     algorithm.batchnorm=True \
#     algorithm.embed_dim=64 \
#     algorithm.ff_dim=128 \
#     algorithm.head_layers='[64,32]' \
#     algorithm.kv_proj_dim=8 \
#     algorithm.optimizer.lr=0.0004295833097630795 \
#     algorithm.optimizer.weight_decay=0.0 \
#     checkpoint_manifest_path=logs/train/hls4ml/linformer/nconst150/aug_lorentz \
#     results_table.invariance_metric=tv \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/hls4ml/linformer/nconst150/aug_lorentz

# ------------------------------------------------------------------------
# linformer  nconst=150  symm_so2  (method=symm_loss, group=so2)
# best trial 54 of linformer_nconst150_symm_loss_so2, val/acc_best=0.702953
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=hls4ml/linformer_150 \
#     data.method=symm_loss \
#     data.augmentation_scheme=so2 \
#     data.n_folds=5 \
#     algorithm.aggregation=max \
#     algorithm.batchnorm=True \
#     algorithm.embed_dim=64 \
#     algorithm.ff_dim=128 \
#     algorithm.head_layers='[32,32,32]' \
#     algorithm.kv_proj_dim=4 \
#     algorithm.optimizer.lr=0.0007465273385464103 \
#     algorithm.optimizer.weight_decay=1e-06 \
#     algorithm.symm_loss_weight=0.06928775176482996 \
#     checkpoint_manifest_path=logs/train/hls4ml/linformer/nconst150/symm_so2 \
#     results_table.invariance_metric=tv \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/hls4ml/linformer/nconst150/symm_so2

# ------------------------------------------------------------------------
# linformer  nconst=150  symm_so3  (method=symm_loss, group=so3)
# best trial 199 of linformer_nconst150_symm_loss_so3, val/acc_best=0.686064
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=hls4ml/linformer_150 \
#     data.method=symm_loss \
#     data.augmentation_scheme=so3 \
#     data.n_folds=5 \
#     algorithm.aggregation=max \
#     algorithm.batchnorm=True \
#     algorithm.embed_dim=64 \
#     algorithm.ff_dim=128 \
#     algorithm.head_layers='[32,32,32]' \
#     algorithm.kv_proj_dim=8 \
#     algorithm.optimizer.lr=0.00045710084985162413 \
#     algorithm.optimizer.weight_decay=1e-05 \
#     algorithm.symm_loss_weight=0.02555419295169649 \
#     checkpoint_manifest_path=logs/train/hls4ml/linformer/nconst150/symm_so3 \
#     results_table.invariance_metric=tv \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/hls4ml/linformer/nconst150/symm_so3

# ------------------------------------------------------------------------
# linformer  nconst=150  symm_lorentz  (method=symm_loss, group=lorentz)
# best trial 191 of linformer_nconst150_symm_loss_lorentz, val/acc_best=0.686470
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=hls4ml/linformer_150 \
#     data.method=symm_loss \
#     data.augmentation_scheme=lorentz \
#     data.n_folds=5 \
#     algorithm.aggregation=max \
#     algorithm.batchnorm=True \
#     algorithm.embed_dim=64 \
#     algorithm.ff_dim=128 \
#     algorithm.head_layers='[64,32]' \
#     algorithm.kv_proj_dim=8 \
#     algorithm.optimizer.lr=0.0009351521232861559 \
#     algorithm.optimizer.weight_decay=1e-07 \
#     algorithm.symm_loss_weight=0.0017900149682060764 \
#     checkpoint_manifest_path=logs/train/hls4ml/linformer/nconst150/symm_lorentz \
#     results_table.invariance_metric=tv \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/hls4ml/linformer/nconst150/symm_lorentz
