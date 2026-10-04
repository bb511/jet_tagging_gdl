#!/usr/bin/env bash
# ========================================================================
# JETFORMER EVALUATION COMMANDS
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
# jetformer  nconst=16
# ========================================================================

# ------------------------------------------------------------------------
# jetformer  nconst=16  baseline  (method=augmentation, group=none)
# best trial 173 of jetformer_nconst16_augmentation_none, val/acc_best=0.554743
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=hls4ml/jetformer_16 \
#     data.method=augmentation \
#     data.augmentation_scheme=none \
#     data.n_folds=5 \
#     algorithm.dropout=0.0 \
#     algorithm.embed_dim=64 \
#     algorithm.num_heads=2 \
#     algorithm.num_transformers=4 \
#     algorithm.optimizer.lr=0.004127539615688498 \
#     algorithm.optimizer.weight_decay=0.0 \
#     checkpoint_manifest_path=logs/train/hls4ml/jetformer/nconst16/baseline \
#     results_table.invariance_metric=wasserstein \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/hls4ml/jetformer/nconst16/baseline

# ------------------------------------------------------------------------
# jetformer  nconst=16  aug_so2  (method=augmentation, group=so2)
# best trial 119 of jetformer_nconst16_augmentation_so2, val/acc_best=0.634388
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=hls4ml/jetformer_16 \
#     data.method=augmentation \
#     data.augmentation_scheme=so2 \
#     data.n_folds=5 \
#     algorithm.dropout=0.1 \
#     algorithm.embed_dim=64 \
#     algorithm.num_heads=4 \
#     algorithm.num_transformers=4 \
#     algorithm.optimizer.lr=0.004521515127642492 \
#     algorithm.optimizer.weight_decay=0.01 \
#     checkpoint_manifest_path=logs/train/hls4ml/jetformer/nconst16/aug_so2 \
#     results_table.invariance_metric=wasserstein \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/hls4ml/jetformer/nconst16/aug_so2

# ------------------------------------------------------------------------
# jetformer  nconst=16  aug_so3  (method=augmentation, group=so3)
# best trial 185 of jetformer_nconst16_augmentation_so3, val/acc_best=0.363194
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=hls4ml/jetformer_16 \
#     data.method=augmentation \
#     data.augmentation_scheme=so3 \
#     data.n_folds=5 \
#     algorithm.dropout=0.0 \
#     algorithm.embed_dim=64 \
#     algorithm.num_heads=4 \
#     algorithm.num_transformers=4 \
#     algorithm.optimizer.lr=0.004441824608072794 \
#     algorithm.optimizer.weight_decay=0.0001 \
#     checkpoint_manifest_path=logs/train/hls4ml/jetformer/nconst16/aug_so3 \
#     results_table.invariance_metric=wasserstein \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/hls4ml/jetformer/nconst16/aug_so3

# ------------------------------------------------------------------------
# jetformer  nconst=16  aug_lorentz  (method=augmentation, group=lorentz)
# best trial 198 of jetformer_nconst16_augmentation_lorentz, val/acc_best=0.550265
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=hls4ml/jetformer_16 \
#     data.method=augmentation \
#     data.augmentation_scheme=lorentz \
#     data.n_folds=5 \
#     algorithm.dropout=0.0 \
#     algorithm.embed_dim=64 \
#     algorithm.num_heads=4 \
#     algorithm.num_transformers=4 \
#     algorithm.optimizer.lr=0.004569987152091947 \
#     algorithm.optimizer.weight_decay=0.01 \
#     checkpoint_manifest_path=logs/train/hls4ml/jetformer/nconst16/aug_lorentz \
#     results_table.invariance_metric=wasserstein \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/hls4ml/jetformer/nconst16/aug_lorentz

# ------------------------------------------------------------------------
# jetformer  nconst=16  symm_so2  (method=symm_loss, group=so2)
# best trial 185 of jetformer_nconst16_symm_loss_so2, val/acc_best=0.575311
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=hls4ml/jetformer_16 \
#     data.method=symm_loss \
#     data.augmentation_scheme=so2 \
#     data.n_folds=5 \
#     algorithm.dropout=0.0 \
#     algorithm.embed_dim=64 \
#     algorithm.num_heads=2 \
#     algorithm.num_transformers=3 \
#     algorithm.optimizer.lr=0.0049952838904663965 \
#     algorithm.optimizer.weight_decay=0.0 \
#     algorithm.symm_loss_weight=0.0017564103460707703 \
#     checkpoint_manifest_path=logs/train/hls4ml/jetformer/nconst16/symm_so2 \
#     results_table.invariance_metric=wasserstein \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/hls4ml/jetformer/nconst16/symm_so2

# ------------------------------------------------------------------------
# jetformer  nconst=16  symm_so3  (method=symm_loss, group=so3)
# best trial 180 of jetformer_nconst16_symm_loss_so3, val/acc_best=0.565680
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=hls4ml/jetformer_16 \
#     data.method=symm_loss \
#     data.augmentation_scheme=so3 \
#     data.n_folds=5 \
#     algorithm.dropout=0.1 \
#     algorithm.embed_dim=64 \
#     algorithm.num_heads=4 \
#     algorithm.num_transformers=3 \
#     algorithm.optimizer.lr=0.0029238763361040827 \
#     algorithm.optimizer.weight_decay=0.0001 \
#     algorithm.symm_loss_weight=0.01152708884782513 \
#     checkpoint_manifest_path=logs/train/hls4ml/jetformer/nconst16/symm_so3 \
#     results_table.invariance_metric=wasserstein \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/hls4ml/jetformer/nconst16/symm_so3

# ------------------------------------------------------------------------
# jetformer  nconst=16  symm_lorentz  (method=symm_loss, group=lorentz)
# best trial 192 of jetformer_nconst16_symm_loss_lorentz, val/acc_best=0.548159
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=hls4ml/jetformer_16 \
#     data.method=symm_loss \
#     data.augmentation_scheme=lorentz \
#     data.n_folds=5 \
#     algorithm.dropout=0.0 \
#     algorithm.embed_dim=96 \
#     algorithm.num_heads=4 \
#     algorithm.num_transformers=3 \
#     algorithm.optimizer.lr=0.00436520077309513 \
#     algorithm.optimizer.weight_decay=0.0001 \
#     algorithm.symm_loss_weight=3.296840429778099 \
#     checkpoint_manifest_path=logs/train/hls4ml/jetformer/nconst16/symm_lorentz \
#     results_table.invariance_metric=wasserstein \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/hls4ml/jetformer/nconst16/symm_lorentz

# ========================================================================
# jetformer  nconst=50
# ========================================================================

# ------------------------------------------------------------------------
# jetformer  nconst=50  baseline  (method=augmentation, group=none)
# best trial 166 of jetformer_nconst50_augmentation_none, val/acc_best=0.662958
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=hls4ml/jetformer_50 \
#     data.method=augmentation \
#     data.augmentation_scheme=none \
#     data.n_folds=5 \
#     algorithm.dropout=0.0 \
#     algorithm.embed_dim=48 \
#     algorithm.num_heads=4 \
#     algorithm.num_transformers=4 \
#     algorithm.optimizer.lr=0.003637751268919896 \
#     algorithm.optimizer.weight_decay=0.001 \
#     checkpoint_manifest_path=logs/train/hls4ml/jetformer/nconst50/baseline \
#     results_table.invariance_metric=wasserstein \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/hls4ml/jetformer/nconst50/baseline

# ------------------------------------------------------------------------
# jetformer  nconst=50  aug_so2  (method=augmentation, group=so2)
# best trial 135 of jetformer_nconst50_augmentation_so2, val/acc_best=0.623499
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=hls4ml/jetformer_50 \
#     data.method=augmentation \
#     data.augmentation_scheme=so2 \
#     data.n_folds=5 \
#     algorithm.dropout=0.0 \
#     algorithm.embed_dim=64 \
#     algorithm.num_heads=4 \
#     algorithm.num_transformers=4 \
#     algorithm.optimizer.lr=0.004183260979433826 \
#     algorithm.optimizer.weight_decay=0.05 \
#     checkpoint_manifest_path=logs/train/hls4ml/jetformer/nconst50/aug_so2 \
#     results_table.invariance_metric=wasserstein \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/hls4ml/jetformer/nconst50/aug_so2

# ------------------------------------------------------------------------
# jetformer  nconst=50  aug_so3  (method=augmentation, group=so3)
# best trial 20 of jetformer_nconst50_augmentation_so3, val/acc_best=0.476359
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=hls4ml/jetformer_50 \
#     data.method=augmentation \
#     data.augmentation_scheme=so3 \
#     data.n_folds=5 \
#     algorithm.dropout=0.2 \
#     algorithm.embed_dim=48 \
#     algorithm.num_heads=4 \
#     algorithm.num_transformers=1 \
#     algorithm.optimizer.lr=0.004115394810483319 \
#     algorithm.optimizer.weight_decay=0.001 \
#     checkpoint_manifest_path=logs/train/hls4ml/jetformer/nconst50/aug_so3 \
#     results_table.invariance_metric=wasserstein \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/hls4ml/jetformer/nconst50/aug_so3

# ------------------------------------------------------------------------
# jetformer  nconst=50  aug_lorentz  (method=augmentation, group=lorentz)
# best trial 176 of jetformer_nconst50_augmentation_lorentz, val/acc_best=0.694468
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=hls4ml/jetformer_50 \
#     data.method=augmentation \
#     data.augmentation_scheme=lorentz \
#     data.n_folds=5 \
#     algorithm.dropout=0.0 \
#     algorithm.embed_dim=64 \
#     algorithm.num_heads=1 \
#     algorithm.num_transformers=3 \
#     algorithm.optimizer.lr=0.004258066396939164 \
#     algorithm.optimizer.weight_decay=0.05 \
#     checkpoint_manifest_path=logs/train/hls4ml/jetformer/nconst50/aug_lorentz \
#     results_table.invariance_metric=wasserstein \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/hls4ml/jetformer/nconst50/aug_lorentz

# ------------------------------------------------------------------------
# jetformer  nconst=50  symm_so2  (method=symm_loss, group=so2)
# best trial 185 of jetformer_nconst50_symm_loss_so2, val/acc_best=0.615752
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=hls4ml/jetformer_50 \
#     data.method=symm_loss \
#     data.augmentation_scheme=so2 \
#     data.n_folds=5 \
#     algorithm.dropout=0.1 \
#     algorithm.embed_dim=64 \
#     algorithm.num_heads=2 \
#     algorithm.num_transformers=4 \
#     algorithm.optimizer.lr=0.003361773810619727 \
#     algorithm.optimizer.weight_decay=0.0 \
#     algorithm.symm_loss_weight=0.0053928852542188465 \
#     checkpoint_manifest_path=logs/train/hls4ml/jetformer/nconst50/symm_so2 \
#     results_table.invariance_metric=wasserstein \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/hls4ml/jetformer/nconst50/symm_so2

# ------------------------------------------------------------------------
# jetformer  nconst=50  symm_so3  (method=symm_loss, group=so3)
# best trial 197 of jetformer_nconst50_symm_loss_so3, val/acc_best=0.599009
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=hls4ml/jetformer_50 \
#     data.method=symm_loss \
#     data.augmentation_scheme=so3 \
#     data.n_folds=5 \
#     algorithm.dropout=0.0 \
#     algorithm.embed_dim=48 \
#     algorithm.num_heads=4 \
#     algorithm.num_transformers=4 \
#     algorithm.optimizer.lr=0.001067701804230896 \
#     algorithm.optimizer.weight_decay=0.0001 \
#     algorithm.symm_loss_weight=0.008411770232979543 \
#     checkpoint_manifest_path=logs/train/hls4ml/jetformer/nconst50/symm_so3 \
#     results_table.invariance_metric=wasserstein \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/hls4ml/jetformer/nconst50/symm_so3

# ------------------------------------------------------------------------
# jetformer  nconst=50  symm_lorentz  (method=symm_loss, group=lorentz)
# best trial 122 of jetformer_nconst50_symm_loss_lorentz, val/acc_best=0.626229
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=hls4ml/jetformer_50 \
#     data.method=symm_loss \
#     data.augmentation_scheme=lorentz \
#     data.n_folds=5 \
#     algorithm.dropout=0.0 \
#     algorithm.embed_dim=64 \
#     algorithm.num_heads=4 \
#     algorithm.num_transformers=4 \
#     algorithm.optimizer.lr=0.0029337520414610455 \
#     algorithm.optimizer.weight_decay=0.0001 \
#     algorithm.symm_loss_weight=0.4856924270287834 \
#     checkpoint_manifest_path=logs/train/hls4ml/jetformer/nconst50/symm_lorentz \
#     results_table.invariance_metric=wasserstein \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/hls4ml/jetformer/nconst50/symm_lorentz

# ========================================================================
# jetformer  nconst=150
# ========================================================================

# ------------------------------------------------------------------------
# jetformer  nconst=150  baseline  (method=augmentation, group=none)
# best trial 155 of jetformer_nconst150_augmentation_none, val/acc_best=0.608801
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=hls4ml/jetformer_150 \
#     data.method=augmentation \
#     data.augmentation_scheme=none \
#     data.n_folds=5 \
#     algorithm.dropout=0.0 \
#     algorithm.embed_dim=96 \
#     algorithm.num_heads=1 \
#     algorithm.num_transformers=4 \
#     algorithm.optimizer.lr=0.0007239951270893974 \
#     algorithm.optimizer.weight_decay=0.0 \
#     checkpoint_manifest_path=logs/train/hls4ml/jetformer/nconst150/baseline \
#     results_table.invariance_metric=wasserstein \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/hls4ml/jetformer/nconst150/baseline

# ------------------------------------------------------------------------
# jetformer  nconst=150  aug_so2  (method=augmentation, group=so2)
# best trial 94 of jetformer_nconst150_augmentation_so2, val/acc_best=0.679955
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=hls4ml/jetformer_150 \
#     data.method=augmentation \
#     data.augmentation_scheme=so2 \
#     data.n_folds=5 \
#     algorithm.dropout=0.0 \
#     algorithm.embed_dim=48 \
#     algorithm.num_heads=4 \
#     algorithm.num_transformers=3 \
#     algorithm.optimizer.lr=0.0014168612949381338 \
#     algorithm.optimizer.weight_decay=0.0001 \
#     checkpoint_manifest_path=logs/train/hls4ml/jetformer/nconst150/aug_so2 \
#     results_table.invariance_metric=wasserstein \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/hls4ml/jetformer/nconst150/aug_so2

# ------------------------------------------------------------------------
# jetformer  nconst=150  aug_so3  (method=augmentation, group=so3)
# best trial 194 of jetformer_nconst150_augmentation_so3, val/acc_best=0.465130
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=hls4ml/jetformer_150 \
#     data.method=augmentation \
#     data.augmentation_scheme=so3 \
#     data.n_folds=5 \
#     algorithm.dropout=0.0 \
#     algorithm.embed_dim=64 \
#     algorithm.num_heads=2 \
#     algorithm.num_transformers=4 \
#     algorithm.optimizer.lr=0.0011655948007607071 \
#     algorithm.optimizer.weight_decay=0.001 \
#     checkpoint_manifest_path=logs/train/hls4ml/jetformer/nconst150/aug_so3 \
#     results_table.invariance_metric=wasserstein \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/hls4ml/jetformer/nconst150/aug_so3

# ------------------------------------------------------------------------
# jetformer  nconst=150  aug_lorentz  (method=augmentation, group=lorentz)
# best trial 176 of jetformer_nconst150_augmentation_lorentz, val/acc_best=0.579777
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=hls4ml/jetformer_150 \
#     data.method=augmentation \
#     data.augmentation_scheme=lorentz \
#     data.n_folds=5 \
#     algorithm.dropout=0.0 \
#     algorithm.embed_dim=96 \
#     algorithm.num_heads=2 \
#     algorithm.num_transformers=3 \
#     algorithm.optimizer.lr=0.0006129940114843682 \
#     algorithm.optimizer.weight_decay=0.0001 \
#     checkpoint_manifest_path=logs/train/hls4ml/jetformer/nconst150/aug_lorentz \
#     results_table.invariance_metric=wasserstein \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/hls4ml/jetformer/nconst150/aug_lorentz

# ------------------------------------------------------------------------
# jetformer  nconst=150  symm_so2  (method=symm_loss, group=so2)
# best trial 132 of jetformer_nconst150_symm_loss_so2, val/acc_best=0.609931
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=hls4ml/jetformer_150 \
#     data.method=symm_loss \
#     data.augmentation_scheme=so2 \
#     data.n_folds=5 \
#     algorithm.dropout=0.0 \
#     algorithm.embed_dim=48 \
#     algorithm.num_heads=2 \
#     algorithm.num_transformers=4 \
#     algorithm.optimizer.lr=0.0005730487798398377 \
#     algorithm.optimizer.weight_decay=0.0001 \
#     algorithm.symm_loss_weight=0.017966227171071714 \
#     checkpoint_manifest_path=logs/train/hls4ml/jetformer/nconst150/symm_so2 \
#     results_table.invariance_metric=wasserstein \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/hls4ml/jetformer/nconst150/symm_so2

# ------------------------------------------------------------------------
# jetformer  nconst=150  symm_so3  (method=symm_loss, group=so3)
# best trial 65 of jetformer_nconst150_symm_loss_so3, val/acc_best=0.595627
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=hls4ml/jetformer_150 \
#     data.method=symm_loss \
#     data.augmentation_scheme=so3 \
#     data.n_folds=5 \
#     algorithm.dropout=0.0 \
#     algorithm.embed_dim=96 \
#     algorithm.num_heads=2 \
#     algorithm.num_transformers=3 \
#     algorithm.optimizer.lr=0.0036831319863668546 \
#     algorithm.optimizer.weight_decay=0.0 \
#     algorithm.symm_loss_weight=0.002902742120470697 \
#     checkpoint_manifest_path=logs/train/hls4ml/jetformer/nconst150/symm_so3 \
#     results_table.invariance_metric=wasserstein \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/hls4ml/jetformer/nconst150/symm_so3

# ------------------------------------------------------------------------
# jetformer  nconst=150  symm_lorentz  (method=symm_loss, group=lorentz)
# best trial 163 of jetformer_nconst150_symm_loss_lorentz, val/acc_best=0.630914
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=hls4ml/jetformer_150 \
#     data.method=symm_loss \
#     data.augmentation_scheme=lorentz \
#     data.n_folds=5 \
#     algorithm.dropout=0.0 \
#     algorithm.embed_dim=64 \
#     algorithm.num_heads=4 \
#     algorithm.num_transformers=3 \
#     algorithm.optimizer.lr=0.004383877627557186 \
#     algorithm.optimizer.weight_decay=0.01 \
#     algorithm.symm_loss_weight=0.6590342899494605 \
#     checkpoint_manifest_path=logs/train/hls4ml/jetformer/nconst150/symm_lorentz \
#     results_table.invariance_metric=wasserstein \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/hls4ml/jetformer/nconst150/symm_lorentz
