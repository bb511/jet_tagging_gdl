#!/usr/bin/env bash
# ========================================================================
# JETFORMER EVALUATION COMMANDS
# ========================================================================
# One block per (constituent count, regime): 1 x 7 = 7 eval runs. Generated
# by scripts/experiments/make_train_catalogues.py; regenerate rather than edit by hand.
# Each block carries the best of the first 100 completed trials, by trial
# number, of its own study under logs/optuna/top_tagging/;
# a block whose study is missing carries none and says so (transferred ablation).
#
# Every command is commented out: copy one to run it by hand, or feed the whole
# file to scripts/experiments/submit_eval.sh, which runs them under nohup.

# ========================================================================
# jetformer  nconst=100
# ========================================================================

# ------------------------------------------------------------------------
# jetformer  nconst=100  baseline  (method=augmentation, group=none)
# best trial 62 of top_tagging_jetformer_nconst100_augmentation_none, val/acc_best=0.935455
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=top_tagging/jetformer_100 \
#     data.method=augmentation \
#     data.augmentation_scheme=none \
#     data.n_folds=5 \
#     algorithm.dropout=0.0 \
#     algorithm.embed_dim=32 \
#     algorithm.num_heads=8 \
#     algorithm.num_transformers=5 \
#     algorithm.optimizer.lr=0.012479868098589031 \
#     algorithm.optimizer.weight_decay=0.0001 \
#     checkpoint_manifest_path=logs/train/top_tagging/jetformer/nconst100/baseline \
#     results_table.invariance_metric=tv \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/top_tagging/jetformer/nconst100/baseline

# ------------------------------------------------------------------------
# jetformer  nconst=100  aug_so2  (method=augmentation, group=so2)
# best trial 94 of top_tagging_jetformer_nconst100_augmentation_so2, val/acc_best=0.936452
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=top_tagging/jetformer_100 \
#     data.method=augmentation \
#     data.augmentation_scheme=so2 \
#     data.n_folds=5 \
#     algorithm.dropout=0.0 \
#     algorithm.embed_dim=48 \
#     algorithm.num_heads=8 \
#     algorithm.num_transformers=4 \
#     algorithm.optimizer.lr=0.008298903761259134 \
#     algorithm.optimizer.weight_decay=0.05 \
#     checkpoint_manifest_path=logs/train/top_tagging/jetformer/nconst100/aug_so2 \
#     results_table.invariance_metric=tv \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/top_tagging/jetformer/nconst100/aug_so2

# ------------------------------------------------------------------------
# jetformer  nconst=100  aug_so3  (method=augmentation, group=so3)
# best trial 97 of top_tagging_jetformer_nconst100_augmentation_so3, val/acc_best=0.937155
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=top_tagging/jetformer_100 \
#     data.method=augmentation \
#     data.augmentation_scheme=so3 \
#     data.n_folds=5 \
#     algorithm.dropout=0.0 \
#     algorithm.embed_dim=128 \
#     algorithm.num_heads=8 \
#     algorithm.num_transformers=4 \
#     algorithm.optimizer.lr=0.004337878230839487 \
#     algorithm.optimizer.weight_decay=0.01 \
#     checkpoint_manifest_path=logs/train/top_tagging/jetformer/nconst100/aug_so3 \
#     results_table.invariance_metric=tv \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/top_tagging/jetformer/nconst100/aug_so3

# ------------------------------------------------------------------------
# jetformer  nconst=100  aug_lorentz  (method=augmentation, group=lorentz)
# best trial 91 of top_tagging_jetformer_nconst100_augmentation_lorentz, val/acc_best=0.936474
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=top_tagging/jetformer_100 \
#     data.method=augmentation \
#     data.augmentation_scheme=lorentz \
#     data.n_folds=5 \
#     algorithm.dropout=0.0 \
#     algorithm.embed_dim=64 \
#     algorithm.num_heads=8 \
#     algorithm.num_transformers=5 \
#     algorithm.optimizer.lr=0.005602498855848973 \
#     algorithm.optimizer.weight_decay=0.05 \
#     checkpoint_manifest_path=logs/train/top_tagging/jetformer/nconst100/aug_lorentz \
#     results_table.invariance_metric=tv \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/top_tagging/jetformer/nconst100/aug_lorentz

# ------------------------------------------------------------------------
# jetformer  nconst=100  symm_so2  (method=symm_loss, group=so2)
# best trial 48 of top_tagging_jetformer_nconst100_symm_loss_so2, val/acc_best=0.936383
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=top_tagging/jetformer_100 \
#     data.method=symm_loss \
#     data.augmentation_scheme=so2 \
#     data.n_folds=5 \
#     algorithm.dropout=0.0 \
#     algorithm.embed_dim=48 \
#     algorithm.num_heads=8 \
#     algorithm.num_transformers=4 \
#     algorithm.optimizer.lr=0.009248600366250297 \
#     algorithm.optimizer.weight_decay=0.01 \
#     algorithm.symm_loss_weight=0.058440764180865426 \
#     checkpoint_manifest_path=logs/train/top_tagging/jetformer/nconst100/symm_so2 \
#     results_table.invariance_metric=tv \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/top_tagging/jetformer/nconst100/symm_so2

# ------------------------------------------------------------------------
# jetformer  nconst=100  symm_so3  (method=symm_loss, group=so3)
# best trial 98 of top_tagging_jetformer_nconst100_symm_loss_so3, val/acc_best=0.938124
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=top_tagging/jetformer_100 \
#     data.method=symm_loss \
#     data.augmentation_scheme=so3 \
#     data.n_folds=5 \
#     algorithm.dropout=0.0 \
#     algorithm.embed_dim=64 \
#     algorithm.num_heads=4 \
#     algorithm.num_transformers=5 \
#     algorithm.optimizer.lr=0.004009572123809718 \
#     algorithm.optimizer.weight_decay=0.05 \
#     algorithm.symm_loss_weight=0.11405498709776797 \
#     checkpoint_manifest_path=logs/train/top_tagging/jetformer/nconst100/symm_so3 \
#     results_table.invariance_metric=tv \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/top_tagging/jetformer/nconst100/symm_so3

# ------------------------------------------------------------------------
# jetformer  nconst=100  symm_lorentz  (method=symm_loss, group=lorentz)
# best trial 99 of top_tagging_jetformer_nconst100_symm_loss_lorentz, val/acc_best=0.935660
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=top_tagging/jetformer_100 \
#     data.method=symm_loss \
#     data.augmentation_scheme=lorentz \
#     data.n_folds=5 \
#     algorithm.dropout=0.0 \
#     algorithm.embed_dim=48 \
#     algorithm.num_heads=8 \
#     algorithm.num_transformers=4 \
#     algorithm.optimizer.lr=0.010353390999549811 \
#     algorithm.optimizer.weight_decay=0.0001 \
#     algorithm.symm_loss_weight=0.012946635828284916 \
#     checkpoint_manifest_path=logs/train/top_tagging/jetformer/nconst100/symm_lorentz \
#     results_table.invariance_metric=tv \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/top_tagging/jetformer/nconst100/symm_lorentz
