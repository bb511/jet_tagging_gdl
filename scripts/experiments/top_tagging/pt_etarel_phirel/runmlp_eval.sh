#!/usr/bin/env bash
# ========================================================================
# MLP EVALUATION COMMANDS
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
# mlp  nconst=100
# ========================================================================

# ------------------------------------------------------------------------
# mlp  nconst=100  baseline  (method=augmentation, group=none)
# best trial 78 of top_tagging_mlp_nconst100_augmentation_none, val/acc_best=0.914842
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=top_tagging/mlp_100 \
#     data.method=augmentation \
#     data.augmentation_scheme=none \
#     data.n_folds=5 \
#     algorithm.batchnorm=True \
#     algorithm.hidden_layers='[256,256,128]' \
#     algorithm.optimizer.lr=0.022414869015640243 \
#     algorithm.optimizer.weight_decay=1e-06 \
#     checkpoint_manifest_path=logs/train/top_tagging/mlp/nconst100/baseline \
#     results_table.invariance_metric=tv \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/top_tagging/mlp/nconst100/baseline

# ------------------------------------------------------------------------
# mlp  nconst=100  aug_so2  (method=augmentation, group=so2)
# best trial 28 of top_tagging_mlp_nconst100_augmentation_so2, val/acc_best=0.913842
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=top_tagging/mlp_100 \
#     data.method=augmentation \
#     data.augmentation_scheme=so2 \
#     data.n_folds=5 \
#     algorithm.batchnorm=True \
#     algorithm.hidden_layers='[256,256,128]' \
#     algorithm.optimizer.lr=0.002685780571681667 \
#     algorithm.optimizer.weight_decay=0.0001 \
#     checkpoint_manifest_path=logs/train/top_tagging/mlp/nconst100/aug_so2 \
#     results_table.invariance_metric=tv \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/top_tagging/mlp/nconst100/aug_so2

# ------------------------------------------------------------------------
# mlp  nconst=100  aug_so3  (method=augmentation, group=so3)
# best trial 85 of top_tagging_mlp_nconst100_augmentation_so3, val/acc_best=0.922334
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=top_tagging/mlp_100 \
#     data.method=augmentation \
#     data.augmentation_scheme=so3 \
#     data.n_folds=5 \
#     algorithm.batchnorm=True \
#     algorithm.hidden_layers='[256,256,128]' \
#     algorithm.optimizer.lr=0.007314676824295033 \
#     algorithm.optimizer.weight_decay=0.0 \
#     checkpoint_manifest_path=logs/train/top_tagging/mlp/nconst100/aug_so3 \
#     results_table.invariance_metric=tv \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/top_tagging/mlp/nconst100/aug_so3

# ------------------------------------------------------------------------
# mlp  nconst=100  aug_lorentz  (method=augmentation, group=lorentz)
# best trial 69 of top_tagging_mlp_nconst100_augmentation_lorentz, val/acc_best=0.913741
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=top_tagging/mlp_100 \
#     data.method=augmentation \
#     data.augmentation_scheme=lorentz \
#     data.n_folds=5 \
#     algorithm.batchnorm=True \
#     algorithm.hidden_layers='[64,64,32]' \
#     algorithm.optimizer.lr=0.011043474445096907 \
#     algorithm.optimizer.weight_decay=0.0 \
#     checkpoint_manifest_path=logs/train/top_tagging/mlp/nconst100/aug_lorentz \
#     results_table.invariance_metric=tv \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/top_tagging/mlp/nconst100/aug_lorentz

# ------------------------------------------------------------------------
# mlp  nconst=100  symm_so2  (method=symm_loss, group=so2)
# best trial 43 of top_tagging_mlp_nconst100_symm_loss_so2, val/acc_best=0.914748
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=top_tagging/mlp_100 \
#     data.method=symm_loss \
#     data.augmentation_scheme=so2 \
#     data.n_folds=5 \
#     algorithm.batchnorm=True \
#     algorithm.hidden_layers='[128,64,32]' \
#     algorithm.optimizer.lr=0.022913694488088714 \
#     algorithm.optimizer.weight_decay=0.0 \
#     algorithm.symm_loss_weight=9.409404234867528 \
#     checkpoint_manifest_path=logs/train/top_tagging/mlp/nconst100/symm_so2 \
#     results_table.invariance_metric=tv \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/top_tagging/mlp/nconst100/symm_so2

# ------------------------------------------------------------------------
# mlp  nconst=100  symm_so3  (method=symm_loss, group=so3)
# best trial 25 of top_tagging_mlp_nconst100_symm_loss_so3, val/acc_best=0.919934
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=top_tagging/mlp_100 \
#     data.method=symm_loss \
#     data.augmentation_scheme=so3 \
#     data.n_folds=5 \
#     algorithm.batchnorm=False \
#     algorithm.hidden_layers='[256,128,64]' \
#     algorithm.optimizer.lr=0.0024695935195930686 \
#     algorithm.optimizer.weight_decay=0.0 \
#     algorithm.symm_loss_weight=0.4184871412968669 \
#     checkpoint_manifest_path=logs/train/top_tagging/mlp/nconst100/symm_so3 \
#     results_table.invariance_metric=tv \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/top_tagging/mlp/nconst100/symm_so3

# ------------------------------------------------------------------------
# mlp  nconst=100  symm_lorentz  (method=symm_loss, group=lorentz)
# best trial 78 of top_tagging_mlp_nconst100_symm_loss_lorentz, val/acc_best=0.914988
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=top_tagging/mlp_100 \
#     data.method=symm_loss \
#     data.augmentation_scheme=lorentz \
#     data.n_folds=5 \
#     algorithm.batchnorm=True \
#     algorithm.hidden_layers='[256,128,64]' \
#     algorithm.optimizer.lr=0.008177671920042017 \
#     algorithm.optimizer.weight_decay=1e-05 \
#     algorithm.symm_loss_weight=1.879347828379632 \
#     checkpoint_manifest_path=logs/train/top_tagging/mlp/nconst100/symm_lorentz \
#     results_table.invariance_metric=tv \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/top_tagging/mlp/nconst100/symm_lorentz
