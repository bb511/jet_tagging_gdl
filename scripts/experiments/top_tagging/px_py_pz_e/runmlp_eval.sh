#!/usr/bin/env bash
# ========================================================================
# MLP EVALUATION COMMANDS
# ========================================================================
# One block per (constituent count, regime): 1 x 7 = 7 eval runs. Generated
# by scripts/experiments/make_train_catalogues.py; regenerate rather than edit by hand.
# Feature set [px, py, pz, e]: studies from logs/optuna/top_tagging/px_py_pz_e/, runs under logs/{train,eval}/top_tagging/px_py_pz_e/.
# Each block carries the best of the first 100 completed trials, by trial
# number, of its own study under logs/optuna/top_tagging/px_py_pz_e/;
# a block whose study is missing carries none and says so (transferred ablation).
#
# Every command is commented out: copy one to run it by hand, or feed the whole
# file to scripts/experiments/submit_eval.sh, which runs them under nohup.

# ========================================================================
# mlp  nconst=100
# ========================================================================

# ------------------------------------------------------------------------
# mlp  nconst=100  baseline  (method=augmentation, group=none)
# best trial 31 of top_tagging_mlp_nconst100_augmentation_none, val/acc_best=0.887788
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=top_tagging/mlp_100 \
#     data.method=augmentation \
#     data.augmentation_scheme=none \
#     data.n_folds=5 \
#     data.features='[px,py,pz,e]' \
#     algorithm.batchnorm=False \
#     algorithm.hidden_layers='[256,128,64]' \
#     algorithm.optimizer.lr=0.00016181146004597548 \
#     algorithm.optimizer.weight_decay=1e-06 \
#     checkpoint_manifest_path=logs/train/top_tagging/px_py_pz_e/mlp/nconst100/baseline \
#     results_table.invariance_metric=tv \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/top_tagging/px_py_pz_e/mlp/nconst100/baseline

# ------------------------------------------------------------------------
# mlp  nconst=100  aug_so2  (method=augmentation, group=so2)
# best trial 68 of top_tagging_mlp_nconst100_augmentation_so2, val/acc_best=0.893965
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=top_tagging/mlp_100 \
#     data.method=augmentation \
#     data.augmentation_scheme=so2 \
#     data.n_folds=5 \
#     data.features='[px,py,pz,e]' \
#     algorithm.batchnorm=True \
#     algorithm.hidden_layers='[256,256,128]' \
#     algorithm.optimizer.lr=0.0011611552887454215 \
#     algorithm.optimizer.weight_decay=0.0 \
#     checkpoint_manifest_path=logs/train/top_tagging/px_py_pz_e/mlp/nconst100/aug_so2 \
#     results_table.invariance_metric=tv \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/top_tagging/px_py_pz_e/mlp/nconst100/aug_so2

# ------------------------------------------------------------------------
# mlp  nconst=100  aug_so3  (method=augmentation, group=so3)
# best trial 43 of top_tagging_mlp_nconst100_augmentation_so3, val/acc_best=0.888048
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=top_tagging/mlp_100 \
#     data.method=augmentation \
#     data.augmentation_scheme=so3 \
#     data.n_folds=5 \
#     data.features='[px,py,pz,e]' \
#     algorithm.batchnorm=False \
#     algorithm.hidden_layers='[256,128,64]' \
#     algorithm.optimizer.lr=0.00018091106557292016 \
#     algorithm.optimizer.weight_decay=1e-05 \
#     checkpoint_manifest_path=logs/train/top_tagging/px_py_pz_e/mlp/nconst100/aug_so3 \
#     results_table.invariance_metric=tv \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/top_tagging/px_py_pz_e/mlp/nconst100/aug_so3

# ------------------------------------------------------------------------
# mlp  nconst=100  aug_lorentz  (method=augmentation, group=lorentz)
# best trial 55 of top_tagging_mlp_nconst100_augmentation_lorentz, val/acc_best=0.890696
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=top_tagging/mlp_100 \
#     data.method=augmentation \
#     data.augmentation_scheme=lorentz \
#     data.n_folds=5 \
#     data.features='[px,py,pz,e]' \
#     algorithm.batchnorm=True \
#     algorithm.hidden_layers='[256,256,128]' \
#     algorithm.optimizer.lr=0.001228909435518509 \
#     algorithm.optimizer.weight_decay=1e-06 \
#     checkpoint_manifest_path=logs/train/top_tagging/px_py_pz_e/mlp/nconst100/aug_lorentz \
#     results_table.invariance_metric=tv \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/top_tagging/px_py_pz_e/mlp/nconst100/aug_lorentz

# ------------------------------------------------------------------------
# mlp  nconst=100  symm_so2  (method=symm_loss, group=so2)
# best trial 24 of top_tagging_mlp_nconst100_symm_loss_so2, val/acc_best=0.893179
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=top_tagging/mlp_100 \
#     data.method=symm_loss \
#     data.augmentation_scheme=so2 \
#     data.n_folds=5 \
#     data.features='[px,py,pz,e]' \
#     algorithm.batchnorm=True \
#     algorithm.hidden_layers='[256,256,128]' \
#     algorithm.optimizer.lr=0.0029317050066283235 \
#     algorithm.optimizer.weight_decay=0.0 \
#     algorithm.symm_loss_weight=0.13190081624244276 \
#     checkpoint_manifest_path=logs/train/top_tagging/px_py_pz_e/mlp/nconst100/symm_so2 \
#     results_table.invariance_metric=tv \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/top_tagging/px_py_pz_e/mlp/nconst100/symm_so2

# ------------------------------------------------------------------------
# mlp  nconst=100  symm_so3  (method=symm_loss, group=so3)
# best trial 94 of top_tagging_mlp_nconst100_symm_loss_so3, val/acc_best=0.891825
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=top_tagging/mlp_100 \
#     data.method=symm_loss \
#     data.augmentation_scheme=so3 \
#     data.n_folds=5 \
#     data.features='[px,py,pz,e]' \
#     algorithm.batchnorm=True \
#     algorithm.hidden_layers='[256,256,128]' \
#     algorithm.optimizer.lr=0.0011800720131626957 \
#     algorithm.optimizer.weight_decay=1e-07 \
#     algorithm.symm_loss_weight=0.10481620671743568 \
#     checkpoint_manifest_path=logs/train/top_tagging/px_py_pz_e/mlp/nconst100/symm_so3 \
#     results_table.invariance_metric=tv \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/top_tagging/px_py_pz_e/mlp/nconst100/symm_so3

# ------------------------------------------------------------------------
# mlp  nconst=100  symm_lorentz  (method=symm_loss, group=lorentz)
# best trial 67 of top_tagging_mlp_nconst100_symm_loss_lorentz, val/acc_best=0.888451
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=top_tagging/mlp_100 \
#     data.method=symm_loss \
#     data.augmentation_scheme=lorentz \
#     data.n_folds=5 \
#     data.features='[px,py,pz,e]' \
#     algorithm.batchnorm=True \
#     algorithm.hidden_layers='[256,256,128]' \
#     algorithm.optimizer.lr=0.0009881216750885274 \
#     algorithm.optimizer.weight_decay=1e-06 \
#     algorithm.symm_loss_weight=0.070826304090537 \
#     checkpoint_manifest_path=logs/train/top_tagging/px_py_pz_e/mlp/nconst100/symm_lorentz \
#     results_table.invariance_metric=tv \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/top_tagging/px_py_pz_e/mlp/nconst100/symm_lorentz
