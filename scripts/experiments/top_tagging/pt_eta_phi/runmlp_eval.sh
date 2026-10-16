#!/usr/bin/env bash
# ========================================================================
# MLP EVALUATION COMMANDS
# ========================================================================
# One block per (constituent count, regime): 1 x 7 = 7 eval runs. Generated
# by scripts/experiments/make_train_catalogues.py; regenerate rather than edit by hand.
# Feature set [pt, eta, phi]: studies from logs/optuna/top_tagging/pt_eta_phi/, runs under logs/{train,eval}/top_tagging/pt_eta_phi/.
# Each block carries the best of the first 100 completed trials, by trial
# number, of its own study under logs/optuna/top_tagging/pt_eta_phi/;
# a block whose study is missing carries none and says so (transferred ablation).
#
# Every command is commented out: copy one to run it by hand, or feed the whole
# file to scripts/experiments/submit_eval.sh, which runs them under nohup.

# ========================================================================
# mlp  nconst=100
# ========================================================================

# ------------------------------------------------------------------------
# mlp  nconst=100  baseline  (method=augmentation, group=none)
# best trial 97 of top_tagging_mlp_nconst100_augmentation_none, val/acc_best=0.889073
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=top_tagging/mlp_100 \
#     data.method=augmentation \
#     data.augmentation_scheme=none \
#     data.n_folds=5 \
#     data.features='[pt,eta,phi]' \
#     algorithm.batchnorm=False \
#     algorithm.hidden_layers='[256,256,128]' \
#     algorithm.optimizer.lr=0.002815199290675051 \
#     algorithm.optimizer.weight_decay=1e-05 \
#     checkpoint_manifest_path=logs/train/top_tagging/pt_eta_phi/mlp/nconst100/baseline \
#     results_table.invariance_metric=tv \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/top_tagging/pt_eta_phi/mlp/nconst100/baseline

# ------------------------------------------------------------------------
# mlp  nconst=100  aug_so2  (method=augmentation, group=so2)
# best trial 73 of top_tagging_mlp_nconst100_augmentation_so2, val/acc_best=0.898836
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=top_tagging/mlp_100 \
#     data.method=augmentation \
#     data.augmentation_scheme=so2 \
#     data.n_folds=5 \
#     data.features='[pt,eta,phi]' \
#     algorithm.batchnorm=True \
#     algorithm.hidden_layers='[256,256,128]' \
#     algorithm.optimizer.lr=0.0026921525037885623 \
#     algorithm.optimizer.weight_decay=0.0 \
#     checkpoint_manifest_path=logs/train/top_tagging/pt_eta_phi/mlp/nconst100/aug_so2 \
#     results_table.invariance_metric=tv \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/top_tagging/pt_eta_phi/mlp/nconst100/aug_so2

# ------------------------------------------------------------------------
# mlp  nconst=100  aug_so3  (method=augmentation, group=so3)
# best trial 55 of top_tagging_mlp_nconst100_augmentation_so3, val/acc_best=0.895620
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=top_tagging/mlp_100 \
#     data.method=augmentation \
#     data.augmentation_scheme=so3 \
#     data.n_folds=5 \
#     data.features='[pt,eta,phi]' \
#     algorithm.batchnorm=False \
#     algorithm.hidden_layers='[256,256,128]' \
#     algorithm.optimizer.lr=0.0022155375080580323 \
#     algorithm.optimizer.weight_decay=1e-07 \
#     checkpoint_manifest_path=logs/train/top_tagging/pt_eta_phi/mlp/nconst100/aug_so3 \
#     results_table.invariance_metric=tv \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/top_tagging/pt_eta_phi/mlp/nconst100/aug_so3

# ------------------------------------------------------------------------
# mlp  nconst=100  aug_lorentz  (method=augmentation, group=lorentz)
# best trial 44 of top_tagging_mlp_nconst100_augmentation_lorentz, val/acc_best=0.890931
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=top_tagging/mlp_100 \
#     data.method=augmentation \
#     data.augmentation_scheme=lorentz \
#     data.n_folds=5 \
#     data.features='[pt,eta,phi]' \
#     algorithm.batchnorm=True \
#     algorithm.hidden_layers='[256,128,64]' \
#     algorithm.optimizer.lr=0.001749013657866682 \
#     algorithm.optimizer.weight_decay=1e-05 \
#     checkpoint_manifest_path=logs/train/top_tagging/pt_eta_phi/mlp/nconst100/aug_lorentz \
#     results_table.invariance_metric=tv \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/top_tagging/pt_eta_phi/mlp/nconst100/aug_lorentz

# ------------------------------------------------------------------------
# mlp  nconst=100  symm_so2  (method=symm_loss, group=so2)
# best trial 97 of top_tagging_mlp_nconst100_symm_loss_so2, val/acc_best=0.897230
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=top_tagging/mlp_100 \
#     data.method=symm_loss \
#     data.augmentation_scheme=so2 \
#     data.n_folds=5 \
#     data.features='[pt,eta,phi]' \
#     algorithm.batchnorm=True \
#     algorithm.hidden_layers='[256,256,128]' \
#     algorithm.optimizer.lr=0.00045861714972164635 \
#     algorithm.optimizer.weight_decay=0.0001 \
#     algorithm.symm_loss_weight=0.012895319197468634 \
#     checkpoint_manifest_path=logs/train/top_tagging/pt_eta_phi/mlp/nconst100/symm_so2 \
#     results_table.invariance_metric=tv \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/top_tagging/pt_eta_phi/mlp/nconst100/symm_so2

# ------------------------------------------------------------------------
# mlp  nconst=100  symm_so3  (method=symm_loss, group=so3)
# best trial 97 of top_tagging_mlp_nconst100_symm_loss_so3, val/acc_best=0.892367
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=top_tagging/mlp_100 \
#     data.method=symm_loss \
#     data.augmentation_scheme=so3 \
#     data.n_folds=5 \
#     data.features='[pt,eta,phi]' \
#     algorithm.batchnorm=True \
#     algorithm.hidden_layers='[256,256,128]' \
#     algorithm.optimizer.lr=0.001792206797944809 \
#     algorithm.optimizer.weight_decay=1e-05 \
#     algorithm.symm_loss_weight=0.09702299441188574 \
#     checkpoint_manifest_path=logs/train/top_tagging/pt_eta_phi/mlp/nconst100/symm_so3 \
#     results_table.invariance_metric=tv \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/top_tagging/pt_eta_phi/mlp/nconst100/symm_so3

# ------------------------------------------------------------------------
# mlp  nconst=100  symm_lorentz  (method=symm_loss, group=lorentz)
# best trial 76 of top_tagging_mlp_nconst100_symm_loss_lorentz, val/acc_best=0.894932
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=top_tagging/mlp_100 \
#     data.method=symm_loss \
#     data.augmentation_scheme=lorentz \
#     data.n_folds=5 \
#     data.features='[pt,eta,phi]' \
#     algorithm.batchnorm=True \
#     algorithm.hidden_layers='[256,256,128]' \
#     algorithm.optimizer.lr=0.0003538291477780032 \
#     algorithm.optimizer.weight_decay=0.0001 \
#     algorithm.symm_loss_weight=0.2655503523460386 \
#     checkpoint_manifest_path=logs/train/top_tagging/pt_eta_phi/mlp/nconst100/symm_lorentz \
#     results_table.invariance_metric=tv \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/top_tagging/pt_eta_phi/mlp/nconst100/symm_lorentz
