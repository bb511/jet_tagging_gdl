#!/usr/bin/env bash
# ========================================================================
# JETFORMER EVALUATION COMMANDS
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
# jetformer  nconst=100
# ========================================================================

# ------------------------------------------------------------------------
# jetformer  nconst=100  baseline  (method=augmentation, group=none)
# best trial 87 of top_tagging_jetformer_nconst100_augmentation_none, val/acc_best=0.931077
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=top_tagging/jetformer_100 \
#     data.method=augmentation \
#     data.augmentation_scheme=none \
#     data.n_folds=5 \
#     data.features='[px,py,pz,e]' \
#     algorithm.dropout=0.0 \
#     algorithm.embed_dim=48 \
#     algorithm.num_heads=8 \
#     algorithm.num_transformers=5 \
#     algorithm.optimizer.lr=0.0043469506414629535 \
#     algorithm.optimizer.weight_decay=0.01 \
#     checkpoint_manifest_path=logs/train/top_tagging/px_py_pz_e/jetformer/nconst100/baseline \
#     results_table.invariance_metric=tv \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/top_tagging/px_py_pz_e/jetformer/nconst100/baseline

# ------------------------------------------------------------------------
# jetformer  nconst=100  aug_so2  (method=augmentation, group=so2)
# best trial 88 of top_tagging_jetformer_nconst100_augmentation_so2, val/acc_best=0.933147
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=top_tagging/jetformer_100 \
#     data.method=augmentation \
#     data.augmentation_scheme=so2 \
#     data.n_folds=5 \
#     data.features='[px,py,pz,e]' \
#     algorithm.dropout=0.0 \
#     algorithm.embed_dim=64 \
#     algorithm.num_heads=8 \
#     algorithm.num_transformers=5 \
#     algorithm.optimizer.lr=0.008984210262547957 \
#     algorithm.optimizer.weight_decay=0.001 \
#     checkpoint_manifest_path=logs/train/top_tagging/px_py_pz_e/jetformer/nconst100/aug_so2 \
#     results_table.invariance_metric=tv \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/top_tagging/px_py_pz_e/jetformer/nconst100/aug_so2

# ------------------------------------------------------------------------
# jetformer  nconst=100  aug_so3  (method=augmentation, group=so3)
# best trial 87 of top_tagging_jetformer_nconst100_augmentation_so3, val/acc_best=0.930951
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=top_tagging/jetformer_100 \
#     data.method=augmentation \
#     data.augmentation_scheme=so3 \
#     data.n_folds=5 \
#     data.features='[px,py,pz,e]' \
#     algorithm.dropout=0.1 \
#     algorithm.embed_dim=128 \
#     algorithm.num_heads=8 \
#     algorithm.num_transformers=4 \
#     algorithm.optimizer.lr=0.004794318797789713 \
#     algorithm.optimizer.weight_decay=0.001 \
#     checkpoint_manifest_path=logs/train/top_tagging/px_py_pz_e/jetformer/nconst100/aug_so3 \
#     results_table.invariance_metric=tv \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/top_tagging/px_py_pz_e/jetformer/nconst100/aug_so3

# ------------------------------------------------------------------------
# jetformer  nconst=100  aug_lorentz  (method=augmentation, group=lorentz)
# best trial 80 of top_tagging_jetformer_nconst100_augmentation_lorentz, val/acc_best=0.933281
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=top_tagging/jetformer_100 \
#     data.method=augmentation \
#     data.augmentation_scheme=lorentz \
#     data.n_folds=5 \
#     data.features='[px,py,pz,e]' \
#     algorithm.dropout=0.0 \
#     algorithm.embed_dim=64 \
#     algorithm.num_heads=8 \
#     algorithm.num_transformers=5 \
#     algorithm.optimizer.lr=0.012241681850747073 \
#     algorithm.optimizer.weight_decay=0.01 \
#     checkpoint_manifest_path=logs/train/top_tagging/px_py_pz_e/jetformer/nconst100/aug_lorentz \
#     results_table.invariance_metric=tv \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/top_tagging/px_py_pz_e/jetformer/nconst100/aug_lorentz

# ------------------------------------------------------------------------
# jetformer  nconst=100  symm_so2  (method=symm_loss, group=so2)
# best trial 39 of top_tagging_jetformer_nconst100_symm_loss_so2, val/acc_best=0.931829
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=top_tagging/jetformer_100 \
#     data.method=symm_loss \
#     data.augmentation_scheme=so2 \
#     data.n_folds=5 \
#     data.features='[px,py,pz,e]' \
#     algorithm.dropout=0.0 \
#     algorithm.embed_dim=96 \
#     algorithm.num_heads=8 \
#     algorithm.num_transformers=5 \
#     algorithm.optimizer.lr=0.007017773461516878 \
#     algorithm.optimizer.weight_decay=0.01 \
#     algorithm.symm_loss_weight=0.016728194669748397 \
#     checkpoint_manifest_path=logs/train/top_tagging/px_py_pz_e/jetformer/nconst100/symm_so2 \
#     results_table.invariance_metric=tv \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/top_tagging/px_py_pz_e/jetformer/nconst100/symm_so2

# ------------------------------------------------------------------------
# jetformer  nconst=100  symm_so3  (method=symm_loss, group=so3)
# best trial 75 of top_tagging_jetformer_nconst100_symm_loss_so3, val/acc_best=0.927684
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=top_tagging/jetformer_100 \
#     data.method=symm_loss \
#     data.augmentation_scheme=so3 \
#     data.n_folds=5 \
#     data.features='[px,py,pz,e]' \
#     algorithm.dropout=0.1 \
#     algorithm.embed_dim=96 \
#     algorithm.num_heads=4 \
#     algorithm.num_transformers=3 \
#     algorithm.optimizer.lr=0.0045124435945967534 \
#     algorithm.optimizer.weight_decay=0.001 \
#     algorithm.symm_loss_weight=0.024647733909911748 \
#     checkpoint_manifest_path=logs/train/top_tagging/px_py_pz_e/jetformer/nconst100/symm_so3 \
#     results_table.invariance_metric=tv \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/top_tagging/px_py_pz_e/jetformer/nconst100/symm_so3

# ------------------------------------------------------------------------
# jetformer  nconst=100  symm_lorentz  (method=symm_loss, group=lorentz)
# best trial 91 of top_tagging_jetformer_nconst100_symm_loss_lorentz, val/acc_best=0.931012
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=top_tagging/jetformer_100 \
#     data.method=symm_loss \
#     data.augmentation_scheme=lorentz \
#     data.n_folds=5 \
#     data.features='[px,py,pz,e]' \
#     algorithm.dropout=0.0 \
#     algorithm.embed_dim=64 \
#     algorithm.num_heads=8 \
#     algorithm.num_transformers=5 \
#     algorithm.optimizer.lr=0.006649643302475974 \
#     algorithm.optimizer.weight_decay=0.01 \
#     algorithm.symm_loss_weight=0.003662237508668213 \
#     checkpoint_manifest_path=logs/train/top_tagging/px_py_pz_e/jetformer/nconst100/symm_lorentz \
#     results_table.invariance_metric=tv \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/top_tagging/px_py_pz_e/jetformer/nconst100/symm_lorentz
