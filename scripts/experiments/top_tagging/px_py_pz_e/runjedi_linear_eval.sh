#!/usr/bin/env bash
# ========================================================================
# JEDI_LINEAR EVALUATION COMMANDS
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
# jedi_linear  nconst=100
# ========================================================================

# ------------------------------------------------------------------------
# jedi_linear  nconst=100  baseline  (method=augmentation, group=none)
# best trial 99 of top_tagging_jedi_linear_nconst100_augmentation_none, val/acc_best=0.895681
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=top_tagging/jedi_linear_100 \
#     data.method=augmentation \
#     data.augmentation_scheme=none \
#     data.n_folds=5 \
#     data.features='[px,py,pz,e]' \
#     algorithm.aggregation=max \
#     algorithm.head_layers='[128,64,32]' \
#     algorithm.hidden=256 \
#     algorithm.optimizer.lr=0.0028111464296847965 \
#     algorithm.optimizer.weight_decay=0.0 \
#     checkpoint_manifest_path=logs/train/top_tagging/px_py_pz_e/jedi_linear/nconst100/baseline \
#     results_table.invariance_metric=tv \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/top_tagging/px_py_pz_e/jedi_linear/nconst100/baseline

# ------------------------------------------------------------------------
# jedi_linear  nconst=100  aug_so2  (method=augmentation, group=so2)
# best trial 97 of top_tagging_jedi_linear_nconst100_augmentation_so2, val/acc_best=0.897706
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=top_tagging/jedi_linear_100 \
#     data.method=augmentation \
#     data.augmentation_scheme=so2 \
#     data.n_folds=5 \
#     data.features='[px,py,pz,e]' \
#     algorithm.aggregation=max \
#     algorithm.head_layers='[128,64,32]' \
#     algorithm.hidden=256 \
#     algorithm.optimizer.lr=0.0020502587449817305 \
#     algorithm.optimizer.weight_decay=1e-07 \
#     checkpoint_manifest_path=logs/train/top_tagging/px_py_pz_e/jedi_linear/nconst100/aug_so2 \
#     results_table.invariance_metric=tv \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/top_tagging/px_py_pz_e/jedi_linear/nconst100/aug_so2

# ------------------------------------------------------------------------
# jedi_linear  nconst=100  aug_so3  (method=augmentation, group=so3)
# best trial 82 of top_tagging_jedi_linear_nconst100_augmentation_so3, val/acc_best=0.893344
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=top_tagging/jedi_linear_100 \
#     data.method=augmentation \
#     data.augmentation_scheme=so3 \
#     data.n_folds=5 \
#     data.features='[px,py,pz,e]' \
#     algorithm.aggregation=max \
#     algorithm.head_layers='[128,64,32]' \
#     algorithm.hidden=256 \
#     algorithm.optimizer.lr=0.002267583894611829 \
#     algorithm.optimizer.weight_decay=1e-07 \
#     checkpoint_manifest_path=logs/train/top_tagging/px_py_pz_e/jedi_linear/nconst100/aug_so3 \
#     results_table.invariance_metric=tv \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/top_tagging/px_py_pz_e/jedi_linear/nconst100/aug_so3

# ------------------------------------------------------------------------
# jedi_linear  nconst=100  aug_lorentz  (method=augmentation, group=lorentz)
# best trial 83 of top_tagging_jedi_linear_nconst100_augmentation_lorentz, val/acc_best=0.912339
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=top_tagging/jedi_linear_100 \
#     data.method=augmentation \
#     data.augmentation_scheme=lorentz \
#     data.n_folds=5 \
#     data.features='[px,py,pz,e]' \
#     algorithm.aggregation=sum_over_n \
#     algorithm.head_layers='[128,64,32]' \
#     algorithm.hidden=256 \
#     algorithm.optimizer.lr=0.0027847853220214535 \
#     algorithm.optimizer.weight_decay=0.0 \
#     checkpoint_manifest_path=logs/train/top_tagging/px_py_pz_e/jedi_linear/nconst100/aug_lorentz \
#     results_table.invariance_metric=tv \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/top_tagging/px_py_pz_e/jedi_linear/nconst100/aug_lorentz

# ------------------------------------------------------------------------
# jedi_linear  nconst=100  symm_so2  (method=symm_loss, group=so2)
# best trial 64 of top_tagging_jedi_linear_nconst100_symm_loss_so2, val/acc_best=0.910696
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=top_tagging/jedi_linear_100 \
#     data.method=symm_loss \
#     data.augmentation_scheme=so2 \
#     data.n_folds=5 \
#     data.features='[px,py,pz,e]' \
#     algorithm.aggregation=sum_over_n \
#     algorithm.head_layers='[64,32,16]' \
#     algorithm.hidden=256 \
#     algorithm.optimizer.lr=0.005048515740569621 \
#     algorithm.optimizer.weight_decay=1e-07 \
#     algorithm.symm_loss_weight=0.016493978716712397 \
#     checkpoint_manifest_path=logs/train/top_tagging/px_py_pz_e/jedi_linear/nconst100/symm_so2 \
#     results_table.invariance_metric=tv \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/top_tagging/px_py_pz_e/jedi_linear/nconst100/symm_so2

# ------------------------------------------------------------------------
# jedi_linear  nconst=100  symm_so3  (method=symm_loss, group=so3)
# best trial 87 of top_tagging_jedi_linear_nconst100_symm_loss_so3, val/acc_best=0.906651
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=top_tagging/jedi_linear_100 \
#     data.method=symm_loss \
#     data.augmentation_scheme=so3 \
#     data.n_folds=5 \
#     data.features='[px,py,pz,e]' \
#     algorithm.aggregation=sum_over_n \
#     algorithm.head_layers='[128,64,32]' \
#     algorithm.hidden=256 \
#     algorithm.optimizer.lr=0.0027373406859394337 \
#     algorithm.optimizer.weight_decay=1e-07 \
#     algorithm.symm_loss_weight=0.0034222446595207896 \
#     checkpoint_manifest_path=logs/train/top_tagging/px_py_pz_e/jedi_linear/nconst100/symm_so3 \
#     results_table.invariance_metric=tv \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/top_tagging/px_py_pz_e/jedi_linear/nconst100/symm_so3

# ------------------------------------------------------------------------
# jedi_linear  nconst=100  symm_lorentz  (method=symm_loss, group=lorentz)
# best trial 76 of top_tagging_jedi_linear_nconst100_symm_loss_lorentz, val/acc_best=0.904435
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=top_tagging/jedi_linear_100 \
#     data.method=symm_loss \
#     data.augmentation_scheme=lorentz \
#     data.n_folds=5 \
#     data.features='[px,py,pz,e]' \
#     algorithm.aggregation=sum_over_n \
#     algorithm.head_layers='[32,32,16]' \
#     algorithm.hidden=128 \
#     algorithm.optimizer.lr=0.004810180017924878 \
#     algorithm.optimizer.weight_decay=0.0 \
#     algorithm.symm_loss_weight=0.014105108112816608 \
#     checkpoint_manifest_path=logs/train/top_tagging/px_py_pz_e/jedi_linear/nconst100/symm_lorentz \
#     results_table.invariance_metric=tv \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/top_tagging/px_py_pz_e/jedi_linear/nconst100/symm_lorentz
