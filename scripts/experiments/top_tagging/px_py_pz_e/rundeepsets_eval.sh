#!/usr/bin/env bash
# ========================================================================
# DEEPSETS EVALUATION COMMANDS
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
# deepsets  nconst=100
# ========================================================================

# ------------------------------------------------------------------------
# deepsets  nconst=100  baseline  (method=augmentation, group=none)
# best trial 12 of top_tagging_deepsets_nconst100_augmentation_none, val/acc_best=0.889936
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=top_tagging/deepsets_100 \
#     data.method=augmentation \
#     data.augmentation_scheme=none \
#     data.n_folds=5 \
#     data.features='[px,py,pz,e]' \
#     algorithm.aggregation=max \
#     algorithm.optimizer.lr=0.002019992745562683 \
#     algorithm.optimizer.weight_decay=0.0 \
#     algorithm.phi_layers='[64,64,64]' \
#     algorithm.rho_layers='[64,32]' \
#     checkpoint_manifest_path=logs/train/top_tagging/px_py_pz_e/deepsets/nconst100/baseline \
#     results_table.invariance_metric=tv \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/top_tagging/px_py_pz_e/deepsets/nconst100/baseline

# ------------------------------------------------------------------------
# deepsets  nconst=100  aug_so2  (method=augmentation, group=so2)
# best trial 96 of top_tagging_deepsets_nconst100_augmentation_so2, val/acc_best=0.890585
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=top_tagging/deepsets_100 \
#     data.method=augmentation \
#     data.augmentation_scheme=so2 \
#     data.n_folds=5 \
#     data.features='[px,py,pz,e]' \
#     algorithm.aggregation=max \
#     algorithm.optimizer.lr=0.002951971006627239 \
#     algorithm.optimizer.weight_decay=1e-06 \
#     algorithm.phi_layers='[128,128,64]' \
#     algorithm.rho_layers='[16]' \
#     checkpoint_manifest_path=logs/train/top_tagging/px_py_pz_e/deepsets/nconst100/aug_so2 \
#     results_table.invariance_metric=tv \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/top_tagging/px_py_pz_e/deepsets/nconst100/aug_so2

# ------------------------------------------------------------------------
# deepsets  nconst=100  aug_so3  (method=augmentation, group=so3)
# best trial 95 of top_tagging_deepsets_nconst100_augmentation_so3, val/acc_best=0.885415
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=top_tagging/deepsets_100 \
#     data.method=augmentation \
#     data.augmentation_scheme=so3 \
#     data.n_folds=5 \
#     data.features='[px,py,pz,e]' \
#     algorithm.aggregation=max \
#     algorithm.optimizer.lr=0.001214754857520886 \
#     algorithm.optimizer.weight_decay=1e-07 \
#     algorithm.phi_layers='[128,128,64]' \
#     algorithm.rho_layers='[128,64]' \
#     checkpoint_manifest_path=logs/train/top_tagging/px_py_pz_e/deepsets/nconst100/aug_so3 \
#     results_table.invariance_metric=tv \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/top_tagging/px_py_pz_e/deepsets/nconst100/aug_so3

# ------------------------------------------------------------------------
# deepsets  nconst=100  aug_lorentz  (method=augmentation, group=lorentz)
# best trial 10 of top_tagging_deepsets_nconst100_augmentation_lorentz, val/acc_best=0.889387
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=top_tagging/deepsets_100 \
#     data.method=augmentation \
#     data.augmentation_scheme=lorentz \
#     data.n_folds=5 \
#     data.features='[px,py,pz,e]' \
#     algorithm.aggregation=max \
#     algorithm.optimizer.lr=0.0037851086154109376 \
#     algorithm.optimizer.weight_decay=0.0 \
#     algorithm.phi_layers='[128,128,64]' \
#     algorithm.rho_layers='[32]' \
#     checkpoint_manifest_path=logs/train/top_tagging/px_py_pz_e/deepsets/nconst100/aug_lorentz \
#     results_table.invariance_metric=tv \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/top_tagging/px_py_pz_e/deepsets/nconst100/aug_lorentz

# ------------------------------------------------------------------------
# deepsets  nconst=100  symm_so2  (method=symm_loss, group=so2)
# best trial 87 of top_tagging_deepsets_nconst100_symm_loss_so2, val/acc_best=0.890682
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=top_tagging/deepsets_100 \
#     data.method=symm_loss \
#     data.augmentation_scheme=so2 \
#     data.n_folds=5 \
#     data.features='[px,py,pz,e]' \
#     algorithm.aggregation=max \
#     algorithm.optimizer.lr=0.0024442830767831014 \
#     algorithm.optimizer.weight_decay=1e-06 \
#     algorithm.phi_layers='[64,64,64]' \
#     algorithm.rho_layers='[32,32,16]' \
#     algorithm.symm_loss_weight=0.0028957876961962356 \
#     checkpoint_manifest_path=logs/train/top_tagging/px_py_pz_e/deepsets/nconst100/symm_so2 \
#     results_table.invariance_metric=tv \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/top_tagging/px_py_pz_e/deepsets/nconst100/symm_so2

# ------------------------------------------------------------------------
# deepsets  nconst=100  symm_so3  (method=symm_loss, group=so3)
# best trial 72 of top_tagging_deepsets_nconst100_symm_loss_so3, val/acc_best=0.890446
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=top_tagging/deepsets_100 \
#     data.method=symm_loss \
#     data.augmentation_scheme=so3 \
#     data.n_folds=5 \
#     data.features='[px,py,pz,e]' \
#     algorithm.aggregation=max \
#     algorithm.optimizer.lr=0.0023984621405352285 \
#     algorithm.optimizer.weight_decay=1e-06 \
#     algorithm.phi_layers='[128,128,64]' \
#     algorithm.rho_layers='[32,16]' \
#     algorithm.symm_loss_weight=0.02065089597780597 \
#     checkpoint_manifest_path=logs/train/top_tagging/px_py_pz_e/deepsets/nconst100/symm_so3 \
#     results_table.invariance_metric=tv \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/top_tagging/px_py_pz_e/deepsets/nconst100/symm_so3

# ------------------------------------------------------------------------
# deepsets  nconst=100  symm_lorentz  (method=symm_loss, group=lorentz)
# best trial 56 of top_tagging_deepsets_nconst100_symm_loss_lorentz, val/acc_best=0.890774
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=top_tagging/deepsets_100 \
#     data.method=symm_loss \
#     data.augmentation_scheme=lorentz \
#     data.n_folds=5 \
#     data.features='[px,py,pz,e]' \
#     algorithm.aggregation=max \
#     algorithm.optimizer.lr=0.0028329558613602963 \
#     algorithm.optimizer.weight_decay=1e-07 \
#     algorithm.phi_layers='[128,128,64]' \
#     algorithm.rho_layers='[32]' \
#     algorithm.symm_loss_weight=0.005668341846373623 \
#     checkpoint_manifest_path=logs/train/top_tagging/px_py_pz_e/deepsets/nconst100/symm_lorentz \
#     results_table.invariance_metric=tv \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/top_tagging/px_py_pz_e/deepsets/nconst100/symm_lorentz
