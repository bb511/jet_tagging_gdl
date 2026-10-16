#!/usr/bin/env bash
# ========================================================================
# JEDI_LINEAR EVALUATION COMMANDS
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
# jedi_linear  nconst=100
# ========================================================================

# ------------------------------------------------------------------------
# jedi_linear  nconst=100  baseline  (method=augmentation, group=none)
# best trial 98 of top_tagging_jedi_linear_nconst100_augmentation_none, val/acc_best=0.913375
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=top_tagging/jedi_linear_100 \
#     data.method=augmentation \
#     data.augmentation_scheme=none \
#     data.n_folds=5 \
#     data.features='[pt,eta,phi]' \
#     algorithm.aggregation=sum_over_n \
#     algorithm.head_layers='[64,32]' \
#     algorithm.hidden=256 \
#     algorithm.optimizer.lr=0.0027490138896887804 \
#     algorithm.optimizer.weight_decay=1e-06 \
#     checkpoint_manifest_path=logs/train/top_tagging/pt_eta_phi/jedi_linear/nconst100/baseline \
#     results_table.invariance_metric=tv \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/top_tagging/pt_eta_phi/jedi_linear/nconst100/baseline

# ------------------------------------------------------------------------
# jedi_linear  nconst=100  aug_so2  (method=augmentation, group=so2)
# best trial 66 of top_tagging_jedi_linear_nconst100_augmentation_so2, val/acc_best=0.912983
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=top_tagging/jedi_linear_100 \
#     data.method=augmentation \
#     data.augmentation_scheme=so2 \
#     data.n_folds=5 \
#     data.features='[pt,eta,phi]' \
#     algorithm.aggregation=sum_over_n \
#     algorithm.head_layers='[128,64,32]' \
#     algorithm.hidden=256 \
#     algorithm.optimizer.lr=0.0009579279350006218 \
#     algorithm.optimizer.weight_decay=0.0 \
#     checkpoint_manifest_path=logs/train/top_tagging/pt_eta_phi/jedi_linear/nconst100/aug_so2 \
#     results_table.invariance_metric=tv \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/top_tagging/pt_eta_phi/jedi_linear/nconst100/aug_so2

# ------------------------------------------------------------------------
# jedi_linear  nconst=100  aug_so3  (method=augmentation, group=so3)
# best trial 96 of top_tagging_jedi_linear_nconst100_augmentation_so3, val/acc_best=0.914183
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=top_tagging/jedi_linear_100 \
#     data.method=augmentation \
#     data.augmentation_scheme=so3 \
#     data.n_folds=5 \
#     data.features='[pt,eta,phi]' \
#     algorithm.aggregation=sum_over_n \
#     algorithm.head_layers='[128,64,32]' \
#     algorithm.hidden=256 \
#     algorithm.optimizer.lr=0.0022212854515915245 \
#     algorithm.optimizer.weight_decay=0.0 \
#     checkpoint_manifest_path=logs/train/top_tagging/pt_eta_phi/jedi_linear/nconst100/aug_so3 \
#     results_table.invariance_metric=tv \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/top_tagging/pt_eta_phi/jedi_linear/nconst100/aug_so3

# ------------------------------------------------------------------------
# jedi_linear  nconst=100  aug_lorentz  (method=augmentation, group=lorentz)
# best trial 99 of top_tagging_jedi_linear_nconst100_augmentation_lorentz, val/acc_best=0.906178
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=top_tagging/jedi_linear_100 \
#     data.method=augmentation \
#     data.augmentation_scheme=lorentz \
#     data.n_folds=5 \
#     data.features='[pt,eta,phi]' \
#     algorithm.aggregation=sum_over_n \
#     algorithm.head_layers='[32,32,16]' \
#     algorithm.hidden=256 \
#     algorithm.optimizer.lr=0.0019267588820390425 \
#     algorithm.optimizer.weight_decay=1e-06 \
#     checkpoint_manifest_path=logs/train/top_tagging/pt_eta_phi/jedi_linear/nconst100/aug_lorentz \
#     results_table.invariance_metric=tv \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/top_tagging/pt_eta_phi/jedi_linear/nconst100/aug_lorentz

# ------------------------------------------------------------------------
# jedi_linear  nconst=100  symm_so2  (method=symm_loss, group=so2)
# best trial 86 of top_tagging_jedi_linear_nconst100_symm_loss_so2, val/acc_best=0.914690
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=top_tagging/jedi_linear_100 \
#     data.method=symm_loss \
#     data.augmentation_scheme=so2 \
#     data.n_folds=5 \
#     data.features='[pt,eta,phi]' \
#     algorithm.aggregation=sum_over_n \
#     algorithm.head_layers='[128,64,32]' \
#     algorithm.hidden=256 \
#     algorithm.optimizer.lr=0.0034171218516182485 \
#     algorithm.optimizer.weight_decay=1e-07 \
#     algorithm.symm_loss_weight=0.005590113037612279 \
#     checkpoint_manifest_path=logs/train/top_tagging/pt_eta_phi/jedi_linear/nconst100/symm_so2 \
#     results_table.invariance_metric=tv \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/top_tagging/pt_eta_phi/jedi_linear/nconst100/symm_so2

# ------------------------------------------------------------------------
# jedi_linear  nconst=100  symm_so3  (method=symm_loss, group=so3)
# best trial 58 of top_tagging_jedi_linear_nconst100_symm_loss_so3, val/acc_best=0.917893
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=top_tagging/jedi_linear_100 \
#     data.method=symm_loss \
#     data.augmentation_scheme=so3 \
#     data.n_folds=5 \
#     data.features='[pt,eta,phi]' \
#     algorithm.aggregation=sum_over_n \
#     algorithm.head_layers='[128,64,32]' \
#     algorithm.hidden=256 \
#     algorithm.optimizer.lr=0.003069571808219641 \
#     algorithm.optimizer.weight_decay=1e-07 \
#     algorithm.symm_loss_weight=0.049881402344431294 \
#     checkpoint_manifest_path=logs/train/top_tagging/pt_eta_phi/jedi_linear/nconst100/symm_so3 \
#     results_table.invariance_metric=tv \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/top_tagging/pt_eta_phi/jedi_linear/nconst100/symm_so3

# ------------------------------------------------------------------------
# jedi_linear  nconst=100  symm_lorentz  (method=symm_loss, group=lorentz)
# best trial 98 of top_tagging_jedi_linear_nconst100_symm_loss_lorentz, val/acc_best=0.914459
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=top_tagging/jedi_linear_100 \
#     data.method=symm_loss \
#     data.augmentation_scheme=lorentz \
#     data.n_folds=5 \
#     data.features='[pt,eta,phi]' \
#     algorithm.aggregation=sum_over_n \
#     algorithm.head_layers='[64,32,16]' \
#     algorithm.hidden=256 \
#     algorithm.optimizer.lr=0.0024182152758687485 \
#     algorithm.optimizer.weight_decay=1e-06 \
#     algorithm.symm_loss_weight=0.0018106217591132546 \
#     checkpoint_manifest_path=logs/train/top_tagging/pt_eta_phi/jedi_linear/nconst100/symm_lorentz \
#     results_table.invariance_metric=tv \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/top_tagging/pt_eta_phi/jedi_linear/nconst100/symm_lorentz
