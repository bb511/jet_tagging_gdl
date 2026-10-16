#!/usr/bin/env bash
# ========================================================================
# JEDI_LINEAR EVALUATION COMMANDS
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
# jedi_linear  nconst=100
# ========================================================================

# ------------------------------------------------------------------------
# jedi_linear  nconst=100  baseline  (method=augmentation, group=none)
# best trial 93 of top_tagging_jedi_linear_nconst100_augmentation_none, val/acc_best=0.929546
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=top_tagging/jedi_linear_100 \
#     data.method=augmentation \
#     data.augmentation_scheme=none \
#     data.n_folds=5 \
#     algorithm.aggregation=sum_over_n \
#     algorithm.head_layers='[64,32]' \
#     algorithm.hidden=256 \
#     algorithm.optimizer.lr=0.0023575868730685524 \
#     algorithm.optimizer.weight_decay=0.0 \
#     checkpoint_manifest_path=logs/train/top_tagging/jedi_linear/nconst100/baseline \
#     results_table.invariance_metric=tv \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/top_tagging/jedi_linear/nconst100/baseline

# ------------------------------------------------------------------------
# jedi_linear  nconst=100  aug_so2  (method=augmentation, group=so2)
# best trial 57 of top_tagging_jedi_linear_nconst100_augmentation_so2, val/acc_best=0.929802
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=top_tagging/jedi_linear_100 \
#     data.method=augmentation \
#     data.augmentation_scheme=so2 \
#     data.n_folds=5 \
#     algorithm.aggregation=sum_over_n \
#     algorithm.head_layers='[128,64,32]' \
#     algorithm.hidden=128 \
#     algorithm.optimizer.lr=0.0037441384983086883 \
#     algorithm.optimizer.weight_decay=1e-06 \
#     checkpoint_manifest_path=logs/train/top_tagging/jedi_linear/nconst100/aug_so2 \
#     results_table.invariance_metric=tv \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/top_tagging/jedi_linear/nconst100/aug_so2

# ------------------------------------------------------------------------
# jedi_linear  nconst=100  aug_so3  (method=augmentation, group=so3)
# best trial 28 of top_tagging_jedi_linear_nconst100_augmentation_so3, val/acc_best=0.929038
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=top_tagging/jedi_linear_100 \
#     data.method=augmentation \
#     data.augmentation_scheme=so3 \
#     data.n_folds=5 \
#     algorithm.aggregation=sum_over_n \
#     algorithm.head_layers='[128,64,32]' \
#     algorithm.hidden=128 \
#     algorithm.optimizer.lr=0.0029543894794126745 \
#     algorithm.optimizer.weight_decay=1e-07 \
#     checkpoint_manifest_path=logs/train/top_tagging/jedi_linear/nconst100/aug_so3 \
#     results_table.invariance_metric=tv \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/top_tagging/jedi_linear/nconst100/aug_so3

# ------------------------------------------------------------------------
# jedi_linear  nconst=100  aug_lorentz  (method=augmentation, group=lorentz)
# best trial 76 of top_tagging_jedi_linear_nconst100_augmentation_lorentz, val/acc_best=0.930468
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=top_tagging/jedi_linear_100 \
#     data.method=augmentation \
#     data.augmentation_scheme=lorentz \
#     data.n_folds=5 \
#     algorithm.aggregation=sum_over_n \
#     algorithm.head_layers='[128,64,32]' \
#     algorithm.hidden=256 \
#     algorithm.optimizer.lr=0.0021367599120210015 \
#     algorithm.optimizer.weight_decay=0.0 \
#     checkpoint_manifest_path=logs/train/top_tagging/jedi_linear/nconst100/aug_lorentz \
#     results_table.invariance_metric=tv \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/top_tagging/jedi_linear/nconst100/aug_lorentz

# ------------------------------------------------------------------------
# jedi_linear  nconst=100  symm_so2  (method=symm_loss, group=so2)
# best trial 22 of top_tagging_jedi_linear_nconst100_symm_loss_so2, val/acc_best=0.929782
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=top_tagging/jedi_linear_100 \
#     data.method=symm_loss \
#     data.augmentation_scheme=so2 \
#     data.n_folds=5 \
#     algorithm.aggregation=sum_over_n \
#     algorithm.head_layers='[64,32,16]' \
#     algorithm.hidden=256 \
#     algorithm.optimizer.lr=0.005283849816063953 \
#     algorithm.optimizer.weight_decay=0.0 \
#     algorithm.symm_loss_weight=0.8938248275879042 \
#     checkpoint_manifest_path=logs/train/top_tagging/jedi_linear/nconst100/symm_so2 \
#     results_table.invariance_metric=tv \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/top_tagging/jedi_linear/nconst100/symm_so2

# ------------------------------------------------------------------------
# jedi_linear  nconst=100  symm_so3  (method=symm_loss, group=so3)
# best trial 95 of top_tagging_jedi_linear_nconst100_symm_loss_so3, val/acc_best=0.929152
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=top_tagging/jedi_linear_100 \
#     data.method=symm_loss \
#     data.augmentation_scheme=so3 \
#     data.n_folds=5 \
#     algorithm.aggregation=sum \
#     algorithm.head_layers='[64,32]' \
#     algorithm.hidden=256 \
#     algorithm.optimizer.lr=0.0013908920067592981 \
#     algorithm.optimizer.weight_decay=1e-07 \
#     algorithm.symm_loss_weight=0.004013627259059247 \
#     checkpoint_manifest_path=logs/train/top_tagging/jedi_linear/nconst100/symm_so3 \
#     results_table.invariance_metric=tv \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/top_tagging/jedi_linear/nconst100/symm_so3

# ------------------------------------------------------------------------
# jedi_linear  nconst=100  symm_lorentz  (method=symm_loss, group=lorentz)
# best trial 0 of top_tagging_jedi_linear_nconst100_symm_loss_lorentz, val/acc_best=0.928942
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=top_tagging/jedi_linear_100 \
#     data.method=symm_loss \
#     data.augmentation_scheme=lorentz \
#     data.n_folds=5 \
#     algorithm.aggregation=sum_over_n \
#     algorithm.head_layers='[64,32,16]' \
#     algorithm.hidden=128 \
#     algorithm.optimizer.lr=0.008641030091777241 \
#     algorithm.optimizer.weight_decay=0.0 \
#     algorithm.symm_loss_weight=0.12837651116271678 \
#     checkpoint_manifest_path=logs/train/top_tagging/jedi_linear/nconst100/symm_lorentz \
#     results_table.invariance_metric=tv \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/top_tagging/jedi_linear/nconst100/symm_lorentz
