#!/usr/bin/env bash
# ========================================================================
# JEDI_LINEAR_MOMENTS EVALUATION COMMANDS
# ========================================================================
# One block per (constituent count, regime): 3 x 1 = 3 eval runs. Generated
# by scripts/optuna/make_train_catalogues.py; regenerate rather than edit by hand.
# Baseline only: this architecture embeds the symmetry rather than acquiring it,
# so the augmentation and SCL regimes are not run on it.
# Each block carries the best of the first 200 completed trials, by trial
# number, of its own study under logs/optuna/hls4ml/;
# a block whose study is missing carries none and says so (transferred ablation).
#
# Every command is commented out: copy one to run it by hand, or feed the whole
# file to scripts/cluster/submit_eval.sh, which queues them on a GPU or slurm.

# ========================================================================
# jedi_linear_moments  nconst=16
# ========================================================================

# ------------------------------------------------------------------------
# jedi_linear_moments  nconst=16  baseline  (method=augmentation, group=none)
# best trial 66 of jedi_linear_moments_nconst16_augmentation_none, val/acc_best=0.721637
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=hls4ml/jedi_linear_moments_16 \
#     data.method=augmentation \
#     data.augmentation_scheme=none \
#     data.n_folds=5 \
#     algorithm.aggregation=max \
#     algorithm.head_layers='[64,32,16]' \
#     algorithm.hidden=128 \
#     algorithm.moments=uniform \
#     algorithm.optimizer.lr=0.0033563666242975 \
#     algorithm.optimizer.weight_decay=1e-05 \
#     checkpoint_manifest_path=logs/train/hls4ml/jedi_linear_moments/nconst16/baseline \
#     results_table.invariance_metric=tv \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/hls4ml/jedi_linear_moments/nconst16/baseline

# ========================================================================
# jedi_linear_moments  nconst=50
# ========================================================================

# ------------------------------------------------------------------------
# jedi_linear_moments  nconst=50  baseline  (method=augmentation, group=none)
# best trial 152 of jedi_linear_moments_nconst50_augmentation_none, val/acc_best=0.818689
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=hls4ml/jedi_linear_moments_50 \
#     data.method=augmentation \
#     data.augmentation_scheme=none \
#     data.n_folds=5 \
#     algorithm.aggregation=max \
#     algorithm.head_layers='[64,32]' \
#     algorithm.hidden=128 \
#     algorithm.moments=rankcut \
#     algorithm.optimizer.lr=0.003660625075400208 \
#     algorithm.optimizer.weight_decay=0.0 \
#     checkpoint_manifest_path=logs/train/hls4ml/jedi_linear_moments/nconst50/baseline \
#     results_table.invariance_metric=tv \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/hls4ml/jedi_linear_moments/nconst50/baseline

# ========================================================================
# jedi_linear_moments  nconst=150
# ========================================================================

# ------------------------------------------------------------------------
# jedi_linear_moments  nconst=150  baseline  (method=augmentation, group=none)
# best trial 117 of jedi_linear_moments_nconst150_augmentation_none, val/acc_best=0.824451
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=hls4ml/jedi_linear_moments_150 \
#     data.method=augmentation \
#     data.augmentation_scheme=none \
#     data.n_folds=5 \
#     algorithm.aggregation=sum_over_n \
#     algorithm.head_layers='[64,32]' \
#     algorithm.hidden=128 \
#     algorithm.moments=rankcut \
#     algorithm.optimizer.lr=0.0077387789876816725 \
#     algorithm.optimizer.weight_decay=0.0 \
#     checkpoint_manifest_path=logs/train/hls4ml/jedi_linear_moments/nconst150/baseline \
#     results_table.invariance_metric=tv \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/hls4ml/jedi_linear_moments/nconst150/baseline
