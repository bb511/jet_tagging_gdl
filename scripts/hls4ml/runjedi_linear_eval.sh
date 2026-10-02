#!/usr/bin/env bash
# ========================================================================
# JEDI_LINEAR EVALUATION COMMANDS
# ========================================================================
# One block per (constituent count, regime): 3 x 7 = 21 eval runs. Generated
# by scripts/optuna/make_train_catalogues.py; regenerate rather than edit by hand.
# Each block carries the best trial of its own study under logs/optuna/hls4ml/;
# a block whose study is missing carries none and says so (transferred ablation).
#
# Every command is commented out: copy one to run it by hand, or feed the whole
# file to scripts/cluster/submit_eval.sh, which queues them on a GPU or slurm.

# ========================================================================
# jedi_linear  nconst=16
# ========================================================================

# ------------------------------------------------------------------------
# jedi_linear  nconst=16  baseline  (method=augmentation, group=none)
# transferred from experiment config (no study yet)
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=hls4ml/jedi_linear_16 \
#     data.method=augmentation \
#     data.augmentation_scheme=none \
#     data.n_folds=5 \
#     checkpoint_manifest_path=logs/train/hls4ml/jedi_linear/nconst16/baseline \
#     results_table.invariance_metric=wasserstein \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/hls4ml/jedi_linear/nconst16/baseline

# ------------------------------------------------------------------------
# jedi_linear  nconst=16  aug_so2  (method=augmentation, group=so2)
# transferred from experiment config (no study yet)
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=hls4ml/jedi_linear_16 \
#     data.method=augmentation \
#     data.augmentation_scheme=so2 \
#     data.n_folds=5 \
#     checkpoint_manifest_path=logs/train/hls4ml/jedi_linear/nconst16/aug_so2 \
#     results_table.invariance_metric=wasserstein \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/hls4ml/jedi_linear/nconst16/aug_so2

# ------------------------------------------------------------------------
# jedi_linear  nconst=16  aug_so3  (method=augmentation, group=so3)
# transferred from experiment config (no study yet)
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=hls4ml/jedi_linear_16 \
#     data.method=augmentation \
#     data.augmentation_scheme=so3 \
#     data.n_folds=5 \
#     checkpoint_manifest_path=logs/train/hls4ml/jedi_linear/nconst16/aug_so3 \
#     results_table.invariance_metric=wasserstein \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/hls4ml/jedi_linear/nconst16/aug_so3

# ------------------------------------------------------------------------
# jedi_linear  nconst=16  aug_lorentz  (method=augmentation, group=lorentz)
# transferred from experiment config (no study yet)
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=hls4ml/jedi_linear_16 \
#     data.method=augmentation \
#     data.augmentation_scheme=lorentz \
#     data.n_folds=5 \
#     checkpoint_manifest_path=logs/train/hls4ml/jedi_linear/nconst16/aug_lorentz \
#     results_table.invariance_metric=wasserstein \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/hls4ml/jedi_linear/nconst16/aug_lorentz

# ------------------------------------------------------------------------
# jedi_linear  nconst=16  symm_so2  (method=symm_loss, group=so2)
# transferred from experiment config (no study yet)
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=hls4ml/jedi_linear_16 \
#     data.method=symm_loss \
#     data.augmentation_scheme=so2 \
#     data.n_folds=5 \
#     checkpoint_manifest_path=logs/train/hls4ml/jedi_linear/nconst16/symm_so2 \
#     results_table.invariance_metric=wasserstein \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/hls4ml/jedi_linear/nconst16/symm_so2

# ------------------------------------------------------------------------
# jedi_linear  nconst=16  symm_so3  (method=symm_loss, group=so3)
# transferred from experiment config (no study yet)
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=hls4ml/jedi_linear_16 \
#     data.method=symm_loss \
#     data.augmentation_scheme=so3 \
#     data.n_folds=5 \
#     checkpoint_manifest_path=logs/train/hls4ml/jedi_linear/nconst16/symm_so3 \
#     results_table.invariance_metric=wasserstein \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/hls4ml/jedi_linear/nconst16/symm_so3

# ------------------------------------------------------------------------
# jedi_linear  nconst=16  symm_lorentz  (method=symm_loss, group=lorentz)
# transferred from experiment config (no study yet)
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=hls4ml/jedi_linear_16 \
#     data.method=symm_loss \
#     data.augmentation_scheme=lorentz \
#     data.n_folds=5 \
#     checkpoint_manifest_path=logs/train/hls4ml/jedi_linear/nconst16/symm_lorentz \
#     results_table.invariance_metric=wasserstein \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/hls4ml/jedi_linear/nconst16/symm_lorentz

# ========================================================================
# jedi_linear  nconst=50
# ========================================================================

# ------------------------------------------------------------------------
# jedi_linear  nconst=50  baseline  (method=augmentation, group=none)
# transferred from experiment config (no study yet)
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=hls4ml/jedi_linear_50 \
#     data.method=augmentation \
#     data.augmentation_scheme=none \
#     data.n_folds=5 \
#     checkpoint_manifest_path=logs/train/hls4ml/jedi_linear/nconst50/baseline \
#     results_table.invariance_metric=wasserstein \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/hls4ml/jedi_linear/nconst50/baseline

# ------------------------------------------------------------------------
# jedi_linear  nconst=50  aug_so2  (method=augmentation, group=so2)
# transferred from experiment config (no study yet)
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=hls4ml/jedi_linear_50 \
#     data.method=augmentation \
#     data.augmentation_scheme=so2 \
#     data.n_folds=5 \
#     checkpoint_manifest_path=logs/train/hls4ml/jedi_linear/nconst50/aug_so2 \
#     results_table.invariance_metric=wasserstein \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/hls4ml/jedi_linear/nconst50/aug_so2

# ------------------------------------------------------------------------
# jedi_linear  nconst=50  aug_so3  (method=augmentation, group=so3)
# transferred from experiment config (no study yet)
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=hls4ml/jedi_linear_50 \
#     data.method=augmentation \
#     data.augmentation_scheme=so3 \
#     data.n_folds=5 \
#     checkpoint_manifest_path=logs/train/hls4ml/jedi_linear/nconst50/aug_so3 \
#     results_table.invariance_metric=wasserstein \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/hls4ml/jedi_linear/nconst50/aug_so3

# ------------------------------------------------------------------------
# jedi_linear  nconst=50  aug_lorentz  (method=augmentation, group=lorentz)
# transferred from experiment config (no study yet)
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=hls4ml/jedi_linear_50 \
#     data.method=augmentation \
#     data.augmentation_scheme=lorentz \
#     data.n_folds=5 \
#     checkpoint_manifest_path=logs/train/hls4ml/jedi_linear/nconst50/aug_lorentz \
#     results_table.invariance_metric=wasserstein \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/hls4ml/jedi_linear/nconst50/aug_lorentz

# ------------------------------------------------------------------------
# jedi_linear  nconst=50  symm_so2  (method=symm_loss, group=so2)
# transferred from experiment config (no study yet)
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=hls4ml/jedi_linear_50 \
#     data.method=symm_loss \
#     data.augmentation_scheme=so2 \
#     data.n_folds=5 \
#     checkpoint_manifest_path=logs/train/hls4ml/jedi_linear/nconst50/symm_so2 \
#     results_table.invariance_metric=wasserstein \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/hls4ml/jedi_linear/nconst50/symm_so2

# ------------------------------------------------------------------------
# jedi_linear  nconst=50  symm_so3  (method=symm_loss, group=so3)
# transferred from experiment config (no study yet)
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=hls4ml/jedi_linear_50 \
#     data.method=symm_loss \
#     data.augmentation_scheme=so3 \
#     data.n_folds=5 \
#     checkpoint_manifest_path=logs/train/hls4ml/jedi_linear/nconst50/symm_so3 \
#     results_table.invariance_metric=wasserstein \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/hls4ml/jedi_linear/nconst50/symm_so3

# ------------------------------------------------------------------------
# jedi_linear  nconst=50  symm_lorentz  (method=symm_loss, group=lorentz)
# transferred from experiment config (no study yet)
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=hls4ml/jedi_linear_50 \
#     data.method=symm_loss \
#     data.augmentation_scheme=lorentz \
#     data.n_folds=5 \
#     checkpoint_manifest_path=logs/train/hls4ml/jedi_linear/nconst50/symm_lorentz \
#     results_table.invariance_metric=wasserstein \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/hls4ml/jedi_linear/nconst50/symm_lorentz

# ========================================================================
# jedi_linear  nconst=150
# ========================================================================

# ------------------------------------------------------------------------
# jedi_linear  nconst=150  baseline  (method=augmentation, group=none)
# transferred from experiment config (no study yet)
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=hls4ml/jedi_linear_150 \
#     data.method=augmentation \
#     data.augmentation_scheme=none \
#     data.n_folds=5 \
#     checkpoint_manifest_path=logs/train/hls4ml/jedi_linear/nconst150/baseline \
#     results_table.invariance_metric=wasserstein \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/hls4ml/jedi_linear/nconst150/baseline

# ------------------------------------------------------------------------
# jedi_linear  nconst=150  aug_so2  (method=augmentation, group=so2)
# transferred from experiment config (no study yet)
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=hls4ml/jedi_linear_150 \
#     data.method=augmentation \
#     data.augmentation_scheme=so2 \
#     data.n_folds=5 \
#     checkpoint_manifest_path=logs/train/hls4ml/jedi_linear/nconst150/aug_so2 \
#     results_table.invariance_metric=wasserstein \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/hls4ml/jedi_linear/nconst150/aug_so2

# ------------------------------------------------------------------------
# jedi_linear  nconst=150  aug_so3  (method=augmentation, group=so3)
# transferred from experiment config (no study yet)
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=hls4ml/jedi_linear_150 \
#     data.method=augmentation \
#     data.augmentation_scheme=so3 \
#     data.n_folds=5 \
#     checkpoint_manifest_path=logs/train/hls4ml/jedi_linear/nconst150/aug_so3 \
#     results_table.invariance_metric=wasserstein \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/hls4ml/jedi_linear/nconst150/aug_so3

# ------------------------------------------------------------------------
# jedi_linear  nconst=150  aug_lorentz  (method=augmentation, group=lorentz)
# transferred from experiment config (no study yet)
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=hls4ml/jedi_linear_150 \
#     data.method=augmentation \
#     data.augmentation_scheme=lorentz \
#     data.n_folds=5 \
#     checkpoint_manifest_path=logs/train/hls4ml/jedi_linear/nconst150/aug_lorentz \
#     results_table.invariance_metric=wasserstein \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/hls4ml/jedi_linear/nconst150/aug_lorentz

# ------------------------------------------------------------------------
# jedi_linear  nconst=150  symm_so2  (method=symm_loss, group=so2)
# transferred from experiment config (no study yet)
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=hls4ml/jedi_linear_150 \
#     data.method=symm_loss \
#     data.augmentation_scheme=so2 \
#     data.n_folds=5 \
#     checkpoint_manifest_path=logs/train/hls4ml/jedi_linear/nconst150/symm_so2 \
#     results_table.invariance_metric=wasserstein \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/hls4ml/jedi_linear/nconst150/symm_so2

# ------------------------------------------------------------------------
# jedi_linear  nconst=150  symm_so3  (method=symm_loss, group=so3)
# transferred from experiment config (no study yet)
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=hls4ml/jedi_linear_150 \
#     data.method=symm_loss \
#     data.augmentation_scheme=so3 \
#     data.n_folds=5 \
#     checkpoint_manifest_path=logs/train/hls4ml/jedi_linear/nconst150/symm_so3 \
#     results_table.invariance_metric=wasserstein \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/hls4ml/jedi_linear/nconst150/symm_so3

# ------------------------------------------------------------------------
# jedi_linear  nconst=150  symm_lorentz  (method=symm_loss, group=lorentz)
# transferred from experiment config (no study yet)
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=hls4ml/jedi_linear_150 \
#     data.method=symm_loss \
#     data.augmentation_scheme=lorentz \
#     data.n_folds=5 \
#     checkpoint_manifest_path=logs/train/hls4ml/jedi_linear/nconst150/symm_lorentz \
#     results_table.invariance_metric=wasserstein \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/hls4ml/jedi_linear/nconst150/symm_lorentz
