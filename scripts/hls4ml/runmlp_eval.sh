#!/usr/bin/env bash
# ========================================================================
# MLP EVALUATION COMMANDS
# ========================================================================
# One block per (constituent count, regime): 3 x 7 = 21 eval runs.
#
# Generated from the study databases under logs/optuna/hls4ml/ by
# scripts/optuna/make_train_catalogues.py; regenerate rather than edit by hand.
# Each block carries the best trial of its own study. Blocks whose study is
# missing carry no overrides and say so: they are the transferred ablation.
#
# Every command is commented out. Copy one to run it by hand, or feed the
# whole file to scripts/cluster/submit_eval.sh, which parses these blocks
# and can queue them on a GPU or send them to slurm.

# ========================================================================
# mlp  nconst=16
# ========================================================================

# ------------------------------------------------------------------------
# mlp  nconst=16  baseline  (method=augmentation, group=none)
# transferred from experiment config (no study yet)
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=hls4ml/mlp_16 \
#     data.method=augmentation \
#     data.augmentation_scheme=none \
#     data.n_folds=5 \
#     checkpoint_manifest_path=logs/train/hls4ml/mlp/nconst16/baseline/checkpoint_manifest.json \
#     callbacks.results_table.invariance_metric=wasserstein \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/hls4ml/mlp/nconst16/baseline

# ------------------------------------------------------------------------
# mlp  nconst=16  aug_so2  (method=augmentation, group=so2)
# transferred from experiment config (no study yet)
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=hls4ml/mlp_16 \
#     data.method=augmentation \
#     data.augmentation_scheme=so2 \
#     data.n_folds=5 \
#     checkpoint_manifest_path=logs/train/hls4ml/mlp/nconst16/aug_so2/checkpoint_manifest.json \
#     callbacks.results_table.invariance_metric=wasserstein \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/hls4ml/mlp/nconst16/aug_so2

# ------------------------------------------------------------------------
# mlp  nconst=16  aug_so3  (method=augmentation, group=so3)
# transferred from experiment config (no study yet)
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=hls4ml/mlp_16 \
#     data.method=augmentation \
#     data.augmentation_scheme=so3 \
#     data.n_folds=5 \
#     checkpoint_manifest_path=logs/train/hls4ml/mlp/nconst16/aug_so3/checkpoint_manifest.json \
#     callbacks.results_table.invariance_metric=wasserstein \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/hls4ml/mlp/nconst16/aug_so3

# ------------------------------------------------------------------------
# mlp  nconst=16  aug_lorentz  (method=augmentation, group=lorentz)
# transferred from experiment config (no study yet)
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=hls4ml/mlp_16 \
#     data.method=augmentation \
#     data.augmentation_scheme=lorentz \
#     data.n_folds=5 \
#     checkpoint_manifest_path=logs/train/hls4ml/mlp/nconst16/aug_lorentz/checkpoint_manifest.json \
#     callbacks.results_table.invariance_metric=wasserstein \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/hls4ml/mlp/nconst16/aug_lorentz

# ------------------------------------------------------------------------
# mlp  nconst=16  symm_so2  (method=symm_loss, group=so2)
# transferred from experiment config (no study yet)
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=hls4ml/mlp_16 \
#     data.method=symm_loss \
#     data.augmentation_scheme=so2 \
#     data.n_folds=5 \
#     checkpoint_manifest_path=logs/train/hls4ml/mlp/nconst16/symm_so2/checkpoint_manifest.json \
#     callbacks.results_table.invariance_metric=wasserstein \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/hls4ml/mlp/nconst16/symm_so2

# ------------------------------------------------------------------------
# mlp  nconst=16  symm_so3  (method=symm_loss, group=so3)
# transferred from experiment config (no study yet)
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=hls4ml/mlp_16 \
#     data.method=symm_loss \
#     data.augmentation_scheme=so3 \
#     data.n_folds=5 \
#     checkpoint_manifest_path=logs/train/hls4ml/mlp/nconst16/symm_so3/checkpoint_manifest.json \
#     callbacks.results_table.invariance_metric=wasserstein \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/hls4ml/mlp/nconst16/symm_so3

# ------------------------------------------------------------------------
# mlp  nconst=16  symm_lorentz  (method=symm_loss, group=lorentz)
# transferred from experiment config (no study yet)
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=hls4ml/mlp_16 \
#     data.method=symm_loss \
#     data.augmentation_scheme=lorentz \
#     data.n_folds=5 \
#     checkpoint_manifest_path=logs/train/hls4ml/mlp/nconst16/symm_lorentz/checkpoint_manifest.json \
#     callbacks.results_table.invariance_metric=wasserstein \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/hls4ml/mlp/nconst16/symm_lorentz

# ========================================================================
# mlp  nconst=50
# ========================================================================

# ------------------------------------------------------------------------
# mlp  nconst=50  baseline  (method=augmentation, group=none)
# transferred from experiment config (no study yet)
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=hls4ml/mlp_50 \
#     data.method=augmentation \
#     data.augmentation_scheme=none \
#     data.n_folds=5 \
#     checkpoint_manifest_path=logs/train/hls4ml/mlp/nconst50/baseline/checkpoint_manifest.json \
#     callbacks.results_table.invariance_metric=wasserstein \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/hls4ml/mlp/nconst50/baseline

# ------------------------------------------------------------------------
# mlp  nconst=50  aug_so2  (method=augmentation, group=so2)
# transferred from experiment config (no study yet)
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=hls4ml/mlp_50 \
#     data.method=augmentation \
#     data.augmentation_scheme=so2 \
#     data.n_folds=5 \
#     checkpoint_manifest_path=logs/train/hls4ml/mlp/nconst50/aug_so2/checkpoint_manifest.json \
#     callbacks.results_table.invariance_metric=wasserstein \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/hls4ml/mlp/nconst50/aug_so2

# ------------------------------------------------------------------------
# mlp  nconst=50  aug_so3  (method=augmentation, group=so3)
# transferred from experiment config (no study yet)
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=hls4ml/mlp_50 \
#     data.method=augmentation \
#     data.augmentation_scheme=so3 \
#     data.n_folds=5 \
#     checkpoint_manifest_path=logs/train/hls4ml/mlp/nconst50/aug_so3/checkpoint_manifest.json \
#     callbacks.results_table.invariance_metric=wasserstein \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/hls4ml/mlp/nconst50/aug_so3

# ------------------------------------------------------------------------
# mlp  nconst=50  aug_lorentz  (method=augmentation, group=lorentz)
# transferred from experiment config (no study yet)
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=hls4ml/mlp_50 \
#     data.method=augmentation \
#     data.augmentation_scheme=lorentz \
#     data.n_folds=5 \
#     checkpoint_manifest_path=logs/train/hls4ml/mlp/nconst50/aug_lorentz/checkpoint_manifest.json \
#     callbacks.results_table.invariance_metric=wasserstein \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/hls4ml/mlp/nconst50/aug_lorentz

# ------------------------------------------------------------------------
# mlp  nconst=50  symm_so2  (method=symm_loss, group=so2)
# transferred from experiment config (no study yet)
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=hls4ml/mlp_50 \
#     data.method=symm_loss \
#     data.augmentation_scheme=so2 \
#     data.n_folds=5 \
#     checkpoint_manifest_path=logs/train/hls4ml/mlp/nconst50/symm_so2/checkpoint_manifest.json \
#     callbacks.results_table.invariance_metric=wasserstein \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/hls4ml/mlp/nconst50/symm_so2

# ------------------------------------------------------------------------
# mlp  nconst=50  symm_so3  (method=symm_loss, group=so3)
# transferred from experiment config (no study yet)
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=hls4ml/mlp_50 \
#     data.method=symm_loss \
#     data.augmentation_scheme=so3 \
#     data.n_folds=5 \
#     checkpoint_manifest_path=logs/train/hls4ml/mlp/nconst50/symm_so3/checkpoint_manifest.json \
#     callbacks.results_table.invariance_metric=wasserstein \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/hls4ml/mlp/nconst50/symm_so3

# ------------------------------------------------------------------------
# mlp  nconst=50  symm_lorentz  (method=symm_loss, group=lorentz)
# transferred from experiment config (no study yet)
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=hls4ml/mlp_50 \
#     data.method=symm_loss \
#     data.augmentation_scheme=lorentz \
#     data.n_folds=5 \
#     checkpoint_manifest_path=logs/train/hls4ml/mlp/nconst50/symm_lorentz/checkpoint_manifest.json \
#     callbacks.results_table.invariance_metric=wasserstein \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/hls4ml/mlp/nconst50/symm_lorentz

# ========================================================================
# mlp  nconst=150
# ========================================================================

# ------------------------------------------------------------------------
# mlp  nconst=150  baseline  (method=augmentation, group=none)
# transferred from experiment config (no study yet)
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=hls4ml/mlp_150 \
#     data.method=augmentation \
#     data.augmentation_scheme=none \
#     data.n_folds=5 \
#     checkpoint_manifest_path=logs/train/hls4ml/mlp/nconst150/baseline/checkpoint_manifest.json \
#     callbacks.results_table.invariance_metric=wasserstein \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/hls4ml/mlp/nconst150/baseline

# ------------------------------------------------------------------------
# mlp  nconst=150  aug_so2  (method=augmentation, group=so2)
# transferred from experiment config (no study yet)
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=hls4ml/mlp_150 \
#     data.method=augmentation \
#     data.augmentation_scheme=so2 \
#     data.n_folds=5 \
#     checkpoint_manifest_path=logs/train/hls4ml/mlp/nconst150/aug_so2/checkpoint_manifest.json \
#     callbacks.results_table.invariance_metric=wasserstein \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/hls4ml/mlp/nconst150/aug_so2

# ------------------------------------------------------------------------
# mlp  nconst=150  aug_so3  (method=augmentation, group=so3)
# transferred from experiment config (no study yet)
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=hls4ml/mlp_150 \
#     data.method=augmentation \
#     data.augmentation_scheme=so3 \
#     data.n_folds=5 \
#     checkpoint_manifest_path=logs/train/hls4ml/mlp/nconst150/aug_so3/checkpoint_manifest.json \
#     callbacks.results_table.invariance_metric=wasserstein \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/hls4ml/mlp/nconst150/aug_so3

# ------------------------------------------------------------------------
# mlp  nconst=150  aug_lorentz  (method=augmentation, group=lorentz)
# transferred from experiment config (no study yet)
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=hls4ml/mlp_150 \
#     data.method=augmentation \
#     data.augmentation_scheme=lorentz \
#     data.n_folds=5 \
#     checkpoint_manifest_path=logs/train/hls4ml/mlp/nconst150/aug_lorentz/checkpoint_manifest.json \
#     callbacks.results_table.invariance_metric=wasserstein \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/hls4ml/mlp/nconst150/aug_lorentz

# ------------------------------------------------------------------------
# mlp  nconst=150  symm_so2  (method=symm_loss, group=so2)
# transferred from experiment config (no study yet)
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=hls4ml/mlp_150 \
#     data.method=symm_loss \
#     data.augmentation_scheme=so2 \
#     data.n_folds=5 \
#     checkpoint_manifest_path=logs/train/hls4ml/mlp/nconst150/symm_so2/checkpoint_manifest.json \
#     callbacks.results_table.invariance_metric=wasserstein \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/hls4ml/mlp/nconst150/symm_so2

# ------------------------------------------------------------------------
# mlp  nconst=150  symm_so3  (method=symm_loss, group=so3)
# transferred from experiment config (no study yet)
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=hls4ml/mlp_150 \
#     data.method=symm_loss \
#     data.augmentation_scheme=so3 \
#     data.n_folds=5 \
#     checkpoint_manifest_path=logs/train/hls4ml/mlp/nconst150/symm_so3/checkpoint_manifest.json \
#     callbacks.results_table.invariance_metric=wasserstein \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/hls4ml/mlp/nconst150/symm_so3

# ------------------------------------------------------------------------
# mlp  nconst=150  symm_lorentz  (method=symm_loss, group=lorentz)
# transferred from experiment config (no study yet)
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=hls4ml/mlp_150 \
#     data.method=symm_loss \
#     data.augmentation_scheme=lorentz \
#     data.n_folds=5 \
#     checkpoint_manifest_path=logs/train/hls4ml/mlp/nconst150/symm_lorentz/checkpoint_manifest.json \
#     callbacks.results_table.invariance_metric=wasserstein \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/hls4ml/mlp/nconst150/symm_lorentz
