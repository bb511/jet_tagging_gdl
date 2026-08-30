#!/usr/bin/env bash
# ========================================================================
# DEEPSETS TRAINING COMMANDS
# ========================================================================
# One block per (constituent count, regime): 3 x 7 = 21 train runs.
#
# Generated from the study databases under logs/optuna/hls4ml/ by
# scripts/optuna/make_train_catalogues.py; regenerate rather than edit by hand.
# Each block carries the best trial of its own study. Blocks whose study is
# missing carry no overrides and say so: they are the transferred ablation.
#
# Every command is commented out. Copy one to run it by hand, or feed the
# whole file to scripts/cluster/submit_train.sh, which parses these blocks
# and can queue them on a GPU or send them to slurm.

# ========================================================================
# deepsets  nconst=16
# ========================================================================

# ------------------------------------------------------------------------
# deepsets  nconst=16  baseline  (method=augmentation, group=none)
# transferred from experiment config (no study yet)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     experiment=hls4ml/deepsets_16 \
#     data.method=augmentation \
#     data.augmentation_scheme=none \
#     data.n_folds=5 \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/train/hls4ml/deepsets/nconst16/baseline

# ------------------------------------------------------------------------
# deepsets  nconst=16  aug_so2  (method=augmentation, group=so2)
# transferred from experiment config (no study yet)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     experiment=hls4ml/deepsets_16 \
#     data.method=augmentation \
#     data.augmentation_scheme=so2 \
#     data.n_folds=5 \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/train/hls4ml/deepsets/nconst16/aug_so2

# ------------------------------------------------------------------------
# deepsets  nconst=16  aug_so3  (method=augmentation, group=so3)
# transferred from experiment config (no study yet)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     experiment=hls4ml/deepsets_16 \
#     data.method=augmentation \
#     data.augmentation_scheme=so3 \
#     data.n_folds=5 \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/train/hls4ml/deepsets/nconst16/aug_so3

# ------------------------------------------------------------------------
# deepsets  nconst=16  aug_lorentz  (method=augmentation, group=lorentz)
# transferred from experiment config (no study yet)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     experiment=hls4ml/deepsets_16 \
#     data.method=augmentation \
#     data.augmentation_scheme=lorentz \
#     data.n_folds=5 \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/train/hls4ml/deepsets/nconst16/aug_lorentz

# ------------------------------------------------------------------------
# deepsets  nconst=16  symm_so2  (method=symm_loss, group=so2)
# transferred from experiment config (no study yet)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     experiment=hls4ml/deepsets_16 \
#     data.method=symm_loss \
#     data.augmentation_scheme=so2 \
#     data.n_folds=5 \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/train/hls4ml/deepsets/nconst16/symm_so2

# ------------------------------------------------------------------------
# deepsets  nconst=16  symm_so3  (method=symm_loss, group=so3)
# transferred from experiment config (no study yet)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     experiment=hls4ml/deepsets_16 \
#     data.method=symm_loss \
#     data.augmentation_scheme=so3 \
#     data.n_folds=5 \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/train/hls4ml/deepsets/nconst16/symm_so3

# ------------------------------------------------------------------------
# deepsets  nconst=16  symm_lorentz  (method=symm_loss, group=lorentz)
# transferred from experiment config (no study yet)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     experiment=hls4ml/deepsets_16 \
#     data.method=symm_loss \
#     data.augmentation_scheme=lorentz \
#     data.n_folds=5 \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/train/hls4ml/deepsets/nconst16/symm_lorentz

# ========================================================================
# deepsets  nconst=50
# ========================================================================

# ------------------------------------------------------------------------
# deepsets  nconst=50  baseline  (method=augmentation, group=none)
# transferred from experiment config (no study yet)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     experiment=hls4ml/deepsets_50 \
#     data.method=augmentation \
#     data.augmentation_scheme=none \
#     data.n_folds=5 \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/train/hls4ml/deepsets/nconst50/baseline

# ------------------------------------------------------------------------
# deepsets  nconst=50  aug_so2  (method=augmentation, group=so2)
# transferred from experiment config (no study yet)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     experiment=hls4ml/deepsets_50 \
#     data.method=augmentation \
#     data.augmentation_scheme=so2 \
#     data.n_folds=5 \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/train/hls4ml/deepsets/nconst50/aug_so2

# ------------------------------------------------------------------------
# deepsets  nconst=50  aug_so3  (method=augmentation, group=so3)
# transferred from experiment config (no study yet)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     experiment=hls4ml/deepsets_50 \
#     data.method=augmentation \
#     data.augmentation_scheme=so3 \
#     data.n_folds=5 \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/train/hls4ml/deepsets/nconst50/aug_so3

# ------------------------------------------------------------------------
# deepsets  nconst=50  aug_lorentz  (method=augmentation, group=lorentz)
# transferred from experiment config (no study yet)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     experiment=hls4ml/deepsets_50 \
#     data.method=augmentation \
#     data.augmentation_scheme=lorentz \
#     data.n_folds=5 \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/train/hls4ml/deepsets/nconst50/aug_lorentz

# ------------------------------------------------------------------------
# deepsets  nconst=50  symm_so2  (method=symm_loss, group=so2)
# transferred from experiment config (no study yet)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     experiment=hls4ml/deepsets_50 \
#     data.method=symm_loss \
#     data.augmentation_scheme=so2 \
#     data.n_folds=5 \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/train/hls4ml/deepsets/nconst50/symm_so2

# ------------------------------------------------------------------------
# deepsets  nconst=50  symm_so3  (method=symm_loss, group=so3)
# transferred from experiment config (no study yet)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     experiment=hls4ml/deepsets_50 \
#     data.method=symm_loss \
#     data.augmentation_scheme=so3 \
#     data.n_folds=5 \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/train/hls4ml/deepsets/nconst50/symm_so3

# ------------------------------------------------------------------------
# deepsets  nconst=50  symm_lorentz  (method=symm_loss, group=lorentz)
# transferred from experiment config (no study yet)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     experiment=hls4ml/deepsets_50 \
#     data.method=symm_loss \
#     data.augmentation_scheme=lorentz \
#     data.n_folds=5 \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/train/hls4ml/deepsets/nconst50/symm_lorentz

# ========================================================================
# deepsets  nconst=150
# ========================================================================

# ------------------------------------------------------------------------
# deepsets  nconst=150  baseline  (method=augmentation, group=none)
# transferred from experiment config (no study yet)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     experiment=hls4ml/deepsets_150 \
#     data.method=augmentation \
#     data.augmentation_scheme=none \
#     data.n_folds=5 \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/train/hls4ml/deepsets/nconst150/baseline

# ------------------------------------------------------------------------
# deepsets  nconst=150  aug_so2  (method=augmentation, group=so2)
# transferred from experiment config (no study yet)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     experiment=hls4ml/deepsets_150 \
#     data.method=augmentation \
#     data.augmentation_scheme=so2 \
#     data.n_folds=5 \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/train/hls4ml/deepsets/nconst150/aug_so2

# ------------------------------------------------------------------------
# deepsets  nconst=150  aug_so3  (method=augmentation, group=so3)
# transferred from experiment config (no study yet)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     experiment=hls4ml/deepsets_150 \
#     data.method=augmentation \
#     data.augmentation_scheme=so3 \
#     data.n_folds=5 \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/train/hls4ml/deepsets/nconst150/aug_so3

# ------------------------------------------------------------------------
# deepsets  nconst=150  aug_lorentz  (method=augmentation, group=lorentz)
# transferred from experiment config (no study yet)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     experiment=hls4ml/deepsets_150 \
#     data.method=augmentation \
#     data.augmentation_scheme=lorentz \
#     data.n_folds=5 \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/train/hls4ml/deepsets/nconst150/aug_lorentz

# ------------------------------------------------------------------------
# deepsets  nconst=150  symm_so2  (method=symm_loss, group=so2)
# transferred from experiment config (no study yet)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     experiment=hls4ml/deepsets_150 \
#     data.method=symm_loss \
#     data.augmentation_scheme=so2 \
#     data.n_folds=5 \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/train/hls4ml/deepsets/nconst150/symm_so2

# ------------------------------------------------------------------------
# deepsets  nconst=150  symm_so3  (method=symm_loss, group=so3)
# transferred from experiment config (no study yet)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     experiment=hls4ml/deepsets_150 \
#     data.method=symm_loss \
#     data.augmentation_scheme=so3 \
#     data.n_folds=5 \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/train/hls4ml/deepsets/nconst150/symm_so3

# ------------------------------------------------------------------------
# deepsets  nconst=150  symm_lorentz  (method=symm_loss, group=lorentz)
# transferred from experiment config (no study yet)
# ------------------------------------------------------------------------
# python3 src/train.py \
#     experiment=hls4ml/deepsets_150 \
#     data.method=symm_loss \
#     data.augmentation_scheme=lorentz \
#     data.n_folds=5 \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/train/hls4ml/deepsets/nconst150/symm_lorentz
