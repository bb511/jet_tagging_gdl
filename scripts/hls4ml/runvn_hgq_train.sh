#!/usr/bin/env bash
# ========================================================================
# VN_HGQ TRAINING COMMANDS
# ========================================================================
# One block per beta at 16 constituents: 5 train runs. Hand-written: this
# row has no study, so scripts/optuna/make_train_catalogues.py neither
# writes nor regenerates this file.
# Baseline only: the quantisation-aware vector neurons embed the symmetry as
# vn does, so the augmentation and SCL regimes are not run on them.
# Each block fine-tunes the main run of the float vn finals
# (algorithm.init_from), in their architecture, at one constant beta; the
# last epoch is the model. The grid 1e-08 ... 1e-06 is provisional: the
# pre-flight D4 on the float checkpoint fixes it, from the initial EBOPs and
# the cross-entropy.
# The five blocks share the label vn_hgq_nconst16_baseline, so select them
# with --only <index> (1 to 5), not by name.
#
# Every command is commented out: copy one to run it by hand, or feed the whole
# file to scripts/cluster/submit_train.sh, which queues them on a GPU or slurm.

# ========================================================================
# vn_hgq  nconst=16
# ========================================================================

# ------------------------------------------------------------------------
# vn_hgq  nconst=16  baseline  (method=augmentation, group=none)  beta=1e-08
# hand-written, no study
# ------------------------------------------------------------------------
# python3 src/train.py \
#     experiment=hls4ml/vn_hgq_16 \
#     data.method=augmentation \
#     data.augmentation_scheme=none \
#     fold=null \
#     algorithm.beta=1e-08 \
#     algorithm.dropout=0.0 \
#     algorithm.head_layers='[64,32]' \
#     algorithm.init_from=logs/train/hls4ml/vn/nconst16/baseline \
#     algorithm.invariants=8 \
#     algorithm.num_blocks=2 \
#     algorithm.scalar_channels=32 \
#     algorithm.stream_channels=8 \
#     algorithm.tensor_invariants=4 \
#     algorithm.vector_channels=16 \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/train/hls4ml/vn_hgq/nconst16/baseline_beta1e-08

# ------------------------------------------------------------------------
# vn_hgq  nconst=16  baseline  (method=augmentation, group=none)  beta=3e-08
# hand-written, no study
# ------------------------------------------------------------------------
# python3 src/train.py \
#     experiment=hls4ml/vn_hgq_16 \
#     data.method=augmentation \
#     data.augmentation_scheme=none \
#     fold=null \
#     algorithm.beta=3e-08 \
#     algorithm.dropout=0.0 \
#     algorithm.head_layers='[64,32]' \
#     algorithm.init_from=logs/train/hls4ml/vn/nconst16/baseline \
#     algorithm.invariants=8 \
#     algorithm.num_blocks=2 \
#     algorithm.scalar_channels=32 \
#     algorithm.stream_channels=8 \
#     algorithm.tensor_invariants=4 \
#     algorithm.vector_channels=16 \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/train/hls4ml/vn_hgq/nconst16/baseline_beta3e-08

# ------------------------------------------------------------------------
# vn_hgq  nconst=16  baseline  (method=augmentation, group=none)  beta=1e-07
# hand-written, no study
# ------------------------------------------------------------------------
# python3 src/train.py \
#     experiment=hls4ml/vn_hgq_16 \
#     data.method=augmentation \
#     data.augmentation_scheme=none \
#     fold=null \
#     algorithm.beta=1e-07 \
#     algorithm.dropout=0.0 \
#     algorithm.head_layers='[64,32]' \
#     algorithm.init_from=logs/train/hls4ml/vn/nconst16/baseline \
#     algorithm.invariants=8 \
#     algorithm.num_blocks=2 \
#     algorithm.scalar_channels=32 \
#     algorithm.stream_channels=8 \
#     algorithm.tensor_invariants=4 \
#     algorithm.vector_channels=16 \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/train/hls4ml/vn_hgq/nconst16/baseline_beta1e-07

# ------------------------------------------------------------------------
# vn_hgq  nconst=16  baseline  (method=augmentation, group=none)  beta=3e-07
# hand-written, no study
# ------------------------------------------------------------------------
# python3 src/train.py \
#     experiment=hls4ml/vn_hgq_16 \
#     data.method=augmentation \
#     data.augmentation_scheme=none \
#     fold=null \
#     algorithm.beta=3e-07 \
#     algorithm.dropout=0.0 \
#     algorithm.head_layers='[64,32]' \
#     algorithm.init_from=logs/train/hls4ml/vn/nconst16/baseline \
#     algorithm.invariants=8 \
#     algorithm.num_blocks=2 \
#     algorithm.scalar_channels=32 \
#     algorithm.stream_channels=8 \
#     algorithm.tensor_invariants=4 \
#     algorithm.vector_channels=16 \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/train/hls4ml/vn_hgq/nconst16/baseline_beta3e-07

# ------------------------------------------------------------------------
# vn_hgq  nconst=16  baseline  (method=augmentation, group=none)  beta=1e-06
# hand-written, no study
# ------------------------------------------------------------------------
# python3 src/train.py \
#     experiment=hls4ml/vn_hgq_16 \
#     data.method=augmentation \
#     data.augmentation_scheme=none \
#     fold=null \
#     algorithm.beta=1e-06 \
#     algorithm.dropout=0.0 \
#     algorithm.head_layers='[64,32]' \
#     algorithm.init_from=logs/train/hls4ml/vn/nconst16/baseline \
#     algorithm.invariants=8 \
#     algorithm.num_blocks=2 \
#     algorithm.scalar_channels=32 \
#     algorithm.stream_channels=8 \
#     algorithm.tensor_invariants=4 \
#     algorithm.vector_channels=16 \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/train/hls4ml/vn_hgq/nconst16/baseline_beta1e-06
