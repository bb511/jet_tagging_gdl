#!/usr/bin/env bash
# ========================================================================
# VN_HGQ TRAINING COMMANDS
# ========================================================================
# Five train runs at 16 constituents: the float reference, then one block
# per beta. Hand-written: vn_hgq has no study and the reference is a variant
# of a study's best, so scripts/optuna/make_train_catalogues.py neither
# writes nor regenerates this file.
# Baseline only: the vector neurons embed the symmetry, so the augmentation
# and SCL regimes are not run on them.
# The first block trains the float reference that algorithm.init_from of
# every later block names: the main run (fold=null) of the vn study's best
# at 16 constituents (lr 0.004986660327956767, weight decay 0.05) with one
# block in place of two, in logs/train/hls4ml/vn/nconst16/baseline_1block.
# One block rather than the two-block finals: it comes within 0.1 points of
# the two-block best at about half the depth and cost, and the paper shows
# the one-block scan alone. It must finish before the beta blocks start.
# Each later block fine-tunes that reference at one constant beta; the last
# epoch is the model. The grid 1e-08 ... 3e-07 was fixed by the dry run of
# 2026-10-07 on the two-block finals: at the seeded 14 data and 8 weight
# bits, beta times the initial EBOPs over the initial cross-entropy ran from
# 0.037 to 1.1.
# The four beta blocks share the label vn_hgq_nconst16_baseline, so select
# them with --only <index> (2 to 5), not by name; the reference is block 1.
#
# Every command is commented out: copy one to run it by hand, or feed the whole
# file to scripts/cluster/submit_train.sh, which queues them on a GPU or slurm.

# ========================================================================
# vn  nconst=16  (the float reference)
# ========================================================================

# ------------------------------------------------------------------------
# vn nconst=16 baseline, one block: the economical VN, the float reference
# the quantised blocks start from
# hand-written: the study's best at nconst=16 with num_blocks=1
# ------------------------------------------------------------------------
# python3 src/train.py \
#     experiment=hls4ml/vn_16 \
#     data.method=augmentation \
#     data.augmentation_scheme=none \
#     fold=null \
#     algorithm.dropout=0.0 \
#     algorithm.head_layers='[64,32]' \
#     algorithm.invariants=8 \
#     algorithm.num_blocks=1 \
#     algorithm.optimizer.lr=0.004986660327956767 \
#     algorithm.optimizer.weight_decay=0.05 \
#     algorithm.scalar_channels=32 \
#     algorithm.stream_channels=8 \
#     algorithm.tensor_invariants=4 \
#     algorithm.vector_channels=16 \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/train/hls4ml/vn/nconst16/baseline_1block

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
#     algorithm.init_from=logs/train/hls4ml/vn/nconst16/baseline_1block \
#     algorithm.invariants=8 \
#     algorithm.num_blocks=1 \
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
#     algorithm.init_from=logs/train/hls4ml/vn/nconst16/baseline_1block \
#     algorithm.invariants=8 \
#     algorithm.num_blocks=1 \
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
#     algorithm.init_from=logs/train/hls4ml/vn/nconst16/baseline_1block \
#     algorithm.invariants=8 \
#     algorithm.num_blocks=1 \
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
#     algorithm.init_from=logs/train/hls4ml/vn/nconst16/baseline_1block \
#     algorithm.invariants=8 \
#     algorithm.num_blocks=1 \
#     algorithm.scalar_channels=32 \
#     algorithm.stream_channels=8 \
#     algorithm.tensor_invariants=4 \
#     algorithm.vector_channels=16 \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/train/hls4ml/vn_hgq/nconst16/baseline_beta3e-07
