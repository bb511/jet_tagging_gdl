#!/usr/bin/env bash
# ========================================================================
# VN_HGQ EVALUATION COMMANDS
# ========================================================================
# Five eval runs at 16 constituents, one per block of runvn_hgq_train.sh:
# the float reference, then one per beta. Hand-written: vn_hgq has no study
# and the reference is a variant of a study's best, so
# scripts/optuna/make_train_catalogues.py neither writes nor regenerates
# this file.
# Baseline only: the vector neurons embed the symmetry, so the augmentation
# and SCL regimes are not run on them.
# The first block evaluates the float reference, the one-block vn main run
# logs/train/hls4ml/vn/nconst16/baseline_1block that algorithm.init_from of
# every later block names. One block rather than the two-block finals: it
# comes within 0.1 points of the two-block best at about half the depth and
# cost, and the paper shows the one-block scan alone. Its row goes to
# results/quantised/ (results_table.subdir), beside the quantised rows,
# rather than to the finals' table in results/embedded/.
# Each later block repeats its training block's overrides and runs Keras in
# float64 (algorithm.floatx) on the CPU (algorithm.jax_platform), where it
# reproduces the fixed-point arithmetic exactly; on a GPU, XLA's float64
# 2**f is off by an ulp for some f and about 2 % of the jets round
# differently. It writes outputs/main_hgq/ beside the probe's
# outputs/main.npz, the input of scripts/synthesis/synthesise_vn.py.
# The grid 1e-08 ... 3e-07 was fixed by the dry run of 2026-10-07 on the
# two-block finals: at the seeded 14 data and 8 weight bits, beta times the
# initial EBOPs over the initial cross-entropy ran from 0.037 to 1.1.
# The four beta blocks share the label vn_hgq_nconst16_baseline, so select
# them with --only <index> (2 to 5), not by name; the reference is block 1.
# Each beta block names its row vn_hgq_beta<tag> (algorithm.name), so that
# the table in results/quantised/ tells the four apart. Each block's
# evaluation merges the main run with the five folds fitted beside it on
# 2026-10-08.
#
# Every command is commented out: copy one to run it by hand, or feed the whole
# file to scripts/cluster/submit_eval.sh, which queues them on a GPU or slurm.

# ========================================================================
# vn  nconst=16  (the float reference)
# ========================================================================

# ------------------------------------------------------------------------
# vn nconst=16 baseline, one block: the economical VN, the float reference
# the quantised blocks start from
# hand-written: the study's best at nconst=16 with num_blocks=1
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=hls4ml/vn_16 \
#     data.method=augmentation \
#     data.augmentation_scheme=none \
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
#     checkpoint_manifest_path=logs/train/hls4ml/vn/nconst16/baseline_1block \
#     results_table.invariance_metric=tv \
#     results_table.subdir=quantised \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/hls4ml/vn/nconst16/baseline_1block

# ========================================================================
# vn_hgq  nconst=16
# ========================================================================

# ------------------------------------------------------------------------
# vn_hgq  nconst=16  baseline  (method=augmentation, group=none)  beta=1e-08
# hand-written, no study
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=hls4ml/vn_hgq_16 \
#     data.method=augmentation \
#     data.augmentation_scheme=none \
#     algorithm.beta=1e-08 \
#     algorithm.dropout=0.0 \
#     algorithm.floatx=float64 \
#     algorithm.head_layers='[64,32]' \
#     algorithm.init_from=logs/train/hls4ml/vn/nconst16/baseline_1block \
#     algorithm.invariants=8 \
#     algorithm.jax_platform=cpu \
#     algorithm.name=vn_hgq_beta1e-08 \
#     algorithm.num_blocks=1 \
#     algorithm.scalar_channels=32 \
#     algorithm.stream_channels=8 \
#     algorithm.tensor_invariants=4 \
#     algorithm.vector_channels=16 \
#     checkpoint_manifest_path=logs/train/hls4ml/vn_hgq/nconst16/baseline_beta1e-08 \
#     results_table.invariance_metric=tv \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/hls4ml/vn_hgq/nconst16/baseline_beta1e-08

# ------------------------------------------------------------------------
# vn_hgq  nconst=16  baseline  (method=augmentation, group=none)  beta=3e-08
# hand-written, no study
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=hls4ml/vn_hgq_16 \
#     data.method=augmentation \
#     data.augmentation_scheme=none \
#     algorithm.beta=3e-08 \
#     algorithm.dropout=0.0 \
#     algorithm.floatx=float64 \
#     algorithm.head_layers='[64,32]' \
#     algorithm.init_from=logs/train/hls4ml/vn/nconst16/baseline_1block \
#     algorithm.invariants=8 \
#     algorithm.jax_platform=cpu \
#     algorithm.name=vn_hgq_beta3e-08 \
#     algorithm.num_blocks=1 \
#     algorithm.scalar_channels=32 \
#     algorithm.stream_channels=8 \
#     algorithm.tensor_invariants=4 \
#     algorithm.vector_channels=16 \
#     checkpoint_manifest_path=logs/train/hls4ml/vn_hgq/nconst16/baseline_beta3e-08 \
#     results_table.invariance_metric=tv \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/hls4ml/vn_hgq/nconst16/baseline_beta3e-08

# ------------------------------------------------------------------------
# vn_hgq  nconst=16  baseline  (method=augmentation, group=none)  beta=1e-07
# hand-written, no study
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=hls4ml/vn_hgq_16 \
#     data.method=augmentation \
#     data.augmentation_scheme=none \
#     algorithm.beta=1e-07 \
#     algorithm.dropout=0.0 \
#     algorithm.floatx=float64 \
#     algorithm.head_layers='[64,32]' \
#     algorithm.init_from=logs/train/hls4ml/vn/nconst16/baseline_1block \
#     algorithm.invariants=8 \
#     algorithm.jax_platform=cpu \
#     algorithm.name=vn_hgq_beta1e-07 \
#     algorithm.num_blocks=1 \
#     algorithm.scalar_channels=32 \
#     algorithm.stream_channels=8 \
#     algorithm.tensor_invariants=4 \
#     algorithm.vector_channels=16 \
#     checkpoint_manifest_path=logs/train/hls4ml/vn_hgq/nconst16/baseline_beta1e-07 \
#     results_table.invariance_metric=tv \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/hls4ml/vn_hgq/nconst16/baseline_beta1e-07

# ------------------------------------------------------------------------
# vn_hgq  nconst=16  baseline  (method=augmentation, group=none)  beta=3e-07
# hand-written, no study
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=hls4ml/vn_hgq_16 \
#     data.method=augmentation \
#     data.augmentation_scheme=none \
#     algorithm.beta=3e-07 \
#     algorithm.dropout=0.0 \
#     algorithm.floatx=float64 \
#     algorithm.head_layers='[64,32]' \
#     algorithm.init_from=logs/train/hls4ml/vn/nconst16/baseline_1block \
#     algorithm.invariants=8 \
#     algorithm.jax_platform=cpu \
#     algorithm.name=vn_hgq_beta3e-07 \
#     algorithm.num_blocks=1 \
#     algorithm.scalar_channels=32 \
#     algorithm.stream_channels=8 \
#     algorithm.tensor_invariants=4 \
#     algorithm.vector_channels=16 \
#     checkpoint_manifest_path=logs/train/hls4ml/vn_hgq/nconst16/baseline_beta3e-07 \
#     results_table.invariance_metric=tv \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/hls4ml/vn_hgq/nconst16/baseline_beta3e-07
