#!/usr/bin/env bash
# ========================================================================
# VN_HGQ EVALUATION COMMANDS
# ========================================================================
# Forty-nine eval runs, one per block of runvn_hgq_train.sh: twenty-eight at
# 16 constituents (the float reference, one per beta, the latency ablation,
# the full-axis designs and the beta scan of the chosen two), then eight at 50
# and 150, then the two-block finals at 16 with the full-axis offset, then
# one picked checkpoint of each of the chosen two trained with JEDI-linear's
# recipe, then the picks of the four designs of blocks 40 to 43, then those of
# the six designs of blocks 48 to 53.
# Hand-written: vn_hgq has no study and the reference is a variant of a
# study's best, so scripts/experiments/make_train_catalogues.py neither writes
# nor regenerates this file.
# Baseline only: the vector neurons embed the symmetry, so the augmentation
# and SCL regimes are not run on them.
# The first block evaluates the float reference, the one-block vn main run
# logs/train/hls4ml/vn/nconst16/baseline_1block that algorithm.init_from of
# blocks 2 to 5 names. One block rather than the two-block finals: it
# comes within 0.1 points of the two-block best at about half the depth and
# cost, and the paper shows the one-block scan alone. Its row goes to
# results/quantised/ (results_table.subdir), beside the quantised rows,
# rather than to the finals' table in results/embedded/.
# Each quantised block repeats its training block's overrides and runs Keras in
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
# the table in results/quantised/ tells the four apart. The evaluation of
# each of blocks 1 to 5 merges the main run with the five folds fitted
# beside it on 2026-10-08.
# Blocks 6 to 13 evaluate the latency ablation of runvn_hgq_train.sh: the
# first four float variants (the gate, the head's 64-wide layer, the second
# layer of the scalar MLP and the stream switched off one after another),
# whose rows vn_<tag> also go to results/quantised/, then their fine-tunes
# at beta=3e-07, rows vn_hgq_<tag>_beta3e-07. Blocks 14 and 15 do the same
# for the fifth variant, which also switches off the tensor invariants.
# Their labels repeat those of blocks 1 to 5, so select them by index too.
# Blocks 16 and 17 evaluate the two float variants of blocks 16 and 17 of
# runvn_hgq_train.sh (no gate, head [32], no stream, at the reference's
# widths and at the one-block search's widths), rows vn_nogate_h32_nostream
# and vn_nogate_h32_nostream_wide; blocks 18 to 20 evaluate the fine-tunes
# of blocks 18 to 20 with the folded normalisations and truncating data
# lanes, rows vn_hgq_<tag>_foldtrn_beta3e-07.
# Blocks 21 to 24 evaluate the full-axis-offset designs A1 and A2 of blocks
# 21 to 24 of runvn_hgq_train.sh, rows vn_axis_<tag> and
# vn_hgq_axis_<tag>_foldtrn_beta3e-07, with the probe off: a transform
# rebases etarel and phirel on the kept constituents, so its invariance
# figures would mean nothing.
# Blocks 25 to 28 evaluate the beta scan and the learning-rate check of blocks
# 25 to 28 of runvn_hgq_train.sh, rows vn_hgq_<tag>_foldtrn[_lr3e-04]_beta<beta>.
# Blocks 29 to 36 evaluate the designs at 50 and 150 constituents of blocks 29
# to 36 of runvn_hgq_train.sh, rows vn_<tag> and vn_hgq_<tag>_foldtrn_beta3e-07
# (tags cheap_h32 and cheap_axis_h32; the row's nconstituents tells the counts
# apart), those with the offset with the probe off.
# Block 37 evaluates block 39 of runvn_hgq_train.sh, the two-block finals with
# the full-axis offset, row vn_axis, with the probe off. Blocks 38 and 39
# evaluate one checkpoint each (ckpt_path) of blocks 37 and 38 of
# runvn_hgq_train.sh, D2 and A2 with JEDI-linear's recipe, rows
# vn_hgq_<tag>_foldtrn_jedirecipe_epoch<epoch>. Each is the point of the
# design's accuracy-EBOPs front, restricted to test accuracies within 2 points
# of the design's best (past it the networks collapse), whose stage count at
# the published JEDI-linear's pipelining (2 ns clock, cutoff 2.7) is nearest
# that of its Tables I and II (23 and 19): epoch 1099 for D2 and epoch 1899
# for A2, both at 24 to 25 stages for either table.
# Blocks 40 to 43 evaluate the designs of blocks 40 to 43 of runvn_hgq_train.sh,
# D2 and A2 with fixed and with trained input words under JEDI-linear's recipe
# and a checkpoint every epoch, picked as the published JEDI-linear picks its
# own: the point of the front of calibrated validation accuracy against traced
# EBOPs nearest 57 237 EBOPs, through the manifests that
# scripts/experiments/quantised_picks.py writes to <run dir>/picks/target57237
# for the main run and each fold; rows vn_hgq_<tag>, the probe off for A2.
# Blocks 44 to 49 evaluate the designs of blocks 48 to 53 of runvn_hgq_train.sh,
# those of the comparison with JEDI-linear at a peak learning rate of 1e-3,
# picked by the rule of the vector neurons: the most accurate point of the
# calibrated front whose latency at a 5 ns clock, as
# scripts/synthesis/synthesise_vn.py --estimate-only estimates it, is at most
# 100 ns, through the manifests that quantised_picks.py pick-hw writes to
# <run dir>/picks/best100ns; rows vn_hgq_<tag>, the probe off for the designs
# with the axis offset.
#
# Every command is commented out: copy one to run it by hand, or feed the whole
# file to scripts/experiments/submit_eval.sh, which queues them on a GPU or slurm.

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

# ========================================================================
# latency ablation of the one-block network  nconst=16
# ========================================================================

# ------------------------------------------------------------------------
# vn nconst=16 baseline, one block, latency ablation nogate:
# no channel gate
# hand-written: the float reference with the switches of the variant
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=hls4ml/vn_16 \
#     data.method=augmentation \
#     data.augmentation_scheme=none \
#     algorithm.dropout=0.0 \
#     algorithm.gate=false \
#     algorithm.head_layers='[64,32]' \
#     algorithm.invariants=8 \
#     algorithm.name=vn_nogate \
#     algorithm.num_blocks=1 \
#     algorithm.optimizer.lr=0.004986660327956767 \
#     algorithm.optimizer.weight_decay=0.05 \
#     algorithm.scalar_channels=32 \
#     algorithm.stream_channels=8 \
#     algorithm.tensor_invariants=4 \
#     algorithm.vector_channels=16 \
#     checkpoint_manifest_path=logs/train/hls4ml/vn/nconst16/baseline_1block_nogate \
#     results_table.invariance_metric=tv \
#     results_table.subdir=quantised \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/hls4ml/vn/nconst16/baseline_1block_nogate

# ------------------------------------------------------------------------
# vn nconst=16 baseline, one block, latency ablation nogate_h32:
# no gate, head [32]
# hand-written: the float reference with the switches of the variant
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=hls4ml/vn_16 \
#     data.method=augmentation \
#     data.augmentation_scheme=none \
#     algorithm.dropout=0.0 \
#     algorithm.gate=false \
#     algorithm.head_layers='[32]' \
#     algorithm.invariants=8 \
#     algorithm.name=vn_nogate_h32 \
#     algorithm.num_blocks=1 \
#     algorithm.optimizer.lr=0.004986660327956767 \
#     algorithm.optimizer.weight_decay=0.05 \
#     algorithm.scalar_channels=32 \
#     algorithm.stream_channels=8 \
#     algorithm.tensor_invariants=4 \
#     algorithm.vector_channels=16 \
#     checkpoint_manifest_path=logs/train/hls4ml/vn/nconst16/baseline_1block_nogate_h32 \
#     results_table.invariance_metric=tv \
#     results_table.subdir=quantised \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/hls4ml/vn/nconst16/baseline_1block_nogate_h32

# ------------------------------------------------------------------------
# vn nconst=16 baseline, one block, latency ablation nogate_h32_mlp1:
# no gate, head [32], one-layer scalar MLP
# hand-written: the float reference with the switches of the variant
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=hls4ml/vn_16 \
#     data.method=augmentation \
#     data.augmentation_scheme=none \
#     algorithm.dropout=0.0 \
#     algorithm.gate=false \
#     algorithm.head_layers='[32]' \
#     algorithm.invariants=8 \
#     algorithm.mlp_layers=1 \
#     algorithm.name=vn_nogate_h32_mlp1 \
#     algorithm.num_blocks=1 \
#     algorithm.optimizer.lr=0.004986660327956767 \
#     algorithm.optimizer.weight_decay=0.05 \
#     algorithm.scalar_channels=32 \
#     algorithm.stream_channels=8 \
#     algorithm.tensor_invariants=4 \
#     algorithm.vector_channels=16 \
#     checkpoint_manifest_path=logs/train/hls4ml/vn/nconst16/baseline_1block_nogate_h32_mlp1 \
#     results_table.invariance_metric=tv \
#     results_table.subdir=quantised \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/hls4ml/vn/nconst16/baseline_1block_nogate_h32_mlp1

# ------------------------------------------------------------------------
# vn nconst=16 baseline, one block, latency ablation nogate_h32_mlp1_nostream:
# no gate, head [32], one-layer scalar MLP, no stream
# hand-written: the float reference with the switches of the variant
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=hls4ml/vn_16 \
#     data.method=augmentation \
#     data.augmentation_scheme=none \
#     algorithm.dropout=0.0 \
#     algorithm.gate=false \
#     algorithm.head_layers='[32]' \
#     algorithm.invariants=8 \
#     algorithm.mlp_layers=1 \
#     algorithm.name=vn_nogate_h32_mlp1_nostream \
#     algorithm.num_blocks=1 \
#     algorithm.optimizer.lr=0.004986660327956767 \
#     algorithm.optimizer.weight_decay=0.05 \
#     algorithm.scalar_channels=32 \
#     algorithm.stream_channels=0 \
#     algorithm.tensor_invariants=4 \
#     algorithm.vector_channels=16 \
#     checkpoint_manifest_path=logs/train/hls4ml/vn/nconst16/baseline_1block_nogate_h32_mlp1_nostream \
#     results_table.invariance_metric=tv \
#     results_table.subdir=quantised \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/hls4ml/vn/nconst16/baseline_1block_nogate_h32_mlp1_nostream

# ------------------------------------------------------------------------
# vn_hgq  nconst=16  baseline  (method=augmentation, group=none)  beta=3e-07
# latency ablation nogate:
# no channel gate
# hand-written, no study
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=hls4ml/vn_hgq_16 \
#     data.method=augmentation \
#     data.augmentation_scheme=none \
#     algorithm.beta=3e-07 \
#     algorithm.dropout=0.0 \
#     algorithm.floatx=float64 \
#     algorithm.gate=false \
#     algorithm.head_layers='[64,32]' \
#     algorithm.init_from=logs/train/hls4ml/vn/nconst16/baseline_1block_nogate \
#     algorithm.invariants=8 \
#     algorithm.jax_platform=cpu \
#     algorithm.name=vn_hgq_nogate_beta3e-07 \
#     algorithm.num_blocks=1 \
#     algorithm.scalar_channels=32 \
#     algorithm.stream_channels=8 \
#     algorithm.tensor_invariants=4 \
#     algorithm.vector_channels=16 \
#     checkpoint_manifest_path=logs/train/hls4ml/vn_hgq/nconst16/baseline_nogate_beta3e-07 \
#     results_table.invariance_metric=tv \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/hls4ml/vn_hgq/nconst16/baseline_nogate_beta3e-07

# ------------------------------------------------------------------------
# vn_hgq  nconst=16  baseline  (method=augmentation, group=none)  beta=3e-07
# latency ablation nogate_h32:
# no gate, head [32]
# hand-written, no study
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=hls4ml/vn_hgq_16 \
#     data.method=augmentation \
#     data.augmentation_scheme=none \
#     algorithm.beta=3e-07 \
#     algorithm.dropout=0.0 \
#     algorithm.floatx=float64 \
#     algorithm.gate=false \
#     algorithm.head_layers='[32]' \
#     algorithm.init_from=logs/train/hls4ml/vn/nconst16/baseline_1block_nogate_h32 \
#     algorithm.invariants=8 \
#     algorithm.jax_platform=cpu \
#     algorithm.name=vn_hgq_nogate_h32_beta3e-07 \
#     algorithm.num_blocks=1 \
#     algorithm.scalar_channels=32 \
#     algorithm.stream_channels=8 \
#     algorithm.tensor_invariants=4 \
#     algorithm.vector_channels=16 \
#     checkpoint_manifest_path=logs/train/hls4ml/vn_hgq/nconst16/baseline_nogate_h32_beta3e-07 \
#     results_table.invariance_metric=tv \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/hls4ml/vn_hgq/nconst16/baseline_nogate_h32_beta3e-07

# ------------------------------------------------------------------------
# vn_hgq  nconst=16  baseline  (method=augmentation, group=none)  beta=3e-07
# latency ablation nogate_h32_mlp1:
# no gate, head [32], one-layer scalar MLP
# hand-written, no study
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=hls4ml/vn_hgq_16 \
#     data.method=augmentation \
#     data.augmentation_scheme=none \
#     algorithm.beta=3e-07 \
#     algorithm.dropout=0.0 \
#     algorithm.floatx=float64 \
#     algorithm.gate=false \
#     algorithm.head_layers='[32]' \
#     algorithm.init_from=logs/train/hls4ml/vn/nconst16/baseline_1block_nogate_h32_mlp1 \
#     algorithm.invariants=8 \
#     algorithm.jax_platform=cpu \
#     algorithm.mlp_layers=1 \
#     algorithm.name=vn_hgq_nogate_h32_mlp1_beta3e-07 \
#     algorithm.num_blocks=1 \
#     algorithm.scalar_channels=32 \
#     algorithm.stream_channels=8 \
#     algorithm.tensor_invariants=4 \
#     algorithm.vector_channels=16 \
#     checkpoint_manifest_path=logs/train/hls4ml/vn_hgq/nconst16/baseline_nogate_h32_mlp1_beta3e-07 \
#     results_table.invariance_metric=tv \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/hls4ml/vn_hgq/nconst16/baseline_nogate_h32_mlp1_beta3e-07

# ------------------------------------------------------------------------
# vn_hgq  nconst=16  baseline  (method=augmentation, group=none)  beta=3e-07
# latency ablation nogate_h32_mlp1_nostream:
# no gate, head [32], one-layer scalar MLP, no stream
# hand-written, no study
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=hls4ml/vn_hgq_16 \
#     data.method=augmentation \
#     data.augmentation_scheme=none \
#     algorithm.beta=3e-07 \
#     algorithm.dropout=0.0 \
#     algorithm.floatx=float64 \
#     algorithm.gate=false \
#     algorithm.head_layers='[32]' \
#     algorithm.init_from=logs/train/hls4ml/vn/nconst16/baseline_1block_nogate_h32_mlp1_nostream \
#     algorithm.invariants=8 \
#     algorithm.jax_platform=cpu \
#     algorithm.mlp_layers=1 \
#     algorithm.name=vn_hgq_nogate_h32_mlp1_nostream_beta3e-07 \
#     algorithm.num_blocks=1 \
#     algorithm.scalar_channels=32 \
#     algorithm.stream_channels=0 \
#     algorithm.tensor_invariants=4 \
#     algorithm.vector_channels=16 \
#     checkpoint_manifest_path=logs/train/hls4ml/vn_hgq/nconst16/baseline_nogate_h32_mlp1_nostream_beta3e-07 \
#     results_table.invariance_metric=tv \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/hls4ml/vn_hgq/nconst16/baseline_nogate_h32_mlp1_nostream_beta3e-07

# ------------------------------------------------------------------------
# vn nconst=16 baseline, one block, latency ablation nogate_h32_mlp1_nostream_notensor:
# no gate, head [32], one-layer scalar MLP, no stream, no tensor invariants
# hand-written: the float reference with the switches of the variant
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=hls4ml/vn_16 \
#     data.method=augmentation \
#     data.augmentation_scheme=none \
#     algorithm.dropout=0.0 \
#     algorithm.gate=false \
#     algorithm.head_layers='[32]' \
#     algorithm.invariants=8 \
#     algorithm.mlp_layers=1 \
#     algorithm.name=vn_nogate_h32_mlp1_nostream_notensor \
#     algorithm.num_blocks=1 \
#     algorithm.optimizer.lr=0.004986660327956767 \
#     algorithm.optimizer.weight_decay=0.05 \
#     algorithm.scalar_channels=32 \
#     algorithm.stream_channels=0 \
#     algorithm.tensor_invariants=0 \
#     algorithm.vector_channels=16 \
#     checkpoint_manifest_path=logs/train/hls4ml/vn/nconst16/baseline_1block_nogate_h32_mlp1_nostream_notensor \
#     results_table.invariance_metric=tv \
#     results_table.subdir=quantised \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/hls4ml/vn/nconst16/baseline_1block_nogate_h32_mlp1_nostream_notensor

# ------------------------------------------------------------------------
# vn_hgq  nconst=16  baseline  (method=augmentation, group=none)  beta=3e-07
# latency ablation nogate_h32_mlp1_nostream_notensor:
# no gate, head [32], one-layer scalar MLP, no stream, no tensor invariants
# hand-written, no study
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=hls4ml/vn_hgq_16 \
#     data.method=augmentation \
#     data.augmentation_scheme=none \
#     algorithm.beta=3e-07 \
#     algorithm.dropout=0.0 \
#     algorithm.floatx=float64 \
#     algorithm.gate=false \
#     algorithm.head_layers='[32]' \
#     algorithm.init_from=logs/train/hls4ml/vn/nconst16/baseline_1block_nogate_h32_mlp1_nostream_notensor \
#     algorithm.invariants=8 \
#     algorithm.jax_platform=cpu \
#     algorithm.mlp_layers=1 \
#     algorithm.name=vn_hgq_nogate_h32_mlp1_nostream_notensor_beta3e-07 \
#     algorithm.num_blocks=1 \
#     algorithm.scalar_channels=32 \
#     algorithm.stream_channels=0 \
#     algorithm.tensor_invariants=0 \
#     algorithm.vector_channels=16 \
#     checkpoint_manifest_path=logs/train/hls4ml/vn_hgq/nconst16/baseline_nogate_h32_mlp1_nostream_notensor_beta3e-07 \
#     results_table.invariance_metric=tv \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/hls4ml/vn_hgq/nconst16/baseline_nogate_h32_mlp1_nostream_notensor_beta3e-07

# ------------------------------------------------------------------------
# vn nconst=16 baseline, one block, latency ablation nogate_h32_nostream:
# no gate, head [32], no stream, the two-layer scalar MLP kept
# hand-written: the float reference with the switches of the variant
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=hls4ml/vn_16 \
#     data.method=augmentation \
#     data.augmentation_scheme=none \
#     algorithm.dropout=0.0 \
#     algorithm.gate=false \
#     algorithm.head_layers='[32]' \
#     algorithm.invariants=8 \
#     algorithm.name=vn_nogate_h32_nostream \
#     algorithm.num_blocks=1 \
#     algorithm.optimizer.lr=0.004986660327956767 \
#     algorithm.optimizer.weight_decay=0.05 \
#     algorithm.scalar_channels=32 \
#     algorithm.stream_channels=0 \
#     algorithm.tensor_invariants=4 \
#     algorithm.vector_channels=16 \
#     checkpoint_manifest_path=logs/train/hls4ml/vn/nconst16/baseline_1block_nogate_h32_nostream \
#     results_table.invariance_metric=tv \
#     results_table.subdir=quantised \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/hls4ml/vn/nconst16/baseline_1block_nogate_h32_nostream

# ------------------------------------------------------------------------
# vn nconst=16 baseline, one block, latency ablation nogate_h32_nostream_wide:
# no gate, head [32], no stream, at the widths of the vn study's best
# one-block trial (8 vector channels, 64 scalar channels, 32 forms, 2 tensors)
# hand-written: the float reference with the switches of the variant
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=hls4ml/vn_16 \
#     data.method=augmentation \
#     data.augmentation_scheme=none \
#     algorithm.dropout=0.0 \
#     algorithm.gate=false \
#     algorithm.head_layers='[32]' \
#     algorithm.invariants=32 \
#     algorithm.name=vn_nogate_h32_nostream_wide \
#     algorithm.num_blocks=1 \
#     algorithm.optimizer.lr=0.004986660327956767 \
#     algorithm.optimizer.weight_decay=0.05 \
#     algorithm.scalar_channels=64 \
#     algorithm.stream_channels=0 \
#     algorithm.tensor_invariants=2 \
#     algorithm.vector_channels=8 \
#     checkpoint_manifest_path=logs/train/hls4ml/vn/nconst16/baseline_1block_nogate_h32_nostream_wide \
#     results_table.invariance_metric=tv \
#     results_table.subdir=quantised \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/hls4ml/vn/nconst16/baseline_1block_nogate_h32_nostream_wide

# ------------------------------------------------------------------------
# vn_hgq  nconst=16  baseline  (method=augmentation, group=none)  beta=3e-07
# latency ablation nogate_h32_mlp1_nostream_foldtrn:
# no gate, head [32], one-layer scalar MLP, no stream, folded
# normalisations, truncating data lanes
# hand-written, no study
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=hls4ml/vn_hgq_16 \
#     data.method=augmentation \
#     data.augmentation_scheme=none \
#     algorithm.beta=3e-07 \
#     algorithm.dropout=0.0 \
#     algorithm.floatx=float64 \
#     algorithm.gate=false \
#     algorithm.head_layers='[32]' \
#     algorithm.init_from=logs/train/hls4ml/vn/nconst16/baseline_1block_nogate_h32_mlp1_nostream \
#     algorithm.invariants=8 \
#     algorithm.jax_platform=cpu \
#     algorithm.mlp_layers=1 \
#     algorithm.name=vn_hgq_nogate_h32_mlp1_nostream_foldtrn_beta3e-07 \
#     algorithm.num_blocks=1 \
#     algorithm.quantisation.fold_norm=true \
#     algorithm.quantisation.round_mode=TRN \
#     algorithm.scalar_channels=32 \
#     algorithm.stream_channels=0 \
#     algorithm.tensor_invariants=4 \
#     algorithm.vector_channels=16 \
#     checkpoint_manifest_path=logs/train/hls4ml/vn_hgq/nconst16/baseline_nogate_h32_mlp1_nostream_foldtrn_beta3e-07 \
#     results_table.invariance_metric=tv \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/hls4ml/vn_hgq/nconst16/baseline_nogate_h32_mlp1_nostream_foldtrn_beta3e-07

# ------------------------------------------------------------------------
# vn_hgq  nconst=16  baseline  (method=augmentation, group=none)  beta=3e-07
# latency ablation nogate_h32_nostream_foldtrn:
# no gate, head [32], no stream, folded normalisations, truncating data lanes
# hand-written, no study
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=hls4ml/vn_hgq_16 \
#     data.method=augmentation \
#     data.augmentation_scheme=none \
#     algorithm.beta=3e-07 \
#     algorithm.dropout=0.0 \
#     algorithm.floatx=float64 \
#     algorithm.gate=false \
#     algorithm.head_layers='[32]' \
#     algorithm.init_from=logs/train/hls4ml/vn/nconst16/baseline_1block_nogate_h32_nostream \
#     algorithm.invariants=8 \
#     algorithm.jax_platform=cpu \
#     algorithm.name=vn_hgq_nogate_h32_nostream_foldtrn_beta3e-07 \
#     algorithm.num_blocks=1 \
#     algorithm.quantisation.fold_norm=true \
#     algorithm.quantisation.round_mode=TRN \
#     algorithm.scalar_channels=32 \
#     algorithm.stream_channels=0 \
#     algorithm.tensor_invariants=4 \
#     algorithm.vector_channels=16 \
#     checkpoint_manifest_path=logs/train/hls4ml/vn_hgq/nconst16/baseline_nogate_h32_nostream_foldtrn_beta3e-07 \
#     results_table.invariance_metric=tv \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/hls4ml/vn_hgq/nconst16/baseline_nogate_h32_nostream_foldtrn_beta3e-07

# ------------------------------------------------------------------------
# vn_hgq  nconst=16  baseline  (method=augmentation, group=none)  beta=3e-07
# latency ablation nogate_h32_nostream_wide_foldtrn:
# no gate, head [32], no stream, the one-block search's widths, folded
# normalisations, truncating data lanes
# hand-written, no study
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=hls4ml/vn_hgq_16 \
#     data.method=augmentation \
#     data.augmentation_scheme=none \
#     algorithm.beta=3e-07 \
#     algorithm.dropout=0.0 \
#     algorithm.floatx=float64 \
#     algorithm.gate=false \
#     algorithm.head_layers='[32]' \
#     algorithm.init_from=logs/train/hls4ml/vn/nconst16/baseline_1block_nogate_h32_nostream_wide \
#     algorithm.invariants=32 \
#     algorithm.jax_platform=cpu \
#     algorithm.name=vn_hgq_nogate_h32_nostream_wide_foldtrn_beta3e-07 \
#     algorithm.num_blocks=1 \
#     algorithm.quantisation.fold_norm=true \
#     algorithm.quantisation.round_mode=TRN \
#     algorithm.scalar_channels=64 \
#     algorithm.stream_channels=0 \
#     algorithm.tensor_invariants=2 \
#     algorithm.vector_channels=8 \
#     checkpoint_manifest_path=logs/train/hls4ml/vn_hgq/nconst16/baseline_nogate_h32_nostream_wide_foldtrn_beta3e-07 \
#     results_table.invariance_metric=tv \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/hls4ml/vn_hgq/nconst16/baseline_nogate_h32_nostream_wide_foldtrn_beta3e-07

# ========================================================================
# vn and vn_hgq  nconst=16  (the full-axis offset, A1 and A2)
# ========================================================================

# ------------------------------------------------------------------------
# vn nconst=16 baseline, one block, full-axis offset axis_a1:
# no gate, no stream, head [64,32], 128 scalar channels, the leader's eta and
# phi from the full jet's axis as two more first-block scalars
# hand-written: trial 134 of vn_cheap_axis_nconst16_augmentation_none
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=hls4ml/vn_16 \
#     data.method=augmentation \
#     data.augmentation_scheme=none \
#     algorithm.axis_offset=true \
#     algorithm.dropout=0.0 \
#     algorithm.gate=false \
#     algorithm.head_layers='[64,32]' \
#     algorithm.invariants=8 \
#     algorithm.name=vn_axis_a1 \
#     algorithm.nbr_feats=5 \
#     algorithm.num_blocks=1 \
#     algorithm.optimizer.lr=0.007221469209238472 \
#     algorithm.optimizer.weight_decay=0.01 \
#     algorithm.scalar_channels=128 \
#     algorithm.stream_channels=0 \
#     algorithm.tensor_invariants=2 \
#     algorithm.vector_channels=16 \
#     callbacks.invariance_metric.probe=false \
#     checkpoint_manifest_path=logs/train/hls4ml/vn/nconst16/baseline_1block_axis_a1 \
#     data.features='[pt,eta,phi,etarel,phirel]' \
#     data.full_axis=true \
#     results_table.invariance_metric=tv \
#     results_table.subdir=quantised \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/hls4ml/vn/nconst16/baseline_1block_axis_a1

# ------------------------------------------------------------------------
# vn nconst=16 baseline, one block, full-axis offset axis_a2:
# no gate, no stream, head [32], 128 scalar channels, the leader's eta and
# phi from the full jet's axis as two more first-block scalars
# hand-written: the best head-[32] trial of vn_cheap_axis_nconst16_augmentation_none
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=hls4ml/vn_16 \
#     data.method=augmentation \
#     data.augmentation_scheme=none \
#     algorithm.axis_offset=true \
#     algorithm.dropout=0.0 \
#     algorithm.gate=false \
#     algorithm.head_layers='[32]' \
#     algorithm.invariants=32 \
#     algorithm.name=vn_axis_a2 \
#     algorithm.nbr_feats=5 \
#     algorithm.num_blocks=1 \
#     algorithm.optimizer.lr=0.008220876046413613 \
#     algorithm.optimizer.weight_decay=0.0 \
#     algorithm.scalar_channels=128 \
#     algorithm.stream_channels=0 \
#     algorithm.tensor_invariants=1 \
#     algorithm.vector_channels=16 \
#     callbacks.invariance_metric.probe=false \
#     checkpoint_manifest_path=logs/train/hls4ml/vn/nconst16/baseline_1block_axis_a2 \
#     data.features='[pt,eta,phi,etarel,phirel]' \
#     data.full_axis=true \
#     results_table.invariance_metric=tv \
#     results_table.subdir=quantised \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/hls4ml/vn/nconst16/baseline_1block_axis_a2

# ------------------------------------------------------------------------
# vn_hgq  nconst=16  baseline  (method=augmentation, group=none)  beta=3e-07
# full-axis offset axis_a1_foldtrn:
# the design of block 21, folded normalisations, truncating data lanes
# hand-written, no study
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=hls4ml/vn_hgq_16 \
#     data.method=augmentation \
#     data.augmentation_scheme=none \
#     algorithm.axis_offset=true \
#     algorithm.beta=3e-07 \
#     algorithm.dropout=0.0 \
#     algorithm.floatx=float64 \
#     algorithm.gate=false \
#     algorithm.head_layers='[64,32]' \
#     algorithm.init_from=logs/train/hls4ml/vn/nconst16/baseline_1block_axis_a1 \
#     algorithm.invariants=8 \
#     algorithm.jax_platform=cpu \
#     algorithm.name=vn_hgq_axis_a1_foldtrn_beta3e-07 \
#     algorithm.nbr_feats=5 \
#     algorithm.num_blocks=1 \
#     algorithm.quantisation.fold_norm=true \
#     algorithm.quantisation.round_mode=TRN \
#     algorithm.scalar_channels=128 \
#     algorithm.stream_channels=0 \
#     algorithm.tensor_invariants=2 \
#     algorithm.vector_channels=16 \
#     callbacks.invariance_metric.probe=false \
#     checkpoint_manifest_path=logs/train/hls4ml/vn_hgq/nconst16/baseline_axis_a1_foldtrn_beta3e-07 \
#     data.features='[pt,eta,phi,etarel,phirel]' \
#     data.full_axis=true \
#     results_table.invariance_metric=tv \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/hls4ml/vn_hgq/nconst16/baseline_axis_a1_foldtrn_beta3e-07

# ------------------------------------------------------------------------
# vn_hgq  nconst=16  baseline  (method=augmentation, group=none)  beta=3e-07
# full-axis offset axis_a2_foldtrn:
# the design of block 22, folded normalisations, truncating data lanes
# hand-written, no study
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=hls4ml/vn_hgq_16 \
#     data.method=augmentation \
#     data.augmentation_scheme=none \
#     algorithm.axis_offset=true \
#     algorithm.beta=3e-07 \
#     algorithm.dropout=0.0 \
#     algorithm.floatx=float64 \
#     algorithm.gate=false \
#     algorithm.head_layers='[32]' \
#     algorithm.init_from=logs/train/hls4ml/vn/nconst16/baseline_1block_axis_a2 \
#     algorithm.invariants=32 \
#     algorithm.jax_platform=cpu \
#     algorithm.name=vn_hgq_axis_a2_foldtrn_beta3e-07 \
#     algorithm.nbr_feats=5 \
#     algorithm.num_blocks=1 \
#     algorithm.quantisation.fold_norm=true \
#     algorithm.quantisation.round_mode=TRN \
#     algorithm.scalar_channels=128 \
#     algorithm.stream_channels=0 \
#     algorithm.tensor_invariants=1 \
#     algorithm.vector_channels=16 \
#     callbacks.invariance_metric.probe=false \
#     checkpoint_manifest_path=logs/train/hls4ml/vn_hgq/nconst16/baseline_axis_a2_foldtrn_beta3e-07 \
#     data.features='[pt,eta,phi,etarel,phirel]' \
#     data.full_axis=true \
#     results_table.invariance_metric=tv \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/hls4ml/vn_hgq/nconst16/baseline_axis_a2_foldtrn_beta3e-07

# ========================================================================
# vn_hgq  nconst=16  (the beta scan and the learning-rate check of D2 and A2)
# ========================================================================

# ------------------------------------------------------------------------
# vn_hgq  nconst=16  baseline  (method=augmentation, group=none)  beta=1e-07
# D2 (block 19) at beta=1e-07
# hand-written, no study
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=hls4ml/vn_hgq_16 \
#     data.method=augmentation \
#     data.augmentation_scheme=none \
#     algorithm.beta=1e-07 \
#     algorithm.dropout=0.0 \
#     algorithm.floatx=float64 \
#     algorithm.gate=false \
#     algorithm.head_layers='[32]' \
#     algorithm.init_from=logs/train/hls4ml/vn/nconst16/baseline_1block_nogate_h32_nostream \
#     algorithm.invariants=8 \
#     algorithm.jax_platform=cpu \
#     algorithm.name=vn_hgq_nogate_h32_nostream_foldtrn_beta1e-07 \
#     algorithm.num_blocks=1 \
#     algorithm.quantisation.fold_norm=true \
#     algorithm.quantisation.round_mode=TRN \
#     algorithm.scalar_channels=32 \
#     algorithm.stream_channels=0 \
#     algorithm.tensor_invariants=4 \
#     algorithm.vector_channels=16 \
#     checkpoint_manifest_path=logs/train/hls4ml/vn_hgq/nconst16/baseline_nogate_h32_nostream_foldtrn_beta1e-07 \
#     results_table.invariance_metric=tv \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/hls4ml/vn_hgq/nconst16/baseline_nogate_h32_nostream_foldtrn_beta1e-07

# ------------------------------------------------------------------------
# vn_hgq  nconst=16  baseline  (method=augmentation, group=none)  beta=3e-08
# D2 (block 19) at beta=3e-08
# hand-written, no study
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=hls4ml/vn_hgq_16 \
#     data.method=augmentation \
#     data.augmentation_scheme=none \
#     algorithm.beta=3e-08 \
#     algorithm.dropout=0.0 \
#     algorithm.floatx=float64 \
#     algorithm.gate=false \
#     algorithm.head_layers='[32]' \
#     algorithm.init_from=logs/train/hls4ml/vn/nconst16/baseline_1block_nogate_h32_nostream \
#     algorithm.invariants=8 \
#     algorithm.jax_platform=cpu \
#     algorithm.name=vn_hgq_nogate_h32_nostream_foldtrn_beta3e-08 \
#     algorithm.num_blocks=1 \
#     algorithm.quantisation.fold_norm=true \
#     algorithm.quantisation.round_mode=TRN \
#     algorithm.scalar_channels=32 \
#     algorithm.stream_channels=0 \
#     algorithm.tensor_invariants=4 \
#     algorithm.vector_channels=16 \
#     checkpoint_manifest_path=logs/train/hls4ml/vn_hgq/nconst16/baseline_nogate_h32_nostream_foldtrn_beta3e-08 \
#     results_table.invariance_metric=tv \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/hls4ml/vn_hgq/nconst16/baseline_nogate_h32_nostream_foldtrn_beta3e-08

# ------------------------------------------------------------------------
# vn_hgq  nconst=16  baseline  (method=augmentation, group=none)  beta=1e-07
# A2 (block 24) at beta=1e-07
# hand-written, no study
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=hls4ml/vn_hgq_16 \
#     data.method=augmentation \
#     data.augmentation_scheme=none \
#     algorithm.axis_offset=true \
#     algorithm.beta=1e-07 \
#     algorithm.dropout=0.0 \
#     algorithm.floatx=float64 \
#     algorithm.gate=false \
#     algorithm.head_layers='[32]' \
#     algorithm.init_from=logs/train/hls4ml/vn/nconst16/baseline_1block_axis_a2 \
#     algorithm.invariants=32 \
#     algorithm.jax_platform=cpu \
#     algorithm.name=vn_hgq_axis_a2_foldtrn_beta1e-07 \
#     algorithm.nbr_feats=5 \
#     algorithm.num_blocks=1 \
#     algorithm.quantisation.fold_norm=true \
#     algorithm.quantisation.round_mode=TRN \
#     algorithm.scalar_channels=128 \
#     algorithm.stream_channels=0 \
#     algorithm.tensor_invariants=1 \
#     algorithm.vector_channels=16 \
#     callbacks.invariance_metric.probe=false \
#     checkpoint_manifest_path=logs/train/hls4ml/vn_hgq/nconst16/baseline_axis_a2_foldtrn_beta1e-07 \
#     data.features='[pt,eta,phi,etarel,phirel]' \
#     data.full_axis=true \
#     results_table.invariance_metric=tv \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/hls4ml/vn_hgq/nconst16/baseline_axis_a2_foldtrn_beta1e-07

# ------------------------------------------------------------------------
# vn_hgq  nconst=16  baseline  (method=augmentation, group=none)  beta=3e-07
# D2 (block 19) at beta=3e-07, the fine-tune starting at a third of the learning rate
# hand-written, no study
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=hls4ml/vn_hgq_16 \
#     data.method=augmentation \
#     data.augmentation_scheme=none \
#     algorithm.beta=3e-07 \
#     algorithm.dropout=0.0 \
#     algorithm.floatx=float64 \
#     algorithm.gate=false \
#     algorithm.head_layers='[32]' \
#     algorithm.init_from=logs/train/hls4ml/vn/nconst16/baseline_1block_nogate_h32_nostream \
#     algorithm.invariants=8 \
#     algorithm.jax_platform=cpu \
#     algorithm.name=vn_hgq_nogate_h32_nostream_foldtrn_lr3e-04_beta3e-07 \
#     algorithm.num_blocks=1 \
#     algorithm.quantisation.fold_norm=true \
#     algorithm.quantisation.round_mode=TRN \
#     algorithm.scalar_channels=32 \
#     algorithm.scheduler.initial_learning_rate=3.0e-4 \
#     algorithm.stream_channels=0 \
#     algorithm.tensor_invariants=4 \
#     algorithm.vector_channels=16 \
#     checkpoint_manifest_path=logs/train/hls4ml/vn_hgq/nconst16/baseline_nogate_h32_nostream_foldtrn_lr3e-04_beta3e-07 \
#     results_table.invariance_metric=tv \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/hls4ml/vn_hgq/nconst16/baseline_nogate_h32_nostream_foldtrn_lr3e-04_beta3e-07

# ========================================================================
# vn  nconst=50 and 150  (the cheap search's head-[32] designs, without and with the offset)
# ========================================================================

# ------------------------------------------------------------------------
# vn nconst=50 baseline, one block, cheap_h32:
# no gate, no stream, head [32], 32 scalar channels
# hand-written: trial 123 of vn_cheap_nconst50_augmentation_none
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=hls4ml/vn_50 \
#     data.method=augmentation \
#     data.augmentation_scheme=none \
#     algorithm.dropout=0.0 \
#     algorithm.gate=false \
#     algorithm.head_layers='[32]' \
#     algorithm.invariants=32 \
#     algorithm.name=vn_cheap_h32 \
#     algorithm.num_blocks=1 \
#     algorithm.optimizer.lr=0.008837575109767657 \
#     algorithm.optimizer.weight_decay=0.01 \
#     algorithm.scalar_channels=32 \
#     algorithm.stream_channels=0 \
#     algorithm.tensor_invariants=4 \
#     algorithm.vector_channels=8 \
#     checkpoint_manifest_path=logs/train/hls4ml/vn/nconst50/baseline_1block_cheap_h32 \
#     results_table.invariance_metric=tv \
#     results_table.subdir=quantised \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/hls4ml/vn/nconst50/baseline_1block_cheap_h32

# ------------------------------------------------------------------------
# vn nconst=50 baseline, one block, cheap_axis_h32:
# no gate, no stream, head [32], 32 scalar channels, the leader's eta and
# phi from the full jet's axis as two more first-block scalars
# hand-written: trial 130 of vn_cheap_axis_nconst50_augmentation_none
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=hls4ml/vn_50 \
#     data.method=augmentation \
#     data.augmentation_scheme=none \
#     algorithm.axis_offset=true \
#     algorithm.dropout=0.0 \
#     algorithm.gate=false \
#     algorithm.head_layers='[32]' \
#     algorithm.invariants=8 \
#     algorithm.name=vn_cheap_axis_h32 \
#     algorithm.nbr_feats=5 \
#     algorithm.num_blocks=1 \
#     algorithm.optimizer.lr=0.00599615570775775 \
#     algorithm.optimizer.weight_decay=0.01 \
#     algorithm.scalar_channels=32 \
#     algorithm.stream_channels=0 \
#     algorithm.tensor_invariants=4 \
#     algorithm.vector_channels=8 \
#     callbacks.invariance_metric.probe=false \
#     checkpoint_manifest_path=logs/train/hls4ml/vn/nconst50/baseline_1block_cheap_axis_h32 \
#     data.features='[pt,eta,phi,etarel,phirel]' \
#     data.full_axis=true \
#     results_table.invariance_metric=tv \
#     results_table.subdir=quantised \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/hls4ml/vn/nconst50/baseline_1block_cheap_axis_h32

# ------------------------------------------------------------------------
# vn nconst=150 baseline, one block, cheap_h32:
# no gate, no stream, head [32], 32 scalar channels
# hand-written: trial 153 of vn_cheap_nconst150_augmentation_none
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=hls4ml/vn_150 \
#     data.method=augmentation \
#     data.augmentation_scheme=none \
#     algorithm.dropout=0.0 \
#     algorithm.gate=false \
#     algorithm.head_layers='[32]' \
#     algorithm.invariants=8 \
#     algorithm.name=vn_cheap_h32 \
#     algorithm.num_blocks=1 \
#     algorithm.optimizer.lr=0.009926512074188387 \
#     algorithm.optimizer.weight_decay=0.05 \
#     algorithm.scalar_channels=32 \
#     algorithm.stream_channels=0 \
#     algorithm.tensor_invariants=4 \
#     algorithm.vector_channels=16 \
#     checkpoint_manifest_path=logs/train/hls4ml/vn/nconst150/baseline_1block_cheap_h32 \
#     results_table.invariance_metric=tv \
#     results_table.subdir=quantised \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/hls4ml/vn/nconst150/baseline_1block_cheap_h32

# ------------------------------------------------------------------------
# vn nconst=150 baseline, one block, cheap_axis_h32:
# no gate, no stream, head [32], 32 scalar channels, the leader's eta and
# phi from the full jet's axis as two more first-block scalars
# hand-written: trial 162 of vn_cheap_axis_nconst150_augmentation_none
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=hls4ml/vn_150 \
#     data.method=augmentation \
#     data.augmentation_scheme=none \
#     algorithm.axis_offset=true \
#     algorithm.dropout=0.0 \
#     algorithm.gate=false \
#     algorithm.head_layers='[32]' \
#     algorithm.invariants=32 \
#     algorithm.name=vn_cheap_axis_h32 \
#     algorithm.nbr_feats=5 \
#     algorithm.num_blocks=1 \
#     algorithm.optimizer.lr=0.006499577920199478 \
#     algorithm.optimizer.weight_decay=0.01 \
#     algorithm.scalar_channels=32 \
#     algorithm.stream_channels=0 \
#     algorithm.tensor_invariants=4 \
#     algorithm.vector_channels=16 \
#     callbacks.invariance_metric.probe=false \
#     checkpoint_manifest_path=logs/train/hls4ml/vn/nconst150/baseline_1block_cheap_axis_h32 \
#     data.features='[pt,eta,phi,etarel,phirel]' \
#     data.full_axis=true \
#     results_table.invariance_metric=tv \
#     results_table.subdir=quantised \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/hls4ml/vn/nconst150/baseline_1block_cheap_axis_h32

# ========================================================================
# vn_hgq  nconst=50 and 150  (their fine-tunes, beta=3e-07, fold+TRN)
# ========================================================================

# ------------------------------------------------------------------------
# vn_hgq  nconst=50  baseline  (method=augmentation, group=none)  beta=3e-07
# cheap_h32_foldtrn: the design of block 29, folded normalisations, truncating data lanes
# hand-written, no study
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=hls4ml/vn_hgq_50 \
#     data.method=augmentation \
#     data.augmentation_scheme=none \
#     algorithm.beta=3e-07 \
#     algorithm.dropout=0.0 \
#     algorithm.floatx=float64 \
#     algorithm.gate=false \
#     algorithm.head_layers='[32]' \
#     algorithm.init_from=logs/train/hls4ml/vn/nconst50/baseline_1block_cheap_h32 \
#     algorithm.invariants=32 \
#     algorithm.jax_platform=cpu \
#     algorithm.name=vn_hgq_cheap_h32_foldtrn_beta3e-07 \
#     algorithm.num_blocks=1 \
#     algorithm.quantisation.fold_norm=true \
#     algorithm.quantisation.round_mode=TRN \
#     algorithm.scalar_channels=32 \
#     algorithm.stream_channels=0 \
#     algorithm.tensor_invariants=4 \
#     algorithm.vector_channels=8 \
#     checkpoint_manifest_path=logs/train/hls4ml/vn_hgq/nconst50/baseline_cheap_h32_foldtrn_beta3e-07 \
#     results_table.invariance_metric=tv \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/hls4ml/vn_hgq/nconst50/baseline_cheap_h32_foldtrn_beta3e-07

# ------------------------------------------------------------------------
# vn_hgq  nconst=50  baseline  (method=augmentation, group=none)  beta=3e-07
# cheap_axis_h32_foldtrn: the design of block 30, folded normalisations, truncating data lanes
# hand-written, no study
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=hls4ml/vn_hgq_50 \
#     data.method=augmentation \
#     data.augmentation_scheme=none \
#     algorithm.axis_offset=true \
#     algorithm.beta=3e-07 \
#     algorithm.dropout=0.0 \
#     algorithm.floatx=float64 \
#     algorithm.gate=false \
#     algorithm.head_layers='[32]' \
#     algorithm.init_from=logs/train/hls4ml/vn/nconst50/baseline_1block_cheap_axis_h32 \
#     algorithm.invariants=8 \
#     algorithm.jax_platform=cpu \
#     algorithm.name=vn_hgq_cheap_axis_h32_foldtrn_beta3e-07 \
#     algorithm.nbr_feats=5 \
#     algorithm.num_blocks=1 \
#     algorithm.quantisation.fold_norm=true \
#     algorithm.quantisation.round_mode=TRN \
#     algorithm.scalar_channels=32 \
#     algorithm.stream_channels=0 \
#     algorithm.tensor_invariants=4 \
#     algorithm.vector_channels=8 \
#     callbacks.invariance_metric.probe=false \
#     checkpoint_manifest_path=logs/train/hls4ml/vn_hgq/nconst50/baseline_cheap_axis_h32_foldtrn_beta3e-07 \
#     data.features='[pt,eta,phi,etarel,phirel]' \
#     data.full_axis=true \
#     results_table.invariance_metric=tv \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/hls4ml/vn_hgq/nconst50/baseline_cheap_axis_h32_foldtrn_beta3e-07

# ------------------------------------------------------------------------
# vn_hgq  nconst=150  baseline  (method=augmentation, group=none)  beta=3e-07
# cheap_h32_foldtrn: the design of block 31, folded normalisations, truncating data lanes
# hand-written, no study
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=hls4ml/vn_hgq_150 \
#     data.method=augmentation \
#     data.augmentation_scheme=none \
#     algorithm.beta=3e-07 \
#     algorithm.dropout=0.0 \
#     algorithm.floatx=float64 \
#     algorithm.gate=false \
#     algorithm.head_layers='[32]' \
#     algorithm.init_from=logs/train/hls4ml/vn/nconst150/baseline_1block_cheap_h32 \
#     algorithm.invariants=8 \
#     algorithm.jax_platform=cpu \
#     algorithm.name=vn_hgq_cheap_h32_foldtrn_beta3e-07 \
#     algorithm.num_blocks=1 \
#     algorithm.quantisation.fold_norm=true \
#     algorithm.quantisation.round_mode=TRN \
#     algorithm.scalar_channels=32 \
#     algorithm.stream_channels=0 \
#     algorithm.tensor_invariants=4 \
#     algorithm.vector_channels=16 \
#     checkpoint_manifest_path=logs/train/hls4ml/vn_hgq/nconst150/baseline_cheap_h32_foldtrn_beta3e-07 \
#     results_table.invariance_metric=tv \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/hls4ml/vn_hgq/nconst150/baseline_cheap_h32_foldtrn_beta3e-07

# ------------------------------------------------------------------------
# vn_hgq  nconst=150  baseline  (method=augmentation, group=none)  beta=3e-07
# cheap_axis_h32_foldtrn: the design of block 32, folded normalisations, truncating data lanes
# hand-written, no study
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=hls4ml/vn_hgq_150 \
#     data.method=augmentation \
#     data.augmentation_scheme=none \
#     algorithm.axis_offset=true \
#     algorithm.beta=3e-07 \
#     algorithm.dropout=0.0 \
#     algorithm.floatx=float64 \
#     algorithm.gate=false \
#     algorithm.head_layers='[32]' \
#     algorithm.init_from=logs/train/hls4ml/vn/nconst150/baseline_1block_cheap_axis_h32 \
#     algorithm.invariants=32 \
#     algorithm.jax_platform=cpu \
#     algorithm.name=vn_hgq_cheap_axis_h32_foldtrn_beta3e-07 \
#     algorithm.nbr_feats=5 \
#     algorithm.num_blocks=1 \
#     algorithm.quantisation.fold_norm=true \
#     algorithm.quantisation.round_mode=TRN \
#     algorithm.scalar_channels=32 \
#     algorithm.stream_channels=0 \
#     algorithm.tensor_invariants=4 \
#     algorithm.vector_channels=16 \
#     callbacks.invariance_metric.probe=false \
#     checkpoint_manifest_path=logs/train/hls4ml/vn_hgq/nconst150/baseline_cheap_axis_h32_foldtrn_beta3e-07 \
#     data.features='[pt,eta,phi,etarel,phirel]' \
#     data.full_axis=true \
#     results_table.invariance_metric=tv \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/hls4ml/vn_hgq/nconst150/baseline_cheap_axis_h32_foldtrn_beta3e-07

# ========================================================================
# vn  nconst=16  (the two-block finals with the full-axis offset)
# ========================================================================

# ------------------------------------------------------------------------
# vn nconst=16 baseline, two blocks, full-axis offset axis:
# the finals' network, the leader's eta and phi from the full jet's axis
# as two more first-block scalars
# hand-written: best trial 155 of vn_nconst16_augmentation_none with the offset
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=hls4ml/vn_16 \
#     data.method=augmentation \
#     data.augmentation_scheme=none \
#     algorithm.axis_offset=true \
#     algorithm.dropout=0.0 \
#     algorithm.head_layers='[64,32]' \
#     algorithm.invariants=8 \
#     algorithm.name=vn_axis \
#     algorithm.nbr_feats=5 \
#     algorithm.num_blocks=2 \
#     algorithm.optimizer.lr=0.004986660327956767 \
#     algorithm.optimizer.weight_decay=0.05 \
#     algorithm.scalar_channels=32 \
#     algorithm.stream_channels=8 \
#     algorithm.tensor_invariants=4 \
#     algorithm.vector_channels=16 \
#     callbacks.invariance_metric.probe=false \
#     checkpoint_manifest_path=logs/train/hls4ml/vn/nconst16/baseline_axis \
#     data.features='[pt,eta,phi,etarel,phirel]' \
#     data.full_axis=true \
#     results_table.invariance_metric=tv \
#     results_table.subdir=quantised \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/hls4ml/vn/nconst16/baseline_axis

# ========================================================================
# vn_hgq  nconst=16  (D2 and A2 trained with JEDI-linear's recipe, the picked checkpoints)
# ========================================================================

# ------------------------------------------------------------------------
# vn_hgq  nconst=16  baseline  (method=augmentation, group=none)  beta 2e-08 -> 3e-06
# D2 with JEDI-linear's recipe (block 37 of runvn_hgq_train.sh), the checkpoint of epoch 1099
# hand-written, no study
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=hls4ml/vn_hgq_16 \
#     data.method=augmentation \
#     data.augmentation_scheme=none \
#     algorithm.beta_schedule='[[0,2e-8,linear],[2000,3e-7,log],[7000,3e-6,constant]]' \
#     algorithm.dropout=0.0 \
#     algorithm.floatx=float64 \
#     algorithm.gate=false \
#     algorithm.head_layers='[32]' \
#     algorithm.init_from=logs/train/hls4ml/vn/nconst16/baseline_1block_nogate_h32_nostream \
#     algorithm.invariants=8 \
#     algorithm.jax_platform=cpu \
#     algorithm.name=vn_hgq_nogate_h32_nostream_foldtrn_jedirecipe_epoch1099 \
#     algorithm.num_blocks=1 \
#     algorithm.optimizer.global_clipnorm=null \
#     algorithm.optimizer.weight_decay=0.0 \
#     algorithm.quantisation.fold_norm=true \
#     algorithm.quantisation.round_mode=TRN \
#     algorithm.scalar_channels=32 \
#     algorithm.scheduler._target_=src.algorithms.utils.keras_schedules.CosineRestarts \
#     algorithm.scheduler.alpha=3.3333333e-4 \
#     algorithm.scheduler.initial_learning_rate=3e-3 \
#     +algorithm.scheduler.cycles=70 \
#     +algorithm.scheduler.flat_fraction=0.1 \
#     algorithm.stream_channels=0 \
#     algorithm.tensor_invariants=4 \
#     algorithm.vector_channels=16 \
#     ckpt_path=logs/train/hls4ml/vn_hgq/nconst16/baseline_nogate_h32_nostream_foldtrn_jedirecipe/checkpoints/epoch_1099.ckpt \
#     results_table.invariance_metric=tv \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/hls4ml/vn_hgq/nconst16/baseline_nogate_h32_nostream_foldtrn_jedirecipe_epoch1099

# ------------------------------------------------------------------------
# vn_hgq  nconst=16  baseline  (method=augmentation, group=none)  beta 2e-08 -> 3e-06
# A2 with JEDI-linear's recipe (block 38 of runvn_hgq_train.sh), the checkpoint of epoch 1899
# hand-written, no study
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=hls4ml/vn_hgq_16 \
#     data.method=augmentation \
#     data.augmentation_scheme=none \
#     algorithm.axis_offset=true \
#     algorithm.beta_schedule='[[0,2e-8,linear],[2000,3e-7,log],[7000,3e-6,constant]]' \
#     algorithm.dropout=0.0 \
#     algorithm.floatx=float64 \
#     algorithm.gate=false \
#     algorithm.head_layers='[32]' \
#     algorithm.init_from=logs/train/hls4ml/vn/nconst16/baseline_1block_axis_a2 \
#     algorithm.invariants=32 \
#     algorithm.jax_platform=cpu \
#     algorithm.name=vn_hgq_axis_a2_foldtrn_jedirecipe_epoch1899 \
#     algorithm.nbr_feats=5 \
#     algorithm.num_blocks=1 \
#     algorithm.optimizer.global_clipnorm=null \
#     algorithm.optimizer.weight_decay=0.0 \
#     algorithm.quantisation.fold_norm=true \
#     algorithm.quantisation.round_mode=TRN \
#     algorithm.scalar_channels=128 \
#     algorithm.scheduler._target_=src.algorithms.utils.keras_schedules.CosineRestarts \
#     algorithm.scheduler.alpha=3.3333333e-4 \
#     algorithm.scheduler.initial_learning_rate=3e-3 \
#     +algorithm.scheduler.cycles=70 \
#     +algorithm.scheduler.flat_fraction=0.1 \
#     algorithm.stream_channels=0 \
#     algorithm.tensor_invariants=1 \
#     algorithm.vector_channels=16 \
#     callbacks.invariance_metric.probe=false \
#     ckpt_path=logs/train/hls4ml/vn_hgq/nconst16/baseline_axis_a2_foldtrn_jedirecipe/checkpoints/epoch_1899.ckpt \
#     data.features='[pt,eta,phi,etarel,phirel]' \
#     data.full_axis=true \
#     results_table.invariance_metric=tv \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/hls4ml/vn_hgq/nconst16/baseline_axis_a2_foldtrn_jedirecipe_epoch1899

# ------------------------------------------------------------------------
# vn_hgq  nconst=16  baseline  (method=augmentation, group=none)  beta 2e-08 -> 3e-06
# D2 (block 40) with JEDI-linear's recipe and a checkpoint every epoch, its pick nearest 57 237 EBOPs
# hand-written, no study
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=hls4ml/vn_hgq_16 \
#     data.method=augmentation \
#     data.augmentation_scheme=none \
#     algorithm.beta_schedule='[[0,2e-8,linear],[2000,3e-7,log],[7000,3e-6,constant]]' \
#     algorithm.dropout=0.0 \
#     algorithm.floatx=float64 \
#     algorithm.gate=false \
#     algorithm.head_layers='[32]' \
#     algorithm.init_from=logs/train/hls4ml/vn/nconst16/baseline_1block_nogate_h32_nostream \
#     algorithm.invariants=8 \
#     algorithm.jax_platform=cpu \
#     algorithm.name=vn_hgq_nogate_h32_nostream_foldtrn_jedirecipe_everyepoch \
#     algorithm.num_blocks=1 \
#     algorithm.optimizer.global_clipnorm=null \
#     algorithm.optimizer.weight_decay=0.0 \
#     algorithm.quantisation.fold_norm=true \
#     algorithm.quantisation.round_mode=TRN \
#     algorithm.scalar_channels=32 \
#     algorithm.scheduler._target_=src.algorithms.utils.keras_schedules.CosineRestarts \
#     algorithm.scheduler.alpha=3.3333333e-4 \
#     +algorithm.scheduler.cycles=70 \
#     +algorithm.scheduler.flat_fraction=0.1 \
#     algorithm.scheduler.initial_learning_rate=3e-3 \
#     algorithm.stream_channels=0 \
#     algorithm.tensor_invariants=4 \
#     algorithm.vector_channels=16 \
#     checkpoint_manifest_path=logs/train/hls4ml/vn_hgq/nconst16/baseline_nogate_h32_nostream_foldtrn_jedirecipe_everyepoch/picks/target57237 \
#     results_table.invariance_metric=tv \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/hls4ml/vn_hgq/nconst16/baseline_nogate_h32_nostream_foldtrn_jedirecipe_everyepoch_target57237

# ------------------------------------------------------------------------
# vn_hgq  nconst=16  baseline  (method=augmentation, group=none)  beta 2e-08 -> 3e-06
# A2 (block 41) with JEDI-linear's recipe and a checkpoint every epoch, its pick nearest 57 237 EBOPs
# hand-written, no study
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=hls4ml/vn_hgq_16 \
#     data.method=augmentation \
#     data.augmentation_scheme=none \
#     algorithm.axis_offset=true \
#     algorithm.beta_schedule='[[0,2e-8,linear],[2000,3e-7,log],[7000,3e-6,constant]]' \
#     algorithm.dropout=0.0 \
#     algorithm.floatx=float64 \
#     algorithm.gate=false \
#     algorithm.head_layers='[32]' \
#     algorithm.init_from=logs/train/hls4ml/vn/nconst16/baseline_1block_axis_a2 \
#     algorithm.invariants=32 \
#     algorithm.jax_platform=cpu \
#     algorithm.name=vn_hgq_axis_a2_foldtrn_jedirecipe_everyepoch \
#     algorithm.nbr_feats=5 \
#     algorithm.num_blocks=1 \
#     algorithm.optimizer.global_clipnorm=null \
#     algorithm.optimizer.weight_decay=0.0 \
#     algorithm.quantisation.fold_norm=true \
#     algorithm.quantisation.round_mode=TRN \
#     algorithm.scalar_channels=128 \
#     algorithm.scheduler._target_=src.algorithms.utils.keras_schedules.CosineRestarts \
#     algorithm.scheduler.alpha=3.3333333e-4 \
#     +algorithm.scheduler.cycles=70 \
#     +algorithm.scheduler.flat_fraction=0.1 \
#     algorithm.scheduler.initial_learning_rate=3e-3 \
#     algorithm.stream_channels=0 \
#     algorithm.tensor_invariants=1 \
#     algorithm.vector_channels=16 \
#     callbacks.invariance_metric.probe=false \
#     checkpoint_manifest_path=logs/train/hls4ml/vn_hgq/nconst16/baseline_axis_a2_foldtrn_jedirecipe_everyepoch/picks/target57237 \
#     data.features='[pt,eta,phi,etarel,phirel]' \
#     data.full_axis=true \
#     results_table.invariance_metric=tv \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/hls4ml/vn_hgq/nconst16/baseline_axis_a2_foldtrn_jedirecipe_everyepoch_target57237

# ------------------------------------------------------------------------
# vn_hgq  nconst=16  baseline  (method=augmentation, group=none)  beta 2e-08 -> 3e-06
# D2 with trained input words (block 42) with JEDI-linear's recipe and a checkpoint every epoch, its pick nearest 57 237 EBOPs
# hand-written, no study
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=hls4ml/vn_hgq_16 \
#     data.method=augmentation \
#     data.augmentation_scheme=none \
#     algorithm.beta_schedule='[[0,2e-8,linear],[2000,3e-7,log],[7000,3e-6,constant]]' \
#     algorithm.dropout=0.0 \
#     algorithm.floatx=float64 \
#     algorithm.gate=false \
#     algorithm.head_layers='[32]' \
#     algorithm.init_from=logs/train/hls4ml/vn/nconst16/baseline_1block_nogate_h32_nostream \
#     algorithm.invariants=8 \
#     algorithm.jax_platform=cpu \
#     algorithm.name=vn_hgq_nogate_h32_nostream_foldtrn_trainedwords_jedirecipe_everyepoch \
#     algorithm.num_blocks=1 \
#     algorithm.optimizer.global_clipnorm=null \
#     algorithm.optimizer.weight_decay=0.0 \
#     algorithm.quantisation.fold_norm=true \
#     algorithm.quantisation.round_mode=TRN \
#     algorithm.quantisation.trained_words=true \
#     algorithm.scalar_channels=32 \
#     algorithm.scheduler._target_=src.algorithms.utils.keras_schedules.CosineRestarts \
#     algorithm.scheduler.alpha=3.3333333e-4 \
#     +algorithm.scheduler.cycles=70 \
#     +algorithm.scheduler.flat_fraction=0.1 \
#     algorithm.scheduler.initial_learning_rate=3e-3 \
#     algorithm.stream_channels=0 \
#     algorithm.tensor_invariants=4 \
#     algorithm.vector_channels=16 \
#     checkpoint_manifest_path=logs/train/hls4ml/vn_hgq/nconst16/baseline_nogate_h32_nostream_foldtrn_trainedwords_jedirecipe_everyepoch/picks/target57237 \
#     results_table.invariance_metric=tv \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/hls4ml/vn_hgq/nconst16/baseline_nogate_h32_nostream_foldtrn_trainedwords_jedirecipe_everyepoch_target57237

# ------------------------------------------------------------------------
# vn_hgq  nconst=16  baseline  (method=augmentation, group=none)  beta 2e-08 -> 3e-06
# A2 with trained input words (block 43) with JEDI-linear's recipe and a checkpoint every epoch, its pick nearest 57 237 EBOPs
# hand-written, no study
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=hls4ml/vn_hgq_16 \
#     data.method=augmentation \
#     data.augmentation_scheme=none \
#     algorithm.axis_offset=true \
#     algorithm.beta_schedule='[[0,2e-8,linear],[2000,3e-7,log],[7000,3e-6,constant]]' \
#     algorithm.dropout=0.0 \
#     algorithm.floatx=float64 \
#     algorithm.gate=false \
#     algorithm.head_layers='[32]' \
#     algorithm.init_from=logs/train/hls4ml/vn/nconst16/baseline_1block_axis_a2 \
#     algorithm.invariants=32 \
#     algorithm.jax_platform=cpu \
#     algorithm.name=vn_hgq_axis_a2_foldtrn_trainedwords_jedirecipe_everyepoch \
#     algorithm.nbr_feats=5 \
#     algorithm.num_blocks=1 \
#     algorithm.optimizer.global_clipnorm=null \
#     algorithm.optimizer.weight_decay=0.0 \
#     algorithm.quantisation.fold_norm=true \
#     algorithm.quantisation.round_mode=TRN \
#     algorithm.quantisation.trained_words=true \
#     algorithm.scalar_channels=128 \
#     algorithm.scheduler._target_=src.algorithms.utils.keras_schedules.CosineRestarts \
#     algorithm.scheduler.alpha=3.3333333e-4 \
#     +algorithm.scheduler.cycles=70 \
#     +algorithm.scheduler.flat_fraction=0.1 \
#     algorithm.scheduler.initial_learning_rate=3e-3 \
#     algorithm.stream_channels=0 \
#     algorithm.tensor_invariants=1 \
#     algorithm.vector_channels=16 \
#     callbacks.invariance_metric.probe=false \
#     checkpoint_manifest_path=logs/train/hls4ml/vn_hgq/nconst16/baseline_axis_a2_foldtrn_trainedwords_jedirecipe_everyepoch/picks/target57237 \
#     data.features='[pt,eta,phi,etarel,phirel]' \
#     data.full_axis=true \
#     results_table.invariance_metric=tv \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/hls4ml/vn_hgq/nconst16/baseline_axis_a2_foldtrn_trainedwords_jedirecipe_everyepoch_target57237

# ========================================================================
# vn_hgq  nconst=16, 50 and 150  (the designs of the comparison at a peak learning rate of 1e-3, picked under 100 ns)
# ========================================================================

# ------------------------------------------------------------------------
# vn_hgq  nconst=16  baseline  (method=augmentation, group=none)  beta 2e-08 -> 3e-06
# D2 (block 48) with JEDI-linear's recipe at a peak learning rate of 1e-3 and a checkpoint every epoch, its pick the most accurate under 100 ns
# hand-written, no study
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=hls4ml/vn_hgq_16 \
#     data.method=augmentation \
#     data.augmentation_scheme=none \
#     algorithm.beta_schedule='[[0,2e-8,linear],[2000,3e-7,log],[7000,3e-6,constant]]' \
#     algorithm.dropout=0.0 \
#     algorithm.floatx=float64 \
#     algorithm.gate=false \
#     algorithm.head_layers='[32]' \
#     algorithm.init_from=logs/train/hls4ml/vn/nconst16/baseline_1block_nogate_h32_nostream \
#     algorithm.invariants=8 \
#     algorithm.jax_platform=cpu \
#     algorithm.name=vn_hgq_nogate_h32_nostream_foldtrn_jedirecipe_lr1e-3_everyepoch \
#     algorithm.num_blocks=1 \
#     algorithm.optimizer.global_clipnorm=null \
#     algorithm.optimizer.weight_decay=0.0 \
#     algorithm.quantisation.fold_norm=true \
#     algorithm.quantisation.round_mode=TRN \
#     algorithm.scalar_channels=32 \
#     algorithm.scheduler._target_=src.algorithms.utils.keras_schedules.CosineRestarts \
#     algorithm.scheduler.alpha=3.3333333e-4 \
#     +algorithm.scheduler.cycles=70 \
#     +algorithm.scheduler.flat_fraction=0.1 \
#     algorithm.scheduler.initial_learning_rate=1e-3 \
#     algorithm.stream_channels=0 \
#     algorithm.tensor_invariants=4 \
#     algorithm.vector_channels=16 \
#     checkpoint_manifest_path=logs/train/hls4ml/vn_hgq/nconst16/baseline_nogate_h32_nostream_foldtrn_jedirecipe_lr1e-3_everyepoch/picks/best100ns \
#     results_table.invariance_metric=tv \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/hls4ml/vn_hgq/nconst16/baseline_nogate_h32_nostream_foldtrn_jedirecipe_lr1e-3_everyepoch_best100ns

# ------------------------------------------------------------------------
# vn_hgq  nconst=16  baseline  (method=augmentation, group=none)  beta 2e-08 -> 3e-06
# A2 with trained input words (block 49) with JEDI-linear's recipe at a peak learning rate of 1e-3 and a checkpoint every epoch, its pick the most accurate under 100 ns
# hand-written, no study
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=hls4ml/vn_hgq_16 \
#     data.method=augmentation \
#     data.augmentation_scheme=none \
#     algorithm.axis_offset=true \
#     algorithm.beta_schedule='[[0,2e-8,linear],[2000,3e-7,log],[7000,3e-6,constant]]' \
#     algorithm.dropout=0.0 \
#     algorithm.floatx=float64 \
#     algorithm.gate=false \
#     algorithm.head_layers='[32]' \
#     algorithm.init_from=logs/train/hls4ml/vn/nconst16/baseline_1block_axis_a2 \
#     algorithm.invariants=32 \
#     algorithm.jax_platform=cpu \
#     algorithm.name=vn_hgq_axis_a2_foldtrn_trainedwords_jedirecipe_lr1e-3_everyepoch \
#     algorithm.nbr_feats=5 \
#     algorithm.num_blocks=1 \
#     algorithm.optimizer.global_clipnorm=null \
#     algorithm.optimizer.weight_decay=0.0 \
#     algorithm.quantisation.fold_norm=true \
#     algorithm.quantisation.round_mode=TRN \
#     algorithm.quantisation.trained_words=true \
#     algorithm.scalar_channels=128 \
#     algorithm.scheduler._target_=src.algorithms.utils.keras_schedules.CosineRestarts \
#     algorithm.scheduler.alpha=3.3333333e-4 \
#     +algorithm.scheduler.cycles=70 \
#     +algorithm.scheduler.flat_fraction=0.1 \
#     algorithm.scheduler.initial_learning_rate=1e-3 \
#     algorithm.stream_channels=0 \
#     algorithm.tensor_invariants=1 \
#     algorithm.vector_channels=16 \
#     callbacks.invariance_metric.probe=false \
#     checkpoint_manifest_path=logs/train/hls4ml/vn_hgq/nconst16/baseline_axis_a2_foldtrn_trainedwords_jedirecipe_lr1e-3_everyepoch/picks/best100ns \
#     data.features='[pt,eta,phi,etarel,phirel]' \
#     data.full_axis=true \
#     results_table.invariance_metric=tv \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/hls4ml/vn_hgq/nconst16/baseline_axis_a2_foldtrn_trainedwords_jedirecipe_lr1e-3_everyepoch_best100ns

# ------------------------------------------------------------------------
# vn_hgq  nconst=50  baseline  (method=augmentation, group=none)  beta 2e-08 -> 3e-06
# the design without the axis offset (block 50) with JEDI-linear's recipe at a peak learning rate of 1e-3 and a checkpoint every epoch, its pick the most accurate under 100 ns
# hand-written, no study
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=hls4ml/vn_hgq_50 \
#     data.method=augmentation \
#     data.augmentation_scheme=none \
#     algorithm.beta_schedule='[[0,2e-8,linear],[2000,3e-7,log],[7000,3e-6,constant]]' \
#     algorithm.dropout=0.0 \
#     algorithm.floatx=float64 \
#     algorithm.gate=false \
#     algorithm.head_layers='[32]' \
#     algorithm.init_from=logs/train/hls4ml/vn/nconst50/baseline_1block_cheap_h32 \
#     algorithm.invariants=32 \
#     algorithm.jax_platform=cpu \
#     algorithm.name=vn_hgq_cheap_h32_foldtrn_jedirecipe_lr1e-3_everyepoch \
#     algorithm.num_blocks=1 \
#     algorithm.optimizer.global_clipnorm=null \
#     algorithm.optimizer.weight_decay=0.0 \
#     algorithm.quantisation.fold_norm=true \
#     algorithm.quantisation.round_mode=TRN \
#     algorithm.scalar_channels=32 \
#     algorithm.scheduler._target_=src.algorithms.utils.keras_schedules.CosineRestarts \
#     algorithm.scheduler.alpha=3.3333333e-4 \
#     +algorithm.scheduler.cycles=70 \
#     +algorithm.scheduler.flat_fraction=0.1 \
#     algorithm.scheduler.initial_learning_rate=1e-3 \
#     algorithm.stream_channels=0 \
#     algorithm.tensor_invariants=4 \
#     algorithm.vector_channels=8 \
#     checkpoint_manifest_path=logs/train/hls4ml/vn_hgq/nconst50/baseline_cheap_h32_foldtrn_jedirecipe_lr1e-3_everyepoch/picks/best100ns \
#     results_table.invariance_metric=tv \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/hls4ml/vn_hgq/nconst50/baseline_cheap_h32_foldtrn_jedirecipe_lr1e-3_everyepoch_best100ns

# ------------------------------------------------------------------------
# vn_hgq  nconst=50  baseline  (method=augmentation, group=none)  beta 2e-08 -> 3e-06
# the design with the full-axis offset and trained input words (block 51) with JEDI-linear's recipe at a peak learning rate of 1e-3 and a checkpoint every epoch, its pick the most accurate under 100 ns
# hand-written, no study
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=hls4ml/vn_hgq_50 \
#     data.method=augmentation \
#     data.augmentation_scheme=none \
#     algorithm.axis_offset=true \
#     algorithm.beta_schedule='[[0,2e-8,linear],[2000,3e-7,log],[7000,3e-6,constant]]' \
#     algorithm.dropout=0.0 \
#     algorithm.floatx=float64 \
#     algorithm.gate=false \
#     algorithm.head_layers='[32]' \
#     algorithm.init_from=logs/train/hls4ml/vn/nconst50/baseline_1block_cheap_axis_h32 \
#     algorithm.invariants=8 \
#     algorithm.jax_platform=cpu \
#     algorithm.name=vn_hgq_cheap_axis_h32_foldtrn_trainedwords_jedirecipe_lr1e-3_everyepoch \
#     algorithm.nbr_feats=5 \
#     algorithm.num_blocks=1 \
#     algorithm.optimizer.global_clipnorm=null \
#     algorithm.optimizer.weight_decay=0.0 \
#     algorithm.quantisation.fold_norm=true \
#     algorithm.quantisation.round_mode=TRN \
#     algorithm.quantisation.trained_words=true \
#     algorithm.scalar_channels=32 \
#     algorithm.scheduler._target_=src.algorithms.utils.keras_schedules.CosineRestarts \
#     algorithm.scheduler.alpha=3.3333333e-4 \
#     +algorithm.scheduler.cycles=70 \
#     +algorithm.scheduler.flat_fraction=0.1 \
#     algorithm.scheduler.initial_learning_rate=1e-3 \
#     algorithm.stream_channels=0 \
#     algorithm.tensor_invariants=4 \
#     algorithm.vector_channels=8 \
#     callbacks.invariance_metric.probe=false \
#     checkpoint_manifest_path=logs/train/hls4ml/vn_hgq/nconst50/baseline_cheap_axis_h32_foldtrn_trainedwords_jedirecipe_lr1e-3_everyepoch/picks/best100ns \
#     data.features='[pt,eta,phi,etarel,phirel]' \
#     data.full_axis=true \
#     results_table.invariance_metric=tv \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/hls4ml/vn_hgq/nconst50/baseline_cheap_axis_h32_foldtrn_trainedwords_jedirecipe_lr1e-3_everyepoch_best100ns

# ------------------------------------------------------------------------
# vn_hgq  nconst=150  baseline  (method=augmentation, group=none)  beta 2e-08 -> 3e-06
# the design without the axis offset (block 52) with JEDI-linear's recipe at a peak learning rate of 1e-3 and a checkpoint every epoch, its pick the most accurate under 100 ns
# hand-written, no study
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=hls4ml/vn_hgq_150 \
#     data.method=augmentation \
#     data.augmentation_scheme=none \
#     algorithm.beta_schedule='[[0,2e-8,linear],[2000,3e-7,log],[7000,3e-6,constant]]' \
#     algorithm.dropout=0.0 \
#     algorithm.floatx=float64 \
#     algorithm.gate=false \
#     algorithm.head_layers='[32]' \
#     algorithm.init_from=logs/train/hls4ml/vn/nconst150/baseline_1block_cheap_h32 \
#     algorithm.invariants=8 \
#     algorithm.jax_platform=cpu \
#     algorithm.name=vn_hgq_cheap_h32_foldtrn_jedirecipe_lr1e-3_everyepoch \
#     algorithm.num_blocks=1 \
#     algorithm.optimizer.global_clipnorm=null \
#     algorithm.optimizer.weight_decay=0.0 \
#     algorithm.quantisation.fold_norm=true \
#     algorithm.quantisation.round_mode=TRN \
#     algorithm.scalar_channels=32 \
#     algorithm.scheduler._target_=src.algorithms.utils.keras_schedules.CosineRestarts \
#     algorithm.scheduler.alpha=3.3333333e-4 \
#     +algorithm.scheduler.cycles=70 \
#     +algorithm.scheduler.flat_fraction=0.1 \
#     algorithm.scheduler.initial_learning_rate=1e-3 \
#     algorithm.stream_channels=0 \
#     algorithm.tensor_invariants=4 \
#     algorithm.vector_channels=16 \
#     checkpoint_manifest_path=logs/train/hls4ml/vn_hgq/nconst150/baseline_cheap_h32_foldtrn_jedirecipe_lr1e-3_everyepoch/picks/best100ns \
#     results_table.invariance_metric=tv \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/hls4ml/vn_hgq/nconst150/baseline_cheap_h32_foldtrn_jedirecipe_lr1e-3_everyepoch_best100ns

# ------------------------------------------------------------------------
# vn_hgq  nconst=150  baseline  (method=augmentation, group=none)  beta 2e-08 -> 3e-06
# the design with the full-axis offset and trained input words (block 53) with JEDI-linear's recipe at a peak learning rate of 1e-3 and a checkpoint every epoch, its pick the most accurate under 100 ns
# hand-written, no study
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=hls4ml/vn_hgq_150 \
#     data.method=augmentation \
#     data.augmentation_scheme=none \
#     algorithm.axis_offset=true \
#     algorithm.beta_schedule='[[0,2e-8,linear],[2000,3e-7,log],[7000,3e-6,constant]]' \
#     algorithm.dropout=0.0 \
#     algorithm.floatx=float64 \
#     algorithm.gate=false \
#     algorithm.head_layers='[32]' \
#     algorithm.init_from=logs/train/hls4ml/vn/nconst150/baseline_1block_cheap_axis_h32 \
#     algorithm.invariants=32 \
#     algorithm.jax_platform=cpu \
#     algorithm.name=vn_hgq_cheap_axis_h32_foldtrn_trainedwords_jedirecipe_lr1e-3_everyepoch \
#     algorithm.nbr_feats=5 \
#     algorithm.num_blocks=1 \
#     algorithm.optimizer.global_clipnorm=null \
#     algorithm.optimizer.weight_decay=0.0 \
#     algorithm.quantisation.fold_norm=true \
#     algorithm.quantisation.round_mode=TRN \
#     algorithm.quantisation.trained_words=true \
#     algorithm.scalar_channels=32 \
#     algorithm.scheduler._target_=src.algorithms.utils.keras_schedules.CosineRestarts \
#     algorithm.scheduler.alpha=3.3333333e-4 \
#     +algorithm.scheduler.cycles=70 \
#     +algorithm.scheduler.flat_fraction=0.1 \
#     algorithm.scheduler.initial_learning_rate=1e-3 \
#     algorithm.stream_channels=0 \
#     algorithm.tensor_invariants=4 \
#     algorithm.vector_channels=16 \
#     callbacks.invariance_metric.probe=false \
#     checkpoint_manifest_path=logs/train/hls4ml/vn_hgq/nconst150/baseline_cheap_axis_h32_foldtrn_trainedwords_jedirecipe_lr1e-3_everyepoch/picks/best100ns \
#     data.features='[pt,eta,phi,etarel,phirel]' \
#     data.full_axis=true \
#     results_table.invariance_metric=tv \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/hls4ml/vn_hgq/nconst150/baseline_cheap_axis_h32_foldtrn_trainedwords_jedirecipe_lr1e-3_everyepoch_best100ns
