#!/usr/bin/env bash
# ========================================================================
# VN_HGQ TRAINING COMMANDS
# ========================================================================
# Twenty-eight train runs at 16 constituents: the float reference, one block per
# beta, then the latency ablation. Hand-written: vn_hgq has no study and the
# reference is a variant of a study's best, so
# scripts/optuna/make_train_catalogues.py neither writes nor regenerates
# this file.
# Baseline only: the vector neurons embed the symmetry, so the augmentation
# and SCL regimes are not run on them.
# The first block trains the float reference that algorithm.init_from of
# blocks 2 to 5 names: the main run (fold=null) of the vn study's best
# at 16 constituents (lr 0.004986660327956767, weight decay 0.05) with one
# block in place of two, in logs/train/hls4ml/vn/nconst16/baseline_1block.
# One block rather than the two-block finals: it comes within 0.1 points of
# the two-block best at about half the depth and cost, and the paper shows
# the one-block scan alone. It must finish before the beta blocks start.
# Blocks 2 to 5 each fine-tune that reference at one constant beta; the last
# epoch is the model. The grid 1e-08 ... 3e-07 was fixed by the dry run of
# 2026-10-07 on the two-block finals: at the seeded 14 data and 8 weight
# bits, beta times the initial EBOPs over the initial cross-entropy ran from
# 0.037 to 1.1.
# The four beta blocks share the label vn_hgq_nconst16_baseline, so select
# them with --only <index> (2 to 5), not by name; the reference is block 1.
# The five folds of blocks 1 to 5 were fitted on 2026-10-08 with the fold
# commands of submit_train.sh --folds 5, run beside the existing main runs
# rather than through the launcher, which would refit the main run.
# The latency ablation of the one-block network switches off, one after
# another, the channel gate (algorithm.gate=false), the head's 64-wide hidden
# layer (head_layers [32]), the second layer of the scalar MLP
# (algorithm.mlp_layers=1), the per-constituent stream
# (algorithm.stream_channels=0) and the tensor invariants
# (algorithm.tensor_invariants=0). Blocks 6 to 9 train the first four float
# variants of the reference, in logs/train/hls4ml/vn/nconst16/
# baseline_1block_<tag>; blocks 10 to 13 fine-tune each at beta=3e-07 from
# its own float run, which must finish first. Block 14 trains the fifth
# variant and block 15 fine-tunes it in the same way, after block 14. Their
# labels repeat those of blocks 1 to 5, so select them by index too.
# Blocks 16 and 17 train two further float variants, both without the gate,
# with head [32], the two-layer scalar MLP and no stream: block 16 at the
# reference's widths (baseline_1block_nogate_h32_nostream) and block 17 at
# the widths of the vn study's best one-block trial, 8 vector channels, 64
# scalar channels, 32 forms and 2 tensor invariants
# (baseline_1block_nogate_h32_nostream_wide). Blocks 18 to 20 fine-tune the
# no-stream variant of block 9 and these two at beta=3e-07 with the batch
# normalisations folded into the dense layers that follow them
# (algorithm.quantisation.fold_norm=true) and truncating data lanes
# (algorithm.quantisation.round_mode=TRN), which take the normalisations'
# constant multiplies and the half-LSB rounding adders out of the design;
# their run directories are baseline_<tag>_foldtrn_beta3e-07.
# Blocks 21 and 22 train the two designs of the cheap search with the
# leader's offset from the full jet's axis (data.full_axis=true,
# algorithm.axis_offset=true; vn_cheap_axis_optuna.yaml) that go on to
# synthesis: A1, the search's best (head [64,32], 128 scalar channels), and
# A2, its best with head [32] (baseline_1block_axis_<tag>). Blocks 23 and 24
# fine-tune them at beta=3e-07 with folded normalisations and truncating data
# lanes (baseline_axis_<tag>_foldtrn_beta3e-07).
# Blocks 25 to 28 place the two chosen designs, D2 (block 19) and A2 (block 24),
# on their accuracy-latency fronts: D2 at beta=1e-07 and 3e-08, A2 at 1e-07,
# and D2 at beta=3e-07 with the fine-tune starting at a third of its learning
# rate (algorithm.scheduler.initial_learning_rate=3e-04), to see what the first
# epoch's drop costs (baseline_<tag>_foldtrn[_lr3e-04]_beta<beta>).
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

# ========================================================================
# latency ablation of the one-block network  nconst=16
# ========================================================================

# ------------------------------------------------------------------------
# vn nconst=16 baseline, one block, latency ablation nogate:
# no channel gate
# hand-written: the float reference with the switches of the variant
# ------------------------------------------------------------------------
# python3 src/train.py \
#     experiment=hls4ml/vn_16 \
#     data.method=augmentation \
#     data.augmentation_scheme=none \
#     fold=null \
#     algorithm.dropout=0.0 \
#     algorithm.gate=false \
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
#     hydra.run.dir=logs/train/hls4ml/vn/nconst16/baseline_1block_nogate

# ------------------------------------------------------------------------
# vn nconst=16 baseline, one block, latency ablation nogate_h32:
# no gate, head [32]
# hand-written: the float reference with the switches of the variant
# ------------------------------------------------------------------------
# python3 src/train.py \
#     experiment=hls4ml/vn_16 \
#     data.method=augmentation \
#     data.augmentation_scheme=none \
#     fold=null \
#     algorithm.dropout=0.0 \
#     algorithm.gate=false \
#     algorithm.head_layers='[32]' \
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
#     hydra.run.dir=logs/train/hls4ml/vn/nconst16/baseline_1block_nogate_h32

# ------------------------------------------------------------------------
# vn nconst=16 baseline, one block, latency ablation nogate_h32_mlp1:
# no gate, head [32], one-layer scalar MLP
# hand-written: the float reference with the switches of the variant
# ------------------------------------------------------------------------
# python3 src/train.py \
#     experiment=hls4ml/vn_16 \
#     data.method=augmentation \
#     data.augmentation_scheme=none \
#     fold=null \
#     algorithm.dropout=0.0 \
#     algorithm.gate=false \
#     algorithm.head_layers='[32]' \
#     algorithm.invariants=8 \
#     algorithm.mlp_layers=1 \
#     algorithm.num_blocks=1 \
#     algorithm.optimizer.lr=0.004986660327956767 \
#     algorithm.optimizer.weight_decay=0.05 \
#     algorithm.scalar_channels=32 \
#     algorithm.stream_channels=8 \
#     algorithm.tensor_invariants=4 \
#     algorithm.vector_channels=16 \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/train/hls4ml/vn/nconst16/baseline_1block_nogate_h32_mlp1

# ------------------------------------------------------------------------
# vn nconst=16 baseline, one block, latency ablation nogate_h32_mlp1_nostream:
# no gate, head [32], one-layer scalar MLP, no stream
# hand-written: the float reference with the switches of the variant
# ------------------------------------------------------------------------
# python3 src/train.py \
#     experiment=hls4ml/vn_16 \
#     data.method=augmentation \
#     data.augmentation_scheme=none \
#     fold=null \
#     algorithm.dropout=0.0 \
#     algorithm.gate=false \
#     algorithm.head_layers='[32]' \
#     algorithm.invariants=8 \
#     algorithm.mlp_layers=1 \
#     algorithm.num_blocks=1 \
#     algorithm.optimizer.lr=0.004986660327956767 \
#     algorithm.optimizer.weight_decay=0.05 \
#     algorithm.scalar_channels=32 \
#     algorithm.stream_channels=0 \
#     algorithm.tensor_invariants=4 \
#     algorithm.vector_channels=16 \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/train/hls4ml/vn/nconst16/baseline_1block_nogate_h32_mlp1_nostream

# ------------------------------------------------------------------------
# vn_hgq  nconst=16  baseline  (method=augmentation, group=none)  beta=3e-07
# latency ablation nogate:
# no channel gate
# hand-written, no study
# ------------------------------------------------------------------------
# python3 src/train.py \
#     experiment=hls4ml/vn_hgq_16 \
#     data.method=augmentation \
#     data.augmentation_scheme=none \
#     fold=null \
#     algorithm.beta=3e-07 \
#     algorithm.dropout=0.0 \
#     algorithm.gate=false \
#     algorithm.head_layers='[64,32]' \
#     algorithm.init_from=logs/train/hls4ml/vn/nconst16/baseline_1block_nogate \
#     algorithm.invariants=8 \
#     algorithm.num_blocks=1 \
#     algorithm.scalar_channels=32 \
#     algorithm.stream_channels=8 \
#     algorithm.tensor_invariants=4 \
#     algorithm.vector_channels=16 \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/train/hls4ml/vn_hgq/nconst16/baseline_nogate_beta3e-07

# ------------------------------------------------------------------------
# vn_hgq  nconst=16  baseline  (method=augmentation, group=none)  beta=3e-07
# latency ablation nogate_h32:
# no gate, head [32]
# hand-written, no study
# ------------------------------------------------------------------------
# python3 src/train.py \
#     experiment=hls4ml/vn_hgq_16 \
#     data.method=augmentation \
#     data.augmentation_scheme=none \
#     fold=null \
#     algorithm.beta=3e-07 \
#     algorithm.dropout=0.0 \
#     algorithm.gate=false \
#     algorithm.head_layers='[32]' \
#     algorithm.init_from=logs/train/hls4ml/vn/nconst16/baseline_1block_nogate_h32 \
#     algorithm.invariants=8 \
#     algorithm.num_blocks=1 \
#     algorithm.scalar_channels=32 \
#     algorithm.stream_channels=8 \
#     algorithm.tensor_invariants=4 \
#     algorithm.vector_channels=16 \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/train/hls4ml/vn_hgq/nconst16/baseline_nogate_h32_beta3e-07

# ------------------------------------------------------------------------
# vn_hgq  nconst=16  baseline  (method=augmentation, group=none)  beta=3e-07
# latency ablation nogate_h32_mlp1:
# no gate, head [32], one-layer scalar MLP
# hand-written, no study
# ------------------------------------------------------------------------
# python3 src/train.py \
#     experiment=hls4ml/vn_hgq_16 \
#     data.method=augmentation \
#     data.augmentation_scheme=none \
#     fold=null \
#     algorithm.beta=3e-07 \
#     algorithm.dropout=0.0 \
#     algorithm.gate=false \
#     algorithm.head_layers='[32]' \
#     algorithm.init_from=logs/train/hls4ml/vn/nconst16/baseline_1block_nogate_h32_mlp1 \
#     algorithm.invariants=8 \
#     algorithm.mlp_layers=1 \
#     algorithm.num_blocks=1 \
#     algorithm.scalar_channels=32 \
#     algorithm.stream_channels=8 \
#     algorithm.tensor_invariants=4 \
#     algorithm.vector_channels=16 \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/train/hls4ml/vn_hgq/nconst16/baseline_nogate_h32_mlp1_beta3e-07

# ------------------------------------------------------------------------
# vn_hgq  nconst=16  baseline  (method=augmentation, group=none)  beta=3e-07
# latency ablation nogate_h32_mlp1_nostream:
# no gate, head [32], one-layer scalar MLP, no stream
# hand-written, no study
# ------------------------------------------------------------------------
# python3 src/train.py \
#     experiment=hls4ml/vn_hgq_16 \
#     data.method=augmentation \
#     data.augmentation_scheme=none \
#     fold=null \
#     algorithm.beta=3e-07 \
#     algorithm.dropout=0.0 \
#     algorithm.gate=false \
#     algorithm.head_layers='[32]' \
#     algorithm.init_from=logs/train/hls4ml/vn/nconst16/baseline_1block_nogate_h32_mlp1_nostream \
#     algorithm.invariants=8 \
#     algorithm.mlp_layers=1 \
#     algorithm.num_blocks=1 \
#     algorithm.scalar_channels=32 \
#     algorithm.stream_channels=0 \
#     algorithm.tensor_invariants=4 \
#     algorithm.vector_channels=16 \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/train/hls4ml/vn_hgq/nconst16/baseline_nogate_h32_mlp1_nostream_beta3e-07

# ------------------------------------------------------------------------
# vn nconst=16 baseline, one block, latency ablation nogate_h32_mlp1_nostream_notensor:
# no gate, head [32], one-layer scalar MLP, no stream, no tensor invariants
# hand-written: the float reference with the switches of the variant
# ------------------------------------------------------------------------
# python3 src/train.py \
#     experiment=hls4ml/vn_16 \
#     data.method=augmentation \
#     data.augmentation_scheme=none \
#     fold=null \
#     algorithm.dropout=0.0 \
#     algorithm.gate=false \
#     algorithm.head_layers='[32]' \
#     algorithm.invariants=8 \
#     algorithm.mlp_layers=1 \
#     algorithm.num_blocks=1 \
#     algorithm.optimizer.lr=0.004986660327956767 \
#     algorithm.optimizer.weight_decay=0.05 \
#     algorithm.scalar_channels=32 \
#     algorithm.stream_channels=0 \
#     algorithm.tensor_invariants=0 \
#     algorithm.vector_channels=16 \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/train/hls4ml/vn/nconst16/baseline_1block_nogate_h32_mlp1_nostream_notensor

# ------------------------------------------------------------------------
# vn_hgq  nconst=16  baseline  (method=augmentation, group=none)  beta=3e-07
# latency ablation nogate_h32_mlp1_nostream_notensor:
# no gate, head [32], one-layer scalar MLP, no stream, no tensor invariants
# hand-written, no study
# ------------------------------------------------------------------------
# python3 src/train.py \
#     experiment=hls4ml/vn_hgq_16 \
#     data.method=augmentation \
#     data.augmentation_scheme=none \
#     fold=null \
#     algorithm.beta=3e-07 \
#     algorithm.dropout=0.0 \
#     algorithm.gate=false \
#     algorithm.head_layers='[32]' \
#     algorithm.init_from=logs/train/hls4ml/vn/nconst16/baseline_1block_nogate_h32_mlp1_nostream_notensor \
#     algorithm.invariants=8 \
#     algorithm.mlp_layers=1 \
#     algorithm.num_blocks=1 \
#     algorithm.scalar_channels=32 \
#     algorithm.stream_channels=0 \
#     algorithm.tensor_invariants=0 \
#     algorithm.vector_channels=16 \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/train/hls4ml/vn_hgq/nconst16/baseline_nogate_h32_mlp1_nostream_notensor_beta3e-07

# ------------------------------------------------------------------------
# vn nconst=16 baseline, one block, latency ablation nogate_h32_nostream:
# no gate, head [32], no stream, the two-layer scalar MLP kept
# hand-written: the float reference with the switches of the variant
# ------------------------------------------------------------------------
# python3 src/train.py \
#     experiment=hls4ml/vn_16 \
#     data.method=augmentation \
#     data.augmentation_scheme=none \
#     fold=null \
#     algorithm.dropout=0.0 \
#     algorithm.gate=false \
#     algorithm.head_layers='[32]' \
#     algorithm.invariants=8 \
#     algorithm.num_blocks=1 \
#     algorithm.optimizer.lr=0.004986660327956767 \
#     algorithm.optimizer.weight_decay=0.05 \
#     algorithm.scalar_channels=32 \
#     algorithm.stream_channels=0 \
#     algorithm.tensor_invariants=4 \
#     algorithm.vector_channels=16 \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/train/hls4ml/vn/nconst16/baseline_1block_nogate_h32_nostream

# ------------------------------------------------------------------------
# vn nconst=16 baseline, one block, latency ablation nogate_h32_nostream_wide:
# no gate, head [32], no stream, at the widths of the vn study's best
# one-block trial (8 vector channels, 64 scalar channels, 32 forms, 2 tensors)
# hand-written: the float reference with the switches of the variant
# ------------------------------------------------------------------------
# python3 src/train.py \
#     experiment=hls4ml/vn_16 \
#     data.method=augmentation \
#     data.augmentation_scheme=none \
#     fold=null \
#     algorithm.dropout=0.0 \
#     algorithm.gate=false \
#     algorithm.head_layers='[32]' \
#     algorithm.invariants=32 \
#     algorithm.num_blocks=1 \
#     algorithm.optimizer.lr=0.004986660327956767 \
#     algorithm.optimizer.weight_decay=0.05 \
#     algorithm.scalar_channels=64 \
#     algorithm.stream_channels=0 \
#     algorithm.tensor_invariants=2 \
#     algorithm.vector_channels=8 \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/train/hls4ml/vn/nconst16/baseline_1block_nogate_h32_nostream_wide

# ------------------------------------------------------------------------
# vn_hgq  nconst=16  baseline  (method=augmentation, group=none)  beta=3e-07
# latency ablation nogate_h32_mlp1_nostream_foldtrn:
# no gate, head [32], one-layer scalar MLP, no stream, folded
# normalisations, truncating data lanes
# hand-written, no study
# ------------------------------------------------------------------------
# python3 src/train.py \
#     experiment=hls4ml/vn_hgq_16 \
#     data.method=augmentation \
#     data.augmentation_scheme=none \
#     fold=null \
#     algorithm.beta=3e-07 \
#     algorithm.dropout=0.0 \
#     algorithm.gate=false \
#     algorithm.head_layers='[32]' \
#     algorithm.init_from=logs/train/hls4ml/vn/nconst16/baseline_1block_nogate_h32_mlp1_nostream \
#     algorithm.invariants=8 \
#     algorithm.mlp_layers=1 \
#     algorithm.num_blocks=1 \
#     algorithm.quantisation.fold_norm=true \
#     algorithm.quantisation.round_mode=TRN \
#     algorithm.scalar_channels=32 \
#     algorithm.stream_channels=0 \
#     algorithm.tensor_invariants=4 \
#     algorithm.vector_channels=16 \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/train/hls4ml/vn_hgq/nconst16/baseline_nogate_h32_mlp1_nostream_foldtrn_beta3e-07

# ------------------------------------------------------------------------
# vn_hgq  nconst=16  baseline  (method=augmentation, group=none)  beta=3e-07
# latency ablation nogate_h32_nostream_foldtrn:
# no gate, head [32], no stream, folded normalisations, truncating data lanes
# hand-written, no study
# ------------------------------------------------------------------------
# python3 src/train.py \
#     experiment=hls4ml/vn_hgq_16 \
#     data.method=augmentation \
#     data.augmentation_scheme=none \
#     fold=null \
#     algorithm.beta=3e-07 \
#     algorithm.dropout=0.0 \
#     algorithm.gate=false \
#     algorithm.head_layers='[32]' \
#     algorithm.init_from=logs/train/hls4ml/vn/nconst16/baseline_1block_nogate_h32_nostream \
#     algorithm.invariants=8 \
#     algorithm.num_blocks=1 \
#     algorithm.quantisation.fold_norm=true \
#     algorithm.quantisation.round_mode=TRN \
#     algorithm.scalar_channels=32 \
#     algorithm.stream_channels=0 \
#     algorithm.tensor_invariants=4 \
#     algorithm.vector_channels=16 \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/train/hls4ml/vn_hgq/nconst16/baseline_nogate_h32_nostream_foldtrn_beta3e-07

# ------------------------------------------------------------------------
# vn_hgq  nconst=16  baseline  (method=augmentation, group=none)  beta=3e-07
# latency ablation nogate_h32_nostream_wide_foldtrn:
# no gate, head [32], no stream, the one-block search's widths, folded
# normalisations, truncating data lanes
# hand-written, no study
# ------------------------------------------------------------------------
# python3 src/train.py \
#     experiment=hls4ml/vn_hgq_16 \
#     data.method=augmentation \
#     data.augmentation_scheme=none \
#     fold=null \
#     algorithm.beta=3e-07 \
#     algorithm.dropout=0.0 \
#     algorithm.gate=false \
#     algorithm.head_layers='[32]' \
#     algorithm.init_from=logs/train/hls4ml/vn/nconst16/baseline_1block_nogate_h32_nostream_wide \
#     algorithm.invariants=32 \
#     algorithm.num_blocks=1 \
#     algorithm.quantisation.fold_norm=true \
#     algorithm.quantisation.round_mode=TRN \
#     algorithm.scalar_channels=64 \
#     algorithm.stream_channels=0 \
#     algorithm.tensor_invariants=2 \
#     algorithm.vector_channels=8 \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/train/hls4ml/vn_hgq/nconst16/baseline_nogate_h32_nostream_wide_foldtrn_beta3e-07

# ========================================================================
# vn and vn_hgq  nconst=16  (the full-axis offset, A1 and A2)
# ========================================================================

# ------------------------------------------------------------------------
# vn nconst=16 baseline, one block, full-axis offset axis_a1:
# no gate, no stream, head [64,32], 128 scalar channels, the leader's eta and
# phi from the full jet's axis as two more first-block scalars
# hand-written: trial 134 of vn_cheap_axis_nconst16_augmentation_none
# ------------------------------------------------------------------------
# python3 src/train.py \
#     experiment=hls4ml/vn_16 \
#     data.method=augmentation \
#     data.augmentation_scheme=none \
#     fold=null \
#     algorithm.axis_offset=true \
#     algorithm.dropout=0.0 \
#     algorithm.gate=false \
#     algorithm.head_layers='[64,32]' \
#     algorithm.invariants=8 \
#     algorithm.nbr_feats=5 \
#     algorithm.num_blocks=1 \
#     algorithm.optimizer.lr=0.007221469209238472 \
#     algorithm.optimizer.weight_decay=0.01 \
#     algorithm.scalar_channels=128 \
#     algorithm.stream_channels=0 \
#     algorithm.tensor_invariants=2 \
#     algorithm.vector_channels=16 \
#     data.features='[pt,eta,phi,etarel,phirel]' \
#     data.full_axis=true \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/train/hls4ml/vn/nconst16/baseline_1block_axis_a1

# ------------------------------------------------------------------------
# vn nconst=16 baseline, one block, full-axis offset axis_a2:
# no gate, no stream, head [32], 128 scalar channels, the leader's eta and
# phi from the full jet's axis as two more first-block scalars
# hand-written: the best head-[32] trial of vn_cheap_axis_nconst16_augmentation_none
# ------------------------------------------------------------------------
# python3 src/train.py \
#     experiment=hls4ml/vn_16 \
#     data.method=augmentation \
#     data.augmentation_scheme=none \
#     fold=null \
#     algorithm.axis_offset=true \
#     algorithm.dropout=0.0 \
#     algorithm.gate=false \
#     algorithm.head_layers='[32]' \
#     algorithm.invariants=32 \
#     algorithm.nbr_feats=5 \
#     algorithm.num_blocks=1 \
#     algorithm.optimizer.lr=0.008220876046413613 \
#     algorithm.optimizer.weight_decay=0.0 \
#     algorithm.scalar_channels=128 \
#     algorithm.stream_channels=0 \
#     algorithm.tensor_invariants=1 \
#     algorithm.vector_channels=16 \
#     data.features='[pt,eta,phi,etarel,phirel]' \
#     data.full_axis=true \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/train/hls4ml/vn/nconst16/baseline_1block_axis_a2

# ------------------------------------------------------------------------
# vn_hgq  nconst=16  baseline  (method=augmentation, group=none)  beta=3e-07
# full-axis offset axis_a1_foldtrn:
# the design of block 21, folded normalisations, truncating data lanes
# hand-written, no study
# ------------------------------------------------------------------------
# python3 src/train.py \
#     experiment=hls4ml/vn_hgq_16 \
#     data.method=augmentation \
#     data.augmentation_scheme=none \
#     fold=null \
#     algorithm.axis_offset=true \
#     algorithm.beta=3e-07 \
#     algorithm.dropout=0.0 \
#     algorithm.gate=false \
#     algorithm.head_layers='[64,32]' \
#     algorithm.init_from=logs/train/hls4ml/vn/nconst16/baseline_1block_axis_a1 \
#     algorithm.invariants=8 \
#     algorithm.nbr_feats=5 \
#     algorithm.num_blocks=1 \
#     algorithm.quantisation.fold_norm=true \
#     algorithm.quantisation.round_mode=TRN \
#     algorithm.scalar_channels=128 \
#     algorithm.stream_channels=0 \
#     algorithm.tensor_invariants=2 \
#     algorithm.vector_channels=16 \
#     data.features='[pt,eta,phi,etarel,phirel]' \
#     data.full_axis=true \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/train/hls4ml/vn_hgq/nconst16/baseline_axis_a1_foldtrn_beta3e-07

# ------------------------------------------------------------------------
# vn_hgq  nconst=16  baseline  (method=augmentation, group=none)  beta=3e-07
# full-axis offset axis_a2_foldtrn:
# the design of block 22, folded normalisations, truncating data lanes
# hand-written, no study
# ------------------------------------------------------------------------
# python3 src/train.py \
#     experiment=hls4ml/vn_hgq_16 \
#     data.method=augmentation \
#     data.augmentation_scheme=none \
#     fold=null \
#     algorithm.axis_offset=true \
#     algorithm.beta=3e-07 \
#     algorithm.dropout=0.0 \
#     algorithm.gate=false \
#     algorithm.head_layers='[32]' \
#     algorithm.init_from=logs/train/hls4ml/vn/nconst16/baseline_1block_axis_a2 \
#     algorithm.invariants=32 \
#     algorithm.nbr_feats=5 \
#     algorithm.num_blocks=1 \
#     algorithm.quantisation.fold_norm=true \
#     algorithm.quantisation.round_mode=TRN \
#     algorithm.scalar_channels=128 \
#     algorithm.stream_channels=0 \
#     algorithm.tensor_invariants=1 \
#     algorithm.vector_channels=16 \
#     data.features='[pt,eta,phi,etarel,phirel]' \
#     data.full_axis=true \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/train/hls4ml/vn_hgq/nconst16/baseline_axis_a2_foldtrn_beta3e-07

# ========================================================================
# vn_hgq  nconst=16  (the beta scan and the learning-rate check of D2 and A2)
# ========================================================================

# ------------------------------------------------------------------------
# vn_hgq  nconst=16  baseline  (method=augmentation, group=none)  beta=1e-07
# D2 (block 19) at beta=1e-07: no gate, head [32], no stream, folded normalisations, truncating data lanes
# hand-written, no study
# ------------------------------------------------------------------------
# python3 src/train.py \
#     experiment=hls4ml/vn_hgq_16 \
#     data.method=augmentation \
#     data.augmentation_scheme=none \
#     fold=null \
#     algorithm.beta=1e-07 \
#     algorithm.dropout=0.0 \
#     algorithm.gate=false \
#     algorithm.head_layers='[32]' \
#     algorithm.init_from=logs/train/hls4ml/vn/nconst16/baseline_1block_nogate_h32_nostream \
#     algorithm.invariants=8 \
#     algorithm.num_blocks=1 \
#     algorithm.quantisation.fold_norm=true \
#     algorithm.quantisation.round_mode=TRN \
#     algorithm.scalar_channels=32 \
#     algorithm.stream_channels=0 \
#     algorithm.tensor_invariants=4 \
#     algorithm.vector_channels=16 \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/train/hls4ml/vn_hgq/nconst16/baseline_nogate_h32_nostream_foldtrn_beta1e-07

# ------------------------------------------------------------------------
# vn_hgq  nconst=16  baseline  (method=augmentation, group=none)  beta=3e-08
# D2 (block 19) at beta=3e-08
# hand-written, no study
# ------------------------------------------------------------------------
# python3 src/train.py \
#     experiment=hls4ml/vn_hgq_16 \
#     data.method=augmentation \
#     data.augmentation_scheme=none \
#     fold=null \
#     algorithm.beta=3e-08 \
#     algorithm.dropout=0.0 \
#     algorithm.gate=false \
#     algorithm.head_layers='[32]' \
#     algorithm.init_from=logs/train/hls4ml/vn/nconst16/baseline_1block_nogate_h32_nostream \
#     algorithm.invariants=8 \
#     algorithm.num_blocks=1 \
#     algorithm.quantisation.fold_norm=true \
#     algorithm.quantisation.round_mode=TRN \
#     algorithm.scalar_channels=32 \
#     algorithm.stream_channels=0 \
#     algorithm.tensor_invariants=4 \
#     algorithm.vector_channels=16 \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/train/hls4ml/vn_hgq/nconst16/baseline_nogate_h32_nostream_foldtrn_beta3e-08

# ------------------------------------------------------------------------
# vn_hgq  nconst=16  baseline  (method=augmentation, group=none)  beta=1e-07
# A2 (block 24) at beta=1e-07: the full-axis offset design, folded
# normalisations, truncating data lanes
# hand-written, no study
# ------------------------------------------------------------------------
# python3 src/train.py \
#     experiment=hls4ml/vn_hgq_16 \
#     data.method=augmentation \
#     data.augmentation_scheme=none \
#     fold=null \
#     algorithm.axis_offset=true \
#     algorithm.beta=1e-07 \
#     algorithm.dropout=0.0 \
#     algorithm.gate=false \
#     algorithm.head_layers='[32]' \
#     algorithm.init_from=logs/train/hls4ml/vn/nconst16/baseline_1block_axis_a2 \
#     algorithm.invariants=32 \
#     algorithm.nbr_feats=5 \
#     algorithm.num_blocks=1 \
#     algorithm.quantisation.fold_norm=true \
#     algorithm.quantisation.round_mode=TRN \
#     algorithm.scalar_channels=128 \
#     algorithm.stream_channels=0 \
#     algorithm.tensor_invariants=1 \
#     algorithm.vector_channels=16 \
#     data.features='[pt,eta,phi,etarel,phirel]' \
#     data.full_axis=true \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/train/hls4ml/vn_hgq/nconst16/baseline_axis_a2_foldtrn_beta1e-07

# ------------------------------------------------------------------------
# vn_hgq  nconst=16  baseline  (method=augmentation, group=none)  beta=3e-07
# D2 (block 19) at beta=3e-07, the fine-tune starting at a third of the learning rate
# hand-written, no study
# ------------------------------------------------------------------------
# python3 src/train.py \
#     experiment=hls4ml/vn_hgq_16 \
#     data.method=augmentation \
#     data.augmentation_scheme=none \
#     fold=null \
#     algorithm.beta=3e-07 \
#     algorithm.dropout=0.0 \
#     algorithm.gate=false \
#     algorithm.head_layers='[32]' \
#     algorithm.init_from=logs/train/hls4ml/vn/nconst16/baseline_1block_nogate_h32_nostream \
#     algorithm.invariants=8 \
#     algorithm.num_blocks=1 \
#     algorithm.quantisation.fold_norm=true \
#     algorithm.quantisation.round_mode=TRN \
#     algorithm.scalar_channels=32 \
#     algorithm.scheduler.initial_learning_rate=3.0e-4 \
#     algorithm.stream_channels=0 \
#     algorithm.tensor_invariants=4 \
#     algorithm.vector_channels=16 \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/train/hls4ml/vn_hgq/nconst16/baseline_nogate_h32_nostream_foldtrn_lr3e-04_beta3e-07
