#!/usr/bin/env bash
# ========================================================================
# JEDI_LINEAR_HGQ TRAINING COMMANDS
# ========================================================================
# Twenty train runs: thirteen at 16 constituents, the float control, then
# the moments design, the control and the tuned control each fine-tuned at
# four betas, six at 50 and 150 at the beta chosen at 16, 1e-07, and the
# tuned control at 16 in float on the full jet's axis.
# Hand-written: jedi_linear_hgq has no study and starts from float runs, so
# scripts/experiments/make_train_catalogues.py neither writes nor regenerates
# this file.
# Baseline only: the moments row that this one quantises is filed with the
# embedded rows and runs the baseline alone, and the Keras step computes no
# SCL.
# The row quantises jedi_linear_moments, JEDI-linear whose head also reads
# the five shift-invariant jet moments, and measures what the moments cost
# against the same network without them, the control, through one flow.
# The first block trains the control in float: the main run (fold=null) of
# jedi_linear on [pt, etarel, phirel] at the settings of the moments design
# (hidden 128, head [64,32,16], max pooling, lr 0.0033563666242975, weight
# decay 1e-05), in logs/train/hls4ml/pt_etarel_phirel/jedi_linear/nconst16/
# baseline_control. The grid's own jedi_linear at 16 on these columns pools
# by the sum with head [32,32,16], so it is not the control.
# Blocks 2 to 5 each fine-tune the moments design at one constant beta; the
# last epoch is the model. They start from the finals' main checkpoint of
# jedi_linear_moments at 16 (trial 66 of its study, val/acc 0.7229), which
# exists only on the NGT shared volume: copy
# logs/train/hls4ml/jedi_linear_moments/nconst16/baseline/ (the checkpoint
# epoch_183.ckpt, .hydra/config.yaml and the manifest) from
# /shared/deodagiu/jet_tagging_gdl/ to the same path here first.
# algorithm.init_from names the checkpoint itself, because that manifest
# holds absolute /shared paths.
# Blocks 6 to 9 fine-tune the control at the same betas and start once block
# 1 has finished.
# Blocks 10 to 13 fine-tune the tuned control at the same betas: the grid's
# own jedi_linear at 16 on [pt, etarel, phirel], from the finals' main
# checkpoint (sum pooling, head [32,32,16], hidden 128, val/acc 0.7163),
# copied from the NGT shared volume as the moments design's is. They were
# added after the scan of 2026-10-10, in which the control, pooling by the
# maximum without the moments, scored 2.5 points under this network in float:
# the moments' gain over a tuned JEDI-linear is read against these blocks.
# Blocks 14 to 19 repeat the measurement at beta=1e-07, the choice at 16, on
# the finals' architectures, hidden 128 and head [64,32] at both counts: at
# 50, the float control (max pooling, as the moments design) in
# logs/train/hls4ml/pt_etarel_phirel/jedi_linear/nconst50/baseline_control,
# then the moments design (rank-cut moments), the control after block 14 and
# the tuned control (sum_over_n); at 150, the moments design (sum_over_n,
# rank-cut moments) and the tuned control, which there is the moments
# design's network without the moments and so serves as the control too. The
# four finals' checkpoints they start from are copied from the NGT shared
# volume as at 16.
# The grid is the vn scan's, 1e-08 ... 3e-07, and here its low end matters:
# at the seeded 14 data and 8 weight bits this network prices 58.8 M EBOPs,
# thirty times the 2.0 M of the one-block vn, so beta=3e-07 would weigh the
# EBOPs 23 times the cross-entropy from the first step, where on the
# two-block vn the same grid weighed them 0.037 to 1.1 times; at 1e-08 the
# ratio is about 0.8.
# The twelve quantised blocks share the label jedi_linear_hgq_nconst16_baseline,
# so select them with --only <index> (2 to 13), not by name; the control is
# block 1. Those at 50 (15 to 17) and 150 (18, 19) share the labels
# jedi_linear_hgq_nconst50_baseline and jedi_linear_hgq_nconst150_baseline in
# the same way. The row's config folds the moments' batch normalisation into the
# head's first layer and truncates every trainable data lane
# (quantisation.fold_norm and round_mode TRN, the recipe of the folded vn
# designs), so no block overrides either.
# Block 20 trains the tuned control at 16 in float with etarel and phirel
# measured from the full jet's axis (data.full_axis=true), in
# logs/train/hls4ml/pt_etarel_phirel/jedi_linear/nconst16/baseline_axis. With
# block 39 of runvn_hgq_train.sh, it separates the full axis, on which the
# published JEDI-linear's inputs are measured, from its 7000-epoch recipe.
#
# Every command is commented out: copy one to run it by hand, or feed the whole
# file to scripts/experiments/submit_train.sh, which queues them on a GPU or slurm.

# ========================================================================
# jedi_linear  nconst=16  (the float control)
# ========================================================================

# ------------------------------------------------------------------------
# jedi_linear nconst=16 baseline on [pt, etarel, phirel]: the control, the
# moments design without the moments
# hand-written: the settings of trial 66 of
# jedi_linear_moments_nconst16_augmentation_none
# ------------------------------------------------------------------------
# python3 src/train.py \
#     experiment=hls4ml/jedi_linear_16 \
#     data.method=augmentation \
#     data.augmentation_scheme=none \
#     fold=null \
#     algorithm.aggregation=max \
#     algorithm.head_layers='[64,32,16]' \
#     algorithm.hidden=128 \
#     algorithm.optimizer.lr=0.0033563666242975 \
#     algorithm.optimizer.weight_decay=1e-05 \
#     data.features='[pt,etarel,phirel]' \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/train/hls4ml/pt_etarel_phirel/jedi_linear/nconst16/baseline_control

# ========================================================================
# jedi_linear_hgq  nconst=16  (the moments design)
# ========================================================================

# ------------------------------------------------------------------------
# jedi_linear_hgq  nconst=16  baseline  (method=augmentation, group=none)  beta=1e-08
# the moments design, fine-tuned from the finals' jedi_linear_moments main run
# hand-written, no study
# ------------------------------------------------------------------------
# python3 src/train.py \
#     experiment=hls4ml/jedi_linear_hgq_16 \
#     data.method=augmentation \
#     data.augmentation_scheme=none \
#     fold=null \
#     algorithm.aggregation=max \
#     algorithm.beta=1e-08 \
#     algorithm.head_layers='[64,32,16]' \
#     algorithm.hidden=128 \
#     algorithm.init_from=logs/train/hls4ml/jedi_linear_moments/nconst16/baseline/checkpoints/epoch_183.ckpt \
#     algorithm.moments=uniform \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/train/hls4ml/jedi_linear_hgq/nconst16/baseline_moments_beta1e-08

# ------------------------------------------------------------------------
# jedi_linear_hgq  nconst=16  baseline  (method=augmentation, group=none)  beta=3e-08
# the moments design, fine-tuned from the finals' jedi_linear_moments main run
# hand-written, no study
# ------------------------------------------------------------------------
# python3 src/train.py \
#     experiment=hls4ml/jedi_linear_hgq_16 \
#     data.method=augmentation \
#     data.augmentation_scheme=none \
#     fold=null \
#     algorithm.aggregation=max \
#     algorithm.beta=3e-08 \
#     algorithm.head_layers='[64,32,16]' \
#     algorithm.hidden=128 \
#     algorithm.init_from=logs/train/hls4ml/jedi_linear_moments/nconst16/baseline/checkpoints/epoch_183.ckpt \
#     algorithm.moments=uniform \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/train/hls4ml/jedi_linear_hgq/nconst16/baseline_moments_beta3e-08

# ------------------------------------------------------------------------
# jedi_linear_hgq  nconst=16  baseline  (method=augmentation, group=none)  beta=1e-07
# the moments design, fine-tuned from the finals' jedi_linear_moments main run
# hand-written, no study
# ------------------------------------------------------------------------
# python3 src/train.py \
#     experiment=hls4ml/jedi_linear_hgq_16 \
#     data.method=augmentation \
#     data.augmentation_scheme=none \
#     fold=null \
#     algorithm.aggregation=max \
#     algorithm.beta=1e-07 \
#     algorithm.head_layers='[64,32,16]' \
#     algorithm.hidden=128 \
#     algorithm.init_from=logs/train/hls4ml/jedi_linear_moments/nconst16/baseline/checkpoints/epoch_183.ckpt \
#     algorithm.moments=uniform \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/train/hls4ml/jedi_linear_hgq/nconst16/baseline_moments_beta1e-07

# ------------------------------------------------------------------------
# jedi_linear_hgq  nconst=16  baseline  (method=augmentation, group=none)  beta=3e-07
# the moments design, fine-tuned from the finals' jedi_linear_moments main run
# hand-written, no study
# ------------------------------------------------------------------------
# python3 src/train.py \
#     experiment=hls4ml/jedi_linear_hgq_16 \
#     data.method=augmentation \
#     data.augmentation_scheme=none \
#     fold=null \
#     algorithm.aggregation=max \
#     algorithm.beta=3e-07 \
#     algorithm.head_layers='[64,32,16]' \
#     algorithm.hidden=128 \
#     algorithm.init_from=logs/train/hls4ml/jedi_linear_moments/nconst16/baseline/checkpoints/epoch_183.ckpt \
#     algorithm.moments=uniform \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/train/hls4ml/jedi_linear_hgq/nconst16/baseline_moments_beta3e-07

# ========================================================================
# jedi_linear_hgq  nconst=16  (the control)
# ========================================================================

# ------------------------------------------------------------------------
# jedi_linear_hgq  nconst=16  baseline  (method=augmentation, group=none)  beta=1e-08
# the control, fine-tuned from the float run of block 1
# hand-written, no study
# ------------------------------------------------------------------------
# python3 src/train.py \
#     experiment=hls4ml/jedi_linear_hgq_16 \
#     data.method=augmentation \
#     data.augmentation_scheme=none \
#     fold=null \
#     algorithm.aggregation=max \
#     algorithm.beta=1e-08 \
#     algorithm.head_layers='[64,32,16]' \
#     algorithm.hidden=128 \
#     algorithm.init_from=logs/train/hls4ml/pt_etarel_phirel/jedi_linear/nconst16/baseline_control \
#     algorithm.moments=none \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/train/hls4ml/jedi_linear_hgq/nconst16/baseline_control_beta1e-08

# ------------------------------------------------------------------------
# jedi_linear_hgq  nconst=16  baseline  (method=augmentation, group=none)  beta=3e-08
# the control, fine-tuned from the float run of block 1
# hand-written, no study
# ------------------------------------------------------------------------
# python3 src/train.py \
#     experiment=hls4ml/jedi_linear_hgq_16 \
#     data.method=augmentation \
#     data.augmentation_scheme=none \
#     fold=null \
#     algorithm.aggregation=max \
#     algorithm.beta=3e-08 \
#     algorithm.head_layers='[64,32,16]' \
#     algorithm.hidden=128 \
#     algorithm.init_from=logs/train/hls4ml/pt_etarel_phirel/jedi_linear/nconst16/baseline_control \
#     algorithm.moments=none \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/train/hls4ml/jedi_linear_hgq/nconst16/baseline_control_beta3e-08

# ------------------------------------------------------------------------
# jedi_linear_hgq  nconst=16  baseline  (method=augmentation, group=none)  beta=1e-07
# the control, fine-tuned from the float run of block 1
# hand-written, no study
# ------------------------------------------------------------------------
# python3 src/train.py \
#     experiment=hls4ml/jedi_linear_hgq_16 \
#     data.method=augmentation \
#     data.augmentation_scheme=none \
#     fold=null \
#     algorithm.aggregation=max \
#     algorithm.beta=1e-07 \
#     algorithm.head_layers='[64,32,16]' \
#     algorithm.hidden=128 \
#     algorithm.init_from=logs/train/hls4ml/pt_etarel_phirel/jedi_linear/nconst16/baseline_control \
#     algorithm.moments=none \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/train/hls4ml/jedi_linear_hgq/nconst16/baseline_control_beta1e-07

# ------------------------------------------------------------------------
# jedi_linear_hgq  nconst=16  baseline  (method=augmentation, group=none)  beta=3e-07
# the control, fine-tuned from the float run of block 1
# hand-written, no study
# ------------------------------------------------------------------------
# python3 src/train.py \
#     experiment=hls4ml/jedi_linear_hgq_16 \
#     data.method=augmentation \
#     data.augmentation_scheme=none \
#     fold=null \
#     algorithm.aggregation=max \
#     algorithm.beta=3e-07 \
#     algorithm.head_layers='[64,32,16]' \
#     algorithm.hidden=128 \
#     algorithm.init_from=logs/train/hls4ml/pt_etarel_phirel/jedi_linear/nconst16/baseline_control \
#     algorithm.moments=none \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/train/hls4ml/jedi_linear_hgq/nconst16/baseline_control_beta3e-07

# ========================================================================
# jedi_linear_hgq  nconst=16  (the tuned control)
# ========================================================================

# ------------------------------------------------------------------------
# jedi_linear_hgq  nconst=16  baseline  (method=augmentation, group=none)  beta=1e-08
# the tuned control, fine-tuned from the finals' jedi_linear main run on [pt, etarel, phirel]
# hand-written, no study
# ------------------------------------------------------------------------
# python3 src/train.py \
#     experiment=hls4ml/jedi_linear_hgq_16 \
#     data.method=augmentation \
#     data.augmentation_scheme=none \
#     fold=null \
#     algorithm.aggregation=sum \
#     algorithm.beta=1e-08 \
#     algorithm.head_layers='[32,32,16]' \
#     algorithm.hidden=128 \
#     algorithm.init_from=logs/train/hls4ml/pt_etarel_phirel/jedi_linear/nconst16/baseline/checkpoints/epoch_184.ckpt \
#     algorithm.moments=none \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/train/hls4ml/jedi_linear_hgq/nconst16/baseline_tuned_beta1e-08

# ------------------------------------------------------------------------
# jedi_linear_hgq  nconst=16  baseline  (method=augmentation, group=none)  beta=3e-08
# the tuned control, fine-tuned from the finals' jedi_linear main run on [pt, etarel, phirel]
# hand-written, no study
# ------------------------------------------------------------------------
# python3 src/train.py \
#     experiment=hls4ml/jedi_linear_hgq_16 \
#     data.method=augmentation \
#     data.augmentation_scheme=none \
#     fold=null \
#     algorithm.aggregation=sum \
#     algorithm.beta=3e-08 \
#     algorithm.head_layers='[32,32,16]' \
#     algorithm.hidden=128 \
#     algorithm.init_from=logs/train/hls4ml/pt_etarel_phirel/jedi_linear/nconst16/baseline/checkpoints/epoch_184.ckpt \
#     algorithm.moments=none \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/train/hls4ml/jedi_linear_hgq/nconst16/baseline_tuned_beta3e-08

# ------------------------------------------------------------------------
# jedi_linear_hgq  nconst=16  baseline  (method=augmentation, group=none)  beta=1e-07
# the tuned control, fine-tuned from the finals' jedi_linear main run on [pt, etarel, phirel]
# hand-written, no study
# ------------------------------------------------------------------------
# python3 src/train.py \
#     experiment=hls4ml/jedi_linear_hgq_16 \
#     data.method=augmentation \
#     data.augmentation_scheme=none \
#     fold=null \
#     algorithm.aggregation=sum \
#     algorithm.beta=1e-07 \
#     algorithm.head_layers='[32,32,16]' \
#     algorithm.hidden=128 \
#     algorithm.init_from=logs/train/hls4ml/pt_etarel_phirel/jedi_linear/nconst16/baseline/checkpoints/epoch_184.ckpt \
#     algorithm.moments=none \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/train/hls4ml/jedi_linear_hgq/nconst16/baseline_tuned_beta1e-07

# ------------------------------------------------------------------------
# jedi_linear_hgq  nconst=16  baseline  (method=augmentation, group=none)  beta=3e-07
# the tuned control, fine-tuned from the finals' jedi_linear main run on [pt, etarel, phirel]
# hand-written, no study
# ------------------------------------------------------------------------
# python3 src/train.py \
#     experiment=hls4ml/jedi_linear_hgq_16 \
#     data.method=augmentation \
#     data.augmentation_scheme=none \
#     fold=null \
#     algorithm.aggregation=sum \
#     algorithm.beta=3e-07 \
#     algorithm.head_layers='[32,32,16]' \
#     algorithm.hidden=128 \
#     algorithm.init_from=logs/train/hls4ml/pt_etarel_phirel/jedi_linear/nconst16/baseline/checkpoints/epoch_184.ckpt \
#     algorithm.moments=none \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/train/hls4ml/jedi_linear_hgq/nconst16/baseline_tuned_beta3e-07

# ========================================================================
# jedi_linear and jedi_linear_hgq  nconst=50  (the moments design, the control and the tuned control)
# ========================================================================

# ------------------------------------------------------------------------
# jedi_linear nconst=50 baseline on [pt, etarel, phirel]: the control, the
# moments design without the moments
# hand-written: the settings of the jedi_linear_moments finals at nconst=50
# ------------------------------------------------------------------------
# python3 src/train.py \
#     experiment=hls4ml/jedi_linear_50 \
#     data.method=augmentation \
#     data.augmentation_scheme=none \
#     fold=null \
#     algorithm.aggregation=max \
#     algorithm.head_layers='[64,32]' \
#     algorithm.hidden=128 \
#     algorithm.optimizer.lr=0.003660625075400208 \
#     algorithm.optimizer.weight_decay=0.0 \
#     data.features='[pt,etarel,phirel]' \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/train/hls4ml/pt_etarel_phirel/jedi_linear/nconst50/baseline_control

# ------------------------------------------------------------------------
# jedi_linear_hgq  nconst=50  baseline  (method=augmentation, group=none)  beta=1e-07
# the moments design, fine-tuned from the finals' jedi_linear_moments main run
# hand-written, no study
# ------------------------------------------------------------------------
# python3 src/train.py \
#     experiment=hls4ml/jedi_linear_hgq_50 \
#     data.method=augmentation \
#     data.augmentation_scheme=none \
#     fold=null \
#     algorithm.aggregation=max \
#     algorithm.beta=1e-07 \
#     algorithm.head_layers='[64,32]' \
#     algorithm.hidden=128 \
#     algorithm.init_from=logs/train/hls4ml/jedi_linear_moments/nconst50/baseline/checkpoints/epoch_176.ckpt \
#     algorithm.moments=rankcut \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/train/hls4ml/jedi_linear_hgq/nconst50/baseline_moments_beta1e-07

# ------------------------------------------------------------------------
# jedi_linear_hgq  nconst=50  baseline  (method=augmentation, group=none)  beta=1e-07
# the control, fine-tuned from the float run of block 14
# hand-written, no study
# ------------------------------------------------------------------------
# python3 src/train.py \
#     experiment=hls4ml/jedi_linear_hgq_50 \
#     data.method=augmentation \
#     data.augmentation_scheme=none \
#     fold=null \
#     algorithm.aggregation=max \
#     algorithm.beta=1e-07 \
#     algorithm.head_layers='[64,32]' \
#     algorithm.hidden=128 \
#     algorithm.init_from=logs/train/hls4ml/pt_etarel_phirel/jedi_linear/nconst50/baseline_control \
#     algorithm.moments=none \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/train/hls4ml/jedi_linear_hgq/nconst50/baseline_control_beta1e-07

# ------------------------------------------------------------------------
# jedi_linear_hgq  nconst=50  baseline  (method=augmentation, group=none)  beta=1e-07
# the tuned control, fine-tuned from the finals' jedi_linear main run on [pt, etarel, phirel]
# hand-written, no study
# ------------------------------------------------------------------------
# python3 src/train.py \
#     experiment=hls4ml/jedi_linear_hgq_50 \
#     data.method=augmentation \
#     data.augmentation_scheme=none \
#     fold=null \
#     algorithm.aggregation=sum_over_n \
#     algorithm.beta=1e-07 \
#     algorithm.head_layers='[64,32]' \
#     algorithm.hidden=128 \
#     algorithm.init_from=logs/train/hls4ml/pt_etarel_phirel/jedi_linear/nconst50/baseline/checkpoints/epoch_194.ckpt \
#     algorithm.moments=none \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/train/hls4ml/jedi_linear_hgq/nconst50/baseline_tuned_beta1e-07

# ========================================================================
# jedi_linear_hgq  nconst=150  (the moments design and the tuned control, also its control)
# ========================================================================

# ------------------------------------------------------------------------
# jedi_linear_hgq  nconst=150  baseline  (method=augmentation, group=none)  beta=1e-07
# the moments design, fine-tuned from the finals' jedi_linear_moments main run
# hand-written, no study
# ------------------------------------------------------------------------
# python3 src/train.py \
#     experiment=hls4ml/jedi_linear_hgq_150 \
#     data.method=augmentation \
#     data.augmentation_scheme=none \
#     fold=null \
#     algorithm.aggregation=sum_over_n \
#     algorithm.beta=1e-07 \
#     algorithm.head_layers='[64,32]' \
#     algorithm.hidden=128 \
#     algorithm.init_from=logs/train/hls4ml/jedi_linear_moments/nconst150/baseline/checkpoints/epoch_188.ckpt \
#     algorithm.moments=rankcut \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/train/hls4ml/jedi_linear_hgq/nconst150/baseline_moments_beta1e-07

# ------------------------------------------------------------------------
# jedi_linear_hgq  nconst=150  baseline  (method=augmentation, group=none)  beta=1e-07
# the tuned control, which at nconst=150 is the moments design's network without the moments
# hand-written, no study
# ------------------------------------------------------------------------
# python3 src/train.py \
#     experiment=hls4ml/jedi_linear_hgq_150 \
#     data.method=augmentation \
#     data.augmentation_scheme=none \
#     fold=null \
#     algorithm.aggregation=sum_over_n \
#     algorithm.beta=1e-07 \
#     algorithm.head_layers='[64,32]' \
#     algorithm.hidden=128 \
#     algorithm.init_from=logs/train/hls4ml/pt_etarel_phirel/jedi_linear/nconst150/baseline/checkpoints/epoch_176.ckpt \
#     algorithm.moments=none \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/train/hls4ml/jedi_linear_hgq/nconst150/baseline_tuned_beta1e-07

# ========================================================================
# jedi_linear  nconst=16  (the tuned control on the full jet's axis)
# ========================================================================

# ------------------------------------------------------------------------
# jedi_linear nconst=16 baseline on [pt, etarel, phirel] from the full jet's
# axis: the tuned control in float
# hand-written: best trial 107 of jedi_linear_nconst16_augmentation_none on
# [pt, etarel, phirel] with data.full_axis=true
# ------------------------------------------------------------------------
# python3 src/train.py \
#     experiment=hls4ml/jedi_linear_16 \
#     data.method=augmentation \
#     data.augmentation_scheme=none \
#     fold=null \
#     algorithm.aggregation=sum \
#     algorithm.head_layers='[32,32,16]' \
#     algorithm.hidden=128 \
#     algorithm.optimizer.lr=0.003110744733957464 \
#     algorithm.optimizer.weight_decay=1e-06 \
#     data.features='[pt,etarel,phirel]' \
#     data.full_axis=true \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/train/hls4ml/pt_etarel_phirel/jedi_linear/nconst16/baseline_axis
