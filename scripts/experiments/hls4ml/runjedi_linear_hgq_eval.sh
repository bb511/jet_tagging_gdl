#!/usr/bin/env bash
# ========================================================================
# JEDI_LINEAR_HGQ EVALUATION COMMANDS
# ========================================================================
# Nineteen eval runs, one per block of runjedi_linear_hgq_train.sh: thirteen at
# 16 constituents, the float control, then the moments design, the control and
# the tuned control each at four betas, and six at 50 and 150 at beta=1e-07
# (the float control, the moments design, the control and the tuned control at
# 50; the moments design and the tuned control, also its control, at 150).
# Hand-written: jedi_linear_hgq has no study and starts from float runs, so
# scripts/experiments/make_train_catalogues.py neither writes nor regenerates this
# file.
# Baseline only: the moments row that this one quantises is filed with the
# embedded rows and runs the baseline alone, and the Keras step computes no
# SCL.
# The first block evaluates the float control, the jedi_linear main run
# logs/train/hls4ml/pt_etarel_phirel/jedi_linear/nconst16/baseline_control
# that algorithm.init_from of blocks 6 to 9 names. Its row,
# jedi_linear_control (algorithm.name), goes to results/quantised/
# (results_table.subdir), beside the quantised rows, rather than to the
# grid's table in results/pt_etarel_phirel/. The float reference of the
# moments design is the finals' evaluation of its main run, outputs/main.npz
# in logs/eval/hls4ml/jedi_linear_moments/nconst16/baseline/ on the NGT shared
# volume, which no block here repeats; that of the tuned control (blocks 10 to
# 13) is the finals' evaluation of the grid's jedi_linear main run on these
# columns, logs/eval/hls4ml/pt_etarel_phirel/jedi_linear/nconst16/baseline/.
# Each quantised block repeats its training block's overrides and runs Keras in
# float64 (algorithm.floatx) on the CPU (algorithm.jax_platform), where it
# reproduces the fixed-point arithmetic exactly; on a GPU, XLA's float64
# 2**f is off by an ulp for some f and about 2 % of the jets round
# differently. It writes outputs/main_hgq/ beside the probe's
# outputs/main.npz, the input of scripts/synthesis/synthesise_vn.py.
# The grid is the vn scan's, 1e-08 ... 3e-07, and here its low end matters:
# at the seeded 14 data and 8 weight bits this network prices 58.8 M EBOPs,
# thirty times the 2.0 M of the one-block vn, so beta=3e-07 would weigh the
# EBOPs 23 times the cross-entropy from the first step, where on the
# two-block vn the same grid weighed them 0.037 to 1.1 times; at 1e-08 the
# ratio is about 0.8.
# The twelve quantised blocks share the label jedi_linear_hgq_nconst16_baseline,
# so select them with --only <index> (2 to 13), not by name; the control is
# block 1; those at 50 (15 to 17) and 150 (18, 19) share the labels
# jedi_linear_hgq_nconst50_baseline and jedi_linear_hgq_nconst150_baseline.
# Each names its row jedi_linear_hgq_{moments,control,tuned}_beta<beta>
# (algorithm.name), so that the table in results/quantised/ tells them apart
# beside its nconstituents column. The probe stays on: a rotation about the beam leaves pt, etarel and
# phirel unchanged up to the float rounding of the transform, which can move an
# input word by one LSB.
#
# Every command is commented out: copy one to run it by hand, or feed the whole
# file to scripts/experiments/submit_eval.sh, which queues them on a GPU or slurm.

# ========================================================================
# jedi_linear  nconst=16  (the float control)
# ========================================================================

# ------------------------------------------------------------------------
# jedi_linear nconst=16 baseline on [pt, etarel, phirel]: the control, the
# moments design without the moments
# hand-written: the settings of trial 66 of
# jedi_linear_moments_nconst16_augmentation_none
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=hls4ml/jedi_linear_16 \
#     data.method=augmentation \
#     data.augmentation_scheme=none \
#     algorithm.aggregation=max \
#     algorithm.head_layers='[64,32,16]' \
#     algorithm.hidden=128 \
#     algorithm.name=jedi_linear_control \
#     algorithm.optimizer.lr=0.0033563666242975 \
#     algorithm.optimizer.weight_decay=1e-05 \
#     checkpoint_manifest_path=logs/train/hls4ml/pt_etarel_phirel/jedi_linear/nconst16/baseline_control \
#     data.features='[pt,etarel,phirel]' \
#     results_table.invariance_metric=tv \
#     results_table.subdir=quantised \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/hls4ml/pt_etarel_phirel/jedi_linear/nconst16/baseline_control

# ========================================================================
# jedi_linear_hgq  nconst=16  (the moments design)
# ========================================================================

# ------------------------------------------------------------------------
# jedi_linear_hgq  nconst=16  baseline  (method=augmentation, group=none)  beta=1e-08
# the moments design, fine-tuned from the finals' jedi_linear_moments main run
# hand-written, no study
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=hls4ml/jedi_linear_hgq_16 \
#     data.method=augmentation \
#     data.augmentation_scheme=none \
#     algorithm.aggregation=max \
#     algorithm.beta=1e-08 \
#     algorithm.floatx=float64 \
#     algorithm.head_layers='[64,32,16]' \
#     algorithm.hidden=128 \
#     algorithm.init_from=logs/train/hls4ml/jedi_linear_moments/nconst16/baseline/checkpoints/epoch_183.ckpt \
#     algorithm.jax_platform=cpu \
#     algorithm.moments=uniform \
#     algorithm.name=jedi_linear_hgq_moments_beta1e-08 \
#     checkpoint_manifest_path=logs/train/hls4ml/jedi_linear_hgq/nconst16/baseline_moments_beta1e-08 \
#     results_table.invariance_metric=tv \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/hls4ml/jedi_linear_hgq/nconst16/baseline_moments_beta1e-08

# ------------------------------------------------------------------------
# jedi_linear_hgq  nconst=16  baseline  (method=augmentation, group=none)  beta=3e-08
# the moments design, fine-tuned from the finals' jedi_linear_moments main run
# hand-written, no study
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=hls4ml/jedi_linear_hgq_16 \
#     data.method=augmentation \
#     data.augmentation_scheme=none \
#     algorithm.aggregation=max \
#     algorithm.beta=3e-08 \
#     algorithm.floatx=float64 \
#     algorithm.head_layers='[64,32,16]' \
#     algorithm.hidden=128 \
#     algorithm.init_from=logs/train/hls4ml/jedi_linear_moments/nconst16/baseline/checkpoints/epoch_183.ckpt \
#     algorithm.jax_platform=cpu \
#     algorithm.moments=uniform \
#     algorithm.name=jedi_linear_hgq_moments_beta3e-08 \
#     checkpoint_manifest_path=logs/train/hls4ml/jedi_linear_hgq/nconst16/baseline_moments_beta3e-08 \
#     results_table.invariance_metric=tv \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/hls4ml/jedi_linear_hgq/nconst16/baseline_moments_beta3e-08

# ------------------------------------------------------------------------
# jedi_linear_hgq  nconst=16  baseline  (method=augmentation, group=none)  beta=1e-07
# the moments design, fine-tuned from the finals' jedi_linear_moments main run
# hand-written, no study
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=hls4ml/jedi_linear_hgq_16 \
#     data.method=augmentation \
#     data.augmentation_scheme=none \
#     algorithm.aggregation=max \
#     algorithm.beta=1e-07 \
#     algorithm.floatx=float64 \
#     algorithm.head_layers='[64,32,16]' \
#     algorithm.hidden=128 \
#     algorithm.init_from=logs/train/hls4ml/jedi_linear_moments/nconst16/baseline/checkpoints/epoch_183.ckpt \
#     algorithm.jax_platform=cpu \
#     algorithm.moments=uniform \
#     algorithm.name=jedi_linear_hgq_moments_beta1e-07 \
#     checkpoint_manifest_path=logs/train/hls4ml/jedi_linear_hgq/nconst16/baseline_moments_beta1e-07 \
#     results_table.invariance_metric=tv \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/hls4ml/jedi_linear_hgq/nconst16/baseline_moments_beta1e-07

# ------------------------------------------------------------------------
# jedi_linear_hgq  nconst=16  baseline  (method=augmentation, group=none)  beta=3e-07
# the moments design, fine-tuned from the finals' jedi_linear_moments main run
# hand-written, no study
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=hls4ml/jedi_linear_hgq_16 \
#     data.method=augmentation \
#     data.augmentation_scheme=none \
#     algorithm.aggregation=max \
#     algorithm.beta=3e-07 \
#     algorithm.floatx=float64 \
#     algorithm.head_layers='[64,32,16]' \
#     algorithm.hidden=128 \
#     algorithm.init_from=logs/train/hls4ml/jedi_linear_moments/nconst16/baseline/checkpoints/epoch_183.ckpt \
#     algorithm.jax_platform=cpu \
#     algorithm.moments=uniform \
#     algorithm.name=jedi_linear_hgq_moments_beta3e-07 \
#     checkpoint_manifest_path=logs/train/hls4ml/jedi_linear_hgq/nconst16/baseline_moments_beta3e-07 \
#     results_table.invariance_metric=tv \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/hls4ml/jedi_linear_hgq/nconst16/baseline_moments_beta3e-07

# ========================================================================
# jedi_linear_hgq  nconst=16  (the control)
# ========================================================================

# ------------------------------------------------------------------------
# jedi_linear_hgq  nconst=16  baseline  (method=augmentation, group=none)  beta=1e-08
# the control, fine-tuned from the float run of block 1
# hand-written, no study
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=hls4ml/jedi_linear_hgq_16 \
#     data.method=augmentation \
#     data.augmentation_scheme=none \
#     algorithm.aggregation=max \
#     algorithm.beta=1e-08 \
#     algorithm.floatx=float64 \
#     algorithm.head_layers='[64,32,16]' \
#     algorithm.hidden=128 \
#     algorithm.init_from=logs/train/hls4ml/pt_etarel_phirel/jedi_linear/nconst16/baseline_control \
#     algorithm.jax_platform=cpu \
#     algorithm.moments=none \
#     algorithm.name=jedi_linear_hgq_control_beta1e-08 \
#     checkpoint_manifest_path=logs/train/hls4ml/jedi_linear_hgq/nconst16/baseline_control_beta1e-08 \
#     results_table.invariance_metric=tv \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/hls4ml/jedi_linear_hgq/nconst16/baseline_control_beta1e-08

# ------------------------------------------------------------------------
# jedi_linear_hgq  nconst=16  baseline  (method=augmentation, group=none)  beta=3e-08
# the control, fine-tuned from the float run of block 1
# hand-written, no study
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=hls4ml/jedi_linear_hgq_16 \
#     data.method=augmentation \
#     data.augmentation_scheme=none \
#     algorithm.aggregation=max \
#     algorithm.beta=3e-08 \
#     algorithm.floatx=float64 \
#     algorithm.head_layers='[64,32,16]' \
#     algorithm.hidden=128 \
#     algorithm.init_from=logs/train/hls4ml/pt_etarel_phirel/jedi_linear/nconst16/baseline_control \
#     algorithm.jax_platform=cpu \
#     algorithm.moments=none \
#     algorithm.name=jedi_linear_hgq_control_beta3e-08 \
#     checkpoint_manifest_path=logs/train/hls4ml/jedi_linear_hgq/nconst16/baseline_control_beta3e-08 \
#     results_table.invariance_metric=tv \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/hls4ml/jedi_linear_hgq/nconst16/baseline_control_beta3e-08

# ------------------------------------------------------------------------
# jedi_linear_hgq  nconst=16  baseline  (method=augmentation, group=none)  beta=1e-07
# the control, fine-tuned from the float run of block 1
# hand-written, no study
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=hls4ml/jedi_linear_hgq_16 \
#     data.method=augmentation \
#     data.augmentation_scheme=none \
#     algorithm.aggregation=max \
#     algorithm.beta=1e-07 \
#     algorithm.floatx=float64 \
#     algorithm.head_layers='[64,32,16]' \
#     algorithm.hidden=128 \
#     algorithm.init_from=logs/train/hls4ml/pt_etarel_phirel/jedi_linear/nconst16/baseline_control \
#     algorithm.jax_platform=cpu \
#     algorithm.moments=none \
#     algorithm.name=jedi_linear_hgq_control_beta1e-07 \
#     checkpoint_manifest_path=logs/train/hls4ml/jedi_linear_hgq/nconst16/baseline_control_beta1e-07 \
#     results_table.invariance_metric=tv \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/hls4ml/jedi_linear_hgq/nconst16/baseline_control_beta1e-07

# ------------------------------------------------------------------------
# jedi_linear_hgq  nconst=16  baseline  (method=augmentation, group=none)  beta=3e-07
# the control, fine-tuned from the float run of block 1
# hand-written, no study
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=hls4ml/jedi_linear_hgq_16 \
#     data.method=augmentation \
#     data.augmentation_scheme=none \
#     algorithm.aggregation=max \
#     algorithm.beta=3e-07 \
#     algorithm.floatx=float64 \
#     algorithm.head_layers='[64,32,16]' \
#     algorithm.hidden=128 \
#     algorithm.init_from=logs/train/hls4ml/pt_etarel_phirel/jedi_linear/nconst16/baseline_control \
#     algorithm.jax_platform=cpu \
#     algorithm.moments=none \
#     algorithm.name=jedi_linear_hgq_control_beta3e-07 \
#     checkpoint_manifest_path=logs/train/hls4ml/jedi_linear_hgq/nconst16/baseline_control_beta3e-07 \
#     results_table.invariance_metric=tv \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/hls4ml/jedi_linear_hgq/nconst16/baseline_control_beta3e-07

# ========================================================================
# jedi_linear_hgq  nconst=16  (the tuned control)
# ========================================================================

# ------------------------------------------------------------------------
# jedi_linear_hgq  nconst=16  baseline  (method=augmentation, group=none)  beta=1e-08
# the tuned control, fine-tuned from the finals' jedi_linear main run on [pt, etarel, phirel]
# hand-written, no study
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=hls4ml/jedi_linear_hgq_16 \
#     data.method=augmentation \
#     data.augmentation_scheme=none \
#     algorithm.aggregation=sum \
#     algorithm.beta=1e-08 \
#     algorithm.floatx=float64 \
#     algorithm.head_layers='[32,32,16]' \
#     algorithm.hidden=128 \
#     algorithm.init_from=logs/train/hls4ml/pt_etarel_phirel/jedi_linear/nconst16/baseline/checkpoints/epoch_184.ckpt \
#     algorithm.jax_platform=cpu \
#     algorithm.moments=none \
#     algorithm.name=jedi_linear_hgq_tuned_beta1e-08 \
#     checkpoint_manifest_path=logs/train/hls4ml/jedi_linear_hgq/nconst16/baseline_tuned_beta1e-08 \
#     results_table.invariance_metric=tv \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/hls4ml/jedi_linear_hgq/nconst16/baseline_tuned_beta1e-08

# ------------------------------------------------------------------------
# jedi_linear_hgq  nconst=16  baseline  (method=augmentation, group=none)  beta=3e-08
# the tuned control, fine-tuned from the finals' jedi_linear main run on [pt, etarel, phirel]
# hand-written, no study
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=hls4ml/jedi_linear_hgq_16 \
#     data.method=augmentation \
#     data.augmentation_scheme=none \
#     algorithm.aggregation=sum \
#     algorithm.beta=3e-08 \
#     algorithm.floatx=float64 \
#     algorithm.head_layers='[32,32,16]' \
#     algorithm.hidden=128 \
#     algorithm.init_from=logs/train/hls4ml/pt_etarel_phirel/jedi_linear/nconst16/baseline/checkpoints/epoch_184.ckpt \
#     algorithm.jax_platform=cpu \
#     algorithm.moments=none \
#     algorithm.name=jedi_linear_hgq_tuned_beta3e-08 \
#     checkpoint_manifest_path=logs/train/hls4ml/jedi_linear_hgq/nconst16/baseline_tuned_beta3e-08 \
#     results_table.invariance_metric=tv \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/hls4ml/jedi_linear_hgq/nconst16/baseline_tuned_beta3e-08

# ------------------------------------------------------------------------
# jedi_linear_hgq  nconst=16  baseline  (method=augmentation, group=none)  beta=1e-07
# the tuned control, fine-tuned from the finals' jedi_linear main run on [pt, etarel, phirel]
# hand-written, no study
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=hls4ml/jedi_linear_hgq_16 \
#     data.method=augmentation \
#     data.augmentation_scheme=none \
#     algorithm.aggregation=sum \
#     algorithm.beta=1e-07 \
#     algorithm.floatx=float64 \
#     algorithm.head_layers='[32,32,16]' \
#     algorithm.hidden=128 \
#     algorithm.init_from=logs/train/hls4ml/pt_etarel_phirel/jedi_linear/nconst16/baseline/checkpoints/epoch_184.ckpt \
#     algorithm.jax_platform=cpu \
#     algorithm.moments=none \
#     algorithm.name=jedi_linear_hgq_tuned_beta1e-07 \
#     checkpoint_manifest_path=logs/train/hls4ml/jedi_linear_hgq/nconst16/baseline_tuned_beta1e-07 \
#     results_table.invariance_metric=tv \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/hls4ml/jedi_linear_hgq/nconst16/baseline_tuned_beta1e-07

# ------------------------------------------------------------------------
# jedi_linear_hgq  nconst=16  baseline  (method=augmentation, group=none)  beta=3e-07
# the tuned control, fine-tuned from the finals' jedi_linear main run on [pt, etarel, phirel]
# hand-written, no study
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=hls4ml/jedi_linear_hgq_16 \
#     data.method=augmentation \
#     data.augmentation_scheme=none \
#     algorithm.aggregation=sum \
#     algorithm.beta=3e-07 \
#     algorithm.floatx=float64 \
#     algorithm.head_layers='[32,32,16]' \
#     algorithm.hidden=128 \
#     algorithm.init_from=logs/train/hls4ml/pt_etarel_phirel/jedi_linear/nconst16/baseline/checkpoints/epoch_184.ckpt \
#     algorithm.jax_platform=cpu \
#     algorithm.moments=none \
#     algorithm.name=jedi_linear_hgq_tuned_beta3e-07 \
#     checkpoint_manifest_path=logs/train/hls4ml/jedi_linear_hgq/nconst16/baseline_tuned_beta3e-07 \
#     results_table.invariance_metric=tv \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/hls4ml/jedi_linear_hgq/nconst16/baseline_tuned_beta3e-07

# ========================================================================
# jedi_linear and jedi_linear_hgq  nconst=50  (the moments design, the control and the tuned control)
# ========================================================================

# ------------------------------------------------------------------------
# jedi_linear nconst=50 baseline on [pt, etarel, phirel]: the control, the
# moments design without the moments
# hand-written: the settings of the jedi_linear_moments finals at nconst=50
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=hls4ml/jedi_linear_50 \
#     data.method=augmentation \
#     data.augmentation_scheme=none \
#     algorithm.aggregation=max \
#     algorithm.head_layers='[64,32]' \
#     algorithm.hidden=128 \
#     algorithm.name=jedi_linear_control \
#     algorithm.optimizer.lr=0.003660625075400208 \
#     algorithm.optimizer.weight_decay=0.0 \
#     checkpoint_manifest_path=logs/train/hls4ml/pt_etarel_phirel/jedi_linear/nconst50/baseline_control \
#     data.features='[pt,etarel,phirel]' \
#     results_table.invariance_metric=tv \
#     results_table.subdir=quantised \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/hls4ml/pt_etarel_phirel/jedi_linear/nconst50/baseline_control

# ------------------------------------------------------------------------
# jedi_linear_hgq  nconst=50  baseline  (method=augmentation, group=none)  beta=1e-07
# the moments design, fine-tuned from the finals' jedi_linear_moments main run
# hand-written, no study
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=hls4ml/jedi_linear_hgq_50 \
#     data.method=augmentation \
#     data.augmentation_scheme=none \
#     algorithm.aggregation=max \
#     algorithm.beta=1e-07 \
#     algorithm.floatx=float64 \
#     algorithm.head_layers='[64,32]' \
#     algorithm.hidden=128 \
#     algorithm.init_from=logs/train/hls4ml/jedi_linear_moments/nconst50/baseline/checkpoints/epoch_176.ckpt \
#     algorithm.jax_platform=cpu \
#     algorithm.moments=rankcut \
#     algorithm.name=jedi_linear_hgq_moments_beta1e-07 \
#     checkpoint_manifest_path=logs/train/hls4ml/jedi_linear_hgq/nconst50/baseline_moments_beta1e-07 \
#     results_table.invariance_metric=tv \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/hls4ml/jedi_linear_hgq/nconst50/baseline_moments_beta1e-07

# ------------------------------------------------------------------------
# jedi_linear_hgq  nconst=50  baseline  (method=augmentation, group=none)  beta=1e-07
# the control, fine-tuned from the float run of block 14
# hand-written, no study
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=hls4ml/jedi_linear_hgq_50 \
#     data.method=augmentation \
#     data.augmentation_scheme=none \
#     algorithm.aggregation=max \
#     algorithm.beta=1e-07 \
#     algorithm.floatx=float64 \
#     algorithm.head_layers='[64,32]' \
#     algorithm.hidden=128 \
#     algorithm.init_from=logs/train/hls4ml/pt_etarel_phirel/jedi_linear/nconst50/baseline_control \
#     algorithm.jax_platform=cpu \
#     algorithm.moments=none \
#     algorithm.name=jedi_linear_hgq_control_beta1e-07 \
#     checkpoint_manifest_path=logs/train/hls4ml/jedi_linear_hgq/nconst50/baseline_control_beta1e-07 \
#     results_table.invariance_metric=tv \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/hls4ml/jedi_linear_hgq/nconst50/baseline_control_beta1e-07

# ------------------------------------------------------------------------
# jedi_linear_hgq  nconst=50  baseline  (method=augmentation, group=none)  beta=1e-07
# the tuned control, fine-tuned from the finals' jedi_linear main run on [pt, etarel, phirel]
# hand-written, no study
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=hls4ml/jedi_linear_hgq_50 \
#     data.method=augmentation \
#     data.augmentation_scheme=none \
#     algorithm.aggregation=sum_over_n \
#     algorithm.beta=1e-07 \
#     algorithm.floatx=float64 \
#     algorithm.head_layers='[64,32]' \
#     algorithm.hidden=128 \
#     algorithm.init_from=logs/train/hls4ml/pt_etarel_phirel/jedi_linear/nconst50/baseline/checkpoints/epoch_194.ckpt \
#     algorithm.jax_platform=cpu \
#     algorithm.moments=none \
#     algorithm.name=jedi_linear_hgq_tuned_beta1e-07 \
#     checkpoint_manifest_path=logs/train/hls4ml/jedi_linear_hgq/nconst50/baseline_tuned_beta1e-07 \
#     results_table.invariance_metric=tv \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/hls4ml/jedi_linear_hgq/nconst50/baseline_tuned_beta1e-07

# ========================================================================
# jedi_linear_hgq  nconst=150  (the moments design and the tuned control, also its control)
# ========================================================================

# ------------------------------------------------------------------------
# jedi_linear_hgq  nconst=150  baseline  (method=augmentation, group=none)  beta=1e-07
# the moments design, fine-tuned from the finals' jedi_linear_moments main run
# hand-written, no study
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=hls4ml/jedi_linear_hgq_150 \
#     data.method=augmentation \
#     data.augmentation_scheme=none \
#     algorithm.aggregation=sum_over_n \
#     algorithm.beta=1e-07 \
#     algorithm.floatx=float64 \
#     algorithm.head_layers='[64,32]' \
#     algorithm.hidden=128 \
#     algorithm.init_from=logs/train/hls4ml/jedi_linear_moments/nconst150/baseline/checkpoints/epoch_188.ckpt \
#     algorithm.jax_platform=cpu \
#     algorithm.moments=rankcut \
#     algorithm.name=jedi_linear_hgq_moments_beta1e-07 \
#     checkpoint_manifest_path=logs/train/hls4ml/jedi_linear_hgq/nconst150/baseline_moments_beta1e-07 \
#     results_table.invariance_metric=tv \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/hls4ml/jedi_linear_hgq/nconst150/baseline_moments_beta1e-07

# ------------------------------------------------------------------------
# jedi_linear_hgq  nconst=150  baseline  (method=augmentation, group=none)  beta=1e-07
# the tuned control, which at nconst=150 is the moments design's network without the moments
# hand-written, no study
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=hls4ml/jedi_linear_hgq_150 \
#     data.method=augmentation \
#     data.augmentation_scheme=none \
#     algorithm.aggregation=sum_over_n \
#     algorithm.beta=1e-07 \
#     algorithm.floatx=float64 \
#     algorithm.head_layers='[64,32]' \
#     algorithm.hidden=128 \
#     algorithm.init_from=logs/train/hls4ml/pt_etarel_phirel/jedi_linear/nconst150/baseline/checkpoints/epoch_176.ckpt \
#     algorithm.jax_platform=cpu \
#     algorithm.moments=none \
#     algorithm.name=jedi_linear_hgq_tuned_beta1e-07 \
#     checkpoint_manifest_path=logs/train/hls4ml/jedi_linear_hgq/nconst150/baseline_tuned_beta1e-07 \
#     results_table.invariance_metric=tv \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/hls4ml/jedi_linear_hgq/nconst150/baseline_tuned_beta1e-07
