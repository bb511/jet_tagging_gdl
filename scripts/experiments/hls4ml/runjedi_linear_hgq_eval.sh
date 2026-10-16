#!/usr/bin/env bash
# ========================================================================
# JEDI_LINEAR_HGQ EVALUATION COMMANDS
# ========================================================================
# Sixteen eval runs of the trainings of runjedi_linear_hgq_train.sh, the variants of
# JEDI-linear as published (see that file). At 16 constituents blocks 1 to 4
# evaluate each variant's design nearest the EBOPs of Table II of
# arXiv:2508.15468 (57 237), blocks 5 and 6 the designs of the reproduction and
# of the kept-axis trigger-word variant nearest the EBOPs of the vector-neuron
# designs they are compared with (blocks 42 and 40 of runvn_hgq_train.sh).
# Blocks 7 to 10 evaluate the two variants of the comparison at 50 and 150
# constituents, each design nearest the EBOPs of the published design at the
# nearest count of arXiv:2508.15468, read off its repository's result reports:
# N = 64 (102 664) for 50, N = 128 (144 427) for 150.
# Blocks 11 to 16 evaluate the two variants of the comparison at 16, 50 and 150
# constituents under the rule of the vector neurons, each the most accurate
# design whose latency at a 5 ns clock is estimated at most 100 ns, so that the
# two architectures are also compared within one budget.
# Hand-written: jedi_linear_hgq has no study, so
# scripts/experiments/make_train_catalogues.py neither writes nor regenerates
# this file.
# Each block evaluates the picks of its training, the main run's and each
# fold's, through the manifests that scripts/experiments/quantised_picks.py
# writes to <run dir>/picks/<tag>: target57237 for blocks 1 to 4, and
# matched_<design> for blocks 5 and 6, picked fold by fold at the EBOPs of the
# same fold of that vector-neuron design, target102664 and target144427 for
# blocks 7 to 10, and best100ns, which quantised_picks.py pick-hw writes, for
# blocks 11 to 16.
# Each block repeats its training block's overrides and runs Keras in float64
# on the CPU (algorithm.jax_platform), where it reproduces the fixed-point
# arithmetic exactly; on a GPU, XLA's float64 2**f is off by an ulp for some f.
# It writes outputs/main_hgq/ beside the probe's outputs/main.npz, the input of
# scripts/synthesis/synthesise_vn.py, and its row, jedi_linear_hgq_<variant>
# (algorithm.name), to results/quantised/.
# The probe is off for the reproduction (blocks 1, 5, 7, 9, 11, 13 and 15): a
# transform rebases etarel and phirel on the kept constituents, so on the full
# jet's axis its invariance figures would mean nothing. It is on for the others.
# Blocks 1 to 6, 11 and 12 share the label jedi_linear_hgq_nconst16_baseline,
# 7, 8, 13 and 14 that of nconst50, 9, 10, 15 and 16 that of nconst150: select
# them with --only <index>.
#
# Every command is commented out: copy one to run it by hand, or feed the whole
# file to scripts/experiments/submit_eval.sh, which queues them on a GPU or slurm.

# ========================================================================
# jedi_linear_hgq  nconst=16  (JEDI-linear as published)
# ========================================================================

# ------------------------------------------------------------------------
# jedi_linear_hgq  nconst=16  baseline  (method=augmentation, group=none)  axis_trained
# the reproduction: etarel and phirel from the full jet's axis, standardised,
# the words trained by HGQ2, as published; the design nearest Table II's EBOPs
# hand-written, no study
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=hls4ml/jedi_linear_hgq_16 \
#     data.method=augmentation \
#     data.augmentation_scheme=none \
#     algorithm.inputs=standardised \
#     algorithm.jax_platform=cpu \
#     algorithm.name=jedi_linear_hgq_axis_trained \
#     callbacks.invariance_metric.probe=false \
#     checkpoint_manifest_path=logs/train/hls4ml/jedi_linear_hgq/nconst16/baseline_axis_trained_allepochs/picks/target57237 \
#     data.features='[pt,etarel,phirel]' \
#     data.full_axis=true \
#     results_table.invariance_metric=tv \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/hls4ml/jedi_linear_hgq/nconst16/baseline_axis_trained_allepochs_target57237

# ------------------------------------------------------------------------
# jedi_linear_hgq  nconst=16  baseline  (method=augmentation, group=none)  kept_words
# etarel and phirel from the kept constituents' axis, the trigger's fixed
# words: the like-for-like row against the vector-neuron designs; the design
# nearest Table II's EBOPs
# hand-written, no study
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=hls4ml/jedi_linear_hgq_16 \
#     data.method=augmentation \
#     data.augmentation_scheme=none \
#     algorithm.inputs=words \
#     algorithm.jax_platform=cpu \
#     algorithm.name=jedi_linear_hgq_kept_words \
#     checkpoint_manifest_path=logs/train/hls4ml/jedi_linear_hgq/nconst16/baseline_kept_words_allepochs/picks/target57237 \
#     data.features='[pt,etarel,phirel]' \
#     results_table.invariance_metric=tv \
#     trainer=gpu \
#     trainer.devices=[1] \
#     hydra.run.dir=logs/eval/hls4ml/jedi_linear_hgq/nconst16/baseline_kept_words_allepochs_target57237

# ------------------------------------------------------------------------
# jedi_linear_hgq  nconst=16  baseline  (method=augmentation, group=none)  abs_words
# no axis: the absolute [pt, eta, phi] as the trigger's fixed words, the
# angle word (1, 2, 8); the design nearest Table II's EBOPs
# hand-written, no study
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=hls4ml/jedi_linear_hgq_16 \
#     data.method=augmentation \
#     data.augmentation_scheme=none \
#     algorithm.inputs=words \
#     algorithm.jax_platform=cpu \
#     algorithm.name=jedi_linear_hgq_abs_words \
#     algorithm.quantisation.angle_kif='[1,2,8]' \
#     checkpoint_manifest_path=logs/train/hls4ml/jedi_linear_hgq/nconst16/baseline_abs_words_allepochs/picks/target57237 \
#     data.features='[pt,eta,phi]' \
#     results_table.invariance_metric=tv \
#     trainer=gpu \
#     trainer.devices=[2] \
#     hydra.run.dir=logs/eval/hls4ml/jedi_linear_hgq/nconst16/baseline_abs_words_allepochs_target57237

# ------------------------------------------------------------------------
# jedi_linear_hgq  nconst=16  baseline  (method=augmentation, group=none)  kept_trained
# etarel and phirel from the kept constituents' axis, standardised, the
# words trained by HGQ2; the design nearest Table II's EBOPs
# hand-written, no study
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=hls4ml/jedi_linear_hgq_16 \
#     data.method=augmentation \
#     data.augmentation_scheme=none \
#     algorithm.inputs=standardised \
#     algorithm.jax_platform=cpu \
#     algorithm.name=jedi_linear_hgq_kept_trained \
#     checkpoint_manifest_path=logs/train/hls4ml/jedi_linear_hgq/nconst16/baseline_kept_trained_allepochs/picks/target57237 \
#     data.features='[pt,etarel,phirel]' \
#     results_table.invariance_metric=tv \
#     trainer=gpu \
#     trainer.devices=[3] \
#     hydra.run.dir=logs/eval/hls4ml/jedi_linear_hgq/nconst16/baseline_kept_trained_allepochs_target57237

# ------------------------------------------------------------------------
# jedi_linear_hgq  nconst=16  baseline  (method=augmentation, group=none)  axis_trained
# the reproduction at the EBOPs of A2 with trained words (block 42 of
# runvn_hgq_train.sh), the full-axis pair of the comparison
# hand-written, no study
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=hls4ml/jedi_linear_hgq_16 \
#     data.method=augmentation \
#     data.augmentation_scheme=none \
#     algorithm.inputs=standardised \
#     algorithm.jax_platform=cpu \
#     algorithm.name=jedi_linear_hgq_axis_trained_matched \
#     callbacks.invariance_metric.probe=false \
#     checkpoint_manifest_path=logs/train/hls4ml/jedi_linear_hgq/nconst16/baseline_axis_trained_allepochs/picks/matched_a2_trained \
#     data.features='[pt,etarel,phirel]' \
#     data.full_axis=true \
#     results_table.invariance_metric=tv \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/hls4ml/jedi_linear_hgq/nconst16/baseline_axis_trained_allepochs_matched_a2_trained

# ------------------------------------------------------------------------
# jedi_linear_hgq  nconst=16  baseline  (method=augmentation, group=none)  kept_words
# the kept-axis trigger-word variant at the EBOPs of D2 (block 40 of
# runvn_hgq_train.sh), the trigger-information pair of the comparison
# hand-written, no study
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=hls4ml/jedi_linear_hgq_16 \
#     data.method=augmentation \
#     data.augmentation_scheme=none \
#     algorithm.inputs=words \
#     algorithm.jax_platform=cpu \
#     algorithm.name=jedi_linear_hgq_kept_words_matched \
#     checkpoint_manifest_path=logs/train/hls4ml/jedi_linear_hgq/nconst16/baseline_kept_words_allepochs/picks/matched_d2 \
#     data.features='[pt,etarel,phirel]' \
#     results_table.invariance_metric=tv \
#     trainer=gpu \
#     trainer.devices=[1] \
#     hydra.run.dir=logs/eval/hls4ml/jedi_linear_hgq/nconst16/baseline_kept_words_allepochs_matched_d2

# ========================================================================
# jedi_linear_hgq  nconst=50 and 150  (the two variants of the comparison with the vector neurons)
# ========================================================================

# ------------------------------------------------------------------------
# jedi_linear_hgq  nconst=50  baseline  (method=augmentation, group=none)  axis_trained
# the reproduction at N = 50: the full jet's axis, the words trained by HGQ2;
# the design nearest the EBOPs of the published N = 64 design (102664)
# hand-written, no study
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=hls4ml/jedi_linear_hgq_50 \
#     data.method=augmentation \
#     data.augmentation_scheme=none \
#     algorithm.inputs=standardised \
#     algorithm.jax_platform=cpu \
#     algorithm.name=jedi_linear_hgq_axis_trained \
#     callbacks.invariance_metric.probe=false \
#     checkpoint_manifest_path=logs/train/hls4ml/jedi_linear_hgq/nconst50/baseline_axis_trained_allepochs/picks/target102664 \
#     data.features='[pt,etarel,phirel]' \
#     data.full_axis=true \
#     results_table.invariance_metric=tv \
#     trainer=gpu \
#     trainer.devices=[1] \
#     hydra.run.dir=logs/eval/hls4ml/jedi_linear_hgq/nconst50/baseline_axis_trained_allepochs_target102664

# ------------------------------------------------------------------------
# jedi_linear_hgq  nconst=50  baseline  (method=augmentation, group=none)  kept_words
# the kept axis and the trigger's fixed words at N = 50;
# the design nearest the EBOPs of the published N = 64 design (102664)
# hand-written, no study
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=hls4ml/jedi_linear_hgq_50 \
#     data.method=augmentation \
#     data.augmentation_scheme=none \
#     algorithm.inputs=words \
#     algorithm.jax_platform=cpu \
#     algorithm.name=jedi_linear_hgq_kept_words \
#     checkpoint_manifest_path=logs/train/hls4ml/jedi_linear_hgq/nconst50/baseline_kept_words_allepochs/picks/target102664 \
#     data.features='[pt,etarel,phirel]' \
#     results_table.invariance_metric=tv \
#     trainer=gpu \
#     trainer.devices=[2] \
#     hydra.run.dir=logs/eval/hls4ml/jedi_linear_hgq/nconst50/baseline_kept_words_allepochs_target102664

# ------------------------------------------------------------------------
# jedi_linear_hgq  nconst=150  baseline  (method=augmentation, group=none)  axis_trained
# the reproduction at N = 150: the full jet's axis, the words trained by HGQ2;
# the design nearest the EBOPs of the published N = 128 design (144427)
# hand-written, no study
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=hls4ml/jedi_linear_hgq_150 \
#     data.method=augmentation \
#     data.augmentation_scheme=none \
#     algorithm.inputs=standardised \
#     algorithm.jax_platform=cpu \
#     algorithm.name=jedi_linear_hgq_axis_trained \
#     callbacks.invariance_metric.probe=false \
#     checkpoint_manifest_path=logs/train/hls4ml/jedi_linear_hgq/nconst150/baseline_axis_trained_allepochs/picks/target144427 \
#     data.features='[pt,etarel,phirel]' \
#     data.full_axis=true \
#     results_table.invariance_metric=tv \
#     trainer=gpu \
#     trainer.devices=[1] \
#     hydra.run.dir=logs/eval/hls4ml/jedi_linear_hgq/nconst150/baseline_axis_trained_allepochs_target144427

# ------------------------------------------------------------------------
# jedi_linear_hgq  nconst=150  baseline  (method=augmentation, group=none)  kept_words
# the kept axis and the trigger's fixed words at N = 150;
# the design nearest the EBOPs of the published N = 128 design (144427)
# hand-written, no study
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=hls4ml/jedi_linear_hgq_150 \
#     data.method=augmentation \
#     data.augmentation_scheme=none \
#     algorithm.inputs=words \
#     algorithm.jax_platform=cpu \
#     algorithm.name=jedi_linear_hgq_kept_words \
#     checkpoint_manifest_path=logs/train/hls4ml/jedi_linear_hgq/nconst150/baseline_kept_words_allepochs/picks/target144427 \
#     data.features='[pt,etarel,phirel]' \
#     results_table.invariance_metric=tv \
#     trainer=gpu \
#     trainer.devices=[2] \
#     hydra.run.dir=logs/eval/hls4ml/jedi_linear_hgq/nconst150/baseline_kept_words_allepochs_target144427

# ========================================================================
# jedi_linear_hgq  nconst=16  (the two variants of the comparison under the vector neurons' rule)
# ========================================================================

# ------------------------------------------------------------------------
# jedi_linear_hgq  nconst=16  baseline  (method=augmentation, group=none)  axis_trained
# the reproduction: etarel and phirel from the full jet's axis, standardised,
# the words trained by HGQ2, as published; the most accurate design under
# 100 ns, the rule of the vector neurons
# hand-written, no study
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=hls4ml/jedi_linear_hgq_16 \
#     data.method=augmentation \
#     data.augmentation_scheme=none \
#     algorithm.inputs=standardised \
#     algorithm.jax_platform=cpu \
#     algorithm.name=jedi_linear_hgq_axis_trained \
#     callbacks.invariance_metric.probe=false \
#     checkpoint_manifest_path=logs/train/hls4ml/jedi_linear_hgq/nconst16/baseline_axis_trained_allepochs/picks/best100ns \
#     data.features='[pt,etarel,phirel]' \
#     data.full_axis=true \
#     results_table.invariance_metric=tv \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/hls4ml/jedi_linear_hgq/nconst16/baseline_axis_trained_allepochs_best100ns

# ------------------------------------------------------------------------
# jedi_linear_hgq  nconst=16  baseline  (method=augmentation, group=none)  kept_words
# etarel and phirel from the kept constituents' axis, the trigger's fixed
# words: the like-for-like row against the vector-neuron designs; the design
# most accurate under 100 ns, the rule of the vector neurons
# hand-written, no study
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=hls4ml/jedi_linear_hgq_16 \
#     data.method=augmentation \
#     data.augmentation_scheme=none \
#     algorithm.inputs=words \
#     algorithm.jax_platform=cpu \
#     algorithm.name=jedi_linear_hgq_kept_words \
#     checkpoint_manifest_path=logs/train/hls4ml/jedi_linear_hgq/nconst16/baseline_kept_words_allepochs/picks/best100ns \
#     data.features='[pt,etarel,phirel]' \
#     results_table.invariance_metric=tv \
#     trainer=gpu \
#     trainer.devices=[1] \
#     hydra.run.dir=logs/eval/hls4ml/jedi_linear_hgq/nconst16/baseline_kept_words_allepochs_best100ns

# ========================================================================
# jedi_linear_hgq  nconst=50 and 150  (the two variants of the comparison under the vector neurons' rule)
# ========================================================================

# ------------------------------------------------------------------------
# jedi_linear_hgq  nconst=50  baseline  (method=augmentation, group=none)  axis_trained
# the reproduction at N = 50: the full jet's axis, the words trained by HGQ2;
# the most accurate design under 100 ns, the rule of the vector neurons
# hand-written, no study
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=hls4ml/jedi_linear_hgq_50 \
#     data.method=augmentation \
#     data.augmentation_scheme=none \
#     algorithm.inputs=standardised \
#     algorithm.jax_platform=cpu \
#     algorithm.name=jedi_linear_hgq_axis_trained \
#     callbacks.invariance_metric.probe=false \
#     checkpoint_manifest_path=logs/train/hls4ml/jedi_linear_hgq/nconst50/baseline_axis_trained_allepochs/picks/best100ns \
#     data.features='[pt,etarel,phirel]' \
#     data.full_axis=true \
#     results_table.invariance_metric=tv \
#     trainer=gpu \
#     trainer.devices=[1] \
#     hydra.run.dir=logs/eval/hls4ml/jedi_linear_hgq/nconst50/baseline_axis_trained_allepochs_best100ns

# ------------------------------------------------------------------------
# jedi_linear_hgq  nconst=50  baseline  (method=augmentation, group=none)  kept_words
# the kept axis and the trigger's fixed words at N = 50;
# the most accurate design under 100 ns, the rule of the vector neurons
# hand-written, no study
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=hls4ml/jedi_linear_hgq_50 \
#     data.method=augmentation \
#     data.augmentation_scheme=none \
#     algorithm.inputs=words \
#     algorithm.jax_platform=cpu \
#     algorithm.name=jedi_linear_hgq_kept_words \
#     checkpoint_manifest_path=logs/train/hls4ml/jedi_linear_hgq/nconst50/baseline_kept_words_allepochs/picks/best100ns \
#     data.features='[pt,etarel,phirel]' \
#     results_table.invariance_metric=tv \
#     trainer=gpu \
#     trainer.devices=[2] \
#     hydra.run.dir=logs/eval/hls4ml/jedi_linear_hgq/nconst50/baseline_kept_words_allepochs_best100ns

# ------------------------------------------------------------------------
# jedi_linear_hgq  nconst=150  baseline  (method=augmentation, group=none)  axis_trained
# the reproduction at N = 150: the full jet's axis, the words trained by HGQ2;
# the most accurate design under 100 ns, the rule of the vector neurons
# hand-written, no study
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=hls4ml/jedi_linear_hgq_150 \
#     data.method=augmentation \
#     data.augmentation_scheme=none \
#     algorithm.inputs=standardised \
#     algorithm.jax_platform=cpu \
#     algorithm.name=jedi_linear_hgq_axis_trained \
#     callbacks.invariance_metric.probe=false \
#     checkpoint_manifest_path=logs/train/hls4ml/jedi_linear_hgq/nconst150/baseline_axis_trained_allepochs/picks/best100ns \
#     data.features='[pt,etarel,phirel]' \
#     data.full_axis=true \
#     results_table.invariance_metric=tv \
#     trainer=gpu \
#     trainer.devices=[1] \
#     hydra.run.dir=logs/eval/hls4ml/jedi_linear_hgq/nconst150/baseline_axis_trained_allepochs_best100ns

# ------------------------------------------------------------------------
# jedi_linear_hgq  nconst=150  baseline  (method=augmentation, group=none)  kept_words
# the kept axis and the trigger's fixed words at N = 150;
# the most accurate design under 100 ns, the rule of the vector neurons
# hand-written, no study
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=hls4ml/jedi_linear_hgq_150 \
#     data.method=augmentation \
#     data.augmentation_scheme=none \
#     algorithm.inputs=words \
#     algorithm.jax_platform=cpu \
#     algorithm.name=jedi_linear_hgq_kept_words \
#     checkpoint_manifest_path=logs/train/hls4ml/jedi_linear_hgq/nconst150/baseline_kept_words_allepochs/picks/best100ns \
#     data.features='[pt,etarel,phirel]' \
#     results_table.invariance_metric=tv \
#     trainer=gpu \
#     trainer.devices=[2] \
#     hydra.run.dir=logs/eval/hls4ml/jedi_linear_hgq/nconst150/baseline_kept_words_allepochs_best100ns
