#!/usr/bin/env bash
# ========================================================================
# JEDI_LINEAR_HGQ TOP-TAGGING EVALUATION COMMANDS
# ========================================================================
# Two eval runs of the trainings of runjedi_linear_hgq_train.sh, the two
# variants of JEDI-linear as published at 100 constituents of the top-tagging
# set (see that file). Each evaluates, under the rule of the vector neurons,
# the most accurate design whose latency at a 5 ns clock is estimated at most
# 100 ns: the main run's and each fold's pick, through the manifests that
# scripts/experiments/quantised_picks.py pick-hw writes to
# <run dir>/picks/best100ns. The paper's protocol, the design nearest its
# EBOPs, has no target here, since it reports no top-tagging design.
# Hand-written: jedi_linear_hgq has no study, so
# scripts/experiments/make_train_catalogues.py neither writes nor regenerates
# this file.
# Each block repeats its training block's overrides and runs Keras in float64
# on the CPU (algorithm.jax_platform), where it reproduces the fixed-point
# arithmetic exactly; on a GPU, XLA's float64 2**f is off by an ulp for some f.
# It writes outputs/main_hgq/ beside the probe's outputs/main.npz, the input of
# scripts/synthesis/synthesise_vn.py, and its row, jedi_linear_hgq_<variant>
# (algorithm.name), to results/top_tagging/quantised/. The probe is on for
# both: the kept axis is the one the transforms rebase on.
# The two blocks share the label jedi_linear_hgq_nconst100_baseline: select
# them with --only <index>.
#
# Every command is commented out: copy one to run it by hand, or feed the whole
# file to scripts/experiments/submit_eval.sh, which runs them under nohup.

# ========================================================================
# jedi_linear_hgq  nconst=100  (the two variants of the comparison with the vector neurons)
# ========================================================================

# ------------------------------------------------------------------------
# jedi_linear_hgq  nconst=100  baseline  (method=augmentation, group=none)  kept_trained
# etarel and phirel from the kept constituents' axis, standardised, the words
# trained by HGQ2, as published; the most accurate design under 100 ns
# hand-written, no study
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=top_tagging/jedi_linear_hgq_100 \
#     data.method=augmentation \
#     data.augmentation_scheme=none \
#     algorithm.inputs=standardised \
#     algorithm.jax_platform=cpu \
#     algorithm.name=jedi_linear_hgq_kept_trained \
#     checkpoint_manifest_path=logs/train/top_tagging/jedi_linear_hgq/nconst100/baseline_kept_trained_allepochs/picks/best100ns \
#     data.features='[pt,etarel,phirel]' \
#     results_table.invariance_metric=tv \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/top_tagging/jedi_linear_hgq/nconst100/baseline_kept_trained_allepochs_best100ns

# ------------------------------------------------------------------------
# jedi_linear_hgq  nconst=100  baseline  (method=augmentation, group=none)  kept_words
# etarel and phirel from the kept constituents' axis, the trigger's fixed
# words; the most accurate design under 100 ns
# hand-written, no study
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=top_tagging/jedi_linear_hgq_100 \
#     data.method=augmentation \
#     data.augmentation_scheme=none \
#     algorithm.inputs=words \
#     algorithm.jax_platform=cpu \
#     algorithm.name=jedi_linear_hgq_kept_words \
#     checkpoint_manifest_path=logs/train/top_tagging/jedi_linear_hgq/nconst100/baseline_kept_words_allepochs/picks/best100ns \
#     data.features='[pt,etarel,phirel]' \
#     results_table.invariance_metric=tv \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/top_tagging/jedi_linear_hgq/nconst100/baseline_kept_words_allepochs_best100ns
