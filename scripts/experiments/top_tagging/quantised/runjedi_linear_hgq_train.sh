#!/usr/bin/env bash
# ========================================================================
# JEDI_LINEAR_HGQ TOP-TAGGING TRAINING COMMANDS
# ========================================================================
# Two train runs of JEDI-linear as published (Table II of arXiv:2508.15468),
# its network, quantisers and recipe ported to HGQ2 and trained from scratch,
# on the leading 100 constituents of the top-tagging reference set (Zenodo
# 2603256): the two variants of the comparison with the vector neurons that
# hls4ml/runjedi_linear_hgq_train.sh trains at 150 (its blocks 7 and 8).
# The cache holds no untruncated axis, so etarel and phirel are measured from
# the kept constituents, the whole jet for all but 0.5 % of the jets, and the
# reproduction's variant axis_trained becomes kept_trained:
# 1. kept_trained: standardised rows, their words trained by HGQ2, as
#    published.
# 2. kept_words: the trigger's fixed p_T and angle words, the like-for-like
#    row against the vector neurons.
# The paper reports no top-tagging design, so the designs are picked under
# 100 ns alone (runjedi_linear_hgq_eval.sh).
# Hand-written: jedi_linear_hgq has no study, so
# scripts/experiments/make_train_catalogues.py neither writes nor regenerates
# this file.
# Baseline only: the Keras step computes no SCL, and setup() refuses it.
# The recipe is the experiment config's (7000 epochs, batches of 2790, Adam at
# 3e-3 under CosineRestarts, beta ramped from 2e-08 to 3e-06 through
# algorithm.beta_schedule); a block carries its variant alone, and a checkpoint
# every epoch, as HGQ2's ParetoFront keeps them in the published code
# (baseline_<variant>_allepochs).
# The two blocks share the label jedi_linear_hgq_nconst100_baseline: select
# them with --only <index>, not by name.
#
# Every command is commented out: copy one to run it by hand, or feed the whole
# file to scripts/experiments/submit_train.sh, which runs them under nohup.

# ========================================================================
# jedi_linear_hgq  nconst=100  (the two variants of the comparison with the vector neurons)
# ========================================================================

# ------------------------------------------------------------------------
# jedi_linear_hgq  nconst=100  baseline  (method=augmentation, group=none)  kept_trained
# etarel and phirel from the kept constituents' axis, standardised, the words
# trained by HGQ2, as published
# hand-written, no study
# ------------------------------------------------------------------------
# python3 src/train.py \
#     experiment=top_tagging/jedi_linear_hgq_100 \
#     data.method=augmentation \
#     data.augmentation_scheme=none \
#     fold=null \
#     algorithm.inputs=standardised \
#     data.features='[pt,etarel,phirel]' \
#     +callbacks.model_checkpoint.every_n_epochs=1 \
#     callbacks.model_checkpoint.save_top_k=-1 \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/train/top_tagging/jedi_linear_hgq/nconst100/baseline_kept_trained_allepochs

# ------------------------------------------------------------------------
# jedi_linear_hgq  nconst=100  baseline  (method=augmentation, group=none)  kept_words
# etarel and phirel from the kept constituents' axis, the trigger's fixed
# words: the like-for-like row against the vector-neuron designs
# hand-written, no study
# ------------------------------------------------------------------------
# python3 src/train.py \
#     experiment=top_tagging/jedi_linear_hgq_100 \
#     data.method=augmentation \
#     data.augmentation_scheme=none \
#     fold=null \
#     algorithm.inputs=words \
#     data.features='[pt,etarel,phirel]' \
#     +callbacks.model_checkpoint.every_n_epochs=1 \
#     callbacks.model_checkpoint.save_top_k=-1 \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/train/top_tagging/jedi_linear_hgq/nconst100/baseline_kept_words_allepochs
