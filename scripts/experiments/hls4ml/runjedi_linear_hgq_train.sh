#!/usr/bin/env bash
# ========================================================================
# JEDI_LINEAR_HGQ TRAINING COMMANDS
# ========================================================================
# Four train runs at 16 constituents of JEDI-linear as published (Table II of
# arXiv:2508.15468), its network, quantisers and recipe ported to HGQ2 and
# trained from scratch, one variant per block and one block per GPU of olgpu.
# Hand-written: jedi_linear_hgq has no study, so
# scripts/experiments/make_train_catalogues.py neither writes nor regenerates
# this file.
# Baseline only: the Keras step computes no SCL, and setup() refuses it.
# The recipe is the experiment config's (7000 epochs, batches of 2790, Adam at
# 3e-3 under CosineRestarts, beta ramped from 2e-08 to 3e-06 through
# algorithm.beta_schedule); a block carries its variant alone, and a checkpoint
# every epoch, as HGQ2's ParetoFront keeps them in the published code: the
# design is picked among the checkpoints on the front of validation accuracy
# against EBOPs (baseline_<variant>_allepochs).
# The variants:
# 1. axis_trained, the reproduction: etarel and phirel measured from the full
#    jet's axis (data.full_axis=true), the rows standardised and their words
#    trained by HGQ2, as published.
# 2. kept_words: the axis of the kept constituents and the trigger's fixed
#    p_T and angle words, the like-for-like row against the vector neurons.
# 3. abs_words: no axis, the absolute [pt, eta, phi] as fixed words, the
#    angle word (1, 2, 8) so that it holds phi.
# 4. kept_trained: the kept axis with standardised rows and trained words.
#    Variant 1 minus 4 is the cost of the kept axis, 4 minus 2 that of the
#    fixed words.
# The four share the label jedi_linear_hgq_nconst16_baseline: select them with
# --only <index>, not by name.
#
# Every command is commented out: copy one to run it by hand, or feed the whole
# file to scripts/experiments/submit_train.sh, which queues them on a GPU or slurm.

# ========================================================================
# jedi_linear_hgq  nconst=16  (JEDI-linear as published)
# ========================================================================

# ------------------------------------------------------------------------
# jedi_linear_hgq  nconst=16  baseline  (method=augmentation, group=none)  axis_trained
# the reproduction: etarel and phirel from the full jet's axis, standardised,
# the words trained by HGQ2, as published
# hand-written, no study
# ------------------------------------------------------------------------
# python3 src/train.py \
#     experiment=hls4ml/jedi_linear_hgq_16 \
#     data.method=augmentation \
#     data.augmentation_scheme=none \
#     fold=null \
#     algorithm.inputs=standardised \
#     data.features='[pt,etarel,phirel]' \
#     data.full_axis=true \
#     +callbacks.model_checkpoint.every_n_epochs=1 \
#     callbacks.model_checkpoint.save_top_k=-1 \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/train/hls4ml/jedi_linear_hgq/nconst16/baseline_axis_trained_allepochs

# ------------------------------------------------------------------------
# jedi_linear_hgq  nconst=16  baseline  (method=augmentation, group=none)  kept_words
# etarel and phirel from the kept constituents' axis, the trigger's fixed
# words: the like-for-like row against the vector-neuron designs
# hand-written, no study
# ------------------------------------------------------------------------
# python3 src/train.py \
#     experiment=hls4ml/jedi_linear_hgq_16 \
#     data.method=augmentation \
#     data.augmentation_scheme=none \
#     fold=null \
#     algorithm.inputs=words \
#     data.features='[pt,etarel,phirel]' \
#     +callbacks.model_checkpoint.every_n_epochs=1 \
#     callbacks.model_checkpoint.save_top_k=-1 \
#     trainer=gpu \
#     trainer.devices=[1] \
#     hydra.run.dir=logs/train/hls4ml/jedi_linear_hgq/nconst16/baseline_kept_words_allepochs

# ------------------------------------------------------------------------
# jedi_linear_hgq  nconst=16  baseline  (method=augmentation, group=none)  abs_words
# no axis: the absolute [pt, eta, phi] as the trigger's fixed words, the
# angle word (1, 2, 8)
# hand-written, no study
# ------------------------------------------------------------------------
# python3 src/train.py \
#     experiment=hls4ml/jedi_linear_hgq_16 \
#     data.method=augmentation \
#     data.augmentation_scheme=none \
#     fold=null \
#     algorithm.inputs=words \
#     algorithm.quantisation.angle_kif='[1,2,8]' \
#     data.features='[pt,eta,phi]' \
#     +callbacks.model_checkpoint.every_n_epochs=1 \
#     callbacks.model_checkpoint.save_top_k=-1 \
#     trainer=gpu \
#     trainer.devices=[2] \
#     hydra.run.dir=logs/train/hls4ml/jedi_linear_hgq/nconst16/baseline_abs_words_allepochs

# ------------------------------------------------------------------------
# jedi_linear_hgq  nconst=16  baseline  (method=augmentation, group=none)  kept_trained
# etarel and phirel from the kept constituents' axis, standardised, the
# words trained by HGQ2
# hand-written, no study
# ------------------------------------------------------------------------
# python3 src/train.py \
#     experiment=hls4ml/jedi_linear_hgq_16 \
#     data.method=augmentation \
#     data.augmentation_scheme=none \
#     fold=null \
#     algorithm.inputs=standardised \
#     data.features='[pt,etarel,phirel]' \
#     +callbacks.model_checkpoint.every_n_epochs=1 \
#     callbacks.model_checkpoint.save_top_k=-1 \
#     trainer=gpu \
#     trainer.devices=[3] \
#     hydra.run.dir=logs/train/hls4ml/jedi_linear_hgq/nconst16/baseline_kept_trained_allepochs
