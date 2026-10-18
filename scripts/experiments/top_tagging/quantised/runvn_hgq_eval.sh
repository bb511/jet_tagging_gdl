#!/usr/bin/env bash
# ========================================================================
# VN_HGQ TOP-TAGGING EVALUATION COMMANDS
# ========================================================================
# Two eval runs of the quantisation-aware trainings of runvn_hgq_train.sh
# (its blocks 3 and 4), the vector-neuron designs of the comparison with
# JEDI-linear at 100 constituents of the top-tagging set. Each evaluates the
# most accurate design whose latency at a 5 ns clock is estimated at most
# 100 ns: the main run's and each fold's pick, through the manifests that
# scripts/experiments/quantised_picks.py pick-hw writes to
# <run dir>/picks/best100ns.
# Hand-written: vn_hgq has no study, so
# scripts/experiments/make_train_catalogues.py neither writes nor regenerates
# this file.
# Each block repeats its training block's overrides and runs Keras in float64
# on the CPU (algorithm.jax_platform), where it reproduces the fixed-point
# arithmetic exactly; on a GPU, XLA's float64 2**f is off by an ulp for some f.
# It writes outputs/main_hgq/ beside the probe's outputs/main.npz, the input of
# scripts/synthesis/synthesise_vn.py, and its row (algorithm.name) to
# results/top_tagging/quantised/. The probe is on for both: the kept axis is
# the one the transforms rebase on.
# The two blocks share the label vn_hgq_nconst100_baseline: select them with
# --only <index>.
#
# Every command is commented out: copy one to run it by hand, or feed the whole
# file to scripts/experiments/submit_eval.sh, which runs them under nohup.

# ========================================================================
# vn_hgq  nconst=100  (the designs of the comparison with JEDI-linear)
# ========================================================================

# ------------------------------------------------------------------------
# vn_hgq  nconst=100  baseline  (method=augmentation, group=none)  beta 2e-08 -> 3e-06
# the design without the offset, the trigger's fixed words; the most accurate
# design under 100 ns
# hand-written, no study
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=top_tagging/vn_hgq_100 \
#     data.method=augmentation \
#     data.augmentation_scheme=none \
#     algorithm.beta_schedule='[[0,2e-8,linear],[2000,3e-7,log],[7000,3e-6,constant]]' \
#     algorithm.dropout=0.0 \
#     algorithm.floatx=float64 \
#     algorithm.gate=false \
#     algorithm.head_layers='[32]' \
#     algorithm.init_from=logs/train/top_tagging/vn/nconst100/baseline_1block_cheap_h32 \
#     algorithm.invariants=8 \
#     algorithm.jax_platform=cpu \
#     algorithm.mlp_layers=2 \
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
#     checkpoint_manifest_path=logs/train/top_tagging/vn_hgq/nconst100/baseline_cheap_h32_foldtrn_jedirecipe_lr1e-3_everyepoch/picks/best100ns \
#     results_table.invariance_metric=tv \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/top_tagging/vn_hgq/nconst100/baseline_cheap_h32_foldtrn_jedirecipe_lr1e-3_everyepoch_best100ns

# ------------------------------------------------------------------------
# vn_hgq  nconst=100  baseline  (method=augmentation, group=none)  beta 2e-08 -> 3e-06
# the design with the offset, trained input words; the most accurate design
# under 100 ns
# hand-written, no study
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=top_tagging/vn_hgq_100 \
#     data.method=augmentation \
#     data.augmentation_scheme=none \
#     algorithm.axis_offset=true \
#     algorithm.beta_schedule='[[0,2e-8,linear],[2000,3e-7,log],[7000,3e-6,constant]]' \
#     algorithm.dropout=0.0 \
#     algorithm.floatx=float64 \
#     algorithm.gate=false \
#     algorithm.head_layers='[32]' \
#     algorithm.init_from=logs/train/top_tagging/vn/nconst100/baseline_1block_cheap_axis_h32 \
#     algorithm.invariants=32 \
#     algorithm.jax_platform=cpu \
#     algorithm.mlp_layers=2 \
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
#     checkpoint_manifest_path=logs/train/top_tagging/vn_hgq/nconst100/baseline_cheap_axis_h32_foldtrn_trainedwords_jedirecipe_lr1e-3_everyepoch/picks/best100ns \
#     data.features='[pt,eta,phi,etarel,phirel]' \
#     results_table.invariance_metric=tv \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/top_tagging/vn_hgq/nconst100/baseline_cheap_axis_h32_foldtrn_trainedwords_jedirecipe_lr1e-3_everyepoch_best100ns
