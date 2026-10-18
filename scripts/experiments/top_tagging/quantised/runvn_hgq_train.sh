#!/usr/bin/env bash
# ========================================================================
# VN_HGQ TOP-TAGGING TRAINING COMMANDS
# ========================================================================
# Four train runs on the leading 100 constituents of the top-tagging
# reference set (Zenodo 2603256): the two vector-neuron designs of the
# comparison with JEDI-linear, as hls4ml/runvn_hgq_train.sh trains them at 150
# (its blocks 31, 32, 52 and 53). Blocks 1 and 2 train them in float, blocks
# 3 and 4 fine-tune each quantisation-aware from its float run, which must
# finish first.
# Each design is the best trial with head [32] and 32 scalar channels, D2's
# latency class, of a cheap search of runvn_cheap_search.sh: without the
# offset (cheap_h32, trial 96 of top_tagging_vn_cheap_nconst100_augmentation_none,
# val/acc_best 0.9298) and with the leader's eta and phi from the jet's axis
# as two more scalars (cheap_axis_h32, trial 0 of
# top_tagging_vn_cheap_axis_nconst100_augmentation_none, val/acc_best 0.9294,
# the hls4ml N = 150 design that the study enqueued first and no later trial
# of the class beat). The axis is that of the kept constituents, the whole jet
# for all but 0.5 % of the jets, since the cache holds no other.
# Blocks 3 and 4 follow the recipe of the published JEDI-linear
# (arXiv:2508.15468) at a peak learning rate of 1e-3, the one deviation, at
# which the hls4ml designs do not collapse: 7000 epochs at batches of 2790,
# Adam without weight decay or clipping under CosineRestarts, beta ramped from
# 2e-08 to 3e-06, folded normalisations, truncating data lanes and a
# checkpoint every epoch. The design without the offset reads the trigger's
# fixed words, the one with it HGQ2-trained words.
# Hand-written: vn_hgq has no study, so
# scripts/experiments/make_train_catalogues.py neither writes nor regenerates
# this file. Fold i of blocks 3 and 4 starts from fold i of its float run, named
# by the checkpoint its manifest records
# (algorithm.init_from=<float run>/fold_<i>/checkpoints/epoch_<k>.ckpt): a run
# directory as init_from is read for its main run alone.
#
# Every command is commented out: copy one to run it by hand, or feed the whole
# file to scripts/experiments/submit_train.sh, which runs them under nohup.

# ========================================================================
# vn  nconst=100  (the float designs)
# ========================================================================

# ------------------------------------------------------------------------
# vn nconst=100 baseline, one block, cheap_h32:
# no gate, no stream, head [32], 32 scalar channels
# hand-written: trial 96 of top_tagging_vn_cheap_nconst100_augmentation_none
# ------------------------------------------------------------------------
# python3 src/train.py \
#     experiment=top_tagging/vn_100 \
#     data.method=augmentation \
#     data.augmentation_scheme=none \
#     fold=null \
#     algorithm.dropout=0.0 \
#     algorithm.gate=false \
#     algorithm.head_layers='[32]' \
#     algorithm.invariants=8 \
#     algorithm.mlp_layers=2 \
#     algorithm.num_blocks=1 \
#     algorithm.optimizer.lr=0.006199864668247216 \
#     algorithm.optimizer.weight_decay=0.05 \
#     algorithm.scalar_channels=32 \
#     algorithm.stream_channels=0 \
#     algorithm.tensor_invariants=4 \
#     algorithm.vector_channels=16 \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/train/top_tagging/vn/nconst100/baseline_1block_cheap_h32

# ------------------------------------------------------------------------
# vn nconst=100 baseline, one block, cheap_axis_h32:
# no gate, no stream, head [32], 32 scalar channels, the leader's eta and
# phi from the jet's axis as two more first-block scalars
# hand-written: trial 0 of top_tagging_vn_cheap_axis_nconst100_augmentation_none
# ------------------------------------------------------------------------
# python3 src/train.py \
#     experiment=top_tagging/vn_100 \
#     data.method=augmentation \
#     data.augmentation_scheme=none \
#     fold=null \
#     algorithm.axis_offset=true \
#     algorithm.dropout=0.0 \
#     algorithm.gate=false \
#     algorithm.head_layers='[32]' \
#     algorithm.invariants=32 \
#     algorithm.mlp_layers=2 \
#     algorithm.nbr_feats=5 \
#     algorithm.num_blocks=1 \
#     algorithm.optimizer.lr=0.006499577920199478 \
#     algorithm.optimizer.weight_decay=0.01 \
#     algorithm.scalar_channels=32 \
#     algorithm.stream_channels=0 \
#     algorithm.tensor_invariants=4 \
#     algorithm.vector_channels=16 \
#     data.features='[pt,eta,phi,etarel,phirel]' \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/train/top_tagging/vn/nconst100/baseline_1block_cheap_axis_h32

# ========================================================================
# vn_hgq  nconst=100  (the designs of the comparison with JEDI-linear)
# ========================================================================

# ------------------------------------------------------------------------
# vn_hgq  nconst=100  baseline  (method=augmentation, group=none)  beta 2e-08 -> 3e-06
# the design without the offset (block 1) with the trigger's fixed words and
# JEDI-linear's recipe at a peak learning rate of 1e-3: 7000 epochs, ramped
# beta, restarting cosine, a checkpoint every epoch
# hand-written, no study
# ------------------------------------------------------------------------
# python3 src/train.py \
#     experiment=top_tagging/vn_hgq_100 \
#     data.method=augmentation \
#     data.augmentation_scheme=none \
#     fold=null \
#     algorithm.beta_schedule='[[0,2e-8,linear],[2000,3e-7,log],[7000,3e-6,constant]]' \
#     algorithm.dropout=0.0 \
#     algorithm.gate=false \
#     algorithm.head_layers='[32]' \
#     algorithm.init_from=logs/train/top_tagging/vn/nconst100/baseline_1block_cheap_h32 \
#     algorithm.invariants=8 \
#     algorithm.mlp_layers=2 \
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
#     data.batch_size=2790 \
#     +callbacks.model_checkpoint.every_n_epochs=1 \
#     callbacks.model_checkpoint.save_top_k=-1 \
#     trainer=gpu \
#     trainer.devices=[0] \
#     trainer.max_epochs=7000 \
#     hydra.run.dir=logs/train/top_tagging/vn_hgq/nconst100/baseline_cheap_h32_foldtrn_jedirecipe_lr1e-3_everyepoch

# ------------------------------------------------------------------------
# vn_hgq  nconst=100  baseline  (method=augmentation, group=none)  beta 2e-08 -> 3e-06
# the design with the offset (block 2) with trained input words and
# JEDI-linear's recipe at a peak learning rate of 1e-3: 7000 epochs, ramped
# beta, restarting cosine, a checkpoint every epoch
# hand-written, no study
# ------------------------------------------------------------------------
# python3 src/train.py \
#     experiment=top_tagging/vn_hgq_100 \
#     data.method=augmentation \
#     data.augmentation_scheme=none \
#     fold=null \
#     algorithm.axis_offset=true \
#     algorithm.beta_schedule='[[0,2e-8,linear],[2000,3e-7,log],[7000,3e-6,constant]]' \
#     algorithm.dropout=0.0 \
#     algorithm.gate=false \
#     algorithm.head_layers='[32]' \
#     algorithm.init_from=logs/train/top_tagging/vn/nconst100/baseline_1block_cheap_axis_h32 \
#     algorithm.invariants=32 \
#     algorithm.mlp_layers=2 \
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
#     data.batch_size=2790 \
#     data.features='[pt,eta,phi,etarel,phirel]' \
#     +callbacks.model_checkpoint.every_n_epochs=1 \
#     callbacks.model_checkpoint.save_top_k=-1 \
#     trainer=gpu \
#     trainer.devices=[0] \
#     trainer.max_epochs=7000 \
#     hydra.run.dir=logs/train/top_tagging/vn_hgq/nconst100/baseline_cheap_axis_h32_foldtrn_trainedwords_jedirecipe_lr1e-3_everyepoch
