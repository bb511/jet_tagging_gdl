#!/usr/bin/env bash
# ========================================================================
# JEDI_LINEAR EVALUATION COMMANDS
# ========================================================================
# One block per (constituent count, regime): 3 x 7 = 21 eval runs. Generated
# by scripts/optuna/make_train_catalogues.py; regenerate rather than edit by hand.
# Each block carries the best of the first 200 completed trials, by trial
# number, of its own study under logs/optuna/hls4ml/;
# a block whose study is missing carries none and says so (transferred ablation).
#
# Every command is commented out: copy one to run it by hand, or feed the whole
# file to scripts/cluster/submit_eval.sh, which queues them on a GPU or slurm.

# ========================================================================
# jedi_linear  nconst=16
# ========================================================================

# ------------------------------------------------------------------------
# jedi_linear  nconst=16  baseline  (method=augmentation, group=none)
# best trial 144 of jedi_linear_nconst16_augmentation_none, val/acc_best=0.553320
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=hls4ml/jedi_linear_16 \
#     data.method=augmentation \
#     data.augmentation_scheme=none \
#     data.n_folds=5 \
#     algorithm.aggregation=max \
#     algorithm.head_layers='[32,32,16]' \
#     algorithm.hidden=64 \
#     algorithm.optimizer.lr=0.003048592048530535 \
#     algorithm.optimizer.weight_decay=0.0 \
#     checkpoint_manifest_path=logs/train/hls4ml/jedi_linear/nconst16/baseline \
#     results_table.invariance_metric=tv \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/hls4ml/jedi_linear/nconst16/baseline

# ------------------------------------------------------------------------
# jedi_linear  nconst=16  aug_so2  (method=augmentation, group=so2)
# best trial 158 of jedi_linear_nconst16_augmentation_so2, val/acc_best=0.574054
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=hls4ml/jedi_linear_16 \
#     data.method=augmentation \
#     data.augmentation_scheme=so2 \
#     data.n_folds=5 \
#     algorithm.aggregation=max \
#     algorithm.head_layers='[64,32]' \
#     algorithm.hidden=128 \
#     algorithm.optimizer.lr=0.0030546033902610716 \
#     algorithm.optimizer.weight_decay=1e-07 \
#     checkpoint_manifest_path=logs/train/hls4ml/jedi_linear/nconst16/aug_so2 \
#     results_table.invariance_metric=tv \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/hls4ml/jedi_linear/nconst16/aug_so2

# ------------------------------------------------------------------------
# jedi_linear  nconst=16  aug_so3  (method=augmentation, group=so3)
# best trial 186 of jedi_linear_nconst16_augmentation_so3, val/acc_best=0.481389
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=hls4ml/jedi_linear_16 \
#     data.method=augmentation \
#     data.augmentation_scheme=so3 \
#     data.n_folds=5 \
#     algorithm.aggregation=max \
#     algorithm.head_layers='[32]' \
#     algorithm.hidden=64 \
#     algorithm.optimizer.lr=0.004664505762124715 \
#     algorithm.optimizer.weight_decay=1e-06 \
#     checkpoint_manifest_path=logs/train/hls4ml/jedi_linear/nconst16/aug_so3 \
#     results_table.invariance_metric=tv \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/hls4ml/jedi_linear/nconst16/aug_so3

# ------------------------------------------------------------------------
# jedi_linear  nconst=16  aug_lorentz  (method=augmentation, group=lorentz)
# best trial 96 of jedi_linear_nconst16_augmentation_lorentz, val/acc_best=0.573080
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=hls4ml/jedi_linear_16 \
#     data.method=augmentation \
#     data.augmentation_scheme=lorentz \
#     data.n_folds=5 \
#     algorithm.aggregation=max \
#     algorithm.head_layers='[32,16]' \
#     algorithm.hidden=128 \
#     algorithm.optimizer.lr=0.003772915054661497 \
#     algorithm.optimizer.weight_decay=1e-06 \
#     checkpoint_manifest_path=logs/train/hls4ml/jedi_linear/nconst16/aug_lorentz \
#     results_table.invariance_metric=tv \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/hls4ml/jedi_linear/nconst16/aug_lorentz

# ------------------------------------------------------------------------
# jedi_linear  nconst=16  symm_so2  (method=symm_loss, group=so2)
# best trial 174 of jedi_linear_nconst16_symm_loss_so2, val/acc_best=0.565760
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=hls4ml/jedi_linear_16 \
#     data.method=symm_loss \
#     data.augmentation_scheme=so2 \
#     data.n_folds=5 \
#     algorithm.aggregation=max \
#     algorithm.head_layers='[64,32]' \
#     algorithm.hidden=64 \
#     algorithm.optimizer.lr=0.003516879580069673 \
#     algorithm.optimizer.weight_decay=1e-07 \
#     algorithm.symm_loss_weight=0.007087398013906303 \
#     checkpoint_manifest_path=logs/train/hls4ml/jedi_linear/nconst16/symm_so2 \
#     results_table.invariance_metric=tv \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/hls4ml/jedi_linear/nconst16/symm_so2

# ------------------------------------------------------------------------
# jedi_linear  nconst=16  symm_so3  (method=symm_loss, group=so3)
# best trial 191 of jedi_linear_nconst16_symm_loss_so3, val/acc_best=0.577580
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=hls4ml/jedi_linear_16 \
#     data.method=symm_loss \
#     data.augmentation_scheme=so3 \
#     data.n_folds=5 \
#     algorithm.aggregation=max \
#     algorithm.head_layers='[64,32]' \
#     algorithm.hidden=128 \
#     algorithm.optimizer.lr=0.0020737975949696075 \
#     algorithm.optimizer.weight_decay=0.0 \
#     algorithm.symm_loss_weight=0.009752183777171774 \
#     checkpoint_manifest_path=logs/train/hls4ml/jedi_linear/nconst16/symm_so3 \
#     results_table.invariance_metric=tv \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/hls4ml/jedi_linear/nconst16/symm_so3

# ------------------------------------------------------------------------
# jedi_linear  nconst=16  symm_lorentz  (method=symm_loss, group=lorentz)
# best trial 182 of jedi_linear_nconst16_symm_loss_lorentz, val/acc_best=0.587170
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=hls4ml/jedi_linear_16 \
#     data.method=symm_loss \
#     data.augmentation_scheme=lorentz \
#     data.n_folds=5 \
#     algorithm.aggregation=max \
#     algorithm.head_layers='[32,16]' \
#     algorithm.hidden=128 \
#     algorithm.optimizer.lr=0.004724117800477397 \
#     algorithm.optimizer.weight_decay=1e-06 \
#     algorithm.symm_loss_weight=0.058712136114606954 \
#     checkpoint_manifest_path=logs/train/hls4ml/jedi_linear/nconst16/symm_lorentz \
#     results_table.invariance_metric=tv \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/hls4ml/jedi_linear/nconst16/symm_lorentz

# ========================================================================
# jedi_linear  nconst=50
# ========================================================================

# ------------------------------------------------------------------------
# jedi_linear  nconst=50  baseline  (method=augmentation, group=none)
# best trial 123 of jedi_linear_nconst50_augmentation_none, val/acc_best=0.579562
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=hls4ml/jedi_linear_50 \
#     data.method=augmentation \
#     data.augmentation_scheme=none \
#     data.n_folds=5 \
#     algorithm.aggregation=max \
#     algorithm.head_layers='[64,32]' \
#     algorithm.hidden=128 \
#     algorithm.optimizer.lr=0.0029683358823681763 \
#     algorithm.optimizer.weight_decay=1e-07 \
#     checkpoint_manifest_path=logs/train/hls4ml/jedi_linear/nconst50/baseline \
#     results_table.invariance_metric=tv \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/hls4ml/jedi_linear/nconst50/baseline

# ------------------------------------------------------------------------
# jedi_linear  nconst=50  aug_so2  (method=augmentation, group=so2)
# best trial 188 of jedi_linear_nconst50_augmentation_so2, val/acc_best=0.573477
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=hls4ml/jedi_linear_50 \
#     data.method=augmentation \
#     data.augmentation_scheme=so2 \
#     data.n_folds=5 \
#     algorithm.aggregation=max \
#     algorithm.head_layers='[32]' \
#     algorithm.hidden=64 \
#     algorithm.optimizer.lr=0.0029220705473181055 \
#     algorithm.optimizer.weight_decay=0.0 \
#     checkpoint_manifest_path=logs/train/hls4ml/jedi_linear/nconst50/aug_so2 \
#     results_table.invariance_metric=tv \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/hls4ml/jedi_linear/nconst50/aug_so2

# ------------------------------------------------------------------------
# jedi_linear  nconst=50  aug_so3  (method=augmentation, group=so3)
# best trial 193 of jedi_linear_nconst50_augmentation_so3, val/acc_best=0.445398
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=hls4ml/jedi_linear_50 \
#     data.method=augmentation \
#     data.augmentation_scheme=so3 \
#     data.n_folds=5 \
#     algorithm.aggregation=sum_over_n \
#     algorithm.head_layers='[64,32]' \
#     algorithm.hidden=128 \
#     algorithm.optimizer.lr=0.008936256846321641 \
#     algorithm.optimizer.weight_decay=1e-06 \
#     checkpoint_manifest_path=logs/train/hls4ml/jedi_linear/nconst50/aug_so3 \
#     results_table.invariance_metric=tv \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/hls4ml/jedi_linear/nconst50/aug_so3

# ------------------------------------------------------------------------
# jedi_linear  nconst=50  aug_lorentz  (method=augmentation, group=lorentz)
# best trial 130 of jedi_linear_nconst50_augmentation_lorentz, val/acc_best=0.503837
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=hls4ml/jedi_linear_50 \
#     data.method=augmentation \
#     data.augmentation_scheme=lorentz \
#     data.n_folds=5 \
#     algorithm.aggregation=sum_over_n \
#     algorithm.head_layers='[64,32]' \
#     algorithm.hidden=128 \
#     algorithm.optimizer.lr=0.005008406327332 \
#     algorithm.optimizer.weight_decay=1e-07 \
#     checkpoint_manifest_path=logs/train/hls4ml/jedi_linear/nconst50/aug_lorentz \
#     results_table.invariance_metric=tv \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/hls4ml/jedi_linear/nconst50/aug_lorentz

# ------------------------------------------------------------------------
# jedi_linear  nconst=50  symm_so2  (method=symm_loss, group=so2)
# best trial 101 of jedi_linear_nconst50_symm_loss_so2, val/acc_best=0.506942
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=hls4ml/jedi_linear_50 \
#     data.method=symm_loss \
#     data.augmentation_scheme=so2 \
#     data.n_folds=5 \
#     algorithm.aggregation=max \
#     algorithm.head_layers='[64,32,16]' \
#     algorithm.hidden=128 \
#     algorithm.optimizer.lr=0.004609758452035376 \
#     algorithm.optimizer.weight_decay=0.0 \
#     algorithm.symm_loss_weight=0.0029685086380761092 \
#     checkpoint_manifest_path=logs/train/hls4ml/jedi_linear/nconst50/symm_so2 \
#     results_table.invariance_metric=tv \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/hls4ml/jedi_linear/nconst50/symm_so2

# ------------------------------------------------------------------------
# jedi_linear  nconst=50  symm_so3  (method=symm_loss, group=so3)
# best trial 151 of jedi_linear_nconst50_symm_loss_so3, val/acc_best=0.581971
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=hls4ml/jedi_linear_50 \
#     data.method=symm_loss \
#     data.augmentation_scheme=so3 \
#     data.n_folds=5 \
#     algorithm.aggregation=max \
#     algorithm.head_layers='[32,16]' \
#     algorithm.hidden=128 \
#     algorithm.optimizer.lr=0.003902661459266972 \
#     algorithm.optimizer.weight_decay=1e-07 \
#     algorithm.symm_loss_weight=0.006078631095641767 \
#     checkpoint_manifest_path=logs/train/hls4ml/jedi_linear/nconst50/symm_so3 \
#     results_table.invariance_metric=tv \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/hls4ml/jedi_linear/nconst50/symm_so3

# ------------------------------------------------------------------------
# jedi_linear  nconst=50  symm_lorentz  (method=symm_loss, group=lorentz)
# best trial 114 of jedi_linear_nconst50_symm_loss_lorentz, val/acc_best=0.588494
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=hls4ml/jedi_linear_50 \
#     data.method=symm_loss \
#     data.augmentation_scheme=lorentz \
#     data.n_folds=5 \
#     algorithm.aggregation=max \
#     algorithm.head_layers='[64,32]' \
#     algorithm.hidden=64 \
#     algorithm.optimizer.lr=0.004216169101865109 \
#     algorithm.optimizer.weight_decay=0.0 \
#     algorithm.symm_loss_weight=0.03611877965968715 \
#     checkpoint_manifest_path=logs/train/hls4ml/jedi_linear/nconst50/symm_lorentz \
#     results_table.invariance_metric=tv \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/hls4ml/jedi_linear/nconst50/symm_lorentz

# ========================================================================
# jedi_linear  nconst=150
# ========================================================================

# ------------------------------------------------------------------------
# jedi_linear  nconst=150  baseline  (method=augmentation, group=none)
# best trial 181 of jedi_linear_nconst150_augmentation_none, val/acc_best=0.545915
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=hls4ml/jedi_linear_150 \
#     data.method=augmentation \
#     data.augmentation_scheme=none \
#     data.n_folds=5 \
#     algorithm.aggregation=max \
#     algorithm.head_layers='[64,32]' \
#     algorithm.hidden=128 \
#     algorithm.optimizer.lr=0.0018143433840631773 \
#     algorithm.optimizer.weight_decay=0.0 \
#     checkpoint_manifest_path=logs/train/hls4ml/jedi_linear/nconst150/baseline \
#     results_table.invariance_metric=tv \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/hls4ml/jedi_linear/nconst150/baseline

# ------------------------------------------------------------------------
# jedi_linear  nconst=150  aug_so2  (method=augmentation, group=so2)
# best trial 158 of jedi_linear_nconst150_augmentation_so2, val/acc_best=0.568325
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=hls4ml/jedi_linear_150 \
#     data.method=augmentation \
#     data.augmentation_scheme=so2 \
#     data.n_folds=5 \
#     algorithm.aggregation=max \
#     algorithm.head_layers='[32,16]' \
#     algorithm.hidden=128 \
#     algorithm.optimizer.lr=0.0029693257793846507 \
#     algorithm.optimizer.weight_decay=0.0 \
#     checkpoint_manifest_path=logs/train/hls4ml/jedi_linear/nconst150/aug_so2 \
#     results_table.invariance_metric=tv \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/hls4ml/jedi_linear/nconst150/aug_so2

# ------------------------------------------------------------------------
# jedi_linear  nconst=150  aug_so3  (method=augmentation, group=so3)
# best trial 180 of jedi_linear_nconst150_augmentation_so3, val/acc_best=0.510979
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=hls4ml/jedi_linear_150 \
#     data.method=augmentation \
#     data.augmentation_scheme=so3 \
#     data.n_folds=5 \
#     algorithm.aggregation=max \
#     algorithm.head_layers='[16]' \
#     algorithm.hidden=128 \
#     algorithm.optimizer.lr=0.0029281970100593067 \
#     algorithm.optimizer.weight_decay=1e-07 \
#     checkpoint_manifest_path=logs/train/hls4ml/jedi_linear/nconst150/aug_so3 \
#     results_table.invariance_metric=tv \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/hls4ml/jedi_linear/nconst150/aug_so3

# ------------------------------------------------------------------------
# jedi_linear  nconst=150  aug_lorentz  (method=augmentation, group=lorentz)
# best trial 196 of jedi_linear_nconst150_augmentation_lorentz, val/acc_best=0.513237
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=hls4ml/jedi_linear_150 \
#     data.method=augmentation \
#     data.augmentation_scheme=lorentz \
#     data.n_folds=5 \
#     algorithm.aggregation=sum_over_n \
#     algorithm.head_layers='[16]' \
#     algorithm.hidden=128 \
#     algorithm.optimizer.lr=0.007636897510095475 \
#     algorithm.optimizer.weight_decay=0.0 \
#     checkpoint_manifest_path=logs/train/hls4ml/jedi_linear/nconst150/aug_lorentz \
#     results_table.invariance_metric=tv \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/hls4ml/jedi_linear/nconst150/aug_lorentz

# ------------------------------------------------------------------------
# jedi_linear  nconst=150  symm_so2  (method=symm_loss, group=so2)
# best trial 148 of jedi_linear_nconst150_symm_loss_so2, val/acc_best=0.538133
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=hls4ml/jedi_linear_150 \
#     data.method=symm_loss \
#     data.augmentation_scheme=so2 \
#     data.n_folds=5 \
#     algorithm.aggregation=max \
#     algorithm.head_layers='[64,32]' \
#     algorithm.hidden=128 \
#     algorithm.optimizer.lr=0.0010867230274439453 \
#     algorithm.optimizer.weight_decay=1e-07 \
#     algorithm.symm_loss_weight=0.0036851990282835744 \
#     checkpoint_manifest_path=logs/train/hls4ml/jedi_linear/nconst150/symm_so2 \
#     results_table.invariance_metric=tv \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/hls4ml/jedi_linear/nconst150/symm_so2

# ------------------------------------------------------------------------
# jedi_linear  nconst=150  symm_so3  (method=symm_loss, group=so3)
# best trial 185 of jedi_linear_nconst150_symm_loss_so3, val/acc_best=0.559124
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=hls4ml/jedi_linear_150 \
#     data.method=symm_loss \
#     data.augmentation_scheme=so3 \
#     data.n_folds=5 \
#     algorithm.aggregation=max \
#     algorithm.head_layers='[64,32]' \
#     algorithm.hidden=128 \
#     algorithm.optimizer.lr=0.0014617448171031474 \
#     algorithm.optimizer.weight_decay=0.0 \
#     algorithm.symm_loss_weight=0.0012094600504099617 \
#     checkpoint_manifest_path=logs/train/hls4ml/jedi_linear/nconst150/symm_so3 \
#     results_table.invariance_metric=tv \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/hls4ml/jedi_linear/nconst150/symm_so3

# ------------------------------------------------------------------------
# jedi_linear  nconst=150  symm_lorentz  (method=symm_loss, group=lorentz)
# best trial 2 of jedi_linear_nconst150_symm_loss_lorentz, val/acc_best=0.526206
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=hls4ml/jedi_linear_150 \
#     data.method=symm_loss \
#     data.augmentation_scheme=lorentz \
#     data.n_folds=5 \
#     algorithm.aggregation=max \
#     algorithm.head_layers='[32,16]' \
#     algorithm.hidden=64 \
#     algorithm.optimizer.lr=0.0032889137796355177 \
#     algorithm.optimizer.weight_decay=1e-07 \
#     algorithm.symm_loss_weight=0.0012861144224156009 \
#     checkpoint_manifest_path=logs/train/hls4ml/jedi_linear/nconst150/symm_lorentz \
#     results_table.invariance_metric=tv \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/hls4ml/jedi_linear/nconst150/symm_lorentz
