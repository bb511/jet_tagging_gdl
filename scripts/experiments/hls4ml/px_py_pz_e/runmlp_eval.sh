#!/usr/bin/env bash
# ========================================================================
# MLP EVALUATION COMMANDS
# ========================================================================
# One block per (constituent count, regime): 3 x 7 = 21 eval runs. Generated
# by scripts/experiments/make_train_catalogues.py; regenerate rather than edit by hand.
# Each block carries the best of the first 200 completed trials, by trial
# number, of its own study under logs/optuna/hls4ml/;
# a block whose study is missing carries none and says so (transferred ablation).
#
# Every command is commented out: copy one to run it by hand, or feed the whole
# file to scripts/experiments/submit_eval.sh, which queues them on a GPU or slurm.

# ========================================================================
# mlp  nconst=16
# ========================================================================

# ------------------------------------------------------------------------
# mlp  nconst=16  baseline  (method=augmentation, group=none)
# best trial 134 of mlp_nconst16_augmentation_none, val/acc_best=0.520063
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=hls4ml/mlp_16 \
#     data.method=augmentation \
#     data.augmentation_scheme=none \
#     data.n_folds=5 \
#     algorithm.batchnorm=True \
#     algorithm.hidden_layers='[256,128,64]' \
#     algorithm.optimizer.lr=0.00044698443798036827 \
#     algorithm.optimizer.weight_decay=1e-05 \
#     checkpoint_manifest_path=logs/train/hls4ml/mlp/nconst16/baseline \
#     results_table.invariance_metric=tv \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/hls4ml/mlp/nconst16/baseline

# ------------------------------------------------------------------------
# mlp  nconst=16  aug_so2  (method=augmentation, group=so2)
# best trial 152 of mlp_nconst16_augmentation_so2, val/acc_best=0.516426
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=hls4ml/mlp_16 \
#     data.method=augmentation \
#     data.augmentation_scheme=so2 \
#     data.n_folds=5 \
#     algorithm.batchnorm=True \
#     algorithm.hidden_layers='[256,128,64]' \
#     algorithm.optimizer.lr=0.0004266162721647833 \
#     algorithm.optimizer.weight_decay=1e-07 \
#     checkpoint_manifest_path=logs/train/hls4ml/mlp/nconst16/aug_so2 \
#     results_table.invariance_metric=tv \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/hls4ml/mlp/nconst16/aug_so2

# ------------------------------------------------------------------------
# mlp  nconst=16  aug_so3  (method=augmentation, group=so3)
# best trial 144 of mlp_nconst16_augmentation_so3, val/acc_best=0.474687
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=hls4ml/mlp_16 \
#     data.method=augmentation \
#     data.augmentation_scheme=so3 \
#     data.n_folds=5 \
#     algorithm.batchnorm=True \
#     algorithm.hidden_layers='[256,128,64]' \
#     algorithm.optimizer.lr=0.0003408676747140133 \
#     algorithm.optimizer.weight_decay=1e-05 \
#     checkpoint_manifest_path=logs/train/hls4ml/mlp/nconst16/aug_so3 \
#     results_table.invariance_metric=tv \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/hls4ml/mlp/nconst16/aug_so3

# ------------------------------------------------------------------------
# mlp  nconst=16  aug_lorentz  (method=augmentation, group=lorentz)
# best trial 191 of mlp_nconst16_augmentation_lorentz, val/acc_best=0.505730
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=hls4ml/mlp_16 \
#     data.method=augmentation \
#     data.augmentation_scheme=lorentz \
#     data.n_folds=5 \
#     algorithm.batchnorm=True \
#     algorithm.hidden_layers='[256,128,64]' \
#     algorithm.optimizer.lr=0.00034725919225115536 \
#     algorithm.optimizer.weight_decay=1e-06 \
#     checkpoint_manifest_path=logs/train/hls4ml/mlp/nconst16/aug_lorentz \
#     results_table.invariance_metric=tv \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/hls4ml/mlp/nconst16/aug_lorentz

# ------------------------------------------------------------------------
# mlp  nconst=16  symm_so2  (method=symm_loss, group=so2)
# best trial 140 of mlp_nconst16_symm_loss_so2, val/acc_best=0.521389
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=hls4ml/mlp_16 \
#     data.method=symm_loss \
#     data.augmentation_scheme=so2 \
#     data.n_folds=5 \
#     algorithm.batchnorm=True \
#     algorithm.hidden_layers='[256,128,64]' \
#     algorithm.optimizer.lr=0.00044022185219354227 \
#     algorithm.optimizer.weight_decay=0.0 \
#     algorithm.symm_loss_weight=0.0084496899469788 \
#     checkpoint_manifest_path=logs/train/hls4ml/mlp/nconst16/symm_so2 \
#     results_table.invariance_metric=tv \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/hls4ml/mlp/nconst16/symm_so2

# ------------------------------------------------------------------------
# mlp  nconst=16  symm_so3  (method=symm_loss, group=so3)
# best trial 160 of mlp_nconst16_symm_loss_so3, val/acc_best=0.505465
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=hls4ml/mlp_16 \
#     data.method=symm_loss \
#     data.augmentation_scheme=so3 \
#     data.n_folds=5 \
#     algorithm.batchnorm=True \
#     algorithm.hidden_layers='[256,128,64]' \
#     algorithm.optimizer.lr=0.0004532086876846241 \
#     algorithm.optimizer.weight_decay=1e-06 \
#     algorithm.symm_loss_weight=0.0013164929586646514 \
#     checkpoint_manifest_path=logs/train/hls4ml/mlp/nconst16/symm_so3 \
#     results_table.invariance_metric=tv \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/hls4ml/mlp/nconst16/symm_so3

# ------------------------------------------------------------------------
# mlp  nconst=16  symm_lorentz  (method=symm_loss, group=lorentz)
# best trial 109 of mlp_nconst16_symm_loss_lorentz, val/acc_best=0.515795
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=hls4ml/mlp_16 \
#     data.method=symm_loss \
#     data.augmentation_scheme=lorentz \
#     data.n_folds=5 \
#     algorithm.batchnorm=True \
#     algorithm.hidden_layers='[256,128,64]' \
#     algorithm.optimizer.lr=0.00043780304487145725 \
#     algorithm.optimizer.weight_decay=1e-06 \
#     algorithm.symm_loss_weight=0.0012346574572959497 \
#     checkpoint_manifest_path=logs/train/hls4ml/mlp/nconst16/symm_lorentz \
#     results_table.invariance_metric=tv \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/hls4ml/mlp/nconst16/symm_lorentz

# ========================================================================
# mlp  nconst=50
# ========================================================================

# ------------------------------------------------------------------------
# mlp  nconst=50  baseline  (method=augmentation, group=none)
# best trial 163 of mlp_nconst50_augmentation_none, val/acc_best=0.549348
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=hls4ml/mlp_50 \
#     data.method=augmentation \
#     data.augmentation_scheme=none \
#     data.n_folds=5 \
#     algorithm.batchnorm=True \
#     algorithm.hidden_layers='[256,128,64]' \
#     algorithm.optimizer.lr=0.0002636492949928615 \
#     algorithm.optimizer.weight_decay=0.0 \
#     checkpoint_manifest_path=logs/train/hls4ml/mlp/nconst50/baseline \
#     results_table.invariance_metric=tv \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/hls4ml/mlp/nconst50/baseline

# ------------------------------------------------------------------------
# mlp  nconst=50  aug_so2  (method=augmentation, group=so2)
# best trial 102 of mlp_nconst50_augmentation_so2, val/acc_best=0.553711
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=hls4ml/mlp_50 \
#     data.method=augmentation \
#     data.augmentation_scheme=so2 \
#     data.n_folds=5 \
#     algorithm.batchnorm=True \
#     algorithm.hidden_layers='[256,128,64]' \
#     algorithm.optimizer.lr=0.00030315048466834297 \
#     algorithm.optimizer.weight_decay=0.0 \
#     checkpoint_manifest_path=logs/train/hls4ml/mlp/nconst50/aug_so2 \
#     results_table.invariance_metric=tv \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/hls4ml/mlp/nconst50/aug_so2

# ------------------------------------------------------------------------
# mlp  nconst=50  aug_so3  (method=augmentation, group=so3)
# best trial 132 of mlp_nconst50_augmentation_so3, val/acc_best=0.517936
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=hls4ml/mlp_50 \
#     data.method=augmentation \
#     data.augmentation_scheme=so3 \
#     data.n_folds=5 \
#     algorithm.batchnorm=True \
#     algorithm.hidden_layers='[256,128,64]' \
#     algorithm.optimizer.lr=0.0003288611702603412 \
#     algorithm.optimizer.weight_decay=1e-05 \
#     checkpoint_manifest_path=logs/train/hls4ml/mlp/nconst50/aug_so3 \
#     results_table.invariance_metric=tv \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/hls4ml/mlp/nconst50/aug_so3

# ------------------------------------------------------------------------
# mlp  nconst=50  aug_lorentz  (method=augmentation, group=lorentz)
# best trial 121 of mlp_nconst50_augmentation_lorentz, val/acc_best=0.546622
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=hls4ml/mlp_50 \
#     data.method=augmentation \
#     data.augmentation_scheme=lorentz \
#     data.n_folds=5 \
#     algorithm.batchnorm=True \
#     algorithm.hidden_layers='[256,128,64]' \
#     algorithm.optimizer.lr=0.00024006932446039094 \
#     algorithm.optimizer.weight_decay=0.0001 \
#     checkpoint_manifest_path=logs/train/hls4ml/mlp/nconst50/aug_lorentz \
#     results_table.invariance_metric=tv \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/hls4ml/mlp/nconst50/aug_lorentz

# ------------------------------------------------------------------------
# mlp  nconst=50  symm_so2  (method=symm_loss, group=so2)
# best trial 162 of mlp_nconst50_symm_loss_so2, val/acc_best=0.560670
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=hls4ml/mlp_50 \
#     data.method=symm_loss \
#     data.augmentation_scheme=so2 \
#     data.n_folds=5 \
#     algorithm.batchnorm=True \
#     algorithm.hidden_layers='[256,128,64]' \
#     algorithm.optimizer.lr=0.00035659985348832795 \
#     algorithm.optimizer.weight_decay=0.0001 \
#     algorithm.symm_loss_weight=0.0013715828700093871 \
#     checkpoint_manifest_path=logs/train/hls4ml/mlp/nconst50/symm_so2 \
#     results_table.invariance_metric=tv \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/hls4ml/mlp/nconst50/symm_so2

# ------------------------------------------------------------------------
# mlp  nconst=50  symm_so3  (method=symm_loss, group=so3)
# best trial 97 of mlp_nconst50_symm_loss_so3, val/acc_best=0.538812
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=hls4ml/mlp_50 \
#     data.method=symm_loss \
#     data.augmentation_scheme=so3 \
#     data.n_folds=5 \
#     algorithm.batchnorm=True \
#     algorithm.hidden_layers='[256,128,64]' \
#     algorithm.optimizer.lr=0.0003052341471196719 \
#     algorithm.optimizer.weight_decay=1e-07 \
#     algorithm.symm_loss_weight=0.004093970867230705 \
#     checkpoint_manifest_path=logs/train/hls4ml/mlp/nconst50/symm_so3 \
#     results_table.invariance_metric=tv \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/hls4ml/mlp/nconst50/symm_so3

# ------------------------------------------------------------------------
# mlp  nconst=50  symm_lorentz  (method=symm_loss, group=lorentz)
# best trial 196 of mlp_nconst50_symm_loss_lorentz, val/acc_best=0.558608
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=hls4ml/mlp_50 \
#     data.method=symm_loss \
#     data.augmentation_scheme=lorentz \
#     data.n_folds=5 \
#     algorithm.batchnorm=True \
#     algorithm.hidden_layers='[256,128,64]' \
#     algorithm.optimizer.lr=0.0002744705759122752 \
#     algorithm.optimizer.weight_decay=1e-05 \
#     algorithm.symm_loss_weight=0.031083203666168003 \
#     checkpoint_manifest_path=logs/train/hls4ml/mlp/nconst50/symm_lorentz \
#     results_table.invariance_metric=tv \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/hls4ml/mlp/nconst50/symm_lorentz

# ========================================================================
# mlp  nconst=150
# ========================================================================

# ------------------------------------------------------------------------
# mlp  nconst=150  baseline  (method=augmentation, group=none)
# best trial 194 of mlp_nconst150_augmentation_none, val/acc_best=0.549590
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=hls4ml/mlp_150 \
#     data.method=augmentation \
#     data.augmentation_scheme=none \
#     data.n_folds=5 \
#     algorithm.batchnorm=True \
#     algorithm.hidden_layers='[256,128,64]' \
#     algorithm.optimizer.lr=0.00013827130924830083 \
#     algorithm.optimizer.weight_decay=0.0001 \
#     checkpoint_manifest_path=logs/train/hls4ml/mlp/nconst150/baseline \
#     results_table.invariance_metric=tv \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/hls4ml/mlp/nconst150/baseline

# ------------------------------------------------------------------------
# mlp  nconst=150  aug_so2  (method=augmentation, group=so2)
# best trial 185 of mlp_nconst150_augmentation_so2, val/acc_best=0.545097
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=hls4ml/mlp_150 \
#     data.method=augmentation \
#     data.augmentation_scheme=so2 \
#     data.n_folds=5 \
#     algorithm.batchnorm=True \
#     algorithm.hidden_layers='[256,128,64]' \
#     algorithm.optimizer.lr=0.0001348287146147312 \
#     algorithm.optimizer.weight_decay=1e-05 \
#     checkpoint_manifest_path=logs/train/hls4ml/mlp/nconst150/aug_so2 \
#     results_table.invariance_metric=tv \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/hls4ml/mlp/nconst150/aug_so2

# ------------------------------------------------------------------------
# mlp  nconst=150  aug_so3  (method=augmentation, group=so3)
# best trial 131 of mlp_nconst150_augmentation_so3, val/acc_best=0.513235
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=hls4ml/mlp_150 \
#     data.method=augmentation \
#     data.augmentation_scheme=so3 \
#     data.n_folds=5 \
#     algorithm.batchnorm=True \
#     algorithm.hidden_layers='[256,128,64]' \
#     algorithm.optimizer.lr=0.00016327062904011942 \
#     algorithm.optimizer.weight_decay=0.0001 \
#     checkpoint_manifest_path=logs/train/hls4ml/mlp/nconst150/aug_so3 \
#     results_table.invariance_metric=tv \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/hls4ml/mlp/nconst150/aug_so3

# ------------------------------------------------------------------------
# mlp  nconst=150  aug_lorentz  (method=augmentation, group=lorentz)
# best trial 199 of mlp_nconst150_augmentation_lorentz, val/acc_best=0.531378
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=hls4ml/mlp_150 \
#     data.method=augmentation \
#     data.augmentation_scheme=lorentz \
#     data.n_folds=5 \
#     algorithm.batchnorm=True \
#     algorithm.hidden_layers='[256,128,64]' \
#     algorithm.optimizer.lr=0.0003531253921191229 \
#     algorithm.optimizer.weight_decay=0.0 \
#     checkpoint_manifest_path=logs/train/hls4ml/mlp/nconst150/aug_lorentz \
#     results_table.invariance_metric=tv \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/hls4ml/mlp/nconst150/aug_lorentz

# ------------------------------------------------------------------------
# mlp  nconst=150  symm_so2  (method=symm_loss, group=so2)
# best trial 97 of mlp_nconst150_symm_loss_so2, val/acc_best=0.543243
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=hls4ml/mlp_150 \
#     data.method=symm_loss \
#     data.augmentation_scheme=so2 \
#     data.n_folds=5 \
#     algorithm.batchnorm=True \
#     algorithm.hidden_layers='[256,128,64]' \
#     algorithm.optimizer.lr=0.00020908988026243334 \
#     algorithm.optimizer.weight_decay=1e-05 \
#     algorithm.symm_loss_weight=0.028476330876315688 \
#     checkpoint_manifest_path=logs/train/hls4ml/mlp/nconst150/symm_so2 \
#     results_table.invariance_metric=tv \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/hls4ml/mlp/nconst150/symm_so2

# ------------------------------------------------------------------------
# mlp  nconst=150  symm_so3  (method=symm_loss, group=so3)
# best trial 127 of mlp_nconst150_symm_loss_so3, val/acc_best=0.523715
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=hls4ml/mlp_150 \
#     data.method=symm_loss \
#     data.augmentation_scheme=so3 \
#     data.n_folds=5 \
#     algorithm.batchnorm=False \
#     algorithm.hidden_layers='[256,128,64]' \
#     algorithm.optimizer.lr=0.005032777936294707 \
#     algorithm.optimizer.weight_decay=1e-07 \
#     algorithm.symm_loss_weight=0.01824071552064335 \
#     checkpoint_manifest_path=logs/train/hls4ml/mlp/nconst150/symm_so3 \
#     results_table.invariance_metric=tv \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/hls4ml/mlp/nconst150/symm_so3

# ------------------------------------------------------------------------
# mlp  nconst=150  symm_lorentz  (method=symm_loss, group=lorentz)
# best trial 149 of mlp_nconst150_symm_loss_lorentz, val/acc_best=0.542810
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=hls4ml/mlp_150 \
#     data.method=symm_loss \
#     data.augmentation_scheme=lorentz \
#     data.n_folds=5 \
#     algorithm.batchnorm=True \
#     algorithm.hidden_layers='[256,128,64]' \
#     algorithm.optimizer.lr=0.00038711018821330456 \
#     algorithm.optimizer.weight_decay=1e-05 \
#     algorithm.symm_loss_weight=0.05608939055444368 \
#     checkpoint_manifest_path=logs/train/hls4ml/mlp/nconst150/symm_lorentz \
#     results_table.invariance_metric=tv \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/hls4ml/mlp/nconst150/symm_lorentz
