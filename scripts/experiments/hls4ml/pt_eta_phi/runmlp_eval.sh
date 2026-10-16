#!/usr/bin/env bash
# ========================================================================
# MLP EVALUATION COMMANDS
# ========================================================================
# One block per (constituent count, regime): 3 x 7 = 21 eval runs. Generated
# by scripts/experiments/make_train_catalogues.py; regenerate rather than edit by hand.
# Feature set [pt, eta, phi]: studies from logs/optuna/hls4ml/pt_eta_phi/, runs under logs/{train,eval}/hls4ml/pt_eta_phi/.
# Each block carries the best of the first 200 completed trials, by trial
# number, of its own study under logs/optuna/hls4ml/pt_eta_phi/;
# a block whose study is missing carries none and says so (transferred ablation).
#
# Every command is commented out: copy one to run it by hand, or feed the whole
# file to scripts/experiments/submit_eval.sh, which runs them under nohup.

# ========================================================================
# mlp  nconst=16
# ========================================================================

# ------------------------------------------------------------------------
# mlp  nconst=16  baseline  (method=augmentation, group=none)
# best trial 91 of mlp_nconst16_augmentation_none, val/acc_best=0.643628
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=hls4ml/mlp_16 \
#     data.method=augmentation \
#     data.augmentation_scheme=none \
#     data.n_folds=5 \
#     data.features='[pt,eta,phi]' \
#     algorithm.batchnorm=False \
#     algorithm.hidden_layers='[256,128,64]' \
#     algorithm.optimizer.lr=0.0013684728754844615 \
#     algorithm.optimizer.weight_decay=1e-07 \
#     checkpoint_manifest_path=logs/train/hls4ml/pt_eta_phi/mlp/nconst16/baseline \
#     results_table.invariance_metric=tv \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/hls4ml/pt_eta_phi/mlp/nconst16/baseline

# ------------------------------------------------------------------------
# mlp  nconst=16  aug_so2  (method=augmentation, group=so2)
# best trial 96 of mlp_nconst16_augmentation_so2, val/acc_best=0.646272
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=hls4ml/mlp_16 \
#     data.method=augmentation \
#     data.augmentation_scheme=so2 \
#     data.n_folds=5 \
#     data.features='[pt,eta,phi]' \
#     algorithm.batchnorm=False \
#     algorithm.hidden_layers='[256,128,64]' \
#     algorithm.optimizer.lr=0.0014799841822756572 \
#     algorithm.optimizer.weight_decay=1e-07 \
#     checkpoint_manifest_path=logs/train/hls4ml/pt_eta_phi/mlp/nconst16/aug_so2 \
#     results_table.invariance_metric=tv \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/hls4ml/pt_eta_phi/mlp/nconst16/aug_so2

# ------------------------------------------------------------------------
# mlp  nconst=16  aug_so3  (method=augmentation, group=so3)
# best trial 186 of mlp_nconst16_augmentation_so3, val/acc_best=0.600801
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=hls4ml/mlp_16 \
#     data.method=augmentation \
#     data.augmentation_scheme=so3 \
#     data.n_folds=5 \
#     data.features='[pt,eta,phi]' \
#     algorithm.batchnorm=False \
#     algorithm.hidden_layers='[256,128,64]' \
#     algorithm.optimizer.lr=0.0012605303666336379 \
#     algorithm.optimizer.weight_decay=1e-06 \
#     checkpoint_manifest_path=logs/train/hls4ml/pt_eta_phi/mlp/nconst16/aug_so3 \
#     results_table.invariance_metric=tv \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/hls4ml/pt_eta_phi/mlp/nconst16/aug_so3

# ------------------------------------------------------------------------
# mlp  nconst=16  aug_lorentz  (method=augmentation, group=lorentz)
# best trial 101 of mlp_nconst16_augmentation_lorentz, val/acc_best=0.641616
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=hls4ml/mlp_16 \
#     data.method=augmentation \
#     data.augmentation_scheme=lorentz \
#     data.n_folds=5 \
#     data.features='[pt,eta,phi]' \
#     algorithm.batchnorm=True \
#     algorithm.hidden_layers='[256,128,64]' \
#     algorithm.optimizer.lr=0.0005529255026723452 \
#     algorithm.optimizer.weight_decay=0.0 \
#     checkpoint_manifest_path=logs/train/hls4ml/pt_eta_phi/mlp/nconst16/aug_lorentz \
#     results_table.invariance_metric=tv \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/hls4ml/pt_eta_phi/mlp/nconst16/aug_lorentz

# ------------------------------------------------------------------------
# mlp  nconst=16  symm_so2  (method=symm_loss, group=so2)
# best trial 167 of mlp_nconst16_symm_loss_so2, val/acc_best=0.641097
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=hls4ml/mlp_16 \
#     data.method=symm_loss \
#     data.augmentation_scheme=so2 \
#     data.n_folds=5 \
#     data.features='[pt,eta,phi]' \
#     algorithm.batchnorm=True \
#     algorithm.hidden_layers='[256,128,64]' \
#     algorithm.optimizer.lr=0.0007923560814008036 \
#     algorithm.optimizer.weight_decay=0.0 \
#     algorithm.symm_loss_weight=0.001172733085335391 \
#     checkpoint_manifest_path=logs/train/hls4ml/pt_eta_phi/mlp/nconst16/symm_so2 \
#     results_table.invariance_metric=tv \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/hls4ml/pt_eta_phi/mlp/nconst16/symm_so2

# ------------------------------------------------------------------------
# mlp  nconst=16  symm_so3  (method=symm_loss, group=so3)
# best trial 94 of mlp_nconst16_symm_loss_so3, val/acc_best=0.631612
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=hls4ml/mlp_16 \
#     data.method=symm_loss \
#     data.augmentation_scheme=so3 \
#     data.n_folds=5 \
#     data.features='[pt,eta,phi]' \
#     algorithm.batchnorm=False \
#     algorithm.hidden_layers='[128,128,64]' \
#     algorithm.optimizer.lr=0.0009315630198214312 \
#     algorithm.optimizer.weight_decay=1e-07 \
#     algorithm.symm_loss_weight=0.010687184361693148 \
#     checkpoint_manifest_path=logs/train/hls4ml/pt_eta_phi/mlp/nconst16/symm_so3 \
#     results_table.invariance_metric=tv \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/hls4ml/pt_eta_phi/mlp/nconst16/symm_so3

# ------------------------------------------------------------------------
# mlp  nconst=16  symm_lorentz  (method=symm_loss, group=lorentz)
# best trial 137 of mlp_nconst16_symm_loss_lorentz, val/acc_best=0.641624
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=hls4ml/mlp_16 \
#     data.method=symm_loss \
#     data.augmentation_scheme=lorentz \
#     data.n_folds=5 \
#     data.features='[pt,eta,phi]' \
#     algorithm.batchnorm=True \
#     algorithm.hidden_layers='[256,128,64]' \
#     algorithm.optimizer.lr=0.0006939520642953449 \
#     algorithm.optimizer.weight_decay=1e-07 \
#     algorithm.symm_loss_weight=0.009614742160584527 \
#     checkpoint_manifest_path=logs/train/hls4ml/pt_eta_phi/mlp/nconst16/symm_lorentz \
#     results_table.invariance_metric=tv \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/hls4ml/pt_eta_phi/mlp/nconst16/symm_lorentz

# ========================================================================
# mlp  nconst=50
# ========================================================================

# ------------------------------------------------------------------------
# mlp  nconst=50  baseline  (method=augmentation, group=none)
# best trial 166 of mlp_nconst50_augmentation_none, val/acc_best=0.640876
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=hls4ml/mlp_50 \
#     data.method=augmentation \
#     data.augmentation_scheme=none \
#     data.n_folds=5 \
#     data.features='[pt,eta,phi]' \
#     algorithm.batchnorm=True \
#     algorithm.hidden_layers='[256,128,64]' \
#     algorithm.optimizer.lr=0.0014067019078310456 \
#     algorithm.optimizer.weight_decay=1e-05 \
#     checkpoint_manifest_path=logs/train/hls4ml/pt_eta_phi/mlp/nconst50/baseline \
#     results_table.invariance_metric=tv \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/hls4ml/pt_eta_phi/mlp/nconst50/baseline

# ------------------------------------------------------------------------
# mlp  nconst=50  aug_so2  (method=augmentation, group=so2)
# best trial 199 of mlp_nconst50_augmentation_so2, val/acc_best=0.645805
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=hls4ml/mlp_50 \
#     data.method=augmentation \
#     data.augmentation_scheme=so2 \
#     data.n_folds=5 \
#     data.features='[pt,eta,phi]' \
#     algorithm.batchnorm=True \
#     algorithm.hidden_layers='[256,128,64]' \
#     algorithm.optimizer.lr=0.0014762405544081797 \
#     algorithm.optimizer.weight_decay=0.0 \
#     checkpoint_manifest_path=logs/train/hls4ml/pt_eta_phi/mlp/nconst50/aug_so2 \
#     results_table.invariance_metric=tv \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/hls4ml/pt_eta_phi/mlp/nconst50/aug_so2

# ------------------------------------------------------------------------
# mlp  nconst=50  aug_so3  (method=augmentation, group=so3)
# best trial 192 of mlp_nconst50_augmentation_so3, val/acc_best=0.619177
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=hls4ml/mlp_50 \
#     data.method=augmentation \
#     data.augmentation_scheme=so3 \
#     data.n_folds=5 \
#     data.features='[pt,eta,phi]' \
#     algorithm.batchnorm=False \
#     algorithm.hidden_layers='[256,128,64]' \
#     algorithm.optimizer.lr=0.0017961045935094794 \
#     algorithm.optimizer.weight_decay=1e-07 \
#     checkpoint_manifest_path=logs/train/hls4ml/pt_eta_phi/mlp/nconst50/aug_so3 \
#     results_table.invariance_metric=tv \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/hls4ml/pt_eta_phi/mlp/nconst50/aug_so3

# ------------------------------------------------------------------------
# mlp  nconst=50  aug_lorentz  (method=augmentation, group=lorentz)
# best trial 75 of mlp_nconst50_augmentation_lorentz, val/acc_best=0.637635
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=hls4ml/mlp_50 \
#     data.method=augmentation \
#     data.augmentation_scheme=lorentz \
#     data.n_folds=5 \
#     data.features='[pt,eta,phi]' \
#     algorithm.batchnorm=True \
#     algorithm.hidden_layers='[256,128,64]' \
#     algorithm.optimizer.lr=0.000955188723452587 \
#     algorithm.optimizer.weight_decay=0.0 \
#     checkpoint_manifest_path=logs/train/hls4ml/pt_eta_phi/mlp/nconst50/aug_lorentz \
#     results_table.invariance_metric=tv \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/hls4ml/pt_eta_phi/mlp/nconst50/aug_lorentz

# ------------------------------------------------------------------------
# mlp  nconst=50  symm_so2  (method=symm_loss, group=so2)
# best trial 133 of mlp_nconst50_symm_loss_so2, val/acc_best=0.642989
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=hls4ml/mlp_50 \
#     data.method=symm_loss \
#     data.augmentation_scheme=so2 \
#     data.n_folds=5 \
#     data.features='[pt,eta,phi]' \
#     algorithm.batchnorm=True \
#     algorithm.hidden_layers='[256,128,64]' \
#     algorithm.optimizer.lr=0.0012523378266637182 \
#     algorithm.optimizer.weight_decay=0.0 \
#     algorithm.symm_loss_weight=0.006295857598067888 \
#     checkpoint_manifest_path=logs/train/hls4ml/pt_eta_phi/mlp/nconst50/symm_so2 \
#     results_table.invariance_metric=tv \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/hls4ml/pt_eta_phi/mlp/nconst50/symm_so2

# ------------------------------------------------------------------------
# mlp  nconst=50  symm_so3  (method=symm_loss, group=so3)
# best trial 114 of mlp_nconst50_symm_loss_so3, val/acc_best=0.629795
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=hls4ml/mlp_50 \
#     data.method=symm_loss \
#     data.augmentation_scheme=so3 \
#     data.n_folds=5 \
#     data.features='[pt,eta,phi]' \
#     algorithm.batchnorm=False \
#     algorithm.hidden_layers='[256,128,64]' \
#     algorithm.optimizer.lr=0.001677002485995432 \
#     algorithm.optimizer.weight_decay=1e-07 \
#     algorithm.symm_loss_weight=0.05936047927687784 \
#     checkpoint_manifest_path=logs/train/hls4ml/pt_eta_phi/mlp/nconst50/symm_so3 \
#     results_table.invariance_metric=tv \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/hls4ml/pt_eta_phi/mlp/nconst50/symm_so3

# ------------------------------------------------------------------------
# mlp  nconst=50  symm_lorentz  (method=symm_loss, group=lorentz)
# best trial 164 of mlp_nconst50_symm_loss_lorentz, val/acc_best=0.647277
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=hls4ml/mlp_50 \
#     data.method=symm_loss \
#     data.augmentation_scheme=lorentz \
#     data.n_folds=5 \
#     data.features='[pt,eta,phi]' \
#     algorithm.batchnorm=True \
#     algorithm.hidden_layers='[256,128,64]' \
#     algorithm.optimizer.lr=0.0010153631974156624 \
#     algorithm.optimizer.weight_decay=1e-05 \
#     algorithm.symm_loss_weight=0.005653202531233223 \
#     checkpoint_manifest_path=logs/train/hls4ml/pt_eta_phi/mlp/nconst50/symm_lorentz \
#     results_table.invariance_metric=tv \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/hls4ml/pt_eta_phi/mlp/nconst50/symm_lorentz

# ========================================================================
# mlp  nconst=150
# ========================================================================

# ------------------------------------------------------------------------
# mlp  nconst=150  baseline  (method=augmentation, group=none)
# best trial 168 of mlp_nconst150_augmentation_none, val/acc_best=0.625378
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=hls4ml/mlp_150 \
#     data.method=augmentation \
#     data.augmentation_scheme=none \
#     data.n_folds=5 \
#     data.features='[pt,eta,phi]' \
#     algorithm.batchnorm=True \
#     algorithm.hidden_layers='[256,128,64]' \
#     algorithm.optimizer.lr=0.000986650387891636 \
#     algorithm.optimizer.weight_decay=1e-05 \
#     checkpoint_manifest_path=logs/train/hls4ml/pt_eta_phi/mlp/nconst150/baseline \
#     results_table.invariance_metric=tv \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/hls4ml/pt_eta_phi/mlp/nconst150/baseline

# ------------------------------------------------------------------------
# mlp  nconst=150  aug_so2  (method=augmentation, group=so2)
# best trial 44 of mlp_nconst150_augmentation_so2, val/acc_best=0.634328
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=hls4ml/mlp_150 \
#     data.method=augmentation \
#     data.augmentation_scheme=so2 \
#     data.n_folds=5 \
#     data.features='[pt,eta,phi]' \
#     algorithm.batchnorm=True \
#     algorithm.hidden_layers='[256,128,64]' \
#     algorithm.optimizer.lr=0.000553265009170144 \
#     algorithm.optimizer.weight_decay=1e-05 \
#     checkpoint_manifest_path=logs/train/hls4ml/pt_eta_phi/mlp/nconst150/aug_so2 \
#     results_table.invariance_metric=tv \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/hls4ml/pt_eta_phi/mlp/nconst150/aug_so2

# ------------------------------------------------------------------------
# mlp  nconst=150  aug_so3  (method=augmentation, group=so3)
# best trial 191 of mlp_nconst150_augmentation_so3, val/acc_best=0.606107
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=hls4ml/mlp_150 \
#     data.method=augmentation \
#     data.augmentation_scheme=so3 \
#     data.n_folds=5 \
#     data.features='[pt,eta,phi]' \
#     algorithm.batchnorm=True \
#     algorithm.hidden_layers='[256,128,64]' \
#     algorithm.optimizer.lr=0.0010094331227907527 \
#     algorithm.optimizer.weight_decay=1e-07 \
#     checkpoint_manifest_path=logs/train/hls4ml/pt_eta_phi/mlp/nconst150/aug_so3 \
#     results_table.invariance_metric=tv \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/hls4ml/pt_eta_phi/mlp/nconst150/aug_so3

# ------------------------------------------------------------------------
# mlp  nconst=150  aug_lorentz  (method=augmentation, group=lorentz)
# best trial 102 of mlp_nconst150_augmentation_lorentz, val/acc_best=0.624319
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=hls4ml/mlp_150 \
#     data.method=augmentation \
#     data.augmentation_scheme=lorentz \
#     data.n_folds=5 \
#     data.features='[pt,eta,phi]' \
#     algorithm.batchnorm=True \
#     algorithm.hidden_layers='[256,128,64]' \
#     algorithm.optimizer.lr=0.0008262581391565444 \
#     algorithm.optimizer.weight_decay=1e-05 \
#     checkpoint_manifest_path=logs/train/hls4ml/pt_eta_phi/mlp/nconst150/aug_lorentz \
#     results_table.invariance_metric=tv \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/hls4ml/pt_eta_phi/mlp/nconst150/aug_lorentz

# ------------------------------------------------------------------------
# mlp  nconst=150  symm_so2  (method=symm_loss, group=so2)
# best trial 145 of mlp_nconst150_symm_loss_so2, val/acc_best=0.631438
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=hls4ml/mlp_150 \
#     data.method=symm_loss \
#     data.augmentation_scheme=so2 \
#     data.n_folds=5 \
#     data.features='[pt,eta,phi]' \
#     algorithm.batchnorm=True \
#     algorithm.hidden_layers='[256,128,64]' \
#     algorithm.optimizer.lr=0.0008233264151535088 \
#     algorithm.optimizer.weight_decay=1e-05 \
#     algorithm.symm_loss_weight=0.007558536541855199 \
#     checkpoint_manifest_path=logs/train/hls4ml/pt_eta_phi/mlp/nconst150/symm_so2 \
#     results_table.invariance_metric=tv \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/hls4ml/pt_eta_phi/mlp/nconst150/symm_so2

# ------------------------------------------------------------------------
# mlp  nconst=150  symm_so3  (method=symm_loss, group=so3)
# best trial 177 of mlp_nconst150_symm_loss_so3, val/acc_best=0.593420
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=hls4ml/mlp_150 \
#     data.method=symm_loss \
#     data.augmentation_scheme=so3 \
#     data.n_folds=5 \
#     data.features='[pt,eta,phi]' \
#     algorithm.batchnorm=False \
#     algorithm.hidden_layers='[256,128,64]' \
#     algorithm.optimizer.lr=0.0019302066049461432 \
#     algorithm.optimizer.weight_decay=1e-05 \
#     algorithm.symm_loss_weight=0.0022259839291159074 \
#     checkpoint_manifest_path=logs/train/hls4ml/pt_eta_phi/mlp/nconst150/symm_so3 \
#     results_table.invariance_metric=tv \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/hls4ml/pt_eta_phi/mlp/nconst150/symm_so3

# ------------------------------------------------------------------------
# mlp  nconst=150  symm_lorentz  (method=symm_loss, group=lorentz)
# best trial 194 of mlp_nconst150_symm_loss_lorentz, val/acc_best=0.633212
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=hls4ml/mlp_150 \
#     data.method=symm_loss \
#     data.augmentation_scheme=lorentz \
#     data.n_folds=5 \
#     data.features='[pt,eta,phi]' \
#     algorithm.batchnorm=True \
#     algorithm.hidden_layers='[256,128,64]' \
#     algorithm.optimizer.lr=0.0005144210656858335 \
#     algorithm.optimizer.weight_decay=0.0001 \
#     algorithm.symm_loss_weight=0.04530647258563953 \
#     checkpoint_manifest_path=logs/train/hls4ml/pt_eta_phi/mlp/nconst150/symm_lorentz \
#     results_table.invariance_metric=tv \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/hls4ml/pt_eta_phi/mlp/nconst150/symm_lorentz
