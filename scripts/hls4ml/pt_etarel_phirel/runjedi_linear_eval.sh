#!/usr/bin/env bash
# ========================================================================
# JEDI_LINEAR EVALUATION COMMANDS
# ========================================================================
# One block per (constituent count, regime): 3 x 7 = 21 eval runs. Generated
# by scripts/optuna/make_train_catalogues.py; regenerate rather than edit by hand.
# Feature set [pt, etarel, phirel]: studies from logs/optuna/hls4ml/pt_etarel_phirel/, runs under logs/{train,eval}/hls4ml/pt_etarel_phirel/.
# Each block carries the best of the first 200 completed trials, by trial
# number, of its own study under logs/optuna/hls4ml/pt_etarel_phirel/;
# a block whose study is missing carries none and says so (transferred ablation).
#
# Every command is commented out: copy one to run it by hand, or feed the whole
# file to scripts/cluster/submit_eval.sh, which queues them on a GPU or slurm.

# ========================================================================
# jedi_linear  nconst=16
# ========================================================================

# ------------------------------------------------------------------------
# jedi_linear  nconst=16  baseline  (method=augmentation, group=none)
# best trial 107 of jedi_linear_nconst16_augmentation_none, val/acc_best=0.712059
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=hls4ml/jedi_linear_16 \
#     data.method=augmentation \
#     data.augmentation_scheme=none \
#     data.n_folds=5 \
#     data.features='[pt,etarel,phirel]' \
#     algorithm.aggregation=sum \
#     algorithm.head_layers='[32,32,16]' \
#     algorithm.hidden=128 \
#     algorithm.optimizer.lr=0.003110744733957464 \
#     algorithm.optimizer.weight_decay=1e-06 \
#     checkpoint_manifest_path=logs/train/hls4ml/pt_etarel_phirel/jedi_linear/nconst16/baseline \
#     results_table.invariance_metric=tv \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/hls4ml/pt_etarel_phirel/jedi_linear/nconst16/baseline

# ------------------------------------------------------------------------
# jedi_linear  nconst=16  aug_so2  (method=augmentation, group=so2)
# best trial 156 of jedi_linear_nconst16_augmentation_so2, val/acc_best=0.713731
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=hls4ml/jedi_linear_16 \
#     data.method=augmentation \
#     data.augmentation_scheme=so2 \
#     data.n_folds=5 \
#     data.features='[pt,etarel,phirel]' \
#     algorithm.aggregation=sum \
#     algorithm.head_layers='[64,32]' \
#     algorithm.hidden=128 \
#     algorithm.optimizer.lr=0.002505468569953152 \
#     algorithm.optimizer.weight_decay=1e-06 \
#     checkpoint_manifest_path=logs/train/hls4ml/pt_etarel_phirel/jedi_linear/nconst16/aug_so2 \
#     results_table.invariance_metric=tv \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/hls4ml/pt_etarel_phirel/jedi_linear/nconst16/aug_so2

# ------------------------------------------------------------------------
# jedi_linear  nconst=16  aug_so3  (method=augmentation, group=so3)
# best trial 179 of jedi_linear_nconst16_augmentation_so3, val/acc_best=0.663358
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=hls4ml/jedi_linear_16 \
#     data.method=augmentation \
#     data.augmentation_scheme=so3 \
#     data.n_folds=5 \
#     data.features='[pt,etarel,phirel]' \
#     algorithm.aggregation=sum_over_n \
#     algorithm.head_layers='[32,16]' \
#     algorithm.hidden=128 \
#     algorithm.optimizer.lr=0.008006643396239792 \
#     algorithm.optimizer.weight_decay=1e-07 \
#     checkpoint_manifest_path=logs/train/hls4ml/pt_etarel_phirel/jedi_linear/nconst16/aug_so3 \
#     results_table.invariance_metric=tv \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/hls4ml/pt_etarel_phirel/jedi_linear/nconst16/aug_so3

# ------------------------------------------------------------------------
# jedi_linear  nconst=16  aug_lorentz  (method=augmentation, group=lorentz)
# best trial 126 of jedi_linear_nconst16_augmentation_lorentz, val/acc_best=0.712995
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=hls4ml/jedi_linear_16 \
#     data.method=augmentation \
#     data.augmentation_scheme=lorentz \
#     data.n_folds=5 \
#     data.features='[pt,etarel,phirel]' \
#     algorithm.aggregation=sum \
#     algorithm.head_layers='[64,32]' \
#     algorithm.hidden=128 \
#     algorithm.optimizer.lr=0.00257645715683098 \
#     algorithm.optimizer.weight_decay=0.0 \
#     checkpoint_manifest_path=logs/train/hls4ml/pt_etarel_phirel/jedi_linear/nconst16/aug_lorentz \
#     results_table.invariance_metric=tv \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/hls4ml/pt_etarel_phirel/jedi_linear/nconst16/aug_lorentz

# ------------------------------------------------------------------------
# jedi_linear  nconst=16  symm_so2  (method=symm_loss, group=so2)
# best trial 53 of jedi_linear_nconst16_symm_loss_so2, val/acc_best=0.713023
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=hls4ml/jedi_linear_16 \
#     data.method=symm_loss \
#     data.augmentation_scheme=so2 \
#     data.n_folds=5 \
#     data.features='[pt,etarel,phirel]' \
#     algorithm.aggregation=sum \
#     algorithm.head_layers='[64,32]' \
#     algorithm.hidden=128 \
#     algorithm.optimizer.lr=0.0023089095697789797 \
#     algorithm.optimizer.weight_decay=1e-06 \
#     algorithm.symm_loss_weight=0.07594477943273266 \
#     checkpoint_manifest_path=logs/train/hls4ml/pt_etarel_phirel/jedi_linear/nconst16/symm_so2 \
#     results_table.invariance_metric=tv \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/hls4ml/pt_etarel_phirel/jedi_linear/nconst16/symm_so2

# ------------------------------------------------------------------------
# jedi_linear  nconst=16  symm_so3  (method=symm_loss, group=so3)
# best trial 104 of jedi_linear_nconst16_symm_loss_so3, val/acc_best=0.712635
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=hls4ml/jedi_linear_16 \
#     data.method=symm_loss \
#     data.augmentation_scheme=so3 \
#     data.n_folds=5 \
#     data.features='[pt,etarel,phirel]' \
#     algorithm.aggregation=sum \
#     algorithm.head_layers='[64,32]' \
#     algorithm.hidden=64 \
#     algorithm.optimizer.lr=0.003789719156745136 \
#     algorithm.optimizer.weight_decay=0.0 \
#     algorithm.symm_loss_weight=0.00469898987570086 \
#     checkpoint_manifest_path=logs/train/hls4ml/pt_etarel_phirel/jedi_linear/nconst16/symm_so3 \
#     results_table.invariance_metric=tv \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/hls4ml/pt_etarel_phirel/jedi_linear/nconst16/symm_so3

# ------------------------------------------------------------------------
# jedi_linear  nconst=16  symm_lorentz  (method=symm_loss, group=lorentz)
# best trial 173 of jedi_linear_nconst16_symm_loss_lorentz, val/acc_best=0.712094
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=hls4ml/jedi_linear_16 \
#     data.method=symm_loss \
#     data.augmentation_scheme=lorentz \
#     data.n_folds=5 \
#     data.features='[pt,etarel,phirel]' \
#     algorithm.aggregation=sum_over_n \
#     algorithm.head_layers='[32,16]' \
#     algorithm.hidden=128 \
#     algorithm.optimizer.lr=0.006922297654155337 \
#     algorithm.optimizer.weight_decay=0.0 \
#     algorithm.symm_loss_weight=0.45893757089235365 \
#     checkpoint_manifest_path=logs/train/hls4ml/pt_etarel_phirel/jedi_linear/nconst16/symm_lorentz \
#     results_table.invariance_metric=tv \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/hls4ml/pt_etarel_phirel/jedi_linear/nconst16/symm_lorentz

# ========================================================================
# jedi_linear  nconst=50
# ========================================================================

# ------------------------------------------------------------------------
# jedi_linear  nconst=50  baseline  (method=augmentation, group=none)
# best trial 130 of jedi_linear_nconst50_augmentation_none, val/acc_best=0.809950
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=hls4ml/jedi_linear_50 \
#     data.method=augmentation \
#     data.augmentation_scheme=none \
#     data.n_folds=5 \
#     data.features='[pt,etarel,phirel]' \
#     algorithm.aggregation=sum_over_n \
#     algorithm.head_layers='[64,32]' \
#     algorithm.hidden=128 \
#     algorithm.optimizer.lr=0.007728326725443656 \
#     algorithm.optimizer.weight_decay=0.0 \
#     checkpoint_manifest_path=logs/train/hls4ml/pt_etarel_phirel/jedi_linear/nconst50/baseline \
#     results_table.invariance_metric=tv \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/hls4ml/pt_etarel_phirel/jedi_linear/nconst50/baseline

# ------------------------------------------------------------------------
# jedi_linear  nconst=50  aug_so2  (method=augmentation, group=so2)
# best trial 157 of jedi_linear_nconst50_augmentation_so2, val/acc_best=0.809665
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=hls4ml/jedi_linear_50 \
#     data.method=augmentation \
#     data.augmentation_scheme=so2 \
#     data.n_folds=5 \
#     data.features='[pt,etarel,phirel]' \
#     algorithm.aggregation=sum_over_n \
#     algorithm.head_layers='[64,32,16]' \
#     algorithm.hidden=128 \
#     algorithm.optimizer.lr=0.009058039757148828 \
#     algorithm.optimizer.weight_decay=1e-06 \
#     checkpoint_manifest_path=logs/train/hls4ml/pt_etarel_phirel/jedi_linear/nconst50/aug_so2 \
#     results_table.invariance_metric=tv \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/hls4ml/pt_etarel_phirel/jedi_linear/nconst50/aug_so2

# ------------------------------------------------------------------------
# jedi_linear  nconst=50  aug_so3  (method=augmentation, group=so3)
# best trial 130 of jedi_linear_nconst50_augmentation_so3, val/acc_best=0.782790
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=hls4ml/jedi_linear_50 \
#     data.method=augmentation \
#     data.augmentation_scheme=so3 \
#     data.n_folds=5 \
#     data.features='[pt,etarel,phirel]' \
#     algorithm.aggregation=sum_over_n \
#     algorithm.head_layers='[64,32]' \
#     algorithm.hidden=64 \
#     algorithm.optimizer.lr=0.008785271500505077 \
#     algorithm.optimizer.weight_decay=0.0 \
#     checkpoint_manifest_path=logs/train/hls4ml/pt_etarel_phirel/jedi_linear/nconst50/aug_so3 \
#     results_table.invariance_metric=tv \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/hls4ml/pt_etarel_phirel/jedi_linear/nconst50/aug_so3

# ------------------------------------------------------------------------
# jedi_linear  nconst=50  aug_lorentz  (method=augmentation, group=lorentz)
# best trial 194 of jedi_linear_nconst50_augmentation_lorentz, val/acc_best=0.809759
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=hls4ml/jedi_linear_50 \
#     data.method=augmentation \
#     data.augmentation_scheme=lorentz \
#     data.n_folds=5 \
#     data.features='[pt,etarel,phirel]' \
#     algorithm.aggregation=sum_over_n \
#     algorithm.head_layers='[64,32]' \
#     algorithm.hidden=128 \
#     algorithm.optimizer.lr=0.009991237533156623 \
#     algorithm.optimizer.weight_decay=1e-07 \
#     checkpoint_manifest_path=logs/train/hls4ml/pt_etarel_phirel/jedi_linear/nconst50/aug_lorentz \
#     results_table.invariance_metric=tv \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/hls4ml/pt_etarel_phirel/jedi_linear/nconst50/aug_lorentz

# ------------------------------------------------------------------------
# jedi_linear  nconst=50  symm_so2  (method=symm_loss, group=so2)
# best trial 189 of jedi_linear_nconst50_symm_loss_so2, val/acc_best=0.809808
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=hls4ml/jedi_linear_50 \
#     data.method=symm_loss \
#     data.augmentation_scheme=so2 \
#     data.n_folds=5 \
#     data.features='[pt,etarel,phirel]' \
#     algorithm.aggregation=sum_over_n \
#     algorithm.head_layers='[32,32,16]' \
#     algorithm.hidden=128 \
#     algorithm.optimizer.lr=0.008190379840986733 \
#     algorithm.optimizer.weight_decay=0.0 \
#     algorithm.symm_loss_weight=0.0065708466145215855 \
#     checkpoint_manifest_path=logs/train/hls4ml/pt_etarel_phirel/jedi_linear/nconst50/symm_so2 \
#     results_table.invariance_metric=tv \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/hls4ml/pt_etarel_phirel/jedi_linear/nconst50/symm_so2

# ------------------------------------------------------------------------
# jedi_linear  nconst=50  symm_so3  (method=symm_loss, group=so3)
# best trial 126 of jedi_linear_nconst50_symm_loss_so3, val/acc_best=0.812290
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=hls4ml/jedi_linear_50 \
#     data.method=symm_loss \
#     data.augmentation_scheme=so3 \
#     data.n_folds=5 \
#     data.features='[pt,etarel,phirel]' \
#     algorithm.aggregation=sum_over_n \
#     algorithm.head_layers='[64,32,16]' \
#     algorithm.hidden=128 \
#     algorithm.optimizer.lr=0.0066064714914338294 \
#     algorithm.optimizer.weight_decay=0.0 \
#     algorithm.symm_loss_weight=0.002329670594205294 \
#     checkpoint_manifest_path=logs/train/hls4ml/pt_etarel_phirel/jedi_linear/nconst50/symm_so3 \
#     results_table.invariance_metric=tv \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/hls4ml/pt_etarel_phirel/jedi_linear/nconst50/symm_so3

# ------------------------------------------------------------------------
# jedi_linear  nconst=50  symm_lorentz  (method=symm_loss, group=lorentz)
# best trial 188 of jedi_linear_nconst50_symm_loss_lorentz, val/acc_best=0.810157
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=hls4ml/jedi_linear_50 \
#     data.method=symm_loss \
#     data.augmentation_scheme=lorentz \
#     data.n_folds=5 \
#     data.features='[pt,etarel,phirel]' \
#     algorithm.aggregation=sum_over_n \
#     algorithm.head_layers='[64,32,16]' \
#     algorithm.hidden=128 \
#     algorithm.optimizer.lr=0.00803264349593718 \
#     algorithm.optimizer.weight_decay=0.0 \
#     algorithm.symm_loss_weight=0.0013057074292595245 \
#     checkpoint_manifest_path=logs/train/hls4ml/pt_etarel_phirel/jedi_linear/nconst50/symm_lorentz \
#     results_table.invariance_metric=tv \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/hls4ml/pt_etarel_phirel/jedi_linear/nconst50/symm_lorentz

# ========================================================================
# jedi_linear  nconst=150
# ========================================================================

# ------------------------------------------------------------------------
# jedi_linear  nconst=150  baseline  (method=augmentation, group=none)
# best trial 183 of jedi_linear_nconst150_augmentation_none, val/acc_best=0.818529
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=hls4ml/jedi_linear_150 \
#     data.method=augmentation \
#     data.augmentation_scheme=none \
#     data.n_folds=5 \
#     data.features='[pt,etarel,phirel]' \
#     algorithm.aggregation=sum_over_n \
#     algorithm.head_layers='[64,32]' \
#     algorithm.hidden=128 \
#     algorithm.optimizer.lr=0.0080066592398403 \
#     algorithm.optimizer.weight_decay=1e-07 \
#     checkpoint_manifest_path=logs/train/hls4ml/pt_etarel_phirel/jedi_linear/nconst150/baseline \
#     results_table.invariance_metric=tv \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/hls4ml/pt_etarel_phirel/jedi_linear/nconst150/baseline

# ------------------------------------------------------------------------
# jedi_linear  nconst=150  aug_so2  (method=augmentation, group=so2)
# best trial 154 of jedi_linear_nconst150_augmentation_so2, val/acc_best=0.817704
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=hls4ml/jedi_linear_150 \
#     data.method=augmentation \
#     data.augmentation_scheme=so2 \
#     data.n_folds=5 \
#     data.features='[pt,etarel,phirel]' \
#     algorithm.aggregation=sum_over_n \
#     algorithm.head_layers='[64,32]' \
#     algorithm.hidden=128 \
#     algorithm.optimizer.lr=0.009845960177576635 \
#     algorithm.optimizer.weight_decay=0.0 \
#     checkpoint_manifest_path=logs/train/hls4ml/pt_etarel_phirel/jedi_linear/nconst150/aug_so2 \
#     results_table.invariance_metric=tv \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/hls4ml/pt_etarel_phirel/jedi_linear/nconst150/aug_so2

# ------------------------------------------------------------------------
# jedi_linear  nconst=150  aug_so3  (method=augmentation, group=so3)
# best trial 158 of jedi_linear_nconst150_augmentation_so3, val/acc_best=0.788154
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=hls4ml/jedi_linear_150 \
#     data.method=augmentation \
#     data.augmentation_scheme=so3 \
#     data.n_folds=5 \
#     data.features='[pt,etarel,phirel]' \
#     algorithm.aggregation=sum_over_n \
#     algorithm.head_layers='[16]' \
#     algorithm.hidden=128 \
#     algorithm.optimizer.lr=0.009381806117006411 \
#     algorithm.optimizer.weight_decay=1e-06 \
#     checkpoint_manifest_path=logs/train/hls4ml/pt_etarel_phirel/jedi_linear/nconst150/aug_so3 \
#     results_table.invariance_metric=tv \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/hls4ml/pt_etarel_phirel/jedi_linear/nconst150/aug_so3

# ------------------------------------------------------------------------
# jedi_linear  nconst=150  aug_lorentz  (method=augmentation, group=lorentz)
# best trial 164 of jedi_linear_nconst150_augmentation_lorentz, val/acc_best=0.817410
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=hls4ml/jedi_linear_150 \
#     data.method=augmentation \
#     data.augmentation_scheme=lorentz \
#     data.n_folds=5 \
#     data.features='[pt,etarel,phirel]' \
#     algorithm.aggregation=sum_over_n \
#     algorithm.head_layers='[64,32,16]' \
#     algorithm.hidden=128 \
#     algorithm.optimizer.lr=0.00911599091917613 \
#     algorithm.optimizer.weight_decay=0.0 \
#     checkpoint_manifest_path=logs/train/hls4ml/pt_etarel_phirel/jedi_linear/nconst150/aug_lorentz \
#     results_table.invariance_metric=tv \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/hls4ml/pt_etarel_phirel/jedi_linear/nconst150/aug_lorentz

# ------------------------------------------------------------------------
# jedi_linear  nconst=150  symm_so2  (method=symm_loss, group=so2)
# best trial 142 of jedi_linear_nconst150_symm_loss_so2, val/acc_best=0.816391
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=hls4ml/jedi_linear_150 \
#     data.method=symm_loss \
#     data.augmentation_scheme=so2 \
#     data.n_folds=5 \
#     data.features='[pt,etarel,phirel]' \
#     algorithm.aggregation=sum_over_n \
#     algorithm.head_layers='[64,32,16]' \
#     algorithm.hidden=128 \
#     algorithm.optimizer.lr=0.009695957560619823 \
#     algorithm.optimizer.weight_decay=1e-06 \
#     algorithm.symm_loss_weight=0.03358382330225255 \
#     checkpoint_manifest_path=logs/train/hls4ml/pt_etarel_phirel/jedi_linear/nconst150/symm_so2 \
#     results_table.invariance_metric=tv \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/hls4ml/pt_etarel_phirel/jedi_linear/nconst150/symm_so2

# ------------------------------------------------------------------------
# jedi_linear  nconst=150  symm_so3  (method=symm_loss, group=so3)
# best trial 183 of jedi_linear_nconst150_symm_loss_so3, val/acc_best=0.819859
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=hls4ml/jedi_linear_150 \
#     data.method=symm_loss \
#     data.augmentation_scheme=so3 \
#     data.n_folds=5 \
#     data.features='[pt,etarel,phirel]' \
#     algorithm.aggregation=sum_over_n \
#     algorithm.head_layers='[64,32]' \
#     algorithm.hidden=128 \
#     algorithm.optimizer.lr=0.009427956030407185 \
#     algorithm.optimizer.weight_decay=0.0 \
#     algorithm.symm_loss_weight=0.006467119954858895 \
#     checkpoint_manifest_path=logs/train/hls4ml/pt_etarel_phirel/jedi_linear/nconst150/symm_so3 \
#     results_table.invariance_metric=tv \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/hls4ml/pt_etarel_phirel/jedi_linear/nconst150/symm_so3

# ------------------------------------------------------------------------
# jedi_linear  nconst=150  symm_lorentz  (method=symm_loss, group=lorentz)
# best trial 158 of jedi_linear_nconst150_symm_loss_lorentz, val/acc_best=0.817205
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=hls4ml/jedi_linear_150 \
#     data.method=symm_loss \
#     data.augmentation_scheme=lorentz \
#     data.n_folds=5 \
#     data.features='[pt,etarel,phirel]' \
#     algorithm.aggregation=sum_over_n \
#     algorithm.head_layers='[64,32,16]' \
#     algorithm.hidden=128 \
#     algorithm.optimizer.lr=0.008641030091777241 \
#     algorithm.optimizer.weight_decay=0.0 \
#     algorithm.symm_loss_weight=0.12837651116271678 \
#     checkpoint_manifest_path=logs/train/hls4ml/pt_etarel_phirel/jedi_linear/nconst150/symm_lorentz \
#     results_table.invariance_metric=tv \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/hls4ml/pt_etarel_phirel/jedi_linear/nconst150/symm_lorentz
