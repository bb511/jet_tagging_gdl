#!/usr/bin/env bash
# ========================================================================
# JETFORMER EVALUATION COMMANDS
# ========================================================================
# One block per (constituent count, regime): 3 x 7 = 21 eval runs. Generated
# by scripts/optuna/make_train_catalogues.py; regenerate rather than edit by hand.
# Feature set [pt, eta, phi]: studies from logs/optuna/hls4ml/pt_eta_phi/, runs under logs/{train,eval}/hls4ml/pt_eta_phi/.
# Each block carries the best of the first 200 completed trials, by trial
# number, of its own study under logs/optuna/hls4ml/pt_eta_phi/;
# a block whose study is missing carries none and says so (transferred ablation).
#
# Every command is commented out: copy one to run it by hand, or feed the whole
# file to scripts/cluster/submit_eval.sh, which queues them on a GPU or slurm.

# ========================================================================
# jetformer  nconst=16
# ========================================================================

# ------------------------------------------------------------------------
# jetformer  nconst=16  baseline  (method=augmentation, group=none)
# best trial 195 of jetformer_nconst16_augmentation_none, val/acc_best=0.716064
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=hls4ml/jetformer_16 \
#     data.method=augmentation \
#     data.augmentation_scheme=none \
#     data.n_folds=5 \
#     data.features='[pt,eta,phi]' \
#     algorithm.dropout=0.0 \
#     algorithm.embed_dim=96 \
#     algorithm.num_heads=4 \
#     algorithm.num_transformers=4 \
#     algorithm.optimizer.lr=0.0014460922959618515 \
#     algorithm.optimizer.weight_decay=0.0 \
#     checkpoint_manifest_path=logs/train/hls4ml/pt_eta_phi/jetformer/nconst16/baseline \
#     results_table.invariance_metric=tv \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/hls4ml/pt_eta_phi/jetformer/nconst16/baseline

# ------------------------------------------------------------------------
# jetformer  nconst=16  aug_so2  (method=augmentation, group=so2)
# best trial 191 of jetformer_nconst16_augmentation_so2, val/acc_best=0.718128
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=hls4ml/jetformer_16 \
#     data.method=augmentation \
#     data.augmentation_scheme=so2 \
#     data.n_folds=5 \
#     data.features='[pt,eta,phi]' \
#     algorithm.dropout=0.0 \
#     algorithm.embed_dim=96 \
#     algorithm.num_heads=4 \
#     algorithm.num_transformers=4 \
#     algorithm.optimizer.lr=0.0012673003363409562 \
#     algorithm.optimizer.weight_decay=0.0001 \
#     checkpoint_manifest_path=logs/train/hls4ml/pt_eta_phi/jetformer/nconst16/aug_so2 \
#     results_table.invariance_metric=tv \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/hls4ml/pt_eta_phi/jetformer/nconst16/aug_so2

# ------------------------------------------------------------------------
# jetformer  nconst=16  aug_so3  (method=augmentation, group=so3)
# best trial 199 of jetformer_nconst16_augmentation_so3, val/acc_best=0.672771
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=hls4ml/jetformer_16 \
#     data.method=augmentation \
#     data.augmentation_scheme=so3 \
#     data.n_folds=5 \
#     data.features='[pt,eta,phi]' \
#     algorithm.dropout=0.0 \
#     algorithm.embed_dim=96 \
#     algorithm.num_heads=4 \
#     algorithm.num_transformers=4 \
#     algorithm.optimizer.lr=0.0032237614583596214 \
#     algorithm.optimizer.weight_decay=0.01 \
#     checkpoint_manifest_path=logs/train/hls4ml/pt_eta_phi/jetformer/nconst16/aug_so3 \
#     results_table.invariance_metric=tv \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/hls4ml/pt_eta_phi/jetformer/nconst16/aug_so3

# ------------------------------------------------------------------------
# jetformer  nconst=16  aug_lorentz  (method=augmentation, group=lorentz)
# best trial 41 of jetformer_nconst16_augmentation_lorentz, val/acc_best=0.715964
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=hls4ml/jetformer_16 \
#     data.method=augmentation \
#     data.augmentation_scheme=lorentz \
#     data.n_folds=5 \
#     data.features='[pt,eta,phi]' \
#     algorithm.dropout=0.0 \
#     algorithm.embed_dim=64 \
#     algorithm.num_heads=4 \
#     algorithm.num_transformers=4 \
#     algorithm.optimizer.lr=0.002477829352310248 \
#     algorithm.optimizer.weight_decay=0.0001 \
#     checkpoint_manifest_path=logs/train/hls4ml/pt_eta_phi/jetformer/nconst16/aug_lorentz \
#     results_table.invariance_metric=tv \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/hls4ml/pt_eta_phi/jetformer/nconst16/aug_lorentz

# ------------------------------------------------------------------------
# jetformer  nconst=16  symm_so2  (method=symm_loss, group=so2)
# best trial 194 of jetformer_nconst16_symm_loss_so2, val/acc_best=0.717896
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=hls4ml/jetformer_16 \
#     data.method=symm_loss \
#     data.augmentation_scheme=so2 \
#     data.n_folds=5 \
#     data.features='[pt,eta,phi]' \
#     algorithm.dropout=0.0 \
#     algorithm.embed_dim=96 \
#     algorithm.num_heads=4 \
#     algorithm.num_transformers=4 \
#     algorithm.optimizer.lr=0.001710198930082827 \
#     algorithm.optimizer.weight_decay=0.001 \
#     algorithm.symm_loss_weight=0.006731026556564273 \
#     checkpoint_manifest_path=logs/train/hls4ml/pt_eta_phi/jetformer/nconst16/symm_so2 \
#     results_table.invariance_metric=tv \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/hls4ml/pt_eta_phi/jetformer/nconst16/symm_so2

# ------------------------------------------------------------------------
# jetformer  nconst=16  symm_so3  (method=symm_loss, group=so3)
# best trial 175 of jetformer_nconst16_symm_loss_so3, val/acc_best=0.716963
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=hls4ml/jetformer_16 \
#     data.method=symm_loss \
#     data.augmentation_scheme=so3 \
#     data.n_folds=5 \
#     data.features='[pt,eta,phi]' \
#     algorithm.dropout=0.0 \
#     algorithm.embed_dim=96 \
#     algorithm.num_heads=4 \
#     algorithm.num_transformers=4 \
#     algorithm.optimizer.lr=0.0013241308451906152 \
#     algorithm.optimizer.weight_decay=0.001 \
#     algorithm.symm_loss_weight=0.002238682661500992 \
#     checkpoint_manifest_path=logs/train/hls4ml/pt_eta_phi/jetformer/nconst16/symm_so3 \
#     results_table.invariance_metric=tv \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/hls4ml/pt_eta_phi/jetformer/nconst16/symm_so3

# ------------------------------------------------------------------------
# jetformer  nconst=16  symm_lorentz  (method=symm_loss, group=lorentz)
# best trial 178 of jetformer_nconst16_symm_loss_lorentz, val/acc_best=0.717016
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=hls4ml/jetformer_16 \
#     data.method=symm_loss \
#     data.augmentation_scheme=lorentz \
#     data.n_folds=5 \
#     data.features='[pt,eta,phi]' \
#     algorithm.dropout=0.0 \
#     algorithm.embed_dim=96 \
#     algorithm.num_heads=4 \
#     algorithm.num_transformers=4 \
#     algorithm.optimizer.lr=0.0018881812298514092 \
#     algorithm.optimizer.weight_decay=0.0001 \
#     algorithm.symm_loss_weight=0.008917646558988244 \
#     checkpoint_manifest_path=logs/train/hls4ml/pt_eta_phi/jetformer/nconst16/symm_lorentz \
#     results_table.invariance_metric=tv \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/hls4ml/pt_eta_phi/jetformer/nconst16/symm_lorentz

# ========================================================================
# jetformer  nconst=50
# ========================================================================

# ------------------------------------------------------------------------
# jetformer  nconst=50  baseline  (method=augmentation, group=none)
# best trial 196 of jetformer_nconst50_augmentation_none, val/acc_best=0.816286
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=hls4ml/jetformer_50 \
#     data.method=augmentation \
#     data.augmentation_scheme=none \
#     data.n_folds=5 \
#     data.features='[pt,eta,phi]' \
#     algorithm.dropout=0.0 \
#     algorithm.embed_dim=64 \
#     algorithm.num_heads=4 \
#     algorithm.num_transformers=4 \
#     algorithm.optimizer.lr=0.0035646536980357157 \
#     algorithm.optimizer.weight_decay=0.0001 \
#     checkpoint_manifest_path=logs/train/hls4ml/pt_eta_phi/jetformer/nconst50/baseline \
#     results_table.invariance_metric=tv \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/hls4ml/pt_eta_phi/jetformer/nconst50/baseline

# ------------------------------------------------------------------------
# jetformer  nconst=50  aug_so2  (method=augmentation, group=so2)
# best trial 194 of jetformer_nconst50_augmentation_so2, val/acc_best=0.814387
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=hls4ml/jetformer_50 \
#     data.method=augmentation \
#     data.augmentation_scheme=so2 \
#     data.n_folds=5 \
#     data.features='[pt,eta,phi]' \
#     algorithm.dropout=0.0 \
#     algorithm.embed_dim=96 \
#     algorithm.num_heads=4 \
#     algorithm.num_transformers=3 \
#     algorithm.optimizer.lr=0.003922130472399923 \
#     algorithm.optimizer.weight_decay=0.05 \
#     checkpoint_manifest_path=logs/train/hls4ml/pt_eta_phi/jetformer/nconst50/aug_so2 \
#     results_table.invariance_metric=tv \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/hls4ml/pt_eta_phi/jetformer/nconst50/aug_so2

# ------------------------------------------------------------------------
# jetformer  nconst=50  aug_so3  (method=augmentation, group=so3)
# best trial 150 of jetformer_nconst50_augmentation_so3, val/acc_best=0.790725
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=hls4ml/jetformer_50 \
#     data.method=augmentation \
#     data.augmentation_scheme=so3 \
#     data.n_folds=5 \
#     data.features='[pt,eta,phi]' \
#     algorithm.dropout=0.0 \
#     algorithm.embed_dim=64 \
#     algorithm.num_heads=4 \
#     algorithm.num_transformers=4 \
#     algorithm.optimizer.lr=0.00458763597065082 \
#     algorithm.optimizer.weight_decay=0.01 \
#     checkpoint_manifest_path=logs/train/hls4ml/pt_eta_phi/jetformer/nconst50/aug_so3 \
#     results_table.invariance_metric=tv \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/hls4ml/pt_eta_phi/jetformer/nconst50/aug_so3

# ------------------------------------------------------------------------
# jetformer  nconst=50  aug_lorentz  (method=augmentation, group=lorentz)
# best trial 95 of jetformer_nconst50_augmentation_lorentz, val/acc_best=0.815571
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=hls4ml/jetformer_50 \
#     data.method=augmentation \
#     data.augmentation_scheme=lorentz \
#     data.n_folds=5 \
#     data.features='[pt,eta,phi]' \
#     algorithm.dropout=0.0 \
#     algorithm.embed_dim=96 \
#     algorithm.num_heads=2 \
#     algorithm.num_transformers=4 \
#     algorithm.optimizer.lr=0.0045743038023335686 \
#     algorithm.optimizer.weight_decay=0.0001 \
#     checkpoint_manifest_path=logs/train/hls4ml/pt_eta_phi/jetformer/nconst50/aug_lorentz \
#     results_table.invariance_metric=tv \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/hls4ml/pt_eta_phi/jetformer/nconst50/aug_lorentz

# ------------------------------------------------------------------------
# jetformer  nconst=50  symm_so2  (method=symm_loss, group=so2)
# best trial 160 of jetformer_nconst50_symm_loss_so2, val/acc_best=0.817077
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=hls4ml/jetformer_50 \
#     data.method=symm_loss \
#     data.augmentation_scheme=so2 \
#     data.n_folds=5 \
#     data.features='[pt,eta,phi]' \
#     algorithm.dropout=0.0 \
#     algorithm.embed_dim=64 \
#     algorithm.num_heads=4 \
#     algorithm.num_transformers=4 \
#     algorithm.optimizer.lr=0.003224405275524019 \
#     algorithm.optimizer.weight_decay=0.05 \
#     algorithm.symm_loss_weight=0.11071845652819601 \
#     checkpoint_manifest_path=logs/train/hls4ml/pt_eta_phi/jetformer/nconst50/symm_so2 \
#     results_table.invariance_metric=tv \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/hls4ml/pt_eta_phi/jetformer/nconst50/symm_so2

# ------------------------------------------------------------------------
# jetformer  nconst=50  symm_so3  (method=symm_loss, group=so3)
# best trial 103 of jetformer_nconst50_symm_loss_so3, val/acc_best=0.815196
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=hls4ml/jetformer_50 \
#     data.method=symm_loss \
#     data.augmentation_scheme=so3 \
#     data.n_folds=5 \
#     data.features='[pt,eta,phi]' \
#     algorithm.dropout=0.0 \
#     algorithm.embed_dim=96 \
#     algorithm.num_heads=4 \
#     algorithm.num_transformers=4 \
#     algorithm.optimizer.lr=0.0018268182622895358 \
#     algorithm.optimizer.weight_decay=0.05 \
#     algorithm.symm_loss_weight=0.01212164581939032 \
#     checkpoint_manifest_path=logs/train/hls4ml/pt_eta_phi/jetformer/nconst50/symm_so3 \
#     results_table.invariance_metric=tv \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/hls4ml/pt_eta_phi/jetformer/nconst50/symm_so3

# ------------------------------------------------------------------------
# jetformer  nconst=50  symm_lorentz  (method=symm_loss, group=lorentz)
# best trial 133 of jetformer_nconst50_symm_loss_lorentz, val/acc_best=0.816126
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=hls4ml/jetformer_50 \
#     data.method=symm_loss \
#     data.augmentation_scheme=lorentz \
#     data.n_folds=5 \
#     data.features='[pt,eta,phi]' \
#     algorithm.dropout=0.0 \
#     algorithm.embed_dim=96 \
#     algorithm.num_heads=2 \
#     algorithm.num_transformers=4 \
#     algorithm.optimizer.lr=0.004207776462162275 \
#     algorithm.optimizer.weight_decay=0.0001 \
#     algorithm.symm_loss_weight=0.0046403943722153574 \
#     checkpoint_manifest_path=logs/train/hls4ml/pt_eta_phi/jetformer/nconst50/symm_lorentz \
#     results_table.invariance_metric=tv \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/hls4ml/pt_eta_phi/jetformer/nconst50/symm_lorentz

# ========================================================================
# jetformer  nconst=150
# ========================================================================

# ------------------------------------------------------------------------
# jetformer  nconst=150  baseline  (method=augmentation, group=none)
# best trial 105 of jetformer_nconst150_augmentation_none, val/acc_best=0.820247
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=hls4ml/jetformer_150 \
#     data.method=augmentation \
#     data.augmentation_scheme=none \
#     data.n_folds=5 \
#     data.features='[pt,eta,phi]' \
#     algorithm.dropout=0.0 \
#     algorithm.embed_dim=96 \
#     algorithm.num_heads=4 \
#     algorithm.num_transformers=4 \
#     algorithm.optimizer.lr=0.004379168210822643 \
#     algorithm.optimizer.weight_decay=0.01 \
#     checkpoint_manifest_path=logs/train/hls4ml/pt_eta_phi/jetformer/nconst150/baseline \
#     results_table.invariance_metric=tv \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/hls4ml/pt_eta_phi/jetformer/nconst150/baseline

# ------------------------------------------------------------------------
# jetformer  nconst=150  aug_so2  (method=augmentation, group=so2)
# best trial 181 of jetformer_nconst150_augmentation_so2, val/acc_best=0.820567
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=hls4ml/jetformer_150 \
#     data.method=augmentation \
#     data.augmentation_scheme=so2 \
#     data.n_folds=5 \
#     data.features='[pt,eta,phi]' \
#     algorithm.dropout=0.0 \
#     algorithm.embed_dim=96 \
#     algorithm.num_heads=4 \
#     algorithm.num_transformers=4 \
#     algorithm.optimizer.lr=0.004661896762807512 \
#     algorithm.optimizer.weight_decay=0.001 \
#     checkpoint_manifest_path=logs/train/hls4ml/pt_eta_phi/jetformer/nconst150/aug_so2 \
#     results_table.invariance_metric=tv \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/hls4ml/pt_eta_phi/jetformer/nconst150/aug_so2

# ------------------------------------------------------------------------
# jetformer  nconst=150  aug_so3  (method=augmentation, group=so3)
# best trial 134 of jetformer_nconst150_augmentation_so3, val/acc_best=0.793674
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=hls4ml/jetformer_150 \
#     data.method=augmentation \
#     data.augmentation_scheme=so3 \
#     data.n_folds=5 \
#     data.features='[pt,eta,phi]' \
#     algorithm.dropout=0.0 \
#     algorithm.embed_dim=64 \
#     algorithm.num_heads=4 \
#     algorithm.num_transformers=4 \
#     algorithm.optimizer.lr=0.003838866196079134 \
#     algorithm.optimizer.weight_decay=0.0 \
#     checkpoint_manifest_path=logs/train/hls4ml/pt_eta_phi/jetformer/nconst150/aug_so3 \
#     results_table.invariance_metric=tv \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/hls4ml/pt_eta_phi/jetformer/nconst150/aug_so3

# ------------------------------------------------------------------------
# jetformer  nconst=150  aug_lorentz  (method=augmentation, group=lorentz)
# best trial 194 of jetformer_nconst150_augmentation_lorentz, val/acc_best=0.813987
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=hls4ml/jetformer_150 \
#     data.method=augmentation \
#     data.augmentation_scheme=lorentz \
#     data.n_folds=5 \
#     data.features='[pt,eta,phi]' \
#     algorithm.dropout=0.0 \
#     algorithm.embed_dim=96 \
#     algorithm.num_heads=4 \
#     algorithm.num_transformers=4 \
#     algorithm.optimizer.lr=0.0008145029416054058 \
#     algorithm.optimizer.weight_decay=0.001 \
#     checkpoint_manifest_path=logs/train/hls4ml/pt_eta_phi/jetformer/nconst150/aug_lorentz \
#     results_table.invariance_metric=tv \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/hls4ml/pt_eta_phi/jetformer/nconst150/aug_lorentz

# ------------------------------------------------------------------------
# jetformer  nconst=150  symm_so2  (method=symm_loss, group=so2)
# best trial 161 of jetformer_nconst150_symm_loss_so2, val/acc_best=0.816906
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=hls4ml/jetformer_150 \
#     data.method=symm_loss \
#     data.augmentation_scheme=so2 \
#     data.n_folds=5 \
#     data.features='[pt,eta,phi]' \
#     algorithm.dropout=0.0 \
#     algorithm.embed_dim=96 \
#     algorithm.num_heads=4 \
#     algorithm.num_transformers=3 \
#     algorithm.optimizer.lr=0.003106746337333645 \
#     algorithm.optimizer.weight_decay=0.0 \
#     algorithm.symm_loss_weight=0.024135708990978486 \
#     checkpoint_manifest_path=logs/train/hls4ml/pt_eta_phi/jetformer/nconst150/symm_so2 \
#     results_table.invariance_metric=tv \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/hls4ml/pt_eta_phi/jetformer/nconst150/symm_so2

# ------------------------------------------------------------------------
# jetformer  nconst=150  symm_so3  (method=symm_loss, group=so3)
# best trial 160 of jetformer_nconst150_symm_loss_so3, val/acc_best=0.816709
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=hls4ml/jetformer_150 \
#     data.method=symm_loss \
#     data.augmentation_scheme=so3 \
#     data.n_folds=5 \
#     data.features='[pt,eta,phi]' \
#     algorithm.dropout=0.0 \
#     algorithm.embed_dim=96 \
#     algorithm.num_heads=4 \
#     algorithm.num_transformers=4 \
#     algorithm.optimizer.lr=0.0011683200070322715 \
#     algorithm.optimizer.weight_decay=0.05 \
#     algorithm.symm_loss_weight=0.004143026956600342 \
#     checkpoint_manifest_path=logs/train/hls4ml/pt_eta_phi/jetformer/nconst150/symm_so3 \
#     results_table.invariance_metric=tv \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/hls4ml/pt_eta_phi/jetformer/nconst150/symm_so3

# ------------------------------------------------------------------------
# jetformer  nconst=150  symm_lorentz  (method=symm_loss, group=lorentz)
# best trial 153 of jetformer_nconst150_symm_loss_lorentz, val/acc_best=0.820834
# ------------------------------------------------------------------------
# python3 src/eval.py \
#     experiment=hls4ml/jetformer_150 \
#     data.method=symm_loss \
#     data.augmentation_scheme=lorentz \
#     data.n_folds=5 \
#     data.features='[pt,eta,phi]' \
#     algorithm.dropout=0.0 \
#     algorithm.embed_dim=96 \
#     algorithm.num_heads=4 \
#     algorithm.num_transformers=4 \
#     algorithm.optimizer.lr=0.0035472957360764743 \
#     algorithm.optimizer.weight_decay=0.0001 \
#     algorithm.symm_loss_weight=0.013708431844576386 \
#     checkpoint_manifest_path=logs/train/hls4ml/pt_eta_phi/jetformer/nconst150/symm_lorentz \
#     results_table.invariance_metric=tv \
#     trainer=gpu \
#     trainer.devices=[0] \
#     hydra.run.dir=logs/eval/hls4ml/pt_eta_phi/jetformer/nconst150/symm_lorentz
