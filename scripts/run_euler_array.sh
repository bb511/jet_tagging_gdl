#!/bin/bash
export HLS4ML_DATA_DIR=/cluster/scratch/agogia/datasets/
# export PYTHONPATH=/cluster/home/agogia/jet_tagging_gdl:$PYTHONPATH

# ==================================================
# BASELINE EXPERIMENTS FOR ALL MODELS
# ==================================================

# python src/train.py \
#     -m \
#     hydra/launcher=submitit_slurm_euler \
#     experiment=baseline/hls4ml_nconst8_none \
#     +experiment_name=hls4ml_nconst8_aug_none \
#     model=jetformer \
#     model.nbr_feat=3 \
#     data.features="[pt,etarel,phirel]" \
#     data.nconstituents=8 \
#     data.n_folds=5 \
#     trainer=gpu \
#     data.data_dir="$HLS4ML_DATA_DIR" \
#     hydra.sweep.dir="logs/slurm/baseline_jetformer_nconst8" \
#     hydra.job.name="baseline_jetformer_nconst8" \
#     separate_train_eval_mode=true \
#     trainer.max_epochs=200 & 


# python src/train.py \
#     -m \
#     hydra/launcher=submitit_slurm_euler \
#     experiment=baseline/hls4ml_nconst16_none \
#     +experiment_name=hls4ml_nconst16_aug_none \
#     model=jetformer \
#     model.nbr_feat=3 \
#     data.features="[pt,etarel,phirel]" \
#     data.nconstituents=16 \
#     data.n_folds=5 \
#     trainer=gpu \
#     data.data_dir="$HLS4ML_DATA_DIR" \
#     hydra.sweep.dir="logs/slurm/baseline_jetformer_nconst16" \
#     hydra.job.name="baseline_jetformer_nconst16" \
#     separate_train_eval_mode=true \
#     trainer.max_epochs=200 & 


# python src/train.py \
#     -m \
#     hydra/launcher=submitit_slurm_euler \
#     experiment=baseline/hls4ml_nconst32_none \
#     +experiment_name=hls4ml_nconst32_aug_none \
#     model=jetformer \
#     model.nbr_feat=3 \
#     data.features="[pt,etarel,phirel]" \
#     data.nconstituents=32 \
#     data.n_folds=5 \
#     trainer=gpu \
#     data.data_dir="$HLS4ML_DATA_DIR" \
#     hydra.sweep.dir="logs/slurm/baseline_jetformer_nconst32" \
#     hydra.job.name="baseline_jetformer_nconst32" \
#     separate_train_eval_mode=true \
#     trainer.max_epochs=200 & 


# ==================================================
# AUGMENTATION EXPERIMENTS FOR ALL MODELS
# ==================================================

# python src/train.py \
#      -m \
#      hydra/launcher=submitit_slurm_euler \
#      experiment=augmentation/hls4ml_nconst8_aug_so2 \
#      +experiment_name=hls4ml_nconst8_aug_so2 \
#      model=jetformer \
#      model.nbr_feat=3 \
#      data.features="[pt,etarel,phirel]" \
#      data.nconstituents=8 \
#      data.n_folds=5 \
#      trainer=gpu \
#      data.data_dir="$HLS4ML_DATA_DIR" \
#      hydra.sweep.dir="logs/slurm/aug_so2_jetformer_nconst8" \
#      hydra.job.name="aug_so2_jetformer_nconst8" \
#      separate_train_eval_mode=true \
#      trainer.max_epochs=200 &


# python src/train.py \
#      -m \
#      hydra/launcher=submitit_slurm_euler \
#      experiment=augmentation/hls4ml_nconst16_aug_so2 \
#      +experiment_name=hls4ml_nconst16_aug_so2 \
#      model=jetformer \
#      model.nbr_feat=3 \
#      data.features="[pt,etarel,phirel]" \
#      data.nconstituents=16 \
#      data.n_folds=5 \
#      trainer=gpu \
#      data.data_dir="$HLS4ML_DATA_DIR" \
#      hydra.sweep.dir="logs/slurm/aug_so2_jetformer_nconst16" \
#      hydra.job.name="aug_so2_jetformer_nconst16" \
#      separate_train_eval_mode=true \
#      trainer.max_epochs=200 &

# python src/train.py \
#     -m \
#     hydra/launcher=submitit_slurm_euler \
#     experiment=augmentation/hls4ml_nconst32_aug_so2 \
#     +experiment_name=hls4ml_nconst32_aug_so2 \
#     model=jetformer \
#     model.nbr_feat=3 \
#     data.features="[pt,etarel,phirel]" \
#     data.nconstituents=32 \
#     data.n_folds=5 \
#     trainer=gpu \
#     data.data_dir="$HLS4ML_DATA_DIR" \
#     hydra.sweep.dir="logs/slurm/aug_so2_jetformer_nconst32" \
#     hydra.job.name="aug_so2_jetformer_nconst32" \
#     separate_train_eval_mode=true \
#     trainer.max_epochs=200 &







# python src/train.py \
#      -m \
#      hydra/launcher=submitit_slurm_euler \
#      experiment=augmentation/hls4ml_nconst8_aug_so3 \
#      +experiment_name=hls4ml_nconst8_aug_so3 \
#      model=jetformer \
#      model.nbr_feat=3 \
#      data.features="[pt,etarel,phirel]" \
#      data.nconstituents=8 \
#      data.n_folds=5 \
#      trainer=gpu \
#      data.data_dir="$HLS4ML_DATA_DIR" \
#      hydra.sweep.dir="logs/slurm/aug_so3_jetformer_nconst8" \
#      hydra.job.name="aug_so3_jetformer_nconst8" \
#      separate_train_eval_mode=true \
#      trainer.max_epochs=200 &


# python src/train.py \
#      -m \
#      hydra/launcher=submitit_slurm_euler \
#      experiment=augmentation/hls4ml_nconst16_aug_so3 \
#      +experiment_name=hls4ml_nconst16_aug_so3 \
#      model=jetformer \
#      model.nbr_feat=3 \
#      data.features="[pt,etarel,phirel]" \
#      data.nconstituents=16 \
#      data.n_folds=5 \
#      trainer=gpu \
#      data.data_dir="$HLS4ML_DATA_DIR" \
#      hydra.sweep.dir="logs/slurm/aug_so3_jetformer_nconst16" \
#      hydra.job.name="aug_so3_jetformer_nconst16" \
#      separate_train_eval_mode=true \
#      trainer.max_epochs=200 &

# python src/train.py \
#     -m \
#     hydra/launcher=submitit_slurm_euler \
#     experiment=augmentation/hls4ml_nconst32_aug_so3 \
#     +experiment_name=hls4ml_nconst32_aug_so3 \
#     model=jetformer \
#     model.nbr_feat=3 \
#     data.features="[pt,etarel,phirel]" \
#     data.nconstituents=32 \
#     data.n_folds=5 \
#     trainer=gpu \
#     data.data_dir="$HLS4ML_DATA_DIR" \
#     hydra.sweep.dir="logs/slurm/aug_so3_jetformer_nconst32" \
#     hydra.job.name="aug_so3_jetformer_nconst32" \
#     separate_train_eval_mode=true \
#     trainer.max_epochs=200 &





# python src/train.py \
#      -m \
#      hydra/launcher=submitit_slurm_euler \
#      experiment=augmentation/hls4ml_nconst8_aug_lorentz \
#      +experiment_name=hls4ml_nconst8_aug_lorentz \
#      model=jetformer \
#      model.nbr_feat=3 \
#      data.features="[pt,etarel,phirel]" \
#      data.nconstituents=8 \
#      data.n_folds=5 \
#      trainer=gpu \
#      data.data_dir="$HLS4ML_DATA_DIR" \
#      hydra.sweep.dir="logs/slurm/aug_lorentz_jetformer_nconst8" \
#      hydra.job.name="aug_lorentz_jetformer_nconst8" \
#      separate_train_eval_mode=true \
#      trainer.max_epochs=200 &


# python src/train.py \
#      -m \
#      hydra/launcher=submitit_slurm_euler \
#      experiment=augmentation/hls4ml_nconst16_aug_lorentz \
#      +experiment_name=hls4ml_nconst16_aug_lorentz \
#      model=jetformer \
#      model.nbr_feat=3 \
#      data.features="[pt,etarel,phirel]" \
#      data.nconstituents=16 \
#      data.n_folds=5 \
#      trainer=gpu \
#      data.data_dir="$HLS4ML_DATA_DIR" \
#      hydra.sweep.dir="logs/slurm/aug_lorentz_jetformer_nconst16" \
#      hydra.job.name="aug_lorentz_jetformer_nconst16" \
#      separate_train_eval_mode=true \
#      trainer.max_epochs=200 &

# python src/train.py \
#     -m \
#     hydra/launcher=submitit_slurm_euler \
#     experiment=augmentation/hls4ml_nconst32_aug_lorentz \
#     +experiment_name=hls4ml_nconst32_aug_lorentz \
#     model=jetformer \
#     model.nbr_feat=3 \
#     data.features="[pt,etarel,phirel]" \
#     data.nconstituents=32 \
#     data.n_folds=5 \
#     trainer=gpu \
#     data.data_dir="$HLS4ML_DATA_DIR" \
#     hydra.sweep.dir="logs/slurm/aug_lorentz_jetformer_nconst32" \
#     hydra.job.name="aug_lorentz_jetformer_nconst32" \
#     separate_train_eval_mode=true \
#     trainer.max_epochs=200 &


# ==================================================
# SYMM_LOSS EXPERIMENTS FOR ALL MODELS
# ==================================================

# python src/train.py \
#     -m \
#     hydra/launcher=submitit_slurm_euler \
#     experiment=symm_loss/hls4ml_symm_loss_nconst8_so2 \
#     +experiment_name=hls4ml_symm_loss_nconst8_so2 \
#     model=jetformer \
#     model.nbr_feat=3 \
#     data.features="[pt,etarel,phirel]" \
#     data.nconstituents=8 \
#     data.n_folds=5 \
#     trainer=gpu \
#     data.data_dir="$HLS4ML_DATA_DIR" \
#     hydra.sweep.dir="logs/slurm/symm_loss_so2_jetformer_nconst8" \
#     hydra.job.name="symm_loss_so2_jetformer_nconst8" \
#     separate_train_eval_mode=true \
#     trainer.max_epochs=200 &

# python src/train.py \
#     -m \
#     hydra/launcher=submitit_slurm_euler \
#     experiment=symm_loss/hls4ml_symm_loss_nconst16_so2 \
#     +experiment_name=hls4ml_symm_loss_nconst16_so2 \
#     model=jetformer \
#     model.nbr_feat=3 \
#     data.features="[pt,etarel,phirel]" \
#     data.nconstituents=16 \
#     data.n_folds=5 \
#     trainer=gpu \
#     data.data_dir="$HLS4ML_DATA_DIR" \
#     hydra.sweep.dir="logs/slurm/symm_loss_so2_jetformer_nconst16" \
#     hydra.job.name="symm_loss_so2_jetformer_nconst16" \
#     separate_train_eval_mode=true \
#     trainer.max_epochs=200 &

# python src/train.py \
#     -m \
#     hydra/launcher=submitit_slurm_euler \
#     experiment=symm_loss/hls4ml_symm_loss_nconst32_so2 \
#     +experiment_name=hls4ml_symm_loss_nconst32_so2 \
#     model=jetformer \
#     model.nbr_feat=3 \
#     data.features="[pt,etarel,phirel]" \
#     data.nconstituents=32 \
#     data.n_folds=5 \
#     trainer=gpu \
#     data.data_dir="$HLS4ML_DATA_DIR" \
#     hydra.sweep.dir="logs/slurm/symm_loss_so2_jetformer_nconst32" \
#     hydra.job.name="symm_loss_so2_jetformer_nconst32" \
#     separate_train_eval_mode=true \
#     trainer.max_epochs=200 & 






# python src/train.py \
#     -m \
#     hydra/launcher=submitit_slurm_euler \
#     experiment=symm_loss/hls4ml_symm_loss_nconst8_so3 \
#     +experiment_name=hls4ml_symm_loss_nconst8_so3 \
#     model=jetformer \
#     model.nbr_feat=3 \
#     data.features="[pt,etarel,phirel]" \
#     data.nconstituents=8 \
#     data.n_folds=5 \
#     trainer=gpu \
#     data.data_dir="$HLS4ML_DATA_DIR" \
#     hydra.sweep.dir="logs/slurm/symm_loss_so3_jetformer_nconst8" \
#     hydra.job.name="symm_loss_so3_jetformer_nconst8" \
#     separate_train_eval_mode=true \
#     trainer.max_epochs=200 &

# python src/train.py \
#     -m \
#     hydra/launcher=submitit_slurm_euler \
#     experiment=symm_loss/hls4ml_symm_loss_nconst16_so3 \
#     +experiment_name=hls4ml_symm_loss_nconst16_so3 \
#     model=jetformer \
#     model.nbr_feat=3 \
#     data.features="[pt,etarel,phirel]" \
#     data.nconstituents=16 \
#     data.n_folds=5 \
#     trainer=gpu \
#     data.data_dir="$HLS4ML_DATA_DIR" \
#     hydra.sweep.dir="logs/slurm/symm_loss_so3_jetformer_nconst16" \
#     hydra.job.name="symm_loss_so3_jetformer_nconst16" \
#     separate_train_eval_mode=true \
#     trainer.max_epochs=200 &

# python src/train.py \
#     -m \
#     hydra/launcher=submitit_slurm_euler \
#     experiment=symm_loss/hls4ml_symm_loss_nconst32_so3 \
#     +experiment_name=hls4ml_symm_loss_nconst32_so3 \
#     model=jetformer \
#     model.nbr_feat=3 \
#     data.features="[pt,etarel,phirel]" \
#     data.nconstituents=32 \
#     data.n_folds=5 \
#     trainer=gpu \
#     data.data_dir="$HLS4ML_DATA_DIR" \
#     hydra.sweep.dir="logs/slurm/symm_loss_so3_jetformer_nconst32" \
#     hydra.job.name="symm_loss_so3_jetformer_nconst32" \
#     separate_train_eval_mode=true \
#     trainer.max_epochs=200 & 




    


# python src/train.py \
#     -m \
#     hydra/launcher=submitit_slurm_euler \
#     experiment=symm_loss/hls4ml_symm_loss_nconst8_lorentz \
#     +experiment_name=hls4ml_symm_loss_nconst8_lorentz \
#     model=jetformer \
#     model.nbr_feat=3 \
#     data.features="[pt,etarel,phirel]" \
#     data.nconstituents=8 \
#     data.n_folds=5 \
#     trainer=gpu \
#     data.data_dir="$HLS4ML_DATA_DIR" \
#     hydra.sweep.dir="logs/slurm/symm_loss_lorentz_jetformer_nconst8" \
#     hydra.job.name="symm_loss_lorentz_jetformer_nconst8" \
#     separate_train_eval_mode=true \
#     trainer.max_epochs=200 &

# python src/train.py \
#     -m \
#     hydra/launcher=submitit_slurm_euler \
#     experiment=symm_loss/hls4ml_symm_loss_nconst16_lorentz \
#     +experiment_name=hls4ml_symm_loss_nconst16_lorentz \
#     model=jetformer \
#     model.nbr_feat=3 \
#     data.features="[pt,etarel,phirel]" \
#     data.nconstituents=16 \
#     data.n_folds=5 \
#     trainer=gpu \
#     data.data_dir="$HLS4ML_DATA_DIR" \
#     hydra.sweep.dir="logs/slurm/symm_loss_lorentz_jetformer_nconst16" \
#     hydra.job.name="symm_loss_lorentz_jetformer_nconst16" \
#     separate_train_eval_mode=true \
#     trainer.max_epochs=200 &

python src/train.py \
    -m \
    hydra/launcher=submitit_slurm_euler \
    experiment=symm_loss/hls4ml_symm_loss_nconst32_lorentz \
    +experiment_name=hls4ml_symm_loss_nconst32_lorentz \
    model=jetformer \
    model.nbr_feat=3 \
    data.features="[pt,etarel,phirel]" \
    data.nconstituents=32 \
    data.n_folds=5 \
    trainer=gpu \
    data.data_dir="$HLS4ML_DATA_DIR" \
    hydra.sweep.dir="logs/slurm/symm_loss_lorentz_jetformer_nconst32" \
    hydra.job.name="symm_loss_lorentz_jetformer_nconst32" \
    separate_train_eval_mode=true \
    trainer.max_epochs=200 & 


# ==================================================
# PERMUTATION EXPERIMENTS FOR DEEPSETS
# ==================================================

# python src/train.py \
#     -m \
#     hydra/launcher=submitit_slurm_euler \
#     experiment=permutation/hls4ml_nconst32_permutation \
#     +experiment_name=hls4ml_nconst32_permutation \
#     model=deepsets_32 \
#     data.nconstituents=32 \
#     data.n_folds=5 \
#     trainer=gpu \
#     data.data_dir="$HLS4ML_DATA_DIR" \
#     hydra.sweep.dir="logs/slurm/permutation_deepsets_nconst32" \
#     hydra.job.name="permutation_deepsets_nconst32" \
#     separate_train_eval_mode=true \
#     trainer.max_epochs=200



# ==================================================
# HYPERPARAMETER SEARCH FOR NANO-PELICAN
# # ================================================
# python src/train.py \
#     -m \
#     hydra/launcher=submitit_slurm_euler \
#     experiment=baseline/hls4ml_nconst32_none \
#     +experiment_name=hls4ml_nconst32_aug_none \
#     model=nano_pelican \
#     data.nconstituents=32 \
#     data.n_folds=0 \
#     trainer=gpu \
#     data.data_dir="$HLS4ML_DATA_DIR" \
#     ~logger \
#     ~callbacks \
#     hydra.sweep.dir="logs/slurm/hyperparameter_search_nano_pelican" \
#     hydra.job.name="hyperparameter_search_nano_pelican" \
#     trainer.max_epochs=50 \
#     hparams_search=nano_pelican_optuna \
#     hydra.sweeper.n_trials=200 \
#     hydra.sweeper.direction="maximize"
