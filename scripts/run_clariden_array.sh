export HLS4ML_DATA_DIR=/capstor/scratch/cscs/agogia/datasets/

# ==================================================
# BASELINE EXPERIEMENTS FOR ALL MODELS
# ==================================================

# python src/train.py \
#     -m \
#     hydra/launcher=submitit_slurm_clariden \
#     experiment=baseline/hls4ml_nconst8_none \
#     +experiment_name=hls4ml_nconst8_aug_none \
#     model=transformer \
#     data.nconstituents=8 \
#     data.n_folds=5 \
#     trainer=gpu \
#     data.data_dir="$HLS4ML_DATA_DIR" \
#     hydra.sweep.dir="logs/slurm/baseline_transformer_nconst8" \
#     hydra.job.name="baseline_transformer_nconst8" \
#     separate_train_eval_mode=true \
#     trainer.max_epochs=200


# python src/train.py \
#     -m \
#     hydra/launcher=submitit_slurm_clariden \
#     experiment=baseline/hls4ml_nconst16_none \
#     +experiment_name=hls4ml_nconst16_aug_none \
#     model=transformer \
#     data.nconstituents=16 \
#     data.n_folds=5 \
#     trainer=gpu \
#     data.data_dir="$HLS4ML_DATA_DIR" \
#     hydra.sweep.dir="logs/slurm/baseline_transformer_nconst16" \
#     hydra.job.name="baseline_transformer_nconst16" \
#     separate_train_eval_mode=true \
#     trainer.max_epochs=200


# python src/train.py \
#     -m \
#     hydra/launcher=submitit_slurm_clariden \
#     experiment=baseline/hls4ml_nconst32_none \
#     +experiment_name=hls4ml_nconst32_aug_none \
#     model=deepsets_32 \
#     data.nconstituents=32 \
#     data.n_folds=5 \
#     trainer=gpu \
#     data.data_dir="$HLS4ML_DATA_DIR" \
#     hydra.sweep.dir="logs/slurm/baseline_deepsets_nconst32" \
#     hydra.job.name="baseline_deepsets_nconst32" \
#     separate_train_eval_mode=true \
#     trainer.max_epochs=200


# ==================================================
# AUGMENTATION EXPERIMENTS FOR ALL MODELS
# ==================================================

# python src/train.py \
#     -m \
#     hydra/launcher=submitit_slurm_clariden \
#     experiment=augmentation/hls4ml_nconst8_aug_so2 \
#     +experiment_name=hls4ml_nconst8_aug_so2 \
#     model=deepsets_8 \
#     data.nconstituents=8 \
#     data.n_folds=5 \
#     trainer=gpu \
#     data.data_dir="$HLS4ML_DATA_DIR" \
#     hydra.sweep.dir="logs/slurm/aug_so2_deepsets_nconst8" \
#     hydra.job.name="augmentation_deepsets_nconst8_so2" \
#     separate_train_eval_mode=true \
#     trainer.max_epochs=200


# python src/train.py \
#     -m \
#     hydra/launcher=submitit_slurm_clariden \
#     experiment=augmentation/hls4ml_nconst16_aug_so2 \
#     +experiment_name=hls4ml_nconst16_aug_so2 \
#     model=deepsets_16 \
#     data.nconstituents=16 \
#     data.n_folds=5 \
#     trainer=gpu \
#     data.data_dir="$HLS4ML_DATA_DIR" \
#     hydra.sweep.dir="logs/slurm/aug_so2_deepsets_nconst16" \
#     hydra.job.name="aug_so2_deepsets_nconst16_so2" \
#     separate_train_eval_mode=true \
#     trainer.max_epochs=200


# python src/train.py \
#     -m \
#     hydra/launcher=submitit_slurm_clariden \
#     experiment=augmentation/hls4ml_nconst32_aug_so2 \
#     +experiment_name=hls4ml_nconst32_aug_so2 \
#     model=deepsets_32 \
#     data.nconstituents=32 \
#     data.n_folds=5 \
#     trainer=gpu \
#     data.data_dir="$HLS4ML_DATA_DIR" \
#     hydra.sweep.dir="logs/slurm/aug_so2_deepsets_nconst32" \
#     hydra.job.name="aug_so2_deepsets_nconst32_so2" \
#     separate_train_eval_mode=true \
#     trainer.max_epochs=200



# # ==================================================
# # SYMM_LOSS EXPERIMENTS FOR ALL MODELS
# # ==================================================

# python src/train.py \
#     -m \
#     hydra/launcher=submitit_slurm_clariden \
#     experiment=symm_loss/hls4ml_symm_loss_nconst8_so2 \
#     +experiment_name=hls4ml_symm_loss_nconst8_so2 \
#     model=deepsets_8 \
#     data.nconstituents=8 \
#     data.n_folds=5 \
#     trainer=gpu \
#     data.data_dir="$HLS4ML_DATA_DIR" \
#     hydra.sweep.dir="logs/slurm/symm_loss_deepsets_nconst8" \
#     hydra.job.name="symm_loss_deepsets_nconst8" \
#     separate_train_eval_mode=true \
#     trainer.max_epochs=200

# python src/train.py \
#     -m \
#     hydra/launcher=submitit_slurm_clariden \
#     experiment=symm_loss/hls4ml_symm_loss_nconst16_so2 \
#     +experiment_name=hls4ml_symm_loss_nconst16_so2 \
#     model=deepsets_16 \
#     data.nconstituents=16 \
#     data.n_folds=5 \
#     trainer=gpu \
#     data.data_dir="$HLS4ML_DATA_DIR" \
#     hydra.sweep.dir="logs/slurm/symm_loss_deepsets_nconst16" \
#     hydra.job.name="symm_loss_deepsets_nconst16" \
#     separate_train_eval_mode=true \
#     trainer.max_epochs=200

# python src/train.py \
#     -m \
#     hydra/launcher=submitit_slurm_clariden \
#     experiment=symm_loss/hls4ml_symm_loss_nconst32_so2 \
#     +experiment_name=hls4ml_symm_loss_nconst32_so2 \
#     model=deepsets_32 \
#     data.nconstituents=32 \
#     data.n_folds=5 \
#     trainer=gpu \
#     data.data_dir="$HLS4ML_DATA_DIR" \
#     hydra.sweep.dir="logs/slurm/symm_loss_deepsets_nconst32" \
#     hydra.job.name="symm_loss_deepsets_nconst32" \
#     separate_train_eval_mode=true \
#     trainer.max_epochs=200


# # ==================================================
# # PERMUTATION EXPERIMENTS FOR DEEPSETS
# # ==================================================

# python src/train.py \
#     -m \
#     hydra/launcher=submitit_slurm_clariden \
#     experiment=permutation/hls4ml_nconst8_permutation \
#     +experiment_name=hls4ml_nconst8_permutation \
#     model=deepsets_8 \
#     data.nconstituents=8 \
#     data.n_folds=5 \
#     trainer=gpu \
#     data.data_dir="$HLS4ML_DATA_DIR" \
#     hydra.sweep.dir="logs/slurm/permutation_deepsets_nconst8" \
#     hydra.job.name="permutation_deepsets_nconst8" \
#     separate_train_eval_mode=true \
#     trainer.max_epochs=200