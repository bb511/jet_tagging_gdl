#!/bin/bash
# #SBATCH --job-name=model-sweep
# #SBATCH --partition=normal
# #SBATCH --gres=gpu:1
# #SBATCH --ntasks=1
# #SBATCH --cpus-per-task=4
# #SBATCH --mem=64G
# #SBATCH --time=12:00:00
# #SBATCH --array=0-0
# #SBATCH --account=a0166
# #SBATCH --output=logs/slurm/null
# #SBATCH --error=logs/slurm/null

# # ---------------------------------------------------------------
# # Job array: runs method-specific experiments for a given model.
# # Clariden variant of run_euler_array.sh.
# #
# # Usage:
# #   sbatch scripts/run_clariden_array.sh deepsets augmentation
# #   sbatch scripts/run_clariden_array.sh transformer augmentation
# #   sbatch scripts/run_clariden_array.sh deepsets symm_loss
# #   sbatch scripts/run_clariden_array.sh irc_deepsets symm_loss
# #   sbatch scripts/run_clariden_array.sh mlp symm_loss
# #   sbatch scripts/run_clariden_array.sh deepsets permutation       # permutation test
# #   sbatch scripts/run_clariden_array.sh irc_deepsets permutation   # no-augmentation baseline for invariant models
# #   sbatch scripts/run_clariden_array.sh nano_pelican permutation   # Lorentz+perm invariant baseline
# #
# # Task-id mapping:
# #   augmentation:        task 0-3  -> nconst=8, task 4-7  -> nconst=16, task 8-11 -> nconst=32
# #   symm_loss:           task 0-2  -> nconst=8, task 3-5  -> nconst=16, task 6-8  -> nconst=32
# #   permutation (deepsets):         task 0 -> nconst=8, task 1 -> nconst=16, task 2 -> nconst=32
# #   permutation (invariant models): task 0 -> nconst=8, task 1 -> nconst=16, task 2 -> nconst=32
# #   (nano_pelican uses a single config; nconst is set by the experiment YAML)
# # ---------------------------------------------------------------

# set -euo pipefail

# # Model from command-line argument (required).
# MODEL="${1:?Usage: sbatch scripts/run_clariden_array.sh <model_name> <augmentation|symm_loss|permutation>}"
# # Method from command-line argument (required).
# METHOD_RAW="${2:?Usage: sbatch scripts/run_clariden_array.sh <model_name> <augmentation|symm_loss|permutation>}"
# METHOD="$(echo "$METHOD_RAW" | tr '[:upper:]' '[:lower:]')"
# # Any extra Hydra overrides passed after model/method, e.g. trainer.max_epochs=200
# EXTRA_ARGS=("${@:3}")

# resolve_model_flag() {
#     local model="$1"
#     local experiment="$2"

#     if [[ "$model" =="deepsets" ]]; then
#         echo ""
#         return
#     fi

#     if [[ "$model" =="mlp" || "$model" =="irc_deepsets" ]]; then
#         local nconst
#         nconst=$(echo "$experiment" | grep -oP 'nconst\K[0-9]+')
#         if [[ -n "$nconst" ]]; then
#             echo "model=${model}_${nconst}"
#             return
#         fi
#     fi

#     echo "model=$model"
# }

# # --- Experiment sets ---
# AUGMENTATION_EXPERIMENTS=(
#     # augmentation/hls4ml_nconst8_aug_none
#     # augmentation/hls4ml_nconst8_aug_so2
#     # augmentation/hls4ml_nconst8_aug_so3
#     # augmentation/hls4ml_nconst8_aug_lorentz
#     # augmentation/hls4ml_nconst16_aug_none
#     # augmentation/hls4ml_nconst16_aug_so2
#     # augmentation/hls4ml_nconst16_aug_so3
#     # augmentation/hls4ml_nconst16_aug_lorentz
#     # augmentation/hls4ml_nconst32_aug_none
#     # augmentation/hls4ml_nconst32_aug_so2
#     # augmentation/hls4ml_nconst32_aug_so3
#     # augmentation/hls4ml_nconst32_aug_lorentz
# )

# SYMM_LOSS_EXPERIMENTS=(
#     # symm_loss/hls4ml_symm_loss_nconst8_so2
#     # symm_loss/hls4ml_symm_loss_nconst8_so3
#     # symm_loss/hls4ml_symm_loss_nconst8_lorentz
#     # symm_loss/hls4ml_symm_loss_nconst16_so2
#     # symm_loss/hls4ml_symm_loss_nconst16_so3
#     # symm_loss/hls4ml_symm_loss_nconst16_lorentz
#     # symm_loss/hls4ml_symm_loss_nconst32_so2
#     # symm_loss/hls4ml_symm_loss_nconst32_so3
#     symm_loss/hls4ml_symm_loss_nconst32_lorentz
# )

# # permutation_EXPERIMENTS=(
# #     permutation/hls4ml_nconst8_permutation
# #     permutation/hls4ml_nconst16_permutation
# #     permutation/hls4ml_nconst32_permutation
# # )

# # permutation_NONE_EXPERIMENTS=(
# #     permutation/hls4ml_nconst8_none
# #     permutation/hls4ml_nconst16_none
# #     permutation/hls4ml_nconst32_none
# # )

# if [[ "$METHOD" =="augmentation" ]]; then
#     EXPERIMENTS=("${AUGMENTATION_EXPERIMENTS[@]}")
# elif [[ "$METHOD" =="symm_loss" ]]; then
#     EXPERIMENTS=("${SYMM_LOSS_EXPERIMENTS[@]}")
# elif [[ "$METHOD" =="permutation" ]]; then
#     if [[ "$MODEL" =="deepsets" ]]; then
#         EXPERIMENTS=("${permutation_EXPERIMENTS[@]}")
#     elif [[ "$MODEL" =="irc_deepsets" || "$MODEL" =="nano_pelican" ]]; then
#         EXPERIMENTS=("${permutation_NONE_EXPERIMENTS[@]}")
#     fi
# else
#     echo "Unsupported method '$METHOD_RAW'. Use one of: augmentation, symm_loss, permutation"
#     exit 1
# fi

# # If the job array is larger than the chosen method's experiment set, skip extras.
# if (( SLURM_ARRAY_TASK_ID >=${#EXPERIMENTS[@]} )); then
#     echo "Task ${SLURM_ARRAY_TASK_ID} has no experiment for method '${METHOD}'. Skipping."
#     echo "Hint: augmentation has 12 tasks (0-11), symm_loss has 9 (0-8), permutation has 3 (0-2)."
#     exit 0
# fi

# EXPERIMENT="${EXPERIMENTS[$SLURM_ARRAY_TASK_ID]}"

# MODEL_FLAG=$(resolve_model_flag "$MODEL" "$EXPERIMENT")
# RUN_NAME="${EXPERIMENT}_${METHOD}"

# # Extract nconst from experiment string if present
# NCONST=""
# if [[ "$EXPERIMENT" =~ nconst([0-9]+) ]]; then
#     NCONST="${BASH_REMATCH[1]}"
# fi

# # Create slurm log directory
# mkdir -p logs/slurm/${SLURM_ARRAY_JOB_ID}

# # Descriptive log file name: <model>_<experiment_basename>_<taskid>
# EXPERIMENT_BASENAME=$(basename "$EXPERIMENT")
# LOG_BASE="logs/slurm/${SLURM_ARRAY_JOB_ID}/${MODEL}_${EXPERIMENT_BASENAME}_${SLURM_ARRAY_TASK_ID}"
# exec > "${LOG_BASE}.out" 2> "${LOG_BASE}.err"

# # Force a unique Hydra run dir per SLURM array task to avoid checkpoint collisions
# HYDRA_RUN_DIR="logs/train/runs/${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}_${MODEL}_${METHOD}_${EXPERIMENT_BASENAME}"

# echo "============================================"
# echo "Job ID:       $SLURM_ARRAY_JOB_ID"
# echo "Task ID:      $SLURM_ARRAY_TASK_ID"
# echo "Node:         $(hostname)"
# echo "GPU:          ${CUDA_VISIBLE_DEVICES:-unset}"
# echo "Model:        $MODEL"
# echo "Method:       $METHOD"
# echo "Experiment:   $EXPERIMENT"
# echo "Nconst:       $NCONST"
# echo "============================================"

# MANIFEST_PATH="${HYDRA_RUN_DIR}/checkpoint_manifest.json"

# # --- Phase 1: Train ---
# echo "--- Phase 1: Training ---"
# if [[ "$MODEL" =="transformer" && -n "$NCONST" ]]; then
#     srun python src/train.py $MODEL_FLAG experiment="$EXPERIMENT" +experiment_name="$RUN_NAME" trainer=gpu data.data_dir="$HLS4ML_DATA_DIR" data.nconstituents="$NCONST" data.n_folds=5 hydra.run.dir="$HYDRA_RUN_DIR" separate_train_eval_mode=true "${EXTRA_ARGS[@]}"
# else
#     srun python src/train.py $MODEL_FLAG experiment="$EXPERIMENT" +experiment_name="$RUN_NAME" trainer=gpu data.data_dir="$HLS4ML_DATA_DIR" data.n_folds=5 hydra.run.dir="$HYDRA_RUN_DIR" separate_train_eval_mode=true "${EXTRA_ARGS[@]}"
# fi

# # --- Phase 2: Eval ---
# srun python src/eval.py $MODEL_FLAG experiment="$EXPERIMENT" +experiment_name="$RUN_NAME" trainer=gpu data.data_dir="$HLS4ML_DATA_DIR" hydra.run.dir="$HYDRA_RUN_DIR" checkpoint_manifest_path="$MANIFEST_PATH" "${EXTRA_ARGS[@]}"



export HLS4ML_DATA_DIR=/capstor/scratch/cscs/agogia/datasets/

# ==================================================
# BASELINE EXPERIEMENTS FOR ALL MODELS
# ==================================================

# python3 src/train.py \
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


# python3 src/train.py \
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


python3 src/train.py \
    -m \
    hydra/launcher=submitit_slurm_clariden \
    experiment=baseline/hls4ml_nconst32_none \
    +experiment_name=hls4ml_nconst32_aug_none \
    model=deepsets_32 \
    data.nconstituents=32 \
    data.n_folds=5 \
    trainer=gpu \
    data.data_dir="$HLS4ML_DATA_DIR" \
    hydra.sweep.dir="logs/slurm/baseline_deepsets_nconst32" \
    hydra.job.name="baseline_deepsets_nconst32" \
    separate_train_eval_mode=true \
    trainer.max_epochs=200


# ==================================================
# AUGMENTATION EXPERIMENTS FOR ALL MODELS
# ==================================================

# python3 src/train.py \
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


# python3 src/train.py \
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


# python3 src/train.py \
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

# python3 src/train.py \
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

# python3 src/train.py \
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

# python3 src/train.py \
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

# python3 src/train.py \
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