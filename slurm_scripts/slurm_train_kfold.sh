#!/bin/sh
#BATCH --job-name=equivariance_net_train
#SBATCH --account=gpu_gres
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=5
#SBATCH --mem-per-cpu=8000M
#SBATCH --time=0-23:30
#SBATCH --output=./logs_kfolds/transformer_loss_kfold_8_gpu_%j.out

# Folder where the data is located for the training of the AE.
# Change so it suits your configuration.
DATA_FOLDER=/work/mbonvin/master_thesis_code/

# Default parameters for running the deepsets training.
name=transformer
config_file=./config_equivariant.yml
data_name=px_py_pz_eta_rel_phi_rel_8


# Gather parameters given by user.
for ARGUMENT in "$@"
do
   KEY=$(echo $ARGUMENT | cut -f1 -d=)

   KEY_LENGTH=${#KEY}
   VALUE="${ARGUMENT:$KEY_LENGTH+1}"

   export "$KEY"="$VALUE"
done

# Set up conda environment and cuda.
source /work/mbonvin/miniconda3/bin/activate equiva_env

# Run the script with print flushing instantaneous.
export PYTHONUNBUFFERED=TRUE
nohup python3 training_kfold_transformer.py --name $name --config_file $config_file --data_name $data_name  --kfold 0 &
nohup python3 training_kfold_transformer.py --name $name --config_file $config_file --data_name $data_name  --kfold 1 &
nohup python3 training_kfold_transformer.py --name $name --config_file $config_file --data_name $data_name  --kfold 2 &
nohup python3 training_kfold_transformer.py --name $name --config_file $config_file --data_name $data_name  --kfold 3 &
nohup python3 training_kfold_transformer.py --name $name --config_file $config_file --data_name $data_name  --kfold 4
export PYTHONUNBUFFERED=FALSE

# Move the logs with the rest of the results of the run.
#mv ./logs/deepsets_gpu_${SLURM_JOBID}.out ./trained_deepsets/${outdir}
