#!/bin/sh
#SBATCH --job-name=equivariance_net_train
#SBATCH --account=gpu_gres
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=5
#SBATCH --mem-per-cpu=8000M
#SBATCH --time=1-23:30
#SBATCH --output=./logs/transformer_px_py_pz_jet_center_32_gpu_%j.out

# Folder where the data is located for the training of the AE.
# Change so it suits your configuration.
DATA_FOLDER=/work/mbonvin/to_export/

# Default parameters for running the deepsets training.
name=transformer
config_file=./config_equivariant.yml
data_name=px_py_pz_jet_center_32

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
python3 train_transformer.py --name $name --config_file $config_file --data_name $data_name
export PYTHONUNBUFFERED=FALSE

# Move the logs with the rest of the results of the run.
#mv ./logs/deepsets_gpu_${SLURM_JOBID}.out ./trained_deepsets/${outdir}
