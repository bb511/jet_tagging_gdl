import paramiko
import time

# SSH credentials
username = 'sstelea'
server = 'euler.ethz.ch'

# User-defined parameter: total_folds
total_folds = 5  # Set to None if no cross-validation is required
run_name = "folds_report"
sleep_time = 10

# Connect to Euler cluster
ssh_client = paramiko.SSHClient()
ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh_client.connect(server, port=22, username=username, allow_agent=True)

# ADD BACK DATA AUGMENTATION
# SLURM job settings
job_configs = [
    {"name": "ds_vect_8", "network": "deepset", "data": "px_py_pz_etarel_phirel_8", 
     "config_file": "config_ds_8.yml", "loss": "default"},
    {"name": "ds_vect_16", "network": "deepset", "data": "px_py_pz_etarel_phirel_16", 
     "config_file": "config_ds_16.yml", "loss": "default"},
    # {"name": "ds_aug_32", "network": "deepset", "data": "px_py_pz_etarel_phirel_data_aug_32", 
    #  "config_file": "config_ds_32.yml", "loss": "default"},
]

job_ids = []  # Store submitted job IDs

for job in job_configs:
    if total_folds is not None:  # Cross-validation mode
        for fold in range(total_folds):  # Loop over folds
            cmds = ['module load stack python_cuda/3.11.6;']  # Load required modules
            cmds.append('cd jet_tagging; \n')  # Change directory to the project folder

            # Define SLURM job submission command with fold argument
            wrap_str = (
                f' --wrap="python3 train.py --name {job["network"]} --data_name {job["data"]} '
                f'--config_file /cluster/home/sstelea/jet_tagging/pre_config_files/{job["config_file"]} '
                f'--loss {job["loss"]} --fold {fold}"'
            )
            log_file = f'./logs_train_folds/{job["name"]}/{run_name}/fold_{fold}.out'  # Custom log file for each fold

            cmds.append(
                f'sbatch -n 1 --cpus-per-task=1 --gpus=gpuhe:1 '
                f'--time=30:00:00 --mem-per-cpu=50G --output={log_file} {wrap_str} \n'
            )

            # Execute the command and capture job ID
            stdin, stdout, stderr = ssh_client.exec_command(''.join(cmds))
            output = stdout.read().decode().strip()

            # Extract job ID from SLURM output (e.g., "Submitted batch job 123456")
            if "Submitted batch job" in output:
                job_id = output.split()[-1]
                job_ids.append(job_id)
                print(f"✅ Submitted job: {job['name']} (Fold {fold}) (Job ID: {job_id})")
            else:
                print(f"❌ Failed to submit job: {job['name']} (Fold {fold})")
                print(stderr.read().decode())
                
             # Sleep before submitting the next job
            # print(f"Sleeping for {sleep_time} seconds before submitting the next job...")
            # time.sleep(sleep_time)
            
    else:  # Original functionality (no cross-validation)
        cmds = ['module load stack python_cuda/3.11.6;']  # Load required modules
        cmds.append('cd jet_tagging; \n')  # Change directory to the project folder

        # Define SLURM job submission command without fold argument
        wrap_str = (
            f' --wrap="python3 train.py --name {job["network"]} --data_name {job["data"]} '
            f'--config_file /cluster/home/sstelea/jet_tagging/pre_config_files/{job["config_file"]} '
            f'--loss {job["loss"]}"'
        )
        log_file = f'./logs_train/{job["name"]}/{run_name}.out'  # Custom log file for each model

        cmds.append(
            f'sbatch -n 1 --cpus-per-task=1 --gpus=gpuhe:1 '
            f'--time=24:00:00 --mem-per-cpu=50G --output={log_file} {wrap_str} \n'
        )

        # Execute the command and capture job ID
        stdin, stdout, stderr = ssh_client.exec_command(''.join(cmds))
        output = stdout.read().decode().strip()

        # Extract job ID from SLURM output (e.g., "Submitted batch job 123456")
        if "Submitted batch job" in output:
            job_id = output.split()[-1]
            job_ids.append(job_id)
            print(f"✅ Submitted job: {job['name']} (Job ID: {job_id})")
        else:
            print(f"❌ Failed to submit job: {job['name']}")
            print(stderr.read().decode())
            
         # Sleep before submitting the next job
        # print(f"Sleeping for {sleep_time} seconds before submitting the next job...")
        # time.sleep(sleep_time)

# Close SSH connection
ssh_client.close()

# Print all submitted job IDs
print("\n📌 Submitted Jobs:", job_ids)