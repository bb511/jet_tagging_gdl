import paramiko
import time

# SSH credentials
username = 'sstelea'
server = 'euler.ethz.ch'

# Connect to Euler cluster
ssh_client = paramiko.SSHClient()
ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh_client.connect(server, port=22, username=username, allow_agent=True)

folds = True
run_name = "folds_report"

# SLURM job settings
job_configs = [
    # {"name":"ds_symlos_8_eval", "network": "deepset", "data": "px_py_pz_etarel_phirel_8", "loss": "so2_eta_phi", "path": "folds_report"},
    # {"name":"ds_symlos_16_eval", "network": "deepset", "data": "px_py_pz_etarel_phirel_16", "loss": "so2_eta_phi", "path": "folds_report"},
    # {"name":"ds_symlos_32_eval", "network": "deepset", "data": "px_py_pz_etarel_phirel_32", "loss": "so2_eta_phi", "path": "folds_report"},
    # {"name":"ds_aug_8_eval", "network": "deepset", "data": "px_py_pz_etarel_phirel_data_aug_8", "loss": "default", "path": "folds_so3_report"},
    # {"name":"ds_aug_16_eval", "network": "deepset", "data": "px_py_pz_etarel_phirel_data_aug_16", "loss": "default", "path": "folds_so3_report"},
    # {"name":"ds_aug_32_eval", "network": "deepset", "data": "px_py_pz_etarel_phirel_data_aug_32", "loss": "default", "path": "folds_so3_report"},
    {"name":"ds_vect_8_eval", "network": "deepset", "data": "px_py_pz_etarel_phirel_8", "loss": "default", "path": "folds_report"},
    {"name":"ds_vect_16_eval", "network": "deepset", "data": "px_py_pz_etarel_phirel_16", "loss": "default", "path": "folds_report"},
    {"name":"ds_vect_32_eval", "network": "deepset", "data": "px_py_pz_etarel_phirel_32", "loss": "default", "path": "folds_report"},
    # {"name":"ds_8_eval", "network": "deepset", "data": "pt_etarel_phirel_8", "loss": "default", "path": "folds_report"},
    # {"name":"ds_16_eval", "network": "deepset", "data": "pt_etarel_phirel_16", "loss": "default", "path": "folds_report"},
    # {"name":"ds_32_eval", "network": "deepset", "data": "pt_etarel_phirel_32", "loss": "default", "path": "folds_report"},
    # {"name":"trans_aug_8_eval", "network": "transformer", "data": "px_py_pz_etarel_phirel_data_aug_8", "loss": "default", "path": "folds_so3_report"},
    # {"name":"trans_aug_16_eval", "network": "transformer", "data": "px_py_pz_etarel_phirel_data_aug_16", "loss": "default", "path": "folds_so3_report"},
    # {"name":"trans_aug_32_eval", "network": "transformer", "data": "px_py_pz_etarel_phirel_data_aug_32", "loss": "default", "path": "folds_so3_report"},
    # {"name":"trans_symlos_8_eval", "network": "transformer", "data": "px_py_pz_etarel_phirel_8", "loss": "so2_eta_phi", "path": "folds_report"},
    # {"name":"trans_symlos_16_eval", "network": "transformer", "data": "px_py_pz_etarel_phirel_16", "loss": "so2_eta_phi", "path": "folds_report"},
    # {"name":"trans_symlos_32_eval", "network": "transformer", "data": "px_py_pz_etarel_phirel_32", "loss": "so2_eta_phi", "path": "folds_report"},
    # {"name":"trans_vect_8_eval", "network": "transformer", "data": "px_py_pz_etarel_phirel_8", "loss": "default", "path": "folds_report"},
    # {"name":"trans_vect_16_eval", "network": "transformer", "data": "px_py_pz_etarel_phirel_16", "loss": "default", "path": "folds_report"},
    # {"name":"trans_vect_32_eval", "network": "transformer", "data": "px_py_pz_etarel_phirel_32", "loss": "default", "path": "folds_report"},
    # {"name":"trans_8_eval", "network": "transformer", "data": "pt_etarel_phirel_8", "loss": "default", "path": "folds_report"},
    # {"name":"trans_16_eval", "network": "transformer", "data": "pt_etarel_phirel_16", "loss": "default", "path": "folds_report"},
    # {"name":"trans_32_eval", "network": "transformer", "data": "pt_etarel_phirel_32", "loss": "default", "path": "folds_report"},
    # {"name":"vn_8_eval", "network": "vector_neuron", "data": "px_py_pz_jet_center_8", "loss": "default", "path": "break_so3"},
    # {"name":"vn_16_eval", "network": "vector_neuron", "data": "px_py_pz_jet_center_16", "loss": "default", "path": "break_so3"},
    # {"name":"vn_32_eval", "network": "vector_neuron", "data": "px_py_pz_jet_center_32", "loss": "default", "path": "break_so3"},
    # {"name":"vn_so2_8_eval", "network": "vector_neuron_so2", "data": "px_py_pz_jet_center_8", "loss": "default", "path": "so2_inv"},
    # {"name":"vn_so2_16_eval", "network": "vector_neuron_so2", "data": "px_py_pz_jet_center_16", "loss": "default", "path": "so2_inv"},
    # {"name":"vn_so2_32_eval", "network": "vector_neuron_so2", "data": "px_py_pz_jet_center_32", "loss": "default", "path": "so2_inv"},
]

job_ids = []  # Store submitted job IDs

for job in job_configs:
    cmds = ['module load stack python_cuda/3.11.6;']  # Load required modules
    cmds.append('cd jet_tagging; \n')  # Change directory to the project folder
    
    # Define SLURM job submission command
    wrap_str = f' --wrap="python3 test_universal.py --name {job["network"]} --data_name {job["data"]} --loss {job["loss"]} --path {job["path"]} --folds {folds}"'
    log_file = f'./logs_eval/{job["name"]}/{run_name}.out'  # 🔥 Custom log file for each model

    # sbatch_command = (
    #     f'sbatch -n 1 --cpus-per-task=1 --gpus=gpuhe:1 '
    #     f'--time=2:00:00 --mem-per-cpu=50G --output=./logs_eval/run_ds_symlos%j.out {wrap_str}'
    # )
    
    cmds.append(f'sbatch -n 1 --cpus-per-task=1 --gpus=gpuhe:1 '
                f'--time=10:00 --mem-per-cpu=50G --output={log_file} {wrap_str} \n')

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

# Close SSH connection
ssh_client.close()

# Print all submitted job IDs
print("\n📌 Submitted Jobs:", job_ids)
