import paramiko

# SSH credentials
username = 'sstelea'
server = 'euler.ethz.ch'

# User-defined parameter: total_folds
total_folds = None  # Set to None if no cross-validation is required
run_name = "data_size_scan"

# Lambda values to scan
size_list = [1e4, 1e5, 2e5, 3e5, 4e5, 5e5, 6e5, 0]
#size_list = [1e4, 1e5, 2e5, 3e5, 4e5, 4.96e5, 2*4.96e5, 3*4.96e5, 4*4.96e5, 5*4.96e5, 6*4.96e5, 7*4.96e5, 8*4.96e5, 9*4.96e5]
#lam = 0

# Job configurations
job_configs = [
    # {"name":"ds32", "network": "deepset", "data": "pt_etarel_phirel_32", "loss": "default"},
    # {"name":"ds_vect_32", "network": "deepset", "data": "px_py_pz_etarel_phirel_32", "loss": "default"},
    # {"name":"ds_symlos_32", "network": "deepset", "data": "px_py_pz_etarel_phirel_32", "loss": "so2_eta_phi"},
    # {"name": "ds_aug_32", "network": "deepset", "data": "px_py_pz_etarel_phirel_data_aug_32", "loss": "default"},
    # {"name":"trans_32", "network": "transformer", "data": "pt_etarel_phirel_32", "loss": "default"},
    # {"name": "trans_vect_32", "network": "transformer", "data": "px_py_pz_etarel_phirel_32", "loss": "default"},
    # {"name": "trans_symlos_32", "network": "transformer", "data": "px_py_pz_etarel_phirel_32", "loss": "so2_eta_phi"},
    # {"name":"trans_aug_32", "network": "transformer", "data": "px_py_pz_etarel_phirel_data_aug_32", "loss": "default"},
    {"name":"vn_so2_32", "network": "vector_neuron_so2", "data": "px_py_pz_jet_center_32", "loss": "default"},
]

# Connect to Euler cluster
ssh_client = paramiko.SSHClient()
ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh_client.connect(server, port=22, username=username, allow_agent=True)

job_ids = []  # Store submitted job IDs

for job in job_configs:
    if total_folds is not None:
        raise NotImplementedError("Cross-validation logic not implemented for lambda scan.")
    else:
        for size in size_list:
            path = f'{run_name}/{size}'
            #name = f'{job["network"]}_lambda{lam}'

            cmds = ['module load stack python_cuda/3.11.6;']
            cmds.append('cd jet_tagging; \n')

            log_file = f'./logs_eval/{job["name"]}/{run_name}/{size}.out'

            wrap_str = (
                f' --wrap="python3 test_universal.py '
                f'--name {job["network"]} '
                f'--data_name {job["data"]} '
                f'--loss {job["loss"]} '
                f'--path {path}"'
            )

            cmds.append(f'sbatch -n 1 --cpus-per-task=1 --gpus=gpuhe:1 '
                f'--time=2:00:00 --mem-per-cpu=50G --output={log_file} {wrap_str} \n')

            stdin, stdout, stderr = ssh_client.exec_command(''.join(cmds))
            output = stdout.read().decode().strip()

            if "Submitted batch job" in output:
                job_id = output.split()[-1]
                job_ids.append(job_id)
                print(f"✅ Submitted job: {job['name']} size={size} → outdir={path} (Job ID: {job_id})")
            else:
                print(f"❌ Failed to submit job: {job['name']} λ={size}")
                print(stderr.read().decode())

# Close SSH connection
ssh_client.close()

# Print all submitted job IDs
print("\n📌 Submitted Jobs:", job_ids)

