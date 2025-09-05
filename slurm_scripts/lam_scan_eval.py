import paramiko

# SSH credentials
username = 'sstelea'
server = 'euler.ethz.ch'

# User-defined parameter: total_folds
total_folds = None  # Set to None if no cross-validation is required
run_name = "lam_scan_so2_compare_to_output_nomask"

# Lambda values to scan
lambda_list = [0, 3e-5, 3e-4, 3e-3, 3e-2, 1e-1, 2e-1, 4e-1, 6e-1, 8e-1, 1]

# Job configurations
job_configs = [
    # {"name":"ds_symlos_8_eval", "network": "deepset", "data": "px_py_pz_etarel_phirel_8", "loss": "so2_eta_phi"},
    # {"name":"ds_symlos_16_eval", "network": "deepset", "data": "px_py_pz_etarel_phirel_16", "loss": "so2_eta_phi"},
    # {"name":"ds_symlos_32_eval", "network": "deepset", "data": "px_py_pz_etarel_phirel_32", "loss": "so2_eta_phi"},
    # {"name":"trans_symlos_8_eval", "network": "transformer", "data": "px_py_pz_etarel_phirel_8", "loss": "so2_eta_phi"},
    # {"name":"trans_symlos_16_eval", "network": "transformer", "data": "px_py_pz_etarel_phirel_16", "loss": "so2_eta_phi"},
    {"name":"trans_symlos_32_eval", "network": "transformer", "data": "px_py_pz_etarel_phirel_32", "loss": "so2_eta_phi"},
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
        for lam in lambda_list:
            path = f'{run_name}/{lam}'
            #name = f'{job["network"]}_lambda{lam}'

            cmds = ['module load stack python_cuda/3.11.6;']
            cmds.append('cd jet_tagging; \n')

            log_file = f'./logs_eval/{job["name"]}/{run_name}/{lam}.out'

            wrap_str = (
                f' --wrap="python3 test_universal.py '
                f'--name {job["network"]} '
                f'--data_name {job["data"]} '
                f'--loss {job["loss"]} '
                f'--path {path}"'
            )

            cmds.append(f'sbatch -n 1 --cpus-per-task=1 --gpus=gpuhe:1 '
                f'--time=10:00 --mem-per-cpu=50G --output={log_file} {wrap_str} \n')

            stdin, stdout, stderr = ssh_client.exec_command(''.join(cmds))
            output = stdout.read().decode().strip()

            if "Submitted batch job" in output:
                job_id = output.split()[-1]
                job_ids.append(job_id)
                print(f"✅ Submitted job: {job['name']} λ={lam} → outdir={path} (Job ID: {job_id})")
            else:
                print(f"❌ Failed to submit job: {job['name']} λ={lam}")
                print(stderr.read().decode())

# Close SSH connection
ssh_client.close()

# Print all submitted job IDs
print("\n📌 Submitted Jobs:", job_ids)
