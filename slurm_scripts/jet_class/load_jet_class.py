import paramiko
import time

username = 'sstelea'
server = 'euler.ethz.ch'

print("🔗 Connecting to Euler cluster...")
ssh_client = paramiko.SSHClient()
ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh_client.connect(server, port=22, username=username, allow_agent=True)
print("✅ Connected to Euler.")

# Prepare SLURM command
job_name = "load_jetclass"
log_file = "./logs_download/load_jetclass.out"
script = "python3 data_jetclass_processing/dataloader.py"
wrap_str = f'--wrap="{script}"'
sbatch_cmd = (
    f"sbatch -n 1 --cpus-per-task=1 --gpus=gpuhe:1 "
    f"--time=10:00:00 --mem-per-cpu=50G --output={log_file} {wrap_str}"
)

cmds = [
    'module load stack python_cuda/3.11.6;',
    'cd jet_tagging;',
    sbatch_cmd
]
full_cmd = ' '.join(cmds)

print(f"🚀 Submitting SLURM job for JetClass download:\n{sbatch_cmd}\n")
stdin, stdout, stderr = ssh_client.exec_command(full_cmd)
output = stdout.read().decode().strip()
error = stderr.read().decode().strip()

if "Submitted batch job" in output:
    job_id = output.split()[-1]
    print(f"✅ Submitted job: {job_name} (Job ID: {job_id})")
else:
    print(f"❌ Failed to submit job: {job_name}")
    print("SLURM output:", output)
    print("SLURM error:", error)

ssh_client.close()
print("🔒 SSH connection closed.")