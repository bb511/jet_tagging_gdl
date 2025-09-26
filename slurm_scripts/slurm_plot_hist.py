import paramiko, time
username = 'sstelea'
server = 'euler.ethz.ch'
ssh_client = paramiko.SSHClient()
ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh_client.connect(server, port=22, username=username, allow_agent=True)
cmds = []
cmds = ['module load stack/2024-06 gcc/12.2.0 python_cuda/3.11.6;']  # Load required modules
cmds.append('cd jet_tagging; \n')
#wrap_str = ' --wrap="python3 hist_aug_test.py"'
wrap_str = ' --wrap="python3 hist_aug_test.py"'
cmds.append('sbatch -n 1 --cpus-per-task=1 --gpus=gpuhe:1 \
    --time=10:00:00 --mem-per-cpu=100G --output=./logs/plot_report.out' + wrap_str + '\n')
ssh_client.exec_command(' '.join(cmds))
time.sleep(20)