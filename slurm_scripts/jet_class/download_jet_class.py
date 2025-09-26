import paramiko
import os

username = 'sstelea'
server   = 'euler.ethz.ch'

# List of files to download (add more dicts if needed)
download_jobs = [
    # {   "url": "https://zenodo.org/record/6619768/files/JetClass_Pythia_train_100M_part0.tar",
    #     "target_dir": "/cluster/scratch/sstelea/JetClass",},
    # {   "url": "https://zenodo.org/record/6619768/files/JetClass_Pythia_train_100M_part1.tar",
    #     "target_dir": "/cluster/scratch/sstelea/JetClass",},
    # {   "url": "https://zenodo.org/record/6619768/files/JetClass_Pythia_train_100M_part2.tar",
    #     "target_dir": "/cluster/scratch/sstelea/JetClass",},
    # {   "url": "https://zenodo.org/record/6619768/files/JetClass_Pythia_train_100M_part3.tar",
    #     "target_dir": "/cluster/scratch/sstelea/JetClass",},
    # {   "url": "https://zenodo.org/record/6619768/files/JetClass_Pythia_train_100M_part4.tar",
    #     "target_dir": "/cluster/scratch/sstelea/JetClass",},
    # {   "url": "https://zenodo.org/record/6619768/files/JetClass_Pythia_train_100M_part5.tar",
    #     "target_dir": "/cluster/scratch/sstelea/JetClass",},
    # {   "url": "https://zenodo.org/record/6619768/files/JetClass_Pythia_train_100M_part6.tar",
    #     "target_dir": "/cluster/scratch/sstelea/JetClass",},
    # {   "url": "https://zenodo.org/record/6619768/files/JetClass_Pythia_train_100M_part7.tar",
    #     "target_dir": "/cluster/scratch/sstelea/JetClass",},
    # {   "url": "https://zenodo.org/record/6619768/files/JetClass_Pythia_train_100M_part8.tar",
    #     "target_dir": "/cluster/scratch/sstelea/JetClass",},
    # {   "url": "https://zenodo.org/record/6619768/files/JetClass_Pythia_train_100M_part9.tar",
    #     "target_dir": "/cluster/scratch/sstelea/JetClass",},
    {   "url": "https://zenodo.org/record/6619768/files/JetClass_Pythia_val_5M.tar",
        "target_dir": "/cluster/scratch/sstelea/JetClass",},
    {   "url": "https://zenodo.org/record/6619768/files/JetClass_Pythia_test_20M.tar",
        "target_dir": "/cluster/scratch/sstelea/JetClass",},        
            ]

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect(server, port=22, username=username, allow_agent=True)

job_ids = []

for job in download_jobs:
    url = job["url"]
    fname = os.path.basename(url)
    target_dir = job["target_dir"]
    log_file = f"/cluster/home/sstelea/jet_tagging/logs_download/download_jetclass_{fname}.out"

    cmds = [
        'module load eth_proxy;',
        'export https_proxy=http://proxy.service.consul:3128;',
        'export http_proxy=http://proxy.service.consul:3128;',
        f'mkdir -p {target_dir} && cd {target_dir};',
        f'echo ">>> START {fname} at $(date)";',
        f'wget -c "{url}" 2>&1 && echo "✅ {fname} at $(date)" '
        f'|| {{ echo "❌ {fname} FAILED at $(date)"; exit 1; }};',
        f'echo ">>> END {fname} at $(date)";'
    ]
    wrapped = " ".join(cmds)
    sbatch_cmd = (
        f"sbatch -n 1 --cpus-per-task=1 --time=10:00:00 --mem-per-cpu=30G "
        f"--output={log_file} --wrap='{wrapped}'"
    )

    stdin, stdout, stderr = ssh.exec_command(sbatch_cmd)
    output = stdout.read().decode().strip()

    if "Submitted batch job" in output:
        job_id = output.split()[-1]
        job_ids.append(job_id)
        print(f"✅ Submitted download job for {fname} (Job ID: {job_id})")
    else:
        print(f"❌ Failed to submit download job for {fname}")
        print(stderr.read().decode())

ssh.close()
print("\n📌 Submitted Download Jobs:", job_ids)