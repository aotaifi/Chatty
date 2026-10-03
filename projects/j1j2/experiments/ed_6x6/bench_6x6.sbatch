#!/bin/bash
#SBATCH --job-name=chatty-ed6-bench
#SBATCH --partition=cluster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/home/a/A.Otaifi/chatty_ed6x6/logs/bench-%j.out
set -euo pipefail
cd /home/a/A.Otaifi/chatty_ed6x6
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
echo "host=$(hostname) cpus=${SLURM_CPUS_PER_TASK}"; lscpu | grep "Model name"
/usr/bin/time -v ls_env/bin/python bench_6x6.py 2>&1
