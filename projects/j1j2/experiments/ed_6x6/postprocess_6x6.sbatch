#!/bin/bash
#SBATCH --job-name=chatty-ed6post
#SBATCH --partition=cluster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/home/a/A.Otaifi/chatty_ed6x6/logs/post-%j.out
set -euo pipefail
cd /home/a/A.Otaifi/chatty_ed6x6
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
echo "host=$(hostname)"
/usr/bin/time -v ls_env/bin/python postprocess_6x6.py "$@"
