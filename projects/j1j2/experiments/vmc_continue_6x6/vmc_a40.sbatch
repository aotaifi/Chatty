#!/bin/bash
#SBATCH --job-name=vmc6
#SBATCH --partition=cip,inter
#SBATCH --gres=gpu:a40:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/home/a/A.Otaifi/chatty_vmc6/logs/%x-%j.out
#SBATCH --error=/home/a/A.Otaifi/chatty_vmc6/logs/%x-%j.err
# Plain-VMC continuation baseline (new files only).  Usage: sbatch --job-name=NAME vmc_a40.sbatch <vmc_continue_6x6.py args>
set -euo pipefail
BASE=/home/a/A.Otaifi/chatty_vmc6
export XLA_PYTHON_CLIENT_PREALLOCATE=false OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
{ echo "job_id=${SLURM_JOB_ID}"; echo "host=$(hostname)"; echo "args=$*"; echo "started=$(date -Is)"; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader; } > "$BASE/logs/meta-${SLURM_JOB_ID}.txt"
"$BASE/../chatty_ll6/venv_gpu/bin/python" "$BASE/code/vmc_continue_6x6.py" "$@"
echo "finished=$(date -Is)" >> "$BASE/logs/meta-${SLURM_JOB_ID}.txt"
