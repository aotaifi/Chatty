#!/bin/bash
#SBATCH --job-name=sl-train
#SBATCH --partition=inter
#SBATCH --gres=gpu:rtx2080ti:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=/home/a/A.Otaifi/chatty_signlearn/logs/%x-%j.out
# Stage B: train + evaluate sign nets. Usage: sbatch [--partition=cip,inter --gres=gpu:a40:1] sl_train.sbatch TASKFILE LO HI
set -euo pipefail
B=/home/a/A.Otaifi/chatty_signlearn
export XLA_PYTHON_CLIENT_PREALLOCATE=false OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
echo "job=$SLURM_JOB_ID host=$(hostname) $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
/home/a/A.Otaifi/chatty_ll6/venv_gpu/bin/python $B/code/sl_train.py --tasks "$1" --lo "$2" --hi "$3"
