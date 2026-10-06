#!/bin/bash
#SBATCH --job-name=sl-prep
#SBATCH --partition=cluster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G
#SBATCH --time=04:00:00
#SBATCH --output=/home/a/A.Otaifi/chatty_signlearn/logs/%x-%A_%a.out
# Stage A: exact targets + training sets. array index -> N in (16 20 24 28 32 36)
set -euo pipefail
B=/home/a/A.Otaifi/chatty_signlearn
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK NUMBA_NUM_THREADS=$SLURM_CPUS_PER_TASK OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export ED_COMMON_DIR=$B/code
NS=(16 20 24 28 32 36)
N=${NS[$SLURM_ARRAY_TASK_ID]}
/home/a/A.Otaifi/chatty_ed6x6/ls_env/bin/python $B/code/sl_prep.py --N $N --cache-dir /home/a/A.Otaifi/ChattyRun/j1j2/sign_design_tests/cache --out $B/data "$@"
