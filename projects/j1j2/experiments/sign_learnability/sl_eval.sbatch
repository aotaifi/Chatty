#!/bin/bash
#SBATCH --job-name=sl-eval
#SBATCH --partition=cluster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G
#SBATCH --time=08:00:00
#SBATCH --output=/home/a/A.Otaifi/chatty_signlearn/logs/%x-%A_%a.out
# Stage C: exact scoring (+ exact Krylov hops) of all trained nets of one N. array index -> N; extra args passed on
set -euo pipefail
B=/home/a/A.Otaifi/chatty_signlearn
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK NUMBA_NUM_THREADS=$SLURM_CPUS_PER_TASK OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export ED_COMMON_DIR=$B/code
NS=(16 20 24 28 32 36)
N=${NS[$SLURM_ARRAY_TASK_ID]}
/home/a/A.Otaifi/chatty_ed6x6/ls_env/bin/python $B/code/sl_eval.py --N $N "$@"
