#!/bin/bash
#SBATCH --job-name=st6up
#SBATCH --partition=cluster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --output=/home/a/A.Otaifi/chatty_stall6/logs/%x-%j.out
# Build the 6x6 symmetric-sector CSR for the stall diagnosis (experiments/stall_6x6).
set -euo pipefail
B=/home/a/A.Otaifi/chatty_stall6
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK NUMBA_NUM_THREADS=$SLURM_CPUS_PER_TASK SLURM_CPUS_PER_TASK
export ED_COMMON_DIR=/home/a/A.Otaifi/chatty_fn6x6/code
/usr/bin/time -v /home/a/A.Otaifi/chatty_ed6x6/ls_env/bin/python -u -X faulthandler $B/code/st6_build_upper.py $B/csr
