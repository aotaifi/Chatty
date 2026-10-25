#!/bin/bash
#SBATCH --job-name=st6csr
#SBATCH --partition=cluster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G
#SBATCH --time=01:30:00
#SBATCH --output=/project/theorie/a/A.Otaifi/Chatty/runs/stall6/logs/%x-%j.out
# Build the 6x6 symmetric-sector CSR for the stall diagnosis (experiments/stall_6x6).
set -euo pipefail
B=/project/theorie/a/A.Otaifi/Chatty/runs/stall6
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK NUMBA_NUM_THREADS=$SLURM_CPUS_PER_TASK SLURM_CPUS_PER_TASK
export ED_COMMON_DIR=/project/theorie/a/A.Otaifi/Chatty/runs/fn6x6/code
/usr/bin/time -v /project/theorie/a/A.Otaifi/Chatty/runs/ed6x6/ls_env/bin/python $B/code/st6_build_csr.py $B/csr $B/data/psi0_6x6_table.npz $B/data/orb_L6.npz
