#!/bin/bash
#SBATCH --job-name=chatty-ft-oracle-r2
#SBATCH --partition=small
#SBATCH --array=0-1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=a.otaifi@lmu.de
#SBATCH --output=/home/a/A.Otaifi/ChattyRun/j1j2/krylov_sign_structure/results/oracle-r2-%A_%a.out
#SBATCH --error=/home/a/A.Otaifi/ChattyRun/j1j2/krylov_sign_structure/results/oracle-r2-%A_%a.err
set -euo pipefail
module purge
module load python/3.12-2024.10
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
BASE=/home/a/A.Otaifi/ChattyRun/j1j2/krylov_sign_structure
MS=(50000 200000)
M=${MS[$SLURM_ARRAY_TASK_ID]}
cd "$BASE/experiments"
python finite_tau_ct_smc_oracle_ratio20.py --M "$M" --beta 3.0 --out "../results/oracle_ratio_M${M}_b3.0.json"
