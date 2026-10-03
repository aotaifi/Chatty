#!/bin/bash
#SBATCH --job-name=chatty-ft-pathratio
#SBATCH --partition=small
#SBATCH --array=0-3
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=a.otaifi@lmu.de
#SBATCH --output=/home/a/A.Otaifi/ChattyRun/j1j2/krylov_sign_structure/results/pathratio-%A_%a.out
#SBATCH --error=/home/a/A.Otaifi/ChattyRun/j1j2/krylov_sign_structure/results/pathratio-%A_%a.err
set -euo pipefail
module purge
module load python/3.12-2024.10
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
BASE=/home/a/A.Otaifi/ChattyRun/j1j2/krylov_sign_structure
MS=(25000 50000 100000 100000)
BETAS=(2.5 2.5 2.5 3.0)
M=${MS[$SLURM_ARRAY_TASK_ID]}
BETA=${BETAS[$SLURM_ARRAY_TASK_ID]}
cd "$BASE/experiments"
python finite_tau_ct_smc_pathratio_longbeta20.py --M "$M" --beta "$BETA" --out "../results/pathratio_M${M}_b${BETA}.json"
