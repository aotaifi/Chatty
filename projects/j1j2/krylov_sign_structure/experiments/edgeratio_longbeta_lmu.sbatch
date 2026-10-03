#!/bin/bash
#SBATCH --job-name=chatty-ft-edgeratio
#SBATCH --partition=small
#SBATCH --array=0-3
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --mail-type=NONE
#SBATCH --output=/home/a/A.Otaifi/ChattyRun/j1j2/krylov_sign_structure/results/edgeratio-%A_%a.out
#SBATCH --error=/home/a/A.Otaifi/ChattyRun/j1j2/krylov_sign_structure/results/edgeratio-%A_%a.err
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
echo "job=$SLURM_JOB_ID task=$SLURM_ARRAY_TASK_ID M=$M beta=$BETA host=$(hostname) start=$(date -Is)"
if [ "$SLURM_ARRAY_TASK_ID" = "0" ]; then
  "$BASE/experiments/chatty_notify.sh" STARTED "$SLURM_ARRAY_JOB_ID" "edge-ratio array running"
fi
python finite_tau_ct_smc_edgeratio_longbeta20.py --M "$M" --beta "$BETA" --epochs 10 --maxedges 50000 --minc 2 --out "../results/finite_tau_ct_smc_edgeratio_M${M}_b${BETA}.json"
echo "finish=$(date -Is)"
