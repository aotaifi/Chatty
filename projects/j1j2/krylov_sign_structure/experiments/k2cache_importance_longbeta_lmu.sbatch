#!/bin/bash
#SBATCH --job-name=chatty-k2imp-long
#SBATCH --partition=small
#SBATCH --array=0-3
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=02:00:00
#SBATCH --mail-type=NONE
#SBATCH --output=/home/a/A.Otaifi/ChattyRun/j1j2/krylov_sign_structure/results/k2implong-%A_%a.out
#SBATCH --error=/home/a/A.Otaifi/ChattyRun/j1j2/krylov_sign_structure/results/k2implong-%A_%a.err
set -euo pipefail
module purge
module load python/3.12-2024.10
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
BASE=/home/a/A.Otaifi/ChattyRun/j1j2/krylov_sign_structure
MS=(10000 25000 10000 25000)
BETAS=(5.0 5.0 10.0 10.0)
i=$SLURM_ARRAY_TASK_ID
M=${MS[$i]}; B=${BETAS[$i]}
ID="${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
notify(){ "$BASE/experiments/chatty_notify.sh" "$1" "$ID" "$2"; }
finish(){ rc=$?; if [ "$rc" -eq 0 ]; then notify FINISHED "K2 importance M=$M beta=$B"; else notify FAILED "K2 importance M=$M beta=$B exit=$rc"; fi; }
trap finish EXIT
notify STARTED "K2 importance M=$M beta=$B"
cd "$BASE/experiments"
python finite_tau_ct_smc_k2cache_importance20.py --M "$M" --beta "$B" --radius 1 --out "../results/k2implong_M${M}_b${B}.json"
