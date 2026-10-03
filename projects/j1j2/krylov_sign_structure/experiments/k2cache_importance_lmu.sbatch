#!/bin/bash
#SBATCH --job-name=chatty-ft-k2imp
#SBATCH --partition=small
#SBATCH --array=0-2
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --mail-type=NONE
#SBATCH --output=/home/a/A.Otaifi/ChattyRun/j1j2/krylov_sign_structure/results/k2imp-%A_%a.out
#SBATCH --error=/home/a/A.Otaifi/ChattyRun/j1j2/krylov_sign_structure/results/k2imp-%A_%a.err
set -euo pipefail
module purge
module load python/3.12-2024.10
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
BASE=/home/a/A.Otaifi/ChattyRun/j1j2/krylov_sign_structure
MS=(5000 10000 25000)
i=$SLURM_ARRAY_TASK_ID
M=${MS[$i]}
ID="${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
notify(){ "$BASE/experiments/chatty_notify.sh" "$1" "$ID" "$2"; }
finish(){ rc=$?; if [ "$rc" -eq 0 ]; then notify FINISHED "K2 importance M=$M"; else notify FAILED "K2 importance M=$M exit=$rc"; fi; }
trap finish EXIT
notify STARTED "K2 importance M=$M"
cd "$BASE/experiments"
python finite_tau_ct_smc_k2cache_importance20.py --M "$M" --beta 3.0 --radius 1 --out "../results/k2imp_M${M}_r1_b3.json"
