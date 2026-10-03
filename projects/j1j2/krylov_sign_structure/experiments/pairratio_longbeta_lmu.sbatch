#!/bin/bash
#SBATCH --job-name=chatty-ft-pairratio
#SBATCH --partition=small
#SBATCH --array=0-5
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --mail-type=NONE
#SBATCH --output=/home/a/A.Otaifi/ChattyRun/j1j2/krylov_sign_structure/results/pairratio-%A_%a.out
#SBATCH --error=/home/a/A.Otaifi/ChattyRun/j1j2/krylov_sign_structure/results/pairratio-%A_%a.err
set -euo pipefail
module purge
module load python/3.12-2024.10
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
BASE=/home/a/A.Otaifi/ChattyRun/j1j2/krylov_sign_structure
MS=(25000 50000 100000 50000 100000 200000)
BETAS=(2.5 2.5 2.5 3.0 3.0 3.0)
M=${MS[$SLURM_ARRAY_TASK_ID]}
BETA=${BETAS[$SLURM_ARRAY_TASK_ID]}
ID="${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
notify(){ "$BASE/experiments/chatty_notify.sh" "$1" "$ID" "$2"; }
finish(){ rc=$?; if [ "$rc" -eq 0 ]; then notify FINISHED "M=$M beta=$BETA"; else notify FAILED "M=$M beta=$BETA exit=$rc"; fi; }
trap finish EXIT
notify STARTED "M=$M beta=$BETA"
cd "$BASE/experiments"
python finite_tau_ct_smc_pairratio_longbeta20.py --M "$M" --beta "$BETA" --epochs 8 --nfit 20000 --K 4 --out "../results/finite_tau_ct_smc_pairratio_M${M}_b${BETA}.json"
