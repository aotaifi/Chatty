#!/bin/bash
#SBATCH --job-name=chatty-ft-matrix
#SBATCH --partition=small
#SBATCH --array=0-18%12
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=a.otaifi@lmu.de
#SBATCH --output=/home/a/A.Otaifi/ChattyRun/j1j2/krylov_sign_structure/results/matrix-%A_%a.out
#SBATCH --error=/home/a/A.Otaifi/ChattyRun/j1j2/krylov_sign_structure/results/matrix-%A_%a.err
set -euo pipefail
module purge
module load python/3.12-2024.10
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
BASE=/home/a/A.Otaifi/ChattyRun/j1j2/krylov_sign_structure
cd "$BASE/experiments"

MODES=(call call call call call call call call rep rep rep dense dense dense dense dense fixed fixed fixed)
MS=(25000 50000 100000 200000 25000 50000 100000 200000 25000 50000 100000 50000 100000 200000 100000 200000 50000 100000 100000)
BETAS=(2.5 2.5 2.5 2.5 3.0 3.0 3.0 3.0 2.0 2.0 2.0 2.5 2.5 2.5 3.0 3.0 2.5 2.5 3.0)
SEEDS=(0 0 0 0 0 0 0 0 1000000 1000000 1000000 0 0 0 0 0 0 0 0)

MODE=${MODES[$SLURM_ARRAY_TASK_ID]}
M=${MS[$SLURM_ARRAY_TASK_ID]}
BETA=${BETAS[$SLURM_ARRAY_TASK_ID]}
SO=${SEEDS[$SLURM_ARRAY_TASK_ID]}
TAG="${MODE}_M${M}_b${BETA}_s${SO}"
echo "job=$SLURM_JOB_ID task=$SLURM_ARRAY_TASK_ID mode=$MODE M=$M beta=$BETA seedoff=$SO host=$(hostname) start=$(date -Is)"

case "$MODE" in
  call|rep)
    python finite_tau_ct_smc_callable_longbeta20.py --M "$M" --beta "$BETA" --seed-offset "$SO" --out "../results/matrix_${TAG}.json"
    ;;
  fixed)
    python finite_tau_ct_smc_callable_longbeta20.py --M "$M" --beta "$BETA" --fixed-sign --out "../results/matrix_${TAG}.json"
    ;;
  dense)
    python finite_tau_ct_smc_longbeta_scaling20.py --M "$M" --beta "$BETA" --out "../results/matrix_${TAG}.json"
    ;;
esac
echo "finish=$(date -Is)"
