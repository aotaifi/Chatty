#!/bin/bash
#SBATCH --job-name=chatty-anderson
#SBATCH --partition=small
#SBATCH --array=0-8
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=06:00:00
#SBATCH --output=/home/a/A.Otaifi/ChattyRun/j1j2/krylov_sign_structure/anderson_loop/logs/anderson-%x-%A_%a.out
#SBATCH --error=/home/a/A.Otaifi/ChattyRun/j1j2/krylov_sign_structure/anderson_loop/logs/anderson-%x-%A_%a.err
# Usage: sbatch [--cpus-per-task/--mem/--time overrides] closed_fn_krylov_anderson_lmu.sbatch <4x4|20> <maxiter> [tag]
set -euo pipefail
module purge
module load python/3.12-2024.10
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
LAT=$1; MAXIT=$2; TAG=${3:-run}
BASE=/home/a/A.Otaifi/ChattyRun/j1j2/krylov_sign_structure/anderson_loop
CFG=(
 "plain|--method plain"
 "and_m3|--method anderson --m 3 --beta 1.0"
 "and_m5|--method anderson --m 5 --beta 1.0"
 "and_m8|--method anderson --m 8 --beta 1.0"
 "or_w1.5|--method overrelax --w 1.5"
 "or_w2|--method overrelax --w 2.0"
 "and_m5_krylovfn|--method anderson --m 5 --beta 1.0 --krylov-amp fn"
 "and_m5_prob|--method anderson --m 5 --beta 1.0 --weight prob"
 "and_m5_b0.5|--method anderson --m 5 --beta 0.5"
)
ENTRY=${CFG[$SLURM_ARRAY_TASK_ID]}
NAME=${ENTRY%%|*}; ARGS=${ENTRY#*|}
OUT=$BASE/results/${TAG}_${LAT}_${NAME}.json
echo "job=${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} host=$(hostname) lat=$LAT cfg=$NAME args=$ARGS start=$(date -Is)"
cd $BASE
python closed_fn_krylov_anderson.py --lattice $LAT --maxiter $MAXIT $ARGS --out $OUT
echo "finish=$(date -Is)"
