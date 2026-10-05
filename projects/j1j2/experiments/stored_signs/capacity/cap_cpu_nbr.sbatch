#!/bin/bash
#SBATCH --job-name=chatty-cap4nbr
#SBATCH --partition=small
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=04:00:00
#SBATCH --output=/home/a/A.Otaifi/ChattyRun/j1j2/stored_signs/capacity/logs/%x-%A_%a.out
#SBATCH --error=/home/a/A.Otaifi/ChattyRun/j1j2/stored_signs/capacity/logs/%x-%A_%a.err
# usage: sbatch --array=0-5 cap_cpu.sbatch <tag> <maxiter>
set -euo pipefail
module purge; module load python/3.12-2024.10
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
TAG=$1; MAXIT=$2
BASE=/home/a/A.Otaifi/ChattyRun/j1j2/stored_signs/capacity
CFG=()
for v in "5k_nbr_b0.5|--beta 0.5 --nbr"; do
  for sd in 0 1; do CFG+=("${v%%|*}_s${sd}|${v#*|}|$sd"); done; done
E=${CFG[$SLURM_ARRAY_TASK_ID]}
NAME=${E%%|*}; R=${E#*|}; ARGS=${R%|*}; SEED=${R##*|}
mkdir -p $BASE/results
python $BASE/stored_sign_capacity_4x4.py $ARGS --seed $SEED --maxiter $MAXIT --out $BASE/results/${TAG}_${NAME}.json
