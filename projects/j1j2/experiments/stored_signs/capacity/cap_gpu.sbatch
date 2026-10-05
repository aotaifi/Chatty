#!/bin/bash
#SBATCH --job-name=chatty-cap4gpu
#SBATCH --partition=inter,cip
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=03:00:00
#SBATCH --output=/home/a/A.Otaifi/ChattyRun/j1j2/stored_signs/capacity/logs/%x-%j.out
#SBATCH --error=/home/a/A.Otaifi/ChattyRun/j1j2/stored_signs/capacity/logs/%x-%j.err
# usage: sbatch cap_gpu.sbatch <tag> <maxiter> ; runs 10 (variant,seed) jobs, 5 concurrently on one A40
set -uo pipefail
module purge; module load python/3.12-2024.10
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
TAG=$1; MAXIT=$2
BASE=/home/a/A.Otaifi/ChattyRun/j1j2/stored_signs/capacity
mkdir -p $BASE/results
CFG=()
for v in "50k_b0.5|--hid 150 --nlayers 3 --beta 0.5" "200k_b0.5|--hid 256 --nlayers 4 --beta 0.5" "50k_b0.3|--hid 150 --nlayers 3 --beta 0.3" "200k_b0.3|--hid 256 --nlayers 4 --beta 0.3" "5k_nbr_b0.5|--beta 0.5 --nbr"; do
  for sd in 0 1; do CFG+=("${v%%|*}_s${sd}|${v#*|}|$sd"); done; done
i=0
for E in "${CFG[@]}"; do
  NAME=${E%%|*}; R=${E#*|}; ARGS=${R%|*}; SEED=${R##*|}
  python $BASE/stored_sign_capacity_4x4.py $ARGS --seed $SEED --maxiter $MAXIT --device cuda --threads 1 --out $BASE/results/${TAG}_${NAME}.json > $BASE/logs/${TAG}_${NAME}.log 2>&1 &
  i=$((i+1)); if (( i % 5 == 0 )); then wait; fi
done
wait
