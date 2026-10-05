#!/bin/bash
#SBATCH --job-name=chatty-stsign4
#SBATCH --partition=small
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=02:00:00
#SBATCH --output=/home/a/A.Otaifi/ChattyRun/j1j2/stored_signs/logs/%x-%A_%a.out
#SBATCH --error=/home/a/A.Otaifi/ChattyRun/j1j2/stored_signs/logs/%x-%A_%a.err
# Usage: sbatch --array=0-N stored_sign_4x4_lmu.sbatch <tag> <maxiter> ; config list below (index = array id)
set -euo pipefail
module purge
module load python/3.12-2024.10
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
TAG=$1; MAXIT=$2
BASE=/home/a/A.Otaifi/ChattyRun/j1j2/stored_signs
CFG=("exact|--method exact|0" "bench|--bench|0")
for v in "symtab|--method symtab" "symtab_b0.5|--method symtab --beta 0.5" "symtabnbr|--method symtab --nbr" "symtabnbr_b0.5|--method symtab --nbr --beta 0.5" "tab|--method tab" "cnn|--method cnn" "cnn_b0.5|--method cnn --beta 0.5"; do
  for n in 1000 10000 100000; do for sd in 0 1; do CFG+=("${v%%|*}_N${n}_s${sd}|${v#*|} --nsamp $n|$sd"); done; done; done
for v in "tabnbr|--method tab --nbr" "tabnbr_b0.5|--method tab --nbr --beta 0.5" "tab_b0.5|--method tab --beta 0.5"; do
  for n in 100 1000 10000; do for sd in 0 1; do CFG+=("${v%%|*}_N${n}_s${sd}|${v#*|} --nsamp $n|$sd"); done; done; done
for v in "symtab|--method symtab" "symtab_b0.5|--method symtab --beta 0.5" "symtabnbr|--method symtab --nbr" "symtabnbr_b0.5|--method symtab --nbr --beta 0.5" "cnn_b0.5|--method cnn --beta 0.5"; do
  for n in 100; do for sd in 0 1; do CFG+=("${v%%|*}_N${n}_s${sd}|${v#*|} --nsamp $n|$sd"); done; done; done
# 44..61: tab variants (N 1e2,1e3,1e4); 62..71: N=100 runs
# index: 0 exact, 1 bench, then 7 variants x 3 N x 2 seeds = 42 -> 2..43
E=${CFG[$SLURM_ARRAY_TASK_ID]}
NAME=${E%%|*}; R=${E#*|}; ARGS=${R%|*}; SEED=${R##*|}
mkdir -p $BASE/results
python $BASE/stored_sign_loop_4x4.py $ARGS --seed $SEED --maxiter $MAXIT --out $BASE/results/${TAG}_${NAME}.json
