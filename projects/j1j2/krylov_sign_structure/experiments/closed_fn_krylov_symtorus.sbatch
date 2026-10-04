#!/bin/bash
#SBATCH --job-name=fntorus
#SBATCH --partition=cluster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=24:00:00
#SBATCH --output=/home/a/A.Otaifi/chatty_fn6x6/logs/fntorus-%x-%j.out
# Usage: sbatch --job-name=NAME closed_fn_krylov_symtorus.sbatch TORUS(t1x,t1y,t2x,t2y) TAGPREFIX
# Runs Anderson(m=3) and plain loops sequentially on the tilted torus (symmetric sector k=0,A1,flip+).
set -euo pipefail
BASE=/home/a/A.Otaifi/chatty_fn6x6
ED=/home/a/A.Otaifi/chatty_ed6x6
cd "$BASE"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK} NUMBA_NUM_THREADS=${SLURM_CPUS_PER_TASK} OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
TORUS="$1"; PFX="$2"
mkdir -p outT cache
for M in anderson plain; do
  EXTRA=""; [ "$M" = anderson ] && EXTRA="--m 3"
  $ED/ls_env/bin/python code/closed_fn_krylov_sym6x6.py --L 0 --torus "$TORUS" --method $M $EXTRA --maxiter 3000 \
    --stop-eps 1e-8 --fn-tol 1e-10 --solver lobpcg --out-dir outT --cache-dir cache --tag ${PFX}_${M} --time-budget 40000 --resume
done
