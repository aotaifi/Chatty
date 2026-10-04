#!/bin/bash
#SBATCH --job-name=fn6x6loop
#SBATCH --partition=cluster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=100G
#SBATCH --time=24:00:00
#SBATCH --output=/home/a/A.Otaifi/chatty_fn6x6/logs/fn6x6-%x-%j.out
# Usage: sbatch --job-name=fn6_and closed_fn_krylov_sym6x6.sbatch TAG [extra args of closed_fn_krylov_sym6x6.py ...]
# e.g.   sbatch --job-name=fn6_and closed_fn_krylov_sym6x6.sbatch sym6x6_and_m3 --method anderson --m 3 --maxiter 400
set -euo pipefail
BASE=/home/a/A.Otaifi/chatty_fn6x6
ED=/home/a/A.Otaifi/chatty_ed6x6
cd "$BASE"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK} NUMBA_NUM_THREADS=${SLURM_CPUS_PER_TASK} OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
TAG="$1"; shift
mkdir -p out6 cache
{ echo "job_id=${SLURM_JOB_ID}"; echo "host=$(hostname)"; echo "cpus=${SLURM_CPUS_PER_TASK}"; echo "tag=${TAG}"; echo "args=$*"; echo "started=$(date -Is)"; } >> "out6/run_metadata_${TAG}_${SLURM_JOB_ID}.txt"
/usr/bin/time -v $ED/ls_env/bin/python code/closed_fn_krylov_sym6x6.py --L 6 \
  --gs-vec $ED/run_gs/vectors_k00_A1_p.npy --gs-states $ED/run_gs/states_k00_A1_p.npy \
  --out-dir out6 --cache-dir cache --tag "$TAG" --fn-tol 1e-10 --solver lobpcg --time-budget 84000 "$@"
echo "finished=$(date -Is)" >> "out6/run_metadata_${TAG}_${SLURM_JOB_ID}.txt"
