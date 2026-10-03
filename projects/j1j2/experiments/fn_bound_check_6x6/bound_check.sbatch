#!/bin/bash
#SBATCH --job-name=chatty-fnbound
#SBATCH --partition=cluster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=/home/a/A.Otaifi/chatty_fnguides/bound_check/logs/fnbound-%x-%j.out
#SBATCH --error=/home/a/A.Otaifi/chatty_fnguides/bound_check/logs/fnbound-%x-%j.err
# Usage: sbatch --job-name=NAME bound_check.sbatch <script.py> <args...>
set -euo pipefail
module load python/3.11-2023.09
BASE=/home/a/A.Otaifi/chatty_fnguides
BC=$BASE/bound_check
OUT="$BC/results/${SLURM_JOB_NAME}_${SLURM_JOB_ID}"
mkdir -p "$OUT"
export JAX_PLATFORMS=cpu OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK} FNGC_DIR=$BASE
cp "$BC"/*.py "$OUT/"
cd "$OUT"
for f in "vit_J2=0.50_N=6x6_k=0.mpack" energy_krylov_vs_vit_6x6_indep.npz; do ln -sf "$BASE/$f" .; done
{ echo "job_id=${SLURM_JOB_ID}"; echo "host=$(hostname)"; echo "args=$*"; echo "started=$(date -Is)"; } > run_metadata.txt
SCRIPT=$1; shift
"$BASE/venv311/bin/python" "$SCRIPT" "$@"
echo "finished=$(date -Is)" >> run_metadata.txt
