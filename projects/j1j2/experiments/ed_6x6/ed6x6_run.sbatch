#!/bin/bash
#SBATCH --job-name=chatty-ed6x6
#SBATCH --partition=cluster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=110G
#SBATCH --time=12:00:00
#SBATCH --output=/home/a/A.Otaifi/chatty_ed6x6/logs/ed6x6-%j.out
# Usage: sbatch ed6x6_run.sbatch OUTDIR SECTOR_SPEC [SECTOR_SPEC ...]   (see ed6x6_run.py)
set -euo pipefail
BASE=/home/a/A.Otaifi/chatty_ed6x6
cd "$BASE"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK} NUMBA_NUM_THREADS=${SLURM_CPUS_PER_TASK}
OUT="$1"; shift
mkdir -p "$OUT"
{ echo "job_id=${SLURM_JOB_ID}"; echo "host=$(hostname)"; echo "cpus=${SLURM_CPUS_PER_TASK}"; echo "args=$*"; echo "started=$(date -Is)"; lscpu | grep "Model name"; } >> "$OUT/run_metadata_${SLURM_JOB_ID}.txt"
/usr/bin/time -v ls_env/bin/python ed6x6_run.py "$OUT" "$@"
echo "finished=$(date -Is)" >> "$OUT/run_metadata_${SLURM_JOB_ID}.txt"
