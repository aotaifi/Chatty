#!/bin/bash
#SBATCH --job-name=ss6
#SBATCH --partition=inter
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=60G
#SBATCH --time=03:00:00
#SBATCH --output=/project/theorie/a/A.Otaifi/Chatty/runs/ss6/logs/%x-%j.out
#SBATCH --error=/project/theorie/a/A.Otaifi/Chatty/runs/ss6/logs/%x-%j.err
# Stored signs 6x6 (experiments/stored_signs_6x6). Full A40 only (a40-NNgb slices fail with jax 0.10).
# Reuses ll6_core.py / vit_dt.py / venv / data of ~/chatty_ll6 read-only.
# Usage: sbatch --job-name=NAME ss6_gpu.sbatch <script.py> <args...>
set -euo pipefail
BASE=/project/theorie/a/A.Otaifi/Chatty/runs/ss6; LL6=/project/theorie/a/A.Otaifi/Chatty/runs/ll6
OUT="$BASE/runs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}"
mkdir -p "$OUT"
export JAX_PLATFORMS=cpu SS6_DATA=/project/theorie/a/A.Otaifi/Chatty/runs/ss6/data_cpu XLA_PYTHON_CLIENT_PREALLOCATE=false OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
cp "$BASE"/code/*.py "$OUT/"; cp "$LL6"/code/ll6_core.py "$LL6"/code/vit_dt.py "$OUT/"
cd "$OUT"
for f in "vit_J2=0.50_N=6x6_k=0.mpack" samples_psi0sq_6x6.npz samples_extra_psi0sq_6x6.npz; do ln -sf "$LL6/data/$f" .; done
for f in "$BASE"/data/*; do [ -e "$f" ] && ln -sf "$f" .; done
{ echo "job_id=${SLURM_JOB_ID}"; echo "host=$(hostname)"; echo "args=$*"; echo "started=$(date -Is)"; nproc; } > run_metadata.txt
"$LL6/venv_gpu/bin/python" "$@"
echo "finished=$(date -Is)" >> run_metadata.txt
