#!/bin/bash
#SBATCH --job-name=sdt-c6
#SBATCH --partition=cip,inter
#SBATCH --gres=gpu:a40:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=03:00:00
#SBATCH --output=/project/theorie/a/A.Otaifi/Chatty/runs/ChattyRun/j1j2/sign_design_tests/logs/%x-%j.out
# Full A40 only (a40-NNgb slices fail with this jax). Reuses ~/chatty_ll6 (ll6_core.py, vit_dt.py, venv_gpu, data) read-only.
set -euo pipefail
B=/project/theorie/a/A.Otaifi/Chatty/runs/ChattyRun/j1j2/sign_design_tests; LL6=/project/theorie/a/A.Otaifi/Chatty/runs/ll6
OUT="$B/runs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}"; mkdir -p "$OUT"
export XLA_PYTHON_CLIENT_PREALLOCATE=false OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
cp "$B"/c_delta_6x6.py "$LL6"/code/ll6_core.py "$LL6"/code/vit_dt.py "$OUT/"
cd "$OUT"
for f in "vit_J2=0.50_N=6x6_k=0.mpack" samples_psi0sq_6x6.npz samples_extra_psi0sq_6x6.npz; do ln -sf "$LL6/data/$f" .; done
{ echo "job_id=${SLURM_JOB_ID}"; echo "host=$(hostname)"; echo "args=$*"; echo "started=$(date -Is)"; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader; } > run_metadata.txt
"$LL6/venv_gpu/bin/python" c_delta_6x6.py --out $OUT/c6.json "$@"
echo "finished=$(date -Is)" >> run_metadata.txt
