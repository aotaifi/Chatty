# Sourced by the deliverable-C Paderborn drafts (NOT SUBMITTED; Paderborn drained since 2026-10-06).
# Staging (from the Mac):  rsync projects/j1j2/experiments/deliverable_C/*.py experiments/loop_8x8/l8_core.py
#   experiments/loop_8x8/review/rv_da.py experiments/learned_loop_6x6/vit_dt.py  -> $BASE/src/delC/
#   inputs: vit_J2=0.50_N=8x8_k=0.mpack (already in $BASE/inputs), psi_1 samples rvD{1,2,3}_samples.npz and psi_P pool
#   l8ref{7000,8000,10000}_X.npz (X arrays only, from ws1 /project/theorie/a/A.Otaifi/chatty_loop8/runs) -> $BASE/inputs/delC/
set -euo pipefail
module load lang
module load Python/3.11.5-GCCcore-13.2.0
source /pc2/users/h/hpcalot/chatty/.venv-j1j2-py311/bin/activate
export JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
BASE=/pc2/users/h/hpcalot/chatty/j1j2
OUT="$BASE/results/delC_${SLURM_JOB_NAME}_${SLURM_JOB_ID}"; mkdir -p "$OUT/code"
cp "$BASE"/src/delC/*.py "$OUT/code/"; cd "$OUT/code"; ln -sf "$BASE/inputs/vit_J2=0.50_N=8x8_k=0.mpack" .
{ echo "job=${SLURM_JOB_ID} host=$(hostname) started=$(date -Is)"; nvidia-smi -L; } > "$OUT/run_metadata.txt"
