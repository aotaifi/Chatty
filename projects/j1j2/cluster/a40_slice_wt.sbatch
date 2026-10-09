#!/bin/bash
#SBATCH --job-name=wtslice
#SBATCH --partition=cip
#SBATCH --gres=gpu:a40-24gb:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=38G
#SBATCH --time=01:00:00
#SBATCH --output=/project/theorie/a/A.Otaifi/chatty_writeback_tail/logs/%x-%j.out
# Tested example: writeback_tail job on a partial ("vGPU slice") A40.  See A40_SLICES_HOWTO.md.
# Slice gres: gpu:a40-8gb:1 | gpu:a40-16gb:1 | gpu:a40-24gb:1  (partition cip only; --gres on the command line overrides).
# Node RAM: a40-8gb / a40-16gb nodes have 26 GB (pass --mem=24G there), a40-24gb nodes 41 GB (--mem<=40G); 12 cores each.
# Usage: sbatch [--gres=gpu:a40-16gb:1] --job-name=NAME a40_slice_wt.sbatch SCRIPT.py [SPEC.json]
set -euo pipefail
B=/home/a/A.Otaifi/chatty_writeback_tail
OUT="/project/theorie/a/A.Otaifi/chatty_writeback_tail/runs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}"
mkdir -p "$OUT/code"
cp "$B"/code/*.py "$OUT/code/"
SCRIPT=$1; shift
ARGS=()
if [ $# -ge 1 ]; then cp "$1" "$OUT/spec.json"; ARGS=("$OUT/spec.json"); fi

# --- slice-specific: jax >= 0.9 / 0.10 needs CUDA VMM, which vGPU slices do not have -> use the jax 0.8.2 venv ---
VENV=/project/theorie/a/A.Otaifi/a40slice/venv_jax_0.8.2
export XLA_FLAGS="--xla_gpu_autotune_level=0" XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=${WT_MEMFRAC:-0.85} OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
# (no CUDA_VISIBLE_DEVICES / MIG handling needed: the slice is simply device 0)

{ echo "job_id=${SLURM_JOB_ID}"; echo "host=$(hostname)"; echo "script=$SCRIPT"; echo "spec=${1:-}"; echo "started=$(date -Is)"; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader; } > "$OUT/run_metadata.txt"
cd "$OUT/code"
"$VENV/bin/python" "$SCRIPT" "$OUT" "${ARGS[@]}"
echo "finished=$(date -Is)" >> "$OUT/run_metadata.txt"
