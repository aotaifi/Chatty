#!/bin/bash
#SBATCH --job-name=wt
#SBATCH --partition=cip,inter
#SBATCH --gres=gpu:a40:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=/home/a/A.Otaifi/chatty_writeback_tail/logs/%x-%j.out
# Write-back with tail sampling (experiments/writeback_tail): exact 6x6 sector + residual CNN on one full A40.
# RTX 2080 Ti alternative: sbatch --partition=inter --gres=gpu:rtx2080ti:1 ...
# Usage: sbatch --job-name=NAME wt.sbatch SCRIPT.py [SPEC.json]
set -euo pipefail
B=/home/a/A.Otaifi/chatty_writeback_tail
OUT="/project/theorie/a/A.Otaifi/chatty_writeback_tail/runs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}"
mkdir -p "$OUT/code"
cp "$B"/code/*.py "$OUT/code/"
SCRIPT=$1; shift
ARGS=()
if [ $# -ge 1 ]; then cp "$1" "$OUT/spec.json"; ARGS=("$OUT/spec.json"); fi
export XLA_FLAGS="--xla_gpu_autotune_level=0" XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=${WT_MEMFRAC:-0.92} OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
{ echo "job_id=${SLURM_JOB_ID}"; echo "host=$(hostname)"; echo "script=$SCRIPT"; echo "spec=${1:-}"; echo "started=$(date -Is)"; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader; } > "$OUT/run_metadata.txt"
cd "$OUT/code"
"/home/a/A.Otaifi/chatty_ll6/venv_gpu/bin/python" "$SCRIPT" "$OUT" "${ARGS[@]}"
echo "finished=$(date -Is)" >> "$OUT/run_metadata.txt"
