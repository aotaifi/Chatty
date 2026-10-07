#!/bin/bash
#SBATCH --job-name=wb
#SBATCH --partition=inter
#SBATCH --gres=gpu:rtx2080ti:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=04:00:00
#SBATCH --output=/home/a/A.Otaifi/chatty_writeback/logs/%x-%j.out
# Write-back test (experiments/writeback): exact 6x6 sector + residual CNN on one GPU.
# RTX 2080 Ti default; A40: sbatch --partition=cip,inter --gres=gpu:a40:1 ...
# Usage: sbatch --job-name=NAME wb.sbatch SPEC.json
set -euo pipefail
B=/home/a/A.Otaifi/chatty_writeback
OUT="/project/theorie/a/A.Otaifi/chatty_writeback/runs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}"
mkdir -p "$OUT/code"
cp "$B"/code/*.py "$OUT/code/"
cp "$1" "$OUT/spec.json"
export XLA_FLAGS="--xla_gpu_autotune_level=0" XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=${WB_MEMFRAC:-0.80} OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
{ echo "job_id=${SLURM_JOB_ID}"; echo "host=$(hostname)"; echo "spec=$1"; echo "started=$(date -Is)"; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader; } > "$OUT/run_metadata.txt"
cd "$OUT/code"
"/home/a/A.Otaifi/chatty_ll6/venv_gpu/bin/python" wb_run.py "$OUT" "$OUT/spec.json"
echo "finished=$(date -Is)" >> "$OUT/run_metadata.txt"
