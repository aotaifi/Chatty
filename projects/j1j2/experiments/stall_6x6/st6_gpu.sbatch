#!/bin/bash
#SBATCH --job-name=st6
#SBATCH --partition=inter
#SBATCH --gres=gpu:rtx2080ti:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=40G
#SBATCH --time=04:00:00
#SBATCH --output=/home/a/A.Otaifi/chatty_stall6/logs/%x-%j.out
# Stall diagnosis 6x6 (experiments/stall_6x6): exact sector + ViT on one GPU (RTX 2080 Ti default; or --partition=cip,inter --gres=gpu:a40:1), fp32 ViT (highest matmul precision).
# Usage: sbatch --job-name=NAME st6_gpu.sbatch MODE [SPEC.json]
set -euo pipefail
B=/home/a/A.Otaifi/chatty_stall6
OUT="/project/theorie/a/A.Otaifi/chatty_stall6/runs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}"
mkdir -p "$OUT/code"
cp "$B"/code/*.py "$OUT/code/"
[ $# -ge 2 ] && cp "$2" "$OUT/spec.json"
export XLA_PYTHON_CLIENT_PREALLOCATE=false TF_GPU_ALLOCATOR=cuda_malloc_async OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
{ echo "job_id=${SLURM_JOB_ID}"; echo "host=$(hostname)"; echo "args=$*"; echo "started=$(date -Is)"; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader; } > "$OUT/run_metadata.txt"
cd "$OUT/code"
"/home/a/A.Otaifi/chatty_ll6/venv_gpu/bin/python" st6_run.py "$1" "$OUT" ${2:+"$OUT/spec.json"}
echo "finished=$(date -Is)" >> "$OUT/run_metadata.txt"
