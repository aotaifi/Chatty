#!/bin/bash
#SBATCH --job-name=lz6
#SBATCH --partition=inter
#SBATCH --gres=gpu:rtx2080ti:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=40G
#SBATCH --time=02:00:00
#SBATCH --output=/home/a/A.Otaifi/chatty_lanczos6/logs/%x-%j.out
# Exact Lanczos vs Krylov baseline on 6x6 J1-J2.  Usage: sbatch run.sbatch   (code copied to /home/a/A.Otaifi/chatty_lanczos6/experiments)
set -euo pipefail
B=/home/a/A.Otaifi/chatty_lanczos6
OUT="/project/theorie/a/A.Otaifi/chatty_stall6/runs_lanczos/${SLURM_JOB_NAME}_${SLURM_JOB_ID}"
mkdir -p "$OUT/experiments"
cp -r "$B"/experiments/* "$OUT/experiments/"
export XLA_FLAGS="--xla_gpu_autotune_level=0" XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.94 OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
{ echo "job_id=${SLURM_JOB_ID}"; echo "host=$(hostname)"; echo "started=$(date -Is)"; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader; } > "$OUT/run_metadata.txt"
cd "$OUT/experiments/lanczos_baseline_6x6"
"/home/a/A.Otaifi/chatty_ll6/venv_gpu/bin/python" run_lanczos6x6.py "$OUT"
echo "finished=$(date -Is)" >> "$OUT/run_metadata.txt"
