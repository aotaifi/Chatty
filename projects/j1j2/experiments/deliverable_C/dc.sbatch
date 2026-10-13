#!/bin/bash
#SBATCH --job-name=dc
#SBATCH --partition=cip
#SBATCH --gres=gpu:a40-24gb:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=38G
#SBATCH --time=08:00:00
#SBATCH --output=/project/theorie/a/A.Otaifi/chatty_delC/logs/%x-%j.out
# Deliverable C (experiments/deliverable_C). jax 0.8.2 venv (A40 slices, full A40, 2080 Ti; cluster/A40_SLICES_HOWTO.md).
# Full A40: sbatch --partition=cip,inter --gres=gpu:a40:1 --mem=48G ...
# Usage: sbatch --job-name=NAME dc.sbatch SCRIPT.py [ARGS...]   (OUT dir is passed as the first argument)
set -euo pipefail
B=/project/theorie/a/A.Otaifi/chatty_delC
OUT="$B/runs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}"
mkdir -p "$OUT/code"
cp "$B"/code/*.py "$OUT/code/"
SCRIPT=$1; shift
export XLA_FLAGS="--xla_gpu_autotune_level=0" XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.85 OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
{ echo "job_id=${SLURM_JOB_ID}"; echo "host=$(hostname)"; echo "script=$SCRIPT"; echo "args=$*"; echo "started=$(date -Is)"; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader; } > "$OUT/run_metadata.txt"
cd "$OUT/code"
ln -sf /project/theorie/a/A.Otaifi/chatty_loop8/data/vit_J2=0.50_N=8x8_k=0.mpack .
"/project/theorie/a/A.Otaifi/a40slice/venv_jax_0.8.2/bin/python" "$SCRIPT" "$OUT" "$@"
echo "finished=$(date -Is)" >> "$OUT/run_metadata.txt"
