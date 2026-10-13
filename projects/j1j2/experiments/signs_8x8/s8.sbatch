#!/bin/bash
#SBATCH --job-name=s8
#SBATCH --partition=cip
#SBATCH --gres=gpu:a40-24gb:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=38G
#SBATCH --time=04:00:00
#SBATCH --output=/project/theorie/a/A.Otaifi/chatty_signs8/logs/%x-%j.out
# Stored signs at 8x8 (experiments/signs_8x8). jax 0.8.2 venv (A40 slices, full A40, 2080 Ti; A40_SLICES_HOWTO.md).
# 16gb slice: --gres=gpu:a40-16gb:1 --mem=24G.  Full A40: --partition=cip,inter --gres=gpu:a40:1 --mem=48G.
# Usage: sbatch --job-name=NAME s8.sbatch SCRIPT.py [ARGS...]  (scripts writing a run dir get OUT as first argument
# when S8_OUTDIR=1; otherwise the arguments are passed unchanged and the working directory is the run dir)
set -euo pipefail
B=/project/theorie/a/A.Otaifi/chatty_signs8
OUT="$B/runs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}"
mkdir -p "$OUT/code"
cp "$B"/code/*.py "$OUT/code/"
SCRIPT=$1; shift
export XLA_FLAGS="${S8_XLA---xla_gpu_autotune_level=0}" XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=${S8_MEMFRAC:-0.85} OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
{ echo "job_id=${SLURM_JOB_ID}"; echo "host=$(hostname)"; echo "script=$SCRIPT"; echo "args=$*"; echo "started=$(date -Is)"; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader; } > "$OUT/run_metadata.txt"
cd "$OUT/code"
ln -sf /project/theorie/a/A.Otaifi/chatty_loop8/data/vit_J2=0.50_N=8x8_k=0.mpack .
PY=/project/theorie/a/A.Otaifi/a40slice/venv_jax_0.8.2/bin/python
if [ "${S8_OUTDIR:-1}" = "1" ]; then "$PY" "$SCRIPT" "$OUT" "$@"; else cd "$OUT"; "$PY" "code/$SCRIPT" "$@"; fi
echo "finished=$(date -Is)" >> "$OUT/run_metadata.txt"
