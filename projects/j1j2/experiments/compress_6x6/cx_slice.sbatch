#!/bin/bash
#SBATCH --job-name=cx
#SBATCH --partition=cip
#SBATCH --gres=gpu:a40-24gb:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=38G
#SBATCH --time=02:00:00
#SBATCH --output=/home/a/A.Otaifi/chatty_compress6/logs/%x-%j.out
# Compression pre-tests (experiments/compress_6x6): exact 6x6 sector + students on one full A40, fp32 networks.
# A40 vGPU slice variant (cluster/A40_SLICES_HOWTO.md: jax 0.8.2 venv). Usage: sbatch --job-name=NAME cx.sbatch SCRIPT.py [SPEC.json]
set -euo pipefail
B=/home/a/A.Otaifi/chatty_compress6
OUT="/project/theorie/a/A.Otaifi/chatty_compress6/runs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}"
mkdir -p "$OUT/code"
cp "$B"/code/*.py "$OUT/code/"
SCRIPT=$1; shift
ARGS=()
if [ $# -ge 1 ]; then cp "$1" "$OUT/spec.json"; ARGS=("$OUT/spec.json"); fi
export XLA_FLAGS="${CX_XLA:---xla_gpu_autotune_level=0}" XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=${CX_MEMFRAC:-0.85} OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
{ echo "job_id=${SLURM_JOB_ID}"; echo "host=$(hostname)"; echo "script=$SCRIPT"; echo "spec=${1:-}"; echo "started=$(date -Is)"; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader; } > "$OUT/run_metadata.txt"
cd "$OUT/code"
"/project/theorie/a/A.Otaifi/a40slice/venv_jax_0.8.2/bin/python" "$SCRIPT" "$OUT" "${ARGS[@]}"
echo "finished=$(date -Is)" >> "$OUT/run_metadata.txt"
