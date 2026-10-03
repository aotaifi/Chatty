#!/bin/bash
#SBATCH --job-name=chatty-j1j2-frust8
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=01:30:00
#SBATCH --array=0-6
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=alyotaifi@gmail.com
#SBATCH --output=/pc2/users/h/hpcalot/chatty/j1j2/results/frust8-%A_%a.out
#SBATCH --error=/pc2/users/h/hpcalot/chatty/j1j2/results/frust8-%A_%a.err
set -euo pipefail
module load lang
module load Python/3.11.5-GCCcore-13.2.0
source /pc2/users/h/hpcalot/chatty/.venv-j1j2-py311/bin/activate
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
BASE=/pc2/users/h/hpcalot/chatty/j1j2
J2S=(0.00 0.20 0.40 0.50 0.60 0.80 1.00)
J2=${J2S[${SLURM_ARRAY_TASK_ID}]}
OUT="$BASE/results/frustration_scan_8x8_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "$OUT"
RUN=$(mktemp -d "${TMPDIR:-/tmp}/chatty-frust8.XXXXXX")
finish(){ cp -f "$RUN"/frustration_scan_8x8_J2_*.json "$OUT/" 2>/dev/null || true; cp -f "$RUN"/frustration_scan_8x8_J2_*.npz "$OUT/" 2>/dev/null || true; cp -f "$RUN"/frustration_scan_8x8.py "$OUT/" 2>/dev/null || true; }
trap finish EXIT
cp "$BASE/src/frustration_scan_8x8.py" "$RUN/"
cp "$BASE/inputs/frustration_scan/vit_J2=${J2}_N=8x8_k=0.mpack" "$RUN/"
cp -r "$BASE/src/nqsmagic" "$RUN/"
cd "$RUN"
export PYTHONPATH="$RUN/nqsmagic:${PYTHONPATH:-}"
python frustration_scan_8x8.py "$J2"
