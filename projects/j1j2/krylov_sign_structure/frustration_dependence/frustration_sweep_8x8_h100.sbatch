#!/bin/bash
#SBATCH --job-name=chatty-j1j2-frust8
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=alyotaifi@gmail.com
#SBATCH --output=/pc2/users/h/hpcalot/chatty/j1j2/results/frust8-%j.out
#SBATCH --error=/pc2/users/h/hpcalot/chatty/j1j2/results/frust8-%j.err
set -euo pipefail
module load lang
module load Python/3.11.5-GCCcore-13.2.0
source /pc2/users/h/hpcalot/chatty/.venv-j1j2-py311/bin/activate
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
BASE=/pc2/users/h/hpcalot/chatty/j1j2
OUT="$BASE/results/frustration_sweep_8x8_${SLURM_JOB_ID}"
mkdir -p "$OUT"
RUN=$(mktemp -d "${TMPDIR:-/tmp}/chatty-frust8.XXXXXX")
finish(){
  cp -f "$RUN"/frustration_sweep_8x8*.json "$OUT/" 2>/dev/null || true
  cp -f "$RUN"/frustration_sweep_8x8.py "$OUT/" 2>/dev/null || true
}
trap finish EXIT
cp "$BASE/src/frustration_sweep_8x8.py" "$RUN/"
cp "$BASE/inputs/frustration/"vit_J2=*_N=8x8_k=0.mpack "$RUN/"
cd "$RUN"
python frustration_sweep_8x8.py
