#!/bin/bash
#SBATCH --job-name=chatty-sr-opt-fn
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=01:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=alyotaifi@gmail.com
#SBATCH --output=/pc2/users/h/hpcalot/chatty/j1j2/results/sr-opt-fn-%j.out
#SBATCH --error=/pc2/users/h/hpcalot/chatty/j1j2/results/sr-opt-fn-%j.err
set -euo pipefail
: "${PARENT_JOB:?need PARENT_JOB}"
module load lang
module load Python/3.11.5-GCCcore-13.2.0
source /pc2/users/h/hpcalot/chatty/.venv-j1j2-py311/bin/activate
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
BASE=/pc2/users/h/hpcalot/chatty/j1j2
PARENT="$BASE/results/sr_prod_build_${PARENT_JOB}"
OUT="$BASE/results/sr_opt_fn_${SLURM_JOB_ID}"
mkdir -p "$OUT"
RUN=$(mktemp -d "${TMPDIR:-/tmp}/sr-opt-fn.XXXXXX")
finish(){ cp -f "$RUN"/*.npz "$RUN"/*.json "$RUN"/*.py "$OUT/" 2>/dev/null || true; }
trap finish EXIT
cp "$BASE/src/gfmc_8x8_sr_optimized.py" "$RUN/"
cp "$PARENT/vit8_fnmle_sr_eta001.mpack" "$RUN/"
cp "$PARENT/vit8_fnmle_sr_eta001_handoff.npz" "$RUN/"
cd "$RUN"
python gfmc_8x8_sr_optimized.py
