#!/bin/bash
#SBATCH --job-name=chatty-hfn4ite
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/pc2/users/h/hpcalot/chatty/j1j2/results/hfn4ite-%j.out
#SBATCH --error=/pc2/users/h/hpcalot/chatty/j1j2/results/hfn4ite-%j.err
set -euo pipefail
: "${LR:?LR required}"
: "${TAG:?TAG required}"
module load lang
module load Python/3.11.5-GCCcore-13.2.0
source /pc2/users/h/hpcalot/chatty/.venv-j1j2-py311/bin/activate
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false
BASE=/pc2/users/h/hpcalot/chatty/j1j2
RUN=$(mktemp -d "${TMPDIR:-/tmp}/hfn4ite.XXXXXX")
OUT="$BASE/results/frozen_hfn_ite_${TAG}_${SLURM_JOB_ID}"
mkdir -p "$OUT"
trap 'cp -f "$RUN"/*.json "$OUT/" 2>/dev/null || true' EXIT
cp "$BASE/src/frozen_hfn_ite_target_exact4x4.py" "$RUN/"
cp "$BASE/src/closed_fn_krylov_exact4x4.py" "$RUN/"
cd "$RUN"
python frozen_hfn_ite_target_exact4x4.py --iteration 10 --lr "$LR" --epochs 20 --energy-samples 100000 --max-inner 2000 --out "ite_${TAG}.json"
