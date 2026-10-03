#!/bin/bash
#SBATCH --job-name=chatty-frozenhfn4
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --output=/pc2/users/h/hpcalot/chatty/j1j2/results/frozenhfn4-%j.out
#SBATCH --error=/pc2/users/h/hpcalot/chatty/j1j2/results/frozenhfn4-%j.err
set -euo pipefail
: "${ITER:?ITER required}"
module load lang
module load Python/3.11.5-GCCcore-13.2.0
source /pc2/users/h/hpcalot/chatty/.venv-j1j2-py311/bin/activate
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false
BASE=/pc2/users/h/hpcalot/chatty/j1j2
RUN=$(mktemp -d "${TMPDIR:-/tmp}/frozenhfn4.XXXXXX")
OUT="$BASE/results/frozen_hfn_factorized_vit4_iter${ITER}_${SLURM_JOB_ID}"
mkdir -p "$OUT"
trap 'cp -f "$RUN"/*.json "$OUT/" 2>/dev/null || true' EXIT
cp "$BASE/src/frozen_hfn_factorized_vit_exact4x4.py" "$RUN/"
cp "$BASE/src/closed_fn_krylov_exact4x4.py" "$RUN/"
cd "$RUN"
python frozen_hfn_factorized_vit_exact4x4.py --iteration "$ITER" --steps 1000 --out "frozen_hfn_factorized_iter${ITER}.json"
