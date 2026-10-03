#!/bin/bash
#SBATCH --job-name=chatty-hfn4minsr
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/pc2/users/h/hpcalot/chatty/j1j2/results/hfn4minsr-%j.out
#SBATCH --error=/pc2/users/h/hpcalot/chatty/j1j2/results/hfn4minsr-%j.err
set -euo pipefail
module load lang
module load Python/3.11.5-GCCcore-13.2.0
source /pc2/users/h/hpcalot/chatty/.venv-j1j2-py311/bin/activate
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false
BASE=/pc2/users/h/hpcalot/chatty/j1j2
RUN=$(mktemp -d "${TMPDIR:-/tmp}/hfn4minsr.XXXXXX")
OUT="$BASE/results/frozen_hfn_minsr_${SLURM_JOB_ID}"
mkdir -p "$OUT"
trap 'cp -f "$RUN"/*.json "$OUT/" 2>/dev/null || true' EXIT
cp "$BASE/src/frozen_hfn_factorized_minsr_exact4x4.py" "$RUN/"
cp "$BASE/src/closed_fn_krylov_exact4x4.py" "$RUN/"
cd "$RUN"
python frozen_hfn_factorized_minsr_exact4x4.py --iteration 10 --trust 0.002 --steps 20 --qgt-n 1024 --out minsr.json
