#!/bin/bash
#SBATCH --job-name=chatty-hfn8micro
#SBATCH --partition=gpu_a40
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/pc2/users/h/hpcalot/chatty/j1j2/results/hfn8micro-%j.out
#SBATCH --error=/pc2/users/h/hpcalot/chatty/j1j2/results/hfn8micro-%j.err
set -euo pipefail
module load lang
module load Python/3.11.5-GCCcore-13.2.0
source /pc2/users/h/hpcalot/chatty/.venv-j1j2-py311/bin/activate
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false
BASE=/pc2/users/h/hpcalot/chatty/j1j2
RUN=$(mktemp -d "${TMPDIR:-/tmp}/hfn8micro.XXXXXX")
OUT="$BASE/results/frozen_hfn_sr8_micro_${SLURM_JOB_ID}"
mkdir -p "$OUT"
trap 'cp -f "$RUN"/*.json "$OUT/" 2>/dev/null || true' EXIT
cp "$BASE/src/frozen_hfn_sr8_microtrust.py" "$RUN/"
cp "$BASE/results/frozen_hfn_sr8_vec256_3554913/a1.mpack" "$RUN/a1.mpack"
cp "$BASE/results/frozen_hfn_sr8_vec256_3554913/frozen_hfn_sr8_signcache256.npz" "$RUN/freeze.npz"
cd "$RUN"
python frozen_hfn_sr8_microtrust.py --a1 a1.mpack --freeze freeze.npz --out microtrust.json
