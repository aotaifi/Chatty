#!/bin/bash
#SBATCH --job-name=chatty-k2ovsr20
#SBATCH --partition=gpu_a40
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/pc2/users/h/hpcalot/chatty/j1j2/results/k2ovsr20-%j.out
#SBATCH --error=/pc2/users/h/hpcalot/chatty/j1j2/results/k2ovsr20-%j.err
set -euo pipefail
module load lang
module load Python/3.11.5-GCCcore-13.2.0
source /pc2/users/h/hpcalot/chatty/.venv-j1j2-py311/bin/activate
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false
BASE=/pc2/users/h/hpcalot/chatty/j1j2
RUN=$(mktemp -d "${TMPDIR:-/tmp}/k2ovsr20.XXXXXX")
OUT="$BASE/results/k2_factorized_overlap_sr20_${SLURM_JOB_ID}"
mkdir -p "$OUT"
trap 'cp -f "$RUN"/*.json "$OUT/" 2>/dev/null || true' EXIT
cp "$BASE/src/k2_factorized_overlap_sr20.py" "$RUN/"
cp "$BASE/src/finite_tau_matching_20site_exact.py" "$RUN/"
cp "$BASE/inputs/reconstruct_M100000_y59279.npz" "$RUN/"
cd "$RUN"
python k2_factorized_overlap_sr20.py   --cache reconstruct_M100000_y59279.npz   --dt 0.0125 --steps 20 --ns 1024 --trust 0.002 --shift 0.1   --out result.json
