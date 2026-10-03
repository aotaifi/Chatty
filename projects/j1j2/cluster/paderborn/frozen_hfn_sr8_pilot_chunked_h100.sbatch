#!/bin/bash
#SBATCH --job-name=chatty-hfn8src
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=/pc2/users/h/hpcalot/chatty/j1j2/results/hfn8src-%j.out
#SBATCH --error=/pc2/users/h/hpcalot/chatty/j1j2/results/hfn8src-%j.err
set -euo pipefail
module load lang
module load Python/3.11.5-GCCcore-13.2.0
source /pc2/users/h/hpcalot/chatty/.venv-j1j2-py311/bin/activate
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false
BASE=/pc2/users/h/hpcalot/chatty/j1j2
RUN=$(mktemp -d "${TMPDIR:-/tmp}/hfn8src.XXXXXX")
OUT="$BASE/results/frozen_hfn_sr8_chunked_${SLURM_JOB_ID}"
mkdir -p "$OUT"
trap 'cp -f "$RUN"/*.json "$RUN"/*.mpack "$OUT/" 2>/dev/null || true' EXIT
cp "$BASE/src/frozen_hfn_sr8_pilot_chunked.py" "$BASE/src/callable_fn_loop.py" "$RUN/"
cp "$BASE/results/fnmle_sr_handoff_3506051/vit8_fnmle_sr_eta001.mpack" "$RUN/a1.mpack"
cp "$BASE/results/frozen_hfn_sr8_3554914/frozen_hfn_sr8_edges.npz" "$RUN/edges.npz"
cd "$RUN"
python frozen_hfn_sr8_pilot_chunked.py --a1 a1.mpack --out-ck frozen_hfn_sr8_candidate.mpack --out-json frozen_hfn_sr8_pilot.json --edge-cache-in edges.npz --nsamp 512 --steps 5 --shift 0.1 --trust 0.002 --qgt-chunk 64 --cg-maxiter 80
