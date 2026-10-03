#!/bin/bash
#SBATCH --job-name=chatty-frozenhfn8
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=03:00:00
#SBATCH --output=/pc2/users/h/hpcalot/chatty/j1j2/results/frozenhfn8-%j.out
#SBATCH --error=/pc2/users/h/hpcalot/chatty/j1j2/results/frozenhfn8-%j.err
set -euo pipefail
module load lang
module load Python/3.11.5-GCCcore-13.2.0
source /pc2/users/h/hpcalot/chatty/.venv-j1j2-py311/bin/activate
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false
BASE=/pc2/users/h/hpcalot/chatty/j1j2
RUN=$(mktemp -d "${TMPDIR:-/tmp}/frozenhfn8.XXXXXX")
OUT="$BASE/results/frozen_hfn_sr8_${SLURM_JOB_ID}"
mkdir -p "$OUT"
trap 'cp -f "$RUN"/*.json "$RUN"/*.mpack "$OUT/" 2>/dev/null || true' EXIT
cp "$BASE/src/frozen_hfn_sr8_one_step.py" "$BASE/src/callable_fn_loop.py" "$RUN/"
cp "$BASE/results/fnmle_sr_handoff_3506051/vit8_fnmle_sr_eta001.mpack" "$RUN/a1.mpack"
cp "$BASE/results/fnmle_sr_handoff_3506051/vit8_fnmle_sr_handoff.npz" "$RUN/h1.npz"
cd "$RUN"
python frozen_hfn_sr8_one_step.py --a1 a1.mpack --handoff1 h1.npz --shift 0.1 --ntrain 256 --nval 256 --fisher-n 256 --out-ck frozen_hfn_sr8_candidate.mpack --out-json frozen_hfn_sr8.json
