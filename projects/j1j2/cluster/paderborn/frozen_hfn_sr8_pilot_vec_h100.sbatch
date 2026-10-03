#!/bin/bash
#SBATCH --job-name=chatty-hfn8srv
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/pc2/users/h/hpcalot/chatty/j1j2/results/hfn8srv-%j.out
#SBATCH --error=/pc2/users/h/hpcalot/chatty/j1j2/results/hfn8srv-%j.err
set -euo pipefail
module load lang
module load Python/3.11.5-GCCcore-13.2.0
source /pc2/users/h/hpcalot/chatty/.venv-j1j2-py311/bin/activate
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false
BASE=/pc2/users/h/hpcalot/chatty/j1j2
RUN=$(mktemp -d "${TMPDIR:-/tmp}/hfn8srv.XXXXXX")
OUT="$BASE/results/frozen_hfn_sr8_vec_${SLURM_JOB_ID}"
mkdir -p "$OUT"
trap 'cp -f "$RUN"/*.json "$RUN"/*.npz "$RUN"/*.mpack "$OUT/" 2>/dev/null || true' EXIT
cp "$BASE/src/frozen_hfn_sr8_pilot_vec.py" "$BASE/src/callable_fn_loop.py" "$RUN/"
cp "$BASE/results/fnmle_sr_handoff_3506051/vit8_fnmle_sr_eta001.mpack" "$RUN/a1.mpack"
cp "$BASE/results/fnmle_sr_handoff_3506051/vit8_fnmle_sr_handoff.npz" "$RUN/h1.npz"
cd "$RUN"
python frozen_hfn_sr8_pilot_vec.py --a1 a1.mpack --handoff1 h1.npz   --out-ck frozen_hfn_sr8_vec.mpack --out-json frozen_hfn_sr8_vec.json   --out-freeze frozen_hfn_sr8_signcache.npz --ntrain 256 --nval 256 --fisher-n 256 --eval-batch 4096
