#!/bin/bash
#SBATCH --job-name=chatty-k1ab
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=/pc2/users/h/hpcalot/chatty/j1j2/results/k1ab-%j.out
#SBATCH --error=/pc2/users/h/hpcalot/chatty/j1j2/results/k1ab-%j.err
set -euo pipefail
: "${MODE:?MODE required}"
module load lang
module load Python/3.11.5-GCCcore-13.2.0
source /pc2/users/h/hpcalot/chatty/.venv-j1j2-py311/bin/activate
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false
BASE=/pc2/users/h/hpcalot/chatty/j1j2
RUN=$(mktemp -d "${TMPDIR:-/tmp}/k1ab.XXXXXX")
OUT="$BASE/results/k1_physical_vs_a1_${MODE}_${SLURM_JOB_ID}"
mkdir -p "$OUT"
trap 'cp -f "$RUN"/*.json "$OUT/" 2>/dev/null || true' EXIT
cp "$BASE/src/current_sign_energy_threshold8_splitamp.py" "$BASE/src/callable_fn_loop.py" "$BASE/src/vit8_callable_amplitude_adapter.py" "$RUN/"
cp "$BASE/results/fnmle_sr_handoff_3506051/vit8_fnmle_sr_eta001.mpack" "$RUN/a1.mpack"
cp "$BASE/results/fnmle_sr_handoff_3506051/vit8_fnmle_sr_handoff.npz" "$RUN/h1.npz"
if [ "$MODE" = "physical" ]; then
  cp "$BASE/results/fixedsign_energy_sr8_3509464/fixedsign_energy_sr.mpack" "$RUN/cand.mpack"
elif [ "$MODE" = "fn" ]; then
  cp "$BASE/results/frozen_hfn_sr8_chunked_3565752/frozen_hfn_sr8_candidate.mpack" "$RUN/cand.mpack"
else
  cp "$RUN/a1.mpack" "$RUN/cand.mpack"
fi
cd "$RUN"
python current_sign_energy_threshold8_splitamp.py --parent-a1 a1.mpack --amp cand.mpack --handoff1 h1.npz --label "$MODE" --ntrain 128 --nval 128 --out "k1_${MODE}.json"
