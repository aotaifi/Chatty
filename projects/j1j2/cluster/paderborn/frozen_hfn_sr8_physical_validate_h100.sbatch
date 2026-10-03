#!/bin/bash
#SBATCH --job-name=chatty-hfn8-phys
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=03:00:00
#SBATCH --output=/pc2/users/h/hpcalot/chatty/j1j2/results/hfn8phys-%j.out
#SBATCH --error=/pc2/users/h/hpcalot/chatty/j1j2/results/hfn8phys-%j.err
set -euo pipefail
: "${CAND_JOB:?CAND_JOB required}"
: "${SEED:?SEED required}"
module load lang
module load Python/3.11.5-GCCcore-13.2.0
source /pc2/users/h/hpcalot/chatty/.venv-j1j2-py311/bin/activate
export JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false
BASE=/pc2/users/h/hpcalot/chatty/j1j2
RUN=$(mktemp -d "${TMPDIR:-/tmp}/hfn8phys.XXXXXX")
OUT="$BASE/results/frozen_hfn_phys_${CAND_JOB}_${SLURM_JOB_ID}"; mkdir -p "$OUT"
trap 'cp -f "$RUN"/*.npz "$OUT/" 2>/dev/null || true' EXIT
cp "$BASE/src/frozen_s1_a2_fn8.py" "$BASE/src/callable_fn_loop.py" "$BASE/src/vit8_callable_amplitude_adapter.py" "$RUN/"
cp "$BASE/results/fnmle_sr_handoff_3506051/vit8_fnmle_sr_eta001.mpack" "$RUN/a1.mpack"
cp "$BASE/results/frozen_hfn_sr8_${CAND_JOB}/frozen_hfn_sr8.mpack" "$RUN/a2.mpack"
cp "$BASE/results/fnmle_sr_handoff_3506051/vit8_fnmle_sr_handoff.npz" "$RUN/h1.npz"
cd "$RUN"
python frozen_s1_a2_fn8.py --a1 a1.mpack --a2 a2.mpack --handoff1 h1.npz --seed "$SEED" --out "frozen_hfn_phys_seed${SEED}.npz"
