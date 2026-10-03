#!/bin/bash
#SBATCH --job-name=chatty-fsE-sr
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --output=/pc2/users/h/hpcalot/chatty/j1j2/results/fixedsign-esr-%j.out
#SBATCH --error=/pc2/users/h/hpcalot/chatty/j1j2/results/fixedsign-esr-%j.err
set -euo pipefail
module load lang
module load Python/3.11.5-GCCcore-13.2.0
source /pc2/users/h/hpcalot/chatty/.venv-j1j2-py311/bin/activate
export JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false
BASE=/pc2/users/h/hpcalot/chatty/j1j2
RUN=$(mktemp -d "${TMPDIR:-/tmp}/fixedsign-esr.XXXXXX")
OUT="$BASE/results/fixedsign_energy_sr8_${SLURM_JOB_ID}"
mkdir -p "$OUT"
trap 'cp -f "$RUN"/fixedsign_energy_sr.* "$OUT/" 2>/dev/null || true' EXIT
cp "$BASE/src/fixedsign_energy_sr8.py" "$BASE/src/callable_fn_loop.py" "$BASE/src/frozen_s1_a2_fn8.py" "$BASE/src/vit8_callable_amplitude_adapter.py" "$RUN/"
cp "$BASE/results/fnmle_sr_handoff_3506051/vit8_fnmle_sr_eta001.mpack" "$RUN/a1.mpack"
cp "$BASE/results/fnmle_sr_handoff_3506051/vit8_fnmle_sr_handoff.npz" "$RUN/h1.npz"
cd "$RUN"
python fixedsign_energy_sr8.py   --a1 a1.mpack --handoff1 h1.npz   --out-ck fixedsign_energy_sr.mpack --out-json fixedsign_energy_sr.json   --ntrain 256 --nval 256 --fisher-n 256 --shift 1.0 --trust-rms 0.05
