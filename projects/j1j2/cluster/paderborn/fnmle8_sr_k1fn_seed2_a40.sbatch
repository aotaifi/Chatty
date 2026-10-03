#!/bin/bash
#SBATCH --job-name=chatty-fnmle-fn2
#SBATCH --partition=gpu_a40
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=05:00:00
#SBATCH --output=/pc2/users/h/hpcalot/chatty/j1j2/results/fnmle-fn2-%j.out
#SBATCH --error=/pc2/users/h/hpcalot/chatty/j1j2/results/fnmle-fn2-%j.err
set -euo pipefail
module load lang
module load Python/3.11.5-GCCcore-13.2.0
source /pc2/users/h/hpcalot/chatty/.venv-j1j2-py311/bin/activate
export JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false
BASE=/pc2/users/h/hpcalot/chatty/j1j2
H="$BASE/results/fnmle_sr_handoff_3506051"
RUN=$(mktemp -d "${TMPDIR:-/tmp}/fnmle-fn2.XXXXXX")
OUT="$BASE/results/fnmle_sr_k1fn_seed2_${SLURM_JOB_ID}"; mkdir -p "$OUT"
trap 'cp -f "$RUN"/*.npz "$OUT/" 2>/dev/null || true' EXIT
cp "$BASE/src/callable_fn_loop.py" "$BASE/src/vit8_callable_amplitude_adapter.py" "$RUN/"
cp "$H/vit8_fnmle_sr_eta001.mpack" "$H/vit8_fnmle_sr_handoff.npz" "$RUN/"
cd "$RUN"
python callable_fn_loop.py  --backend vit8_callable_amplitude_adapter.py --checkpoint vit8_fnmle_sr_eta001.mpack  --L 8 --J2 0.5  --threshold-npz vit8_fnmle_sr_handoff.npz  --threshold-states-key threshold_states --threshold-weights-key threshold_weights  --pool-npz vit8_fnmle_sr_handoff.npz  --pool-states-key pool_states --pool-weights-key pool_weights  --M 128 --beta-target 1.2 --burn-beta 0.4 --tau-max 0.025  --seed 13002 --out fnmle_sr_k1fn_seed13002.npz
