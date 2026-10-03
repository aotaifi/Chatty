#!/bin/bash
#SBATCH --job-name=chatty-i2cand
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=01:30:00
#SBATCH --output=/pc2/users/h/hpcalot/chatty/j1j2/results/i2cand-%j.out
#SBATCH --error=/pc2/users/h/hpcalot/chatty/j1j2/results/i2cand-%j.err
set -euo pipefail
module load lang
module load Python/3.11.5-GCCcore-13.2.0
source /pc2/users/h/hpcalot/chatty/.venv-j1j2-py311/bin/activate
export JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false
BASE=/pc2/users/h/hpcalot/chatty/j1j2
RUN=$(mktemp -d "${TMPDIR:-/tmp}/i2cand.XXXXXX")
OUT="$BASE/results/fnmle_sr_iter2_candidates_${SLURM_JOB_ID}"; mkdir -p "$OUT"
trap 'cp -f "$RUN"/vit8_fnmle_sr_iter2_eta*.mpack "$OUT/" 2>/dev/null || true; cp -f "$RUN"/vit8_fnmle_sr_iter2_train.json "$OUT/" 2>/dev/null || true' EXIT
cp "$BASE/src/fnmle8_sr_iter2_save_candidates.py" "$BASE/src/fnmle8_sr_iter2_train.py" "$RUN/"
cp "$BASE/inputs/vit_J2=0.50_N=8x8_k=0.mpack" "$RUN/"
cp "$BASE/results/fnmle_sr_handoff_3506051/vit8_fnmle_sr_eta001.mpack" "$RUN/"
cp "$BASE/results/fnmle_sr_k1fn_3506052/fnmle_sr_k1fn_seed13001.npz" "$RUN/"
cp "$BASE/results/fnmle_sr_k1fn_seed2_3506054/fnmle_sr_k1fn_seed13002.npz" "$RUN/"
cp "$BASE/inputs/krylov_phys8_a2_fixedT.npz" "$RUN/"
cp "$BASE/inputs/krylov_scaling_8x8_a1.20_tr4096_va2048.npz" "$RUN/"
cp "$BASE/results/fnmle_sr_handoff_3506051/vit8_fnmle_sr_handoff.npz" "$RUN/"
cd "$RUN"
python fnmle8_sr_iter2_save_candidates.py
