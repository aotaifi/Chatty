#!/bin/bash
#SBATCH --job-name=chatty-neighaudit
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=01:30:00
#SBATCH --output=/pc2/users/h/hpcalot/chatty/j1j2/results/neighaudit-%j.out
#SBATCH --error=/pc2/users/h/hpcalot/chatty/j1j2/results/neighaudit-%j.err
set -euo pipefail
module load lang
module load Python/3.11.5-GCCcore-13.2.0
source /pc2/users/h/hpcalot/chatty/.venv-j1j2-py311/bin/activate
export JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false
BASE=/pc2/users/h/hpcalot/chatty/j1j2
RUN=$(mktemp -d "${TMPDIR:-/tmp}/neighaudit.XXXXXX")
OUT="$BASE/results/fnmle8_neighbor_shell_audit_${SLURM_JOB_ID}"
mkdir -p "$OUT"
trap 'cp -f "$RUN"/neighbor_audit.json "$OUT/" 2>/dev/null || true' EXIT
cp "$BASE/src/fnmle8_neighbor_shell_audit.py" "$BASE/src/callable_fn_loop.py" "$BASE/src/vit8_callable_amplitude_adapter.py" "$RUN/"
cp "$BASE/inputs/vit_J2=0.50_N=8x8_k=0.mpack" "$RUN/a0.mpack"
cp "$BASE/results/fnmle_sr_handoff_3506051/vit8_fnmle_sr_eta001.mpack" "$RUN/a1.mpack"
cp "$BASE/results/fnmle_sr_iter2_train_3506071/vit8_fnmle_sr_iter2.mpack" "$RUN/a2.mpack"
cp "$BASE/results/fnmle_sr_handoff_3506051/vit8_fnmle_sr_handoff.npz" "$RUN/h1.npz"
cp "$BASE/results/fnmle_sr_k1fn_3506052/fnmle_sr_k1fn_seed13001.npz" "$RUN/r1.npz"
cp "$BASE/results/fnmle_sr_k1fn_seed2_3506054/fnmle_sr_k1fn_seed13002.npz" "$RUN/r2.npz"
cd "$RUN"
python fnmle8_neighbor_shell_audit.py --a0 a0.mpack --a1 a1.mpack --a2 a2.mpack --h1 h1.npz --rep1 r1.npz --rep2 r2.npz --out neighbor_audit.json --nbase 96
