#!/bin/bash
#SBATCH --job-name=chatty-fnmle-i2tr
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:45:00
#SBATCH --output=/pc2/users/h/hpcalot/chatty/j1j2/results/fnmle-i2tr-%j.out
#SBATCH --error=/pc2/users/h/hpcalot/chatty/j1j2/results/fnmle-i2tr-%j.err
set -euo pipefail
module load lang
module load Python/3.11.5-GCCcore-13.2.0
source /pc2/users/h/hpcalot/chatty/.venv-j1j2-py311/bin/activate
export JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false TF_GPU_ALLOCATOR=cuda_malloc_async
BASE=/pc2/users/h/hpcalot/chatty/j1j2
RUN=$(mktemp -d "${TMPDIR:-/tmp}/fnmle-i2tr.XXXXXX"); OUT="$BASE/results/fnmle_sr_iter2_train_${SLURM_JOB_ID}"; mkdir -p "$OUT"
trap 'cp -f "$RUN"/*.mpack "$RUN"/*.npz "$RUN"/*.json "$OUT/" 2>/dev/null || true' EXIT
cp "$BASE/src/fnmle8_sr_iter2_train.py" "$RUN/"
cp "$BASE/inputs/vit_J2=0.50_N=8x8_k=0.mpack" "$BASE/inputs/krylov_phys8_a2_fixedT.npz" "$BASE/inputs/krylov_scaling_8x8_a1.20_tr4096_va2048.npz" "$RUN/"
cp "$BASE/results/fnmle_sr_handoff_3506051/vit8_fnmle_sr_eta001.mpack" "$BASE/results/fnmle_sr_handoff_3506051/vit8_fnmle_sr_handoff.npz" "$RUN/"
cp "$BASE/results/fnmle_sr_k1fn_3506052/fnmle_sr_k1fn_seed13001.npz" "$RUN/"
cp "$BASE/results/fnmle_sr_k1fn_seed2_3506054/fnmle_sr_k1fn_seed13002.npz" "$RUN/"
cd "$RUN"; python fnmle8_sr_iter2_train.py
