#!/bin/bash
#SBATCH --job-name=chatty-mixefn-sr8
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/pc2/users/h/hpcalot/chatty/j1j2/results/mixefn-sr8-%j.out
#SBATCH --error=/pc2/users/h/hpcalot/chatty/j1j2/results/mixefn-sr8-%j.err
set -euo pipefail
module load lang
module load Python/3.11.5-GCCcore-13.2.0
source /pc2/users/h/hpcalot/chatty/.venv-j1j2-py311/bin/activate
export JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false TF_GPU_ALLOCATOR=cuda_malloc_async
BASE=/pc2/users/h/hpcalot/chatty/j1j2
RUN=$(mktemp -d "${TMPDIR:-/tmp}/mixefn-sr8.XXXXXX")
OUT="$BASE/results/mixed_efn_force_sr8_${SLURM_JOB_ID}"
mkdir -p "$OUT"
trap 'cp -f "$RUN"/*.mpack "$RUN"/*.json "$RUN"/*.npz "$OUT/" 2>/dev/null || true' EXIT
cp "$BASE/src/mixed_efn_force_sr8.py" "$BASE/src/callable_fn_loop.py" "$BASE/src/vit8_callable_amplitude_adapter.py" "$RUN/"
cp "$BASE/results/fnmle_sr_handoff_3506051/vit8_fnmle_sr_eta001.mpack" "$RUN/a1.mpack"
cp "$BASE/results/fnmle_sr_handoff_3506051/vit8_fnmle_sr_handoff.npz" "$RUN/h1.npz"
cp "$BASE/results/fnmle_sr_k1fn_3506052/fnmle_sr_k1fn_seed13001.npz" "$RUN/r1.npz"
cp "$BASE/results/fnmle_sr_k1fn_seed2_3506054/fnmle_sr_k1fn_seed13002.npz" "$RUN/r2.npz"
cd "$RUN"
python mixed_efn_force_sr8.py --a1 a1.mpack --handoff1 h1.npz --rep1 r1.npz --rep2 r2.npz \
  --out-ck mixed_efn_force_sr8.mpack --out-json mixed_efn_force_sr8.json \
  --nforce 128 --fisher-n 256 --shift 1.0 --grad-base-batch 8 --edge-cache edge_cache.npz
