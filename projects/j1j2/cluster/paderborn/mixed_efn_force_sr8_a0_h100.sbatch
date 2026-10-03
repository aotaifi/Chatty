#!/bin/bash
#SBATCH --job-name=chatty-mixefn-a0
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/pc2/users/h/hpcalot/chatty/j1j2/results/mixefn-a0-%j.out
#SBATCH --error=/pc2/users/h/hpcalot/chatty/j1j2/results/mixefn-a0-%j.err
set -euo pipefail
module load lang
module load Python/3.11.5-GCCcore-13.2.0
source /pc2/users/h/hpcalot/chatty/.venv-j1j2-py311/bin/activate
export JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false TF_GPU_ALLOCATOR=cuda_malloc_async
BASE=/pc2/users/h/hpcalot/chatty/j1j2
RUN=$(mktemp -d "${TMPDIR:-/tmp}/mixefn-a0.XXXXXX")
OUT="$BASE/results/mixed_efn_force_sr8_a0_${SLURM_JOB_ID}"
mkdir -p "$OUT"
trap 'cp -f "$RUN"/*.mpack "$RUN"/*.json "$RUN"/*.npz "$OUT/" 2>/dev/null || true' EXIT
cp "$BASE/src/mixed_efn_force_sr8.py" "$BASE/src/callable_fn_loop.py" "$RUN/"
cp "$BASE/inputs/vit_J2=0.50_N=8x8_k=0.mpack" "$RUN/a0.mpack"
cp "$BASE/results/8x8_krylov_3471544/krylov_phys8_a2_fixedT.npz" "$RUN/pool0.npz"
cp "$BASE/results/8x8_krylov_3471544/gfmc_8x8_grscale_M128_seed10501.npz" "$RUN/r1.npz"
cp "$BASE/results/8x8_krylov_3471544/gfmc_8x8_grscale_M128_seed10502.npz" "$RUN/r2.npz"
cd "$RUN"
python mixed_efn_force_sr8.py --a1 a0.mpack --handoff1 pool0.npz --rep1 r1.npz --rep2 r2.npz \
  --threshold -28.37107876288694 --pool-states-key states \
  --out-ck mixed_efn_force_sr8_a0.mpack --out-json mixed_efn_force_sr8_a0.json \
  --nforce 128 --fisher-n 256 --shift 1.0 --grad-base-batch 8 --edge-cache edge_cache.npz
