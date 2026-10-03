#!/bin/bash
#SBATCH --job-name=chatty-sr-prod-build
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=alyotaifi@gmail.com
#SBATCH --output=/pc2/users/h/hpcalot/chatty/j1j2/results/sr-prod-build-%j.out
#SBATCH --error=/pc2/users/h/hpcalot/chatty/j1j2/results/sr-prod-build-%j.err
set -euo pipefail
module load lang
module load Python/3.11.5-GCCcore-13.2.0
source /pc2/users/h/hpcalot/chatty/.venv-j1j2-py311/bin/activate
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export TF_GPU_ALLOCATOR=cuda_malloc_async
BASE=/pc2/users/h/hpcalot/chatty/j1j2
OUT="$BASE/results/sr_prod_build_${SLURM_JOB_ID}"
mkdir -p "$OUT"
RUN=$(mktemp -d "${TMPDIR:-/tmp}/sr-prod-build.XXXXXX")
finish(){
  cp -f "$RUN"/*.mpack "$RUN"/*.npz "$RUN"/*.json "$RUN"/*.py "$OUT/" 2>/dev/null || true
}
trap finish EXIT
cp "$BASE/src/fnmle8_sr_production_checkpoint.py" "$RUN/"
cp "$BASE/src/audit_sr_k1_handoff_8x8.py" "$RUN/"
cp "$BASE/src/vit8_callable_amplitude_adapter.py" "$RUN/"
cp "$BASE/src/callable_fn_loop.py" "$RUN/"
cp "$BASE/inputs/vit_J2=0.50_N=8x8_k=0.mpack" "$RUN/"
cp "$BASE/inputs/krylov_phys8_a2_fixedT.npz" "$RUN/"
cp "$BASE/inputs/krylov_scaling_8x8_a1.20_tr4096_va2048.npz" "$RUN/"
cp "$BASE/inputs/gfmc_8x8_grscale_M128_seed10501.npz" "$RUN/"
cp "$BASE/inputs/gfmc_8x8_grscale_M128_seed10502.npz" "$RUN/"
cd "$RUN"
python fnmle8_sr_production_checkpoint.py
python audit_sr_k1_handoff_8x8.py
