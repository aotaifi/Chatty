#!/bin/bash
#SBATCH --job-name=chatty-fnmle-srls
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=alyotaifi@gmail.com
#SBATCH --output=/pc2/users/h/hpcalot/chatty/j1j2/results/fnmle-srls-%j.out
#SBATCH --error=/pc2/users/h/hpcalot/chatty/j1j2/results/fnmle-srls-%j.err
set -euo pipefail
module load lang
module load Python/3.11.5-GCCcore-13.2.0
source /pc2/users/h/hpcalot/chatty/.venv-j1j2-py311/bin/activate
export JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false TF_GPU_ALLOCATOR=cuda_malloc_async
export SR_NS=512
BASE=/pc2/users/h/hpcalot/chatty/j1j2
RUN=$(mktemp -d "${TMPDIR:-/tmp}/fnmle-srls.XXXXXX")
OUT="$BASE/results/fnmle_srls_${SLURM_JOB_ID}"; mkdir -p "$OUT"
trap 'cp -f "$RUN"/*.json "$OUT/" 2>/dev/null || true' EXIT
cp "$BASE/src/fnmle8_sr_lambda1_linesearch.py" "$RUN/"
cp "$BASE/inputs/vit_J2=0.50_N=8x8_k=0.mpack" "$RUN/"
cp "$BASE/inputs/gfmc_8x8_grscale_M128_seed10501.npz" "$RUN/"
cp "$BASE/inputs/gfmc_8x8_grscale_M128_seed10502.npz" "$RUN/"
cp "$BASE/inputs/krylov_phys8_a2_fixedT.npz" "$RUN/"
cd "$RUN"
python fnmle8_sr_lambda1_linesearch.py
