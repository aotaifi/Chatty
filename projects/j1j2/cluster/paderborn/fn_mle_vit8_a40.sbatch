#!/bin/bash
#SBATCH --job-name=chatty-fnmle-vit8-a40
#SBATCH --partition=gpu_a40
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=alyotaifi@gmail.com
#SBATCH --output=/pc2/users/h/hpcalot/chatty/j1j2/results/fnmle-vit8-a40-%j.out
#SBATCH --error=/pc2/users/h/hpcalot/chatty/j1j2/results/fnmle-vit8-a40-%j.err
set -euo pipefail
module load lang
module load Python/3.11.5-GCCcore-13.2.0
source /pc2/users/h/hpcalot/chatty/.venv-j1j2-py311/bin/activate
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
BASE=/pc2/users/h/hpcalot/chatty/j1j2
OUT="$BASE/results/fnmle_vit8_${SLURM_JOB_ID}"
mkdir -p "$OUT"
RUN=$(mktemp -d "${TMPDIR:-/tmp}/fnmle-vit8.XXXXXX")
finish(){ cp -f "$RUN"/*.json "$OUT/" 2>/dev/null || true; cp -f "$RUN"/*.mpack "$OUT/" 2>/dev/null || true; cp -f "$RUN"/*.py "$OUT/" 2>/dev/null || true; }
trap finish EXIT
cp "$BASE/src/train_fn_mle_vit8.py" "$RUN/"
cp "$BASE/inputs/vit_J2=0.50_N=8x8_k=0.mpack" "$RUN/"
cp "$BASE/inputs/gfmc_8x8_grscale_M128_seed10501.npz" "$RUN/"
cp "$BASE/inputs/gfmc_8x8_grscale_M128_seed10502.npz" "$RUN/"
cp "$BASE/inputs/krylov_phys8_a2_fixedT.npz" "$RUN/"
cd "$RUN"
export MLE_BATCH=512 MLE_STEPS=160 MLE_LR=2e-5 MLE_EVAL_EVERY=10
export OUT_CK=vit8_fnmle_rep1_best.mpack
export OUT_JSON=vit8_fnmle_rep1_metrics.json
python train_fn_mle_vit8.py
