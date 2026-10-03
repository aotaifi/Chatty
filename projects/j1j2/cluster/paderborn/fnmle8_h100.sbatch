#!/bin/bash
#SBATCH --job-name=chatty-fnmle8-h100
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=alyotaifi@gmail.com
#SBATCH --output=/pc2/users/h/hpcalot/chatty/j1j2/results/fnmle8-h100-%j.out
#SBATCH --error=/pc2/users/h/hpcalot/chatty/j1j2/results/fnmle8-h100-%j.err
set -euo pipefail
module load lang
module load Python/3.11.5-GCCcore-13.2.0
source /pc2/users/h/hpcalot/chatty/.venv-j1j2-py311/bin/activate
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
BASE=/pc2/users/h/hpcalot/chatty/j1j2
OUT="$BASE/results/fnmle8_${SLURM_JOB_ID}"
mkdir -p "$OUT"
RUN=$(mktemp -d "${TMPDIR:-/tmp}/chatty-fnmle8.XXXXXX")
finish(){ cp -f "$RUN"/*.mpack "$RUN"/*.json "$RUN"/*.py "$OUT/" 2>/dev/null || true; }
trap finish EXIT
cp "$BASE/src/train_fn_mle_vit8.py" "$BASE/src/diag_mle_vit8_node.py" "$BASE/src/vit8_callable_amplitude_adapter.py" "$BASE/src/callable_fn_loop.py" "$RUN/"
cp "$BASE/inputs/vit_J2=0.50_N=8x8_k=0.mpack" "$BASE/inputs/krylov_phys8_a2_fixedT.npz" "$BASE/inputs/krylov_scaling_8x8_a1.20_tr4096_va2048.npz" "$RUN/"
cp "$BASE/results/8x8_krylov_3471544/gfmc_8x8_grscale_M128_seed10501.npz" "$BASE/results/8x8_krylov_3471544/gfmc_8x8_grscale_M128_seed10502.npz" "$RUN/"
cd "$RUN"
export MLE_BATCH=512 MLE_STEPS=160 MLE_LR=2e-5 MLE_EVAL_EVERY=10
python train_fn_mle_vit8.py
BASE_CK=vit_J2=0.50_N=8x8_k=0.mpack NEW_CK=vit8_fnmle_rep1_best.mpack python diag_mle_vit8_node.py
