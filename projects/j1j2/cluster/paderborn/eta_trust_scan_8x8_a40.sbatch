#!/bin/bash
#SBATCH --job-name=chatty-eta-scan8
#SBATCH --partition=gpu_a40
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=alyotaifi@gmail.com
#SBATCH --output=/pc2/users/h/hpcalot/chatty/j1j2/results/eta-scan8-%j.out
#SBATCH --error=/pc2/users/h/hpcalot/chatty/j1j2/results/eta-scan8-%j.err
set -euo pipefail
module load lang
module load Python/3.11.5-GCCcore-13.2.0
source /pc2/users/h/hpcalot/chatty/.venv-j1j2-py311/bin/activate
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false
BASE=/pc2/users/h/hpcalot/chatty/j1j2
OUT="$BASE/results/eta_scan8_${SLURM_JOB_ID}"
mkdir -p "$OUT"
RUN=$(mktemp -d "${TMPDIR:-/tmp}/chatty-eta8.XXXXXX")
finish(){ cp -f "$RUN"/eta_trust_scan_8x8.* "$OUT/" 2>/dev/null || true; }
trap finish EXIT
cp "$BASE/src/eta_trust_scan_8x8.py" "$BASE/src/callable_fn_loop.py" "$BASE/src/learned_fn8_callable_amplitude_adapter.py" "$RUN/"
cp "$BASE/inputs/fn_residual_cnn_8x8.mpack" "$BASE/inputs/learned_fn_handoff_8x8.npz" "$BASE/inputs/krylov_scaling_8x8_a1.20_tr4096_va2048.npz" "$BASE/inputs/vit_J2=0.50_N=8x8_k=0.mpack" "$RUN/"
cd "$RUN"
export BASE_VIT_CHECKPOINT="$RUN/vit_J2=0.50_N=8x8_k=0.mpack"
python eta_trust_scan_8x8.py
