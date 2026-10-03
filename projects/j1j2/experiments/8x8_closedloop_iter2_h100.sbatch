#!/bin/bash
#SBATCH --job-name=chatty-j1j2-8x8-iter2
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=alyotaifi@gmail.com
#SBATCH --output=/pc2/users/h/hpcalot/chatty/j1j2/results/8x8-iter2-%j.out
#SBATCH --error=/pc2/users/h/hpcalot/chatty/j1j2/results/8x8-iter2-%j.err
set -euo pipefail
: "${PARENT_JOB:?submit with --export=ALL,PARENT_JOB=<8x8 first-FN job>}"
module load lang
module load Python/3.11.5-GCCcore-13.2.0
source /pc2/users/h/hpcalot/chatty/.venv-j1j2-py311/bin/activate
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
BASE=/pc2/users/h/hpcalot/chatty/j1j2
PARENT="$BASE/results/8x8_krylov_${PARENT_JOB}"
OUT="$BASE/results/8x8_closedloop_iter2_${SLURM_JOB_ID}"
mkdir -p "$OUT"
RUN=$(mktemp -d "${TMPDIR:-/tmp}/chatty-8x8-iter2.XXXXXX")
finish(){
  cp -f "$RUN"/*.npz "$OUT/" 2>/dev/null || true
  cp -f "$RUN"/gfmc_8x8_closedloop_iter2.py "$OUT/" 2>/dev/null || true
  cp -f "$RUN"/gfmc_8x8_gr_population_size.py "$OUT/" 2>/dev/null || true
}
trap finish EXIT
cp "$BASE/src/gfmc_8x8_closedloop_iter2.py" "$RUN/"
cp "$BASE/src/gfmc_8x8_gr_population_size.py" "$RUN/"
cp "$BASE/inputs/vit_J2=0.50_N=8x8_k=0.mpack" "$RUN/"
cp "$BASE/inputs/krylov_phys8_a2_fixedT.npz" "$RUN/"
cp "$BASE/inputs/krylov_scaling_8x8_a1.20_tr4096_va2048.npz" "$RUN/"
cp "$PARENT/gr_8x8_population_scaling_M128.npz" "$RUN/"
cd "$RUN"
{
  echo "job=${SLURM_JOB_ID} parent=${PARENT_JOB} host=$(hostname) started=$(date -Is)"
  nvidia-smi -L
  python - <<'PY'
import jax
print("backend",jax.default_backend(),"devices",jax.devices())
PY
} > "$OUT/run_metadata.txt"
python gfmc_8x8_closedloop_iter2.py
echo "finished=$(date -Is)" >> "$OUT/run_metadata.txt"
