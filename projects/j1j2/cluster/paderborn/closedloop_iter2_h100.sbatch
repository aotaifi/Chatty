#!/bin/bash
#SBATCH --job-name=chatty-j1j2-iter2
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=hpcalot
#SBATCH --output=/pc2/users/h/hpcalot/chatty/j1j2/results/closedloop-iter2-%j.out
#SBATCH --error=/pc2/users/h/hpcalot/chatty/j1j2/results/closedloop-iter2-%j.err

set -euo pipefail
module load lang
module load Python/3.11.5-GCCcore-13.2.0
source /pc2/users/h/hpcalot/chatty/.venv-j1j2-py311/bin/activate
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

BASE=/pc2/users/h/hpcalot/chatty/j1j2
OUTDIR="$BASE/results/closedloop_iter2_${SLURM_JOB_ID}"
mkdir -p "$OUTDIR"
RUN=$(mktemp -d "${TMPDIR:-/tmp}/chatty-j1j2-iter2.XXXXXX")

finish() {
  cp -f "$RUN"/*.npz "$OUTDIR/" 2>/dev/null || true
  cp -f "$RUN"/*.json "$OUTDIR/" 2>/dev/null || true
  cp -f "$RUN"/gfmc_6x6_closedloop_iter2.py "$OUTDIR/" 2>/dev/null || true
  cp -f "$RUN"/gfmc_6x6_gr_population_size.py "$OUTDIR/" 2>/dev/null || true
}
trap finish EXIT

cp "$BASE/src/gfmc_6x6_closedloop_iter2.py" "$RUN/"
cp "$BASE/src/gfmc_6x6_gr_population_size.py" "$RUN/"
cp "$BASE/inputs/"* "$RUN/"
cd "$RUN"

{
  echo "job_id=${SLURM_JOB_ID}"
  echo "host=$(hostname)"
  echo "started=$(date -Is)"
  echo "cpus=${SLURM_CPUS_PER_TASK}"
  nvidia-smi -L
  python - <<'PY'
import jax, flax, netket, numpy, numba, scipy
print("jax",jax.__version__,"backend",jax.default_backend(),"devices",jax.devices())
print("flax",flax.__version__,"netket",netket.__version__)
print("numpy",numpy.__version__,"numba",numba.__version__,"scipy",scipy.__version__)
PY
  sha256sum vit_J2=0.50_N=6x6_k=0.mpack *.npz
} > "$OUTDIR/run_metadata.txt"

python gfmc_6x6_closedloop_iter2.py

echo "finished=$(date -Is)" >> "$OUTDIR/run_metadata.txt"
