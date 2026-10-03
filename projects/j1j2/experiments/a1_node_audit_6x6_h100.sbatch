#!/bin/bash
#SBATCH --job-name=j1j2-a1-node
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=hpcalot
#SBATCH --output=/pc2/users/h/hpcalot/chatty/j1j2/results/a1-node-audit-%j.out
#SBATCH --error=/pc2/users/h/hpcalot/chatty/j1j2/results/a1-node-audit-%j.err

set -euo pipefail
module load lang
module load Python/3.11.5-GCCcore-13.2.0
source /pc2/users/h/hpcalot/chatty/.venv-j1j2-py311/bin/activate
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

BASE=/pc2/users/h/hpcalot/chatty/j1j2
OUTDIR="$BASE/results/a1_node_audit_${SLURM_JOB_ID}"
mkdir -p "$OUTDIR"
RUN=$(mktemp -d "${TMPDIR:-/tmp}/chatty-j1j2-a1node.XXXXXX")

finish() {
  cp -f "$RUN"/*.npz "$OUTDIR/" 2>/dev/null || true
  cp -f "$RUN"/*.json "$OUTDIR/" 2>/dev/null || true
  cp -f "$RUN"/*.msgpack "$OUTDIR/" 2>/dev/null || true
  cp -f "$RUN"/a1_node_audit_6x6.py "$OUTDIR/" 2>/dev/null || true
}
trap finish EXIT

cp "$BASE/src/a1_node_audit_6x6.py" "$RUN/"
cp "$BASE/src/gfmc_6x6_gr_population_size.py" "$RUN/"
cp "$BASE/inputs/"* "$RUN/"
cd "$RUN"

{
  echo "job_id=${SLURM_JOB_ID}"
  echo "host=$(hostname)"
  echo "started=$(date -Is)"
  nvidia-smi -L
  python - <<'PY'
import jax
print("backend",jax.default_backend(),"devices",jax.devices())
PY
} > "$OUTDIR/run_metadata.txt"

python a1_node_audit_6x6.py
echo "finished=$(date -Is)" >> "$OUTDIR/run_metadata.txt"
