#!/bin/bash
#SBATCH --job-name=chatty-k1rep-6x6
#SBATCH --partition=normal
#SBATCH --cpus-per-task=128
#SBATCH --mem=200G
#SBATCH --time=06:00:00
#SBATCH --output=/pc2/users/h/hpcalot/chatty/j1j2/results/k1rep-6x6-%j.out
#SBATCH --error=/pc2/users/h/hpcalot/chatty/j1j2/results/k1rep-6x6-%j.err
# Usage: sbatch k1_repeat_fixed_amp_6x6_cpu.sbatch [script args...]
#   pilot: sbatch --time=00:30:00 k1_repeat_fixed_amp_6x6_cpu.sbatch --ntrain-chains 4 --ntrain-rounds 2 --neval 4 --nab 64 --nab-k3 8
set -euo pipefail
module load lang
module load Python/3.11.5-GCCcore-13.2.0
source /pc2/users/h/hpcalot/chatty/.venv-j1j2-py311/bin/activate
export JAX_PLATFORMS=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
BASE=/pc2/users/h/hpcalot/chatty/j1j2
SRC=$BASE/k1_repeat_fixed_amp_6x6
OUT="$BASE/results/k1_repeat_fixed_amp_6x6_${SLURM_JOB_ID}"
mkdir -p "$OUT"
RUN=$(mktemp -d "${TMPDIR:-/tmp}/chatty-k1rep.XXXXXX")
finish(){ cp -f "$RUN"/k1_repeat_fixed_amp_6x6*.json "$RUN"/k1_repeat_fixed_amp_6x6*.npz "$OUT/" 2>/dev/null || true; }
trap finish EXIT
cp "$SRC/k1_repeat_fixed_amp_6x6.py" "$OUT/"
cp "$SRC/k1_repeat_fixed_amp_6x6.py" "$RUN/"
cp "$BASE/inputs/vit_J2=0.50_N=6x6_k=0.mpack" "$BASE/inputs/energy_krylov_vs_vit_6x6_indep.npz" \
   "$BASE/inputs/krylov_scaling_6x6_a1.20_tr4096_va2048.npz" "$BASE/inputs/krylov_scaling_6x6_a1.20_tr4096_va2048.json" "$RUN/"
cd "$RUN"
{ echo "job_id=${SLURM_JOB_ID}"; echo "host=$(hostname)"; echo "args=$*"; echo "started=$(date -Is)"; lscpu | head -20; } > "$OUT/run_metadata.txt"
python k1_repeat_fixed_amp_6x6.py "$@"
echo "finished=$(date -Is)" >> "$OUT/run_metadata.txt"
