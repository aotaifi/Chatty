[Reading 35 lines from start (total: 35 lines, 0 remaining)]

#!/bin/bash
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=alyotaifi@gmail.com
set -euo pipefail
module load lang
module load Python/3.11.5-GCCcore-13.2.0
source /pc2/users/h/hpcalot/chatty/.venv-j1j2-py311/bin/activate
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
BASE=/pc2/users/h/hpcalot/chatty/j1j2
#SBATCH --job-name=chatty-j1j2-8x8-K
#SBATCH --output=/pc2/users/h/hpcalot/chatty/j1j2/results/8x8-krylov-%j.out
#SBATCH --error=/pc2/users/h/hpcalot/chatty/j1j2/results/8x8-krylov-%j.err
OUT="$BASE/results/8x8_krylov_${SLURM_JOB_ID}"
mkdir -p "$OUT"
RUN=$(mktemp -d "${TMPDIR:-/tmp}/chatty-8x8-krylov.XXXXXX")
finish(){ cp -f "$RUN"/*.npz "$OUT/" 2>/dev/null || true; cp -f "$RUN"/gfmc_8x8_gr_population_size.py "$OUT/" 2>/dev/null || true; }
trap finish EXIT
cp "$BASE/src/gfmc_8x8_gr_population_size.py" "$RUN/"
cp "$BASE/inputs/vit_J2=0.50_N=8x8_k=0.mpack" "$RUN/"
cp "$BASE/inputs/krylov_phys8_a2_fixedT.npz" "$RUN/"
cp "$BASE/inputs/krylov_scaling_8x8_a1.20_tr4096_va2048.npz" "$RUN/"
cd "$RUN"
{ echo "job=${SLURM_JOB_ID} host=$(hostname) started=$(date -Is)"; nvidia-smi -L; python - <<'PY'
import jax; print("backend",jax.default_backend(),"devices",jax.devices())
PY
} > "$OUT/run_metadata.txt"
python gfmc_8x8_gr_population_size.py 128
echo "finished=$(date -Is)" >> "$OUT/run_metadata.txt"

[executed on device: MacBook-Air-2.local (c492ee85-a74b-4ec6-9be7-d51e7192ffe3)]