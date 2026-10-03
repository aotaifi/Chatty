#!/bin/bash
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --job-name=chatty-fnguides
#SBATCH --output=/pc2/users/h/hpcalot/chatty/j1j2/results/fnguides-%j.out
#SBATCH --error=/pc2/users/h/hpcalot/chatty/j1j2/results/fnguides-%j.err
# Usage: sbatch fn_guides_compare_h100.sbatch L
#   L=6: E_ViT (VMC), ViT-sign FN, Marshall FN (new), Krylov FN with the seeds of the
#        existing M=128 run (9501/9502) as a reproduction check.
#   L=8: E_ViT (VMC), ViT-sign FN, Marshall FN seed 11501 as reproduction check of 3471545.
# All FN: M=128, beta_target=1.2, burn_beta=0.4, tau_max=0.025.
set -euo pipefail
LL=$1
module load lang
module load Python/3.11.5-GCCcore-13.2.0
source /pc2/users/h/hpcalot/chatty/.venv-j1j2-py311/bin/activate
export JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
BASE=/pc2/users/h/hpcalot/chatty/j1j2
SRC=$BASE/fn_guides_compare
OUT="$BASE/results/fn_guides_compare_${LL}x${LL}_${SLURM_JOB_ID}"
mkdir -p "$OUT"
RUN=$(mktemp -d "${TMPDIR:-/tmp}/chatty-fnguides.XXXXXX")
finish(){ cp -f "$RUN"/*.json "$RUN"/*.npz "$OUT/" 2>/dev/null || true; }
trap finish EXIT
cp "$SRC/fn_guides_compare.py" "$OUT/"; cp "$SRC/fn_guides_compare.py" "$RUN/"
cp "$BASE/inputs/vit_J2=0.50_N=${LL}x${LL}_k=0.mpack" "$RUN/"
if [ "$LL" = 6 ]; then cp "$BASE/inputs/energy_krylov_vs_vit_6x6_indep.npz" "$RUN/"; else cp "$BASE/inputs/krylov_phys8_a2_fixedT.npz" "$RUN/"; fi
cd "$RUN"
{ echo "job=${SLURM_JOB_ID} host=$(hostname) L=$LL started=$(date -Is)"; nvidia-smi -L; } > "$OUT/run_metadata.txt"
python fn_guides_compare.py vmc $LL 1024 16 4 100 20261003
PHI=$(python -c "import json;print(json.load(open('vmc_vit_${LL}x${LL}.json'))['phi'])")
echo "phi=$PHI" >> "$OUT/run_metadata.txt"
if [ "$LL" = 6 ]; then
  python fn_guides_compare.py fn 6 vit 128 13601 13602 --phi "$PHI"
  python fn_guides_compare.py fn 6 marshall 128 12601 12602
  python fn_guides_compare.py fn 6 krylov 128 9501 9502
else
  python fn_guides_compare.py fn 8 vit 128 13501 13502 --phi "$PHI"
  python fn_guides_compare.py fn 8 marshall 128 11501
fi
echo "finished=$(date -Is)" >> "$OUT/run_metadata.txt"
