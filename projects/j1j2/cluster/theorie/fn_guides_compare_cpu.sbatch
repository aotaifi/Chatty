#!/bin/bash
#SBATCH --job-name=chatty-fnguides
#SBATCH --partition=cluster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/home/a/A.Otaifi/chatty_fnguides/logs/fnguides-%j.out
#SBATCH --error=/home/a/A.Otaifi/chatty_fnguides/logs/fnguides-%j.err
# LMU theorie CPU fallback (Paderborn GPUs blocked by the Oct 5-9 shutdown).
# Usage: sbatch fn_guides_compare_cpu.sbatch <fn_guides_compare.py args...>
#   e.g. sbatch fn_guides_compare_cpu.sbatch vmc 8 256 16 3 60 20261003
#        sbatch fn_guides_compare_cpu.sbatch fn 8 vit 128 13501 13502 --phi 0.056
set -euo pipefail
module load python/3.11-2023.09
BASE=/home/a/A.Otaifi/chatty_fnguides
OUT="$BASE/results/fnguides_${SLURM_JOB_ID}"
mkdir -p "$OUT"
export JAX_PLATFORMS=cpu OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
cp "$BASE/fn_guides_compare.py" "$OUT/"
cd "$OUT"
for f in "vit_J2=0.50_N=6x6_k=0.mpack" "vit_J2=0.50_N=8x8_k=0.mpack" energy_krylov_vs_vit_6x6_indep.npz krylov_phys8_a2_fixedT.npz; do ln -sf "$BASE/$f" .; done
{ echo "job_id=${SLURM_JOB_ID}"; echo "host=$(hostname)"; echo "cpus=${SLURM_CPUS_PER_TASK}"; echo "args=$*"; echo "started=$(date -Is)"; lscpu | grep -E "Model name|^CPU\(s\)"; } > run_metadata.txt
"$BASE/venv311/bin/python" fn_guides_compare.py "$@"
echo "finished=$(date -Is)" >> run_metadata.txt
