#!/bin/bash
#SBATCH --job-name=chatty-j1j2-8x8-node2diag
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=01:30:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=alyotaifi@gmail.com
#SBATCH --output=/pc2/users/h/hpcalot/chatty/j1j2/results/8x8-node2diag-%j.out
#SBATCH --error=/pc2/users/h/hpcalot/chatty/j1j2/results/8x8-node2diag-%j.err
set -euo pipefail
module load lang
module load Python/3.11.5-GCCcore-13.2.0
source /pc2/users/h/hpcalot/chatty/.venv-j1j2-py311/bin/activate
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
BASE=/pc2/users/h/hpcalot/chatty/j1j2
OUT="$BASE/results/8x8_node2diag_${SLURM_JOB_ID}"
mkdir -p "$OUT"
RUN=$(mktemp -d "${TMPDIR:-/tmp}/chatty-8x8-node2diag.XXXXXX")
finish(){ cp -f "$RUN"/*.npz "$OUT/" 2>/dev/null || true; cp -f "$RUN"/diag_8x8_fnrefresh_signstep.py "$OUT/" 2>/dev/null || true; }
trap finish EXIT
cp "$BASE/src/diag_8x8_fnrefresh_signstep.py" "$RUN/"
cp "$BASE/src/gfmc_8x8_gr_population_size.py" "$RUN/"
cp "$BASE/inputs/vit_J2=0.50_N=8x8_k=0.mpack" "$RUN/"
cp "$BASE/inputs/krylov_scaling_8x8_a1.20_tr4096_va2048.npz" "$RUN/"
cp "$BASE/results/8x8_krylov_3471544/gr_8x8_population_scaling_M128.npz" "$RUN/"
cd "$RUN"
python diag_8x8_fnrefresh_signstep.py
