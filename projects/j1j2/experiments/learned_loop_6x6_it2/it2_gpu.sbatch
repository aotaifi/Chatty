#!/bin/bash
#SBATCH --job-name=it2
#SBATCH --partition=cip,inter
#SBATCH --gres=gpu:a40:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/home/a/A.Otaifi/chatty_ll6it2/logs/%x-%j.out
#SBATCH --error=/home/a/A.Otaifi/chatty_ll6it2/logs/%x-%j.err
# Learned loop 6x6 iterations 2-3 with stored signs (experiments/learned_loop_6x6_it2).
# Full A40 (a40-NNgb slices and V100 fail with jax 0.10 / cuDNN); rtx2080ti works for small jobs:
#   sbatch --partition=inter --gres=gpu:rtx2080ti:1 ...
# Usage: sbatch --job-name=NAME it2_gpu.sbatch <script.py> <args...>   (runs in runs/NAME_JOBID)
set -euo pipefail
BASE=/home/a/A.Otaifi/chatty_ll6it2; LL6=/home/a/A.Otaifi/chatty_ll6; SS6=/home/a/A.Otaifi/chatty_ss6
OUT="$BASE/runs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}"
mkdir -p "$OUT"
export XLA_PYTHON_CLIENT_PREALLOCATE=false OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
cp "$BASE"/code/*.py "$OUT/"; cp "$LL6"/code/ll6_core.py "$LL6"/code/vit_dt.py "$SS6"/code/ss6_core.py "$OUT/"
cd "$OUT"
for f in "vit_J2=0.50_N=6x6_k=0.mpack" samples_psi0sq_6x6.npz samples_extra_psi0sq_6x6.npz energy_krylov_vs_vit_6x6_indep.npz psi0_6x6_table.npz; do ln -sf "$LL6/data/$f" .; done
for f in "$BASE"/data/*; do [ -e "$f" ] && ln -sf "$f" .; done
{ echo "job_id=${SLURM_JOB_ID}"; echo "host=$(hostname)"; echo "args=$*"; echo "started=$(date -Is)"; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader; } > run_metadata.txt
"$LL6/venv_gpu/bin/python" "$@"
echo "finished=$(date -Is)" >> run_metadata.txt
