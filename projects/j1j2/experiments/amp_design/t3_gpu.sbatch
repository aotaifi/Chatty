#!/bin/bash
#SBATCH --job-name=t3
#SBATCH --partition=cip,inter
#SBATCH --gres=gpu:a40:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --output=/home/a/A.Otaifi/chatty_t3/logs/%x-%j.out
#SBATCH --error=/home/a/A.Otaifi/chatty_t3/logs/%x-%j.err
# T3 (amp_design): 6x6 loop = stored composite sign + hop, fixed-sign RGN amplitude steps, FN referee.
# Small jobs: sbatch --partition=inter --gres=gpu:rtx2080ti:1 ...
# Usage: sbatch --job-name=NAME t3_gpu.sbatch <script.py> <args...>   (runs in runs/NAME_JOBID)
set -euo pipefail
BASE=/home/a/A.Otaifi/chatty_t3; LL6=/home/a/A.Otaifi/chatty_ll6; SS6=/home/a/A.Otaifi/chatty_ss6; IT2=/home/a/A.Otaifi/chatty_ll6it2
OUT="$BASE/runs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}"
mkdir -p "$OUT"
export XLA_PYTHON_CLIENT_PREALLOCATE=false OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK} XLA_FLAGS=--xla_gpu_autotune_level=0
cp "$IT2"/code/*.py "$OUT/"; cp "$LL6"/code/ll6_core.py "$LL6"/code/vit_dt.py "$SS6"/code/ss6_core.py "$OUT/"; cp "$BASE"/code/*.py "$OUT/"
cd "$OUT"
for f in "vit_J2=0.50_N=6x6_k=0.mpack" samples_psi0sq_6x6.npz samples_extra_psi0sq_6x6.npz energy_krylov_vs_vit_6x6_indep.npz psi0_6x6_table.npz; do ln -sf "$LL6/data/$f" .; done
{ echo "job_id=${SLURM_JOB_ID}"; echo "host=$(hostname)"; echo "args=$*"; echo "started=$(date -Is)"; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader; } > run_metadata.txt
"$LL6/venv_gpu/bin/python" "$@"
echo "finished=$(date -Is)" >> run_metadata.txt
