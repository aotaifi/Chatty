#!/bin/bash
#SBATCH --job-name=fncal
#SBATCH --partition=cip,inter
#SBATCH --gres=gpu:a40:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=/project/theorie/a/A.Otaifi/chatty_fncal6/logs/%x-%j.out
#SBATCH --error=/project/theorie/a/A.Otaifi/chatty_fncal6/logs/%x-%j.err
# FN referee calibration on 6x6 (experiments/fn_calibration_6x6). Full A40 (a40-NNgb slices do not work with jax 0.10).
# Usage: sbatch [--job-name=NAME] [--gres=gpu:rtx2080ti:1 --partition=inter] fncal.sbatch <script.py> <args...>
# Runs in $BASE/runs/NAME_JOBID (script copied there); FNCAL_OUT = that directory.
set -euo pipefail
BASE=/project/theorie/a/A.Otaifi/chatty_fncal6
OUT="$BASE/runs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}"
mkdir -p "$OUT"
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
cp "$BASE"/code/*.py "$OUT/"
cd "$OUT"
export FNCAL_OUT="$OUT"
{ echo "job_id=${SLURM_JOB_ID}"; echo "host=$(hostname)"; echo "args=$*"; echo "started=$(date -Is)"; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader; } > run_metadata.txt
/home/a/A.Otaifi/chatty_ll6/venv_gpu/bin/python "$@"
echo "finished=$(date -Is)" >> run_metadata.txt
