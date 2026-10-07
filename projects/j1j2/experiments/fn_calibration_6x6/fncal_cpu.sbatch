#!/bin/bash
#SBATCH --job-name=fnccpu
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=03:00:00
#SBATCH --output=/project/theorie/a/A.Otaifi/chatty_fncal6/logs/%x-%j.out
#SBATCH --error=/project/theorie/a/A.Otaifi/chatty_fncal6/logs/%x-%j.err
# CPU reference (fncal_ref_cpu.py): 4 independent processes. Usage: sbatch fncal_cpu.sbatch GUIDE M NPOP_PER_PROC
set -euo pipefail
BASE=/project/theorie/a/A.Otaifi/chatty_fncal6
OUT="$BASE/runs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}"; mkdir -p "$OUT"; cp "$BASE"/code/*.py "$OUT/"; cd "$OUT"
export JAX_PLATFORMS=cpu OMP_NUM_THREADS=1
for k in 0 1 2 3; do
  /home/a/A.Otaifi/chatty_ll6/venv_gpu/bin/python fncal_ref_cpu.py "$1" "$2" "$3" 1.2 0.4 $((1000000 + k * 100000)) ref_$k.json > ref_$k.log 2>&1 &
done
wait
