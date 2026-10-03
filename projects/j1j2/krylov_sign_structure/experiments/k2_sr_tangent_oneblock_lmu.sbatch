#!/bin/bash
#SBATCH --job-name=chatty-k2sr1
#SBATCH --partition=small
#SBATCH --array=0-3
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:45:00
#SBATCH --mail-type=NONE
#SBATCH --output=/home/a/A.Otaifi/ChattyRun/j1j2/krylov_sign_structure/results/k2sr1-%A_%a.out
#SBATCH --error=/home/a/A.Otaifi/ChattyRun/j1j2/krylov_sign_structure/results/k2sr1-%A_%a.err
set -euo pipefail
module purge
module load python/3.12-2024.10
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
BASE=/home/a/A.Otaifi/ChattyRun/j1j2/krylov_sign_structure
MS=(25000 25000 100000 100000)
NS=(2000 8000 2000 8000)
i=$SLURM_ARRAY_TASK_ID
M=${MS[$i]}; N=${NS[$i]}
ID="${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
notify(){ "$BASE/experiments/chatty_notify.sh" "$1" "$ID" "$2"; }
finish(){ rc=$?; if [ "$rc" -eq 0 ]; then notify FINISHED "K2 SR oneblock M=$M ns=$N"; else notify FAILED "K2 SR oneblock M=$M ns=$N exit=$rc"; fi; }
trap finish EXIT
notify STARTED "K2 SR oneblock M=$M ns=$N"
cd "$BASE/experiments"
python k2_sr_tangent_oneblock20.py --M "$M" --ns "$N" --h 8 --out "../results/k2sr1_M${M}_ns${N}_h8.json"
