#!/bin/bash
#SBATCH --job-name=chatty-k2srdtrec
#SBATCH --partition=small
#SBATCH --array=0-1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=03:00:00
#SBATCH --mail-type=NONE
#SBATCH --output=/home/a/A.Otaifi/ChattyRun/j1j2/krylov_sign_structure/results/k2srdtrec-%A_%a.out
#SBATCH --error=/home/a/A.Otaifi/ChattyRun/j1j2/krylov_sign_structure/results/k2srdtrec-%A_%a.err
set -euo pipefail
module purge
module load python/3.12-2024.10
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
BASE=/home/a/A.Otaifi/ChattyRun/j1j2/krylov_sign_structure
DTS=(0.025 0.0125)
i=$SLURM_ARRAY_TASK_ID
DT=${DTS[$i]}
ID="${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
notify(){ "$BASE/experiments/chatty_notify.sh" "$1" "$ID" "$2"; }
finish(){ rc=$?; if [ "$rc" -eq 0 ]; then notify FINISHED "K2 SR recursive M=100k dt=$DT"; else notify FAILED "K2 SR recursive M=100k dt=$DT exit=$rc"; fi; }
trap finish EXIT
notify STARTED "K2 SR recursive M=100k dt=$DT"
cd "$BASE/experiments"
python k2_sr_tangent_recursive20.py --M 100000 --beta 3.0 --dt "$DT" --ns 2000 --h 8 --out "../results/k2srdtrec_M100000_dt${DT}_b3.json"
