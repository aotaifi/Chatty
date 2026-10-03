#!/bin/bash
#SBATCH --job-name=chatty-k2armle
#SBATCH --partition=small
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --time=01:00:00
#SBATCH --mail-type=NONE
#SBATCH --output=/home/a/A.Otaifi/ChattyRun/j1j2/krylov_sign_structure/results/k2armle-%j.out
#SBATCH --error=/home/a/A.Otaifi/ChattyRun/j1j2/krylov_sign_structure/results/k2armle-%j.err
set -euo pipefail
module purge
module load python/3.12-2024.10
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
BASE=/home/a/A.Otaifi/ChattyRun/j1j2/krylov_sign_structure
ID="$SLURM_JOB_ID"
notify(){ "$BASE/experiments/chatty_notify.sh" "$1" "$ID" "$2"; }
finish(){ rc=$?; if [ "$rc" -eq 0 ]; then notify FINISHED "K2 AR MLE compression"; else notify FAILED "K2 AR MLE compression exit=$rc"; fi; }
trap finish EXIT
notify STARTED "K2 AR MLE compression"
cd "$BASE/experiments"
python k2_ar_mle_compression20.py --M 25000 --epochs 30 --hidden 96 --out ../results/k2armle_M25000_h96.json
