#!/bin/bash
#SBATCH --job-name=chatty-k2sraudit
#SBATCH --partition=small
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:45:00
#SBATCH --mail-type=NONE
#SBATCH --output=/home/a/A.Otaifi/ChattyRun/j1j2/krylov_sign_structure/results/k2sraudit-%j.out
#SBATCH --error=/home/a/A.Otaifi/ChattyRun/j1j2/krylov_sign_structure/results/k2sraudit-%j.err
set -euo pipefail
module purge
module load python/3.12-2024.10
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
BASE=/home/a/A.Otaifi/ChattyRun/j1j2/krylov_sign_structure
ID="$SLURM_JOB_ID"
notify(){ "$BASE/experiments/chatty_notify.sh" "$1" "$ID" "$2"; }
finish(){ rc=$?; if [ "$rc" -eq 0 ]; then notify FINISHED "K2 SR unseen-neighbor audit"; else notify FAILED "K2 SR unseen-neighbor audit exit=$rc"; fi; }
trap finish EXIT
notify STARTED "K2 SR unseen-neighbor audit"
cd "$BASE/experiments"
python k2_sr_unseen_neighbor_audit20.py --M 100000 --dt .0125 --ns 2000 --h 8 --out ../results/k2sraudit_M100000_dt0125.json
