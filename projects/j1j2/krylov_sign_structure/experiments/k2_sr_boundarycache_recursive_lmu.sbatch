#!/bin/bash
#SBATCH --job-name=chatty-k2srcache
#SBATCH --partition=small
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --mail-type=NONE
#SBATCH --output=/home/a/A.Otaifi/ChattyRun/j1j2/krylov_sign_structure/results/k2srcache-%j.out
#SBATCH --error=/home/a/A.Otaifi/ChattyRun/j1j2/krylov_sign_structure/results/k2srcache-%j.err
set -euo pipefail
module purge
module load python/3.12-2024.10
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
BASE=/home/a/A.Otaifi/ChattyRun/j1j2/krylov_sign_structure
ID="$SLURM_JOB_ID"
notify(){ "$BASE/experiments/chatty_notify.sh" "$1" "$ID" "$2"; }
finish(){ rc=$?; if [ "$rc" -eq 0 ]; then notify FINISHED "K2 SR boundary-cache recursive"; else notify FAILED "K2 SR boundary-cache recursive exit=$rc"; fi; }
trap finish EXIT
notify STARTED "K2 SR boundary-cache recursive"
cd "$BASE/experiments"
python k2_sr_boundarycache_recursive20.py --beta 3.0 --dt 0.0125 --ns 2000 --h 8 --margin 0.2 --out ../results/k2srcache_M100000_dt0125_b3.json
