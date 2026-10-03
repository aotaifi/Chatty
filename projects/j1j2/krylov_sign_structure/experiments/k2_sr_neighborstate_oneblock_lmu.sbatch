#!/bin/bash
#SBATCH --job-name=chatty-k2srnbr
#SBATCH --partition=small
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --mail-type=NONE
#SBATCH --output=/home/a/A.Otaifi/ChattyRun/j1j2/krylov_sign_structure/results/k2srnbr-%j.out
#SBATCH --error=/home/a/A.Otaifi/ChattyRun/j1j2/krylov_sign_structure/results/k2srnbr-%j.err
set -euo pipefail
module purge
module load python/3.12-2024.10
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
BASE=/home/a/A.Otaifi/ChattyRun/j1j2/krylov_sign_structure
ID="$SLURM_JOB_ID"
notify(){ "$BASE/experiments/chatty_notify.sh" "$1" "$ID" "$2"; }
finish(){ rc=$?; if [ "$rc" -eq 0 ]; then notify FINISHED "K2 SR neighbor-state"; else notify FAILED "K2 SR neighbor-state exit=$rc"; fi; }
trap finish EXIT
notify STARTED "K2 SR neighbor-state"
cd "$BASE/experiments"
python k2_sr_neighborstate_oneblock20.py --M 100000 --dt .0125 --ns 2000 --K 4 --h 8 --out ../results/k2srnbr_M100000_dt0125_K4.json
