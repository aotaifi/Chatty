#!/bin/bash
#SBATCH --job-name=chatty-tdvp1
#SBATCH --partition=small
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --mail-type=NONE
#SBATCH --output=/home/a/A.Otaifi/ChattyRun/j1j2/krylov_sign_structure/results/tdvp1-%j.out
#SBATCH --error=/home/a/A.Otaifi/ChattyRun/j1j2/krylov_sign_structure/results/tdvp1-%j.err
set -euo pipefail
module purge
module load python/3.12-2024.10
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
BASE=/home/a/A.Otaifi/ChattyRun/j1j2/krylov_sign_structure
ID="$SLURM_JOB_ID"
notify(){ "$BASE/experiments/chatty_notify.sh" "$1" "$ID" "$2"; }
finish(){ rc=$?; if [ "$rc" -eq 0 ]; then notify FINISHED "persistent TDVP oneblock"; else notify FAILED "persistent TDVP oneblock exit=$rc"; fi; }
trap finish EXIT
notify STARTED "persistent TDVP oneblock"
cd "$BASE/experiments"
python k2_tdvp_persistent_oneblock20.py --M 100000 --dt .0125 --ns 2000 --h 8 --out ../results/tdvp1_M100000_dt0125.json
