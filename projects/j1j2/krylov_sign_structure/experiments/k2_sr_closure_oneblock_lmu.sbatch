#!/bin/bash
#SBATCH --job-name=chatty-k2srclose
#SBATCH --partition=small
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=01:30:00
#SBATCH --mail-type=NONE
#SBATCH --output=/home/a/A.Otaifi/ChattyRun/j1j2/krylov_sign_structure/results/k2srclose-%j.out
#SBATCH --error=/home/a/A.Otaifi/ChattyRun/j1j2/krylov_sign_structure/results/k2srclose-%j.err
set -euo pipefail
module purge
module load python/3.12-2024.10
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
BASE=/home/a/A.Otaifi/ChattyRun/j1j2/krylov_sign_structure
ID="$SLURM_JOB_ID"
notify(){ "$BASE/experiments/chatty_notify.sh" "$1" "$ID" "$2"; }
finish(){ rc=$?; if [ "$rc" -eq 0 ]; then notify FINISHED "K2 SR two-hop closure"; else notify FAILED "K2 SR two-hop closure exit=$rc"; fi; }
trap finish EXIT
notify STARTED "K2 SR two-hop closure"
cd "$BASE/experiments"
python k2_sr_closure_tangent_oneblock20.py --M 100000 --dt .0125 --ns 2000 --h 8 --out ../results/k2srclose_M100000_dt0125.json
