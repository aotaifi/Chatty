#!/bin/bash
#SBATCH --job-name=chatty-k1cost
#SBATCH --partition=small
#SBATCH --cpus-per-task=1
#SBATCH --mem=12G
#SBATCH --time=00:15:00
#SBATCH --mail-type=NONE
#SBATCH --output=/home/a/A.Otaifi/ChattyRun/j1j2/krylov_sign_structure/results/k1cost-%j.out
#SBATCH --error=/home/a/A.Otaifi/ChattyRun/j1j2/krylov_sign_structure/results/k1cost-%j.err
set -euo pipefail
module purge
module load python/3.12-2024.10
BASE=/home/a/A.Otaifi/ChattyRun/j1j2/krylov_sign_structure
ID="$SLURM_JOB_ID"
notify(){ "$BASE/experiments/chatty_notify.sh" "$1" "$ID" "$2"; }
finish(){ rc=$?; if [ "$rc" -eq 0 ]; then notify FINISHED "recursive K1 cost"; else notify FAILED "recursive K1 cost exit=$rc"; fi; }
trap finish EXIT
notify STARTED "recursive K1 cost"
cd "$BASE/experiments"
python diag_recursive_k1_cost20.py
