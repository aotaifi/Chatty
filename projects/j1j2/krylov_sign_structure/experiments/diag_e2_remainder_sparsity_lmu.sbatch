#!/bin/bash
#SBATCH --job-name=chatty-e2sparse
#SBATCH --partition=small
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --mail-type=NONE
#SBATCH --output=/home/a/A.Otaifi/ChattyRun/j1j2/krylov_sign_structure/results/e2sparse-%j.out
#SBATCH --error=/home/a/A.Otaifi/ChattyRun/j1j2/krylov_sign_structure/results/e2sparse-%j.err
set -euo pipefail
module purge
module load python/3.12-2024.10
BASE=/home/a/A.Otaifi/ChattyRun/j1j2/krylov_sign_structure
ID="$SLURM_JOB_ID"
notify(){ "$BASE/experiments/chatty_notify.sh" "$1" "$ID" "$2"; }
finish(){ rc=$?; if [ "$rc" -eq 0 ]; then notify FINISHED "E2 remainder sparsity"; else notify FAILED "E2 remainder sparsity exit=$rc"; fi; }
trap finish EXIT
notify STARTED "E2 remainder sparsity"
cd "$BASE/experiments"
python diag_e2_remainder_sparsity20.py
