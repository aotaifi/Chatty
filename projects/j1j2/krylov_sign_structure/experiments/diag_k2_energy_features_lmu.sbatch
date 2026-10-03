#!/bin/bash
#SBATCH --job-name=chatty-k2efeat
#SBATCH --partition=small
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --mail-type=NONE
#SBATCH --output=/home/a/A.Otaifi/ChattyRun/j1j2/krylov_sign_structure/results/k2efeat-%j.out
#SBATCH --error=/home/a/A.Otaifi/ChattyRun/j1j2/krylov_sign_structure/results/k2efeat-%j.err
set -euo pipefail
module purge
module load python/3.12-2024.10
BASE=/home/a/A.Otaifi/ChattyRun/j1j2/krylov_sign_structure
ID="$SLURM_JOB_ID"
notify(){ "$BASE/experiments/chatty_notify.sh" "$1" "$ID" "$2"; }
finish(){ rc=$?; if [ "$rc" -eq 0 ]; then notify FINISHED "K2 energy-feature diagnostic"; else notify FAILED "K2 energy-feature diagnostic exit=$rc"; fi; }
trap finish EXIT
notify STARTED "K2 energy-feature diagnostic"
cd "$BASE/experiments"
python diag_k2_energy_features20.py
