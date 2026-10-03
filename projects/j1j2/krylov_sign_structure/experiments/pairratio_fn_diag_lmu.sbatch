#!/bin/bash
#SBATCH --job-name=chatty-ft-fndiag
#SBATCH --partition=small
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --mail-type=NONE
#SBATCH --output=/home/a/A.Otaifi/ChattyRun/j1j2/krylov_sign_structure/results/fndiag-%j.out
#SBATCH --error=/home/a/A.Otaifi/ChattyRun/j1j2/krylov_sign_structure/results/fndiag-%j.err
set -euo pipefail
module purge
module load python/3.12-2024.10
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
BASE=/home/a/A.Otaifi/ChattyRun/j1j2/krylov_sign_structure
ID="$SLURM_JOB_ID"
notify(){ "$BASE/experiments/chatty_notify.sh" "$1" "$ID" "$2"; }
finish(){ rc=$?; if [ "$rc" -eq 0 ]; then notify FINISHED "pairratio FN diagnostic"; else notify FAILED "pairratio FN diagnostic exit=$rc"; fi; }
trap finish EXIT
notify STARTED "pairratio FN diagnostic"
cd "$BASE/experiments"
python finite_tau_ct_smc_pairratio_longbeta20.py --M 100000 --beta 3.0 --epochs 2 --nfit 100000 --K 4 --out ../results/pairratio_fndiag_M100000_b3.json
