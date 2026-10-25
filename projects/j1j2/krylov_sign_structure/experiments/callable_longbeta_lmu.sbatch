#!/bin/bash
#SBATCH --job-name=chatty-ft-callable
#SBATCH --partition=small
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=a.otaifi@lmu.de
#SBATCH --output=/project/theorie/a/A.Otaifi/Chatty/runs/ChattyRun/j1j2/krylov_sign_structure/results/callable-longbeta-%j.out
#SBATCH --error=/project/theorie/a/A.Otaifi/Chatty/runs/ChattyRun/j1j2/krylov_sign_structure/results/callable-longbeta-%j.err
set -euo pipefail
module purge
module load python/3.12-2024.10
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
BASE=/project/theorie/a/A.Otaifi/Chatty/runs/ChattyRun/j1j2/krylov_sign_structure
cd "$BASE/experiments"
echo "job=$SLURM_JOB_ID host=$(hostname) start=$(date -Is)"
python finite_tau_ct_smc_callable_longbeta20.py --M 25000 --beta 2.0 --out ../results/finite_tau_ct_smc_callable_longbeta_M25000.json
echo "finish=$(date -Is)"
