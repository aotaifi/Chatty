#!/bin/bash
#SBATCH --job-name=chatty-ft-pairdata
#SBATCH --partition=small
#SBATCH --array=0-5
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --mail-type=NONE
#SBATCH --output=/home/a/A.Otaifi/ChattyRun/j1j2/krylov_sign_structure/results/pairdata-%A_%a.out
#SBATCH --error=/home/a/A.Otaifi/ChattyRun/j1j2/krylov_sign_structure/results/pairdata-%A_%a.err
set -euo pipefail
module purge
module load python/3.12-2024.10
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
BASE=/home/a/A.Otaifi/ChattyRun/j1j2/krylov_sign_structure

MS=(100000 100000 200000 200000 200000 100000)
BETAS=(3.0 3.0 3.0 3.0 3.0 2.5)
NFITS=(50000 100000 50000 100000 200000 100000)
KS=(4 4 4 4 2 4)
EPOCHS=(3 2 3 2 2 2)

i=$SLURM_ARRAY_TASK_ID
M=${MS[$i]}; BETA=${BETAS[$i]}; NFIT=${NFITS[$i]}; K=${KS[$i]}; EP=${EPOCHS[$i]}
ID="${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
notify(){ "$BASE/experiments/chatty_notify.sh" "$1" "$ID" "$2"; }
finish(){ rc=$?; if [ "$rc" -eq 0 ]; then notify FINISHED "M=$M beta=$BETA nfit=$NFIT"; else notify FAILED "M=$M beta=$BETA nfit=$NFIT exit=$rc"; fi; }
trap finish EXIT
notify STARTED "M=$M beta=$BETA nfit=$NFIT"
cd "$BASE/experiments"
python finite_tau_ct_smc_pairratio_longbeta20.py --M "$M" --beta "$BETA" --epochs "$EP" --nfit "$NFIT" --K "$K" --out "../results/pairdata_M${M}_b${BETA}_n${NFIT}_K${K}_e${EP}.json"
