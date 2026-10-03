#!/bin/bash
#SBATCH --job-name=chatty-k2arhyb
#SBATCH --partition=small
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --time=01:00:00
#SBATCH --mail-type=NONE
#SBATCH --output=/home/a/A.Otaifi/ChattyRun/j1j2/krylov_sign_structure/results/k2arhyb-%j.out
#SBATCH --error=/home/a/A.Otaifi/ChattyRun/j1j2/krylov_sign_structure/results/k2arhyb-%j.err
set -euo pipefail
module purge
module load python/3.12-2024.10
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
BASE=/home/a/A.Otaifi/ChattyRun/j1j2/krylov_sign_structure
ID="$SLURM_JOB_ID"
notify(){ "$BASE/experiments/chatty_notify.sh" "$1" "$ID" "$2"; }
finish(){ rc=$?; if [ "$rc" -eq 0 ]; then notify FINISHED "K2 AR hybrid MLE-edge"; else notify FAILED "K2 AR hybrid MLE-edge exit=$rc"; fi; }
trap finish EXIT
notify STARTED "K2 AR hybrid MLE-edge"
cd "$BASE/experiments"
python k2_ar_hybrid_mle_edge20.py --M 25000 --nedges 30000 --epochs 30 --hidden 96 --alpha .8 --out ../results/k2arhyb_M25000_h96_a08.json
