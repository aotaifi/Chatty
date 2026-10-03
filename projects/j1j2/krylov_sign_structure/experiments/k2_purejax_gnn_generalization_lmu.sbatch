#!/bin/bash
#SBATCH --job-name=chatty-k2gnn
#SBATCH --partition=small
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:30:00
#SBATCH --mail-type=NONE
#SBATCH --output=/home/a/A.Otaifi/ChattyRun/j1j2/krylov_sign_structure/results/k2gnn-%j.out
#SBATCH --error=/home/a/A.Otaifi/ChattyRun/j1j2/krylov_sign_structure/results/k2gnn-%j.err
set -euo pipefail
module purge
module load python/3.12-2024.10
export JAX_PLATFORMS=cpu
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
BASE=/home/a/A.Otaifi/ChattyRun/j1j2/krylov_sign_structure
ID="$SLURM_JOB_ID"
notify(){ "$BASE/experiments/chatty_notify.sh" "$1" "$ID" "$2"; }
finish(){ rc=$?; if [ "$rc" -eq 0 ]; then notify FINISHED "K2 GNN generalization"; else notify FAILED "K2 GNN generalization exit=$rc"; fi; }
trap finish EXIT
notify STARTED "K2 GNN generalization"
cd "$BASE/experiments"
python k2_purejax_gnn_generalization20.py --M 100000 --dt .0125 --ns 2000 --epochs 20 --hidden 32 --layers 3 --out ../results/k2gnn_M100000_dt0125.json
