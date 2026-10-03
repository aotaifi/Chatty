#!/bin/bash
#SBATCH --job-name=chatty-k2gnnrecon
#SBATCH --partition=small
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --mail-type=NONE
#SBATCH --output=/home/a/A.Otaifi/ChattyRun/j1j2/krylov_sign_structure/results/k2gnnrecon-%j.out
#SBATCH --error=/home/a/A.Otaifi/ChattyRun/j1j2/krylov_sign_structure/results/k2gnnrecon-%j.err
set -euo pipefail
module purge
module load python/3.12-2024.10
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
BASE=/home/a/A.Otaifi/ChattyRun/j1j2/krylov_sign_structure
ID="$SLURM_JOB_ID"
notify(){ "$BASE/experiments/chatty_notify.sh" "$1" "$ID" "$2"; }
finish(){ rc=$?; if [ "$rc" -eq 0 ]; then notify FINISHED "K2 GNN reconstructed M100k"; else notify FAILED "K2 GNN reconstructed exit=$rc"; fi; }
trap finish EXIT
notify STARTED "K2 GNN reconstructed M100k"
cd "$BASE/experiments"
python k2_gnn_tempered_reconstructed20.py --M 100000 --steps 5 --dt .0125 --alpha .8 --centers 2000 --epochs 4 --out ../results/k2gnnrecon_M100000_dt0125_a08.json
