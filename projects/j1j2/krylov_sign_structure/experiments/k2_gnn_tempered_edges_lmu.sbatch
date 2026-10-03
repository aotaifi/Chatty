#!/bin/bash
#SBATCH --job-name=chatty-k2temp
#SBATCH --partition=small
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:30:00
#SBATCH --mail-type=NONE
#SBATCH --output=/home/a/A.Otaifi/ChattyRun/j1j2/krylov_sign_structure/results/k2temp-%j.out
#SBATCH --error=/home/a/A.Otaifi/ChattyRun/j1j2/krylov_sign_structure/results/k2temp-%j.err
set -euo pipefail
module purge
module load python/3.12-2024.10
export JAX_PLATFORMS=cpu
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
BASE=/home/a/A.Otaifi/ChattyRun/j1j2/krylov_sign_structure
ID="$SLURM_JOB_ID"
notify(){ "$BASE/experiments/chatty_notify.sh" "$1" "$ID" "$2"; }
finish(){ rc=$?; if [ "$rc" -eq 0 ]; then notify FINISHED "K2 tempered edge GNN"; else notify FAILED "K2 tempered edge GNN exit=$rc"; fi; }
trap finish EXIT
notify STARTED "K2 tempered edge GNN"
cd "$BASE/experiments"
python k2_gnn_tempered_edges_oneblock20.py --M 100000 --dt .0125 --ntrain 30000 --nval 10000 --epochs 15 --alpha .6 --mix .5 --out ../results/k2temp_M100000_dt0125_a06_mix05.json
