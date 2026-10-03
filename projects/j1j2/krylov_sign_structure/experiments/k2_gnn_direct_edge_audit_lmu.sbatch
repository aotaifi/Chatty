#!/bin/bash
#SBATCH --job-name=chatty-k2edgept
#SBATCH --partition=small
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --array=0-1
#SBATCH --mail-type=NONE
#SBATCH --output=/home/a/A.Otaifi/ChattyRun/j1j2/krylov_sign_structure/results/k2edgept-%A_%a.out
#SBATCH --error=/home/a/A.Otaifi/ChattyRun/j1j2/krylov_sign_structure/results/k2edgept-%A_%a.err
set -euo pipefail
module purge
module load python/3.12-2024.10
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
BASE=/home/a/A.Otaifi/ChattyRun/j1j2/krylov_sign_structure
MODE=exact
[[ "$SLURM_ARRAY_TASK_ID" -eq 1 ]] && MODE=reconstructed
ID="${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
notify(){ "$BASE/experiments/chatty_notify.sh" "$1" "$ID" "$2"; }
finish(){ rc=$?; if [ "$rc" -eq 0 ]; then notify FINISHED "K2 direct-edge PairGNN $MODE"; else notify FAILED "K2 direct-edge PairGNN $MODE exit=$rc"; fi; }
trap finish EXIT
notify STARTED "K2 direct-edge PairGNN $MODE"
cd "$BASE/experiments"
python k2_gnn_direct_edge_audit20.py --mode "$MODE" --M 100000 --dt .0125 --alpha .8 --mix .5 --ntrain 30000 --nval 10000 --epochs 15 --out "../results/k2edgept_${MODE}_M100000_dt0125_a08.json"
