#!/bin/bash
#SBATCH --job-name=chatty-k2srcap
#SBATCH --partition=small
#SBATCH --array=0-3
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --mail-type=NONE
#SBATCH --output=/home/a/A.Otaifi/ChattyRun/j1j2/krylov_sign_structure/results/k2srcap-%A_%a.out
#SBATCH --error=/home/a/A.Otaifi/ChattyRun/j1j2/krylov_sign_structure/results/k2srcap-%A_%a.err
set -euo pipefail
module purge
module load python/3.12-2024.10
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
BASE=/home/a/A.Otaifi/ChattyRun/j1j2/krylov_sign_structure
MS=(25000 25000 100000 100000)
HS=(8 16 8 16)
ITS=(3 1 3 1)
i=$SLURM_ARRAY_TASK_ID
M=${MS[$i]}; H=${HS[$i]}; IT=${ITS[$i]}
ID="${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
notify(){ "$BASE/experiments/chatty_notify.sh" "$1" "$ID" "$2"; }
finish(){ rc=$?; if [ "$rc" -eq 0 ]; then notify FINISHED "K2 SR cap M=$M h=$H it=$IT"; else notify FAILED "K2 SR cap M=$M h=$H it=$IT exit=$rc"; fi; }
trap finish EXIT
notify STARTED "K2 SR cap M=$M h=$H it=$IT"
cd "$BASE/experiments"
python k2_sr_tangent_iterative_oneblock20.py --M "$M" --ns 2000 --h "$H" --iters "$IT" --out "../results/k2srcap_M${M}_h${H}_it${IT}.json"
