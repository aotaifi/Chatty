#!/bin/bash
#SBATCH --job-name=chatty-k2sr0125
#SBATCH --partition=small
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:30:00
#SBATCH --mail-type=NONE
#SBATCH --output=/home/a/A.Otaifi/ChattyRun/j1j2/krylov_sign_structure/results/k2sr0125-%j.out
#SBATCH --error=/home/a/A.Otaifi/ChattyRun/j1j2/krylov_sign_structure/results/k2sr0125-%j.err
set -euo pipefail
module purge
module load python/3.12-2024.10
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
BASE=/home/a/A.Otaifi/ChattyRun/j1j2/krylov_sign_structure
ID="$SLURM_JOB_ID"
notify(){ "$BASE/experiments/chatty_notify.sh" "$1" "$ID" "$2"; }
finish(){ rc=$?; if [ "$rc" -eq 0 ]; then notify FINISHED "K2 SR recursive dt=.0125 M=100k"; else notify FAILED "K2 SR recursive dt=.0125 exit=$rc"; fi; }
trap finish EXIT
notify STARTED "K2 SR recursive dt=.0125 M=100k"
cd "$BASE/experiments"
python k2_sr_tangent_recursive20.py --M 100000 --beta 3.0 --dt .0125 --ns 2000 --h 8 --out ../results/k2srrec_M100000_b3_dt0125.json
