#!/bin/bash
#SBATCH --job-name=rung2-4x4
#SBATCH --partition=cluster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=/home/a/A.Otaifi/ChattyRun/j1j2/learning_ladder_rung2/logs/rung2-%j.out
#SBATCH --error=/home/a/A.Otaifi/ChattyRun/j1j2/learning_ladder_rung2/logs/rung2-%j.err
# Rung 2 of the 4x4 amplitude-learning ladder (LMU theorie CPU).
# Usage: sbatch rung2_lmu.sbatch <tag> <rung2_sampled_sr_loop_4x4.py args...>
set -euo pipefail
module load python/3.11-2023.09
BASE=/home/a/A.Otaifi/ChattyRun/j1j2/learning_ladder_rung2
TAG=$1; shift
mkdir -p "$BASE/results" "$BASE/logs"
export JAX_PLATFORMS=cpu OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK} OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK} MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export PYTHONPATH=/home/a/A.Otaifi/chatty_fnguides/nqsmagic
cd "$BASE"
echo "job=${SLURM_JOB_ID} host=$(hostname) tag=$TAG args=$* start=$(date -Is)"
/home/a/A.Otaifi/chatty_fnguides/venv311/bin/python "$BASE/rung2_sampled_sr_loop_4x4.py" --out "$BASE/results/${TAG}_${SLURM_JOB_ID}.json" "$@"
echo "end=$(date -Is)"
