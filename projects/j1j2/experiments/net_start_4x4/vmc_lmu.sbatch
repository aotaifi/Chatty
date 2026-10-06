#!/bin/bash
#SBATCH --job-name=nets4-vmc
#SBATCH --partition=cluster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=/home/a/A.Otaifi/ChattyRun/j1j2/net_start_4x4/logs/%x-%j.out
#SBATCH --error=/home/a/A.Otaifi/ChattyRun/j1j2/net_start_4x4/logs/%x-%j.err
# Usage: sbatch [--job-name=..] [--time=..] vmc_lmu.sbatch <script.py> <args...>   (script in the run dir)
set -euo pipefail
module load python/3.11-2023.09
BASE=/home/a/A.Otaifi/ChattyRun/j1j2/net_start_4x4
SCRIPT=$1; shift
export JAX_PLATFORMS=cpu OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK} OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK} MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export PYTHONPATH=/home/a/A.Otaifi/chatty_fnguides/nqsmagic:$BASE
cd "$BASE"
echo "job=${SLURM_JOB_ID} host=$(hostname) script=$SCRIPT args=$* start=$(date -Is)"
/home/a/A.Otaifi/chatty_fnguides/venv311/bin/python "$BASE/$SCRIPT" --cpus ${SLURM_CPUS_PER_TASK} "$@"
echo "end=$(date -Is)"
