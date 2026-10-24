#!/bin/bash
#SBATCH --job-name=pilA40
#SBATCH --partition=cip
#SBATCH --gres=gpu:a40:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --no-requeue
#SBATCH --output=/scratch/a/A.Otaifi/rbmppL/%x-%j.out
# Timing pilot on the theorie cluster's FULL A40 (cip-cl-nv01, 48 GB). usage: sbatch --export=ALL,L=10,ARGS="--sr 2048:30,8192:4 ..." pilot_a40.sbatch
set -uo pipefail
L=${L:?L}; B=/scratch/a/A.Otaifi/rbmppL; RUN=${RUNDIR:-$B/pilot_a40_L${L}_${SLURM_JOB_ID}}; mkdir -p $RUN/code
cp $B/code/*.py $RUN/code/
export XLA_FLAGS="--xla_gpu_autotune_level=0" XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
{ echo job_id=$SLURM_JOB_ID; hostname; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader; date -Is; sha256sum $RUN/code/*.py; } >> $RUN/run_metadata.txt
cd $RUN/code
/project/theorie/a/A.Otaifi/Chatty/runs/a40slice/venv_jax_0.8.2/bin/python pilot_driver.py $RUN $L $(cat ${ARGSF:-/dev/null}) > $RUN/driver.out 2>&1; rc=$?
echo "exit=$rc finished=$(date -Is)" >>> $RUN/run_metadata.txt; exit $rc
