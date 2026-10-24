#!/bin/bash
#SBATCH --job-name=pilH100
#SBATCH --account=hpc-prf-ppdmrg
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --no-requeue
#SBATCH --output=/scratch/hpc-prf-ppdmrg/hpcalot/rbmppL/runs/%x-%j.out
# Timing pilot on one H100 (express: timing only, <= 0.5 H100-h in total for both sizes). usage:
#   sbatch --qos=express_hpcalot --time=00:20:00 --export=ALL,L=10,ARGSF=/path/args.txt   (a file: --export cannot carry commas) pilot_h100.sbatch
set -uo pipefail
L=${L:?L}; B=/scratch/hpc-prf-ppdmrg/hpcalot/rbmppL; RUN=$B/runs/pilot_L${L}_${SLURM_JOB_ID}; mkdir -p $RUN/code
cp $B/code/*.py $RUN/code/
export XLA_FLAGS="--xla_gpu_autotune_level=0" XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
{ echo job_id=$SLURM_JOB_ID; hostname; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader; date -Is; sha256sum $RUN/code/*.py; } > $RUN/run_metadata.txt
cd $RUN/code
/scratch/hpc-prf-ppdmrg/hpcalot/venv_jax_0.8.2/bin/python pilot_driver.py $RUN $L $(cat ${ARGSF:-/dev/null}) > $RUN/driver.out 2>&1; rc=$?
echo "exit=$rc finished=$(date -Is)" >> $RUN/run_metadata.txt; exit $rc
