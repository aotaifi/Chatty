#!/bin/bash
#SBATCH --job-name=rb8
#SBATCH --account=hpc-prf-ppdmrg
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --no-requeue
#SBATCH --output=/scratch/hpc-prf-ppdmrg/hpcalot/rbmpp8/runs/%x-%j.out
# Usage (QoS and --time on the command line):
#   sbatch --qos=express_hpcalot --time=00:30:00 --job-name=NAME rbmpp8_h100.sbatch RUNNAME SCRIPT.py SPEC.json
# The run directory is runs/RUNNAME and is REUSED if it exists (spec "resume": true continues from state.npz).
set -uo pipefail
B=/scratch/hpc-prf-ppdmrg/hpcalot/rbmpp8
RUN=$B/runs/$1; SCRIPT=$2; SPEC=$3
mkdir -p $RUN/code; cp $B/code/*.py $RUN/code/; cp $B/specs/$SPEC $RUN/spec.json
export XLA_FLAGS="--xla_gpu_autotune_level=0" XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.90 OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
{ echo job_id=$SLURM_JOB_ID; hostname; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader; echo started=$(date -Is); sha256sum $RUN/code/*.py; } >> $RUN/run_metadata.txt
cd $RUN/code
/scratch/hpc-prf-ppdmrg/hpcalot/venv_jax_0.8.2/bin/python $SCRIPT $RUN $RUN/spec.json
rc=$?
echo exit=$rc finished=$(date -Is) >> $RUN/run_metadata.txt; exit $rc
