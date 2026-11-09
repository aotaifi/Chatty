#!/bin/bash
#SBATCH --job-name=rb8sh
#SBATCH --account=hpc-prf-ppdmrg
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --no-requeue
#SBATCH --output=/scratch/hpc-prf-ppdmrg/hpcalot/rbmpp8/runs/%x-%j.out
# SR shift scan, one arm per job: warm start rb8_prod params_final.npz (sha 0520c5be), 'simple' projection (256 images),
# lr 0.02 constant, relative diagonal shift = arm value (constant), 280 steps x 8192 samples. Code = rb8_prod copies
# (rbmpp_lib 5fa08acc, rbmpp_train 475c175f). Hard cap 1900 s on the python process. Usage: sbatch ... shiftscan.sbatch ARM
set -uo pipefail
ARM=$1; B=/scratch/hpc-prf-ppdmrg/hpcalot/rbmpp8; RUN=$B/runs/rb8_shiftscan/$ARM
mkdir -p $RUN/code $RUN/train; cp $B/runs/rb8_prod/code/rbmpp_lib.py $B/runs/rb8_prod/code/rbmpp_train.py $RUN/code/
cp $B/specs_shiftscan/$ARM.json $RUN/train/spec.json
export XLA_FLAGS="--xla_gpu_autotune_level=0" XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.90 OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
{ echo job_id=$SLURM_JOB_ID arm=$ARM; hostname; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader; echo started=$(date -Is); sha256sum $RUN/code/*.py $B/runs/rb8_prod/params_final.npz $RUN/train/spec.json; } >> $RUN/run_metadata.txt
cd $RUN/code
timeout 1900 /scratch/hpc-prf-ppdmrg/hpcalot/venv_jax_0.8.2/bin/python rbmpp_train.py $RUN/train $RUN/train/spec.json; rc=$?
echo "exit=$rc finished=$(date -Is)" >> $RUN/run_metadata.txt
echo "DONE_RBMPP8 exit=0 job=$SLURM_JOB_ID arm=$ARM train_rc=$rc $(date -Is)" >> $RUN/VERDICT_RUN
exit 0
