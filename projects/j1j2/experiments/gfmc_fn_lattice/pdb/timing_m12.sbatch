#!/bin/bash
#SBATCH --job-name=gft
#SBATCH --account=hpc-prf-ppdmrg
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:h100:1
#SBATCH --qos=express_hpcalot
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --no-requeue
#SBATCH --output=/scratch/hpc-prf-ppdmrg/hpcalot/chatty_gfmc/runs/%x-%j.out
set -uo pipefail
B=/scratch/hpc-prf-ppdmrg/hpcalot/chatty_gfmc; C=/scratch/hpc-prf-ppdmrg/hpcalot/chatty_compress6
RUN=$B/runs/timing_${SLURM_JOB_ID}; mkdir -p $RUN/code; cp $B/code/*.py $RUN/code/
export XLA_FLAGS="--xla_gpu_autotune_level=0" XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
{ echo job_id=$SLURM_JOB_ID; hostname; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader; date -Is; } > $RUN/run_metadata.txt
cd $RUN/code
$C/../venv_jax_0.8.2/bin/python timing_m12.py $RUN/timing.json 6:full:256 6:trans:256 6:none:512 8:none:512 8:trans:64 8:full:16 14:none:128 14:trans:8 14:full:1 > $RUN/progress.txt 2>&1; rc=$?
echo "GFMC_DONE exit=$rc" >> $RUN/progress.txt; echo exit=$rc finished=$(date -Is) >> $RUN/run_metadata.txt; exit $rc
