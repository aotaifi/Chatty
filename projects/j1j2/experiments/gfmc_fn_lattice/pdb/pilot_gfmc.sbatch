#!/bin/bash
#SBATCH --job-name=gfp
#SBATCH --account=hpc-prf-ppdmrg
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --no-requeue
#SBATCH --qos=express_hpcalot
#SBATCH --output=/scratch/hpc-prf-ppdmrg/hpcalot/chatty_gfmc/runs/%x-%j.out
set -uo pipefail
B=/scratch/hpc-prf-ppdmrg/hpcalot/chatty_gfmc; C=/scratch/hpc-prf-ppdmrg/hpcalot/chatty_compress6
RUN=$B/runs/pilot_${SLURM_JOB_ID}; mkdir -p $RUN/code; cp $B/code/*.py $RUN/code/
export GFMC_PARAMS=$B/data/params_final.npz GFMC_TABLE_A=$B/data/table_A.json
export XLA_FLAGS="--xla_gpu_autotune_level=0" XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.85 OMP_NUM_THREADS=8
{ echo job_id=$SLURM_JOB_ID; hostname; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader; date -Is; sha256sum $RUN/code/*.py; } > $RUN/run_metadata.txt
cd $RUN/code
$C/../venv_jax_0.8.2/bin/python gfmc6_run.py $RUN --M ${PM:-512} --dtau ${PDT:-0.1} --blocks 100000 --burn 10 --seed 1 --chunk ${PCH:-512} --maxsec ${PMAXSEC:-1080} > $RUN/stdout.txt 2>&1; rc=$?
echo exit=$rc finished=$(date -Is) >> $RUN/run_metadata.txt; exit $rc
