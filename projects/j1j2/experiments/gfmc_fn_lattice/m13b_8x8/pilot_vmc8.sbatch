#!/bin/bash
#SBATCH --job-name=g8vmc
#SBATCH --account=hpc-prf-ppdmrg
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --no-requeue
#SBATCH --qos=express_hpcalot
#SBATCH --output=/scratch/hpc-prf-ppdmrg/hpcalot/gfmc8/runs/%x-%j.out
# M1.3b pilot: eval timing vs chunk, then the 8x8 VMC p=0 / p=1 moment estimator (vmc_moments.py) on a pilot parameter file.
# usage: sbatch [--export=ALL,PARAMS=...,CH=256,CHAINS=1024,SWEEPS=16,SEEDS=1,2] pilot_vmc8.sbatch
set -uo pipefail
B=/scratch/hpc-prf-ppdmrg/hpcalot/gfmc8
PARAMS=${PARAMS:-/scratch/hpc-prf-ppdmrg/hpcalot/rbmpp8/runs/pilot8_long/params_final.npz}
RUN=$B/runs/vmcpilot_${SLURM_JOB_ID}; mkdir -p $RUN/code; cp $B/code/*.py $RUN/code/
export XLA_FLAGS="--xla_gpu_autotune_level=0" XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 OMP_NUM_THREADS=8
{ echo job_id=$SLURM_JOB_ID; hostname; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader; date -Is; echo params=$PARAMS; sha256sum $PARAMS $RUN/code/*.py; } > $RUN/run_metadata.txt
cd $RUN/code; PY=$B/../venv_jax_0.8.2/bin/python
$PY timing8.py $RUN/timing8.json $PARAMS 8 translations 64 128 256 512 > $RUN/timing.txt 2>&1
$PY vmc_moments.py $RUN --params $PARAMS --L 8 --proj translations --chains ${CHAINS:-1024} --sweeps ${SWEEPS:-16} --seeds ${SEEDS:-1,2} --chunk ${CH:-256} --maxsec_seed ${MAXSEC:-420} > $RUN/stdout.txt 2>&1; rc=$?
echo exit=$rc finished=$(date -Is) >> $RUN/run_metadata.txt; echo "GFMC_DONE exit=$rc" >> $RUN/progress.txt; exit $rc
