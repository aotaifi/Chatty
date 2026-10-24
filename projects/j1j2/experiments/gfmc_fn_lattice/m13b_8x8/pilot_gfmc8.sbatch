#!/bin/bash
#SBATCH --job-name=g8gf
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
# M1.3b pilot: 8x8 fixed-node GFMC, guide p = 0 or 1.  usage:
#   sbatch --export=ALL,PARAMS=...,CONSTS=.../vmc_s1.json,P=1,M=128,DT=0.1,MAXSEC=1200,CH=256 pilot_gfmc8.sbatch
set -uo pipefail
B=/scratch/hpc-prf-ppdmrg/hpcalot/gfmc8
PARAMS=${PARAMS:-/scratch/hpc-prf-ppdmrg/hpcalot/rbmpp8/runs/pilot8_long/params_final.npz}
CONSTS=${CONSTS:?CONSTS (vmc_s1.json)}; P=${P:-1}
RUN=$B/runs/gfpilot_p${P}_${SLURM_JOB_ID}; mkdir -p $RUN/code; cp $B/code/*.py $RUN/code/
export XLA_FLAGS="--xla_gpu_autotune_level=0" XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 OMP_NUM_THREADS=8
{ echo job_id=$SLURM_JOB_ID; hostname; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader; date -Is; echo params=$PARAMS consts=$CONSTS; sha256sum $PARAMS $CONSTS $RUN/code/*.py; } > $RUN/run_metadata.txt
cd $RUN/code
$B/../venv_jax_0.8.2/bin/python gfmc_run.py $RUN --params $PARAMS --consts $CONSTS --p $P --L 8 --proj translations --M ${M:-128} --dtau ${DT:-0.1} --blocks 100000 --burn ${BURN:-20} --seed ${SEED:-1} --chunk ${CH:-256} --maxsec ${MAXSEC:-1200} > $RUN/stdout.txt 2>&1; rc=$?
echo exit=$rc finished=$(date -Is) >> $RUN/run_metadata.txt; exit $rc
