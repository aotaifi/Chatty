#!/bin/bash
#SBATCH --job-name=l8diag
#SBATCH --account=hpc-prf-ppdmrg
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --no-requeue
#SBATCH --output=/scratch/hpc-prf-ppdmrg/hpcalot/l8/runs/diag-%j.out
# L8 refit diagnosis (l8_diag.py) on the production dumps D_s1, D_s2 at v_0 = rb8_prod; no GFMC.  usage:
#   sbatch --qos=Q --time=T --export=ALL,PARTS=a|b|a,b diag_l8.sbatch
set -uo pipefail
B=/scratch/hpc-prf-ppdmrg/hpcalot; PY=$B/venv_jax_0.8.2/bin/python; R=$B/l8/runs/prod
RUN=$B/l8/runs/diag_$SLURM_JOB_ID; mkdir -p $RUN/code; cp $B/l8/code/*.py $RUN/code/
export XLA_FLAGS="--xla_gpu_autotune_level=0" XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 OMP_NUM_THREADS=8
{ echo job_id=$SLURM_JOB_ID; hostname; nvidia-smi --query-gpu=name --format=csv,noheader; date -Is; echo PARTS=${PARTS:-a,b}; sha256sum $B/rbmpp8/runs/rb8_prod/params_final.npz $RUN/code/*.py; } > $RUN/run_metadata.txt
cd $RUN/code
$PY l8_diag.py $RUN --params0 $B/rbmpp8/runs/rb8_prod/params_final.npz --consts $B/gfmc8/runs/rb8/V/consts.json --dumps $R/D_s1,$R/D_s2 --L 8 \
   --parts ${PARTS:-a,b} > $RUN/stdout.txt 2>&1; rc=$?
echo exit=$rc finished=$(date -Is) >> $RUN/run_metadata.txt; exit $rc
