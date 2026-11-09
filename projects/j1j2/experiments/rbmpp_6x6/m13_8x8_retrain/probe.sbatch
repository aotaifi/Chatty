#!/bin/bash
#SBATCH --job-name=rb8pg
#SBATCH --account=hpc-prf-ppdmrg
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --no-requeue
#SBATCH --output=/scratch/hpc-prf-ppdmrg/hpcalot/rbmpp8/runs/%x-%j.out
# Cost probe for the 8x8 base retrain (point-group projection, warm start from rb8_prod params_final.npz sha 0520c5be).
# Code = rb8_prod's own copies (rbmpp_lib 5fa08acc, rbmpp_train 475c175f). Three stages, each under a hard timeout:
#  a0: 'simple' (T x {E,C4} x flip, 256 images), lr 0, 20 steps  -> step-0 energy with the published projection, s/step
#  a : 'simple', SR lr 0.02 shift 0.01, 100 steps                 -> energy trend under SR, s/step
#  b : 'full' (T x C4v x flip, 1024 images), lr 0, 12 steps       -> step-0 energy with full projection, s/step
set -uo pipefail
B=/scratch/hpc-prf-ppdmrg/hpcalot/rbmpp8; RUN=$B/runs/rb8_pgprobe
mkdir -p $RUN/code; cp $B/runs/rb8_prod/code/rbmpp_lib.py $B/runs/rb8_prod/code/rbmpp_train.py $RUN/code/
export XLA_FLAGS="--xla_gpu_autotune_level=0" XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.90 OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
{ echo job_id=$SLURM_JOB_ID; hostname; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader; echo started=$(date -Is); sha256sum $RUN/code/*.py $B/runs/rb8_prod/params_final.npz; } >> $RUN/run_metadata.txt
PY=/scratch/hpc-prf-ppdmrg/hpcalot/venv_jax_0.8.2/bin/python
cd $RUN/code; rcs=""
for st in "a0 a0_simple_lr0 330" "a a_simple_sr 840" "b b_full_lr0 480"; do
  set -- $st; mkdir -p $RUN/$1; cp $B/specs_pgprobe/$2.json $RUN/$1/spec.json
  echo "STAGE $1 start $(date -Is)"; timeout $3 $PY rbmpp_train.py $RUN/$1 $RUN/$1/spec.json; rc=$?; rcs="$rcs $1=$rc"
  echo "STAGE $1 rc=$rc $(date -Is)"
done
echo "exit:$rcs finished=$(date -Is)" >> $RUN/run_metadata.txt
echo "DONE_RBMPP8 exit=0 job=$SLURM_JOB_ID stages:$rcs $(date -Is)" >> $RUN/VERDICT_RUN
exit 0
