#!/bin/bash
#SBATCH --job-name=rbLprod
#SBATCH --account=hpc-prf-ppdmrg
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --no-requeue
#SBATCH --output=/scratch/hpc-prf-ppdmrg/hpcalot/rbmppL/runs/%x-%j.out
# M1 LxL production (generalisation of ../m13_8x8/rbmpp8_prod.sbatch, which is unchanged): train (resumes from state.npz when the run directory
# exists) then the independent evaluation.
# Usage: sbatch --qos=QOS --time=HH:MM:SS rbmppL_prod.sbatch RUNNAME TRAIN_SPEC.json EVAL_SPEC.json
# Re-submitting the same RUNNAME continues the run (spec "resume": true); the evaluation runs only when train finished.
# The last line of RUN/VERDICT_RUN is "DONE_RBMPP8 exit=0" only if both stages finished (same marker as 8x8, so the watcher is shared).
set -uo pipefail
B=/scratch/hpc-prf-ppdmrg/hpcalot/rbmppL
RUN=$B/runs/$1; TS=$2; ES=$3
mkdir -p $RUN/code; cp $B/code/*.py $RUN/code/; cp $B/specs/$TS $RUN/spec.json
sed "s#@RUN@#$RUN#g" $B/specs/$ES > $RUN/eval_spec.json
export XLA_FLAGS="--xla_gpu_autotune_level=0" XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.90 OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
{ echo job_id=$SLURM_JOB_ID; hostname; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader; echo started=$(date -Is); sha256sum $RUN/code/*.py; } >> $RUN/run_metadata.txt
PY=/scratch/hpc-prf-ppdmrg/hpcalot/venv_jax_0.8.2/bin/python
cd $RUN/code
$PY rbmpp_train.py $RUN $RUN/spec.json; rc=$?
if [ $rc -eq 0 ] && [ -f $RUN/params_final.npz ]; then $PY rbmpp_eval.py $RUN/eval $RUN/eval_spec.json; rc=$?; fi
echo "exit=$rc finished=$(date -Is)" >> $RUN/run_metadata.txt
echo "DONE_RBMPP8 exit=$rc job=$SLURM_JOB_ID $(date -Is)" >> $RUN/VERDICT_RUN
exit $rc
