#!/bin/bash
#SBATCH --job-name=rb8ev
#SBATCH --account=hpc-prf-ppdmrg
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --no-requeue
#SBATCH --output=/scratch/hpc-prf-ppdmrg/hpcalot/rbmpp8/runs/%x-%j.out
# One independent VMC evaluation chain-set of the retrained 8x8 RBM+PP (fixed parameters): 1024 Markov chains from fresh random
# starts with their own RNG seed, 100 thermalisation sweeps, 1.6e6 samples; writes eval/E_s<seed>.npy (per-sweep means, input of
# quotable.py series) and eval/eval.json. Code = rb8_prod's rbmpp_eval.py (1bfd6576) + rbmpp_lib.py (5fa08acc).
# Usage: sbatch --time=... eval.sbatch PSET PROJ SEED   (spec specs_eval/PSET_PROJ_SEED.json)
set -uo pipefail
PSET=$1; PROJ=$2; SEED=$3; B=/scratch/hpc-prf-ppdmrg/hpcalot/rbmpp8; RUN=$B/runs/rb8_retrain/$PSET/ev_${PROJ}_$SEED
mkdir -p $RUN/code $RUN/eval; cp $B/runs/rb8_prod/code/rbmpp_lib.py $B/runs/rb8_prod/code/rbmpp_eval.py $RUN/code/
cp $B/specs_eval/${PSET}_${PROJ}_$SEED.json $RUN/eval_spec.json
export XLA_FLAGS="--xla_gpu_autotune_level=0" XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.90 OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
{ echo job_id=$SLURM_JOB_ID pset=$PSET proj=$PROJ seed=$SEED; hostname; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader; echo started=$(date -Is); sha256sum $RUN/code/*.py $B/runs/rb8_retrain/$PSET/train/params_final.npz $RUN/eval_spec.json; } >> $RUN/run_metadata.txt
cd $RUN/code
/scratch/hpc-prf-ppdmrg/hpcalot/venv_jax_0.8.2/bin/python rbmpp_eval.py $RUN/eval $RUN/eval_spec.json; rc=$?
echo "exit=$rc finished=$(date -Is)" >> $RUN/run_metadata.txt
echo "DONE_RBMPP8 exit=0 job=$SLURM_JOB_ID pset=$PSET proj=$PROJ seed=$SEED eval_rc=$rc $(date -Is)" >> $RUN/VERDICT_RUN
exit 0
