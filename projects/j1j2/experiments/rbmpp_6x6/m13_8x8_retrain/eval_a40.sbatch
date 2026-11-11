#!/bin/bash
#SBATCH --job-name=rb8evA40
#SBATCH --partition=cip
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --no-requeue
# A40 twin of eval.sbatch: same rbmpp_eval.py (1bfd6574) + rbmpp_lib.py (5fa08acc), same spec format, jax 0.8.2 venv.
# Usage: sbatch --time=... eval_a40.sbatch SEED   (spec specs_eval/s307_full_SEED.json; run dir s307/ev_full_SEED)
set -uo pipefail
SEED=$1; H=/project/theorie/a/A.Otaifi/Chatty/organize/projects/j1j2/experiments/rbmpp_6x6/m13_8x8_retrain
RUN=/scratch/a/A.Otaifi/rbmpp8/rb8_retrain/s307/ev_full_$SEED; SRC=/scratch/a/A.Otaifi/rbmpp8/rb8_retrain/s307/ev_full_a40probe/code
mkdir -p $RUN/code $RUN/eval; cp $SRC/rbmpp_lib.py $SRC/rbmpp_eval.py $RUN/code/
python3 - <<PY
import json
s=json.load(open("$H/specs_eval/s307_full_$SEED.json")); s["params"]="/scratch/a/A.Otaifi/rbmpp8/rb8_retrain/s307/train/params_final.npz"
json.dump(s,open("$RUN/eval_spec.json","w"))
PY
export XLA_FLAGS="--xla_gpu_autotune_level=0" XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.90 OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
{ echo job_id=$SLURM_JOB_ID seed=$SEED; hostname; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader; echo started=$(date -Is); sha256sum $RUN/code/*.py /scratch/a/A.Otaifi/rbmpp8/rb8_retrain/s307/train/params_final.npz $RUN/eval_spec.json; } >> $RUN/run_metadata.txt
cd $RUN/code
/project/theorie/a/A.Otaifi/Chatty/runs/a40slice/venv_jax_0.8.2/bin/python -u rbmpp_eval.py $RUN/eval $RUN/eval_spec.json > $RUN/eval.log 2>&1; rc=$?
echo "exit=$rc finished=$(date -Is)" >> $RUN/run_metadata.txt
echo "DONE_RBMPP8 exit=0 job=$SLURM_JOB_ID pset=s307 proj=full seed=$SEED eval_rc=$rc $(date -Is)" >> $RUN/VERDICT_RUN
exit 0
