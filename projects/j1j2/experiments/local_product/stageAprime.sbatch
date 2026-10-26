#!/bin/bash
#SBATCH --partition=cip
#SBATCH --gres=gpu:a40-16gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --no-requeue
#SBATCH --hint=nomultithread
# Stage A' (PREREG.md Amendment 1): cached-G cost test.  usage: sbatch --job-name=NAME --time=T --output=RUN/slurm-%j.out stageAprime.sbatch RUN STAGE
# STAGE = probe | full | full_from16 (full without the L=6,8 correctness part, already in RUN/correct.json).  Writes RUN/progress.txt (PROGRESS lines) and, at the end, 'RUN_VERDICT OK' or 'RUN_VERDICT FAIL rc'.
# XLA autotuning left ON: bench job 17513142 measured conv 0.51 vs 0.21 TFLOP/s with --xla_gpu_autotune_level=0 on the A40-16Q slice.
set -u
RUN=$1; STAGE=$2
export OMPI_MCA_pml=ob1 OMPI_MCA_btl="vader,self,tcp"
VENV=/project/theorie/a/A.Otaifi/Chatty/runs/a40slice/venv_jax_0.8.2
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.85 OMP_NUM_THREADS=4
mkdir -p "$RUN/code"; cp /project/theorie/a/A.Otaifi/Chatty/organize/projects/j1j2/experiments/local_product/gcache.py "$RUN/code/"
{ echo "job_id=$SLURM_JOB_ID"; echo "host=$(hostname)"; echo "stage=$STAGE"; echo "started=$(date -Is)"; sha256sum "$RUN/code/gcache.py";
  git -C /project/theorie/a/A.Otaifi/Chatty/organize rev-parse HEAD; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader; } > "$RUN/run_metadata.txt"
P="$VENV/bin/python $RUN/code/gcache.py $RUN"
rc=0
if [ "$STAGE" = probe ]; then
  $P timing --L 8 20 --B 256 --Bel 16 --nrep 1 --tag _probe || rc=$?
else
  if [ "$STAGE" = full ]; then $P correct --L 6 8 16 --ncfg 100 100 10 || rc=$?; else $P correct --L 16 --ncfg 10 || rc=$?; fi
  [ $rc = 0 ] && { $P timing --L 6 8 12 16 20 --B 256 --Bel 16 --nrep 2 || rc=$?; }
  [ $rc = 0 ] && { $P timing --L 24 32 --B 256 --Bel 16 --nrep 2 --modes cached --tag _ext || rc=$?; }
  [ $rc = 0 ] && { $P timing --L 6 8 12 16 20 --R 1 --B 256 --Bel 16 --nrep 2 --tag _R1 || rc=$?; }
fi
echo "finished=$(date -Is) rc=$rc" >> "$RUN/run_metadata.txt"
if [ $rc = 0 ]; then echo "$(date -Is) RUN_VERDICT OK" >> "$RUN/progress.txt"; else echo "$(date -Is) RUN_VERDICT FAIL $rc" >> "$RUN/progress.txt"; fi
