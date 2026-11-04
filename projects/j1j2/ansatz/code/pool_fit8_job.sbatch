#!/bin/bash
#SBATCH --partition=cluster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --no-requeue
#SBATCH --hint=nomultithread
# pool_fit 8x8 / 6x6 checks as one CPU job per piece.  usage: sbatch --job-name=pf8_X --cpus-per-task=N --mem=MG --time=HH:MM:SS --export=ALL,WHAT=...,OUTD=... pool_fit8_job.sbatch
# WHAT: t8abcd (pool_fit8_test.py a b c d) | t8e (e, small batches) | t8e2 (param-space solve timing) | c6a | c6b | c6c  (existing 6x6 checks of pool_checks.py, unchanged)
# Sizing is in each run dir's ANALYSIS_BRIEF.md.  The run's own verdict line is OUTD/VERDICT_LINE (written last).
set -uo pipefail
C=/project/theorie/a/A.Otaifi/Chatty/organize/projects/j1j2/ansatz/code
V=/scratch/a/A.Otaifi/ansatz_runs/pfaffian_v; source $V/cpu/env.sh
export OMPI_MCA_pml=ob1 OMPI_MCA_btl="vader,self,tcp"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK} OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK} MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
mkdir -p $OUTD; echo "START $(date -Is) $(hostname) job=${SLURM_JOB_ID} cpus=${SLURM_CPUS_PER_TASK}" > $OUTD/verdict.txt
T="/usr/bin/time -v"
case $WHAT in
  t8abcd) for k in a d b c; do $T $PY $C/pool_fit8_test.py $OUTD $k >> $OUTD/run.log 2>&1 || rc=$?; done;;
  recheck) $T $PY $C/pool_fit8_test.py $OUTD a >> $OUTD/run.log 2>&1 || rc=$?; $T $PY $C/pool_fit8_test.py $OUTD d >> $OUTD/run.log 2>&1 || rc=$?
          mkdir -p $OUTD/dedup; PP2_DEFAULT_PATH=dedup $T $PY $C/pool_fit8_test.py $OUTD/dedup a >> $OUTD/run.log 2>&1 || rc=$?
          PP2_DEFAULT_PATH=dedup $T $PY $C/pool_fit8_test.py $OUTD/dedup d >> $OUTD/run.log 2>&1 || rc=$?;;
  t8e)    $T $PY $C/pool_fit8_test.py $OUTD e >> $OUTD/run.log 2>&1; rc=$?;;
  t8e2)   $T $PY $C/pool_fit8_test.py $OUTD e2 >> $OUTD/run.log 2>&1; rc=$?;;
  c6a)    $T $PY $C/pool_checks.py a $OUTD/check_a $V/cpu/checka_job/ref_r1 >> $OUTD/run.log 2>&1; rc=$?;;
  c6b)    mkdir -p $OUTD/check_b; cp $V/cpu/check_b/W_exact.npy $V/cpu/check_b/W_exact.npy.key $OUTD/check_b/; $T $PY $C/pool_checks.py b $OUTD/check_b $V/pool >> $OUTD/run.log 2>&1; rc=$?;;
  c6c)    mkdir -p $OUTD/check_c_pert; ln -sfn $V/pool $OUTD/check_c_pert/pool_small   # the baseline run read the V pool through this symlink
          $T $PY $C/pool_checks.py c $OUTD/check_c_pert $V/cpu/spec_check_c_pert.json >> $OUTD/run.log 2>&1; rc=$?;;
  *) echo "bad WHAT $WHAT" >> $OUTD/run.log; rc=9;;
esac
rc=${rc:-0}
echo "END $(date -Is) rc=$rc" >> $OUTD/verdict.txt
python3 $C/pool_fit8_gate.py $OUTD $WHAT > $OUTD/VERDICT_LINE 2>> $OUTD/run.log
exit $rc
