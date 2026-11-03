#!/bin/bash
#SBATCH --job-name=pool_chka
#SBATCH --partition=cluster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --no-requeue
# pool_fit check (a): (1) pool_ref_r1.py = cx_r1fit.py's own loss/grad/GN step on table-fit batches and its population loss on an orbit subset U;
# (2) pool_checks.py a = the same quantities through pool_fit.py.  Sizing: ref MaxRSS 17.5 GB on ws1 when killed at model setup (Lab 13.7 GB, r5 selftest) -> 48G;
# runtime: r5 selftest_pp2 (12288-orbit Jacobian) 15 min on ws1; here 2 x (~5.6k + ~13k configs fwd+jac) -> ~1 h expected against 4 h.
# usage: sbatch pool_checks_a.sbatch RUNDIR
set -uo pipefail
J=$1; V=/scratch/a/A.Otaifi/ansatz_runs/pfaffian_v; source $V/cpu/env.sh
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
echo "START $(date -Is) $(hostname) job=${SLURM_JOB_ID}" >> $J/readout.log
/usr/bin/time -v $PY $CODE/pool_ref_r1.py $J/ref_r1 $V/cpu/spec_ref_r1.json >> $J/readout.log 2>&1; rc1=$?
echo "ref_rc=$rc1 $(date -Is)" >> $J/readout.log
rc2=9; [ $rc1 = 0 ] && { /usr/bin/time -v $PY $CODE/pool_checks.py a $J/check_a $J/ref_r1 >> $J/readout.log 2>&1; rc2=$?; }
echo "check_rc=$rc2 $(date -Is)" >> $J/readout.log
[ $rc1 = 0 ] && [ $rc2 = 0 ] && echo '{"done": true}' > $J/readout.json
exit $(( rc1 > rc2 ? rc1 : rc2 ))
