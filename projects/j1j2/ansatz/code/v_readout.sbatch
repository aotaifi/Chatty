#!/bin/bash
#SBATCH --job-name=pf_vrd
#SBATCH --partition=cluster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --no-requeue
# Gate V CPU readout (v_bare.py; copy of r5_readout.sbatch's frame): K, Vm (s_G) once; control theta_4 table e = 1.2184484428720862e-05 (rel 1e-8, abort on mismatch);
# bare e of the pool-fitted theta_bar's table, r_V, verdict.  Sizing: r4 readout MaxRSS 17.1 GB (job 17640414, same K/Vm) -> 96G; K build ~8 min + 2 solves at ~85 s
# + 1 optional augmented scan (~12 solves) -> ~0.5 h expected against 3 h.
set -uo pipefail
C=/project/theorie/a/A.Otaifi/Chatty/organize/projects/j1j2/ansatz/code; O=/scratch/a/A.Otaifi/ansatz_runs/pfaffian_v/readout
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK} OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK} MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
mkdir -p $O; echo "START $(date -Is) $(hostname) job=${SLURM_JOB_ID}" > $O/verdict.txt
/usr/bin/env python3 $C/v_bare.py $O >> $O/readout.log 2>> $O/readout.err; rc=$?
echo "END $(date -Is) readout_rc=$rc" >> $O/verdict.txt
exit $rc
