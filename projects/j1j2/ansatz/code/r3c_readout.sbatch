#!/bin/bash
#SBATCH --job-name=pf_r3crd
#SBATCH --partition=cluster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --no-requeue
# R3-cont CPU readout: r3c_bare.py (control on the round-3 1500 table first, abort on mismatch; bare e at 10 tables; augmented scan at the 2 final tables).
# Sizing: MaxRSS 22.1 GB of the r3_1500 readout (job 17629941, 8 cpu, 14 min, same K/Vm) -> 96G = 4.3x; ~85 s per solve -> ~1.2 h expected against 8 h.  Output pfaffian_r3c/readout/{readout.json,readout.log,verdict.txt}
set -uo pipefail
C=/project/theorie/a/A.Otaifi/Chatty/organize/projects/j1j2/ansatz/code; O=/scratch/a/A.Otaifi/ansatz_runs/pfaffian_r3c/readout
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK} OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK} MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
mkdir -p $O; echo "START $(date -Is) $(hostname) job=${SLURM_JOB_ID}" > $O/verdict.txt
/usr/bin/env python3 $C/r3c_bare.py $O >> $O/readout.log 2>> $O/readout.err; rc=$?
echo "END $(date -Is) readout_rc=$rc" >> $O/verdict.txt
exit $rc
