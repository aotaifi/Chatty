#!/bin/bash
#SBATCH --partition=cluster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --no-requeue
# fn_s0.py: exact-sign additions A-D (see fn_s0.py docstring); sbatch --export=ALL,OUTD=...
# Sizing: MaxRSS 22.1 GB of r3_1500 readout (job 17629941, same K/Vm), 96G = 4.3x; ~90 s/solve at tol 1e-6 (21 it), tol 1e-8 maybe ~35 it -> ~150 s x 41 + ~30 min build ~ 2.2 h against 6 h wall (limit asked by lead).
set -uo pipefail
C=/project/theorie/a/A.Otaifi/Chatty/organize/projects/j1j2/ansatz/code
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK} OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK} MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
mkdir -p $OUTD; echo "START $(date -Is) $(hostname) job=${SLURM_JOB_ID}" > $OUTD/verdict.txt
/usr/bin/env python3 $C/fn_s0.py $OUTD >> $OUTD/floor.log 2>> $OUTD/floor.err; rc=$?
echo "END $(date -Is) rc=$rc" >> $OUTD/verdict.txt
exit $rc
