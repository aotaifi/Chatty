#!/bin/bash
#SBATCH --job-name=pf_r5rd
#SBATCH --partition=cluster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=05:00:00
#SBATCH --no-requeue
# Round-5 CPU readout (copy of r4_readout.sbatch with r5_bare.py): K, Vm once; control = theta_4's table must give e 1.2184484428720862e-05 rel 1e-8 (abort on mismatch);
# per arm in readout/arms.txt: bare e of theta_5, r_5, augmented c scan.
# Sizing: MaxRSS 17.1 GB of the r4 readout (job 17640414, 8 cpu, same K/Vm, 31.6 min for ~22 solves) -> 96G = 5.6x; up to 4 arms x (1 bare + 10-17 scan) + 1 control
# = 45-73 solves at ~85 s + K build ~8 min -> 1.2-1.9 h expected against 5 h wall.
set -uo pipefail
C=/project/theorie/a/A.Otaifi/Chatty/organize/projects/j1j2/ansatz/code; O=/scratch/a/A.Otaifi/ansatz_runs/pfaffian_r5/readout
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK} OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK} MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
mkdir -p $O; echo "START $(date -Is) $(hostname) job=${SLURM_JOB_ID}" > $O/verdict.txt
/usr/bin/env python3 $C/r5_bare.py $O >> $O/readout.log 2>> $O/readout.err; rc=$?
echo "END $(date -Is) readout_rc=$rc" >> $O/verdict.txt
exit $rc
