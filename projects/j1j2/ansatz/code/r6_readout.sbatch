#!/bin/bash
#SBATCH --job-name=pf_r6rd
#SBATCH --partition=cluster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --no-requeue
# Round-6 CPU readout (copy of r5_readout.sbatch with r6_bare.py): K, Vm once; controls = theta_4's table e 1.2184484428720862e-05 and pp2_s1's theta_5 table e 8.331857874447715e-06
# (rel 1e-8 each, abort on mismatch); per arm in readout/arms.txt (r6, m5): bare e of theta_R, r_R, augmented c scan.
# Sizing: MaxRSS 17.2 GB of the r5 readout (job 17672082, 8 cpu, same K/Vm, 1.81 h for 1 control + 4 arms x (bare + 10-12 scan) ~ 50 solves) -> 96G = 5.6x;
# round 6 = 2 controls + 2 arms x (1 bare + 10-17 scan) = 24-38 solves -> 0.9-1.4 h expected against 4 h wall.
set -uo pipefail
C=/project/theorie/a/A.Otaifi/Chatty/organize/projects/j1j2/ansatz/code; O=/scratch/a/A.Otaifi/ansatz_runs/pfaffian_r6/readout
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK} OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK} MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
mkdir -p $O; echo "START $(date -Is) $(hostname) job=${SLURM_JOB_ID}" > $O/verdict.txt
/usr/bin/env python3 $C/r6_bare.py $O >> $O/readout.log 2>> $O/readout.err; rc=$?
echo "END $(date -Is) readout_rc=$rc" >> $O/verdict.txt
exit $rc
