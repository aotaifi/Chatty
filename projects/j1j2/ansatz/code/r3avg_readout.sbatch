#!/bin/bash
#SBATCH --job-name=pf_r3avgrd
#SBATCH --partition=cluster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --no-requeue
# R3-avg CPU readout: r3avg_bare.py (controls, theta(1200) augmented scan, then waits for theta_bar's table, bare + augmented scan).
# Sizing: MaxRSS 22.1 GB of r3_1500 readout (17629941, same K/Vm) -> 96G = 4.3x; ~85 s/solve, ~30 solves + K build ~ 1 h of work against 4 h wall (wait for the table is idle time, capped at 3.25 h elapsed).
set -uo pipefail
C=/project/theorie/a/A.Otaifi/Chatty/organize/projects/j1j2/ansatz/code; O=/scratch/a/A.Otaifi/ansatz_runs/pfaffian_r3avg/readout
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK} OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK} MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
mkdir -p $O; echo "START $(date -Is) $(hostname) job=${SLURM_JOB_ID}" > $O/verdict.txt
/usr/bin/env python3 $C/r3avg_bare.py $O >> $O/readout.log 2>> $O/readout.err; rc=$?
echo "END $(date -Is) readout_rc=$rc" >> $O/verdict.txt
exit $rc
