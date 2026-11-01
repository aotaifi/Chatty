#!/bin/bash
#SBATCH --job-name=pf_r4rd
#SBATCH --partition=cluster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --no-requeue
# Round-4 CPU readout: r4_bare.py (K, Vm once; control = theta_bar's table must give e 1.2521612014025783e-05 rel 1e-8, abort on mismatch; theta_4 bare e + augmented c scan; bare e of the final iterate and the 6 checkpoints).
# Sizing: MaxRSS 22.1 GB of the r3_1500 readout (job 17629941, 8 cpu, same K/Vm) -> 96G = 4.3x; ~85 s per solve, ~22 solves + K build (~30 min) ~ 1.2 h expected against 4 h wall.
# Output pfaffian_r4/readout/{readout.json,readout.log,verdict.txt}; winner cluster from readout/winner.txt (written by r4_chain.sh).
set -uo pipefail
C=/project/theorie/a/A.Otaifi/Chatty/organize/projects/j1j2/ansatz/code; O=/scratch/a/A.Otaifi/ansatz_runs/pfaffian_r4/readout
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK} OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK} MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
mkdir -p $O; echo "START $(date -Is) $(hostname) job=${SLURM_JOB_ID}" > $O/verdict.txt
/usr/bin/env python3 $C/r4_bare.py $O >> $O/readout.log 2>> $O/readout.err; rc=$?
echo "END $(date -Is) readout_rc=$rc" >> $O/verdict.txt
exit $rc
