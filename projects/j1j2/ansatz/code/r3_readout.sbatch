#!/bin/bash
#SBATCH --job-name=pf_r3rd
#SBATCH --partition=cluster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --no-requeue
# Round 3, CPU readout of the step-1500 table (arm r3_1500), resuming the c-scan from readout/r3_1500/readout.json.  The workstation runs of this arm were killed 4x (exit 143/137, ~17 GB RSS; cause unconfirmed).
# usage: r3_readout.sbatch   (writes readout/r3_1500/{readout.json,readout.log,verdict.txt})
set -uo pipefail
C=/project/theorie/a/A.Otaifi/Chatty/organize/projects/j1j2/ansatz/code; O=/scratch/a/A.Otaifi/ansatz_runs/pfaffian_r3/readout/r3_1500
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK} OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK} MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
echo "START $(date -Is) $(hostname) job=${SLURM_JOB_ID}" > $O/verdict.txt
/usr/bin/env python3 $C/r3_readout.py r3_1500 $O >> $O/readout.log 2>> $O/readout.err; rc=$?
echo "END $(date -Is) readout_rc=$rc" >> $O/verdict.txt
exit $rc
