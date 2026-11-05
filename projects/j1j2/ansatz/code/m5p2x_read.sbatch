#!/bin/bash
#SBATCH --job-name=pf_m5p2e1rd
#SBATCH --partition=cluster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --no-requeue
# M5' RERUN, EXTENSION SEGMENT 1 CPU readout: m5p2x_read.py final (tail average 1200..2100) + m5p2x_plateau.py.  usage: m5p2x_read.sbatch OUTDIR [final]
# Sizing (measured, 1500 run readout job 17761847 and test job 17748850): MaxRSS 17.3 GB, 96G = 5.5x; ~1 h against the 04:00 wall.
set -uo pipefail
C=/project/theorie/a/A.Otaifi/Chatty/organize/projects/j1j2/ansatz/code; O=$1; MODE=${2:-final}
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK} OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK} MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
mkdir -p $O; echo "START $(date -Is) $(hostname) job=${SLURM_JOB_ID} mode=$MODE" >> $O/verdict.txt
rc=0
/usr/bin/env python3 $C/m5p2x_read.py $O final >> $O/readout.log 2>> $O/readout.err; rc=$?
echo "m5p2x_read rc=$rc $(date -Is)" >> $O/verdict.txt
/usr/bin/env python3 $C/m5p2x_plateau.py $O >> $O/plateau.log 2>> $O/plateau.err; rp=$?
[ $rp -gt $rc ] && rc=$rp
echo "END $(date -Is) readout_rc=$rc plateau_rc=$rp" >> $O/verdict.txt
exit $rc
