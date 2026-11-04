#!/bin/bash
#SBATCH --job-name=pf_m5p2rd
#SBATCH --partition=cluster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --no-requeue
# M5' RERUN CPU readout = m5p_read.sbatch (m5p2_read.py = m5p_read.py with only Q changed) + the plateau step (m5p2_plateau.py).  usage: m5p2_read.sbatch OUTDIR [final|selftest_plateau]
# Sizing (measured): MaxRSS 17.3 GB (selftest 17703286) / 17.0 GB (rel readout 17720247), 96G = 5.5x; each step builds K,Vm (3 min) and does ~2.6 min per exact solve.
# final: control + e_rel + e_end + 5-8 scan solves ~0.5-0.8 h; plateau step: build + 2 solves ~0.15 h; total ~1 h against 04:00 wall (<= half).
set -uo pipefail
C=/project/theorie/a/A.Otaifi/Chatty/organize/projects/j1j2/ansatz/code; O=$1; MODE=${2:-final}
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK} OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK} MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
mkdir -p $O; echo "START $(date -Is) $(hostname) job=${SLURM_JOB_ID} mode=$MODE" >> $O/verdict.txt
rc=0
if [ "$MODE" = final ]; then
  /usr/bin/env python3 $C/m5p2_read.py $O final >> $O/readout.log 2>> $O/readout.err; rc=$?
  echo "m5p2_read rc=$rc $(date -Is)" >> $O/verdict.txt
fi
# plateau step runs even if the readout failed (independent data), exit code = max
/usr/bin/env python3 $C/m5p2_plateau.py $O >> $O/plateau.log 2>> $O/plateau.err; rp=$?
[ $rp -gt $rc ] && rc=$rp
echo "END $(date -Is) readout_rc=$rc plateau_rc=$rp" >> $O/verdict.txt
exit $rc
