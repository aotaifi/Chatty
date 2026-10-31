#!/bin/bash
# usage: r2g_readout_run.sh ARM (g1|g2): readout then c-fill on this node, verdict file in the readout dir
ARM=$1; C=/project/theorie/a/A.Otaifi/Chatty/organize/projects/j1j2/ansatz/code; O=/scratch/a/A.Otaifi/ansatz_runs/pfaffian_r2g/readout/$ARM
mkdir -p $O; echo "START $(date -Is) $(hostname) ppid=$PPID" > $O/verdict.txt
python3 $C/r2g_readout.py $ARM $O > $O/readout.log 2> $O/readout.err; rc1=$?
if [ $rc1 -eq 0 ] && grep -q "DONE" $O/readout.log; then python3 $C/r2g_cfill.py $ARM > $O/cfill.log 2> $O/cfill.err; rc2=$?; else rc2=skipped; fi
echo "END $(date -Is) readout_rc=$rc1 cfill_rc=$rc2" >> $O/verdict.txt
