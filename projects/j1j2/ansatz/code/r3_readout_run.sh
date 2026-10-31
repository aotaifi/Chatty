#!/bin/bash
# usage: r3_readout_run.sh ARM (r3_900|r3_1200|r3_1500): CPU readout on this node, verdict file in the readout dir
ARM=$1; C=/project/theorie/a/A.Otaifi/Chatty/organize/projects/j1j2/ansatz/code; O=/scratch/a/A.Otaifi/ansatz_runs/pfaffian_r3/readout/$ARM
mkdir -p $O; echo "START $(date -Is) $(hostname) ppid=$PPID" > $O/verdict.txt
python3 $C/r3_readout.py $ARM $O > $O/readout.log 2> $O/readout.err; rc1=$?
echo "END $(date -Is) readout_rc=$rc1" >> $O/verdict.txt
