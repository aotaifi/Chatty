#!/bin/bash
# T-BAD10 round 1: submit the five stages chained with afterok and arm a durable setsid watcher per stage in the same action.  NOT RUN until the lead approves.
# usage: submit_chain.sh [--go]   (without --go: prints what it would do)
R=/project/theorie/a/A.Otaifi/runs/j1j2/sign_bad10/r1; GO=0; [ "$1" = --go ] && GO=1
sub() { if [ $GO = 1 ]; then sbatch --parsable "$@"; else echo "DRY sbatch $*" >&2; echo DRY$RANDOM; fi; }
arm() { # job marker dir stall
  if [ $GO = 1 ]; then setsid nohup $R/watch2.sh $1 $2 $3/slurm_$1.out $3/all.log $4 $3 > /dev/null 2>&1 < /dev/null & else echo "DRY watcher job $1 marker $2 dir $3 stall $4" >&2; fi; }
J1=$(sub $R/e0ext.sbatch);                                  arm $J1 E0EXT_DONE $R/e0ext 900
J2=$(sub --dependency=afterok:$J1 $R/vmc.sbatch);          arm $J2 VMC_DONE $R/vmc 1200
J3=$(sub --dependency=afterok:$J2 $R/labels.sbatch);       arm $J3 LABELS_DONE $R/labels 2700
J4=$(sub --dependency=afterok:$J3 $R/fit.sbatch);          arm $J4 FIT_DONE $R/fit 1500
J5=$(sub --dependency=afterok:$J4 $R/fn.sbatch);           arm $J5 FN_R1_DONE $R/fn_net 900
echo "STAGE_JOBS e0ext=$J1 vmc=$J2 labels=$J3 fit=$J4 fn=$J5"; [ $GO = 1 ] && echo "$J1 $J2 $J3 $J4 $J5" > $R/chain_jobs.txt
