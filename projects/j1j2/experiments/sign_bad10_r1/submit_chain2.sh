#!/bin/bash
# T-BAD10 round-1 RESTART chain (after the VMC gate stopped the first chain): e0_m64_c2 -> vmc_exact -> labels -> fit -> fn64, afterok, one setsid watcher each.
# usage: submit_chain2.sh [--after JOBID] [--go]
R=/project/theorie/a/A.Otaifi/runs/j1j2/sign_bad10/r1; GO=0; AFTER=""
while [ $# -gt 0 ]; do case $1 in --go) GO=1;; --after) AFTER=$2; shift;; esac; shift; done
sub() { if [ $GO = 1 ]; then sbatch --parsable "$@"; else echo "DRY sbatch $*" >&2; echo DRY$RANDOM; fi; }
arm() { if [ $GO = 1 ]; then setsid nohup $R/watch2.sh $1 $2 $3/slurm_$1.out $4 $5 $3 > /dev/null 2>&1 < /dev/null & else echo "DRY watcher job $1 marker $2 dir $3" >&2; fi; }
DEP=""; [ -n "$AFTER" ] && DEP="--dependency=afterok:$AFTER"
J1=$(sub $DEP $R/e0m64c2.sbatch);                         arm $J1 E0M64C2_DONE $R/m64 $R/m64/all_c2.log 1200
J2=$(sub --dependency=afterok:$J1 $R/vmc_exact.sbatch);   arm $J2 VMCX_DONE $R/vmc_exact $R/vmc_exact/all.log 1500
J3=$(sub --dependency=afterok:$J2 $R/labels.sbatch);      arm $J3 LABELS_DONE $R/labels $R/labels/all.log 2700
J4=$(sub --dependency=afterok:$J3 $R/fit.sbatch);         arm $J4 FIT_DONE $R/fit $R/fit/all.log 1500
J5=$(sub --dependency=afterok:$J4 $R/fn64.sbatch);        arm $J5 FN64_DONE $R/fn_net64 $R/fn_net64/all.log 1500
echo "STAGE_JOBS e0m64c2=$J1 vmc_exact=$J2 labels=$J3 fit=$J4 fn64=$J5"; [ $GO = 1 ] && echo "restart chain: $J1 $J2 $J3 $J4 $J5" >> $R/chain_jobs.txt
