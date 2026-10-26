#!/bin/bash
# submit diag_l8.sbatch (QOS, TIME, PARTS from the environment) and arm its durable watcher in the same action.
set -u
HERE=$(cd "$(dirname "$0")" && pwd); CODE=$HERE/..; RBMPP=$CODE/../rbmpp_6x6
B=/scratch/hpc-prf-ppdmrg/hpcalot; SSH="ssh -o BatchMode=yes -o HostName=login4"; SL=/opt/software/slurm/25.11.8-1/bin
QOS=${QOS:?}; TIME=${TIME:?}; PARTS=${PARTS:-a,b}
rsync -a -e "$SSH" $CODE/{gfmc_fn_lib,gfmc_run,vmc_moments}.py $HERE/{l8_lib,l8_refit,l8_vmc_prefix,l8_diag}.py $RBMPP/rbmpp_lib.py $HERE/diag_l8.sbatch paderborn:$B/l8/code/ || exit 1
jid=$($SSH -n paderborn "export PATH=$SL:\$PATH; cd $B/l8/runs; sbatch --parsable --qos=$QOS --time=$TIME --export=ALL,PARTS=$PARTS $B/l8/code/diag_l8.sbatch")
[[ "$jid" =~ ^[0-9]+$ ]] || { echo "SUBMIT FAILED: '$jid'"; exit 1; }
L=/scratch/a/A.Otaifi/l8/diag/$jid; mkdir -p $L
cat > $L/ANALYSIS_BRIEF.md <<EOB
# L8 refit diagnosis (Paderborn job $jid, QoS $QOS, parts $PARTS): remote $B/l8/runs/diag_$jid
Question: after F 3923271 (NO_RISE), is the refit step-size limited or data-starved?  (a) cross-chain S-metric cosine of the per-chain natural
gradients at v_0 (shift 0.1, 1, 10; jackknife over 20 block groups; reference: the two halves of chain 1); (b) 5-step refits lr {0.01,0.03,0.1}
x shift {0.1,1}, both train/held assignments.  Files: progress.txt (PROGRESS (a)/(b) lines, last DIAG_DONE), diag.json.
Gate: grep PROGRESS progress.txt.  Branches (lead, pre-registered): GO = some (lr, shift) has held-out R > 1 by > 3 sigma at its best step in BOTH
assignments -> resubmit F with it (V, G behind); DATA-STARVED = cross-chain cosine within 2 sigma of 0 at every shift -> stop, report the data factor;
otherwise stop and report.
EOB
GF_REMRUN=$B/l8/runs/diag_$jid GF_DONE_PAT=DIAG_DONE STALL=1200 POLL=60 setsid nohup bash $CODE/m13b_8x8/watch_gfmc8.sh $jid $L > $L/watcher.log 2>&1 < /dev/null &
sleep 1; echo "SUBMITTED diag job $jid; watcher pid $! ; local $L"
