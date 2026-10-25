#!/bin/bash
# Submit the L8 8x8 pilot (pilot_l8.sbatch, express_hpcalot, <= 22 min) and arm its durable watcher in the same action.
#   GO=1 submit_pilot.sh      (dry run without GO=1)
set -u
GO=${GO:-0}; HERE=$(cd "$(dirname "$0")" && pwd); CODE=$HERE/..; RBMPP=$CODE/../rbmpp_6x6
B=/scratch/hpc-prf-ppdmrg/hpcalot; SSH="ssh -o BatchMode=yes -o HostName=login4"; SL=/opt/software/slurm/25.11.8-1/bin
LOC=/scratch/a/A.Otaifi/l8/pilot
if [ "$GO" != 1 ]; then echo "DRY: rsync code -> $B/l8/code; sbatch pilot_l8.sbatch; watcher -> $LOC/<jobid>"; exit 0; fi
$SSH paderborn "mkdir -p $B/l8/code $B/l8/runs" || exit 1
rsync -a -e "$SSH" $CODE/{gfmc_fn_lib,gfmc_run,vmc_moments}.py $HERE/{l8_lib,l8_refit,l8_vmc_prefix}.py $RBMPP/rbmpp_lib.py $HERE/pilot_l8.sbatch paderborn:$B/l8/code/ || exit 1
jid=$($SSH -n paderborn "export PATH=$SL:\$PATH; cd $B/l8/runs; sbatch --parsable $B/l8/code/pilot_l8.sbatch")
[[ "$jid" =~ ^[0-9]+$ ]] || { echo "SUBMIT FAILED: '$jid'"; exit 1; }
L=$LOC/$jid; mkdir -p $L
cat > $L/ANALYSIS_BRIEF.md <<EOB
# L8 8x8 pilot (Paderborn job $jid, express_hpcalot): remote run dir $B/l8/runs/pilot_$jid
Question: (1) s/block of the p=1 FN-GFMC resumed from a copy of rb8/p1_s1/ckpt.npz with --dump, and the dump size per block; (2) control (i):
the zero-step refit gives params bitwise = rb8_prod and R = 1 exactly, and the stage-3 VMC on them reproduces rb8/V (stage A Ec, B0 E_ref, the
first 2 sweeps of seed 1, consts from the stored sums) to <= 1e-8 relative, while a 1e-7 perturbation of one f entry does not; (3) weight health
of the refit weights x v/g on the dumped walkers (n_eff/n >= 0.1, max share <= 1%), and the held-out R curve of a short refit.
Files (pulled here): progress.txt (stage lines, last PILOT_DONE), run_metadata.txt, D1/progress.txt, D1/blocks.npz, Z/, ZV/prefix.json, ZB/prefix.json,
F/progress.txt (WEIGHT_HEALTH, step lines), F/refit.json, F/health.json.
Gate: grep -h "PREFIX_DONE\|WEIGHT_HEALTH\|REFIT_DONE\|PILOT_DONE" -r . ; ZV verdict PASS and ZB verdict FAIL; WEIGHT_HEALTH OK; R_held curve in F/refit.json.
Next: all as required -> size the production stages from D1 block seconds and F step seconds, fill launch_l8.sh SIZING, report. Weight health
STOP -> stop and report the numbers (design failure). ZV FAIL -> report which quantity differs (GPU nondeterminism or a code change).
EOB
GF_REMRUN=$B/l8/runs/pilot_$jid GF_DONE_PAT=PILOT_DONE STALL=900 POLL=60 setsid nohup bash $CODE/m13b_8x8/watch_gfmc8.sh $jid $L > $L/watcher.log 2>&1 < /dev/null &
sleep 1; echo "SUBMITTED pilot job $jid; watcher pid $! ppid $(ps -o ppid= -p $! | tr -d ' '); local $L"
