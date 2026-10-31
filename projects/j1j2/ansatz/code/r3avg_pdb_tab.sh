#!/bin/bash
# R3-avg: ONE Paderborn H100 table-only job for theta_bar (reuses r3c_pdb_tab.sbatch with a fake step label "bar": FITRUN/round_1/theta_stepbar.npy = theta_bar).  --submit to run.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd); Q=/scratch/a/A.Otaifi/ansatz_runs/pfaffian_r3avg; QC=/scratch/a/A.Otaifi/ansatz_runs/pfaffian_r3c
P=/scratch/hpc-prf-ppdmrg/hpcalot/chatty_compress6; RF=$P/runs/r3avg_fit; RT=$P/runs/r3avg_tab
SSHO="ssh -o BatchMode=yes -o HostName=login4"; rem() { $SSHO paderborn "export PATH=/opt/software/slurm/25.11.8-1/bin:\$PATH; $1"; }
[ -e "$Q/tabs/job_id.txt" ] && { echo already submitted; exit 1; }
rem "mkdir -p $RF/round_1 $RT/tab_bar"
rsync -a -e "$SSHO" $Q/theta_bar.npy paderborn:$RF/round_1/theta_stepbar.npy
l=$(sha256sum $Q/theta_bar.npy | cut -d' ' -f1); r=$(rem "sha256sum $RF/round_1/theta_stepbar.npy" | cut -d' ' -f1); [ "$l" = "$r" ] || { echo theta sha mismatch; exit 1; }
mkdir -p $Q/tabs/tab_bar
python3 - $QC/B_tabs/tab_150/spec_r1.json $Q/tabs/tab_bar/spec_r1.json $RF/round_1/theta_stepbar.npy <<'PY'
import json,sys
s=json.load(open(sys.argv[1])); assert s['max_steps']==0 and s['fit_overrides']=={'seed':1}
s['warm_theta']=sys.argv[3]; json.dump(s,open(sys.argv[2],'w'),indent=1)
PY
rsync -a -e "$SSHO" $Q/tabs/tab_bar/spec_r1.json paderborn:$RT/tab_bar/
{ echo "# R3-avg table job (remote $RT, local $Q/tabs)"; sed -n '2,$p' $Q/ANALYSIS_BRIEF.md; } > $Q/tabs/ANALYSIS_BRIEF.md
rsync -a -e "$SSHO" $Q/tabs/ANALYSIS_BRIEF.md paderborn:$RT/
rsync -a -e "$SSHO" $HERE/r3c_pdb_tab.sbatch paderborn:$P/
for f in r3c_pdb_tab.sbatch; do rem "sha256sum $P/$f" | cut -d' ' -f1 | grep -qx "$(sha256sum $HERE/$f | cut -d' ' -f1)" || { echo sbatch sha differs; exit 1; }; done
echo staged
[ "${1:-}" = "--submit" ] || exit 0
OUT=$(rem "sbatch --parsable --no-requeue --job-name=pf_r3avgtab --time=00:30:00 --output=$RT/slurm-%j.out $P/r3c_pdb_tab.sbatch $RT $RF bar" 2>/dev/null); JT=${OUT%%;*}
[[ "$JT" =~ ^[0-9]+$ ]] || { echo "sbatch failed: $OUT"; exit 1; }
echo "$JT" > $Q/tabs/job_id.txt; echo "r3avg_tab $JT $(date -Is)" >> $Q/job_ids.txt
R1_REMRUN=$RT R1_SUBS=tab_bar setsid nohup $HERE/r3_pdb_tab_watch.sh $JT $Q/tabs > $Q/tabs/watcher.log 2>&1 < /dev/null &
echo $! > $Q/tabs/watcher_pid.txt; sleep 1
echo "job $JT watcher pid $(cat $Q/tabs/watcher_pid.txt) ppid=$(ps -o ppid= -p $(cat $Q/tabs/watcher_pid.txt))"
rem "sacct -j $JT -X -n -o JobID,State,Reason%30"
