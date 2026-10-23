#!/bin/bash
# Durable watcher for a Paderborn H1 job (adapted from cx_b0_pdb_watch.sh).  usage: cx_h1_pdb_watch.sh JOBID LOCALRUN   (env H1_REMRUN = remote run dir)
# Polls the job by `ssh paderborn sacct` (through the VPN).  ssh/VPN failures are tolerated for FAILSEC (1800 s) of CONTINUOUS failure, then DISAPPEARED;
# a sacct that answers but lists no such job 5 polls in a row is also DISAPPEARED.  Every RSYNC_EVERY polls and at the end it rsyncs the small outputs of
# the remote run dir (cx_h1.json, progress files, round_1/{cx_t1s.json,fit.log,theta.npy,progress.txt}, run_metadata, slurm log; files > 3 MB excluded,
# so not la_table.npy) into LOCALRUN.
# Fires on COMPLETED (cx_h1.json "done": true and no "stop" key: no CONTROL FAILED / FIT FAILED), FAILED/CANCELLED/NODE_FAIL/OOM, TIMEOUT, DISAPPEARED,
# STALLED (RUNNING and no progress file or run_metadata touched for STALL s; the fit writes progress every 10 steps, the table every 120 s).
# Last step: job_finish.sh --wake none (VERDICT line in delta_today.md).
# Test hooks: H1_SSH (executable taking the remote command string as $1), H1_REMOTE_PREFIX ("paderborn:" default; "" for a local source), H1_FIN, H1_DELTA.
JOB=$1; LOC=$2; STALL=${STALL:-2700}; POLL=${POLL:-120}; FAILSEC=${FAILSEC:-1800}; RSYNC_EVERY=${RSYNC_EVERY:-10}
REMRUN=${H1_REMRUN:?set H1_REMRUN}
PFX=${H1_REMOTE_PREFIX-paderborn:}
FIN=${H1_FIN:-/project/theorie/a/A.Otaifi/tools/job_finish.sh}
DELTA=${H1_DELTA:-/project/theorie/a/A.Otaifi/Chatty-organize/projects/j1j2/delta_today.md}
SLURMBIN=/opt/software/slurm/25.11.8-1/bin
rem() { if [ -n "${H1_SSH:-}" ]; then "$H1_SSH" "$1"; else ssh -o BatchMode=yes -o ConnectTimeout=30 -o HostName=login4 paderborn "$1"; fi; }
pull() { rsync -a -e "ssh -o BatchMode=yes -o ConnectTimeout=30 -o HostName=login4" --max-size=3M \
           --exclude='tables/' --exclude='ANALYSIS_BRIEF.md' --exclude='spec_h1.json' "${PFX}${REMRUN}/" "$LOC/" >> "$LOC/watcher_rsync.log" 2>&1; }
fail0=0; miss=0; n=0
while true; do
  n=$((n+1))
  out=$(rem "export PATH=$SLURMBIN:\$PATH; echo STATE=\$(sacct -j $JOB -X -n -P -o State | head -1 | sed 's/ .*//'); echo NOW=\$(date +%s); echo REF=\$(stat -c %Y $REMRUN/progress.txt $REMRUN/run_metadata.txt 2>/dev/null | sort -n | tail -1); echo END" 2>/dev/null)
  if ! echo "$out" | grep -q '^END'; then      # ssh / VPN failure
    now=$(date +%s); [ $fail0 -eq 0 ] && fail0=$now
    [ $((now-fail0)) -ge $FAILSEC ] && { state=DISAPPEARED; break; }
    sleep "$POLL"; continue
  fi
  fail0=0
  st=$(echo "$out" | sed -n 's/^STATE=//p'); now=$(echo "$out" | sed -n 's/^NOW=//p'); ref=$(echo "$out" | sed -n 's/^REF=//p')
  if [ -z "$st" ]; then miss=$((miss+1)); [ $miss -ge 5 ] && { state=DISAPPEARED; break; }
  else miss=0
    case "$st" in
      COMPLETED) pull; if grep -q '"done": true' "$LOC/cx_h1.json" 2>/dev/null && ! grep -q '"stop"' "$LOC/cx_h1.json"; then state=COMPLETED; else state=FAILED; fi; break;;
      FAILED|CANCELLED*|NODE_FAIL|OUT_OF_MEMORY|BOOT_FAIL) state=FAILED; break;;
      TIMEOUT) state=TIMEOUT; break;;
      RUNNING) [ -n "$ref" ] && [ $((now-ref)) -gt $STALL ] && { state=STALLED; break; };;
    esac
  fi
  [ $((n % RSYNC_EVERY)) -eq 0 ] && pull
  sleep "$POLL"
done
for i in 1 2 3; do pull && break; sleep 20; done
echo "$state $(date -Is)" > "$LOC/watcher_verdict.txt"
exec "$FIN" --run "$LOC" --job "$JOB" --state "$state" --delta "$DELTA" --wake none
