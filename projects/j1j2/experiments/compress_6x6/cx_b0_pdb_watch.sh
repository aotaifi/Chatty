#!/bin/bash
# Durable watcher for the Paderborn B0 seed-2 job (adapted from cx_b0_watch.sh).  usage: cx_b0_pdb_watch.sh JOBID LOCALRUN
# Polls the job by `ssh paderborn sacct` (through the VPN).  ssh/VPN failures are tolerated for FAILSEC (1800 s) of CONTINUOUS failure, then DISAPPEARED;
# a sacct that answers but lists no such job 5 polls in a row is also DISAPPEARED.  Every RSYNC_EVERY polls and at the end it rsyncs the small outputs of
# the remote run dir (train.json, cx_b0.json, progress files, run_metadata, slurm log, params_final; NOT tables/state/intermediate params) into LOCALRUN.
# Fires on COMPLETED (cx_b0.json "done": true and no CONTROL FAILED), FAILED/CANCELLED/NODE_FAIL/OOM, TIMEOUT, DISAPPEARED, STALLED (RUNNING and no progress
# file touched for STALL s; the table writes progress every ckpt_sec=1200 s).  Last step: job_finish.sh --wake none (VERDICT line in delta_today.md).
# Test hooks: B0_SSH (executable taking the remote command string as $1), B0_REMOTE_PREFIX ("paderborn:" default; "" for a local source), B0_FIN, B0_DELTA.
JOB=$1; LOC=$2; STALL=${STALL:-2700}; POLL=${POLL:-120}; FAILSEC=${FAILSEC:-1800}; RSYNC_EVERY=${RSYNC_EVERY:-10}
REMRUN=${B0_REMRUN:-/scratch/hpc-prf-ppdmrg/hpcalot/chatty_compress6/runs/b0_ppbf_simple_s2}
PFX=${B0_REMOTE_PREFIX-paderborn:}
FIN=${B0_FIN:-/project/theorie/a/A.Otaifi/tools/job_finish.sh}
DELTA=${B0_DELTA:-/project/theorie/a/A.Otaifi/Chatty/organize/projects/j1j2/delta_today.md}
SLURMBIN=/opt/software/slurm/25.11.8-1/bin
rem() { if [ -n "${B0_SSH:-}" ]; then "$B0_SSH" "$1"; else ssh -o BatchMode=yes -o ConnectTimeout=30 -o HostName=login4 paderborn "$1"; fi; }
pull() { rsync -a -e "ssh -o BatchMode=yes -o ConnectTimeout=30 -o HostName=login4" --max-size=3M \
           --exclude='table_*.npz' --exclude='params_0*.npz' --exclude='state.npz' --exclude='code/' --exclude='tables/' --exclude='ANALYSIS_BRIEF.md' \
           --exclude='spec_*.json' "${PFX}${REMRUN}/" "$LOC/" >> "$LOC/watcher_rsync.log" 2>&1; }
fail0=0; miss=0; n=0
while true; do
  n=$((n+1))
  out=$(rem "export PATH=$SLURMBIN:\$PATH; echo STATE=\$(sacct -j $JOB -X -n -P -o State | head -1 | sed 's/ .*//'); echo NOW=\$(date +%s); echo REF=\$(stat -c %Y $REMRUN/progress.txt $REMRUN/train/progress.txt $REMRUN/run_metadata.txt 2>/dev/null | sort -n | tail -1); echo END" 2>/dev/null)
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
      COMPLETED) pull; if grep -q '"done": true' "$LOC/cx_b0.json" 2>/dev/null && ! grep -q 'CONTROL FAILED' "$LOC/cx_b0.json"; then state=COMPLETED; else state=FAILED; fi; break;;
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
