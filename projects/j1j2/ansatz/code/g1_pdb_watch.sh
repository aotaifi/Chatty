#!/bin/bash
# Durable watcher for a Paderborn Test G job (copy of c1_pdb_watch.sh, json name cx_g1.json).  usage: c1_pdb_watch.sh JOBID LOCALRUN   env: C1_REMRUN (remote run dir, required), C1_SUBS (subdirs with cx_g1.json, default ".")
# Polls `ssh paderborn sacct`; ssh/VPN failure tolerated for FAILSEC of continuous failure, then DISAPPEARED; 5 polls with no such job: DISAPPEARED.
# Pulls small outputs (<3 MB) of the remote run dir every RSYNC_EVERY polls and at the end.  Fires on COMPLETED (every sub's cx_g1.json has "done": true and no "stop"),
# FAILED/CANCELLED/NODE_FAIL/OOM, TIMEOUT, DISAPPEARED, STALLED (RUNNING and newest progress.txt/run_metadata.txt under the run dir older than STALL s).
# Last step: job_finish.sh --wake none (VERDICT line in delta_today.md).  Test hooks: C1_SSH, C1_REMOTE_PREFIX, C1_FIN, C1_DELTA.
JOB=$1; LOC=$2; STALL=${STALL:-1800}; POLL=${POLL:-120}; FAILSEC=${FAILSEC:-1800}; RSYNC_EVERY=${RSYNC_EVERY:-10}
REMRUN=${C1_REMRUN:?set C1_REMRUN}; SUBS=${C1_SUBS:-.}
PFX=${C1_REMOTE_PREFIX-paderborn:}
FIN=${C1_FIN:-/project/theorie/a/A.Otaifi/tools/job_finish.sh}
DELTA=${C1_DELTA:-/project/theorie/a/A.Otaifi/Chatty/organize/projects/j1j2/delta_today.md}
SLURMBIN=/opt/software/slurm/25.11.8-1/bin
rem() { if [ -n "${C1_SSH:-}" ]; then "$C1_SSH" "$1"; else ssh -o BatchMode=yes -o ConnectTimeout=30 -o HostName=login4 paderborn "$1"; fi; }
pull() { rsync -a -e "ssh -o BatchMode=yes -o ConnectTimeout=30 -o HostName=login4" --max-size=3M --exclude='tables/' --exclude='ANALYSIS_BRIEF.md' --exclude='spec_g1.json' --exclude='la_table.npy' "${PFX}${REMRUN}/" "$LOC/" >> "$LOC/watcher_rsync.log" 2>&1; }
alldone() { for s in $SUBS; do f="$LOC/$s/cx_g1.json"; grep -q '"done": true' "$f" 2>/dev/null && ! grep -q '"stop"' "$f" || return 1; done; return 0; }
fail0=0; miss=0; n=0
while true; do
  n=$((n+1))
  out=$(rem "export PATH=$SLURMBIN:\$PATH; echo STATE=\$(sacct -j $JOB -X -n -P -o State | head -1 | sed 's/ .*//'); echo NOW=\$(date +%s); echo REF=\$(find $REMRUN \( -name progress.txt -o -name run_metadata.txt \) -printf '%T@\n' 2>/dev/null | cut -d. -f1 | sort -n | tail -1); echo END" 2>/dev/null)
  if ! echo "$out" | grep -q '^END'; then
    now=$(date +%s); [ $fail0 -eq 0 ] && fail0=$now
    [ $((now-fail0)) -ge $FAILSEC ] && { state=DISAPPEARED; break; }
    sleep "$POLL"; continue
  fi
  fail0=0
  st=$(echo "$out" | sed -n 's/^STATE=//p'); now=$(echo "$out" | sed -n 's/^NOW=//p'); ref=$(echo "$out" | sed -n 's/^REF=//p')
  if [ -z "$st" ]; then miss=$((miss+1)); [ $miss -ge 5 ] && { state=DISAPPEARED; break; }
  else miss=0
    case "$st" in
      COMPLETED) pull; if alldone; then state=COMPLETED; else state=FAILED; fi; break;;
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
