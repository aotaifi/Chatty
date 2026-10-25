#!/bin/bash
# Durable watcher for a Paderborn GFMC job (adapted from compress_6x6/cx_n0_pdb_watch.sh).
# usage: GF_REMRUN=<remote run dir> gfmc_pdb_watch.sh JOBID LOCALRUN      (start with: setsid nohup ... &)
# Polls `sacct` through ssh (VPN). ssh/VPN failure tolerated FAILSEC of continuous failure, then DISAPPEARED; no such job 5 polls in a row: DISAPPEARED.
# COMPLETED needs the run's own verdict line `GFMC_DONE` in progress.txt (a COMPLETED job without it is FAILED); STALLED = RUNNING and
# progress.txt / run_metadata.txt untouched for STALL s (blocks write a PROGRESS line each; checkpoint every 10 blocks).
# Small outputs (progress.txt, result.json, blocks.npz, run_metadata, slurm log; not ckpt.npz) are rsynced every RSYNC_EVERY polls and at the end.
# Last step: job_finish.sh --wake none (VERDICT line in delta_today.md).  Test hooks: GF_SSH, GF_REMOTE_PREFIX, GF_FIN, GF_DELTA.
JOB=$1; LOC=$2; STALL=${STALL:-1800}; POLL=${POLL:-120}; FAILSEC=${FAILSEC:-1800}; RSYNC_EVERY=${RSYNC_EVERY:-5}
REMRUN=${GF_REMRUN:?set GF_REMRUN}
PFX=${GF_REMOTE_PREFIX-paderborn:}
FIN=${GF_FIN:-/project/theorie/a/A.Otaifi/tools/job_finish.sh}
DELTA=${GF_DELTA:-/project/theorie/a/A.Otaifi/Chatty/organize/projects/j1j2/delta_today.md}
SLURMBIN=/opt/software/slurm/25.11.8-1/bin
mkdir -p "$LOC"
rem() { if [ -n "${GF_SSH:-}" ]; then "$GF_SSH" "$1"; else ssh -o BatchMode=yes -o ConnectTimeout=30 -o HostName=login4 paderborn "$1"; fi; }
pull() { rsync -a -e "ssh -o BatchMode=yes -o ConnectTimeout=30 -o HostName=login4" --exclude='ckpt.npz*' --exclude='code/' --exclude='ANALYSIS_BRIEF.md' "${PFX}${REMRUN}/" "$LOC/" >> "$LOC/watcher_rsync.log" 2>&1; }
fail0=0; miss=0; n=0
while true; do
  n=$((n+1))
  out=$(rem "export PATH=$SLURMBIN:\$PATH; echo STATE=\$(sacct -j $JOB -X -n -P -o State | head -1 | sed 's/ .*//'); echo NOW=\$(date +%s); echo REF=\$(stat -c %Y $REMRUN/progress.txt $REMRUN/run_metadata.txt 2>/dev/null | sort -n | tail -1); echo DONE=\$(grep -c GFMC_DONE $REMRUN/progress.txt 2>/dev/null); echo END" 2>/dev/null)
  if ! echo "$out" | grep -q '^END'; then
    now=$(date +%s); [ $fail0 -eq 0 ] && fail0=$now
    [ $((now-fail0)) -ge $FAILSEC ] && { state=DISAPPEARED; break; }
    sleep "$POLL"; continue
  fi
  fail0=0
  st=$(echo "$out" | sed -n 's/^STATE=//p'); now=$(echo "$out" | sed -n 's/^NOW=//p'); ref=$(echo "$out" | sed -n 's/^REF=//p'); dn=$(echo "$out" | sed -n 's/^DONE=//p')
  if [ -z "$st" ]; then miss=$((miss+1)); [ $miss -ge 5 ] && { state=DISAPPEARED; break; }
  else miss=0
    case "$st" in
      COMPLETED) pull; if [ "${dn:-0}" -ge 1 ]; then state=COMPLETED; else state=FAILED; fi; break;;
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
