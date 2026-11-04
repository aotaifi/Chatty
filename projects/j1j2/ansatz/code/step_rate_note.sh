#!/bin/bash
# Records the measured s/step of a Paderborn fit once it has logged N (default 50) steps, in delta_today.md and the run's job_ids.txt, then exits.  Durable (setsid), polls ssh every POLL s.
# usage: step_rate_note.sh JOBID REMOTE_PROGRESS_FILE JOBIDS_FILE LABEL   (env N, POLL, R1_SSH, R1_DELTA)
# Reports: the logged median "s/step" of the line at step N, and the wall-clock rate between step 10 and step N from the progress timestamps (both numbers, with the files).  Job ended first: one line, exit.
JOB=$1; PF=$2; JIDS=$3; LABEL=$4; N=${N:-50}; POLL=${POLL:-120}
DELTA=${R1_DELTA:-/project/theorie/a/A.Otaifi/Chatty/organize/projects/j1j2/delta_today.md}; SLURMBIN=/opt/software/slurm/25.11.8-1/bin
rem() { if [ -n "${R1_SSH:-}" ]; then "$R1_SSH" "$1"; else ssh -o BatchMode=yes -o ConnectTimeout=30 -o HostName=login4 paderborn "$1"; fi; }
miss=0
while true; do
  out=$(rem "export PATH=$SLURMBIN:\$PATH; echo STATE=\$(sacct -j $JOB -X -n -P -o State | head -1 | sed 's/ .*//'); echo TZ=\$(date +%z); echo NODE=\$(sacct -j $JOB -X -n -P -o NodeList | head -1); echo PROG_BEGIN; grep -E 'PROGRESS +step [0-9]+:' $PF 2>/dev/null; echo PROG_END; echo END" 2>/dev/null)
  if echo "$out" | grep -q '^END'; then
    st=$(echo "$out" | sed -n 's/^STATE=//p'); node=$(echo "$out" | sed -n 's/^NODE=//p')
    r=$(echo "$out" | sed -n '/^PROG_BEGIN/,/^PROG_END/p' | grep PROGRESS | python3 -c '
import re, sys, datetime
N = int(sys.argv[1]); P = {}
for l in sys.stdin:
    m = re.match(r"(\S+)\s+PROGRESS\s+step (\d+):.* ([0-9.]+)s/step", l)
    if m: P[int(m.group(2))] = (datetime.datetime.fromisoformat(m.group(1)), float(m.group(3)))
if N in P and 10 in P and N > 10:
    print("%.2f %.2f" % (P[N][1], (P[N][0] - P[10][0]).total_seconds() / (N - 10)))
' $N)
    if [ -n "$r" ]; then
      set -- $r; L="$(date '+%Y-%m-%d %H:%M') [pfaffian V2] $LABEL job $JOB at step $N: logged median $1 s/step, wall-clock steps 10..$N $2 s/step (node $node; $PF; expectation 5.8 s/step from R6)"
      echo "$L" >> "$DELTA"; echo "$LABEL $JOB measured s/step at step $N: logged $1, wall-clock 10..$N $2 (node $node) $(date -Is)" >> "$JIDS"; exit 0
    fi
    case "$st" in COMPLETED|FAILED|TIMEOUT|CANCELLED*|NODE_FAIL|OUT_OF_MEMORY|BOOT_FAIL) echo "$(date '+%Y-%m-%d %H:%M') [pfaffian V2] $LABEL job $JOB ended ($st) before logging step $N; no s/step recorded" >> "$DELTA"; exit 0;; esac
    if [ -z "$st" ]; then miss=$((miss+1)); [ $miss -ge 5 ] && { echo "$(date '+%Y-%m-%d %H:%M') [pfaffian V2] $LABEL job $JOB not found in sacct 5 polls; step-rate note gives up" >> "$DELTA"; exit 0; }; else miss=0; fi
  fi
  sleep "$POLL"
done
