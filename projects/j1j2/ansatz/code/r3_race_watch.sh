#!/bin/bash
# L-SWO round-3 "first start wins" race watcher, usage: r3_race_watch.sh (no args); copy of r2g_race_watch.sh (lead 2026-10-30).
# Jobs: Paderborn H100 (id in pfaffian_lswo/run/job_id.txt, remote run dir below) and theorie A40 (id in pfaffian_lswo/run_a40/job_id.txt).  Every POLL s (300) checks both.
# STARTED = sacct State RUNNING AND >= 1 fit step logged ("step N:" line in <run>/round_1/progress.txt, written by cx_r1fit.py progress()).
# One started -> scancel the other (remote scancel for Paderborn), verify its state, append one line [pfaffian R3 race] to delta_today.md, exit.  Both started in the same
# poll -> keep Paderborn, cancel the A40.  Either job ENDED (COMPLETED/FAILED/TIMEOUT/CANCELLED/..., or gone for 2 polls) before the other starts -> line, exit, cancel nothing.
# ssh failure -> retry next round; 6 consecutive failures -> one line, keep going.  Test hooks: R1_SSH, R1_SACCT, R1_SCANCEL, R1_DELTA, R1_PDB_RUN, R1_A40_RUN, R1_PDB_ID, R1_A40_ID, POLL.
ARM=r3; LL=/scratch/a/A.Otaifi/ansatz_runs/pfaffian_r3
PDB=${R1_PDB_ID:-$(cat $LL/pdb_fit/job_id.txt)}; A40=${R1_A40_ID:-$(cat $LL/fit/job_id.txt)}
PDBT=${R1_PDBT_ID:-$(cat $LL/pdb_tabs/job_id.txt)}; A40T=${R1_A40T_ID:-$(cat $LL/tabs/job_id.txt)}
[[ "$PDB" =~ ^[0-9]+$ && "$A40" =~ ^[0-9]+$ && "$PDBT" =~ ^[0-9]+$ && "$A40T" =~ ^[0-9]+$ ]] || { echo "job ids missing: pdb=$PDB a40=$A40"; exit 1; }
PDB_RUN=${R1_PDB_RUN:-/scratch/hpc-prf-ppdmrg/hpcalot/chatty_compress6/runs/r3_fit}
A40_RUN=${R1_A40_RUN:-/scratch/a/A.Otaifi/ansatz_runs/pfaffian_r3/fit}
DELTA=${R1_DELTA:-/project/theorie/a/A.Otaifi/Chatty/organize/projects/j1j2/delta_today.md}
POLL=${POLL:-300}; SACCT=${R1_SACCT:-sacct}; SCANCEL=${R1_SCANCEL:-scancel}
SLURMBIN=/opt/software/slurm/25.11.8-1/bin
rem() { if [ -n "${R1_SSH:-}" ]; then "$R1_SSH" "$1"; else ssh -o BatchMode=yes -o ConnectTimeout=30 -o HostName=login4 paderborn "$1"; fi; }
ts() { date '+%Y-%m-%d %H:%M'; }
say() { echo "$(ts) $*"; }                       # log
note() { echo "$(ts) [pfaffian R3 race] $*" >> "$DELTA"; say "DELTA: $*"; }
ended() { case "$1" in COMPLETED|FAILED|TIMEOUT|CANCELLED*|NODE_FAIL|OUT_OF_MEMORY|BOOT_FAIL|DEADLINE|PREEMPTED|GONE) return 0;; *) return 1;; esac; }
laststep() { grep -oE ' step [0-9]+:' "$1" 2>/dev/null | tail -1 | tr -dc 0-9; }
fail=0; failnote=0; gonep=0; gonea=0
while true; do
  # --- A40 (local)
  sa=$($SACCT -j $A40 -X -n -P -o State 2>/dev/null | head -1 | sed 's/ .*//'); na=$(laststep "$A40_RUN/round_1/progress.txt")
  if [ -z "$sa" ]; then gonea=$((gonea+1)); [ $gonea -ge 2 ] && sa=GONE || sa=UNKNOWN; else gonea=0; fi
  # --- Paderborn (ssh)
  out=$(rem "export PATH=$SLURMBIN:\$PATH; echo STATE=\$(sacct -j $PDB -X -n -P -o State | head -1 | sed 's/ .*//'); echo STEP=\$(grep -oE ' step [0-9]+:' $PDB_RUN/round_1/progress.txt 2>/dev/null | tail -1 | tr -dc 0-9); echo END" 2>/dev/null)
  if echo "$out" | grep -q '^END'; then
    fail=0; sp=$(echo "$out" | sed -n 's/^STATE=//p'); np=$(echo "$out" | sed -n 's/^STEP=//p')
    if [ -z "$sp" ]; then gonep=$((gonep+1)); [ $gonep -ge 2 ] && sp=GONE || sp=UNKNOWN; else gonep=0; fi
  else
    fail=$((fail+1)); sp=SSHFAIL; np=
    say "ssh failure $fail"
    if [ $fail -ge 6 ] && [ $failnote -eq 0 ]; then note "Paderborn ssh failed $fail consecutive polls (job $PDB state unknown; A40 $A40 state $sa); watcher keeps running, nothing cancelled"; failnote=1; fi
  fi
  say "poll: pdb $PDB $sp step=${np:-none} | a40 $A40 $sa step=${na:-none}"
  # --- decisions
  if ended "$sa" || ended "$sp"; then
    note "job(s) ended before the race was decided: A40 $A40 $sa (step ${na:-none}), Paderborn $PDB $sp (step ${np:-none}); nothing cancelled, race watcher exits"
    say "EXIT ended"; exit 0
  fi
  ps_=0; as_=0
  [ "$sp" = RUNNING ] && [ -n "$np" ] && [ "$np" -ge 1 ] && ps_=1
  [ "$sa" = RUNNING ] && [ -n "$na" ] && [ "$na" -ge 1 ] && as_=1
  if [ $ps_ -eq 1 ]; then     # Paderborn started (also when both started: keep the H100)
    $SCANCEL $A40T 2>&1 | sed 's/^/scancel a40 tabs: /'; $SCANCEL $A40 2>&1 | sed 's/^/scancel a40: /'
    for i in 1 2 3 4 5 6; do sleep "${VERIFY_WAIT:-5}"; v=$($SACCT -j $A40 -X -n -P -o State 2>/dev/null | head -1 | sed 's/ .*//'); case "$v" in CANCELLED*) break;; esac; done
    case "$v" in CANCELLED*) note "$PDB started (step $np) -> cancelled $A40 (PI pre-approval via lead)"; [ $as_ -eq 1 ] && say "both started; kept Paderborn";; *) note "$PDB started (step $np) -> scancel $A40 issued, state afterwards '$v' (NOT verified cancelled) (PI pre-approval via lead)";; esac
    say "EXIT paderborn won"; exit 0
  elif [ $as_ -eq 1 ] && [ "$sp" != SSHFAIL ]; then   # A40 started alone; need ssh for the Paderborn scancel
    rem "export PATH=$SLURMBIN:\$PATH; scancel $PDBT $PDB" 2>&1 | sed 's/^/scancel pdb: /'
    for i in 1 2 3 4 5 6; do sleep "${VERIFY_WAIT:-5}"; v=$(rem "export PATH=$SLURMBIN:\$PATH; sacct -j $PDB -X -n -P -o State | head -1 | sed 's/ .*//'" 2>/dev/null); case "$v" in CANCELLED*) break;; esac; done
    case "$v" in CANCELLED*) note "$A40 started (step $na) -> cancelled $PDB (PI pre-approval via lead)";; *) note "$A40 started (step $na) -> scancel $PDB issued, state afterwards '$v' (NOT verified cancelled) (PI pre-approval via lead)";; esac
    say "EXIT a40 won"; exit 0
  fi
  sleep "$POLL"
done
