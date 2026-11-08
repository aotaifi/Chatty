#!/bin/bash
# ETA race watcher (PI 2026-11-04; replaces the "first start wins" rule of v_/r6_/m5p_race_watch.sh, which on 2026-11-03 cancelled the ~9x faster H100 copy 4107178 because the A40 copy
# 17695322 had logged a step 10 minutes earlier).  Use for FUTURE launches only: launch_{r6,m5p,v}_th.sh start it via ${RACE_WATCH:-$HERE/race_watch_eta.sh}.
# usage: RACE_KIND=v|r6|m5p race_watch_eta.sh [ARM]   (ARM required for r6 and m5p; same paths, ids files and delta tags as the old watchers; override R1_Q, R1_PDB_RUN, R1_A40_RUN)
# Rule.  BEFORE both copies have logged a fit step ("step N:" in round_1/progress.txt) it does NOTHING (a copy that started first and is alone is not a reason to cancel the other).
# When both are RUNNING with >= 1 logged step, each copy's rate is measured over its latest >= MINSPAN (20) logged steps (wall-clock between the two progress timestamps, so compile and
# start-up are excluded); until both have a >= 20-step window it waits.  Then expected finish = t_last + (MAXSTEPS - step) x s/step + table time (TAB_S_PDB 600 s: r5/r6 H100 tables 4-8 min;
# TAB_S_TH 4800 s: A40 table ~80 min), which equals now + remaining steps x s/step.  The copy with the earlier expected finish is kept; the other (fit and table job, PI pre-approval for
# race losers) is cancelled and the state verified.  Difference <= TIE_S (300 s): keep TIE_PREFER (default th: spares the H100 allocation).  Every poll logs both rates and expected finishes.
# One copy ended (COMPLETED/FAILED/...) or gone: line to delta_today.md, nothing cancelled, exit.  Paderborn ids not yet written: waits (theorie copy ended -> exit; Q/pdb_skip -> exit).
# ssh failure -> retry next poll; 6 consecutive -> one delta line, keep going.  MAXSTEPS: env, else max_steps of $A40_RUN/spec_*.json, else 1500.
# Test hooks: R1_SSH, R1_SACCT, R1_SCANCEL, R1_DELTA, R1_PDB_ID, R1_PDBT_ID, R1_A40_ID, R1_A40T_ID, POLL, VERIFY_WAIT, ONESHOT (decide once from current files then exit).
ARM=${1:-}; KIND=${RACE_KIND:?set RACE_KIND=v|r6|m5p}
PB=/scratch/a/A.Otaifi/ansatz_runs; PR=/scratch/hpc-prf-ppdmrg/hpcalot/chatty_compress6/runs
case $KIND in
  v)   LL=${R1_Q:-$PB/pfaffian_v};                 PDB_RUN=${R1_PDB_RUN:-$PR/v_fit};              TAG="V";;
  r6)  : ${ARM:?arm}; LL=${R1_Q:-$PB/pfaffian_r6}/$ARM;   PDB_RUN=${R1_PDB_RUN:-$PR/r6_${ARM}_fit};  TAG="R6 $ARM";;
  m5p) : ${ARM:?arm}; LL=${R1_Q:-$PB/pfaffian_m5p}/$ARM;  PDB_RUN=${R1_PDB_RUN:-$PR/m5p_${ARM}_fit}; TAG="M5P $ARM";;
  b2r1) LL=${R1_Q:-$PB/pfaffian_8x8_r1/r1};          PDB_RUN=${R1_PDB_RUN:-$PR/b2r1_r1_fit};         TAG="B2 R1"; NOTAB=1; TAB_S_PDB=${TAB_S_PDB:-0}; TAB_S_TH=${TAB_S_TH:-0};;   # 2026-11-08: fit only, no table jobs
  *)   echo "RACE_KIND must be v, r6, m5p or b2r1"; exit 1;;
esac
A40=${R1_A40_ID:-$(cat $LL/fit_th/job_id.txt 2>/dev/null)}; A40T=${R1_A40T_ID:-$(cat $LL/tab_th/job_id.txt 2>/dev/null)}; [ -n "${NOTAB:-}" ] && A40T=0
[[ "$A40" =~ ^[0-9]+$ && "$A40T" =~ ^[0-9]+$ ]] || { echo "theorie job ids missing: a40=$A40"; exit 1; }
A40_RUN=${R1_A40_RUN:-$LL/fit_th}
DELTA=${R1_DELTA:-/project/theorie/a/A.Otaifi/Chatty/organize/projects/j1j2/delta_today.md}
POLL=${POLL:-300}; SACCT=${R1_SACCT:-sacct}; SCANCEL=${R1_SCANCEL:-scancel}
MINSPAN=${MINSPAN:-20}; TAB_S_PDB=${TAB_S_PDB:-600}; TAB_S_TH=${TAB_S_TH:-4800}; TIE_S=${TIE_S:-300}; TIE_PREFER=${TIE_PREFER:-th}
MAXSTEPS=${MAXSTEPS:-$(python3 - "$A40_RUN" <<'PY' 2>/dev/null
import glob, json, sys
for f in sorted(glob.glob(sys.argv[1] + '/spec_*.json')):
    try:
        print(int(json.load(open(f))['max_steps'])); break
    except Exception: pass
PY
)}; MAXSTEPS=${MAXSTEPS:-1500}
SLURMBIN=/opt/software/slurm/25.11.8-1/bin
rem() { if [ -n "${R1_SSH:-}" ]; then "$R1_SSH" "$1"; else ssh -o BatchMode=yes -o ConnectTimeout=30 -o HostName=login4 paderborn "$1"; fi; }
ts() { date '+%Y-%m-%d %H:%M'; }
say() { echo "$(ts) $*"; }
note() { echo "$(ts) [pfaffian $TAG race-eta] $*" >> "$DELTA"; say "DELTA: $*"; }
ended() { case "$1" in COMPLETED|FAILED|TIMEOUT|CANCELLED*|NODE_FAIL|OUT_OF_MEMORY|BOOT_FAIL|DEADLINE|PREEMPTED|GONE) return 0;; *) return 1;; esac; }
tzsec() { local z=$1 s=${1:0:1}; local h=${z:1:2} m=${z:3:2}; echo $(( ${s}1 * (10#$h * 3600 + 10#$m * 60) )); }
# decide FILE_PDB TZ_PDB FILE_TH TZ_TH -> key=value lines (DECISION=wait|keep_pdb|keep_th, INFO=...)
decide() { python3 - "$@" "$MAXSTEPS" "$MINSPAN" "$TAB_S_PDB" "$TAB_S_TH" "$TIE_S" "$TIE_PREFER" <<'PY'
import re, sys, datetime
fp, zp, ft, zt, mx, span, tp, tt, tie, pref = sys.argv[1:11]
mx, span, tp, tt, tie = int(mx), int(span), float(tp), float(tt), float(tie)
def parse(f, tz):
    pts = []
    tzi = datetime.timezone(datetime.timedelta(seconds=int(tz)))
    for l in open(f).read().splitlines():
        m = re.match(r'(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d)\s+PROGRESS\s+step (\d+):', l)
        if m:
            t = datetime.datetime.strptime(m.group(1), '%Y-%m-%dT%H:%M:%S').replace(tzinfo=tzi).timestamp()
            s = int(m.group(2))
            if pts and s <= pts[-1][0]: pts = [p for p in pts if p[0] < s]      # restart / duplicate: keep the newest
            pts.append((s, t))
    return pts
def meas(pts):
    if not pts: return None
    s1, t1 = pts[-1]; c = [p for p in pts if p[0] <= s1 - span]
    if not c: return dict(step=s1, t=t1, rate=None, s0=None)
    s0, t0 = c[-1]; return dict(step=s1, t=t1, rate=(t1 - t0) / (s1 - s0), s0=s0)
P, T = meas(parse(fp, zp)), meas(parse(ft, zt))
def fmt(n, d, tab):
    if d is None: return f'{n}: no step logged'
    if d['rate'] is None: return f"{n}: step {d['step']}, window < {span} steps"
    d['exp'] = d['t'] + (mx - d['step']) * d['rate'] + tab
    return f"{n}: step {d['step']} rate {d['rate']:.2f} s/step (steps {d['s0']}..{d['step']}) expected finish {datetime.datetime.fromtimestamp(d['exp']).strftime('%m-%d %H:%M')} (remaining {mx - d['step']} steps x rate + table {tab:.0f} s; epoch {d['exp']:.0f})"
info = '; '.join([fmt('pdb', P, tp), fmt('th', T, tt)])
if P is None or T is None: print('DECISION=nothing'); print('INFO=' + info + ' | not both logged: nothing done'); sys.exit()
if P['rate'] is None or T['rate'] is None: print('DECISION=wait'); print('INFO=' + info + f' | waiting for a >= {span}-step window on both'); sys.exit()
d = P['exp'] - T['exp']
if abs(d) <= tie: k = 'keep_' + pref; why = f'tie (|diff| {abs(d):.0f} s <= {tie:.0f} s) -> keep {pref}'
elif d < 0: k = 'keep_pdb'; why = f'pdb earlier by {-d:.0f} s ({-d/3600:.2f} h)'
else: k = 'keep_th'; why = f'th earlier by {d:.0f} s ({d/3600:.2f} h)'
print('DECISION=' + k); print('INFO=' + info + ' | ' + why)
PY
}
# --- wait for the Paderborn ids
while true; do
  PDB=${R1_PDB_ID:-$(cat $LL/fit/job_id.txt 2>/dev/null)}; PDBT=${R1_PDBT_ID:-$(cat $LL/tab/job_id.txt 2>/dev/null)}; [ -n "${NOTAB:-}" ] && PDBT=0
  [[ "$PDB" =~ ^[0-9]+$ && "$PDBT" =~ ^[0-9]+$ ]] && break
  [ -e "$LL/pdb_skip" ] && { note "pdb_skip present: no Paderborn copy for $TAG; race watcher exits (nothing cancelled)"; exit 0; }
  sa=$($SACCT -j $A40 -X -n -P -o State 2>/dev/null | head -1 | sed 's/ .*//')
  if ended "$sa"; then note "theorie copy $A40 ended ($sa) before any Paderborn job; race watcher exits"; exit 0; fi
  say "waiting for Paderborn ids; a40 $A40 $sa (nothing cancelled in this phase)"; sleep "$POLL"
done
TMPD=$(mktemp -d); trap 'rm -rf $TMPD' EXIT
LTZ=$(tzsec "$(date +%z)")
fail=0; failnote=0; gonep=0; gonea=0
while true; do
  sa=$($SACCT -j $A40 -X -n -P -o State 2>/dev/null | head -1 | sed 's/ .*//')
  if [ -z "$sa" ]; then gonea=$((gonea+1)); [ $gonea -ge 2 ] && sa=GONE || sa=UNKNOWN; else gonea=0; fi
  grep -E 'PROGRESS +step [0-9]+:' "$A40_RUN/round_1/progress.txt" 2>/dev/null | tail -n 80 > $TMPD/th.txt
  out=$(rem "export PATH=$SLURMBIN:\$PATH; echo STATE=\$(sacct -j $PDB -X -n -P -o State | head -1 | sed 's/ .*//'); echo TZ=\$(date +%z); echo PROG_BEGIN; grep -E 'PROGRESS +step [0-9]+:' $PDB_RUN/round_1/progress.txt 2>/dev/null | tail -n 80; echo PROG_END; echo END" 2>/dev/null)
  if echo "$out" | grep -q '^END'; then
    fail=0; sp=$(echo "$out" | sed -n 's/^STATE=//p'); PTZ=$(tzsec "$(echo "$out" | sed -n 's/^TZ=//p')")
    echo "$out" | sed -n '/^PROG_BEGIN/,/^PROG_END/p' | grep -E 'PROGRESS' > $TMPD/pdb.txt
    if [ -z "$sp" ]; then gonep=$((gonep+1)); [ $gonep -ge 2 ] && sp=GONE || sp=UNKNOWN; else gonep=0; fi
  else
    fail=$((fail+1)); sp=SSHFAIL; : > $TMPD/pdb.txt; PTZ=$LTZ; say "ssh failure $fail"
    if [ $fail -ge 6 ] && [ $failnote -eq 0 ]; then note "Paderborn ssh failed $fail consecutive polls (job $PDB state unknown; theorie $A40 state $sa); watcher keeps running, nothing cancelled"; failnote=1; fi
  fi
  if ended "$sa" || ended "$sp"; then
    note "job(s) ended before the race was decided: theorie $A40 $sa, Paderborn $PDB $sp; nothing cancelled, race watcher exits"; say "EXIT ended"; exit 0
  fi
  if [ "$sp" = RUNNING ] && [ "$sa" = RUNNING ]; then
    D=$(decide $TMPD/pdb.txt $PTZ $TMPD/th.txt $LTZ); dec=$(echo "$D" | sed -n 's/^DECISION=//p'); inf=$(echo "$D" | sed -n 's/^INFO=//p')
  else dec=nothing; inf="states pdb $PDB $sp, theorie $A40 $sa: not both running"; fi
  say "poll: $dec | $inf"
  case "$dec" in
    keep_pdb|keep_th)
      if [ $dec = keep_pdb ]; then   # cancel the theorie copy
        [ "$A40T" != 0 ] && $SCANCEL $A40T 2>&1 | sed 's/^/scancel th tab: /'; $SCANCEL $A40 2>&1 | sed 's/^/scancel th fit: /'
        for i in 1 2 3 4 5 6; do sleep "${VERIFY_WAIT:-5}"; v=$($SACCT -j $A40 -X -n -P -o State 2>/dev/null | head -1 | sed 's/ .*//'); case "$v" in CANCELLED*) break;; esac; done
        case "$v" in CANCELLED*) note "ETA rule: keep Paderborn $PDB, cancelled theorie $A40 (state $v; PI pre-approval for race losers). $inf";; *) note "ETA rule: keep Paderborn $PDB, scancel $A40 issued, state afterwards '$v' (NOT verified cancelled). $inf";; esac
      else                           # cancel the Paderborn copy
        rem "export PATH=$SLURMBIN:\$PATH; scancel $([ "$PDBT" != 0 ] && echo $PDBT) $PDB" 2>&1 | sed 's/^/scancel pdb: /'
        for i in 1 2 3 4 5 6; do sleep "${VERIFY_WAIT:-5}"; v=$(rem "export PATH=$SLURMBIN:\$PATH; sacct -j $PDB -X -n -P -o State | head -1 | sed 's/ .*//'" 2>/dev/null); case "$v" in CANCELLED*) break;; esac; done
        case "$v" in CANCELLED*) note "ETA rule: keep theorie $A40, cancelled Paderborn $PDB (state $v; PI pre-approval for race losers). $inf";; *) note "ETA rule: keep theorie $A40, scancel $PDB issued, state afterwards '$v' (NOT verified cancelled). $inf";; esac
      fi
      say "EXIT decided $dec"; exit 0;;
  esac
  [ -n "${ONESHOT:-}" ] && { say "ONESHOT: exit"; exit 0; }
  sleep "$POLL"
done
