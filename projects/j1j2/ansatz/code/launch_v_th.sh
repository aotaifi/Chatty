#!/bin/bash
# Gate V theorie A40 race copy (PI standing rule: Paderborn start estimate > 1 day -> race a theorie copy; first copy to log a fit step wins, the loser's fit and table jobs are
# cancelled by v_race_watch.sh, PI pre-approved).  Adapted from launch_r6_th.sh.  Modes: (none) STAGE ONLY; --test: + sbatch --test-only; --submit.
# RUN ON th-ws-7010m51 ONLY (VPN host).  Run launch_v_pdb.sh (stage) first: the specs are derived from its specs (only paths and fit_sec 90000 change; diffs in $Q/spec_diff_th_*.txt).
# Fit pf_vth (v_th.sbatch, cip,inter a40, 26:00; round-5 pp2 A40 53.3 s/step -> 22.2 h) + table pf_vtth (r5_tab_th.sbatch, code pfaffian_r5/code_th = the r5 pins, afterok, 03:00;
# est. ~80 min for pp2 on an A40, r5_tab_th header).  Watchers: lswo_a40_watch.sh (fit_th), r5_tab_watch.sh (tab_th), v_race_watch.sh.
set -euo pipefail
case "$(hostname -s)" in th-ws-7010m51) ;; *) echo "run on th-ws-7010m51 (VPN host), not $(hostname -s)"; exit 1;; esac
HERE=$(cd "$(dirname "$0")" && pwd)
# Race rule (PI 2026-11-04): expected-finish rule via race_watch_eta.sh (RACE_KIND=v); RACE_WATCH=$HERE/v_race_watch.sh restores the old first-start rule.
RACE_WATCH=${RACE_WATCH:-$HERE/race_watch_eta.sh}
Q=/scratch/a/A.Otaifi/ansatz_runs/pfaffian_v; R5=/scratch/a/A.Otaifi/ansatz_runs/pfaffian_r5; P=/scratch/hpc-prf-ppdmrg/hpcalot/chatty_compress6; L=/project/theorie/a/A.Otaifi/Chatty/runs
MODE=${1:-stage}; PART=${TH_PART:-cip,inter}; GRES=${TH_GRES:-gpu:a40:1}; CODE="pool_fit.py cx_lib.py rbmpp_lib.py"
[ -s $Q/fit/spec_pool_fit.json ] && [ -s $Q/tab/tab_t5/spec_r1.json ] || { echo "run launch_v_pdb.sh (stage) first"; exit 1; }
CD=$Q/code_th; mkdir -p $CD; for f in $CODE; do cp "$HERE/$f" "$CD/$f"; done
( cd $CD && sha256sum $CODE ) | sort > $Q/code_th_sha256.txt
diff $Q/code_th_sha256.txt <(sort $Q/code_sha256.txt) > /dev/null || { echo "code_th differs from the Paderborn pins ($Q/code_sha256.txt)"; exit 1; }
TCD=$R5/code_th; ( cd $TCD && sha256sum cx_r1.py cx_r1fit.py cx_lib.py rbmpp_lib.py ) | diff - $R5/code_th_sha256.txt > /dev/null || { echo "pfaffian_r5/code_th drifted"; exit 1; }
python3 - "$Q" "$P" "$L" "$R5" <<'PY'
import json, sys, os, hashlib
Q, P, L, R5 = sys.argv[1:5]; DD = f'{P}/data'; sha = lambda p: hashlib.sha256(open(p, 'rb').read()).hexdigest()
loc = {f'{DD}/v/pool.npz': f'{Q}/pool/pool.npz', f'{DD}/r5/theta_4.npy': '/scratch/a/A.Otaifi/ansatz_runs/pfaffian_r4/pdb_tabs/theta_4.npy',
       f'{DD}/r5/theta_lswo_r1.npy': '/scratch/a/A.Otaifi/ansatz_runs/pfaffian_lswo/run/round_1/theta.npy',
       f'{DD}/r5/target_T5_sG.npy': f'{R5}/target_T5_sG.npy', f'{DD}/lswo2/s_G.npy': '/scratch/a/A.Otaifi/ansatz_runs/pfaffian_lswo2/s_G.npy',
       f'{DD}/refresh_17409419': f'{L}/compress6/runs/refresh_17409419', f'{DD}/rpT1_pp_17163493/params_final.npz': f'{L}/rbmpp6/runs/rpT1_pp_17163493/params_final.npz',
       f'{DD}/rpTabPP_17164048/sym_tables_PP.npz': f'{L}/rbmpp6/runs/rpTabPP_17164048/sym_tables_PP.npz',
       f'{DD}/o2b_r1/round_1_spec_fit.json': f'{L}/compress6/runs/o2b_17431886/round_1/spec_fit.json', f'{P}/runs/v_tab/theta_5.npy': f'{Q}/tab_th/theta_5.npy'}
def fix(v):
    if isinstance(v, str) and v in loc: return loc[v]
    if isinstance(v, dict): return {k: fix(x) for k, x in v.items()}
    return v
sp = fix(json.load(open(f'{Q}/fit/spec_pool_fit.json'))); sp['fit_sec'] = 90000.0
assert not [k for k, v in sp.items() if isinstance(v, str) and v.startswith('/scratch/hpc-prf')], sp
assert sha(sp['pool']) == sp['pool_sha256'] and all(os.path.exists(sp[k]) for k in ('warm_theta', 'warm_theta2'))
os.makedirs(f'{Q}/fit_th/round_1', exist_ok=True); json.dump(sp, open(f'{Q}/fit_th/spec_pool_fit.json', 'w'), indent=1)
t = fix(json.load(open(f'{Q}/tab/tab_t5/spec_r1.json')))
assert not [k for k, v in t.items() if isinstance(v, str) and v.startswith('/scratch/hpc-prf')], t
for k in ('target_la', 'sign_path', 'pp_params', 'pp_table', 'o2b_fit_spec', 'refresh'): assert os.path.exists(t[k]), (k, t[k])
os.makedirs(f'{Q}/tab_th/tab_t5', exist_ok=True); json.dump(t, open(f'{Q}/tab_th/tab_t5/spec_r1.json', 'w'), indent=1)
PY
diff -u $Q/fit/spec_pool_fit.json $Q/fit_th/spec_pool_fit.json > $Q/spec_diff_th_fit.txt || true; diff -u $Q/tab/tab_t5/spec_r1.json $Q/tab_th/tab_t5/spec_r1.json > $Q/spec_diff_th_tab.txt || true
for D in fit_th tab_th; do { echo "# Gate V theorie race copy ($D).  Full brief: $Q/ANALYSIS_BRIEF.md"; sed -n '2,$p' $Q/ANALYSIS_BRIEF.md; } > $Q/$D/ANALYSIS_BRIEF.md; done
echo "staged theorie copy ($PART $GRES): $Q/{fit_th,tab_th}; code $CD (fit), $TCD (table); diffs $Q/spec_diff_th_*.txt"
fitf="--parsable --no-requeue --partition=$PART --gres=$GRES --job-name=pf_vth --time=26:00:00 --output=$Q/fit_th/slurm-%j.out"
tabf="--parsable --no-requeue --partition=$PART --gres=$GRES --job-name=pf_vtth --time=03:00:00 --output=$Q/tab_th/slurm-%j.out"
if [ "$MODE" = --test ]; then
  echo "--- test-only fit"; sbatch --test-only $fitf "$HERE/v_th.sbatch" "$Q/fit_th" "$CD" 2>&1
  echo "--- test-only tab"; sbatch --test-only $tabf "$HERE/r5_tab_th.sbatch" "$Q/tab_th" "$Q/fit_th" "$TCD" 2>&1
fi
[ "$MODE" = --submit ] || exit 0
[ -e "$Q/fit_th/job_id.txt" ] && { echo "theorie copy already submitted"; exit 1; }
JOB=$(sbatch $fitf "$HERE/v_th.sbatch" "$Q/fit_th" "$CD"); JOB=${JOB%%;*}; [[ "$JOB" =~ ^[0-9]+$ ]] || { echo "sbatch failed (fit): $JOB"; exit 1; }
echo "$JOB" > $Q/fit_th/job_id.txt; echo "V fit_th $JOB $PART $GRES $(date -Is)" >> $Q/job_ids.txt
setsid nohup "$HERE/lswo_a40_watch.sh" "$JOB" "$Q/fit_th" > "$Q/fit_th/watcher.log" 2>&1 < /dev/null &
JT=$(sbatch $tabf --dependency=afterok:$JOB "$HERE/r5_tab_th.sbatch" "$Q/tab_th" "$Q/fit_th" "$TCD"); JT=${JT%%;*}; [[ "$JT" =~ ^[0-9]+$ ]] || { echo "sbatch failed (tab): $JT; fit $JOB submitted"; exit 1; }
echo "$JT" > $Q/tab_th/job_id.txt; echo "V tab_th $JT afterok:$JOB $(date -Is)" >> $Q/job_ids.txt
R1_TSUBS=t5 setsid nohup "$HERE/r5_tab_watch.sh" "$JT" "$Q/tab_th" > "$Q/tab_th/watcher.log" 2>&1 < /dev/null &
RACE_KIND=v setsid nohup "$RACE_WATCH" V > "$Q/race.log" 2>&1 < /dev/null &
sleep 3
for X in "lswo_a40_watch.sh $JOB:fit_th/watcher_pid.txt" "r5_tab_watch.sh $JT:tab_th/watcher_pid.txt" "$(basename $RACE_WATCH) V:race_pid.txt"; do
  pat=${X%%:*}; W=$(pgrep -f "$pat" | head -1); echo "$W" > $Q/${X##*:}; echo "$pat: pid $W ppid=$(ps -o ppid= -p $W 2>/dev/null)"; done
echo "V theorie: fit_th=$JOB tab_th=$JT"; sacct -j $JOB,$JT -X -n -o JobID,JobName%12,State,Reason%25
