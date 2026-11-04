#!/bin/bash
# Round 6 theorie race copies of the 2 Paderborn arms (r6, m5) (PI standing rule: race a theorie copy; first copy to log a fit step wins, the loser's fit and table jobs are
# cancelled by r6_race_watch.sh ARM, PI pre-approved).  Adapted from launch_r5_th.sh.  Modes: (none) STAGE ONLY; --test: stage + `sbatch --test-only`; --submit.
# RUN ON th-ws-7010m51 ONLY (the race watcher talks to Paderborn through the VPN there).  Run launch_r6_pdb.sh (stage) first: the specs are derived from its specs.
# GPU: env TH_PART / TH_GRES (default cip,inter / gpu:a40:1, as round 5 ran).  Walls sized for an A40: both arms fit_sec 90000 wall 26:00, table 02:00
# (r6: round-5 pp2 A40 53.3 s/step -> 1500 steps 22.2 h; m5: round-5 single PP A40 26.8 s/step x 1.5 (brief) -> 16.8 h, see ANALYSIS_BRIEF.md for the CPU-measured ratio).
# Specs = the Paderborn specs (pfaffian_r6/<arm>/fit/spec_r1.json) with ONLY fit_sec and paths changed; diff in $Q/spec_diff_th_<arm>.txt.
# Per arm: fit pf_r6th_<arm> (r6_th.sbatch) + table pf_r6tth_<arm> (r6_tab_th.sbatch, afterok); watchers lswo_a40_watch.sh (fit), r5_tab_watch.sh (table, R1_TSUBS), r6_race_watch.sh ARM.
set -euo pipefail
case "$(hostname -s)" in th-ws-7010m51) ;; *) echo "run launch_r6_th.sh on th-ws-7010m51 (VPN host), not $(hostname -s)"; exit 1;; esac
HERE=$(cd "$(dirname "$0")" && pwd)
Q=/scratch/a/A.Otaifi/ansatz_runs/pfaffian_r6; R5=/scratch/a/A.Otaifi/ansatz_runs/pfaffian_r5; R4=/scratch/a/A.Otaifi/ansatz_runs/pfaffian_r4
P=/scratch/hpc-prf-ppdmrg/hpcalot/chatty_compress6
L=/project/theorie/a/A.Otaifi/Chatty/runs
MODE=${1:-stage}; ARMS=${R6_ARMS:-"r6 m5"}; CODE="cx_r1.py cx_r1fit.py cx_lib.py rbmpp_lib.py"
PART=${TH_PART:-cip,inter}; GRES=${TH_GRES:-gpu:a40:1}
fs() { echo 90000; }
tmf() { echo 26:00:00; }
tmt() { case $1 in m5) echo 03:00:00;; *) echo 02:00:00;; esac; }   # m5: two A40 table passes (tab_t5 + tab_t0, ~40 min each)
tag() { case $1 in r6) echo t6;; m5) echo t5;; esac; }
chk() { h=$(sha256sum "$1" | cut -d' ' -f1); [ "$h" = "$2" ] || { echo "sha256 mismatch $1"; exit 1; }; echo "input ok ${h:0:12} $1"; }
chk $Q/target_T6_sG.npy $(cut -d' ' -f1 $Q/target_T6_sG.npy.sha256)
chk $R5/target_T5_sG.npy $(cut -d' ' -f1 $R5/target_T5_sG.npy.sha256)
chk $R4/pdb_tabs/theta_4.npy 3114250142513bce80f6d1436e0465a44d509cee13248903bf5ce64f9f4bc9b6
chk $R5/pp2_s1/tab/theta_5.npy 4558d78f7c46e9e6f27911582a7dd5d2f15026e33c053ee4bee17951a449b7ac
chk /scratch/a/A.Otaifi/ansatz_runs/pfaffian_lswo2/s_G.npy 94eb19c95063ea22d2bee540e551d99870c6bb815210656fe20cc2d01e87c46a
chk $HERE/cx_r1fit.py f1374bed9c7529a53ea8d95177eb709f6a79c2fbd78906351a66c7f0319babc5
chk $HERE/rbmpp_lib.py 97eade567b035835ba6d74d4b5f21a11612cdf264769152ee3d1533f812d82fb
chk $HERE/cx_r1.py 5ebae9d83497d0805137ae0487a0878b53afcb8fbf93f79d6213bd50eb3ab19d
chk $HERE/cx_lib.py f320a0954d53c6155cc462d33b91602de7478a5e0cfdc2ee74d5a125cf281a45
chk $L/compress6/runs/o2b_17431886/round_1/spec_fit.json 3a89451e312a5f55fd2413a69d4d62665743a45707ac6ec692ee0b067615ec12
chk $L/rbmpp6/runs/rpT1_pp_17163493/params_final.npz e765e8fada11f3cc147c9d864c3720207cc022bb95275bc43318e5d41886b424
chk $L/rbmpp6/runs/rpTabPP_17164048/sym_tables_PP.npz d8a4e96c9a03b7ac18c0ef5bbaa9f4bcfe7632b34e01400cc2e6fb2f7253a934
chk $L/stall6/data/psi0_6x6_table.npz e2d0c838d7983f29e61805c9feef67d3aea24584058c2b703f9fbd8c5da1660d
chk $L/stall6/data/sym_tables.npz 1f8576537870f2737ad6200ea709da953fb214d3a24b65677efe8248254944d9
chk $L/stall6/csr/norb.npy 9f37ec985d777ec56bfe82e249d0b7a2601f1cee1d518fcfb08c0c8e5ecf5730
chk $L/compress6/runs/refresh_17409419/s2.npy fd4f1ef0638dc726f9652f81bbf5751eadc796601393fb4dddf5f1f031154451
for f in up_codes.npy up_inc.npy up_indices.npy meta.json; do [ -s $L/stall6/csr/$f ] || { echo "missing csr/$f"; exit 1; }; done
for A in $ARMS; do [ -s $Q/$A/fit/spec_r1.json ] || { echo "$Q/$A/fit/spec_r1.json missing: run launch_r6_pdb.sh (stage) first"; exit 1; }; done
CD=$Q/code_th; mkdir -p $CD; for f in $CODE; do cp "$HERE/$f" "$CD/$f"; done; ( cd $CD && sha256sum $CODE ) > $Q/code_th_sha256.txt
diff <(sort $Q/code_th_sha256.txt) <(sort $Q/code_sha256.txt) > /dev/null || { echo "code_th differs from the Paderborn pins"; exit 1; }
python3 - "$Q" "$P" "$L" <<'PY'
import json, sys, os
Q, P, L = sys.argv[1:4]
DD = f'{P}/data'; R5 = '/scratch/a/A.Otaifi/ansatz_runs/pfaffian_r5'
loc = {f'{DD}/r6/target_T6_sG.npy': f'{Q}/target_T6_sG.npy', f'{DD}/r6/theta_5_pp2s1.npy': f'{R5}/pp2_s1/tab/theta_5.npy',
       f'{DD}/r5/target_T5_sG.npy': f'{R5}/target_T5_sG.npy', f'{DD}/r5/theta_4.npy': '/scratch/a/A.Otaifi/ansatz_runs/pfaffian_r4/pdb_tabs/theta_4.npy',
       f'{DD}/lswo2/s_G.npy': '/scratch/a/A.Otaifi/ansatz_runs/pfaffian_lswo2/s_G.npy', f'{DD}/refresh_17409419': f'{L}/compress6/runs/refresh_17409419',
       f'{DD}/rpT1_pp_17163493/params_final.npz': f'{L}/rbmpp6/runs/rpT1_pp_17163493/params_final.npz',
       f'{DD}/rpTabPP_17164048/sym_tables_PP.npz': f'{L}/rbmpp6/runs/rpTabPP_17164048/sym_tables_PP.npz',
       f'{DD}/o2b_r1/round_1_spec_fit.json': f'{L}/compress6/runs/o2b_17431886/round_1/spec_fit.json'}
def fix(v):
    if isinstance(v, str) and v in loc: return loc[v]
    if isinstance(v, dict): return {k: fix(x) for k, x in v.items()}
    return v
for A, TG in (('r6', 't6'), ('m5', 't5')):
    sp = fix(json.load(open(f'{Q}/{A}/fit/spec_r1.json'))); sp['fit_sec'] = 90000.0
    bad = [k for k, v in list(sp.items()) + list(sp['fit_overrides'].items()) if isinstance(v, str) and v.startswith('/scratch/hpc-prf')]; assert not bad, bad
    for k in ('target_la', 'warm_theta', 'sign_path', 'pp_params', 'pp_table', 'o2b_fit_spec', 'refresh'): assert os.path.exists(sp[k]), (k, sp[k])
    os.makedirs(f'{Q}/{A}/fit_th', exist_ok=True); json.dump(sp, open(f'{Q}/{A}/fit_th/spec_r1.json', 'w'), indent=1)
    t = json.loads(json.dumps(sp)); t.update(warm_theta=f'{Q}/{A}/tab_th/theta_{TG[1]}.npy', fit_sec=1.0, max_steps=0); t['fit_overrides'] = {'seed': sp['fit_overrides']['seed']}
    os.makedirs(f'{Q}/{A}/tab_th/tab_{TG}', exist_ok=True); json.dump(t, open(f'{Q}/{A}/tab_th/tab_{TG}/spec_r1.json', 'w'), indent=1)
    if A == 'm5':                                      # initial theta: the fit spec with max_steps 0 -> e_start
        t0 = json.loads(json.dumps(sp)); t0.update(fit_sec=1.0, max_steps=0); os.makedirs(f'{Q}/{A}/tab_th/tab_t0', exist_ok=True)
        json.dump(t0, open(f'{Q}/{A}/tab_th/tab_t0/spec_r1.json', 'w'), indent=1)
PY
for A in $ARMS; do
  diff -u $Q/$A/fit/spec_r1.json $Q/$A/fit_th/spec_r1.json > $Q/spec_diff_th_$A.txt || true
  for D in fit_th tab_th; do { echo "# Round 6 theorie race copy ($A $D, run dir $Q/$A/$D); full brief $Q/ANALYSIS_BRIEF.md"; sed -n '2,$p' "$Q/ANALYSIS_BRIEF.md"; } > "$Q/$A/$D/ANALYSIS_BRIEF.md"; done
done
echo "staged theorie copies ($PART $GRES): $Q/<arm>/{fit_th,tab_th}; code $CD; spec diffs $Q/spec_diff_th_*.txt"
fitf() { echo "--parsable --no-requeue --partition=$PART --gres=$GRES --job-name=pf_r6th_$1 --time=$(tmf $1) --output=$Q/$1/fit_th/slurm-%j.out"; }
tabf() { echo "--parsable --no-requeue --partition=$PART --gres=$GRES --job-name=pf_r6tth_$1 --time=$(tmt $1) --output=$Q/$1/tab_th/slurm-%j.out"; }
if [ "$MODE" = --test ]; then
  for A in $ARMS; do echo "--- test-only fit $A"; sbatch --test-only $(fitf $A) "$HERE/r6_th.sbatch" "$Q/$A/fit_th" "$CD" 2>&1
    echo "--- test-only tab $A"; sbatch --test-only $(tabf $A) "$HERE/r6_tab_th.sbatch" "$Q/$A/tab_th" "$Q/$A/fit_th" "$CD" "$(tag $A)" 2>&1; done
fi
[ "$MODE" = --submit ] || exit 0
for A in $ARMS; do
  [ -e "$Q/$A/fit_th/job_id.txt" ] && { echo "$A: theorie copy already submitted, skipped"; continue; }
  FT=$Q/$A/fit_th; TT=$Q/$A/tab_th; TG=$(tag $A)
  JOB=$(sbatch $(fitf $A) "$HERE/r6_th.sbatch" "$FT" "$CD"); [[ "$JOB" =~ ^[0-9]+$ ]] || { echo "sbatch failed (fit $A): $JOB"; exit 1; }
  echo "$JOB" > $FT/job_id.txt; echo "$A fit_th $JOB $PART $GRES $(date -Is)" >> $Q/job_ids.txt
  setsid nohup "$HERE/lswo_a40_watch.sh" "$JOB" "$FT" > "$FT/watcher.log" 2>&1 < /dev/null &
  echo $! > $FT/watcher_pid.txt
  JT=$(sbatch $(tabf $A) --dependency=afterok:$JOB "$HERE/r6_tab_th.sbatch" "$TT" "$FT" "$CD" "$TG"); [[ "$JT" =~ ^[0-9]+$ ]] || { echo "sbatch failed (tab $A): $JT; fit $JOB already submitted"; exit 1; }
  echo "$JT" > $TT/job_id.txt; echo "$A tab_th $JT afterok:$JOB $(date -Is)" >> $Q/job_ids.txt
  R1_TSUBS="$TG$([ $A = m5 ] && echo " t0")" setsid nohup "$HERE/r5_tab_watch.sh" "$JT" "$TT" > "$TT/watcher.log" 2>&1 < /dev/null &
  echo $! > $TT/watcher_pid.txt
  setsid nohup "$HERE/r6_race_watch.sh" "$A" > "$Q/$A/race.log" 2>&1 < /dev/null &
  echo $! > $Q/$A/race_pid.txt
  echo "$A: fit_th=$JOB (watcher $(cat $FT/watcher_pid.txt) ppid=$(ps -o ppid= -p $(cat $FT/watcher_pid.txt))) tab_th=$JT (watcher $(cat $TT/watcher_pid.txt) ppid=$(ps -o ppid= -p $(cat $TT/watcher_pid.txt))) race $(cat $Q/$A/race_pid.txt) ppid=$(ps -o ppid= -p $(cat $Q/$A/race_pid.txt))"
done
sacct -u $USER -S now-10minutes -X -n -o JobID,JobName%20,State,Reason%25 | grep pf_r6
