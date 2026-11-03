#!/bin/bash
# Round 5 theorie race copies of the 4 Paderborn arms (PI standing rule: Paderborn ETA > 1 day -> race a theorie copy; first copy to log a fit step wins, the loser's fit and table
# jobs are cancelled by r5_race_watch.sh ARM, PI pre-approved).  Adapted from launch_r4_a40.sh.  Modes: (none) STAGE ONLY; --test: stage + `sbatch --test-only`; --submit.
# GPU: env TH_PART / TH_GRES (default inter / gpu:h200nvl:1; fallback cip,inter / gpu:a40:1).  Walls sized for an A40: ctl fit_sec 42500 wall 13:00, table 01:00;
# pp2 fit_sec 90000 wall 26:00, table 02:00.  Specs = the Paderborn specs (pfaffian_r5/<arm>/fit/spec_r1.json) with ONLY fit_sec and paths changed; diff in $Q/spec_diff_th_<arm>.txt.
# Per arm: fit pf_r5th_<arm> (r5_th.sbatch) + table pf_r5tth_<arm> (r5_tab_th.sbatch, afterok); watchers lswo_a40_watch.sh (fit), r5_tab_watch.sh (table), r5_race_watch.sh ARM.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
Q=/scratch/a/A.Otaifi/ansatz_runs/pfaffian_r5; R4=/scratch/a/A.Otaifi/ansatz_runs/pfaffian_r4
P=/scratch/hpc-prf-ppdmrg/hpcalot/chatty_compress6
L=/project/theorie/a/A.Otaifi/Chatty/runs
MODE=${1:-stage}; ARMS="pp2_s1 pp2_s2 ctl_s1 ctl_s2"; CODE="cx_r1.py cx_r1fit.py cx_lib.py rbmpp_lib.py"
PART=${TH_PART:-inter}; GRES=${TH_GRES:-gpu:h200nvl:1}
fs() { case $1 in pp2_*) echo 90000;; *) echo 42500;; esac; }
tmf() { case $1 in pp2_*) echo 26:00:00;; *) echo 13:00:00;; esac; }
tmt() { case $1 in pp2_*) echo 02:00:00;; *) echo 01:00:00;; esac; }
chk() { h=$(sha256sum "$1" | cut -d' ' -f1); [ "$h" = "$2" ] || { echo "sha256 mismatch $1"; exit 1; }; echo "input ok ${h:0:12} $1"; }
T5SHA=$(cut -d' ' -f1 $Q/target_T5_sG.npy.sha256)
chk $Q/target_T5_sG.npy $T5SHA
chk $R4/pdb_tabs/theta_4.npy 3114250142513bce80f6d1436e0465a44d509cee13248903bf5ce64f9f4bc9b6
chk /scratch/a/A.Otaifi/ansatz_runs/pfaffian_lswo/run/round_1/theta.npy 7cf4129453ab7e1ab8f450aa02c9b5902066a6680d0b66534c668c42a00f7d57
chk /scratch/a/A.Otaifi/ansatz_runs/pfaffian_lswo2/s_G.npy 94eb19c95063ea22d2bee540e551d99870c6bb815210656fe20cc2d01e87c46a
chk $HERE/cx_r1fit.py 8f89a324cea3bb6aca9069e462a397c094e43729933c9d0bf87e2e4055bc6f7f
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
CD=$Q/code_th; mkdir -p $CD; for f in $CODE; do cp "$HERE/$f" "$CD/$f"; done; ( cd $CD && sha256sum $CODE ) > $Q/code_th_sha256.txt
diff <(sort $Q/code_th_sha256.txt) <(sort $Q/code_sha256.txt) > /dev/null || { echo "code_th differs from the Paderborn pins"; exit 1; }
python3 - "$Q" "$P" "$L" <<'PY'
import json, sys, os
Q, P, L = sys.argv[1:4]
D5 = f'{P}/data/r5'; DD = f'{P}/data'
loc = {f'{D5}/target_T5_sG.npy': f'{Q}/target_T5_sG.npy', f'{D5}/theta_4.npy': '/scratch/a/A.Otaifi/ansatz_runs/pfaffian_r4/pdb_tabs/theta_4.npy',
       f'{D5}/theta_lswo_r1.npy': '/scratch/a/A.Otaifi/ansatz_runs/pfaffian_lswo/run/round_1/theta.npy',
       f'{DD}/lswo2/s_G.npy': '/scratch/a/A.Otaifi/ansatz_runs/pfaffian_lswo2/s_G.npy', f'{DD}/refresh_17409419': f'{L}/compress6/runs/refresh_17409419',
       f'{DD}/rpT1_pp_17163493/params_final.npz': f'{L}/rbmpp6/runs/rpT1_pp_17163493/params_final.npz',
       f'{DD}/rpTabPP_17164048/sym_tables_PP.npz': f'{L}/rbmpp6/runs/rpTabPP_17164048/sym_tables_PP.npz',
       f'{DD}/o2b_r1/round_1_spec_fit.json': f'{L}/compress6/runs/o2b_17431886/round_1/spec_fit.json'}
def fix(v):
    if isinstance(v, str) and v in loc: return loc[v]
    if isinstance(v, dict): return {k: fix(x) for k, x in v.items()}
    return v
for A in ('pp2_s1', 'pp2_s2', 'ctl_s1', 'ctl_s2'):
    sp = fix(json.load(open(f'{Q}/{A}/fit/spec_r1.json'))); sp['fit_sec'] = 90000.0 if A.startswith('pp2') else 42500.0
    bad = [k for k, v in list(sp.items()) + list(sp['fit_overrides'].items()) if isinstance(v, str) and v.startswith('/scratch/hpc-prf')]; assert not bad, bad
    for k in ('target_la', 'warm_theta', 'sign_path', 'pp_params', 'pp_table', 'o2b_fit_spec', 'refresh'): assert os.path.exists(sp[k]), (k, sp[k])
    if 'warm_theta2' in sp['fit_overrides']: assert os.path.exists(sp['fit_overrides']['warm_theta2'])
    os.makedirs(f'{Q}/{A}/fit_th', exist_ok=True); json.dump(sp, open(f'{Q}/{A}/fit_th/spec_r1.json', 'w'), indent=1)
    t = json.loads(json.dumps(sp)); t.update(warm_theta=f'{Q}/{A}/tab_th/theta_5.npy', fit_sec=1.0, max_steps=0); t['fit_overrides'] = {'seed': sp['fit_overrides']['seed']}
    os.makedirs(f'{Q}/{A}/tab_th/tab_t5', exist_ok=True); json.dump(t, open(f'{Q}/{A}/tab_th/tab_t5/spec_r1.json', 'w'), indent=1)
PY
for A in $ARMS; do
  diff -u $Q/$A/fit/spec_r1.json $Q/$A/fit_th/spec_r1.json > $Q/spec_diff_th_$A.txt || true
  for D in fit_th tab_th; do { echo "# Round 5 theorie race copy ($A $D, run dir $Q/$A/$D); full brief $Q/ANALYSIS_BRIEF.md"; sed -n '2,$p' "$Q/ANALYSIS_BRIEF.md"; } > "$Q/$A/$D/ANALYSIS_BRIEF.md"; done
done
echo "staged theorie copies ($PART $GRES): $Q/<arm>/{fit_th,tab_th}; code $CD; spec diffs $Q/spec_diff_th_*.txt"
fitf() { echo "--parsable --no-requeue --partition=$PART --gres=$GRES --job-name=pf_r5th_$1 --time=$(tmf $1) --output=$Q/$1/fit_th/slurm-%j.out"; }
tabf() { echo "--parsable --no-requeue --partition=$PART --gres=$GRES --job-name=pf_r5tth_$1 --time=$(tmt $1) --output=$Q/$1/tab_th/slurm-%j.out"; }
if [ "$MODE" = --test ]; then
  for A in $ARMS; do echo "--- test-only fit $A"; sbatch --test-only $(fitf $A) "$HERE/r5_th.sbatch" "$Q/$A/fit_th" "$CD" 2>&1
    echo "--- test-only tab $A"; sbatch --test-only $(tabf $A) "$HERE/r5_tab_th.sbatch" "$Q/$A/tab_th" "$Q/$A/fit_th" "$CD" 2>&1; done
fi
[ "$MODE" = --submit ] || exit 0
for A in $ARMS; do
  [ -e "$Q/$A/fit_th/job_id.txt" ] && { echo "$A: theorie copy already submitted, skipped"; continue; }
  FT=$Q/$A/fit_th; TT=$Q/$A/tab_th
  JOB=$(sbatch $(fitf $A) "$HERE/r5_th.sbatch" "$FT" "$CD"); [[ "$JOB" =~ ^[0-9]+$ ]] || { echo "sbatch failed (fit $A): $JOB"; exit 1; }
  echo "$JOB" > $FT/job_id.txt; echo "$A fit_th $JOB $PART $GRES $(date -Is)" >> $Q/job_ids.txt
  setsid nohup "$HERE/lswo_a40_watch.sh" "$JOB" "$FT" > "$FT/watcher.log" 2>&1 < /dev/null &
  echo $! > $FT/watcher_pid.txt
  JT=$(sbatch $(tabf $A) --dependency=afterok:$JOB "$HERE/r5_tab_th.sbatch" "$TT" "$FT" "$CD"); [[ "$JT" =~ ^[0-9]+$ ]] || { echo "sbatch failed (tab $A): $JT; fit $JOB already submitted"; exit 1; }
  echo "$JT" > $TT/job_id.txt; echo "$A tab_th $JT afterok:$JOB $(date -Is)" >> $Q/job_ids.txt
  setsid nohup "$HERE/r5_tab_watch.sh" "$JT" "$TT" > "$TT/watcher.log" 2>&1 < /dev/null &
  echo $! > $TT/watcher_pid.txt
  setsid nohup "$HERE/r5_race_watch.sh" "$A" > "$Q/$A/race.log" 2>&1 < /dev/null &
  echo $! > $Q/$A/race_pid.txt
  echo "$A: fit_th=$JOB (watcher $(cat $FT/watcher_pid.txt) ppid=$(ps -o ppid= -p $(cat $FT/watcher_pid.txt))) tab_th=$JT (watcher $(cat $TT/watcher_pid.txt) ppid=$(ps -o ppid= -p $(cat $TT/watcher_pid.txt))) race $(cat $Q/$A/race_pid.txt) ppid=$(ps -o ppid= -p $(cat $Q/$A/race_pid.txt))"
done
sacct -u $USER -S now-10minutes -X -n -o JobID,JobName%20,State,Reason%25 | grep pf_r5
