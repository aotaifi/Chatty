#!/bin/bash
# M5' theorie race copy of one PHASE (env M5P_PHASE = rel1|rel2|rel3|round), adapted from launch_r6_th.sh.  PI standing rule: race a theorie copy; first copy to log a fit step wins, the loser's
# fit and table jobs are cancelled by m5p_race_watch.sh PHASE (PI pre-approved).  Modes: (none) STAGE ONLY; --test: stage + `sbatch --test-only`; --submit.  RUN ON th-ws-7010m51 (race watcher uses the VPN).
# Run launch_m5p_pdb.sh (stage) for the same phase first: the specs are derived from its specs.
# GPU: default H200 NVL (TH_PART=inter TH_GRES=gpu:h200nvl:1, a40slice venv; lead: tested OK).  If no H200 is free: TH_PART=cip,inter TH_GRES=gpu:a40:1 (A40 27 s/step planned; fit_sec sized for it).
# Walls: relax fit_sec 60000 wall 18:00 (1000 steps x 27 s = 7.5 h, x2.2 margin); round fit_sec 90000 wall 26:00 (1500 x 27 s = 11.3 h, x2.2); tables 03:00 (A40 pass ~40 min).
# Specs = the Paderborn specs with ONLY fit_sec and paths changed (diff in $Q/spec_diff_th_<phase>.txt).  Jobs: fit pf_m5pth_<phase> (r6_th.sbatch unchanged) + table pf_m5ptth_<phase> (m5p_tab_th.sbatch, afterok).
set -euo pipefail
case "$(hostname -s)" in th-ws-7010m51) ;; *) echo "run launch_m5p_th.sh on th-ws-7010m51 (VPN host), not $(hostname -s)"; exit 1;; esac
HERE=$(cd "$(dirname "$0")" && pwd)
# Race rule (PI 2026-11-04): expected-finish rule via race_watch_eta.sh (RACE_KIND=m5p); RACE_WATCH=$HERE/m5p_race_watch.sh restores the old first-start rule.
RACE_WATCH=${RACE_WATCH:-$HERE/race_watch_eta.sh}
PH=${M5P_PHASE:?set M5P_PHASE}; Q=/scratch/a/A.Otaifi/ansatz_runs/pfaffian_m5p; R4=/scratch/a/A.Otaifi/ansatz_runs/pfaffian_r4; V=/scratch/a/A.Otaifi/ansatz_runs/pfaffian_v
P=/scratch/hpc-prf-ppdmrg/hpcalot/chatty_compress6; L=/project/theorie/a/A.Otaifi/Chatty/runs
MODE=${1:-stage}; CODE="cx_r1.py cx_r1fit.py cx_lib.py rbmpp_lib.py"
PART=${TH_PART:-inter}; GRES=${TH_GRES:-gpu:h200nvl:1}
case $PH in rel*) FS=60000; TMF=18:00:00; TAG=rel;; round) FS=90000; TMF=26:00:00; TAG=t5;; *) echo "bad phase"; exit 1;; esac; TMT=03:00:00
chk() { h=$(sha256sum "$1" | cut -d' ' -f1); [ "$h" = "$2" ] || { echo "sha256 mismatch $1"; exit 1; }; echo "input ok ${h:0:12} $1"; }
chk $R4/pdb_tabs/theta_4.npy 3114250142513bce80f6d1436e0465a44d509cee13248903bf5ce64f9f4bc9b6
chk $V/target/target_T5_b1_sG.npy 35ef133508403ddbdfd7e459fa690d348034e0af8ababaf28b9c92f9025c8e99
chk /scratch/a/A.Otaifi/ansatz_runs/pfaffian_lswo2/s_G.npy 94eb19c95063ea22d2bee540e551d99870c6bb815210656fe20cc2d01e87c46a
chk $HERE/cx_r1fit.py f1374bed9c7529a53ea8d95177eb709f6a79c2fbd78906351a66c7f0319babc5; chk $HERE/rbmpp_lib.py 97eade567b035835ba6d74d4b5f21a11612cdf264769152ee3d1533f812d82fb
chk $HERE/cx_r1.py 5ebae9d83497d0805137ae0487a0878b53afcb8fbf93f79d6213bd50eb3ab19d; chk $HERE/cx_lib.py f320a0954d53c6155cc462d33b91602de7478a5e0cfdc2ee74d5a125cf281a45
chk $L/compress6/runs/o2b_17431886/round_1/spec_fit.json 3a89451e312a5f55fd2413a69d4d62665743a45707ac6ec692ee0b067615ec12
chk $L/rbmpp6/runs/rpT1_pp_17163493/params_final.npz e765e8fada11f3cc147c9d864c3720207cc022bb95275bc43318e5d41886b424
chk $L/rbmpp6/runs/rpTabPP_17164048/sym_tables_PP.npz d8a4e96c9a03b7ac18c0ef5bbaa9f4bcfe7632b34e01400cc2e6fb2f7253a934
chk $L/stall6/data/psi0_6x6_table.npz e2d0c838d7983f29e61805c9feef67d3aea24584058c2b703f9fbd8c5da1660d; chk $L/stall6/data/sym_tables.npz 1f8576537870f2737ad6200ea709da953fb214d3a24b65677efe8248254944d9
chk $L/stall6/csr/norb.npy 9f37ec985d777ec56bfe82e249d0b7a2601f1cee1d518fcfb08c0c8e5ecf5730; chk $L/compress6/runs/refresh_17409419/s2.npy fd4f1ef0638dc726f9652f81bbf5751eadc796601393fb4dddf5f1f031154451
for f in up_codes.npy up_inc.npy up_indices.npy meta.json; do [ -s $L/stall6/csr/$f ] || { echo "missing csr/$f"; exit 1; }; done
[ -s $Q/$PH/fit/spec_r1.json ] || { echo "$Q/$PH/fit/spec_r1.json missing: run launch_m5p_pdb.sh (stage) for $PH first"; exit 1; }
CD=$Q/code_th; mkdir -p $CD; for f in $CODE; do cp "$HERE/$f" "$CD/$f"; done; ( cd $CD && sha256sum $CODE ) > $Q/code_th_sha256.txt
diff <(sort $Q/code_th_sha256.txt) <(sort $Q/code_sha256.txt) > /dev/null || { echo "code_th differs from the Paderborn pins"; exit 1; }
python3 - "$Q" "$P" "$L" "$PH" "$TAG" "$FS" <<'PY'
import json, sys, os
Q, P, L, PH, TAG, FS = sys.argv[1:7]; DD = f'{P}/data'; DM = f'{DD}/m5p'; R4 = '/scratch/a/A.Otaifi/ansatz_runs/pfaffian_r4'; V = '/scratch/a/A.Otaifi/ansatz_runs/pfaffian_v'
loc = {f'{DM}/target_la_theta4.npy': f'{R4}/pdb_tabs/tab_t4/round_1/la_table.npy', f'{DM}/target_T5_b1_sG.npy': f'{V}/target/target_T5_b1_sG.npy', f'{DD}/r5/theta_4.npy': f'{R4}/pdb_tabs/theta_4.npy',
       f'{DM}/warm_round.npy': f'{Q}/theta_rel.npy', f'{DM}/warm_rel2.npy': f'{Q}/rel1/tab/theta_rel.npy', f'{DM}/warm_rel3.npy': f'{Q}/rel2/tab/theta_rel.npy',
       f'{DD}/lswo2/s_G.npy': '/scratch/a/A.Otaifi/ansatz_runs/pfaffian_lswo2/s_G.npy', f'{DD}/refresh_17409419': f'{L}/compress6/runs/refresh_17409419',
       f'{DD}/rpT1_pp_17163493/params_final.npz': f'{L}/rbmpp6/runs/rpT1_pp_17163493/params_final.npz', f'{DD}/rpTabPP_17164048/sym_tables_PP.npz': f'{L}/rbmpp6/runs/rpTabPP_17164048/sym_tables_PP.npz',
       f'{DD}/o2b_r1/round_1_spec_fit.json': f'{L}/compress6/runs/o2b_17431886/round_1/spec_fit.json'}
def fix(v):
    if isinstance(v, str) and v in loc: return loc[v]
    if isinstance(v, dict): return {k: fix(x) for k, x in v.items()}
    return v
sp = fix(json.load(open(f'{Q}/{PH}/fit/spec_r1.json'))); sp['fit_sec'] = float(FS)
bad = [k for k, v in list(sp.items()) + list(sp['fit_overrides'].items()) if isinstance(v, str) and v.startswith('/scratch/hpc-prf')]; assert not bad, bad
for k in ('target_la', 'warm_theta', 'sign_path', 'pp_params', 'pp_table', 'o2b_fit_spec', 'refresh'): assert os.path.exists(sp[k]), (k, sp[k])
os.makedirs(f'{Q}/{PH}/fit_th', exist_ok=True); json.dump(sp, open(f'{Q}/{PH}/fit_th/spec_r1.json', 'w'), indent=1)
t = json.loads(json.dumps(sp)); t.update(warm_theta=f'{Q}/{PH}/tab_th/theta_{"rel" if TAG == "rel" else "5"}.npy', fit_sec=1.0, max_steps=0); t['fit_overrides'] = {'seed': sp['fit_overrides']['seed']}
os.makedirs(f'{Q}/{PH}/tab_th/tab_{TAG}', exist_ok=True); json.dump(t, open(f'{Q}/{PH}/tab_th/tab_{TAG}/spec_r1.json', 'w'), indent=1)
PY
diff -u $Q/$PH/fit/spec_r1.json $Q/$PH/fit_th/spec_r1.json > $Q/spec_diff_th_$PH.txt || true
for D in fit_th tab_th; do { echo "# M5' $PH theorie race copy ($D, run dir $Q/$PH/$D); full brief $Q/ANALYSIS_BRIEF.md"; sed -n '2,$p' "$Q/ANALYSIS_BRIEF.md"; } > "$Q/$PH/$D/ANALYSIS_BRIEF.md"; done
echo "staged theorie copy ($PART $GRES) phase $PH: $Q/$PH/{fit_th,tab_th}; code $CD; spec diff $Q/spec_diff_th_$PH.txt"
FT=$Q/$PH/fit_th; TT=$Q/$PH/tab_th
fitf() { echo "--parsable --no-requeue --partition=$PART --gres=$GRES --job-name=pf_m5pth_$PH --time=$TMF --output=$FT/slurm-%j.out"; }
tabf() { echo "--parsable --no-requeue --partition=$PART --gres=$GRES --job-name=pf_m5ptth_$PH --time=$TMT --output=$TT/slurm-%j.out"; }
if [ "$MODE" = --test ]; then echo "--- test-only fit $PH"; sbatch --test-only $(fitf) "$HERE/r6_th.sbatch" "$FT" "$CD" 2>&1
  echo "--- test-only tab $PH"; sbatch --test-only $(tabf) "$HERE/m5p_tab_th.sbatch" "$TT" "$FT" "$CD" "$TAG" 2>&1; fi
[ "$MODE" = --submit ] || exit 0
[ -e "$FT/job_id.txt" ] && { echo "$PH: theorie copy already submitted"; exit 1; }
JOB=$(sbatch $(fitf) "$HERE/r6_th.sbatch" "$FT" "$CD"); [[ "$JOB" =~ ^[0-9]+$ ]] || { echo "sbatch failed (fit): $JOB"; exit 1; }
echo "$JOB" > $FT/job_id.txt; echo "$PH fit_th $JOB $PART $GRES $(date -Is)" >> $Q/job_ids.txt
setsid nohup "$HERE/lswo_a40_watch.sh" "$JOB" "$FT" > "$FT/watcher.log" 2>&1 < /dev/null & echo $! > $FT/watcher_pid.txt
JT=$(sbatch $(tabf) --dependency=afterok:$JOB "$HERE/m5p_tab_th.sbatch" "$TT" "$FT" "$CD" "$TAG"); [[ "$JT" =~ ^[0-9]+$ ]] || { echo "sbatch failed (tab): $JT; fit $JOB already submitted"; exit 1; }
echo "$JT" > $TT/job_id.txt; echo "$PH tab_th $JT afterok:$JOB $(date -Is)" >> $Q/job_ids.txt
R1_TSUBS="$TAG" setsid nohup "$HERE/r5_tab_watch.sh" "$JT" "$TT" > "$TT/watcher.log" 2>&1 < /dev/null & echo $! > $TT/watcher_pid.txt
RACE_KIND=m5p setsid nohup "$RACE_WATCH" "$PH" > "$Q/$PH/race.log" 2>&1 < /dev/null & echo $! > $Q/$PH/race_pid.txt
echo "$PH: fit_th=$JOB (watcher $(cat $FT/watcher_pid.txt) ppid=$(ps -o ppid= -p $(cat $FT/watcher_pid.txt))) tab_th=$JT (watcher $(cat $TT/watcher_pid.txt) ppid=$(ps -o ppid= -p $(cat $TT/watcher_pid.txt))) race $(cat $Q/$PH/race_pid.txt) ppid=$(ps -o ppid= -p $(cat $Q/$PH/race_pid.txt))"
sacct -u $USER -S now-10minutes -X -n -o JobID,JobName%20,State,Reason%25 | grep pf_m5p
