#!/bin/bash
# K=3 ROUND-6 TEST at 6x6 (PLAN_pfaffian.md "K=3 ROUND-6 TEST at 6x6", lead + PI GO 2026-11-09), theorie A40 (Paderborn regular QoS 'cont' start estimate > 2 days; PLAN placement rule).
# R6's fit exactly (pfaffian_r6/r6/fit_th/spec_r1.json: seed 1, 1500 steps, theta_6 = mean of 7 iterates 600..1500, target T_6 = pfaffian_r6/target_T6_sG.npy, warm = pp2_s1 theta_5)
# with ONLY the class/init keys changed: arm pp2 -> pp3 (B = P_1 + lam P_2 + mu P_3), code pins (cx_r1fit.py, rbmpp_lib.py), fit_overrides += p3_theta (pfaffian_r1/run_b trained PP), p3_lam_median_target 0.095 (K=3 AMENDMENT),
# and the time budget fit_sec (K=3 is ~1.5-2x per step, A40 R6 53.3 s/step -> 1500 steps 22 h at K=2).  No race copy.  Modes: (none) STAGE ONLY; --test: stage + `sbatch --test-only`; --submit.
# RUN ON th-ws-7010m51 ONLY (same host as every launcher of this angle).  Code: ansatz/code/k3/ (cx_r1.py and cx_lib.py byte-identical to R6's; cx_r1fit.py, rbmpp_lib.py changed).
# Jobs: fit pf_k3th (r6_th.sbatch, unchanged) -> table pf_k3tth (r6k3_tab_th.sbatch, afterok) -> readout pf_k3rd (r6k3_readout.sbatch, submitted by r6k3_chain.sh when both watchers say COMPLETED).
# Watchers (durable setsid, ppid 1): lswo_a40_watch.sh (fit), r5_tab_watch.sh (table, R1_TSUBS=t6), r6k3_chain.sh -> r3_cpu_watch.sh (readout).  Every watcher ends in job_finish.sh --wake none.
set -euo pipefail
case "$(hostname -s)" in th-ws-7010m51) ;; *) echo "run launch_r6k3_th.sh on th-ws-7010m51, not $(hostname -s)"; exit 1;; esac
HERE=$(cd "$(dirname "$0")" && pwd); K3=$HERE/k3
Q=/scratch/a/A.Otaifi/ansatz_runs/pfaffian_r6k3; Q6=/scratch/a/A.Otaifi/ansatz_runs/pfaffian_r6; R5=/scratch/a/A.Otaifi/ansatz_runs/pfaffian_r5
L=/project/theorie/a/A.Otaifi/Chatty/runs
MODE=${1:-stage}; PART=${TH_PART:-cip,inter}; GRES=${TH_GRES:-gpu:a40:1}; CODE="cx_r1.py cx_r1fit.py cx_lib.py rbmpp_lib.py"
FITSEC=${K3_FITSEC:-150000}; WALL=${K3_WALL:-47:00:00}; TABWALL=${K3_TABWALL:-04:00:00}   # fit_sec 41.7 h inside a 47 h wall (cip/inter limit 2 d); table pass ~2-3 h on an A40 (pp3)
chk() { h=$(sha256sum "$1" | cut -d' ' -f1); [ "$h" = "$2" ] || { echo "sha256 mismatch $1"; exit 1; }; echo "input ok ${h:0:12} $1"; }
chk $Q6/target_T6_sG.npy $(cut -d' ' -f1 $Q6/target_T6_sG.npy.sha256)
chk $R5/pp2_s1/tab/theta_5.npy 4558d78f7c46e9e6f27911582a7dd5d2f15026e33c053ee4bee17951a449b7ac
chk /scratch/a/A.Otaifi/ansatz_runs/pfaffian_lswo2/s_G.npy 94eb19c95063ea22d2bee540e551d99870c6bb815210656fe20cc2d01e87c46a
chk $K3/cx_r1.py 5ebae9d83497d0805137ae0487a0878b53afcb8fbf93f79d6213bd50eb3ab19d
chk $K3/cx_lib.py f320a0954d53c6155cc462d33b91602de7478a5e0cfdc2ee74d5a125cf281a45
chk $K3/cx_r1fit.py 0920539fe61f35be64e8482df21551f1ac290deb0b0f9f2377e2524001e06885
chk /scratch/a/A.Otaifi/ansatz_runs/pfaffian_r1/run_b/round_1/theta.npy f03e1325343bf425dc7466b55a0728cb86c873e1e5237347499b558a15b681fd   # P_3 (trained 6x6 PP, K=3 AMENDMENT)
chk $K3/rbmpp_lib.py 04b97143a7ed0e48f814fbe7604bd619009d77231517cf4a2b9b9c21d029523c
chk $L/compress6/runs/o2b_17431886/round_1/spec_fit.json 3a89451e312a5f55fd2413a69d4d62665743a45707ac6ec692ee0b067615ec12
chk $L/rbmpp6/runs/rpT1_pp_17163493/params_final.npz e765e8fada11f3cc147c9d864c3720207cc022bb95275bc43318e5d41886b424
chk $L/rbmpp6/runs/rpTabPP_17164048/sym_tables_PP.npz d8a4e96c9a03b7ac18c0ef5bbaa9f4bcfe7632b34e01400cc2e6fb2f7253a934
chk $L/stall6/data/psi0_6x6_table.npz e2d0c838d7983f29e61805c9feef67d3aea24584058c2b703f9fbd8c5da1660d
chk $L/stall6/data/sym_tables.npz 1f8576537870f2737ad6200ea709da953fb214d3a24b65677efe8248254944d9
chk $L/stall6/csr/norb.npy 9f37ec985d777ec56bfe82e249d0b7a2601f1cee1d518fcfb08c0c8e5ecf5730
chk $L/compress6/runs/refresh_17409419/s2.npy fd4f1ef0638dc726f9652f81bbf5751eadc796601393fb4dddf5f1f031154451
for f in up_codes.npy up_inc.npy up_indices.npy meta.json; do [ -s $L/stall6/csr/$f ] || { echo "missing csr/$f"; exit 1; }; done
# the CPU checks must have passed on THESE code files (self-test json written by the pinned cx_r1fit.py)
python3 - "$Q" <<'PY'
import json, sys
Q = sys.argv[1]; s = json.load(open(f'{Q}/cpu/selftest_pp3/self_test_pp3.json')); sp = json.load(open(f'{Q}/cpu/selftest_pp3/cx_t1s.json'))
assert sp['code_sha256']['running']['cx_r1fit.py'] == '0920539fe61f35be64e8482df21551f1ac290deb0b0f9f2377e2524001e06885', 'self test ran another cx_r1fit.py'
assert s['a_pass'] and s['b_pass'] and s['c_pass'], 'CPU checks (a)(b)(c) not all passed'
sp2 = json.load(open(f'{Q}/cpu/splice_e.json')); assert sp2['pass_mu0_rel_1e-9'] and sp2['broken_fails'], 'spliced exact-table e check not passed'
ident = json.load(open(f'{Q}/cpu/ident_k2.json')); assert ident['identical'], 'K=2 path not bit-identical'
PY
CD=$Q/code_th; mkdir -p $CD; for f in $CODE; do cp "$K3/$f" "$CD/$f"; done; ( cd $CD && sha256sum $CODE ) | sort > $Q/code_sha256.txt
python3 - "$Q" "$Q6" "$FITSEC" "$CD" <<'PY'
import json, sys, os
Q, Q6, FITSEC, CD = sys.argv[1], sys.argv[2], float(sys.argv[3]), sys.argv[4]
sha = {l.split()[1]: l.split()[0] for l in open(f'{Q}/code_sha256.txt')}
base = json.load(open(f'{Q6}/r6/fit_th/spec_r1.json')); sp = json.loads(json.dumps(base))
assert base['arm'] == 'pp2' and base['max_steps'] == 1500 and base['fit_overrides']['seed'] == 1 and base['fit_overrides']['min_steps'] == 1500 and base['fit_overrides']['save_theta_at'] == [600, 750, 900, 1050, 1200, 1350]
assert base['target_la'] == f'{Q6}/target_T6_sG.npy' and base['warm_theta'].endswith('pfaffian_r5/pp2_s1/tab/theta_5.npy')
sp['arm'] = 'pp3'; sp['code_sha256']['cx_r1fit.py'] = sha['cx_r1fit.py']; sp['code_sha256']['rbmpp_lib.py'] = sha['rbmpp_lib.py']; sp['fit_sec'] = FITSEC
sp['fit_overrides'].update(p3_theta='/scratch/a/A.Otaifi/ansatz_runs/pfaffian_r1/run_b/round_1/theta.npy', p3_lam_median_target=0.095)
# assert the diff: exactly these keys changed
chg = {k for k in set(base) | set(sp) if base.get(k) != sp.get(k)}; assert chg == {'arm', 'code_sha256', 'fit_sec', 'fit_overrides'}, chg
assert {k for k in sp['code_sha256'] if sp['code_sha256'][k] != base['code_sha256'][k]} == {'cx_r1fit.py', 'rbmpp_lib.py'}
assert {k for k in set(sp['fit_overrides']) | set(base['fit_overrides']) if sp['fit_overrides'].get(k) != base['fit_overrides'].get(k)} == {'p3_theta', 'p3_lam_median_target'}
for k in ('target_la', 'warm_theta', 'sign_path', 'pp_params', 'pp_table', 'o2b_fit_spec', 'refresh'): assert os.path.exists(sp[k]), (k, sp[k])
os.makedirs(f'{Q}/k3/fit_th', exist_ok=True); json.dump(sp, open(f'{Q}/k3/fit_th/spec_r1.json', 'w'), indent=1)
t = json.loads(json.dumps(sp)); t.update(warm_theta=f'{Q}/k3/tab_th/theta_6.npy', fit_sec=1.0, max_steps=0); t['fit_overrides'] = {'seed': 1}
os.makedirs(f'{Q}/k3/tab_th/tab_t6', exist_ok=True); json.dump(t, open(f'{Q}/k3/tab_th/tab_t6/spec_r1.json', 'w'), indent=1)
PY
diff -u $Q6/r6/fit_th/spec_r1.json $Q/k3/fit_th/spec_r1.json > $Q/spec_diff_r6.txt || true
diff -u $Q6/r6/tab_th/tab_t6/spec_r1.json $Q/k3/tab_th/tab_t6/spec_r1.json > $Q/spec_diff_r6_tab.txt || true
for D in fit_th tab_th; do { echo "# K=3 round-6 test, theorie copy ($D, run dir $Q/k3/$D); full brief $Q/ANALYSIS_BRIEF.md"; sed -n '2,$p' "$Q/ANALYSIS_BRIEF.md"; } > "$Q/k3/$D/ANALYSIS_BRIEF.md"; done
mkdir -p $Q/readout; cp "$Q/ANALYSIS_BRIEF.md" "$Q/readout/ANALYSIS_BRIEF.md"
echo "staged ($PART $GRES): $Q/k3/{fit_th,tab_th}; code $CD; spec diffs $Q/spec_diff_r6.txt $Q/spec_diff_r6_tab.txt; fit_sec $FITSEC wall $WALL"
fitf() { echo "--parsable --no-requeue --partition=$PART --gres=$GRES --job-name=pf_k3th --time=$WALL --output=$Q/k3/fit_th/slurm-%j.out"; }
tabf() { echo "--parsable --no-requeue --partition=$PART --gres=$GRES --job-name=pf_k3tth --time=$TABWALL --output=$Q/k3/tab_th/slurm-%j.out"; }
if [ "$MODE" = --test ]; then
  echo "--- test-only fit";  sbatch --test-only $(fitf) "$HERE/r6_th.sbatch" "$Q/k3/fit_th" "$CD" 2>&1
  echo "--- test-only tab";  sbatch --test-only $(tabf) "$HERE/r6k3_tab_th.sbatch" "$Q/k3/tab_th" "$Q/k3/fit_th" "$CD" t6 2>&1
fi
[ "$MODE" = --submit ] || exit 0
[ -e "$Q/k3/fit_th/job_id.txt" ] && { echo "already submitted ($Q/k3/fit_th/job_id.txt)"; exit 1; }
FT=$Q/k3/fit_th; TT=$Q/k3/tab_th
JOB=$(sbatch $(fitf) "$HERE/r6_th.sbatch" "$FT" "$CD"); [[ "$JOB" =~ ^[0-9]+$ ]] || { echo "sbatch failed (fit): $JOB"; exit 1; }
echo "$JOB" > $FT/job_id.txt; echo "k3 fit_th $JOB $PART $GRES $(date -Is)" >> $Q/job_ids.txt
setsid nohup "$HERE/lswo_a40_watch.sh" "$JOB" "$FT" > "$FT/watcher.log" 2>&1 < /dev/null &
echo $! > $FT/watcher_pid.txt
JT=$(sbatch $(tabf) --dependency=afterok:$JOB "$HERE/r6k3_tab_th.sbatch" "$TT" "$FT" "$CD" t6); [[ "$JT" =~ ^[0-9]+$ ]] || { echo "sbatch failed (tab): $JT; fit $JOB already submitted"; exit 1; }
echo "$JT" > $TT/job_id.txt; echo "k3 tab_th $JT afterok:$JOB $(date -Is)" >> $Q/job_ids.txt
R1_TSUBS="t6" setsid nohup "$HERE/r5_tab_watch.sh" "$JT" "$TT" > "$TT/watcher.log" 2>&1 < /dev/null &
echo $! > $TT/watcher_pid.txt
setsid nohup "$HERE/r6k3_chain.sh" > "$Q/readout/chain.log" 2>&1 < /dev/null &
echo $! > $Q/readout/chain_pid.txt
echo "k3: fit_th=$JOB (watcher $(cat $FT/watcher_pid.txt) ppid=$(ps -o ppid= -p $(cat $FT/watcher_pid.txt))) tab_th=$JT (watcher $(cat $TT/watcher_pid.txt) ppid=$(ps -o ppid= -p $(cat $TT/watcher_pid.txt))) chain $(cat $Q/readout/chain_pid.txt) ppid=$(ps -o ppid= -p $(cat $Q/readout/chain_pid.txt))"
sacct -u $USER -S now-10minutes -X -n -o JobID,JobName%20,State,Reason%25 | grep pf_k3 || true
