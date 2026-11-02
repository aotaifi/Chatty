#!/bin/bash
# Round 5 (PLAN_pfaffian.md "Two-PP class (d)"), Paderborn H100 only (no A40 copies).  Adapted from launch_r4_pdb.sh.  Modes: (none) STAGE ONLY; --test: stage + `sbatch --test-only`
# of every job with the exact submit flags; --submit: sbatch for real.
# Arms (all sign s_G, target T_5, damped GN as round 4, exactly 1500 steps, theta saved at 600..1350, theta_5 = mean of the 7 iterates):
#   pp2_s1, pp2_s2: arm 'pp2' (B = P_1 + lam P_2), P_1 = theta_4 (pfaffian_r4/pdb_tabs/theta_4.npy), P_2 = pfaffian_lswo/run/round_1/theta.npy, lam = 0.1; wall 05:00, fit_sec 16000;
#   ctl_s1, ctl_s2: arm 'rbmpp' (single PP) from theta_4; wall 03:00, fit_sec 9000.     Seeds 1 and 2 (fit_overrides.seed).
# Per arm: fit job pf_r5_<arm> (r5_pdb.sbatch, --qos cont) + table job pf_r5t_<arm> (r5_pdb_tab.sbatch, 01:00, afterok): theta_5 = mean of the 7 iterates, then tab_t5 only.
# Watchers (durable setsid, --wake none): lswo_pdb_watch.sh (fit), r3_pdb_tab_watch.sh (tab), r5_chain.sh (all arms settled -> theorie CPU readout r5_readout.sbatch).
# Specs = pfaffian_r4/pdb_fit/spec_r1.json with ONLY: code_sha256 (cx_r1fit.py, rbmpp_lib.py), target_la (T_5), warm_theta (theta_4), fit_overrides.seed, and for pp2: arm, fit_sec,
# fit_overrides.{warm_theta2, lambda_init_re, lambda_init_im}.  Diffs vs round 4 in $Q/spec_diff_<arm>.txt.   Every ssh/rsync uses -o HostName=login4.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
R4=/scratch/a/A.Otaifi/ansatz_runs/pfaffian_r4; Q=/scratch/a/A.Otaifi/ansatz_runs/pfaffian_r5
TH4=$R4/pdb_tabs/theta_4.npy; TH4SHA=3114250142513bce80f6d1436e0465a44d509cee13248903bf5ce64f9f4bc9b6
THL=/scratch/a/A.Otaifi/ansatz_runs/pfaffian_lswo/run/round_1/theta.npy; THLSHA=7cf4129453ab7e1ab8f450aa02c9b5902066a6680d0b66534c668c42a00f7d57
P=/scratch/hpc-prf-ppdmrg/hpcalot/chatty_compress6; D=$P/data; D5=$D/r5; CR=$P/codeR5
SSHO="ssh -o BatchMode=yes -o HostName=login4"
rem() { $SSHO paderborn "export PATH=/opt/software/slurm/25.11.8-1/bin:\$PATH; $1"; }
MODE=${1:-stage}
ARMS="pp2_s1 pp2_s2 ctl_s1 ctl_s2"; QOS=cont; TMT=01:00:00
tmf() { case $1 in pp2_*) echo 05:00:00;; *) echo 03:00:00;; esac; }
CODE="cx_r1.py cx_r1fit.py cx_lib.py rbmpp_lib.py"
NEW_R1FIT=8f89a324cea3bb6aca9069e462a397c094e43729933c9d0bf87e2e4055bc6f7f; NEW_RBMPP=97eade567b035835ba6d74d4b5f21a11612cdf264769152ee3d1533f812d82fb
T5SHA=$(cut -d' ' -f1 $Q/target_T5_sG.npy.sha256)
[ "$(sha256sum $Q/target_T5_sG.npy | cut -d' ' -f1)" = "$T5SHA" ] || { echo "target_T5 sha mismatch"; exit 1; }
[ "$(sha256sum $TH4 | cut -d' ' -f1)" = "$TH4SHA" ] || { echo "theta_4 sha mismatch"; exit 1; }
[ "$(sha256sum $THL | cut -d' ' -f1)" = "$THLSHA" ] || { echo "lswo round_1 theta sha mismatch"; exit 1; }
python3 -c "
import json; d=json.load(open('$Q/prep_result.json')); assert d.get('done') and d['E_T5_sG']['pass_1e-6_rel'] and d['T5_sha256']=='$T5SHA', 'prep control not passed'"
python3 -c "
import json; d=json.load(open('$Q/cpu/selftest_pp2/self_test_pp2.json')); assert all(d[k] for k in ('a_pass','b_pass','c_pass')), 'pp2 CPU checks (a,b,c) not passed'
assert d['d_pass'] or '${D_ACK:-}' == '1' or '$MODE' != '--submit', 'check (d) failed: d/d lam lies in span(P_1 columns, const) (scale/phase gauge, self_test_pp2.json d.gauge_identity_max_abs_dev); the lead must acknowledge with D_ACK=1'"
if [ "$MODE" = --submit ]; then for A in $ARMS; do [ -e "$Q/$A/fit/job_id.txt" ] && { echo "$Q/$A/fit/job_id.txt exists: already submitted"; exit 1; }; done; fi
chk() { # LOCALFILE REMOTEFILE
  l=$(sha256sum "$1" | cut -d' ' -f1); r=$(rem "sha256sum $2 2>/dev/null" | cut -d' ' -f1 || true)
  if [ "$l" != "$r" ]; then rsync -a -e "$SSHO" "$1" paderborn:$2; r=$(rem "sha256sum $2" | cut -d' ' -f1); fi
  [ "$l" = "$r" ] || { echo "sha256 mismatch $2"; exit 1; }; echo "input ok ${l:0:12} $2"; }
# 0. code: local pins, then codeR5 on Paderborn (copied, sha-checked)
[ "$(sha256sum $HERE/cx_r1fit.py | cut -d' ' -f1)" = "$NEW_R1FIT" ] || { echo "local cx_r1fit.py sha != $NEW_R1FIT"; exit 1; }
[ "$(sha256sum $HERE/rbmpp_lib.py | cut -d' ' -f1)" = "$NEW_RBMPP" ] || { echo "local rbmpp_lib.py sha != $NEW_RBMPP"; exit 1; }
rem "mkdir -p $CR"; for f in $CODE; do chk $HERE/$f $CR/$f; done
for f in st6_sector.py vit_dt.py; do chk /project/theorie/a/A.Otaifi/Chatty/runs/compress6/code/$f $CR/$f; done     # helper modules (identical to codeR3c's: 711b7afb, 5476fff3)
rem "cd $CR && sha256sum $CODE" | sort > $Q/code_sha256_remote.txt
{ for f in $CODE; do sha256sum $HERE/$f | sed "s#  .*#  $f#"; done; } | sort > $Q/code_sha256.txt
diff $Q/code_sha256.txt $Q/code_sha256_remote.txt > /dev/null || { echo "codeR5 sha256 differs from local pins"; exit 1; }
# 1. data: shared inputs sha-checked (as launch_r4_pdb.sh) and the round-5 inputs
chk /project/theorie/a/A.Otaifi/Chatty/runs/compress6/runs/o2b_17431886/round_1/spec_fit.json $D/o2b_r1/round_1_spec_fit.json
chk /project/theorie/a/A.Otaifi/Chatty/runs/rbmpp6/runs/rpT1_pp_17163493/params_final.npz $D/rpT1_pp_17163493/params_final.npz
chk /project/theorie/a/A.Otaifi/Chatty/runs/rbmpp6/runs/rpTabPP_17164048/sym_tables_PP.npz $D/rpTabPP_17164048/sym_tables_PP.npz
chk /project/theorie/a/A.Otaifi/Chatty/runs/stall6/data/psi0_6x6_table.npz $D/psi0_6x6_table.npz
chk /project/theorie/a/A.Otaifi/Chatty/runs/stall6/data/sym_tables.npz $D/sym_tables.npz
chk /project/theorie/a/A.Otaifi/Chatty/runs/stall6/csr/norb.npy $D/csr/norb.npy
chk /project/theorie/a/A.Otaifi/Chatty/runs/compress6/runs/refresh_17409419/s2.npy $D/refresh_17409419/s2.npy
rem "cd $D/csr; ls -l meta.json up_indices.npy up_codes.npy up_inc.npy" > /dev/null
rem "mkdir -p $D5"
chk /scratch/a/A.Otaifi/ansatz_runs/pfaffian_lswo2/s_G.npy $D/lswo2/s_G.npy
chk $Q/target_T5_sG.npy $D5/target_T5_sG.npy
chk $TH4 $D5/theta_4.npy
chk $THL $D5/theta_lswo_r1.npy
# 2. specs
for A in $ARMS; do mkdir -p $Q/$A/fit $Q/$A/tab/tab_t5; done; mkdir -p $Q/readout
python3 - "$R4/pdb_fit/spec_r1.json" "$Q" "$P" "$D5" "$NEW_R1FIT" "$NEW_RBMPP" <<'PY'
import json, sys
src, Q, P, D5, sha_fit, sha_rbm = sys.argv[1:7]; base = json.load(open(src))
assert base['fit_overrides'] == {'seed': 2, 'min_steps': 1500, 'save_theta_at': [600, 750, 900, 1050, 1200, 1350]} and base['max_steps'] == 1500 and base['fit_sec'] == 9000.0
assert base['arm'] == 'rbmpp' and base['rbm_alpha'] == 4 and base['sign_path'].endswith('lswo2/s_G.npy') and base['skip_efn'] and base['skip_fn_readouts']
for A in ('pp2_s1', 'pp2_s2', 'ctl_s1', 'ctl_s2'):
    seed = int(A[-1]); sp = json.loads(json.dumps(base))
    sp['code_sha256']['cx_r1fit.py'] = sha_fit; sp['code_sha256']['rbmpp_lib.py'] = sha_rbm
    sp.update(target_la=f'{D5}/target_T5_sG.npy', warm_theta=f'{D5}/theta_4.npy')
    sp['fit_overrides'] = {'seed': seed, 'min_steps': 1500, 'save_theta_at': [600, 750, 900, 1050, 1200, 1350]}
    if A.startswith('pp2'):
        sp['arm'] = 'pp2'; sp['fit_sec'] = 16000.0
        sp['fit_overrides'].update(warm_theta2=f'{D5}/theta_lswo_r1.npy', lambda_init_re=0.1, lambda_init_im=0.0)
    json.dump(sp, open(f'{Q}/{A}/fit/spec_r1.json', 'w'), indent=1)
    RT = f'{P}/runs/r5_{A}_tab'
    t = json.loads(json.dumps(sp)); t.update(warm_theta=f'{RT}/theta_5.npy', fit_sec=1.0, max_steps=0); t['fit_overrides'] = {'seed': seed}
    json.dump(t, open(f'{Q}/{A}/tab/tab_t5/spec_r1.json', 'w'), indent=1)
PY
for A in $ARMS; do diff -u $R4/pdb_fit/spec_r1.json $Q/$A/fit/spec_r1.json > $Q/spec_diff_$A.txt || true; diff -u $R4/pdb_tabs/tab_t4/spec_r1.json $Q/$A/tab/tab_t5/spec_r1.json > $Q/spec_diff_${A}_tab.txt || true; done
# 3. run dirs, briefs, sbatch files
for A in $ARMS; do
  RUNF=$P/runs/r5_${A}_fit; RUNT=$P/runs/r5_${A}_tab; rem "mkdir -p $RUNF $RUNT/tab_t5"
  for K in fit tab; do
    [ $K = fit ] && { LOC=$Q/$A/fit; REM=$RUNF; } || { LOC=$Q/$A/tab; REM=$RUNT; }
    { echo "# Round 5 Paderborn ($A $K job; remote $REM, local $LOC).  Full brief and verdict rules: $Q/ANALYSIS_BRIEF.md"; sed -n '2,$p' "$Q/ANALYSIS_BRIEF.md"; } > "$LOC/ANALYSIS_BRIEF.md"
    rsync -a -e "$SSHO" "$LOC/ANALYSIS_BRIEF.md" paderborn:$REM/
  done
  rsync -a -e "$SSHO" "$Q/$A/fit/spec_r1.json" paderborn:$RUNF/
  rsync -a -e "$SSHO" "$Q/$A/tab/tab_t5/spec_r1.json" paderborn:$RUNT/tab_t5/
done
cp "$Q/ANALYSIS_BRIEF.md" "$Q/readout/ANALYSIS_BRIEF.md"
rsync -a -e "$SSHO" "$HERE/r5_pdb.sbatch" "$HERE/r5_pdb_tab.sbatch" paderborn:$P/
for f in r5_pdb.sbatch r5_pdb_tab.sbatch; do rem "sha256sum $P/$f" | cut -d' ' -f1 | grep -qx "$(sha256sum "$HERE/$f" | cut -d' ' -f1)" || { echo "$f sha256 differs after copy"; exit 1; }; done
echo "staged Paderborn: code $CR, data $D5, runs $P/runs/r5_{$(echo $ARMS | tr ' ' ',')}_{fit,tab}; local $Q/<arm>/{fit,tab}; spec diffs $Q/spec_diff_*.txt"
fitf() { echo "--parsable --no-requeue --qos=$QOS --job-name=pf_r5_$1 --time=$(tmf $1) --output=$P/runs/r5_$1_fit/slurm-%j.out"; }
tabf() { echo "--parsable --no-requeue --qos=$QOS --job-name=pf_r5t_$1 --time=$TMT --output=$P/runs/r5_$1_tab/slurm-%j.out"; }
if [ "$MODE" = --test ]; then
  for A in $ARMS; do
    echo "--- test-only fit $A";  rem "sbatch --test-only $(fitf $A) $P/r5_pdb.sbatch $P/runs/r5_${A}_fit" 2>&1
    echo "--- test-only tab $A (no dependency: a test-only cannot see the not-yet-existing fit job)"; rem "sbatch --test-only $(tabf $A) $P/r5_pdb_tab.sbatch $P/runs/r5_${A}_tab $P/runs/r5_${A}_fit t5" 2>&1
  done
fi
[ "$MODE" = --submit ] || exit 0
for A in $ARMS; do
  RUNF=$P/runs/r5_${A}_fit; RUNT=$P/runs/r5_${A}_tab
  OUTL=$(rem "sbatch $(fitf $A) $P/r5_pdb.sbatch $RUNF" 2>/dev/null); JOB=${OUTL%%;*}
  [[ "$JOB" =~ ^[0-9]+$ ]] || { echo "sbatch failed (fit $A): $OUTL"; exit 1; }
  echo "$JOB" > "$Q/$A/fit/job_id.txt"; echo "$A fit $JOB $(date -Is)" >> "$Q/job_ids.txt"
  R1_REMRUN=$RUNF setsid nohup "$HERE/lswo_pdb_watch.sh" "$JOB" "$Q/$A/fit" > "$Q/$A/fit/watcher.log" 2>&1 < /dev/null &
  echo "$!" > "$Q/$A/fit/watcher_pid.txt"
  OUTT=$(rem "sbatch $(tabf $A) --dependency=afterok:$JOB $P/r5_pdb_tab.sbatch $RUNT $RUNF t5" 2>/dev/null); JT=${OUTT%%;*}
  [[ "$JT" =~ ^[0-9]+$ ]] || { echo "sbatch failed (tab $A): $OUTT; fit $JOB already submitted"; exit 1; }
  echo "$JT" > "$Q/$A/tab/job_id.txt"; echo "$A tab $JT afterok:$JOB $(date -Is)" >> "$Q/job_ids.txt"
  R1_REMRUN=$RUNT R1_SUBS="tab_t5" setsid nohup "$HERE/r3_pdb_tab_watch.sh" "$JT" "$Q/$A/tab" > "$Q/$A/tab/watcher.log" 2>&1 < /dev/null &
  echo "$!" > "$Q/$A/tab/watcher_pid.txt"
  echo "$A: fit=$JOB (watcher $(cat $Q/$A/fit/watcher_pid.txt) ppid=$(ps -o ppid= -p $(cat $Q/$A/fit/watcher_pid.txt))) tab=$JT (watcher $(cat $Q/$A/tab/watcher_pid.txt) ppid=$(ps -o ppid= -p $(cat $Q/$A/tab/watcher_pid.txt)))"
done
setsid nohup "$HERE/r5_chain.sh" > "$Q/readout/chain.log" 2>&1 < /dev/null &
echo "$!" > "$Q/readout/chain_pid.txt"; echo "chain pid $(cat $Q/readout/chain_pid.txt) ppid=$(ps -o ppid= -p $(cat $Q/readout/chain_pid.txt))"
rem "sacct -u \$USER -S now-1hour -X -n -o JobID,JobName%20,State,Reason%30" 2>/dev/null; rem "squeue --start -u \$USER" 2>/dev/null || true
