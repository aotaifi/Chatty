#!/bin/bash
# Round 6 (PLAN_pfaffian.md "Round 6 two-PP + parameter-matched control"), Paderborn H100.  Adapted from launch_r5_pdb.sh.  Modes: (none) STAGE ONLY; --test: stage + `sbatch --test-only`
# of every job with the exact submit flags; --submit: sbatch for real.  RUN ON th-ws-7010m51 ONLY (the PC2 VPN client runs there; the setsid watchers must live there).
# Arms (sign s_G, damped GN as round 5, seed 1, exactly 1500 steps, theta saved at 600..1350, theta_R = mean of the 7 iterates):
#   r6: arm 'pp2' warm from pp2_s1's theta_5 (pfaffian_r5/pp2_s1/tab/theta_5.npy, 5334 = P_1, P_2, lambda, taken as is), target T_6 (pfaffian_r6/target_T6_sG.npy); wall 05:00, fit_sec 16000;
#   m5: arm 'rbmpp', rbm_alpha 72, warm = theta_4's PP block + RBM block 72 x 37 ~ N(0, 0.003^2) (rbm_init_sigma 0.003, lead decision 2026-11-03 after the sigma sweep pfaffian_r6/cpu/sigma_sweep.txt; rbm_init_seed 72);
#       its table job also tabulates the INITIAL theta (tab_t0: same spec, max_steps 0) -> e_start for r_5(own), target T_5; wall 05:00, fit_sec 16000.
# Per arm: fit job pf_r6_<arm> (r6_pdb.sbatch, --qos cont) + table job pf_r6t_<arm> (r6_pdb_tab.sbatch, 01:00, afterok; r6 -> theta_6 + tab_t6, m5 -> theta_5 + tab_t5).
# Watchers (durable setsid, --wake none): lswo_pdb_watch.sh (fit), r3_pdb_tab_watch.sh (tab), r6_chain.sh (both arms settled -> theorie CPU readout r6_readout.sbatch).
# Specs: r6 = pfaffian_r5/pp2_s1/fit/spec_r1.json, m5 = pfaffian_r5/ctl_s1/fit/spec_r1.json, with ONLY: code_sha256 (cx_r1fit.py), target_la, warm_theta, fit_sec, and
#   r6: fit_overrides minus warm_theta2 / lambda_init_* (not read for a 5334 warm theta);  m5: rbm_alpha 72, fit_overrides + rbm_init_sigma, rbm_init_seed.
#   Diffs in $Q/spec_diff_<arm>.txt.   Every ssh/rsync uses -o HostName=login4.
set -euo pipefail
case "$(hostname -s)" in th-ws-7010m51) ;; *) echo "run launch_r6_pdb.sh on th-ws-7010m51 (VPN host), not $(hostname -s)"; exit 1;; esac
HERE=$(cd "$(dirname "$0")" && pwd)
R4=/scratch/a/A.Otaifi/ansatz_runs/pfaffian_r4; R5=/scratch/a/A.Otaifi/ansatz_runs/pfaffian_r5; Q=/scratch/a/A.Otaifi/ansatz_runs/pfaffian_r6
TH4=$R4/pdb_tabs/theta_4.npy; TH4SHA=3114250142513bce80f6d1436e0465a44d509cee13248903bf5ce64f9f4bc9b6
TH5=$R5/pp2_s1/tab/theta_5.npy; TH5SHA=4558d78f7c46e9e6f27911582a7dd5d2f15026e33c053ee4bee17951a449b7ac
P=/scratch/hpc-prf-ppdmrg/hpcalot/chatty_compress6; D=$P/data; D6=$D/r6; CR=$P/codeR6
SSHO="ssh -o BatchMode=yes -o HostName=login4 -o ControlMaster=auto -o ControlPath=/tmp/a.otaifi-cm-pdb-r6 -o ControlPersist=300"   # one multiplexed connection: the staging burst got login4 to close new connections (2026-11-03 09:00)
rem() { $SSHO paderborn "export PATH=/opt/software/slurm/25.11.8-1/bin:\$PATH; $1"; }
MODE=${1:-stage}
ARMS=${R6_ARMS:-"r6 m5"}; QOS=cont; TMT=01:00:00   # R6_ARMS=r6 submits one arm (M5 waited for its sigma, 2026-11-03)
tmf() { echo 05:00:00; }
tag() { case $1 in r6) echo t6;; m5) echo t5;; esac; }
CODE="cx_r1.py cx_r1fit.py cx_lib.py rbmpp_lib.py"
NEW_R1FIT=f1374bed9c7529a53ea8d95177eb709f6a79c2fbd78906351a66c7f0319babc5; RBMPP=97eade567b035835ba6d74d4b5f21a11612cdf264769152ee3d1533f812d82fb
T6SHA=$(cut -d' ' -f1 $Q/target_T6_sG.npy.sha256); T5SHA=$(cut -d' ' -f1 $R5/target_T5_sG.npy.sha256)
[ "$(sha256sum $Q/target_T6_sG.npy | cut -d' ' -f1)" = "$T6SHA" ] || { echo "target_T6 sha mismatch"; exit 1; }
[ "$(sha256sum $R5/target_T5_sG.npy | cut -d' ' -f1)" = "$T5SHA" ] || { echo "target_T5 sha mismatch"; exit 1; }
[ "$(sha256sum $TH4 | cut -d' ' -f1)" = "$TH4SHA" ] || { echo "theta_4 sha mismatch"; exit 1; }
[ "$(sha256sum $TH5 | cut -d' ' -f1)" = "$TH5SHA" ] || { echo "pp2_s1 theta_5 sha mismatch"; exit 1; }
python3 -c "
import json; d=json.load(open('$Q/prep_result.json')); assert d.get('done') and d['E_T6_sG']['pass_1e-6_rel'] and d['T6_sha256']=='$T6SHA', 'prep control (E_FN[T_6]) not passed'
s=json.load(open('$Q/cpu/selftest_m5_s3em3/self_test_m5.json')); assert s['rbm_init']['sigma']==0.003 and s['theta_start_is_pp_plus_sigma_z'] and all(s[k] for k in ('a_pass','b_pass','c_pass','d_pass')), 'M5 CPU checks (a-d) not passed'
i=json.load(open('$Q/cpu/ident_result.json')); assert i['all_identical'], 'old arms not byte-identical under the new code'"
# 1a. specs (local only, written before any remote action so that launch_r6_th.sh can stage without the VPN)
for A in $ARMS; do mkdir -p $Q/$A/fit $Q/$A/tab/tab_$(tag $A); done; mkdir -p $Q/readout
python3 - "$R5" "$Q" "$P" "$D" "$D6" "$NEW_R1FIT" <<'PY'
import json, sys
R5, Q, P, D, D6, sha_fit = sys.argv[1:7]
SAVE = [600, 750, 900, 1050, 1200, 1350]
for A, src in (('r6', f'{R5}/pp2_s1/fit/spec_r1.json'), ('m5', f'{R5}/ctl_s1/fit/spec_r1.json')):
    base = json.load(open(src)); sp = json.loads(json.dumps(base))
    assert base['max_steps'] == 1500 and base['fit_overrides']['seed'] == 1 and base['fit_overrides']['save_theta_at'] == SAVE and base['fit_overrides']['min_steps'] == 1500
    assert base['rbm_alpha'] == 4 and base['sign_path'].endswith('lswo2/s_G.npy') and base['skip_efn'] and base['skip_fn_readouts']
    sp['code_sha256']['cx_r1fit.py'] = sha_fit; sp['fit_sec'] = 16000.0
    if A == 'r6':
        assert base['arm'] == 'pp2'
        sp.update(target_la=f'{D6}/target_T6_sG.npy', warm_theta=f'{D6}/theta_5_pp2s1.npy')
        sp['fit_overrides'] = {'seed': 1, 'min_steps': 1500, 'save_theta_at': SAVE}
    else:
        assert base['arm'] == 'rbmpp' and base['target_la'] == f'{D}/r5/target_T5_sG.npy' and base['warm_theta'] == f'{D}/r5/theta_4.npy'
        sp['rbm_alpha'] = 72
        sp['fit_overrides'] = {'seed': 1, 'min_steps': 1500, 'save_theta_at': SAVE, 'rbm_init_sigma': 0.003, 'rbm_init_seed': 72}
    json.dump(sp, open(f'{Q}/{A}/fit/spec_r1.json', 'w'), indent=1)
    TG = {'r6': 't6', 'm5': 't5'}[A]; RT = f'{P}/runs/r6_{A}_tab'
    t = json.loads(json.dumps(sp)); t.update(warm_theta=f'{RT}/theta_{TG[1]}.npy', fit_sec=1.0, max_steps=0); t['fit_overrides'] = {'seed': 1}
    json.dump(t, open(f'{Q}/{A}/tab/tab_{TG}/spec_r1.json', 'w'), indent=1)
    if A == 'm5':                                      # initial theta (theta_4 PP + RBM init): the fit spec with max_steps 0 -> la table of theta_start = e_start
        t0 = json.loads(json.dumps(sp)); t0.update(fit_sec=1.0, max_steps=0); __import__('os').makedirs(f'{Q}/{A}/tab/tab_t0', exist_ok=True)
        json.dump(t0, open(f'{Q}/{A}/tab/tab_t0/spec_r1.json', 'w'), indent=1)
PY
diff -u $R5/pp2_s1/fit/spec_r1.json $Q/r6/fit/spec_r1.json > $Q/spec_diff_r6.txt || true; diff -u $R5/pp2_s1/tab/tab_t5/spec_r1.json $Q/r6/tab/tab_t6/spec_r1.json > $Q/spec_diff_r6_tab.txt || true
diff -u $R5/ctl_s1/fit/spec_r1.json $Q/m5/fit/spec_r1.json > $Q/spec_diff_m5.txt || true; diff -u $R5/ctl_s1/tab/tab_t5/spec_r1.json $Q/m5/tab/tab_t5/spec_r1.json > $Q/spec_diff_m5_tab.txt || true
[ "$(sha256sum $HERE/cx_r1fit.py | cut -d' ' -f1)" = "$NEW_R1FIT" ] || { echo "local cx_r1fit.py sha != $NEW_R1FIT"; exit 1; }
{ for f in $CODE; do sha256sum $HERE/$f | sed "s#  .*#  $f#"; done; } | sort > $Q/code_sha256.txt
chk() { # LOCALFILE REMOTEFILE
  l=$(sha256sum "$1" | cut -d' ' -f1); r=$(rem "sha256sum $2 2>/dev/null" | cut -d' ' -f1 || true)
  if [ "$l" != "$r" ]; then rsync -a -e "$SSHO" "$1" paderborn:$2; r=$(rem "sha256sum $2" | cut -d' ' -f1); fi
  [ "$l" = "$r" ] || { echo "sha256 mismatch $2"; exit 1; }; echo "input ok ${l:0:12} $2"; }
# 0. code: local pins, then codeR6 on Paderborn (copied, sha-checked)
[ "$(sha256sum $HERE/cx_r1fit.py | cut -d' ' -f1)" = "$NEW_R1FIT" ] || { echo "local cx_r1fit.py sha != $NEW_R1FIT"; exit 1; }
[ "$(sha256sum $HERE/rbmpp_lib.py | cut -d' ' -f1)" = "$RBMPP" ] || { echo "local rbmpp_lib.py sha != $RBMPP"; exit 1; }
rem "mkdir -p $CR"; for f in $CODE; do chk $HERE/$f $CR/$f; done
for f in st6_sector.py vit_dt.py; do chk /project/theorie/a/A.Otaifi/Chatty/runs/compress6/code/$f $CR/$f; done     # helper modules (as codeR5)
rem "cd $CR && sha256sum $CODE" | sort > $Q/code_sha256_remote.txt
diff $Q/code_sha256.txt $Q/code_sha256_remote.txt > /dev/null || { echo "codeR6 sha256 differs from local pins"; exit 1; }
# 1. data: shared inputs sha-checked (as launch_r5_pdb.sh) and the round-6 inputs
chk /project/theorie/a/A.Otaifi/Chatty/runs/compress6/runs/o2b_17431886/round_1/spec_fit.json $D/o2b_r1/round_1_spec_fit.json
chk /project/theorie/a/A.Otaifi/Chatty/runs/rbmpp6/runs/rpT1_pp_17163493/params_final.npz $D/rpT1_pp_17163493/params_final.npz
chk /project/theorie/a/A.Otaifi/Chatty/runs/rbmpp6/runs/rpTabPP_17164048/sym_tables_PP.npz $D/rpTabPP_17164048/sym_tables_PP.npz
chk /project/theorie/a/A.Otaifi/Chatty/runs/stall6/data/psi0_6x6_table.npz $D/psi0_6x6_table.npz
chk /project/theorie/a/A.Otaifi/Chatty/runs/stall6/data/sym_tables.npz $D/sym_tables.npz
chk /project/theorie/a/A.Otaifi/Chatty/runs/stall6/csr/norb.npy $D/csr/norb.npy
chk /project/theorie/a/A.Otaifi/Chatty/runs/compress6/runs/refresh_17409419/s2.npy $D/refresh_17409419/s2.npy
rem "cd $D/csr; ls -l meta.json up_indices.npy up_codes.npy up_inc.npy" > /dev/null
rem "mkdir -p $D6"
chk /scratch/a/A.Otaifi/ansatz_runs/pfaffian_lswo2/s_G.npy $D/lswo2/s_G.npy
chk $Q/target_T6_sG.npy $D6/target_T6_sG.npy
chk $R5/target_T5_sG.npy $D/r5/target_T5_sG.npy
chk $TH4 $D/r5/theta_4.npy
chk $TH5 $D6/theta_5_pp2s1.npy
# 3. run dirs, briefs, sbatch files
for A in $ARMS; do
  RUNF=$P/runs/r6_${A}_fit; RUNT=$P/runs/r6_${A}_tab; TG=$(tag $A); rem "mkdir -p $RUNF $RUNT/tab_$TG"
  for K in fit tab; do
    [ $K = fit ] && { LOC=$Q/$A/fit; REM=$RUNF; } || { LOC=$Q/$A/tab; REM=$RUNT; }
    { echo "# Round 6 Paderborn ($A $K job; remote $REM, local $LOC).  Full brief and verdict rules: $Q/ANALYSIS_BRIEF.md"; sed -n '2,$p' "$Q/ANALYSIS_BRIEF.md"; } > "$LOC/ANALYSIS_BRIEF.md"
    rsync -a -e "$SSHO" "$LOC/ANALYSIS_BRIEF.md" paderborn:$REM/
  done
  rsync -a -e "$SSHO" "$Q/$A/fit/spec_r1.json" paderborn:$RUNF/
  rsync -a -e "$SSHO" "$Q/$A/tab/tab_$TG/spec_r1.json" paderborn:$RUNT/tab_$TG/
  [ $A = m5 ] && { rem "mkdir -p $RUNT/tab_t0"; rsync -a -e "$SSHO" "$Q/$A/tab/tab_t0/spec_r1.json" paderborn:$RUNT/tab_t0/; }
done
cp "$Q/ANALYSIS_BRIEF.md" "$Q/readout/ANALYSIS_BRIEF.md"
rsync -a -e "$SSHO" "$HERE/r6_pdb.sbatch" "$HERE/r6_pdb_tab.sbatch" paderborn:$P/
for f in r6_pdb.sbatch r6_pdb_tab.sbatch; do rem "sha256sum $P/$f" | cut -d' ' -f1 | grep -qx "$(sha256sum "$HERE/$f" | cut -d' ' -f1)" || { echo "$f sha256 differs after copy"; exit 1; }; done
echo "staged Paderborn: code $CR, data $D6, runs $P/runs/r6_{r6,m5}_{fit,tab}; local $Q/<arm>/{fit,tab}; spec diffs $Q/spec_diff_*.txt"
fitf() { echo "--parsable --no-requeue --qos=$QOS --job-name=pf_r6_$1 --time=$(tmf $1) --output=$P/runs/r6_$1_fit/slurm-%j.out"; }
tabf() { echo "--parsable --no-requeue --qos=$QOS --job-name=pf_r6t_$1 --time=$TMT --output=$P/runs/r6_$1_tab/slurm-%j.out"; }
if [ "$MODE" = --test ]; then
  for A in $ARMS; do
    echo "--- test-only fit $A";  rem "sbatch --test-only $(fitf $A) $P/r6_pdb.sbatch $P/runs/r6_${A}_fit" 2>&1
    echo "--- test-only tab $A (no dependency: a test-only cannot see the not-yet-existing fit job)"; rem "sbatch --test-only $(tabf $A) $P/r6_pdb_tab.sbatch $P/runs/r6_${A}_tab $P/runs/r6_${A}_fit $(tag $A)" 2>&1
  done
fi
[ "$MODE" = --submit ] || exit 0
for A in $ARMS; do
  [ -e "$Q/$A/fit/job_id.txt" ] && { echo "$A: $Q/$A/fit/job_id.txt exists (already submitted), skipped"; continue; }
  [ -e "$Q/$A/pdb_skip" ] && { echo "$A: theorie race copy already won ($(cat $Q/$A/pdb_skip)), Paderborn not submitted"; continue; }
  RUNF=$P/runs/r6_${A}_fit; RUNT=$P/runs/r6_${A}_tab; TG=$(tag $A)
  OUTL=$(rem "sbatch $(fitf $A) $P/r6_pdb.sbatch $RUNF" 2>/dev/null); JOB=${OUTL%%;*}
  [[ "$JOB" =~ ^[0-9]+$ ]] || { echo "sbatch failed (fit $A): $OUTL"; exit 1; }
  echo "$JOB" > "$Q/$A/fit/job_id.txt"; echo "$A fit $JOB $(date -Is)" >> "$Q/job_ids.txt"
  R1_REMRUN=$RUNF setsid nohup "$HERE/lswo_pdb_watch.sh" "$JOB" "$Q/$A/fit" > "$Q/$A/fit/watcher.log" 2>&1 < /dev/null &
  echo "$!" > "$Q/$A/fit/watcher_pid.txt"
  OUTT=$(rem "sbatch $(tabf $A) --dependency=afterok:$JOB $P/r6_pdb_tab.sbatch $RUNT $RUNF $TG" 2>/dev/null); JT=${OUTT%%;*}
  [[ "$JT" =~ ^[0-9]+$ ]] || { echo "sbatch failed (tab $A): $OUTT; fit $JOB already submitted"; exit 1; }
  echo "$JT" > "$Q/$A/tab/job_id.txt"; echo "$A tab $JT afterok:$JOB $(date -Is)" >> "$Q/job_ids.txt"
  R1_REMRUN=$RUNT R1_SUBS="tab_$TG$([ $A = m5 ] && echo " tab_t0")" setsid nohup "$HERE/r3_pdb_tab_watch.sh" "$JT" "$Q/$A/tab" > "$Q/$A/tab/watcher.log" 2>&1 < /dev/null &
  echo "$!" > "$Q/$A/tab/watcher_pid.txt"
  echo "$A: fit=$JOB (watcher $(cat $Q/$A/fit/watcher_pid.txt) ppid=$(ps -o ppid= -p $(cat $Q/$A/fit/watcher_pid.txt))) tab=$JT (watcher $(cat $Q/$A/tab/watcher_pid.txt) ppid=$(ps -o ppid= -p $(cat $Q/$A/tab/watcher_pid.txt)))"
done
if [ -s "$Q/readout/chain_pid.txt" ] && kill -0 "$(cat $Q/readout/chain_pid.txt)" 2>/dev/null; then echo "r6_chain.sh already running (pid $(cat $Q/readout/chain_pid.txt)), not restarted"
else setsid nohup "$HERE/r6_chain.sh" > "$Q/readout/chain.log" 2>&1 < /dev/null &
  echo "$!" > "$Q/readout/chain_pid.txt"; echo "chain pid $(cat $Q/readout/chain_pid.txt) ppid=$(ps -o ppid= -p $(cat $Q/readout/chain_pid.txt))"; fi
rem "sacct -u \$USER -S now-1hour -X -n -o JobID,JobName%20,State,Reason%30" 2>/dev/null; rem "squeue --start -u \$USER" 2>/dev/null || true
