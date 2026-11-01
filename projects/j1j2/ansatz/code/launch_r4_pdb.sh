#!/bin/bash
# Round 4 (PLAN_pfaffian.md "Round 4"), Paderborn H100.  Modes: (none) STAGE ONLY; --test: stage + `sbatch --test-only` of every Paderborn job with the exact submit flags; --submit: sbatch for real.
# Jobs: fit pf_r4 (03:00:00, --qos cont) from theta_bar to T_4; table job pf_r4tab (01:00:00, afterok, --qos cont): theta_4 = mean of the 7 iterates, then tab_t4 + 6 checkpoint tables.
# Watchers (durable setsid, --wake none): lswo_pdb_watch.sh (fit), r3_pdb_tab_watch.sh (tabs), r4_chain.sh (gate -> theorie CPU readout).  Every ssh/rsync uses -o HostName=login4.
# Spec = pfaffian_r3/pdb_fit/spec_r1.json with ONLY: code_sha256.cx_r1fit.py (41f766cc), target_la (T_4), warm_theta (theta_bar), fit_overrides.save_theta_at [600..1350].  Diff vs round 3 written to $Q/spec_diff_pdb.txt.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
Q3=/scratch/a/A.Otaifi/ansatz_runs/pfaffian_r3; QA=/scratch/a/A.Otaifi/ansatz_runs/pfaffian_r3avg; Q=/scratch/a/A.Otaifi/ansatz_runs/pfaffian_r4
P=/scratch/hpc-prf-ppdmrg/hpcalot/chatty_compress6; D=$P/data; D4=$D/r4
SSHO="ssh -o BatchMode=yes -o HostName=login4"
rem() { $SSHO paderborn "export PATH=/opt/software/slurm/25.11.8-1/bin:\$PATH; $1"; }
MODE=${1:-stage}
FS=9000; TMF=03:00:00; TMT=01:00:00; QOS=cont; SUBS="t4 600 750 900 1050 1200 1350"
CODE="cx_r1.py cx_r1fit.py cx_lib.py rbmpp_lib.py"
NEW_R1FIT=41f766ccaa71396d3fc12f57c027bd45bb2a08bafb042eefb9bbb4ba85175fc5
T4SHA=$(cut -d' ' -f1 $Q/target_T4_sG.npy.sha256); BARSHA=1ec8f71516a98e96d414f763cf5baa15359e492467bf30bd2f44fa6b09c0d3f6
[ "$(sha256sum $Q/target_T4_sG.npy | cut -d' ' -f1)" = "$T4SHA" ] || { echo "target_T4 sha mismatch"; exit 1; }
[ "$(sha256sum $QA/theta_bar.npy | cut -d' ' -f1)" = "$BARSHA" ] || { echo "theta_bar sha mismatch"; exit 1; }
python3 -c "
import json,sys; d=json.load(open('$Q/prep_result.json')); assert d.get('done') and d['E_T4_sG']['pass_1e-6_rel'] and d['T4_sha256']=='$T4SHA', 'prep control not passed'"
[ "$MODE" = --submit ] && [ -e "$Q/pdb_fit/job_id.txt" ] && { echo "$Q/pdb_fit/job_id.txt exists: already submitted"; exit 1; }
chk() { # LOCALFILE REMOTEFILE
  l=$(sha256sum "$1" | cut -d' ' -f1); r=$(rem "sha256sum $2 2>/dev/null" | cut -d' ' -f1 || true)
  if [ "$l" != "$r" ]; then rsync -a -e "$SSHO" "$1" paderborn:$2; r=$(rem "sha256sum $2" | cut -d' ' -f1); fi
  [ "$l" = "$r" ] || { echo "sha256 mismatch $2"; exit 1; }; echo "input ok ${l:0:12} $2"; }
# 0. code (codeR3c, already on Paderborn) sha-checked against the local pins
[ "$(sha256sum $HERE/cx_r1fit.py | cut -d' ' -f1)" = "$NEW_R1FIT" ] || { echo "local cx_r1fit.py sha != $NEW_R1FIT"; exit 1; }
mkdir -p $Q/pdb_fit $Q/pdb_tabs $Q/readout
rem "cd $P/codeR3c && sha256sum $CODE" | sort > $Q/code_sha256_remote.txt
{ for f in $CODE; do sha256sum $HERE/$f | sed "s#  .*#  $f#"; done; } | sort > $Q/code_sha256.txt
diff $Q/code_sha256.txt $Q/code_sha256_remote.txt > /dev/null || { echo "codeR3c sha256 differs from local pins"; exit 1; }
# 1. data (shared inputs sha-checked, as launch_r3c_pdb.sh) and the round-4 inputs
chk /project/theorie/a/A.Otaifi/Chatty/runs/compress6/runs/o2b_17431886/round_1/spec_fit.json $D/o2b_r1/round_1_spec_fit.json
chk /project/theorie/a/A.Otaifi/Chatty/runs/rbmpp6/runs/rpT1_pp_17163493/params_final.npz $D/rpT1_pp_17163493/params_final.npz
chk /project/theorie/a/A.Otaifi/Chatty/runs/rbmpp6/runs/rpTabPP_17164048/sym_tables_PP.npz $D/rpTabPP_17164048/sym_tables_PP.npz
chk /project/theorie/a/A.Otaifi/Chatty/runs/stall6/data/psi0_6x6_table.npz $D/psi0_6x6_table.npz
chk /project/theorie/a/A.Otaifi/Chatty/runs/stall6/data/sym_tables.npz $D/sym_tables.npz
chk /project/theorie/a/A.Otaifi/Chatty/runs/stall6/csr/norb.npy $D/csr/norb.npy
chk /project/theorie/a/A.Otaifi/Chatty/runs/compress6/runs/refresh_17409419/s2.npy $D/refresh_17409419/s2.npy
rem "cd $D/csr; ls -l meta.json up_indices.npy up_codes.npy up_inc.npy" > /dev/null
rem "mkdir -p $D4"
chk /scratch/a/A.Otaifi/ansatz_runs/pfaffian_lswo2/s_G.npy $D/lswo2/s_G.npy
chk $Q/target_T4_sG.npy $D4/target_T4_sG.npy
chk $QA/theta_bar.npy $D4/theta_bar.npy
python3 -c "
import numpy as np; a=np.load('$QA/theta_bar.npy'); assert a.size==2740 and a.dtype==np.float64 and np.isfinite(a).all()"
# 2. specs
mkdir -p $Q/pdb_tabs; for L in $SUBS; do mkdir -p $Q/pdb_tabs/tab_$L; done
python3 - "$Q3/pdb_fit/spec_r1.json" "$Q" "$P" "$D4" "$FS" "$NEW_R1FIT" <<'PY'
import json, sys, os
src, Q, P, D4, fs, newsha = sys.argv[1:7]; base = json.load(open(src))
assert base['fit_overrides'] == {'seed': 2, 'min_steps': 1500, 'save_theta_at': [900, 1200]} and base['max_steps'] == 1500 and base['fit_sec'] == float(fs) and base['sign_path'].endswith('lswo2/s_G.npy')
sp = json.loads(json.dumps(base)); sp['code_sha256']['cx_r1fit.py'] = newsha
sp.update(target_la=f'{D4}/target_T4_sG.npy', warm_theta=f'{D4}/theta_bar.npy')
sp['fit_overrides'] = {'seed': 2, 'min_steps': 1500, 'save_theta_at': [600, 750, 900, 1050, 1200, 1350]}
json.dump(sp, open(f'{Q}/pdb_fit/spec_r1.json', 'w'), indent=1)
RT = f'{P}/runs/r4_pdb_tab'
for L in 't4 600 750 900 1050 1200 1350'.split():
    t = json.loads(json.dumps(sp)); t.update(warm_theta=f'{RT}/theta_4.npy' if L == 't4' else f'{P}/runs/r4_pdb_fit/round_1/theta_step{L}.npy', fit_sec=1.0, max_steps=0); t['fit_overrides'] = {'seed': 2}
    json.dump(t, open(f'{Q}/pdb_tabs/tab_{L}/spec_r1.json', 'w'), indent=1)
PY
diff -u $Q3/pdb_fit/spec_r1.json $Q/pdb_fit/spec_r1.json > $Q/spec_diff_pdb.txt || true
# 3. run dirs, briefs, sbatch files
RUNF=$P/runs/r4_pdb_fit; RUNT=$P/runs/r4_pdb_tab; rem "mkdir -p $RUNF $RUNT"
for K in fit tabs; do
  [ $K = fit ] && { LOC=$Q/pdb_fit; REM=$RUNF; } || { LOC=$Q/pdb_tabs; REM=$RUNT; }
  { echo "# Round 4 Paderborn ($K job; remote $REM, local $LOC).  Full brief and verdict rules: $Q/ANALYSIS_BRIEF.md"; sed -n '2,$p' "$Q/ANALYSIS_BRIEF.md"; } > "$LOC/ANALYSIS_BRIEF.md"
  rsync -a -e "$SSHO" "$LOC/ANALYSIS_BRIEF.md" paderborn:$REM/
done
rsync -a -e "$SSHO" "$Q/pdb_fit/spec_r1.json" paderborn:$RUNF/
for L in $SUBS; do rem "mkdir -p $RUNT/tab_$L"; rsync -a -e "$SSHO" "$Q/pdb_tabs/tab_$L/spec_r1.json" paderborn:$RUNT/tab_$L/; done
cp "$Q/ANALYSIS_BRIEF.md" "$Q/readout/ANALYSIS_BRIEF.md"
rsync -a -e "$SSHO" "$HERE/r3c_pdb.sbatch" "$HERE/r4_pdb_tab.sbatch" paderborn:$P/
for f in r3c_pdb.sbatch r4_pdb_tab.sbatch; do rem "sha256sum $P/$f" | cut -d' ' -f1 | grep -qx "$(sha256sum "$HERE/$f" | cut -d' ' -f1)" || { echo "$f sha256 differs after copy"; exit 1; }; done
echo "staged Paderborn: fit $RUNF (wall $TMF fit_sec $FS qos $QOS), tabs $RUNT (wall $TMT); local $Q/{pdb_fit,pdb_tabs}; spec diff $Q/spec_diff_pdb.txt"
FITF="--parsable --no-requeue --qos=$QOS --job-name=pf_r4 --time=$TMF --output=$RUNF/slurm-%j.out"
TABF="--parsable --no-requeue --qos=$QOS --job-name=pf_r4tab --time=$TMT --output=$RUNT/slurm-%j.out"
if [ "$MODE" = --test ]; then
  echo "--- test-only fit";  rem "sbatch --test-only $FITF $P/r3c_pdb.sbatch $RUNF" 2>&1
  echo "--- test-only tabs (no dependency: a test-only cannot see the not-yet-existing fit job)"; rem "sbatch --test-only $TABF $P/r4_pdb_tab.sbatch $RUNT $RUNF '$SUBS'" 2>&1
fi
[ "$MODE" = --submit ] || exit 0
OUTL=$(rem "sbatch $FITF $P/r3c_pdb.sbatch $RUNF" 2>/dev/null); JOB=${OUTL%%;*}
[[ "$JOB" =~ ^[0-9]+$ ]] || { echo "sbatch failed (fit): $OUTL"; exit 1; }
echo "$JOB" > "$Q/pdb_fit/job_id.txt"; echo "pdb_fit $JOB $(date -Is)" >> "$Q/job_ids.txt"
R1_REMRUN=$RUNF setsid nohup "$HERE/lswo_pdb_watch.sh" "$JOB" "$Q/pdb_fit" > "$Q/pdb_fit/watcher.log" 2>&1 < /dev/null &
echo "$!" > "$Q/pdb_fit/watcher_pid.txt"
OUTT=$(rem "sbatch $TABF --dependency=afterok:$JOB $P/r4_pdb_tab.sbatch $RUNT $RUNF '$SUBS'" 2>/dev/null); JT=${OUTT%%;*}
[[ "$JT" =~ ^[0-9]+$ ]] || { echo "sbatch failed (tabs): $OUTT; fit $JOB already submitted"; exit 1; }
echo "$JT" > "$Q/pdb_tabs/job_id.txt"; echo "pdb_tabs $JT afterok:$JOB $(date -Is)" >> "$Q/job_ids.txt"
SB=""; for L in $SUBS; do SB="$SB tab_$L"; done
R1_REMRUN=$RUNT R1_SUBS="${SB# }" setsid nohup "$HERE/r3_pdb_tab_watch.sh" "$JT" "$Q/pdb_tabs" > "$Q/pdb_tabs/watcher.log" 2>&1 < /dev/null &
echo "$!" > "$Q/pdb_tabs/watcher_pid.txt"
setsid nohup "$HERE/r4_chain.sh" > "$Q/readout/chain.log" 2>&1 < /dev/null &
echo "$!" > "$Q/readout/chain_pid.txt"
echo "fit=$JOB (watcher $(cat $Q/pdb_fit/watcher_pid.txt) ppid=$(ps -o ppid= -p $(cat $Q/pdb_fit/watcher_pid.txt))) tabs=$JT (watcher $(cat $Q/pdb_tabs/watcher_pid.txt) ppid=$(ps -o ppid= -p $(cat $Q/pdb_tabs/watcher_pid.txt))) chain pid $(cat $Q/readout/chain_pid.txt) ppid=$(ps -o ppid= -p $(cat $Q/readout/chain_pid.txt))"
rem "sacct -j $JOB,$JT -X -n -o JobID,JobName,State,Reason%30" 2>/dev/null; rem "squeue --start -u \$USER" 2>/dev/null || true
