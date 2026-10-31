#!/bin/bash
# R3-cont (PLAN_pfaffian.md "R3-cont"), Paderborn H100.  Default: STAGE ONLY (rsync + sha-check code/data, write specs and briefs; sbatch --test-only is run separately).
# --submit: sbatch  A fit (pf_r3cA, 01:30:00) + A tables (pf_r3cAtab, afterok, 00:30:00) and  B fit (pf_r3cB, 02:00:00) + B tables (pf_r3cBtab, afterok, 00:45:00); one durable setsid watcher per job
#   (lswo_pdb_watch.sh fits; r3_pdb_tab_watch.sh tables); then the chain gate r3c_chain.sh (waits for all four verdicts + the 10 la_tables pulled back, then sbatch r3c_readout.sbatch on the theorie cluster + r3_cpu_watch.sh).
# Needs the VPN.  Specs = pfaffian_r3/pdb_fit/spec_r1.json with: code_sha256.cx_r1fit.py = new, warm_theta, max_steps, fit_overrides {seed, min_steps, save_theta_at, lam0, tau 0.01, tau_min 0.01, [lam_floor]}.
#   A: warm theta(1500) = pfaffian_r3/pdb_fit/round_1/theta.npy, seed 2, lam0 = final lam_rel of that fit (0.03125 = hist[-1].lam_rel = lam_next, cx_t1s.json), lam_floor default 1e-6, 600 steps, save [150,300,450]
#   B: warm theta(1200) = .../theta_step1200.npy, seed 1, lam0 = lam_floor = 0.0078125 (= 2^-7 = the 7.8e-3 of the plan = lam_rel used at step 1200), 900 steps, save [150,300,450,600,750]
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
LS=/scratch/a/A.Otaifi/ansatz_runs/pfaffian_lswo2; Q3=/scratch/a/A.Otaifi/ansatz_runs/pfaffian_r3; Q=/scratch/a/A.Otaifi/ansatz_runs/pfaffian_r3c
P=/scratch/hpc-prf-ppdmrg/hpcalot/chatty_compress6; D=$P/data/lswo2; D3=$P/data/r3; DC=$P/data/r3c
SSHO="ssh -o BatchMode=yes -o HostName=login4"
rem() { $SSHO paderborn "export PATH=/opt/software/slurm/25.11.8-1/bin:\$PATH; $1"; }
FS=9000; TMA=01:30:00; TMB=02:00:00; TTA=00:30:00; TTB=00:45:00
CODE="cx_r1.py cx_r1fit.py cx_lib.py rbmpp_lib.py"
NEW_R1FIT=41f766ccaa71396d3fc12f57c027bd45bb2a08bafb042eefb9bbb4ba85175fc5
[ -e "$Q/A_fit/job_id.txt" ] && { echo "$Q/A_fit/job_id.txt exists: already submitted"; exit 1; }
chk() { # LOCALFILE REMOTEFILE
  l=$(sha256sum "$1" | cut -d' ' -f1); r=$(rem "sha256sum $2 2>/dev/null" | cut -d' ' -f1 || true)
  if [ "$l" != "$r" ]; then rsync -a -e "$SSHO" "$1" paderborn:$2; r=$(rem "sha256sum $2" | cut -d' ' -f1); fi
  [ "$l" = "$r" ] || { echo "sha256 mismatch $2"; exit 1; }; echo "input ok ${l:0:12} $2"; }
# 0. code: codeR3c = codeR1 pinned files + st6_sector.py, vit_dt.py, with the NEW cx_r1fit.py; every file sha-checked against the local pins
[ "$(sha256sum $HERE/cx_r1fit.py | cut -d' ' -f1)" = "$NEW_R1FIT" ] || { echo "local cx_r1fit.py sha != $NEW_R1FIT"; exit 1; }
rem "mkdir -p $P/codeR3c $DC; cd $P/codeR1 && cp -p cx_r1.py cx_lib.py rbmpp_lib.py st6_sector.py vit_dt.py $P/codeR3c/"
rsync -a -e "$SSHO" "$HERE/cx_r1fit.py" paderborn:$P/codeR3c/
rem "cd $P/codeR3c && sha256sum $CODE" | sed 's#  #  #' | sort > $Q/code_sha256_remote.txt
{ for f in $CODE; do sha256sum $HERE/$f | sed "s#  .*#  $f#"; done; } | sort > $Q/code_sha256.txt
diff $Q/code_sha256.txt $Q/code_sha256_remote.txt > /dev/null || { echo "codeR3c sha256 differs from local pins"; exit 1; }
for f in st6_sector.py vit_dt.py; do l=$(sha256sum /project/theorie/a/A.Otaifi/Chatty/runs/compress6/code/$f | cut -d' ' -f1); r=$(rem "sha256sum $P/codeR3c/$f" | cut -d' ' -f1); [ "$l" = "$r" ] || { echo "$f differs from compress6/code"; exit 1; }; echo "input ok ${l:0:12} codeR3c/$f"; done
# 1. data (shared inputs sha-checked) and the R3-cont inputs
chk /project/theorie/a/A.Otaifi/Chatty/runs/compress6/runs/o2b_17431886/round_1/spec_fit.json $P/data/o2b_r1/round_1_spec_fit.json
chk /project/theorie/a/A.Otaifi/Chatty/runs/rbmpp6/runs/rpT1_pp_17163493/params_final.npz $P/data/rpT1_pp_17163493/params_final.npz
chk /project/theorie/a/A.Otaifi/Chatty/runs/rbmpp6/runs/rpTabPP_17164048/sym_tables_PP.npz $P/data/rpTabPP_17164048/sym_tables_PP.npz
chk /project/theorie/a/A.Otaifi/Chatty/runs/stall6/data/psi0_6x6_table.npz $P/data/psi0_6x6_table.npz
chk /project/theorie/a/A.Otaifi/Chatty/runs/stall6/data/sym_tables.npz $P/data/sym_tables.npz
chk /project/theorie/a/A.Otaifi/Chatty/runs/stall6/csr/norb.npy $P/data/csr/norb.npy
chk /project/theorie/a/A.Otaifi/Chatty/runs/compress6/runs/refresh_17409419/s2.npy $P/data/refresh_17409419/s2.npy
rem "cd $P/data/csr; ls -l meta.json up_indices.npy up_codes.npy up_inc.npy" > /dev/null
chk $LS/s_G.npy $D/s_G.npy
chk $Q3/target_T3_sG.npy $D3/target_T3_sG.npy
chk $Q3/pdb_fit/round_1/theta.npy $DC/A_theta.npy
chk $Q3/pdb_fit/round_1/theta_step1200.npy $DC/B_theta.npy
python3 -c "
import numpy as np
for f in ('theta.npy','theta_step1200.npy'):
    a=np.load('$Q3/pdb_fit/round_1/'+f); assert a.size==2740 and a.dtype==np.float64 and np.isfinite(a).all(), f"
for A in A B; do mkdir -p $Q/${A}_fit; done
mkdir -p $Q/A_tabs/tab_{150,300,450} $Q/B_tabs/tab_{150,300,450,600,750} $Q/readout
python3 - "$Q3/pdb_fit/spec_r1.json" "$Q" "$P" "$DC" "$FS" "$NEW_R1FIT" <<'PY'
import json, sys
src, Q, P, DC, fs, newsha = sys.argv[1:7]; base = json.load(open(src))
assert base['fit_overrides'] == {'seed': 2, 'min_steps': 1500, 'save_theta_at': [900, 1200]} and base['max_steps'] == 1500 and base['sign_path'].endswith('lswo2/s_G.npy')
ARMS = dict(A=dict(warm=f'{DC}/A_theta.npy', ms=600, save=[150, 300, 450], ov=dict(seed=2, lam0=0.03125)),
            B=dict(warm=f'{DC}/B_theta.npy', ms=900, save=[150, 300, 450, 600, 750], ov=dict(seed=1, lam0=0.0078125, lam_floor=0.0078125)))
for arm, a in ARMS.items():
    sp = json.loads(json.dumps(base)); sp['code_sha256']['cx_r1fit.py'] = newsha
    sp.update(warm_theta=a['warm'], fit_sec=float(fs), max_steps=a['ms'])
    sp['fit_overrides'] = dict(a['ov'], min_steps=a['ms'], save_theta_at=a['save'], tau=0.01, tau_min=0.01)
    json.dump(sp, open(f'{Q}/{arm}_fit/spec_r1.json', 'w'), indent=1)
    RUNF = f'{P}/runs/r3c_{arm}_fit'
    for N in a['save']:
        t = json.loads(json.dumps(sp)); t.update(warm_theta=f'{RUNF}/round_1/theta_step{N}.npy', fit_sec=1.0, max_steps=0); t['fit_overrides'] = {'seed': a['ov']['seed']}
        import os; os.makedirs(f'{Q}/{arm}_tabs/tab_{N}', exist_ok=True); json.dump(t, open(f'{Q}/{arm}_tabs/tab_{N}/spec_r1.json', 'w'), indent=1)
PY
declare -A NS=([A]="150 300 450" [B]="150 300 450 600 750")
for A in A B; do
  RUNF=$P/runs/r3c_${A}_fit; RUNT=$P/runs/r3c_${A}_tab
  rem "mkdir -p $RUNF $RUNT"
  for D_ in fit tabs; do
    [ $D_ = fit ] && { LOC=$Q/${A}_fit; REM=$RUNF; } || { LOC=$Q/${A}_tabs; REM=$RUNT; }
    { echo "# R3-cont arm $A ($D_ job; remote $REM, local $LOC).  Full brief and verdict table: $Q/ANALYSIS_BRIEF.md"; sed -n '2,$p' "$Q/ANALYSIS_BRIEF.md"; } > "$LOC/ANALYSIS_BRIEF.md"
    rsync -a -e "$SSHO" "$LOC/ANALYSIS_BRIEF.md" paderborn:$REM/
  done
  rsync -a -e "$SSHO" "$Q/${A}_fit/spec_r1.json" paderborn:$RUNF/
  for N in ${NS[$A]}; do rem "mkdir -p $RUNT/tab_$N"; rsync -a -e "$SSHO" "$Q/${A}_tabs/tab_$N/spec_r1.json" paderborn:$RUNT/tab_$N/; done
done
cp "$Q/ANALYSIS_BRIEF.md" "$Q/readout/ANALYSIS_BRIEF.md"
rsync -a -e "$SSHO" "$HERE/r3c_pdb.sbatch" "$HERE/r3c_pdb_tab.sbatch" paderborn:$P/
echo "staged: A fit $P/runs/r3c_A_fit (wall $TMA, tabs $TTA), B fit $P/runs/r3c_B_fit (wall $TMB, tabs $TTB); local $Q/{A,B}_{fit,tabs}"
[ "${1:-}" = "--submit" ] || exit 0
for f in r3c_pdb.sbatch r3c_pdb_tab.sbatch; do rem "sha256sum $P/$f" | cut -d' ' -f1 | grep -qx "$(sha256sum "$HERE/$f" | cut -d' ' -f1)" || { echo "$f sha256 differs after copy"; exit 1; }; done
for A in A B; do
  [ $A = A ] && { TMF=$TMA; TMT=$TTA; } || { TMF=$TMB; TMT=$TTB; }
  RUNF=$P/runs/r3c_${A}_fit; RUNT=$P/runs/r3c_${A}_tab; LF=$Q/${A}_fit; LT=$Q/${A}_tabs
  OUTL=$(rem "sbatch --parsable --no-requeue --job-name=pf_r3c$A --time=$TMF --output=$RUNF/slurm-%j.out $P/r3c_pdb.sbatch $RUNF" 2>/dev/null); JOB=${OUTL%%;*}
  [[ "$JOB" =~ ^[0-9]+$ ]] || { echo "sbatch failed (fit $A): $OUTL"; exit 1; }
  echo "$JOB" > "$LF/job_id.txt"; echo "r3c_${A}_fit $JOB $(date -Is)" >> "$Q/job_ids.txt"
  R1_REMRUN=$RUNF setsid nohup "$HERE/lswo_pdb_watch.sh" "$JOB" "$LF" > "$LF/watcher.log" 2>&1 < /dev/null &
  echo "$!" > "$LF/watcher_pid.txt"
  OUTT=$(rem "sbatch --parsable --no-requeue --job-name=pf_r3c${A}tab --time=$TMT --dependency=afterok:$JOB --output=$RUNT/slurm-%j.out $P/r3c_pdb_tab.sbatch $RUNT $RUNF '${NS[$A]}'" 2>/dev/null); JT=${OUTT%%;*}
  [[ "$JT" =~ ^[0-9]+$ ]] || { echo "sbatch failed (tabs $A): $OUTT; fit $JOB already submitted"; exit 1; }
  echo "$JT" > "$LT/job_id.txt"; echo "r3c_${A}_tabs $JT afterok:$JOB $(date -Is)" >> "$Q/job_ids.txt"
  SUBS=""; for N in ${NS[$A]}; do SUBS="$SUBS tab_$N"; done
  R1_REMRUN=$RUNT R1_SUBS="${SUBS# }" setsid nohup "$HERE/r3_pdb_tab_watch.sh" "$JT" "$LT" > "$LT/watcher.log" 2>&1 < /dev/null &
  echo "$!" > "$LT/watcher_pid.txt"
  echo "arm $A: fit=$JOB (watcher $(cat $LF/watcher_pid.txt) ppid=$(ps -o ppid= -p $(cat $LF/watcher_pid.txt))) tabs=$JT (watcher $(cat $LT/watcher_pid.txt) ppid=$(ps -o ppid= -p $(cat $LT/watcher_pid.txt)))"
done
setsid nohup "$HERE/r3c_chain.sh" > "$Q/readout/chain.log" 2>&1 < /dev/null &
echo "$!" > "$Q/readout/chain_pid.txt"; echo "chain gate pid $(cat $Q/readout/chain_pid.txt) ppid=$(ps -o ppid= -p $(cat $Q/readout/chain_pid.txt))"
rem "sacct -j $(cat $Q/A_fit/job_id.txt),$(cat $Q/A_tabs/job_id.txt),$(cat $Q/B_fit/job_id.txt),$(cat $Q/B_tabs/job_id.txt) -X -n -o JobID,JobName,State,Reason%30" 2>/dev/null; rem "squeue --start -u \$USER" 2>/dev/null || true
