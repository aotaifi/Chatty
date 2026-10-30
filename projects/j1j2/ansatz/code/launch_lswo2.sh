#!/bin/bash
# L-SWO round 2 launcher (PLAN_pfaffian.md "L-SWO round 2"; copy of launch_lswo.sh), two arms on Paderborn H100s:
#   s1 : target_T2_s1.npy (ln T2 = ln|B_1| + 4.0 f_J[B_1], normalised), sign s1;   sG : target_TG_sG.npy (ln T_G = ln|B_1| + c_G f_J^G[B_1]), sign s_G.npy.
# Default: STAGE ONLY.  --submit: stage, sbatch both (gpu_h100, hpc-prf-ppdmrg, --no-requeue, wall 03:00:00), record job ids (run dirs, pfaffian_lswo2/job_ids.txt),
# arm one durable watcher per job (setsid, lswo_pdb_watch.sh -> job_finish.sh --wake none).
# Differences from launch_lswo.sh (spec): target_la = the arm's target; warm_theta = B_1's theta.npy (2740 = rbmpp NPAR, taken as is by cx_r1fit.py, RBM block 0 as in B_1);
#   max_steps 3000, fit_sec 9000, wall 03:00:00; sign_path = s_G.npy for arm sG.  Class, code af662d3 (cx_r1.py 5ebae9d8, cx_r1fit.py 09f66f19), seed 1, fit knobs unchanged.
# Run dirs: remote runs/lswo2_s1, runs/lswo2_sG; local /scratch/a/A.Otaifi/ansatz_runs/pfaffian_lswo2/run_s1, run_sG.  Needs the VPN.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
RUNS=/project/theorie/a/A.Otaifi/Chatty/runs/compress6/runs
R1L=/scratch/a/A.Otaifi/ansatz_runs/pfaffian_r1; L0=/scratch/a/A.Otaifi/ansatz_runs/pfaffian_lswo; LL=/scratch/a/A.Otaifi/ansatz_runs/pfaffian_lswo2
P=/scratch/hpc-prf-ppdmrg/hpcalot/chatty_compress6; D=$P/data/lswo2; D1=$P/data/r1; DO=$P/data/o2b_r1
SSHO="ssh -o BatchMode=yes -o HostName=login4"
rem() { $SSHO paderborn "export PATH=/opt/software/slurm/25.11.8-1/bin:\$PATH; $1"; }
TM=03:00:00; FS=9000; MX=3000
CODE="cx_r1.py cx_r1fit.py cx_lib.py rbmpp_lib.py"
B1TH=$L0/run/round_1/theta.npy
for ARM in s1 sG; do [ -e "$LL/run_$ARM/job_id.txt" ] && { echo "$LL/run_$ARM/job_id.txt exists: already submitted"; exit 1; }; done
chk() { # LOCALFILE REMOTEFILE
  l=$(sha256sum "$1" | cut -d' ' -f1); r=$(rem "sha256sum $2 2>/dev/null" | cut -d' ' -f1 || true)
  if [ "$l" != "$r" ]; then rsync -a -e "$SSHO" "$1" paderborn:$2; r=$(rem "sha256sum $2" | cut -d' ' -f1); fi
  [ "$l" = "$r" ] || { echo "sha256 mismatch $2"; exit 1; }; echo "input ok ${l:0:12} $2"; }
# 0. code pinned by sha256 (identical to the L-SWO / R1 run)
rem "mkdir -p $P/codeR1 $D"
( cd "$HERE" && sha256sum $CODE ) > "$LL/code_sha256.txt"
diff "$LL/code_sha256.txt" $L0/run/code_sha256.txt || { echo "code sha256 differs from the L-SWO run (cx_r1.py 5ebae9d8, cx_r1fit.py 09f66f19)"; exit 1; }
for f in $CODE; do rsync -a -e "$SSHO" "$HERE/$f" paderborn:$P/codeR1/; done
rem "cd $P/codeR1 && sha256sum $CODE" | diff - "$LL/code_sha256.txt" > /dev/null || { echo "code sha256 differs after copy"; exit 1; }
( cd "$HERE" && git rev-parse HEAD && git status --short -- $CODE ) > "$LL/code_git_status.txt" || true
# 1. inputs (shared with L-SWO, sha-checked) and the round-2 inputs
chk /scratch/a/A.Otaifi/ansatz_runs/pfaffian_z/s1.npy $D1/s1.npy
chk $RUNS/o2b_17431886/round_1/spec_fit.json $DO/round_1_spec_fit.json
chk /project/theorie/a/A.Otaifi/Chatty/runs/rbmpp6/runs/rpT1_pp_17163493/params_final.npz $P/data/rpT1_pp_17163493/params_final.npz
chk /project/theorie/a/A.Otaifi/Chatty/runs/rbmpp6/runs/rpTabPP_17164048/sym_tables_PP.npz $P/data/rpTabPP_17164048/sym_tables_PP.npz
chk /project/theorie/a/A.Otaifi/Chatty/runs/stall6/data/psi0_6x6_table.npz $P/data/psi0_6x6_table.npz
chk /project/theorie/a/A.Otaifi/Chatty/runs/stall6/data/sym_tables.npz $P/data/sym_tables.npz
chk /project/theorie/a/A.Otaifi/Chatty/runs/stall6/csr/norb.npy $P/data/csr/norb.npy
rem "cd $P/data/csr; ls -l meta.json up_indices.npy up_codes.npy up_inc.npy" > /dev/null
chk $B1TH $D/B1_theta.npy
chk $LL/target_T2_s1.npy $D/target_T2_s1.npy
chk $LL/target_TG_sG.npy $D/target_TG_sG.npy
chk $LL/s_G.npy $D/s_G.npy
python3 -c "import numpy as np,sys; a=np.load('$B1TH'); assert a.size==2740 and a.dtype==np.float64 and (a[2592:]==0).all()" || { echo "B_1 theta not 2740 / RBM block not zero"; exit 1; }
for ARM in s1 sG; do
  NAME=lswo2_$ARM; LOC=$LL/run_$ARM; RUN=$P/runs/$NAME; mkdir -p "$LOC"; rem "mkdir -p $RUN"
  cp "$LL/code_sha256.txt" "$LL/code_git_status.txt" "$LOC/"
  if [ $ARM = s1 ]; then TGT=$D/target_T2_s1.npy; SGN=$D1/s1.npy; else TGT=$D/target_TG_sG.npy; SGN=$D/s_G.npy; fi
  # 2. spec = L-SWO's spec_r1.json except target_la, warm_theta, max_steps, fit_sec (and sign_path for sG)
  python3 - "$LOC" "$P" "$D" "$DO" "$FS" "$MX" "$TGT" "$SGN" <<'PY'
import json, sys
loc, P, D, DO, fs, mx, tgt, sgn = sys.argv[1:9]
sha = {l.split()[1]: l.split()[0] for l in open(f'{loc}/code_sha256.txt')}
sp = dict(refresh=f'{P}/data/refresh_17409419', pp_params=f'{P}/data/rpT1_pp_17163493/params_final.npz', pp_table=f'{P}/data/rpTabPP_17164048/sym_tables_PP.npz',
          code_sha256=sha, target_la=tgt, sign_path=sgn, warm_theta=f'{D}/B1_theta.npy',
          o2b_fit_spec=f'{DO}/round_1_spec_fit.json', window=3, freeze_outer=False, arm='rbmpp', rbm_alpha=4, skip_efn=True, skip_fn_readouts=True,
          fit_sec=float(fs), max_steps=int(mx), fit_overrides=dict(seed=1))
json.dump(sp, open(f'{loc}/spec_r1.json', 'w'), indent=1)
PY
  rsync -a -e "$SSHO" "$LOC/spec_r1.json" paderborn:$RUN/
  # 3. brief
  if [ $ARM = s1 ]; then BASE="sign s1; target T2 = B_1 e^{4.0 f_J[B_1]} (E_FN[T2,s1] = 8.3376e-6 per site), Delta_J2 = e(B_1) - e(T2) = 1.12e-5"; \
    GATE="e(B_1,s1) = 1.957e-5; prediction bare e(B_2) 1.45-1.75e-5; augmented B_2 e^{c f_J[B_2]}: c* 3.5-4.5, 6.5-7.8e-6; KILL (candidate (1)): e(B_1) - e(B_2) <= 0.5e-6"
  else BASE="sign s_G (Lanczos sign, s_G.npy); target T_G = B_1 e^{c_G f_J^G[B_1]} (e(T_G,s_G) from prep_result.json); confound: B_1 was trained under s1"; \
    GATE="e(B_1,s_G) 1.7-1.9e-5 (prep_result.json E_B1_sG); prediction e(T_G) 7.0-7.5e-6, bare e(B_2^G,s_G) 1.3-1.6e-5, augmented 5.8-7.0e-6 (competitor bar 5.6e-6); KILL: e(B_1,s_G) - e(B_2^G,s_G) <= 0.5e-6"; fi
  cat > "$LOC/ANALYSIS_BRIEF.md" <<EOB
# L-SWO round 2 GPU fit, arm $ARM, Paderborn (remote $RUN, local $LOC), PLAN_pfaffian.md "L-SWO round 2" (projects/j1j2); $BASE
Question: fit B_2 = e^R x complex PP (real symmetric RBM alpha 4, no backflow), warm-started from B_1's theta (2740 values, RBM block 0), to the arm's target; same fit spec as L-SWO
(seed 1, GN/minSR, flatness rule), max_steps $MX, fit_sec $FS, wall $TM; samples and bond weights from the target itself.  Prep (targets, s_G, line search): $LL/prep_result.json.
Files: $LOC/cx_r1.json (steps, converged, reason, median_sec_per_step, L_ho_curve, table_sec), round_1/{cx_t1s.json, fit.log, progress.txt, theta.npy}, round_1/la_table.npy (pulled by the watcher).
Gate (fit only; no Monte Carlo, quotable.py does not apply): python3 $HERE/r1_gate.py $LOC
  Physics gate = the CPU readout per PLAN "L-SWO round 2": $GATE.
Branches: FIT OK, FIT NOT OK, TIMEOUT, STALLED, FAILED -> next step: report to the angle lead (steps, converged, s/step, wall, H100-h, la_table path); the CPU readout is run by the lead.
  No resubmission without the lead.
EOB
  rsync -a -e "$SSHO" "$LOC/ANALYSIS_BRIEF.md" paderborn:$RUN/
  echo "staged arm $ARM: remote $RUN, local $LOC; wall $TM fit_sec $FS max_steps $MX"
done
[ "${1:-}" = "--submit" ] || exit 0
rsync -a -e "$SSHO" "$HERE/lswo2.sbatch" paderborn:$P/lswo2.sbatch
rem "sha256sum $P/lswo2.sbatch" | cut -d' ' -f1 | grep -qx "$(sha256sum "$HERE/lswo2.sbatch" | cut -d' ' -f1)" || { echo "lswo2.sbatch sha256 differs after copy"; exit 1; }
for ARM in s1 sG; do
  NAME=lswo2_$ARM; LOC=$LL/run_$ARM; RUN=$P/runs/$NAME
  OUTL=$(rem "sbatch --parsable --no-requeue --job-name=$NAME --time=$TM --output=$RUN/slurm-%j.out $P/lswo2.sbatch $RUN" 2>/dev/null)
  JOB=${OUTL%%;*}; [[ "$JOB" =~ ^[0-9]+$ ]] || { echo "sbatch failed ($ARM): $OUTL"; exit 1; }
  echo "$JOB" > "$LOC/job_id.txt"; echo "$ARM $JOB $(date -Is)" >> "$LL/job_ids.txt"
  R1_REMRUN=$RUN setsid nohup "$HERE/lswo_pdb_watch.sh" "$JOB" "$LOC" > "$LOC/watcher.log" 2>&1 < /dev/null &
  echo "$!" > "$LOC/watcher_pid.txt"
  echo "arm=$ARM job=$JOB run=$RUN watcher_pid=$(cat $LOC/watcher_pid.txt) ppid=$(ps -o ppid= -p $(cat $LOC/watcher_pid.txt))"
  rem "sacct -j $JOB -X -n -o State,Reason%30" 2>/dev/null
done
rem "squeue --start -u \$USER" 2>/dev/null || true
