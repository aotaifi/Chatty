#!/bin/bash
# R2G-conv launcher (PLAN_pfaffian.md "R2G-conv"; copy of launch_lswo2.sh, sG arm only), Paderborn H100, seeds 1 and 2 (run dirs runs/r2g_s1, runs/r2g_s2).
# Spec = the round-2 s_G arm's (pfaffian_lswo2/run_sG/spec_r1.json: target_TG_sG.npy, s_G.npy, B_1 warm theta, code af662d3) except fit_overrides.seed = k (fit_sec 9000, max_steps 3000, wall 03:00:00 as in round 2).
# Default: STAGE ONLY.  --submit: stage, sbatch both seeds (gpu_h100, hpc-prf-ppdmrg, --no-requeue), record ids (pfaffian_r2g/job_ids.txt), arm one durable watcher per job (lswo_pdb_watch.sh -> job_finish.sh --wake none).
# Local mirror /scratch/a/A.Otaifi/ansatz_runs/pfaffian_r2g/run_s{1,2}.  Needs the VPN.  Then launch_r2g_a40.sh --submit (race watchers read these ids).
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
RUNS=/project/theorie/a/A.Otaifi/Chatty/runs/compress6/runs
R1L=/scratch/a/A.Otaifi/ansatz_runs/pfaffian_r1; L0=/scratch/a/A.Otaifi/ansatz_runs/pfaffian_lswo; LL=/scratch/a/A.Otaifi/ansatz_runs/pfaffian_r2g; LS=/scratch/a/A.Otaifi/ansatz_runs/pfaffian_lswo2
P=/scratch/hpc-prf-ppdmrg/hpcalot/chatty_compress6; D=$P/data/lswo2; D1=$P/data/r1; DO=$P/data/o2b_r1
SSHO="ssh -o BatchMode=yes -o HostName=login4"
rem() { $SSHO paderborn "export PATH=/opt/software/slurm/25.11.8-1/bin:\$PATH; $1"; }
TM=03:00:00; FS=9000; MX=3000
CODE="cx_r1.py cx_r1fit.py cx_lib.py rbmpp_lib.py"
B1TH=$L0/run/round_1/theta.npy
mkdir -p "$LL"
for K in 1 2; do [ -e "$LL/run_s$K/job_id.txt" ] && { echo "$LL/run_s$K/job_id.txt exists: already submitted"; exit 1; }; done
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
chk $RUNS/o2b_17431886/round_1/spec_fit.json $DO/round_1_spec_fit.json
chk /project/theorie/a/A.Otaifi/Chatty/runs/rbmpp6/runs/rpT1_pp_17163493/params_final.npz $P/data/rpT1_pp_17163493/params_final.npz
chk /project/theorie/a/A.Otaifi/Chatty/runs/rbmpp6/runs/rpTabPP_17164048/sym_tables_PP.npz $P/data/rpTabPP_17164048/sym_tables_PP.npz
chk /project/theorie/a/A.Otaifi/Chatty/runs/stall6/data/psi0_6x6_table.npz $P/data/psi0_6x6_table.npz
chk /project/theorie/a/A.Otaifi/Chatty/runs/stall6/data/sym_tables.npz $P/data/sym_tables.npz
chk /project/theorie/a/A.Otaifi/Chatty/runs/stall6/csr/norb.npy $P/data/csr/norb.npy
rem "cd $P/data/csr; ls -l meta.json up_indices.npy up_codes.npy up_inc.npy" > /dev/null
chk $B1TH $D/B1_theta.npy
chk $LS/target_TG_sG.npy $D/target_TG_sG.npy
chk $LS/s_G.npy $D/s_G.npy
python3 -c "import numpy as np,sys; a=np.load('$B1TH'); assert a.size==2740 and a.dtype==np.float64 and (a[2592:]==0).all()" || { echo "B_1 theta not 2740 / RBM block not zero"; exit 1; }
for K in 1 2; do
  NAME=r2g_s$K; LOC=$LL/run_s$K; RUN=$P/runs/$NAME; mkdir -p "$LOC"; rem "mkdir -p $RUN"
  cp "$LL/code_sha256.txt" "$LL/code_git_status.txt" "$LOC/"
  TGT=$D/target_TG_sG.npy; SGN=$D/s_G.npy
  # 2. spec = round-2 s_G spec_r1.json except fit_overrides.seed = K (fit_sec 9000, max_steps 3000 as in round 2)
  python3 - "$LOC" "$P" "$D" "$DO" "$FS" "$MX" "$TGT" "$SGN" "$K" <<'PY'
import json, sys
loc, P, D, DO, fs, mx, tgt, sgn, k = sys.argv[1:10]
sha = {l.split()[1]: l.split()[0] for l in open(f'{loc}/code_sha256.txt')}
sp = dict(refresh=f'{P}/data/refresh_17409419', pp_params=f'{P}/data/rpT1_pp_17163493/params_final.npz', pp_table=f'{P}/data/rpTabPP_17164048/sym_tables_PP.npz',
          code_sha256=sha, target_la=tgt, sign_path=sgn, warm_theta=f'{D}/B1_theta.npy',
          o2b_fit_spec=f'{DO}/round_1_spec_fit.json', window=3, freeze_outer=False, arm='rbmpp', rbm_alpha=4, skip_efn=True, skip_fn_readouts=True,
          fit_sec=float(fs), max_steps=int(mx), fit_overrides=dict(seed=int(k)))
json.dump(sp, open(f'{loc}/spec_r1.json', 'w'), indent=1)
PY
  rsync -a -e "$SSHO" "$LOC/spec_r1.json" paderborn:$RUN/
  cat > "$LOC/ANALYSIS_BRIEF.md" <<EOB
# R2G-conv GPU fit, seed $K, Paderborn (remote $RUN, local $LOC), PLAN_pfaffian.md "R2G-conv" (projects/j1j2); sign s_G (s_G.npy), target T_G = B_1 e^{4.0 f_J^G[B_1]} (target_TG_sG.npy), warm B_1 theta
Question: converge round 2 under s_G: B_2 = e^R x complex PP (real symmetric RBM alpha 4, no backflow), warm-started from B_1's theta (2740 values, RBM block 0), flatness < 0.02 rule as R1b, fit seed $K,
max_steps $MX, fit_sec $FS, wall $TM.  Spec identical to the round-2 s_G arm ($LS/run_sG) except fit_overrides.seed = $K.  Race partner: theorie A40 copy ($LL/run_s${K}_a40, fit_sec 43200, wall 13:00:00).
Files: $LOC/cx_r1.json (steps, converged, reason, median_sec_per_step, L_ho_curve, table_sec), round_1/{cx_t1s.json, fit.log, progress.txt, theta.npy}, round_1/la_table.npy (pulled by the watcher).
Gate (fit only; no Monte Carlo, quotable.py does not apply): python3 $HERE/r1_gate.py $LOC
  Physics gate = the CPU readout per PLAN "R2G-conv" (run by the lead): predictions converged bare e(B_2^G) 1.36-1.42e-5; augmented minima 5.2-5.5e-6; seed spread <= 0.15e-6; KILL of candidate (1): bare drop <= 0.5e-6 vs 1.5453e-5 in either seed.
Branches: FIT OK, FIT NOT OK, TIMEOUT, STALLED, FAILED -> next step: report to the angle lead (steps, converged and reason, s/step, wall, GPU-h, la_table path); the CPU readout is run by the lead.
  No resubmission without the lead.
EOB
  rsync -a -e "$SSHO" "$LOC/ANALYSIS_BRIEF.md" paderborn:$RUN/
  echo "staged seed $K: remote $RUN, local $LOC; wall $TM fit_sec $FS max_steps $MX"
done
[ "${1:-}" = "--submit" ] || exit 0
rsync -a -e "$SSHO" "$HERE/r2g.sbatch" paderborn:$P/r2g.sbatch
rem "sha256sum $P/r2g.sbatch" | cut -d' ' -f1 | grep -qx "$(sha256sum "$HERE/r2g.sbatch" | cut -d' ' -f1)" || { echo "r2g.sbatch sha256 differs after copy"; exit 1; }
for K in 1 2; do
  NAME=r2g_s$K; LOC=$LL/run_s$K; RUN=$P/runs/$NAME
  OUTL=$(rem "sbatch --parsable --no-requeue --job-name=$NAME --time=$TM --output=$RUN/slurm-%j.out $P/r2g.sbatch $RUN" 2>/dev/null)
  JOB=${OUTL%%;*}; [[ "$JOB" =~ ^[0-9]+$ ]] || { echo "sbatch failed (seed $K): $OUTL"; exit 1; }
  echo "$JOB" > "$LOC/job_id.txt"; echo "s$K $JOB $(date -Is)" >> "$LL/job_ids.txt"
  R1_REMRUN=$RUN setsid nohup "$HERE/lswo_pdb_watch.sh" "$JOB" "$LOC" > "$LOC/watcher.log" 2>&1 < /dev/null &
  echo "$!" > "$LOC/watcher_pid.txt"
  echo "seed=$K job=$JOB run=$RUN watcher_pid=$(cat $LOC/watcher_pid.txt) ppid=$(ps -o ppid= -p $(cat $LOC/watcher_pid.txt))"
  rem "sacct -j $JOB -X -n -o State,Reason%30" 2>/dev/null
done
rem "squeue --start -u \$USER" 2>/dev/null || true
