#!/bin/bash
# R1 launcher (PLAN_pfaffian.md "R1"; built on launch_c1.sh) on one Paderborn H100.  Default: STAGE ONLY.  --submit: stage, sbatch (gpu_h100,
# hpc-prf-ppdmrg, --no-requeue, wall 01:30:00), record the job id, arm the durable watcher (setsid, r1_pdb_watch.sh -> job_finish.sh --wake none).
# Fit B_1 = e^R x complex PP (rbmpp, alpha 4, no backflow) to phi'' = FN(A_s1, s1); sign s1 everywhere (sign_path); fit_sec 4200, max_steps 1100;
# no E_FN / hops on the GPU (skip_fn_readouts, skip_efn).  Needs the VPN.  Local /scratch/a/A.Otaifi/ansatz_runs/pfaffian_r1/run,
# remote /scratch/hpc-prf-ppdmrg/hpcalot/chatty_compress6/runs/r1.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
RUNS=/project/theorie/a/A.Otaifi/Chatty/runs/compress6/runs
R1L=/scratch/a/A.Otaifi/ansatz_runs/pfaffian_r1
P=/scratch/hpc-prf-ppdmrg/hpcalot/chatty_compress6; D=$P/data/r1; DO=$P/data/o2b_r1
SSHO="ssh -o BatchMode=yes -o HostName=login4"
rem() { $SSHO paderborn "export PATH=/opt/software/slurm/25.11.8-1/bin:\$PATH; $1"; }
NAME=r1; TM=01:30:00; FS=4200; MX=1100
LOC=$R1L/run; RUN=$P/runs/$NAME
CODE="cx_r1.py cx_r1fit.py cx_lib.py rbmpp_lib.py"
[ -e "$LOC/job_id.txt" ] && { echo "$LOC/job_id.txt exists: already submitted"; exit 1; }
mkdir -p "$LOC"
# 1. code pinned by sha256 (git state recorded; cx_r1*.py are new files)
( cd "$HERE" && git status --short -- $CODE ) > "$LOC/code_git_status.txt" || true
rem "mkdir -p $P/codeR1 $D $RUN"
for f in $CODE; do rsync -a -e "$SSHO" "$HERE/$f" paderborn:$P/codeR1/; done
( cd "$HERE" && sha256sum $CODE ) > "$LOC/code_sha256.txt"
rem "cd $P/codeR1 && sha256sum $CODE" | diff - "$LOC/code_sha256.txt" > /dev/null || { echo "code sha256 differs after copy"; exit 1; }
# 2. inputs, checked by sha256 (copied when absent or different)
chk() { # LOCALFILE REMOTEFILE
  l=$(sha256sum "$1" | cut -d' ' -f1); r=$(rem "sha256sum $2 2>/dev/null" | cut -d' ' -f1 || true)
  if [ "$l" != "$r" ]; then rsync -a -e "$SSHO" "$1" paderborn:$2; r=$(rem "sha256sum $2" | cut -d' ' -f1); fi
  [ "$l" = "$r" ] || { echo "sha256 mismatch $2"; exit 1; }; echo "input ok ${l:0:12} $2"; }
chk $R1L/target_la_s1.npy $D/target_la_s1.npy
chk /scratch/a/A.Otaifi/ansatz_runs/pfaffian_z/s1.npy $D/s1.npy
chk /scratch/a/A.Otaifi/ansatz_runs/pfaffian_c/runs/c1_cplx_s1/round_1/theta.npy $D/c1_cplx_s1_theta.npy
chk $RUNS/o2b_17431886/round_1/spec_fit.json $DO/round_1_spec_fit.json
chk /project/theorie/a/A.Otaifi/Chatty/runs/rbmpp6/runs/rpT1_pp_17163493/params_final.npz $P/data/rpT1_pp_17163493/params_final.npz
chk /project/theorie/a/A.Otaifi/Chatty/runs/rbmpp6/runs/rpTabPP_17164048/sym_tables_PP.npz $P/data/rpTabPP_17164048/sym_tables_PP.npz
chk /project/theorie/a/A.Otaifi/Chatty/runs/stall6/data/psi0_6x6_table.npz $P/data/psi0_6x6_table.npz
chk /project/theorie/a/A.Otaifi/Chatty/runs/stall6/data/sym_tables.npz $P/data/sym_tables.npz
chk /project/theorie/a/A.Otaifi/Chatty/runs/stall6/csr/norb.npy $P/data/csr/norb.npy
rem "cd $P/data/csr; ls -l meta.json up_indices.npy up_codes.npy up_inc.npy" > /dev/null
# 3. spec (fit knobs = the O2b round-1 spec_fit.json as test C ran them: B 8208, K 2, tau 0.05 -> 0.01, flat_tol 0.02, ...; seed 1)
python3 - "$LOC" "$P" "$D" "$DO" "$FS" "$MX" <<'PY'
import json, sys
loc, P, D, DO, fs, mx = sys.argv[1:7]
sha = {l.split()[1]: l.split()[0] for l in open(f'{loc}/code_sha256.txt')}
sp = dict(refresh=f'{P}/data/refresh_17409419', pp_params=f'{P}/data/rpT1_pp_17163493/params_final.npz', pp_table=f'{P}/data/rpTabPP_17164048/sym_tables_PP.npz',
          code_sha256=sha, target_la=f'{D}/target_la_s1.npy', sign_path=f'{D}/s1.npy', warm_theta=f'{D}/c1_cplx_s1_theta.npy',
          o2b_fit_spec=f'{DO}/round_1_spec_fit.json', window=3, freeze_outer=False, arm='rbmpp', rbm_alpha=4, skip_efn=True, skip_fn_readouts=True,
          fit_sec=float(fs), max_steps=int(mx), fit_overrides=dict(seed=1))
json.dump(sp, open(f'{loc}/spec_r1.json', 'w'), indent=1)
PY
rsync -a -e "$SSHO" "$LOC/spec_r1.json" paderborn:$RUN/
# 4. brief
cat > "$LOC/ANALYSIS_BRIEF.md" <<EOB
# R1 GPU fit (remote $RUN, local $LOC), PLAN_pfaffian.md "R1" (projects/j1j2), sign s1 throughout
Question: fit B_1 = e^R x complex PP (real symmetric RBM alpha 4, no backflow; warm f = complex copy s1's f, RBM 0) to phi'' = FN(A_s1, s1)
(target_la_s1.npy, E_FN[phi''] = 1.608e-5); fit_sec $FS, max_steps $MX; test-C fit pipeline (GN/minSR, flatness rule). No E_FN on the GPU.
Files: $LOC/cx_r1.json (fit: steps, converged, reason, median_sec_per_step, L_ho_curve, table_sec; err_by_decade; bond_error_total at s1),
  round_1/{cx_t1s.json, fit.log, progress.txt, theta.npy}, round_1/la_table.npy (pulled by the watcher at the end).
Gate (fit only; no Monte Carlo, quotable.py does not apply): python3 $HERE/r1_gate.py $LOC
Branches: FIT OK -> next step: report to the angle lead; the physics gate is the CPU readout at s1 in PLAN "R1" (E_FN[B_1] bare; c scan
  A_1' = B_1 e^{c f_J[B_1]}, c 0..6 refined to 0.1; zero-error reference phi'' e^{c f_J[phi'']}; tau/Delta <= 0.7 releases R2/R2c; kill of the
  class E_FN[B_1] > 4e-5 or tau_ref/Delta > 1). FIT NOT OK, TIMEOUT, STALLED, FAILED -> report to the angle lead; no resubmission without the lead.
The worker reports numbers to the lead only; the lead decides.
EOB
rsync -a -e "$SSHO" "$LOC/ANALYSIS_BRIEF.md" paderborn:$RUN/
echo "staged remote $RUN, local $LOC; wall $TM fit_sec $FS max_steps $MX"
[ "${1:-}" = "--submit" ] || exit 0
rsync -a -e "$SSHO" "$HERE/r1.sbatch" paderborn:$P/r1.sbatch
rem "sha256sum $P/r1.sbatch" | cut -d' ' -f1 | grep -qx "$(sha256sum "$HERE/r1.sbatch" | cut -d' ' -f1)" || { echo "r1.sbatch sha256 differs after copy"; exit 1; }
OUTL=$(rem "sbatch --parsable --no-requeue --time=$TM --output=$RUN/slurm-%j.out $P/r1.sbatch $RUN" 2>/dev/null)
JOB=${OUTL%%;*}; [[ "$JOB" =~ ^[0-9]+$ ]] || { echo "sbatch failed: $OUTL"; exit 1; }
echo "$JOB" > "$LOC/job_id.txt"; echo "$JOB" > "$R1L/job_id.txt"
R1_REMRUN=$RUN setsid nohup "$HERE/r1_pdb_watch.sh" "$JOB" "$LOC" > "$LOC/watcher.log" 2>&1 < /dev/null &
WPID=$!; echo "$WPID" > "$LOC/watcher_pid.txt"
echo "job=$JOB run=$RUN watcher_pid=$WPID"
rem "sacct -j $JOB -X -n -o State,Reason%30" 2>/dev/null
