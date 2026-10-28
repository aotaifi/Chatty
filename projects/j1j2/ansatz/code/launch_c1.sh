#!/bin/bash
# Test C launcher (PLAN_pfaffian.md "Test C") on one Paderborn H100.  Default: STAGE ONLY.  --submit: stage, sbatch (gpu_h100, hpc-prf-ppdmrg, --no-requeue), arm the durable watcher.
#   MODE=probe  one job, both arms in sequence, rate_only, max_steps 3, production B 8208, seed 1, complex im_sigma 0.01  (dirs RUN/real, RUN/cplx)
#   MODE=real   production real arm  (no --complex, warm F1 A theta, --seed SEED): full cx_c1.py (fit to flatness, table, hops, E_FN, t_real.npy)
#   MODE=cplx   production complex arm (--complex --im_sigma 0.01 --seed SEED, max_steps MAX_STEPS default 600)
# Env: SEED (real/cplx, 1..3), SBATCH_TIME (required with --submit), FIT_SEC (real/cplx: default = SBATCH_TIME - 25 min in s), MAX_STEPS (cplx 600; real unset).
# Needs the VPN (paderborn/README.md).  Local run dir /scratch/a/A.Otaifi/ansatz_runs/pfaffian_c/runs/NAME; remote /scratch/hpc-prf-ppdmrg/hpcalot/chatty_compress6/runs/NAME.
set -euo pipefail
MODE=${MODE:?set MODE=probe|real|cplx}
HERE=$(cd "$(dirname "$0")" && pwd)
RUNS=/project/theorie/a/A.Otaifi/Chatty/runs/compress6/runs
O2B=$RUNS/o2b_17431886; F1A=$RUNS/f1_A
P=/scratch/hpc-prf-ppdmrg/hpcalot/chatty_compress6; D=$P/data/o2b_r1
SSHO="ssh -o BatchMode=yes -o HostName=login4"
rem() { $SSHO paderborn "export PATH=/opt/software/slurm/25.11.8-1/bin:\$PATH; $1"; }
SEED=${SEED:-1}
case $MODE in
  probe) NAME=c1_probe; TM=${SBATCH_TIME:-00:45:00}; SUBS="real cplx"; ITEMS=("real:--seed~1" "cplx:--complex~--im_sigma~0.01~--seed~1");;
  real)  NAME=c1_real_s$SEED; TM=${SBATCH_TIME:-}; SUBS="."; ITEMS=(".:--seed~$SEED");;
  cplx)  NAME=c1_cplx_s$SEED; TM=${SBATCH_TIME:-}; SUBS="."; ITEMS=(".:--complex~--im_sigma~0.01~--seed~$SEED");;
  *) echo bad MODE; exit 1;;
esac
LOC=/scratch/a/A.Otaifi/ansatz_runs/pfaffian_c/runs/$NAME; RUN=$P/runs/$NAME
if [ "$MODE" != probe ]; then
  [ -n "$TM" ] || { echo "set SBATCH_TIME (HH:MM:SS)"; exit 1; }
  IFS=: read -r h m s <<< "$TM"; TS=$((10#$h*3600 + 10#$m*60 + 10#${s:-0})); FS=${FIT_SEC:-$((TS-1500))}
else FS=3000; fi
mkdir -p "$LOC"
# 1. code: committed files (git HEAD must equal the working copies), pinned by sha256
( cd "$HERE" && git diff --quiet HEAD -- cx_c1.py cx_c1fit.py cx_lib.py rbmpp_lib.py ) || { echo "code differs from git HEAD"; exit 1; }
rem "[ -d $P/codeC1 ] || cp -a $P/codeF1 $P/codeC1; mkdir -p $D $RUN"
for f in cx_c1.py cx_c1fit.py cx_lib.py rbmpp_lib.py; do rsync -a -e "$SSHO" "$HERE/$f" paderborn:$P/codeC1/; done
( cd "$HERE" && sha256sum cx_c1.py cx_c1fit.py cx_lib.py rbmpp_lib.py ) > "$LOC/code_sha256.txt"
rem "cd $P/codeC1 && sha256sum cx_c1.py cx_c1fit.py cx_lib.py rbmpp_lib.py" | diff - "$LOC/code_sha256.txt" > /dev/null || { echo "code sha256 differs after copy"; exit 1; }
# 2. inputs: reuse what is on Paderborn if the sha256 equals the local original; else copy
chk() { # LOCALFILE REMOTEFILE
  l=$(sha256sum "$1" | cut -d' ' -f1); r=$(rem "sha256sum $2 2>/dev/null" | cut -d' ' -f1 || true)
  if [ "$l" != "$r" ]; then rsync -a -e "$SSHO" "$1" paderborn:$2; r=$(rem "sha256sum $2" | cut -d' ' -f1); fi
  [ "$l" = "$r" ] || { echo "sha256 mismatch $2"; exit 1; }; echo "input ok ${l:0:12} $2"; }
chk $O2B/round_1/target_la.npy $D/round_1_target_la.npy
chk $O2B/round_1/spec_fit.json $D/round_1_spec_fit.json
chk $RUNS/refresh_17409419/s2.npy $P/data/refresh_17409419/s2.npy
chk /project/theorie/a/A.Otaifi/Chatty/runs/rbmpp6/runs/rpT1_pp_17163493/params_final.npz $P/data/rpT1_pp_17163493/params_final.npz
chk /project/theorie/a/A.Otaifi/Chatty/runs/rbmpp6/runs/rpTabPP_17164048/sym_tables_PP.npz $P/data/rpTabPP_17164048/sym_tables_PP.npz
chk /project/theorie/a/A.Otaifi/Chatty/runs/stall6/data/psi0_6x6_table.npz $P/data/psi0_6x6_table.npz
chk /project/theorie/a/A.Otaifi/Chatty/runs/stall6/data/sym_tables.npz $P/data/sym_tables.npz
chk /project/theorie/a/A.Otaifi/Chatty/runs/stall6/csr/norb.npy $P/data/csr/norb.npy
chk $F1A/round_1/theta.npy $D/f1A_theta.npy
rem "cd $P/data/csr; ls -l meta.json up_indices.npy up_codes.npy up_inc.npy" > /dev/null
# 3. specs (one per dir): production F1 A values come from the O2b round-1 spec_fit.json (B 8208, K 2, tau 0.05 -> 0.01, ...) exactly as F1 A (spec_f1.json: window 3, freeze_outer false)
python3 - "$LOC" "$P" "$D" "$FS" "$MODE" "${MAX_STEPS:-}" "$SUBS" <<'PY'
import json, sys, os
loc, P, D, fs, mode, mx, subs = sys.argv[1:8]
sha = {l.split()[1]: l.split()[0] for l in open(f'{loc}/code_sha256.txt')}
sp = dict(refresh=f'{P}/data/refresh_17409419', pp_params=f'{P}/data/rpT1_pp_17163493/params_final.npz', pp_table=f'{P}/data/rpTabPP_17164048/sym_tables_PP.npz',
          code_sha256=sha, target_la=f'{D}/round_1_target_la.npy', warm_theta=f'{D}/f1A_theta.npy', o2b_fit_spec=f'{D}/round_1_spec_fit.json',
          window=3, freeze_outer=False, fit_sec=float(fs))
if mode == 'probe': sp.update(rate_only=True, max_steps=3)
elif mx: sp['max_steps'] = int(mx)
elif mode == 'cplx': sp['max_steps'] = 600
for s in subs.split():
    d = loc if s == '.' else f'{loc}/{s}'; os.makedirs(d, exist_ok=True); json.dump(sp, open(f'{d}/spec_c1.json', 'w'), indent=1)
PY
for s in $SUBS; do [ "$s" = . ] && sd= || sd=/$s; rem "mkdir -p $RUN$sd"; rsync -a -e "$SSHO" "$LOC${sd}/spec_c1.json" paderborn:$RUN$sd/; done
# 4. brief
if [ "$MODE" = probe ]; then cat > "$LOC/ANALYSIS_BRIEF.md" <<EOB
# Test C rate probe $NAME (remote $RUN, local $LOC)
Question: s/step of the real and complex arm at the production batch (B 8208, K 2) on one H100; ratio complex/real. rate_only, max_steps 3, warm theta F1 A, complex im_sigma 0.01 seed 1.
Files: $LOC/{real,cplx}/cx_c1.json (key rate: median_sec_per_step, median_excl_first), {real,cplx}/round_1/fit.log. Gate: no Monte Carlo (timing); read the numbers:
  python3 -c "import json;[print(a,json.load(open('$LOC/%s/cx_c1.json'%a))['rate']) for a in ('real','cplx')]"
Next: ratio > 6 or six production runs do not fit in 10 H100-h minus this probe: report the numbers, write no production launcher. Else launch_c1.sh MODE=real|cplx.
EOB
else cat > "$LOC/ANALYSIS_BRIEF.md" <<EOB
# Test C production run $NAME (remote $RUN, local $LOC), seed $SEED
Question: PLAN_pfaffian.md section "Test C" (projects/j1j2): does complex PP (|A| = 0 needs Re A = Im A = 0) cut the copy's error vs the real PP, specifically on the bonds the real node crosses?
Run: cx_c1.py ($MODE arm; warm F1 A theta f1_A/round_1/theta.npy; F1 A fit spec; fit_sec $FS, max_steps ${MAX_STEPS:-none/600 for cplx}); full driver: fit to F1's flatness criterion, table, hops, E_FN.
Files: $LOC/cx_c1.json: fit {steps, converged, reason, E_FN_zero_hop, median_sec_per_step, L_ho_curve}, tax_0 (= E_FN zero hop - 9.85e-6), tax (one hop, F1 definition), hop_a2, hop_phi1, control, stop (only on failure);
  round_1/{cx_t1s.json, fit.log, theta.npy}; la_table.npy and (real arm) t_real.npy stay on Paderborn $RUN/round_1 (pull them for c1_mech.py).
Gate: no Monte Carlo (exact solves): quotable.py does not apply; read the numbers directly:
  python3 -c "import json;d=json.load(open('$LOC/cx_c1.json'));print(d.get('stop'),d['fit']['steps'],d['fit']['converged'],d['fit']['reason'][:150],d['tax_0'],d['tax'])"
Verdict rule: PLAN_pfaffian.md "Test C" (Pass: complex tax_0 <= 0.75 x real AND cut_X >= 2 cut_R; Kill: >= 0.95 x real OR cut_X <= cut_R; seed scatter std of tax_0 ratio <= 0.05). It needs all six runs (real and complex, seeds 1-3); this job is one of them: report its numbers only.
Mechanism readout (after all six): ansatz/code/c1_mech.py OUT.json --sets <real seed 1 dir (la_table.npy, t_real.npy)> --real <real dirs> --cplx <complex dirs> -> C(S_x), C(S_rest), cut_X, cut_R.
The worker reports the numbers to the lead only; the lead decides.
EOB
fi
rem "true"; rsync -a -e "$SSHO" "$LOC/ANALYSIS_BRIEF.md" paderborn:$RUN/
echo "staged remote $RUN, local $LOC; mode $MODE time ${TM:-unset} fit_sec $FS"
[ "${1:-}" = "--submit" ] || exit 0
[ -n "$TM" ] || { echo "SBATCH_TIME required"; exit 1; }
rsync -a -e "$SSHO" "$HERE/c1.sbatch" paderborn:$P/c1.sbatch
OUTL=$(rem "sbatch --parsable --job-name=c1$MODE$SEED --time=$TM --output=$RUN/slurm-%j.out $P/c1.sbatch $RUN ${ITEMS[*]@Q}" 2>/dev/null)
JOB=${OUTL%%;*}; [[ "$JOB" =~ ^[0-9]+$ ]] || { echo "sbatch failed: $OUTL"; exit 1; }
echo "$JOB" > "$LOC/job_id.txt"
C1_REMRUN=$RUN C1_SUBS="$SUBS" setsid nohup "$HERE/c1_pdb_watch.sh" "$JOB" "$LOC" > "$LOC/watcher.log" 2>&1 < /dev/null &
echo "job=$JOB run=$RUN watcher_pid=$!"
rem "sacct -j $JOB -X -n -o State,Reason%30" 2>/dev/null
