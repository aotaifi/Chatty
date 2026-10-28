#!/bin/bash
# Test G launcher (PLAN.md "TEST G") on one Paderborn H100, after launch_c1.sh.  Default: STAGE ONLY.  --submit: stage, sbatch
# (gpu_h100, hpc-prf-ppdmrg, --no-requeue, hard --time), arm the durable watcher (g1_pdb_watch.sh -> job_finish.sh --wake none).
#   MODE=probe  one job: rates + controls (1) (stored c1 table), (2a) HF FD in table space, (3) broken chi*, (2b) theta FD of the
#               sampled loss incl. Re/Im-only, (2c) sampled vs exact gradient, Im-only step; n_probe_steps inner steps.  Seed 1.
#   MODE=probe2 (lead 2026-10-28) optimiser check, <= 0.1 H100-h: stored c1 table, chi_0, 20 inner steps (mixture sampler, line search,
#               B_pairs 12288, B_acc 16384, lam0 0.1), paired MM bound on E_FN[A_20] - E_FN[A_0] from 65536 fresh pairs.  SBATCH_TIME=00:06:00
#   MODE=gate   (lead 2026-10-28; the staged launcher) ONE frozen-chi* block, seed 1, <= 0.75 H100-h: A_0 table (control 1 gate), chi_0,
#               E_FN[chi_0], K_INNER steps, A_1 table, E_FN[A_1]; readout Delta_0, tau_0.  Usage: MODE=gate SEED=1 SBATCH_TIME=00:45:00 K_INNER=k
#   MODE=prod   (superseded by gate, kept) production seed SEED (1|2): warm start c1_cplx_s$SEED theta; full table at step 0 = control (1) gate; MM loop.
# Env: SEED, SBATCH_TIME (HH:MM:SS, required), TIME_SEC (default SBATCH_TIME - 600 s), K_INNER, B_PAIRS (prod sizing from the probe).
# Needs the VPN (paderborn/README.md).  Local /scratch/a/A.Otaifi/ansatz_runs/pfaffian_g/runs/NAME; remote $P/runs/NAME.
# Production needs cx_g1fit.py committed (git HEAD == working copy); the probe records the sha256 only.
set -euo pipefail
MODE=${MODE:?set MODE=probe|prod}
HERE=$(cd "$(dirname "$0")" && pwd)
P=/scratch/hpc-prf-ppdmrg/hpcalot/chatty_compress6; D=$P/data/o2b_r1
SSHO="ssh -o BatchMode=yes -o HostName=login4"
rem() { $SSHO paderborn "export PATH=/opt/software/slurm/25.11.8-1/bin:\$PATH; $1"; }
SEED=${SEED:-1}
TM=${SBATCH_TIME:?set SBATCH_TIME}
IFS=: read -r h m s <<< "$TM"; TS=$((10#$h*3600 + 10#$m*60 + 10#${s:-0})); TSEC=${TIME_SEC:-$((TS-600))}
case $MODE in
  probe) NAME=g1_probe;;
  probe2) NAME=g1_probe2;;
  gate)  NAME=g1_gate_s$SEED;;
  prod)  NAME=g1_s$SEED;;
  *) echo bad MODE; exit 1;;
esac
case $SEED in
  1) REFTAX=1.1067110978066649e-05;;      # pfaffian_d/split/split_s1.json dE_FN_site
  2) REFTAX=9.112477741663524e-06;;       # pfaffian_d/split/split_s2.json dE_FN_site
  *) echo "SEED 1|2"; exit 1;;
esac
LOC=/scratch/a/A.Otaifi/ansatz_runs/pfaffian_g/runs/$NAME; RUN=$P/runs/$NAME
mkdir -p "$LOC"
# 1. code: codeG1 = codeC1 (st6_sector, cx_lib, rbmpp_lib as Test C ran them) + cx_g1fit.py, pinned by sha256
ST6=$HERE/../../experiments/stall_6x6/st6_sector.py
if [ "$MODE" = prod ] || [ "$MODE" = gate ]; then ( cd "$HERE" && git diff --quiet HEAD -- cx_g1fit.py cx_lib.py rbmpp_lib.py && git ls-files --error-unmatch cx_g1fit.py > /dev/null ) || { echo "prod: cx_g1fit.py not committed / differs from git HEAD"; exit 1; }; fi
rem "[ -d $P/codeG1 ] || cp -a $P/codeC1 $P/codeG1; mkdir -p $RUN"
for f in cx_g1fit.py cx_lib.py rbmpp_lib.py; do rsync -a -e "$SSHO" "$HERE/$f" paderborn:$P/codeG1/; done
rsync -a -e "$SSHO" "$ST6" paderborn:$P/codeG1/
( cd "$HERE" && sha256sum cx_g1fit.py cx_lib.py rbmpp_lib.py; cd "$(dirname "$ST6")" && sha256sum st6_sector.py ) > "$LOC/code_sha256.txt"
rem "cd $P/codeG1 && sha256sum cx_g1fit.py cx_lib.py rbmpp_lib.py st6_sector.py" | diff - "$LOC/code_sha256.txt" > /dev/null || { echo "code sha256 differs after copy"; exit 1; }
# 2. inputs on Paderborn: sha256 against the local originals
chk() { l=$(sha256sum "$1" | cut -d' ' -f1); r=$(rem "sha256sum $2 2>/dev/null" | cut -d' ' -f1 || true)
  [ "$l" = "$r" ] || { echo "sha256 mismatch $2"; exit 1; }; echo "input ok ${l:0:12} $2"; }
C1L=/scratch/a/A.Otaifi/ansatz_runs/pfaffian_c/runs/c1_cplx_s$SEED/round_1
chk /project/theorie/a/A.Otaifi/Chatty/runs/compress6/runs/o2b_17431886/round_1/target_la.npy $D/round_1_target_la.npy
chk /project/theorie/a/A.Otaifi/Chatty/runs/compress6/runs/refresh_17409419/s2.npy $P/data/refresh_17409419/s2.npy
chk $C1L/theta.npy $P/runs/c1_cplx_s$SEED/round_1/theta.npy
chk $C1L/la_table.npy $P/runs/c1_cplx_s$SEED/round_1/la_table.npy
# 3. spec
python3 - "$LOC" "$P" "$D" "$TSEC" "$MODE" "$SEED" "$REFTAX" "${K_INNER:-20}" "${B_PAIRS:-8192}" <<'PY'
import json, sys
loc, P, D, ts, mode, seed, reftax, kin, bp = sys.argv[1:10]
sha = {l.split()[1]: l.split()[0] for l in open(f'{loc}/code_sha256.txt')}
sp = dict(mode=mode, seed=int(seed), refresh=f'{P}/data/refresh_17409419', target_la=f'{D}/round_1_target_la.npy',
          warm_theta=f'{P}/runs/c1_cplx_s{seed}/round_1/theta.npy', ref_table=f'{P}/runs/c1_cplx_s{seed}/round_1/la_table.npy',
          ref_tax0=float(reftax), ref_dE_target_split=9.853014655922562e-06, code_sha256=sha, time_sec=float(ts),
          k_inner=int(kin), B_pairs=int(bp), B_acc=int(bp), delta=0.5, lam0=0.01, lam_cap=1.0, z_acc=2.0, max_backtrack=1,
          flat_tol=1e-8, flat_n=3, B_gauge=2048, fn_tol=1e-10, fn_maxit=400)
if mode == 'probe2': sp.update(n_probe_steps=20, B_pairs=12288, B_acc=16384, lam0=0.1, B_mm=65536, line_search=[1.0, 0.5, 0.25, 0.125])
if mode == 'gate': sp.update(B_acc=16384, lam0=0.1, line_search=[1.0, 0.5, 0.25, 0.125])
if mode == 'probe': sp.update(probe_table_n=1 << 20, fd_eps=[1e-3, 1e-4, 1e-2], c2b_pairs=512, c2c_pairs=65536, n_probe_steps=6)
json.dump(sp, open(f'{loc}/spec_g1.json', 'w'), indent=1)
PY
rsync -a -e "$SSHO" "$LOC/spec_g1.json" paderborn:$RUN/
# 4. brief
if [ "$MODE" = probe2 ]; then cat > "$LOC/ANALYSIS_BRIEF.md" <<EOB
# Test G optimiser check $NAME (remote $RUN, local $LOC); job id in $LOC/job_id.txt
Question (lead 2026-10-28): after probe 4049890 (5 of 6 steps raised L), do the fixed inner steps (all-orbit defensive mixture sampler
l/L + |l'|/G, line search 1..1/8 on a paired acceptance batch, lam0 0.1) lower L_chi0?  Acceptance: >= 70% of the 20 steps lower L by
> 2 sigma, and the paired MM bound on E_FN[A_20] - E_FN[A_0] (65536 fresh pairs) is < 0 at > 2 sigma.
No Monte Carlo chains.  Files: $LOC/cx_g1.json {probe2, probe_steps[]}, progress.txt.
Gate:
  python3 $HERE/g1_gate_read.py $LOC/cx_g1.json
Next: report to the worker that submitted; it reports to the lead (no further submission).
EOB
elif [ "$MODE" = probe ]; then cat > "$LOC/ANALYSIS_BRIEF.md" <<EOB
# Test G rate probe $NAME (remote $RUN, local $LOC); job id in $LOC/job_id.txt
Question: PLAN.md "TEST G" build step: s per chi* solve (cold, warm), s per inner step (B_pairs pairs + acceptance batch), s per
full table (from a 2^20-rep subset), s per pair-kernel pass; controls (1) tax_0 vs split_s1 (rel <= 1e-9, stored c1 table),
(2a) HF vs central FD of E_FN in table space (3 directions, rel <= 1e-3 at eps 1e-3), (3) broken chi* must fail (2a),
(2b) theta FD of the sampled loss (Re, Im, mixed; rel <= 1e-3), (2c) sampled vs exact HF gradient (z-scores), Im-only step lowers L.
No Monte Carlo chains (exact solves; quotable.py does not apply).  Files: $LOC/cx_g1.json, progress.txt.
Gate:
  python3 -c "import json;d=json.load(open('$LOC/cx_g1.json'));c=d['controls'];print('done',d['done'],'stop',d.get('stop'),'c1',c['c1_tax0']['ok'],c['c1_tax0']['rel'],'c2',c.get('c2_pass'),'c3',c.get('c3_broken_fails'),'2c z',[round(q['z'],2) for q in c['c2c_sampled_vs_exact']]);print('rates',d['rates'])"
Next: report the rates and control lines to the worker that submitted (it stages the production launcher); no production submission.
EOB
elif [ "$MODE" = gate ]; then cat > "$LOC/ANALYSIS_BRIEF.md" <<EOB
# Test G gate $NAME (remote $RUN, local $LOC), seed $SEED; job id in $LOC/job_id.txt
Question (lead 2026-10-28, after the second opinion cj-34203b7a8e09): one frozen-chi* block from A_0 = c1_cplx_s$SEED.  Per site:
E_FN[A_0], E_FN[chi_0] (chi_0 = FN ground state of H_FN[A_0], used as guide amplitude with sign s2), E_FN[A_1] (after K_INNER=${K_INNER:-20}
sampled steps on L_chi0, all-orbit pair sampling), Delta_0 = E_FN[A_0] - E_FN[chi_0], tau_0 = E_FN[A_1] - E_FN[chi_0].
Controls in the run: c1_tax0 (tax_0 of A_0 vs split_s$SEED, rel <= 1e-9, else exit 2 before any step); MM_decrease (exact P(A_1; chi_0) < P(A_0; chi_0)).
No Monte Carlo chains (exact solves; quotable.py does not apply).  Files: $LOC/cx_g1.json {controls, gate, inner[]}; theta.npy;
la_table.npy (A_1) stays on Paderborn $RUN.
Gate:
  python3 $HERE/g1_gate_read.py $LOC/cx_g1.json
Next: report these numbers to the lead; the lead decides (no further submission by the worker).
EOB
else cat > "$LOC/ANALYSIS_BRIEF.md" <<EOB
# Test G production $NAME (remote $RUN, local $LOC), seed $SEED; job id in $LOC/job_id.txt
Question: PLAN.md "TEST G": does fitting the TRUE E_FN (MM with the HF gradient, chi* re-solved every k_inner=${K_INNER:-20} sampled
GN steps) lower tax_0 of the complex copy c1_cplx_s$SEED (start $REFTAX per site)?  Prediction: falls 20-40% (to 0.55-0.75e-5 mean
over s1, s2), E_FN decreases monotonically; kill of 'the fit's weighting is a lever': drop < 10%.
No Monte Carlo chains (exact solves; quotable.py does not apply).  Files: $LOC/cx_g1.json {controls.c1_tax0, outer[] (dE_FN_site,
tax0, P_old_block/P_new_block, bond_dmax_block, solve.res_abs_final, solve.sign_cert, local_energy), plain_iteration[], summary};
theta_k*.npy; la_table.npy stays on Paderborn $RUN.
Gate:
  python3 -c "import json;d=json.load(open('$LOC/cx_g1.json'));s=d.get('summary',{});print('stop',d.get('stop'),'c1',d['controls']['c1_tax0']['ok'],'reason',s.get('stop_reason'),'tax0',s.get('tax0_start'),'->',s.get('tax0_end'),'drop',s.get('drop_rel'),'monotone',s.get('monotone'),'plain',s.get('plain_E_end'),'rates',s.get('rate_main',{}).get('median'),s.get('rate_plain',{}).get('median'))"
Verdict (needs both seeds; this job is one): report its numbers only; the lead applies PLAN.md TEST G.
EOB
fi
rsync -a -e "$SSHO" "$LOC/ANALYSIS_BRIEF.md" paderborn:$RUN/
echo "staged remote $RUN, local $LOC; mode $MODE seed $SEED time $TM time_sec $TSEC"
[ "${1:-}" = "--submit" ] || exit 0
rsync -a -e "$SSHO" "$HERE/g1.sbatch" paderborn:$P/g1.sbatch
OUTL=$(rem "sbatch --parsable --job-name=g1$MODE$SEED --time=$TM --output=$RUN/slurm-%j.out $P/g1.sbatch $RUN $MODE" 2>/dev/null)
JOB=${OUTL%%;*}; [[ "$JOB" =~ ^[0-9]+$ ]] || { echo "sbatch failed: $OUTL"; exit 1; }
echo "$JOB" > "$LOC/job_id.txt"
C1_REMRUN=$RUN C1_SUBS="." setsid nohup "$HERE/g1_pdb_watch.sh" "$JOB" "$LOC" > "$LOC/watcher.log" 2>&1 < /dev/null &
echo "job=$JOB run=$RUN watcher_pid=$!"
rem "sacct -j $JOB -X -n -o State,Reason%30" 2>/dev/null
