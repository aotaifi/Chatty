#!/bin/bash
# Test O2b launcher (README "De-oracling the G1 guide" O2 + Amendment O2-1): the fitted loop, ONE job on a full A40 (cx.sbatch),
# rounds 0..4 + the closing hop, per-round checkpoints in RUN/round_k (resume: spec key resume_from). Run AFTER the lead's go.
# Amendment O2b-1: round 0 fit 10800 s (one-time initial training), rounds 1-4 fit 2400 s.
# Wall 19:15 = 1.3 x (round 0 ~17270 s + 4 x ~8870 s + controls and closing hop ~410 s), per round from measured rates: 2 hops
# x 110 s (o2a_17430739), fit + one step (71 s, t1s2_ppbf), table 5782 s (t1s2_ppbf, slice), fit-job overhead ~400 s.
# Partition MaxTime cip/inter 2 days (sinfo 2026-10-22).
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd); B=/project/theorie/a/A.Otaifi/Chatty/runs/compress6
RUNS=/project/theorie/a/A.Otaifi/Chatty/runs/compress6/runs
cp "$HERE/cx_lib.py" "$HERE/cx_t1s.py" "$HERE/cx_o2b.py" "$HERE/../rbmpp_6x6/rbmpp_lib.py" "$B/code/"
for f in cx_o2b.py cx_t1s.py cx_lib.py; do cmp -s "$HERE/$f" "$B/code/$f" || { echo "copy of $f differs"; exit 1; }; done
sha256sum "$B/code/cx_o2b.py" "$B/code/cx_t1s.py" "$B/code/cx_lib.py" "$B/code/rbmpp_lib.py"
# code drift guard: cx.sbatch copies $B/code/*.py at job START; cx_o2b.py (and each round's cx_t1s.py) exit 3 on a mismatch
SHAJSON=$(python3 -c "import hashlib,json;print(json.dumps({f:hashlib.sha256(open('$B/code/'+f,'rb').read()).hexdigest() for f in ('cx_o2b.py','cx_t1s.py','cx_lib.py','rbmpp_lib.py')}))")
SPECD=$B/specs_o2b/$(date +%Y%m%dT%H%M%S); mkdir -p "$SPECD"; SP=$SPECD/o2b.json
python3 -c "import json,sys;d=json.load(open('$HERE/specs/o2b.json'));d['code_sha256']=json.loads(sys.argv[1]);json.dump(d,open('$SP','w'))" "$SHAJSON"
cd "$HERE"
OUTL=$(sbatch --parsable --no-requeue --job-name=o2b --time=19:15:00 "$B/cx.sbatch" cx_o2b.py "$SP")
JOB=${OUTL%%;*}; RUN=$RUNS/o2b_$JOB; mkdir -p "$RUN"
cat > "$RUN/ANALYSIS_BRIEF.md" <<EOB
# O2b analysis brief (job $JOB, run $RUN)
Question (README "De-oracling the G1 guide", O2 and Amendment O2-1, projects/j1j2/experiments/compress_6x6/README.md): does the
FITTED loop (hop -> FN solve -> fit PP + 3x3 backflow to log phi_k -> hop) from the oracle-free VMC PP start reach E_FN <= 1e-5
within 4 rounds, and what does the fit cost against the fit-free chain (O2a, o2a_17430739)? Sign s_2 is still the oracle.
Files: $RUN/cx_o2b.json: control {EFN_zero_PP, EFN_g0, EFN_hop_phi0 vs O2a}; rounds[k] {EFN = E_FN(g_k), hop_a {alpha_V, H_V,
  expelled {w_psi0_s2_eq_s0, w_psi0_s2_ne_s0, ...}}, EFN_hop_phi = E_FN(hop(phi_k)), c (on E_FN - 9.15e-7), fit_tax, exact_gain,
  viability_tax_lt_half_gain, fit {converged, steps, tau_hist, err_by_decade {mean, rms, rms_T0, rms_heldout}, E_FN_zero_hop,
  E_FN_splice}}; stop; $RUN/round_k/ (each fit: cx_t1s.json, la_table.npy, theta.npy, target_la.npy, fit.log, progress.txt).
  Code: cx_o2b.py + cx_t1s.py (git, experiments/compress_6x6), spec $SP.
Gate: no Monte Carlo chains (exact solves, one fit per round): quotable.py does not apply. Read the numbers directly:
  python3 -c "import json;d=json.load(open('$RUN/cx_o2b.json'));print(d['control'],d['stop'],[(r['k'],r['EFN'],r.get('c'),r.get('fit_tax')) for r in d['rounds']])"
Readings (pre-registered): PASS E_FN(g_k) <= 1e-5 within 4 rounds; STALL two consecutive c_k > 0.9 while E_FN > 1.5e-5; RISE c_k > 1;
  viability: fit tax < 0.5 x exact gain in each of the first 3 rounds; reading a STALL per Amendment O2-1 item 6.
Next per verdict: the worker reports the numbers to the lead only; the lead decides (projects/j1j2/PLAN.md).
EOB
setsid nohup "$HERE/cx_o2b_watch.sh" "$JOB" "$RUN" > "$RUN/watcher.log" 2>&1 < /dev/null &
echo "job=$JOB run=$RUN watcher_pid=$!"
sacct -j "$JOB" -X -n -o State,Reason%30
