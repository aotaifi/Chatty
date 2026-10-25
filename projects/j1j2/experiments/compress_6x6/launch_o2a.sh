#!/bin/bash
# Test O2a launcher (A40 slice via cx_slice.sbatch), modelled on launch_g1.sh. ~13 solves/round x 5 rounds (measured ~85 s per solve in O1), 2 h wall.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd); B=/project/theorie/a/A.Otaifi/Chatty/runs/compress6; WALL=${WALL:-02:00:00}
RUNS=/project/theorie/a/A.Otaifi/Chatty/runs/compress6/runs
cp "$HERE/cx_lib.py" "$HERE/cx_o2a.py" "$B/code/"
sha256sum "$B/code/cx_o2a.py" "$B/code/cx_lib.py"
for f in cx_o2a.py cx_lib.py; do cmp -s "$HERE/$f" "$B/code/$f" || { echo "copy of $f differs"; exit 1; }; done
SHAJSON=$(python3 -c "import hashlib,json;print(json.dumps({f:hashlib.sha256(open('$B/code/'+f,'rb').read()).hexdigest() for f in ('cx_o2a.py','cx_lib.py')}))")
cd "$HERE"
python3 - <<PY
import json,os
d=json.load(open('specs/o2a.json'))
for p in [d['pp'],d['refresh']+'/s2.npy']: assert os.path.exists(p),p
PY
SPECD=$B/specs_o2a/$(date +%Y%m%dT%H%M%S); mkdir -p "$SPECD"; SP=$SPECD/o2a.json
python3 -c "import json,sys;d=json.load(open('specs/o2a.json'));d['code_sha256']=json.loads(sys.argv[1]);json.dump(d,open('$SP','w'),indent=1)" "$SHAJSON"
OUTL=$(sbatch --parsable --no-requeue --job-name=o2a --time="$WALL" "$B/cx_slice.sbatch" cx_o2a.py "$SP")
JOB=${OUTL%%;*}; RUN=$RUNS/o2a_$JOB; mkdir -p "$RUN"
cat > "$RUN/ANALYSIS_BRIEF.md" <<EOB
# O2a analysis brief (job $JOB, run $RUN)
Question: the fit-free reference chain (README "De-oracling the G1 guide" O2 + Amendment O2-1 item 1, pre-registered 2026-10-22, not edited here).
Round k = 0..4: g_k = |exp(la_k)||1 + alpha_V (E_L - E)| (E_L of (|exp(la_k)|, s_2), alpha_V = O1 rule: argmin of exact <H> of the one-hop state);
E_FN(g_k) and phi_k = FN solution of g_k (sec.fn_solve, s_2); la_{k+1} = log phi_k normalised. Start la_0 = VMC-trained PP table (sym_tables_PP.npz lP), sign s_2 (refresh_17409419).
A sixth evaluation (g_5) closes c_4.
Amendment O2-1: item 1 (O2a first: the fit-free chain phi_{k+1} = FN(hop(phi_k)), same VMC-PP start, same alpha rule, no training, 4 rounds, ~1.5 A40-h, measures the rate of the map);
 item 3 (contraction c_k defined on E_FN - 9.15e-7); item 4 (new verdict RISE: c_k > 1 in any round -> stop);
 item 6 (reading a STALL: O2a rate > 0.9 = the map is too slow; fit tax >= exact gain = representation floor; small tax but rising s_2-right expelled weight = spurious-attractor mechanism of self-healing DMC, arXiv:2609.01301).
 Reference: the exact iteration contracts by 0.6-0.65 per solve from a fitted ViT (9.2 -> 4.3 -> 2.5 -> 1.7e-5). Fit-free chain from 2.37e-5 estimated marginal at 4 rounds for the 1e-5 bar (review estimate, not measured).
Files: $RUN/cx_o2a.json (control{EFN_zero_PP, ref 5.3677e-5, ok}; rounds[k]{alpha_V, H_V, EFN, c, expelled{count_frac_config, count_frac_rep, w_psi0, w_psi0_s2_eq_s0, w_psi0_s2_ne_s0}, at_bound, n_floored}; stop; code_sha256.match);
  $RUN/progress.txt; $RUN/la_5.npy. Spec $SP. Code cx_o2a.py. Energies per site relative to E0.
Control: zero-hop E_FN of the PP table must reproduce 5.3677e-5 (T0); otherwise exit 2: report and stop.
Gate: exact solves, no Monte Carlo chains: quotable.py does not apply. Read numbers directly:
  python3 -c "import json;d=json.load(open('$RUN/cx_o2a.json'));print(d['control'],d.get('stop'));[print(r['k'],r['alpha_V'],r['H_V'],r['EFN'],r.get('c'),r.get('expelled')) for r in d['rounds']]"
Next: report control and the per-round table to the lead only; no interpretation against the readings (lead decides). If a round shows c_k > 1 the job stops (RISE): report that.
EOB
setsid nohup "$HERE/cx_o2a_watch.sh" "$JOB" "$RUN" > "$RUN/watcher.log" 2>&1 < /dev/null &
echo "job=$JOB run=$RUN watcher_pid=$!"; sacct -j "$JOB" -X -n -o State,Reason%30
