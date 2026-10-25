#!/bin/bash
# Test T1s-2 launcher (README "Test T1s-2"): 3 jobs, one per arm, each warm from its own T1s theta.npy; run AFTER the lead's go.
#   B = B_acc = 8208 (912 per decade), tau 0.05 -> 0.01 anneal, splice E_FN.  Fit budgets: amendment T1s2-1.
#   Walls: (fit budget + measured T1s overhead) x ~1.3; partition MaxTime cip/inter 2 days (sinfo, 2026-10-21).
#   t1s2_pp   (b) PP          fit <= 3600 s,  A40 24 GB slice (cx_slice.sbatch), wall 2:20
#   t1s2_ppbf (c) PP+backflow fit <= 10800 s, A40 24 GB slice,                    wall 6:15
#   t1s2_vit  (a) ViT         fit <= 9000 s,  FULL A40 (cx.sbatch: G alone is 24624 x 154980 fp32 = 15.3 GB, the slice has 19.15 GB), wall 4:10
# Stages code like launch_g0.sh (cx_slice.sbatch copies $B/code/*.py into the run), submits, writes ANALYSIS_BRIEF.md,
# arms one durable setsid watcher per job (cx_t1s_watch.sh -> job_finish.sh --wake none).
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd); B=/project/theorie/a/A.Otaifi/Chatty/runs/compress6
RUNS=/project/theorie/a/A.Otaifi/Chatty/runs/compress6/runs
cp "$HERE/cx_lib.py" "$HERE/cx_t1s.py" "$HERE/../rbmpp_6x6/rbmpp_lib.py" "$B/code/"
sha256sum "$B/code/cx_t1s.py" "$B/code/cx_lib.py" "$B/code/rbmpp_lib.py"
# code drift guard: cx_slice.sbatch copies $B/code/*.py at job START; the job compares these hashes and exits 3 on a mismatch
SHAJSON=$(python3 -c "import hashlib,json;print(json.dumps({f:hashlib.sha256(open('$B/code/'+f,'rb').read()).hexdigest() for f in ('cx_t1s.py','cx_lib.py','rbmpp_lib.py')}))")
for f in cx_t1s.py cx_lib.py; do cmp -s "$HERE/$f" "$B/code/$f" || { echo "copy of $f differs"; exit 1; }; done
SPECD=$B/specs_t1s2/$(date +%Y%m%dT%H%M%S); mkdir -p "$SPECD"
cd "$HERE"
for spec in specs/t1s2_vit.json specs/t1s2_pp.json specs/t1s2_ppbf.json; do [ -f "$spec" ] || { echo "missing $spec"; exit 1; }; done
submit() {   # name spec wall sbatch-file
  local OUTL JOB RUN SP
  SP=$SPECD/$(basename "$2")
  python3 -c "import json,sys;d=json.load(open('$2'));d['code_sha256']=json.loads(sys.argv[1]);json.dump(d,open('$SP','w'))" "$SHAJSON"
  OUTL=$(sbatch --parsable --no-requeue --job-name="$1" --time="$3" "$B/$4" cx_t1s.py "$SP")
  JOB=${OUTL%%;*}; RUN=$RUNS/${1}_$JOB; mkdir -p "$RUN"
  ARM=$(python3 -c "import json;print(json.load(open('$2'))['arm'])")
  cat > "$RUN/ANALYSIS_BRIEF.md" <<EOB
# T1s-2 analysis brief (job $JOB, arm $ARM, run $RUN)
Question (README "Test T1s-2", projects/j1j2/experiments/compress_6x6/README.md, pre-registered 2026-10-21): is the T1s plateau
a sampling noise floor? Arm '$ARM' warm from its own T1s endpoint, B = B_acc = 8208, tau annealed 0.05 -> 0.01.
Files: $RUN/cx_t1s.json (keys: readout.primary_table = count-weighted rms log error per decade -5..-13; readout.rms_by_decade
  (rms_count, rms_count_heldout, rms_T0, eps_T0); converged {converged, stalled, reason, acc_window, lam_below_cap, heldout_rel_change,
  median_move_accepted_window} (README amendment T1s-1); code_sha256 {expected, running, match};
  fit.hist (per step: ok, verdict, lam_rel, dmean, dse, trust_scale), fit.curve (held-out loss and per-decade rms every 10 steps);
  E_FN (one sec.fn_solve with s_2 and the fitted table); control2 (readout code on the warm table vs cx_t0.json eps; warm net vs
  table); control1 (vit_nostrat only: profile next to the refresh fit's)); $RUN/progress.txt; $RUN/la_table.npy; $RUN/theta.npy.
  Code: cx_t1s.py (git, experiments/compress_6x6), spec $SP (= $2 + code_sha256; copied to $RUN/spec.json at start).
Gate: no Monte Carlo chains (exact tables, one fit): quotable.py does not apply. The numbers are read directly from cx_t1s.json:
  python3 -c "import json;d=json.load(open('$RUN/cx_t1s.json'));print(d['readout']['primary_table'],d['converged'],d.get('E_FN'),d['control2'].get('rel_diff_eps'))"
Readings (pre-registered, per arm, r = decade -11 rms vs its T1s value): noise floor if r falls >= 1.5x; converged limit if r
  falls < 1.2x and the arm converged (T1s-1 rule + tau reached 0.01); otherwise inconclusive. Pass: E_FN <= 1e-5. Also E_FN_splice
  (fitted >= -9, exact phi* below). New keys: E_FN_splice, fit.tau_hist, converged.tau_at_floor.
Next per verdict: the worker reports the numbers to the lead only (no interpretation against the decision rule); the lead decides
  (projects/j1j2/PLAN.md).
EOB
  setsid nohup "$HERE/cx_t1s_watch.sh" "$JOB" "$RUN" > "$RUN/watcher.log" 2>&1 < /dev/null &
  echo "job=$JOB arm=$ARM run=$RUN watcher_pid=$!"
  sacct -j "$JOB" -X -n -o State,Reason%30
}
submit t1s2_pp   specs/t1s2_pp.json   02:20:00 cx_slice.sbatch
submit t1s2_ppbf specs/t1s2_ppbf.json 06:15:00 cx_slice.sbatch
submit t1s2_vit  specs/t1s2_vit.json  04:10:00 cx.sbatch
