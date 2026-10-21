#!/bin/bash
# Test T1s launcher (README "Test T1s"): 4 jobs on A40 vGPU slices (cx_slice.sbatch), one per arm, run AFTER the lead's go.
#   t1s_vit  (a) ViT warm psi_P          fit <= 1800 s, wall 1:45
#   t1s_vitns control 1: (a) with decade weights off, fit <= 900 s (the refresh fit's budget), wall 1:15
#   t1s_pp   (b) PP f-only warm          fit <= 1800 s, wall 2:00
#   t1s_ppbf (c) PP + 3x3 backflow       fit <= 1800 s, wall 2:15
# Stages code like launch_g0.sh (cx_slice.sbatch copies $B/code/*.py into the run), submits, writes ANALYSIS_BRIEF.md,
# arms one durable setsid watcher per job (cx_t1s_watch.sh -> job_finish.sh --wake none).
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd); B=/home/a/A.Otaifi/chatty_compress6
RUNS=/project/theorie/a/A.Otaifi/chatty_compress6/runs
cp "$HERE/cx_lib.py" "$HERE/cx_t1s.py" "$HERE/../rbmpp_6x6/rbmpp_lib.py" "$B/code/"
sha256sum "$B/code/cx_t1s.py" "$B/code/cx_lib.py" "$B/code/rbmpp_lib.py"
cd "$HERE"
for spec in specs/t1s_vit.json specs/t1s_vitns.json specs/t1s_pp.json specs/t1s_ppbf.json; do [ -f "$spec" ] || { echo "missing $spec"; exit 1; }; done
submit() {   # name spec wall
  local OUTL JOB RUN
  OUTL=$(sbatch --parsable --no-requeue --job-name="$1" --time="$3" "$B/cx_slice.sbatch" cx_t1s.py "$2")
  JOB=${OUTL%%;*}; RUN=$RUNS/${1}_$JOB; mkdir -p "$RUN"
  ARM=$(python3 -c "import json;print(json.load(open('$2'))['arm'])")
  cat > "$RUN/ANALYSIS_BRIEF.md" <<EOB
# T1s analysis brief (job $JOB, arm $ARM, run $RUN)
Question (README "Test T1s", projects/j1j2/experiments/compress_6x6/README.md, pre-registered 2026-10-21): is the tail error of
the fit to log phi*(s_2) a representability limit or the training objective? Asked per arm; this run is arm '$ARM'
(vit = (a) ViT warm psi_P; vit_nostrat = control 1, (a) with the decade weights off; pp = (b) PP f-only; ppbf = (c) PP + 3x3 backflow).
Files: $RUN/cx_t1s.json (keys: readout.primary_table = count-weighted rms log error per decade -5..-13; readout.rms_by_decade
  (rms_count, rms_count_heldout, rms_T0, eps_T0); converged {converged, reason, acc_window, lam_below_cap, heldout_rel_change};
  fit.hist (per step: ok, verdict, lam_rel, dmean, dse, trust_scale), fit.curve (held-out loss and per-decade rms every 10 steps);
  E_FN (one sec.fn_solve with s_2 and the fitted table); control2 (readout code on the warm table vs cx_t0.json eps; warm net vs
  table); control1 (vit_nostrat only: profile next to the refresh fit's)); $RUN/progress.txt; $RUN/la_table.npy; $RUN/theta.npy.
  Code: cx_t1s.py (git, experiments/compress_6x6), spec $2.
Gate: no Monte Carlo chains (exact tables, one fit): quotable.py does not apply. The numbers are read directly from cx_t1s.json:
  python3 -c "import json;d=json.load(open('$RUN/cx_t1s.json'));print(d['readout']['primary_table'],d['converged'],d.get('E_FN'),d['control2'].get('rel_diff_eps'))"
Readings (pre-registered, per arm): objective was the cause if rms at decade -11 <= 0.03 with the bulk (decades >= -8) <= 0.01;
  representability limit if a verifiably converged fit still has >= 0.1 at -11; otherwise inconclusive. Physics pass: E_FN <= 1e-5.
Next per verdict: the worker reports the numbers to the lead only (no interpretation against the decision rule); the lead decides
  (projects/j1j2/PLAN.md).
EOB
  setsid nohup "$HERE/cx_t1s_watch.sh" "$JOB" "$RUN" > "$RUN/watcher.log" 2>&1 < /dev/null &
  echo "job=$JOB arm=$ARM run=$RUN watcher_pid=$!"
  sacct -j "$JOB" -X -n -o State,Reason%30
}
submit t1s_vit   specs/t1s_vit.json   01:45:00
submit t1s_vitns specs/t1s_vitns.json 01:15:00
submit t1s_pp    specs/t1s_pp.json    02:00:00
submit t1s_ppbf  specs/t1s_ppbf.json  02:15:00
