#!/bin/bash
# Test G1 launcher (A40 vGPU slice via cx_slice.sbatch). Run from experiments/compress_6x6 AFTER the lead's go.
# Stages code like launch_g0.sh / launch_t1s2.sh (cx_slice.sbatch copies $B/code/*.py into the run), adds the sha256 code-drift guard
# (the job exits 3 on mismatch), submits, writes ANALYSIS_BRIEF.md, arms a durable setsid watcher (cx_g1_watch.sh -> job_finish.sh --wake none).
# Wall: T0-phys measured ~30 s per E_FN solve on this slice (job 17410698: 918 s for 2 one-hop bases + physics). G1 = 7 one-hop runs x 15 solves
# = ~52 min, so 1 h is not half-wall; default WALL=02:00:00 (override: WALL=01:00:00 ./launch_g1.sh). cx_g1.json is dumped after every base.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd); B=/project/theorie/a/A.Otaifi/Chatty/runs/compress6; WALL=${WALL:-02:00:00}
RUNS=/project/theorie/a/A.Otaifi/Chatty/runs/compress6/runs
cp "$HERE/cx_lib.py" "$HERE/cx_g1.py" "$B/code/"
sha256sum "$B/code/cx_g1.py" "$B/code/cx_lib.py"
for f in cx_g1.py cx_lib.py; do cmp -s "$HERE/$f" "$B/code/$f" || { echo "copy of $f differs"; exit 1; }; done
SHAJSON=$(python3 -c "import hashlib,json;print(json.dumps({f:hashlib.sha256(open('$B/code/'+f,'rb').read()).hexdigest() for f in ('cx_g1.py','cx_lib.py')}))")
cd "$HERE"
python3 - <<PY
import json,os
d=json.load(open('specs/g1.json'))
for p in list(d['bases'].values())+[d['control_ref'],d['refresh']+'/s2.npy',d['refresh']+'/la_star_s2.npy']: assert os.path.exists(p),p
PY
SPECD=$B/specs_g1/$(date +%Y%m%dT%H%M%S); mkdir -p "$SPECD"; SP=$SPECD/g1.json
python3 -c "import json,sys;d=json.load(open('specs/g1.json'));d['code_sha256']=json.loads(sys.argv[1]);json.dump(d,open('$SP','w'),indent=1)" "$SHAJSON"
OUTL=$(sbatch --parsable --no-requeue --job-name=g1 --time="$WALL" "$B/cx_slice.sbatch" cx_g1.py "$SP")
JOB=${OUTL%%;*}; RUN=$RUNS/g1_$JOB; mkdir -p "$RUN"
cat > "$RUN/ANALYSIS_BRIEF.md" <<EOB
# G1 analysis brief (job $JOB, run $RUN)
Question (README "Test G1", projects/j1j2/experiments/compress_6x6/README.md, pre-registered 2026-10-22): one hop a = |base| |1 + alpha (E_L - E)|
(E_L of (|base|, s_2), bounded alpha search +-3/sdE, 12 evaluations, as cx_t0.py (c) / cx_t0phys.py (1)) on the tail-trained tables:
what is E_FN(s_2, a), with and without the oracle veto?
Files: $RUN/cx_g1.json (control{rec, ref_EFN, rel_EFN, ok}; bases.<name>{EFN_zero, EFN_zero_veto, alpha, EFN, EFN_veto, scan, X_frac};
  code_sha256.match); $RUN/progress.txt. Bases: t1s2_ppbf_17418857, t1s2_pp_17418856, t1s2_vit_17418858, t1s_ppbf_17412550, t1s_pp_17412549,
  t1s_vit_17412547 (la_table.npy each). Code cx_g1.py; spec $SP (= specs/g1.json + code_sha256). All energies per site relative to E0.
Control (runs first; exit 2 and no bases if it fails): ViT VMC table lab.lP0 must reproduce T0-phys (t0phys_17410698 cx_t0phys.json c.vit: EFN 2.9048e-05,
  EFN_veto 2.6138e-05) to rel 1e-3.
Gate: no Monte Carlo chains (exact tables, no training): quotable.py does not apply. Numbers read directly:
  python3 -c "import json;d=json.load(open('$RUN/cx_g1.json'));print(d['control']['ok'],{k:(v['EFN_zero'],v['EFN'],v['EFN_veto'],v['alpha']) for k,v in d['bases'].items()})"
Readings (README, best tail-trained Pfaffian arm, no veto; equal-hop refs psi_P+1 Lanczos 1.89e-5, ViT 2.90e-5, PP 2.37e-5, RBM+PP 1.29e-5):
  PASS: E_FN <= 1e-5 and below every reference. Partial: below 1.29e-5, above 1e-5. No edge: >= 1.29e-5.
Next per verdict: the worker reports the per-base table and the control to the lead only, no interpretation against the readings; the lead decides (projects/j1j2/PLAN.md).
EOB
setsid nohup "$HERE/cx_g1_watch.sh" "$JOB" "$RUN" > "$RUN/watcher.log" 2>&1 < /dev/null &
echo "job=$JOB run=$RUN watcher_pid=$!"; sacct -j "$JOB" -X -n -o State,Reason%30
