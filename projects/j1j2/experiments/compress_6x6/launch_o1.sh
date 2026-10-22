#!/bin/bash
# Test O1 launcher (A40 slice via cx_slice.sbatch), modelled on launch_g1.sh. ~6 FN solves, 1 h wall.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd); B=/home/a/A.Otaifi/chatty_compress6; WALL=${WALL:-01:00:00}
RUNS=/project/theorie/a/A.Otaifi/chatty_compress6/runs
cp "$HERE/cx_lib.py" "$HERE/cx_o1.py" "$B/code/"
sha256sum "$B/code/cx_o1.py" "$B/code/cx_lib.py"
for f in cx_o1.py cx_lib.py; do cmp -s "$HERE/$f" "$B/code/$f" || { echo "copy of $f differs"; exit 1; }; done
SHAJSON=$(python3 -c "import hashlib,json;print(json.dumps({f:hashlib.sha256(open('$B/code/'+f,'rb').read()).hexdigest() for f in ('cx_o1.py','cx_lib.py')}))")
cd "$HERE"
python3 - <<PY
import json,os
d=json.load(open('specs/o1.json'))
for p in list(d['bases'].values())+[d['g1_json'],d['refresh']+'/s2.npy',d['refresh']+'/la_star_s2.npy']: assert os.path.exists(p),p
PY
SPECD=$B/specs_o1/$(date +%Y%m%dT%H%M%S); mkdir -p "$SPECD"; SP=$SPECD/o1.json
python3 -c "import json,sys;d=json.load(open('specs/o1.json'));d['code_sha256']=json.loads(sys.argv[1]);json.dump(d,open('$SP','w'),indent=1)" "$SHAJSON"
OUTL=$(sbatch --parsable --no-requeue --job-name=o1 --time="$WALL" "$B/cx_slice.sbatch" cx_o1.py "$SP")
JOB=${OUTL%%;*}; RUN=$RUNS/o1_$JOB; mkdir -p "$RUN"
cat > "$RUN/ANALYSIS_BRIEF.md" <<EOB
# O1 analysis brief (job $JOB, run $RUN)
Question (README "De-oracling the G1 guide", O1, pre-registered 2026-10-22; not edited): alpha_V = argmin of the exact variational energy <H> of the
one-hop state s_2 |a| |1 + alpha (E_L - E)| (bounds +-3/sdE, same E_L/E/sdE as cx_g1); report E_FN(s_2, alpha_V) next to the G1 alpha and its E_FN.
Files: $RUN/cx_o1.json (control{EFN, ref 7.056e-6, ok}; bases.<name>{alpha_V, H_V, EFN_V, alpha_G1, H_G1, EFN_G1_recomputed, EFN_G1_file, at_bound, grid_min});
  $RUN/progress.txt; code_sha256.match. Bases t1s2_ppbf_17418857, t1s2_pp_17418856, t1s2_vit_17418858. Spec $SP. Code cx_o1.py. Energies per site relative to E0.
Control: ppbf E_FN at the G1 alpha must reproduce 7.056e-6 to 3 digits; otherwise the job exits 2 and the worker stops and reports.
Gate: no Monte Carlo chains (exact solves): quotable.py does not apply. Read numbers directly:
  python3 -c "import json;d=json.load(open('$RUN/cx_o1.json'));print(d['control'],{k:(v['alpha_V'],v['H_V'],v['EFN_V'],v['alpha_G1'],v['EFN_G1_recomputed']) for k,v in d['bases'].items()})"
Next: the worker reports control and per-base table to the lead only; no interpretation against the PASS/FAIL rule (lead decides).
EOB
setsid nohup "$HERE/cx_o1_watch.sh" "$JOB" "$RUN" > "$RUN/watcher.log" 2>&1 < /dev/null &
echo "job=$JOB run=$RUN watcher_pid=$!"; sacct -j "$JOB" -X -n -o State,Reason%30
