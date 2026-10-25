#!/bin/bash
# Test S1 launcher (README compress_6x6 "Test S1: the ceiling of the Pfaffian's own sign"): ONE job, one A40 slice (gpu:a40-24gb), fit-free chain, sign 'own'.
#   launch_s1.sh [--stage-only]
# Stages RUN=runs/s1_own (code copy + sha256, spec, ANALYSIS_BRIEF.md); unless --stage-only submits, arms the durable watcher
# (cx_s1_watch.sh -> job_finish.sh --wake none) and prints the job id.  Run AFTER the lead's go.
# Sizing: O2a (job 17430739, same code path and hardware, A40-24Q) took 110 s per round (736 s for the control + 6 evaluations);
#   S1 does <= 16 evaluations (15 maps) + zero-hop control: <= ~35 min, wall 1 h (<= 1 A40-h on a slice, README budget <= 2 A40-h).
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd); B=/project/theorie/a/A.Otaifi/Chatty/runs/compress6; WALL=${WALL:-01:00:00}
RUN=/project/theorie/a/A.Otaifi/Chatty/runs/compress6/runs/s1_own
mkdir -p "$RUN/code" /project/theorie/a/A.Otaifi/Chatty/runs/compress6/logs
# code: everything cx_lib needs comes from $B/code (as cx_slice.sbatch); cx_lib.py must equal the repo's, cx_s1.py is copied from the repo
cp "$B"/code/*.py "$RUN/code/"; cp "$HERE/cx_s1.py" "$RUN/code/"
cmp -s "$HERE/cx_lib.py" "$RUN/code/cx_lib.py" || { echo "cx_lib.py differs from $B/code"; exit 1; }
sha256sum "$RUN"/code/cx_s1.py "$RUN"/code/cx_lib.py | tee "$RUN/code_sha256.txt"
python3 - "$RUN" "$HERE" <<'PY'
import json, hashlib, os, sys
run, here = sys.argv[1:3]
sha = {f: hashlib.sha256(open(f'{run}/code/{f}', 'rb').read()).hexdigest() for f in ('cx_s1.py', 'cx_lib.py')}
d = json.load(open(f'{here}/specs/s1_own.json')); d['code_sha256'] = sha
for p in [d['pp'], d['refresh'] + '/s2.npy']: assert os.path.exists(p), p
json.dump(d, open(f'{run}/spec_s1.json', 'w'), indent=1)
PY
if [ ! -f "$RUN/ANALYSIS_BRIEF.md" ]; then cp "$HERE/specs/s1_ANALYSIS_BRIEF.md" "$RUN/ANALYSIS_BRIEF.md"; sed -i "s#@RUN@#$RUN#g" "$RUN/ANALYSIS_BRIEF.md"; fi
echo "staged $RUN"
[ "${1:-}" = "--stage-only" ] && exit 0
OUTL=$(sbatch --parsable --no-requeue --job-name=s1 --time="$WALL" "$HERE/s1.sbatch" "$RUN")
JOB=${OUTL%%;*}; echo "$JOB" > "$RUN/job_id.txt"
setsid nohup "$HERE/cx_s1_watch.sh" "$JOB" "$RUN" > "$RUN/watcher.log" 2>&1 < /dev/null &
echo "job=$JOB run=$RUN watcher_pid=$!"; sacct -j "$JOB" -X -n -o State,Reason%30
