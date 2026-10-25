#!/bin/bash
# Test B0 launcher (README compress_6x6 "Test B0"): ONE job on a full A40: SR VMC training of PP + 3x3 backflow, then the exact readout.
#   launch_b0.sh [full|simple] [--stage-only]        default full (projection of the TRAINING; the readout is always 'full', 576 images)
# Stages RUN=runs/b0_ppbf_<mode> (code copy + sha256, specs, ANALYSIS_BRIEF.md), then (unless --stage-only) submits, arms the durable
# watcher (cx_b0_watch.sh -> job_finish.sh --wake none) and prints the job id.  Run AFTER the lead's go.
# Sizing (measured, see the B0 report): per SR step with eloc float32, 2048 chains, RTX 2080 Ti (A40 smoke job 17441980 queued; A40 not yet measured):
#   full 17.8 s, simple 4.55 s.  Readout: table of 15.8M orbits x 576 images ~ 5782 s (t1s2_ppbf, same network and code path) + exact
#   solves/controls ~ 600 s = ~1.8 h.  Training wall_sec 19080 s (5.3 h) -> full ~1070 steps, simple ~4180 steps (steps = lr-schedule horizon
#   1100 / 4300).  Total ~7.1 h of the 8 h limit (README budget <= 8 A40-h), partition MaxTime cip/inter 2 days.
set -euo pipefail
MODE=${1:-full}; STAGE_ONLY=${2:-}
case "$MODE" in full|simple) ;; *) echo "mode full|simple"; exit 1;; esac
HERE=$(cd "$(dirname "$0")" && pwd); EXP=$HERE/..; B=/project/theorie/a/A.Otaifi/Chatty/runs/compress6
RUN=/project/theorie/a/A.Otaifi/Chatty/runs/compress6/runs/b0_ppbf_$MODE
mkdir -p "$RUN/code"
for f in cx_lib.py cx_b0.py; do cp "$HERE/$f" "$RUN/code/"; done
for f in rbmpp_lib.py rbmpp_bf_lib.py rbmpp_bf_train.py; do cp "$EXP/rbmpp_6x6/$f" "$RUN/code/"; done
for f in st6_sector.py lanczos_lib.py vit_dt.py; do cp "$B/code/$f" "$RUN/code/"; done      # as cx.sbatch (cp $B/code/*.py)
for f in cx_lib.py; do cmp -s "$HERE/$f" "$B/code/$f" || { echo "$f differs from $B/code"; exit 1; }; done
cmp -s "$EXP/rbmpp_6x6/rbmpp_lib.py" "$B/code/rbmpp_lib.py" || { echo "rbmpp_lib.py differs from $B/code"; exit 1; }
sha256sum "$RUN"/code/{cx_b0.py,cx_lib.py,rbmpp_bf_lib.py,rbmpp_bf_train.py,rbmpp_lib.py}
SHAJSON=$(python3 -c "import hashlib,json;print(json.dumps({f:hashlib.sha256(open('$RUN/code/'+f,'rb').read()).hexdigest() for f in ('cx_b0.py','cx_lib.py','rbmpp_bf_lib.py','rbmpp_bf_train.py','rbmpp_lib.py')}))")
python3 - "$SHAJSON" "$HERE" "$RUN" "$MODE" <<'EOF'
import json, sys
sha, here, run, mode = json.loads(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4]
t = json.load(open(f'{here}/specs/b0_train_{mode}.json')); t['code_sha256'] = {k: sha[k] for k in ('rbmpp_bf_train.py', 'rbmpp_bf_lib.py', 'rbmpp_lib.py')}
r = json.load(open(f'{here}/specs/b0_readout.json')); r['code_sha256'] = {k: sha[k] for k in ('cx_b0.py', 'cx_lib.py', 'rbmpp_bf_lib.py', 'rbmpp_lib.py')}
r['params'] = f'{run}/train/params_final.npz'
json.dump(t, open(f'{run}/spec_train.json', 'w'), indent=1); json.dump(r, open(f'{run}/spec_readout.json', 'w'), indent=1)
EOF
for p in $(python3 -c "import json;d=json.load(open('$RUN/spec_readout.json'));print(d['refresh']+'/s2.npy',d['pp_table'],d['pp_params'])") $(python3 -c "import json;print(json.load(open('$RUN/spec_train.json'))['init'])"); do [ -e "$p" ] || { echo "missing $p"; exit 1; }; done
if [ ! -f "$RUN/ANALYSIS_BRIEF.md" ]; then cp "$HERE/specs/b0_ANALYSIS_BRIEF.md" "$RUN/ANALYSIS_BRIEF.md"; sed -i "s#@RUN@#$RUN#g; s#@MODE@#$MODE#g" "$RUN/ANALYSIS_BRIEF.md"; fi
echo "staged $RUN"
[ "$STAGE_ONLY" = "--stage-only" ] && exit 0
OUTL=$(sbatch --parsable --no-requeue --job-name=b0 --time=08:00:00 "$HERE/b0.sbatch" "$RUN")
JOB=${OUTL%%;*}; echo "$JOB" > "$RUN/job_id.txt"
setsid nohup "$HERE/cx_b0_watch.sh" "$JOB" "$RUN" > "$RUN/watcher.log" 2>&1 < /dev/null &
echo "job=$JOB run=$RUN watcher_pid=$!"
sacct -j "$JOB" -X -n -o State,Reason%30
