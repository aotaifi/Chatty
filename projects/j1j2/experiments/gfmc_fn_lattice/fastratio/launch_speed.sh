#!/bin/bash
# Stage (default) or --submit the fast-vs-ref speed + equivalence job on ONE full theorie A40 (Paderborn H100 estimated start 2026-11-03, > 1 day).
# Code is taken from git commit $COMMIT (pinned by sha256), inputs by sha256.  --submit: sbatch, record job id, arm the durable watcher (setsid, watch_local.sh -> job_finish.sh --wake none).
set -euo pipefail
COMMIT=${COMMIT:?set COMMIT}; HERE=$(cd "$(dirname "$0")" && pwd); REL=projects/j1j2/experiments/gfmc_fn_lattice/fastratio
RUN=/scratch/a/A.Otaifi/ansatz_runs/fastratio/gpu_job; D=/scratch/a/A.Otaifi/ansatz_runs/fastratio/data
mkdir -p $RUN/code
[ -e $RUN/JOB_ID ] && { echo "already submitted: $(cat $RUN/JOB_ID)"; exit 1; }
for f in rbmpp_lib.py gfmc_fn_lib.py gfmc_run.py bench_gpu.py job_speed.sh gate_speed.py; do git -C $HERE show $COMMIT:$REL/$f > $RUN/code/$f; done
chmod +x $RUN/code/job_speed.sh; ( cd $RUN/code && sha256sum *.py *.sh ) > $RUN/code_sha256.txt; echo $COMMIT > $RUN/COMMIT
chk() { h=$(sha256sum "$1" | cut -d' ' -f1); [ "$h" = "$2" ] || { echo "sha256 mismatch $1"; exit 1; }; echo "input ok ${h:0:12} $1"; }
chk $D/params_final.npz 0520c5be4a7770d84e1968f3669fbbf8063415388c1b50a47ac043317754cbe9
chk $D/params10_random.npz d3ff5e6e7f34b72df46527c191adcda56e7e2fbeb3acd7e8a3644807360b5b0b
chk $D/consts10_fake.json f0db6661eee1b02c1937776b719f3bc6b67d6a69160670fe8476222108bc7b59
cat > $RUN/ANALYSIS_BRIEF.md <<EOB
# fastratio speed + equivalence job (one full theorie A40; run dir $RUN; code commit $COMMIT)
## Question
Does the determinant-contraction path (--ratio fast) give (a) >= 2x per-call speed over --ratio ref at 10x10 (16 hidden, random params) and (b) energies equal to <= 1e-10 relative per block in a short 8x8 GFMC (M 64, 3 blocks, seed 1)?
## Files
bench.json (kernel timing, 8x8 production params / 10x10 random), eq_ref/ eq_fast/ (blocks.npz, progress.txt), g10_ref/ g10_fast/ (optional 10x10 GFMC), progress.txt (FASTRATIO_DONE verdict line), run_metadata.txt.
## Gate command
python3 $HERE/gate_speed.py $RUN   (prints GATE lines and one VERDICT line; writes gate_speed.json)
## Next step per verdict
speedup(10x10) >= 2 and equivalence True -> report PASS to the lead (project lead decides integration).  speedup < 2 -> report the measured factor and the reason (kernel times, temp memory); do not keep optimising.
equivalence False -> report the per-block numbers; the fast path is not integrable.  This is an A40 measurement, NOT the H100 the gate was stated for (H100 est. start > 1 day).
No quotable.py: nothing is fitted or averaged.
EOB
if [ "${1:-}" = "--submit" ]; then
  jid=$(sbatch --parsable $HERE/speed.sbatch $RUN); [[ "$jid" =~ ^[0-9]+$ ]] || { echo "SUBMIT FAILED: '$jid'"; exit 1; }
  echo $jid > $RUN/JOB_ID; echo "SUBMITTED job $jid"
  setsid nohup bash $HERE/watch_local.sh $jid $RUN > $RUN/watcher.log 2>&1 < /dev/null &
  sleep 1; echo "WATCHER armed pid $! ppid $(ps -o ppid= -p $! | tr -d ' ')"
  squeue -j $jid -o "%i %T %R %S"
else echo "STAGED (dry): $RUN ; submit with --submit"; fi
