#!/bin/bash
# Test G0 launcher (A40 vGPU slice, 1 h). Run from experiments/compress_6x6 AFTER the lead's go.
# Stages code like t0_17410220 / t0b_17410565 (cx.sbatch copies $B/code/*.py into the run), submits, writes ANALYSIS_BRIEF.md,
# arms a durable setsid watcher (job_finish.sh --wake none).
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd); B=/project/theorie/a/A.Otaifi/Chatty/runs/compress6
cp "$HERE/cx_lib.py" "$HERE/cx_g0.py" "$B/code/"
cd "$HERE"
OUTL=$(sbatch --parsable --no-requeue --job-name=g0 --time=01:00:00 "$B/cx_slice.sbatch" cx_g0.py specs/g0.json)
JOB=${OUTL%%;*}
RUN=/project/theorie/a/A.Otaifi/Chatty/runs/compress6/runs/g0_$JOB
mkdir -p "$RUN"
cat > "$RUN/ANALYSIS_BRIEF.md" <<EOB
# G0 analysis brief (job $JOB, run $RUN)
Question: does gamma-FN (Sorella-Capriotti), on exact tables with sign s_2 and ratios from ViT / PP / RBM+PP, give a reported
energy <H> on phi^gamma that does not depend on the tail? (README "Test G0", projects/j1j2/experiments/compress_6x6/README.md)
Files: $RUN/cx_g0.json (controls.i/ii/iii, then grid.{vit,pp,rbmpp}: gamma, E_gamma, H_phi, neg_weight, res, chord_gamma_m1, min_H_phi);
  code cx_g0.py (git, experiments/compress_6x6); spec specs/g0.json. All energies per site relative to E0. No Monte Carlo chains: quotable.py does not apply.
Controls (grid runs only if all pass; controls.all_ok): (i) gamma=0 ViT reproduces E_FN 1.0186e-4; (ii) exact a=|psi0|, sign s0: E^gamma=E0 to 1e-9 for gamma 0,0.5,2;
  (iii) sign of gamma B flipped must FAIL (ii) at gamma 0.5, 2.
Decision rule (README, primary = min over gamma of <H> on phi^gamma, ViT ratios):
- PASS if <= 1e-5 or within 3x of the ceiling 9.15e-7. The reported energy then does not need a tail network: T1 is moot for the energy, and the open problem is the sign labels.
- FAIL if it removes < 50% of the gap from 1.0186e-4 to the ceiling.
- In between: partial; report the fraction removed, per base.
- The chord estimate is reported but never decides.
Next per verdict: the lead decides (projects/j1j2/PLAN.md). A worker only reports the numbers.
EOB
setsid nohup "$HERE/cx_g0_watch.sh" "$JOB" "$RUN" > "$RUN/watcher.log" 2>&1 < /dev/null &
echo "job=$JOB run=$RUN"; sacct -j "$JOB" -X -n -o State,Reason%30
