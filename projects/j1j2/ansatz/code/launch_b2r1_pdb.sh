#!/bin/bash
# Branch 2 R1 fit on Paderborn H100 (PLAN_pfaffian.md "Branch 2 GO" step 3; job b2r1_pdb.sbatch).  PREPARED 2026-11-05, NOT SUBMITTED: submission waits for the bond-ESS gate (step 1),
# the labelled pool (step 2) and the lead's go.  Modes:
#   --local POOLDIR [LOCALRUN]   write and check the spec locally only (no ssh): LOCALRUN/spec_r1.json with the REMOTE paths
#   (none) POOLDIR               stage: --local + copy code / pool / rb8_prod params to Paderborn (sha256-checked)
#   --test POOLDIR               stage + `sbatch --test-only` with the exact submit flags
#   --submit POOLDIR             stage + sbatch + durable watcher (lswo_pdb_watch.sh, setsid, --wake none) + job id record
# POOLDIR holds pool.npz, pool.npz.sha256, pool_summary.json from b2_prep.py merge.  Env: R1_QOS (default cont; the lead's order is express_hpcalot first when its November
# remainder covers the run: check `pc2status`), R1_TAG (default r1), R1_SPEC_OVER (json dict merged into the spec, e.g. '{"lambda_init_re": 0.0}').
# Run on th-ws-7010m51 (the PC2 VPN host).  Every ssh/rsync uses -o HostName=login4.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
MODE=${1:?mode}; [ "$MODE" = --local ] || [ "$MODE" = --test ] || [ "$MODE" = --submit ] || { MODE=stage; set -- stage "$@"; }
POOLDIR=${2:?POOLDIR}; TAG=${R1_TAG:-r1}; QOS=${R1_QOS:-cont}; OVER=${R1_SPEC_OVER:-"{}"}
Q=/scratch/a/A.Otaifi/ansatz_runs/pfaffian_8x8_r1/$TAG; LOC=${3:-$Q/fit}
P=/scratch/hpc-prf-ppdmrg/hpcalot/chatty_compress6; D=$P/data/b2r1; CR=$P/codeB2R1; RUNF=$P/runs/b2r1_${TAG}_fit
PROD8=/scratch/a/A.Otaifi/rbmpp8/rb8_prod/params_final.npz; PROD8SHA=0520c5be4a7770d84e1968f3669fbbf8063415388c1b50a47ac043317754cbe9
CODE="b2_prep.py pool_fit.py pp2_lemma.py cx_lib.py rbmpp_lib.py"
SSHO="ssh -o BatchMode=yes -o HostName=login4"
rem() { $SSHO paderborn "export PATH=/opt/software/slurm/25.11.8-1/bin:\$PATH; $1"; }
[ "$(sha256sum $PROD8 | cut -d' ' -f1)" = "$PROD8SHA" ] || { echo "rb8_prod params sha mismatch"; exit 1; }
POOLSHA=$(cut -d' ' -f1 $POOLDIR/pool.npz.sha256)
[ "$(sha256sum $POOLDIR/pool.npz | cut -d' ' -f1)" = "$POOLSHA" ] || { echo "pool.npz sha mismatch"; exit 1; }
mkdir -p $LOC $Q
# 1. spec (local file, remote paths); the pool must be a b2_prep merge (decades -2..-9 present, K 2, med recorded) and the code shas are pinned
PY=/project/theorie/a/A.Otaifi/Chatty/runs/a40slice/venv_jax_0.8.2/bin/python
JAX_PLATFORMS=cpu $PY - "$HERE" "$POOLDIR" "$POOLSHA" "$D" "$LOC" "$OVER" <<'PY'
import json, sys, os
HERE, POOLDIR, PS, D, LOC, OVER = sys.argv[1:7]; sys.path.insert(0, HERE)
import numpy as np, b2_prep as B2
s = json.load(open(f'{POOLDIR}/pool_summary.json')); assert s['pool_sha256'] == PS and s['K'] == 2 and s['betas'] == [1.0, 0.5, 0.25], 'pool summary'
assert s['decs_rel'] == B2.DECS_R1, f"pool decades {s['decs_rel']} != {B2.DECS_R1}"
assert all(s['counts'][str(d)]['train'] > 0 and s['counts'][str(d)]['heldout'] > 0 for d in B2.DECS_R1), 'a decade without train / held-out entries'
z = np.load(f'{POOLDIR}/pool.npz'); assert float(z['med2lnB']) == float(s['med2lnB'])
sp = B2.r1_spec(f'{D}/pool.npz', PS, warm_npz=f'{D}/params_final.npz', code_dir=HERE, **json.loads(OVER))
sp = B2.resolve_lam(dict(sp, warm_npz=B2.PROD8), f'{POOLDIR}/pool.npz'); sp['warm_npz'] = f'{D}/params_final.npz'   # lam from the LOCAL pool + params (same files, sha-checked)
json.dump(sp, open(f'{LOC}/spec_r1.json', 'w'), indent=1)
print(f"spec {LOC}/spec_r1.json: pool S {s['S']} med {s['med2lnB']} decs {s['decs_rel']} train/heldout " + ' '.join(f"{d}:{s['counts'][str(d)]['train']}/{s['counts'][str(d)]['heldout']}" for d in B2.DECS_R1)
      + f"  B {sp['B']} steps {sp['max_steps']} fit_sec {sp['fit_sec']} lam_init {sp['lambda_init_re']} p2_sigma_rel {sp['p2_sigma_rel']}")
PY
{ for f in $CODE; do sha256sum $HERE/$f | sed "s#  .*#  $f#"; done; } | sort > $LOC/code_sha256.txt
cat > $LOC/ANALYSIS_BRIEF.md <<EOF
# Branch 2 R1 fit ($TAG; Paderborn run dir $RUNF, local $LOC)
Question: does the two-PP fit to T_1 = sgn(M1)|B_0| e^{4 f_J[B_0; sgn(M1)]} (mode 'fix', f0 1.22) converge on the pool $POOLDIR (sha ${POOLSHA:0:12})?
Files: round_1/pool_fit.json (curve, hist), round_1/theta_step{150..1350}.npy (every 150), round_1/theta_tail.npy (tail average of the last 900 steps: theta at 600..1350
and the final 1500, 7 iterates; the round's theta_1), round_1/theta_bar.npy (pool_fit's mean of ALL saved iterates: not used), cx_r1.json (done flag), run_metadata.txt.
Spec: spec_r1.json (lam recomputed on this pool: p2_logratio_median, p2_logratio_q, lambda_init_re); pool sha in pool.npz.sha256; med2lnB read from the pool.
Registered before submission: fit estimate 1.3-5 H100-h (3-12 s/step x 1500, derived, not measured at 8x8). STOP (cancel and report to the lead) if the measured
s/step projects above 7.5 H100-h (1.5x the top, i.e. > 18 s/step); fit_sec 27000 enforces the same ceiling inside the job.
Gate: cx_r1.json "done": true, steps == 1500, no "stop"; held-out loss L_ho falls from step 0 and its last 300 steps are flat within the anneal tolerance; per-decade held-out
loss with decade -2 not an outlier (PLAN "pool used" condition).  H100-h = wall x 1; > 7.5 h (1.5x the 5 h estimate) -> stop and tell the lead.
Next: COMPLETED -> bare c = 0 mechanism grade (gfmc_b2.BareM1Sign on theta_bar) after the lead's go; FAILED / TIMEOUT / STALLED -> report to the lead with the curve.
EOF
[ "$MODE" = --local ] && { echo "local spec written: $LOC/spec_r1.json"; exit 0; }
chk() { # LOCALFILE REMOTEFILE
  l=$(sha256sum "$1" | cut -d' ' -f1); r=$(rem "sha256sum $2 2>/dev/null" | cut -d' ' -f1 || true)
  if [ "$l" != "$r" ]; then rsync -a -e "$SSHO" "$1" paderborn:$2; r=$(rem "sha256sum $2" | cut -d' ' -f1); fi
  [ "$l" = "$r" ] || { echo "sha256 mismatch $2"; exit 1; }; echo "input ok ${l:0:12} $2"; }
rem "mkdir -p $CR $D $RUNF/round_1"
for f in $CODE; do chk $HERE/$f $CR/$f; done
rem "cd $CR && sha256sum $CODE" | sort > $LOC/code_sha256_remote.txt
diff $LOC/code_sha256.txt $LOC/code_sha256_remote.txt > /dev/null || { echo "codeB2R1 sha256 differs from local"; exit 1; }
chk $HERE/b2r1_pdb.sbatch $P/b2r1_pdb.sbatch; chk $POOLDIR/pool.npz $D/pool.npz; chk $PROD8 $D/params_final.npz
cp $POOLDIR/pool.npz.sha256 $LOC/pool.npz.sha256
rsync -a -e "$SSHO" $LOC/spec_r1.json $LOC/ANALYSIS_BRIEF.md $LOC/pool.npz.sha256 paderborn:$RUNF/
echo "staged Paderborn: code $CR, pool $D/pool.npz (${POOLSHA:0:12}), run $RUNF; local $LOC"
fitf="--parsable --no-requeue --qos=$QOS --job-name=pf_b2${TAG} --time=15:00:00 --output=$RUNF/slurm-%j.out"
[ "$MODE" = --test ] && { rem "sbatch --test-only $fitf $P/b2r1_pdb.sbatch $RUNF" 2>&1; exit 0; }
[ "$MODE" = --submit ] || exit 0
[ -e "$LOC/job_id.txt" ] && { echo "$LOC/job_id.txt exists (already submitted)"; exit 1; }
OUTL=$(rem "sbatch $fitf $P/b2r1_pdb.sbatch $RUNF" 2>/dev/null); JOB=${OUTL%%;*}
[[ "$JOB" =~ ^[0-9]+$ ]] || { echo "sbatch failed: $OUTL"; exit 1; }
echo "$JOB" > $LOC/job_id.txt; echo "b2 $TAG fit $JOB qos $QOS $(date -Is)" >> $Q/job_ids.txt
R1_REMRUN=$RUNF setsid nohup "$HERE/lswo_pdb_watch.sh" "$JOB" "$LOC" > "$LOC/watcher.log" 2>&1 < /dev/null &
sleep 3; W=$(pgrep -f "lswo_pdb_watch.sh $JOB" | head -1); echo "$W" > $LOC/watcher_pid.txt; echo "watcher pid $W ppid=$(ps -o ppid= -p $W 2>/dev/null)"
echo "b2 R1 fit: job $JOB"; rem "sacct -j $JOB -X -n -o JobID,JobName%20,State,Reason%30" 2>/dev/null || true
