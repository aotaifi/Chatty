#!/bin/bash
# Gate V (PLAN_pfaffian.md "B design" (5) V, "V (narrowed)"), Paderborn H100.  Adapted from launch_r5_pdb.sh.  Modes: (none) STAGE ONLY; --test: stage + `sbatch --test-only`
# of both jobs with the exact submit flags; --submit: sbatch for real (needs the CPU checks passed; the lead's go).
# One arm: round 5's pp2_s1 (theta_4 -> T_5, s_G, B = P_1 + lam P_2, lam 0.1, seed 1, exactly 1500 steps, theta saved at 600..1350) with the batches drawn from the
# SAMPLED POOL (pool_fit.py fit; pool built locally by `pool_fit.py build $Q/pool $Q/pool/spec_pool.json`, S = 4e5, beta 1, 1/2, 1/4, K = 2, B-proposal partners).
# Jobs: pf_v (v_pdb.sbatch, 05:00) + pf_vt (r5_pdb_tab.sbatch, unchanged, codeR5, 01:00, afterok): theta_V = mean of the 7 iterates, tab_t5 table.
# Watchers (durable setsid, --wake none): lswo_pdb_watch.sh (fit), r3_pdb_tab_watch.sh (tab), v_chain.sh (both COMPLETED -> theorie CPU readout v_readout.sbatch).
# The pool's target is whatever pool/pool_summary.json records (OLD T_5 now; rebuild the pool with the fj_fix mode-b1 table when it lands, then re-stage).
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
Q=/scratch/a/A.Otaifi/ansatz_runs/pfaffian_v; R4=/scratch/a/A.Otaifi/ansatz_runs/pfaffian_r4; R5=/scratch/a/A.Otaifi/ansatz_runs/pfaffian_r5
TH4=$R4/pdb_tabs/theta_4.npy; TH4SHA=3114250142513bce80f6d1436e0465a44d509cee13248903bf5ce64f9f4bc9b6
THL=/scratch/a/A.Otaifi/ansatz_runs/pfaffian_lswo/run/round_1/theta.npy; THLSHA=7cf4129453ab7e1ab8f450aa02c9b5902066a6680d0b66534c668c42a00f7d57
P=/scratch/hpc-prf-ppdmrg/hpcalot/chatty_compress6; D=$P/data; D5=$D/r5; DV=$D/v; CV=$P/codeV
RUNF=$P/runs/v_fit2; RUNT=$P/runs/v_tab2; QP=$Q/pdb2   # FRESH copy (PI 2026-11-04): new dirs, nothing of the first copy 4107178/4107179 (cancelled by the old race rule) is touched
SSHO="ssh -o BatchMode=yes -o HostName=login4"
rem() { $SSHO paderborn "export PATH=/opt/software/slurm/25.11.8-1/bin:\$PATH; $1"; }
MODE=${1:-stage}; QOS=cont
CODE="pool_fit.py cx_lib.py rbmpp_lib.py"; SS6=/project/theorie/a/A.Otaifi/Chatty/runs/compress6/code/st6_sector.py
[ "$(sha256sum $TH4 | cut -d' ' -f1)" = "$TH4SHA" ] || { echo "theta_4 sha mismatch"; exit 1; }
[ "$(sha256sum $THL | cut -d' ' -f1)" = "$THLSHA" ] || { echo "lswo round_1 theta sha mismatch"; exit 1; }
POOLSHA=$(cut -d' ' -f1 $Q/pool/pool.npz.sha256)
[ "$(sha256sum $Q/pool/pool.npz | cut -d' ' -f1)" = "$POOLSHA" ] || { echo "pool.npz sha mismatch"; exit 1; }
python3 - "$Q" "$POOLSHA" <<'PY'
import json, sys
Q, ps = sys.argv[1:3]; d = json.load(open(f'{Q}/pool/pool_summary.json')); assert d['pool_sha256'] == ps and d['S'] == 400000 and d['K'] == 2 and d['betas'] == [1.0, 0.5, 0.25], 'pool summary'
print('pool target', d['spec']['target'], d['target_sha256'][:12], '| B table', d['spec']['la_B'][-60:], '| sign', d['spec']['sign'][-40:])
PY
if [ "$MODE" = --submit ]; then
  python3 - "$Q" <<'PY'
import json, sys
Q = sys.argv[1]
for k, p in (('a', f'{Q}/cpu/checka_job/check_a/check_a.json'), ('b', f'{Q}/cpu/check_b/check_b.json'), ('c', f'{Q}/cpu/check_c_pert/check_c.json')):
    assert json.load(open(p))['pass'], f'CPU check ({k}) not passed: {p}'
PY
fi
chk() { # LOCALFILE REMOTEFILE
  l=$(sha256sum "$1" | cut -d' ' -f1); r=$(rem "sha256sum $2 2>/dev/null" | cut -d' ' -f1 || true)
  if [ "$l" != "$r" ]; then rsync -a -e "$SSHO" "$1" paderborn:$2; r=$(rem "sha256sum $2" | cut -d' ' -f1); fi
  [ "$l" = "$r" ] || { echo "sha256 mismatch $2"; exit 1; }; echo "input ok ${l:0:12} $2"; }
# 0. code: codeV on Paderborn (copied, sha-checked); the table job uses codeR5 as round 5 did (r5_pdb_tab.sbatch, its pins)
mkdir -p $QP; rem "mkdir -p $CV $DV $RUNF/round_1 $RUNT/tab_t5"
for f in $CODE; do chk $HERE/$f $CV/$f; done; chk $SS6 $CV/st6_sector.py
{ for f in $CODE; do sha256sum $HERE/$f | sed "s#  .*#  $f#"; done; } | sort > $QP/code_sha256.txt
rem "cd $CV && sha256sum $CODE" | sort > $QP/code_sha256_remote.txt
diff $QP/code_sha256.txt $QP/code_sha256_remote.txt > /dev/null || { echo "codeV sha256 differs from local"; exit 1; }
chk $HERE/r5_pdb_tab.sbatch $P/r5_pdb_tab.sbatch; chk $HERE/v_pdb.sbatch $P/v_pdb.sbatch
# 1. data
chk $Q/pool/pool.npz $DV/pool.npz; chk $TH4 $D5/theta_4.npy; chk $THL $D5/theta_lswo_r1.npy
# 2. specs: the fit spec (round 5 pp2_s1's knobs from o2b round_1 spec_fit.json + its overrides; the pool replaces the table), the table spec (pp2_s1's, warm_theta -> V's theta_5)
mkdir -p $QP/fit $QP/tab/tab_t5 $Q/readout
python3 - "$QP" "$P" "$D5" "$DV" "$POOLSHA" "$R5" "$HERE" <<'PY'
import json, sys, hashlib
Q, P, D5, DV, PS, R5, HERE = sys.argv[1:8]; sha = lambda p: hashlib.sha256(open(p, 'rb').read()).hexdigest()
o2b = json.load(open('/project/theorie/a/A.Otaifi/Chatty/runs/compress6/runs/o2b_17431886/round_1/spec_fit.json'))
r5 = json.load(open(f'{R5}/pp2_s1/fit/spec_r1.json'))
sp = {k: o2b[k] for k in ('B', 'K', 'B_acc', 'B_ho', 'lam_b', 'lam0', 'lam_cap', 'tau', 'tau_min', 'anneal_steps', 'z_acc', 'ho_every', 'flat_tol')}
sp.update(arm='pp2', rbm_alpha=r5['rbm_alpha'], warm_theta=f'{D5}/theta_4.npy', warm_theta2=f'{D5}/theta_lswo_r1.npy', lambda_init_re=r5['fit_overrides']['lambda_init_re'],
          lambda_init_im=r5['fit_overrides']['lambda_init_im'], seed=r5['fit_overrides']['seed'], max_steps=r5['max_steps'], save_theta_at=r5['fit_overrides']['save_theta_at'],
          fit_sec=r5['fit_sec'], pool=f'{DV}/pool.npz', pool_sha256=PS, code_sha256={f: sha(f'{HERE}/{f}') for f in ('pool_fit.py', 'cx_lib.py', 'rbmpp_lib.py')})
assert sp['B'] == 8208 and sp['K'] == 2 and sp['max_steps'] == 1500 and sp['seed'] == 1 and sp['tau_min'] == 0.01
json.dump(sp, open(f'{Q}/fit/spec_pool_fit.json', 'w'), indent=1)
t = json.load(open(f'{R5}/pp2_s1/tab/tab_t5/spec_r1.json')); t['warm_theta'] = f'{P}/runs/v_tab2/theta_5.npy'
json.dump(t, open(f'{Q}/tab/tab_t5/spec_r1.json', 'w'), indent=1)
PY
diff -u $R5/pp2_s1/tab/tab_t5/spec_r1.json $QP/tab/tab_t5/spec_r1.json > $QP/spec_diff_tab.txt || true
rsync -a -e "$SSHO" $QP/fit/spec_pool_fit.json paderborn:$RUNF/; rsync -a -e "$SSHO" $QP/tab/tab_t5/spec_r1.json paderborn:$RUNT/tab_t5/
for K in fit tab; do { echo "# Gate V fresh Paderborn copy 'pdb2' ($K; remote ${RUNF} / ${RUNT}, local $QP/$K); raced against the A40 copy 17695322/3 (first completed copy's table is read out by v_chain2.sh).  Full brief: $Q/ANALYSIS_BRIEF.md"; sed -n '2,$p' $Q/ANALYSIS_BRIEF.md; } > $QP/$K/ANALYSIS_BRIEF.md; done
rsync -a -e "$SSHO" $QP/fit/ANALYSIS_BRIEF.md paderborn:$RUNF/; rsync -a -e "$SSHO" $QP/tab/ANALYSIS_BRIEF.md paderborn:$RUNT/
echo "staged Paderborn: code $CV, pool $DV/pool.npz (${POOLSHA:0:12}), runs $RUNF $RUNT; local $QP/{fit,tab}"
fitf="--parsable --no-requeue --qos=$QOS --job-name=pf_v2 --time=05:00:00 --output=$RUNF/slurm-%j.out"
tabf="--parsable --no-requeue --qos=$QOS --job-name=pf_vt2 --time=01:00:00 --output=$RUNT/slurm-%j.out"
if [ "$MODE" = --test ]; then
  echo "--- test-only fit"; rem "sbatch --test-only $fitf $P/v_pdb.sbatch $RUNF" 2>&1
  echo "--- test-only tab (no dependency: a test-only cannot see the not-yet-existing fit job)"; rem "sbatch --test-only $tabf $P/r5_pdb_tab.sbatch $RUNT $RUNF t5" 2>&1
fi
[ "$MODE" = --submit ] || exit 0
[ -e "$QP/fit/job_id.txt" ] && { echo "$QP/fit/job_id.txt exists (already submitted)"; exit 1; }
OUTL=$(rem "sbatch $fitf $P/v_pdb.sbatch $RUNF" 2>/dev/null); JOB=${OUTL%%;*}
[[ "$JOB" =~ ^[0-9]+$ ]] || { echo "sbatch failed (fit): $OUTL"; exit 1; }
echo "$JOB" > $QP/fit/job_id.txt; echo "V2 fit $JOB $(date -Is)" >> $Q/job_ids.txt
R1_REMRUN=$RUNF setsid nohup "$HERE/lswo_pdb_watch.sh" "$JOB" "$QP/fit" > "$QP/fit/watcher.log" 2>&1 < /dev/null &
OUTT=$(rem "sbatch $tabf --dependency=afterok:$JOB $P/r5_pdb_tab.sbatch $RUNT $RUNF t5" 2>/dev/null); JT=${OUTT%%;*}
[[ "$JT" =~ ^[0-9]+$ ]] || { echo "sbatch failed (tab): $OUTT; fit $JOB already submitted"; exit 1; }
echo "$JT" > $QP/tab/job_id.txt; echo "V2 tab $JT afterok:$JOB $(date -Is)" >> $Q/job_ids.txt
R1_REMRUN=$RUNT R1_SUBS="tab_t5" setsid nohup "$HERE/r3_pdb_tab_watch.sh" "$JT" "$QP/tab" > "$QP/tab/watcher.log" 2>&1 < /dev/null &
setsid nohup "$HERE/v_chain2.sh" > "$Q/readout/chain2.log" 2>&1 < /dev/null &
sleep 3   # setsid forks: record the re-parented pids (ppid 1), not $!
for X in "lswo_pdb_watch.sh $JOB:pdb2/fit" "r3_pdb_tab_watch.sh $JT:pdb2/tab" "v_chain2.sh:readout"; do
  pat=${X%%:*}; dir=${X##*:}; W=$(pgrep -f "$pat" | head -1); f=$Q/$dir/watcher_pid.txt; [ $dir = readout ] && f=$Q/readout/chain2_pid.txt
  echo "$W" > $f; echo "$pat: pid $W ppid=$(ps -o ppid= -p $W 2>/dev/null)"; done
echo "V2: fit=$JOB tab=$JT"
rem "sacct -u \$USER -S now-1hour -X -n -o JobID,JobName%20,State,Reason%30" 2>/dev/null || true
