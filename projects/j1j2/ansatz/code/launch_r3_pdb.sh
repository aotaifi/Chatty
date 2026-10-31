#!/bin/bash
# Round 3 under s_G, Paderborn H100 copy (race partner of launch_r3_a40.sh).  Default: STAGE ONLY (rsync + sha-check code/data, write specs/briefs, sbatch --test-only is run separately).
# --submit: sbatch the fit (pf_r3, 03:00:00, gpu_h100, hpc-prf-ppdmrg, --no-requeue) and the table job (pf_r3tab, 00:30:00, --dependency=afterok:fit), record ids, arm one durable watcher per job
#   (lswo_pdb_watch.sh for the fit; r3_pdb_tab_watch.sh for the tables).  Needs the VPN.  Then run launch_r3_a40.sh --submit (arms the race watcher r3_race_watch.sh, which reads these ids).
# Spec = R2G seed-2 Paderborn spec (pfaffian_r2g/run_s2/spec_r1.json) with: target_la = data/r3/target_T3_sG.npy, warm_theta = data/r3/B2_theta.npy (= run_s2/round_1/theta.npy), fit_sec 9000, max_steps 1500,
#   fit_overrides {seed 2, min_steps 1500, save_theta_at [900,1200]}, cx_r1fit.py sha = the A40 copy's.  Code dir on Paderborn: codeR3 (codeR1 untouched).  Table specs: max_steps 0, warm_theta = fit/round_1/theta_step{N}.npy.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
LL=/scratch/a/A.Otaifi/ansatz_runs/pfaffian_r2g; LS=/scratch/a/A.Otaifi/ansatz_runs/pfaffian_lswo2; Q=/scratch/a/A.Otaifi/ansatz_runs/pfaffian_r3
P=/scratch/hpc-prf-ppdmrg/hpcalot/chatty_compress6; D=$P/data/lswo2; D3=$P/data/r3
SSHO="ssh -o BatchMode=yes -o HostName=login4"
rem() { $SSHO paderborn "export PATH=/opt/software/slurm/25.11.8-1/bin:\$PATH; $1"; }
TM=03:00:00; FS=9000; MS=1500
RUNF=$P/runs/r3_fit; RUNT=$P/runs/r3_tab; LF=$Q/pdb_fit; LT=$Q/pdb_tabs
CODE="cx_r1.py cx_r1fit.py cx_lib.py rbmpp_lib.py"
NEW_R1FIT=4f3103d1d4775f85b2764861b64c2521936fd751eccf5a63d24914a8ace3bc76
[ -e "$LF/job_id.txt" ] && { echo "$LF/job_id.txt exists: already submitted"; exit 1; }
[ -s "$Q/fit/code_sha256.txt" ] && [ -s "$Q/fit/code/cx_r1fit.py" ] || { echo "stage launch_r3_a40.sh first (code and brief)"; exit 1; }
chk() { # LOCALFILE REMOTEFILE
  l=$(sha256sum "$1" | cut -d' ' -f1); r=$(rem "sha256sum $2 2>/dev/null" | cut -d' ' -f1 || true)
  if [ "$l" != "$r" ]; then rsync -a -e "$SSHO" "$1" paderborn:$2; r=$(rem "sha256sum $2" | cut -d' ' -f1); fi
  [ "$l" = "$r" ] || { echo "sha256 mismatch $2"; exit 1; }; echo "input ok ${l:0:12} $2"; }
# 0. code: codeR3 = codeR1 (remote copy of the pinned files, st6_sector.py, vit_dt.py) with the new cx_r1fit.py; every file sha-checked against the local pins
[ "$(sha256sum $HERE/cx_r1fit.py | cut -d' ' -f1)" = "$NEW_R1FIT" ] || { echo "local cx_r1fit.py sha != $NEW_R1FIT"; exit 1; }
rem "mkdir -p $P/codeR3 $D3; cd $P/codeR1 && cp -p cx_r1.py cx_lib.py rbmpp_lib.py st6_sector.py vit_dt.py $P/codeR3/"
rsync -a -e "$SSHO" "$HERE/cx_r1fit.py" paderborn:$P/codeR3/
rem "cd $P/codeR3 && sha256sum $CODE" | diff - "$Q/fit/code_sha256.txt" > /dev/null || { echo "codeR3 sha256 differs from $Q/fit/code_sha256.txt"; exit 1; }
for f in st6_sector.py vit_dt.py; do l=$(sha256sum /project/theorie/a/A.Otaifi/Chatty/runs/compress6/code/$f | cut -d' ' -f1); r=$(rem "sha256sum $P/codeR3/$f" | cut -d' ' -f1); [ "$l" = "$r" ] || { echo "$f differs from compress6/code"; exit 1; }; echo "input ok ${l:0:12} codeR3/$f"; done
# 1. data (shared inputs sha-checked) and the round-3 inputs
chk /project/theorie/a/A.Otaifi/Chatty/runs/compress6/runs/o2b_17431886/round_1/spec_fit.json $P/data/o2b_r1/round_1_spec_fit.json
chk /project/theorie/a/A.Otaifi/Chatty/runs/rbmpp6/runs/rpT1_pp_17163493/params_final.npz $P/data/rpT1_pp_17163493/params_final.npz
chk /project/theorie/a/A.Otaifi/Chatty/runs/rbmpp6/runs/rpTabPP_17164048/sym_tables_PP.npz $P/data/rpTabPP_17164048/sym_tables_PP.npz
chk /project/theorie/a/A.Otaifi/Chatty/runs/stall6/data/psi0_6x6_table.npz $P/data/psi0_6x6_table.npz
chk /project/theorie/a/A.Otaifi/Chatty/runs/stall6/data/sym_tables.npz $P/data/sym_tables.npz
chk /project/theorie/a/A.Otaifi/Chatty/runs/stall6/csr/norb.npy $P/data/csr/norb.npy
chk /project/theorie/a/A.Otaifi/Chatty/runs/compress6/runs/refresh_17409419/s2.npy $P/data/refresh_17409419/s2.npy
rem "cd $P/data/csr; ls -l meta.json up_indices.npy up_codes.npy up_inc.npy" > /dev/null
chk $LS/s_G.npy $D/s_G.npy
chk $Q/target_T3_sG.npy $D3/target_T3_sG.npy
chk $LL/run_s2/round_1/theta.npy $D3/B2_theta.npy
python3 -c "import numpy as np; a=np.load('$LL/run_s2/round_1/theta.npy'); assert a.size==2740 and a.dtype==np.float64"
mkdir -p "$LF" "$LT/tab_900" "$LT/tab_1200"; rem "mkdir -p $RUNF $RUNT/tab_900 $RUNT/tab_1200"
cp "$Q/code_sha256.txt" "$LF/"
python3 - "$LL/run_s2/spec_r1.json" "$LF" "$LT" "$P" "$D" "$D3" "$FS" "$MS" "$NEW_R1FIT" "$RUNF" <<'PY'
import json, sys, os
src, LF, LT, P, D, D3, fs, ms, newsha, RUNF = sys.argv[1:11]; sp = json.load(open(src))
assert sp['fit_overrides'] == {'seed': 2} and sp['sign_path'] == f'{D}/s_G.npy' and sp['o2b_fit_spec'].startswith(P)
sp['code_sha256']['cx_r1fit.py'] = newsha
sp.update(target_la=f'{D3}/target_T3_sG.npy', warm_theta=f'{D3}/B2_theta.npy', fit_sec=float(fs), max_steps=int(ms))
sp['fit_overrides'] = {'seed': 2, 'min_steps': int(ms), 'save_theta_at': [900, 1200]}
json.dump(sp, open(f'{LF}/spec_r1.json', 'w'), indent=1)
for N in (900, 1200):
    t = dict(sp); t.update(warm_theta=f'{RUNF}/round_1/theta_step{N}.npy', fit_sec=1.0, max_steps=0, fit_overrides={'seed': 2})
    json.dump(t, open(f'{LT}/tab_{N}/spec_r1.json', 'w'), indent=1)
PY
for D_ in fit tabs; do
  [ $D_ = fit ] && { LOC=$LF; REM=$RUNF; GATE=$Q/pdb_fit; } || { LOC=$LT; REM=$RUNT; GATE=$Q/pdb_tabs; }
  { echo "# Round 3 under s_G, Paderborn H100 copy ($D_ job; remote $REM, local $LOC); full brief $Q/ANALYSIS_BRIEF.md.  Race partner: the theorie A40 copy ($Q/fit, $Q/tabs); the first to log a fit step wins, the loser (fit and table job) is cancelled."
    sed -n '2,$p' "$Q/ANALYSIS_BRIEF.md" | sed "s#r3_gate.py [^ ]*/fit  (fit job) and  r3_gate.py [^ ]*/tabs  (table job)#r3_gate.py $GATE#; s#fit_sec 42500#fit_sec 42500 (A40; Paderborn copy fit_sec $FS, wall $TM, ~3 s/step)#"; } > "$LOC/ANALYSIS_BRIEF.md"
  rsync -a -e "$SSHO" "$LOC/ANALYSIS_BRIEF.md" paderborn:$REM/
done
rsync -a -e "$SSHO" "$LF/spec_r1.json" paderborn:$RUNF/
for N in 900 1200; do rsync -a -e "$SSHO" "$LT/tab_$N/spec_r1.json" paderborn:$RUNT/tab_$N/; done
rsync -a -e "$SSHO" "$HERE/r3_pdb.sbatch" "$HERE/r3_pdb_tab.sbatch" paderborn:$P/
echo "staged Paderborn: fit $RUNF (local $LF; wall $TM fit_sec $FS max_steps $MS), tabs $RUNT (local $LT; wall 00:30:00)"
[ "${1:-}" = "--submit" ] || exit 0
for f in r3_pdb.sbatch r3_pdb_tab.sbatch; do rem "sha256sum $P/$f" | cut -d' ' -f1 | grep -qx "$(sha256sum "$HERE/$f" | cut -d' ' -f1)" || { echo "$f sha256 differs after copy"; exit 1; }; done
OUTL=$(rem "sbatch --parsable --no-requeue --job-name=pf_r3 --time=$TM --output=$RUNF/slurm-%j.out $P/r3_pdb.sbatch $RUNF" 2>/dev/null); JOB=${OUTL%%;*}
[[ "$JOB" =~ ^[0-9]+$ ]] || { echo "sbatch failed (fit): $OUTL"; exit 1; }
echo "$JOB" > "$LF/job_id.txt"; echo "pdb_fit $JOB $(date -Is)" >> "$Q/job_ids.txt"
R1_REMRUN=$RUNF setsid nohup "$HERE/lswo_pdb_watch.sh" "$JOB" "$LF" > "$LF/watcher.log" 2>&1 < /dev/null &
echo "$!" > "$LF/watcher_pid.txt"
OUTT=$(rem "sbatch --parsable --no-requeue --job-name=pf_r3tab --time=00:30:00 --dependency=afterok:$JOB --output=$RUNT/slurm-%j.out $P/r3_pdb_tab.sbatch $RUNT $RUNF" 2>/dev/null); JT=${OUTT%%;*}
[[ "$JT" =~ ^[0-9]+$ ]] || { echo "sbatch failed (tabs): $OUTT; fit $JOB already submitted"; exit 1; }
echo "$JT" > "$LT/job_id.txt"; echo "pdb_tabs $JT afterok:$JOB $(date -Is)" >> "$Q/job_ids.txt"
R1_REMRUN=$RUNT R1_SUBS="tab_900 tab_1200" setsid nohup "$HERE/r3_pdb_tab_watch.sh" "$JT" "$LT" > "$LT/watcher.log" 2>&1 < /dev/null &
echo "$!" > "$LT/watcher_pid.txt"
echo "pdb fit=$JOB watcher_pid=$(cat $LF/watcher_pid.txt) ppid=$(ps -o ppid= -p $(cat $LF/watcher_pid.txt)); tabs=$JT watcher_pid=$(cat $LT/watcher_pid.txt) ppid=$(ps -o ppid= -p $(cat $LT/watcher_pid.txt))"
rem "sacct -j $JOB,$JT -X -n -o JobID,State,Reason%30" 2>/dev/null; rem "squeue --start -u \$USER" 2>/dev/null || true
