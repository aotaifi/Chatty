#!/bin/bash
# Round 3 under s_G (PLAN_pfaffian.md "Round 3 under s_G"), theorie A40 copy (race partner of launch_r3_pdb.sh, which must be submitted first).  Default: STAGE ONLY.   --submit: sbatch the fit (pf_r3, 13:00:00), sbatch the table job
# (pf_r3tab, 02:00:00, --dependency=afterok:fit), arm one durable watcher per job (lswo_a40_watch.sh / r3_tab_watch.sh -> job_finish.sh --wake none) and the race watcher r3_race_watch.sh, record ids in pfaffian_r3/job_ids.txt.
# Spec = R2G seed-2 A40 spec (pfaffian_r2g/run_s2_a40/spec_r1.json) with: target_la = pfaffian_r3/target_T3_sG.npy, warm_theta = R2G seed-2 theta (B_2), fit_sec 42500, max_steps 1500,
#   fit_overrides {seed 2, min_steps 1500, save_theta_at [900,1200]}, code_sha256 of cx_r1fit.py (new: save_theta_at).  Table specs: same, max_steps 0, warm_theta = fit/round_1/theta_step{N}.npy, fit_sec 1.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
L=/project/theorie/a/A.Otaifi/Chatty/runs; L0=/scratch/a/A.Otaifi/ansatz_runs/pfaffian_lswo; LL=/scratch/a/A.Otaifi/ansatz_runs/pfaffian_r2g; Q=/scratch/a/A.Otaifi/ansatz_runs/pfaffian_r3
CODE="cx_r1.py cx_r1fit.py cx_lib.py rbmpp_lib.py"; FS=42500; TM=13:00:00; MS=1500
NEW_R1FIT=4f3103d1d4775f85b2764861b64c2521936fd751eccf5a63d24914a8ace3bc76
chk() { h=$(sha256sum "$1" | cut -d' ' -f1); [ "$h" = "$2" ] || { echo "sha256 mismatch $1"; exit 1; }; echo "input ok ${h:0:12} $1"; }
[ -e "$Q/fit/job_id.txt" ] && { echo "fit/job_id.txt exists: already submitted"; exit 1; }
chk $Q/target_T3_sG.npy 4e2d542e3e17ae411c4dc401804e71ca27d6841c67ccd026c2c9addc170fc9da
chk $LL/run_s2/round_1/theta.npy 7166d084c054458ab87726fb85284e395a42e54e7f425d5bf3e609d46e16663a
chk $LL/run_s2/round_1/la_table.npy b00dd2dadc694c8629808feab6a0a11838be1c3e0c010a97ead3951dfcf487a0
chk $HERE/cx_r1fit.py $NEW_R1FIT
chk /scratch/a/A.Otaifi/ansatz_runs/pfaffian_z/s1.npy 69adeaca7c4603a21d15313c037c70d4c85b379a3fbe0341cacf7a264336d84a
chk $L/compress6/runs/o2b_17431886/round_1/spec_fit.json 3a89451e312a5f55fd2413a69d4d62665743a45707ac6ec692ee0b067615ec12
chk $L/rbmpp6/runs/rpT1_pp_17163493/params_final.npz e765e8fada11f3cc147c9d864c3720207cc022bb95275bc43318e5d41886b424
chk $L/rbmpp6/runs/rpTabPP_17164048/sym_tables_PP.npz d8a4e96c9a03b7ac18c0ef5bbaa9f4bcfe7632b34e01400cc2e6fb2f7253a934
chk $L/stall6/data/psi0_6x6_table.npz e2d0c838d7983f29e61805c9feef67d3aea24584058c2b703f9fbd8c5da1660d
chk $L/stall6/data/sym_tables.npz 1f8576537870f2737ad6200ea709da953fb214d3a24b65677efe8248254944d9
chk $L/stall6/csr/norb.npy 9f37ec985d777ec56bfe82e249d0b7a2601f1cee1d518fcfb08c0c8e5ecf5730
chk $L/compress6/runs/refresh_17409419/s2.npy fd4f1ef0638dc726f9652f81bbf5751eadc796601393fb4dddf5f1f031154451
for f in up_codes.npy up_inc.npy up_indices.npy meta.json; do [ -s $L/stall6/csr/$f ] || { echo "missing csr/$f"; exit 1; }; done
RUN=$Q/fit; mkdir -p "$RUN/code" "$Q/tabs"
for f in cx_r1.py cx_lib.py rbmpp_lib.py; do git -C "$HERE" show af662d3:projects/j1j2/ansatz/code/$f > "$RUN/code/$f"; done
cp "$HERE/cx_r1fit.py" "$RUN/code/cx_r1fit.py"
( cd "$RUN/code" && sha256sum $CODE ) > "$RUN/code_sha256.txt"
DD=$(diff "$RUN/code_sha256.txt" "$L0/run/code_sha256.txt" || true)
[ "$(echo "$DD" | grep -c '^[<>]')" = 2 ] || { echo "code sha256 must differ from the L-SWO run in exactly the cx_r1fit.py line"; exit 1; }
echo "$DD" | grep -q cx_r1fit.py || { echo "differing line is not cx_r1fit.py"; exit 1; }
cp "$RUN/code_sha256.txt" "$Q/code_sha256.txt"; ( cd "$HERE" && git diff -- cx_r1fit.py ) > "$Q/cx_r1fit.diff"
python3 - "$LL/run_s2_a40/spec_r1.json" "$Q" "$FS" "$MS" "$NEW_R1FIT" "$LL" <<'PY'
import json, sys, os
src, Q, fs, ms, newsha, LL = sys.argv[1:7]; sp = json.load(open(src))
sp['code_sha256']['cx_r1fit.py'] = newsha
sp.update(target_la=f'{Q}/target_T3_sG.npy', warm_theta=f'{LL}/run_s2/round_1/theta.npy', fit_sec=float(fs), max_steps=int(ms))
assert sp['sign_path'].endswith('pfaffian_lswo2/s_G.npy') and sp['fit_overrides'] == {'seed': 2}
sp['fit_overrides'] = {'seed': 2, 'min_steps': int(ms), 'save_theta_at': [900, 1200]}
json.dump(sp, open(f'{Q}/fit/spec_r1.json', 'w'), indent=1)
for N in (900, 1200):                      # table-only specs: max_steps 0, warm_theta = checkpoint, no save, no min_steps
    t = dict(sp); t.update(warm_theta=f'{Q}/fit/round_1/theta_step{N}.npy', fit_sec=1.0, max_steps=0, fit_overrides={'seed': 2})
    os.makedirs(f'{Q}/tabs/tab_{N}', exist_ok=True); json.dump(t, open(f'{Q}/tabs/tab_{N}/spec_r1.json', 'w'), indent=1)
PY
for k in target_la sign_path warm_theta; do p=$(python3 -c "import json;print(json.load(open('$RUN/spec_r1.json'))['$k'])"); [ -s "$p" ] || { echo "missing $k $p"; exit 1; }; done
for D in fit tabs; do { echo "# Round 3 under s_G ($D job, run dir $Q/$D); full brief $Q/ANALYSIS_BRIEF.md"; sed -n '2,$p' "$Q/ANALYSIS_BRIEF.md" | sed "s#r3_gate.py [^ ]*/fit  (fit job) and  r3_gate.py [^ ]*/tabs  (table job)#r3_gate.py $Q/$D#"; } > "$Q/$D/ANALYSIS_BRIEF.md"; done
echo "staged: fit $RUN (wall $TM fit_sec $FS max_steps $MS), tabs $Q/tabs (wall 02:00:00)"
[ "${1:-}" = "--submit" ] || exit 0
for f in $Q/pdb_fit/job_id.txt $Q/pdb_tabs/job_id.txt; do [ -s "$f" ] || { echo "Paderborn job id missing ($f): submit launch_r3_pdb.sh first"; exit 1; }; done
JOB=$(sbatch --parsable --no-requeue --job-name=pf_r3 --time=$TM --output=$RUN/slurm-%j.out "$HERE/r3_a40.sbatch" "$RUN")
[[ "$JOB" =~ ^[0-9]+$ ]] || { echo "sbatch failed (fit): $JOB"; exit 1; }
echo "$JOB" > "$RUN/job_id.txt"; echo "fit $JOB $(date -Is)" >> "$Q/job_ids.txt"
setsid nohup "$HERE/lswo_a40_watch.sh" "$JOB" "$RUN" > "$RUN/watcher.log" 2>&1 < /dev/null &
echo $! > "$RUN/watcher_pid.txt"
JT=$(sbatch --parsable --no-requeue --job-name=pf_r3tab --time=02:00:00 --dependency=afterok:$JOB --output=$Q/tabs/slurm-%j.out "$HERE/r3_tab_a40.sbatch" "$Q")
[[ "$JT" =~ ^[0-9]+$ ]] || { echo "sbatch failed (tabs): $JT; fit $JOB already submitted"; exit 1; }
echo "$JT" > "$Q/tabs/job_id.txt"; echo "tabs $JT afterok:$JOB $(date -Is)" >> "$Q/job_ids.txt"
setsid nohup "$HERE/r3_tab_watch.sh" "$JT" "$Q/tabs" > "$Q/tabs/watcher.log" 2>&1 < /dev/null &
echo $! > "$Q/tabs/watcher_pid.txt"
setsid nohup "$HERE/r3_race_watch.sh" > "$Q/race.log" 2>&1 < /dev/null &
echo $! > "$Q/race_pid.txt"
echo "race_pid=$(cat $Q/race_pid.txt) ppid=$(ps -o ppid= -p $(cat $Q/race_pid.txt))"
echo "fit=$JOB watcher_pid=$(cat $RUN/watcher_pid.txt) ppid=$(ps -o ppid= -p $(cat $RUN/watcher_pid.txt)); tabs=$JT watcher_pid=$(cat $Q/tabs/watcher_pid.txt) ppid=$(ps -o ppid= -p $(cat $Q/tabs/watcher_pid.txt))"
sacct -j $JOB,$JT -X -n -o JobID,State,Reason%30
