#!/bin/bash
# Round 4 (PLAN_pfaffian.md "Round 4"), theorie A40 race copy of launch_r4_pdb.sh (which must be submitted first).  Modes: (none) STAGE ONLY; --test: stage + `sbatch --test-only` of the A40 jobs; --submit.
# --submit: sbatch fit (pf_r4, 13:00:00, fit_sec 42500), table job (pf_r4tab, 10:00:00, afterok: theta_4 = mean of the 7 iterates + tab_t4 + 6 checkpoint tables), one durable watcher per job
# (lswo_a40_watch.sh / r4_tab_watch.sh -> job_finish.sh --wake none) and the race watcher r4_race_watch.sh (cancels the copy that starts second, with its follow-on job).
# Spec = pfaffian_r3/fit/spec_r1.json (A40 round-3 spec) with ONLY: code_sha256.cx_r1fit.py (41f766cc), target_la (T_4), warm_theta (theta_bar), fit_overrides.save_theta_at [600..1350].  Diff: $Q/spec_diff_a40.txt.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
Q3=/scratch/a/A.Otaifi/ansatz_runs/pfaffian_r3; QA=/scratch/a/A.Otaifi/ansatz_runs/pfaffian_r3avg; Q=/scratch/a/A.Otaifi/ansatz_runs/pfaffian_r4
L=/project/theorie/a/A.Otaifi/Chatty/runs
MODE=${1:-stage}; CODE="cx_r1.py cx_r1fit.py cx_lib.py rbmpp_lib.py"; FS=42500; TMF=13:00:00; TMT=10:00:00; SUBS="t4 600 750 900 1050 1200 1350"
NEW_R1FIT=41f766ccaa71396d3fc12f57c027bd45bb2a08bafb042eefb9bbb4ba85175fc5
T4SHA=$(cut -d' ' -f1 $Q/target_T4_sG.npy.sha256); BARSHA=1ec8f71516a98e96d414f763cf5baa15359e492467bf30bd2f44fa6b09c0d3f6
chk() { h=$(sha256sum "$1" | cut -d' ' -f1); [ "$h" = "$2" ] || { echo "sha256 mismatch $1"; exit 1; }; echo "input ok ${h:0:12} $1"; }
[ "$MODE" = --submit ] && [ -e "$Q/fit/job_id.txt" ] && { echo "fit/job_id.txt exists: already submitted"; exit 1; }
python3 -c "
import json; d=json.load(open('$Q/prep_result.json')); assert d.get('done') and d['E_T4_sG']['pass_1e-6_rel'] and d['T4_sha256']=='$T4SHA', 'prep control not passed'"
chk $Q/target_T4_sG.npy $T4SHA
chk $QA/theta_bar.npy $BARSHA
chk $HERE/cx_r1fit.py $NEW_R1FIT
chk $HERE/cx_r1.py 5ebae9d83497d0805137ae0487a0878b53afcb8fbf93f79d6213bd50eb3ab19d
chk $HERE/cx_lib.py f320a0954d53c6155cc462d33b91602de7478a5e0cfdc2ee74d5a125cf281a45
chk $HERE/rbmpp_lib.py 6d09da0665a4d13a2420c0c0ad9ae14862ea111e07b0122ccef23a1f04782a4a
chk /scratch/a/A.Otaifi/ansatz_runs/pfaffian_z/s1.npy 69adeaca7c4603a21d15313c037c70d4c85b379a3fbe0341cacf7a264336d84a
chk $L/compress6/runs/o2b_17431886/round_1/spec_fit.json 3a89451e312a5f55fd2413a69d4d62665743a45707ac6ec692ee0b067615ec12
chk $L/rbmpp6/runs/rpT1_pp_17163493/params_final.npz e765e8fada11f3cc147c9d864c3720207cc022bb95275bc43318e5d41886b424
chk $L/rbmpp6/runs/rpTabPP_17164048/sym_tables_PP.npz d8a4e96c9a03b7ac18c0ef5bbaa9f4bcfe7632b34e01400cc2e6fb2f7253a934
chk $L/stall6/data/psi0_6x6_table.npz e2d0c838d7983f29e61805c9feef67d3aea24584058c2b703f9fbd8c5da1660d
chk $L/stall6/data/sym_tables.npz 1f8576537870f2737ad6200ea709da953fb214d3a24b65677efe8248254944d9
chk $L/stall6/csr/norb.npy 9f37ec985d777ec56bfe82e249d0b7a2601f1cee1d518fcfb08c0c8e5ecf5730
chk $L/compress6/runs/refresh_17409419/s2.npy fd4f1ef0638dc726f9652f81bbf5751eadc796601393fb4dddf5f1f031154451
for f in up_codes.npy up_inc.npy up_indices.npy meta.json; do [ -s $L/stall6/csr/$f ] || { echo "missing csr/$f"; exit 1; }; done
RUN=$Q/fit; mkdir -p "$RUN/code" "$Q/tabs" "$Q/readout"
for f in $CODE; do cp "$HERE/$f" "$RUN/code/$f"; done
( cd "$RUN/code" && sha256sum $CODE ) > "$RUN/code_sha256.txt"
python3 - "$Q3/fit/spec_r1.json" "$Q" "$FS" "$NEW_R1FIT" "$QA" <<'PY'
import json, sys, os
src, Q, fs, newsha, QA = sys.argv[1:6]; sp = json.load(open(src))
assert sp['fit_overrides'] == {'seed': 2, 'min_steps': 1500, 'save_theta_at': [900, 1200]} and sp['max_steps'] == 1500 and sp['fit_sec'] == float(fs) and sp['sign_path'].endswith('pfaffian_lswo2/s_G.npy')
sp['code_sha256']['cx_r1fit.py'] = newsha
sp.update(target_la=f'{Q}/target_T4_sG.npy', warm_theta=f'{QA}/theta_bar.npy')
sp['fit_overrides'] = {'seed': 2, 'min_steps': 1500, 'save_theta_at': [600, 750, 900, 1050, 1200, 1350]}
json.dump(sp, open(f'{Q}/fit/spec_r1.json', 'w'), indent=1)
for N in 't4 600 750 900 1050 1200 1350'.split():
    t = json.loads(json.dumps(sp)); t.update(warm_theta=f'{Q}/tabs/theta_4.npy' if N == 't4' else f'{Q}/fit/round_1/theta_step{N}.npy', fit_sec=1.0, max_steps=0); t['fit_overrides'] = {'seed': 2}
    os.makedirs(f'{Q}/tabs/tab_{N}', exist_ok=True); json.dump(t, open(f'{Q}/tabs/tab_{N}/spec_r1.json', 'w'), indent=1)
PY
for k in target_la sign_path warm_theta; do p=$(python3 -c "import json;print(json.load(open('$RUN/spec_r1.json'))['$k'])"); [ -s "$p" ] || { echo "missing $k $p"; exit 1; }; done
diff -u $Q3/fit/spec_r1.json $RUN/spec_r1.json > $Q/spec_diff_a40.txt || true
for D in fit tabs; do { echo "# Round 4 theorie A40 ($D job, run dir $Q/$D); full brief $Q/ANALYSIS_BRIEF.md"; sed -n '2,$p' "$Q/ANALYSIS_BRIEF.md"; } > "$Q/$D/ANALYSIS_BRIEF.md"; done
echo "staged A40: fit $RUN (wall $TMF fit_sec $FS), tabs $Q/tabs (wall $TMT); spec diff $Q/spec_diff_a40.txt"
FITF="--parsable --no-requeue --job-name=pf_r4 --time=$TMF --output=$RUN/slurm-%j.out"
TABF="--parsable --no-requeue --job-name=pf_r4tab --time=$TMT --output=$Q/tabs/slurm-%j.out"
if [ "$MODE" = --test ]; then
  echo "--- test-only fit"; sbatch --test-only $FITF "$HERE/r3_a40.sbatch" "$RUN" 2>&1
  echo "--- test-only tabs (no dependency)"; sbatch --test-only $TABF "$HERE/r4_tab_a40.sbatch" "$Q" 2>&1
fi
[ "$MODE" = --submit ] || exit 0
for f in $Q/pdb_fit/job_id.txt $Q/pdb_tabs/job_id.txt; do [ -s "$f" ] || { echo "Paderborn job id missing ($f): submit launch_r4_pdb.sh first"; exit 1; }; done
JOB=$(sbatch $FITF "$HERE/r3_a40.sbatch" "$RUN")
[[ "$JOB" =~ ^[0-9]+$ ]] || { echo "sbatch failed (fit): $JOB"; exit 1; }
echo "$JOB" > "$RUN/job_id.txt"; echo "fit $JOB $(date -Is)" >> "$Q/job_ids.txt"
setsid nohup "$HERE/lswo_a40_watch.sh" "$JOB" "$RUN" > "$RUN/watcher.log" 2>&1 < /dev/null &
echo $! > "$RUN/watcher_pid.txt"
JT=$(sbatch $TABF --dependency=afterok:$JOB "$HERE/r4_tab_a40.sbatch" "$Q")
[[ "$JT" =~ ^[0-9]+$ ]] || { echo "sbatch failed (tabs): $JT; fit $JOB already submitted"; exit 1; }
echo "$JT" > "$Q/tabs/job_id.txt"; echo "tabs $JT afterok:$JOB $(date -Is)" >> "$Q/job_ids.txt"
setsid nohup "$HERE/r4_tab_watch.sh" "$JT" "$Q/tabs" > "$Q/tabs/watcher.log" 2>&1 < /dev/null &
echo $! > "$Q/tabs/watcher_pid.txt"
setsid nohup "$HERE/r4_race_watch.sh" > "$Q/race.log" 2>&1 < /dev/null &
echo $! > "$Q/race_pid.txt"
echo "race_pid=$(cat $Q/race_pid.txt) ppid=$(ps -o ppid= -p $(cat $Q/race_pid.txt))"
echo "fit=$JOB watcher_pid=$(cat $RUN/watcher_pid.txt) ppid=$(ps -o ppid= -p $(cat $RUN/watcher_pid.txt)); tabs=$JT watcher_pid=$(cat $Q/tabs/watcher_pid.txt) ppid=$(ps -o ppid= -p $(cat $Q/tabs/watcher_pid.txt))"
sacct -j $JOB,$JT -X -n -o JobID,State,Reason%30
