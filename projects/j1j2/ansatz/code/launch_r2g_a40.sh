#!/bin/bash
# R2G-conv, theorie full-A40 copies of both seeds (cip,inter; --gres=gpu:a40:1; wall 13:00:00); copy of launch_lswo2_a40.sh.  Default: STAGE ONLY.
# --submit: sbatch both seeds, record ids (run dirs, pfaffian_r2g/job_ids.txt), arm one durable job watcher per job (lswo_a40_watch.sh -> job_finish.sh --wake none) and one
# durable race watcher per seed (r2g_race_watch.sh K; first copy with a logged fit step wins, the other is cancelled; both started in the same poll -> keep Paderborn).
# REQUIRES the Paderborn jobs to be submitted first (launch_r2g.sh --submit), since the race watcher reads their ids from run_s<k>/job_id.txt.
# Spec = the Paderborn seed spec (pfaffian_r2g/run_s<k>/spec_r1.json, written by launch_r2g.sh) with local paths and fit_sec 43200 only (max_steps stays 3000, seed = k).
# Sizing: A40 ~30 s/step (L-SWO) -> ~1440 steps within fit_sec 43200 ~12 h against 13 h wall (~1600 steps nominal); the Paderborn copy (H100) runs the 3000-step budget in 3 h.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
L=/project/theorie/a/A.Otaifi/Chatty/runs; L0=/scratch/a/A.Otaifi/ansatz_runs/pfaffian_lswo; LL=/scratch/a/A.Otaifi/ansatz_runs/pfaffian_r2g; LS=/scratch/a/A.Otaifi/ansatz_runs/pfaffian_lswo2
CODE="cx_r1.py cx_r1fit.py cx_lib.py rbmpp_lib.py"; FS=43200; TM=13:00:00
chk() { h=$(sha256sum "$1" | cut -d' ' -f1); [ "$h" = "$2" ] || { echo "sha256 mismatch $1"; exit 1; }; echo "input ok ${h:0:12} $1"; }
for K in 1 2; do [ -e "$LL/run_s${K}_a40/job_id.txt" ] && { echo "run_s${K}_a40/job_id.txt exists: already submitted"; exit 1; }
  [ -s "$LL/run_s$K/spec_r1.json" ] || { echo "stage launch_r2g.sh first ($LL/run_s$K/spec_r1.json missing)"; exit 1; }; done
chk /scratch/a/A.Otaifi/ansatz_runs/pfaffian_z/s1.npy 69adeaca7c4603a21d15313c037c70d4c85b379a3fbe0341cacf7a264336d84a
chk $L/compress6/runs/o2b_17431886/round_1/spec_fit.json 3a89451e312a5f55fd2413a69d4d62665743a45707ac6ec692ee0b067615ec12
chk $L/rbmpp6/runs/rpT1_pp_17163493/params_final.npz e765e8fada11f3cc147c9d864c3720207cc022bb95275bc43318e5d41886b424
chk $L/rbmpp6/runs/rpTabPP_17164048/sym_tables_PP.npz d8a4e96c9a03b7ac18c0ef5bbaa9f4bcfe7632b34e01400cc2e6fb2f7253a934
chk $L/stall6/data/psi0_6x6_table.npz e2d0c838d7983f29e61805c9feef67d3aea24584058c2b703f9fbd8c5da1660d
chk $L/stall6/data/sym_tables.npz 1f8576537870f2737ad6200ea709da953fb214d3a24b65677efe8248254944d9
chk $L/stall6/csr/norb.npy 9f37ec985d777ec56bfe82e249d0b7a2601f1cee1d518fcfb08c0c8e5ecf5730
chk $L/compress6/runs/refresh_17409419/s2.npy fd4f1ef0638dc726f9652f81bbf5751eadc796601393fb4dddf5f1f031154451
for f in up_codes.npy up_inc.npy up_indices.npy meta.json; do [ -s $L/stall6/csr/$f ] || { echo "missing csr/$f"; exit 1; }; done
for K in 1 2; do
  RUN=$LL/run_s${K}_a40; PD=$LL/run_s$K; mkdir -p "$RUN/code"
  for f in $CODE; do git -C "$HERE" show af662d3:projects/j1j2/ansatz/code/$f > "$RUN/code/$f"; done
  ( cd "$RUN/code" && sha256sum $CODE ) > "$RUN/code_sha256.txt"
  diff "$RUN/code_sha256.txt" "$L0/run/code_sha256.txt" || { echo "code sha256 differs from the L-SWO run"; exit 1; }
  python3 - "$PD/spec_r1.json" "$RUN/spec_r1.json" "$FS" "$LS" <<'PY'
import json, sys
src, dst, fs, Q = sys.argv[1:5]; sp = json.load(open(src)); L = '/project/theorie/a/A.Otaifi/Chatty/runs'; R = '/scratch/a/A.Otaifi/ansatz_runs'
sp.update(refresh=f'{L}/compress6/runs/refresh_17409419', pp_params=f'{L}/rbmpp6/runs/rpT1_pp_17163493/params_final.npz',
          pp_table=f'{L}/rbmpp6/runs/rpTabPP_17164048/sym_tables_PP.npz', o2b_fit_spec=f'{L}/compress6/runs/o2b_17431886/round_1/spec_fit.json',
          target_la=f'{Q}/target_TG_sG.npy', sign_path=f'{Q}/s_G.npy',
          warm_theta=f'{R}/pfaffian_lswo/run/round_1/theta.npy', fit_sec=float(fs))
json.dump(sp, open(dst, 'w'), indent=1)
PY
  for k in target_la sign_path warm_theta; do p=$(python3 -c "import json;print(json.load(open('$RUN/spec_r1.json'))['$k'])"); [ -s "$p" ] || { echo "missing $k $p"; exit 1; }; done
  sed -e "s#^\# R2G-conv GPU fit, seed $K, Paderborn.*#\# R2G-conv, seed $K, full theorie A40 copy (run dir $RUN); race partner of the Paderborn copy $PD; spec = that copy's with local paths and fit_sec $FS#" \
      -e "s#max_steps 3000, fit_sec 9000, wall 03:00:00#max_steps 3000, fit_sec $FS (A40 ~30 s/step: ~1440 steps expected), wall $TM#" "$PD/ANALYSIS_BRIEF.md" > "$RUN/ANALYSIS_BRIEF.md"
  sed -i "s#python3 .*r1_gate.py [^ ]*#python3 $HERE/r1_gate.py $RUN#; s#Files: [^ ]*/cx_r1.json#Files: $RUN/cx_r1.json#" "$RUN/ANALYSIS_BRIEF.md"
  echo "staged seed $K A40: $RUN; wall $TM fit_sec $FS max_steps 3000"
done
[ "${1:-}" = "--submit" ] || exit 0
for K in 1 2; do [ -s "$LL/run_s$K/job_id.txt" ] || { echo "Paderborn job id missing for seed $K (submit launch_r2g.sh first)"; exit 1; }; done
for K in 1 2; do
  RUN=$LL/run_s${K}_a40
  JOB=$(sbatch --parsable --no-requeue --job-name=r2ga40_s$K --time=$TM --output=$RUN/slurm-%j.out "$HERE/r2g_a40.sbatch" "$RUN")
  [[ "$JOB" =~ ^[0-9]+$ ]] || { echo "sbatch failed (seed $K): $JOB"; exit 1; }
  echo "$JOB" > "$RUN/job_id.txt"; echo "s$K-a40 $JOB $(date -Is)" >> "$LL/job_ids.txt"
  setsid nohup "$HERE/lswo_a40_watch.sh" "$JOB" "$RUN" > "$RUN/watcher.log" 2>&1 < /dev/null &
  echo $! > "$RUN/watcher_pid.txt"
  setsid nohup "$HERE/r2g_race_watch.sh" $K > "$RUN/race.log" 2>&1 < /dev/null &
  echo $! > "$RUN/race_pid.txt"
  echo "seed=$K a40 job=$JOB watcher_pid=$(cat $RUN/watcher_pid.txt) race_pid=$(cat $RUN/race_pid.txt) ppid=$(ps -o ppid= -p $(cat $RUN/race_pid.txt))"
  sacct -j $JOB -X -n -o State,Reason%30
done
