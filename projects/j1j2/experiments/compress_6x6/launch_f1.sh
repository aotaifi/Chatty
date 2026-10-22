#!/bin/bash
# Test F1 launcher on one Paderborn H100 (README compress_6x6 "Test F1"): ONE round-1 fit of the O2b loop, tax = E_FN(hop(a_2)) - E_FN(hop(phi_1)).
#   ARM=ctl1  control 1: window 3, 80 steps (O2b round-1 count), also checks hop(O2b a_2); tax must reproduce 5.93e-6 within 5%, hop(phi_1) 4.951e-6
#   ARM=ctl2  control 2: window 5 with the outer ring frozen, 80 steps; must equal ctl1 (tax 1e-3 rel, log-amplitudes on 4096 orbits 1e-10; cx_f1_compare.py)
#   ARM=ctl3  control 3: window 5 free, 20 steps, rate only (s/step); ARM=brk: broken embedding, 1 step, rate only (step-0 held-out loss must differ from 0.0436815)
#   ARM=A     production arm A: window 3, fit budget FIT_SEC (10800 s)       ARM=B  production arm B: window 5, outer ring free, same budget
#   launch_f1.sh            stage only (default): code (sha256-pinned), O2b round-1 inputs, spec, local run dir + ANALYSIS_BRIEF.md
#   launch_f1.sh --submit   stage, then sbatch on gpu_h100 (account hpc-prf-ppdmrg, --no-requeue) and arm the durable watcher (cx_f1_pdb_watch.sh)
# Needs the VPN (paderborn/README.md): ssh -o HostName=login4 paderborn.  Env: ARM (required), SBATCH_TIME (A, B: required; controls default 00:30:00), FIT_SEC (A, B: 10800).
set -euo pipefail
ARM=${ARM:?set ARM=ctl1|ctl2|ctl3|brk|A|B}
HERE=$(cd "$(dirname "$0")" && pwd)
RUNS=/project/theorie/a/A.Otaifi/chatty_compress6/runs
O2B=$RUNS/o2b_17431886                                                  # O2b: round_0/theta.npy (a_1), round_1/{target_la.npy, la_table.npy, spec_fit.json}
NAME=f1_$ARM; LOC=$RUNS/$NAME
P=/scratch/hpc-prf-ppdmrg/hpcalot/chatty_compress6; RUN=$P/runs/$NAME; D=$P/data/o2b_r1
SSHO="ssh -o BatchMode=yes -o HostName=login4"
rem() { $SSHO paderborn "export PATH=/opt/software/slurm/25.11.8-1/bin:\$PATH; $1"; }
case $ARM in
  ctl1) WIN=3; FRZ=false; EXTRA='"max_steps": 80, "o2b_table": "@D@/round_1_la_table.npy", "expect_tax": 5.93021221797743e-06'; FS=${FIT_SEC:-3000}; TM=${SBATCH_TIME:-00:30:00};;
  ctl2) WIN=5; FRZ=true;  EXTRA='"max_steps": 80'; FS=${FIT_SEC:-3000}; TM=${SBATCH_TIME:-00:30:00};;
  ctl1s) WIN=3; FRZ=false; EXTRA='"max_steps": 10'; FS=3000; TM=${SBATCH_TIME:-00:20:00};;   # control 2 at 10 steps (before roundoff amplification)
  ctl2s) WIN=5; FRZ=true;  EXTRA='"max_steps": 10'; FS=3000; TM=${SBATCH_TIME:-00:20:00};;
  ctl3) WIN=5; FRZ=false; EXTRA='"max_steps": 20, "rate_only": true'; FS=${FIT_SEC:-3000}; TM=${SBATCH_TIME:-00:20:00};;
  brk)  WIN=5; FRZ=true;  EXTRA='"max_steps": 1, "rate_only": true, "break_embed": true'; FS=${FIT_SEC:-3000}; TM=${SBATCH_TIME:-00:15:00};;
  A)    WIN=3; FRZ=false; EXTRA='"expect_tax": null'; FS=${FIT_SEC:-10800}; TM=${SBATCH_TIME:?set SBATCH_TIME for A};;
  B)    WIN=5; FRZ=false; EXTRA='"expect_tax": null'; FS=${FIT_SEC:-10800}; TM=${SBATCH_TIME:?set SBATCH_TIME for B};;
  *) echo "bad ARM"; exit 1;;
esac
mkdir -p "$LOC"
# 1. code: pinned by sha256 (copy of the Paderborn code/ dir once, then the F1 files + rbmpp_lib, compared after the copy)
rem "[ -d $P/codeF1 ] || cp -a $P/code $P/codeF1; mkdir -p $D $RUN"
for f in cx_f1.py cx_f1fit.py cx_lib.py; do rsync -a -e "$SSHO" "$HERE/$f" paderborn:$P/codeF1/; done
rsync -a -e "$SSHO" "$HERE/../rbmpp_6x6/rbmpp_lib.py" paderborn:$P/codeF1/
( cd "$HERE" && sha256sum cx_f1.py cx_f1fit.py cx_lib.py ../rbmpp_6x6/rbmpp_lib.py | sed 's#../rbmpp_6x6/##' ) > "$LOC/code_sha256.txt"
rem "cd $P/codeF1 && sha256sum cx_f1.py cx_f1fit.py cx_lib.py rbmpp_lib.py" | diff - "$LOC/code_sha256.txt" > /dev/null || { echo "code sha256 differs after copy"; exit 1; }
# 2. O2b round-1 inputs (copied once; sizes printed, sha256 compared with the local O2b files)
rsync -a --size-only -e "$SSHO" "$O2B/round_1/target_la.npy" paderborn:$D/round_1_target_la.npy
rsync -a --size-only -e "$SSHO" "$O2B/round_1/la_table.npy" paderborn:$D/round_1_la_table.npy
rsync -a -e "$SSHO" "$O2B/round_0/theta.npy" paderborn:$D/round_0_theta.npy
rsync -a -e "$SSHO" "$O2B/round_1/spec_fit.json" paderborn:$D/round_1_spec_fit.json
rem "cd $D && ls -l && sha256sum round_0_theta.npy round_1_spec_fit.json" ; ( cd "$O2B" && sha256sum round_0/theta.npy round_1/spec_fit.json )
rem "cd $P/data; ls -l psi0_6x6_table.npz sym_tables.npz csr/{meta.json,norb.npy,up_indices.npy,up_codes.npy,up_inc.npy} refresh_17409419/s2.npy rpTabPP_17164048/sym_tables_PP.npz rpT1_pp_17163493/params_final.npz" >/dev/null
# 3. spec
python3 - "$LOC" "$P" "$D" "$WIN" "$FRZ" "$FS" "$EXTRA" <<'PY'
import json, sys
loc, P, D, win, frz, fs, extra = sys.argv[1:8]
sha = {l.split()[1]: l.split()[0] for l in open(f'{loc}/code_sha256.txt')}
sp = dict(refresh=f'{P}/data/refresh_17409419', pp_params=f'{P}/data/rpT1_pp_17163493/params_final.npz', pp_table=f'{P}/data/rpTabPP_17164048/sym_tables_PP.npz',
          code_sha256=sha, target_la=f'{D}/round_1_target_la.npy', warm_theta=f'{D}/round_0_theta.npy', o2b_fit_spec=f'{D}/round_1_spec_fit.json',
          window=int(win), freeze_outer=frz == 'true', fit_sec=float(fs))
sp.update({k: v for k, v in json.loads('{' + extra.replace('@D@', D) + '}').items() if v is not None})
json.dump(sp, open(f'{loc}/spec_f1.json', 'w'), indent=1)
PY
rsync -a -e "$SSHO" "$LOC/spec_f1.json" paderborn:$RUN/
# 4. brief
sed "s#@NAME@#$NAME#g; s#@ARM@#$ARM#g; s#@LOC@#$LOC#g; s#@REM@#$RUN#g; s#@WIN@#$WIN#g; s#@FS@#$FS#g" "$HERE/specs/f1_ANALYSIS_BRIEF.md" > "$LOC/ANALYSIS_BRIEF.md"
echo "staged $RUN (remote) and $LOC (local); arm $ARM window $WIN freeze_outer $FRZ fit_sec $FS time $TM"
[ "${1:-}" = "--submit" ] || exit 0
rsync -a -e "$SSHO" "$HERE/f1.sbatch" paderborn:$P/f1.sbatch
OUTL=$(rem "sbatch --parsable --job-name=f1$ARM --time=$TM --output=$RUN/slurm-%j.out $P/f1.sbatch $RUN" 2>/dev/null)
JOB=${OUTL%%;*}; [[ "$JOB" =~ ^[0-9]+$ ]] || { echo "sbatch failed: $OUTL"; exit 1; }
echo "$JOB" > "$LOC/job_id.txt"
STALL=${STALL:-2700} F1_REMRUN=$RUN setsid nohup "$HERE/cx_f1_pdb_watch.sh" "$JOB" "$LOC" > "$LOC/watcher.log" 2>&1 < /dev/null &
echo "job=$JOB run=$RUN watcher_pid=$!"
rem "sacct -j $JOB -X -n -o State,Reason%30" 2>/dev/null
