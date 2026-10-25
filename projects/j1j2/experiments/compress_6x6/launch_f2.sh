#!/bin/bash
# Test F2 launcher on one Paderborn H100 (README compress_6x6 "Test F2"): ONE round-1 fit of the O2b loop with the violating-bond loss lambda_v L_v
# (cx_f2fit.py), tax = E_FN(hop(a_2)) - E_FN(hop(phi_1)) exactly as in F1.
#   ARM=ctl1  control 1: window 3, lambda_v = 0, 80 steps; must reproduce F1 arm A's 80-step control tax 5.959e-6 (f1_ctl1) within 2%
#   ARM=lvchk control 2: brute-force L_v at step 0 on the warm network (no fit; cx_f2_lvcheck.py): sampled bonds, README formula by enumeration, mislabelled flag
#   ARM=rC / rD  control 3: rate only (20 steps, no readout), window 3 / 5, with lambda_v (LAMBDA_V, a number or 'auto')
#   ARM=C  production arm C: window 3, lambda_v = LAMBDA_V (number, required), fit budget FIT_SEC (10800 s), the fit's own convergence rule
#   ARM=D  production arm D: window 5 (outer ring free), same lambda_v, same budget
#   launch_f2.sh            stage only (default): code (sha256-pinned), O2b round-1 inputs, spec, local run dir + ANALYSIS_BRIEF.md
#   launch_f2.sh --submit   stage, then sbatch on gpu_h100 (account hpc-prf-ppdmrg, --no-requeue, QoS express_hpcalot) and arm the durable watcher (cx_f2_pdb_watch.sh)
# Needs the VPN (paderborn/README.md): ssh -o HostName=login4 paderborn.
# Env: ARM (required), SBATCH_TIME (C, D: required; controls default), FIT_SEC (C, D: 10800), LAMBDA_V (C, D, rD: required; controls: auto), QOS (express_hpcalot).
set -euo pipefail
ARM=${ARM:?set ARM=ctl1|lvchk|rC|rD|C|D}
HERE=$(cd "$(dirname "$0")" && pwd)
RUNS=/project/theorie/a/A.Otaifi/Chatty/runs/compress6/runs
O2B=$RUNS/o2b_17431886                                                  # O2b: round_0/theta.npy (a_1), round_1/{target_la.npy, la_table.npy, spec_fit.json}
NAME=f2_$ARM; LOC=$RUNS/$NAME
P=/scratch/hpc-prf-ppdmrg/hpcalot/chatty_compress6; RUN=$P/runs/$NAME; D=$P/data/o2b_r1
SSHO="ssh -o BatchMode=yes -o HostName=login4"
QOS=${QOS:-express_hpcalot}
rem() { $SSHO paderborn "export PATH=/opt/software/slurm/25.11.8-1/bin:\$PATH; $1"; }
LV=${LAMBDA_V:-}
case $ARM in
  ctl1)  WIN=3; FRZ=false; LV=0; EXTRA='"max_steps": 80, "expect_tax": 5.958984511463446e-06'; FS=${FIT_SEC:-3000}; TM=${SBATCH_TIME:-00:30:00};;
  lvchk) WIN=3; FRZ=false; LV=${LV:-auto}; EXTRA='"rate_only": true, "lv_check": true, "lv_check_reps": 2000'; FS=3000; TM=${SBATCH_TIME:-00:30:00};;
  diag)  WIN=3; FRZ=false; LV=${LV:-auto}; EXTRA='"rate_only": true, "lv_diag": true'; FS=3000; TM=${SBATCH_TIME:-00:30:00};;
  pa)    WIN=3; FRZ=false; LV=${LV:?set LAMV}; EXTRA='"max_steps": 150, "rate_only": true, "old_weight": 0, "fit_overrides": {"min_steps": 150}'; FS=7000; TM=${SBATCH_TIME:-02:00:00};;
  pb)    WIN=3; FRZ=false; LV=${LV:?set LAMV}; EXTRA='"max_steps": 150, "rate_only": true, "fit_overrides": {"min_steps": 150}'; FS=7000; TM=${SBATCH_TIME:-02:00:00};;
  rC)    WIN=3; FRZ=false; LV=${LV:-auto}; EXTRA='"max_steps": 20, "rate_only": true'; FS=3000; TM=${SBATCH_TIME:-00:20:00};;
  rD)    WIN=5; FRZ=false; LV=${LV:?set LAMBDA_V for rD}; EXTRA='"max_steps": 20, "rate_only": true'; FS=3000; TM=${SBATCH_TIME:-00:20:00};;
  C)     WIN=3; FRZ=false; LV=${LV:?set LAMBDA_V for C}; EXTRA='"expect_tax": null'; FS=${FIT_SEC:-10800}; TM=${SBATCH_TIME:?set SBATCH_TIME for C};;
  D)     WIN=5; FRZ=false; LV=${LV:?set LAMBDA_V for D}; EXTRA='"expect_tax": null'; FS=${FIT_SEC:-10800}; TM=${SBATCH_TIME:?set SBATCH_TIME for D};;
  *) echo "bad ARM"; exit 1;;
esac
mkdir -p "$LOC"
# 1. code: pinned by sha256 (copy of the Paderborn code/ dir once, then the F2 files + rbmpp_lib, compared after the copy)
rem "[ -d $P/codeF2 ] || cp -a $P/code $P/codeF2; mkdir -p $D $RUN"
for f in cx_f2.py cx_f2fit.py cx_f2_lvcheck.py cx_lib.py; do rsync -a -e "$SSHO" "$HERE/$f" paderborn:$P/codeF2/; done
rsync -a -e "$SSHO" "$HERE/../rbmpp_6x6/rbmpp_lib.py" paderborn:$P/codeF2/
( cd "$HERE" && sha256sum cx_f2.py cx_f2fit.py cx_f2_lvcheck.py cx_lib.py ../rbmpp_6x6/rbmpp_lib.py | sed 's#../rbmpp_6x6/##' ) > "$LOC/code_sha256.txt"
rem "cd $P/codeF2 && sha256sum cx_f2.py cx_f2fit.py cx_f2_lvcheck.py cx_lib.py rbmpp_lib.py" | diff - "$LOC/code_sha256.txt" > /dev/null || { echo "code sha256 differs after copy"; exit 1; }
# 2. O2b round-1 inputs (copied once; sizes printed, sha256 compared with the local O2b files)
rsync -a --size-only -e "$SSHO" "$O2B/round_1/target_la.npy" paderborn:$D/round_1_target_la.npy
rsync -a --size-only -e "$SSHO" "$O2B/round_1/la_table.npy" paderborn:$D/round_1_la_table.npy
rsync -a -e "$SSHO" "$O2B/round_0/theta.npy" paderborn:$D/round_0_theta.npy
rsync -a -e "$SSHO" "$O2B/round_1/spec_fit.json" paderborn:$D/round_1_spec_fit.json
rem "cd $D && ls -l && sha256sum round_0_theta.npy round_1_spec_fit.json" ; ( cd "$O2B" && sha256sum round_0/theta.npy round_1/spec_fit.json )
rem "cd $P/data; ls -l psi0_6x6_table.npz sym_tables.npz csr/{meta.json,norb.npy,up_indices.npy,up_codes.npy,up_inc.npy} refresh_17409419/s2.npy rpTabPP_17164048/sym_tables_PP.npz rpT1_pp_17163493/params_final.npz" >/dev/null
# 3. spec
python3 - "$LOC" "$P" "$D" "$WIN" "$FRZ" "$FS" "$EXTRA" "$LV" <<'PY'
import json, sys
loc, P, D, win, frz, fs, extra, lv = sys.argv[1:9]
sha = {l.split()[1]: l.split()[0] for l in open(f'{loc}/code_sha256.txt')}
sp = dict(refresh=f'{P}/data/refresh_17409419', pp_params=f'{P}/data/rpT1_pp_17163493/params_final.npz', pp_table=f'{P}/data/rpTabPP_17164048/sym_tables_PP.npz',
          code_sha256=sha, target_la=f'{D}/round_1_target_la.npy', warm_theta=f'{D}/round_0_theta.npy', o2b_fit_spec=f'{D}/round_1_spec_fit.json',
          window=int(win), freeze_outer=frz == 'true', fit_sec=float(fs), lambda_v=(lv if lv == 'auto' else float(lv)))
sp.update({k: v for k, v in json.loads('{' + extra.replace('@D@', D) + '}').items() if v is not None})
json.dump(sp, open(f'{loc}/spec_f2.json', 'w'), indent=1)
PY
rsync -a -e "$SSHO" "$LOC/spec_f2.json" paderborn:$RUN/
# 4. brief
sed "s#@NAME@#$NAME#g; s#@ARM@#$ARM#g; s#@LOC@#$LOC#g; s#@REM@#$RUN#g; s#@WIN@#$WIN#g; s#@FS@#$FS#g; s#@LV@#$LV#g" "$HERE/specs/f2_ANALYSIS_BRIEF.md" > "$LOC/ANALYSIS_BRIEF.md"
echo "staged $RUN (remote) and $LOC (local); arm $ARM window $WIN freeze_outer $FRZ lambda_v $LV fit_sec $FS time $TM qos $QOS"
[ "${1:-}" = "--submit" ] || exit 0
rsync -a -e "$SSHO" "$HERE/f2.sbatch" paderborn:$P/f2.sbatch
OUTL=$(rem "sbatch --parsable --qos=$QOS --job-name=f2$ARM --time=$TM --output=$RUN/slurm-%j.out $P/f2.sbatch $RUN" 2>/dev/null)
JOB=${OUTL%%;*}; [[ "$JOB" =~ ^[0-9]+$ ]] || { echo "sbatch failed: $OUTL"; exit 1; }
echo "$JOB" > "$LOC/job_id.txt"
STALL=${STALL:-2700} F2_REMRUN=$RUN setsid nohup "$HERE/cx_f2_pdb_watch.sh" "$JOB" "$LOC" > "$LOC/watcher.log" 2>&1 < /dev/null &
echo "job=$JOB run=$RUN watcher_pid=$!"
rem "sacct -j $JOB -X -n -o State,Reason%30" 2>/dev/null
