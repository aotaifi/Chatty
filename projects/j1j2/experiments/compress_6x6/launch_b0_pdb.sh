#!/bin/bash
# Test B0, SEED 2, on one Paderborn H100 (README compress_6x6 "Test B0" + Amendment B0-1; second seed of runs/b0_ppbf_simple, job 17445407).
# Amendment B0-2 version: NAME (default b0_ppbf_simple_s2), SEED (2), LR (start lr of the cosine; end lr 0.002), plateau stop disabled.  Env: NAME=b0_ppbf_simple_s1h SEED=1 LR=0.01 launch_b0_pdb.sh [--submit]
#   launch_b0_pdb.sh            stage only (default): code (sha256-identical to seed 1's code/), specs, inputs check, local run dir + ANALYSIS_BRIEF.md
#   launch_b0_pdb.sh --submit   stage, then sbatch on gpu_h100 (account hpc-prf-ppdmrg, --no-requeue) and arm the durable watcher (cx_b0_pdb_watch.sh)
# Needs the VPN (paderborn/README.md): ssh -o HostName=login4 paderborn.  Sizing (H100, probe job 3806779: 1.1 s/step, table 49k orbits/s): WALL_SEC training ceiling, SBATCH_TIME job limit.
set -euo pipefail
WALL_SEC=${WALL_SEC:-5200}; SBATCH_TIME=${SBATCH_TIME:-02:30:00}
HERE=$(cd "$(dirname "$0")" && pwd)
S1=/project/theorie/a/A.Otaifi/Chatty/runs/compress6/runs/b0_ppbf_simple               # seed 1: code, specs
NAME=${NAME:-b0_ppbf_simple_s2}; SEED=${SEED:-2}; LR=${LR:?set LR}
LOC=/project/theorie/a/A.Otaifi/Chatty/runs/compress6/runs/$NAME            # local: specs, brief, pulled-back small outputs
P=/scratch/hpc-prf-ppdmrg/hpcalot/chatty_compress6; RUN=$P/runs/$NAME   # remote
SSHO="ssh -o BatchMode=yes -o HostName=login4"; export PATH=$PATH
rem() { $SSHO paderborn "export PATH=/opt/software/slurm/25.11.8-1/bin:\$PATH; $1"; }
case "$WALL_SEC$SBATCH_TIME" in *@*) echo "WALL_SEC / SBATCH_TIME not set"; exit 1;; esac
mkdir -p "$LOC"
# 1. code: same files as seed 1 (sha256 compared after the copy)
rem "mkdir -p $RUN/code $P/data/rpT1_pp_17163493"
rsync -a -e "$SSHO" "$S1/code/" paderborn:$RUN/code/
( cd "$S1/code" && sha256sum *.py ) > "$LOC/code_sha256.txt"
rem "cd $RUN/code && sha256sum *.py" | diff - "$LOC/code_sha256.txt" > /dev/null || { echo "code sha256 differs from seed 1"; exit 1; }
# 2. specs: seed 1's, seed 2, paths -> Paderborn, wall_sec set
python3 - "$S1" "$LOC" "$P" "$RUN" "$WALL_SEC" "$SEED" "$LR" <<'PY'
import json, sys
s1, loc, P, run, wall, seed, lr = sys.argv[1:8]
t = json.load(open(f'{s1}/spec_train.json')); r = json.load(open(f'{s1}/spec_readout.json'))
t['seed'] = int(seed); t['wall_sec'] = int(wall); t['lr'] = float(lr); t['lr_end'] = 0.002; t['plateau_min_step'] = 100000;  # > steps: plateau() returns None, never stops
 t['init'] = f'{P}/data/rpT1_pp_17163493/params_final.npz'
r['refresh'] = f'{P}/data/refresh_17409419'; r['pp_table'] = f'{P}/data/rpTabPP_17164048/sym_tables_PP.npz'; r['pp_params'] = t['init']; r['params'] = f'{run}/train/params_final.npz'
json.dump(t, open(f'{loc}/spec_train.json', 'w'), indent=1); json.dump(r, open(f'{loc}/spec_readout.json', 'w'), indent=1)
PY
rsync -a -e "$SSHO" "$LOC/spec_train.json" "$LOC/spec_readout.json" paderborn:$RUN/
# 3. inputs exist (sizes printed); the PP start params must be byte-identical to seed 1's
rem "cd $P/data; ls -l psi0_6x6_table.npz sym_tables.npz csr/{meta.json,norb.npy,up_indices.npy,up_codes.npy,up_inc.npy} refresh_17409419/s2.npy rpTabPP_17164048/sym_tables_PP.npz rpT1_pp_17163493/params_final.npz; sha256sum rpT1_pp_17163493/params_final.npz"
sha256sum /project/theorie/a/A.Otaifi/Chatty/runs/rbmpp6/runs/rpT1_pp_17163493/params_final.npz
# 4. brief
if [ ! -f "$LOC/ANALYSIS_BRIEF.md" ]; then sed "s#@NAME@#$NAME#g; s#@SEED@#$SEED#g; s#@LR@#$LR#g" "$HERE/specs/b0_s2_ANALYSIS_BRIEF.md" > "$LOC/ANALYSIS_BRIEF.md"; fi
echo "staged $RUN (remote) and $LOC (local); wall_sec=$WALL_SEC time=$SBATCH_TIME"
[ "${1:-}" = "--submit" ] || exit 0
rsync -a -e "$SSHO" "$HERE/b0_pdb.sbatch" paderborn:$P/b0_pdb.sbatch
OUTL=$(rem "sbatch --parsable --job-name=${NAME#b0_ppbf_simple_} --time=$SBATCH_TIME --output=$RUN/slurm-%j.out $P/b0_pdb.sbatch $RUN" 2>/dev/null)
JOB=${OUTL%%;*}; [[ "$JOB" =~ ^[0-9]+$ ]] || { echo "sbatch failed: $OUTL"; exit 1; }
echo "$JOB" > "$LOC/job_id.txt"
B0_REMRUN=$RUN setsid nohup "$HERE/cx_b0_pdb_watch.sh" "$JOB" "$LOC" > "$LOC/watcher.log" 2>&1 < /dev/null &
echo "job=$JOB run=$RUN watcher_pid=$!"
rem "sacct -j $JOB -X -n -o State,Reason%30" 2>/dev/null
