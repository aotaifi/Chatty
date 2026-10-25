#!/bin/bash
# Gates G-A2 / G-N0c / G-A3 launcher (Paderborn H100, probe-sized, no training): launch_gate.sh a2|n0c|a3 [--submit]
set -euo pipefail
WHICH=${1:?a2|n0c|a3}
HERE=$(cd "$(dirname "$0")" && pwd)
RUNS=/project/theorie/a/A.Otaifi/Chatty/runs/compress6/runs
P=/scratch/hpc-prf-ppdmrg/hpcalot/chatty_compress6; D=$P/data; NAME=gate_$WHICH; LOC=$RUNS/$NAME
SSHO="ssh -o BatchMode=yes -o HostName=login4"
rem() { $SSHO paderborn "export PATH=/opt/software/slurm/25.11.8-1/bin:\$PATH; $1"; }
mkdir -p "$LOC"
rem "[ -d $P/codeGate ] || cp -a $P/codeH1 $P/codeGate; mkdir -p $P/runs/$NAME $D/rpTabA_17168764"
FILES="cx_ga2.py cx_gn0c.py cx_ga3.py cx_n0.py cx_lib.py"
for f in $FILES; do rsync -a -e "$SSHO" "$HERE/$f" paderborn:$P/codeGate/; done
rsync -a -e "$SSHO" "$HERE/../rbmpp_6x6/rbmpp_lib.py" paderborn:$P/codeGate/
( cd "$HERE" && sha256sum $FILES ../rbmpp_6x6/rbmpp_lib.py | sed 's#../rbmpp_6x6/##' ) > "$LOC/code_sha256.txt"
rem "cd $P/codeGate && sha256sum $FILES rbmpp_lib.py" | diff - "$LOC/code_sha256.txt" > /dev/null || { echo "code sha256 differs"; exit 1; }
if [ "$WHICH" = a3 ]; then
  rsync -aL -e "$SSHO" /project/theorie/a/A.Otaifi/Chatty/runs/rbmpp6/runs/rpTabA_17168764/sym_tables_A.npz paderborn:$D/rpTabA_17168764/
  ( sha256sum /project/theorie/a/A.Otaifi/Chatty/runs/rbmpp6/runs/rpTabA_17168764/sym_tables_A.npz | cut -c1-64 ) > "$LOC/sym_tables_A.sha256"
  rem "sha256sum $D/rpTabA_17168764/sym_tables_A.npz | cut -c1-64" | diff - "$LOC/sym_tables_A.sha256" || { echo "A table sha256 differs"; exit 1; }
fi
COMMON="{\"refresh\": \"$D/refresh_17409419\", \"o2b_table\": \"$D/o2b_r1/round_1_la_table.npy\", \"target_la\": \"$D/o2b_r1/round_1_target_la.npy\""
case $WHICH in
 a2)  echo "$COMMON}" > "$LOC/spec_$WHICH.json"; JSONF=cx_ga2.json
      RUNCMD="\$PY cx_ga2.py \$RUN \$RUN/spec_$WHICH.json";;
 a3)  echo "{\"table_A\": \"$D/rpTabA_17168764/sym_tables_A.npz\", \"rounds\": 6}" > "$LOC/spec_$WHICH.json"; JSONF=cx_ga3.json
      RUNCMD="\$PY cx_ga3.py \$RUN \$RUN/spec_$WHICH.json";;
 n0c) echo "$COMMON}" > "$LOC/spec_$WHICH.json"; JSONF=final.json
      for v in lin quad; do echo "{\"refresh\": \"$D/refresh_17409419\", \"a2_table\": \"$P/runs/$NAME/pert_${v}_la.npy\", \"target_la\": \"$D/o2b_r1/round_1_target_la.npy\", \"nbonds\": 1048576, \"seed\": 7777}" > "$LOC/spec_n0_$v.json"; done
      RUNCMD="\$PY cx_gn0c.py \$RUN \$RUN/spec_$WHICH.json; rc=\$?; for v in lin quad; do [ \$rc = 0 ] && \$PY cx_n0.py \$RUN/n0_\$v \$RUN/spec_n0_\$v.json; rc=\$?; done; [ \$rc = 0 ] && \$PY - <<PY
import json; ok = all(json.load(open('\$RUN/n0_%s/cx_n0.json' % v)).get('done') and 'stop' not in json.load(open('\$RUN/n0_%s/cx_n0.json' % v)) for v in ('lin','quad')); json.dump(dict(done=ok), open('\$RUN/final.json','w'))
PY";;
esac
cat > "$LOC/$WHICH.sbatch" <<J
#!/bin/bash
#SBATCH --job-name=gate_$WHICH
#SBATCH --account=hpc-prf-ppdmrg
#SBATCH --partition=gpu_h100
#SBATCH --qos=express_hpcalot
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=${SBATCH_TIME:-00:30:00}
#SBATCH --no-requeue
#SBATCH --output=$P/runs/$NAME/slurm-%j.out
set -uo pipefail
RUN=$P/runs/$NAME
export ST6_CSR=$P/data/csr ST6_TABLE=$P/data/psi0_6x6_table.npz ST6_SYM=$P/data/sym_tables.npz CX_LOOP8=/unused CX_TABDIR=\$RUN/tables
export XLA_FLAGS="--xla_gpu_autotune_level=0" XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.85 OMP_NUM_THREADS=8
{ echo job_id=\$SLURM_JOB_ID; hostname; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader; date -Is; } > \$RUN/run_metadata.txt
cd $P/codeGate
PY=$P/../venv_jax_0.8.2/bin/python
$RUNCMD
rc=\$?
echo exit=\$rc finished=\$(date -Is) >> \$RUN/run_metadata.txt; exit \$rc
J
sed -i 's#^\(\$P/\.\./venv\)#\1#' "$LOC/$WHICH.sbatch"
cat > "$LOC/ANALYSIS_BRIEF.md" <<J
# Gate $WHICH analysis brief (local copy of Paderborn run $P/runs/$NAME; local dir $LOC)
Question: README compress_6x6 "M1 + gate for the loop", item $( case $WHICH in a2) echo G-A2;; n0c) echo G-N0c;; a3) echo G-A3;; esac ). No training; exact tables.
File: $LOC/$JSONF (done, stop, control, summary / chain).  Gate: python3 -c "import json;d=json.load(open('$LOC/$JSONF'));print(d.get('stop'),d.get('control'),d.get('summary'),d.get('chain'))"
Next step per verdict: COMPLETED with control.ok -> report the numbers to the lead (the lead applies the pre-registered reading); CONTROL FAILED / FAILED / TIMEOUT -> report, do not interpret.
Exact computation, no Monte Carlo chains (quotable.py n/a).
J
rsync -a -e "$SSHO" "$LOC/spec_"*.json "$LOC/$WHICH.sbatch" paderborn:$P/runs/$NAME/
echo staged $NAME
[ "${2:-}" = "--submit" ] || exit 0
JID=$(rem "sbatch --parsable $P/runs/$NAME/$WHICH.sbatch" | cut -d';' -f1)
[[ "$JID" =~ ^[0-9]+$ ]] || { echo "sbatch failed: $JID"; exit 1; }
echo "$JID" > "$LOC/job_id.txt"; echo submitted $JID
GATE_JSON=$JSONF STALL=${STALL:-1500} GATE_REMRUN=$P/runs/$NAME setsid nohup "$HERE/cx_gate_pdb_watch.sh" "$JID" "$LOC" > "$LOC/watcher.log" 2>&1 < /dev/null &
echo "watcher_pid=$!"
rem "sacct -j $JID -X -n -o State,Reason%30"
