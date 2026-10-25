#!/bin/bash
# Test N0 launcher (Paderborn H100, probe-sized, no training): launch_n0.sh [--submit]  (stage; with --submit also sbatch + arm the durable watcher)
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
RUNS=/project/theorie/a/A.Otaifi/Chatty/runs/compress6/runs
P=/scratch/hpc-prf-ppdmrg/hpcalot/chatty_compress6; D=$P/data; NAME=n0; LOC=$RUNS/$NAME
SSHO="ssh -o BatchMode=yes -o HostName=login4"
rem() { $SSHO paderborn "export PATH=/opt/software/slurm/25.11.8-1/bin:\$PATH; $1"; }
mkdir -p "$LOC"
rem "[ -d $P/codeN0 ] || cp -a $P/code $P/codeN0; mkdir -p $P/runs/$NAME"
for f in cx_n0.py cx_lib.py; do rsync -a -e "$SSHO" "$HERE/$f" paderborn:$P/codeN0/; done
rsync -a -e "$SSHO" "$HERE/../rbmpp_6x6/rbmpp_lib.py" paderborn:$P/codeN0/
( cd "$HERE" && sha256sum cx_n0.py cx_lib.py ../rbmpp_6x6/rbmpp_lib.py | sed 's#../rbmpp_6x6/##' ) > "$LOC/code_sha256.txt"
rem "cd $P/codeN0 && sha256sum cx_n0.py cx_lib.py rbmpp_lib.py" | diff - "$LOC/code_sha256.txt" > /dev/null || { echo "code sha256 differs"; exit 1; }
cat > "$LOC/spec_n0.json" <<J
{"refresh": "$D/refresh_17409419", "a2_table": "$D/o2b_r1/round_1_la_table.npy", "target_la": "$D/o2b_r1/round_1_target_la.npy", "nbonds": 1048576, "seed": 7777}
J
cat > "$LOC/n0.sbatch" <<J
#!/bin/bash
#SBATCH --job-name=n0
#SBATCH --account=hpc-prf-ppdmrg
#SBATCH --partition=gpu_h100
#SBATCH --qos=express_hpcalot
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=${SBATCH_TIME:-00:30:00}
#SBATCH --no-requeue
#SBATCH --output=$P/runs/$NAME/slurm-%j.out
set -uo pipefail
RUN=$P/runs/$NAME
export ST6_CSR=$P/data/csr ST6_TABLE=$P/data/psi0_6x6_table.npz ST6_SYM=$P/data/sym_tables.npz CX_LOOP8=/unused CX_TABDIR=\$RUN/tables
export XLA_FLAGS="--xla_gpu_autotune_level=0" XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.85 OMP_NUM_THREADS=8
{ echo job_id=\$SLURM_JOB_ID; hostname; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader; date -Is; } > \$RUN/run_metadata.txt
cd $P/codeN0
$P/../venv_jax_0.8.2/bin/python cx_n0.py \$RUN \$RUN/spec_n0.json; rc=\$?
echo exit=\$rc finished=\$(date -Is) >> \$RUN/run_metadata.txt; exit \$rc
J
sed "s#@LOC@#$LOC#g; s#@REM@#$P/runs/$NAME#g" "$HERE/specs/n0_ANALYSIS_BRIEF.md" > "$LOC/ANALYSIS_BRIEF.md"
rsync -a -e "$SSHO" "$LOC/spec_n0.json" "$LOC/n0.sbatch" paderborn:$P/runs/$NAME/
echo staged $NAME
[ "${1:-}" = "--submit" ] || exit 0
JID=$(rem "sbatch --parsable $P/runs/$NAME/n0.sbatch" | cut -d';' -f1)
[[ "$JID" =~ ^[0-9]+$ ]] || { echo "sbatch failed: $JID"; exit 1; }
echo "$JID" > "$LOC/job_id.txt"; echo submitted $JID
STALL=${STALL:-1500} N0_REMRUN=$P/runs/$NAME setsid nohup "$HERE/cx_n0_pdb_watch.sh" "$JID" "$LOC" > "$LOC/watcher.log" 2>&1 < /dev/null &
echo "watcher_pid=$!"
rem "sacct -j $JID -X -n -o State,Reason%30"
