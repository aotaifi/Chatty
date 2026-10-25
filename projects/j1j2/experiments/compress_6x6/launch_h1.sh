#!/bin/bash
# Test H1 launcher (Paderborn H100): STAGE=control (k=1 controls only) or STAGE=full.  launch_h1.sh [--submit]
set -euo pipefail
STAGE=${STAGE:?set STAGE=control|full}
HERE=$(cd "$(dirname "$0")" && pwd)
RUNS=/project/theorie/a/A.Otaifi/Chatty/runs/compress6/runs
P=/scratch/hpc-prf-ppdmrg/hpcalot/chatty_compress6; D=$P/data; NAME=h1_$STAGE; LOC=$RUNS/$NAME
SSHO="ssh -o BatchMode=yes -o HostName=login4"
rem() { $SSHO paderborn "export PATH=/opt/software/slurm/25.11.8-1/bin:\$PATH; $1"; }
mkdir -p "$LOC"
rem "[ -d $P/codeH1 ] || cp -a $P/code $P/codeH1"
for f in cx_h1.py cx_lib.py; do rsync -a -e "$SSHO" "$HERE/$f" paderborn:$P/codeH1/; done
rsync -a -e "$SSHO" "$HERE/../rbmpp_6x6/rbmpp_lib.py" paderborn:$P/codeH1/
( cd "$HERE" && sha256sum cx_h1.py cx_lib.py ../rbmpp_6x6/rbmpp_lib.py | sed 's#../rbmpp_6x6/##' ) > "$LOC/code_sha256.txt"
rem "cd $P/codeH1 && sha256sum cx_h1.py cx_lib.py rbmpp_lib.py" | diff - "$LOC/code_sha256.txt" > /dev/null || { echo "code sha256 differs"; exit 1; }
cat > "$LOC/spec_h1.json" <<J
{"stage": "$STAGE", "refresh": "$D/refresh_17409419", "o2b_table": "$D/o2b_r1/round_1_la_table.npy", "target_la": "$D/o2b_r1/round_1_target_la.npy",
 "f1_table_A": "$P/runs/f1_A/round_1/la_table.npy", "f1_table_B": "$P/runs/f1_B/round_1/la_table.npy"}
J
cat > "$LOC/h1.sbatch" <<J
#!/bin/bash
#SBATCH --job-name=h1_$STAGE
#SBATCH --account=hpc-prf-ppdmrg
#SBATCH --partition=gpu_h100
#SBATCH --qos=express_hpcalot
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=${SBATCH_TIME:-00:40:00}
#SBATCH --no-requeue
#SBATCH --output=$P/runs/$NAME/slurm-%j.out
set -uo pipefail
RUN=$P/runs/$NAME
export ST6_CSR=$P/data/csr ST6_TABLE=$P/data/psi0_6x6_table.npz ST6_SYM=$P/data/sym_tables.npz CX_LOOP8=/unused CX_TABDIR=\$RUN/tables
export XLA_FLAGS="--xla_gpu_autotune_level=0" XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.85 OMP_NUM_THREADS=8
{ echo job_id=\$SLURM_JOB_ID; hostname; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader; date -Is; } > \$RUN/run_metadata.txt
cd $P/codeH1
$P/../venv_jax_0.8.2/bin/python cx_h1.py \$RUN \$RUN/spec_h1.json; rc=\$?
echo exit=\$rc finished=\$(date -Is) >> \$RUN/run_metadata.txt; exit \$rc
J
rem "mkdir -p $P/runs/$NAME"
rsync -a -e "$SSHO" "$LOC/spec_h1.json" "$LOC/h1.sbatch" paderborn:$P/runs/$NAME/
echo staged $NAME
[ "${1:-}" = "--submit" ] || exit 0
JID=$(rem "sbatch --parsable $P/runs/$NAME/h1.sbatch" | cut -d';' -f1)
echo "$JID" > "$LOC/job_id.txt"; echo submitted $JID
