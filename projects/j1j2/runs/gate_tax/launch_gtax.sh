#!/bin/bash
# G-tax launcher (Paderborn H100).  STAGE=control|full SBATCH_TIME=hh:mm:ss launch_gtax.sh [--submit]
set -euo pipefail
STAGE=${STAGE:?set STAGE=control|full}
HERE=$(cd "$(dirname "$0")" && pwd)
RUNS=/project/theorie/a/A.Otaifi/Chatty/runs/compress6/runs
P=/scratch/hpc-prf-ppdmrg/hpcalot/chatty_compress6; D=$P/data; NAME=gtax_$STAGE; LOC=$RUNS/$NAME
AR=/project/theorie/a/A.Otaifi/Chatty/runs/rbmpp6/runs
SSHO="ssh -o BatchMode=yes -o HostName=login4"
rem() { $SSHO paderborn "export PATH=/opt/software/slurm/25.11.8-1/bin:\$PATH; $1"; }
mkdir -p "$LOC"
rem "[ -d $P/codeGT ] || cp -a $P/code $P/codeGT; mkdir -p $D/gtax"
for f in cx_gtax.py gtax_net.py rbmpp_lib.py; do rsync -a -e "$SSHO" "$HERE/$f" paderborn:$P/codeGT/; done
rsync -a -e "$SSHO" "$HERE/../../experiments/compress_6x6/cx_lib.py" paderborn:$P/codeGT/
rsync -a -e "$SSHO" $AR/rpT4_sr2_17164867/params_final.npz $AR/rpTabA_17168764/sym_tables_A.npz $AR/rpTabA_17168764/table_A.json paderborn:$D/gtax/
( cd "$HERE" && sha256sum cx_gtax.py gtax_net.py rbmpp_lib.py ../../experiments/compress_6x6/cx_lib.py | sed 's#../../experiments/compress_6x6/##' ) > "$LOC/code_sha256.txt"
rem "cd $P/codeGT && sha256sum cx_gtax.py gtax_net.py rbmpp_lib.py cx_lib.py" | sort -k2 | diff - <(sort -k2 "$LOC/code_sha256.txt") > /dev/null || { echo "code sha256 differs"; exit 1; }
SHAJ=$(awk '{printf "\"%s\": \"%s\", ", $2, $1}' "$LOC/code_sha256.txt" | sed 's/, $//')
cat > "$LOC/spec_gtax.json" <<J
{"stage": "$STAGE", "steps": 400, "seed": 0, "params": "$D/gtax/params_final.npz", "table_A": "$D/gtax/sym_tables_A.npz", "table_A_json": "$D/gtax/table_A.json",
 "code_sha256": {$SHAJ}}
J
cat > "$LOC/gtax.sbatch" <<J
#!/bin/bash
#SBATCH --job-name=gtax_$STAGE
#SBATCH --account=hpc-prf-ppdmrg
#SBATCH --partition=gpu_h100
#SBATCH --qos=express_hpcalot
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=${SBATCH_TIME:-00:40:00}
#SBATCH --no-requeue
#SBATCH --output=$P/runs/$NAME/slurm-%j.out
set -uo pipefail
RUN=$P/runs/$NAME
export ST6_CSR=$D/csr ST6_TABLE=$D/psi0_6x6_table.npz ST6_SYM=$D/sym_tables.npz CX_LOOP8=/unused CX_TABDIR=\$RUN/tables
export XLA_FLAGS="--xla_gpu_autotune_level=0" XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.85 OMP_NUM_THREADS=8
{ echo job_id=\$SLURM_JOB_ID; hostname; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader; date -Is; } > \$RUN/run_metadata.txt
cd $P/codeGT
$P/../venv_jax_0.8.2/bin/python cx_gtax.py \$RUN \$RUN/spec_gtax.json; rc=\$?
echo exit=\$rc finished=\$(date -Is) >> \$RUN/run_metadata.txt; exit \$rc
J
rem "mkdir -p $P/runs/$NAME"
rsync -a -e "$SSHO" "$LOC/spec_gtax.json" "$LOC/gtax.sbatch" paderborn:$P/runs/$NAME/
echo staged $NAME
[ "${1:-}" = "--submit" ] || exit 0
JID=$(rem "sbatch --parsable $P/runs/$NAME/gtax.sbatch" | cut -d';' -f1)
echo "$JID" > "$LOC/job_id.txt"; echo submitted $JID
