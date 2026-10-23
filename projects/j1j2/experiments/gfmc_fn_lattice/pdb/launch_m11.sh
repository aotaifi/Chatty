#!/bin/bash
# Staged launcher for the M1.1 production chains (6x6, RBM+PP A + 1 signed hop, fixed-node GFMC) on Paderborn gpu_h100.
# DRY RUN by default: prints every sbatch, ssh and watcher command it would run and writes nothing remote.
#   launch_m11.sh STAGE           STAGE in T (3-minute smoke test) | A (reference point, 2 seeds) | B (dtau scan) | C (M scan)
#   GO=1 launch_m11.sh STAGE      rsync code, submit one job per chain, arm one durable watcher per job, write ANALYSIS_BRIEF.md
# Environment: QOS (default cont = the PI's normal QoS; express_hpcalot only if the lead says so), CHUNK (default 512), LOCALROOT.
# Prerequisite (done 2026-10-23): verify job 3825550 PASS (alpha, <H>, amplitude 1.8e-10, exact E_FN 5.5738332e-06).
# Each chain: gfmc6_run.py with --maxsec just under the sbatch time; it checkpoints every 10 blocks; if a chain stops on --maxsec before
# its block count, resubmit with RESUME=1 (same name) -- nothing else changes.  A chain is an independent seed (init Metropolis, RNG).
# Chain table: name seed M dtau blocks burn walltime_h   (blocks counted after burn; burn = 10 tau).
# Sizing (pilots 3825551 / 3825752, H100): 0.17 s per walker*tau, block sigma 9.3e-4/sqrt(M) per site (dtau 0.1, blocks ~independent)
#   -> every chain below = 3.5e6 walker-blocks at dtau 0.2 (equal tau budget in B) = ~33-36 GPU-h, error ~5e-7 per site (9% of 5.57e-6).
set -u
STAGE=${1:?stage A|B|C}; GO=${GO:-0}; QOS=${QOS:-cont}; CHUNK=${CHUNK:-512}; RESUME=${RESUME:-0}
HERE=$(cd "$(dirname "$0")" && pwd); CODE=$HERE/..; DELTA=/project/theorie/a/A.Otaifi/Chatty-organize/projects/j1j2/delta_today.md
RB=/scratch/hpc-prf-ppdmrg/hpcalot/chatty_gfmc; LOCALROOT=${LOCALROOT:-/scratch/a/A.Otaifi/gfmc_m11/prod}
SSH="ssh -o BatchMode=yes -o HostName=login4"; SL=/opt/software/slurm/25.11.8-1/bin
case $STAGE in
A) CHAINS="
refA_s1  1 2048 0.2 1700  50 40
refA_s2  2 2048 0.2 1700  50 40";;
B) CHAINS="
dt01_s3  3 2048 0.1 3400 100 40
dt04_s3  3 2048 0.4  850  25 40";;
C) CHAINS="
M0512_s3 3  512 0.2 6800  50 40
M8192_s3 3 8192 0.2  430  50 40";;
T) CHAINS="
test_s9  9   64 0.2    6   4  1";;   # smoke test of the launcher path (~3 min), not part of the production
*) echo bad stage; exit 1;; esac
run() { if [ "$GO" = 1 ]; then eval "$@"; else echo "DRY: $*"; fi; }
if [ "$GO" = 1 ]; then rsync -a -e "ssh -o BatchMode=yes -o HostName=login4" $CODE/gfmc_fn_lib.py $CODE/gfmc6_run.py $CODE/gfmc6_analyze.py $CODE/../rbmpp_6x6/rbmpp_lib.py paderborn:$RB/code/ || exit 1; fi
echo "$CHAINS" | while read name seed M dt nb burn wh; do
  [ -z "$name" ] && continue
  maxsec=$(( wh*3600 - 900 ))
  sb=$RB/runs/$name.sbatch
  body="#!/bin/bash
#SBATCH --job-name=g_$name
#SBATCH --account=hpc-prf-ppdmrg
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:h100:1
#SBATCH --qos=$QOS
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=$wh:00:00
#SBATCH --no-requeue
#SBATCH --output=$RB/runs/%x-%j.out
set -uo pipefail
RUN=$RB/runs/$name; mkdir -p \$RUN/code; cp $RB/code/*.py \$RUN/code/
export GFMC_PARAMS=$RB/data/params_final.npz GFMC_TABLE_A=$RB/data/table_A.json
export XLA_FLAGS=\"--xla_gpu_autotune_level=0\" XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.85 OMP_NUM_THREADS=8
{ echo job_id=\$SLURM_JOB_ID; hostname; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader; date -Is; sha256sum \$RUN/code/*.py; } >> \$RUN/run_metadata.txt
cd \$RUN/code
$RB/../venv_jax_0.8.2/bin/python gfmc6_run.py \$RUN --M $M --dtau $dt --blocks $nb --burn $burn --seed $seed --chunk $CHUNK --maxsec $maxsec $([ "$RESUME" = 1 ] && echo --resume) > \$RUN/stdout.txt 2>&1; rc=\$?
echo exit=\$rc finished=\$(date -Is) >> \$RUN/run_metadata.txt; exit \$rc"
  if [ "$GO" = 1 ]; then
    printf '%s\n' "$body" | $SSH paderborn "cat > $sb"
    jid=$($SSH -n paderborn "export PATH=$SL:\$PATH; sbatch --parsable $sb")
    echo "SUBMITTED $name job $jid"
    [ -n "$jid" ] || { echo "no job id for $name"; continue; }
    L=$LOCALROOT/$name; mkdir -p $L
    cat > $L/ANALYSIS_BRIEF.md <<EOB
# M1.1 6x6 GFMC chain $name (job $jid): seed $seed, M=$M, dtau=$dt, $nb blocks after $burn burn blocks, QoS $QOS
Question: does fixed-node GFMC with the RBM+PP A + 1 signed hop guide reproduce exact E_FN = 5.5738332e-06 per site above E0 (results/rbmpp_6x6/runs/table_A.json lanczos[1].dE_FN_site, re-derived exactly in verify job 3825550)?
Files: progress.txt (PROGRESS block lines, last line GFMC_DONE), blocks.npz (emix, egrow per block), result.json in $L (pulled from $RB/runs/$name).
Gate: this chain alone gives no verdict. When all chains of stages A-C are done run: python3 $CODE/gfmc6_analyze.py $LOCALROOT   (per-chain blocking error, tau_int, drift in fifths; seed difference; dtau and 1/M extrapolation; FINAL line = PASS if within 2 sigma). For a single chain also run /project/theorie/a/A.Otaifi/tools/quotable.py on blocks.npz-derived series with another seed's chain.
Next: seeds disagree beyond errors or drift in fifths > 3 sigma -> treat as relaxation/trapping, continue the chain (RESUME=1) before any fit; FINAL PASS -> report to lead, M1.2 (cost per walker step at 8x8, 14x14); FINAL FAIL -> stop, report dtau and 1/M slopes, do not extrapolate further.
EOB
    GF_REMRUN=$RB/runs/$name setsid nohup $HERE/gfmc_pdb_watch.sh $jid $L > $L/watcher.log 2>&1 < /dev/null &
    echo "WATCHER armed for $jid (ppid $(ps -o ppid= -p $! 2>/dev/null))"
  else
    echo "DRY: submit $name: M=$M dtau=$dt blocks=$nb burn=$burn seed=$seed time=${wh}h maxsec=$maxsec qos=$QOS -> $SSH paderborn sbatch $sb ; watcher gfmc_pdb_watch.sh <jid> $LOCALROOT/$name"
  fi
done
