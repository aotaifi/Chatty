#!/bin/bash
# M1.3b production launcher (STAGED, NOT RUN): 8x8 fixed-node GFMC with the translations x spin-flip RBM+PP guide, p = 0 and p = 1.
# DRY RUN by default: prints every sbatch / ssh / watcher command it would run and writes nothing remote.
#   launch_m13b.sh STAGE PARAMS_PATH [TAG]
#     STAGE   T  smoke test (3 min): 4 VMC sweeps + 3 blocks of each guide, checks the whole path
#             V  VMC p = 0 / p = 1 production (vmc_moments.py, two independent seeds, pooled) -> consts.json (alpha, E, w) for the walkers
#             A  GFMC p = 0 chains (two independent seeds)           [needs V's consts.json: submitted afterok:V]
#             B  GFMC p = 1 chains (two independent seeds)           [needs V's consts.json: submitted afterok:V]
#             X  bias scans on p = 0 (cheap guide): dtau and M variants of one chain (the bias at 8x8 is NOT assumed small)
#     PARAMS_PATH  a params .npz ON PADERBORN (e.g. /scratch/hpc-prf-ppdmrg/hpcalot/rbmpp8/runs/rb8_prod/params_final.npz)
#     TAG     run family name (default: the parameter file's run directory name); everything lives in $RB/runs/$TAG/
#   GO=1 launch_m13b.sh STAGE PARAMS [TAG]   rsync code, submit, arm one durable watcher per job, write ANALYSIS_BRIEF.md per chain.
# Environment: QOS (default express_hpcalot, 20 H100-h per month, personal; the lead may choose cont), CHUNK (default 256), LOCALROOT, DEP (job id
#   to depend on instead of the V job read from LOCALROOT/TAG/JOB_V), RESUME=1 (continue chains after a TIMEOUT: same names, --resume).
# Each chain: gfmc_run.py with --maxsec just under the sbatch time; checkpoint every 10 blocks; a stop on --maxsec before its block count
# prints GFMC_DONE completed False: resubmit with RESUME=1 (nothing else changes).  A chain is an independent seed (own init walkers and RNG).
# SIZING (pilots of 2026-10-24, pilot parameters pilot8_long; see SIZING in ANALYSIS_BRIEF.md): chain table below, written by the sizing step.
set -u
STAGE=${1:?stage T|V|A|B|X}; PARAMS=${2:?PARAMS_PATH on Paderborn}; GO=${GO:-0}; QOS=${QOS:-express_hpcalot}; CHUNK=${CHUNK:-256}; RESUME=${RESUME:-0}
TAG=${3:-$(basename "$(dirname "$PARAMS")")}
HERE=$(cd "$(dirname "$0")" && pwd); CODE=$HERE/..; RBMPP=$CODE/../rbmpp_6x6
DELTA=/project/theorie/a/A.Otaifi/Chatty/organize/projects/j1j2/delta_today.md
RB=/scratch/hpc-prf-ppdmrg/hpcalot/gfmc8; LOCALROOT=${LOCALROOT:-/scratch/a/A.Otaifi/gfmc_m13b/prod}; LR=$LOCALROOT/$TAG
SSH="ssh -o BatchMode=yes -o HostName=login4"; SL=/opt/software/slurm/25.11.8-1/bin; VENV=/scratch/hpc-prf-ppdmrg/hpcalot/venv_jax_0.8.2
RUNS=$RB/runs/$TAG
# ---- chain table: name p seed M dtau blocks burn walltime_h  (blocks counted after burn; burn = 10 tau at least)
# SIZING_TABLE_BEGIN
# Sizing from the pilots of 2026-10-24 (H100; pilot parameters pilot8_long; jobs 3831879 VMC, 3832040 p=0, 3832041 p=1):
#   p=0: 0.0038 s/walker-tau, sigma_walker/N 5.9e-3, tau_int 0.75 blocks (dtau 0.1); p=1: 0.52 s/walker-tau, sigma_walker/N 2.8e-3, tau_int 1.3 blocks;
#   VMC 0.13 ms/eval, 20 s per sweep of 1024 chains (147 evals per sample), E_1 error 4.4e-5 per site from 32k samples.
#   error^2 = sigma_w^2 * 2 tau_int_blocks / (walker-blocks).  Targets: GFMC 1e-5 (p=1, two chains) and ~4e-6 (p=0); VMC E_1 ~2e-5.
#   Cost (H100-h): V 1.1, A 0.5, B 4.2 (burn 15 tau = 0.55 h of each chain), X 1.5; total about 7.3 (budget 10).  The production parameters
#   will have a different sigma_walker: the first chains' progress lines give it; if the error is off by a factor f, cost scales by f^2.
case $STAGE in
T) VMC="--chains 64 --sweeps 4 --stageA_sweeps 4 --stageA_chains 64 --therm 20 --seeds 1,2"; VWH=1
   CHAINS="
p0_s9 0 9 64 0.2 3 2 1
p1_s9 1 9 32 0.2 3 2 1";;
V) VMC="--chains 1024 --sweeps 80 --seeds 1,2"; VWH=3; CHAINS="";;
A) CHAINS="
p0_s1 0 1 1024 0.1 2000 100 1
p0_s2 0 2 1024 0.1 2000 100 1";;
B) CHAINS="
p1_s1 1 1 256 0.1 420 150 4
p1_s2 1 2 256 0.1 420 150 4";;
X) CHAINS="
p0_s3_M0256_dt0.1 0 3 256 0.1 4000 100 1
p0_s3_M4096_dt0.1 0 3 4096 0.1 300 100 1
p0_s3_M1024_dt0.2 0 3 1024 0.2 1000 50 1
p0_s3_M1024_dt0.4 0 3 1024 0.4 500 25 1
p1_s3_M0064_dt0.1 1 3 64 0.1 600 150 2";;
*) echo "bad stage"; exit 1;; esac
# SIZING_TABLE_END
run() { if [ "$GO" = 1 ]; then eval "$@"; else echo "DRY: $*"; fi; }
if [ "$GO" = 1 ]; then
  $SSH paderborn "test -f $PARAMS" || { echo "params not found on Paderborn: $PARAMS"; exit 1; }
  rsync -a -e "$SSH" $CODE/{gfmc_fn_lib,gfmc_run,vmc_moments,timing8}.py $RBMPP/rbmpp_lib.py paderborn:$RB/code/ || exit 1
  mkdir -p $LR
else
  echo "DRY: rsync code {gfmc_fn_lib,gfmc_run,vmc_moments,timing8}.py rbmpp_lib.py -> paderborn:$RB/code/ ; local dir $LR"
fi
submit() {  # name sbatch-body dependency-or-empty -> prints job id
  local name=$1 body=$2 dep=$3 sb=$RUNS/$1.sbatch
  if [ "$GO" = 1 ]; then
    printf '%s\n' "$body" | $SSH paderborn "mkdir -p $RUNS/$name; cat > $sb"
    jid=$($SSH -n paderborn "export PATH=$SL:\$PATH; sbatch --parsable ${dep:+--dependency=afterok:$dep} $sb")
    [[ "$jid" =~ ^[0-9]+$ ]] || { echo "SUBMIT FAILED for $name: '$jid'" >&2; return 1; }
    echo "SUBMITTED $name job $jid" >&2; echo "$jid"
  else
    echo "DRY: submit $name ${dep:+(afterok:$dep)} via sbatch $sb" >&2; echo "DRYJOB"
  fi
}
header() {  # name hours
  echo "#!/bin/bash
#SBATCH --job-name=m13b_$1
#SBATCH --account=hpc-prf-ppdmrg
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:h100:1
#SBATCH --qos=$QOS
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=$2:00:00
#SBATCH --no-requeue
#SBATCH --output=$RUNS/$1/slurm-%j.out
set -uo pipefail
RUN=$RUNS/$1; mkdir -p \$RUN/code; cp $RB/code/*.py \$RUN/code/
export XLA_FLAGS=\"--xla_gpu_autotune_level=0\" XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 OMP_NUM_THREADS=8
{ echo job_id=\$SLURM_JOB_ID; hostname; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader; date -Is; echo params=$PARAMS; sha256sum $PARAMS \$RUN/code/*.py; } >> \$RUN/run_metadata.txt
cd \$RUN/code"
}
watch() {  # name jid
  local L=$LR/$1
  if [ "$GO" = 1 ]; then
    mkdir -p $L
    GF_REMRUN=$RUNS/$1 STALL=${STALL:-1800} POLL=${POLL:-120} setsid nohup bash $HERE/watch_gfmc8.sh $2 $L > $L/watcher.log 2>&1 < /dev/null &
    sleep 1; echo "WATCHER armed for $2 ($1), ppid $(ps -o ppid= -p $! 2>/dev/null | tr -d ' ')"
  else
    echo "DRY: watcher watch_gfmc8.sh $2 $L (GF_REMRUN=$RUNS/$1)"
  fi
}
brief_chain() {  # name jid p seed M dt nb burn wh
  local L=$LR/$1
  [ "$GO" = 1 ] || return 0
  mkdir -p $L; cat > $L/ANALYSIS_BRIEF.md <<EOB
# M1.3b 8x8 GFMC chain $1 (job $2): guide p=$3 (translations x spin-flip RBM+PP; p=1 adds one signed Lanczos hop), seed $4, M=$5, dtau=$6, $7 blocks after $8 burn blocks, QoS $QOS
Question: fixed-node energy E_FN per site at 8x8 (J2/J1 = 0.5, PBC) for this guide, with the two independent chains of the same p, against the VMC
p=0 / p=1 energies of the SAME guide (vmc_moments.py, $RUNS/V) and the references: published RBM+PP -0.498886, extrapolated -0.49906, DMRG -0.4992.
Parameters: $PARAMS  (consts alpha, E, w from $RUNS/V/consts.json).  GPU-h per number = sum of block seconds of the chains.
Files per chain (pulled to $L): progress.txt (last line GFMC_DONE), blocks.npz (emix, egrow, ess, elvar, hops, sec, evals per block), result.json.
Gate: python3 $CODE/gfmc8_analyze.py $LR   (when both chains of a guide are done): per-chain blocking error, tau_int, drift in fifths, chain
difference, tools/quotable.py verdict on the pair (series with the burn removed), table E_FN vs VMC vs references with GPU-h per number.
Next per verdict: QUOTABLE and chains agree within 3 sigma and no drift over fifths beyond 3 sigma -> report E_FN(p), gap to VMC, gap to the
references, GPU-h. Drift > 3 sigma in fifths or chain difference > 3 sigma -> relaxation/trapping: continue the chains (RESUME=1) before any number.
GFMC_DONE with completed False = --maxsec was hit: RESUME=1. NOT QUOTABLE -> report which check failed, no number.  dtau and 1/M bias: stage X (p=0 chains).
Tests before production: check_general.py (4x4, complex RBM), verify_general_6x6.py (6x6 regression), pilots (see README of the step).
EOB
}
if [ "$STAGE" = V ] || [ "$STAGE" = T ]; then
  name=V; [ "$STAGE" = T ] && name=Vtest
  body="$(header $name $VWH)
PY=$VENV/bin/python
\$PY timing8.py \$RUN/timing8.json $PARAMS 8 translations 128 256 512 > \$RUN/timing.txt 2>&1
\$PY vmc_moments.py \$RUN --params $PARAMS --L 8 --proj translations $VMC --chunk $CHUNK --maxsec_seed $(( (VWH*3600 - 1800) / 2 )) > \$RUN/stdout.txt 2>&1; rc=\$?
echo exit=\$rc finished=\$(date -Is) >> \$RUN/run_metadata.txt; echo \"GFMC_DONE exit=\$rc\" >> \$RUN/progress.txt; exit \$rc"
  jid=$(submit $name "$body" "") || exit 1
  [ "$GO" = 1 ] && echo $jid > $LR/JOB_$name
  if [ "$GO" = 1 ]; then mkdir -p $LR/$name; cat > $LR/$name/ANALYSIS_BRIEF.md <<EOB
# M1.3b 8x8 VMC p=0 / p=1 (job $jid): params $PARAMS
Question: VMC <H> and variance of the real guide v = Re[w psi_P] (p=0) and of the one-hop Lanczos state (p=1, Sorella style), per site, with errors
from the chain-group jackknife; the walker constants alpha, E, w (consts.json).  Files in $LR/$name: progress.txt (PROGRESS RESULT lines, last GFMC_DONE),
vmc_s1.json, vmc_s2.json, vmc_pooled.json, E0_s*.npy, E1_s*.npy (per-sweep series), timing8.json.
Gate: python3 /project/theorie/a/A.Otaifi/tools/quotable.py series E0_s1.npy E0_s2.npy --name 'E VMC p=0'  and the same for E1; check_h2 (<h1^2>/<vt h2>) must be 1.000 +- 0.01 and
ratio_vs_fastupdate_maxrel < 1e-8 in vmc_s1.json.  Next: gate passes -> stages A, B (queued afterok); fails -> do not run A, B, report.
EOB
  fi
  watch $name $jid
  [ "$STAGE" = V ] && exit 0
  VJOB=$jid
else
  VJOB=${DEP:-$( [ -f $LR/JOB_V ] && cat $LR/JOB_V )}
  [ "$GO" = 1 ] && [ -z "${VJOB:-}" ] && { echo "no V job id: run stage V first (or DEP=<jobid>, or consts exist already)"; exit 1; }
fi
echo "$CHAINS" | while read name p seed M dt nb burn wh; do
  [ -z "$name" ] && continue
  maxsec=$(( wh*3600 - 900 ))
  [ "$STAGE" = T ] && cname=${name}_test || cname=$name
  CONSTS=$RUNS/V/consts.json; [ "$STAGE" = T ] && CONSTS=$RUNS/Vtest/consts.json
  body="$(header $cname $wh)
$VENV/bin/python gfmc_run.py \$RUN --params $PARAMS --consts $CONSTS --p $p --L 8 --proj translations --M $M --dtau $dt --blocks $nb --burn $burn --seed $seed --chunk $CHUNK --maxsec $maxsec $([ "$RESUME" = 1 ] && echo --resume) > \$RUN/stdout.txt 2>&1; rc=\$?
echo exit=\$rc finished=\$(date -Is) >> \$RUN/run_metadata.txt; exit \$rc"
  jid=$(submit $cname "$body" "${VJOB:-}") || continue
  brief_chain $cname $jid $p $seed $M $dt $nb $burn $wh
  watch $cname $jid
done
