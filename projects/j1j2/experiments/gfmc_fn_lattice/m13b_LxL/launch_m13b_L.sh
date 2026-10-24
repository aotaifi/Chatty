#!/bin/bash
# M1 LxL production launcher (STAGED, NOT RUN; generalisation of ../m13b_8x8/launch_m13b.sh, which is unchanged): LxL (L = 10, 12) fixed-node GFMC with the translations x spin-flip RBM+PP guide, p = 0 and p = 1.
# DRY RUN by default: prints every sbatch / ssh / watcher command it would run and writes nothing remote.
#   [GO=1] LSZ=10|12 launch_m13b_L.sh STAGE PARAMS_PATH [TAG]      (LSZ is an environment variable, required)
#     STAGE   T  smoke test (3 min): 4 VMC sweeps + 3 blocks of each guide, checks the whole path
#             V  VMC p = 0 / p = 1 production (vmc_moments.py, two independent seeds, pooled) -> consts.json (alpha, E, w) for the walkers
#             A  GFMC p = 0 chains (two independent seeds)           [needs V's consts.json: submitted afterok:V]
#             B  GFMC p = 1 chains (two independent seeds)           [needs V's consts.json: submitted afterok:V]
#             X  bias scans on p = 0 (cheap guide): dtau and M variants of one chain (the bias at 8x8 is NOT assumed small)
#     PARAMS_PATH  a params .npz ON PADERBORN (e.g. /scratch/hpc-prf-ppdmrg/hpcalot/rbmpp8/runs/rb8_prod/params_final.npz)
#     TAG     run family name (default: the parameter file's run directory name); everything lives in $RB/runs/$TAG/
#   GO=1 LSZ=10 launch_m13b_L.sh STAGE PARAMS [TAG]   rsync code, submit, arm one durable watcher per job, write ANALYSIS_BRIEF.md per chain.
# Environment: QOS (default cont; express_hpcalot has ~0.5 H100-h left in October, the pilots used 0.49; lowcont is the fallback), CHUNK (default 256 at L=10, 128 at L=12), LOCALROOT, DEP (job id
#   to depend on instead of the V job read from LOCALROOT/TAG/JOB_V), RESUME=1 (continue chains after a TIMEOUT: same names, --resume).
# Each chain: gfmc_run.py with --maxsec just under the sbatch time; checkpoint every 10 blocks; a stop on --maxsec before its block count
# prints GFMC_DONE completed False: resubmit with RESUME=1 (nothing else changes).  A chain is an independent seed (own init walkers and RNG).
# SIZING (pilots of 2026-10-24, pilot parameters pilot8_long; see SIZING in ANALYSIS_BRIEF.md): chain table below, written by the sizing step.
set -u
LSZ=${LSZ:?set LSZ=10 or LSZ=12}; case $LSZ in 10|12) ;; *) echo "LSZ must be 10 or 12"; exit 1;; esac
STAGE=${1:?stage T|V|A|B|X}; PARAMS=${2:?PARAMS_PATH on Paderborn}; GO=${GO:-0}; QOS=${QOS:-cont}; CHUNK=${CHUNK:-}; RESUME=${RESUME:-0}
TAG=${3:-$(basename "$(dirname "$PARAMS")")}
HERE=$(cd "$(dirname "$0")" && pwd); CODE=$HERE/..; RBMPP=$CODE/../rbmpp_6x6; WATCH=$CODE/m13b_8x8/watch_gfmc8.sh
DELTA=/project/theorie/a/A.Otaifi/Chatty/organize/projects/j1j2/delta_today.md
RB=/scratch/hpc-prf-ppdmrg/hpcalot/gfmcL/L$LSZ; LOCALROOT=${LOCALROOT:-/scratch/a/A.Otaifi/gfmc_m13b_L/L$LSZ}; LR=$LOCALROOT/$TAG
SSH="ssh -o BatchMode=yes -o HostName=login4"; SL=/opt/software/slurm/25.11.8-1/bin; VENV=/scratch/hpc-prf-ppdmrg/hpcalot/venv_jax_0.8.2
RUNS=$RB/runs/$TAG
case $LSZ in 10) REFE=-0.497629;; 12) REFE=-0.496791;; esac   # printed d_ref in the progress lines only (Nomura-Imada RBM+PP)
# ---- chain table: name p seed M dtau blocks burn walltime_h  (blocks counted after burn; burn = 10 tau at least)
# SIZING_TABLE_BEGIN
# Sizing (2026-10-25) from the pilots on H100 (express jobs 3860368 L10, 3861243 L12, 3862046 L12 chains 8192) and A40 (cip jobs 17485251/17486801, 17485252/17486881);
# model and every input in ../../rbmpp_6x6/m13_LxL/sizing.py (python3 sizing.py prints the table).  Production guide quality is NOT measured yet: sigma_walker is ASSUMED
# = the 8x8 production value x 8/L (p0 4.9e-3, p1 2.1e-3 per site at 8x8; constant Var(E_loc)/N of the guide); the cost of every chain scales with (sigma_w)^2.
#   L=10: s per walker-tau p0 0.032, p1 7.6 (pilot, poor guide: 0.026 / 4.2); VMC 0.14 s/sample (143 s per sweep of 1024).  H100-h: V 11.9 (E1 error 1e-5; 3.0 at 2e-5), A 1.0, B 19.6, X 1.8.
#   L=12: s per walker-tau p0 0.13, p1 44 (pilot: 0.117 / 20.1); VMC 0.56 s/sample.                                                  H100-h: V 32.8 (8.2 at 2e-5), A 3.0, B 85.8, X 7.
#   The p1 burn-in (15 tau at M=64) is 4 h (L10) / 23.5 h (L12) of those B hours; A40 hours = H100 hours x 7-10 (see README).
# After the first chain of each family: read sigma_walker from its progress lines (std of E_mix per block x sqrt(M)); if f = sigma_measured / sigma_assumed, cost x f^2.
case $LSZ in 10) CHUNK=${CHUNK:-256};; 12) CHUNK=${CHUNK:-128};; esac    # eloc memory: 20 GB (L10 chunk 256), 30 GB (L12 chunk 128) on H100
case $STAGE in
T) VMC="--chains 64 --sweeps 4 --stageA_sweeps 4 --stageA_chains 64 --therm 20 --seeds 1,2"; VWH=1
   CHAINS="
p0_s9 0 9 64 0.2 3 2 1
p1_s9 1 9 32 0.2 3 2 1";;
V) case $LSZ in 10) VMC="--chains 1024 --sweeps 150 --seeds 1,2"; VWH=14;; 12) VMC="--chains 1024 --sweeps 100 --seeds 1,2"; VWH=36;; esac; CHAINS="";;
A) case $LSZ in
   10) CHAINS="
p0_s1 0 1 1024 0.1 500 100 2
p0_s2 0 2 1024 0.1 500 100 2";;
   12) CHAINS="
p0_s1 0 1 1024 0.1 350 100 3
p0_s2 0 2 1024 0.1 350 100 3";;
   esac;;
B) case $LSZ in
   10) CHAINS="
p1_s1 1 1 64 0.1 600 150 12
p1_s2 1 2 64 0.1 600 150 12";;
   12) CHAINS="
p1_s1 1 1 64 0.1 400 150 48
p1_s2 1 2 64 0.1 400 150 48";;
   esac;;
X) case $LSZ in
   10) CHAINS="
p0_s3_M0256_dt0.1 0 3 256 0.1 1600 100 1
p0_s3_M4096_dt0.1 0 3 4096 0.1 150 100 2
p0_s3_M1024_dt0.2 0 3 1024 0.2 450 50 2";;
   12) CHAINS="
p0_s3_M0256_dt0.1 0 3 256 0.1 1600 100 4
p0_s3_M4096_dt0.1 0 3 4096 0.1 150 100 8
p0_s3_M1024_dt0.2 0 3 1024 0.2 450 50 8";;
   esac;;
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
    GF_REMRUN=$RUNS/$1 STALL=${STALL:-1800} POLL=${POLL:-120} setsid nohup bash $WATCH $2 $L > $L/watcher.log 2>&1 < /dev/null &
    sleep 1; echo "WATCHER armed for $2 ($1), ppid $(ps -o ppid= -p $! 2>/dev/null | tr -d ' ')"
  else
    echo "DRY: watcher watch_gfmc8.sh $2 $L (GF_REMRUN=$RUNS/$1)"
  fi
}
brief_chain() {  # name jid p seed M dt nb burn wh
  local L=$LR/$1
  [ "$GO" = 1 ] || return 0
  mkdir -p $L; cat > $L/ANALYSIS_BRIEF.md <<EOB
# M1 ${LSZ}x${LSZ} GFMC chain $1 (job $2): guide p=$3 (translations x spin-flip RBM+PP; p=1 adds one signed Lanczos hop), seed $4, M=$5, dtau=$6, $7 blocks after $8 burn blocks, QoS $QOS
Question: fixed-node energy E_FN per site at ${LSZ}x${LSZ} (J2/J1 = 0.5, PBC) for this guide, with the two independent chains of the same p, against the VMC
p=0 / p=1 energies of the SAME guide (vmc_moments.py, $RUNS/V) and the references: the references listed in $HERE/ANALYSIS_BRIEF.md for L=$LSZ.
Parameters: $PARAMS  (consts alpha, E, w from $RUNS/V/consts.json).  GPU-h per number = sum of block seconds of the chains.
Files per chain (pulled to $L): progress.txt (last line GFMC_DONE), blocks.npz (emix, egrow, ess, elvar, hops, sec, evals per block), result.json.
Gate: python3 $CODE/gfmc_L_analyze.py $LR --L $LSZ   (when both chains of a guide are done): per-chain blocking error, tau_int, drift in fifths, chain
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
\$PY timing8.py \$RUN/timing8.json $PARAMS $LSZ translations 128 256 512 > \$RUN/timing.txt 2>&1
\$PY vmc_moments.py \$RUN --params $PARAMS --L $LSZ --proj translations $VMC --chunk $CHUNK --maxsec_seed $(( (VWH*3600 - 1800) / 2 )) > \$RUN/stdout.txt 2>&1; rc=\$?
echo exit=\$rc finished=\$(date -Is) >> \$RUN/run_metadata.txt; echo \"GFMC_DONE exit=\$rc\" >> \$RUN/progress.txt; exit \$rc"
  jid=$(submit $name "$body" "") || exit 1
  [ "$GO" = 1 ] && echo $jid > $LR/JOB_$name
  if [ "$GO" = 1 ]; then mkdir -p $LR/$name; cat > $LR/$name/ANALYSIS_BRIEF.md <<EOB
# M1 ${LSZ}x${LSZ} VMC p=0 / p=1 (job $jid): params $PARAMS
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
$VENV/bin/python gfmc_run.py \$RUN --ref $REFE --params $PARAMS --consts $CONSTS --p $p --L $LSZ --proj translations --M $M --dtau $dt --blocks $nb --burn $burn --seed $seed --chunk $CHUNK --maxsec $maxsec $([ "$RESUME" = 1 ] && echo --resume) > \$RUN/stdout.txt 2>&1; rc=\$?
echo exit=\$rc finished=\$(date -Is) >> \$RUN/run_metadata.txt; exit \$rc"
  jid=$(submit $cname "$body" "${VJOB:-}") || continue
  brief_chain $cname $jid $p $seed $M $dt $nb $burn $wh
  watch $cname $jid
done
