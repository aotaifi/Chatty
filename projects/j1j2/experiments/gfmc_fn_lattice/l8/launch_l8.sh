#!/bin/bash
# Test L8 production launcher (STAGED, NOT SUBMITTED by the builder): one loop round at 8x8 with walkers (README compress_6x6, "Test L8").
# DRY RUN by default (prints every rsync / sbatch / watcher action, writes nothing remote).  GO=1 submits the whole chain at once:
#   D  stage 1  walker dump: resume COPIES of M1.3b's p1_s1 / p1_s2 ckpt.npz (equilibrated, no burn) with --dump for NB_D blocks (2 jobs)
#   F  stage 2  refit (l8_refit.py, SR on R; train D_s1, held-out = chain D_s2; early stopping)      afterok:D_s1:D_s2
#               exit 3 = weight-health STOP, exit 4 = held-out R did not rise by 3 sigma -> job FAILED, stage 3 never starts
#   V  stage 3a (V and G: --kill-on-invalid-dep=yes) VMC on the refitted params (vmc_moments.py: phase w, E, alpha by Ritz) -> consts.json    afterok:F
#   G  stage 3b 2 p=1 FN chains with g_1 (M 256, dtau 0.1, as M1.3b)                                    afterok:V
#   usage: [GO=1] [QOS=cont] launch_l8.sh [all|D|F|V|G]      (a single stage needs DEP=<jobid[:jobid]> for its afterok, or none)
# Decision gate after G: python3 l8_decide.py $LOCALROOT/g1  (L8_VERDICT GAIN|NO_GAIN|LOSS|UNDECIDED|NOT_QUOTABLE).
# SIZING from the 8x8 pilot (job 3922180), see the SIZING block and l8/STATUS.md.
set -u
STAGE=${1:-all}; GO=${GO:-0}; QOS=${QOS:-cont}; CHUNK=${CHUNK:-256}
HERE=$(cd "$(dirname "$0")" && pwd); CODE=$HERE/..; RBMPP=$CODE/../rbmpp_6x6
DELTA=/project/theorie/a/A.Otaifi/Chatty/organize/projects/j1j2/delta_today.md
B=/scratch/hpc-prf-ppdmrg/hpcalot; PY=$B/venv_jax_0.8.2/bin/python
PARAMS0=$B/rbmpp8/runs/rb8_prod/params_final.npz; V0=$B/gfmc8/runs/rb8/V; SRC=$B/gfmc8/runs/rb8
RUNS=$B/l8/runs/prod; LOCALROOT=${LOCALROOT:-/scratch/a/A.Otaifi/l8/prod}
SSH="ssh -o BatchMode=yes -o HostName=login4"; SL=/opt/software/slurm/25.11.8-1/bin
# SIZING_BEGIN  (pilot Paderborn job 3922180, H100, 2026-10-25; run dir /scratch/hpc-prf-ppdmrg/hpcalot/l8/runs/pilot_3922180)
#   D: 13.1 s/block with --dump (median of 20 resumed p1_s1 blocks; 13.0 without), dump 21.9 kB/block -> 400 blocks = 1.46 h per chain (against 3 h wall), 8.8 MB per chain.
#   F: guide values 0.41 ms/walker; SR step 2.7 s with 4096 training walkers + 4096 VMC samples (17 s first step incl. compile).  Production: 2 x 320 training
#      blocks = 164k walkers; walker reverse pass scales with them: <= 40 x 2.7 s ~ 2 min per step (upper bound) -> 20 steps <= 0.7 h (against 3 h wall).
#   V: 0.95 h (rb8/V, same code and sizes, job 3832999) against 3 h.  G: 13.1 s/block -> 1080 blocks (930 + 150 burn, as M1.3b p1) = 3.9 h per chain against 8 h.
#   Total: D 2.9 + F <= 0.7 + V 0.95 + G 7.9 = ~12.5 H100-h (budget 15).
NB_D=${NB_D:-400}; WH_D=${WH_D:-3}          # dump blocks per chain, wall hours per dump job
STEPS_F=${STEPS_F:-20}; LR_F=${LR_F:-0.3}; MAXN_F=${MAXN_F:-0.02}; WH_F=${WH_F:-3}
WH_V=3; VMC="--chains 1024 --sweeps 80 --seeds 1,2"
NB_G=${NB_G:-930}; BURN_G=${BURN_G:-150}; WH_G=${WH_G:-8}   # per chain; a --maxsec stop is resumed by resubmitting stage G (same names, --resume)
# SIZING_END
run() { if [ "$GO" = 1 ]; then eval "$@"; else echo "DRY: $*"; fi; }
if [ "$GO" = 1 ]; then
  $SSH paderborn "test -f $PARAMS0 && test -f $V0/consts.json && test -f $SRC/p1_s1/ckpt.npz && test -f $SRC/p1_s2/ckpt.npz" || { echo "inputs missing on Paderborn"; exit 1; }
  $SSH paderborn "mkdir -p $B/l8/code $RUNS" || exit 1
  rsync -a -e "$SSH" $CODE/{gfmc_fn_lib,gfmc_run,vmc_moments}.py $HERE/{l8_lib,l8_refit,l8_vmc_prefix}.py $RBMPP/rbmpp_lib.py paderborn:$B/l8/code/ || exit 1
  mkdir -p $LOCALROOT
else
  echo "DRY: rsync gfmc_fn_lib gfmc_run vmc_moments l8_lib l8_refit l8_vmc_prefix rbmpp_lib -> paderborn:$B/l8/code/; local $LOCALROOT"
fi
submit() {  # name body dep -> job id
  local name=$1 body=$2 dep=$3 sb=$RUNS/$1.sbatch
  if [ "$GO" = 1 ]; then
    printf '%s\n' "$body" | $SSH paderborn "mkdir -p $RUNS/$name; cat > $sb"
    jid=$($SSH -n paderborn "export PATH=$SL:\$PATH; sbatch --parsable ${dep:+--dependency=afterok:$dep} $sb")
    [[ "$jid" =~ ^[0-9]+$ ]] || { echo "SUBMIT FAILED for $name: '$jid'" >&2; return 1; }
    echo "SUBMITTED $name job $jid ${dep:+afterok:$dep}" >&2; echo "$jid"
  else
    echo "DRY: submit $name ${dep:+(afterok:$dep)} via sbatch $sb" >&2; echo "DRY_$name"
  fi
}
header() {  # name hours [kill: cancel the job if its afterok dependency can never be satisfied]
  echo "#!/bin/bash${3:+
#SBATCH --kill-on-invalid-dep=yes}
#SBATCH --job-name=l8_${1//\//_}
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
RUN=$RUNS/$1; mkdir -p \$RUN/code; cp $B/l8/code/*.py \$RUN/code/
export XLA_FLAGS=\"--xla_gpu_autotune_level=0\" XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 OMP_NUM_THREADS=8
{ echo job_id=\$SLURM_JOB_ID; hostname; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader; date -Is; sha256sum $PARAMS0 \$RUN/code/*.py; } >> \$RUN/run_metadata.txt
cd \$RUN/code"
}
watch() {  # name jid donepat
  local L=$LOCALROOT/$1
  if [ "$GO" = 1 ]; then
    mkdir -p $L
    GF_REMRUN=$RUNS/$1 GF_DONE_PAT=$3 STALL=${STALL:-1800} POLL=${POLL:-120} setsid nohup bash $CODE/m13b_8x8/watch_gfmc8.sh $2 $L > $L/watcher.log 2>&1 < /dev/null &
    sleep 1; echo "WATCHER armed for $2 ($1), pid $!, ppid $(ps -o ppid= -p $! 2>/dev/null | tr -d ' ')"
  else
    echo "DRY: watcher watch_gfmc8.sh $2 $L (GF_REMRUN=$RUNS/$1, verdict line $3)"
  fi
}
brief() {  # name jid text
  [ "$GO" = 1 ] || { echo "DRY: ANALYSIS_BRIEF.md for $1"; return 0; }
  mkdir -p $LOCALROOT/$1; printf '%s\n' "$3" > $LOCALROOT/$1/ANALYSIS_BRIEF.md
}
COMMON="Test L8 (README compress_6x6 'Test L8'; l8/STATUS.md): one loop round at 8x8 with walkers. g_0 = rb8_prod RBM+PP + one signed Lanczos hop
(consts $V0/consts.json: alpha -0.17543187, E -31.91513584), E_FN(g_0) p=1 from M1.3b (gfmc_m13b/prod/rb8/analysis_final.json)."
JD=""; jF=""; jV=""
if [ "$STAGE" = all ] || [ "$STAGE" = D ]; then
  for s in 1 2; do
    name=D_s$s
    body="$(header $name $WH_D)
[ -f \$RUN/ckpt.npz ] || cp $SRC/p1_s$s/ckpt.npz \$RUN/ckpt.npz
NB0=\$($PY -c \"import numpy as np; print(int(np.load('\$RUN/ckpt.npz', allow_pickle=True)['nb']))\")
[ -f \$RUN/NB_START ] || echo \$NB0 > \$RUN/NB_START; NBS=\$(cat \$RUN/NB_START)
$PY gfmc_run.py \$RUN --params $PARAMS0 --consts $V0/consts.json --p 1 --L 8 --proj translations --M 256 --dtau 0.1 --blocks \$((NBS - 150 + $NB_D)) --burn 150 \\
  --seed $s --chunk $CHUNK --resume --dump --maxsec $((WH_D*3600 - 900)) > \$RUN/stdout.txt 2>&1; rc=\$?
echo exit=\$rc finished=\$(date -Is) >> \$RUN/run_metadata.txt; exit \$rc"
    jid=$(submit $name "$body" "${DEP:-}") || exit 1
    JD="${JD:+$JD:}$jid"
    brief $name $jid "# L8 stage 1 walker dump $name (job $jid), QoS $QOS
$COMMON
Question: walker configurations of the p=1 FN-GFMC with g_0 (mixed distribution g_0 phi_0): resume of a COPY of $SRC/p1_s$s/ckpt.npz (seed $s chain,
equilibrated over 1080 blocks), $NB_D more blocks with --dump (pre-resampling X, logw, EL per block in dump/blk_NNNNN.npz on Paderborn: $RUNS/$name/dump).
Files (pulled to here, dump stays remote): progress.txt (last line GFMC_DONE ... completed True|False), blocks.npz, result.json.
Gate: the last line is 'GFMC_DONE ... completed True' and the E_mix/N of the new blocks agrees with the M1.3b p=1 value -0.498946 within 3 x its
blocking error (python3 $CODE/gfmc8_analyze.py is not needed; read result.json 'mix').  Count: ssh paderborn ls $RUNS/$name/dump | wc -l = $NB_D.
Next: gate passes -> nothing to do, stage F (refit) is queued afterok.  completed False (maxsec) -> resubmit this stage with RESUME (rerun
launch_l8.sh D with the same names: it resumes from \$RUN/ckpt.npz) and re-chain F/V/G with DEP.  E_mix off by > 3 sigma -> stop, report (a
resume that changed the walk)."
    watch $name $jid GFMC_DONE
  done
fi
if [ "$STAGE" = all ] || [ "$STAGE" = F ]; then
  name=F; dep=${JD:-${DEP:-}}
  body="$(header $name $WH_F)
$PY l8_refit.py \$RUN --params0 $PARAMS0 --consts $V0/consts.json --dumps $RUNS/D_s1,$RUNS/D_s2 --L 8 --proj translations --steps $STEPS_F \\
  --lr $LR_F --maxn $MAXN_F --shift 0.1 --sweeps_per_step 8 --chains 1024 --holdout_chain 1 --binb 2 --seed 1 --chunk $CHUNK --gate_sigma 3 --save_every 1 --select best --maxsec $((WH_F*3600 - 1200)) > \$RUN/stdout.txt 2>&1; rc=\$?
echo exit=\$rc finished=\$(date -Is) >> \$RUN/run_metadata.txt; exit \$rc"
  jF=$(submit $name "$body" "$dep") || exit 1
  brief $name $jF "# L8 stage 2 refit (job $jF), QoS $QOS
$COMMON
Question: does the refit of the same RBM+PP class to phi_0 (fidelity, walker samples only) raise the held-out relative fidelity
R = F(v)/F(v_0) beyond its error?  $STEPS_F SR steps, lr $LR_F, |d|_S <= $MAXN_F, 1024 VMC chains x 8 sweeps per step; trained on D_s1 (all blocks),
held-out = the independent chain D_s2 (all blocks); params_final.npz = the step with the largest held-out R (early stopping; line SELECTED step k).
Files: progress.txt (WEIGHT_HEALTH line; one PROGRESS line per step; last line REFIT_DONE verdict PASS|NO_RISE|HEALTH_STOP), refit.json (curve:
R_held, R_train with errors per step; final.selected_step), health.json, params_final.npz and params_tNNN.npz (remote only: $RUNS/F/).
Gate: grep 'SELECTED\|REFIT_DONE' progress.txt; WEIGHT_HEALTH must show neff/n >= 0.1 and max_share <= 1e-2; held-out R at the selected step > 1 by
> 3 sigma (re-evaluated with fresh VMC samples; the job exits 0 only then).  Report the selected step and both curves (refit.json); a selected
step that is the last one means the curve had not turned down.  Caveat: selecting the maximum of a noisy curve biases its value up (4x4 check: z in STATUS.md).
Next: PASS -> stage V and G run automatically (afterok).  NO_RISE (exit 4, job FAILED) -> stage 3 does not run: V and G carry --kill-on-invalid-dep=yes, so Slurm
cancels them and their watchers end with FAILED (check sacct shows them CANCELLED); report the curve: the refit does
not gain fidelity on held-out walkers (lead decides: more dump blocks or a different lr, not a different estimator).  HEALTH_STOP (exit 3) ->
design failure of the a/g weights at 8x8: V and G are cancelled by Slurm (kill-on-invalid-dep); report the numbers, no retuning."
  watch $name $jF REFIT_DONE
fi
if [ "$STAGE" = all ] || [ "$STAGE" = V ]; then
  name=V; dep=${jF:-${DEP:-}}
  body="$(header $name $WH_V kill)
$PY vmc_moments.py \$RUN --params $RUNS/F/params_final.npz --L 8 --proj translations $VMC --chunk $CHUNK --maxsec_seed $(( (WH_V*3600 - 1800) / 2 )) > \$RUN/stdout.txt 2>&1; rc=\$?
sha256sum $RUNS/F/params_final.npz >> \$RUN/run_metadata.txt
echo exit=\$rc finished=\$(date -Is) >> \$RUN/run_metadata.txt; [ \$rc = 0 ] && echo \"VMC_DONE_L8 exit=\$rc\" >> \$RUN/progress.txt; exit \$rc"
  jV=$(submit $name "$body" "$dep") || exit 1
  brief $name $jV "# L8 stage 3a VMC of g_1 (job $jV): params $RUNS/F/params_final.npz (the refit a_1)
$COMMON
Question: VMC p=0 / p=1 energies of a_1 and the walker constants alpha (Ritz), E, w for g_1 = a_1 + one signed hop (consts.json).
Files: progress.txt (PROGRESS RESULT lines, VMC_DONE), vmc_s1.json, vmc_s2.json, vmc_pooled.json, consts.json, E0_s*.npy, E1_s*.npy.
Gate: python3 /project/theorie/a/A.Otaifi/tools/quotable.py series E0_s1.npy E0_s2.npy --name 'E VMC p=0 a_1' (same for E1_s*); check_h2 within 2 jackknife
errors of 1 and ratio_vs_fastupdate_maxrel < 1e-6 in vmc_s1.json.  Compare E_1 with g_0's (rb8/V: p=1 E1/N -0.4989 range).
Next: gate passes -> stage G runs (afterok).  Fails -> scancel the G jobs and report."
  watch $name $jV VMC_DONE_L8
fi
if [ "$STAGE" = all ] || [ "$STAGE" = G ]; then
  dep=${jV:-${DEP:-}}
  for s in 1 2; do
    name=g1/p1_s$s
    body="$(header g1/p1_s$s $WH_G kill)
$PY gfmc_run.py \$RUN --params $RUNS/F/params_final.npz --consts $RUNS/V/consts.json --p 1 --L 8 --proj translations --M 256 --dtau 0.1 \\
  --blocks $NB_G --burn $BURN_G --seed $s --chunk $CHUNK --maxsec $((WH_G*3600 - 900)) --resume > \$RUN/stdout.txt 2>&1; rc=\$?
echo exit=\$rc finished=\$(date -Is) >> \$RUN/run_metadata.txt; exit \$rc"
    jid=$(submit $name "$body" "$dep") || exit 1
    brief $name $jid "# L8 stage 3b FN chain g_1 p1_s$s (job $jid): params $RUNS/F/params_final.npz, consts $RUNS/V/consts.json, M 256, dtau 0.1, $NB_G blocks after $BURN_G burn
$COMMON
Question: E_FN(g_1) per site vs E_FN(g_0) = -0.498946 +- 4.6e-6 (M1.3b p=1 final, 2 chains x 1080 blocks, QUOTABLE); decision rule (sigma = combined error, ~7e-6):
dE < -3 sigma GAIN (loop gains at 8x8: next a second round, then 10x10); |dE| <= 2 sigma NO_GAIN (as at 6x6: the loop's future is the storage object,
ansatz angle); dE > +3 sigma LOSS (same conclusion, stronger).
Files (pulled to $LOCALROOT/g1/p1_s$s): progress.txt (GFMC_DONE ... completed True|False), blocks.npz, result.json.
Gate (when BOTH g_1 chains are done): python3 $HERE/l8_decide.py $LOCALROOT/g1   -> line 'L8_VERDICT ...' (runs gfmc8_analyze.py + quotable.py).
Next per verdict: GAIN / NO_GAIN / LOSS -> report it with dE, sigma, H100-h; NOT_QUOTABLE or chain difference > 3 sigma or drift > 3 sigma in
fifths -> continue both chains (resubmit stage G with the same names: --resume) before any verdict; GFMC_DONE completed False -> resubmit stage G
(it resumes); UNDECIDED (2-3 sigma) -> lead decides (extend chains)."
    watch $name $jid GFMC_DONE
  done
fi
