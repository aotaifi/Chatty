#!/bin/bash
#SBATCH --job-name=l8pilot
#SBATCH --account=hpc-prf-ppdmrg
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:22:00
#SBATCH --no-requeue
#SBATCH --qos=express_hpcalot
#SBATCH --output=/scratch/hpc-prf-ppdmrg/hpcalot/l8/runs/pilot-%j.out
# Test L8 8x8 pilot (one job, ~11 min): (1) resume a COPY of M1.3b p1_s1's ckpt.npz with --dump for NB blocks (s/block, dump size);
# (2) control (i): zero-step refit -> stage-3 VMC prefix vs rb8/V (and the perturbed broken version); (3) refit pilot: weight health of
# x v/g on the dumped walkers and a few SR steps with the held-out R curve.  The original p1_s1 run dir is only read.
set -uo pipefail
B=/scratch/hpc-prf-ppdmrg/hpcalot; G8=$B/gfmc8; PY=$B/venv_jax_0.8.2/bin/python
PARAMS=$B/rbmpp8/runs/rb8_prod/params_final.npz; V=$G8/runs/rb8/V; SRC=$G8/runs/rb8/p1_s1
NB=${NB:-20}; STEPS=${STEPS:-8}
RUN=$B/l8/runs/pilot_$SLURM_JOB_ID; mkdir -p $RUN/code $RUN/D1; cp $B/l8/code/*.py $RUN/code/
export XLA_FLAGS="--xla_gpu_autotune_level=0" XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 OMP_NUM_THREADS=8
{ echo job_id=$SLURM_JOB_ID; hostname; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader; date -Is; echo params=$PARAMS src=$SRC NB=$NB STEPS=$STEPS
  sha256sum $PARAMS $V/consts.json $SRC/ckpt.npz $RUN/code/*.py; } > $RUN/run_metadata.txt
P() { echo "$(date -Is) PROGRESS $*" >> $RUN/progress.txt; }
cp $SRC/ckpt.npz $RUN/D1/ckpt.npz
cd $RUN/code
NB0=$($PY -c "import numpy as np; print(int(np.load('$RUN/D1/ckpt.npz', allow_pickle=True)['nb']))")
P "stage D: resume copy of $SRC at block $NB0 for $NB blocks with --dump"
$PY gfmc_run.py $RUN/D1 --params $PARAMS --consts $V/consts.json --p 1 --L 8 --proj translations --M 256 --dtau 0.1 --blocks $((NB0 - 150 + NB)) --burn 150 \
   --seed 1 --chunk 256 --resume --dump --maxsec 360 > $RUN/D1/stdout.txt 2>&1; rcD=$?
P "stage D exit $rcD; $(grep -c 'PROGRESS block' $RUN/D1/progress.txt) block lines; dump $(du -s --apparent-size --block-size=1 $RUN/D1/dump | cut -f1) bytes in $(ls $RUN/D1/dump | wc -l) files"
P "stage Z: zero-step refit"
$PY l8_refit.py $RUN/Z --params0 $PARAMS --consts $V/consts.json --dumps $RUN/D1 --steps 0 --no_gate --no_health_stop --chains 256 --binb 1 > $RUN/Z.log 2>&1; rcZ=$?
P "stage Z exit $rcZ; $(tail -1 $RUN/Z/progress.txt)"
$PY l8_vmc_prefix.py $RUN/ZV --params $RUN/Z/params_final.npz --ref $V --sweeps 2 > $RUN/ZV.log 2>&1; rcV=$?
P "stage ZV exit $rcV; $(tail -1 $RUN/ZV/progress.txt)"
$PY l8_vmc_prefix.py $RUN/ZB --params $RUN/Z/params_final.npz --ref $V --perturb 1e-7 --stageA_only > $RUN/ZB.log 2>&1; rcB=$?
P "stage ZB (broken: f[0,1] x (1+1e-7)) exit $rcB; $(tail -1 $RUN/ZB/progress.txt)"
P "stage F: refit pilot $STEPS steps"
$PY l8_refit.py $RUN/F --params0 $PARAMS --consts $V/consts.json --dumps $RUN/D1 --steps $STEPS --no_gate --no_health_stop --chains 1024 --binb 1 \
   --save_every 0 --final_sweeps 4 --maxsec 240 > $RUN/F.log 2>&1; rcF=$?
P "stage F exit $rcF; $(tail -1 $RUN/F/progress.txt)"
echo exit=$rcD,$rcZ,$rcV,$rcB,$rcF finished=$(date -Is) >> $RUN/run_metadata.txt
P "PILOT_DONE exits D $rcD Z $rcZ ZV $rcV ZB $rcB F $rcF"
