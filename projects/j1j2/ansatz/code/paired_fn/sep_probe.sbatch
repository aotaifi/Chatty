#!/bin/bash
#SBATCH --job-name=sepprobe
#SBATCH --partition=cip
#SBATCH --gres=gpu:a40-24gb:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=36G
#SBATCH --time=00:50:00
#SBATCH --no-requeue
# T-SEP rho probe on the 2nd cip slice. usage: sbatch --output RUN/slurm-%j.out sep_probe.sbatch RUN ARM [MAXSEC] [CTL_BLOCKS] [BURN]
# ARM exact | stored4.  M 256, control A=A first (ctl blocks), then the paired run until MAXSEC (inside sep_probe.py); ~0.4 A40-h of work against a 50 min wall.
# stored4 reads n_4 (path sha256) from sign_bad/r4/final_net.txt and refuses to start without it.  XLA autotune ON (net convs 9-12x slower off), as r3_fn_net.sbatch.
set -uo pipefail
RUN=$1; ARM=$2; MAXSEC=${3:-1500}; CTL=${4:-10}; BURN=${5:-10}; mkdir -p "$RUN"
C=/project/theorie/a/A.Otaifi/Chatty/organize/projects/j1j2/ansatz/code/paired_fn
export OMPI_MCA_pml=ob1 OMPI_MCA_btl="vader,self,tcp"
export PYTHONDONTWRITEBYTECODE=1 XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.85 OMP_NUM_THREADS=1
unset JAX_COMPILATION_CACHE_DIR XLA_FLAGS
PY=/project/theorie/a/A.Otaifi/Chatty/runs/a40slice/venv_jax_0.8.2/bin/python
NET4=(); if [ "$ARM" = stored4 ]; then
  F=/scratch/a/A.Otaifi/ansatz_runs/sign_bad/r4/final_net.txt; [ -s $F ] || { echo "SEP_PROBE_FAILED no final_net.txt" >> $RUN/progress.txt; exit 3; }
  read N4P N4SHA < $F; [ "$(sha256sum $N4P | cut -d' ' -f1)" = "$N4SHA" ] || { echo "SEP_PROBE_FAILED n4 sha" >> $RUN/progress.txt; exit 3; }; NET4=(--net4 $N4P); fi
{ echo job_id=$SLURM_JOB_ID; hostname; nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader; date -Is; echo "arm $ARM maxsec $MAXSEC ctl $CTL burn $BURN"; (cd $C; sha256sum engine.py providers.py analysis.py cnn_sign.py sep_probe.py); } > $RUN/run_metadata.txt
cd $C; $PY sep_probe.py $RUN --arm $ARM "${NET4[@]}" --M 256 --ctl_blocks $CTL --burn $BURN --maxsec $MAXSEC > $RUN/sep_probe.log 2>&1; rc=$?
echo "sep_probe_exit=$rc $(date -Is)" >> $RUN/run_metadata.txt; sacct -j $SLURM_JOB_ID --format=JobID,Elapsed,MaxRSS,State -P >> $RUN/run_metadata.txt 2>/dev/null
exit $rc
