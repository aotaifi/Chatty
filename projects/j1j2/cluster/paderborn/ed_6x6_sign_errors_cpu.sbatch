#!/bin/bash
#SBATCH --job-name=chatty-ed6x6-sign
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --cpus-per-task=128
#SBATCH --mem=200G
#SBATCH --time=04:00:00
#SBATCH --output=/pc2/users/h/hpcalot/chatty/j1j2/results/ed6x6sign-%j.out
#SBATCH --error=/pc2/users/h/hpcalot/chatty/j1j2/results/ed6x6sign-%j.err
# Usage: sbatch [--time=..] --export=ALL,KMAX=2,N=100000,NSHARDS=16,CHUNK=256,MAXSEC=3000,START=0 ed_6x6_sign_errors_cpu.sbatch
set -euo pipefail
module load lang
module load Python/3.11.5-GCCcore-13.2.0
source /pc2/users/h/hpcalot/chatty/.venv-j1j2-py311/bin/activate
export JAX_PLATFORMS=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false
KMAX=${KMAX:-2}; N=${N:-100000}; NSHARDS=${NSHARDS:-16}; CHUNK=${CHUNK:-256}; MAXSEC=${MAXSEC:-1e9}; START=${START:-0}; SAMPLES=${SAMPLES:-samples_psi0sq_6x6.npz}; TAG=${TAG:-ed6x6_sign}
BASE=/pc2/users/h/hpcalot/chatty/j1j2
SRC=$BASE/ed_6x6_sign_errors
OUT="$BASE/results/ed_6x6_sign_errors_${SLURM_JOB_ID}"
mkdir -p "$OUT"
RUN=$(mktemp -d "${TMPDIR:-/tmp}/chatty-ed6sign.XXXXXX")
finish(){ cp -f "$RUN"/${TAG}*.npz "$OUT/" 2>/dev/null || true; cp -f "$RUN"/shard*.log "$OUT/" 2>/dev/null || true; }
trap finish EXIT
cp "$SRC/ed_6x6_sign_errors.py" "$OUT/"; cp "$SRC/ed_6x6_sign_errors.py" "$RUN/"
cp "$BASE/inputs/vit_J2=0.50_N=6x6_k=0.mpack" "$BASE/inputs/krylov_scaling_6x6_a1.20_tr4096_va2048.json" "$BASE/inputs/$SAMPLES" "$RUN/"
cd "$RUN"
{ echo "job_id=${SLURM_JOB_ID}"; echo "host=$(hostname)"; echo "KMAX=$KMAX N=$N NSHARDS=$NSHARDS CHUNK=$CHUNK MAXSEC=$MAXSEC START=$START"; echo "started=$(date -Is)"; } > "$OUT/run_metadata.txt"
CPS=$(( SLURM_CPUS_PER_TASK / NSHARDS ))
for ((j=0; j<NSHARDS; j++)); do
  lo=$(( j*CPS )); hi=$(( (j+1)*CPS - 1 ))
  OMP_NUM_THREADS=$CPS taskset -c ${lo}-${hi} python ed_6x6_sign_errors.py --kmax $KMAX --n $N --shard $j --nshards $NSHARDS \
     --chunk $CHUNK --batch 4096 --max-seconds $MAXSEC --start $START --samples $SAMPLES --out $TAG > shard${j}.log 2>&1 &
done
wait
echo "finished=$(date -Is)" >> "$OUT/run_metadata.txt"
