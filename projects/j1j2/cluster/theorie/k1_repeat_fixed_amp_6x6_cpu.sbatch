#!/bin/bash
#SBATCH --job-name=chatty-k1rep-6x6
#SBATCH --partition=cluster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=/home/a/A.Otaifi/chatty_k1rep/logs/k1rep-%j.out
#SBATCH --error=/home/a/A.Otaifi/chatty_k1rep/logs/k1rep-%j.err
# LMU theorie CPU fallback (Paderborn GPUs blocked by the Oct 5-9 shutdown).
# Usage: sbatch [--cpus-per-task N] k1_repeat_fixed_amp_6x6_cpu.sbatch [script args...]
# Env: ~/chatty_k1rep/venv311 = jax 0.10.2 / flax 0.12.8 / netket 3.22.4 (CPU) + nqsmagic copied from Paderborn src.
set -euo pipefail
module load python/3.11-2023.09
BASE=/home/a/A.Otaifi/chatty_k1rep
OUT="$BASE/results/k1_repeat_fixed_amp_6x6_${SLURM_JOB_ID}"
mkdir -p "$OUT"
export JAX_PLATFORMS=cpu OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export XLA_FLAGS="--xla_force_host_platform_device_count=1"
cp "$BASE/k1_repeat_fixed_amp_6x6.py" "$OUT/"
cd "$OUT"
for f in "vit_J2=0.50_N=6x6_k=0.mpack" energy_krylov_vs_vit_6x6_indep.npz krylov_scaling_6x6_a1.20_tr4096_va2048.npz krylov_scaling_6x6_a1.20_tr4096_va2048.json; do ln -sf "$BASE/$f" .; done
{ echo "job_id=${SLURM_JOB_ID}"; echo "host=$(hostname)"; echo "cpus=${SLURM_CPUS_PER_TASK}"; echo "args=$*"; echo "started=$(date -Is)"; lscpu | grep -E "Model name|^CPU\(s\)"; } > run_metadata.txt
"$BASE/venv311/bin/python" k1_repeat_fixed_amp_6x6.py "$@"
echo "finished=$(date -Is)" >> run_metadata.txt
