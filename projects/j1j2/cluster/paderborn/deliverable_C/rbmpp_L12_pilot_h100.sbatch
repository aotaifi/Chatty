#!/bin/bash
# DRAFT, not submitted. 1-hour cost pilot for the 12x12 guide (PLAN.md option b): PP alone (Nomura-Imada ansatz, own
# JAX code experiments/rbmpp_6x6, L-generic, tests pass at 4x4/6x6), 300 SR steps, then a short full-projection eval.
# Measures: seconds per SR step, memory of S (~21k parameters at L = 12), energy trend. Decides the 12x12/14x14 budget.
# Staging: experiments/rbmpp_6x6/*.py -> $BASE/src/rbmpp/
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --job-name=pp12pilot
#SBATCH --output=/pc2/users/h/hpcalot/chatty/j1j2/results/delC-pp12pilot-%j.out
set -euo pipefail
module load lang
module load Python/3.11.5-GCCcore-13.2.0
source /pc2/users/h/hpcalot/chatty/.venv-j1j2-py311/bin/activate
export JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
BASE=/pc2/users/h/hpcalot/chatty/j1j2
OUT="$BASE/results/delC_pp12pilot_${SLURM_JOB_ID}"; mkdir -p "$OUT"; cd "$BASE/src/rbmpp"
cat > "$OUT/spec.json" <<J
{"L": 12, "rbm": false, "seed": 1, "steps": 300, "chains": 4096, "lr": 0.05, "lr_end": 0.03, "shift": 1e-3,
 "max_dnorm": 0.3, "chunk": 128, "ckpt_every": 25, "eval_samples": 16384, "eval_proj": ["simple"]}
J
python rbmpp_train.py "$OUT" "$OUT/spec.json"
