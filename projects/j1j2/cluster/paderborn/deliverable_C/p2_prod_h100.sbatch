#!/bin/bash
# DRAFT, not submitted. PREREG E3 production: paired E_2 - E_1 at 8x8 from the mixture psi_P^2 (f^2 + g^2).
# Size (from pilot P2-0 on ws1, results/deliverable_C/README.md): see PLAN.md; one job = 512 chains x 2 samples.
# Submit 8 jobs with seeds 1..8 (array) -> 8192 samples.
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --array=1-8
#SBATCH --job-name=p2prod
#SBATCH --output=/pc2/users/h/hpcalot/chatty/j1j2/results/delC-p2prod-%A_%a.out
source /pc2/users/h/hpcalot/chatty/j1j2/src/delC/dc_common.sh
I=$BASE/inputs/delC
cat > spec.json <<J
{"phi": 0.056006, "alpha": -0.160951, "Eref": -31.92677568, "b1": B1_FROM_PILOT, "b2": B2_FROM_PILOT,
 "init": ["$I/rvD1_samples.npz", "$I/rvD2_samples.npz", "$I/rvD3_samples.npz"],
 "nchains": 512, "burn": 4, "nsamp": 2, "thin": 2, "seed": ${SLURM_ARRAY_TASK_ID}, "batch": 4096}
J
python dc_p2_prod.py "$OUT" spec.json
echo "finished=$(date -Is)" >> "$OUT/run_metadata.txt"
