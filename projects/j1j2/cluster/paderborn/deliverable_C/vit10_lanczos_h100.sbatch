#!/bin/bash
# DRAFT, not submitted (scope decision 2026-10-19: route C compute kept small). 10x10 public ViT, PREREG E4:
# (1) moments run = reproduction gate (binarised and complex energy vs the card) + alpha; (2) direct VMC of psi_1.
# Code: experiments/deliverable_C/dc_lanczos10.py + dc_vit10.py + dc_fn.py + rv_da.py + vit_dt.py (UNTESTED on real
# weights; first run a 10-minute smoke with nchains 16). Checkpoint: download into its own new directory
# $BASE/inputs/hf_j1j2_10x10/<revision>/ (model.safetensors, symm_indexes.txt; verify sizes/sha256 against the HF API
# listing), load as data only (dc_vit10.load_safetensors), never execute the downloaded python files.
# Cost: ~2.6e4 symmetric evaluations per two-hop sample at ~1.1e4/s on an H100 -> ~2.4 s/sample; 6000 samples ~4 h.
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --job-name=vit10lz
#SBATCH --output=/pc2/users/h/hpcalot/chatty/j1j2/results/delC-vit10lz-%j.out
source /pc2/users/h/hpcalot/chatty/j1j2/src/delC/dc_common.sh
H=$BASE/inputs/hf_j1j2_10x10/symm_trxy_ising
cat > spec.json <<J
{"mode": "moments", "ckpt": "$H/model.safetensors", "symm": "$H/symm_indexes.txt", "flip": true, "batch": 1024,
 "nchains": 1024, "burn": 20, "nsamp": 6, "thin": 1, "seed": 1, "chunk": 16}
J
python dc_lanczos10.py "$OUT" spec.json
# then: mode "direct" with alpha, Eref, phi from $OUT/lanczos10.json, init $OUT/samples.npz, nchains 1024, burn 4, nsamp 4, thin 2
