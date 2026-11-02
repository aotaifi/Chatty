# vit10_jax: own JAX version of the 10x10 J1-J2 (J2/J1=0.5) ViT NQS of Rende et al. (HF nqs-models/j1j2_square_10x10)

Files: `vit10_jax.py` (model; `logpsi_plain`, `logpsi_symm`, sigma = +-1, shape (B,100), complex log psi), `validate_vit10.py`
(energy / symm / bench), `vit10_a40.sbatch`, `run_17657387_a40.out` (raw log). Weights: safetensors from /scratch/a/A.Otaifi/ext/hf_vit10
(plain = main, symm = symm_trxy_ising, sha 1e3275...). Reference .py files were read as text, never imported.
Choices: flax LayerNorm eps 1e-6, nn.gelu approximate=True (flax default, the reference calls nn.gelu unchanged). The energy cannot
tell exact from tanh GELU (see below), so this choice rests on the reference code, not on the energy.
Energy: H = sum J S.S, spin-1/2 units (Pauli sigma.sigma divided by 4): diag J s_i s_j/4, off-diagonal J/2 per antiparallel bond times psi(s')/psi(s); 200 NN + 200 NNN bonds, PBC.

Results (job 17657387, one full A40, f64, 30 min; Paderborn H100 est. start was 8 days away):
- symm, first 1024 of the shipped 16384 spins: E/N = -0.497628 +- 0.000030 (target -0.497676: 1.6 SE). Im part of mean 2e-7.
- broken: roll2d transposed, 128 samples: -0.3989 +- 0.149 (clearly wrong). GELU exact instead of tanh, 128 samples: -0.497652 +- 0.000039 (NOT distinguishable).
- plain net (main weights) on the symm-branch spins: -0.497774 +- 0.000154 (n=1024); its own card value is -0.497505.
- symmetry (CPU, 32 spins): translation 1.4e-14, rot90 3.3e-15, Re under spin flip 1.8e-15; Im is invariant only mod 2 pi (log branch).
- throughput A40 f64 (evals/s): plain 12370/13229/12653 at B=256/1024/4096; symm 200/201/201. Mean antiparallel (J1+J2) bonds = distinct one-swap neighbours: 225.3 +- 5.1 (512 spins).
