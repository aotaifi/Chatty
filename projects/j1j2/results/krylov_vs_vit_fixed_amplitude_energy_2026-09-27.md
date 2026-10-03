# Krylov-sign vs ViT fixed-amplitude energy — 2026-09-27

Question: does the label-free one-step Krylov sign rule produce a lower physical variational energy than the ViT's own phase/sign structure when both use exactly the same ViT amplitudes?

## 6x6 preliminary paired test
- Threshold chosen independently and without sign labels by robust Otsu on an alpha=0.8 training sample: t = -14.75594475197843.
- Physical energy sample: 256 direct |a_ViT|^2 configurations (ESS=256).
- Marshall and ViT local energies use all Hamiltonian neighbors exactly.
- Krylov-minus-ViT difference uses 12 uniformly sampled connected edges per physical configuration; the required neighbor Krylov signs are evaluated by an additional exact r_M shell.

Results:
- Marshall: E = -17.85108 ± 0.07581, E/N = -0.4958633
- ViT binary sign: E = -18.15794 ± 0.01783, E/N = -0.5043873
- actual complex ViT phase: E = -18.15794 ± 0.01783, E/N = -0.5043873
- Krylov sign estimate: E ≈ -18.10392 ± 0.04135, E/N ≈ -0.5028868
- paired Krylov - actual-ViT: +0.05402 ± 0.03657 total energy = +0.0015005/site (~1.48 sigma)

The first 16-chain sample was found to be autocorrelation-dominated: one sticky configuration occurred twice on the same chain on consecutive collection rounds and carried a very large local K-ViT penalty. A cleaner follow-up used 64 independent chains, four widely separated rounds per chain (256 direct |a|^2 samples total), and 16 uniformly sampled Hamiltonian edges per central configuration. Statistical errors are estimated from the 64 independent chain means.

## 6x6 cleaner independent-chain verdict
- Marshall: E = -17.9335223 +/- 0.0577865, E/N = -0.49815340
- actual ViT: E = -18.1242128 +/- 0.0059251, E/N = -0.50345036
- Krylov sign: E = -18.1200134 +/- 0.0134083, E/N = -0.50333370
- paired Krylov - actual ViT: +0.0041995 +/- 0.0103486 total energy = +0.00011665 +/- 0.00028746 per site (0.41 sigma)
- paired Krylov - Marshall: -0.1724717 +/- 0.0448928 total energy = -0.00479088 +/- 0.00124702 per site (3.84 sigma)

Verdict: the label-free one-step Krylov signs do NOT currently beat the ViT signs at fixed ViT amplitudes. The point estimate is slightly higher, but only 0.41 sigma; statistically the two energies are indistinguishable at this precision. Crucially, the Krylov rule lowers the energy substantially relative to Marshall and reaches essentially the ViT fixed-amplitude energy without using ViT sign labels. This is stronger evidence than sign overlap alone that the physically important sign structure has been reconstructed.

8x8 related fact: an independent direct |a|^2 validation sample of 2048 states gives 100% central-state sign agreement between all tested label-free Krylov thresholds and the ViT signs, including the sampled non-Marshall configurations. This does not by itself prove equal energy because neighboring configurations entering E_loc must also be classified.
