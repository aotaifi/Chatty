# 8x8 FN-walker MLE + SR closed-loop result — iteration 1

## Verdict
The scalable amplitude/sign loop passes its first full 8x8 physical gate.

Starting from the established 8x8 ViT/K1 guide, use fixed-sign FN mixed walkers as the
maximum-likelihood target, precondition the MLE gradient with matrix-free SR, update the ViT
amplitude, recompute the K1 threshold/signs with the corrected alpha=1.2 threshold measure,
then rerun fixed-node GFMC.

## SR amplitude update
- SR diagonal shift: lambda = 1
- accepted line-search step: eta = 0.01
- Fisher sample count: 512
- CG relative residual: 8.26e-6
- held-out FN-replica log-likelihood gain: +0.015575
- guide-pool importance ESS: 3235.7 / 4096
- corrected alpha=1.2 threshold ESS: 2140.5 / 4096
- delta log-amplitude RMS on guide pool: 0.21594

The raw Euclidean MLE gradients from the two independent baseline FN replicas were strongly
anti-aligned (cosine -0.8529), while SR rotated the rep1 direction into a transferable
direction: g2^T S^{-1} g1 = +3.85 at lambda=1. This is why Adam/plain gradient failed while
SR succeeded.

## Corrected K1 update
Historical baseline threshold:
T0 = -28.37107876288694

SR-updated threshold, using the original states sampled from a0^1.2 and reweighting only by
(a1/a0)^1.2:
T1 = -28.824166903057147

Node change on the threshold sample:
- raw: 184 / 4096 = 4.4922%
- correctly reweighted alpha=1.2 mass: 9.2730%

Thus the successful energy change is accompanied by a nontrivial sign-structure update.

## Matched FN energy
Baseline 8x8 K1-FN:
E0 = -31.8854241096 +/- 0.00159161
(two-replica spread proxy)

SR-loop iteration 1, M=128:
- seed 13001: E = -31.8967878286
- seed 13002: E = -31.9056697050

Combined:
E1 = -31.9012287668 +/- 0.00444094

Difference:
Delta E = E1 - E0 = -0.0158046572

Using the two-replica spread proxies in quadrature gives a diagnostic separation of about
3.35 sigma. With only two replicas this should not be treated as a precision statistical
claim; the robust conclusion is that the new loop does not degrade the FN solution and gives
a reproducible lower mean in two independent runs.

## Project-level conclusion
We now have an explicit scalable 8x8 realization of

sign -> fixed-node walkers -> MLE amplitude force -> SR/natural-gradient update
     -> corrected K1 sign refresh -> lower fixed-node energy.

This is the first production result that validates the intended FN-amplitude/sign feedback
loop rather than only one of its substeps.

Iteration 2 is being used as a convergence/fixed-point test, not as a prerequisite for the
iteration-1 viability verdict.
