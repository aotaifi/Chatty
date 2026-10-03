# Why one Krylov step works — 6x6 mechanism verdict
**Date:** 2026-09-30

## Core result
For the 6x6 square-lattice J1-J2 model at J2/J1=1/2, the K1 sign rule is best understood as a threshold on the amplitude-weighted **off-diagonal Marshall local field**.

Write
[
r_M(x)=D(x)+F_{J1}(x)+F_{J2}(x),
]
where the Marshall gauge makes the NN J1 hopping contribution negative and the frustrating NNN J2 hopping contribution positive.

The useful sign information is almost entirely in
[
F(x)=F_{J1}(x)+F_{J2}(x).
]

On an independent cached 6x6 train/validation set:
- diagonal term alone: essentially no sign improvement;
- J1 off-diagonal alone: weak;
- J2 off-diagonal alone: weak/moderate;
- ratio F_J2/|F_J1|: strong (~0.994 overlap) but incomplete;
- physical off-diagonal sum: ~0.9984 train, ~0.9986 val;
- full local energy: ~0.9990 train, ~0.9989 val.

Thus K1 is not reading a classical diagonal frustration motif. It detects the competition/cancellation between unfrustrated J1 and frustrating J2 hopping channels, weighted by neighboring amplitudes.
## Physical coupling is special
Sweep
[
F_lambda=F_{J1}+lambda F_{J2}
]
where F_J2 already contains the physical J2=0.5 factor.

The optimum is broad around lambda=1--1.25:
- train: lambda=1 -> 0.99839; lambda=1.25 -> 0.99876;
- validation: lambda=1 -> 0.99862; lambda=1.25 -> 0.99840.

So the actual Hamiltonian weighting is already near-optimal; this is not an arbitrary fitted separator.

## Why the first step looks miraculous
On the alpha=1.2 6x6 benchmark, physical wrong-sign mass is:
- Marshall: 0.01635 train, 0.01811 val;
- after K1: 0.000808 train, 0.000802 val.

K1 therefore removes about 95% of the physically relevant wrong-sign mass in one step, a 20--23x reduction.

## Why naive K2 gets worse
True K2 uses
[
|psi_1(x)|=|T_1-r_0(x)|a_0(x),
]
so the first Krylov operation changes amplitudes as well as signs.
On the physical diagnostic sample, the multiplier has CV ~0.18 and the true-K2 coordinate is strongly correlated with it.
Restoring the original ViT amplitudes removes the K2 pathology for 3 of the 5 directly observed bad K2 flips.
The remaining two restored-amplitude anomalies are already extreme local-energy outliers in the original ViT state (98.8th and 99.6th percentiles), so they are consistent with amplitude/local-energy error rather than missing central sign corrections.

More importantly, r1 still contains a tiny residual sign signal, but only in an ultra-rare tail:
- useful oracle tail mass: ~3e-4 to 6e-4;
- precision: ~78--91%;
- recall of remaining wrong mass: ~36--55%;
- the train-selected extreme-tail threshold also improves validation.

By contrast, the three unsupervised K2 thresholds flip ~1.0--1.8% of physical weight.
They have ~91--96% recall but only ~4--8% precision, so they catch nearly all residual sign defects while introducing far more false flips.

## Mechanism summary
[
oxed{	ext{K1 = bulk off-diagonal frustration-field correction}}
]
followed by
[
oxed{	ext{residual defects become rare; amplitude noise/heavy tails dominate ordinary K2 clustering}.}
]

This reconciles:
1. excellent one-step sign reconstruction;
2. successful projected/FN iterations when amplitudes are restored or re-solved;
3. worsening of the direct two-Krylov continuation.

## Verdict
ACCEPT / mechanism resolved on 6x6.
The remaining question is cross-size/model generalization, which belongs to the separate frustration/generalization parked angles rather than this mechanism thread.
