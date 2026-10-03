# Full-network walker-MLE amplitude route: exact 4x4 gate — 2026-09-30

## Identity
For a guide amplitude a and fixed-node ground-state amplitude phi_FN[a,s], the
importance-sampled FN walkers have stationary mixed distribution

f(x) proportional to a(x) phi_FN(x).

If an amplitude model is maximum-likelihood trained so that its sampling distribution
a_new(x)^2 matches f, the ideal optimum is

a_new proportional to sqrt(a phi_FN).

Because the lattice fixed-node sign-flip potential depends on the current guide ratios,
phi_FN is itself a function of the current guide. The real damped map is therefore

a_k -> sqrt(a_k phi_FN[a_k,s_k]),

interleaved with the sign update; it is not convergence toward one immutable phi_FN.

## Exact ideal half-step map
Exact 4x4 J1-J2 at J2/J1=0.5, Sz=0, Hilbert dimension 12,870.
Exact ground energy: -8.45792335139.

Starting from exact ground-state amplitudes with Marshall signs:
- initial guide energy: -8.27290914;
- first FN energy: -8.34152412;
- first ideal half-step + K1 energy: -8.44578657;
- first half-step sign overlap with exact signs: 0.99941675.
- weighted log-distance to the current phi_FN is halved exactly by the geometric-mean update.

The iteration remains near the correct basin and reaches about -8.4553 by 12 rounds.

Hard start: uniform amplitude + Marshall signs:
- initial energy: -5.06666667;
- first half-step is non-monotone and worsens to -4.40235;
- then the loop recovers;
- round 11 energy: -8.42491442;
- round 29 energy: -8.44962196;
- round 29 sign overlap: 0.997918.

Thus the ideal MLE half-step map is viable and contracting toward each current FN target,
but global energy need not improve monotonically from a very poor guide.

## Finite-sample network diagnostic
A fresh compact full-amplitude MLP was first trained on the exact guide distribution a_g^2,
then fine-tuned by maximum likelihood on 10,000 exact mixed walkers drawn from
f proportional to a_g phi_FN.

This intentionally tests whether a newly trained small network can realize the subtle
half-step correction.

Measured:
- walker samples: 10,000; unique states: 2,085;
- guide-pretraining amplitude fidelity: 0.967764;
- learned amplitude fidelity to exact ideal half-step: 0.949841;
- learned amplitude fidelity to phi_FN: 0.949223;
- ideal half-step K1 energy: -8.44578657;
- learned-network K1 energy: -7.98753401;
- learned vs ideal-half K1 disagreement under exact physical weight: 0.003771.

Verdict: the small fresh network fails because it cannot even represent the starting guide
accurately enough; this is not a failure of the walker-MLE identity. Production should
fine-tune the existing expressive guide network (or an equivalently capable model) from its
current parameters. Do not replace a high-quality guide with a newly initialized small model.

## Consequence for 8x8
The current discriminative CNN density-ratio update failed at eta=0.3597 because its local
neighbor ratios produce a large nodal move. The full-network MLE route is now the preferred
replacement if the strongly damped classifier direction does not pass its eta=0.20 pilot.
