# J1-J2 Krylov/FN project decision — 2026-09-27

## Current state
The label-free one-step Krylov sign rule is
s_t(x)=s_M(x) sign[t-r_M(x)], with r_M=(H a s_M)/(a s_M).
No hidden signs are used to construct the rule or choose the threshold.

Validated results:
- 6x6 alpha=1.2 stress: weighted hidden-sign overlap ~0.99776-0.99840 depending unsupervised threshold; robust two-means 10-90 gives 0.9977645.
- 8x8 alpha=1.2 stress: weighted overlap ~0.9975-0.99784; robust two-means 10-90 gives 0.9977270.
- 8x8 physical alpha=2 evaluation with the alpha=1.2 threshold frozen: O=0.9995117 on 4096 samples; sampled wrong-sign mass=0.0002441; Marshall baseline O=0.9599609.
- Earlier 8x8 alpha=0.8 O~0.99923 was statistically optimistic because ESS was only ~22.
- Exact 4x4 closed lattice FN -> amplitude -> Krylov loop: Marshall O=0.974538; first update O=0.998122; after 12 iterations O~0.997910. Exact E0=-8.45792335; loop guide energy=-8.45249689, residual ~3.39e-4/site. Therefore the rule is highly accurate but not exact.

## Project decision
Do NOT spend the next effort on more standalone sign-overlap scaling first.
The next decisive experiment is to close the actual FN -> amplitude -> Krylov-sign loop at 6x6.

Reason:
1. Standalone sign reconstruction has already survived 6x6 -> 8x8.
2. The scientific claim requires the signs to improve the sign-free fixed-node amplitude solve, not merely agree with a ViT diagnostic.
3. A 6x6 loop can falsify the route cheaply: if Krylov-guided FN fails to lower energy materially relative to Marshall FN, or the next Krylov update degrades/oscillates, stop.
4. If 6x6 closes stably, repeat at 8x8 and only then measure scaling of walker population, projection length, variance/autocorrelation, and residual energy bias.

## Immediate 6x6 experiment
Run, with matched FN/QMC controls:
Marshall signs -> FN amplitudes a0 -> label-free one-step Krylov signs s1 -> FN amplitudes a1 -> s2 -> ...
Track each iteration:
- FN energy per site and uncertainty;
- variational energy of a_k s_k when available;
- change in signs under physical FN weight;
- local-energy variance;
- walker population / population-control bias;
- projection length / autocorrelation;
- overlap with hidden ViT signs only as an offline diagnostic, never for construction.

Baselines:
A) Marshall FN.
B) Krylov signs applied once to the same starting amplitudes.
C) Closed Krylov/FN iterations.
D) Hidden-ViT signs/amplitudes only as a diagnostic upper/reference benchmark, not as input.

Stop criteria:
FAIL if 6x6 loop does not materially improve FN energy over Marshall, becomes unstable/cyclic with non-negligible weight, or requires rapidly growing walker/projection resources.
CONTINUE if energy improves monotonically or settles to a stable substantially lower bias with controlled cost; then run the identical 8x8 loop and scaling study.

Interpretation:
The promising hypothesis is not that one Krylov step gives the exact node. It is that sign reconstruction may be polynomially cheap while sign-free FN supplies the amplitudes, producing a scalable approximate self-consistent loop.
