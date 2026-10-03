# Krylov sign scaling + fixed-node loop — 2026-09-27

## Canonical rule
Use the label-free one-step Krylov correction
s_t(x)=s_M(x) sign[t-r_M(x)],
with r_M=(H a s_M)/(a s_M). Discover t on a flattened amplitude sample; evaluate physical performance separately.

## 8x8 stress test
At alpha=1.2 with 4096 train / 2048 validation samples:
- importance ESS: 461 train / 309 validation;
- robust two-means 10-90 threshold t=-28.3711;
- validation weighted hidden-sign overlap O=0.997727;
- nearby unsupervised thresholds give O about 0.9975-0.99784.
This downgrades the earlier alpha=0.8 O~0.99923 estimate, whose ESS was only ~22.

## 8x8 physical alpha=2 test
Freeze t=-28.3711 learned label-free at alpha=1.2, then score on 4096 alpha=2 physical samples:
- O=0.99951171875;
- wrong-sign sample mass = 0.000244140625 (1/4096 sampled configurations);
- Marshall baseline O=0.9599609375;
- hidden non-Marshall fraction ~0.0200; predicted flip fraction ~0.0198.
Importance-weight ESS is exactly 4096 because alpha=2; this does not by itself remove MCMC autocorrelation.

## 6x6 matched alpha=1.2 stress test
4096 train / 2048 validation:
- ESS: 941 train / 442 validation;
- robust two-means 10-90 threshold t=-15.7922;
- validation weighted overlap O=0.9977645;
- less aggressive two-means thresholds give O~0.99807;
- best unlabeled Otsu 5-95 gives O=0.998397.
Thus flattened-distribution error does not vanish, but remains around the 10^-3 overlap-deficit scale.

## Exact 4x4 fixed-node -> amplitude -> Krylov loop
Exact S^z=0 Hilbert dimension 12870. Exact ground energy E0=-8.4579233514.
With exact amplitudes + Marshall signs:
- Marshall hidden-sign overlap O=0.9745381;
- direct one-step Krylov O=0.99926294, E=-8.44847413.
Closing the exact lattice fixed-node loop using alpha=1.2 two-means sign updates:
- first FN->Krylov step jumps to O=0.9981217;
- after 12 iterations, O~0.99790994;
- guide energy reaches -8.45249689, still above exact by ~0.00542647 total (~3.39e-4/site).
The sign update occasionally flips a tiny true-weight sector (~7.3e-5), but the loop remains stable and strongly improved over Marshall. It does not converge to the exact node.

## Verdict
The one-step Krylov construction is not exact: exact 4x4 closure leaves a small fixed-node/sign bias. No evidence yet of catastrophic scaling from 6x6 to 8x8; the physical 8x8 test remains extremely accurate. Therefore this is a promising approximate scalable sign guide, not an exact sign-problem solution. The remaining decisive scaling question is whether the residual FN energy-per-site bias stays bounded/decreases or grows with L when the full FN projector is run at 6x6/8x8.
