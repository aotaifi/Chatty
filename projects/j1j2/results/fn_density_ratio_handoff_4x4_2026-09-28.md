# 4x4 FN amplitude-handoff falsification — 2026-09-28

## Goal
Test whether FN mixed-distribution walkers can be compressed into a callable amplitude correction cheaply enough to feed the one-step Krylov sign rule.

## Construction
Reference guide amplitude a_k gives q(x) proportional to a_k(x)^2.
Exact FN mixed distribution gives f(x) proportional to a_k(x) a_FN(x).
Therefore f/q is proportional to a_FN/a_k.

Represent the correction only through a scalar coordinate r(x), using four equal-q-mass bins:
g(x) = log[f_bin/q_bin],
a_hat(x) proportional to a_k(x) exp(eta g(x)).
The best stable coordinate is the current-guide local energy; eta=0.5 under-relaxation suppresses finite-sample noise.

## Single-step result
Exact first FN from Marshall:
E_FN = -8.34152411548.
Exact FN-amplitude Krylov sign overlap: O = 0.9981217230.

A four-bin density-ratio model already reproduces the physically important sign update:
- n=3000 samples from q and 3000 from f: median downstream O~0.999263; worst of 20 one-step repeats ~0.99810.
- n=10000+10000: every 4-bin repeat in the one-step sweep reached O=0.999263.
This is 2-3 orders of magnitude fewer samples than the raw walker histogram route (~1e6-3e6 ideal walkers).

## Closed-loop result
Tested FN -> sampled 4-bin amplitude correction -> Krylov signs -> FN repeatedly for 12 iterations.

With eta=0.5 and 10000 q + 10000 f samples per iteration:
- all four random-seed runs kept O=0.9992629413 through essentially the full loop; no catastrophic sign drift.
- final guide energies across seeds:
  -8.45038883,
  -8.44845081,
  -8.45106727,
  -8.44838239.
- corresponding FN energies near the end remain around -8.4523 to -8.4525.
- amplitude fidelity to the exact FN amplitude is typically 0.9994-0.9999 after the first few iterations.
- sampled sign disagreement with the exact FN->Krylov update is usually zero at physical weight to numerical precision in the stable 10k runs.

At 3000+3000 samples the loop is often good but occasional noisy updates can kick the variational energy upward; under-relaxation substantially improves stability. Thus the remaining issue is sampling noise, not representational capacity of this 4-bin handoff on 4x4.

## Verdict
PASS for the controlled 4x4 handoff.
A raw histogram is not needed. The FN amplitude correction is highly compressible in this test: four scalar bins plus one damping parameter are enough to preserve the Krylov sign update over many iterations with O(10^4) samples, rather than O(10^6).

This does NOT yet prove scalability. The next decisive experiment is the real 6x6 fixed-node walker loop using the same callable 4-bin density-ratio handoff. Measure:
- FN energy/site and uncertainty,
- stability across iterations/seeds,
- walker/sample count needed for fixed amplitude/sign accuracy,
- local-energy variance and projection/autocorrelation cost,
- whether bin count/model complexity must grow with L.

Stop if sample count or model complexity grows sharply, or if the loop becomes unstable. Continue to 8x8 only if 6x6 closes cleanly.
