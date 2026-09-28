# Results log

No subproject-specific calculations yet. Append decisive tests here with date, model, system size, baseline sign convention, threshold rule, observables, and verdict.

## 2026-09-28 — Exact square J1-J2 mechanism sweep

**Model/system:** periodic 4x4 spin-1/2 square-lattice J1-J2 Heisenberg model, S^z=0, exact diagonalization.

**One-step family:** (s_1(x)=s_0(x),mathrm{sign}[t-r_0(x)]), with (r_0=(H a s_0)/(a s_0)).

**Threshold rule:** primary results choose (t) without sign labels by minimizing the exact fixed-amplitude variational energy over all r0 thresholds. Exact ground-state signs are used only to score O_S. An oracle-overlap threshold is stored only as a capacity diagnostic.

**Marshall sweep:**
- J2/J1=0.0: O_S 1.000000 -> 1.000000; energy error/site ~0 -> ~0; P_stable=1.000000.
- 0.2: O_S 1.000000 -> 1.000000; energy error/site ~0 -> ~0; P_stable=1.000000.
- 0.4: O_S 0.999128 -> 0.999983; error/site 6.05e-4 -> 1.18e-5; P_stable=0.999992.
- 0.5: O_S 0.974538 -> 0.999319; error/site 1.156e-2 -> 5.51e-4; P_stable=0.999658.
- 0.6: O_S 0.795572 -> 0.957131; error/site 7.570e-2 -> 3.714e-2; P_stable=0.976555.
- 0.8: O_S 0.123664 -> 0.061971; oracle coordinate capacity only O_S=0.442614; error/site 5.897e-1 -> 4.243e-1; P_stable=0.584007.
- 1.0: O_S 0.064215 -> 0.171346; oracle coordinate capacity only O_S=0.284533; error/site 8.700e-1 -> 6.135e-1; P_stable=0.494345.

**Verdict:** a fixed Marshall gauge does not support a universal one-dimensional Krylov correction. At large J2 the r0 coordinate itself loses the target sign information; this is not just threshold-selection failure.

## 2026-09-28 — J2-adapted baseline comparison

**Baseline:** stripe bipartition, appropriate to the dominant J2 antiferromagnetic structure. x- and y-stripe gauges are symmetry-equivalent on the 4x4 torus.

- J2/J1=0.8: O_S 0.899845 -> 0.975927; fixed-amplitude error/site 5.009e-2 -> 2.401e-2; P_stable=0.986240.
- J2/J1=1.0: O_S 0.955819 -> 0.994884; fixed-amplitude error/site 2.649e-2 -> 5.776e-3; P_stable=0.997445.
- In both cases the label-free energy-optimal threshold reaches the same O_S^(1) as the hidden-sign oracle threshold to numerical precision.

**Verdict:** the one-step miracle is strongly baseline/gauge dependent. It reappears when the baseline is placed in the correct broad sign basin.

## 2026-09-28 — Triangular-lattice falsification

**Model/system:** nearest-neighbor spin-1/2 triangular-lattice Heisenberg antiferromagnet, periodic 6x3 triangular Bravais torus, 18 spins, S^z=0, exact diagonalization.

**Order of operations:** run only after completing the square J2 sweep and J2-adapted baseline comparison.

**Baselines tested:** three simple two-color gauges (maxcut_x, parity_y, parity_xy).

**Results:**
- Best hidden-sign oracle threshold on the same scalar r0 coordinate reaches only O_S^(1)=0.0604 across the tested gauges.
- Best fixed-amplitude energy among the tested baselines improves only from ~0.2222/site error to ~0.1972/site.
- The corresponding P_stable^(1) is nevertheless ~0.947.

**Verdict:** decisive controlled failure of the one-step scalar sign coordinate under intrinsic non-bipartite frustration. High local-field stability can coexist with a globally wrong sign sector, exposing metastable/glassy sign-Ising basins.

## Mechanism conclusion

**ACCEPT:** good-gauge / large-basin mechanism. A one-step Krylov coordinate works when (i) the baseline gauge is already in the correct broad sign basin and (ii) the remaining physically important sign defects are approximately monotone in r0.

**REJECT:** generic-frustration simplification; universal Marshall rule; using P_stable^(1) alone as evidence that the correct sign sector has been found.

Full tables and interpretation: `MECHANISM_VERDICT_2026-09-28.md`.

## 2026-09-28 — Iterated fixed-amplitude Krylov contraction

Controlled exact 4x4 test with exact ground-state amplitudes held fixed. At each iteration:
r_k = H(a s_k)/(a s_k), followed by s_(k+1) = s_k sign(t_k-r_k).
The threshold is selected without sign labels by exact fixed-amplitude energy minimization over realizable r_k threshold groups.

Results:
- J2/J1=0.5, Marshall: O_S = 0.974538 -> 0.999319 -> 0.99999823 -> 1.000000. Exact signs after 3 updates.
- J2/J1=0.6, Marshall: O_S = 0.795572 -> 0.957131 -> 0.991267 -> 0.999444 -> 1.000000. Exact after 4.
- J2/J1=1.0, stripe: O_S = 0.955819 -> 0.994884 -> 0.9996918 -> 0.99999944 -> 1.000000. Exact after 4.

In every nontrivial step, the label-free energy-selected threshold reaches the same next-step O_S as the hidden-sign oracle threshold on the same r_k coordinate.

Verdict: PASS. On the tested exact square clusters, a good starting sign basin is not merely sufficient for one excellent Krylov correction; the iterated map is contractive all the way to the exact ground-state sign structure. This does not yet establish a scalable algorithm because the amplitudes and fixed-amplitude energy objective were exact.

## 2026-09-28 — Basin dynamics correction and closure

The basin-radius and wrong-gauge follow-ups materially revise the earlier “good gauge required” interpretation.

**Square wrong-gauge recovery**
- J2/J1=0.8, Marshall: O_S 0.123664 -> 0.061971 -> 0.098225 -> 0.219270 -> 0.469895 -> 0.507588 -> 0.848923 -> 0.962882 -> 0.991197 -> 0.999846 -> 1. Exact after 10 updates.
- J2/J1=1.0, Marshall: O_S 0.064215 -> 0.171346 -> 0.290203 -> 0.339260 -> 0.910391 -> 0.990778 -> 0.998562 -> 0.999975 -> 0.99999941 -> 1. Exact after 9.

Thus good gauge is not required for eventual convergence on the tested 4x4 square cluster; it mainly makes the first update efficient.

**Random-corruption profile**
With exact amplitudes and exact signs randomly corrupted by physical probability weight q, all 32/32 sampled square starts converged through q=0.42. This corresponds to initial sign overlap only about 0.16. The transition becomes pattern/model dependent for q>=0.44, so this is an empirical random-corruption profile, not a rigorous worst-case basin radius.

**Triangular repeated dynamics**
On the 6x3 triangular Heisenberg model, maxcut_x/parity_y/parity_xy flow to wrong fixed points after 13/18/18 updates with O_S essentially zero. Yet random corruptions around the exact signs converge in all 8/8 trials through q=0.45 and in 4/8 at q=0.49.

Interpretation: the exact signs are a strong attractor even on triangular, but simple structured gauges occupy competing basins. Basin membership is not determined by overlap alone.

**Degeneracy correction**
The older triangular one-step script could split exactly/near-degenerate r values by sweeping configurations individually. That can produce states not realizable by any single threshold. The grouped-threshold implementation is authoritative; the previous triangular oracle <=0.0604 is a legacy artifact/upper bound.

Current mechanism: projected-Krylov sign descent with monotone fixed-amplitude energy and multiple attraction basins. See `BASIN_DYNAMICS_VERDICT_2026-09-28.md`.
