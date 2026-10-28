# Review: loop C fixed point E* = T/(1-r) (PLAN.md lines 98-110), 2026-10-28

Worker review, read and compute only (no jobs). Claim reviewed: PLAN.md "LOOP C PREDICTION (lead, takeover 2026-10-28)".
All energies are E_FN - E0 per site, in units of 1e-6. Every number below is exact-solve output (deterministic; one fitting
seed per chain unless stated), not Monte Carlo, so quotable.py (MC chain gate) does not apply; fits on one chain are labelled.
Script: /scratch/a/A.Otaifi/tmp/claude-29196/-project-theorie-a-A-Otaifi/9d020707-2c67-4609-bec6-53f62eb11c06/scratchpad/fit.py

## Conclusion
"No accuracy edge at 6x6" holds only on the s2 (fixed oracle sign) chain, and there it holds because of the s2 floor, not
because of r. The comparison with the 1.95 competitor mixes chains. On the competitor's own-sign chain the existing data
put loop round 1 at 1.29-1.40 (0.66-0.72 x competitor) if the complex copies' T transfers, the break-even being a 1.31-1.51x
larger T on that target (real copies measured 0.87-0.90x). The r used (0.33) has an inconsistent denominator; the
consistent single-target r is 0.23, and the same recipe back-tested on the O2b one-hop loop over-predicts E* by 43%.

## Raw data verified
- O2b: /project/theorie/a/A.Otaifi/Chatty/runs/compress6/runs/o2b_17431886/cx_o2b.json ('rounds'). E_k = EFN (one-hop guide of
  a_k), h_k = EFN_hop_phi = E_FN(K1(phi_k)), T_k = fit_tax; E_{k+1} = h_k + T_k exactly (max deviation 9e-22).
  E = 23.692, 14.119, 10.881, 9.262, 9.247, 8.423; h = 8.106, 4.951, 3.576, 2.897, 2.539; T = 6.013, 5.930, 5.686, 6.349,
  5.884 (mean 5.972, sd 0.242). README table reproduced.
- G-A3: projects/j1j2/experiments/compress_6x6/gate_results/cx_ga3.json 'chain': 1.9474, 0.3182, 0.0917, 0.0331, 0.0138,
  0.0062, 0.0030; round ratios 0.163, 0.288, 0.361, 0.416, 0.453, 0.486.
- G-tax: projects/j1j2/runs/gate_tax/out_full/cx_gtax.json 'tax': E_FN(K2(a_1)) 1.9749, E_FN(K2(phi_0)) 0.3182, tax_2 1.6568,
  no-loop E_FN(K2(A)) 1.9474. a_1 sign = table s_g0 (only amplitude fitted; cx_gtax.py header lines 5-13).
- T2: /scratch/a/A.Otaifi/ansatz_runs/pfaffian_c/runs/t2_cplx/s{1,2,3}/cx_h1.json (cases.F1B = complex copy, cases.phi1 =
  target): tax_1 3.754/3.231/2.845, tax_2 1.244/1.070/0.925 (mean 1.080, sd 0.160), ratio 0.331/0.331/0.325; controls
  o2b_a2 tax_2 1.903 and F1A 1.846 reproduced. E_FN(K2(phi_1)) = 3.2442, copies' E_FN(K2) 4.488/4.315/4.169.
  Note: t2_cplx/ANALYSIS.md says ANALYSIS BLOCKED (gate not run by the watcher); the numbers above are read directly.
- Zero hop: /scratch/a/A.Otaifi/ansatz_runs/pfaffian_d/split/split_{s1,s2,s3,o2b_round_1}.json: target phi_1 9.853; complex
  copies 20.92/18.97/17.71 (tax_0 11.07/9.11/7.86); O2b real copy 29.84 (tax_0 19.99).
- s2 floor: projects/j1j2/results/compress_6x6/runs/cx_s2ceil.json arms.s2a_flip_all.history: 40 solves, last 0.9149, still
  falling: last step ratio 0.926, Aitken 0.475. So 0.915 is an upper bound on the s2 floor, not a converged floor.

## (1) Linear map with constant T, O2b one-hop data (one chain)
- T constant: yes, to 4% (sd 0.24 on mean 5.97, no trend over E 23.7 -> 9.2). Supported.
- Free fit E_{k+1} = r E_k + T (5 pairs): r = 0.370 +- 0.031, T = 5.41 +- 0.45 (corr -0.93), E* = 8.59; residuals
  -0.06, +0.24, -0.18, +0.41, -0.41 (rms 0.29). The fitted T is 0.56 below the measured mean T. Rounds 1-4 only: r 0.421,
  T 4.87, E* 8.42 (round 5 is already 8.42).
- The h-part is NOT a function of E: h = rE through the origin gives r = 0.334 with systematic residuals +0.19, +0.23,
  -0.06, -0.20, -0.55; r_k = h_k/E_k falls 0.342, 0.351, 0.329, 0.313, 0.275. Rounds 3 and 4 have the same E (9.262 vs
  9.247) but h drops 12% (2.897 -> 2.539). The target keeps improving while the energy stalls: the loop carries memory,
  and E* = h_inf + T with h_inf set by the slowed exact chain (h ratios 0.61, 0.72, 0.81, 0.88), not by r E*.
- Loop states contract faster than exact-chain states at equal E (O2b round 3: r 0.31 at E 9.3; O2a exact chain at E 8.1:
  0.55), because the tax excess in E_k is mostly hop-removable. A single r taken from an exact target does not describe
  loop states.
Verdict: "constant T" supported; "r constant" refuted at the 20% level within 5 rounds; the linear map fits E to rms 0.29
(3% of E) by accident of T >> h here. It is a one-seed chain.

## (2) Single-target r, and the floor
- Denominator inconsistency. The recursion propagates E_k = E_FN of the guide that made phi_k (PLAN uses E(phi_2) = 4.32 =
  copies' E_FN(K2)). For phi_1 that quantity is 14.119, not the zero-hop self-FN 9.853. Consistent r = 3.244/14.119 = 0.230
  (H1's own number), E* = 1.40. The plan's r = 0.329 gives 1.61.
- Back-test of the plan's recipe on the one-hop loop (the only loop data): r = E_FN(K1(phi_1))/E_FN(phi_1) = 4.951/9.853 =
  0.503 -> E* = 12.0, sequence 13.07, 12.54, 12.27, 12.14; observed 10.88, 9.26, 9.25, 8.42. Over by 43%. Consistent r 0.351
  -> 10.92, 9.80, 9.41, 9.27, E* 9.20 (9% over, and the observed sequence is still falling).
- G-A3's r range (0.16-0.49) is from the own-sign chain, which has no s2 floor; it does not transfer to the s2 chain.
  On the s2 exact one-hop chain (O2a) r rises 0.34 -> 0.79 as E falls to 1.7.
- Floor. With sign s2 fixed, every h = E_FN(K2(phi)) >= F_s2, so every loop round has E_k >= F_s2 + T, whatever r is:
  F = 0.915 (lowest s2 value measured) -> E_k >= 1.995 > competitor 1.947; F = 0.475 (Aitken, unconverged) -> >= 1.555.
  Floor model h - F = r_F (E - F) with phi_1: r_F 0.176 (F 0.915) or 0.203 (F 0.475) -> E* = F + T/(1-r_F) = 2.23 or 1.83.
  With the largest seed T (1.244): 2.43. Material: the floor moves E* from 1.40-1.61 to 1.83-2.23, i.e. from 0.72-0.83 x to
  0.94-1.14 x the competitor. The "1.5" end of the plan's range is excluded unless the s2 floor is below 0.42.

## (3) Fairness of the competitor comparison
Not fair. The competitor 1.947 is E_FN(K2(A)) with the RBM+PP's own (signed-polynomial) sign; the loop runs on the O2b chain
with the oracle sign s2 and its floor (<= 0.915, never seen below). Two fair comparisons exist in the data:
- Same own-sign chain, real copy, measured (G-tax): loop round 1 1.975 vs no-loop 1.947 (1.014 x): no edge for real copies.
- Same own-sign chain, complex copy, derived: loop round 1 = E_FN(K2(phi_0)) + T = 0.318 + T. With T = 1.080 (mean on phi_1)
  -> 1.398 (0.72 x); with T scaled by the real copies' target transfer 0.87-0.90 (1.657/1.903, 1.657/1.846; different
  architectures, so loose) -> T 0.94-0.97, 1.26-1.29 (0.65-0.66 x); worst seed 1.244 -> 1.562 (0.80 x). Further rounds go
  lower (exact own-sign chain 0.318 -> 0.092 -> 0.033), so E* in about 1.0-1.5 (0.51-0.78 x). Break-even for round 1:
  T = 1.629, a factor 1.51 (mean) or 1.31 (worst seed) above the measured complex T.
- An s2-chain no-loop competitor (VMC network + two s2 hops + FN) is not in the data (only VMC PP s2: zero hop 53.7, one hop
  23.7, cx_o2b.json control); so the s2-vs-s2 comparison cannot be made from existing files.

## (4) The assumption the conclusion rests on most
That the loop is run on the s2 chain (sign frozen at the oracle s2) while the competitor is on its own-sign chain; i.e. that
the r/T/floor of the s2 chain decide the comparison. On the s2 chain the conclusion is forced by the floor (E_k >= F + T
= 1.995 for F = 0.915), not by r. Existing data refute "no edge" for the like-for-like own-sign chain in the derived sense:
G-tax gives the loop map there (0.318 + T), T2 gives T = 0.93-1.24 for complex copies, and the result is 0.66-0.80 x the
competitor at round 1. What is not measured, and decides it: the complex copy's tax_2 on the own-sign target phi_0 =
FN(K2(A)) (G-tax target). If it is <= 1.63 the loop beats the competitor at round 1; <= 0.66 is needed for a 2x edge at
round 1 (0.318 + T <= 0.974).

## Not checked
No new computation of E_FN; the own-sign complex tax and any s2-chain no-loop competitor are not in the data. O2b is one fitting
seed with unconverged fits; complex T is 3 seeds on one target. Whether storing the own-sign chain's evolving sign as a table
is equivalent to s2's table (both are tables at 6x6, neither is a network) was not examined beyond cx_gtax.py's header.
