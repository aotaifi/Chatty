# PLAN_signgap: sign gap as a size-converging correction (angle lead, opened 2026-10-29)

Owner: angle lead session (started by the j1j2 project lead). Reports to the project lead when the angle closes or its
result changes PLAN.md. PLAN.md `## Queue` last item holds the origin. Equations: board blocks 16-18
(artifact LwUGJoTxGJku3k6mTVgMVP, version 44).

## Open question (one)
Does delta(N) = (E0(N) - E_B(N))/N converge fast enough that E0 = E_B(QMC) + N delta(fit to ED) predicts the J2 = 0.5
energy at 8x8 / 10x10 to 1e-4 per site? E_B = ground energy of H_B (Marshall frame, every off-diagonal -|.|; J2 bond
becomes J2[SzSz - SxSx - SySy], easy-plane Neel).
- Mechanism (board 17): delta(L) = delta_inf + (c0 - c_B)/L^3. Two-point (4x4, 6x6) estimate c0 - c_B ~ -0.16,
  delta(8x8) ~ 0.3854, delta_inf ~ 0.3858. Lead's confidence ~20%.
- Term that breaks it (board 18): H_B has a Goldstone L^-3 term c_B; E0 at J2 = 0.5 likely non-magnetic, so its
  finite-size form differs and the L^-3 terms need not cancel.
- Exact identity (board 16): delta = (1/N) int_0^1 <V>_lambda dlambda, V = J2 sum_nnn (S+S- + h.c.), path H_B -> H.
- Scope: exact Lanczos (QuSpin or own code) for stage A; sign-free SSE for stage B. Passed over: fitting E0 directly
  (that is the control, not the method); DMRG for E_B (QMC is exact-in-principle and cheaper for a stoquastic H).
- Budget: stage A (ED + fits, kill test a-c) 50 core-hours; stage B (QMC E_B at 8x8, 10x10, plus E_B(40)) 200
  core-hours; no GPU. Output /scratch/a/A.Otaifi/ansatz_runs/signgap/.

## References (checked 2026-10-29)
- N=40 exact (Richter & Schulenburg, arXiv:0909.3723, Table 1, total energies): J2 0.3 E0 = -22.42728643
  (-0.560682161/site); J2 0.5 E0 = -19.96304839 (-0.499076210/site). Note: above 8x8 DMRG and 6x6: non-monotone in N.
- 6x6 exact -0.50380965 (TRIED I1). 4x4 exact -0.5286202095, E_B -0.9118324446 (ed4.json).
- 8x8: DMRG torus -0.4992(1); our FN bound -0.498946(5) (TRIED I2, I3).
- 10x10: best variational -0.4976921(4) (Chen-Heyl arXiv:2302.01941); extrapolations ZVE -0.497715(9) and Hu var-extrap
  -0.49781(2) disagree by ~1e-4 (results/literature/J1J2_BENCHMARKS_COST.md section 1.3).

## Kill test (FROZEN 2026-10-29, after Astra review reviews/REVIEW_astra_signgap_2026-10-29.md; before any fit)
Identity (review, exact): with the same fit operator P, r_a = P[e_a] - e_a(N*), the delta route misses by r0 - r_B, the direct
route by r0. The method works only if the finite-size residuals of E0 and E_B are correlated across clusters.
Clusters (periodic, J2 along (+-1,+-1), same cluster and checked sector for E0 and E_B): N = 16, 18, 20, 26, 32 (fit),
34, 40 (clean holdouts), 36 (holdout reported but contaminated: delta(6x6) informed the proposal).
Forms: F1 = d + a N^-3/2 (primary); F2 = F1 + b N^-2 (the O(n) L^-4 term, Hasenfratz-Niedermayer hep-lat/9212022 Eq. 2.18);
F3 = d + a N^-1. Weights equal; no exclusions after seeing data.
- (0) J2 = 0: delta = 0 on every cluster (identity; worker check).
- (a) J2 = 0.3, comparison regime (NOT a guaranteed positive control: c0 - c_B = -0.719(2 v0 - v_B), no symmetry fixes it):
  F1 misses at 34 and 40 <= 1e-4/site.
- (b) J2 = 0.5: F1 misses at 34 and 40 <= 1e-4/site, AND the spread of F1-F3 (fit on all N <= 40) at N = 64 and 100 <= 1e-4
  (catches the review's counterexample: an omitted 0.5 L^-4 term passes 36/40 and misses 64/100 by 1.5-2.2e-4).
- (c) Control at each J2: max holdout |r0 - r_B| <= max |r0| / 5 (same forms on E0/N directly).
- (d) 8x8: E_B/N (QMC, stat error) + delta(64) with total error (form spread (+) stat) <= 1e-4 and centre in
  [-0.4994, -0.4990] (DMRG -0.4992(1), itself an extrapolation). The FN-bound inequality is dropped: the window lies below it.
- (e) 10x10: refutation only. Refuted if the prediction lies above the best variational energy (CNN-MPS -0.4976939(2),
  arXiv:2603.14425 Table 1 per Astra, not yet read by me; else Chen-Heyl -0.4976921(4)) by more than its error. Agreement
  establishes no accuracy: a prediction is neither a state nor a bound.
Order: E_B(40) is not in the stage-A ED budget; it comes first in stage B (SSE, validated against exact E_B at N <= 36).
Until then (a)-(c) are judged at N = 34 alone, provisionally; the verdict needs 40.
Verdicts: (0) fails -> code bug, fix. (a) fails -> KILL, cause = cluster-shape/sector scatter of delta. (a) passes, (b) fails ->
KILL, cause = board-18 term (non-cancelling finite-size forms). (c) fails -> KILL, cause = residuals uncorrelated (delta no better
than E0). (a)-(c) pass -> stage B. (d) fails -> KILL at scale, cause from the size of the miss vs form spread.
Value if it passes (review point 5): a cheap energy estimator with validated error, not a variational result; the project lead
judges whether that serves the goal.

## VERDICT 2026-10-29: KILL at stage A (no QMC spent)
Exact E0, E_B on N = 16, 18, 20, 26, 32, 34, 36 (A0_ed/energies.json; 6x6 E0 matches published to 4e-9; J2 = 0 gives delta = 0 to 1e-13).
Fits (A1_fits/fits.py, fits.out), misses per site at N = 34 / 36:

| J2 | form | delta miss | E0-direct miss | ratio |
|---|---|---|---|---|
| 0.3 | F1 | 1.47e-3 / 2.87e-3 | 1.74e-3 / 3.21e-3 | 1.12 |
| 0.3 | F2 | -1.93e-3 / -1.63e-3 | -2.11e-3 / -1.87e-3 | 1.09 |
| 0.5 | F1 | 5.47e-3 / 1.10e-2 | 5.68e-3 / 1.12e-2 | 1.02 |
| 0.5 | F2 | -4.97e-3 / -2.78e-3 | -4.92e-3 / -2.78e-3 | 0.99 |

(a), (b), (c) all fail, by 15-110x the 1e-4 bar; the form spread of delta at N = 64 is 1e-2 (J2 0.3) and 3e-2 (J2 0.5).
A failure at N = 34 is decisive without N = 40 (a pass needed every holdout). Mechanism: r_delta = r0 - r_B with r_B ~ 2e-4 while
r0 ~ 1e-2: the partner's energy is smooth in N, the frustrated E0 carries the cluster-to-cluster scatter, so delta inherits it
whole; it fails equally at J2 0.3 (both Neel), so the cause is not the non-magnetic phase named on board 18. Squares alone
(4x4 0.383212, 6x6 0.385308) give no holdout, and two-point L^-3 vs L^-4 forms differ by 1.6e-4 at 8x8. Stage B not run.
Hypothesis, untested: in delta = (1/N) int <V>_lambda dlambda the shape-sensitive part sits at lambda > 1/2, where V's J2
hops turn positive (the frustrated half of the path).

## Dead ends
- Sign gap as a size-converging correction: KILL, see VERDICT (A1_fits/fits.out).

## Cockpit
- STATUS (done, low): The sign gap per site does not converge with cluster size: it inherits the full cluster-to-cluster scatter of the exact energy, so it cannot predict 8x8 or 10x10.
- NEXT (done, low): Angle closed; nothing to run.
