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

## Kill test (DRAFT, pending blind review cj-438ea329c07d; frozen before any fit is run)
Clusters (periodic, J2 along (+-1,+-1)): N = 16 (4x4), 18, 20, 26, 32, 34, 36 (6x6), 40 (tilted, Richter-Schulenburg).
Fit set N in {16, 18, 20, 26, 32}; predicted N in {36, 40}. Primary form F1: delta_inf + a N^-3/2. Secondary: F2 = F1 +
b N^-5/2; F3 = delta_inf + a N^-1.
- (a) J2 0.3 (both Neel): F1 predicts 36 and 40 within 1e-4/site.
- (b) J2 0.5: F1 predicts 36 and 40 within 1e-4/site, and F1-F3 predictions agree within 1e-4.
- (c) Control: the same forms on E0/N; max |miss| of E0 fit >= 5x max |miss| of delta fit, at each J2.
- (d) 8x8: E_B(QMC) + 64 delta(64) below -0.498946 and within 2e-4 of -0.4992.
- (e) 10x10: prediction below -0.4976921 (an exact energy must lie below a variational one) and inside
  [-0.49790, -0.49765].
Verdicts: (a) fails -> KILL, cause = cluster-shape scatter in delta (not the phase). (a) passes, (b) fails -> KILL,
cause = the board-18 term. (a)-(c) pass -> stage B. (d) or (e) fails -> KILL at scale, cause named from the 8x8 miss.

## Running
- Blind review: Astra cj-438ea329c07d (submitted 2026-10-29).

## Dead ends
(none yet)

## Cockpit
- STATUS (in progress, medium): Testing whether the sign-gap per site converges with cluster size fast enough to predict 8x8 and 10x10 energies.
- NEXT (open, medium): Exact energies of both Hamiltonians on clusters up to 36 sites, then the pre-registered fits.
