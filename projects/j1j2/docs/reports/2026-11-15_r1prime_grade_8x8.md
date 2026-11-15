# One loop round on the strong 8x8 network lowers the fixed-node energy by 1.20(23)e-5 per site; the sealed prediction was 4-7x too large

- **Result:** E(R1') - E(warm start) = -1.20(23)e-5 per site, 5.2 sigma, all 8 chain-wise differences negative (bare c = 0 FN-GFMC, J2 = 0.5, 8x8, s306 base, M 1024, burn 150 + 600 blocks, 8 chains per arm, unpaired chain-scatter error).
- **Arms:** R1' -0.4988919(16) and warm start -0.4988799(16), both quotable; base re-run -0.4988795(27), not quotable (chain chi2/dof 2.11), matching the old base -0.4988780(27).
- **Warm start alone:** E(warm) - E(base) = -0.04(31)e-5, consistent with zero, as sealed (-1.2(1.4)e-6).
- **Prediction:** sealed (2'') -4.6..-8.3e-5 (an extrapolation) MISSED, at 0.14-0.26 of it. The trimmed held-out proxy (stored fraction 0.145 x G_s306 = -1.36(18)e-5) matched at 0.5 sigma.
- **Missed term:** the stored fraction f = 0.145 on held-out, against the 0.51-0.90 that the tangent share allowed. Why it is 3.5-6x short is not understood. Candidates: near-node non-linearity (d ln|P_1| ~ 1/P_1, tangent 26-73x at the worst rows), train vs held-out tangent share, finite steps. The Pfaffian angle diagnoses it on paper before R2'.
- **Not checked:** one seed of the fit; a fit-to-fit spread of +-0.2e-6 was seen earlier at 6x6 (LESSONS).
- **Next:** R2' early kill: |gain_2| < 6.0e-6.
- **Figure:** figs/2026-11-15_r1prime_grade_raw.png (25-block means, all 24 chains, flat).
- **Sources:** runs/j1j2/r1prime_grade/{manual_gate_8chain.txt, raw_grade_8chain.png, prod, prod2} (Paderborn 4271870, 4271928); fit 4255548 (theta_tail sha 15cff596). Lead re-derived the arm means and the difference from E_mix.npy. PLAN_pfaffian 'R1' DESIGN FINAL', 'R1' END RULE'; TRIED_IDEAS P33.
