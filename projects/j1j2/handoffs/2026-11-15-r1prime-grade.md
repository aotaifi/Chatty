# Handoff 2026-11-15: R1' (first loop round on the strong 8x8 base) graded

**Built:** one label-free loop round on s306 (RBM+PP 8x8, 'simple' projection): warm start B_0' + lam P_2 (s307's PP), fit to f_J labels
(c 4) for 1500 steps (Paderborn 4255548), then graded by bare FN-GFMC against the warm start and the base (3 arms x 8 chains,
Paderborn 4271870 + 4271928).
**Result:** loop gain -1.20(23)e-5/site (5.2 sigma); the warm start alone does nothing (-0.04(31)e-5). Report: docs/reports/2026-11-15_r1prime_grade_8x8.md.
**Why this design:** the warm arm isolates the fit from the warm start's own change; the base re-run used the same burn (150 blocks),
because the old base arm had a ~1e-5 burn bias. Eight chains per arm, because headline B (4222359) showed chain offsets up to 2.5e-5 that 4 chains cannot detect.
**Rejected:** grading against the old base arm (different burn); a trimmed held-out gate (it drops the highest-weight rows, which matter most for the energy);
a paired FN estimator for R1' (separate branching runs decorrelate, LESSONS; the decision uses the unpaired error).
**Three things to understand:**
1. The held-out loss on one 4096-row batch was a one-row statistic: a few near-node rows with bond weight ~200 dominate. Only the full held-out set (424585 rows) with a bootstrap error means anything.
2. Near a node of P_1, d ln|P_1| = (grad P_1 . dtheta)/P_1 diverges, so those rows jump by O(3) in log under bulk updates. This was transient (step 150 worse, step 300 flat), not a runaway.
3. The miss is in f, the fraction of the copy loss the fit realises on held-out (0.145 vs 0.51-0.90). The trimmed proxy predicted the grade at 0.5 sigma, so it can serve as a cheap pre-grade for R2'.
