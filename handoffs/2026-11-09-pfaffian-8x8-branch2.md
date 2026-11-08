# Pfaffian angle: 8x8 loop, branch 2 (2026-11-04 to 11-09)

## What was built
- The 8x8 tools, each with a check that failed when deliberately broken:
  - pool8 sampler;
  - pool_fit arm pp8 (two-PP + RBM);
  - gfmc_aug / gfmc_aug8 / gfmc_b2 grading guides;
  - pp2_lemma speedups (endpoint dedup, PhaseSign with a direct fallback).
- A 2.0e6-entry 8x8 pool labelled under the production network's own sign sgn(M1) (pool_s2b, sha 426487e9).
- The first 8x8 round-1 refit, R1 (Paderborn 4143668, 1.73 H100-h).
- Its first energy grade, at c = 0, is running (Paderborn 4165381).

## Why branch 2 (fixed sign) and not the stored-sign loop
- Every trained 8x8 sign network labels the round's correction worse than the frozen production sign: eps^2 0.54-1.47 against 0.40.
- Training lowers the wrong-sign count, but it moves the errors onto high-amplitude bonds, and those are the bonds the labels weight most.
- So the round keeps sgn(M1). The sign is improved only at the end, by one Lanczos step (option (ii)).
- The headline comparison is B_k + p=1 against M1 + p=1, -0.498946(5).

## Alternatives rejected, and why
- Stored-sign loop (branch 1): gate (a) failed on 13 checkpoints. The alpha mismatch was ruled out (eps^2 0.024).
- Direct s_G labels: about 150 A40-h. They are also unnecessary, since branch 2 labels with sgn(M1).
- An 8.6e6-draw pool: that size was set under s_G. Under the own sign the heavy f_J tail almost vanishes (0.00 vs 0.09 on the same x), and 2.0e6 entries passed the gate.
- A paired c1 estimator: unbiased, but it saves only about 1.6x, so it is sized as if there were no saving.

## Three things to understand
1. The correction f_J = ln SK - ln den depends on the sign through which bonds count as kept. A sign whose nodes cut through regions where |B| has no node leaves some x with few kept bonds and a very negative f_J. That is s_G; the network's own sign does not do this.
2. The copy loss weights a pair by |T(y)/T(x)|. A few deep x with a heavy partner dominate the loss. R1 improved 99% of the held-out entries but left those bonds unfitted, with |B_1| too large at x. It is not yet known whether sampling or the network class is the cause (see the separator in PLAN_pfaffian.md).
3. Lattice fixed-node energy depends on amplitude ratios across sign-violating bonds, not only on the signs. That is why an amplitude-only round can lower E_FN at a fixed sign. Whether it does at 8x8 is what the running grade measures, against same-sign p=0 -0.4987565(28).

State, branches and job ids: projects/j1j2/PLAN_pfaffian.md, section "STATE AT MIGRATION (2026-11-09)".
