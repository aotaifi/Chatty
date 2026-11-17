# 2026-11-17 sign angle: the copy floor is the label rule; 10x10 round 1 resolved

## What was found
- **T-BAD10 round 1 (10x10, declared alpha -0.170).** g_1 = 1.94(10)e-3 per site.
  - e_0 = -0.494079(64), e_1 = -0.496019(79); both QUOTABLE.
  - RESOLVED against the bias bound B64 = 2.64e-4.
  - Path: runs/j1j2/sign_bad10/r1/fn_net64/.
- **The 8x8 label rule reverses signs by construction.** The label is y = sgn(C u + Q), where:
  - u is the old sign;
  - C = 1 + alpha (D - E_REF) is the diagonal self term;
  - Q = alpha u (EL2 - D) is the neighbour term.
- **Why C < 0 everywhere.** On 8x8, D ranges over -14..+5, so D - E_REF >= 17.9 > 1/|alpha| = 5.7. Hence C < 0 for every configuration.
  - Wherever |Q| < |C|, the label is -u whatever u is.
  - 97% of the flip labels are of this kind.
  - 78.5% point away from sgn Q, the sign that minimises the off-diagonal local energy given the neighbours.
  - Sources: Astra cj-f4b619de888a; recomputed from table_r3 with 0 mismatches.
- **The network partly filters the harmful flips.** In decades -9..-5, 39% of the flips it copies are harmful, against 74% of the flips it misses.

## Tests, in order (all in PLAN_sign.md)
- **T-GROW: KILL.** Capacity (residual head, 2x width) and bond-weighted loss: cut < 2%. The class-balanced arm trades keeps for flips.
- **T-NEAR: PARTIAL.** Near-threshold flips are missed more (0.935 vs 0.685), but most misses are clear-cut.
- **T-CF: KILL on copy.** A refit on reversal-free labels sgn Q reproduces good flips at 0.42 vs n_3's 0.52, because it trains on 5x fewer flips. It keeps the former harmful flips at 0.93 vs 0.80.
- **First-order energy.** dE(CF - n_3) = -3.2(1.0)e-5 per site. This is below the lead's 4e-5 bar, so there is no FN grade.

## Alternatives rejected
- More width or a reweighted loss: train error falls, held-out error stays flat. That does not change the label term.
- An FN grade of CF vs n_3 now: its expected effect is under ~1.5x grade4's sigma (1.9e-5), so the probe cannot fail.

## Three things to understand
1. **The label decides.** With alpha this large, the step 1 + alpha(H - E_REF) is not a positive projector. On a bad amplitude, its flips are mostly reversals of whatever the net said.
2. **The joint-loop label rule must change.** Two options:
   - C-free: y = sgn(alpha u (EL - D)), flipping 2.65% of rows;
   - positive projector: |alpha'| < 1/(D_max - E_REF) = 0.0271 sampled (0.0125 global), flipping <= 0.8% of rows per step.
   My recommendation is C-free.
3. **Copying good flips stays at ~0.4-0.5** whatever the label set. That is the next open term, not yet diagnosed.

GPU used tonight: ~5.3 A40-h (T-GROW 4.48, T-CF ~0.8).
