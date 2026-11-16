# Pfaffian angle: why R1' stored so little (2026-11-15/16)

## What was built
- R1' grade (8x8, retrained base s306): loop gain E(R1') - E(warm) = -1.20(23)e-5/site, 8 chains per arm.
  - The sealed prediction (2'') was -4.6..-8.3e-5, so it missed.
  - Source: runs/j1j2/r1prime_grade/manual_gate_8chain.txt.
- The term that missed is f, the share of the copy loss the fit removes on held-out rows.
  - (2'') assumed f = 0.51-0.90.
  - The fit realised 0.145(19) with 5 rows trimmed, and 0.009(90) untrimmed.
- Diagnosis tools, all under runs/j1j2/r1prime_fit/diag_ho/:
  - full held-out evaluation (full_ho.py);
  - jvp linearisation (full_lin.py);
  - Gauss-Newton tangent share on 373k train rows (gn/);
  - tail-index and loss-ESS scaling (hill_tail.json, loss_ess_scaling.json).

## Result (a hypothesis, two supporting measurements)
- The fit creates heavy near-node rows at decade -9, where d ln|P_1| ~ 1/P_1 (tangent norm 34x the median). Those rows then dominate the loss and its gradient.
- The loss-weighted ESS is 2463 rows at the start and 640 at the end, against 18466 parameters.
- The decade -9 ESS exponent goes from 0.78 at the start to -0.7 at the end. Decades -8..-2 scale ~linearly.

## Rejected and why
- The 4096-row held-out batch as a gate: one decade -9 row carried 80% of it. Replaced by the full held-out set with a row bootstrap.
- The trimmed gate: rejected by the lead. It drops exactly the rows where amplitude errors cost FN energy most.
- Linearising the total step: |g.dtheta| reaches ~590, and even small-step rows get only 1/3 from the linear term, so the test was invalid.
- The GN tangent share as a class-vs-optimiser test: not identifiable, because loss-ESS << P.
  - The held-out share is <= 0.01.
  - The in-sample share is optimism: a shuffled control already reaches 0.11-0.39.

## Three things to understand
1. Every copy-loss number here is an importance-sampled mean with a tail index ~1-1.7. Quote it only on the full set with a row bootstrap, and check the top-row share first.
2. The bond-ESS gate (>= 3P per decade on the weights) passed while the loss-weighted ESS was ~10^3. The relevant ESS includes residual^2.
3. The bond-only loss is the theory-preferred estimand: FN sees the guide only through blocked bonds. Our two graded points (R1, R1') cannot separate it from the total loss.

Next step and state: PLAN_pfaffian.md, the "STATE (2026-11-16 ...)" block.
