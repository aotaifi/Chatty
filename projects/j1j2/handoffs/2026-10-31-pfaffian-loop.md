# Pfaffian angle handoff, 2026-10-31

## What was built
- A label-free loop at 6x6, J2 = 0.5, scored by exact fixed-node (FN) energies. Each round refits a complex pair product B
  (no backflow; the RBM block stays inactive) to the target B_prev e^{c f_J[B_prev]}. Here f_J is the guide's own Jacobi
  factor (board block 20) and c is about 4.
- The guide that gets used is the augmented guide B e^{c f_J[B]}, one extra Jacobi level. At 6x6 a walker step with it costs
  5.4x a plain PP, measured in T2.
- Results under the competitor's sign s_G, as energy per site above E0:
  - round 2: augmented guide 5.21e-6 and 5.29e-6 (two seeds);
  - round 3: 4.79e-6 (official value, kill rule) and 4.64e-6 (tail-averaged).
  - The competitor (RBM+PP plus one signed Lanczos step) is at 5.574e-6.
- Report: docs/reports/2026-10-31_label_free_loop_6x6.md. Plan and every pre-registration: PLAN_pfaffian.md.
  Tried ideas: P14-P19 in TRIED_IDEAS.md.

## Why this design
- The FN correction a round needs is about the current guide's own f_J (tests K and K3). It is orthogonal to the PP's linear
  tangent space, which holds 0-4% of it (Θ v2, calibrated).
- So f_J stays an explicit layer. The nonlinear refit then banks r = 0.11-0.18 of each round's gain.
- Label-free targets store more than exact labels: 37% (L-SWO) against 15% (R1b) in round 1.

## Rejected, and why
- Backflow base: it makes the Jacobi factor cost 36.5x a plain PP (T2), because each swap becomes rank 12-14.
- RBM residual banks and a feature dictionary: ε² 0.95-0.98 (tests N and K). They miss the direction.
- A stop rule on loss flatness: in R2G and round 3 the held-out loss rose while E_FN fell.
- A stop rule on an E_FN tolerance: at a fixed target, E_FN walks 0.1-0.2e-6 per 150 steps.
- Training the sign: out of scope by the PI's rule. The sign comes from Krylov, the amplitude from FN.

## Three things to understand
1. E_FN sees a copy only through V_A on blocked pairs. The copy loss weights errors differently, so loss and E_FN disagree in
   loss-flat directions. That disagreement is why stopping on the loss failed and why averaging the tail iterates works
   (0.35e-6 better than the best single iterate).
2. Whether r declines from round to round is NOT established: q is 0.63-0.95 between single iterates, and round 3 averaged
   kept 0.178. Round 4 (averaged, about 1.5 H100-h) would settle it; its prediction is in the report.
3. On equal cost at 6x6:
   - training ties or loses: 4.4 H100-h on top of the competitor's PP start, against their 1.6-7 H100-h;
   - the walker step is unresolved: the competitor's RBM part decides it, and it is unmeasured;
   - the exact FN solves used for E' and c* exist only at 6x6.

## State at handoff
- No jobs running and no workers.
- The lead accepted both wording corrections and gave a GO for round 4 (averaged protocol, about 1.5 H100-h). It is
  pre-registered in PLAN_pfaffian.md under "Round 4" and NOT yet staged. Its q decides whether the loop is worth scaling.
- The walker-step cost against the competitor is unresolved until measured.
