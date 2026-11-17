# Why the 8x8 log fit fails: image cancellation at the FN wall (2026-11-17, lead + PI, theorist mode)

## What was found
- R1''s stiff rows (log-loss gradient 26-73x typical) are configurations where |B| is small because the 256 symmetry images
  of the projected pair product cancel: kappa = |sum_g t_g| / sum_g |t_g| = 6e-4..9e-3 vs 0.27 for typical decade -9 rows.
  No single det f[U,D] is near-singular (sigma_min/median ~ typical). Data: runs/j1j2/r1prime_fit/diag_ho/image_cancel.json.
- |grad ln B| ~ 1/kappa: corr(ln kappa, ln tangent norm) = -0.93 over 200 random rows. The derivative dB/dtheta is ordinary;
  the 1/B is what explodes, because images cancel but their gradients do not.
- Those rows sit on the small side of a sign-violating bond (partners e^3-e^7.5 larger); the Jacobi target agrees they are small.
- Exact: the FN state obeys (H_xx - E)|phi(x)| = K(x) - Vs(x) (kept minus violating neighbour weight), so K > Vs everywhere at the
  fixed point. Rows with Vs > K are amplitude-sign inconsistent; a C-free sign round flips exactly those rows.
- Account with every step and tag: STORAGE_PROBLEM.md 'CURRENT 2026-11-17'. Log: delta_today.md 11-17.

## Why
The PI asked why the copy fails at 8x8 and insisted on deriving rather than measuring; the only computations were CPU reads of
existing files (no new run).

## Rejected and why
- Smaller steps (clip, backtracking, hinge): a step inside the linear region of a cancelling row is kappa_x/kappa_bulk ~ 1/30-1/450
  of the bulk step; the bulk stalls. Five patches met this.
- Square loss on a trainable base: no 1/B in the gradient, but bulk updates still scramble cancelling rows by (1/kappa)|delta|.
- 'Spurious dips with large f_J' (lead's conjecture): killed, f_J in [-3.13, 2.2] on all 2.1M pool rows.
- Calling the PP a 'Pfaffian' with one-particle hops: wrong; it is det f[U,D], an exchange changes a row and a column.

## Three things to understand
1. d ln B/d theta = (dB/d theta)/B: the trouble is B small by cancellation with a normal slope, not a large slope.
2. Frozen base x e^r (PI's form) removes 1/B: the log loss on r has no B in its gradient; its target c f_J is bounded within a round.
3. The limit is the moving sign: the rows a sign round flips (Vs > K) are the cancelling wall rows; after the flip their target
   rises by e^3-e^7.5, which a smooth r cannot make in one swap. Storage and sign meet at the inequality K(x) > Vs(x).
