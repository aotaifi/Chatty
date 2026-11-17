# Why the 8x8 log fit fails: two cancellations at the node (2026-11-17, lead + PI, theorist mode; reviewed by Astra)

## What was found
- R1''s stiff rows (log-loss Jacobian 26-73x typical) are configurations where |B| is small because the symmetry images of the
  projected pair product cancel: kappa = |sum_g t_g| / sum_g |t_g| = 6e-4..9e-3 (P_1) vs 0.27 for typical decade -9 rows.
  No single det f[U,D] is near-singular. Data: runs/j1j2/r1prime_fit/diag_ho/image_cancel.json.
- |grad ln B| ~ 1/kappa statistically: corr(ln kappa, ln Jacobian norm) = -0.93, log-log slope -0.99 (200 random decade -9 rows).
  The slope dB/dtheta is ordinary; the 1/B is what explodes, because images cancel and their gradients do not.
- At those rows the kept and violating neighbour sums nearly balance: K/Vs = 0.71-0.98, both 28-503x larger than D|B(x)|.
  The local FN formula |B(x)| ~ (K - Vs)/D makes such rows small by a physical cancellation; the network copies it by cancelling images.
- Flipping the sign at such a row swaps K and Vs: the target moves by only 2|ln(K/Vs)| ~ 0.03-0.68 (x c), so the frozen-base form
  survives a sign round there. Account with every step: STORAGE_PROBLEM.md 'CURRENT 2026-11-17' (review block: Astra cj-7018a90b8916).

## Why
The PI asked why the copy fails at 8x8 and insisted on deriving rather than measuring; the only computations were CPU reads of
existing files and Astra's independent recomputation (no new run).

## Rejected and why
- Smaller steps (clip, backtracking, hinge): a step inside a cancelling row's linear region is ~kappa_x/kappa_bulk of the bulk step.
- 'Freezing the base is necessary': withdrawn (Astra): the norm bounds sensitivity, a given direction may avoid it; freezing is sufficient.
- 'Spurious dips with large f_J': none in the 2.1M sampled rows (raw f_J in [-0.78, 0.55]); unsampled deeper dips not excluded.
- 'After a sign flip the target rises e^3-e^7.5' (lead): wrong, it assumed one dominant neighbour; measured K ~ Vs.
- Calling the PP a 'Pfaffian' with one-particle hops: it is det f[U,D]; an exchange changes a row and a column.

## Three things to understand
1. d ln B/d theta = (dB/d theta)/B: the trouble is B small by cancellation with a normal slope, not a large slope.
2. Frozen base x e^r (PI's form) has no moving 1/B: the log loss on r differentiates only r. What stays is a fixed pair weight
   rho ~ Vs/|B(x)| (pool max 16131) and the target clip at |c f| <= 4.88.
3. The bad rows are near the node of S, where the exact amplitude is a difference K - Vs of two large sums.
