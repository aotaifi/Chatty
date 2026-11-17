# Handoff 2026-11-17: storage step, from failed heads to an explicit teacher (j1j2 lead, goal run, PI away)

What was built
- Rung 2 (exact 6x6 tables): the round's FN correction is u = ln(p/A) ~ l + c f_J[G], l = ln|1 + alpha(E_L - E)| (the Lanczos factor),
  f_J[G] one Jacobi step of the FN equation, c ~ 1.3. Applied exactly for 3 rounds it keeps 98.1% and 96.3% of each round's FN gain
  (rangelimit_6x6/rounds23/rounds.json; code ansatz/code/rangelimit_features.py, rangelimit_rounds.py).
- Proposal v2, Astra-reviewed: docs/theory/2026-11-17_next_method_proposal.md. Freeze the network, evaluate the formula as a teacher
  (alpha, c by energy search), consolidate a full network with a frozen-relative pair loss, accept a round only if E_H[chi] < E_FN[g].
- B10 harness built and CPU-tested, not submitted (runs/j1j2/b10_6x6/).

Why
- Rung 2b: an RBM head stores 2-3% of the correction (below a random vector). The sampled B10 gradient needs ~5e9 samples at 6x6.
  So step 4's consensus route was blocked twice before any GPU hour.

Rejected and why
- B10 (frozen-H_FN Rayleigh on a head): head class too weak, gradient unsampleable (STORAGE_PROBLEM 'RUNG 2b FINAL', 'B10 6x6 HARNESS').
- Nesting the explicit formula: radius 2k after k rounds.
- Literal log-D loss: brings back the moving 1/B (Astra eq. 19).
- My v1 claim 'tax/gain = eps^2 in the D-norm': wrong, the M-R law was an empirical calibration on two poor copies (ratio 4.7-18 here).

Three things to understand
1. The correction is not something to learn inside a round; it is a formula. What must be learned is the whole network that reproduces it
   (consolidation), and that is where every earlier attempt failed.
2. The certificate E_H[chi_k] < E_FN[g_k] (Astra eq. 10) guarantees descent; on 6x6 the teacher leaves only ~5% of a round's gain for
   consolidation error. Earlier storage fits kept ~6%, i.e. lost ~94%.
3. Everything so far uses the fixed exact sign s2 and fitted c; moving Lanczos signs and energy-searched c are the next rung-2 check
   (CPU, kill in the proposal's A1). Any network run needs the PI's 'now test it'.
