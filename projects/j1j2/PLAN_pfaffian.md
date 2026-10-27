# PLAN_pfaffian (angle lead "pfaffian angle", started 2026-10-27 by the ansatz angle; PI asked for it)

## Question
Can a Pfaffian-type network copy the fixed-node (FN) state accurately in the rare configurations,
where today's copies fail? Words: configuration x, neighbour y (one swap), amplitude psi(x),
true ratio r* = phi(x)/phi(y), the copy's ratio r.

## Facts this rests on (verified 2026-10-27; STORAGE_PROBLEM.md, TRIED_IDEAS.md)
- Best-case copy: O2b round 1, Chatty/runs/compress6/runs/o2b_17431886, PP + 3x3 linear backflow, 1944 parameters
  (f 1260 used + u1,u2 648), symmetrised over 576 images, no RBM (code/cx_t1s.py l.38, l.154-176; board block 18).
  Log loss on the exact FN state; stopped at 80 steps, not converged.
- Its error: 75% of the cost from pairs where r < r*/3 (copy's small side too small), r* 0.01-1, phi^2 1e-9..1e-12.
- Failures to quote: B31 (bigger backflow 5x5 does not cut the tax: 5.60e-6 vs 4.58e-6 vs 5.93e-6), G4/C11 (PP+bf by
  plain VMC: no gain in 400 SR steps), G3 (fixed PP killed in T0; backflow route inconclusive, no tail-weighted
  training), B29 (tail-weighted fits: none below 1e-5), G11 (projected BCS + visons killed at 4x6),
  D13 (residual over PP is not local at J2 = 0.5). Pass: G14/G1 7.06e-6 with oracle sign (one seed).
- Published: RBM+PP (Nomura-Imada arXiv:2005.14142); neural backflow (Luo-Clark 2019).

## Scope
Paper discussion with the PI first; no run until he agrees. Candidates: PP + linear backflow (what we used),
neural backflow, sum of Pfaffians e^J sum_k Pf[M_k], plus RBM/Jastrow factor.
Waits on: ansatz angle's 4x4 ceiling test (R_ceil) and guide-vs-FN-vs-copy split (results into STORAGE_PROBLEM.md).
Budget: none spent; to set with the PI when a test is proposed.

## Running
none

## Dead ends
none yet

## Cockpit
- STATUS (open, medium): Discussing with the PI whether a pair-product network can copy the fixed-node state in rare configurations.
- NEXT (open, medium): Agree on paper which Pfaffian variant to test and the test that would kill it.
