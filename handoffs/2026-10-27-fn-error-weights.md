# 2026-10-27 What the fixed-node energy weighs, versus what a network fit weighs

**What was built.** A derivation plus an exact 4x4 check of how storage (copy) errors in a guide wavefunction cost
lattice fixed-node (FN) energy. Code: projects/j1j2/ansatz/code/fn_weight_check.py. Log:
/scratch/a/A.Otaifi/ansatz_runs/fn_weights/fn_weight_check_4x4.log. Equations page:
https://claude.ai/artifact/LwUGJoTxGJku3k6mTVgMVP.

**Result.**
- Log error: psi_T(x) = psi(x) e^{eps(x)}.
- Every copy loss weighs configuration x by psi(x)^2 at small eps: log regression, fidelity, and L2 in psi alike.
- The FN energy changes only on "violating" exchange pairs, those with psi_T(x) H_xy psi_T(y) > 0:
  dE ~ c sum |H_xy| psi(y)^2 (r - r*)^2 / r, where r* = |psi(x)/psi(y)| <= 1, r is the copy's ratio, and c = 0.63-0.66 at 4x4.
- At small error this is the geometric-mean weight |H psi(x) psi(y)| on (eps_y - eps_x)^2.
- Near a node the copy usually has r >> r*, and the cost is |H| psi(y)^2 r, the large side's weight (FN-loop lead's point).
- Measured at 4x4: the same copy loss costs 19x more energy at psi^2 ~ 1e-9 than at 1e-3.
- 43% of exchange pairs are violating for the exact state. Every J2 pair is violating in a Marshall-like frame, not only
  pairs at nodes.

**Alternatives this rules out or explains.** All three designs these equations suggest are already in TRIED_IDEAS.md:
- learn the wall ratios directly (bond field, E8/E9: oracle ceiling works, trained ratios plateau 1.92e-5);
- hop only near nodes (D12: 76-80% of the gain, but ~50 network calls per flagged configuration);
- fit with the FN pair objective (B30: does not generalise; the top 0.1% of bonds carry 55% of the loss).

**Three things to understand.**
1. The fixed-node energy sees a guide only through its signs and its amplitude ratios on violating pairs.
2. Fitting error is cheap where psi is large and expensive across pairs where psi jumps. No loss of the form
   sum psi^2 (...) can see that.
3. Near nodes, whether a network can make a deep dip in one spin flip (linear zeros versus log output) decides the
   cost. The open diagnostic is to split the measured 6x6 refit tax bond by bond into |H| psi(y)^2 (r - r*)^2 / r.
