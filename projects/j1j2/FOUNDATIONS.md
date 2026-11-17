# FOUNDATIONS: the non-obvious claims the j1j2 project rests on (CLAUDE.md point 7)
NAME CORRECTION 2026-11-17 (PI): our 'Pfaffian' / 'PP' is the Gutzwiller-projected SINGLET pair product, psi_PP = eps det f[Up, Dn]
(Nomura-Imada 2005.14142), the special case of a Pfaffian with only up-down pairing; a spin exchange changes one row AND one column (rank-2,
2x2-determinant ratio). The 8x8 base is B = sum_g phi_RBM(z_g)[det f_1 + lam det f_2] over 256 images, f, b, W complex, so B is complex
(ansatz/code/simple/pp2_lemma.py l.1-12). Angle, directory and file names keep 'pfaffian'. Definitions: STORAGE_PROBLEM.md 'Notation'.
Only claims that are subtle or were worked out by our own thinking: textbook results (Perron-Frobenius, the lattice FN bound
of ten Haaf et al. 1995, zero sets of real vs complex functions) do not get a row (PI 2026-10-28). A row enters once the lead has
read or re-derived it; otherwise it is marked HYPOTHESIS. Mechanisms on the equation board (artifact LwUGJoTxGJku3k6mTVgMVP)
that rest on such a claim cite its row.

| # | claim | theorem / equation | assumption | source | where used | status |
|---|---|---|---|---|---|---|
| F1 | Pair cost of a copy error, any size, frozen phi: c_xy = abs(H_xy) phi(y)^2 (r - r_true)^2 / r | algebraic identity from the FN diagonal term (board 3) at fixed phi (delta_today 2026-10-27) | phi frozen: an UPPER BOUND on the true loss, not the loss; over-weights deep dips (bound/true 17.7 for O2b, 6.05-7.81 for complex copies; pfaffian_d/split/split_*.json) | derived here | bond splits (Test D); not to be used for shares of the true tax | read; bound only |
