# Storing the fixed-node state in a network: the three issues (PI + ansatz lead, 2026-10-27)

Equations: equation board https://claude.ai/artifact/LwUGJoTxGJku3k6mTVgMVP (block numbers below).
Derivation and 4x4 check: ../../handoffs/2026-10-27-fn-error-weights.md. Tried ideas: TRIED_IDEAS.md (row ids below).

| # | issue | cause (equation) | what was tried | status |
|---|---|---|---|---|
| 1 | log problem | output ln|psi_T| and loss in ln weigh errors by psi^2 (13); FN energy weighs pairs across the node by the big side, |H| psi(y)^2 (r - r*)^2 / r (5); a smooth log net cannot dip ln(1/r*) in one swap (7) | edge/pair losses with log-output nets: B13 (4x4 66% vs 46%), B30 (6x6, does not generalise), B36 (weak) | fix on paper: e^J Pf output + pair loss L_fix (15). L_fix needs phi pointwise (exact tables at 6x6 only). E_FN training (16) is C15: tried, failed sampled (issue 3). Linear-zero output + L_fix at 6x6: untested |
| 2 | no walkers at the node | walkers ~ g phi, rare near nodes (AVSL-08: 0/198 profitable orbits visited) | flatter importance g = |psi_T|^beta: F1 parked (ESS 5% at beta 1/2, 6x6); eigen-equation labels from visited neighbours (10): B18 itfit semi-implicit step 87-97%, inconclusive | open |
| 3 | gradient denominator | E_FN force F(x) = sum_viol K_xy a(y)/a(x) [O(y) - O(x)] (16): a(x) small near nodes | C15/E1: exact 4x4 force works (1.74x descent with SR); SAMPLED 8x8 FAILED: walkers for force cosine 0.7 grow 8e4 -> 2.5e8 near the fixed point, physical E_FN worse by +0.046 total (3549023/24). Factorised residual a_new = a_k e^{r} gives finite-variance gradients (Tim, seq72-74); B10 exact 4x4 99.7%, B11 8x8 inconclusive (OOM). Clipped Lanczos factor (B27/D7): works as one-step target, retention 0.84 -> 0.09 | open: the noise problem is the measured blocker of E_FN training |

Reading (lead): issue 3 is not new; the factorised residual is the known handle. Issue 1's fix and issue 3's fix
are compatible: e^{r} J-type correction on a guide whose node (Pf) is kept, trained on E_FN with SR.
Next decision for the PI: which of 1-3 to test first, on paper before any run.

## Test T-Lfix (pre-registered 2026-10-27, PI asked; before any run)
Does the pair loss L_fix (board 15) store the fixed-node state better than the log loss in our code (board 13), same network?
- Target phi = exact FN state of a fixed guide (4x4: the existing exact loop targets; 6x6: the G-tax target phi_0 or an O2b round).
- Arms, same network class (RBM+PP, has Pfaffian zeros), same parameters, same optimiser and step budget, 3 seeds:
  (A) log loss as per_sample in ansatz/code/c2le_gtax.py l.471-473; (B) L_fix. Optional (C): L_fix on a log-only net
  (positive RBM x Marshall sign) to separate loss from architecture.
- Readout: tax_k = E_FN(K_k fit) - E_FN(K_k phi), k = 0, 1, 2; plus fit error by |phi|^2 decade and by pair class.
- Pass: tax_2(B) <= 0.5 x tax_2(A) at both sizes. Kill: tax_2(B) >= 0.9 x tax_2(A) at both sizes. Between: inconclusive.
- 4x4 on workstation CPU first. 6x6 needs a GPU job: launcher read by the lead before submission.
