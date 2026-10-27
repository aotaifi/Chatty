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

## Test T-flat (pre-registered 2026-10-27, PI said yes; before any run)
Does flattening the walker distribution cut the walkers needed for a usable E_FN gradient (issue 3)?
- Exact 4x4, the guide(s) used in C15 / Tim seq48 (early Marshall-like guide and a near-fixed-point guide), exact phi_FN.
- Exact E_FN gradient (Hellmann-Feynman, board 16; C15 force F(x) = sum_viol K_xy a(y)/a(x) [O(y) - O(x)]) as reference.
- Sampled estimator: draw M configs i.i.d. from g phi / Z with g = |a|^beta, beta in {1, 0.75, 0.5, 0.25}; reweight to phi^2 exactly
  (w = phi/g, self-normalised); gradient estimate = weighted mean of F. Cosine with the exact gradient vs M; M_0.7 = walkers for
  mean cosine 0.7 (fit over M = 1e2..1e7, 20 repeats); also effective sample size per M.
- Pass: M_0.7(beta best) <= 0.1 x M_0.7(beta 1) at the near-fixed-point guide. Kill: >= 0.5 x. Between: inconclusive.
- Check: beta = 1 reproduces C15's order of magnitude (8e4 early, 2.5e8 near fixed point, on that guide) or the deviation is explained.

## Step back 2026-10-27 (PI: "you should excel at training networks on this")
1. Verdict: continue the two tests, but add the missing control: an ideal ceiling arm in T-Lfix, training the same
   network directly on the exact E_FN(K_2 fit) at 4x4 (full sum, no sampling, no loss proxy). Without it a T-Lfix
   miss cannot tell a wrong loss from a network class that cannot represent the near-node structure.
2. Thought that changed it: as an ML problem this is task-metric loss + coverage of a rare boundary + the right
   inductive bias, and we never measured the class's best possible tax (representability). G1/G14 gave 7.06e-6
   with an oracle sign at 6x6. That hints the class itself may be the floor.
3. Cheapest test: the ceiling arm, minutes of CPU at 4x4. Independent Opus review of the logic requested by the PI.
Literature: one arXiv query (neural guide + lattice fixed node + guiding) returned nothing; guess stands as unpublished.

## Independent review 2026-10-27 (Opus, ML/NQS; verified by the lead where marked) - CORRECTS issue 1
- VERIFIED (c2le_gtax.py l.426-464): our 6x6 copy loss is NOT psi^2-weighted. Configurations are drawn equally per decade
  (-5..-13), and bond terms are drawn proportional to |H| phi(y)/phi(x) on violating bonds: already FN-shaped (small-error
  form of board 5). So "the loss weighs errors by psi^2" is wrong for our code. Issue 1 as stated does not explain the G-tax.
- Derived by the reviewer: E_FN[psi_T] <= <phi|H_FN[phi]|phi> + L_fix (exact upper bound at frozen phi, same signs).
  Storing with L_fix is therefore majorise-minimise, and its fixed point is a stationary point of E_FN over the class,
  the same as direct E_FN training (C15). Issues 1 and 3 are one issue. No loss can beat the class's E_FN ceiling.
- 4x4 T-Lfix as built was limited by a sign floor and by bulk capacity (392 parameters for 441 orbits), not by the node.
  Amended: target sign in all arms, tax_0 primary, ceiling arm via the stage (c) code at cfc2681.
- T-flat crashed at import (Python 3.9) and measured in table space. Not rerun; redesign in network parameter space if needed.
- New deciding number: R_ceil = tax(ceiling) / tax(warm start). R_ceil >= 0.7 means the network class is the floor.
  Then storage is a representability (ansatz) question, not a loss question.
- Reviewer's ideas, not in TRIED_IDEAS: shell-only flattening, g = |psi_T| max(1, eps max_viol |psi_T(y)/psi_T(x)|)
  (lattice Attaccalite-Sorella / Pathak-Wagner); zero-variance control variates for the force (Assaraf-Caffarel 1999/2003).

## Bond split of the 6x6 refit tax (O2b round 1; worker, ws CPU; raw split read by the lead; /scratch/a/A.Otaifi/ansatz_runs/bond_split/)
- Target phi_1 and copy a_2 share the oracle sign s2, so the sign-mismatch share is 0 by construction.
- Measured zero-hop E_FN(copy) - E_FN(phi_1) = 1.999e-5 per site (own LOBPCG matches stored 2.98e-5 / 9.85e-6).
- Frozen-phi pair cost (board 5 summed) = 3.53e-4 per site, 17.7x the measured difference: at a real 6x6 fit the bound is very
  loose (the FN state relaxes), so its decomposition is indicative only.
- Share of that cost, by r* decade x r/r*: 75% from r < r*/3 (the copy's small side too SMALL, an over-deep wall), only 3.5% from
  r > 3 r* (copy too large). r* mostly 1e-2..1e-1 (52%) and 1e-1..1 (33%); under 0.3% below r* = 1e-3.
- By phi(y)^2 decade: peaks at 1e-9..1e-12 (75%). J1 bonds 65%, J2 35%.
- Reading: the tax is not the "near-node r >> r*" regime; it is moderate-ratio pairs in the tail where the copy makes the small side
  too small. Checks: copy = phi gives 0; phi e^{0.01 g} matches board formula to 1.00005; noise 0.3 gives 2.1e-3; 1% sign flips
  route 5% to mismatch classes. Round 3 running (unchecked).
