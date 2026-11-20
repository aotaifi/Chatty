# Copying rounds at flat cost: two-hop inputs (2026-11-20)

**Built.** A copy loop for the PI's round map M on 6x6: chi_k = B e^{g_k}, B the frozen pair-product table (A_0), g_k a cubic head
refit each round on the target T_k = M[chi_{k-1}] (frozen previous copy, as in arXiv:2307.15521). Code: branch j1j2-ratnet,
capacity_fit.py (extra=, shufcol=), round_map.py (--save-parts), commit 24517de. Runs: /scratch/a/A.Otaifi/ansatz_runs/consol_6x6/ref2hop.

**Result (grade N = best FN energy per site above exact, e-6).**
| copy | round 2 | round 3 |
|---|---|---|
| one-hop inputs (arm A) | 3.061 | 2.711 |
| + two-hop input l_1 (arm B) | 2.776 | 2.432 |
| exact rounds | 2.416 | 1.918 (derived) |
Arm B keeps 94% / 92% of the exact rounds' gain from A_0 (8.683). Copy error eps_k = N(chi_k) - N(T_k): A 0.49 / 0.58, B 0.21 / 0.39.

**Why.** The round map needs the local energy of the previous copy, which contains the head's value at every neighbouring
configuration, h(y_b). A head that sees only x's one-hop ratios of B cannot hold that, so it loses a near-constant amount each round,
and the loop stalls (fixed point N_inf + eps/(1 - kappa), kappa 0.32, N_inf 1.68). l_1 = ln|1 + a(E_L[chi_1] - E)| from the frozen
round-1 copy carries it at a fixed cost (z^2 ratio evaluations per amplitude, independent of the round).

**Rejected.** Refit the GPU ratnet from scratch each round: N 3.39 at round 2, because it drops the previous head (keep it as input or
warm start). Feed the previous round's exact local energy: nests, cost z^k. Spin-input ViT add-on: could not rebuild B's ratios (11-18).
Refitting B itself: October refits absorbed 0-37% per round.

**Three things to understand.**
1. Per-round copy fraction q falls by construction when the absolute error eps is constant; judge by eps and by the cumulative share.
2. The loop is an optimiser of the stored class: it converges to that class's best energy, so lower energy needs more input reach, not a better copy.
3. l_1 comes from round 1 and goes stale (eps 0.21 -> 0.39); running now: refresh it each round from a one-hop projection of the latest copy
   (STORAGE 'REFRESHED ONE-HOP REFERENCE'), cost still flat.
