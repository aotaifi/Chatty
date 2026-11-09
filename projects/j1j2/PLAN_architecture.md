# PLAN_architecture: an amplitude network built for the Jacobi loop (angle lead, started 2026-11-09)

Brief: proposals/2026-11-05_architecture_angle_brief.md (incl. 2026-11-08 gated hybrid). Reports to the project lead ("FN-loop lead").
Paper + CPU only. No GPU, no hybrid build before the Pfaffian angle's R1 separator (a)/(b) says CLASS (b) (PLAN_pfaffian "Separator ...").

## Open question (one)
Which network class carries the loop's amplitude at 14x14, i.e. passes C1 and C2 below? Budget: ~2 days paper/CPU (to 2026-11-11).
Scope: algebraic classes (PP sums, Pfaffians, tied PP sums) and local log-additive factors; generic NNs (CNN-MPS, ViT, RBM) passed over
because their two-swap ratio costs ~2N plain (board 23; proposals/2026-11-05_cnn_mps_twoswap_cost.md), except as a gated or local add-on.

## C1, C2 as equations
Notation: N sites, n = N/2, d ~ 2N one-swap neighbours, |G| projection images, U = per-move cache update, R1/R2 = cost of one one-/two-swap
ratio from the cache at x. A Jacobi step per walker move needs all d^2 two-swap ratios (board 23).
- (C1) kappa(N) = [U + d R1 + d^2 R2] / [U + d R1] stays bounded as N grows AND kappa(14x14) <= kappa_max ~ 30
  (measured: one PP 5.5x, two PPs 10.9x plain at 10x10). Equivalent: R2 = O(U / d^2). Determinant tables: U ~ n^2, R2 = det4 = O(1): pass.
  Generic net: R2 = R1 = C_eval ~ N: kappa ~ d, fail.
- (C2) the round's correction Delta* = ln T_k - ln|B_k| = c f_J[B_k] (fixed form, b2_prep.py) must be reachable by a refit of the class, in the
  energy's own metric (board 5): with eta = ln|B_new| - ln|B_k| - Delta*,
  eps^2 = sum_{blocked xy} |H_xy| phi_x phi_y (eta_x - eta_y)^2 / sum_{blocked xy} |H_xy| phi_x phi_y (Delta*_x - Delta*_y)^2 <= 1 - r_min,
  r = 1 - eps^2 stored per round (M-R law), r_min = 0.35 (V gate). Only BOND DIFFERENCES of the correction matter.
  Linear reach = Theta v2 share C = b^+ G^+ b / D_RR (board 27); the nonlinear refit is what was measured (r).

## Exact identity found here (verified, /scratch/a/A.Otaifi/ansatz_runs/arch_paper/ident.py, 4x4)
With pp2_lemma's parity par(z) (= site-ordered fermion sign), for every bond b and every x (aligned or not):
  PP_f(P_b x) = - PP_{f o pi_b}(x),  pi_b = transposition of b's two sites, f o pi_b (k,l) = f(pi_b k, pi_b l).   [max err 5e-15; broken + sign: 2.0]
  => H B_f = c0 B_f - sum_b (J_b/2) B_{f o pi_b}  (Heisenberg exchange = site permutation), for any projection.   [summed, full proj.: 1.3e-14]
  If the unprojected PP (with its RBM) is invariant under a translation subgroup T' of index m, the 4N terms collapse to one per T'-orbit:
  H B = c0 B - sum_{O=1..4m} (|O| J_O/2) B_{f o pi_{b_O}}.   [m = 1, full proj.: 1.5e-15 vs 3.8 / 4.5 for non-equivalent / general-f controls]
  So ONE signed Lanczos step (1 + alpha(H - E)) B of a T'-invariant PP is EXACTLY a sum of 1 + 4m projected PPs (5 for m = 1, 17 for 2x2
  sublattice f), each with its own fast updates: cost (1 + 4m) x plain, constant in N. For general f (production rb8_prod, m = N) it is 4N + 1.
  Two arXiv searches found no statement of this; novelty unverified.

## Candidates on paper
| candidate | C1 | C2 | verdict |
|---|---|---|---|
| sum of K independent PPs | pass: kappa ~ K x one PP (10.9 vs 5.5 measured); the table method (Clark/Morales; not read) needs a shared reference, absent for independent f_k: no saving | r 0.49 at K = 2 (R5), 0.25 round 6. A general-f PP's Lanczos step has PP-rank 4N+1 (identity): fixed K cannot hold repeated rounds; candidate mechanism of the r decline | pass C1; C2 partial, falls with rounds (hypothesis) |
| tied PP sum S_{1+4m}: B_f + sum_O c_O B_{f o pi_O}, f T'-invariant | pass: (1+4m) x plain; a Jacobi head on top costs (1+4m) x one level, i.e. two levels at constant cost (board 20 had ~2e3 x at 14x14) | one Lanczos step EXACT (identity); Jacobi vs Lanczos differ ~8% at 6x6 (PLAN.md); round 2 needs compression (f o pi_O not invariant) | PASS on paper for one step; open: bare cost of imposing T'-invariance on f |
| local dressing of the swapped sites' pair amplitudes (backflow-type) | backflow measured 36.5x at 6x6, grows ~N^2 (board 23). Finite-pattern dressing: O(1) in N but det of ~4w rows (w ~ 9-16) -> kappa >> 30 | not reached | FAIL C1 (practical) |
| 2-body Jastrow | pass (cached fields, O(1)) | fail: 2-body features eps^2 0.973 (Test L, P7) | FAIL C2 |
| Pfaffian sums (S_z = 0 sector) | pass: rank-4 Pfaffian update, ~4 x a PP per term | triplet blocks A (up-up), D (down-down) enter only as products A^k D^k (pairs must balance), so dPf/dA = 0 at A = D = 0: zero linear reach at the PP point and a bilinear saddle for a warm-started refit (LESSONS "bilinear saddle") | FAIL: dominated by PP sums (derived, not run) |
| gated hybrid G = B exp[w(|B|^2) g], g global CNN | FAIL at 8x8: kappa_g = d^2 p_z C_g / C_plain with d^2 = 1.01e4 distinct z per walker, p_z(dec <= -7) = 0.0619(27), p_z(<= -9) = 0.0116(9) (arch_tail, W 2000, 160-chain errors; lemma vs direct 8.4e-8) -> kappa_g ~ 600 / ~120 at C_g ~ C_plain; a gate deep enough for kappa ~ 5 (<= -12, p_z 5.5e-4) misses the heavy class (87% in -7..-9). Scaling makes it worse: std of log10|B|^2 1.30 (6x6, T5 table) -> 1.80 (8x8), ratio 1.38 vs sqrt(N) 1.33; extrapolated 3.15 decades at 14x14 | g sees only the deep region; heavy class 17462 training entries (1.1%) | FAIL C1 (global gate); the local log-additive form replaces it |
| local log-additive g (ln g = sum_i phi(window_i), no gate) | pass at all N: a two-swap changes only windows touching 4 sites; kappa ~ 1-2 at 8x8 (vs |G| det4 per PP endpoint) | needs H-loc: the correction's bond difference is a local function of the configuration near the bond. Untested (Test L's 2-body set is a subset) | PENDING H-loc test |

## Next (registered before any number)
1. DONE arch_tail (/scratch/a/A.Otaifi/ansatz_runs/arch_tail/summary.json): p_z(<= -7) 0.0619(27), p_z(<= -9) 0.0116(9); p_y 0.0213 / 0.0034.
   Predicted 1-5% / 0.1-1%: both slightly exceeded (two-swap shift -2.23 decades as assumed; walker tail heavier than the 450k-draw estimate).
   Rule fired (> 1e-3): global-gate hybrid fails C1 at 8x8. Tail carried by few walkers: per-walker z fraction at -7 median 0.018, 99th pct 0.58.
2. H-loc (proposed, CPU, ~2 h worker): on pool_s2b, bond-weighted held-out least squares of add_x - add_y (and of R1's residual e_x - e_y) on
   local features around the swapped bond, radius r = 1, 2, 3 (pattern counts, then a small window MLP). Prediction (mine, ~50%): explained
   share 0.2-0.4 at r = 2, because f_J's bond difference runs through the PP's Green function, which need not be short-ranged at J2 = 0.5.
   Pass: >= 0.5 on heavy bonds (hybrid's job), >= 0.35 in the bulk.
3. Tied PP: (a) exact 6x6 check that (1 + alpha(H - E)) B_{f_inv} as S_5 reproduces the d-sum guide's E_FN; (b) bare cost of a
   translation-invariant (m = 1) and 2x2 (m = 4) f vs general f at 6x6; (c) does R5's learned P_2 lie near span{B_{f1 o pi_O}}?
   Derivations go to a breaking review (agreement 6); Astra before any GPU proposal (agreement 8).

## Dead ends
(none yet)

## Cockpit
- STATUS (in progress, medium): A tied pair-product sum holds one Lanczos step exactly at constant cost; Pfaffians, backflow and the |psi|-gated CNN hybrid fail the cost or reach test.
- NEXT (in progress, medium): Test whether the round correction is local (decides the cheap local add-on) and check the tied pair-product sum exactly at 6x6.
