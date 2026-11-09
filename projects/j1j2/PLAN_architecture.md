# PLAN_architecture: an amplitude network built for the Jacobi loop (angle lead, started 2026-11-09)

Brief: proposals/2026-11-05_architecture_angle_brief.md (incl. 2026-11-08 gated hybrid). Reports to the project lead ("FN-loop lead").
Paper + CPU only. No GPU, no hybrid build before the Pfaffian angle's R1 separator (a)/(b) says CLASS (b) (PLAN_pfaffian "Separator ...").

## CLOSED 2026-11-09 (lead agreed)
No candidate passes C1 and C2. The gated hybrid fails C1 (extensive gate, A1). Local add-ons fail C2 (non-local correction, A2). The tied PP
sum gives no cost gain (Astra, A3). Pfaffian sums are dominated (A4). Backflow fails C1 (measured earlier), the 2-body Jastrow C2 (Test L).
More PPs (K-PP sums, cost ~K x one PP) remain the only cheap route. Side finding handed to the Pfaffian angle: pp2_lemma dedup keys collide
above 64 sites. The 6x6 symmetric-f accuracy run (arch_sym6) was kept running at the PI's request; its result is for the record, not a gate.

## Open question (one) [closed]
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

REVIEW (Opus worker, 2026-11-09, arch_paper/review/REVIEW.md; exact 4x4, every broken variant fails): A holds (3.3e-15) and needs the
site-ordered par (Marshall parity breaks J1-v and J2: 2.0). B holds for the SUM over bonds only (per bond under projection: 1.75). 'inner' RBM
closes if (f, W) are permuted together (1.3e-14); 'outer' does not. C collapse needs ALL of: f T'-invariant, T' permutes hidden units (W, b),
T' inside the projection group. PRODUCTION rb8_prod (M = 16, W general) -> no collapse, 1 + 4N = 257 terms at 8x8. 4m is an upper bound
(point-symmetric f, full proj: 1 + 2). 6x6 trap: c4 is an odd site permutation, so a c4-invariant f gives B_full == 0. D: (1+a(H-E))^2 B
needs 8N + 21 projected PPs (149 / 309 / 533 at L = 4 / 6 / 8; rank 134 at 4x4, the missing 15 are f-independent spin-1/2 identities).
=> Tied-PP class requires building a symmetric PP (+ symmetric RBM or none); the 6x6 checks must use translation-invariant f with
translations-only or c4-safe projection.
ASTRA (cj-852432069699, 2026-11-09, /scratch/a/A.Otaifi/astra/arch_tied/ANSWER.md; job ended on a usage limit AFTER writing it):
identity with a relabelled Jastrow holds (4x4, 1e-15; broken 0.07-1.97; base coefficient is 1 + alpha(c0 - E), not 1). MY COST CLAIM WAS WRONG:
each defect term B_O = Proj[PP_{f o pi_O}] has trivial translation stabilizer, so it needs all N translation images; the 5 (17) projected terms
are pm(1+4N) primitive determinants, i.e. 257 x (8x8) / 785 x (14x14) a tied base, ~4m x a general-f projected PP. No asymptotic gain over a
general-f PP + one signed Lanczos step at equal image count; vs the naive two-level route (d^3) a bounded factor survives (Astra: not a timing).
Accuracy risk: Nomura-Imada arXiv:2005.14142 App. B (Astra-read), 8x8 J2 = 0.5 RBM+PP, a 4x4 pairing cell costs 4.26e-4/site in E_var;
Hu-Becca-Parola-Sorella 1304.2630 symmetric projected BCS 6x6 p = 0 -0.50117, p = 2 -0.503571 vs ED -0.50381 (Astra-read, not by me).
Implementation gaps found: pp2_lemma recomputes inverses per call (no persistent update); dedup keys are one uint64 -> collide above 64 sites
(10x10, 14x14). => Tied class: no cost advantage; the identity is a correctness/representation fact, not an architecture. Pending: arch_sym6 accuracy.
Only the FIRST step collapses: H acting on B_{f o pi_O} (no longer T'-invariant) gives back the full bond sum, so H^2 B needs one PP per
T'-orbit of bond PAIRS, ~16 m N / 2 (extensive). p nested steps are NOT (1+4m)^p PPs (lead's reading corrected 2026-11-09).
Literature (2026-11-09, wiki + arXiv, 6 searches): Becca, Hu, Iqbal, Parola, Poilblanc, Sorella arXiv:1412.2656 Sec. 4 (read that section):
p Lanczos steps on Gutzwiller-projected BCS states (translation-invariant pairing, i.e. m = 1) cost O(N^p) per VMC ratio on top of the
O(N^2) update, evaluated by the H sum. The PP-sum form is not stated in the parts read. So for VMC (one ratio per move) the identity saves
nothing at p <= 2. Its gain is in FN-GFMC with a Jacobi head (d ratios per move): two levels at 5 d^2 det4 instead of d^3 det6, a factor
~d/5 (~80 at 14x14). Related: Dutta-Chen-Henderson-Scuseria arXiv:2101.08911 (abstract only) write correlated AGP states as linear
combinations of non-orthogonal AGPs (chemistry, pairing Hamiltonian). Novelty of the spin-exchange instance: modest, not claimed.

## Candidates on paper
| candidate | C1 | C2 | verdict |
|---|---|---|---|
| sum of K independent PPs | pass: kappa ~ K x one PP (10.9 vs 5.5 measured); the table method (Clark/Morales; not read) needs a shared reference, absent for independent f_k: no saving | r 0.49 at K = 2 (R5), 0.25 round 6. A general-f PP's Lanczos step has PP-rank 4N+1 (identity): fixed K cannot hold repeated rounds; candidate mechanism of the r decline | pass C1; C2 partial, falls with rounds (hypothesis) |
| tied PP sum S_{1+4m}: B_f + sum_O c_O B_{f o pi_O}, f T'-invariant | WITHDRAWN (Astra): pm(1+4N) primitive dets, no gain over general f + Lanczos; was: (1+4m) x plain; a Jacobi head on top costs (1+4m) x one level, i.e. two levels at constant cost (board 20 had ~2e3 x at 14x14) | one Lanczos step EXACT (identity); Jacobi vs Lanczos differ ~8% at 6x6 (PLAN.md); round 2 needs compression (f o pi_O not invariant) | PASS on paper for one step; open: bare cost of imposing T'-invariance on f |
| local dressing of the swapped sites' pair amplitudes (backflow-type) | backflow measured 36.5x at 6x6, grows ~N^2 (board 23). Finite-pattern dressing: O(1) in N but det of ~4w rows (w ~ 9-16) -> kappa >> 30 | not reached | FAIL C1 (practical) |
| 2-body Jastrow | pass (cached fields, O(1)) | fail: 2-body features eps^2 0.973 (Test L, P7) | FAIL C2 |
| Pfaffian sums (S_z = 0 sector) | pass: rank-4 Pfaffian update, ~4 x a PP per term | triplet blocks A (up-up), D (down-down) enter only as products A^k D^k (pairs must balance), so dPf/dA = 0 at A = D = 0: zero linear reach at the PP point and a bilinear saddle for a warm-started refit (LESSONS "bilinear saddle") | FAIL: dominated by PP sums (derived, not run) |
| gated hybrid G = B exp[w(|B|^2) g], g global CNN | FAIL at 8x8: kappa_g = d^2 p_z C_g / C_plain with d^2 = 1.01e4 distinct z per walker, p_z(dec <= -7) = 0.0619(27), p_z(<= -9) = 0.0116(9) (arch_tail, W 2000, 160-chain errors; lemma vs direct 8.4e-8) -> kappa_g ~ 600 / ~120 at C_g ~ C_plain; a gate deep enough for kappa ~ 5 (<= -12, p_z 5.5e-4) misses the heavy class (87% in -7..-9). Scaling makes it worse: std of log10|B|^2 1.30 (6x6, T5 table) -> 1.80 (8x8), ratio 1.38 vs sqrt(N) 1.33; extrapolated 3.15 decades at 14x14 | g sees only the deep region; heavy class 17462 training entries (1.1%) | FAIL C1 (global gate); the local log-additive form replaces it |
| local log-additive g (ln g = sum_i phi(window_i), no gate) | pass at all N | FAIL (H-loc, arch_hloc/summary.json): held-out weighted explained share of the correction's bond difference, r = 1-3, linear and MLP: all 0.08-0.13, bulk 0.02-0.04, heavy -0.26..-0.40; R1 residual difference also non-local (all <= 0.03). Controls: shuffled ~0.00, local synthetic 1.00 / 0.96. Predicted 0.2-0.4 at r = 2: missed low | FAIL C2 |

## Next (registered before any number)
1. DONE arch_tail (/scratch/a/A.Otaifi/ansatz_runs/arch_tail/summary.json): p_z(<= -7) 0.0619(27), p_z(<= -9) 0.0116(9); p_y 0.0213 / 0.0034.
   Predicted 1-5% / 0.1-1%: both slightly exceeded (two-swap shift -2.23 decades as assumed; walker tail heavier than the 450k-draw estimate).
   Rule fired (> 1e-3): global-gate hybrid fails C1 at 8x8. Tail carried by few walkers: per-walker z fraction at -7 median 0.018, 99th pct 0.58.
2. H-loc (proposed, CPU, ~2 h worker): on pool_s2b, bond-weighted held-out least squares of add_x - add_y (and of R1's residual e_x - e_y) on
   local features around the swapped bond, radius r = 1, 2, 3 (pattern counts, then a small window MLP). Prediction (mine, ~50%): explained
   share 0.2-0.4 at r = 2, because f_J's bond difference runs through the PP's Green function, which need not be short-ranged at J2 = 0.5.
   Pass: >= 0.5 on heavy bonds (hybrid's job), >= 0.35 in the bulk.
3. Lead's order 2026-11-09: (1) breaking review of the identity FIRST (Opus worker, arch_paper/REVIEW_BRIEF.md, running; incl. the RBM
   invariance condition and the bond-pair count); (2) exact 6x6: S_5 vs the sampled p = 1 energy of the same network; bare cost of
   translation-invariant f (and RBM) vs general f; a Jacobi head on S_5 vs the competitor's two-step 1.95e-6; (c) does R5's P_2 lie near
   span{B_{f1 o pi_O}}? H-loc runs in parallel (Sonnet worker, arch_hloc/ANALYSIS_BRIEF.md, running). Astra before any GPU proposal.

## Dead ends
- Gated |B|^2 hybrid with a global CNN: C1 fails, p_z(<= -7) 0.0619(27) at 8x8 (arch_tail).
- Local log-additive add-on: C2 fails, the round correction's bond difference is not local within r <= 3 (share <= 0.13, heavy < 0; arch_hloc).
  Mechanism consistent with f_J running through the PP's Green function; unchecked: C4/spin-flip invariance of the target, a second MLP seed.

## Cockpit
- STATUS (done, low): Closed: no network beats plain pair-product sums on both cost and reach for the 14x14 loop; three proposed add-ons were ruled out before building.
- NEXT (done, low): Nothing further on this angle; the symmetric-pairing accuracy number at 6x6 is recorded when it lands.
