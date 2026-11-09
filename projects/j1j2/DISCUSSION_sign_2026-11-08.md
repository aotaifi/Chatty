# Sign discussion with the PI, 2026-11-08 (open points; nothing here is a result)

Status key: OPEN = to discuss; RUNNING = test launched; PARKED = noted for later.

## Facts the discussion rests on
- 8x8 stored sign: best loop score (decades8 eps^2) is the old network's own sign, 0.40; trained signs 0.54-0.72 (bar 0.3). (PLAN_sign "GATE (a)")
- 6x6 exact: one final Lanczos step at c = 0 beats the competitor (5.57e-6): s_G|B_6| 2.65e-6, sgn(psi_P)|B_6| 3.71e-6; fixed own sign + Jacobi head 7.20e-6. (T-B2P1, A47)
- At s_G the step's 7.06 -> 2.65e-6 gain is almost all amplitude (sign change on 1.2e-6 of the weight).
- 6x6: the first sign step carries almost all sign gain; later steps < 0.5e-6 (A41). 8x8 first step weight 28x the 6x6 one.
- Pfaffian angle: under the net's own sign, stranded configurations (f_J < -0.5, few kept bonds) are ~100x rarer than under s_G.

## Points
P1 (OPEN) Double work. FN = projection to convergence inside the node; Lanczos = the same projection, a few steps, allowed to cross nodes.
   We keep only the sign of the Lanczos step and throw away its amplitude gain, which FN then rebuilds. Revisit how the two are combined.
P2 (RUNNING, T-GATE) Keep the old sign where the stored network is unsure, store the new sign elsewhere. Test on existing checkpoints, no training.
P3 (OPEN) Why rare configurations matter: FN and the labels use ratios psi(y)/psi(x), not psi^2; the gains chased are ~1e-6/site;
   eps^2 -> energy at 8x8 not measured.
   P3 first number (11-08, from existing runs; lead's caveat: this is the sign as the NODE on a fixed amplitude, a different quantity from the
   loop's label eps^2; record both): same amplitude |M1|, c = 0, FN mixed energy per site:
   sgn(M1) -0.4987565(28) (gfmc_m13b/prod/rb8/analysis_final.json) vs stored phase sign (ppw_s1, sign_sha256 aac1b0e6 confirmed by the lead from jobB run_info.json; gate-(a) eps^2 0.55)
   -0.498716(13) QUOTABLE (h100bench/jobB/summary_B.json, c0_M256): the stored sign is WORSE by 4.0(1.3)e-5/site (3.0 sigma).
   Direct s_G on |M1| at c = 0: timed, energy NOT measured. Reading: at 8x8 a stored sign's node cost is ~1e-5 scale, so the storage barrier is real
   for the node, not only for the labels. One point; the PP-full sign also differs from sgn(M1) in the bulk.
P4 (OPEN) Is Lanczos the only sign update? Alternatives: energy-based sign optimisation (Marshall trap at J2 = 0.5); node update from the FN state
   itself (self-healing DMC, Reboredo et al. 2009; our 6x6 V > K rule; Astra's cut certificate), which does not nest neighbour shells.
   P4 worked through (11-08): CORRECTION of the angle lead's earlier claim: a local FN-state rule evaluated on the fly also chains one layer per
   round (it needs the current sign at x's neighbours); it escapes only if each round's sign is stored (self-healing DMC refits the guide).
   Its value is a possibly better TARGET sign (from the converged FN state, not the guide). Evidence weak: A27 greedy flips 94% of K1 with exact
   amplitudes, fail at log-amplitude noise 0.3; certified hybrid gained ~1e-8 at 6x6; SHDMC tested on two electrons only, Caffarel 2026 biased
   fixed point (wiki notes, papers not read). PI 11-08: 6x6 evidence is weak either way (guide already very good there); P4 is UNDECIDED, not weak. Ranked after P6 because it does not avoid storage. P9(a) (Astra's cut certificate) is the careful version of this rule.
   P4 SETTLED with the PI (11-08): the rule flips the sign label of a visited x (not the walker): K(x) = sum of allowed terms |H_xy psi(y)/psi(x)|
   (H_xy psi_y/psi_x < 0), V(x) = sum of blocked ones (> 0); flip iff V > K. Same neighbour sums as the Lanczos sign (flip iff
   V - K > 1/|alpha| - (H_xx - E)), same cost; only the THRESHOLD differs. Justified once the amplitude is close to the FN state (after the
   amplitude loop). => P4 = alternative threshold for the same update; candidate for the FINAL sign step after the amplitude loop. Does not help
   storage or the chain.
P5 (OPEN, PI; CORRECTED by the PI 11-08) The PI's loop is: guide -> FN -> Jacobi corrections and storage -> Lanczos on the stored guide -> FN ->
   Jacobi corrections and storage -> ... (NOT 'store psi_1 jointly'; that was the angle lead's reading, kept below as P5b).
   Constraint found: Lanczos on the stored guide WITH its Jacobi head needs the head at every neighbour, i.e. signs three swaps out (d^3, lead);
   at c = 0 (bare B) it is affordable (T-B2P1). After the step the new sign sgn(psi_1) must reach the next round: stored, or recomputed (nesting).
P5b (OPEN) Angle lead's reading: store psi_1 = (1 + alpha(H - E)) psi as ONE object.
   Angle lead: this is storing psi_1 = (1 + alpha(H - E)) psi as ONE object (sign and amplitude together) = T-JOINT, and it keeps the step's
   amplitude gain (P1). Connects P1, P4, T-JOINT and the own-sign observation.
P6 (OPEN, PI) The amplitude side succeeded with a label-free target A e^{c f_J}, the Jacobi head computed at walker time from neighbours.
   Angle lead: the transferable part is "store the smooth part, compute the hard near-node part exactly at walker time". For the sign: stored (or
   old) sign in the bulk, exact one-step sign s_G computed on the fly only at stranded configurations (small SK), where it matters.
   Cost = (fraction of stranded evaluations) x 2N per evaluation; the stranded fraction is small under the net's own sign. Hypothesis, untested.
   P6 worked through (11-08): exact s_G(x) = sgn(M1)(x) sgn[1 + alpha(E_L^M1(x) - E)] costs one H row (~2N evaluations); the expense is that the
   walls need it at ~(2N)^2 configurations per move. A gate only saves if it is CHEAPER than E_L itself (the flip criterion IS E_L, so gating on
   the flip condition costs as much as computing s_G). Cheap gate candidate: the stored network's own margin |m| (small at image cancellations,
   i.e. near its node). Cost multiplier = fraction of evaluations gated. Test: T-GATE arm S' (stored where |m| >= tau, exact s_G elsewhere),
   eps^2 + gated fraction, offline on decades8.
   Multi-round cost (PI's question, 11-08): with gated fraction f, the exact sign of round k chains back only through unsure configurations:
   cost per evaluation ~ 1 + (2N f) + (2N f)^2 + ...; bounded for any number of rounds iff 2N f < 1 (8x8: f < ~0.8%; 14x14: f < ~0.25%).
   Caveat: unsure configurations may cluster (neighbours of unsure are unsure), raising the effective f. Decision numbers from T-GATE arm S':
   eps^2 < 0.3 AND f < 1/(2N).
P7 (OPEN, PI) Make the ratio explosion numerically stable (logs/exponentials instead of ratios).
   Angle lead: the code already works in logs (log|B|, f_J = ln SK - ln den); the large values are physical, not round-off. Logs avoid overflow but
   not the size of the effect. What can change it is a modelling choice: the f0 cap (already used), or computing those terms exactly (P6).
P8 (PARKED) T-JOINT, one complex network for sign and amplitude: Astra 45%, own-sign observation raises it to ~50%; PI decides.

P9 (OPEN, PI asks for an explanation) Astra's ideas the PI did not follow; to explain step by step:
   (a) the cut certificate: whether flipping the sign of a set of configurations lowers E_FN can be read from the CURRENT FN state u, through the
       bonds crossing the set's boundary (Astra cj-807a618b54f5, ASTRA_rule.md); (b) eps^2 is a weighted graph-Laplacian quadratic form in the label
       errors, so per-bond costs do not add (ASTRA_eps2.md eq 1-4). PI to confirm which one was meant.

## New ideas for a better sign every round (angle lead, 11-08; PI wants this line kept alive beside Branch 2)
I1 (RUNNING, T-GATE arm 3, PI go 11-08) Ensemble-disagreement gate (strengthens P6): two independently trained sign nets; where they DISAGREE, compute the exact one-step sign.
   Evidence: stored-sign errors are seed-specific (6x6: seeds share 0.61e-6 of 1.10e-6 / 1.96e-6 wrong weight; Opus review), so disagreement
   flags most of each net's errors, likely better than |m|. Offline test on existing pairs (T-IW g05/g10 finals; ppw_s1 + a second seed).
I2 Store the stepped wavefunction jointly (P5b / T-JOINT): node follows the amplitude, avoids stranding (Pfaffian own-sign observation). ~50%.
I3 Continuation (homotopy): smaller sign steps per round (alpha scaled down), warm-start each round's sign net from the last. A small change of
   the node is easier to track without false flips (T-SIGN8: big change -> false flips). More rounds, each cheap. ~25%.
I4 Astra's proper loss: form the full kept/blocked sums before the logs, accept by the hard-sign label metric, on a small development panel. ~20%.
I5 (from the PI's point: we learn a +-1 step function on a discrete domain) Train on the continuous signed value v(x) = 1 + alpha(E_L(x) - E)
   (already computed for the labels; its sign IS the label, its size is the distance to flipping) by regression, then take the sign.
   Soft targets / distillation: strong signal near the node, where +-1 labels carry none. Same principle as I2 (learn a smooth signed function).
   TRAP: plain squared regression on v underweights wrong signs near the node (v small: v=+0.01 vs prediction -0.01 costs 4e-4), the psi^2 mistake
   again. Use a loss that protects the sign near zero (v / local scale, or regression + BCE). ~30%; run after I1/P6 numbers set the bar.
Low: a sign cache of visited configurations (walkers rarely revisit at 8x8); released-node transients (give energy, not a stored sign).

## Astra on I1-I5 (cj-60be7531368f, /scratch/a/A.Otaifi/astra/sign_ideas/ASTRA_ideas.md, read)
Ranking: I5 (amended) > I2 > I1 > I4 > I3 (I5 30%; I2 45% one-round 6x6, 25% two stored 8x8 rounds; I1 20%; I4 15%; I3 10%).
- I5 amended: do NOT regress v = P/psi (poles at the old guide's zeros); regress the pole-free absolute target u = P/D, D = sum of absolute terms of
  P = [1 + alpha(H - E)] psi, |u| <= 1, sgn u = sgn P; loss Huber(g - u) + lambda log(1 + exp(-t g / T)); accept by hard-sign eps^2.
  Cheapest test: development-panel replay, <= 1 core-h if rows are saved.
- T-GATE discrepancy: the running worker computes only the KEEP-OLD-SIGN fallbacks; exact-s_G repair arms (P6 S', I1 exact) are not computed
  (~80-110 core-h estimated). Its result will not test exact repair.
- Multi-round cost: 2N f < 1 is an independence estimate only. Clustered unsure configurations blow up (toy: same f, adjacent unsure: 1.5e6 calls at
  depth 20 vs 1.016 separated). The relevant quantity is the spectral radius of the unsure-set adjacency across rounds.
- NEW (Astra's strongest addition): CERTIFIED PARTIAL SUMS. P_r(x) = sum_j a_j S_{r-1}(x_j). Resolve cheap children first; if their signed sum K
  and the unresolved absolute weight R satisfy |K| > R, the sign of P_r(x) is certain without resolving the rest (branch and bound). Prunes the
  recursion exactly. Test: replay exact 6x6 rows with a gate mask, count signs resolved vs all-child fallback, <= 1 core-h.

## Where supervised sign fitting stands (11-08, after T-CERT, T-SOFT1, T-ROUND3, T-GATE)
Reliable in common configurations (0 errors above decade -5); confidently wrong in rare ones (decades <= -6); soft targets do not address it.
New division (PI's amplitude idea, T-AMP running): rare configurations -> exact Lanczos sign with pruning; common configurations -> a stored sign.
At 8x8 the step barely changes common-configuration signs, so the old sign suffices there and no trained net is needed yet. Supervised fitting returns
when later rounds or larger lattices move the node into common configurations: it then stores the new sign where it is reliable.

## Order proposed
P2 runs now (cheap). Then P1 + P5 + P6 together on paper (they are one question: what to store and what to compute at walker time), then P4.

## T-NODE result (2026-11-09; for the PI, plain)
The gated exact sign S_-7 does NOT make a better node at 8x8: energy +8.0(4.1)e-6/site vs the old sign (no gain; the predicted gain >= 5e-6 is
excluded at 3.2 sigma). Likely reason: it changes the sign only in configurations where walkers spend ~1/10000 of their time, while the loop's label
score (eps^2) counts those configurations heavily. Open: whether it still helps the next round's LABELS. Astra asked (cj-9c2bd0991219) before any verdict.
Correction (same day, Astra cj-9c2bd0991219): my "walkers spend ~1/10000 of their time there" was wrong; that number counts evaluations, not
time. The energy test could see a gain of 1e-5 with ~70% chance, so the lack of gain is real for the node. Labels remain open for a different reason:
the new sign changes the loop's correction strongly where few bonds are kept. Next: a 1-core-hour pilot to price relabelling; the decisive test
(refit + grade with the new sign) costs ~5.5 H100-h plus relabelling and needs the lead's OK.
