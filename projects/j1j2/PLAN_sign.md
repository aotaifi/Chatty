# PLAN_sign: sign-storage angle (angle lead, session started 2026-10-30; reports to the FN-loop lead)

## Question (one, open)
Can a second network STORE each round's sign s_{k+1} = sgn psi'_k (psi'_k = one signed Lanczos step of the guide
g_k = S_k|A_k|) well enough that the storage costs < 0.5e-6/site of exact E_FN at 6x6, at a walker cost that stays a
constant factor (board 23)? Needed because without storage round k nests k steps (d^(k+1) ratios).
Scope: supervised storage of a known target sign (labels exact at any x with one neighbour sum, so Test W's label noise
does not apply). Not mine: the amplitude network (Pfaffian angle). Passed over: stacked factor nets (A10, unlearnable),
sign from the amplitude network itself (one network two jobs; couples the angles), CNN as the deployed class (cost below).
Budget: S1 ~2 h CPU (worker); S2 <= 3 H100-h, only after S1 and the PI's go.

## Mechanism (derived 2026-10-30, lead of this angle; board blocks 24-26 when posted; Astra cj-34b8f6b3a4c9 asked blind)
Notation: s target sign, phi FN ground state at s (self-consistent magnitude), S stored sign, F = {x : S(x) != s(x)},
K(x) = sum_{y kept} |H_xy| phi(y)/phi(x), V(x) = sum_{y blocked} |H_xy| phi(y)/phi(x), u(x) = E_L^s(x) - E, alpha < 0.
1. Gauge: E_FN depends on S only through bond products S(x)S(y) (blocked set). The error object is the set W of
   bonds with S(x)S(y) != s(x)s(y); a region flipped as a whole costs only on its boundary bonds.
2. Bare cost: delta E_0 = E_FN[S,|phi|] - E_FN[s,|phi|] ~= 2B, B = sum_{(x,y) in W, kept under s} |H_xy| phi(x)phi(y)
   (isolated x: trial chi = phi with chi(x) = 0 gives 2 phi(x)^2 K(x) = 2 phi(x) sum_y |H_xy| phi(y); unrelaxed chi = S|phi|
   gives 4B). Per configuration the cost is 2 phi^2 K, and K grows in the tail, so the phi^2-weighted wrong-sign weight w
   misprices tail errors. Calibration (records): dE/w = 0.61 (Test X, s1 vs s_G at |L|), 1.09 (at |G|), ~0.7 (ViT, s_2).
3. The step repairs isolated errors: a signed Lanczos factor f = 1 + alpha(E_L - E) flips the sign where E_L - E > 1/|alpha|
   (= a Krylov step with T = E + 1/|alpha|; Test X alpha* = -0.183, 1/|alpha| = 5.5). At isolated x in F,
   E_L^S - E = u + 2(K - V), so x is restored iff 2|alpha|(K - V) > f_s = 1 + alpha u. Restored x keeps a magnitude error
   e_x = ln[(2|alpha|(K - V) - f_s)/f_s], which costs only on x's BLOCKED bonds: c_x = sum_{y blocked} 2|H_xy|phi(x)phi(y)(cosh e_x - 1)
   (board 5). Boundary (wall) configurations have only a few wrong bonds, 2|alpha| K_W < f_s, and are not restored.
   G1 = E_FN[step(S|A|)] - E_FN[step(s|A|)] ~= 2 B_wall + sum_{restored x} c_x.
4. Over rounds: s_{k+1} = Krylov-type map of S_k, contraction q ~ 0.02-0.04 per step (sign_learnability), so storage
   errors do not accumulate: steady-state tax ~ G1/(1 - q) per round (perturbed contraction).
Records it rests on: sign_learnability (composite learnable, factors not; one sign hop removes 96-99% of net error;
errors sit in the |psi|^2 < 1e-9 tail; tempered beta 0.5 + one-hop neighbours essential), A8 (self-sealing).

## Pre-registered predictions (written before any run; lead's confidence in brackets)
- P1 (injected errors, exact FN): delta E_0 / (2B) in [0.5, 1.2] for isolated AND wall errors; dE/w differs > 2x between
  the two error types at equal w. [65%]
- P2: after the step, isolated tail errors keep G1/delta E_0 <= 0.3; wall errors keep >= 0.7. [55%]
- P3 (network, S2): complex-PP phase net on s' at 6x6: delta E_0 3-10e-6 (bare fails the bar), G1 0.3-2e-6, central
  0.8e-6. Pass (G1 <= 0.5e-6) [~35%].

## Tests
- S1 (CPU, exact 6x6 tables, no training; worker after the PI's go): target s_G, magnitude |G| (Test X objects, E_FN
  5.574e-6 control). Inject (i) isolated tail errors (orbits drawn ~|G|, no two adjacent), (ii) one wall (flip the orbits
  satisfying a local predicate), each at 2B/N ~ 5e-6. Read delta E_0, G1 (step alpha* of the reference and line-searched),
  B, B_wall, w. ~8 FN solves (~2 min each) + steps.
  Added 2026-10-30 after the lead's Test X note (a sign is judged with an amplitude refitted under it): third readout
  G_relax = floor(S) - floor(s), floor = E_FN at the self-consistent magnitude under that sign (iterated exact FN, as for the
  s1 floor). The fixed-magnitude delta E_0 is the upper bound for a storage error, G_relax the lower; the loop's refit sits
  between. Prediction: G_relax/delta E_0 <= 0.5 for isolated errors (the refit can push phi(x) down at x in F), >= 0.7 for
  walls [50%]. If G_relax already passes the bar where G1 fails, the grade in S2 becomes the refitted one.
  Decision: P1 outside [0.4, 1.5] -> mechanism wrong, re-derive, no network. Isolated G1/dE_0 <= 0.3 and walls >= 0.7 ->
  S2 loss carries a bond-product term; budget B_wall/N < 0.25e-6. Isolated G1/dE_0 > 0.7 -> the step does not repair; need
  B/N < 0.25e-6 bare, beyond any tested net at 36 sites (best composite w ~1e-6..1e-5): kill S2 as specified; next is a
  sign-hop-then-Lanczos guide (same d^2 cost), a change to the loop's step that goes to the lead.
- S2 (GPU, after S1): network = complex Pfaffian pair product used for its PHASE only, S = M sgn cos(theta - theta0)
  (fast det4 updates: walker cost ~2x the amplitude's; 6x6 ~11x a plain single-network guide with one step, 14x14 ~6x).
  CNN classifier excluded at scale: d^2 full evaluations per move (~1.6e5 at 14x14), ~30-500x the amplitude.
  Target s' = sgn step(s1 |L| e^{4.0 f_J}) (the loop's current guide), magnitude matched. Samples ~|psi'|^(2 beta),
  beta 0.5, from the TARGET (never from an FN state built with S: avoids A8's self-sealing) plus all one-swap neighbours;
  loss logistic on s'(x) cos(theta - theta0) (+ bond-product term if S1 says walls dominate).
  Grade: exact G1 and delta E_0. Kill: G1 >= 2e-6 converged, 2 seeds agree.

## Running
- Astra cj-34b8f6b3a4c9 (xhigh, blind derivation of 1-4 above). Nothing on the cluster.

## Dead ends
(none yet)
