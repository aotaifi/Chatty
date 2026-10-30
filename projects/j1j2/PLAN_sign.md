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

## Correction and Astra (2026-10-30, cj-344a55c0d108, /scratch/a/A.Otaifi/astra/sign_angle/ASTRA_sign_design.md, read in full)
- My mechanism 2 had a factor-2 error: for an eigenstate teacher the frozen first-order cost is B_F = 4 sum_F a^2 kappa
  (= E_var[S a] - E0, exact Rayleigh bound), and deleting F gives the floor bound sum_F a^2 kappa/(1 - w) (kappa = H_xx - E0);
  my '2 phi^2 K' was neither. Re-derived the deletion bound myself (cross terms of (H - E0)psi = 0); checked Astra's three
  certificates = var_error/4 from its json. P1's band [0.5,1.2] x 2B is therefore superseded by the bounds below.
- Agreed: only boundary edges of F cost (exact operator identity, Astra eq 2); w misprices; the step repairs isolated signs
  iff 2|alpha| kappa > 1 (2.73) but leaves magnitude ratio rho = 2|alpha|kappa - 1 at the repaired configuration (exact at
  kappa = 5.45); immediate tax = floor difference + difference of amplitude excess (Astra eq 14), i.e. my G1 and G_relax.
- New from Astra: (i) floor certificates on EXISTING N=36 CNN signs (exact ED amplitude, bound (8), not measured floors):
  gs L 4.77e-7, gs XL 1.73e-7 per site bare; composite m2 M 1.66e-5 bare, 2.71e-7 after the sign-only hop. So a stored sign
  with floor < 0.5e-6 exists; the open term is the full step's rho. (ii) PLAN.md's 'each magnitude is best with the sign it
  was built with' is contradicted by Test X itself (|L| improves with s_G); told the lead. (iii) loss: edge-weighted phase
  loss c_xy ~ |H_xy| a_x a_y + small vertex anchor (eq 19); samples must keep support where the FN amplitude is ~0 (A8).
- Astra's forecast for a complex-PP phase net: immediate tax ~2e-6/site (0.2-20e-6, 25% to pass), floor difference ~0.2e-6.
  Mine (P3): 0.8e-6 central, 35%. Both below 70%: no GPU before the replay.

## Tests
- R0 (replaces my S1; Astra's design; CPU, no training, no GPU; worker after the PI's go): frozen N=36 checkpoints from
  /project/theorie/a/A.Otaifi/Chatty/runs/archive/chatty_signlearn (gs XL and composite m2 M). Teacher psi0 (exact), S =
  Marshall x sgn(logit), v = S a0 (a0 = sqrt(p0/n_orb)), v' = v + alpha(Hv - E0 v), alpha = -0.18338 fixed, E = E0.
  Readouts per site: tau = E_FN[sgn v', |v'|] - E0 (immediate tax); tau_sign = E_FN[sgn v', a0] - E0 (sign part only);
  tau_mag = tau - tau_sign; w, weighted boundary C_F before and after; floor bound (8) for sgn v'. ~1 matrix build (15-18 min)
  + ~6 FN solves (~2 min) per checkpoint; ~1.2e9 sparse entries (memory!).
  Pre-registered (lead, before any number; mechanism: repaired configurations keep rho = 2|alpha|kappa - 1, which costs on their
  blocked bonds ~ (0.1-0.3) x the bare floor scale, scaling with w):
    gs XL: tau 2-8e-8 (PASS by ~10x). m2 M: tau 1.5-6e-6 (FAIL), tau_mag >= 0.6 tau. [55%]
  Decision: XL pass and m2 fail with tau_mag dominant -> mechanism holds; the stored sign must reach w ~1e-6 against its
  teacher OR the step must not distort repaired magnitudes (next: derive a repair-aware step, lead's call since it changes
  the loop's step). XL fails -> the full step, not storage, breaks the budget even in the best case: to the lead before any
  network. m2 passes -> repair is benign; go to S2 directly.
- S2 (GPU, only after R0 and the PI's go): complex-PP phase net, S = M sgn Re(e^{-i theta0} P/|P|), independent of the
  amplitude net, on the loop teacher s' = sgn step(s1 |L| e^{4.0 f_J}); edge loss (Astra eq 19); samples tempered beta 0.3-0.5
  from the teacher + all one-swap neighbours + replay of near-zero-FN configurations; CNN as the accuracy control. Grade tau
  and the floor difference. Kill: tau >= 2e-6 converged, 2 seeds. Walker cost (board 23): ~2x the amplitude's (fast det4
  updates); conv complex RBM phase also O(1) per two-swap (Astra eq 18); generic CNN excluded at 14x14.

## Running
- Astra cj-344a55c0d108 DONE (above). Board blocks 24-26 posted (v51). R0 awaits the PI's go.
- Astra cj-34b8f6b3a4c9 stopped at the Codex usage limit 10:14; resume after 11:36
  (codex session 01a18c0d-d25b-70c1-b260-5512e8cd8aec). Partial CPU checks, /scratch/a/A.Otaifi/astra/sign_angle/check_sign_identities.json
  (read, not yet verified by me; predictions above NOT changed after seeing them):
  toy ring, equal w = 0.25: isolated error first-order 2.0, exact FN 0.586; connected 1.0, 0.268; global flip 0 (bond picture
  holds; relaxation 0.27-0.29 x first order, below my P1 band). Repair threshold K > 1/(2|alpha|) = 2.73, exact magnitude at
  K = 1/|alpha| = 5.45 (alpha -0.183). sign_learnability nets, N=36, exact amplitude, variational sign error per site:
  m2 M net 6.6e-5 -> 1.08e-6 after one sign hop; gs L 1.9e-6 -> 3.9e-9; gs XL 6.9e-7 -> 9.0e-10.
  Nothing on the cluster.

## Dead ends
(none yet)
