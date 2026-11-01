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

## R0 RESULT 2026-11-01 (job 17640186, /scratch/a/A.Otaifi/ansatz_runs/sign_r0/r0_result.json; verified by me from the json)
Exact psi0 teacher, alpha = -0.18338 fixed, E = E0, per site; controls C1, C2, C3a, C4 pass, lobpcg residuals <= 3.6e-10,
E0 <= E_FN <= E_var for every solve; C3b (ARPACK cross-check) timed out.
| net | w before | w after step | tau_bare (no step) | tau (full step) | tau_sign | tau - tau_sign | floor bound after step |
|---|---|---|---|---|---|---|---|
| gs XL | 4.37e-7 | 4.95e-9 | 6.00e-7 | 1.498e-6 | 7.4e-9 | 1.490e-6 | 2.3e-9 |
| m2 M | 5.16e-5 | 6.13e-6 | 4.88e-5 | 7.57e-5 | 6.39e-6 | 6.93e-5 | 2.5e-6 |
Predictions: XL tau 2-8e-8 MISSED (1.50e-6, 20-75x); m2 tau 1.5-6e-6 MISSED (7.6e-5); m2 magnitude share >= 0.6 HELD (0.92).
Reading: the step repairs the sign (w 4.4e-7 -> 5e-9, sign-only part 7e-9) but its magnitude at repaired configurations
costs 1.49e-6, 2.5x MORE than not stepping (6.0e-7). Named term: rho_x = 2|alpha| kappa_x - 1 (board 25). My coefficient for
it was 10-30x low: per unit repaired weight the cost is ~124 (total energy), not ~1.4-4, because the network's errors sit in the
tail at large kappa (rho >> 1) and blocked-bond cost grows as cosh(ln rho), not quadratically.
Consequence: stored sign + the loop's plain signed Lanczos step fails the 0.5e-6 bar even for the best stored sign; the
failure is the step's magnitude rule, not storage (floor after the step 2.3e-9).

## R1 (next; CPU; changes rho)
Repair-aware step: where the step flips the stored sign, take the magnitude from the local energy of the REPAIRED sign,
which is exact from data already computed: flipping only S(x) gives E_L' (x) = 2 H_xx - E_L^S(x). Guide:
psi''(x) = S_h(x)|A(x)| |1 + alpha (E_L^{S_h}(x) - E)|, S_h = S sgn(T - E_L^S), T = E + 1/|alpha| (same d^2 cost as the step).
For an isolated error with an eigenstate teacher this gives rho = 1 exactly.
- R1a (N=36 replay, as R0): predicted XL tau <= 3e-8 (pass), m2 tau 6-15e-6 (fail, unrepaired errors). [60%]
- R1b (Test X basis, competitor guide psi_P, no stored sign): does the repair-aware rule keep the competitor's one-step
  gain? Plain 5.574e-6 (control); prediction: within +-1e-6 of it [55%]. Adopt the rule only if R1b <= 5.574e-6 + 0.3e-6
  and R1a XL passes. The loop's step is the lead's call; R1 only measures.

## Running
- Astra cj-344a55c0d108 DONE (above). Board blocks 24-26 posted (v51). R0 RUNNING (lead's go under the PI's takeover, 2026-11-01): worker, run dir /scratch/a/A.Otaifi/ansatz_runs/sign_r0 (ANALYSIS_BRIEF.md, controls C1-C4). Next per verdict: as in Tests/R0.
- Input from the lead 2026-11-01: label-free loop under s_G at 6x6 reached 4.79e-6 after round 3 (4.64e-6 averaged) vs competitor 5.57e-6 (docs/reports/2026-10-31_label_free_loop_6x6.md); S2's bar: a stored sign must keep that, judged with the magnitude refit under the stored sign.
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
