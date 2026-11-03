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

## R1c (added 2026-11-01 on the lead's question; pre-registered before any number): stored sign in the JACOBI loop
Deployable Jacobi guide: psi(x) = S(x) B(x) e^{c f_J[B; S](x)}, f_J with S's OWN walls; no sign step at walker time.
(a) No rho term: the magnitude never comes from |1 + alpha(E_L - E)|. (b) A walker-time sign step S_h feeding the Jacobi
walls is NOT one-level: psi(y) needs f_J(y), i.e. S_h at y's neighbours z, and S_h(z) needs E_L at z, i.e. the guide at w ~ z:
three swaps from the walker, cost d^3. (The same holds for R2G's s_G, computed from psi_P: it must be stored to be one-level.)
So the sign update happens offline (training labels), and the walker uses S directly.
Mechanism: at a wrong-sign x, S blocks the bonds that s kept, so f_J(x) = ln[K_S/(H_xx + V_S - E')] ~ ln(small/2 kappa) << 0 and
e^{c f_J} expels x; neighbours shift by c f_J(y) ~ -2c|H_xy| (a_x/a_y)/K(y), small for tail x. The guide approaches 'delete F',
whose energy is the floor bound of block 24.
Replay (N=36, exact teacher, B = a0, E' = E0, c in {2, 4, 8}; c = 0 is R0's tau_bare 6.0e-7):
  XL: tau_J(c=4) in [0.8, 2.5]e-7, i.e. near the floor bound 1.73e-7, PASS [55%]; decreasing in c from 6.0e-7 toward the bound.
  m2: tau_J(c=4) in [1, 3]e-5 (floor bound 1.66e-5), fail.
Decision: XL pass -> the Jacobi loop's storage tax at the best tested sign accuracy is below the bar; S2 then trains on s_G and
is graded in the loop's own guide (refit magnitude), bar: keep 4.79e-6 within 0.5e-6. XL fail with tau_J >> bound -> the
neighbour term (c f_J at y) is mispredicted: re-derive before S2.

## R1 RESULT 2026-11-01 (jobs 17640421, 17640422; /scratch/a/A.Otaifi/ansatz_runs/sign_r1/r1_result.json, r1b_result.json; verified by me)
Per site above E0; lobpcg residuals <= 6.7e-10, E0 <= E_FN <= E_var for all solves; plain arms reproduce R0 and Test X G (3.5e-10).
Control miss: fJ(s0) max 1.16e-7 instead of < 1e-10. My threshold was set below the stored a0's accuracy on tiny orbits
(max|f - 1| 1.3e-4 there). The effect on the energy is second order (~1e-14), so the readouts stand.
| guide | XL (w 4.4e-7) | m2 (w 5.2e-5) |
|---|---|---|
| S a0, no step (R0) | 6.00e-7 | 4.88e-5 |
| S a0 e^{c f_J[S walls]}, c = 2 / 4 / 8 (R1c) | 4.20e-7 / 3.56e-7 / 4.09e-7 | 2.69e-5 / 2.29e-5 / 3.49e-5 |
| plain signed step (R0) | 1.50e-6 | 7.57e-5 |
| repair-aware step (R1a) | 2.23e-7 | 3.45e-5 |
R1b: repair-aware step on the competitor guide 1.394e-5 vs plain 5.574e-6. It destroys the genuine flips' magnitudes
(repaired/plain magnitude 8.2 on the f < 0 orbits): REJECTED for the Lanczos loop.
Predictions: R1c XL [0.8, 2.5]e-7 MISSED (3.56e-7; still below the 5e-7 bar); m2 held; R1a both missed (XL 2.2e-7 vs <= 3e-8);
R1b missed. Reading of R1c: the cost falls from bare toward the floor bound (1.73e-7) up to c ~4, then rises at c = 8, because the
neighbour term c f_J(y) grows with c. I underestimated that term (cost at c = 4 is 2.1x the bound, not ~1x).
CLOSED (best case): in the deployable Jacobi loop (stored sign used directly, f_J with its own walls, no walker-time step,
one-level cost), a stored sign with ground-state-net accuracy costs 3.6e-7/site with an exact teacher: PASSES the 0.5e-6 bar.
Measured slope tau ~ 0.8 w (XL) / 0.44 w (m2), so the bar needs w <~ 6e-7 against the teacher.
Not shown: (1) a composite loop sign (s_G) stored to w ~6e-7 (the M net on a composite reached 5e-5; XL on a composite untested);
(2) the deployable class (the XL CNN costs ~d full evaluations per two-swap shell at 14x14); (3) a non-eigenstate teacher.
## S2 (proposal, GPU; the lead decides spend under takeover)
Train sign networks on s_G (the 6x6 loop's sign) with the edge-weighted loss (Astra eq 19), tempered samples from the teacher plus
neighbours; two classes: XL CNN (accuracy reference) and complex-PP phase (deployable). Grade: E_FN[S B_3 e^{c f_J[B_3; S]}]
vs the same with s_G (4.786e-6), c rescanned; bar +0.5e-6. Prediction (before any run): XL CNN w 2-8e-6, tax 1-4e-6 (fails)
[50%]; PP phase worse. Value check first (CPU, minutes): the same loop guide under the FREE one-level sign sgn(psi_P) sets what
storing s_G is worth; if s_G's advantage over sgn(psi_P) is < the expected tax, do not store, and use the free sign.

## V1 (lead's GO 2026-11-01, CPU; pre-registered before any number): what storing s_G is worth
Round-3 guide (theta(1200), theta_bar) as sigma B e^{c f_J[B; sigma]} with sigma in {s_G (control), sgn psi_P, s1}, c rescanned.
Run dir /scratch/a/A.Otaifi/ansatz_runs/sign_v1. Only sgn(psi_P) is one-level free: s1 is a Krylov step of psi_P, which with Jacobi
walls needs three swaps, as s_G does. Prediction: sgn psi_P 1.0-2.5e-5, s1 6-10e-6 (B was trained under s_G, so both lose more
than the sign alone) [70%]. Decision (lead's rule): store iff E(sgn psi_P) - E(s_G) > 2e-6 -> then S2 (<= 3 H100-h, lead-approved);
else the deployable loop uses sgn psi_P and S2 is not run.

## V1 RESULT 2026-11-01 (jobs 17640513-15; /scratch/a/A.Otaifi/ansatz_runs/sign_v1/v1_*.json; verified by me)
Round-3 loop guide sigma B e^{c* f_J[B; sigma]}, per site above E0 (c*): theta(1200): s_G 4.786e-6 (3.9), sgn psi_P 8.965e-6 (4.0),
s1 7.278e-6 (3.9); theta_bar: s_G 4.644e-6 (4.0), sgn psi_P 8.755e-6 (4.0), s1 7.133e-6 (3.9). Controls reproduce the readouts
(<= 5e-5 relative); orbit mismatch sgn psi_P vs s_G 0.1163 = Test X. Predictions: sgn psi_P band 1.0-2.5e-5 MISSED (8.8-9.0e-6;
I mixed bare and augmented: bare is 1.9-2.0e-5); s1 held. Decision (lead's rule): s_G's value over the free sign = 4.1-4.2e-6 > 2e-6
-> STORE; S2 approved by the lead under the takeover (<= 3 H100-h).
## S2a (RUNNING; pre-registered in /scratch/a/A.Otaifi/ansatz_runs/sign_s2/ANALYSIS_BRIEF.md before any number)
XL CNN (tested class, accuracy reference, not deployable at scale) trained on s_G with samples from the loop's own augmented guide
(not psi0, not an FN state built with S). Graded inside the loop: tax = E(S B e^{c f_J[B;S]}) - E(s_G ...), c rescanned.
Prediction: w_T 2-8e-6; tax 1-4e-6 (fails 0.5e-6) [50%]; tax < 4.1e-6 (beats the free sign) [65%].
Decision: tax <= 0.5e-6 -> next the deployable class (complex-PP phase) on the same target; 0.5e-6 < tax < 4.1e-6 -> storage pays
but is taxed: next changes the measured error term (boundary weight C_F via the edge loss, Astra eq 19); tax >= 4.1e-6 -> the loop
uses sgn psi_P and the angle closes on storage.

## S2a RESULT 2026-11-01 (Paderborn 4085740, 0.28 H100-h; CPU 17641159/60, 17641316; /scratch/a/A.Otaifi/ansatz_runs/sign_s2/; verified by me)
XL CNN trained on s_G, samples from the loop's augmented theta_bar guide. Orbit order identical to the loop tables (E_var check 4e-16).
w_T 1.07e-6 (6.5% of orbits wrong, all in the tail); C_F 1.08e-5/site. Loop guide S B e^{c* f_J[B;S]} vs s_G, per site:
theta_bar 5.068e-6 vs 4.644e-6 (tax 4.25e-7, c* 4.0); theta(1200) 5.216e-6 vs 4.786e-6 (tax 4.30e-7, c* 3.9).
PASSES the 0.5e-6 bar by 15%. SEED 2 (Paderborn 4085913, new init and samples; v2_S_seed1.json, verified): tax 4.20e-7 / 4.26e-7, w_T 1.12e-6; seeds agree to 5e-9 in energy. Predictions: w_T 2-8e-6 missed (better); tax 1-4e-6 missed (better); tax < 4.1e-6 held.
Slope tax/w_T = 0.40 (R1c XL 0.82, m2 0.44). Worker's note, not verified by me: s_G = -sgn psi0 up to weight 8.4e-7, so the
competitor's one-step sign is nearly the exact ground-state sign, and this target is ground-state-like (cf. sign_learnability gs vs m2).
The stored loop at 6x6 (one seed) is 5.07e-6, still below the competitor's 5.57e-6.
Next: (1) seed 2 DONE, agrees; (2) the deployable class (complex-PP phase, fast det4 updates),
designed on paper first; the XL CNN is the accuracy reference, not deployable at 14x14 (board 23).

## S2b (RUNNING from 2026-11-01, worker; run dir /scratch/a/A.Otaifi/ansatz_runs/sign_s2b; inside the lead's S2 budget, ~2.4 H100-h left)
Why: the Jacobi guide needs S at every two-swap configuration z (walls of f_J at each neighbour y), d^2 ~ 1.5e5 per walker move at
14x14; an XL CNN evaluation there is ~1.7e8 MAC (no cone saving: a 6-layer 3x3 cone covers the 14x14 torus), so ~3e13 MAC per move.
The deployable class needs O(1)-in-N two-swap updates (board 23; Astra eq 17-18):
 (i) complex-PP phase: S = M sgn Re(e^{-i theta0} P/|P|), P = sum over the 576 images of Pf[F(gx)], F complex, no backflow, own
     parameters (independent of the amplitude net); two-swap ratio via det4 with the cached inverse, same cost class as the amplitude.
 (ii) one-layer convolutional complex RBM phase, Im log R = sum over windows: a two-swap updates O(C k^2) hidden fields.
Training as S2a (teacher samples, beta 0.5, one-swap neighbours, logistic loss on the margin s_G(x) Re(e^{-i theta0} P)/|P|),
complex random init (Astra: a real init sits at a zero-gradient extremum). Graded as S2a.
Prediction (before any number): (i) w_T 3e-6-2e-5, tax 1-8e-6 [central 3e-6; passes 0.5e-6: 15%]; (ii) worse than (i) unless
C >= 32 channels with k = 5 [30%]. Evidence: A20 (VMC-trained PP sign, w 2.4e-5 vs exact), sign_learnability (gs at N36 to 1e-3
relative needs ~30k CNN parameters; the PP has ~2.6k).
Decision: (i) or (ii) tax <= 0.5e-6 -> the deployable stored sign exists at 6x6; report and close the angle at this size, next
is 8x8. Both tax >= 4.1e-6 -> no deployable class beats the free sign: the loop's scale claim must use sgn psi_P or a faster
sign class (name the term first). Between -> storage pays with a tax; report the trade to the lead.

## S2b RESULT 2026-11-02 (Paderborn 4086773 PP, 4086774/4086902 RBM, 0.99 H100-h; LMU 17641815, 17642239; /scratch/a/A.Otaifi/ansatz_runs/sign_s2b; verified by me)
(i) complex-PP phase, 2593 params (full f, no translation tying), Marshall explicit: w_T 2.92e-6, C_F 2.69e-5/site; loop grade
theta_bar 6.271e-6 (tax 1.63e-6), theta(1200) 6.450e-6 (tax 1.66e-6), c* 4.0. BETWEEN: fails 0.5e-6, beats the free sign (8.76e-6) by 2.5e-6,
but the stored loop (6.27e-6) is ABOVE the competitor's 5.57e-6. Held-out loss still falling 4% over the last 20% of a schedule that
reached lr 0: the measurement is not at plateau. Slope tax/w_T 0.56 (XL 0.40): the named term is w_T, the class's tail error.
(ii) conv complex RBM (k 5, C 32, 1665 params): did not train (accuracy 0.500, two runs, lr 3e-3); NOT MEASURED, not a capacity result.
Predictions: (i) tax 1-8e-6 held, pass 15% did not pass; w_T band just missed (better).
## S2b' (completing the measurement; inside S2 budget, 1.45 H100-h left; pre-registered before any number)
(i') warm continuation of the PP to a held-out plateau (change < 1% over 20% of steps), <= 0.5 H100-h: predicted tax 1.2-1.6e-6 [70%].
(ii') conv RBM at lr 3e-4, otherwise as (ii), <= 0.15 H100-h: predicted trains (accuracy > 0.9) [60%], tax 2-8e-6 if it trains.
Decision after S2b': best deployable tax <= 0.5e-6 -> pass; else the deployable stored sign costs ~1.5e-6/site at 6x6, which erases
the loop's 6x6 margin over the competitor: report to the lead with the trade, and the next term to change is the class capacity on
the tail (a fast class with more tail resolution: PP x conv-RBM phase product, both O(1) updates), proposed, not run.

## STEP-BACK 2026-11-02 (PI request via the lead; Astra cj-ffe80dbfda01, /scratch/a/A.Otaifi/astra/signstep/EVIDENCE.md, read)
Astra: continue; change the TRAINING before the class. The PP was under-trained (lr -> 0 while the held-out loss was still falling).
The loss was a uniform per-configuration BCE, not the edge loss of Astra's eq 19 (mine: S2b brief specified it that way, my error);
60% of the PP's wrong weight sits outside the training set. No representability barrier is shown.
Two tests, pre-registered here before any number:
T-OTF (lead's idea; CPU; runs first): do not store s_G; compute it at the walker's neighbours (d^2, fast det updates of psi_P), with
the Jacobi walls from the free sign sgn psi_P (no d^3). Guide = s_G B e^{c f_J[B; sgn psi_P walls]}, theta_bar and theta(1200), c rescanned.
Predictions: lead 5.0-6.0e-6; mine 5.5-7.0e-6 (central 6.2e-6: the walls of sgn psi_P differ from s_G's on 11.6% of orbits; V1 showed
the walls carry ~4e-6 between the two signs). Useful iff < 5.57e-6 (beats the competitor) at no storage.
T-EDGE (Astra rank 1; GPU <= 0.5 H100-h per arm): edge loss L = sum_pairs w_xy softplus(-4 y_x y_y cos(phi_x - phi_y)) / sum w
+ 0.1 * mean_x softplus(-4 y_x cos(phi_x - theta0)), w_xy = |H_xy| a_x a_y / q_x for x ~ q (tempered teacher, beta 0.5), y in x's shell;
refreshed teacher samples (new seed); warm start from the S2b PP; constant lr 3e-4 then a short decay. CONTROL = S2b' (i'): the same
start, loss and schedule, uniform BCE, matched budget. Predictions: Astra tax 0.4-0.9e-6; mine 0.8-1.4e-6; edge beats the control by
>= 0.2e-6 [55%]. Kill (lead's): no gain over the control, or settled tax >= 0.93e-6.

T-OTF RESULT 2026-11-02 (CPU 17642561 control, 17642562; /scratch/a/A.Otaifi/ansatz_runs/sign_otf/otf_sGsP.json; verified by me):
guide s_G B e^{c f_J[B; sgn psi_P walls]}: theta_bar 6.483e-6 (c* 3.5), theta(1200) 6.644e-6 (c* 3.4); +1.84/1.86e-6 over s_G's own walls.
Controls reproduce v1_sG. Lead's 5.0-6.0e-6 missed; mine 5.5-7.0e-6 held. Above the competitor (5.57e-6) and above the stored PP
phase (6.27e-6): on-the-fly s_G with free walls is NOT useful. The Jacobi walls must match the guide sign (Test X lesson, again).

## TESTBED QUESTION (PI 2026-11-02: agree with lead and Astra what to test storage on)
Verified by me (exact ED sign sg0, N36 data; p0-weighted wrong weight): Marshall 1.96e-2, s1 7.2e-6, s_G 8.4e-7. At 6x6 from psi_P one
signed Lanczos step already gives a near-exact sign: the loop never updates it, so 6x6-from-psi_P cannot test stored sign UPDATES.
Lead's view: 8x8 infeasible for a 0.5e-6 tax (8x8 FN p=0 2.8e-6/site per 0.4 H100-h -> 1e-7 needs ~800x); test a MOVING target at 6x6
with exact grading: 2-3 sign rounds S_{k+1} = sgn[G_k(1 + alpha_k(E_L[G_k] - E))], G_k the stored guide, each stored by the deployable
class and fed forward, from a start where one step changes the sign on a scale-like weight (weaker psi_P checkpoint or sgn psi_P;
not Marshall). Readout: does E_FN still fall per round, and does the tax accumulate. Astra cj-ed149d8d5be8 asked blind.

## DECISION (PI 2026-11-02, after lead and Astra cj-ed149d8d5be8 /scratch/a/A.Otaifi/astra/sign_target/ASTRA_target.md): MOVING-TARGET TEST at 6x6
Bar (ii): per sign round k, tau_k <= 0.25 Delta_k (Delta_k = e_keep - e_teacher, tau_k = e_stored - e_teacher, each sign with its own
walls, matched amplitude), for three useful updates; and the stored loop keeps >= 75% of the cumulative gain of the same loop without
storage. Two storage seeds. Deployable class only. FIXED AMPLITUDE first (refit only if storage passes).
STEP 0 (qualify a start; CPU; pre-registered before any number): candidate starts = existing exact tables of weaker guides in
Chatty/runs/rbmpp6 (PP-only rpTabPP, rpTabB1/B2, and A for reference). Fixed B = |psi_start|. Chain, exact, no storage:
S_0 = sgn psi_start; G_k = S_k B e^{c_k f_J[B; S_k]} (c_k by exact scan, reported as optimistic); S_{k+1} = sgn[G_k (1 + alpha_k(E_L[G_k] - E_k))],
alpha_k = Ritz one-step Lanczos as in lp6x6.py / test_x.py, E_k = E_var[G_k]. Readouts per k = 0, 1, 2: d_k = Pr_pT(S_{k+1} != S_k),
ED mismatch of S_k, e_keep = E_FN[G_k-type guide with S_k], e_teacher = same with S_{k+1}, Delta_k.
Qualifies iff for k = 0, 1, 2: d_k >= 1e-5 and Delta_k >= 0.5e-6/site.
Prediction: table A fails at k = 1 (S_1 = s_G-like, near exact; d_1 < 1e-5) [80%]; the PP-only table qualifies [55%].
If none qualifies: the 6x6 learned-guide family cannot host three useful sign updates; report to the PI with the chain numbers
(then the 8x8 snapshot test (c) is the only route).

## S2b'/T-EDGE RESULT 2026-11-02 (Paderborn 4087758, 4089690, 4087759; sign_s2b/v2_S_pp2.json, v2_S_edge.json; verified by me)
PP phase to plateau (control): tax 1.447e-6 (theta_bar) / 1.481e-6, w_T 2.57e-6. Edge loss: 1.651e-6 / 1.687e-6, w_T 2.89e-6.
T-EDGE KILL FIRES (no gain over the matched control; settled tax >= 0.93e-6). More training gained 11% (1.63 -> 1.45e-6).
Conv RBM at lr 3e-4: held-out accuracy 0.771 (< 0.9), not graded. One seed each.
## STEP-BACK 2 (2026-11-02; two failed fixes of the PP-phase storage: more training, edge loss)
1. Verdict: change the evidence, not the loss: separate representability from generalisation before any further training variant.
2. Thought: 61% of the PP's wrong weight sits on orbits it never saw, train/held-out loss 0.062/0.081, and the edge loss started
   at its own optimum (batch loss 0.0253 -> 0.0247). My guess, written before the search: generalisation in the tail, not capacity.
   It agrees with Westerhout et al. arXiv:1907.08186 (abstract only): for frustrated magnets, generalisation from a subset, not
   expressibility, is the bottleneck, and the sign is harder than the amplitude. What 6x6 hides: at 14x14 the training set is a vanishing
   fraction of the space, so generalisation is the scale question anyway.
3. Cheapest next test (FULL-SET FIT, ~0.5 H100-h): train the same PP phase on ALL 15.8M orbits (teacher-weighted, no held-out), grade w_T
   and the tax. Prediction: w_T drops to 0.5-1.2e-6 and the tax to 0.3-0.7e-6 (generalisation) [55%]; if w_T stays >= 2e-6, the PP class
   cannot represent s_G to the bar (capacity), and the next step changes the class (parity features, Astra rank 2), not the data.

## CORRECTION to STEP-BACK 2 (2026-11-02; independent Opus review /scratch/a/A.Otaifi/astra/sign_warm_opus/OPUS_warm.md)
My 'generalisation' reading was wrong: the statistic I read it from ('61% of wrong weight off the training set') is 76% for the passing
XL too, so it does not discriminate. Clean test (reviewer, CPU): error rate on own training draw vs a second draw of the same law: PP
1.60% vs 1.72%, XL 0.75% vs 1.16%. The PP UNDERFITS (capacity). The warm start W is ill-posed: psi_P's RBM is complex and carries sign;
the PP-only start is wrong on ~9e-6 of teacher weight. Rule adopted: compute any diagnostic on the passing control first.
## T-CAP (PI 'agreed go' 2026-11-02; consensus with the Opus reviewer ~75%; pre-registered before any number; <= ~5 H100-h)
Arms, each trained to the plateau rule (held-out change < 1% over the last 20% of steps, then 10% linear decay), caps from measured
step counts (1-PP plateau took ~1.0 H100-h: pp 0.59 + pp2 0.41):
 (1) 1-PP phase, NEW random seed, same history as pp + pp2 (cosine run, then warm constant-lr phase), cap 1.0 H100-h: seed scatter.
 (2) 2-PP sum phase, P = P_a + P_b (two independent complex pairings, ~5.2k params), random init, cap 2.0 H100-h: capacity.
 (3) sgn psi_P x sgn(Re(e^{-i theta0}(P_a + P_b))/|.| + kappa), kappa learned from 1.5 (starts exactly at sgn psi_P), cap 2.0 H100-h:
     the PI's step-by-step hypothesis at the same capacity as (2).
Same data, loss (uniform BCE on the margin), grading as S2b (exact FN tax inside the loop guide, c rescanned, theta_bar and theta(1200)).
Decision: scatter s = |tax(1) - 1.447e-6|. Capacity lever iff tax(2) <= tax(1) - max(0.3e-6, 2s). PI hypothesis supported iff
tax(3) <= tax(2) - max(0.3e-6, 2s). An arm that hits its cap is 'not at plateau' and its comparison is not read.
Predictions (lead, before any number): (1) 1.35-1.55e-6; (2) 1.0-1.3e-6 [55% capacity lever]; (3) within 0.3e-6 of (2) [60%].
Reviewer: T-RES 1.0-1.5e-6, <= 1.0e-6 at 20%.
Amendment (takeover, before any number): arm (3) margin normalised l = sP M (m_res + kappa)/(1 + |kappa|) so |l| <= 1 as in (2);
moved to theorie A40 (Paderborn ETA 2 days), caps scaled by the measured 2.2x A40/H100 ratio: (1) 2.2, (2)-(3) 4.5 A40-h each.

## STEP 0 RESULT 2026-11-02 (CPU 17652276-79, 17659148; /scratch/a/A.Otaifi/ansatz_runs/sign_q0/q0_*.json; verified by me)
Fixed amplitude B = |psi_start|, exact chain, no storage (E_FN - E0 per site):
| start (E_var) | k | d_k | Delta_k | ED mismatch S_k -> S_{k+1} |
|---|---|---|---|---|
| A (4.74e-5) | 0 | 1.67e-6 | 1.73e-6 | 1.10e-5 -> 7.8e-7 |
| PP (6.30e-5) | 0 | 2.16e-6 | 2.20e-6 | 2.38e-5 -> 3.8e-6 |
| B2 (1.54e-4) | 0 | 5.31e-6 | 5.27e-6 | 9.25e-5 -> 1.85e-5 |
| B1 (3.87e-4) | 0 | 1.30e-5 | 1.14e-5 | 3.12e-4 -> 6.4e-5 |
| B1 | 1 | 3.51e-5 | -4.3e-7 | 6.4e-5 -> 6.1e-5 |
Control: A's c = 0 step reproduces s_G exactly (alpha -0.18338). NO START QUALIFIES (pre-registered d_k >= 1e-5 and Delta_k >= 0.5e-6
for k = 0, 1, 2). Predictions: A fails at k = 1 MISSED (fails at k = 0 on d); PP qualifies MISSED.
Reading (CORRECTED after Astra cj-35a7f4402d6c): A, PP and B2 stopped on the d gate, so 'only the first step pays' is UNTESTED for them; the second (B1, k = 1) moves the sign on 3.5e-5 of the weight and LOSES
energy. This is A4 again (TRIED_IDEAS: a second Krylov step at fixed modulus fails; 'K1 -> FN refresh -> projected K1'): a sign step
pays only with a magnitude matched to it. So the fixed-amplitude moving-target test, as the PI and I set it up, cannot host several
useful sign updates. I should have checked A4 before proposing 'fixed amplitude first'.

## STEP 0' (AMENDMENT, takeover 2026-11-02, Astra cj-35a7f4402d6c /scratch/a/A.Otaifi/astra/sign_q0/ASTRA_q0.md, 85%; pre-registered before any number)
The original step 0 FAILED its registered gates; that stands. Amendment: the d_k >= 1e-5 gate is dropped as a qualification veto (no
conversion from vertex mass to FN energy; A gains 1.73e-6 at d 1.67e-6); Delta_k >= 0.5e-6/site for three rounds is kept and not lowered.
Start B2 only (largest first gain among the d-stopped starts). Two arms after the shared round 0:
 (a) fixed base B (as step 0);
 (b) exact refresh after each sign step: B_{k+1} = |Phi_FN[J(B_k, T_k)]| / sqrt(n_orb), one converged FN solve, J(B,S) = S B e^{c* f_J[B;S]}.
Grade each round at a COMMON amplitude: Delta_k = F_{B_k}(S_k) - F_{B_k}(T_k) (refresh lowers E_FN even at unchanged sign, so post-refresh
falls do not count). Stop an arm at the first Delta_k < 0.5e-6. Cap 48 CPU core-h. Sensitivity: the fJ-undefined rows and the 1e-15 floor.
Predictions (lead, before any number): (a) fails at k = 1 [65%]; (b) three useful updates [45%].
Decision: an arm qualifies -> propose the storage test on it to the PI at check-in (no GPU in takeover beyond the T-CAP default);
neither -> stop the qualification search, report the chain numbers; no more starts, no alpha change, no size change in takeover.

## STEP 0' RESULT 2026-11-02 (CPU 17662380/1, 27 core-h; /scratch/a/A.Otaifi/ansatz_runs/sign_q0r/q0r_{a,b}.json; verified by me)
B2 start, per site above E0, Delta graded at a common amplitude. Round 0 (shared, reproduces q0_B2): Delta_0 = +5.27e-6.
(a) fixed B, k = 1: Delta_1 = -7.6e-8 (d 4.6e-6). (b) exact refresh: refresh alone lowers F(S_1) 5.88e-5 -> 3.13e-5; then
k = 1 sign step Delta_1 = -2.39e-6 (d 1.4e-6). NEITHER QUALIFIES. Floor/fJ-row sensitivity <= 9e-13. Predictions: (a) fails at k = 1 held;
(b) three useful updates missed. Not checked: c-scan resolution at T_1 (c* moved 5.2 -> 3.0).
Reading: in this 6x6 learned-guide family (A, PP, B1, B2), ONE signed-Lanczos sign step carries all the sign gain that exists;
every second sign step measured (B1 fixed, B2 fixed, B2 refreshed) gains <= 0 while the amplitude refresh gains 2.7e-5. The 6x6 loop
therefore needs ONE stored sign (the s_G type, already shown storable by the XL CNN at 0.42e-6), not a moving one; a moving-target
storage test cannot be hosted at 6x6. Whether further sign steps pay at 14x14 is open and not testable here.
Per the pre-registered rule: qualification search STOPPED (no more starts, no alpha change, no size change in takeover).

## NEXT QUESTION AT 8x8 (Astra cj-9be89de86cff, /scratch/a/A.Otaifi/astra/sign_8x8/ASTRA_8x8.md, 90% in the sequence; awaiting the PI)
Test whether a second sign step pays at 8x8 BEFORE any storage. VMC cannot judge it (exact 6x6 counterexample, B2 first update: FN gain
+5.27e-6, VMC gain -2.04e-6 on the same guides), so FN differences are needed, which means a paired FN estimator validated against exact
6x6. Gates, each a stop point: (1) time the full 8x8 teacher (nested shells up to the 5th), <= 0.25 H100-h; (2) validate paired FN
differences at exact 6x6, <= 48 CPU core-h (+1-2 days of building the driver); (3) 8x8 pilot, <= 1.75 H100-h; (4) second-sign FN
comparison, <= 8 H100-h; (5) storage only if (4) is positive, <= 16 H100-h. Total cap 26 H100-h. Lead's view: run gates (1)-(2) only,
then decide.

## GATES 1-2 (PI 'go with what you both agree on', 2026-11-03; pre-registered before any number)
GATE 1 (teacher cost, <= 0.25 H100-h; run dir /scratch/a/A.Otaifi/ansatz_runs/sign_g1): with the M1 8x8 RBM+PP checkpoint, time complete
GFMC rows for the keep guide G1 = S1 B e^{c f_J[B;S1]} (S1 = one signed-Lanczos sign of psi_M1) and the teacher guide with T1 = sgn step(G1),
each with its own walls, on ~50 held-out roots: deduplicated base queries per row and seconds per row on one H100, and the ratio q to M1's
p=1 row cost (results/m1_8x8). Prediction: keep row 3 shells, teacher row 5 shells; q(teacher) 10-300 [60%].
Rule: q <= 1 -> gate 3 needs kappa <= 0.048; 1 < q <= 5 -> continue only if gate 2 gives kappa <= 0.048/q; q > 5 -> stop the 8x8 FN route
(needs kappa < 0.01, not credible without evidence) and report.
GATE 2 (paired FN estimator at exact 6x6, <= 48 CPU core-h; run dir /scratch/a/A.Otaifi/ansatz_runs/sign_g2): couple two FN populations
(common event randomness, coupled resampling, each arm's own rates and weights), difference over >= 4 independent pair ensembles.
Calibration set (exact differences known): B2 first update +5.27e-6; B2 fixed second -7.6e-8; B2 refreshed second -2.39e-6; S2a stored XL
vs s_G +4.25e-7; zero controls (same sign; global flip); broken control (wall term omitted) must fail.
Pass: every case within 2 SE of exact, replicated, blocking stable; report kappa = Var(paired)/Var(unpaired) at matched work.
Prediction: passes with kappa 0.05-0.3 [45%]; kappa below 0.048 [20%].
Decision after both: gate 3 (8x8 pilot, 1.75 H100-h) only if q*kappa <= 0.048; otherwise close the current investigation as
'second sign step at 8x8 unaffordable to measure with this estimator', 8x8 physics left open.

## PAPER CHECK: is our sign-update rule the wrong one? (lead, 2026-11-03; PI: at scale later sign updates must pay, so a rule that
## never pays at 6x6 is suspect; gates 1-2 PAUSED by the PI until this is settled)
Guide frame of G: E_L(x) = H_xx + V(x) - K(x), V = sum_blocked |H_xy| |G_y/G_x|, K = sum_kept |H_xy| |G_y/G_x|.
(1) Coordinate-wise FN rule (flip x alone, amplitude fixed): in the bound <a|H_FN|a> a kept bond gives -2|H|a_x a_y, a blocked one +2|H|a_x a_y,
    so flipping x lowers it iff V(x) > K(x), i.e. iff E_L(x) - H_xx > 0. Threshold: the off-diagonal local energy, no diagonal term.
(2) Our rule (sign of one signed Lanczos step): flip iff 1 + alpha(E_L - E) < 0, i.e. iff V - K > 1/|alpha| - kappa_x, kappa_x = H_xx - E.
    Its threshold is offset by theta_x = 1/|alpha| - kappa_x, which depends on x through the DIAGONAL energy (1/|alpha| ~ 11-13 here).
    Tail configurations (kappa_x > 1/|alpha|) are flipped even when V < K (wrong by (1)); low-kappa configurations are under-flipped.
    In the full Lanczos vector those wrong flips carry magnitude |1 + alpha(E_L - E)|, small near the threshold, so they are harmless;
    extracting the SIGN alone and rebuilding the magnitude with the Jacobi factor drops that protection.
(3) Mechanism for 'first step pays, second does not': the first step's flips include clear V >> K flips that pass any threshold; once
    those are done, the remaining flips are marginal, and the theta_x offset errors dominate.
Prediction (falsifiable, cheap, existing B2 tables, CPU ~2 h): (a) of the fixed-arm second step's flip weight (T_1 != S_1, p_T-weighted),
>= 50% has V - K < 0 under the parent G_1 [60%]; (b) the rule-(1) sign at the same B gains >= 0.5e-6/site where ours lost 7.6e-8 [40%;
risk: A27, greedy local flips fail under imperfect amplitudes]. Kill of the mechanism: (a) < 20%.
Re-think (lead, 2026-11-03, before Astra):
- (1) is exact as the first-order FN change only if the FN ground state u is close to the guide a. In general the per-bond weight is
  (u_x sqrt(a_y/a_x) + u_y sqrt(a_x/a_y))^2 (Astra eq 3, cj-344a55c0d108), not 4 a_x a_y. So rule (1) is right after an exact refresh
  (u ~ a) and only approximate at fixed B.
- Simultaneous flips: (1) and (2) both decide x alone; two flipped neighbours leave their shared bond unchanged, so Jacobi-style
  updates of all x at once can over-flip clusters (same for both rules).
- Prior art: A27 (greedy signed-bond flips recover 94% of the K1 gain with exact amplitudes, fail at log-amplitude noise 0.3) and A21
  (signed-bond heuristic gives 0 gain in a 20-site chamber transfer). So rule (1) depends on the amplitude's tail accuracy; the test must
  report it.
- Interpretation: (2) is the right sign for the magnitude |1 + alpha(E_L - E)| a; we pair it with a different magnitude (B e^{c f_J}).
  This is the Test X lesson again: a sign must be chosen for the magnitude it will be used with.
Lead's review (2026-11-03): the algebra holds (re-derived). Competing explanation: after step 1 the sign is near exact, so there is nothing
left to fix and any rule's errors dominate. (Against it: B1 after step 1 is 6.4e-5 from ED, 80x s_G, and its step 2 still loses.)
Lead advises against adopting V > K (greedy, single-site, tail ratios: the A27/A21 failure mode) and against pairing the Lanczos sign
with the Lanczos magnitude (that is the competitor, and R0 failed with a stored sign).
T-RULE (lead's design; pre-registered BEFORE Astra's review; CPU ~10 core-h; existing tables; run dir /scratch/a/A.Otaifi/ansatz_runs/sign_rule):
cases B1 k=1 fixed, B2 k=1 fixed, B2 k=1 refreshed. (i) Classify each step-2 flip (T != S) as toward or away from the ED sign, p_T-weighted,
histogrammed by kappa_x and by V - K under the parent. (ii) Hybrid rule: flip iff (Lanczos rule AND V > K), same magnitude; graded Delta.
Prediction (offset story): >= 60% of the away-flip weight lies in the window theta_x < V - K < 0; hybrid Delta >= 0 in all three cases [45%].
Kill: away-flip weight in the window < 30%, OR hybrid Delta < 0 in >= 2 of 3 cases -> the offset is not the cause.
Review: the derivation goes to an independent reviewer to break before the CPU check (agreement point 6).

## Running
- T-CAP on theorie A40 (cip/inter; c128, 0.657 s/step measured; /scratch/a/A.Otaifi/ansatz_runs/sign_tcap, jobids.txt): arm (1) a1pp 17669369
  RUNNING since 2026-11-03 08:34 (ETA ~12:45), then a1pp2 17669370; arms (2) sum_m4 17678709 and (3) res_m4 17678710 PENDING
  (first copies OOM on A40; resubmitted with 4-way gradient micro-batching, same batch and maths, grad diff 9.8e-15). Watchers armed (worker). Next per verdict: the pre-registered T-CAP decision rule.
- Takeover (PI away since 2026-11-02 16:50); next check-in 2026-11-04 11:00.

## Dead ends
(none yet)
