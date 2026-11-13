# PLAN_sign: sign-storage angle (angle lead, session started 2026-10-30; reports to the FN-loop lead)
Shared lessons with the lead: LESSONS.md (read before designing any test; add a row when a lesson costs a run).
Open discussion with the PI (points P1-P8, nothing settled): DISCUSSION_sign_2026-11-08.md.

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

## T-SPLIT (PI go 2026-11-03; lead's call after the project lead's and Astra's reviews; pre-registered before any number; CPU <= 8 core-h)
Astra cj-807a618b54f5 (/scratch/a/A.Otaifi/astra/sign_rule/ASTRA_rule.md): the offset algebra is right, but 'V > K lowers E_FN' is false
in general (counterexample); the certificate uses the current FN state u (cut score Q). B2 fixed: 97.7% of the second step's flip weight
has V - K < 0 (first step 44%); refreshed: 48%. The hybrid H = Lanczos AND V > K certifiably improves frozen-magnitude FN in B2 fixed,
by >= 7.8e-9/site.
Lead's judgement: B2 fixed is a near-null case (loss 7.6e-8); the real loss is B2 refreshed (-2.39e-6), so the decisive split is there.
Split (Astra eq D): loop change = [E_FN(T, a) - E_FN(S, a)] at the FROZEN parent magnitude a + [rebuild of T's Jacobi magnitude].
Solves, both B2 arms at k = 1: (1) E_FN(L, a) with L = the Lanczos sign T_1, at the parent's frozen a = B e^{c_S f_J[B;S]}
(refreshed: B = B_1, c_S = 5.2; fixed: B = B_0, c_S = 4.7); (3) the hybrid H in the deployable family: its own f_J, c in {3.0, 4.0, 4.7, 5.2}.
Prediction (refreshed arm): (1) frozen sign change in [-5e-7, +2e-7]/site, so most of the -2.39e-6 comes from the rebuild [65%];
(3) H also loses in the rebuilt family [60%].
Decision: |frozen change| < 0.5e-6 and the loop loss is large -> the REBUILD causes it: hand to the project lead (loop design), not this
angle. Frozen change <= -1e-6 -> the FLIP choice causes it: next is a sign rule matched to its magnitude (certificate Q). H gains >= 0.5e-6
in the rebuilt family -> candidate rule, to the lead.

## T-SPLIT RESULT 2026-11-03 (CPU 17683003/4, 17685127, 5.2 core-h; /scratch/a/A.Otaifi/ansatz_runs/sign_split/split_{a,b}.json, diag/; verified)
Energy change per site (positive = loss): fixed arm: frozen-magnitude sign change +6.2e-8, loop +7.6e-8 (both ~0). Refreshed arm: frozen
sign change -3.49e-7 (a GAIN), loop +2.39e-6. Prediction (1) held.
BUT the refreshed loop number is an ARTIFACT of the Jacobi family: orbit 148821 has B_1 ~ e^{-37.9} and no kept bond under S_1
(f_J undefined, set to 0); T_1 flips it, giving f_J = +16.5, so at c >= 3 that one orbit carries 1 - 1e-9 of the guide weight
(diag/collapse_T1B1.json, w_top1 0.999999999 at c = 3). The step 0' refreshed Delta_1 = -2.39e-6 and the H grade in that arm are INVALID.
Corrected reading of step 0': the second sign step gains 0.35e-6/site at frozen magnitude in the refreshed arm and ~0 at fixed B. That is
below the 0.5e-6 'useful' bar, but it is not a loss, so 'every second step loses' is withdrawn; 'gains little at 6x6' stands.
The Jacobi factor is unbounded on tiny-amplitude orbits (f_J ~ ln of a neighbour sum over a tiny B), and a sign flip there can make one
orbit dominate. This is a loop-design defect, handed to the project lead. Not checked: whether any earlier stored-sign grade (S2a, S2b, R1c)
touched such an orbit (their scans were smooth and their energies sane).

T-CAP INTERIM 2026-11-03 (verified from sign_tcap/v2_S_sum.json): arm (2) 2-PP sum, at plateau (17451 steps): tax 2.758e-6 (theta_bar),
2.806e-6 (theta(1200)); w_T 4.65e-6, C_F 3.75e-5. WORSE than the 1-PP's 1.447e-6, so the 'capacity lever' cannot fire whatever arm (1)'s
scatter is, unless s > 0.65e-6. Prediction (2) 1.0-1.3e-6 MISSED. Arms (1) a1pp2 17693076 and (3) res_m4 17678710 running on A40.

T-CAP AMENDMENT 2026-11-03 (lead, before any arm-(3) number): arm (3) as specified was frozen by construction. m_res in [-1, 1], so with
kappa > 1, sgn(m_res + kappa) = +1 everywhere; kappa rose 1.5 -> 3.38 by step 1260. My spec error (and the reviewer's design). Cancelled and
redesigned: l = sP M (1 + A m_res)/(1 + A), A = softplus(.) >= 0 from 1e-3. It starts exactly at sgn psi_P, and flips need A > 1. Prediction
and decision rule unchanged.

## RE-GRADE WITH FIXED f_J (lead's request 2026-11-03; ansatz/code/fj_fix.py mode b1: rows with no kept bond become nodes;
## c_eff = c min(1, f0/|f_J|), f0 = the 0.9999 quantile; pre-registered before any number; CPU <= 24 core-h; run dir /scratch/a/A.Otaifi/ansatz_runs/sign_fix)
Re-grade with the fixed family, at a common B, c rescanned: B1 k=0 and k=1 (fixed B); B2 k=0, k=1 fixed, k=1 refreshed (B_1 recomputed with the
fixed family). Same sign sequences T_k as before (q0_B1s, q0r); controls: the s_G guide unchanged to <= 1.2e-8 (lead's check).
Predictions: B2 fixed k=1 |Delta_1| < 0.2e-6 [75%]; B2 refreshed k=1 Delta_1 in [+0.1, +0.7]e-6 [55%]; B1 fixed k=1 Delta_1 >= 0 [50%];
first steps stay >= +1.5e-6 [80%].
Decision: every second-step Delta_1 < 0.5e-6 -> 'the second sign step gains < 0.5e-6/site at 6x6' CONFIRMED (no longer 'loses');
any Delta_1 >= 0.5e-6 -> the claim is WRONG: report the case, and a moving-target test becomes hostable at 6x6 from that start.

## FIXED-f_J RE-GRADE RESULT 2026-11-04 (CPU 17696058/9, 18.3 core-h; /scratch/a/A.Otaifi/ansatz_runs/sign_fix/fix_B{1,2}.json; verified by me)
Fixed Jacobi family (fj_fix.py b1), common B, c refined to 0.1; top-1 orbit weight <= 9.3e-3 in all grades (no collapse). Delta per site:
B1 k=0 +1.137e-5, B1 k=1 fixed -4.31e-7; B2 k=0 +5.27e-6, B2 k=1 fixed -7.6e-8, B2 k=1 refreshed +3.94e-7 (old -2.39e-6, artifact).
CONFIRMED (pre-registered rule): no second sign step reaches 0.5e-6/site at 6x6 (range -0.43 to +0.39e-6); first steps 5.3-11.4e-6.
Predictions: B2 fixed held; B2 refreshed held; B1 >= 0 missed; first steps held. Not checked: a separate c minimum below 3.

T-CAP RESULT (arms 1-2) 2026-11-04 (verified from sign_tcap/v2_S_a1.json, v2_S_sum.json):
| arm | tax theta_bar | tax theta(1200) | w_T |
|---|---|---|---|
| 1-PP seed 0 (S2b', H100) | 1.447e-6 | 1.481e-6 | 2.57e-6 |
| 1-PP seed 1 (arm 1, A40) | 0.847e-6 | 0.865e-6 | 1.72e-6 |
| 2-PP sum (arm 2) | 2.758e-6 | 2.806e-6 | 4.65e-6 |
Seed scatter s = 0.60e-6 (one pair; different hardware and phase caps). Capacity lever: NO (the sum is worse than either 1-PP seed).
Predictions: arm 1 1.35-1.55e-6 missed (0.85); arm 2 missed. Reading: at this class size the training seed moves the tax by 0.6e-6, as much
as anything else tried; a summed pairing trains worse, not better. Neither a capacity nor a loss change has beaten seed noise.
Arm 3 (residual, redesign 2): A stalls at 0.08 (cos steps 3780-4410), far below the 1 needed to flip any orbit; held-out stays exactly
sgn psi_P. Mechanism: the gradient on m is scaled by A/(1+A), and the 88% agreeing orbits pull A down, so m cannot learn the flips while
A is small: a saddle. This is A10 again: learning only the flips on top of a fixed base does not train. Second failed parametrisation of the
residual -> step-back on arm 3; recommend cancelling 17697011 (it will reproduce sgn psi_P, tax ~4.1e-6) to save ~6 A40-h (PI's call).

## Running
- T-BAD (PI GO): cost probe done (neither design fits 20 A40-h); design finalised after Astra QUESTION8 (6) (resubmit 23:45).
- TAKEOVER from 2026-11-09 19:50, EXTENDED 2026-11-10 ~01:00 (PI: "I won't be here at 9, you have the lead till I come back, together with decisions").
  Rules I set for myself: size every job from measured rates (j1j2 GPU budget is efficiency, no ceiling; theorie A40/H200, Paderborn cont; no express);
  each stage pre-registered in this file before its numbers; Astra on every fired kill; no cancels of others' jobs, deletions, paper or emails.
  T-BAD round 1 FN grades: Marshall e_0 = -0.495522(23), net n_1 e_1 = -0.498158(17), g_1 = 2.636(29)e-3 USEFUL, theorie A40, PENDING (Priority); watchers 3953856 / 3954632 (PPID 1),
  VERDICT lines in delta_today.md; gate sign_bad/r1/code/r1_analyze_fn.py. Next: g_1 = e_M - e_n1 per T-BAD FINAL; then round 2 (labels K[n_1], fit with
  the recipe chosen by the fit check, FN of n_2). Round 2: e_2 = -0.498658(9), g_2 = 5.00(19)e-4 USEFUL. Round 3: e_3 = -0.498709(10), g_3 = 5.1(13)e-5 USEFUL, gains slowing; T-CAP8 inconclusive (deeper net -2.2(14)e-5, 1.6 sigma); round 4 HELD until the paired estimator; CONTROL C DONE: loop beats SR at matched compute by 1.06e-3/site (59 sigma); storage screen done (round 4: warm start, half steps).
  FN-trained sign: T-FNI done (estimator fails on a good parent's tiny margins; bad start untested), parked behind T-BAD.
  Defaults (PI may override): CPU <= 40 core-h, theorie A40 <= 8 A40-h, no Paderborn express, no cancels/deletes/paper/emails beyond takeover rules.
- Fresh-eyes consult done (section FRESH-EYES CONSULT). T-LEV stopped (wrong regime; see T-LEV).
- Astra on the FN-trained stored sign (QUESTION8.md -> ASTRA_fnsign.md): cj-3cde43955281 failed on Codex quota; resubmitted as cj-9a275217f646 (23:45, items 1-7). Next: cheap test (<= 8 core-h
  or <= 4 A40-h) pre-registered and run if Astra rates it viable; otherwise record and bring to the PI.

## Where the angle stands (2026-11-09)
- Question now: a BETTER sign every round (PI). One round solved at 8x8 on the loop metric, every round not.
- Stored trained sign nets (PP, frozen-RBM, IW, soft target): KILLED for this recipe (A48; fresh eps^2 0.69-0.70 vs sgn(M1) 0.40; node cost of a stored
  sign on |M1| +4.0(1.3)e-5/site).
- Amplitude-gated exact sign S_-7 (sgn(M1) where decade > -7, exact s_G below): eps^2 0.191 [0.097, 0.285] on 20 fresh blocks (T-K7) = first 8x8 sign
  under the 0.3 bar, nothing stored; first-update cost 0.30 s/request (proxy), real per-move cost 2.65x the plain c = 0 step (T-NODE probe).
  Repeated rounds with exact lower levels: C3/C2 = 22 (rejected). As a NODE at c = 0 on |M1|: no gain (T-NODE, +8.0(4.1)e-6, kill fired; diagnosis + Astra pending).
- Components kept: exact interval pruning (exact, ~2x per level); decade gate. Open ideas: recursively gated lower levels (Astra: worth one capped test,
  no convergence guarantee); k = -6; T-JOINT (one complex net, PI: ask Astra only). Discussion record: DISCUSSION_sign_2026-11-08.md (P1-P9, I1-I5).
- Placement (lead): S_-7 not in the R1 headline; candidate sign for R2 labels/walls once T-NODE shows a node gain and the c > 0 wall cost is timed.

## STEP-BACK 3 (2026-11-04; deployable sign class; third failed fix; provisional until Astra cj-98ed296751e4 and the lead reply)
1. Verdict (provisional): continue on the 1-PP class, but change the lever from class/loss to the OPTIMISER; do not grade arm 3.
2. Thought: zero-compute, from existing run.json: train accuracy orders all five taxes monotonically, the passing XL control included
   (XL 0.9925 -> 0.43e-6; 1-PP s1 0.9864 -> 0.85e-6; 1-PP s0 0.9840 -> 1.45e-6; 2-PP sum 0.9793 -> 2.76e-6; residual 0.9729 -> ~4.1e-6),
   and train-heldout gaps are alike (no overfitting). The 2-PP sum CONTAINS the 1-PP yet fits the training set worse: the optimiser,
   not the class, sets where these runs land (term: optimisation gap L(theta_Adam) - min_C L). Literature (wiki only, papers not read):
   Szabo2020 'Marshall trap', VMC phase optimisation gets stuck; Westerhout 2020 sign generalisation fails in frustrated phases.
   Caveat: unweighted train accuracy is a proxy; linear XL<->s1 interpolation says the bar needs ~0.991 (derived, unvalidated).
3. Cheapest test: 1-PP seed spread, 3 new seeds x ~2.2 A40-h + 3 x ~1 h CPU grading (~7 A40-h). Prediction: spread >= 0.4e-6 [65%],
   best seed <= 0.5e-6 [25%]. Kill of 'optimisation-limited': all seeds within 0.3e-6 and train acc <= 0.988 -> representability
   floor, close the class at the bar.
Lead's view (2026-11-04, blind): optimiser limit, not representability (one tax-vs-train-error curve, tax ~ err^1.75; superset 2-PP worse);
accept ~1e-6 for now (keeps 50-70% of the 2.76e-6 loop margin; V is binding); one discriminator only: 2-PP warm-started from 1-PP seed 1,
lambda ~1e-3, P_2 small. My concern: lambda*P_2 small is the arm-3 bilinear saddle (gradient on P_2 scaled by lambda; A stalled at 0.08),
so 'flat' would not separate a local optimum from a stall. Variant: P_2 = P_1 + small noise, lambda = 1 (same start sign, O(1) gradients);
a run whose ||P_1 - P_2|| never grows is 'stalled, not read'. Decision after Astra.
Arm 3 grading: no outcome changes a decision (held-out = sgn psi_P to 4e-4 accuracy, free-sign tax known), so not run; recorded as
'did not train'.

## T-WARM (lead GO 2026-11-04, cap 4.5 A40-h; pre-registered before any number; submission waits for Astra cj-98ed296751e4)
Question: is the 1-PP seed-1 point (train acc 0.9864, tax 0.847e-6) a local optimum of the 2-PP sum class, or did the optimiser stop there?
Start: P = P_1 + P_2, P_1 = 1-PP seed 1 (sign_tcap/runs_a40s/a1_pp2_m4j), P_2 = P_1 with relative noise ||F_2 - F_1||/||F_1|| = 0.1
(lambda = 1, no small factor: escape from the symmetric point P_1 = P_2 runs on the antisymmetric curvature, ~ ln(1/noise)/|curv|).
Training: as T-CAP arm 2 (same data, uniform BCE on the margin, const lr 3e-4 phase with the plateau rule, 10% decay); grading v2.py.
Checks: step 0 tax within seed scatter of 0.847e-6 (else the start is not the 1-PP point: stop); log ||F_1 - F_2||/||F_1|| at every checkpoint.
Reads: (a) difference grows AND tax <= 0.5e-6 -> optimiser-limited, larger class worth having; (b) difference grows AND tax within
0.847e-6 +- 0.6e-6 -> local optimum of the larger class: accept ~1e-6/site, park the class; (c) tax > 1.45e-6 -> training leaves a good
point for a worse one: the optimiser, not the class, is broken (same reading as arm 2); (d) difference never exceeds 2x its start -> stalled,
decides nothing. An arm at its cap is not read.
Predictions (angle lead, before any number): (b) 55%, (a) 15%, (c) 15%, (d) 15%.

## T-COMB (REPLACES T-WARM's start, 2026-11-04, angle lead's call after the lead and Astra cj-98ed296751e4
## /scratch/a/A.Otaifi/astra/sign_class/ASTRA_class.md (read in full); pre-registered before any number; run dir /scratch/a/A.Otaifi/ansatz_runs/sign_comb)
Astra: containment P_{F,F} = 2P_F gives a constructive optimisation gap >= 0.018 in train BCE for the random-start sum (agrees with lead and me);
but nothing yet shows the best 1-PP can reach 0.5e-6. Its frozen normalised-margin combination m_t = (1-t) m_seed1 + t m_seed0, t = 0.436214
(fit on train BCE only): train BCE 0.0539 -> 0.0474, held-out 0.0732 -> 0.0624, w_T 1.715e-6 -> 1.409e-6. Deployable (two det4 PPs).
Why this start over T-WARM's noisy copy: two different trained optima, not the symmetric point P_1 = P_2, so no saddle escape is needed.
My correction: seed spread costs ~5.2 A40-h per seed (Astra, from the recorded A40 timings), not 2.2; STEP-BACK 3's '~7 A40-h' was wrong.
Step 1 (CPU 4-14 core-h): grade S_t = M sgn m_t with v2.py on theta_bar and theta(1200), own walls, c rescanned; check f_J finite rows,
top-1 orbit share, solver residuals. Prediction: tax 0.55-0.75e-6 (slope tax/w_T 0.40-0.49 x 1.41e-6) [70%]; <= 0.5e-6 [10%].
Step 2 (only if step 1 misses; <= 3 A40-h, inside the lead's 4.5 A40-h): joint warm continuation of both PPs, t fixed at start, same data and
loss, 4000 const 3e-4 + 400 decay, checkpoint chosen by held-out loss, graded as step 1. Prediction: <= 0.5e-6 on both tables [25%].
Reads: any step <= 0.5e-6 on both tables -> deployable stored sign at 6x6, report; step 2 within 0.3e-6 of step 1 -> stop: 'no passing class
found within the tested budget' (not 'none exists'), fallback = best measured deployable sign; step 2 worse than step 1 -> same stop.
Open (Astra): the 1-PP's stored loop is only 0.08e-6 below the competitor on theta_bar and 0.08e-6 above on theta(1200); the lead's '50-70% of
the margin kept' uses R6's amplitude, where no sign tax has been measured. Which amplitude the tax must be measured on goes to the lead.

## T-R6 (lead's request 2026-11-04: grade the stored sign on the amplitude the loop claim rests on; pre-registered before any number; CPU)
Lead retracted 'keeps 50-70% of the margin' (tax measured on the round-3 single-PP tables, not transferable: Test X). Grade S_a1 (1-PP seed 1),
and S_t if T-COMB beats S_a1, against s_G on two two-PP amplitude tables: R6 theta_6 (pfaffian_r6/r6/tab/tab_t6/round_1/la_table.npy, sha 6bb5918d)
and R5 pp2_s1 (pfaffian_r5/pp2_s1/tab/tab_t5/round_1/la_table.npy, sha f5d9d1aa). Grader: v2.py logic with these tables added (copy, not edit).
Controls: s_G on R6 reproduces the readout's bare 7.0612e-6 and aug 2.814e-6 (c* 3.9); on R5 s1 bare 8.3319e-6.
Readout: E_aug(S) per site above E0 vs the competitor 5.574e-6, margin in units of the fit-to-fit scatter 0.2e-6; tax = E_aug(S) - E_aug(s_G).
Bias: theta_6 was trained under s_G, so the grade favours s_G (tax is an upper estimate for a loop trained under S).
Predictions (angle lead, before any number): R6 tax 0.6-1.5e-6 [60%]; stored-sign loop below the competitor by >= 5 x 0.2e-6 [75%].

## T-COMB step 1 + T-R6 RESULT 2026-11-04 (CPU 17715330, 17715406/07, 17717616; 58 core-h, T-R6 14.9 core-h over its cap; /scratch/a/A.Otaifi/ansatz_runs/sign_comb{,/r6grade}; verified by me from the json)
Energy per site above E0, augmented loop guide, c rescanned. Controls: s_G on R6 bare 7.061207e-6, aug 2.813615e-6 (readout 2.814e-6); t=0 reproduces S_a1 exactly.
| amplitude | s_G | S_a1 (1-PP) | S_t (2 PPs mixed) | S_t vs competitor 5.574e-6 |
|---|---|---|---|---|
| theta_bar (round 3) | 4.644e-6 | 5.491e-6 (tax 0.85e-6) | 5.339e-6 (tax 0.70e-6) | -0.23e-6 |
| theta(1200) | 4.786e-6 | 5.651e-6 (0.87e-6) | 5.499e-6 (0.71e-6) | -0.08e-6 |
| R5 two-PP s1 | 3.258e-6 | 4.117e-6 (0.86e-6) | 3.974e-6 (0.72e-6) | -1.60e-6 (8.0 x 0.2e-6) |
| R6 two-PP theta_6 | 2.814e-6 | 3.697e-6 (0.88e-6) | 3.557e-6 (0.74e-6) | -2.02e-6 (10.1 x 0.2e-6) |
The sign tax is ~0.7-0.9e-6/site on all four amplitudes (it does not depend on the amplitude here). On the R6 amplitude the stored-sign
loop beats the competitor by 2.0e-6 (10 x the fit-to-fit scatter). Predictions: T-COMB 0.55-0.75e-6 held; T-R6 tax 0.6-1.5e-6 held; >= 5 x 0.2e-6
below competitor held. S_t misses 0.5e-6. No f_J bad rows for S_t (S_a1: row 0); top-1 orbit 0.93% everywhere.
Step 2 (joint warm training) NOT started: the PI's reframing (below) makes the 6x6 0.5e-6 bar the wrong target.
PI 2026-11-04 (reframing, angle lead agrees, withdrew 'fails the bar'): the bar is the sign tax against the competitor's error AT THE TARGET SIZE;
existing networks are ~100x worse at 10x10 than at 6x6, so a ~1e-6/site tax that does not grow with size is a clear win. Open: does the
per-site tax grow with L? Proposed next (awaiting the PI's yes): stored-PP wrong-sign weight w_T against the on-the-fly one-step target at 8x8 and
10x10 on guide samples (no FN, no exact), tax ~ 0.4-0.6 w_T from 6x6; a few GPU-h per size.

## T-SCALE (PI 'yes' 2026-11-04; design on paper, pre-registered before any number; Astra review of the design before steps 1-2)
Question: does the per-site tax of a cheap stored sign grow with L? (PI: at 10x10 existing networks are ~100x worse than at 6x6, so a tax that
stays ~1e-6/site is a clear win; one that grows like the competitor's error is not.)
Mechanism (6x6, measured): tax_total = N x tax_site ~= kbar w_T, w_T = target-weighted wrong-sign weight. kbar = 36 x 0.847e-6/1.715e-6 = 17.8 (S_a1),
36 x 0.696e-6/1.409e-6 = 17.8 (S_t), 14.3 (XL CNN). So tax_site ~= kbar (w_T/N). Local sign defects at constant density give w_T ~ N, i.e. a flat
per-site tax; worse generalisation at larger L (arXiv:1907.08186, abstract) gives w_T/N growing. Readout: w_T/N at L = 6 (exact 4.76e-8, S_a1), 8, 10.
Target at each L: T_L = sgn[v + alpha_L (H v - E v)], v = that size's best guide (8x8 M1 RBM+PP, 10x10 rbL10_prod RBM+PP), alpha_L from the
VMC energy minimum of the step (6x6 rule: alpha* = -0.18338). Labels exact at any x from one H row (~4N guide calls). Stored class: 1-PP complex
phase, same recipe as S_a1 (tempered beta 0.5 samples + one-swap neighbours, uniform BCE on the margin, plateau rule).
Estimator: w_T = E_{|v'|^2}[1(S != T)], by MCMC on |v'|^(2 beta) with beta 0.5 and reweighting (wrong signs sit in the tail).
Step 0 (worker; <= 1 H100-h + 10 core-h): checkpoints, label cost per sample, PP step cost at L = 8, 10, alpha_L scan.
Step 1 (CPU <= 24 core-h): estimator check at exact 6x6: sampled w_T of S_a1 = 1.715e-6 within 2 SE over >= 4 independent chains; broken control
(no reweighting) must disagree.
Step 2 (<= 10 H100-h total): train and measure at L = 8, 10.
Predictions (angle lead, before any number): w_T/N grows <= 2x from L = 6 to 10 [50%]; grows > 5x [20%].
Reads: per-site proxy kbar w_T/N at 10x10 <= 3e-6 (1% of the ~3e-4 competitor error; 10x10 RBM+PP E/N -0.497272, H7) -> cheap stored sign viable at
scale; growth as fast as the competitor's error -> not viable, report. Not covered: kbar's own L dependence (needs an FN grade at 8x8, unaffordable).

## T-SCALE AMENDMENT (angle lead, 2026-11-04, PI asked whether the prediction was checked conceptually; before any number). Steps 1-2 HELD.
It was not: the 50% prediction was a guess. Flaw found on paper: the target T_L is one global signed-Lanczos step. Its flips need
E_L(x) - E > 1/|alpha_L|. With alpha_L ~ 1/sigma_H^2 ~ 1/N and bulk fluctuations of E_L ~ sqrt(N), the flipped weight shrinks with N
(flips survive only next to the guide's nodes). Equivalently, a fixed number of Lanczos steps improves E/N less and less as N grows
(standard; Becca-Sorella book, not re-read). So the stored sign's absolute per-site tax at 10x10 would be small even if storage were poor:
a pass would be uninformative. Fix: read the RELATIVE retention r = 1 - w_T/d_L, with d_L = weight(T_L != guide sign); at 6x6 the stored PP keeps
79% of the s_G gain (Astra, theta_bar). r can fail at any size. Step 0 also reports d_L. [lead's confidence in the shrinkage argument ~70%]
Consequence for the loop (to the project lead): if one step's sign gain per site shrinks with N, the loop at 14x14 needs many sign updates,
which is the PI's premise, so the scaling question is whether storage errors accumulate across rounds (mechanism 4, contraction q).

## T-SCALE REVIEW (Astra cj-17cae01c48b2, /scratch/a/A.Otaifi/astra/sign_scale/ASTRA_scale.md, read in full; job exit 1 after the file was written)
- My calibration error: the 6x6 w_T used the round-3 Jacobi weight (sign_s2/data p0), not the Lanczos weight |v'|^2 the design samples. Corrected
  (exact, all orbits): w_G(S_a1) 9.355e-7, S_t 6.974e-7, XL 4.503e-7; kbar 32.6 / 35.9 / 33.9 (consistent across copies at 6x6).
- The w -> tax transfer is unsupported across L: a scattered wrong configuration changes O(N) edges (kbar ~ N, factor 2.78 from 6 to 10);
  C_W/N is 21-23x the tax (relaxation factor unmeasured). Tempered estimator feasible: ~5e5 independent draws for a 20% half-width.
- Astra's recommendation (85%): 8x8 first, 10x10 only if the 8x8 result and rates allow; add the boundary statistic C_W/N, two training seeds,
  the no-update control sgn(v) with missed vs false flips; pre-register a 20% half-width; 0.25 H100-h sampler pilot. FN viability needs a paired
  FN grade at size (driver not built), not w.
Angle lead's call: adopt all of it; keep the relative retention r = 1 - w_T/d_L (same idea as the no-update control) as the headline readout,
with C_W/N as the energy-relevant second readout. The test answers 'does the copy stay as good as at 6x6', NOT 'is the FN tax small at 10x10'.
Steps 1-2 re-sized from step 0's timings before any submission.

## T-LCNN (lead's precheck for the 8x8 loop build, 2026-11-04; pre-registered before training; CPU; run dir /scratch/a/A.Otaifi/ansatz_runs/sign_lcnn)
Train the sign class the 8x8 plan uses, the L CNN (sl_train.py SIZES L = 64 channels x 6 layers), on s_G at 6x6 with S2a's recipe
(same data sign_s2/data/N36, beta 0.5, one-swap neighbours, 80000 steps), 2 seeds; grade with v2r.py on R6 theta_6 and R5 pp2_s1, s_G controls as T-R6.
Smaller 'saving' variant: graded too once the lead names it.
Prediction (angle lead): R6 tax 0.5-1.0e-6 [70%] (XL 0.43e-6 on theta_bar, tax independent of amplitude in T-R6; L's floor certificate
4.8e-7 is 2.8x XL's); > 1.5e-6 [10%]. Stop rule (lead): R6 tax > 1.5e-6 stops the 8x8 build.

## T-SCALE STEP 0 RESULT 2026-11-04 (A40 17722680, 17724740, 17721088/9; 1.43 A40-h, over its 1 h; /scratch/a/A.Otaifi/ansatz_runs/sign_scale/STEP0.md; worker's numbers, spot-checks pending)
Guides: 8x8 M1 rbmpp8/rb8_prod/params_final.npz (E/N -0.498672); 10x10 rbmppL/prod/rbL10_prod/backup_paderborn/params_04150.npz (stopped run, E/N -0.497272, no eval chains).
Labels: one guide call per label via rank-2 bond ratios: 1.0 ms (8x8), 3.9 ms (10x10) on A40. alpha_8 = -0.18 (two seeds; analytic -0.171(7), -0.191(7); not quotable, 7 sweeps).
alpha_10 NOT scanned (~2 A40-h).
Flip weight d_L = P_|psi|^2(T != sgn guide) at alpha -0.18: 8x8 1/16384 (6e-5, 95% upper 2.9e-4); 10x10 4/8192 = 4.9(2.2)e-4. Not a test of my shrinkage
argument (alpha_10 unfitted; 10x10 guide is a stopped, worse checkpoint; 1 and 4 events). Taken at face value it does NOT shrink from 8 to 10.
PP sign net: 2.9 s/step (8x8), 15.9 s/step (10x10) vs 0.66 s at 6x6; s2b_train_m.py is hard-coded to 6x6 (lattice, neighbour table, ED loaders):
the data pipeline must be rebuilt from guide labels. Projected: 8x8 labels 0.4-0.6 + training 3.3-7 A40-h; 10x10 training 18-37 A40-h (over the cap).
Consequence: T-SCALE is an 8x8 test (Astra's advice holds on cost grounds too); 10x10 needs the PI's spend decision after 8x8.

## LOCAL-CNN REVIEW (PI request 2026-11-04; Opus reviewer, /scratch/a/A.Otaifi/astra/sign_localcnn/OPUS_localcnn.md; CPU only; fit3.json checked by me)
Verdict: no GPU on the shallow local CNN sign. The best convex correction from all translation-summed 3x3-patch features, added to the 1-PP margin,
moves w_T 1.724e-6 -> 1.712e-6 (-0.7%; -20% needed); on the two-PP mix 1.413e-6 -> 1.408e-6. A planted local rule is recovered (test can see it).
The PP's errors are seed-specific (seeds share 0.61e-6 of their wrong weight), not a shared local motif: this is why mixing two PPs helped.
Parity: an additive local term cannot express 'odd number of local defects'; the right form for local defects is PP x local phase (product).
Cost: L-independent but not cheaper than the PP (2-layer C16: 1.1e10 MAC/move vs PP full projection 4.7e10). Reviewer's better bet (~40%, guess):
a third frozen PP seed in the mix (~5.2 A40-h). Not checked: the product form (stored logits keep only cos of the phase).

## TAKEOVER 2026-11-04 evening (PI: 'discuss with the lead and decide; you are also taking over')
Decision (proposed to the lead, default unless it objects): 8x8 sign network = 1-PP complex phase, 2 seeds + margin mix, on T_8 (M1, alpha -0.18);
readouts per T-SCALE REVIEW. Tonight CPU: general-L pipeline + 6x6 gates G1 (labels = s_G exactly), G2 (sampled w_T(S_a1) = 9.355e-7 within a 20%
half-width; broken control), G3 (margin equality). From 11-05 ~10:00 (lead's takeover GPU default, after Astra): 8x8 labels ~0.5 A40-h, training
2 x 3-7 A40-h on A40. No H100, no 10x10 without the PI. Result that changes the plan: G1 or G2 failing (pipeline wrong) or T-LCNN R6 tax > 1.5e-6
(lead stops the 8x8 build; the PP 8x8 copy test still runs, it does not depend on the build).
Proposed to the lead for the 8x8 loop build: store the sign after every sign update and track the tax over rounds (accumulation at scale).

## T-SCALE 6x6 GATES (2026-11-04 evening; sign_scale/val6; verified from wT_table4.json, exact_S_a1.json, g3_margin.txt)
G2 (table backend, estimator only): sampled w(S_a1) 9.207e-7 [8.87, 9.54]e-7 vs exact 9.355e-7, half-width 3.6%, 4 chain sets, QUOTABLE; broken (no reweight) 6.6e-4.
Exact 6x6 no-update control (Lanczos measure): d_6 = 5.853e-6 (step flip weight), stored S_a1 missed 3.67e-7 + false 5.68e-7 -> retention r_6 = 1 - w/d = 0.84.
G3 margin: identical to s2b_train_m.py (max rel 0; broken 0.30). G1, G3 training, G2 guide backend: running (new worker).
Paderborn H100 queue ~11-21: all GPU on theorie A40; 0 H100-h used.
Added before any 8x8 number: MATCHED-BUDGET 6x6 reference (same pipeline, same 8x8 schedule 2000 cosine + cap 4500 steps, 2 seeds), because 6x6's
r 0.84 came from ~25k steps and 8x8 cannot afford that (2.9 s/step). Read: r_8 within 0.1 of matched r_6 -> copy holds; lower by > 0.1 -> degrades.

## T-LCNN RESULT 2026-11-04 (A40 17747340/44 train+eval, CPU grade 17747342/46; 0.8 A40-h; sign_lcnn/seed{0,1}_gpu/v2_S.json; verified by me)
L CNN (64 ch x 6 layers) on s_G at 6x6, two seeds: R6 tax 1.081e-6 / 1.077e-6, R5 1.065e-6 / 1.059e-6; w_T 2.17e-6 / 2.15e-6; no f_J bad rows.
Stop rule (> 1.5e-6) does NOT fire: the lead's 8x8 build continues. Prediction 0.5-1.0e-6 MISSED (higher). The L CNN is worse than the deployable
1-PP (0.88e-6) and the two-PP mix (0.74e-6) on the same grade. top1 (fixed rerun 17749149/50): orbit 15802042, 0.933% share, top-10 4.17%, = the bulk orbit of all earlier grades (no concentration).

## T-DEEP (lead's lever 2026-11-05; pre-registered before any number)
Lead (Pfaffian angle 9d37e93): at 8x8 the loop's LABELS use the stored sign. At 6x6, S_a1 shifts the round correction by eps^2 0.17, all in the deep
decades (-15 0.61, -14 0.50, -13 0.31, < 0.005 above -8); S_a1 disagrees with s_G on 6.1/2.1/0.35% of orbits at -15/-14/-13. Pass bar at 8x8: eps^2 < 0.3
on decades8's 6332 draws.
Record against loss reweighting: T-EDGE (bond-weighted loss) worsened the tax (1.65 vs 1.45e-6). Untested lever: sample COVERAGE of the deep decades.
(1) 8x8 PP pair (beta 0.5) = baseline; eps^2 measured when it exists. (2) 6x6 coverage arm: 1-PP on a beta mixture {0.5, 0.25}, same pipeline and
schedule as the matched-budget 6x6 reference; grade: R6 tax and eps^2 by decade vs S_a1's 0.17. ~1 A40-h + CPU.
Predictions: -15/-14 disagreement halves [50%]; tax changes < 0.15e-6 [60%]. (3) 8x8 deep arm only if (1) misses 0.3 and (2) cuts eps^2 >= 30%.

## T-SCALE 8x8 FIRST ATTEMPT 2026-11-05 (A40 17749271/72; sign_scale/r8/pp_s*/progress.txt; read by me)
Gates: G1 PASS (0 label differences vs s_G on 665,928 configs; broken alpha=0 9784); G2 guide backend PASS (w 9.07e-7 [7.6, 10.5]e-7 vs exact 9.355e-7,
half-width 16%; d 5.97e-6; r 0.848; one chain set warm-started from table_s13); G3 PASS (200 CPU steps).
8x8 random-init 1-PP does NOT train: held-out acc 0.502 / 0.510 at cosine step 2000 and 0.503 at const step 3000 (BCE 1.41 -> 1.41); the same code and
schedule leave the random start at 6x6 between steps 500 and 1000. Not a symmetry mismatch (Marshall x target invariant, +1.000 translations/flip).
Hypothesis (not derived): the random start is a plateau whose escape time grows with N and |G| (1024 images at 8x8 vs 576). Cancelled (~7 A40-h saved).
Change (before any number): WARM START from the guide's own pairing (M1's PP factor + 1e-2 imaginary part), constant lr 3e-4 + plateau rule, cap 4500;
step 0 = the no-update sign (checked). Matched 6x6 reference with the same warm start (table A parent's PP). One 8x8 seed first. Cap 20 A40-h.
This is also how deployment would start each round. Prediction: trains, and r_8 within 0.1 of matched warm r_6 [45%].

## T-HYB (lead's design 2026-11-05; pre-registered before any number; CPU; run dir /scratch/a/A.Otaifi/ansatz_runs/sign_hyb)
Lead: no stored Jacobi head exists (Theta v2, Test N, M-R failed); walls need S two swaps away, d^2 cost stands. Hybrid: accurate sign at the node,
cheap PP sign in the walls: G = S_node(y) B(y) exp(c f_J[B; S_wall](y)); cost ~2N S_node evaluations + d^2 PP lemma-path evaluations per move.
Grades on R6 theta_6 (and R5), c rescanned, tax vs s_G 2.8136e-6:
 H1 node S_XL, walls S_a1 (the candidate); H2 node s_G, walls S_a1 (pure wall cost); C1 S_XL/S_XL (XL on R6, new); C2 S_a1/S_a1 (must reproduce 0.883e-6);
 C0 s_G/s_G (2.8136e-6).
Prediction: wall cost scales A36's 1.85e-6 (sgn psi_P walls, 11.6% orbit disagreement with s_G) by the |H|-weighted wall disagreement ratio
D(S_a1, S_XL)/D(sP, s_G), computed exactly from the tables before grading. Angle lead's prior: H1 tax 0.5-0.9e-6 [60%]; pass (< 0.6e-6) [35%];
C1 0.40-0.50e-6 [70%].
Pass: H1 tax < 0.6e-6 -> 14x14 sign = CNN/ViT at the node + PP walls. Else the PP-only class stands.

## T-DEEP RESULT 2026-11-05 (A40 17750067/79/145/146, 3.03 A40-h; sign_deep/grade/v2_S_{A,B}.json verified by me; storedsign6 controls pass)
INCONCLUSIVE on coverage: arm B (beta {0.5, 0.25}) stalled at chance through the cosine phase (held-out 0.53 at step 2000; A 0.975 at step 1000) and hit
the 4500 cap still improving. R6 tax A 2.32e-6, B 23.1e-6, S_a1 0.88e-6; eps^2 (old) A 0.473, B 7.76, S_a1 0.171.
Two facts: (1) the 4500-step budget costs 2.6x in tax at 6x6 (A 2.32 vs S_a1 0.88, same data class, ~25k steps): under-training dominates; (2) the lead's
eps^2 tracks the tax: tax/eps^2 = 5.1, 4.9, 3.0 (x1e-6) for S_a1, A, B, so eps^2 on decade draws is a usable cheap proxy at 8x8.
The random-init stall now appears at 6x6 too, with deeper data (B), not only at 8x8: hypothesis (not derived): harder (deeper) label sets lengthen the
random-start plateau. Next (pre-registered before any number): T-DEEP2, warm continuation of the converged S_a1 parameters on A-data vs B-data, const lr 3e-4,
4000 steps each, same seed: isolates coverage without the init stall. Predictions: B2 lowers -15/-14 disagreement vs A2 by >= 30% [50%]; R6 tax
difference |B2 - A2| < 0.15e-6 [55%]. Cap 2 A40-h.

## T-HYB RESULT 2026-11-05 (CPU 17751313-17, ~40 core-h; sign_hyb/{C0,C1,C2,H1,H2}.json verified by me)
R6 tax vs s_G (node/walls): C1 XL/XL 0.439e-6; C2 S_a1/S_a1 0.883e-6 (reproduces T-R6 exactly); H1 XL/S_a1 0.998e-6; H2 s_G/S_a1 0.351e-6.
FAILS the bar (< 0.6e-6): the hybrid is no better than the PP alone (0.998 vs 0.883). The PP walls cost 0.35-0.56e-6 even under an exact node sign,
and wall damage adds to the node tax (H1 ~ C1 + 0.56). The PP's 0.88e-6 splits as ~0.35 walls + ~0.53 node. D-scaling of A36 over-predicted by 1.5-2x.
Reading: the walls must be as accurate as the node (A36 again); a cheap wall sign under an accurate node sign does not pay. PP-only class stands; best deployable
remains the two-PP mix (0.74e-6). Predictions: C1 held; H1 0.5-0.9 missed (higher); pass 35% did not pass.

## T-SIGN8 (critical path, lead 2026-11-05: production 8x8 grading NO-GO until the trained 8x8 sign passes eps^2 < 0.3 on decades8's 6332 draws;
## Astra cj-6a20013aa7ba /scratch/a/A.Otaifi/astra/v_go/REVIEW.md; pre-registered before any number)
Class: a copy of the M1 guide (RBM x PP, M1's own projection) with the RBM parameters FROZEN and the pairing f trainable as a complex phase:
m(x) = Re(e^{-i theta0} Psi(x))/|Psi(x)|, Psi = M1 with f -> f (1 + i 1e-2 xi). Step 0 = sgn(M1) exactly on every sample (gate; fixes the 1344/20000
mismatch of the PP-only start). Deployable: the guide class itself. Train on T_8 data (r8/d81, d82) with const lr 3e-4 and the plateau rule, cap 20000
steps (A40, 'as needed'); checkpoint every 1000 steps. Hand-off: the plateau checkpoint's sign to the Pfaffian angle for eps^2 on decades8 draws.
Prediction: plateau before 12000 steps [55%]; eps^2 < 0.3 [45%] (6x6 S_a1 0.17 at ~25k steps; matched-budget 6x6 arm A 0.47).
Coefficient note (lead): the w -> tax proxy coefficient is kbar = 32.6 (Lanczos measure, Astra), superseding 17.8 in T-SCALE above.

## T-SIGN8 GATE 0 FAIL and fix (2026-11-05; r8f/gate0/gate0.json read by me; 0.09 A40-h)
f -> f(1 + i 1e-2 xi) flips the step-0 sign on 2.08% of held-out configs (5963/287306); eps = 0 gives 0 (model correct). Broken controls reproduce
5.5% (RBM dropped) and 6.7% (RBM dropped + full projection).
Equation: M1's projected Psi = sum_g R(gx) det f(gx) is REAL. With f(1 + i eps xi): Psi = Psi_R + i eps Psi_1 + eps^2 Psi_2 + O(eps^3), so
Re Psi = Psi_R + eps^2 Re Psi_2: the sign flips wherever |Psi_R| < eps^2 |Psi_2|, i.e. at the image-cancellation configs next to the nodes (2%).
And the normalised margin m = Re(e^{-i theta0} Psi)/|Psi| = +-1 has zero gradient at real Psi (d cos/dphi = 0 at phi = 0, pi). The two needs
(exact start, nonzero gradient) conflict for ANY eps in this margin: the term at fault is the |Psi| normalisation.
Fix (changes that term): keep f REAL and use the cancellation-ratio margin m = Psi(x)/A(x), A(x) = sum_g |R(gx) det f(gx)| > 0, |m| <= 1.
sgn m = sgn Psi exactly at step 0; dm/df != 0 at a real f; |m| is small exactly where images cancel, near the nodes. Loss softplus(-s y m) with
a trainable scale s (init 4). Deployable: same evaluations as the guide plus the |terms| sum.
Gate 0 unchanged (0 mismatches). Prediction: Gate 0 passes [95%]; training leaves step 0 within 500 steps [70%].

## T-DEEP2 RESULT 2026-11-05 (A40 17751657/58, CPU 17751659/60; 1.8 A40-h; sign_deep2/code/tdeep2_table.py re-run by me, controls PASS)
Warm continuation of S_a1 (4000 steps): R6 tax S_a1 0.883 / A2 (beta 0.5) 1.148 / B2 (beta mix) 0.999e-6; eps^2 old 0.171 / 0.201 / 0.187;
-15/-14 disagreement B2/A2 0.97/0.98. Coverage lever: NO (eps^2 cut 7%, needed 30%). No 8x8 deep arm. Predictions: >= 30% cut missed; |dtax| < 0.15e-6 borderline.
New fact: continuing S_a1 RAISES held-out accuracy (0.979 -> 0.990, on the pipeline's held-out set, not S_a1's own) while the tax RISES (+0.27e-6):
equal-or-better classification, worse tax. That is the loss-objective mismatch the lead named (uniform BCE does not track the FN tax at this level).
Third proxy point: tax/eps^2 = 5.1 (S_a1), 5.7 (A2), 5.3 (B2) x1e-6 per unit eps^2.
Consequence for T-SIGN8: select and report checkpoints by eps^2 / w_T on separate draws, not by held-out BCE.

## T-SIGN8 RESULT 2026-11-05 (A40 17753590, 17754056; 0.62 A40-h; r8f/HANDOFF.md, grade/dwT_*.json read by me)
Gate 0 PASS (frozen RBM + real pairing, margin Psi/sum|t_g|: 0/287306 mismatches; broken controls 5.1% / 6.7%). Plateau at step 9000.
No trained checkpoint beats the no-update sign: w - d (paired) = +6.6e-5 (s3000), +4.7e-5 (s6000), +4.5e-5 (s9000), +0.9 [-0.9, +2.6]e-5 (final);
d = 1.63e-4 (NOT quotable, chains 2.0 sigma apart). Training fixes ~72% of the missed flips (1.63e-4 -> 4.6e-5) but adds false flips 1.26-1.78e-4.
Held-out BCE fell 0.249 -> 0.149 meanwhile (BCE does not track w, as in T-DEEP2). Predictions: plateau < 12000 held; eps^2 pending (Pfaffian angle, 5 checkpoints).
Note: d_8 = 1.6e-4 is 28x d_6 (5.85e-6): the one-step sign change carries far more weight at 8x8, so storing it matters more there.
Labels used alpha -0.18, E -31.915008; the loop/decades8 use alpha -0.17543, E -31.915136 (target mismatch, size unmeasured).

STEP-BACK 4 (third failed 8x8 sign version: random init stuck; PP-only start wrong; frozen-RBM start adds false flips)
1. Verdict: continue, changing the one term the equation names, then stop. 2. Thought: the graded quantity is w = E_{p1}[1(S != T)], p1 ~ |v'|^2,
but the loss is uniform over samples from q ~ |v'|^(2 beta), beta 0.5, i.e. it weights each configuration by q/p1 ~ |v'|^-1 relative to the grade:
the tail is over-weighted by orders and the bulk, where false flips cost, under-weighted. This is covariate shift; the standard fix is importance-weighted
ERM with a flattening power gamma (Shimodaira 2000, J. Stat. Plan. Inference 90, 227: weights (p/q)^gamma, 0 <= gamma <= 1; from memory, not re-read).
T-EDGE (bond weights on the same samples, 6x6) is a different weighting and does not cover this. 3. Cheapest test (T-IW): from the same step-0 start,
two arms gamma = 0.5 and 1.0, omega_i = (|v'(x_i)|^(2-2 beta))^gamma normalised, clipped at the 99.9% quantile; labels regenerated at the loop's
alpha -0.17543, E -31.915136; same schedule; grade w - d paired, missed/false by |v'| decile. ~1.5 A40-h.
Predictions: false-flip weight of the unweighted final checkpoint sits in the upper half of |v'|^2 [65%]; one arm gives w - d < 0 beyond its 95% interval [40%].
Kill: neither arm below d -> this class + BCE cannot beat the free no-update sign at 8x8 at this data size; report to the lead (8x8 then uses sgn(M1)
or the exact step with its cost).

## 8x8 PP-FULL WARM RESULT 2026-11-05 (A40 17750218 train, CPU 17751790 grade; sign_scale/r8/gradew/wT_ppw_s1.json verified by me)
1-PP complex phase, FULL space-group projection (1024 images), warm-started from M1's pairing without the RBM factor, 4500 steps (cap, not plateau), alpha -0.18 labels:
w_T 5.87e-5 [5.23, 6.50]e-5 vs no-update d 1.63e-4 [1.47, 1.79]e-4 -> r_8 = 0.64; missed 2.22e-5, false 3.65e-5. QUOTABLE (2 chains).
Matched 6x6 (same class and start, 4500 steps): w 2.89e-6 vs d 5.97e-6 -> r_6 = 0.52. Copy quality does NOT degrade from 6x6 to 8x8 at matched budget
(r_8 - r_6 = +0.12 +- 0.03, errors treat w and d as independent). The warm starts differ (8x8 step 0 r = 0.33, 6x6 step 0 r = -0.41), one seed each.
Contrast: the frozen-RBM M1 copy (T-SIGN8, translation projection, exact start) could not beat d. Hypothesis (not derived): the full point-group projection
is the inductive bias that generalises to unseen configurations; the frozen-RBM model's held-out BCE fell while its fresh-chain w rose.
This is the 8x8 stored sign to hand to the Pfaffian angle (export in r8iw/HANDOFF_ppw.md). T-IW extended: PP-full arms gamma 0 / 0.5 / 1 on the loop's alpha/E labels.

## GATE (a) on ppw_s1 (Pfaffian angle, 2026-11-05; PLAN_pfaffian.md "Result gate (a) on ppw_s1"): FAIL
decades8 eps^2 (bond metric with (Delta*_x - Delta*_y)^2, half B): ppw_s1 0.539(44); its PP-full step 0 0.568(32); frozen sgn(M1) 0.400(10); bar 0.3.
w_T fell 3x but eps^2 did not move: 80% of eps^2 comes from bonds the sign wrongly UNBLOCKS (~0.4% of bond weight). w_T is the wrong target metric.
My error: I ranked ppw_s1 ahead of the r8f checkpoints by w_T; the PP-full class starts at 0.57 (RBM sign dropped), r8f starts at sgn(M1) = 0.40.
Next (my call): gate (a) on r8f checkpoints (alpha -0.18) and on T-IW frozen-RBM gamma arms (matched alpha); PP-full gamma arms cancelled.
If none passes 0.3 on B: bond-weighted loss on the loop's measure (|H_xy| B_x B_y (Delta*_x - Delta*_y)^2, unblock errors separately), scored by
split-half eps^2, after an Astra review. If that fails: recommend the lead's Branch 2 (fixed sgn(M1)).

## T-BOND (joint design, lead + angle lead, 2026-11-05, PI: "work together on this one"; pre-registered before any number; Astra review after 15:00)
Lead's lessons (amplitude side): gains came from target and class, not weighting; never weight per configuration by psi^2 (|H| a_x a_y is psi^2-type:
why T-EDGE failed); draw equally per decade (decade -13 band weighs ~120x its |A|^2 share); production loss ansatz/code/pool_fit.py l.11-22.
Cost of one sign error (lead, from fj_fix.py l.14-21; re-derived by me): f_J(x) = ln SK_x - ln den_x, SK = sum_kept |H| r, den = diag - E' + sum_blocked |H| r.
Misclassifying bond (x,y) moves h = |H_xy| |B_y/B_x| between SK and den:
  wrong UNBLOCK: Df_J(x) = ln(1 + h/SK_x) - ln(1 - h/den_x);  wrong BLOCK: Df_J(x) = ln(1 - h/SK_x) - ln(1 + h/den_x).
den_x >~ 9 but SK_x is tiny next to the node, so unblocking a large-r bond there is catastrophic: this derives '80% of eps^2 from unblocking on 0.4%
of bond weight'. w_T (per configuration) and |H| a_x a_y (per bond, psi^2-type) both miss it.
Design: frozen-RBM model at exactly sgn(M1) (eps^2 0.400; B half 0.406). Data: pool8 x-draws (sign-free, rungs beta 1/.5/.25, -7..-9 tail expansion),
ALL ~2N bonds of each x, labels t_xy = s_G(x) s_G(y) from M1 consts (alpha -0.1754318714549749, E -31.915135844116154). Loss: BCE on the bond
product m_x m_y vs t_xy, weight w_xy = |c Df_J(x)|^2 for the error direction (f0-capped as the labels: n-weighted signed q.9999, fj_fix mode 'fix'),
x-end only (y symmetrised by drawing it as x), times the decade band weight. Bond-ESS per decade gate on the sign pool. ~2e5 x: labels ~7 A40-h.
Gate S (angle lead's addition, before training, CPU): on the 6332 decades8 draws (both ends stored), the surrogate W = sum over wrong bonds of w_xy for
sgn(M1), ppw_s1 step 0 and ppw_s1 must order like their eps^2 (0.400 < 0.539 < 0.568) and W/eps^2 must agree across the three within 30%.
Fails -> the weight is not the loop's cost; no training.
PASS: half-B eps^2 < 0.3 for the best-on-A checkpoint, both seeds. KILL: half-A eps^2 not below 0.37 within ~3000 steps in either seed (AMENDED after T-IW, before any T-BOND number: at the plateau or 15000 steps).
Ruled out up front (lead's list): psi^2 or per-configuration weights; walker-count labels; selection on the gate half; parameter-space gates;
f_J non-finite rows -> node; one constant set; exact warm start; strict optimiser acceptance; no 6x6 coefficients; two seeds before any pass.
Predictions (angle lead): Gate S passes [55%]; PASS [30%]; KILL fires [40%].

## T-IW RESULT 2026-11-05 (A40 17754337-39; ~2.7 A40-h; r8iw/HANDOFF.md, train_g0*/grade/dwT_*.json verified by me)
Frozen-RBM class (start = sgn(M1), Gate 0 0/287306), matched labels (alpha -0.17543, E -31.915136; 2640/1.51M training labels changed vs -0.18),
importance-weighted BCE omega = |v'|^((2-2 beta) gamma): paired w - d at the final (cap + decay, step 19998): gamma 0.5 -4.40 [-5.60, -3.21]e-5,
gamma 1.0 -2.16 [-3.31, -1.00]e-5; d = 1.51e-4 (not quotable, chains 2 sigma). Both arms are WORSE than d at steps 3000 and 6000
(gamma 0.5 +7.3e-5, +3.6e-5) and cross below d only after ~15000 steps. Not at plateau. One seed each. Prediction 'one arm below d' [40%] held.
Weighting moved the false flips out of the top |v'|^2 deciles (8-9: 1.75e-5/1.57e-5 -> 8.3e-6/1.8e-6), as the covariate-shift term predicted.
eps^2 (the deciding metric) not yet measured: checkpoints handed to the Pfaffian angle (r8iw/HANDOFF.md).
Consequence for T-BOND (amendment before any T-BOND number): the trajectory gets worse before better, so a KILL at ~3000 steps could fire on a run
that would pass; KILL moved to the plateau or 15000 steps, whichever first.

## ASTRA on T-BOND and T-JOINT (cj-a2cbf28c7311, /scratch/a/A.Otaifi/astra/sign_eps2/ASTRA_eps2.md, read in full; audit.py reproduces the saved eps^2 to 1e-12)
- T-BOND REJECTED as written (not run). The single-edge Delta f_J formula is right, but eps^2 is a weighted graph-Laplacian quadratic form in the label
  errors, eps^2 ~ eta^T L_mu eta (eq 1): endpoint cross terms; a configuration flip changes f_J at x AND all its neighbours; errors at one x add INSIDE
  the logs (A_x = sum sigma_e h_xe), not as squares (counterexample: block + unblock at one x costs 0, the surrogate charges 0.08). Multiple wrong bonds
  per root are the rule (5085/5724 M1 roots). Saved decomposition: x-end square 0.07-0.09 of eps^2 0.40-0.57; the y end carries most.
- Gate S is not a validation: an incomplete diagonal surrogate passes its ordering and 30% tests on these files (false positive shown). Gate S stopped.
  Partial Gate S numbers (worker, from the Pfaffian angle's per-file outputs; sign_bond/gateS/dev_on_reference_outputs/): x-end W = 4.87(92)e-4 (M1),
  4.44(76)e-4 (step 0), 6.7(3.4)e-4 (ppw_s1): wrong ordering, W/eps^2 spread 1.56x -> the x-end surrogate fails Gate S anyway, consistent with Astra.
- Consistent eps^2 (full / A / B): sgn(M1) 0.400 / 0.394 / 0.406; ppw_s1 0.551 / 0.564 / 0.539; PP step 0 0.568 / 0.543 / 0.593. Half B is now a reused benchmark.
- Recommendation (Astra): finish the A/B eps^2 grades of the T-IW exports (~18-20 core-h, no new labels); continue the lead's Branch 2 meanwhile.
  P(this PP/frozen-RBM family reaches eps^2 < 0.3 on honest held-out data in one practical two-seed attempt, corrected objective) 20% (10-35%);
  T-BOND unchanged ~10%. Any new loss must form the full kept/blocked sums before the logs and be checked against the hard-sign evaluator directionally.
- T-JOINT: worth one bounded 6x6 comparison after fixing definitions (own sign defines own walls; no literal complex regression to a real target;
  cross-grade amplitude/sign to separate losses); P(joint <= separate) 45%. No topological reason that a sign change forces small |A|.
My call: agree. T-BOND not trained; T-IW eps^2 grades are the next number (with the Pfaffian angle); recommend Branch 2 to the lead now; T-JOINT is
the PI's to schedule.

## T-B2P1 (lead's request 2026-11-05: 6x6 analogue of the Branch-2 headline; pre-registered before any solve; CPU; run dir /scratch/a/A.Otaifi/ansatz_runs/sign_b2p1)
Guide psi_1 = (1 + alpha (E_L - E)) S |B_k|, E_L of S|B_k|, alpha from the moment formula (Astra eq) or a line search as for the competitor;
exact 6x6 FN with psi_1's own sign and magnitude, c = 0. Arms: (a) S = s_G on B_6 (R6 theta_6) and B_5 (R5 pp2_s1); (b) S = sgn(psi_P) on B_6.
Bar: competitor 5.574e-6 (psi_P + p=1 + FN). Controls: S = sgn(psi_P), B = |psi_P| reproduces the competitor's 5.574e-6; alpha = 0 reproduces
bare E_FN[S, |B_k|] (R6 s_G bare 7.0612e-6).
Predictions (angle lead): (a) B_6 3.5-5.0e-6, beats the bar [65%]; (b) 4.0-6.5e-6, beats the bar [45%]. (Lead: (a) 60%, (b) weaker.)
MECHANISM (written after launch, before any number; PI: think first): both factors are one relaxation step of the FN equation (H_FN - E) psi = 0.
Jacobi: e^{f_J} = SK/(H_xx - E' + V), i.e. the step preconditioned by the local diagonal D(x) = H_xx + V(x) - E'. Lanczos: 1 + alpha (E_L - E) =
1 + |alpha| (K - V - H_xx + E), the same step with ONE constant preconditioner |alpha|. Next to the node V(x) and D(x) vary by large factors, so the
constant-alpha step cannot adapt there and should recover only part of the bare -> augmented gain (7.06 -> 2.81e-6). The Lanczos step also flips the
sign where 1 + alpha(E_L - E) < 0: on s_G that is a second sign step (<= 0.5e-6 at 6x6, A41); on sgn(psi_P) it is the first (~4e-6).
Revised predictions: (a1) 4.0-6.0e-6, beats the bar [50%] (was 65%); (b) beats the bar [40%].
Design point the equations show: at a FIXED sign, Branch 2 can keep the Jacobi head at walker time (the walls need the fixed sign two swaps out, which a
lemma-path evaluator of sgn(M1) gives cheaply); the final Lanczos step at c = 0 must DROP the head (c > 0 with a stepped sign is d^3). So Branch 2 must
choose between (i) fixed own sign + Jacobi head and (ii) one Lanczos sign step without the head. Arm (c) added: S = sgn(psi_P) on B_6 with the head.
Read: (b) beats the bar -> Branch 2's 8x8 headline has a 6x6 precedent; (a) beats but (b) not -> the gain needs the stepped sign inside the loop
(stored s_G), and Branch 2 at fixed own sign loses it; neither -> the loop's 6x6 advantage lives in the augmentation, which the c = 0 final step drops.

## T-B2P1 RESULT 2026-11-05 (CPU 21 jobs, 11.9 core-h; sign_b2p1/*.json verified by me; controls: Test X 5.573833e-6 rel 2e-10, alpha=0 7.0612e-6)
Exact 6x6 FN per site above E0, one signed Lanczos step on S|B_k| at c = 0, moment-formula alpha (= line-search optimum within 0.02e-6):
| arm | E_FN | vs competitor 5.574e-6 |
|---|---|---|
| (a1) s_G, B_6 + step | 2.646e-6 | -2.93e-6 (also below the Jacobi-augmented 2.814e-6) |
| (a2) s_G, B_5 + step | 3.211e-6 | -2.36e-6 |
| (b) sgn(psi_P), B_6 + step | 3.705e-6 | -1.87e-6 |
| (c) sgn(psi_P), B_6 + Jacobi head (c* 4.0), no step | 7.202e-6 | +1.63e-6 |
MY MECHANISM WAS WRONG: I predicted one constant-alpha Lanczos step recovers only part of the Jacobi gain (a1 4.0-6.0e-6); it recovers MORE
(2.65 vs 2.81e-6). At s_G the step changes the sign on only 1.2e-6 of the weight, so nearly all of the 7.06 -> 2.65 gain is magnitude. The diagonal
preconditioning argument does not hold here; not yet understood why (candidate: alpha is variationally optimal for the step, c = 4 on f_J is a fixed
heuristic power).
Branch 2 consequence: a 6x6 precedent exists for 'fixed sign + amplitude loop + one final Lanczos step at c = 0' (b: 1.87e-6 below the competitor),
and the alternative 'fixed own sign + Jacobi head' (c) is WORSE than the competitor. So Branch 2 must end with the Lanczos step, not the head.
Caveats: one network per table; B_6 was trained under s_G (helps (a), and (b) uses that amplitude under another sign); exact tables, no sampling.

## GATE (a) on T-IW finals (Pfaffian angle 2026-11-05 16:05; gate_a/round2/{g05_19998,g10_19998}; step-0 control reproduces sgn(M1) 0.4002 exactly)
eps^2 (full, -2..-9): gamma 0.5 final 0.642(23), gamma 1.0 final 0.724(37), gamma 0.5 s6000 0.941(47); all WORSE than sgn(M1) 0.400. The w_T gains
(w - d -4.4e-5) do not carry to the loop metric. Branch 1 (stored sign in the loop) is NOT reopened by this class; the lead started Branch 2.
Angle status: storing the 8x8 sign with the deployable PP/frozen-RBM class does not reach the loop's bar (best eps^2 = the no-update sign itself).
What the angle delivers instead: T-B2P1 (one final Lanczos step at c = 0 beats the competitor at fixed own sign at 6x6; the head at fixed own sign does not).

## T-CERT and T-SOFT1 (PI go 2026-11-08: cheap checks with kill tests; a fired kill is diagnosed (budget, tweak) and sent to Astra before any
## verdict; pre-registered before any number; both run on the 8x8 decades8 draws (s_G at x and y, ln|B_y/B_x| stored), not 6x6)
T-CERT (Astra's certified partial sums, ASTRA_ideas.md eq 8): P(x) = (1 + alpha(H_xx - E)) S(x)|B(x)| + alpha sum_y H_xy S(y)|B(y)| with S = s_G (round-1
sign), coefficients from |M1| ratios. A child y is CHEAP if the stored net (T-IW g05 final) is confident there (|m_y| >= tau), else EXPENSIVE. Add cheap
children (using the stored sign), order expensive ones by |a_j|, resolve until |K| > R. Report per x: certified with 0 expensive resolves (fraction),
mean expensive resolves vs all expensive children, and the certification error rate vs the exact sgn P (cheap children may be wrong).
Controls: all-exact signs -> 0 errors; deliberately underestimated R -> errors appear. Taus 0.01/0.03/0.1/0.3.
Kill: mean resolves >= 50% of the expensive children on rows with any expensive child, at every tau. Prediction: passes at some tau [45%].
T-SOFT1 (I5 amended, information check only, no training): on the same draws compute u(x) = P(x)/D(x) (D = sum of absolute terms of P) and v = P/psi.
Question: does small |u| mark the bonds that carry eps^2 (the costly wrongly-unblocked bonds of ppw_s1 / T-IW g05)? Report the share of the eps^2
numerator on bonds whose endpoint |u| is in the lowest 1/5/10%, the ROC AUC of |u| for 'costly bond', and the pole fraction of v (|v| > 10).
Kill: AUC < 0.6 for u AND for v. Prediction: AUC >= 0.7 [55%]. Pass -> stage 2 (training BCE vs Huber + sign term, two seeds, ~3-4 A40-h + grades).

## T-CERT RESULT (first pass) 2026-11-08 (CPU ~3 core-h; sign_cert/full.json, 6332 decades8 draws; numbers from the worker, read)
Pruning WORKS: resolves / expensive children 0.02 / 0.04 / 0.11 / 0.25 at tau 0.01 / 0.03 / 0.1 / 0.3 (kill 0.5 does not fire); with NO stored
signs at all the certificate alone resolves 0.48 of the children. BUT the certified sign is wrong on 8.9 / 8.6 / 7.5 / 3.7% of draws, all in decades
-7 and below (4-30%), because cheap children are confidently wrong (stored sign wrong on 10% of children with |m| >= 0.1). Order does not matter;
sgn(M1) as the cheap sign is worse. Not usable as is. Diagnosis (PI rule) running: V1 cheap = confident AND g05/g10 agree; V2 safety margin from
p_wrong(|m|) calibrated on A, evaluated on B; V1+V2. Then to Astra before any verdict.

## T-CERT VARIANTS 2026-11-08 (CPU ~4.5 core-h; sign_cert/cert2.json; calibration half A, evaluation half B, 3160 draws; worker's numbers, read)
V1 (agreement gate) does NOT help: g05 and g10 agree on 94% of children and are wrong together (13.3% wrong given agreement vs 14.6% overall):
at 8x8 the two nets' errors are SHARED (unlike the 6x6 seed-specific finding), which weakens I1.
V2 (safety margin from p_wrong(|m|)) WORKS for the error: 2x margin 0.54-0.82% at tau <= 0.1; 3x margin 0.28-0.38%; 3x + p_wrong per decade group
0.16-0.22%; with 11-20% of all ~140 children resolved (27-43% in decades -12..-9, ~0 above -5). Small counts (4-12 error rows of 3160).
MY CRITERION WAS WRONG for many rounds: each resolved child needs the previous round's exact sign, i.e. recursion. Bounded cost over rounds needs
mean resolves per row < 1 in the recursion's own distribution, not a fraction < 0.5. Here mean resolves ~ 0.15 x 140 ~ 21 per row -> ~21^k instead
of 140^k: a big saving for 2-3 rounds (e.g. two sign rounds ~6x cheaper than full nesting), not unbounded rounds. To Astra (PI rule) before verdict.

## ASTRA on T-CERT (cj-54e43141d188, /scratch/a/A.Otaifi/astra/sign_ideas/ASTRA_cert.md, read; replay reproduces all 112 cert2 summaries)
CONTINUE narrowly. Pruning screen PASSES; not yet a certified bounded-cost every-round updater: V2's margin is a smoothed expectation, not a bound
(one-child counterexample); B is now development data (thresholds compared on it); the replay gave the exact SELF sign sx for free, which is itself a
previous-round request (a self chain alone prevents bounded cost); deep decades -12..-9 need ~51-56 children per row. 21^k is a conditional forecast.
I1 drops to last (shared errors). Ranking now I5 > I2 > I4 > I3 > I1.
Next (T-TRACE, frozen V2deck3 tau 0.01; pre-registered before any number): real two-sign-round call trace on 32 decade-stratified roots + the 5 failure
rows, previous-step signs from the actual evaluator incl. the self dependency; compare unpruned / exact interval pruning / V2: raw and memoized calls,
CPU time, sign differences. <= 1 core-h. Exact pruning must equal unpruned (else fix code); if V2/exact pruning does not cut evaluator cost -> stop
this accelerator; if it does -> next gate is hard-sign eps^2 and a matched-cost refreshed-round comparison on fresh blocks.

## T-TRACE RESULT 2026-11-08 (CPU ~0.3 core-h; sign_cert/trace/{trace.json,analysis.json}; worker's numbers, read)
Real two-sign-round evaluator (ELnp2 s_G reproduces all stored labels). 32 decade-stratified roots (deep-heavy) + 5 failure rows:
(a) unpruned 4342 round-1 calls, 198 s; (b) EXACT interval pruning 2140 calls, 99.5 s, signs identical to (a) on 37/37 (broken-bound controls differ
on 8-21 roots); (c) V2deck3 1087 calls + gate evals, 62 s, wrong only on the 5 known failure rows. Split: decades >= -8: (c) 15 calls vs (b) 1066;
decades <= -9: (c) 1040 vs (b) 1042 (stored signs unreliable there). Self chain costs 1 call per round (no branching). Oracle ordering would save <= 1/3 of (b).
Reading: exact pruning is a free, exact 2x per level; stored signs add large savings in the bulk only. The walker-weighted cost (deep decades are
rarely requested) is NOT measured and decides the practical value. To Astra with that question.

## ASTRA on T-TRACE (cj-4a0834d51330, /scratch/a/A.Otaifi/astra/sign_ideas/ASTRA_trace.md, read; replay reproduces analysis.json)
CONTINUE. Exact pruning PASSES as a two-round accelerator (2.0x CPU). Weighted by a REQUEST proxy (MBAR weight x R_B, one-swap requests), the hybrid's
mean resolved children per request is 0.59-0.79 (< 1), vs 21 on the decade-stratified panel; roots weighted by |B|^2: 0.038. Forecast at 8x8: one
extra sign round costs ~1.5-3x the previous round's sign work (proxy, conditional). Not passed as an every-round updater: bulk substitution errors remain,
gate evaluation dominates time (a shared shell evaluation could remove it), third-round composition unmeasured. Scaling with N: no constant factor shown.
Next (T-ROUND3, pre-registered before any number): third sign round on request-proxy-sampled configurations (MBAR x R_B) + the 2 surviving failure rows
as stress cases; exact pruning vs hybrid, explicit self calls, cache counted; record CPU, gate cost, unique (round, config) evaluations, hard children
that produce hard children. Criterion (Astra's): request-weighted third/second-round CPU ratio <= 3, not dominated by a few hard branches.
If it fails: diagnose (gate cost vs hard-descendant expansion) before changing the policy, then Astra. If it passes: next gate = hard-sign eps^2 < 0.3.

## T-SOFT1 RESULT 2026-11-08 (sign_soft1/{soft1_result.json,yend_result.json}; eps^2 reproduced 0.5514 / 0.6416; worker's numbers, read)
KILL FIRES on the registered (unweighted) reading: AUC of small |u| at x for costly bonds 0.545 (ppw_s1) / 0.557 (g05); v 0.501 / 0.540; per decade
-9/-8 ~0.44 (chance or below). Weighted pooled u 0.66 / 0.63 is carried by the decade mix, not supported per decade. Diagnosis: at the OTHER end y the
signal is REVERSED: LARGE |u_y| marks costly bonds (AUC of +|u_y| ~0.61-0.69; 1185 sampled bonds, not importance-weighted; may be a decade proxy).
Reading (angle lead): the costly errors are NOT where the step's target is near zero (ambiguous); they are where it is clearly nonzero and the stored
net is confidently wrong (cf. T-CERT: 10% wrong among confident children). So I5's mechanism (soft targets inform the boundary) does not address the
costly errors; their cause is generalisation/coverage in the deep decades. Not yet a verdict: to Astra (PI rule).

## T-ROUND3 RESULT 2026-11-08 (CPU 0.69 core-h; sign_cert/round3/analysis3.json; request PROXY MBAR x R_B; worker's numbers, read)
Third/second-round CPU ratio, request-weighted (bootstrap 16-84%): exact pruning 40.3 (38-42); hybrid with recursive fallback 4.25 (3.3-5.3);
hybrid with exact fallback 6.0 (4.8-7.3). Criterion <= 3 FAILS in every mode (measured, not inconclusive).
Causes: exact pruning = hard-descendant expansion (21-93 resolved children, each a full round-2 evaluation). Hybrid: bulk (97% of requests) ratio 2.15 =
pure GATE cost (removable in principle by a shared shell evaluation, Astra); the excess comes from mid decades -8..-6 (3.2% of requests, 50% of round-3
CPU): 2 of 3 mid requests resolve 43/52 children, 35/50 of which resolve their own. Dominated by a few hard branches (only 3 mid samples).
Signs: hybrids agree with exact on 15/15 proxy requests. Caveats: proxy, not an FN request stream; g05 stands in for a round-2 net; no amplitude refresh.
Angle lead's reading: vs the original ~140x per extra sign round, ~4x per round is a large reduction, but not bounded; the mid-decade hard branches are
the term to change (a better stored sign there, or a shared-shell gate). Not a verdict: to Astra with T-SOFT1 at 00:10 (quota).

## T-GATE RESULT 2026-11-08 (job 17843726, ~40 core-h; sign_gate/result.json; controls reproduce 0.6416 / 0.7244 / 0.4002; worker's numbers, read)
Keep-old-sign hybrid (stored g05 where |m| >= tau, sgn(M1) elsewhere): eps^2 falls MONOTONICALLY as the stored sign is used less: tau 0 0.642, 0.01 0.588,
0.03 0.523, 0.1 0.422, 0.3 0.397(10), sgn(M1) 0.400(10). Best on A (tau 0.3) vs sgn(M1): -0.0035 +- 0.0020 (1.7 sigma, not beyond error); B -0.0022 +- 0.0021.
g10 same (no tau beats sgn(M1)). I1keep (disagreement -> sgn(M1)) 0.569. Disagreement covers only 21% of g05's eps^2 error.
Reading: on the loop metric the stored sign adds no value over sgn(M1) ANYWHERE, even where it is confident; its confident changes are net harmful
(consistent with T-CERT and T-SOFT1: confidently wrong). Exact-s_G repair arms (S', I1 exact) not computed: 17-450 core-h (control 1000 core-h).
Prediction (best tau beats sgn(M1)) [30%] missed.

## T-AMP (PI's idea 2026-11-08: the amplitude decides what is hard; pre-registered before any number; CPU)
S_k(x) = sgn(M1)(x) if decade(x) > k, else exact s_G (Lanczos one-step sign; with exact interval pruning for cost). k in {-4,-5,-6,-7,-8,-9}.
(1) Accuracy: decades8 eps^2 (A/B/full) of S_k; exact s_G needed only at deep configurations (x, y, two-swap z) -> cost only there.
(2) Cost: T-ROUND3 machinery with the decade gate instead of the confidence gate: request-weighted third/second-round CPU ratio, breakdown.
Pass: some k with eps^2 < 0.3 on A (then B) AND ratio <= 3. Predictions: eps^2 < 0.3 at k >= -7 [65%]; ratio <= 3 at that k [35%].
If a kill fires: diagnose (deep children of shallow roots, -8..-6 hard branches), variants, then Astra.

## T-AMP PARTIAL 2026-11-08 23:51 (sign_amp/{parent_result.json,count_deep.log}; ~15 CPU-min; worker's numbers, read)
Pre-registered S_k not run (exact s_G at ~20k two-swap z per x: 330-3400 core-h offline). Zero-cost VARIANT (z inherits its parent's gate):
eps^2 full k = -4 0.012, -5 0.048, -6 0.100(9), -7 0.197(12), -8 0.329(12), -9 0.380(10); A and B agree; controls sgn(M1) 0.4002, exact 0 pass.
sgn(M1)'s eps^2 sits mostly in decades -3..-7 (not only the deepest). eps^2 < 0.3 for k >= -7 (variant). Cost part not run. My brief swapped the two
controls' k labels (worker ran both correctly). To Astra with the cost question.

## ASTRA combined review (cj-abd055cffe3d, /scratch/a/A.Otaifi/astra/sign_ideas/ASTRA_soft1.md, read)
MY ERROR: T-SOFT1's spec built the target with psi = s_G|B|, i.e. P_2 (the NEXT round's numerator), not P_1 = [1 + alpha(H - E)](sgn(M1)|B|), whose sign
s_G the nets copy (u_2 disagrees with s_G on 1604/6332 roots; u_1 on 0). Corrected (no new labels): AUC of small |u_1| at y for costly bonds 0.785 (ppw_s1)
/ 0.768 (g05). The I5 kill is WITHDRAWN, and so is my reading 'costly errors are confident-wrong, not borderline': small target magnitude IS enriched on
costly bonds (student confidence and teacher margin are different quantities).
Hybrid (confidence gate): FAILS the every-round cost screen; exact pruning kept as a component (H3r ~25x faster than E3 on the proxy).
T-AMP: the parent variant's 0.197 is NOT the eps^2 of S_-7; its bias is uncontrolled (also affects f_J(x)); k = -7 is a candidate, not a pass.
ONE next step (Astra; T-SOFT2, pre-registered before any number): corrected I5 stage 2: matched BCE vs bounded-numerator Huber + explicit sign term,
both storing sgn P_1 = s_G, same frozen-RBM class, exact warm start at sgn(M1), same label constants, coverage and optimizer budget, two seeds per arm;
select on development data, judge full hard-sign eps^2 on FRESH process blocks; pass eps^2 < 0.3 in both seeds of the Huber arm, M1 control reproduced.
Planning estimate ~3-4 A40-h training + grading (~12 core-h per candidate) + fresh blocks. PI GO 2026-11-09. Predictions (angle lead): Huber arm eps^2 < 0.3 in both seeds [30%]; Huber below BCE by > 2 sigma [50%]. A fired kill is diagnosed, varied and sent to Astra first.

## T-SOFT2 RESULT 2026-11-09 (A40 ~5.4 h, CPU ~53 core-h fresh blocks; sign_soft2/grade/fresh/paired_fresh.json verified by me)
20 FRESH process blocks (960 draws), jackknife over blocks; QUOTABLE halves. eps^2: sgn(M1) 0.401(34); BCE 0.690(94) / 0.701(92); HUBER 0.699(79) / 0.701(86).
HUBER - BCE paired: +0.009(35), -0.000(22). Every trained net ~0.30(10) ABOVE sgn(M1). PASS FAILS (bar 0.3); predictions [30%] / [50%] missed.
Spec change by the worker (judged OK): trainable output scale g = a m (literal Huber pulled margins down 6x; dev eps^2 4-10).
Diagnosis: the excess is in SHALLOW decades -6..-2 (M1 0.03-0.47, nets 0.41-0.84); deep decades -10..-8 ~ M1; wrongly blocked weight 0.0037 -> 0.0064;
82% still from wrongly unblocked; coverage is not the cause (123k-228k configs per decade -9..-6); more steps / lambda x4 give 0.68-0.69.
BCE and Huber end as the SAME sign function (held-out error vs s_G 0.061 in all four runs).
Reading: supervised training of this class does not carry the step's sign onto the loop metric; it adds damage where sgn(M1) is already nearly right.
Sixth failed training variant (BCE, IW, PP-full, gate/keep, soft target...). Step back: stop training stored signs of this class; the live route is the
EXACT sign where it matters (amplitude gate + exact pruning), whose accuracy looked good (parent variant) and whose cost is unmeasured. To Astra (PI rule).

## ASTRA on T-SOFT2 (cj-37ea7c904156, /scratch/a/A.Otaifi/astra/sign_ideas/ASTRA_soft2.md, read; all fresh numbers reproduced to 3e-16)
KILL further BCE/IW/Huber training of the frozen-RBM real-pairing sign class under this recipe (a practical stop, not an impossibility theorem).
Term: training moves toward the target, but its induced change of the kept-bond sum (small denominator next to the node) is too distorted to pay:
in the exact decomposition the squared-change term outweighs the cross term in every arm. (Correction: BCE and Huber are NOT the same sign function:
27/960 roots, 4351 neighbours differ.) Fresh blocks are time-limited (done=False), so errors are block jackknife, not a converged FN measurement.
Parent-spliced eps^2 KILLED as the gate's acceptance test (it grades a different correction). Exact pruning CONTINUES as a component.
NEXT (T-K7, pre-registered before any number; CPU 3-4 core-h, cap 8): the TRUE amplitude-gated sign S_-7 (eq 4: s_0 = sgn(M1) where decade > -7, s_G
where <= -7, applied separately at x, y and every z). Accuracy: exact mixed f_J(x) on the 20 fresh blocks; 256 outer reference bonds stratified by block
and y decade (cover -8/-7/-6) with inclusion probabilities, complete one-swap neighbourhood of each sampled y, parent result as control variate (eq 5).
Cost: complete caller request scopes on a small fixed trace, gate decided from log amplitudes, r = 2 and 3 timed with exact ungated lower levels, plus the
absolute cost of the first-update Jacobi caller. Pass: eps^2 interval < 0.3 AND C3/C2 interval < 3 (absolute cost reported). Lower bound above a limit
rejects that claim; cap reached unresolved = inconclusive. Predictions (angle lead): eps^2 < 0.3 [55%]; C3/C2 < 3 [35%].

## T-K7 RESULT 2026-11-09 (job 17852399, ~4.6 core-h; sign_k7/result.json verified by me; controls pass)
ACCURACY PASSES: true amplitude-gated sign S_-7 (sgn(M1) where decade > -7, exact s_G where <= -7, at every x, y, z): eps^2 0.191 +- 0.048,
95% [0.097, 0.285] on 20 fresh blocks (252 sampled outer bonds, eq 5; halves 0.154 / 0.231, QUOTABLE). First 8x8 sign below the 0.3 bar.
Remaining error sits in shallow decades -6..-3 (0.157 of 0.191), where S_-7 = sgn(M1); deep decades nearly clean.
COST: first-update caller C1 = 0.30 s per request (7.6 ELnp2 calls; proxy trace) = CHEAP. Repeated rounds REJECTED: C3/C2 = 22.3 [19.9, 25.1] with exact
UNGATED lower levels (hard-descendant expansion, 2.4k-50k calls per request at r = 3; heavy requests capped, lower bounds).
Reading: a gated exact one-step sign is cheap AND passes the loop's bar for ONE sign update, with nothing stored. Every-round updates with exact
lower levels are unaffordable. Untested: recursively GATED lower levels (a different updater, Astra) and a gate at k = -6/-5 to remove the shallow part.
Predictions: eps^2 < 0.3 [55%] held; C3/C2 < 3 [35%] missed.

## T-NODE (lead 2026-11-09; pre-registered before any number; runs AFTER Astra's preconditions (cj-e7eef3719cc1): real walker cost, walls, f0)
Question: is S_-7 a better NODE than sgn(M1) at 8x8? FN-GFMC at c = 0 on the fixed amplitude |M1| with sign S_-7 (sgn(M1) where decade > -7, exact s_G
where <= -7; s_G needed only at deep walker neighbours, one level), same code and settings as M1's p=0 (-0.4987565(28), gfmc_m13b/prod/rb8).
Also measure the REAL per-move cost (time per walker step, fraction of moves needing exact s_G) from that run; the proxy trace does not count.
Placement (lead): not in the R1 headline; candidate sign for R2's labels and walls only.
Prediction (angle lead): gain over sgn(M1) of 5e-6 to 4e-5 per site [55%] (cf. M1 p=1 full step, sign + amplitude, -0.498946: 1.9e-4).
KILL: no gain beyond 2 sigma -> S_-7 is not used (a fired kill is diagnosed and sent to Astra first). Precision target sigma(diff) <= 4e-6
(two chains; error of the difference). A40 inside the lead's ceiling.
Code (lead): copy of /scratch/a/A.Otaifi/ansatz_runs/pfaffian_8x8_r1/grade_bare (gfmc_b2.BareM1Sign, M 1024, dtau 0.1; code shas in prod/run_metadata.txt:
gfmc_b2.py f768e05b, grade_bare.py aa0cf6af, gfmc_aug8.py 8c383172, pp2_lemma.py ef50e129), S_-7 added as a sign option, all else identical; sgn(M1) arm = its B0_k3..k5
(-0.498762 / -0.498755 / -0.498749) reused if the code path is unchanged for that arm, else rerun. Ran on Paderborn H100 originally; A40 or H100 express.

## ASTRA on T-K7 (cj-e7eef3719cc1, ASTRA_k7.md complete despite the job's quota exit; read) 2026-11-09
T-K7 accuracy PASS is sound (replayed exactly); recursion with exact ungated lower levels rejected (C3/C2 >= 22.3, censored). S_-7 = viable candidate
for ONE frozen sign update; FN-energy benefit and real walker cost unproved; recursively gated lower levels = a different updater, worth a later capped
test, no convergence guarantee; k = -6 plausible later (first-round calls 15.4 vs 7.6), -5 not justified now.
Preconditions for T-NODE: freeze B0 = |M1| inside the gate even when the amplitude is refitted; s0 = sign Re[w M1] with the SAME global phase on both
sides of the gate; m_ref -67.03209121446969, alpha -0.1754318714549749, step E -31.915135844116154 (distinct from the Jacobi E' -31.920416); c = 0
needs only x/y signs (simplest first test); time the complete production move (not 0.30 s / 7.6 calls); fix the wall-time ceiling before launch.

## T-NODE STATUS 2026-11-09 (no energy yet; sign_node/; worker's numbers, read)
Controls pass: (1) forced-s0 path reproduces the B0 code path bitwise on one device (30 blocks; broken input differs); (2) S_-7 = stored decades8
labels at 954 x / 134,684 y (0 disagreements; gate off: 1254); (3) walker E_L vs direct ELnp2 5.0e-11 at flipped deep signs; global phase 1.8e-16.
REAL COST (A40, 6 chains): S_-7 2.65x the plain c = 0 step (0.029 vs 0.011 s/move); s_G needed in 45% of moves, 2.1% of neighbour evaluations,
2.9 s_G per move; s_G differs from s0 at 24% of deep neighbours. Spec change: s_G via the pp2_lemma GPU path (= make_sG8; matches labels exactly).
BLOCKED: Paderborn express November quota is nearly used (hlp1_prodA reserves 1072 of the remaining minutes). sigma(diff) <= 4e-6 needs both arms
rerun (~3.0 H100-h +-30%); reusing B0 gives sigma(diff) >= 4.8e-6 (detects gains >= ~1e-5). A40 production ~63 h per S_-7 chain: impractical.
Probe 4186269 pending (express). Decision with the lead: QoS / timing.

## T-NODE BRANCHES (written 2026-11-09 before the verdict of job 4187008)
Measured cost already: S_-7 costs 2.65x the plain c = 0 step per move (A40 probe). Branches by the energy gain G = E(sgn M1) - E(S_-7) per site:
(A) G >= ~2e-5 (large): the first sign update is worth a lot at 8x8 -> (1) hand S_-7 to the lead for R2 labels/walls; (2) every-round becomes the priority:
    next test = Astra's capped RECURSIVELY GATED second round (lower levels also S_-7-gated): its eps^2 and C3/C2 on the proxy, then a real trace;
    (3) k = -6 accuracy/cost (first-round calls ~15 vs 7.6).
(B) 2 sigma < G < ~1e-5 (small but real): S_-7 is useful but modest -> hand it to the lead (their call whether 2.65x per move pays in R2); every-round
    stays open at lower priority; the amplitude loop is the bigger lever at 8x8; next sign test only if the lead's R2 needs it.
(C) G not beyond 2 sigma (KILL as registered): do not use S_-7 as a node. Diagnose first: eps^2 measures damage to the round labels, the node energy
    measures FN at c = 0; walkers rarely visit decades <= -7, so the node test may be blind to where S_-7 differs. Then the sign's value, if any, is in
    the labels/walls (c > 0): the decisive test becomes an R2 label comparison (labels with S_-7 vs sgn(M1)) rather than the node. To Astra before any verdict.
Cost axis for all branches: c > 0 walls need S_-7 two swaps out (d^2 per move); that real cost is unmeasured and must be timed before R2 uses S_-7 in walls.

## T-NODE RESULT 2026-11-09 (Paderborn H100 express 4187008, ~3.4 H100-h; sign_node/prod/{ANALYSIS.md,summary_node.json,post.log}; verified by me from the json)
- KILL FIRED (as registered): E(S_-7) - E(sgn M1) = +7.96e-6 +- 4.1e-6 /site paired (unpaired +8.4(3.6)e-6): no gain; S_-7 leans WORSE at 1.9 sigma.
  Predicted gain >= 5e-6 excluded at 3.2 sigma; >= 2e-5 at 6.8 sigma. sigma(diff) 4.1e-6 vs target 4e-6 (just missed: S_-7 chains hit maxsec at
  2171/2160/2150 of 2950 blocks; F arm complete).
- Arms (quotable.py, 3 chains each, chains agree): S_-7 -0.498746(2.7e-6) [chi2/dof 0.15]; sgn(M1) -0.498754(2.3e-6) [chi2/dof 1.33] (B0 earlier -0.4987565(28)).
- CTRL1 on H100: F_k3..5 = B0_k3..5 bitwise, 1550 blocks each, 0 differing fields; broken arm differs.
- Cost on H100: 5.47 vs 2.01 s/block (2.72x); s_G 75% of S_-7 time; 45% of moves need s_G.
- [WRONG, see ASTRA below] Why the node test cannot see the eps^2 gain (my equation reading, Astra asked blind): S_-7 differs from sgn(M1) only at d <= -7. Walkers spend
  frac_x_deep = 1.8e-3 of their time at such x, and s_G flips 5.4% of those (~1e-4 of walker time); flips at deep neighbours 23.6% of 2.1% of evaluations.
  eps^2 draws equally per decade, so its 0.40 -> 0.19 gain sits where the walker measure is ~1e-3. Per-decade E_mix differences are noise-limited
  (largest |diff| 2.7e-4 at d = -2, where the signs are identical). 6x6 already said this: s_G's step gain was almost all amplitude (A47).
- ASTRA on T-NODE (cj-9c2bd0991219, /scratch/a/A.Otaifi/astra/sign_ideas/ASTRA_node.md, read in full; audit files there): run sound (controls
  re-checked, raw curves audit_raw_*.png, diff k3 fifths chi2/dof 2.40 p 0.048 > cutoff 0.01); keep the operational stop on S_-7 as a node on B0,
  but it is not proof of zero benefit (+-2 sigma interval [-0.1, +16.1]e-6). CORRECTION of my reading above: frac_x_deep x flip_x_frac (1e-4) is a
  fraction of ROW EVALUATIONS, not walker time, and the test was NOT blind: power 68% at a 1e-5 gain, ~100% at 2e-5. My prediction (5e-6..4e-5)
  was a bet without a mechanism; the run lowers it. Exact fixed-amplitude identity: dE_FN = -sum_{dF} t_xy a_x a_y int (u_x/a_x + u_y/a_y)^2 (eq 5).
  The +8e-6 lean may be noise or a sign-amplitude mismatch; not separable from this run (eq 7), and NOT a tax to carry onto B1.
  Label route OPEN with a mechanism: delta f(x) = log(1 + v_x/K) - log(1 - v_x/D), so a sign change moves the Jacobi target strongly where the kept sum
  K is small (eq 9, checked on 960 roots to 1.6e-15). Decisive test = R1' sign ablation: same B0, pool roots, warm start, budget, c, f0, E' as R1;
  S_-7 frozen on B0, its own labels and RE-SAMPLED blocked partners (old pool partners miss newly blocked bonds); grade the bare guide S_-7 x B1_S vs
  R1's -0.498803(3.2e-6), sigma(diff) <= 4e-6, gain > 2 sigma. Cost NOT 3.1 H100-h: ~5.5 H100-h fit + grade (2.72x grade) plus relabelling of
  2.0e6 roots x 2 partners, possibly ~500 core-h (unmeasured). "Walls must match the node" is the chosen iteration, not a theorem.
  Next (Astra): <= 1 core-h offline relabelling/cost pilot (64 roots over the 8 fitted decades, 2 partners each, full neighbourhoods, gate, labels, rho;
  gate-off control reproduces old labels; wrong-endpoint-sign broken control). Not approved by Astra: k = -6, longer k = -7, online c > 0.
  Code defects to fix before reusing the provider: GatedSign.__init__ sets self._elnp = None (shadows the method; small-margin fallback would crash;
  sg_recheck = 0 in all runs, so these results are unaffected); analyze_node.py reports complete=true from status OK (S7 chains completed=false).
- Branch C: S_-7 is not used as a NODE. Its value, if any, is in the labels (c > 0); test only on the lead's loop metric.
- Lead 2026-11-09 (verified the gate; paired per chain +10.9 / +1.5 / +7.8e-6; kill stands). Label-route metric = the refit net's bare FN energy at
  c = 0, same code, vs R1's -0.498803(3); eps^2 is only a screen (8x8 link to energy uncalibrated). T-HYB constraint: walls must match the node, so
  S_-7 walls in f_J force an S_-7 node, which costs +8e-6 -> a label gain must exceed ~1e-5 to show. Cost R1' fit ~1.7 + grade ~1.4 H100-h, after the
  headline and only if Astra leaves a mechanism. Lead's prior 25%. Design nothing until Astra replies.

## RELABEL PILOT RESULT 2026-11-09 (sign_relabel_pilot/{PILOT.md,pilot.json}; 0.51 core-h; worker; numbers checked by me in pilot.json)
- Full-pool S_-7 relabelling of R1's pool (2,000,175 roots, K = 2) = 15,458 (+-524) core-h, 4.9e8 exact s_G calls (0.11 CPU s each); decades -7..-9
  are 83% of it (s/root 4.2 at d -2 -> 41.9 at d -9; s_G calls/root 8 -> 381). Stored gate-off labels 0.38 s/root. Cache hit ~30%.
- Controls: (a) gate off reproduces stored R1 labels/partners/rho on 64 roots to 8.9e-13; (b) negating gated endpoint signs changes 63/64 labels
  (max |d lnT| 5.3; the 64th root never reaches the gate). Fixed-provider fallback agrees; the original crashes (TypeError), as Astra found.
- Blocked sets change at 59/64 roots (11.7% of S_-7-blocked bonds not blocked under sgn(M1)); 6/128 sampled partners new (all at d -8/-9): old
  partners not reusable. Caveats: shared workstation (load ~10), sg_chunk 16; Wbar not computed.
- Bill of the R1' ablation: ~15,500 core-h relabel (a GPU port is unmeasured; my rough scaling from T-NODE's s_G throughput is ~25 H100-h, a
  guess) + ~5.5 H100-h fit/grade. My recommendation: park the S_-7 label route (one round only, lead's and my prior 25%). Cheapest untried variant:
  a 10% sub-pool (~1,550 core-h) with its own s0 control fit (+2 fits, +2 grades, ~9 H100-h). Lead decides.
- PARKED by the lead 2026-11-09: 15.5k core-h for one round at a 25% prior does not pay; 10% sub-pool variant recorded as cheapest untried (TRIED_IDEAS A49).
  Every-round question goes back to the PI; lead's input: 8x8 headline interim lean -1.5e-5 (2.5 sigma, final tonight), its limits ranked overlap ~50%, sign ~30%, deep fit ~20%.

## T-RG (PI 2026-11-09: "the exact sign where the wavefunction is tiny had a lot of hope, and repeating it"; pre-registered before any number)
Question: can S_-7 be REPEATED every round at bounded cost if the lower rounds are gated too (Astra's eq 1, ASTRA_k7.md (3)):
t_0 = s0; t_{r+1}(x) = s0(x) outside D (decade > -7), sgn[b_x t_r(x) + sum_y a_xy t_r(y)] inside D; b_x = 1 + alpha(H_xx - E), a_xy = alpha H_xy B_y/B_x.
Equation for the cost (why T-K7's repetition blew up): a deep root has on average 132 deep neighbours (ASTRA_k7, 385 fresh deep roots), so
C_r ~ (unpruned deep children)^r; shallow children cost nothing (known s0). The only lever is exact partial-sum pruning: stop expanding when
|known sum| > |b_x| + sum of unresolved |a_xy| (cheap shortcut certified 0/385 before any deep child is resolved; adaptive pruning untested).
Test: T-K7's cost machinery (sign_k7/cost, same proxy trace and request scopes) with eq-1 recursion, shallow children from s0, deep children resolved
cheapest-first with certified pruning, cache per (round, configuration). Report C1, C2, C3 (request-weighted CPU s), C3/C2 with bootstrap interval,
deep children resolved per request, and roots whose t_2 differs from t_1 (Astra: 175/960). Controls: r = 1 reproduces S_-7 at every root (T-K7 labels);
pruning off gives identical signs (exactness); a broken certificate (drop the unresolved sum) must give wrong signs.
PASS: C3/C2 interval < 3. KILL: lower bound >= 3 (then diagnose: fan-out per level, Astra). Cap 8 CPU core-h; capped requests = lower bounds.
Prediction (angle lead): pass 20% (deep configurations cluster). Cost only: round-2 accuracy needs its own reference, designed only if this passes.
Run dir /scratch/a/A.Otaifi/ansatz_runs/sign_rg (worker).

## FRESH-EYES CONSULT (Fable, no CLAUDE.md, PI request 2026-11-09; /scratch/a/A.Otaifi/astra/fable_fresh/FRESH.md, read)
Main claim (80%): the second sign step changes only node-shell configurations, so its gain is second order and need not grow with L (6x6: second
steps -0.43..+0.39e-6 vs first 5.3-11.4e-6); the lever is the base guide (our M1 VMC -0.498674(12) vs published RBM+PP -0.498886(1),
2005.14142; Varbench RBM+PP -0.498963(2) below our p=1 FN -0.498946(5): checked against TRIED_IDEAS I2 and PLAN.md l.19-20).
Ranked: guide first then p=1 + FN + variance extrapolation 75%; restricted-class E_FN gradient on the real PP pairing only (C15 stage (c), never run)
45%; Ising-ground-state sign labels at fixed amplitude (2207.10675) 40%; forward-walked ln g instead of the Jacobi head 30%.
My check: its "0.18% of walker weight" reuses frac_x_deep, which Astra showed counts row evaluations (walker residence below -7 is tau 1.1e-3);
its claim that the S_-7 null was predictable by weight counting is therefore unsupported (Astra: 68% power at 1e-5). "6x6 second steps small" is 6x6 (PI: not a verdict).

FABLE AFTER THE PI'S PUSHBACK (FRESH.md section 10, read): concedes the frac_x_deep error and that the S_-7 null was a measurement, not predictable.
2a holds only near convergence: round r's gain ~ remaining wrong-sign weight, contracting by c_r per round (A1: 18x, then 5x from Marshall), so the
loop needs r* = log(w0/w_floor)/log c rounds (1 from M1, 3-4 from Marshall). Case for the loop over SR: the Marshall trap (2002.04613), since a
projection step flips bulk signs without driving amplitudes through zero. Proposed 8x8 bad-start test (Marshall x |M1|, A13 reference/held-out):
A = rounds 2-3 of the stored loop (one-hop labels); B = Ising-ground-state labels at fixed amplitude (2207.10675); C = SR training of the same net at
equal GPU-h (Marshall-trap control, run first). PASS c_2 >= 3 and c_3 >= 2; KILL c_2 < 2; loop unnecessary if C reaches w_2(A). ~12 A40-h + 0.4 H100-h.
My check: c_r is a wrong-sign-WEIGHT metric, and LESSONS says cutting w_T can raise the label error (8x8 gate (a)), so the grade must be the FN energy
per round (lead's metric), with w only as a screen. 12 A40-h exceeds my takeover ceiling (8): design at the check-in with Astra's answer (QUESTION8 (6)).

## T-LEV (takeover, angle lead's call on Fable's step; pre-registered before any number; CPU <= 10 core-h)
Question: does the fixed-amplitude SECOND sign step have more leverage at 8x8 than at 6x6 (PI's premise "a better sign every round" vs Fable's 2a)?
m2(x) = [1 - alpha(H_xx - E)] s_G(x)|B_x| - alpha sum_y H_xy s_G(y)|B_y| (s_G = first-step sign, fixed |B|);
Lambda_L = (1/N) sum_x p(x) |V(x) - K(x)| [sgn m2(x) != s_G(x)], p = walker residence weight by decade with the panel's inclusion weights
(8x8: sign_k7 960-root panel, p from sign_node prod tau_dec of the sgn(M1) arm; 6x6: exact B2-fixed tables, p exact).
CALIBRATION GATE (first, LESSONS: check a proxy on known cases): at 6x6 the same proxy for the FIRST step (s0 -> s_G) and the SECOND step must order the
measured gains (first 5.3-11.4e-6, second -7.6e-8 /site, STEP 0' arm (a)): Lambda_6(first)/Lambda_6(second) >= 10, else the proxy is uninformative
and nothing below is read. Broken control: proxy with p uniform over decades must change the 8x8 number (shows p acts).
Reads (recommendations to the PI, not automatic): R = Lambda_8/Lambda_6 <= 3 -> second step at 8x8 likely ~1e-6/site: recommend the PI move effort
to the guide; R >= 10 -> premise alive: GATE 1-2 plan; between -> GATE 1 only. Bootstrap interval over roots; a read needs the whole interval in a band.
Caveat: misses the refreshed-amplitude second step (needs the second shell). Prediction (angle lead): R <= 3 [45%], R >= 10 [20%].
Run dir /scratch/a/A.Otaifi/ansatz_runs/sign_lev (worker).
STOPPED 2026-11-09 (partial, 0.15 core-h, sign_lev/lev8.json): Lambda_8 = 4.9e-5 [95% 1.5e-5, 1.1e-4], one d -5 root ~half; 177/960 roots flip; no calibration, no Lambda_6, no read. (PI: the goal is a method that scales; at large L the starting network is bad). T-LEV measures leverage from an
already-good sign (M1), so no outcome changes the decision. Fable's 2a holds only from a good start. Our own records from a BAD start contradict it:
A1 (6x6 from Marshall, wrong weight 2.0e-2 -> 1.1e-3 -> 2.3e-4: the second step gains ~5x again); C1 (exact loop from Marshall converges, iteration
count flat 16 -> 36 sites); A13 (8x8 from Marshall: stored net + exact hop closes 98.5% of the Marshall -> ViT gap).
Hypothesis (angle lead, ~55%): storage is easy while the remaining sign errors sit in high-amplitude configurations (bad start, A13), and hard once
only deep-tail errors remain (good start, A48); the tail is then the endgame, where Fable's second-order argument applies. Test regime for
"every round" = a deliberately bad start at 8x8 (Marshall or a weak network) standing in for large L; design at the check-in, Astra first.

## T-BAD (PI GO 2026-11-09 21:10: "go for testing"; DRAFT pre-registration, finalised after Astra QUESTION8 (6) and the cost probe, before any number)
Question: from a BAD start at 8x8 (stand-in for large L), does the stored loop keep lowering the energy in rounds 2 and 3? Round r: labels = the
one-hop threshold-step sign of guide r; store them in a sign net; guide r+1 sign = net_{r+1} + one exact hop (constant cost per round, no nesting).
Base: A13 machinery (experiments/signs_8x8, results/signs_8x8; runs Chatty/runs/signs8): amplitude a = projected ViT fixed, start Marshall M;
round 1 is DONE there: s_net^M + hop, paired VMC-referee E -0.498711(71) vs ViT sign -0.498785(52), Marshall -0.493748(238), K1_M -0.498186(129).
Arms: A = rounds 2 and 3 of the stored loop; C = SR (variational-energy) training of the same sign-net class from M at equal GPU-h (Marshall-trap
control; Fable: run first). B (Ising-ground-state labels) deferred.
DESIGN ISSUE found while drafting (not in Fable's cost): A13 trained on samples + ALL their one-hop neighbours. Round-2 labels at a NEIGHBOUR y need
s_1 on N1(y), i.e. r_net at two hops from y = a on the THREE-hop set of the sample (~3-4e5 ViT evals, 70-90 s/sample on a 2080 Ti); labels on samples
x alone need only the stored two-hop set. So round >= 2 either trains on samples only (many more samples: two-hop data ~2 s/sample) or pays three hops.
This is the real per-round cost of storage; it is constant in r but sets the sample budget.
Metric: screen = A13's paired variational referee (RB-type, per-chain SE); deciding = FN energy at c = 0 on fixed |amplitude| per round (lead's metric
class; asked to confirm). Reads (draft): PASS g2 = E1 - E2 and g3 = E2 - E3 both beyond 2 sigma (paired); KILL g2 not beyond 2 sigma (storage caps
the loop at one round); loop unnecessary at 8x8 if arm C reaches E <= E2(A) at equal GPU-h. Prediction (angle lead): g2 > 2 sigma 60%, g3 40%.
Budget: PI GO; proposed cap 20 A40-h + 1 H100-h for stage 1 (A, C screens), FN grade only if the screen passes.
SCREEN CALIBRATION (lead 2026-11-09, registered before running): FN c = 0 on the fixed amplitude for the round-0 (Marshall) and round-1
(s_net^M + hop) signs. The VMC screen is trusted for rounds 2-3 only if FN ranks round 1 below round 0 as VMC does AND the FN gain E0 - E1 is within a
factor 2 of the VMC gain (5.04e-3 - 7.4e-5 = 4.97e-3 /site, paired, results/signs_8x8). Otherwise FN grades every round from the start. Inside the 20 A40-h cap.
Now: cost probe (worker, <= 1 A40-h): two-hop data rate on A40, round-1 net + hop evaluation rate, three-hop rate on 20 samples. Run dir sign_bad.
COST PROBE RESULT (A40 job 17870133, 0.21 A40-h; sign_bad/probe/PROBE.md, read): two-hop data 2.13 s/sample (10,157 states); round-1 net + hop sign
0.037 s at a fresh x, its local energy 2.35 s/sample; three-hop set 4.6e5 states, 87 s/sample. Rounds 2+3 without training: (a) samples-only, 32k/round
+ referee 55.7 A40-h (33.0 with the ViT data cached, fixed amplitude); (b) A13-style three-hop 130.9 (77.3). Per 1000 samples per round: (a) 0.70 h,
(b) 29.3 h. Neither fits 20 A40-h; (a) fits at ~10-12k samples/round uncached. Risk: samples-only training gives far fewer training states than A13
(2048 samples + ~3e5 neighbours), and A13's nets were data-limited, so a weak round-2 net could fire a false kill. ViT parity with the old run exact.
/scratch is at 958G of 972.8G soft quota: any cache goes under /project/theorie/a/A.Otaifi/runs/j1j2, not /scratch.
Decision after Astra (QUESTION8 (6)): round 2 only first (the kill is g2), sample count vs data-limit risk, and whether the cap must rise (PI's call).

ASTRA ON T-BAD (cj-9a275217f646, /scratch/a/A.Otaifi/astra/sign_ideas/ASTRA_fnsign.md section 6, read in full): GO on the question, amended:
(1) recurrence (6a): n_{r+1} <- fit(S_r), S_{r+1} = K_T[n_{r+1}] (fit the DEPLOYED sign, then one hop). My draft fitted K[S_r] = two ideal steps per round and
needed three-hop data; (6a) needs only the two-hop set for neighbour labels (CPU check 2.85e-14). (2) 8,192 tempered root packets/round with all ~145
neighbour labels, learning curve 2k/4k/8k, doubling to 16k if still improving; failure while still data-limited = no verdict. (3) VMC referee = diagnostic,
NOT a veto (disagrees with the lead's calibration logic: a large round-0->1 agreement does not calibrate the small round-1->2 sign). (4) Decide by FN c = 0
of the DEPLOYED S_r; g* = 1e-5, sigma(g) <= 3.5e-6; kill only if g + 2 sigma < g* after data adequacy. (5) SR control with a continuous phase, same capacity,
matched compute, FN-graded; Ising arm deferred (2207.10675 is a greedy heuristic). Budget A + C ~22 A40-h (35 with doubling) + an UNMEASURED FN bill:
a full row of the ViT + hop sign costs 2.35 s on an A40, so FN on the A13 amplitude may be unaffordable (100k rows ~65 A40-h).
MY DECISION (takeover): price FN before any learning spend. Stage 0 on the |M1| amplitude (T-NODE-validated GFMC, cheap amplitude) instead of the ViT:
time FN c = 0 per move for Marshall x |M1| (bare), a stored CNN sign (net-only), and net + hop (deployed S_r); short Marshall x |M1| FN run (round-0 point).
Cost <= 2 A40-h. With |M1| the A13 round 1 must be redone (its labels used the ViT amplitude). Full T-BAD budget goes to the PI at 09:00 with these numbers.
STAGE 0 RESULT 2026-11-10 (A40 jobs 17872472/3, 17874751; 2.28 A40-h; sign_bad/stage0/STAGE0.md, read): s/move at M 1024: sgn(M1) 4.85e-3, Marshall 4.9e-3,
stored CNN (net-only) 1.61e-2, deployed net + one hop 0.77 (+-30%; 87% sign evaluation, ~1e4 unique net states per move). Controls pass (generic path =
BareM1Sign bitwise 20 blocks; broken differs 6554/6554; hop sign vs recomputation 4.5e-11, 0 mismatches). Round 0 Marshall x |M1|: -0.495471 / -0.495477
(225 blocks/chain, NOT QUOTABLE, plot only), ~3.3e-3 above sgn(M1). FN on the deployed hop sign: 292-1837 A40-h per chain at sigma 1e-5 = UNAFFORDABLE.
DECISION (takeover, branch pre-stated in Running): deploy and FN-grade the STORED NET ALONE each round (the scalable deployment: per walker step cost =
one net evaluation per neighbour, 3.3x bare); the hop is used only offline to make labels. Scope change recorded: this tests "store each round's one-step
sign and use the stored sign", not Astra's "net + hop" deployment. Finding in itself: a one-hop dressed sign costs ~150x per FN walker step at 8x8.
T-BAD FINAL (registered 2026-11-10 before any round-1 number):
 recurrence: n_0 = Marshall M; labels_r = K[n_r](z) = n_r(z) sgn[1 + alpha(E_L(|B0| n_r; z) - E)] on training states z (root + all valid one-swap neighbours),
 alpha -0.1754318714549749, E -31.915135844116154 (the loop's sign step, = threshold T = E - 1/alpha); fit n_{r+1} (A13 CNN class, D4 x flip averaged,
 Marshall x sgn(logit)); deploy n_{r+1}. Amplitude fixed |B0| = |M1| everywhere. Data: tempered (beta 1/2) root packets with two-hop |B0| sets;
 learning curve 2k/4k/8k packets (held-out label disagreement and held-out complete-neighbourhood E_L error); final net from 8k; doubling to 16k if 4k->8k
 still improves held-out error by > 10% (relative).
 Metric: e_r = FN c = 0 energy/site of (|B0|, n_r), 2 chains from the T-NODE starts, quotable.py; sigma per arm <= 3e-5 (gains from a bad start are
 >> 1e-5; refine only where a decision needs it). g_r = e_{r-1} - e_r, sigma(g) from both arms.
 Reads: round r USEFUL if g_r > 2 sigma(g_r). KILL of "storage carries the loop past one round": g_2 not beyond 2 sigma with data adequate
 (4k->8k change < 10%) -> diagnose (storage loss: compare held-out label error of n_2 vs n_1; and the exact one-step leverage of K[n_1] on a panel), Astra.
 Data-limited (still improving) -> no verdict, double data.
 Predictions (angle lead): g_1 in [1.5e-3, 3.3e-3] 75%; g_2 > 2 sigma 60%; g_3 > 2 sigma 45%. Control C (SR, continuous phase) after round 2.
 Budget from measured rates: data ~0.5 A40-h/round, fit ~1-2, FN ~2 chains x (1-4) A40-h per sign; ~10 A40-h per round.
ROUND 1 PHASE A RESULT 2026-11-10 (A40 17875220 data 1.55 h, 17877190 fit 1.33 h; runs/j1j2/sign_bad/r1, sign_bad/r1/; worker's numbers, read):
 10240 tempered roots (8192 train / 2048 held-out by chain), 1,468,952 states, labels K[M] (minus share 25.6%). Label controls: 20 states vs direct
 recomputation 1.5e-10, 0 sign mismatches; wrong alpha changes 3.9-25.6% of labels. Learning curve (held-out disagreement / E_L mean abs err, total units):
 2k 0.110 / 3.52; 4k 0.106 / 3.22; 8k 0.103 / 3.00 (no-flip 0.257; Marshall E_L err 4.42). 4k->8k 3% (< 10%): no 16k. Train error RISES with n
 (0.084 -> 0.099): optimisation/capacity-limited, not data-limited. By |B0|^2 decade: 0.1% at >= -1, 3.3% at -5, 12% at -7, 22% at -9, chance below -13.
 n_1 = runs/j1j2/sign_bad/r1/fit/net_8192.pkl, sha256 33349343...93ddc. Worse fit than A13 arm M (held-out 6.7% on neighbours), on another amplitude.
 DECISION: grade round 1 now (launchers read by me; Marshall 2 h, net 16 h ceilings, 350 blocks/chain, 2 chains each) and, in parallel, a FIT-RECIPE
 CHECK before round 2 (registered now): 8k roots, steps 60/root and a second seed; adopt for round 2 if held-out disagreement drops > 10% relative;
 n_1 is regraded only if g_1's decision depends on it. <= 2 A40-h (raised to 4 A40-h on measured cost 0.75 h per 30-step fit: three fits ~3.7 h).
FIT-RECIPE CHECK RESULT 2026-11-10 (A40 17885650-52, 3.70 A40-h; runs/j1j2/sign_bad/r1/fitcheck; worker's table, read): held-out disagreement
 0.1029 (30 steps, seed 0) / 0.1019 (60, seed 0) / 0.1028 (30, seed 1) / 0.1025 (60, seed 2); seed spread ~0.1%, recipe effect <= 1% (threshold 10%):
 KEEP 30 steps/root. Train error falls slightly with steps, held-out does not: the CNN class (or optimiser), not data or length, limits the fit; deepest
 decades at chance for every fit. Caveat for the kill rule: a round-2 null could be capacity-limited; the diagnosis must test a larger class before a verdict.
ROUND 2 PHASE A started 2026-11-10 (before g_1, to save wall time; g_1 partial is far below Marshall): labels K[n_1] on the round-1 roots/states (same split),
 fit n_2 with the round-1 recipe, stage FN of n_2 as for n_1. Run dir sign_bad/r2.
ROUND 1 FN RESULT 2026-11-11 (A40 17885644 Marshall, 17885645 net; sign_bad/r1/fn_*/ANALYSIS.md; raw curves quotable_*.png looked at by me: stationary,
 one single-bin excursion in chain k4 near block 80): e_0 (Marshall x |M1|) = -0.495522(23), e_1 (stored n_1 x |M1|) = -0.498158(17), both QUOTABLE
 (2 chains each, chi2/dof 0.35 / 0.01). g_1 = 2.636(29)e-3 /site: USEFUL, inside the predicted [1.5, 3.3]e-3 band. Reference sgn(M1) -0.498754(2.3e-6):
 n_1 closes 82% of the Marshall -> sgn(M1) gap (remaining 5.96e-4). Round 2 FN 17893815 decides whether storage carries a second round.
ROUND 2 PHASE A RESULT 2026-11-11 (A40 17888468, 1.86 A40-h; runs/j1j2/sign_bad/r2; worker's numbers, minus-share definition checked by me in r2_labels.py):
 labels K[n_1]: controls pass (20 states vs direct 1.2e-11, 0 mismatches; Marshall in place of n_1 reproduces round 1 on all 1,468,952 states; broken differs).
 K[n_1] != n_1 on 17.1% of states (0.1% at d >= -1, 6.3% at -5, 20.6% at -7, 38-45% at -9..-11); many repair n_1's own fit errors (n_1 held-out 10.3%).
 (minus_share 0.499 is the ABSOLUTE sign share, not comparable to round 1's relative 25.6%.)
 n_2 fit: held-out disagreement 0.1042 (n_1: 0.1029), flips 0.198, keeps 0.069, E_L err 3.37 vs Marshall 4.69; deep decades near chance as before.
 n_2 = runs/j1j2/sign_bad/r2/fit/net_8192.pkl sha256 c7b16b7b...6c (checked). FN of n_2 submitted (launcher read by me: copy of r1 with paths/seeds 41/42).
ROUND 2 FN RESULT 2026-11-11 (A40 17893815; sign_bad/r2/fn_net/ANALYSIS.md; raw curve quotable_n2.png looked at by me: stationary, chains overlap):
 e_2 (stored n_2 x |M1|) = -0.498658(9), QUOTABLE (chi2/dof 0.02). g_2 = e_1 - e_2 = 5.00(19)e-4 /site: USEFUL (26 sigma). Prediction (g_2 > 2 sigma, 60%) held.
 Gap to sgn(M1) -0.498754: 3.232e-3 (Marshall) -> 5.96e-4 (n_1) -> 0.96e-4 (n_2): contraction x5.4 then x6.2 per round.
 READING: from a bad start at 8x8, storing each round's one-step sign in a CNN and deploying the stored net carries the loop through TWO rounds with
 large, contracting gains, at a per-walker-step cost of one net evaluation per neighbour (3.3x bare). Fixed amplitude |M1|; the net class is capacity-limited.
 Next: round 3 (labels K[n_2], same pipeline; g_3 now ~1e-4 scale, sigma per arm ~1e-5 suffices); then control C (SR, continuous phase, matched compute).
ROUND 3 PHASE A RESULT 2026-11-11 (A40 17899661, 2.94 A40-h; runs/j1j2/sign_bad/r3; worker's numbers, launcher diff vs r2 read in its report):
 labels K[n_2]: controls pass (direct 1.2e-11, 0 mismatches; n_1 in place reproduces round 2 on all states; broken 16.6%). Flips K[n_2] != n_2 12.3%
 (round 2: 17.1%); 13.5% at d -7, 31-40% at -9..-12. n_3 held-out disagreement 0.0986 (n_1 0.1029, n_2 0.1042), E_L err 2.62 (heavy-tailed, rms 7.9).
 n_3 sha256 5b8b6dc4...a8f1. FN of n_3: 17900939 (16 h), watcher 2544452 (PPID 1); gate r3/code/r3_analyze_fn.py (g_3 = e_2 - e_3).
STORAGE SCREEN (PI asked 2026-11-11 whether this is the best training; registered before any number; offline on the round-3 labels, <= 3 A40-h):
 arms at 8192 roots, same held-out table: (a) warm start from n_2, same steps; (b) warm start, half the steps; (c) wider CNN 128 channels x 6 from scratch;
 (d) deeper 64 x 10 from scratch. Baseline n_3 (0.0986 / 2.62). Adopt for round 4 an arm that lowers held-out E_L error or disagreement by > 10% relative,
 or matches them at < 60% of the cost (seed spread ~0.1-2%). Label accuracy is a screen only; the FN grade of round 4 decides. Pair-product full-projection
 class: a larger build, deferred.
STORAGE SCREEN RESULT 2026-11-11 (A40 17901025-27, 17900956 bench; 2.51 A40-h; runs/j1j2/sign_bad/screen; worker's table, read):
 held-out disagreement / E_L abs err: n_3 0.0986 / 2.62; (a) warm, full steps 0.0984 / 2.64; (b) warm, half steps 0.0989 / 2.65 at 51% cost; (d) 64x10
 0.0913 / 2.62 at 1.8x cost (-7.4% disagreement, better at d -7..-13 by 0.01-0.03); (c) 128x6 not run (3.5x per step). Rule: (b) adopted for round 4
 (same accuracy, half the training cost); warm start gives no accuracy gain. (d) is the only real accuracy gain, below the 10% bar and costlier per walker
 step; kept as the candidate if round 3/4 stall. Depth helps more than training length: consistent with a class limit.
ROUND 3 FN RESULT 2026-11-11 (A40 17900939; sign_bad/r3/fn_net/ANALYSIS.md; raw curve quotable_n3.png looked at by me: stationary within noise, chain k3
 possibly drifting up ~1e-4 over the last third; k4 fifths chi2/dof 2.08, p 0.08; chains agree chi2/dof 1.17; scatter error quoted):
 e_3 = -0.498709(10). g_3 = e_2 - e_3 = 5.1(13)e-5: USEFUL (3.9 sigma). Prediction (g_3 > 2 sigma, 45%) held.
 Sequence e_0..e_3: -0.495522(23), -0.498158(17), -0.498658(9), -0.498709(10). Gap to sgn(M1) -0.498754: 3.23e-3, 5.96e-4, 9.6e-5, 4.5(1.0)e-5;
 contraction x5.4, x6.2, x2.1: the gains are slowing; the loop has not yet reached the VMC-trained sign.
 Reading (hypothesis): the storage loss (CNN at chance below d -11) is starting to dominate the one-step gain. Test of it below (T-CAP8).
NEXT, registered 2026-11-11 before any number (takeover):
 T-CAP8 (capacity attribution at fixed labels): FN c = 0 of the screen arm (d) net (64x10, round-3 labels, sha 86c5dc8e...499e) vs e_3 (64x6, same labels).
  Prediction: e(d) below e_3 by > 2 sigma, 40%. If yes, storage capacity limits the loop -> round 4 with the deeper class; if no, labels/step limit -> round 4
  with the current class to test convergence. 2 chains, 350 blocks, seeds 61/62.
 CONTROL C (SR, PI's question 2026-11-11 "is the loop cheaper than VMC"): same 64x6 CNN class with a continuous phase output theta(x) (psi = |B0| e^{i theta}),
  from Marshall (theta = 0 relative to Marshall), SR/minSR on the variational energy at fixed |B0|, matched compute = labels + fits of rounds 1-3
  (~7.7 A40-h); hard sign extracted from cos(theta) with a frozen convention; graded by the same FN c = 0. Reads: e_C <= e_3 at matched compute -> the loop has
  no demonstrated advantage at 8x8 (not a statement about larger L); e_C clearly above e_3 -> loop beats gradient training of the same class.
T-CAP8 RESULT 2026-11-12 (A40 17905447; sign_bad/cap8/fn_net/ANALYSIS.md; raw curve quotable_deep.png looked at by me: stationary, chains overlap):
 e_deep (64x10, round-3 labels) = -0.498731(9), e_3 (64x6) = -0.498709(10): d = 2.2(14)e-5, 1.6 sigma: NOT beyond 2 sigma (prediction 40% did not hold),
 but a lean of half the remaining gap to sgn(M1) (e_deep is 2.3e-5 above it). Reading: INCONCLUSIVE at this resolution (Astra's rule: "not beyond 2 sigma"
 alone is not a verdict). Registered branch said "round 4 with the current class"; I HOLD round 4 instead (decision, takeover): its expected gain (~1e-5) is
 below the unpaired resolution (sigma(g) ~1.3e-5), so it would read null whatever happens. The paired FN estimator is now the bottleneck for T-BAD round 4,
 T-CAP8's resolution and T-STRONG alike.
CONTROL C PHASE 1 RESULT 2026-11-12 (A40 17924353 7.72 h + probe 17905531 0.2 h; sign_bad/ctrlC; worker's numbers, launcher read by me):
 minSR (written here; validated: gradient vs autodiff 8e-14 / FD 2e-8, broken Re-for-Im off by 4.2; E_L of theta = pi set vs r1_lib 2.1e-14) on psi = |M1| M e^{i theta},
 64x6 CNN phase, 1457 steps at matched compute 7.72 A40-h. theta = 0 is a stationary point (E_L real): a 0.02 rad start collapsed back to Marshall on 4x4,
 so production started at 0.3 rad rms (accepted by me; recorded deviation from "small"). E_var -0.49367 -> -0.49695(32), still falling slowly;
 hard sign flips only ~1.6% of sampled configurations vs Marshall (loop n_1: 28% of tempered states); hard-sign variational E -0.4958..-0.4971, flat
 after 50%. Suggestive of the Marshall trap (gradient training barely moves the sign), NOT a verdict: FN of the hard sign decides (gate reports D vs e_0..e_3).
CONTROL C RESULT 2026-11-12 (cip a40-24gb slices 17931121 k3, 17931262 k4 (k4 stopped at maxsec, 322 of 350 measured blocks); ctrlC/fn_net_slice/fn_summary.json, w_k4/ANALYSIS.md):
 e_C (SR-trained phase net, hard sign, 7.7 A40-h, matched to the loop's rounds 1-3) = -0.497651(15), QUOTABLE (chains chi2/dof 0.58).
 vs the loop: D_0 = -2.13e-3 (C beats Marshall), D_1 = +5.07e-4, D_2 = +1.007e-3, D_3 = +1.058(18)e-3: LOOP_BEATS_C at 59 sigma (registered read on e_3).
 The loop's ROUND 1 alone (~2.9 A40-h of labels + fit) is already 5.1e-4 below SR at 7.7 A40-h. Scope: one SR setting (minSR, eta 0.05, 0.3 rad start, 1024 samples),
 still falling slowly at the end; hard sign moved only 1.6% of configurations from Marshall (the Marshall-trap picture of 2002.04613). It does NOT show SR cannot
 get there with other settings or more compute; it shows the loop is far cheaper for this class at 8x8 from a bad start.
## T-RG RESULT 2026-11-09 (Slurm 17867357, 36 tasks rc 0, ~1.5 core-h; sign_rg/{RESULT.md,result.json}; numbers checked by me in result.json)
KILL FIRED: C3/C2 = 18.3, 95% [17.1, 19.5] (C3 a lower bound: 5/18 requests capped at 600 s). C1 0.25 s (7.6 calls), C2 5.46 s (168), C3 >= 99.7 s (>= 3171).
Gating the lower rounds cut C2 ~10x vs T-K7's ungated recursion, but the ratio per round stays ~18.
Diagnosis (equation): C_{r+1}/C_r ~ (deep children per deep node) x (fraction left after pruning) ~ 25 x 0.3-0.4. Measured: level 2 634 -> 191 resolved
per request (70% pruned); mode 3 level 2 10877 -> 4249 (61%), level 3 365 -> 99 (73%). The certificate almost never closes before a deep child is
resolved (0 at level 2, 0.71/request at level 3). The term is the fan-out of deep children, ~25 per level: deep configurations cluster (ASTRA_k7: ~132 deep
neighbours per deep root). Pruning would need to remove ~95% per level to reach C3/C2 < 3.
Controls: r = 1 reproduces T-K7 S_-7 0/351; pruning off identical 0/351 (r = 2; r = 3 only 0/2, affordable on 2 requests); broken certificate wrong 99/351 (r = 2), 33/145 (r = 3).
t_2 != t_1 at 97/351 deep configurations (27.6%): the second gated round changes deep signs a lot.
Cheap untried variants (for Astra, before any verdict): other pruning orders (largest |a| first was used); a deeper gate (k = -8 or -9: fewer deep
children per node); exact sign for the deepest decades only plus storage for the rest. Verdict deferred to Astra (QUESTION8 item 7).

ASTRA ON THE FN-TRAINED SIGN (same file, sections 1-5): the finite-cut majorisation is exact (dE <= Q = -sum_cut t a a (g_x+g_y)^2, g = u/a), and it
removes the explicit 1/K label pole, but mixed walkers alone do not give g (off-diagonal FN-state ratios); hard signs have no gradient (a continuous
phase changes the problem). The P4 rule with FN ratios is not a descent certificate unless a ~ u. Precedents: McFarland-Manousakis 2112.14182 (direct
DMC-energy gradient, molecules), Nakano JCTC 2024 (restricted-class FN optimisation), Caffarel 2609.01301 (SHDMC can RAISE E_FN). Prior that a
restricted FN-aware learner gives a useful stored second update at 8x8: 30-40%. Cheapest test: <= 8 CPU core-h on the 6x6 exact B2 tables (sign_q0r),
an information-cost test of the estimator, not a verdict on 8x8. Not run yet (T-BAD first); offered at the check-in.
T-RG (section 7): cost kill stands; my diagnosis double-counted pruning (~25 is already the post-pruning paid fan-out; 87-111 candidates per level
would need ~97% pruning). Only variant worth a capped screen: deeper gate k = -8/-9 (<= 4 core-h), if the angle keeps exact recursion open.
## T-FNI (takeover; Astra ASTRA_fnsign.md section 5, registered there before any number; CPU <= 8 core-h)
Information-cost test of an FN-trained stored sign on the exact 6x6 fixed-B2 parent (sign_q0r): (1) reproduce full-cut Q for saved cuts + controls
(g = 1, off-diagonal omitted must fail); (2) train one frozen sign class's small parameter subset once on the exact-u cut objective vs the supervised
target, same data/cost, held-out roots; (3) exact Q (and FN solve if the margin warrants) for the hardened sign; (4) error of a deployable estimate of
g = u/a vs exact g, and the effective sample size it implies. Reads as in ASTRA_fnsign.md section 5 (mechanics pass; estimator passes only if its error
< descent margin). Scope: estimator information cost on a GOOD 6x6 parent; no verdict on 8x8 or on bad starts. Run dir sign_fni (worker).

T-FNI RESULT 2026-11-10 (CPU 2.9 core-h; Slurm 17875420; sign_fni/{RESULT.md,result.json}; key numbers checked by me in result.json):
 Step 1 PASS: cut bound Q reproduced to 8e-14 (lanczos +8.97e-8, hybrid -7.84e-9 /site); g = 1, diagonal-only, mixed-as-pure controls all fail (6-91%).
 Step 2 (deviation: no stored class fits this parent; 6-parameter head on the frozen 1-PP margin): FN-objective head Q -1.05e-11 /site (746x smaller than
 the hybrid cut), supervised head +9.18e-8 (worse); optimiser hit its 48-evaluation cap. Step 3: mechanics pass, dE_FN <= Q holds for every sign
 (hybrid dE -1.25e-8 vs Q -7.84e-9; random cut +3.1e-8); no false certified decrease. Step 4 FAIL (certificate level): the deployable g estimate
 (one Jacobi ratio, median error 0.65-1.8%) has Sum|dW| 6-10x the descent margin; certifying the hybrid margin needs ~1e9 independent roots.
 Reading: FN-energy training of a stored sign works in exact arithmetic but its walker estimator cannot certify the TINY margins of a good 6x6 parent
 (~1e-8/site). Scope: good parent only; from a bad start margins are ~1e-3, so the same estimator error may be affordable there (untested). Parked behind T-BAD.
## T-STRONG (lead's question 2026-11-11, PI yes; DESIGN ONLY, nothing runs before T-CAP8 and CONTROL C close)
Question: can ONE stored sign round started FROM a strong network's own sign end BELOW it? Base: rb8_retrain s306 (sha 45fa54e6, 'simple' projection,
full-projection VMC -0.49891 on 2 seeds). Bar: FN c = 0 p = 0 of the base with its own sign (Pfaffian angle runs it), same code path as T-BAD.
Mechanism (the term a sign round can lower): at fixed amplitude a, E_FN[a, s] changes by a sign change on a set F through the boundary bonds only
(ASTRA_node eq 3/5: dE = -sum_{dF} t_xy a_x a_y int (u_x/a_x + u_y/a_y)^2 dlambda). One Lanczos step flips x where 1 + alpha(E_L(x) - E) < 0, i.e. where the
guide's local energy is far above E because its node blocks bonds (large V(x)); those flips turn blocked bonds into kept ones where u is still large and
lower the sign-flip potential the walkers feel. The network's own sign has no such step built in; VMC optimised it for the variational energy, which can
move against E_FN (6x6: E_FN -5.27e-6 while E_VMC +2.04e-6). So the available gain is G_exact = E_FN[a, s_base] - E_FN[a, s_G], the exact one-step
sign at fixed amplitude. A stored round can only reach G_exact minus its storage loss L.
Why the gate (a) failures need not apply (and where they still might): gate (a) graded TRAINED REPLACEMENT signs on the label metric eps^2, and training moved
errors into high-amplitude decades (-2..-5) where the old sign was nearly right. Here: (i) storage is RESIDUAL, s = s_base x c(x) with c = flip indicator
trained on K[s_base]/s_base with a strong no-flip prior, so the base sign is kept exactly wherever c = +1 and copy errors can only be false or missed flips;
(ii) grading is the deployed FN energy, not eps^2. Where they still might apply: false flips in the bulk (P3: a stored sign differing in the bulk cost
+4.0(1.3)e-5 on |M1|) and the 64x6 class (held-out 10% in T-BAD, chance below d -11).
Size of G_exact (estimates, not measurements): 6x6 exact, sgn(psi_P) -> s_G on B_6: 1.06e-6 (A47); 8x8 first-step weight ~28x -> <= ~3e-5. Against:
A13 arm V (ViT sign) one Krylov step was a VMC null step at +-3e-6; T-NODE: the deep part (d <= -7) gives no node gain. So G_exact is likely 0.5-3e-5.
Prediction (angle lead): stored round below the base by > 2 sigma: 20% (lead 25%). Lower because T-BAD round 3 already stalls 4.5(1.0)e-5 above sgn(M1)
with gains shrinking x2, which suggests a storage loss of a few 1e-5 for this class near a strong sign, the same size as G_exact. Rises to ~30% if T-CAP8
shows the 64x10 net lowers e_3 beyond 2 sigma at fixed labels (storage loss reducible).
Kill test: g_S = E_FN[a, s_base] - E_FN[a, s_base c] not beyond 2 sigma(g_S) with sigma(g_S) <= 6e-6 -> KILL for this class at 8x8 (diagnose: false/missed
flip rates by decade; Astra). Stage 0 gate (cheap, before any FN): labels K[s_base] on a T-BAD-style sample at fixed |base| (flip fraction per decade);
residual CNN fit; if held-out FALSE flips in decades >= -5 exceed the true flip rate there, stop without FN (the stored round would add bulk errors).
Cost: stage 0 <= 3 A40-h (labels need |base| on two-hop sets; base is RBM+PP, cheaper than the CNN); stage 1 FN, two arms, sigma per arm ~4.2e-6:
~4.5x T-BAD's 350 blocks -> ~60 A40-h (net arm) + ~20 (base arm), i.e. ~80 A40-h or ~25 H100-h; theorie H200 from 11-22 or Paderborn cont preferred.

LEAD'S CONDITIONS (2026-11-12): stage 0 GO after T-CAP8 + CONTROL C close (no re-ask). Before FN: (1) resolution: grade g_S PAIRED with sigma <= 2.5e-6, or
state why not and the unpaired cost; (2) reuse the Pfaffian angle's B_0' FN p=0 arm on the same s306 base if the codes match; send launcher + GPU-h before
submission; R1 of the loop has priority on shared GPUs.
MY ANSWER on (1): common-random-number pairing in this GFMC code gives nothing: T-NODE measured block-energy correlations +0.039 / -0.016 / -0.015 after
burn, population hashes never re-align (ASTRA_node.md section 1), paired sigma 3.8e-6 vs unpaired 3.6e-6. Unpaired sigma(g) 2.5e-6 needs sigma per arm
1.8e-6 = ~26x T-BAD's 350 blocks: ~360 A40-h for the net arm alone: not reasonable. A real paired estimator (correlated sampling of the two guides on one
walk, validated against exact 6x6 = gate 2 of "NEXT QUESTION AT 8x8": <= 48 CPU core-h + 1-2 days of driver build) is the route; build and validate it
during stage 0, then FN. On (2): our generic path equals BareM1Sign bitwise (stage 0 control); the s306 amplitude loader and the Pfaffian angle's
launcher are compared at stage 0 before reusing their arm.

PAIRED FN ESTIMATOR (lead 2026-11-12: build first, SHARED tool under ansatz/code/ with README; validate vs exact 6x6 incl. a broken reweighting that
must miss; report weight distribution near flips at 6x6 and an 8x8 pilot; pointer to the lead when gate 2 passes; <= 48 CPU core-h + build).
Design asked of Astra first (cj-1470845f5b1a failed on quota; resubmitted cj-c9a7d4cfa462, astra/paired_fn/QUESTION.md -> ASTRA_paired.md); build by a worker to that spec.

T-STRONG STAGE 0 (lead GO 2026-11-12, start now; registered before any number; <= 3 A40-h; NO FN until the paired estimator passes its 6x6 gate at sigma <~ 3e-6):
 base s306 (rb8_retrain params_final sha 45fa54e6, 'simple' projection, 256 images; own sign sgn Re w psi), amplitude |s306|, code path = the Pfaffian angle's
 base_s306_p0 run (runs/j1j2/base_s306_p0/code; FN p=0 bar -0.4988780(27), Paderborn 4222088). Sample: T-BAD layout (tempered beta 1/2 of |s306|^2, 2560 chains x 4
 roots, 80/20 by chain, roots + all one-swap neighbours). Labels K[s_base](z) = s_base(z) sgn[1 + alpha(E_L(|s306| s_base; z) - E)], alpha and E the loop's
 sign-step constants for this base (from the base's own consts; state them). Report flip rate K != s_base per |s306|^2 decade.
 Residual mask: c(z) = K/s_base learned by the 64x6 CNN (BCE), deployed flip only where logit < -tau, tau chosen on the TRAIN split so that train false flips
 in decades >= -5 do not exceed train true flips there. Held-out gate: false-flip rate vs true-flip rate per decade; STOP (no FN ever for this design) if
 held-out false flips in decades >= -5 exceed the true flip rate there. Also report missed flips per decade and the fraction of true flips captured.
 Prediction (angle lead): flip rate above d -5 below 0.5% [70%]; gate passes [45%].
 SEALED PREDICTION (2026-11-13, before any stage-0 number; constants = headline_s306/inputs/consts_s306_p1.json sha 54add61c..., alpha -0.17605(337e-5),
 E -31.926142, read not recomputed). Basis: labels from a near-converged sign put flips in the tail: T-BAD K[n_2] vs n_2 (n_2 ~1e-4 above sgn(M1)): 0 at d >= -1, 0.2% at -3,
 2.5% at -5, 13.5% at -7, 31% at -9..-12; S_-7 on sgn(M1): s_G != s0 at 5.4% of deep roots, 24% of deep neighbours. s306 is a better sign than n_2 and than M1 (VMC
 1.6e-4 below M1), so per decade I predict flip rates of: <= 0.1% at d >= -3; 0.3-1.5% at -5; 3-10% at -7; 15-30% at -9..-11 (each band ~60%); overall ~2-6% of tempered
 states. Gate (held-out false flips in d >= -5 do not exceed true flips there): passes 45% (the 64x6 CNN had ~10% held-out disagreement on T-BAD labels; with
 flips ~1% at -5 the no-flip prior must hold false flips below ~1%). If the gate passes, G_exact stays 0.5-3e-5 and the stored round below the base by > 2 sigma 20%.
STAGE 0 (previously stopped 2026-11-12: no sign-step constants (alpha, E) for s306 exist (consts_s306.json has only w); lead: one constant set only, from the headline VMC job (headline_s306/scripts/vmc.sbatch, rerun by the Pfaffian angle on theorie A40, ~10 A40-h), installed at headline_s306/inputs/consts_s306_p1.json; the lead sends path + sha when its gate passes; stage 0 waits.
 Lead's follow-up to Astra (after cj-c9a7d4cfa462 lands): can the paired estimator grade a p=1 guide (Pfaffian headline B_k + p=1 vs B_0' + p=1)?
ASTRA ON THE PAIRED ESTIMATOR (cj-c9a7d4cfa462, /scratch/a/A.Otaifi/astra/paired_fn/ASTRA_paired.md, read): A-only path reweighting is INVALID for sign
 changes (B opens bonds absent from A's walk; exact counterexample). Recommended: one joint run of two COUPLED populations, each with its own configuration,
 rates, weights and multinomial resampling marginal: per bond b, both arms swap at min(r_A, r_B), A-only / B-only at the remainders; at block ends a maximal
 categorical coupling of the two parent laws by configuration. Cut formula Q_A is a derivative/bound, not the finite difference (FNI: Q -7.84e-9 vs
 Delta -1.25e-8); measuring both E_L on A's walk misses a first-order response term. Efficiency: sigma^2 x GPU-h = c_pair (S_A + S_B - 2 S_AB); reaching
 2.5e-6 at the old budget needs kappa_work <= 0.20 (rho >= 0.80 at equal cost); book NO speedup until a pilot measures rho. Staged validation:
 rows/finite propagators -> exact 6x6 calibration (>= 4 ensembles, sigma <= |Delta|/3) -> systematics (M, dtau, memory) -> broken versions miss
 (wrong ratio, omitted V_sf) -> 8x8 pilot (owner's OK). Astra 80% that this is the right estimator to test, LOW that it reaches 2.5e-6 at the old budget.
BUILD (takeover decision, lead GO): shared tool ansatz/code/paired_fn/ (README, Astra's interface section 5), validation stages 1-4 on CPU (<= 48 core-h),
 worker; 8x8 pilot only after I read stage 4.
PLAN ADOPTED (lead + PI, 2026-11-12): estimator build with an EARLY 8x8 RHO PROBE right after stage 1 (~0.25 A40-h; s306 vs s306 x random sparse flip mask at the
 expected density; A = B sanity arm must give rho = 1, a broken coupling must not): report separation time, recoupling rate, long-lag rho; STOP the build if
 rho < 0.5. Guesses (mine, recorded as guesses): rho mask 0.6-0.9 > size round 0.3-0.6 > p=1 < 0.3. If rho is low, sign tests make the effect bigger
 (cumulative 2-3 sign rounds, weaker start, 10x10) with a coarser bar stated as "cannot see below 1.2e-5"; no 360 A40-h budget. Parallel: R1-R3 unpaired
 (Pfaffian angle), stage 0 waits only for the s306 constants (A40 VMC job by the Pfaffian angle, 11-12 10:32; the lead sends the sha).
PAIRED ESTIMATOR STAGE 1 PASS 2026-11-12 (ansatz/code/paired_fn, committed; /scratch/a/A.Otaifi/ansatz_runs/paired_fn/stage1/stage1.json, VERDICT_STAGE1 PASS; worker's
 numbers): rows equal independent rows exactly; joint-generator marginals 4e-16; resampler marginals z <= 1.03; finite-T propagator matches on amplitude-only,
 sign-only and both pairs (|z| < 2.6); reweight_A raises SupportViolation on sign changes; broken versions miss at 10-437 sigma (inverse ratio, omitted V_sf,
 omitted changed walls, forced parent law, unchecked reweighting). 6x6: 95-97% of walker pairs still coupled after 30 blocks (bare S0->T0).
 8x8 RHO PROBE approved and submitted by the builder (launcher read by me: 30 min, 0.25 A40-h; A = B control 40 blocks first; s306 vs s306 x hash-random flip
 mask at density 1e-3 of configurations, i.e. uniform over the walker measure: HARSHER than T-STRONG's mask, which flips mostly deep configurations, so its rho
 is a conservative bound). Gate: stop the build if long-lag rho < 0.5.
PAIRED ESTIMATOR STAGE 2 + 8x8 PROBE 2026-11-12 (builder's numbers; probe Paderborn H100 4227108, paired_fn/probe8_20261112d/result.json; 6x6 job 17932399,
 stage2/analysis6.json; ~32 CPU core-h + 0.37 H100-h + 0.30 A40-h wasted on jit shapes):
 6x6 stage 2 PASS (coarse): 8 exact pairs x 4 ensembles all within 2 sigma (max |z| 1.34); micro-scale pairs resolved: first -5.68(34)e-6 vs exact -5.27e-6,
 refr2 +2.35(31)e-6 vs +2.39e-6, bareS0T0 +1.59(23)e-6 vs +1.30e-6; rho 0.60-0.998, kappa_work 0.003-0.6 (best for small differences).
 8x8 probe: A = B control exact (Delta == 0, zero exclusive hops). s306 vs s306 x hash-random mask pF = 1e-3: Delta = +7.0e-4/site (a LARGE perturbation,
 walker weight share in F 1e-3), separation time 32 tau, resampling recoupling alpha = matched fraction (0.335: NO net recoupling at 8x8), long-lag rho 0.25
 (binned 0.34(16)), kappa_work 0.65. REGISTERED GATE FIRED (rho < 0.5): stages 3-4 stopped.
 DIAGNOSIS (before any verdict, PI rule): the probe mask was harsher than T-STRONG's by design (uniform over the walker measure, 1e-3 weight share; Delta 7e-4
 is ~25-100x the expected T-STRONG gain). At 8x8 nothing re-pairs once arms separate, so rho is set by the separation time against the energy memory, and
 separation scales with the exclusive-edge rate ~ the walker weight touching F. A residual mask flips mostly deep configurations (walker weight ~1e-4 or less).
 Cheap variant (takeover decision): two more probes at pF = 1e-4 and 1e-5 (<= 0.5 H100-h each). Read: rho >= 0.5 at the density matching T-STRONG's
 stage-0 flip weight -> keep the tool for sign-only residual grades; else stop the build for 8x8 and use the fallback (cumulative effects, coarser bar).
 Prediction (angle lead): rho >= 0.8 at pF = 1e-5 [55%], >= 0.5 at 1e-4 [50%]. For amplitude/size rounds (B_k vs B_0') rho stays unmeasured at 8x8.
SEALED PREDICTION for the pF 1e-4 / 1e-5 probes (PI rule, projects/j1j2/CLAUDE.md point 1; written 2026-11-12 BEFORE either probe ran):
 Mechanism: at 8x8 separated arms never re-pair (alpha = matched fraction), so rho = mbar(T) x h(pF):
   mbar(T) = (tau_s/T)(1 - exp(-T/tau_s))   [run-averaged matched fraction],   tau_s(pF) = tau_s(1e-3) x (1e-3/pF)   [separation flux ~ walker weight on F ~ pF],
   h(pF) = 1/(1 + kappa_h x 146 pF)          [matched walkers with a flipped neighbour have different E_L; 146 = valid one-swap neighbours].
 Calibration on probe8_20261112d (pF 1e-3, T = 421 blocks x dtau 0.1 = 42.1 tau): measured mbar = alpha_mean 0.335 -> effective tau_s(1e-3) = 15 tau (the per-walker
 separation time is 32.3 tau; population branching roughly halves it); h = rho/mbar = 0.25/0.335 = 0.74 -> kappa_h = 2.4. Check: kappa_work = r(1 - rho) = 0.93 x 0.75 = 0.70
 vs measured 0.65. 6x6 consistency: rho -> 0.88-0.998 as the pair difference shrinks (1e-7, 1e-8 pairs), as the model says.
 Predictions at the same run length (42 tau):
   pF 1e-4: tau_s 150 tau, mbar 0.87, h 0.97 -> rho 0.84 (band 0.70-0.93); separation rate 3.1e-3/tau (+-x2).
   pF 1e-5: tau_s 1500 tau, mbar 0.99, h 0.997 -> rho 0.98 (band 0.93-1.0); separation rate 3.1e-4/tau (+-x2).
 Miss = the result: separation rate not proportional to pF within x2 names the flux term; rho(1e-4) < 0.6 with the rate as predicted names h (E_L differences at
 matched walkers) or genealogy.
 T-STRONG density: pF_eq = walker weight on flipped configurations. Proxy (s306's mask is unmeasured until stage 0): T-NODE S_-7 on |M1|: residence below d -7 1.1e-3
 x 5.4% root flips = 6e-5 (range 2e-5..2e-4). Production with chains of T_c = 50 tau from shared starts: pF_eq 6e-5 -> tau_s 250 tau, mbar 0.91, h 0.98 -> rho 0.89;
 pessimistic 2e-4 -> rho 0.68.
 Need: kappa_work = r(1 - rho), r = 0.93. sigma(diff) 2.5e-6 at the old 5.6e-6 budget needs kappa <= 0.20 -> rho >= 0.785. Any rho > 0 is cheaper than unpaired.
 So rho at the stage-0 density is predicted > 0.5 (0.68-0.89): the probes are worth their <= 1 H100-h.
 AMENDMENT (same day, before any probe number was seen: the builder had already submitted Paderborn 4227404 / 4227405 when the hold arrived; it reports
 them only after this seal). The probes run ~1100 blocks = 110 tau, not 42 tau; the SAME equations at T = 110 tau give: pF 1e-4: mbar 0.71, rho 0.68 (band 0.50-0.85);
 pF 1e-5: mbar 0.96, rho 0.96 (band 0.88-0.99). These 110-tau numbers are the ones to score. Placement note: the builder used Paderborn EXPRESS (<= 1 H100-h)
 against my takeover rule (no express; reserved for the lead's headline); reported to the lead.
T-STRONG STAGE 0 RESULT 2026-11-13 (A40 jobs 17950928-30 + labels 17951033/34/107; runs/j1j2/sign_strong/s0/r/{ctl1,ctl2}.json, fit/fit_8192.json; read and recomputed per
 decade by me): controls pass (s306 evaluator vs the Pfaffian angle's: dlnpsi 2e-13, 0 sign mismatches over 2656 neighbours, broken w differs 1241; labels vs direct 1.6e-10,
 0 mismatches; alpha = 0 -> no flips; wrong alpha changes 3.8-5.6%). Constants read from consts_s306_p1.json (sha 54add61c).
 Flip rate K[s306] != s306 on 1,455,822 tempered states: 11.4% overall; by |s306|^2 decade: 0 at d >= -2, 1.2e-5 at -3, 3.4e-4 at -4, 0.22% at -5, 0.94% at -6, 3.0% at -7,
 7.5% at -8, 15% at -9, 24% at -10, 33% at -11, 42-60% below -12. vs the sealed bands: d >= -3 hit; -5 0.22% (band 0.3-1.5%: miss low); -7 3.0% (band 3-10%: edge);
 -9..-11 15-33% (band 15-30%: hit); overall 11% (2-6%: MISS: the tempered sample puts most states in deep decades; my overall share used the wrong weight).
 Mask (64x6 CNN, BCE on c = K/s306, tau = 0 by the rule): registered GATE PASS (held-out false flips d >= -5: 1 vs 88 true), but VACUOUSLY: the mask captures
 0/79 true flips at -5, 0.3% at -6, 0.5% at -7, 1.1% at -8, 3.2% at -9, 7.1% at -10, 16% at -11, 33-100% only below -12 (where it also has 9-43% false flips).
 Held-out disagreement 10.0% vs no-flip 11.1%. MY ERROR: the gate checked false flips only, not capture; a mask that never flips passes it.
 Reading: the stored round would change the node only in the deepest decades, where T-NODE found no node gain even for EXACT flips (+8.0(4.1)e-6). The FN stage on this
 mask is not worth running. Diagnosis (equation): BCE with positives at 0.2-7% in d -5..-8 is minimised by "no flip" there (class imbalance; the loss gain from a correct
 rare flip is smaller than the loss of the false flips it brings). Cheapest variant (registered now, before any number): class-weighted BCE (weight on flips = 1/flip rate per
 decade, capped at 100), same class and data, tau chosen on train so that false flips <= true flips captured in d -8..-5. Reads: captured fraction >= 50% at d -7..-5 with
 false/captured <= 1 -> a usable mask, then decide FN (paired estimator permitting); captured < 20% -> the 64x6 class cannot store the strong-start sign step at 8x8 (record,
 Astra before verdict). Prediction: usable 35%. Cost ~1 A40-h.
 AMENDED (lead, 2026-11-13): first a THRESHOLD SWEEP of the existing net on held-out states (minutes; a class weight w only shifts the BCE-optimal logit by ~log w), per
 decade and in energy cost (cost(z) = |B0(z)| sum_y |H_zy| |B0(y)|, tempered-to-|B0|^2 reweighted bond weight). If no threshold reaches capture >= 50% at d -7..-5 with
 false <= captured (counts and cost), skip the weighted run: the limit is the features -> class question to Astra. Else seal the operating point, then run it.
THRESHOLD SWEEP RESULT 2026-11-13 (worker; sign_strong/s0w/{sweep.json,sweep_table.txt}; 41 thresholds on the stage-0 net's held-out logits): read (2): NO threshold works.
 First threshold with capture >= 50% at each of d -7/-6/-5 (t = 2.47): captured 0.65/0.55/0.56 (~710/200/44) vs false 6701/2985/980, i.e. 5-20x more false than captured;
 at t = 3.0 pooled d -8..-5: 3605 captured vs 34385 false, held-out disagreement 32% (no-flip 11%). Where false <= captured, capture < 1%. Energy-weighted (lower bound:
 only 3.6% of neighbours present in the table): false cost > captured cost at every threshold (t 3.0: 1.06 vs 0.154 over d -8..-5). The weighted BCE run is SKIPPED.
 Reading (kill of "imbalance" fired; diagnosis, not a verdict): the CNN's logit does not separate flip from no-flip states at mid decades, so the limit is the features/class.
 Equation: the label is sgn[1 + alpha(E_L^{s306}(z) - E)], a threshold on the base guide's local energy = sum over ~146 neighbours of H_zy psi(y)/psi(z) of a Pfaffian+RBM;
 a spin-input CNN must reproduce that whole sum near its threshold, and the near-threshold states are exactly the rare mid-decade flips. Not tested: a pass of a weighted net
 (the sweep tests only a logit shift); a regression target (the continuous v = 1 + alpha(E_L - E), Astra's pole-free u = P/D) instead of the hard sign; inputs from the base
 network itself. Next: Astra on the class question (QUESTION9) before any verdict.
## Queue (not run; PI sets priority)
- WATCH (PI 2026-11-11, DISCUSSION P14/P15): keep the global Lanczos sign step up to 14x14 (dilution measured only beyond ~1e3 sites for good guides,
  TRIED_IDEAS D4). No patch/local rule (PI: sign is non-local; local rule lost at 8x8, D3; AR2, A44). Above ~16x16 revisit with more projection (e^{-tau H}), not patches.
- Evidence for T-JOINT (Pfaffian angle OBSERVATION 2026-11-05, PLAN_pfaffian.md; same-x check sg_check.py running): on the 8x8 pool the heavy-x tail
  (f_J(x) < -0.5) is 8 / 35 / 205 of 2e6 draws at decades -7/-8/-9 under the net's own sign sgn(M1), vs 3-10% of x under s_G. Mechanism: B's own
  nodes leave kept bonds around every x (SK stays large); s_G's nodes cut through regions where |B| has no node and strand x with few kept bonds,
  exactly the costly unblocking regime (T-BOND). A sign whose node follows the amplitude avoids it by construction.
- T-JOINT (PI: ask Astra, do not run; question 6 in astra/sign_eps2/QUESTION.md) (lead's design 2026-11-05, on the PI's question): one complex two-PP net storing s_G x |T_5| at 6x6 exact (refit from theta_4; modulus
  loss as R5 + bond phase loss with the T-BOND weight); grade exact FN with its own sign and walls vs separate B_5 + S_t (R5 aug 3.97e-6).
  ~1.5 H100-h + CPU. Untried (a joint REAL net is ruled out: own node fought the imposed sign, q 0.424, P1; complex node-free copy halved the tax).
  Lead's prior joint <= separate 35%; mine 40%.

## Dead ends
- S_-7 as a c = 0 NODE on |M1|: +8.0(4.1)e-6/site vs sgn(M1), kill fired (T-NODE RESULT, sign_node/prod, 4187008). Label use PARKED on cost (15.5k core-h relabel, sign_relabel_pilot; A49).
- Trained stored 8x8 sign (BCE / IW / PP-full / keep-old gates / soft-target Huber): fresh eps^2 0.69-0.70 vs sgn(M1) 0.40 (T-SOFT2, A48); induced
  kept-bond-sum change outweighs the gain (ASTRA_soft2.md). Practical stop for this recipe, not an impossibility theorem.
- Confidence-gated stored+exact hybrid for every round: C3/C2 4.25 (T-ROUND3), stored confident signs carry no usable information (T-GATE).
- Two-net disagreement gate (I1): 8x8 nets share their errors (13.3% wrong given agreement; T-CERT V1).
- T-BOND per-bond surrogate loss: rejected before running (eps^2 is a Laplacian form; Gate S false positive; ASTRA_eps2.md).
- Local shallow CNN sign: 3x3-patch correction moves w_T -0.7% (A44). Hybrid XL node + PP walls: 0.998e-6 (A45). Coverage (beta mix): A46.
